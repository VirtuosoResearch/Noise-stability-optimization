import os
import torch
import collections
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.loss import nll_loss, bce_loss
from parse_config import ConfigParser
import data_loader.data_loaders as module_data
import model.model as module_arch
import argparse
from utils import prepare_device, deep_copy
import random
from model.modeling_vit import VisionTransformer, CONFIGS
from utils.hessian import set_seed, compute_hessians_trace, compute_eigenvalue
from data_loader.random_noise import label_noise

criterion = nll_loss
def compute_loss(model, data_loader, device = "cpu"):
    loss = 0
    batch_count = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for idx, (data, labels, index) in enumerate(data_loader):
            data, labels = data.to(device), labels.to(device)

            loss += criterion(model(data), labels)
            batch_count += 1

    return loss/batch_count

def main(config, args):
    set_seed(args.seed)
    logger = config.get_logger('generate')
    logger = config.get_logger('train')

    # setup data_loader instances
    if config["data_loader"]["type"] == "CaltechDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, idx_start = 0, img_num = 30, phase = "train")
        valid_data_loader = config.init_obj('data_loader', module_data, idx_start = 30, img_num = 20, phase = "val")
        test_data_loader = config.init_obj('data_loader', module_data, idx_start = 50, img_num = 20, phase = "test")
    elif config["data_loader"]["type"] == "AircraftsDataLoader" \
        or config["data_loader"]["type"] == "DomainNetDataLoader"\
        or config["data_loader"]["type"] == "CXRDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, phase = "train")
        valid_data_loader = config.init_obj('data_loader', module_data, phase = "val")
        test_data_loader = config.init_obj('data_loader', module_data, phase = "test")
    elif config["data_loader"]["type"] == "BirdsDataLoader" or \
        config["data_loader"]["type"] == "CarsDataLoader" or \
        config["data_loader"]["type"] == "DogsDataLoader" or \
        config["data_loader"]["type"] == "IndoorDataLoader" or \
        config["data_loader"]["type"] == "Cifar10DataLoader" or \
        config["data_loader"]["type"] == "Cifar100DataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, valid_split = 0.1, phase = "train")
        valid_data_loader = train_data_loader.split_validation()
        test_data_loader = config.init_obj('data_loader', module_data, phase = "test")
    elif config["data_loader"]["type"] == "FlowerDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data)
        valid_data_loader = train_data_loader.split_validation()
        test_data_loader = train_data_loader.split_test()
    elif config["data_loader"]["type"] == "AnimalAttributesDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data)
        valid_data_loader = train_data_loader.split_validation()
        test_data_loader = train_data_loader.split_test()
    elif config["data_loader"]["type"] == "MessidorDataLoader" or \
        config["data_loader"]["type"] == "AptosDataLoader" or \
        config["data_loader"]["type"] == "JinchiDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, valid_split = 0.2, test_split=0.2, phase = "train")
        valid_data_loader = train_data_loader.split_validation()
        test_data_loader = train_data_loader.split_test()

    # If small data, shrink training data size
    assert 0 < args.data_frac <= 1
    if args.data_frac < 1:
        train_data_len = len(train_data_loader.sampler)
        train_data_loader.sampler.indices = train_data_loader.sampler.indices[:int(train_data_len*args.data_frac)]
        train_labels_old = None # 
        if args.downsample_test:
            test_data_len = len(test_data_loader.sampler)
            val_data_len = len(valid_data_loader.sampler)
            test_data_loader.sampler.indices = test_data_loader.sampler.indices[:int(test_data_len*args.data_frac)]
            valid_data_loader.sampler.indices = valid_data_loader.sampler.indices[:int(val_data_len*args.data_frac)]
    logger.info("Train Size: {} Valid Size: {} Test Size: {}".format(
        len(train_data_loader.sampler), 
        len(valid_data_loader.sampler), 
        len(test_data_loader.sampler)))

    
    if args.synthetic_noise:
        if config["data_loader"]["type"] == "DomainNetDataLoader" or config["data_loader"]["type"] == "AnimalAttributesDataLoader":
            train_data_loader.dataset.labels = train_data_loader.dataset.true_labels
        wrong_indices, train_labels_old = label_noise(
            train_data_loader.dataset, train_data_loader.sampler.indices, args.noise_rate, symmetric=True
        )
        logger.info("Randomizing {} number of labels".format(wrong_indices.shape[0]))

    if args.is_vit:
        vit_config = CONFIGS[args.vit_type]
        model = config.init_obj('arch', module_arch, config = vit_config, img_size = args.img_size, zero_head=True)
    else:
        model = config.init_obj('arch', module_arch, pretrained=True)

    file = os.path.join("./saved_hessians/", args.checkpoint_dir)
    device, device_ids = prepare_device(config['n_gpu'])
    if args.load_multiple_points:
        assert len(args.checkpoint_names) > 0
        num_points = len(args.checkpoint_names)
        # average state dict
        final_state_dict = deep_copy(model.state_dict())
        for i in range(num_points):
            state_dict = torch.load(os.path.join(file, args.checkpoint_names[i]+".pth"))["state_dict"]
            for key, val in state_dict.items():
                if i == 0:
                    final_state_dict[key] = state_dict[key]
                else:
                    final_state_dict[key] += state_dict[key]
        for key, val in final_state_dict.items():
            final_state_dict[key] = final_state_dict[key]/float(num_points)
        model.load_state_dict(final_state_dict)
    else:
        model.load_state_dict(
                torch.load(os.path.join(file, args.checkpoint_name+".pth"))["state_dict"]
            )
    model.to(device)

    model.eval()
    train_loss = compute_loss(model, train_data_loader, device)
    test_loss = compute_loss(model, test_data_loader, device)
    print(test_loss, train_loss)
    print("Generalization error: {}".format(test_loss - train_loss))

    hessian_traces = []
    hessian_lambdas = []
    max_layer_trace = []; max_lambda_1 = []

    sample_size = 0
    not_improving = 0
    model.eval()
    data_loader = test_data_loader if args.use_test else train_data_loader 
    for data, target, index in data_loader:
        num_samples = data.shape[0]
        data, target = data.to(device), target.to(device)
        # model.load_state_dict(
        #     torch.load(os.path.join(file, args.checkpoint_name+".pth"))["state_dict"]
        #     )
        # model.to(device)
        # model.eval()
        output = model(data)
        loss = criterion(output, target)

        print(loss)
        layer_traces = compute_hessians_trace(model, loss, device=device)
        lambda_1, _ = compute_eigenvalue(model, loss, device=device, top_n=1) 
        
        hessian_traces.append(layer_traces)
        hessian_lambdas.append(np.array(lambda_1[0]))

        if max_layer_trace == []:
            max_layer_trace = layer_traces
            max_lambda_1 = np.array(lambda_1[0])
        else:
            max_layer_trace = np.maximum(max_layer_trace, layer_traces)
            max_lambda_1 = np.maximum(max_lambda_1, np.array(lambda_1[0]))

        logger.info(max_layer_trace); logger.info(max_layer_trace.sum())
        logger.info(max_lambda_1); logger.info(max_lambda_1.sum())
        print(np.mean(np.array(hessian_traces), axis=0), np.mean(np.array(hessian_traces), axis=0).sum())
        print(np.mean(np.array(hessian_lambdas), axis=0), np.mean(np.array(hessian_lambdas), axis=0).sum())
        logger.info("========== Batch Complete ==========")

        sample_size += num_samples
        if sample_size > args.sample_size:
            break
    print("Sum of trace: {}".format(np.mean(np.array(hessian_traces), axis=0).sum()))
    print("Sum of top-1 eigenvalues: {}".format(np.mean(np.array(hessian_lambdas), axis=0).sum()))
        

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="cpu", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--seed', default=0, type=int)
    args.add_argument('--use_test', action="store_true")
    args.add_argument('--data_frac', type=float, default=1.0)
    args.add_argument('--downsample_test', action="store_true")
    
    args.add_argument('--synthetic_noise', action="store_true")
    args.add_argument('--noise_rate', type=float, default=0.0)
    
    args.add_argument('--is_vit', action="store_true")
    args.add_argument('--img_size', type=int, default=224)
    args.add_argument("--vit_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    args.add_argument("--vit_pretrained_dir", type=str, default="checkpoints/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")

    args.add_argument("--checkpoint_dir", type=str, default="ResNet50_IndoorDataLoader_none_none_1.0000_1.0000_rand_init_True")
    args.add_argument("--checkpoint_name", type=str, default="model_epoch_30")
    args.add_argument("--save_name", type=str, default="finetuned_train")
    args.add_argument("--sample_size", type=int, default=100)
    # args.add_argument("--num_layers", type=int, default=18)

    args.add_argument('--load_multiple_points', action="store_true")
    args.add_argument("--checkpoint_names", type=str, nargs="+", default=["model_epoch_20", "model_epoch_25", "model_epoch_30"])

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--model'], type=str, target="arch;type"),
        CustomArgs(['--domain'], type=str, target="data_loader;args;domain"),
        CustomArgs(['--sample'], type=int, target="data_loader;args;sample"),
    ]
    config, args = ConfigParser.from_args(args, options)
    print(config)
    main(config, args)