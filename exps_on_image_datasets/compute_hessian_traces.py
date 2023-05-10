import os
import torch
import collections
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.loss import nll_loss
from parse_config import ConfigParser
import data_loader.data_loaders as module_data
import model.model as module_arch
import argparse
from utils import prepare_device, deep_copy
import random
from model.modeling_vit import VisionTransformer, CONFIGS
from utils.hessian import set_seed, compute_hessians_trace, compute_eigenvalue
from data_loader.random_noise import label_noise


def main(config, args):
    set_seed(0)
    logger = config.get_logger('generate')
    logger = config.get_logger('train')

    # setup data_loader instances
    if config["data_loader"]["type"] == "CaltechDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, idx_start = 0, img_num = 30, phase = "train")
        valid_data_loader = config.init_obj('data_loader', module_data, idx_start = 30, img_num = 20, phase = "val")
        test_data_loader = config.init_obj('data_loader', module_data, idx_start = 50, img_num = 20, phase = "test")
    elif config["data_loader"]["type"] == "AircraftsDataLoader" or config["data_loader"]["type"] == "DomainNetDataLoader":
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
    model.load_state_dict(
            torch.load(os.path.join(file, args.checkpoint_name+".pth"))["state_dict"]
        )
    model.to(device)

    hessian_traces = []
    hessian_lambdas = []
    max_layer_trace = []; max_lambda_1 = []

    sample_size = 0
    not_improving = 0
    model.eval()
    for data, target, index in train_data_loader:
        num_samples = data.shape[0]
        data, target = data.to(device), target.to(device)
        model.load_state_dict(
            torch.load(os.path.join(file, args.checkpoint_name+".pth"))["state_dict"]
            )
        model.to(device)

        model.eval()
        output = model(data)
        loss = nll_loss(output, target)

        print(loss)
        layer_traces = compute_hessians_trace(model, loss, device=device)
        lambda_1, _ = compute_eigenvalue(model, loss, device=device, top_n=1) 
        
        hessian_traces.append(layer_traces)
        hessian_lambdas.append(np.array(lambda_1[0]))

        # if max_layer_trace == []:
        #     max_layer_trace = layer_traces
        #     max_lambda_1 = np.array(lambda_1[0])
        # else:
        #     max_layer_trace = np.maximum(max_layer_trace, layer_traces)
        #     max_lambda_1 = np.maximum(max_lambda_1, np.array(lambda_1[0]))

        # logger.info(max_layer_trace); logger.info(max_layer_trace.sum())
        # logger.info(max_lambda_1); logger.info(max_lambda_1.sum())
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