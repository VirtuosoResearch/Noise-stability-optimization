import os
import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import *
from utils import prepare_device, deep_copy
from data_loader.data_loaders import MnistDataLoader

def main(config, args):
    logger = config.get_logger('train')

    train_data_loader = MnistDataLoader(data_dir="./data", batch_size=args.batch_size, shuffle=True, num_workers=4, training=True)
    test_data_loader = MnistDataLoader(data_dir="./data", batch_size=args.batch_size, shuffle=False, num_workers=4, training=False)
    valid_data_loader = test_data_loader

    model = module_arch.MLP(input_dim=28*28, hidden_dim=args.hidden_dim, n_classes=10, n_layers=1)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    source_state_dict = deep_copy(model.state_dict())
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    accuracies = []
    for run in range(args.runs):
        model.reset_parameters(source_state_dict)
        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        checkpoint_dir = os.path.join(
        "./saved", 
        "{}_{}_hidden_{}_run_{}".format(config["arch"]["type"], config["data_loader"]["type"], args.hidden_dim, run))
        trainer = Trainer(model, criterion, metrics, optimizer,
                        config=config,
                        device=device,
                        train_data_loader=train_data_loader,
                        valid_data_loader=valid_data_loader,
                        test_data_loader=test_data_loader,
                        lr_scheduler=lr_scheduler,
                        checkpoint_dir = checkpoint_dir)
                        
        trainer.train()
        accuracies.append(trainer.test())
    logger.info("Test Accuracy {:1.4f} +/- {:1.4f}".format(np.mean(accuracies), np.std(accuracies)))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--runs', type=int, default=3)
    args.add_argument('--batch_size', type=int, default=256)
    args.add_argument('--hidden_dim', type=int, default=32)

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--model'], type=str, target="arch;type"),
        CustomArgs(['--weight_decay'], type=float, target="optimizer;args;weight_decay"),
        CustomArgs(['--early_stop'], type=int, target="trainer;early_stop"),
        CustomArgs(['--epochs'], type=int, target="trainer;epochs"),
    ]
    config, args = ConfigParser.from_args(args, options)
    print(config)
    main(config, args)
