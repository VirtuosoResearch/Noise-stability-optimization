import argparse
import collections
import os
import random
from collections import OrderedDict

import datasets
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, load_metric

import transformers
from data_loader.load_data_fns import load_glue_tasks
from parse_config import ConfigParser
from trainer import *
from transformers import (AdamW, AutoModelForSequenceClassification,
                          SchedulerType, get_scheduler)
from utils import deep_copy, prepare_device
from utils.util import prepare_inputs
from utils.hessian import get_layers, compute_hessian_traces, set_seed

''' Define a function to calculate stability '''
def perturbe_model_weights(state_dict, eps=0.001, use_neg = False, perturbation = {}):
    if not use_neg:
        perturbation = {} 
    for key, value in state_dict.items():
        if ("encoder" in key or "classifier" in key) and "weight" in key and ("LayerNorm" not in key):
            if use_neg:
                state_dict[key] -= perturbation[key]
            else:
                tmp_perturb = torch.randn_like(value)*eps
                state_dict[key] += tmp_perturb
                perturbation[key] = tmp_perturb
    return state_dict, perturbation

def compute_loss(model, data_loader, device = "cpu", batch_num=100):
    loss = 0
    batch_count = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            batch = prepare_inputs(batch, device)
            outputs = model(**batch)
            loss += outputs.loss
            batch_count += 1
            if batch_count > batch_num:
                break

    return loss/batch_count

def calculate_stability(model, data_loader, eps=1e-3, device = "cpu", runs = 20, batch_num = 100):
    ''' Calculate pred_vectors for model before perturbation '''
    loss_before = compute_loss(model, data_loader, device=device, batch_num=batch_num)
    print(f"Loss before: {loss_before}")
    state_dict_before = deep_copy(model.state_dict())

    '''
    Calculate the perturbed loss
    '''
    differences = []
    for i in range(runs):
        differece = 0
        state_dict_after = deep_copy(state_dict_before)
        state_dict_after, perturbations = perturbe_model_weights(state_dict_after, eps = eps)
        model.load_state_dict(state_dict_after)
        # print(list(perturbations.keys()))
        
        loss_after = compute_loss(model, data_loader, device=device)
        differece += loss_after - loss_before
        print(f"Loss after: {loss_after}")
        # differences.append(differece.cpu().item())

        state_dict_after = deep_copy(state_dict_before)
        state_dict_after, _ = perturbe_model_weights(state_dict_after, eps = eps, use_neg=True, perturbation = perturbations)
        model.load_state_dict(state_dict_after)
        
        loss_after = compute_loss(model, data_loader, device=device)
        differece += loss_after - loss_before
        print(f"Loss after: {loss_after}")
        differences.append(differece.cpu().item()/2)
    return differences


def main(config, args):
    set_seed(0)
    logger = config.get_logger('train')
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    
    # Load dataset
    train_data_loader, valid_data_loader, test_data_loader, transformers_config = load_glue_tasks(
        args.task_name, logger=logger,
        model_name_or_path=args.model_name_or_path,
        pad_to_max_length=config["data_loader"]["args"]["pad_to_max_length"],
        max_length=config["data_loader"]["args"]["max_length"],
        train_batch_size=config["data_loader"]["args"]["train_batch_size"],
        eval_batch_size=config["data_loader"]["args"]["eval_batch_size"]
    )
    # test_data_loader = None

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)

    # Load pretrained model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=transformers_config,
    )

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    file = os.path.join("./saved/", args.checkpoint_dir)
    model.load_state_dict(
            torch.load(os.path.join(file, f"model_epoch_{args.epoch}.pth"))["state_dict"]
        )
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    train_loss = compute_loss(model, train_data_loader, device=device, batch_num=10000)
    test_loss = compute_loss(model, test_data_loader, device=device, batch_num=10000)
    print("Training loss: {}".format(train_loss))
    print("Test loss: {}".format(test_loss))
    data_loader = train_data_loader
    if not args.compute_hessian_trace:
        diff_losses = calculate_stability(model, data_loader, eps=args.eps, device = device, batch_num=args.sample_size)
        logger.info("Noise stability of {}: {:.6f} +/- {:.6f}".format(
            args.eps, np.mean(diff_losses), np.std(diff_losses)
        ))

    if args.compute_hessian_trace:
        traces = []
        max_traces = np.zeros(73); max_loss = 0
        sample_count = 0
        model.eval()
        for _, batch in enumerate(data_loader):
            model.load_state_dict(
                torch.load(os.path.join(file, f"model_epoch_{args.epoch}.pth"))["state_dict"]
            )

            batch = prepare_inputs(batch, device)
            outputs = model(**batch)
            loss = outputs.loss

            layer_traces, loss = compute_hessian_traces(model, loss, device = device)
            max_traces = np.maximum(max_traces, layer_traces)
            max_loss = max(max_loss, loss)

            traces.append(np.sum(layer_traces))

            logger.info("Current layer traces: {}".format(np.sum(layer_traces)))
            logger.info("Traces mean: {}".format(np.mean(traces)))
            logger.info("Max layer traces: {}".format(max_traces))
            logger.info("Max loss: {}".format(max_loss))

            sample_count += 1
            if sample_count > args.sample_size:
                break

        pretrained_state_dict = torch.load(os.path.join(file, "model_epoch_0.pth"))["state_dict"]
        state_dict = torch.load(os.path.join(file, f"model_epoch_{args.epoch}.pth"))["state_dict"]

        weights = []
        for key, val in state_dict.items():
            if "weight" in key and ("LayerNorm" not in key and "embeddings" not in key and "pooler" not in key):
                weights.append(val - pretrained_state_dict[key])

        norms = np.array([torch.norm(w).item() for w in weights])
        train_num = len(train_data_loader.dataset)
        bound = max_loss*np.math.sqrt((max_traces.sum()*np.square(norms).sum())/train_num)

        print("Traces: {}".format(max_traces.sum()))
        print("Train loss: {}".format(train_loss))
        print("Test loss: {}".format(test_loss))
        print("Generalization error: {}".format(test_loss - train_loss))
        print("Bound: {}".format(bound))
        

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument("--epoch", type=int, default=1)

    args.add_argument('--task_name', type=str, default="mrpc")
    args.add_argument('--model_name_or_path', type=str, default="bert-base-cased")
    args.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    args.add_argument("--eps", type=float, default=1e-3)
    args.add_argument("--checkpoint_dir", type=str, default="mrpc_False")
    args.add_argument("--checkpoint_name", type=str, default="model_best")
    args.add_argument("--save_name", type=str, default="finetuned_train")
    args.add_argument("--sample_size", type=int, default=100)
    args.add_argument("--early_stop", type=int, default=20)
    args.add_argument("--compute_hessian_trace", action="store_true")

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--epochs'], type=int, target='trainer;num_train_epochs'),
        CustomArgs(['--warm_up'], type=int, target='trainer;num_warmup_steps'),
    ]
    config, args = ConfigParser.from_args(args, options)
    main(config, args)
