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
from data_loader.random_noise import label_noise

from model.modeling_vit import VisionTransformer, CONFIGS
from utils.dual_t import get_transition_matrices, compose_T_matrices
from utils.sam import SAM
from utils.nsm import NSM
from utils.rsam import RSAM 
from utils.bsam import BSAM

def main(config, args):
    logger = config.get_logger('train')

    # setup data_loader instances
    if config["data_loader"]["type"] == "CaltechDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, idx_start = 0, img_num = 30, phase = "train", use_augmentation = args.use_augmentation)
        valid_data_loader = config.init_obj('data_loader', module_data, idx_start = 30, img_num = 20, phase = "val", use_augmentation = args.use_augmentation)
        test_data_loader = config.init_obj('data_loader', module_data, idx_start = 50, img_num = 20, phase = "test", use_augmentation = args.use_augmentation)
    elif config["data_loader"]["type"] == "AircraftsDataLoader" \
        or config["data_loader"]["type"] == "DomainNetDataLoader" \
        or config["data_loader"]["type"] == "CXRDataLoader" :
        train_data_loader = config.init_obj('data_loader', module_data, phase = "train", use_augmentation = args.use_augmentation)
        valid_data_loader = config.init_obj('data_loader', module_data, phase = "val", use_augmentation = args.use_augmentation)
        test_data_loader = config.init_obj('data_loader', module_data, phase = "test", use_augmentation = args.use_augmentation)
        # valid_data_loader = test_data_loader
    elif config["data_loader"]["type"] == "BirdsDataLoader" or \
        config["data_loader"]["type"] == "CarsDataLoader" or \
        config["data_loader"]["type"] == "DogsDataLoader" or \
        config["data_loader"]["type"] == "IndoorDataLoader" or \
        config["data_loader"]["type"] == "Cifar10DataLoader" or \
        config["data_loader"]["type"] == "Cifar100DataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, valid_split = 0.1, phase = "train", use_augmentation = args.use_augmentation)
        valid_data_loader = train_data_loader.split_validation()
        test_data_loader = config.init_obj('data_loader', module_data, phase = "test", use_augmentation = args.use_augmentation)
    elif config["data_loader"]["type"] == "FlowerDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, use_augmentation = args.use_augmentation)
        valid_data_loader = train_data_loader.split_validation()
        test_data_loader = train_data_loader.split_test()
    elif config["data_loader"]["type"] == "AnimalAttributesDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, use_augmentation = args.use_augmentation)
        valid_data_loader = train_data_loader.split_validation()
        test_data_loader = train_data_loader.split_test()
    elif config["data_loader"]["type"] == "MessidorDataLoader" or \
        config["data_loader"]["type"] == "AptosDataLoader" or \
        config["data_loader"]["type"] == "JinchiDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, valid_split = 0.2, test_split=0.2, phase = "train")
        valid_data_loader = train_data_loader.split_validation()
        test_data_loader = train_data_loader.split_test()

    if args.synthetic_noise:
        if config["data_loader"]["type"] == "DomainNetDataLoader" or config["data_loader"]["type"] == "AnimalAttributesDataLoader":
            train_data_loader.dataset.labels = train_data_loader.dataset.true_labels
        wrong_indices, train_labels_old = label_noise(
            train_data_loader.dataset, train_data_loader.sampler.indices, args.noise_rate, symmetric=True
        )
        logger.info("Randomizing {} number of labels".format(wrong_indices.shape[0]))
    else:
        if config["data_loader"]["type"] == "DomainNetDataLoader" or config["data_loader"]["type"] == "AnimalAttributesDataLoader":
            train_labels_old = train_data_loader.dataset.true_labels[train_data_loader.sampler.indices]
        else:
            train_labels_old = None

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

    # build model architecture, then print to console
    if args.is_vit:
        vit_config = CONFIGS[args.vit_type]
        model = config.init_obj('arch', module_arch, config = vit_config, img_size = args.img_size, zero_head=True)
        model.load_from(np.load(args.vit_pretrained_dir))
    else:
        model = config.init_obj('arch', module_arch)
    logger.info(model)

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
    
    if 'domain' in config['data_loader']['args']:
        domain_name = config['data_loader']['args']['domain']
    else:
        domain_name = None
    
    for run in range(args.runs):
        model.reset_parameters(source_state_dict)
        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        if args.train_ls:
            checkpoint_dir = os.path.join(
            "./saved_hessians", 
            "{}_{}_ls_{}".format(config["arch"]["type"], config["data_loader"]["type"], args.ls_alpha))
            trainer = LabelSmoothTrainer(model, criterion, metrics, optimizer,
                        config=config,
                        device=device,
                        train_data_loader=train_data_loader,
                        valid_data_loader=valid_data_loader,
                        test_data_loader=test_data_loader,
                        lr_scheduler=lr_scheduler,
                        checkpoint_dir=checkpoint_dir,
                        alpha=args.ls_alpha)
        elif args.train_mixup:
            checkpoint_dir = os.path.join(
            "./saved_hessians", 
            "{}_{}_mixup_{}".format(config["arch"]["type"], config["data_loader"]["type"],  args.mixup_alpha))
            trainer = MixupTrainer(model, criterion, metrics, optimizer,
                        config=config,
                        device=device,
                        train_data_loader=train_data_loader,
                        valid_data_loader=valid_data_loader,
                        test_data_loader=test_data_loader,
                        lr_scheduler=lr_scheduler,
                        checkpoint_dir=checkpoint_dir,
                        alpha=args.mixup_alpha)
        elif args.train_swa:
            checkpoint_dir = os.path.join(
            "./saved_hessians", 
            "{}_{}_swa_{}_{}".format(config["arch"]["type"], config["data_loader"]["type"], args.swa_epoch, args.swa_lr))
            trainer = SWATrainer(model, criterion, metrics, optimizer,
                        config=config,
                        device=device,
                        train_data_loader=train_data_loader,
                        valid_data_loader=valid_data_loader,
                        test_data_loader=test_data_loader,
                        lr_scheduler=lr_scheduler,
                        checkpoint_dir=checkpoint_dir,
                        swa_start=args.swa_epoch,
                        swa_lr=args.swa_lr)
        elif args.train_sam:
            checkpoint_dir = os.path.join(
            "./saved_revision", 
            "{}_{}_sam_{}_ada_{}_bs_{}".format(config["arch"]["type"], config["data_loader"]["type"], args.sam_rho, args.sam_adaptive, config["data_loader"]["args"]["batch_size"]) + \
                ("_unnormalize" if args.sam_unnormalize else "")
            )
            base_optimizer = getattr(torch.optim, config["optimizer"]["type"])
            optimizer = SAM(model.parameters(), base_optimizer, rho=args.sam_rho, 
                            adaptive=args.sam_adaptive, unnormalize=args.sam_unnormalize,
                            **dict(config["optimizer"]["args"]))
            lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer.base_optimizer)
            trainer = SAMTrainer(model, criterion, metrics, optimizer,
                            config=config,
                            device=device,
                            train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            test_data_loader=test_data_loader,
                            lr_scheduler=lr_scheduler,
                            checkpoint_dir=checkpoint_dir)
        elif args.train_rsam:
            checkpoint_dir = os.path.join(
            "./saved", 
            "{}_{}_rsam_{}_{}_{}".format(config["arch"]["type"], config["data_loader"]["type"], args.rsam_rho, args.rsam_sigma, args.rsam_lam))
            base_optimizer = getattr(torch.optim, config["optimizer"]["type"])
            optimizer = RSAM(model.parameters(), base_optimizer, rho=args.rsam_rho, 
                             sigma=args.rsam_sigma, lam=args.rsam_lam, **dict(config["optimizer"]["args"]))
            lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer.base_optimizer)
            trainer = RSAMTrainer(model, criterion, metrics, optimizer,
                            config=config,
                            device=device,
                            train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            test_data_loader=test_data_loader,
                            lr_scheduler=lr_scheduler,
                            checkpoint_dir=checkpoint_dir)
        elif args.train_bsam:
            checkpoint_dir = os.path.join(
            "./saved", 
            "{}_{}_bsam_{}_{}".format(config["arch"]["type"], config["data_loader"]["type"], args.bsam_rho, args.bsam_sigma))
            optimizer = BSAM(model.parameters(), rho=args.bsam_rho, sigma=args.bsam_sigma,
                             **dict(config["optimizer"]["args"]))
            lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
            trainer = BSAMTrainer(model, criterion, metrics, optimizer,
                            config=config,
                            device=device,
                            train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            test_data_loader=test_data_loader,
                            lr_scheduler=lr_scheduler,
                            checkpoint_dir=checkpoint_dir)
        elif args.train_nsm:
            checkpoint_dir = os.path.join(
                "./saved_revision",
                "{}_{}_nsm_{}_{}_{}_{}_distribution_{}_bs_{}".format(config["arch"]["type"], config["data_loader"]["type"], 
                                               args.nsm_lam, args.nsm_sigma, args.num_perturbs, args.use_neg, args.nsm_distribution, config["data_loader"]["args"]["batch_size"]))
            if config["optimizer"]["args"]["weight_decay"] != 0:
                checkpoint_dir = checkpoint_dir + "_wd_{}".format(config["optimizer"]["args"]["weight_decay"])
            if args.use_augmentation:
                checkpoint_dir = checkpoint_dir + "_aug"
            if config["reg_method"] == "penalty":
                checkpoint_dir = checkpoint_dir + "_penalty"
            base_optimizer = getattr(torch.optim, config["optimizer"]["type"])
            optimizer = NSM(model.parameters(), base_optimizer, sigma=args.nsm_sigma, distribution=args.nsm_distribution, **dict(config["optimizer"]["args"]))
            lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer.base_optimizer)
            trainer = NSMTrainer(model, criterion, metrics, optimizer,
                            config=config,
                            device=device,
                            train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            test_data_loader=test_data_loader,
                            lr_scheduler=lr_scheduler,
                            checkpoint_dir=checkpoint_dir,
                            nsm_lam=args.nsm_lam,
                            num_perturbs=args.num_perturbs,
                            use_neg=args.use_neg,
                            nsm_sigma=args.nsm_sigma,
                            nsm_sigma_schedule=args.nsm_sigma_schedule)

            if config["reg_method"] == "penalty":
                lambda_extractor = config["reg_extractor"]
                lambda_pred_head = config["reg_predictor"]
                scale_factor = config["scale_factor"]
                print(lambda_extractor, lambda_pred_head, scale_factor)
                trainer.add_penalty(
                    norm = config["reg_norm"], lambda_extractor = lambda_extractor, lambda_pred_head=lambda_pred_head, 
                    state_dict = source_state_dict, scale_factor=scale_factor
                )
        elif args.train_nsmswa:
            checkpoint_dir = os.path.join(
                "./saved",
                "{}_{}_nsm_{}_{}_{}_{}_swa_{}_{}".format(
                config["arch"]["type"], config["data_loader"]["type"], 
                args.nsm_lam, args.nsm_sigma, args.num_perturbs, args.use_neg,
                args.swa_epoch, args.swa_lr))
            base_optimizer = getattr(torch.optim, config["optimizer"]["type"])
            optimizer = NSM(model.parameters(), base_optimizer, sigma=args.nsm_sigma, **dict(config["optimizer"]["args"]))
            lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer.base_optimizer)
            trainer = NSMSWATrainer(model, criterion, metrics, optimizer,
                            config=config,
                            device=device,
                            train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            test_data_loader=test_data_loader,
                            lr_scheduler=lr_scheduler,
                            checkpoint_dir=checkpoint_dir,
                            nsm_lam=args.nsm_lam,
                            num_perturbs=args.num_perturbs,
                            use_neg=args.use_neg,
                            swa_start=args.swa_epoch,
                            swa_lr=args.swa_lr)
        else:
            checkpoint_dir = os.path.join(
            "./saved", 
            "{}_{}_{}_{}_{:.4f}_{:.4f}_run_{}".format(config["arch"]["type"], config["data_loader"]["type"], 
                                    config["reg_method"], config["reg_norm"],
                                    config["reg_extractor"], config["reg_predictor"], run))
            checkpoint_dir = checkpoint_dir + ("_synthethic_{}".format(args.noise_rate) if args.synthetic_noise else "")
            trainer = ConstraintTrainer(model, criterion, metrics, optimizer,
                            config=config,
                            device=device,
                            train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            test_data_loader=test_data_loader,
                            lr_scheduler=lr_scheduler,
                            checkpoint_dir = checkpoint_dir,
                            gradient_update_step=args.gradient_update_step)
                            
            lambda_extractor = config["reg_extractor"]
            lambda_pred_head = config["reg_predictor"]
            scale_factor = config["scale_factor"]
            print(lambda_extractor, lambda_pred_head, scale_factor)
            if config["reg_method"] == "constraint":
                trainer.add_constraint(
                    norm = config["reg_norm"], lambda_extractor = lambda_extractor, lambda_pred_head=lambda_pred_head, 
                    state_dict = source_state_dict, scale_factor=scale_factor
                )
            if config["reg_method"] == "penalty":
                trainer.add_penalty(
                    norm = config["reg_norm"], lambda_extractor = lambda_extractor, lambda_pred_head=lambda_pred_head, 
                    state_dict = source_state_dict, scale_factor=scale_factor
                )

        trainer.train()
        accuracies.append(trainer.test())
    logger.info("Test Accuracy {:1.4f} +/- {:1.4f}".format(np.mean(accuracies), np.std(accuracies)))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, nargs='+',
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--runs', type=int, default=3)
    args.add_argument('--data_frac', type=float, default=1.0)
    args.add_argument('--downsample_test', action="store_true")
    
    args.add_argument('--is_vit', action="store_true")
    args.add_argument('--img_size', type=int, default=224)
    args.add_argument("--vit_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    args.add_argument("--vit_pretrained_dir", type=str, default="checkpoints/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")

    args.add_argument('--synthetic_noise', action="store_true")
    args.add_argument('--noise_rate', type=float, default=0.0)

    args.add_argument('--constraint_reweight', action="store_true")
    args.add_argument('--load_matrix', action="store_true")
    args.add_argument('--reweight_epoch', type=int, default=1)
    args.add_argument('--reweight_noise_rate', type=float, default=0.8)

    args.add_argument('--gradient_update_step', type=int, default=1)

    args.add_argument('--train_ls', action="store_true")
    args.add_argument('--ls_alpha', type=float, default=0.15)

    args.add_argument('--train_mixup', action="store_true")
    args.add_argument('--mixup_alpha', type=float, default=0.2)

    args.add_argument('--train_swa', action="store_true")
    args.add_argument('--swa_epoch', type=int, default=20)
    args.add_argument('--swa_lr', type=float, default=0.005)

    args.add_argument('--train_sam', action="store_true")
    args.add_argument('--sam_rho', type=float, default=0.05)
    args.add_argument('--sam_adaptive', action="store_true")
    args.add_argument('--sam_unnormalize', action="store_true")

    args.add_argument('--train_rsam', action="store_true")
    args.add_argument('--rsam_rho', type=float, default=0.05)
    args.add_argument('--rsam_sigma', type=float, default=0.01)
    args.add_argument('--rsam_lam', type=float, default=1.0)

    args.add_argument('--train_bsam', action="store_true")
    args.add_argument('--bsam_rho', type=float, default=0.05)
    args.add_argument('--bsam_sigma', type=float, default=0.01)

    args.add_argument('--train_nsm', action="store_true")
    args.add_argument('--use_neg', action="store_true")
    args.add_argument('--nsm_sigma', type=float, default=0.01)
    args.add_argument('--num_perturbs', type=int, default=1)
    args.add_argument('--nsm_lam', type=float, default=0.5)
    args.add_argument('--nsm_distribution', type=str, default="normal")
    args.add_argument('--nsm_sigma_schedule', type=str, default="none")

    args.add_argument('--train_nsmswa', action="store_true")
    args.add_argument('--use_augmentation', action="store_true")

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--model'], type=str, target="arch;type"),
        CustomArgs(['--weight_decay'], type=float, target="optimizer;args;weight_decay"),
        CustomArgs(['--reg_method'], type=str, target='reg_method'),
        CustomArgs(['--reg_norm'], type=str, target='reg_norm'),
        CustomArgs(['--reg_extractor'], type=float, target='reg_extractor'),
        CustomArgs(['--reg_predictor'], type=float, target='reg_predictor'),
        CustomArgs(['--scale_factor'], type=float, target="scale_factor"),
        CustomArgs(['--domain'], type=str, target="data_loader;args;domain"),
        CustomArgs(['--sample'], type=int, target="data_loader;args;sample"),
        CustomArgs(['--early_stop'], type=int, target="trainer;early_stop"),
        CustomArgs(['--epochs'], type=int, target="trainer;epochs"),
    ]
    config, args = ConfigParser.from_args(args, options)
    print(config)
    main(config, args)
