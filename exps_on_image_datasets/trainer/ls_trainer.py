import torch
import os
import numpy as np
from torch._C import dtype
from .base_trainer import Trainer
import torch.nn.functional as F

class LabelSmoothTrainer(Trainer):
    
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, 
        train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir, alpha = 0):
        super().__init__(model, criterion, metric_ftns, optimizer, config, device, 
            train_data_loader, valid_data_loader=valid_data_loader, test_data_loader=test_data_loader, 
            lr_scheduler=lr_scheduler, checkpoint_dir=checkpoint_dir)
        self.alpha = alpha
        num_classes = config["arch"]["args"]["n_classes"]
        self.smoothed_label = torch.ones(1, num_classes, dtype=torch.float).to(device) 
        self.smoothed_label = self.smoothed_label / num_classes

    def _label_smooth_loss(self, output, target, index):
        ce_loss = F.nll_loss(output, target)
        smooth_loss = torch.sum(- output * self.smoothed_label, dim=1).mean()
        loss = ce_loss * (1-self.alpha) + smooth_loss * self.alpha
        return loss

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target, index) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)
            index = index.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            
            loss = self._label_smooth_loss(output, target, index)
            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log