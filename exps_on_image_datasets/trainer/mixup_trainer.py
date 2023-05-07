import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from .base_trainer import Trainer

class MixupTrainer(Trainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, 
        train_data_loader, valid_data_loader=None, test_data_loader=None, lr_scheduler=None, checkpoint_dir=None,
        alpha = 0.2):
        super().__init__(model, criterion, metric_ftns, optimizer, config, device, 
        train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir)
        self.alpha = alpha

    def mixup_data(self, x, y, alpha=1.0):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

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
            data, targets_a, targets_b, lam = self.mixup_data(data, target, self.alpha)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.mixup_criterion(self.criterion, output, targets_a, targets_b, lam)
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