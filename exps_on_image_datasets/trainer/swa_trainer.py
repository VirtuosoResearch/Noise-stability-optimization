import torch
import torch.nn.functional as F
import os
from .base_trainer import Trainer
from torch.optim.swa_utils import update_bn

class SWATrainer(Trainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, 
                train_data_loader, valid_data_loader=None, test_data_loader=None, lr_scheduler=None, checkpoint_dir=None,
                swa_start = 20, swa_lr = 0.0005):
        super().__init__(model, criterion, metric_ftns, optimizer, config, device, 
                train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir)
        
        self.swa_model = AveragedModel(self.model)
        self.swa_start = swa_start
        self.swa_lr = swa_lr
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.swa_lr, anneal_epochs=5)
        self.save_epoch = 1

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, batch in enumerate(self.train_data_loader):
            if len(batch) == 3:
                data, target, index = batch
            else:
                data, target = batch
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
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

        if epoch >= self.swa_start:
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()
        else:
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        return log
    
    def test(self):
        update_bn(self.train_data_loader, self.swa_model, device=self.device)

        self.swa_model.eval()

        total_loss = 0.0
        total_metrics = torch.zeros(len(self.metric_ftns))

        with torch.no_grad():
            for i, batch in enumerate(self.test_data_loader):
                if len(batch) == 3:
                    data, target, index = batch
                else:
                    data, target = batch
                data, target = data.to(self.device), target.to(self.device)
                output = self.swa_model(data)

                # computing loss, metrics on test set
                loss = self.criterion(output, target)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(self.metric_ftns):
                    total_metrics[i] += metric(output, target) * batch_size

        n_samples = len(self.test_data_loader.sampler)
        log = {'loss': total_loss / n_samples}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(self.metric_ftns)
        })
        self.logger.info(log)

        return log[self.metric_ftns[0].__name__]
