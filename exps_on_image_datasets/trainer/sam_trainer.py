from trainer.base_trainer import Trainer
from utils.bypass_bn import enable_running_stats, disable_running_stats


class SAMTrainer(Trainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, 
        train_data_loader, valid_data_loader=None, test_data_loader=None, lr_scheduler=None, checkpoint_dir=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config, device, 
        train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir)

    
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

            # first forward-backward step
            enable_running_stats(self.model)
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(self.model)
            self.criterion(self.model(data), target).backward()
            self.optimizer.second_step(zero_grad=True)

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