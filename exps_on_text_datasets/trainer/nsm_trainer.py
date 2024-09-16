from trainer.glue_trainer import GLUETrainer
from utils import MetricTracker, prepare_inputs

class NSMTrainer(GLUETrainer):

    def __init__(self, model, metric, optimizer, lr_scheduler, config, device, 
        train_data_loader, valid_data_loader=None, test_data_loader=None, checkpoint_dir=None, criterion=None,
        num_perturbs = 1, nsm_lam=0.5, use_neg=False):
        super().__init__(model, metric, optimizer, lr_scheduler, config, device, 
        train_data_loader, valid_data_loader, test_data_loader, checkpoint_dir, criterion)
        self.num_perturbs = num_perturbs
        self.nsm_lam = nsm_lam
        self.use_neg = use_neg

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for step, batch in enumerate(self.train_data_loader):
            batch = prepare_inputs(batch, self.device)

            # first forward-backward step
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            self.optimizer.store_gradients(zero_grad=True, store_weights=True, update_weight=self.nsm_lam)

            # second forward-backward step
            if self.num_perturbs != 0:
                update_weight = (1-self.nsm_lam)/(2*self.num_perturbs) if self.use_neg else (1-self.nsm_lam)/(self.num_perturbs)
                for i in range(self.num_perturbs):
                    self.optimizer.first_step(zero_grad=True, store_perturb=True)
                    self.model(**batch).loss.backward()
                    self.optimizer.store_gradients(zero_grad=True, store_weights=False, update_weight=update_weight)
                    if self.use_neg:
                        self.optimizer.first_step(zero_grad=True, store_perturb=False)
                        self.model(**batch).loss.backward()
                        self.optimizer.store_gradients(zero_grad=True, store_weights=False, update_weight=update_weight)
            self.optimizer.second_step(zero_grad=True)
            self.completed_steps += 1

            # outputs = self.model(**batch)
            # loss = outputs.loss
            # loss = loss / self.cfg_trainer["gradient_accumulation_steps"]
            # loss.backward()
            # if step % self.cfg_trainer["gradient_accumulation_steps"] == 0 or step == len(self.train_data_loader) - 1:
            #     self.optimizer.step()
            #     self.lr_scheduler.step()
            #     self.optimizer.zero_grad()
            #     self.completed_steps += 1

            if self.completed_steps >= self.cfg_trainer["max_train_steps"]:
                break

            # update training metrics
            self.train_metrics.update('loss', loss.item())
            
            predictions = outputs.logits.argmax(dim=-1)
            self.metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )

        log = self.train_metrics.result()
        train_metrics = self.metric.compute()
        log.update(**train_metrics)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        return log