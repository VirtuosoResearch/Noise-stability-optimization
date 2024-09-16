import copy
import json
import logging
from typing import List, Dict
import wandb
from collections import defaultdict
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
# from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import PreTrainedTokenizerBase
import numpy as np
from utils.sam import SAM
import os

class AlpacaModel(pl.LightningModule):
    validation_predictions: Dict

    def __init__(self, model, tokenizer: PreTrainedTokenizerBase, model_type: str, use_cpu_offload=False,
                lr=3e-4, truncate_early=True, max_length=1024, weight_decay=1e-4, use_wandb=False,
                intialize_project_matrix=False, run_seed = 0, project_dim = 200, gradient_dir = "test", predict_steps = 2000, use_sgd=True,
                train_sam = False, sam_rho = 0.05, sam_adaptive = False, sam_unnormalize = False):
        """
        - completion_metadata: metaddata used to save completions. If None, completions are not saved.
          `epoch_N` is appended to the `train_key` when saving intermediate validation completions.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.use_cpu_offload = use_cpu_offload
        self.lr = lr
        self.max_length = max_length
        self.truncate_early = truncate_early
        self.weight_decay = weight_decay
        self.use_wandb = use_wandb
        self.validation_step_outputs = []
        gradient_dim = 0
        self.use_sgd = use_sgd

        self.train_sam = train_sam
        self.sam_rho = sam_rho
        self.sam_adaptive = sam_adaptive
        self.sam_unnormalize = sam_unnormalize

        if self.train_sam:
            self.automatic_optimization = False
        
        self.removing_keys = ["shared", "lm_head", "wte", "wpe", "ln", "embed_tokens", "norm", "word_embeddings" ]
        for name, param in model.named_parameters():
            if any([key in name for key in self.removing_keys]):
                continue
            if param.requires_grad:
                gradient_dim += param.numel()
        print("Creating project matrix with dimensions: ", gradient_dim, project_dim)
        if intialize_project_matrix:
            np.random.seed(run_seed)
            self.project_matrix = (2 * np.random.randint(2, size=(gradient_dim, project_dim)) - 1).astype(float)
            self.project_matrix *= 1 / np.sqrt(project_dim)
            self.gradient_dir = f"./gradients/{gradient_dir}"
            if not os.path.exists(self.gradient_dir):
                os.makedirs(self.gradient_dir)
            self.param_names = [name for name, param in model.named_parameters() if param.requires_grad]
        self.predict_steps = predict_steps

    def get_trainable_parameters(self):
        return [param for name, param in self.model.named_parameters()\
                if (name in self.param_names) and (not any([key in name for key in self.removing_keys]))]

    def on_validation_end(self) -> None:
        if not self.automatic_optimization:
            # Save a checkpoint of the model
            ckpt_path = os.path.join(self.trainer.log_dir, 'checkpoints', 'ckpt.pt')
            self.trainer.save_checkpoint(ckpt_path)
        return super().on_validation_end()

    def training_step(self, batch, batch_idx):
        kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        if self.model_type == "encoder_decoder":
            kwargs["decoder_attention_mask"] = batch["decoder_attention_mask"]
        if self.train_sam:
            optimizer = self.optimizers()
            # `optimizer` is a `LightningOptimizer` wrapping the optimizer.
            # To access it, do the following.
            # However, it won't work on TPU, AMP, etc...
            sam_optimizer = optimizer.optimizer

            # first forward-backward pass
            loss_1 = self.model(**kwargs)["loss"]
            self.manual_backward(loss_1)
            sam_optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            loss_2 = self.model(**kwargs)["loss"]
            self.manual_backward(loss_2)
            sam_optimizer.second_step(zero_grad=True)
        else:
            loss = self.model(**kwargs)["loss"]
            if self.use_wandb:
                wandb.log({"train_loss": loss})
            return loss

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Returns outputs in dictionary format, since it's the only way that seems to work with `all_gather`
        """        
        input_ids = batch['input_ids']
        labels = batch["labels"]

        outputs = self.model(input_ids, labels=labels)
        lm_logits = outputs.logits 
        assert self.model_type == "decoder"
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        losses = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")

        context_length = self.max_length - 1
        losses = losses.view(-1, context_length)
        keep = losses != 0
        losses = (losses).sum(dim = 1) / keep.sum(dim = 1)

        output_dict = {
            "skills": batch['skill'],
            "losses": losses,
        }
        self.validation_step_outputs.append(output_dict)
        return output_dict
    
    def on_validation_epoch_end(self) -> None:
        """
        Gather outputs from all GPUs and save validation predictions as a CompletionDataset and
        log validation metrics.

        Note, `all_gather` *concatenates* tensors from all GPUs along the first dimension.
        """
        outputs = self.validation_step_outputs
        loss_dict = defaultdict(list)
        for batch in outputs:
            skills = batch["skills"]
            losses = batch["losses"]
            for j, skill in enumerate(skills):
                loss_dict[skill].append(losses[j])   

        summary = {}
        for skill, losses in loss_dict.items():
            summary[f"loss_{skill}"] = torch.stack(losses).mean().item()
        summary["loss"] = torch.cat([torch.stack(losses) for losses in loss_dict.values()]).mean().item()

        # Log metrics
        logging.info(summary)
        if summary:
            for key, value in summary.items():
                if key == "accuracy":
                    self.log(key, value, prog_bar=True, logger=True)
                else:
                    if self.use_wandb:
                        wandb.log({f"val_{key}": value})
                    self.log(key, value, prog_bar=False, logger=True)
        self.validation_step_outputs.clear()
        return summary

    def forward(self, batch, batch_idx):
        kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        outputs = self.model(**kwargs)
        return outputs

    def predict_step(self, batch, batch_idx):
        # collect the gradients and project them to the embedding space
        torch.set_grad_enabled(True)

        kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        model_outputs = self.model(**kwargs)
        logits = model_outputs.logits
        
        labels = kwargs["labels"]
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        gradients = []; outputs = np.zeros(labels.shape)

        if self.predict_steps < batch_idx:
            torch.set_grad_enabled(False)
            return outputs

        for i in range(len(labels)):
            tmp_mask = labels[i] != -100
            tmp_logits = logits[i][tmp_mask]
            tmp_probs = torch.softmax(tmp_logits, dim=-1)
            tmp_labels = labels[i][tmp_mask]

            tmp_outputs = tmp_probs[range(tmp_probs.size(0)), tmp_labels]
            tmp_outputs[tmp_outputs>0.9] -= 1e-3 # in case (1-tmp_outputs) is less than zero            
            tmp_outputs = torch.log(tmp_outputs/(1-tmp_outputs))
            tmp_loss = tmp_outputs.mean()

            tmp_gradients = torch.autograd.grad(tmp_loss, self.get_trainable_parameters(), retain_graph=True, create_graph=False)
            tmp_gradients = torch.cat([gradient.reshape(-1) for gradient in tmp_gradients]).cpu().type(torch.float32).numpy() # flatten gradients
            tmp_gradients = (tmp_gradients.reshape(1, -1) @ self.project_matrix).flatten()
            gradients.append(tmp_gradients)
            outputs[i, :tmp_outputs.size(0)] = tmp_outputs.clone().detach().cpu().type(torch.float32).numpy()
        gradients = np.array(gradients); logits.detach()
        np.save(f"{self.gradient_dir}/train_batch_{batch_idx}_gradients.npy", gradients)
        return outputs

    def configure_optimizers(self):
        if self.use_cpu_offload:
            optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.lr)
        elif self.train_sam:
            base_optimizer = torch.optim.AdamW
            optimizer = SAM(self.parameters(), base_optimizer, rho=self.sam_rho, adaptive=self.sam_adaptive, unnormalize=self.sam_unnormalize, lr=self.lr, weight_decay=self.weight_decay)
        elif self.use_sgd:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            
        return optimizer