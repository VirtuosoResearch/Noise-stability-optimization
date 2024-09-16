"""
LightningModule for training encoder OR encoder_decoder models which provides:
- Saving intermediate validation predictions as CompletionDataset
- Logging intermediate validation metrics (from the CompletionDataset)
"""
import copy
import json
import logging
from typing import List, Dict

import pytorch_lightning as pl
import torch
# from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import PreTrainedTokenizerBase

from data.completion_dataset import CompletionDataset, CompletionMetadata
from data.dataset import Dataset
from evaluation.evaluator import Evaluator
from evaluation.summary import summarize_evaluation
from utils.sam import SAM
from utils.nsm import NSM
import os

class Model(pl.LightningModule):
    validation_predictions: Dict

    def __init__(self, model, tokenizer: PreTrainedTokenizerBase, model_type: str, use_cpu_offload=False,
                 completion_metadata: CompletionMetadata = None, lr=3e-4, truncate_early=True, max_length=1024, weight_decay=1e-4,
                 train_sam = False, sam_rho = 0.05, sam_adaptive = False, sam_unnormalize = False,
                 train_nsm=False, nsm_use_neg=False, nsm_sigma=0.01, nsm_num_perturbs=1, nsm_lam = 0):
        """
        - completion_metadata: metaddata used to save completions. If None, completions are not saved.
          `epoch_N` is appended to the `train_key` when saving intermediate validation completions.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.use_cpu_offload = use_cpu_offload
        self.completion_metadata = completion_metadata
        self.lr = lr
        self.max_length = max_length
        self.truncate_early = truncate_early
        self.weight_decay = weight_decay
        self.validation_step_outputs = []
        self.train_sam = train_sam
        self.sam_rho = sam_rho
        self.sam_adaptive = sam_adaptive
        self.sam_unnormalize = sam_unnormalize

        self.train_nsm = train_nsm
        self.nsm_use_neg = nsm_use_neg
        self.nsm_sigma = nsm_sigma
        self.nsm_num_perturbs = nsm_num_perturbs
        self.nsm_lam = nsm_lam

        if self.train_sam or self.train_nsm:
            self.automatic_optimization = False

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
        elif self.train_nsm:
            optimizer = self.optimizers()
            # `optimizer` is a `LightningOptimizer` wrapping the optimizer.
            # To access it, do the following.
            # However, it won't work on TPU, AMP, etc...
            nsm_optimizer = optimizer.optimizer

            loss = self.model(**kwargs)["loss"]
            self.manual_backward(loss)
            nsm_optimizer.store_gradients(zero_grad=True, store_weights=True, update_weight=self.nsm_lam)

            # second forward-backward step
            if self.nsm_num_perturbs != 0:
                update_weight = (1-self.nsm_lam)/(2*self.nsm_num_perturbs) if self.nsm_use_neg else (1-self.nsm_lam)/(self.nsm_num_perturbs)
                for i in range(self.nsm_num_perturbs):
                    nsm_optimizer.first_step(zero_grad=True, store_perturb=True)
                    loss = self.model(**kwargs)["loss"]
                    self.manual_backward(loss)
                    nsm_optimizer.store_gradients(zero_grad=True, store_weights=False, update_weight=update_weight)
                    if self.nsm_use_neg:
                        nsm_optimizer.first_step(zero_grad=True, store_perturb=False)
                        loss = self.model(**kwargs)["loss"]
                        self.manual_backward(loss)
                        nsm_optimizer.store_gradients(zero_grad=True, store_weights=False, update_weight=update_weight)
            nsm_optimizer.second_step(zero_grad=True)
        else:
            return self.model(**kwargs)["loss"]

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Returns outputs in dictionary format, since it's the only way that seems to work with `all_gather`
        """
        if self.current_epoch < 2 and self.truncate_early:
            max_length = 256
        else:
            max_length = self.max_length

        if self.model_type == "encoder_decoder":
            output = self.model.generate(batch["input_ids"], max_length=max_length).detach()
        elif self.model_type == "decoder":
            output = self.model.generate(batch["input_ids"], max_length=max_length,
                                         pad_token_id=self.tokenizer.pad_token_id,
                                         eos_token_id=self.tokenizer.eos_token_id).detach()
        else:
            raise NotImplementedError("model_type='{}' not supported".format(self.model_type))

        output_dict = {
            "sample_index": batch["sample_index"],
            "input": batch["input_ids"],
            "output": output,
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
        # Determine total sample count and local max input/output length
        local_max_output_length = 0
        local_max_input_length = 0
        total_samples = 0
        for batch in outputs:
            local_max_input_length = max(local_max_input_length, batch["input"].shape[-1])
            local_max_output_length = max(local_max_output_length, batch["output"].shape[-1])
            total_samples += batch["sample_index"].shape[0]

        # Determine global max input/output length
        max_input_length = self.all_gather(torch.tensor(local_max_input_length, dtype=torch.long)).max()
        max_output_length = self.all_gather(torch.tensor(local_max_output_length, dtype=torch.long)).max()

        # Create local padded tensors
        local_outputs: dict = {
            "sample_index": torch.ones((total_samples,), dtype=torch.long) * self.tokenizer.pad_token_id,
            "input": torch.ones((total_samples, max_input_length), dtype=torch.long) * self.tokenizer.pad_token_id,
            "output": torch.ones((total_samples, max_output_length), dtype=torch.long) * self.tokenizer.pad_token_id,
        }

        # Populate local tensors
        start_index = 0
        for i, batch in enumerate(outputs):
            batch_size = batch["sample_index"].shape[0]
            end_index = start_index + batch_size
            local_outputs["sample_index"][start_index:end_index] = batch["sample_index"]
            input_width = batch["input"].shape[-1]
            output_width = batch["output"].shape[-1]
            if self.model_type == "encoder_decoder":
                local_outputs["input"][start_index:end_index, :input_width] = batch["input"]
                local_outputs["output"][start_index:end_index, :output_width] = batch["output"]
            elif self.model_type == "decoder":
                output_only_width = output_width - input_width
                local_outputs["input"][start_index:end_index, :input_width] = batch["input"]
                local_outputs["output"][start_index:end_index, :output_only_width] = batch["output"][:, input_width:]
            else:
                raise NotImplementedError("model_type='{}' not supported".format(self.model_type))

            start_index = end_index

        global_outputs = self.all_gather(local_outputs)
        if self.global_rank == 0:
            if global_outputs["sample_index"].dim() == 2:  # world_size > 1
                global_outputs["sample_index"] = global_outputs["sample_index"].flatten(start_dim=0, end_dim=1)
                global_outputs["output"] = global_outputs["output"].flatten(start_dim=0, end_dim=1)
                global_outputs["input"] = global_outputs["input"].flatten(start_dim=0, end_dim=1)

            final_output = {
                "sample_index": global_outputs["sample_index"].tolist(),
                "input": self.tokenizer.batch_decode(global_outputs["input"], skip_special_tokens=True),
                "output": self.tokenizer.batch_decode(global_outputs["output"], skip_special_tokens=True),
            }

            if self.completion_metadata is not None:
                # Save outputs as CompletionDataset
                cd = self._generate_completion_dataset(self.completion_metadata, final_output, epoch=self.current_epoch)
                cd.save()

                # Log validation examples
                examples = []
                for i in cd.indices[:5]:
                    examples.append(cd[i])
                logging.info("VALIDATION_EXAMPLES".center(80, "-"))
                logging.info(json.dumps(examples, indent=4))

                # Log metrics
                evaluation = Evaluator.evaluate_completion_dataset(cd)
                summary = summarize_evaluation(evaluation)
                logging.info(summary)
                if summary:
                    for key, value in summary.items():
                        if key == "accuracy":
                            self.log(key, value, prog_bar=True, logger=True)
                        else:
                            self.log(key, value, prog_bar=False, logger=True)

        self.validation_step_outputs.clear()
        return summary

    def configure_optimizers(self):
        if self.use_cpu_offload:
            optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.lr)
        elif self.train_sam:
            base_optimizer = torch.optim.AdamW
            optimizer = SAM(self.parameters(), base_optimizer, rho=self.sam_rho, adaptive=self.sam_adaptive, unnormalize=self.sam_unnormalize, lr=self.lr, weight_decay=self.weight_decay)
        elif self.train_nsm:
            base_optimizer = torch.optim.AdamW
            optimizer = NSM(self.parameters(), base_optimizer, sigma=self.nsm_sigma, distribution="normal", lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    @staticmethod
    def _generate_completion_dataset(completion_metadata, output: Dict[str, List], epoch=None,
                                     completions_per_sample=1) -> CompletionDataset:
        """
        Initialize and populate a CompletionDataset from model output.

        - output: {
            sample_index: List[int],
            input: List[str],
            output: List[str],
          }
        - completions_per_sample: limit the number of completions used per sample. This is useful when model output
          is obtained from distributed inference, where some samples may be duplicated to match batch sizes. Will use
          all completions if None. Existing completions count towards the limit.
        """
        if completions_per_sample is not None and completions_per_sample < 1:
            raise ValueError("completions_per_sample must be at least 1")

        # Add/assign epoch to train key of completion_identifier
        completion_metadata = copy.deepcopy(completion_metadata)
        if epoch is not None:
            completion_metadata.epoch = epoch

        # Initialize completion dataset
        cd = CompletionDataset.init(completion_metadata)

        # Populate completion dataset with model output
        dataset = Dataset.load(cd.dataset_key)
        for sample_index, input, output in zip(output["sample_index"], output["input"], output["output"]):
            if len(dataset) <= sample_index:
                raise KeyError(
                    "Sample index {} not found in dataset {}".format(sample_index, cd.dataset_key))

            if sample_index in cd.data:
                completions = cd.data[sample_index]
            else:
                completions = list()
                cd.data[sample_index] = completions

            completion_index = len(completions)
            if completions_per_sample is None or completion_index < completions_per_sample:
                completions.append({
                    "sample_index": sample_index,
                    "completion_index": completion_index,
                    "question": dataset[sample_index]["question"],
                    "answer": dataset[sample_index]["answer"],
                    "prompt": input,
                    "completion": output,
                })
            cd.data[sample_index] = completions

        return cd