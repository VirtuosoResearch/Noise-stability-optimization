"""
Run custom fine-tuning based experiments, i.e., fine-tuning models such as T5, and GPT-2 on GPUs.

Note, to check distributed errors used `TORCH_DISTRIBUTED_DEBUG=DETAIL`
Note, if deepspeed hangs at initialization, use `NCCL_P2P_DISABLE=1`. Thought, this seems to slow down the training a lot...
Note, to see more NCCL errors, use NCCL_DEBUG=WARN
"""
import argparse
import logging
import os

from src.custom.data_module import DataModule
from src.data.completion_dataset import CompletionMetadata

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pytorch_lightning as pl
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.custom.model import Model
from peft import get_peft_model, LoraConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from evaluation.evaluator import Evaluator
from evaluation.summary import summarize_evaluation
import pandas as pd

logging.basicConfig(level=logging.INFO)

torch.set_float32_matmul_precision("high")
import time

def evaluate(outputs, model, tokenizer):
    """
    Gather outputs from all GPUs and save validation predictions as a CompletionDataset and
    log validation metrics.

    Note, `all_gather` *concatenates* tensors from all GPUs along the first dimension.
    """
    # Determine total sample count and local max input/output length
    local_max_output_length = 0
    local_max_input_length = 0
    total_samples = 0
    for batch in outputs:
        local_max_input_length = max(local_max_input_length, batch["input"].shape[-1])
        local_max_output_length = max(local_max_output_length, batch["output"].shape[-1])
        total_samples += batch["sample_index"].shape[0]

    max_input_length = local_max_input_length
    max_output_length = local_max_output_length
    # Create local padded tensors
    local_outputs: dict = {
        "sample_index": torch.ones((total_samples,), dtype=torch.long) * tokenizer.pad_token_id,
        "input": torch.ones((total_samples, max_input_length), dtype=torch.long) * tokenizer.pad_token_id,
        "output": torch.ones((total_samples, max_output_length), dtype=torch.long) * tokenizer.pad_token_id,
    }

    # Populate local tensors
    start_index = 0
    for i, batch in enumerate(outputs):
        batch_size = batch["sample_index"].shape[0]
        end_index = start_index + batch_size
        local_outputs["sample_index"][start_index:end_index] = batch["sample_index"]
        input_width = batch["input"].shape[-1]
        output_width = batch["output"].shape[-1]
        if model.model_type == "encoder_decoder":
            local_outputs["input"][start_index:end_index, :input_width] = batch["input"]
            local_outputs["output"][start_index:end_index, :output_width] = batch["output"]
        elif model.model_type == "decoder":
            output_only_width = output_width - input_width
            local_outputs["input"][start_index:end_index, :input_width] = batch["input"]
            local_outputs["output"][start_index:end_index, :output_only_width] = batch["output"][:, input_width:]
        else:
            raise NotImplementedError("model_type='{}' not supported".format(model.model_type))

        start_index = end_index

    global_outputs = local_outputs
    if model.global_rank == 0:
        if global_outputs["sample_index"].dim() == 2:  # world_size > 1
            global_outputs["sample_index"] = global_outputs["sample_index"].flatten(start_dim=0, end_dim=1)
            global_outputs["output"] = global_outputs["output"].flatten(start_dim=0, end_dim=1)
            global_outputs["input"] = global_outputs["input"].flatten(start_dim=0, end_dim=1)

        final_output = {
            "sample_index": global_outputs["sample_index"].tolist(),
            "input": tokenizer.batch_decode(global_outputs["input"], skip_special_tokens=True),
            "output": tokenizer.batch_decode(global_outputs["output"], skip_special_tokens=True),
        }

        assert model.completion_metadata is not None
        # Save outputs as CompletionDataset
        cd = model._generate_completion_dataset(model.completion_metadata, final_output)
        cd.save()

        # Log metrics
        evaluation = Evaluator.evaluate_completion_dataset(cd)
        summary = summarize_evaluation(evaluation)
    return summary

def add_result_to_csv(result_datapoint, file_name):
    for key, val in result_datapoint.items():
        result_datapoint[key] = [val, ]
    
    if os.path.exists(file_name):
        result_df = pd.read_csv(file_name, index_col=0)
        tmp_df = pd.DataFrame(result_datapoint)
        result_df = pd.concat([result_df, tmp_df], ignore_index = True)
        result_df.to_csv(file_name)
    else:
        result_df = pd.DataFrame(result_datapoint)  
        result_df.to_csv(file_name)   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_key", type=str, default="multiarith")
    parser.add_argument("--model_key", type=str, default="t5_base")
    parser.add_argument("--train_key", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--preset_key", type=str, default="ft_cot")
    parser.add_argument("--inference_batch_size", type=int, default=None)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--disable_checkpointing", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)

    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--feature_transfer", action="store_true")

    parser.add_argument("--data_index_dir", type=str, default=None)
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--runs", type=int, default=3)

    parser.add_argument("--load_model_dir", type=str, default="test")

    parser.add_argument('--train_sam', action="store_true")
    parser.add_argument('--sam_rho', type=float, default=0.05)
    parser.add_argument('--sam_adaptive', action="store_true")
    parser.add_argument('--sam_unnormalize', action="store_true")

    parser.add_argument('--train_nsm', action="store_true")
    parser.add_argument('--nsm_use_neg', action="store_true")
    parser.add_argument('--nsm_sigma', type=float, default=0.01)
    parser.add_argument('--nsm_num_perturbs', type=int, default=1)
    parser.add_argument('--nsm_lam', type=float, default=0.5)

    # for writing results
    parser.add_argument("--write_results", action="store_true")
    parser.add_argument("--subset_idxes", type=int, nargs="+", default=None)
    args = parser.parse_args()
    args.enable_checkpointing = not args.disable_checkpointing
    print("arguments".upper().center(80, "-"))
    print(args)
    print("-" * 80)

    time_start = time.time()

    dataset_key = args.dataset_key
    model_key = args.model_key
    train_key = args.train_key

    model_name = args.model_key.replace("/", "_")
    save_name = f"{args.dataset_key}_{model_name}_{args.preset_key}" + \
                (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                (f"_{args.save_name}" if args.save_name else "")
    file_dir = os.path.join("./results/", save_name)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    metrics = {}
    for run in range(args.runs):
        if "flan" in model_key:
            hf_key = "google/{}".format(model_key.replace("_", "-"))
            model = AutoModelForSeq2SeqLM.from_pretrained(hf_key)
            tokenizer = AutoTokenizer.from_pretrained(hf_key, model_max_length=512)
            model_type = "encoder_decoder"
            append_eos = False  # t5 tokenizers already append eos
        elif "t5" in model_key:
            hf_key = model_key.replace("_", "-")
            model = T5ForConditionalGeneration.from_pretrained(hf_key)
            tokenizer = T5TokenizerFast.from_pretrained(hf_key, model_max_length=512)
            model_type = "encoder_decoder"
            append_eos = False
        elif "gpt2" in model_key:
            from transformers import GPT2Tokenizer, GPT2LMHeadModel

            hf_key = model_key.replace("_", "-")
            tokenizer = GPT2Tokenizer.from_pretrained(hf_key)
            model = GPT2LMHeadModel.from_pretrained(hf_key)
            model_type = "decoder"
            append_eos = True
        elif "Llama" in model_key:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            hf_key = model_key
            tokenizer = AutoTokenizer.from_pretrained(hf_key)
            model = AutoModelForCausalLM.from_pretrained(hf_key)
            model_type = "decoder"
            append_eos = True
        else:
            raise NotImplementedError(model_key)
        
        if args.train_lora:
            if "gpt" in model_key or "Llama" in model_key:
                config = LoraConfig(
                    r=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    target_modules=["q_proj", "k_proj", "v_proj"],
                    lora_dropout=0.1,
                    bias="lora_only",
                    modules_to_save=[],
                )
                model = get_peft_model(model, config)
                model.print_trainable_parameters()
            else:
                config = LoraConfig(
                    r=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    target_modules=["q", "k", "v"],
                    lora_dropout=0.1,
                    bias="lora_only",
                    modules_to_save=[],
                )
                model = get_peft_model(model, config)
                model.print_trainable_parameters()
        
        if args.feature_transfer:
            for name, param in model.named_parameters():
                if "lm_head" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                print(name, param.requires_grad)

        if "ft_cot" in args.preset_key:
            completion_key = "ft_cot"
        elif args.preset_key == "ft":
            completion_key = "ft"
        elif args.preset_key == "fs_cot":
            raise NotImplementedError("We don't train models on fs_cot")
        else:
            raise NotImplementedError(args.preset_key)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        batch_size = args.batch_size
        if args.inference_batch_size is None:
            inference_batch_size = batch_size
        else:
            inference_batch_size = args.inference_batch_size
        data_module = DataModule(dataset_key, args.preset_key, tokenizer, model_type, batch_size=batch_size,
                                inference_batch_size=inference_batch_size, num_workers=8, append_eos=append_eos,
                                data_index_dir=args.data_index_dir)

        cm = CompletionMetadata(model_key, completion_key, dataset_key, data_module.finetune_key,
                                data_module.prediction_template, train_key=args.train_key,
                                train_lora=args.train_lora, lora_rank=args.lora_rank)
        use_cpu_offload = args.strategy and "offload" in args.strategy
        lm = Model(model, tokenizer, model_type, use_cpu_offload=use_cpu_offload, completion_metadata=cm, lr=args.lr, weight_decay=args.weight_decay,
                    train_sam=args.train_sam, sam_rho=args.sam_rho, sam_adaptive=args.sam_adaptive, sam_unnormalize=args.sam_unnormalize,
                    train_nsm=args.train_nsm, nsm_use_neg=args.nsm_use_neg, nsm_sigma=args.nsm_sigma, nsm_num_perturbs=args.nsm_num_perturbs, nsm_lam=args.nsm_lam)
        
        load_model_dir = args.load_model_dir
        load_model_dir = os.path.join("external_lightning_logs", load_model_dir)
        if load_model_dir is not None and os.path.exists(load_model_dir + ".ckpt"):
            lm = lm.load_from_checkpoint(load_model_dir + ".ckpt", model=model, tokenizer=tokenizer, model_type=model_type)
            logging.info(f"Loaded model from {load_model_dir}")
        lm.completion_metadata = cm

        if not os.path.exists("external_lightning_logs"):
            raise Exception("external_lightning_logs/ does not exist")
        default_root_dir = os.path.join("external_lightning_logs", 
                                        "{}_{}_{}".format(model_name, args.dataset_key, args.preset_key) + \
                                            (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                                            (f"_feature_transfer" if args.feature_transfer else "") + \
                                            (f"_{args.save_name}" if args.save_name else "") + \
                                            (f"_sam_rho_{args.sam_rho}" if args.train_sam else "") + \
                                            (f"_nsm_sigma_{args.nsm_sigma}" if args.train_nsm else "") 
                                        )
        # remove previous checkpoints
        if args.save_name and os.path.exists(default_root_dir):
            os.system(f"rm -rf {default_root_dir}")
        
        checkpoint_callback = ModelCheckpoint(
            monitor="accuracy",
            dirpath=default_root_dir,
            filename="epoch_{epoch}",
            # every_n_epochs=1,
            save_top_k = -1,
            mode="max",
        )
        trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=args.strategy,
                            default_root_dir=default_root_dir, min_epochs=args.epochs, max_epochs=args.epochs,
                            accumulate_grad_batches=args.accumulate, precision=args.precision,
                            enable_checkpointing=args.enable_checkpointing,
                            callbacks=[checkpoint_callback]
                            )

        trainer.fit(lm, datamodule=data_module)

        # evaluate the best checkpoint
        if args.epochs > 0:
            # lm = lm.load_from_checkpoint(checkpoint_callback.best_model_path, 
            #                             model=model, tokenizer=tokenizer, 
            #                             model_type=model_type, completion_metadata=cm)
            # test_loader = data_module.test_dataloader()
            # outputs = []; lm.model.eval()
            # for batch_idx, batch in enumerate(test_loader):
            #     batch = {k: v.to(lm.device) for k, v in batch.items()}
            #     batch_output = lm.validation_step(batch, batch_idx)
            #     outputs.append(batch_output)
            # summary = evaluate(outputs, lm, tokenizer)
            summary = trainer.validate(lm, datamodule=data_module,  ckpt_path=checkpoint_callback.best_model_path)[0]
            logging.info(summary)
        else:
            summary = trainer.validate(lm, datamodule=data_module)[0]
            logging.info(summary)

        # save indexes 
        if args.write_results and run == 0:
            subset_idxes = args.subset_idxes
            result_datapoint = {
                "Data indices": " ".join([str(idx) for idx in subset_idxes])
            }
            for key, val in summary.items():
                result_datapoint[key] = val
            file_name = os.path.join(file_dir, "results.csv")
            add_result_to_csv(result_datapoint, file_name)
            
        for key in summary:
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(summary[key])
    
    for key in metrics:
        logging.info(f"{key}: {np.mean(metrics[key])} +/- {np.std(metrics[key])}")

    time_end = time.time()
    logging.info(f"Total time: {time_end - time_start}")