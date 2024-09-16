### Overview

We provide the implementation of a noise stability optimization algorithm in order to find flat minimizers. The algorithm adds random perturbations to the model weights and computes gradients from the perturbed weights to conduct gradient descent. We evaluate the Hessian trace and largest eigenvalue to validate the improved sharpness by our algorithm.

### Usage

The main implementation of the algorithm is the `NSO` optimizer as in `./exps_on_image_datasets/utils/nso.py`. We provide a simple code structure to use the algorithm in the following: 

```python
from utils.nso import NSO
from utils.bypass_bn import enable_running_stats, disable_running_stats
...

class args:
  # Smoothed SGD parameters
  nsm_sigma = 0.01 # standard deviation sigma of the isotropic Gaussian distribution 
  nsm_perturbs = 1 # how many perturbations sampled (k in the algorithm), default: 1
  use_neg = True # if use two-point estimate, default: True
  nso_lam = 0 # weight of the unperturbed weight loss, default: 0
  

model = YourModel()
base_optimizer = torch.optim.SGD  # define an optimizer for the gradient descent update
optimizer = NSO(model.parameters(), base_optimizer, sigma=args.nso_sigma, **dict(config["optimizer"]["args"])) # pass in the sigma of sample distribution and other optimizer parameters, such as weight_decay
...

for input, output in data:
  
    # first forward-backward step: compute the gradients on the original weight (can be skipped if nso_lam == 0)
    enable_running_stats(model)

    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.store_gradients(zero_grad=True, store_weights=True, update_weight=self.nso_lam)

    # second forward-backward step: taking perturbations and computing gradients (main part)
    disable_running_stats(model)
    if args.num_perturbs != 0:
        update_weight = (1-args.nso_lam)/(2*args.num_perturbs) if args.use_neg else (1-args.nso_lam)/(args.num_perturbs)
        for i in range(self.num_perturbs):
            optimizer.first_step(zero_grad=True, store_perturb=True)
            criterion(model(data), target).backward()
            optimizer.store_gradients(zero_grad=True, store_weights=False, update_weight=update_weight)
            if args.use_neg:
                optimizer.first_step(zero_grad=True, store_perturb=False)
                criterion(model(data), target).backward()
                optimizer.store_gradients(zero_grad=True, store_weights=False, update_weight=update_weight)
    optimizer.second_step(zero_grad=True)
...
```

The full training logic is specified in the `NSOTrainer` from `./exps_on_image_datasets/trainer/nso_trainer.py`. 

### Experiments

We evaluate our algorithm across various settings

**Fine-tuning ResNets on image classification datasets.** Enter the `./exps_on_image_datasets` folder to conduct fine-tuning experiments on image classification datasets. Please refer to `data/README.md` for the introductions to preparing data. 

To install requirements, see the package versions in `requirements.txt`

- Use `train_label_noise.py` to run experiments of fine-tuning ResNet/VisionTransformers on image datasets. 
- Use `compute_hessian_traces.py` to compute the trace of loss's Hessian of each layer in a neural network. 

We provide examples of how to run the scripts in the `exps_on_image_datasets/scripts` folder. 

**Fine-tuning RoBERTa/BERT model on text classification datasets.** Enter the `./exps_on_text_datasets` folder to conduct fine-tuning experiments on image classification datasets.

- Use `train_glue_label_noise.py` to run the experiments of fine-tuning RoBERTa-Base. Follow the bash script example in the `exps_on_text_datasets/scripts` folder to run the command. 

- Use `compute_hessian_traces.py` to compute the trace of loss's Hessian of each layer in a neural network. 

We provide examples to run the scripts in the `exps_on_text_datasets/scripts` folder. 

**Pretraining contrastive language-image models.** Enter the `./open_clip` folder to conduct experiments pretraining CLIP models on paired text-image datasets. Please refer to `open_clip/README.md` to set up the Python environment and download datasets. 

- Use `src/open_clip_train/main.py` to run the pretraining experiments. 

- Use `src/open_clip_train/compute_hessian_traces.py` to compute the trace of loss's Hessian of the network. 

Follow the bash script examples in `open_clip/scripts` to run the experiments. 

**Chain-of-thought fine-tuning of language models.** Enter the `./chain-of-thought-finetuning` folder to conduct experiments of chain-of-thought fine-tuning. Please refer to `chain-of-thought-finetuning/README.md` to set up the Python environment and download datasets. 

- Use `custom_train.py` to fine-tune language models on chain-of-thought data.

- Use `compute_hessian_traces.py` to compute the trace of loss's Hessian of the network. 

Follow the bash script examples in `chain-of-thought-finetuning/scripts` to run the experiments. 


### Acknowledgment

Thanks to the authors of the following repositories for providing their implementation publicly available.

- **[SAM Optimizer (In PyTorch)](https://github.com/davda54/sam)**
- **[PyHessian](https://github.com/amirgholami/PyHessian)**
- **[OpenCLIP](https://github.com/mlfoundations/open_clip)**
- **[Reasoning Teacher](https://github.com/itsnamgyu/reasoning-teacher)**
