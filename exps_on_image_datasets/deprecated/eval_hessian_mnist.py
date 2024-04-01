# %%
import torch
from data_loader.data_loaders import MnistDataLoader
from model.model import MLP

state_dict_dir = "./saved/MLP_MNISTDataLoader_run_0/model_best.pth"

train_data_loader = MnistDataLoader(data_dir="./data", batch_size=256, shuffle=True, num_workers=4, training=True)
test_data_loader = MnistDataLoader(data_dir="./data", batch_size=256, shuffle=False, num_workers=4, training=False)

model = MLP(input_dim=28*28, hidden_dim=128, n_classes=10, n_layers=1)
model.load_state_dict(torch.load(state_dict_dir)['state_dict'])

# %%
import numpy as np
from model.loss import nll_loss
from utils.util import deep_copy

class args:

    eps = 0.001
    sample_size = 10000
    device = 0

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data_loader = test_data_loader

# %%
''' Evaluate the Hessian trace '''
import os
from utils.hessian import compute_hessians_trace

def get_layers(model):
    layers = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            layers[name] = module
    return layers

def compute_hessians_trace(model, criterion, data, target, device = "cpu", maxIter=100, tol=1e-3):
    # Get parameters and gradients of corresponding layer
    data, target = data.to(device), target.to(device)
    model = model.to(device)    
    output = model(data)
    loss = criterion(output, target)

    layers = get_layers(model)
    weights = [module.weight for name, module in layers.items()]
    model.zero_grad()
    gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)

    layer_traces = []
    trace_vhv = []
    trace = 0.

    # Start Iterations
    for _ in range(maxIter):
        vs = [torch.randint_like(weight, high=2) for weight in weights]
            
        # generate Rademacher random variables
        for v in vs:
            v[v == 0] = -1

        model.zero_grad()  
        Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)
        tmp_layer_traces = np.array([torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)])

        layer_traces.append(tmp_layer_traces)
        trace_vhv.append(np.sum(tmp_layer_traces))

        if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
            break
        else:
            trace = np.mean(trace_vhv)
    return np.mean(np.array(layer_traces), axis=0), loss.cpu().item()

model.eval()
traces = []
max_traces = np.zeros((2))
max_loss = 0
for data, target in data_loader:
    layer_traces, loss = compute_hessians_trace(model, nll_loss, data, target, device=device)
    
    traces.append(np.sum(layer_traces))
    max_traces = np.maximum(max_traces, layer_traces)
    max_loss = max(max_loss, loss)
    print(layer_traces)
    print("sum of trace: {}".format(np.sum(layer_traces)))
    print("sum of hessian traces: {}".format(np.mean(traces)))
    print("max of hessian traces: {}\tmax loss: {}".format(max_traces, max_loss))
        
# %%
''' Define a function to calculate stability '''
def perturbe_model_weights(state_dict, eps=0.001, use_neg = False, perturbation = {}):
    if not use_neg:
        perturbation = {} 
    for key, value in state_dict.items():
        if ("weight" in key) and (len(value.size())!=1):
            if use_neg:
                state_dict[key] -= perturbation[key]
            else:
                tmp_perturb = torch.randn_like(value)*eps
                state_dict[key] += tmp_perturb
                perturbation[key] = tmp_perturb
    return state_dict, perturbation

def compute_loss(model, data_loader, device = "cpu"):
    loss = 0
    batch_count = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for idx, (data, labels) in enumerate(data_loader):
            data, labels = data.to(device), labels.to(device)

            loss += nll_loss(model(data), labels)
            batch_count += 1

    return loss/batch_count

def calculate_stability(model, data_loader, eps=1e-3, device = "cpu", runs = 10):
    ''' Calculate pred_vectors for model before perturbation '''
    loss_before = compute_loss(model, data_loader, device=device)
    state_dict_before = deep_copy(model.state_dict())
    print(f"Loss before: {loss_before}")

    '''
    Calculate the perturbed loss
    '''
    differences = []
    for i in range(runs):
        differece = 0
        state_dict_after = deep_copy(state_dict_before)
        state_dict_after, perturbations = perturbe_model_weights(state_dict_after, eps = eps)
        model.load_state_dict(state_dict_after)
        
        loss_after = compute_loss(model, data_loader, device=device)
        differece += loss_after - loss_before
        print(f"Loss after: {loss_after}")
        # differences.append(differece.cpu().item())

        state_dict_after = deep_copy(state_dict_before)
        state_dict_after, _ = perturbe_model_weights(state_dict_after, eps = eps, use_neg=True, perturbation = perturbations)
        model.load_state_dict(state_dict_after)
        
        loss_after = compute_loss(model, data_loader, device=device)
        differece += loss_after - loss_before
        print(f"Loss after: {loss_after}")
        differences.append(differece.cpu().item()/2)
    return differences

# %%
if args.sample_size < len(data_loader.sampler.indices):
        data_loader.sampler.indices = data_loader.sampler.indices[:args.sample_size]
for eps in np.arange(0.02, 0.041, 0.001):
    model.load_state_dict(torch.load(state_dict_dir)['state_dict'])
    diff_losses = calculate_stability(model, data_loader, eps=eps, device = device)
    print("Noise stability of {}: {:.4f} +/- {:.4f}".format(
        eps, np.mean(diff_losses), np.std(diff_losses)
    ))

# %%
train_loss = compute_loss(model, train_data_loader, device)
test_loss = compute_loss(model, test_data_loader, device)


# %%
train_loss, test_loss = 0.02720494568347931, 0.06853332370519638

max_traces = np.array([37.96546761, 57.63991912])
norms = np.array([16.30604362,  4.63564253])
max_loss = 0.1378830522298813
train_num = 60000

bound = max_loss*np.math.sqrt((max_traces.sum()*np.square(norms).sum())/train_num)

# %% 
state_dict = torch.load(state_dict_dir)['state_dict']
weights = []
for key, value in state_dict.items():
    if ("weight" in key) and (len(value.size())!=1):
        weights.append(value)
norms = np.array([torch.norm(weight).cpu().item() for weight in weights])