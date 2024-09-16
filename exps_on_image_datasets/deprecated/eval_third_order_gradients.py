# %%
import torch
from data_loader.data_loaders import MnistDataLoader
from model.model import MLP

state_dict_dir = "./saved/MLP_MNISTDataLoader_hidden_16_run_0/model_best.pth"

train_data_loader = MnistDataLoader(data_dir="./data", batch_size=256, shuffle=True, num_workers=4, training=True)
test_data_loader = MnistDataLoader(data_dir="./data", batch_size=256, shuffle=False, num_workers=4, training=False)

model = MLP(input_dim=28*28, hidden_dim=16, n_classes=10, n_layers=1)
model.load_state_dict(torch.load(state_dict_dir)['state_dict'])

# %%
import torch 
import torch.nn.functional as F

device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
criterion = F.nll_loss
model.to(device)
for batch_idx, (data, target) in enumerate(train_data_loader):
    data, target = data.to(device), target.to(device)
    data = data.view(data.shape[0], -1)
    def func(weights_1, bias_1, weights_2, bias_2):
        output = F.linear(data, weights_1, bias_1)
        output = F.relu(output)
        output = F.linear(output, weights_2, bias_2)
        output = F.log_softmax(output, dim=1)
        return criterion(output, target)
    
    hessians = torch.autograd.functional.hessian(func, (model.feature_extractor[0].weight, model.feature_extractor[0].bias, model.pred_head.weight, model.pred_head.bias), create_graph=True)

    # %%
    def get_hessian_trace(hessians):
        trace = 0.
        for i in range(len(hessians)):
            for j in range(len(hessians[i])):
                if i == j:
                    tmp_tensor = hessians[i][j]
                    if len(tmp_tensor.shape) == 2:
                        trace += torch.trace(tmp_tensor)
                    else:
                        tmp_tensor = tmp_tensor.view(tmp_tensor.shape[0]*tmp_tensor.shape[1], -1)
                        trace += torch.trace(tmp_tensor)
        return trace

    trace = get_hessian_trace(hessians)
    gradients = torch.autograd.grad(trace, model.parameters(), retain_graph=True)
    gradient_norms = [torch.norm(gradient) for gradient in gradients]
    trace_gradient_norm = torch.sum(torch.stack(gradient_norms))
    print(trace_gradient_norm)
    # %%

    def sample_hessian_gradient_norm(hessians):
        new_hessians = []
        for i in range(len(hessians)):
            for j in range(len(hessians[i])):
                new_hessians.append(hessians[i][j])
        hessians = [hessian.view(1, -1) for hessian in new_hessians]
        hessians = torch.cat(hessians, dim=1)

        sample_entry = torch.randint(0, hessians.shape[1], (1,))[0]
        sample_hessian = hessians[:, sample_entry]
        gradients = torch.autograd.grad(sample_hessian, model.parameters(), retain_graph=True)
        gradient_norms = [torch.norm(gradient) for gradient in gradients]
        gradient_norms = torch.sum(torch.stack(gradient_norms))
        return gradient_norms

    norm = 0
    for i in range(1620):
        norm += sample_hessian_gradient_norm(hessians)
    print(norm*162052900/1620)
    break