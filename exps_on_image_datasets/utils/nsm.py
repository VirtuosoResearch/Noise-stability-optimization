import torch
import torch.distributions as dist
import numpy as np

class NSM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, sigma=0.05, distribution=None, **kwargs):
        assert sigma >= 0.0, f"Invalid sigma, should be non-negative: {sigma}"

        defaults = dict(sigma=sigma, **kwargs)
        super(NSM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

        if distribution == "laplace":
            self.distribution = dist.Laplace(0, 1/np.math.sqrt(2))
        elif distribution == "uniform":
            self.distribution = dist.Uniform(-np.math.sqrt(3), np.math.sqrt(3))
        elif distribution == "binomial":
            self.distribution = dist.Binomial(4, 0.5)
        elif distribution == "cauchy":
            self.distribution = dist.Cauchy(0, 1)
        else:
            self.distribution = dist.Normal(0, 1)

    @torch.no_grad()
    def store_gradients(self, zero_grad=False, store_weights=False, update_weight = 0.5):
        ''' store the gradients of original weights '''
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad: continue
                if store_weights: 
                    self.state[p]["old_p"] = p.data.clone()
                    self.state[p]["old_gradients"] = p.grad.data.clone()*update_weight
                else:
                    assert ("old_gradients" in self.state[p])
                    self.state[p]["old_gradients"] += p.grad.data.clone()*update_weight

        if zero_grad: self.zero_grad(set_to_none=False)

    @torch.no_grad()
    def first_step(self, zero_grad=False, store_perturb=True):
        ''' take a perturbation step of the original weights '''
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad: continue
                p.data = self.state[p]["old_p"].clone()  # restore original weights 
                if store_perturb:
                    e_w = self.distribution.sample(p.data.shape).to(p.device) * group["sigma"]
                    self.state[p]["perturb"] = e_w
                    p.add_(e_w)  # climb to the local maximum "w + e(w)"
                else:
                    e_w = self.state[p]["perturb"]
                    p.sub_(e_w)

        if zero_grad: self.zero_grad(set_to_none=False)

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad: continue
                p.data = self.state[p]["old_p"]  # get back to original weights
                p.grad.data = self.state[p]["old_gradients"]

        self.base_optimizer.step()  # do the actual weight update

        if zero_grad: 
            self.zero_grad(set_to_none=False)

    @torch.no_grad()
    def step(self, closure=None):
        ''' Deprecated for now '''
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups