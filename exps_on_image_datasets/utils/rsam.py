import torch

class RSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, sigma=0.05, lam=1, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, sigma=sigma, lam = lam, **kwargs)
        super(RSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def store_weights(self, zero_grad=False):
        ''' store the gradients of original weights '''
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad: continue
                self.state[p]["old_p"] = p.data.clone()    

        if zero_grad: self.zero_grad(set_to_none=False)

    # @torch.no_grad()
    # def store_gradients(self, zero_grad=False, restore_gradients=True, update_weight = 1.0):
    #     ''' store the gradients of original weights '''
    #     for group in self.param_groups:
    #         for p in group["params"]:
    #             if not p.requires_grad: continue
    #             if restore_gradients: 
    #                 self.state[p]["old_gradients"] = p.grad.data.clone()*update_weight
    #             else:
    #                 assert ("old_gradients" in self.state[p])
    #                 self.state[p]["old_gradients"] += p.grad.data.clone()*update_weight

    #     if zero_grad: self.zero_grad(set_to_none=False)

    @torch.no_grad()
    def perturb_weights(self, zero_grad=False):
        ''' take a perturbation step of the original weights '''
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad: continue

                # compute per-channel norm
                if len(p.data.size()) == 4:
                    weight_norm = torch.norm(p.data.view(p.data.size(0), p.data.size(1), -1), p="fro", dim=2)
                    
                    e_w = torch.randn_like(p.data) * group["sigma"] * weight_norm.view(weight_norm.size(0), weight_norm.size(1), 1, 1)
                    self.state[p]["perturb"] = e_w
                if len(p.data.size()) == 2:
                    weight_norm = torch.norm(p.data, p="fro", dim=1)
                    e_w = torch.randn_like(p.data) * group["sigma"] * weight_norm.view(weight_norm.size(0), 1)
                    self.state[p]["perturb"] = e_w
                else:
                    e_w = torch.randn_like(p.data) * group["sigma"]
                    self.state[p]["perturb"] = e_w
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad(set_to_none=False)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"] # retore the weights before perturbation

        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad(set_to_none=False)

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad(set_to_none=False)

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        (((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)*group["lam"] +\
                             self.state[p]["perturb"].norm(p=2).to(shared_device)) # added the perturbation weight here 
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups