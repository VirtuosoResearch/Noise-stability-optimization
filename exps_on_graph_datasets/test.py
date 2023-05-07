# %%
import torch 
import numpy as np

state_dict = torch.load("./saved/COLLAB_gcn_layer_2_aggr_add_fold_0_run_1.pth")

weights = []
for key in state_dict.keys():
    if "weight" in key and "bn" not in key:
        weights.append(state_dict[key])

norms = (np.array([torch.norm(w).cpu().item() for w in weights]))
# %%
import numpy as np

norms = np.array([81.1944046 , 39.84794235,  9.04224682])
max_traces = np.array([[107.31640137, 31.56732254, 20.74490004]])
max_loss = 0.8333492279052734
train_num = 4500

bound = max_loss*np.math.sqrt((max_traces.sum()*norms.sum())/train_num)
