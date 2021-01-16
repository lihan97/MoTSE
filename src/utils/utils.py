import torch
import dgl
import numpy as np
import random
import sys
import os
sys.path.append('..')
def set_random_seed(seed=22):
    random.seed(seed)
    np.random.seed(seed)
    dgl.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def makedir(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)
def load_model(n_tasks, device=torch.device('cpu'),
               source_model_path=None):
    from models import GCN
    model = GCN(n_tasks=n_tasks)
    

    if source_model_path is not None:
        print(f"loading pretrained model: {source_model_path}")
        model.load_state_dict(torch.load(source_model_path)['model_state_dict'])
    return model.to(device)