import glob
import random

import numpy as np
import torch
import torch_geometric

from preprocessing.clean_datasets import clean_data_path, Graph_


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    torch_geometric.seed_everything(0)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def numOfGraphs():
    return len(glob.glob(f'{clean_data_path}{Graph_}*'))


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
