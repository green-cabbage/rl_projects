import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
from typing import TypeVar, List, Tuple

def addActivation(modules: List, activation: str) -> List:
    """
    author: Hyeon-Seo Yun Aug 2021
    """
    if activation == "elu":
        modules.append(nn.ELU())
    elif activation == "relu":
        modules.append(nn.ReLU())
    elif activation == "tanh":
        modules.append(nn.Hardtanh())
    elif activation == "leaky":
        modules.append(nn.LeakyReLU())
    else:
        pass
    return modules

def fillModules(
    modules: List,
    input_dim: int,
    output_dim: int,
    layer_depth: int,
    hidden_nodes: int,
    device: str,
    activation = "relu",
    bias = True,
    put_batchnorm = False) -> List:
    """
    author: Hyeon-Seo Yun Aug 2021
    we typically expect modules parameter to be an empty list
    We have it like this to leave the option to add modularity
    when put_batchnorm == True, we add batchnorm before every 
    dense layer, except before the input layer
    params:
    layer_depth == num of hidden layers + two to account for
    input and output layers
    """
    print("fillModules device: ", device)
    # add the input layer
    # print("input_dim ", input_dim)
    # print("hidden_nodes: ", hidden_nodes)
    # print("modules: ", len(modules))
    modules.append(nn.Linear(input_dim, hidden_nodes, bias=bias).to(device))
    modules = addActivation(modules, activation)
    for _ in range(layer_depth-2):
        if put_batchnorm:
            modules.append(nn.BatchNorm1d(hidden_nodes).to(device))
        modules.append(nn.Linear(hidden_nodes, hidden_nodes, bias=bias).to(device))
        modules = addActivation(modules, activation)  
    # last layer
    if put_batchnorm:
        modules.append(nn.BatchNorm1d(hidden_nodes).to(device))
    modules.append(nn.Linear(hidden_nodes, output_dim, bias=bias).to(device))
    modules = addActivation(modules, activation)
    return modules