from utils import addActivation, fillModules
import torch 
import torch.nn as nn 

class ConvBlock(nn.Module):
    """
    Convolutional portion of the rl model
    """
    def __init__(
        self,
        parameters : List[Tuple[int]],
        activation = "relu"):
        """
        params:
        parameters: a list of tuples for Conv2ds. Each tuple is for one conv2d
        We assume the tuple to be sizes of four,
        which are (in_channels, out_channels, kernel_size, stride) informations
        the length of the parameters is the number of Conv2ds inserted, and in
        the order the tuples in parameters are listed
        
        activation: activation function to use. Options are relu, leaky
        """
        super(ConvBlock, self).__init__()
        modules = []
        for in_channels, out_channels, kernel_size, stride in parameters:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
            modules.append(conv)
            modules = addActivation(modules, activation)
        self.block_ = nn.Sequential(*modules)
        
    def forward(self, X):
        return self.block_(X)

class BreakOutAgent(nn.Module):
    def __init__(
        self,
        conv_params : List[Tuple[int]],
        fcn_params : Tuple[int],
        device : str,
        activation = "relu"):
        """
        params:
        conv_params: a list of tuples for Conv2ds. Each tuple is for one conv2d
        We assume the tuple to be sizes of four,
        which are (in_channels, out_channels, kernel_size, stride) informations
        the length of the parameters is the number of Conv2ds inserted, and in
        the order the tuples in parameters are listed

        fcn_params: tuple of dimensions for linear layers
        the tuple represents: 
        (flatten_nodes, hidden_nodes, output_nodes, layer_depth)
        flatten_nodes is the number of nodes of the flattened conv_block
        output. hidden_nodes is the number of nodes in the hidden layer.
        output_nodes is the number of nodes (equivalent to number of 
        actions relevant to the game that's being played)
        flatten_nodes can be inferred from the conv_params, but nobody
        got time for that
        """
        super(BreakOutAgent, self).__init__()
        # intialize conv block
        self.conv_block_ = ConvBlock(
            conv_params, 
            activation = activation
        )
        # initialize fcn block
        flatten_nodes, hidden_nodes, output_nodes, layer_depth = fcn_params
        # flatten_nodes == input_nodes
        fcn_modules = []
        fcn_modules = fillModules(
            flatten_nodes,
            output_nodes,
            layer_depth,
            hidden_nodes,
            device,
            activation = activation
        )
        self.fcn_block_ = nn.Sequential(*fcn_modules)


    def forward(self, X, entropy=0):
        # pass through conv block
        X = self.conv_block_(X)
        # flatten except the batch dim
        X = X.flatten(start_dim = 1)
        X = self.fcn_block_(X)
        # if entropy > 0:
        #     X += entropy
        #     X = t
        return X