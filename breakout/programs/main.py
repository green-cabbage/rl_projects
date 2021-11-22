
# input shape: 84 ×84 image with 4 channels
conv_params = [
    (4, 16, 8, 4),
    (16, 32, 4, 2),
]
flatten_nodes = 32*9*9
hidden_nodes = 256
output_nodes = 4
layer_depth = 3 # one hidden layer
fcn_params = (
    flatten_nodes,
    hidden_nodes,
    output_nodes,
    layer_depth
)
activation = "relu"
