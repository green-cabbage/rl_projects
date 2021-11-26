from models import BreakOutAgent
from pipeline import train
import gym

def main():
    # input shape: 210x160, image with 3 channels
    conv_params = [
        (3, 16, 8, 4),
        (16, 32, 4, 2),
    ]
    flatten_nodes = 32*24*18
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
    dev = "cpu"
    model = BreakOutAgent(conv_params,
        fcn_params,
        dev,
        activation = activation
    )
    # start gym breakout
    env = gym.make('ALE/Breakout-v5', render_mode='human')
    game_step_limit
    def train(
    model,
    nepochs,
    saveEveryN, 
    dev)
    env.close()


if __name__ == '__main__':
    main()