from models import BreakOutAgent
from pipeline import train
import gym

def main():
    # input shape: 210x160, image with 3 channels, but we 
    # are filtering to just 1 channel
    n_channels = 1
    conv_params = [
        (n_channels, 16, 8, 4),
        (16, 32, 4, 2),
    ]
    flatten_nodes = 32*24*18
    hidden_nodes = 256
    output_nodes = 18
    layer_depth = 3 # one hidden layer
    fcn_params = (
        flatten_nodes,
        hidden_nodes,
        output_nodes,
        layer_depth
    )
    activation = "relu"
    dev = "cuda"
    model = BreakOutAgent(conv_params,
        fcn_params,
        dev,
        activation = activation
    )
    # start gym breakout
    env = gym.make('ALE/Breakout-v5', render_mode='human')
    nepochs = 200
    game_step_limit =  200
    sample_size = game_step_limit //2
    saveEveryN = 5
    lr = 0.0001
    train(
        model,
        env,
        nepochs,
        saveEveryN, 
        game_step_limit,
        sample_size,
        dev,
        lr = lr
    )
    env.close()


if __name__ == '__main__':
    main()