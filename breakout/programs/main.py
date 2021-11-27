from models import BreakOutAgent
from pipeline import train_loop
import gym
from datetime import datetime
import os

def main():
    # input shape: 210x160, image with 3 channels, but we 
    # are filtering to just 1 channel
    # n_timesteps = 1
    # # add in two more frames to manifest ball movement
    n_timesteps = 3
    conv_params = [
        (n_timesteps, 16, 8, 4),
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
    env = gym.make('ALE/Breakout-v5')#, render_mode='human')
    nepochs = 200000
    # game_step_limit =  10000
    game_step_limit =  int(10000 *0.38)
    print("game_step_limit: ", game_step_limit)
    batch_size = game_step_limit //2
    saveEveryN = 200
    lr = 0.00025
    gamma = 0.99
    hidden_layer_depth = layer_depth-2
    cnn_depth = len(conv_params)
    save_path = \
        f"../results/modelSaves/NCh{n_timesteps}_ConvD{cnn_depth}_HN{hidden_nodes}_HLD{hidden_layer_depth}_A{activation}_GSL{game_step_limit}_BchS{batch_size}_Lr{lr}_G{gamma}_Date{datetime.now().strftime('%b%d_%H-%M-%S')}"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    train_loop(
        model,
        n_timesteps,
        env,
        nepochs,
        saveEveryN, 
        game_step_limit,
        batch_size,
        dev,
        lr = lr,
        gamma = gamma,
        save_path = save_path
    )
    
    env.close()


if __name__ == '__main__':
    main()