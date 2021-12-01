from models import BreakOutAgent
# from pipeline import trainloop, Epsilon
from pipeline_numba import trainloop, Epsilon
import gym
from datetime import datetime
import os
import time

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
    hidden_nodes = 512
    output_nodes = 3
    hidden_layer_depth = 1 # one hidden layer
    fcn_params = (
        flatten_nodes,
        hidden_nodes,
        output_nodes,
        hidden_layer_depth
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
    # nepochs = 400
    nepochs = 2000000
    # game_step_limit =  10000
    # game_step_limit =  int(10000 *0.4)
    # game_step_limit =  int(10000 *0.01) # 0.35
    
    saveEveryN = 100
    
    gamma = 0.99
    cnn_depth = len(conv_params)
    # loss_type = "MSE"
    loss_type = "Huber"
    # total_epsilon_decrease_steps = 1000000.0
    # random_action_counter_limit = 50000
    eps_const = 10
    batch_size = 32
    # game_step_limit =  batch_size*eps_const
    game_step_limit =  10000
    train_num_per_run = 112
    lr = 0.00025 #* eps_const/2
    total_epsilon_decrease_steps = 30* 1000000.0 #4*1000000.0 #/ eps_const
    random_action_counter_limit = 50000 #// eps_const
    # train after every 4 actions and batch_size is satisfactory
    save_path = \
        f"../results/modelSaves/Loss{loss_type}_NCh{n_timesteps}_ConvD{cnn_depth}_HN{hidden_nodes}_HLD{hidden_layer_depth}_A{activation}_GSL{game_step_limit}_BchS{batch_size}_TNPR{train_num_per_run}_Lr{lr}_G{gamma}_EpsD{total_epsilon_decrease_steps}_RACL{random_action_counter_limit}_Date{datetime.now().strftime('%b%d_%H-%M-%S')}"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    epsilon = Epsilon(
        total_epsilon_decrease_steps = total_epsilon_decrease_steps, 
        random_action_counter_limit = random_action_counter_limit
        )
    trainloop(
        model,
        n_timesteps,
        env,
        nepochs,
        epsilon,
        saveEveryN, 
        game_step_limit,
        batch_size,
        train_num_per_run,
        dev,
        lr = lr,
        gamma = gamma,
        loss_type = loss_type,
        save_path = save_path
        
    )
    
    env.close()
    return save_path


if __name__ == '__main__':
    start = time.time()
    save_path = main()
    end = time.time()
    print(f"run {save_path}")
    print("Elapsed time = %s" % (end - start))