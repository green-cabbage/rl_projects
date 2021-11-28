"""
this file defines the training pipeline and evaluations

the pipeline is inspired by the code at
https://keras.io/examples/rl/deep_q_network_breakout/
and dqn breakout paper
https://arxiv.org/pdf/1312.5602.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import TypeVar, List, Tuple
import os
import pandas as pd
import copy
# import sklearn.preprocessing as pre 

gp_data = TypeVar("Gameplay Data")
eps_class = TypeVar("Epsilong Class")
tch_tensor = TypeVar("tch_tensor")
np_array = TypeVar("np_array")


class GameplayData():
    """
    a simple class that groups
    state_history, next_state_history, action_history,
    reward_history, done_history
    into one clean class variable
    We assume the data types of these class member variables
    are consistent within the same class instance
    However, we also assume that the data type to not be 
    static, ie using np arrays or torch tensors instead
    of lists as the class member variable type
    """
    def __init__(
        self,
        state_history_size,
        next_state_history_size,
        action_history_size,
        reward_history_size,
        done_history_size,
        dev =  "cpu"):
        self.state_history = torch.empty(state_history_size, device = dev)
        self.next_state_history = torch.empty(next_state_history_size, device = dev)
        # action has long type bc it will layer go through one hot
        self.action_history = torch.empty(action_history_size, device = dev).long()
        self.reward_history = torch.empty(reward_history_size, device = dev)
        self.done_history = torch.empty(done_history_size, device = dev)

    def __len__(self):
        """
        all the elements in this class should have the 
        same length
        """
        return len(self.state_history)
    
    def __del__(self):
        del self.state_history
        del self.next_state_history
        del self.action_history
        del self.reward_history
        del self.done_history

class Epsilon():
    """
    simple class to record epsilon value throughout the 
    function calls
    """
    def __init__(
        self, 
        total_epsilon_decrease_steps = 1000000.0,
        random_action_counter_limit = 50000):
        self.epsilon_ = 1.0  # Epsilon greedy parameter
        self.epsilon_min_ = 0.1  # Minimum epsilon greedy parameter
        self.epsilon_max_ = 1.0  # Maximum epsilon greedy parameter
        self.total_epsilon_decrease_steps_ =  total_epsilon_decrease_steps 
        # Rate at which to reduce chance of random action being taken
        self.epsilon_interval_ = (
            (self.epsilon_max_ - self.epsilon_min_) /self.total_epsilon_decrease_steps_ 
        )
        print("self.epsilon_interval_: ", self.epsilon_interval_)
        self.random_action_counter_limit_ = random_action_counter_limit
        self.counter_ = 0
    def update(self):
        self.epsilon_ -= self.epsilon_interval_
        self.epsilon_ = max(self.epsilon_, self.epsilon_min_)
        self.counter_ += 1
        # print("self.epsilon_: ", self.epsilon_)

    def doRandomAction_q(self):
        return True if self.counter_ < self.random_action_counter_limit_ else False

    def __str__(self):
        if self.doRandomAction_q():
            return str(1.0)
        else:
            return str(self.epsilon_)

def filterState(
    current_state : tch_tensor, 
    raw_state : np_array, 
    timestep_size : int,
    dev= "cpu") -> tch_tensor:
    """
    we assume the breakout state is np array of
    (210, 160, 3) shape
    we only take one channel, reshape it to (1,210, 160)
    and turn all non zero values to one before
    returining the array as torch tensor
    """
    # print("raw state shape: ", state.shape)
    if current_state == None: # if start of the game
        H, W, C = raw_state.shape
        current_state = torch.empty(
            ((1, timestep_size, H, W)), 
            device = dev
        )
        for idx in range(timestep_size):
            current_state[:,idx,:,:] = torch.tensor(
                raw_state[:,:,0],
                device = dev
            ).float()
        # turn various color values to just one
        current_state[current_state != 0] = 1
        # saving filter for test
        # np.savetxt(f"current_state.csv", current_state[0,0,:,:], delimiter=",")
    else:
        # note timestep_size == T  
        N, T, H, W = current_state.shape
        if T > 1 :# reassign values like putting appending a queue
        # idx == 0 being the most recent
            for idx in range(1, T):
                current_state[:,idx-1,:,:] = current_state[:,idx,:,:]
        # plug in a new element
        raw_state = torch.tensor(raw_state[:,:,0], device = dev).float()
        # turn various color values to just one
        raw_state[raw_state != 0] = 1
        current_state[:,0,:,:] = raw_state

    # print("current_state: ", current_state.shape )
    return current_state




# def takeRandomAction(policy: tch_tensor, epsilon =0):
#     """
#     we assume policy shape == (1, model.num_actions)
#     in the future
#     """
#     # print("original policy: ", policy)
#     with torch.no_grad():
#         np_policy = policy.cpu().numpy()
#     # floor the negative values to zero and normalize
#     np_policy[np_policy < 0] = 0
#     # we can now assume np policy is all positive
#     # normalize
    

#     if sum == 0 or np.random.random() < epsilon:
#         # give uniform distribution for 100% 
#         # random action
#         # print("total random action")
#         num_actions = np_policy.shape[-1]
#         np_policy[:] = 1/num_actions
#     else: # normal normalization
#         np_policy = np_policy / np.sum(np_policy) 
#     # turn to one dimensional array
#     np_policy = np_policy[0]
#     # print("normalized policy: ", np_policy)
#     action = np.random.choice(len(np_policy), p = np_policy)
#     # print("action: ", action)
#     return action


def takeRandomAction(policy: tch_tensor, epsilon =0):
    """
    we assume policy shape == (1, model.num_actions)
    in the future
    """
    # print("original policy: ", policy)
    with torch.no_grad():
        np_policy = policy.cpu().numpy()

    # turn to one dimensional array
    np_policy = np_policy[0]
    # print("np_policy shape: ", np_policy.shape)

    if np.random.random() < epsilon:
        # give uniform distribution for 100% 
        # random action
        action = np.random.choice(len(np_policy))
        # print("exploration")
    else: # normal normalization
        action = np.argmax(np_policy)
        # print("exploitation")
    
    # print("action: ", action)
    return action

def playGameForTraining(
    model, 
    n_timesteps : int,
    env, 
    game_step_limit : int, 
    epsilon : eps_class,
    dev = "cpu"):
    """
    params:
    epsilon: Epsilon class instance
    """
    # print("epsilon: ", epsilon)
    # gameplay_data = GameplayData()
    

    # state_history_size = (game_step_limit, 1, 210, 160)
    # next_state_history_size = (game_step_limit, 1, 210, 160)
    state_history_size = (game_step_limit, n_timesteps, 210, 160)
    next_state_history_size = (game_step_limit, n_timesteps, 210, 160)
    action_history_size = (game_step_limit,)
    reward_history_size = (game_step_limit,)
    done_history_size = (game_step_limit,)
    gameplay_data = GameplayData(
            state_history_size,
            next_state_history_size,
            action_history_size,
            reward_history_size,
            done_history_size,
            dev = dev
    )
    # do reset twice bc there's a bug where
    # if you do it once, it sometimes doesn't give
    # the ball
    raw_state = env.reset()
    raw_state = env.reset()
    # filter the state
    state = filterState(None, raw_state, n_timesteps, dev = dev)
    # print("starting playGameForTraining")
    counter = 0
    reward_tally = 0 
    """
    9 is left  10 is do nothing 11 is right
    """
    action_map = {0 : 9, 1 : 10, 2: 11}
    while True:
        
        # print("state shape: ", state.shape)
        with torch.no_grad():
            # state = torch.from_numpy(state).to(dev)
            policy = model(state)
        
        # print("policy shape: ", policy.shape)
        # print("policy[0] shape: ", policy[0].shape)
 
        if epsilon.doRandomAction_q():
            current_eps = 1.0
        else: 
            current_eps = epsilon.epsilon_
        action = takeRandomAction(policy, epsilon = current_eps)
        raw_next_state, reward, done, _ = env.step(action_map[action])
        reward_tally += reward
        # filter the next state

            

        next_state = filterState(state, raw_next_state, n_timesteps, dev = dev)
        # print("next state shape: ", next_state.shape)

        # take gameplay data

        gameplay_data.state_history[counter] = state
        gameplay_data.next_state_history[counter] = next_state
        gameplay_data.action_history[counter] = action
        gameplay_data.reward_history[counter] = reward
        gameplay_data.done_history[counter] = float(done)

        counter += 1
        state = next_state
        # stop playing the game if number of steps taken
        # exceeds the limit
        if counter == game_step_limit:
            break
        
        if done:
            raw_state = env.reset()
            # filter the state
            state = filterState(None, raw_state, n_timesteps, dev = dev)
            # print("done!")
    avg_reward = reward_tally/game_step_limit
    return gameplay_data, avg_reward

def loss_fn(
    prediction : tch_tensor,
    label : tch_tensor,
    loss_type = "MSE"):
    """
    loss function used in model training
    return: loss scalar value (idk the official name)
    """
    if loss_type == "MSE":
        loss_variable = nn.MSELoss()
    elif loss_type == "Huber":
        loss_variable = nn.HuberLoss()
    return loss_variable(prediction, label)

def preprocessGameplayData(
    gameplay_data : gp_data,
    n_timesteps : int,
    sample_size : int,
    dev = "cpu"):
    """
    preprocess the gameplay data for training
    delte gameplay_data when done
    """
    # random sample from gameplay_data
    sample_idxs = np.random.choice(len(gameplay_data), size=sample_size)



    # print("gameplay_data.state_history: ", gameplay_data.state_history)

    # initialize new preprocessed gameplay data
    state_history_size = (sample_size, n_timesteps, 210, 160)
    next_state_history_size = (sample_size, n_timesteps, 210, 160)
    action_history_size = (sample_size,)
    reward_history_size = (sample_size,)
    done_history_size = (sample_size,)
    preprocessed_gameplay_data = GameplayData(
        state_history_size,
        next_state_history_size,
        action_history_size,
        reward_history_size,
        done_history_size,
        dev = dev
    )
    sample_idxs = np.random.choice(len(gameplay_data), size=sample_size)
    preprocessed_gameplay_data.state_history = gameplay_data.state_history[sample_idxs]
    preprocessed_gameplay_data.next_state_history = gameplay_data.next_state_history[sample_idxs]
    preprocessed_gameplay_data.action_history = gameplay_data.action_history[sample_idxs]
    preprocessed_gameplay_data.reward_history = gameplay_data.reward_history[sample_idxs]
    preprocessed_gameplay_data.done_history = gameplay_data.done_history[sample_idxs]



    # print("state_sample shape: ", state_sample.shape)
    # print("state_sample shape: ", next_state_sample.shape)
    # print("action_sample shape: ", action_sample.shape)
    # print("reward_sample shape: ", reward_sample.shape)
    # print("done_sample shape: ", done_sample.shape)

    
    # delete preprocessed_gameplay_data
    # del gameplay_data
    return preprocessed_gameplay_data
    
def trainOneBatch(
    model,
    optim,
    preprocessed_gameplay_data: gp_data,
    gamma : float,
    loss_type = "MSE"):
    """
    params:
    preprocessed_gameplay_data
    """

    # Build the updated Q-values for the sampled future states
    next_state_array = preprocessed_gameplay_data.next_state_history
    with torch.no_grad():
        next_state_q_values = model(next_state_array)
        
    # Q value = reward + discount factor * expected future reward
    # expected future reward == max q value from the next state
    rewards_array =  preprocessed_gameplay_data.reward_history
    done_array = preprocessed_gameplay_data.done_history
    # print("next_state_q_values shape: ", next_state_q_values)
    max_vals, _ = torch.max(next_state_q_values, dim=1)
    
    # print("max_vals shape: ", max_vals.shape) 
    # print("rewards_array shape: ", rewards_array.shape)
    # print("done_array shape: ", done_array.shape)

    
    updated_q_values = rewards_array + gamma *  max_vals
    # print("updated_q_values shape: ", updated_q_values.shape)
    # multiply (1-done_array) to get discounted values when it's not done
    # and make reward -1 when the game has ended
    # print("done_array: ", done_array)
    # print("updated_q_values b4 done array mult: ", updated_q_values)
    updated_q_values = updated_q_values*(1-done_array) -  done_array

    # print("updated_q_values after: ", updated_q_values)
    # print("updated_q_values shape: ", updated_q_values.shape)

    """
    updated_q_values == y from the deepmind paper
    note that 
    if done:
        y = reward
    else: 
        y = reward + gamma*torch.max(model(next_state))

    y/ updated_q_values are the labels that we are training the model
    to predict
    """

    optim.zero_grad()
    
    state_array = preprocessed_gameplay_data.state_history
    q_pred = model(state_array)
    # mask so only the q predictions of actual actions taken are given
    action_array = preprocessed_gameplay_data.action_history
    masks = F.one_hot(action_array, num_classes = model.num_actions)
    # print("q_pred.shape: ", q_pred.shape)
    # print("masks shape: ", masks.shape)
    final_prediction = torch.sum(torch.multiply(q_pred, masks), dim = 1)
    # print("final_prediction shape: ", final_prediction.shape)
    loss = loss_fn(final_prediction, updated_q_values, loss_type = loss_type)
    # print("loss value: ", loss)
    loss.backward()
    optim.step()

# def test():
#     """
#     """


def train_loop(
    model,
    n_timesteps : int,
    env,
    nepochs,
    epsilon,
    saveEveryN,
    game_step_limit,
    sample_size,
    dev,
    lr = 0.001,
    gamma = 0.99,
    loss_type = "MSE",
    save_path = "."):
    

    # initialize optim
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    column_names = ["epoch", "avg_reward", "epsilon"]
    avg_reward_df = pd.DataFrame(columns = column_names)
    for epoch in range(nepochs):
        # print("epoch: ", epoch)
        epsilon.update()
        # we intreprete epsilon == 0 being no exploration
        # and epsilon == 1 being always exploration
        gameplay_data, avg_reward = playGameForTraining(
            model,
            n_timesteps,
            env,
            game_step_limit, 
            dev = dev,
            epsilon = epsilon)
        preprocessed_gameplay_data = preprocessGameplayData(
            gameplay_data, 
            n_timesteps,
            sample_size, 
            dev = dev
        )
        trainOneBatch(
            model, 
            optim, 
            preprocessed_gameplay_data, 
            gamma, 
            loss_type = loss_type
        )
        # delete preprocessed_gameplay_data to save memory
        del preprocessed_gameplay_data

        
        # add data to avg_reward_df
        if epoch % (saveEveryN//2) ==0:
            eps = print(epsilon)
            avg_reward_df = avg_reward_df.append(
                            {
                                column_names[0]: epoch, 
                                column_names[1]: avg_reward,
                                column_names[2]: eps
                            },  
                            ignore_index = True
            )
        if epoch % saveEveryN ==0:
            print("epoch: ", epoch)
            print("epsilon: ", epsilon)
            # save model
            model_save_path = save_path + f"/Epoch{epoch}"
            if not os.path.exists(model_save_path):
                os.mkdir(model_save_path)
            torch.save(model,model_save_path + "/model.pt")

            
            avg_reward_df.to_csv(model_save_path + "/avg_rewards.csv")
            # # test
            # test()

        
