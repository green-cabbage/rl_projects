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
import numba
# import sklearn.preprocessing as pre 

gp_data = TypeVar("Gameplay Data")
eps_class = TypeVar("Epsilong Class")
tch_tensor = TypeVar("tch_tensor")
np_array = TypeVar("np_array")


# class GameplayData():
#     """
#     a simple class that groups
#     state_history, next_state_history, action_history,
#     reward_history, done_history
#     into one clean class variable
#     We assume the data types of these class member variables
#     are consistent within the same class instance
#     However, we also assume that the data type to not be 
#     static, ie using np arrays or torch tensors instead
#     of lists as the class member variable type
#     """
#     def __init__(
#         self,
#         state_history_size,
#         next_state_history_size,
#         action_history_size,
#         reward_history_size,
#         done_history_size,
#         dev =  "cpu"):
#         self.state_history = torch.empty(state_history_size, device = dev)
#         self.next_state_history = torch.empty(next_state_history_size, device = dev)
#         # action has long type bc it will layer go through one hot
#         self.action_history = torch.empty(action_history_size, device = dev).long()
#         self.reward_history = torch.empty(reward_history_size, device = dev)
#         self.done_history = torch.empty(done_history_size, device = dev)

#     def __len__(self):
#         """
#         all the elements in this class should have the 
#         same length
#         """
#         return len(self.state_history)
    
#     def __del__(self):
#         del self.state_history
#         del self.next_state_history
#         del self.action_history
#         del self.reward_history
#         del self.done_history

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
        # print("self.epsilon_interval_: ", self.epsilon_interval_)
        self.random_action_counter_limit_ = random_action_counter_limit
        self.counter_ = 0
    def update(self):
        # self.epsilon_ -= self.epsilon_interval_
        # self.epsilon_ = max(self.epsilon_, self.epsilon_min_)
        if self.counter_ > self.random_action_counter_limit_:
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

# @numba.njit(numba.uint8[:,:](numba.uint8[:,:]), parallel=True)
# def changeNonzeroValues(raw_state : np_array):
#     """
#     We assume raw_state is 2 dimensional
#     """
#     H, W = raw_state.shape
#     for idx in numba.prange(H):
#         for jdx in numba.prange(W):
#             val = raw_state[idx, jdx]
#             if val != 0:
#                 raw_state[idx, jdx] = 1
#     return raw_state

@numba.njit(numba.uint8[:,:](numba.uint8[:,:]))
def changeNonzeroValues(raw_state : np_array):
    """
    We assume raw_state is 2 dimensional
    """
    H, W = raw_state.shape
    for idx in range(H):
        for jdx in range(W):
            val = raw_state[idx, jdx]
            if val != 0:
                raw_state[idx, jdx] = 1
    return raw_state

@numba.njit(
    numba.float32[:,:,:,:](numba.float32[:,:,:,:], numba.uint8[:,:,:], numba.boolean),
    # parallel=True
)
def filterState(
    current_state : np_array, 
    raw_state : np_array, 
    beginning : bool) -> np_array:
    """
    we assume the breakout state is np array of
    (210, 160, 3) shape
    we only take one channel, reshape it to (1,210, 160)
    and turn all non zero values to one before
    returining the array as torch tensor
    params:
    current_state: float32 dtype
    raw_state: uint8 dtype
    """
    # print("raw state shape: ", state.shape)
    # take one of the three rgb frames
    raw_state = raw_state[:,:,0]
    # turn various nonzero values to 1
    raw_state = changeNonzeroValues(raw_state)
    N, T, H, W = current_state.shape
    if beginning: #  current_state == empty np array
        H, W = raw_state.shape
        # current_state_element = np.empty((1, 1, H, W), dtype = np.float32)
        # current_state_element[0,0,:,:] = raw_state[:,:]
        # current_state = np.concatenate(
        #    [current_state_element for _ in range(timestep_size)],
        #    axis = 1,
        # #    dtype = np.float32
        # )
        # current_state = np.repeat(current_state_element, timestep_size, axis=1)
        # turn various color values to just one
        
        for idx in range(T):
            current_state[0,idx,:,:] = raw_state[:,:]
        # saving filter for test
        # np.savetxt(f"current_state.csv", current_state[0,0,:,:], delimiter=",")
    else:
        # note timestep_size == T  
        # N should == 1
        if T > 1 :# reassign values like putting appending a queue
        # on the channel dimension with idx == 0 being the most recent
            # current_state[:,1:,:,:] = current_state[:,:-1,:,:]
            for idx in range(1,T):
                current_state[:,-(idx),:,:] = current_state[:,-(idx+1),:,:]
        # plug in a new element
        
        current_state[0,0,:,:] = raw_state

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

@numba.njit(
    numba.int64(numba.float32[:,:], numba.int64), 
    # parallel=True
)
def takeRandomAction(policy: np_array, epsilon = 0):
    """
    we assume policy shape == (1, model.num_actions)
    in the future
    """
    # print("original policy: ", policy)
    # turn to one dimensional array
    policy = policy[0]
    # print("policy shape: ", policy.shape)

    if np.random.random() < epsilon:
        # give uniform distribution for 100% 
        # random action
        action = np.random.choice(len(policy))
        # print("exploration")
    else: # normal normalization
        action = np.argmax(policy)
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
    state_history = np.empty((game_step_limit, n_timesteps, 210, 160), dtype = np.float32)
    next_state_history = np.empty((game_step_limit, n_timesteps, 210, 160), dtype = np.float32)
    action_history = np.empty((game_step_limit,), dtype = np.int64)
    reward_history = np.empty((game_step_limit,), dtype = np.float32)
    done_history = np.empty((game_step_limit,), dtype = np.float32)

    # do reset twice bc there's a bug where
    # if you do it once, it sometimes doesn't give
    # the ball
    raw_state = env.reset()
    raw_state = env.reset()
    H, W, _ = raw_state.shape

    # start with empty state
    state = np.empty((1, n_timesteps, H, W), dtype = np.float32)
    # filter the raw observation from the game
    state = filterState(state, raw_state, True)

    # print("starting playGameForTraining")
    counter = 0
    reward_tally = 0 
    """
    9 is left  10 is do nothing 11 is right
    """
    action_map = {0 : 9, 1 : 10, 2: 11}
    while True:
        epsilon.update()
        # print("state shape: ", state.shape)
        with torch.no_grad():
            policy = model(torch.from_numpy(state).to(dev)).cpu().numpy()
        
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

            

        next_state = filterState(state, raw_next_state, False)
        # print("next state shape: ", next_state.shape)

        # take gameplay data

        state_history[counter] = state
        next_state_history[counter] = next_state
        action_history[counter] = action
        reward_history[counter] = reward
        done_history[counter] = float(done)

        counter += 1
        state = next_state
        # stop playing the game if number of steps taken
        # exceeds the limit
        if counter == game_step_limit:
            break
        
        if done:
            raw_state = env.reset()
            # start with empty state
            state = np.empty((1, n_timesteps, H, W), dtype = np.float32)
            # filter the raw observation from the game
            state = filterState(state, raw_state, True)
            # print("done!")
    avg_reward = reward_tally/game_step_limit
    gameplay_tuple = (
        state_history,
        next_state_history,
        action_history,
        reward_history,
        done_history
    )
    return gameplay_tuple, avg_reward

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
    gameplay_tuple : gp_data,
    n_timesteps : int,
    sample_size : int,
    dev = "cpu"):
    """
    preprocess the gameplay data for training
    delte gameplay_data when done
    """
    state_history, next_state_history, action_history, reward_history, done_history \
        = gameplay_tuple
    # random sample from gameplay_data
    sample_idxs = np.random.choice(len(done_history), size=sample_size)



    # print("gameplay_data.state_history: ", gameplay_data.state_history)

    # initialize new preprocessed gameplay data
    state_history_size = (sample_size, n_timesteps, 210, 160)
    next_state_history_size = (sample_size, n_timesteps, 210, 160)
    action_history_size = (sample_size,)
    reward_history_size = (sample_size,)
    done_history_size = (sample_size,)

    sample_idxs = np.random.choice(len(done_history), size=sample_size)
    state_history = state_history[sample_idxs]
    next_state_history = next_state_history[sample_idxs]
    action_history = action_history[sample_idxs]
    reward_history = reward_history[sample_idxs]
    done_history = done_history[sample_idxs]

    gameplay_tuple = (
        state_history,
        next_state_history,
        action_history,
        reward_history,
        done_history
    )

    # print("state_sample shape: ", state_sample.shape)
    # print("state_sample shape: ", next_state_sample.shape)
    # print("action_sample shape: ", action_sample.shape)
    # print("reward_sample shape: ", reward_sample.shape)
    # print("done_sample shape: ", done_sample.shape)

    
    # delete preprocessed_gameplay_data
    # del gameplay_data
    return gameplay_tuple
    
def trainOneBatch(
    model,
    optim,
    gameplay_tuple,
    gamma : float,
    loss_type = "MSE",
    dev = "cpu"):
    """
    params:
    preprocessed_gameplay_data
    """
    state_array, next_state_array, action_array, rewards_array, done_array \
        = gameplay_tuple
    
    state_array = torch.tensor(state_array, device = dev)
    next_state_array = torch.tensor(next_state_array, device = dev)
    action_array = torch.tensor(action_array, device = dev)
    rewards_array =   torch.tensor(rewards_array, device = dev)
    done_array =  torch.tensor(done_array, device = dev)

    # Build the updated Q-values for the sampled future states
    with torch.no_grad():
        next_state_q_values = model(next_state_array)

    
    # Q value = reward + discount factor * expected future reward
    # expected future reward == max q value from the next state
    
    # print("next_state_q_values shape: ", next_state_q_values)
    max_vals, _ = torch.max(next_state_q_values, dim=1)
    

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
    
    
    q_pred = model(state_array)
    # mask so only the q predictions of actual actions taken are given
    
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


def trainloop(
    model,
    n_timesteps : int,
    env,
    nepochs,
    epsilon,
    saveEveryN,
    game_step_limit,
    sample_size,
    train_num_per_run : int,
    dev,
    lr = 0.001,
    gamma = 0.99,
    loss_type = "MSE",
    save_path = "."):
    

    # initialize optim
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    column_names = ["epoch", "avg_reward", "epsilon", "total steps played"]
    avg_reward_df = pd.DataFrame(columns = column_names)
    for epoch in range(nepochs):
        # print("epoch: ", epoch)
        
        # we intreprete epsilon == 0 being no exploration
        # and epsilon == 1 being always exploration
        gameplay_tuple, avg_reward = playGameForTraining(
            model,
            n_timesteps,
            env,
            game_step_limit, 
            dev = dev,
            epsilon = epsilon)
        for _ in range(train_num_per_run):    
            preprocessed_gameplay_data = preprocessGameplayData(
                gameplay_tuple, 
                n_timesteps,
                sample_size, 
                dev = dev
            )
            trainOneBatch(
                model, 
                optim, 
                gameplay_tuple,
                gamma, 
                loss_type = loss_type,
                dev = dev
            )
            # delete preprocessed_gameplay_data to save memory
            del preprocessed_gameplay_data

            
        if epoch % saveEveryN ==0:
            print("epoch: ", epoch)
            print("epsilon: ", epsilon)
            eps = str(epsilon)
            eps_counter = epsilon.counter_
            avg_reward_df = avg_reward_df.append(
                            {
                                column_names[0]: epoch, 
                                column_names[1]: avg_reward,
                                column_names[2]: eps,
                                column_names[3]: eps_counter
                            },  
                            ignore_index = True
            )
            # save model
            model_save_path = save_path + f"/Epoch{epoch}"
            if not os.path.exists(model_save_path):
                os.mkdir(model_save_path)
            torch.save(model,model_save_path + "/model.pt")

            
            avg_reward_df.to_csv(model_save_path + "/avg_rewards.csv")
            # # test
            # test()

        
# def play(
#     env_param : str, 
#     model_save_path : str):
#     """
#     allow human to see the performance of 
#     """
#     env = gym.make(env_param, render_mode='human')
#     model = torch.load(model_save_path)
#     for step in range(n_steps):
        

def resumeTrainloop():



    trainloop