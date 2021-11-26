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
import sklearn.preprocessing as pre 

gp_data = TypeVar("Gameplay Data")
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
        state_history = [],
        next_state_history = [],
        action_history = [],
        reward_history = [],
        done_history = [] ):
        self.state_history = state_history
        self.next_state_history = next_state_history
        self.action_history = action_history
        self.reward_history = reward_history
        self.done_history = done_history

    def __len__():
        """
        all the elements in this class should have the 
        same length
        """
        return len(self.state_history)

# def getGameAction(policy, entropy, env, game_state):
#     """
#     params:
    
#     policy: np array or torch tensor (haven't decided which)
#     that represent's the agent model's policy
#     entropy: the entropy value to take into account when
#     sampling a action from the given policy
#     env: game env, probably using this when certain actions
#     are not allowed according to the game
#     game_state: game state, probably using this when certain 
#     actions are not allowed according to the game
#     returns:
#     game compatible action from the given policy
#     """

def filterState(state : np_array, dev= "cpu") -> tch_tensor:
    """
    we assume the breakout state is np array of
    (210, 160, 3) shape
    we only take one channel, reshape it to (1,210, 160)
    and turn all non zero values to one before
    returining the array as torch tensor
    """
    print("raw state shape: ", state.shape)
    H, W, C = state.shape
    state_filtered = state[:,:,0].reshape((1, 1, H, W)) 
    state_filtered[state_filtered != 0] = 1
    # saving filter for test
    # np.savetxt(f"state_filtered.csv", state_filtered[0,0,:,:], delimiter=",")
    return torch.tensor(state_filtered, device = dev).float()

def takeRandomAction(policy: tch_tensor, epsilon =0):
    """
    we assume policy shape == (1, model.num_actions)
    in the future
    """
    print("original policy: ", policy)
    with torch.no_grad():
        np_policy = policy.cpu().numpy()
    # floor the negative values to zero and normalize
    np_policy[np_policy < 0] = 0
    np_policy = pre.normalize(np_policy, axis=0)
    # turn to one dimensional array
    np_policy = np_policy[0]
    print("normalized policy: ", np_policy)
    action = np.random.choice(len(np_policy), p = np_policy)
    return action

def playGameForTraining(
    model, 
    env, 
    game_step_limit, 
    dev = "cpu"):
    """
    """
    gameplay_data = GameplayData()
    raw_state = env.reset()
    # filter the state
    state = filterState(raw_state, dev = dev)

    counter = 0
    while True:
        counter += 1
        print("state shape: ", state.shape)
        policy = model(state)
        
        print("policy shape: ", policy.shape)
        print("policy[0] shape: ", policy[0].shape)
        
        
       
            
            
        action = takeRandomAction(policy)
        raw_next_state, reward, done, _ = env.step(action)
        # filter the next state
        next_state = filterState(raw_next_state, dev = dev)


        # take gameplay data
        gameplay_data.state_history.append(state)
        gameplay_data.next_state_history.append(next_state)
        gameplay_data.action_history.append(action)
        gameplay_data.reward_history.append(reward)
        gameplay_data.done_history.append(done)

        state = next_state
        # stop playing the game if number of steps taken
        # exceeds the limit
        if counter == game_step_limit:
            break
        
        if done:
            raw_state = env.reset()
            # filter the state
            state = filterState(raw_state, dev = dev)
    return gameplay_data

def loss_fn(
    prediction : tch_tensor,
    label : tch_tensor):
    """
    loss function used in model training
    return: loss scalar value (idk the official name)
    """
    return nn.MSELoss()(prediction, label)

def preprocessGameplayData(
    gameplay_data : gp_data,
    sample_size : int,
    dev = "cpu"):
    """
    preprocess the gameplay data for training
    """
    # random sample from gameplay_data
    sample_idxs = np.random.choice(range(len(gameplay_data)), size=sample_size)

    state_sample = torch.tensor(
        [gameplay_data.state_history[idx] for idx in sample_idxs],
        device = dev
    )
    next_state_sample = torch.tensor(
        [gameplay_data.next_state_history[idx] for idx in sample_idxs],
        device = dev
    )
    action_sample = torch.tensor(
        [gameplay_data.action_history[idx] for idx in sample_idxs],
        device = dev
    )
    reward_sample = torch.tensor(
        [gameplay_data.reward_history[idx] for idx in sample_idxs],
        device = dev
    )
    done_sample = torch.tensor(
        [gameplay_data.done_history[idx] for idx in sample_idxs],
        device = dev
    )
    
    # initialize new preprocessed gameplay data
    # except now the values are torch tensors
    preprocessed_gameplay_data = GameplayData(
        state_history = state_sample,
        next_state_history = next_state_sample,
        action_history = action_sample,
        reward_history = reward_sample,
        done_history = done_sample 
    )
    return preprocessed_gameplay_data
    
def trainOneBatch(
    model,
    optim,
    preprocessed_gameplay_data: gp_data):
    """
    params:
    preprocessed_gameplay_data
    """

    # Build the updated Q-values for the sampled future states
    next_state_array = preprocessed_gameplay_data.next_state_history
    with torch.no_grad():
        model.eval()
        next_state_q_values = model(next_state_array)
        # put model back to train mode
        model.train()
    # Q value = reward + discount factor * expected future reward
    # expected future reward == max q value from the next state
    rewards_array =  preprocessed_gameplay_data.reward_history
    updated_q_values = rewards_array + gamma * torch.max(next_state_q_values, dim=1)


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
    final_prediction = torch.multiply(q_pred, masks)
    loss = loss_fn(final_prediction, updated_q_values)
    loss.backward()
    optim.step()


def train(
    model,
    env,
    nepochs,
    saveEveryN, 
    game_step_limit,
    sample_size,
    dev):
    # initialize model and optim
    for epoch in range(nepochs):
        gameplay_data = playGameForTraining(
            model, 
            env,
            game_step_limit, 
            dev = dev)
        preprocessed_gameplay_data = preprocessGameplayData(
            gameplay_data, 
            sample_size, 
            dev = dev
        )
        trainOneBatch(model, optim, preprocessed_gameplay_data)

# def test():
#     """
#     """
