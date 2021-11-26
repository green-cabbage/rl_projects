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
# import sklearn.preprocessing as pre 

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

    def __len__(self):
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
    # print("raw state shape: ", state.shape)
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
    # print("original policy: ", policy)
    with torch.no_grad():
        np_policy = policy.cpu().numpy()
    # floor the negative values to zero and normalize
    np_policy[np_policy < 0] = 0
    # we can now assume np policy is all positive
    # normalize
    sum = np.sum(np_policy)

    if sum == 0 or np.random.random() < epsilon:
        # give uniform distribution for 100% 
        # random action
        # print("total random action")
        num_actions = np_policy.shape[-1]
        np_policy[:] = 1/num_actions
    else: # normal normalization
        np_policy = np_policy / sum 
    # turn to one dimensional array
    np_policy = np_policy[0]
    # print("normalized policy: ", np_policy)
    action = np.random.choice(len(np_policy), p = np_policy)
    # print("action: ", action)
    return action

def playGameForTraining(
    model, 
    env, 
    game_step_limit, 
    dev = "cpu",
    epsilon = 0):
    """
    """
    print("epsilon: ", epsilon)
    gameplay_data = GameplayData()
    # do reset twice bc there's a bug where
    # if you do it once, it sometimes doesn't give
    # the ball
    raw_state = env.reset()
    raw_state = env.reset()
    # filter the state
    state = filterState(raw_state, dev = dev)
    print("starting playGameForTraining")
    counter = 0
    while True:
        counter += 1
        # print("state shape: ", state.shape)
        policy = model(state)
        
        # print("policy shape: ", policy.shape)
        # print("policy[0] shape: ", policy[0].shape)
        
        
       
            

        action = takeRandomAction(policy, epsilon = epsilon)
        raw_next_state, reward, done, _ = env.step(action)
        # filter the next state
        if done:
            print("done!")

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
    sample_idxs = np.random.choice(len(gameplay_data), size=sample_size)

    # for element in [gameplay_data.done_history[idx] for idx in sample_idxs]:
    #     print("element type: ", type(element))

    # print("gameplay_data.state_history: ", gameplay_data.state_history)
    state_sample = torch.cat([gameplay_data.state_history[idx] for idx in sample_idxs])
    next_state_sample = torch.cat([gameplay_data.next_state_history[idx] for idx in sample_idxs])
    action_sample = torch.tensor(
        [gameplay_data.action_history[idx] for idx in sample_idxs],
        device = dev
    )
    reward_sample = torch.tensor(
        [gameplay_data.reward_history[idx] for idx in sample_idxs],
        device = dev
    )
    done_sample = torch.tensor(
        [float(gameplay_data.done_history[idx]) for idx in sample_idxs],
        device = dev
    )
    # print("state_sample shape: ", state_sample.shape)
    # print("state_sample shape: ", next_state_sample.shape)
    # print("action_sample shape: ", action_sample.shape)
    # print("reward_sample shape: ", reward_sample.shape)
    # print("done_sample shape: ", done_sample.shape)

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
    preprocessed_gameplay_data: gp_data,
    gamma : float):
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
    done_array = preprocessed_gameplay_data.done_history
    # print("next_state_q_values shape: ", next_state_q_values)
    # print("max(next_state_q_values, dim=1): ", torch.max(next_state_q_values, dim=1))
    max_vals, _ = torch.max(next_state_q_values, dim=1)
    
    # print("max_vals shape: ", max_vals.shape) 
    # print("rewards_array shape: ", rewards_array.shape)
    # print("done_array shape: ", done_array.shape)

    
    updated_q_values = rewards_array + gamma *  max_vals
    # multiply (1-done_array) to get discounted values when it's not done
    # and make reward -1 when the game has ended
    print("done_array: ", done_array)
    print("updated_q_values b4 done array mult: ", updated_q_values)
    updated_q_values = updated_q_values*(1-done_array) -  done_array

    print("updated_q_values after: ", updated_q_values)

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
    print("final_prediction shape: ", final_prediction.shape)
    loss = loss_fn(final_prediction, updated_q_values)
    print("loss value: ", loss)
    loss.backward()
    optim.step()


def train(
    model,
    env,
    nepochs,
    saveEveryN, 
    game_step_limit,
    sample_size,
    dev,
    lr = 0.001,
    gamma = 0.99):
    # initialize optim
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    epsilon_min = 0.05
    epsilon_counter = 1.0
    for epoch in range(nepochs):
        epsilon_counter -= 0.02
        epsilon = max(epsilon_counter, epsilon_min)
        # we intreprete epsilon == 0 being no exploration
        # and epsilon == 1 being always exploration
        gameplay_data = playGameForTraining(
            model,
            env,
            game_step_limit, 
            dev = dev,
            epsilon = epsilon)
        preprocessed_gameplay_data = preprocessGameplayData(
            gameplay_data, 
            sample_size, 
            dev = dev
        )
        trainOneBatch(model, optim, preprocessed_gameplay_data, gamma)

# def test():
#     """
#     """
