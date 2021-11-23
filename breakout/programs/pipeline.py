"""
this file defines the training pipeline and evaluations

the pipeline is inspired by the code at
https://keras.io/examples/rl/deep_q_network_breakout/
and dqn breakout paper
https://arxiv.org/pdf/1312.5602.pdf
"""

from typing import TypeVar, List, Tuple

gp_data = TypeVar("Gameplay Data")

class GameplayData():
    """
    a simple class that groups
    state_history, next_state_history, action_history,
    reward_history, done_history
    into one clean class variable
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
        self.reward_history = reward_history]
        self.done_history = done_history

    def __len__():
        """
        all the elements in this class should have the 
        same length
        """
        return len(self.state_history)

def getGameAction(policy, entropy, env, game_state):
    """
    params:
    
    policy: np array or torch tensor (haven't decided which)
    that represent's the agent model's policy
    entropy: the entropy value to take into account when
    sampling a action from the given policy
    env: game env, probably using this when certain actions
    are not allowed according to the game
    game_state: game state, probably using this when certain 
    actions are not allowed according to the game
    returns:
    game compatible action from the given policy
    """


def playGameForTraining(model, env, game_step_limit):
    """
    """
    gameplay_data = GameplayData()
    state = env.reset()

    counter = 0
    while True:
        counter += 1
        policy = model(state)
        action, np_action = getGameAction(policy, entropy)
        next_state, reward, done = env.step(action)
        # take gameplay data
        gameplay_data.state_history.append(state)
        gameplay_data.next_state_history.append(next_state)
        gameplay_data.action_history.append(np_action)
        gameplay_data.reward_history.append(reward)
        gameplay_data.done_history.append(done)

        state = next_state
        # stop playing the game if number of steps taken
        # exceeds the limit
        if counter == game_step_limit:
            break
        
        if done:
            state = env.reset()

def preprocessGameplayData(gameplay_data : gp_data, sample_size : int):
    """
    preprocess the gameplay data for training
    """
    # random sample from gameplay_data
    sample_indices = np.random.choice(range(len(gameplay_data)), size=sample_size)

    state_sample = np.array([gameplay_data.state_history[idx] for idx in sample_indices])
    next_state_sample = np.array([gameplay_data.next_state_history[idx] for idx in sample_indices])
    action_sample = np.array([gameplay_data.action_history[idx] for idx in sample_indices])
    reward_sample = np.array([gameplay_data.reward_history[idx] for idx in sample_indices])
    done_sample = np.array([gameplay_data.done_history[idx] for idx in sample_indices])
    
    # initialize new preprocessed gameplay data
    preprocessed_gameplay_data = GameplayData(
        state_history = state_sample,
        next_state_history = next_state_sample,
        action_history = action_sample,
        reward_history = reward_sample,
        done_history = done_sample 
    )
    return preprocessed_gameplay_data
    
def trainFromPreprocesssedGameplay(
    model,
    preprocessed_gameplay_data: gp_data):
    """

    """

    if done:
        y = reward
    else: 
        y = reward + gamma*torch.max(model(next_state))

        