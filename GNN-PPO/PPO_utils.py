import torch
from torch_geometric.data import Batch

import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os

####################################
# Replay Buffer
####################################
class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.rewards = []
        self.done = []

        self.old_graph = []

        self.advantages = []
        self.returns = []

        self.batch_size = batch_size

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.values.clear()
        self.rewards.clear()
        self.done.clear()

        self.old_graph = []

        self.advantages.clear()
        self.returns.clear()
    
    def push(self, obs, action, logprob, value, reward, done, old_graph):
        self.obs.append(obs.detach().clone())
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.rewards.append(reward)
        self.done.append(done)

        self.old_graph.append(old_graph)

    def push_advantages(self, advantages, returns):
        self.advantages.extend(advantages)
        self.returns.extend(returns)

    def sample_idx(self, idx):
        obs_tensor = torch.stack(self.obs)
        actions_tensor = torch.stack(self.actions)
        old_log_probs_tensor = torch.stack(self.logprobs)
        advantages_tensor = torch.stack(self.advantages)
        returns_tensor = torch.stack(self.returns)

        old_graph_idx = [self.old_graph[i] for i in idx]
        old_graph_tensor = Batch.from_data_list(old_graph_idx)

        mb_obs = obs_tensor[idx]
        mb_actions = actions_tensor[idx]
        mb_old_log_probs = old_log_probs_tensor[idx]
        mb_advantages = advantages_tensor[idx]
        mb_returns = returns_tensor[idx]

        return mb_obs, mb_actions, mb_old_log_probs, mb_advantages, mb_returns, old_graph_tensor
    
    def __len__(self):
        assert(len(self.obs)==len(self.advantages))
        return len(self.obs)
    
####################################
# Returns
####################################
def compute_returns(next_value, rewards, values, masks, gamma=0.99, lam=0.95):
    '''Compute returns for Generalized Advantage Estimator (GAE)'''
    # values: list of tensors (1D) per timestep
    returns = []
    gae = 0
    # Append an extra value for bootstrap (assumed 0 for terminal state)
    values = list(values) + [next_value]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

####################################
# Convert action list of type [vert, horiz] -->  to dict of shape (5,)
####################################
def convert_action_to_env(pursuer_names=None, evader_names=None, pursuer_actions=None, evader_actions=None):
    ''' 
    Input: team_action lists of shape (n_agents, 2)
    Output: dict of actions
    '''
    env_actions = {}

    def populate_pursuer():
        for pursuer, (vert, horiz) in zip(pursuer_names, pursuer_actions):
            action = np.zeros(5, dtype='float32')
            action[4 if vert > 0 else 3] = abs(vert)
            action[2 if horiz > 0 else 1] = abs(horiz)

            env_actions[pursuer] = action

    def populate_evader():
        for evader, (vert, horiz) in zip(evader_names, evader_actions):
            action = np.zeros(5, dtype='float32')
            action[4 if vert > 0 else 3] = abs(vert)
            action[2 if horiz > 0 else 1] = abs(horiz)

            env_actions[evader] = action

    if pursuer_names is not None:
        populate_pursuer()
    if evader_names is not None:
        populate_evader()
    
    return env_actions

####################################
# Visualize Rewards
####################################
def visualize_reward_graph():
    ''' Plot rewards using matplotlib. '''
    rewards = np.load("GNN-PPO/rewards.npy")
    
    def running_avg(data, window=10):
        ''' Return x_vals, y_vals to plot the running average. '''
        n = window - 1
        running_avg = [np.mean(data[i-n:i+1]) for i in range(n, len(data))]
        return list(range(n, len(rewards))), running_avg

    running_x, running_averages = running_avg(rewards)

    rewards_x = list(range(len(rewards)))

    plt.plot(rewards_x, rewards, alpha=0.5, label="Reward")
    plt.plot(running_x, running_averages, color="orangered", label="Running Avg (last 10)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

####################################
# Save/Load Actor and Critic Models
####################################
def save_models(actorcritics, model_name):
    '''Save models according to model_name.'''

    save_dir = os.path.join("GNN-PPO", "Models")
    torch.save(actorcritics['pursuer'].state_dict(), os.path.join(save_dir, f'pursuer_{model_name}.pth'))
    torch.save(actorcritics['evader'].state_dict(), os.path.join(save_dir, f'evader_{model_name}.pth'))

    print("Checkpoint. Models saved. ")

def load_models(actorcritics, model_name):
    '''
    Load models into 'actors' and 'critics' from model_name.
    '''
    save_dir = os.path.join("GNN-PPO", "Models")

    actorcritics['pursuer'].load_state_dict(torch.load(os.path.join(save_dir, f'pursuer_{model_name}.pth')))
    actorcritics['evader'].load_state_dict(torch.load(os.path.join(save_dir, f'evader_{model_name}.pth')))


####################################
# Visualize one rollout
####################################
def visualize_policy(env, actorcritics):
    '''Perform one rollout of environment. '''
    actorcritics['pursuer'].eval()
    actorcritics['evader'].eval()

    observations, _ = env.reset()
    # Constants
    pursuer_names = [agent for agent in env.agents if 'adversary' in agent]
    evader_names = [agent for agent in env.agents if 'agent' in agent]

    reward_accum = 0
    while True:
        env.render()
        # Get actions
        p_obs = torch.tensor(np.array([observations[p].copy() for p in pursuer_names]))
        e_obs = torch.tensor(np.array([observations[e].copy() for e in evader_names]))

        p_actions, p_logprobs, p_values = actorcritics['pursuer'].get_action(p_obs) # Get pursuer actions. Shape (num_pursuer, 2)
        e_actions, e_logprobs, e_values = actorcritics['evader'].get_action(e_obs) # Get evader actions. Shape (num_evader, 2)

        # Convert action list to dict
        env_actions = convert_action_to_env(pursuer_names, evader_names, p_actions, e_actions)

        # 1.2: Take action and observe result 
        next_observations, rewards, terminations, truncations, _ = env.step(env_actions)
        done = terminations[pursuer_names[0]] or truncations[pursuer_names[0]]
        reward_accum += rewards[pursuer_names[0]]

        if(rewards[pursuer_names[0]]) > 0:
            print("Reward!")
        elif(rewards[pursuer_names[0]]) < 0:
            print("Collision!")

        if done:
            break
        observations = next_observations

    print(f"Reward: {reward_accum:.2f}")
    env.close()