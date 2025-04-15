####################################
# Imports
####################################
import torch
from torch_geometric.data import Batch

from DDPG_graph import GraphEncoder

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

####################################
# Replay Buffer
####################################
class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
    
    def push(self, obs, next_obs, actions, rewards, done, graph, next_graph):
        self.buffer.append((obs, next_obs, actions, rewards, done, graph, next_graph))
    
    def sample(self):
        ''' Return batch of samples. '''
        batch = random.sample(self.buffer, self.batch_size) # Shape: [batch size, 7 items, # of pursuer/evader names]
        return batch
    
    def sample_vectorized(self):
        ''' Return vectorized batch of samples. '''
        batch = self.sample()

        obs_list, next_obs_list, actions_list, rewards_list, done_list, graph_list, next_graph_list = zip(*batch)
        obs_batch = torch.stack(obs_list)
        next_obs_batch = torch.stack(next_obs_list)
        actions_batch = torch.stack(actions_list)
        rewards_batch = torch.stack(rewards_list).unsqueeze(2)
        done_batch = torch.stack(done_list).unsqueeze(2)

        graph_batch = Batch.from_data_list(graph_list)
        next_graph_batch = Batch.from_data_list(next_graph_list)
        
        return obs_batch, next_obs_batch, actions_batch, rewards_batch, done_batch, graph_batch, next_graph_batch # Each shape: [batch_size, n_agents, type size]
    
    def __len__(self):
        return len(self.buffer)

####################################
# Gaussian Noise
####################################
def get_Gaussian_noise(episode, total_episodes, num_agents, SIGMA_INIT = 0.9, SIGMA_END = 0.1):
    ''' 
    Input: Episodes elapsed, total episodes.
    Output: Noise array of shape (num_agents, 2).
            Mean = 0, standard deviation reaches SIGMA_END after total_episodes/2. 
    '''
    anneal_episode = total_episodes//2
    sigma = SIGMA_END if (episode > anneal_episode) else (SIGMA_END + (SIGMA_INIT-SIGMA_END)*(anneal_episode-episode)/anneal_episode)

    return np.random.normal(0, sigma, (num_agents, 2)) # Shape [num_agents, 2]

####################################
# Convert action list of type [vert, horiz] -->  to dict of shape (5,)
####################################
def convert_actions_to_env(pursuer_names=None, evader_names=None, pursuer_actions=None, evader_actions=None):
    ''' 
    Input: Team_action lists of shape (n_agents, 2).
    Output: Dict of actions.
    '''
    env_actions = {}

    # Populate pursuer
    for pursuer, (horiz, vert) in zip(pursuer_names, pursuer_actions):
        action = np.zeros(5, dtype='float32')
        action[2 if horiz > 0 else 1] = abs(horiz)
        action[4 if vert > 0 else 3] = abs(vert)

        env_actions[pursuer] = action

    # Populate evader
    for evader, (horiz, vert) in zip(evader_names, evader_actions):
        action = np.zeros(5, dtype='float32')
        action[2 if horiz > 0 else 1] = abs(horiz)
        action[4 if vert > 0 else 3] = abs(vert)
        env_actions[evader] = action
        
    return env_actions
        

####################################
# Save/Load Actor and Critic Models
####################################
def save_models(actors, critics, model_name):
    ''' Save models according to model_name. '''
    save_dir = os.path.join("GNN-DDPG", "Models")

    torch.save(actors['pursuer'].state_dict(), os.path.join(save_dir, f'a_pursuer_{model_name}.pth'))
    torch.save(actors['evader'].state_dict(), os.path.join(save_dir, f'a_evader_{model_name}.pth'))
    torch.save(critics['pursuer'].state_dict(), os.path.join(save_dir, f'c_pursuer_{model_name}.pth'))
    torch.save(critics['evader'].state_dict(), os.path.join(save_dir, f'c_evader_{model_name}.pth'))

    print("Checkpoint. Models saved. ")

def load_actors(actors, model_name):
    ''' Load actor and critic models from disk. '''
    save_dir = os.path.join("GNN-DDPG", "Models")

    actors['pursuer'].load_state_dict(torch.load(os.path.join(save_dir, f'a_pursuer_{model_name}.pth'), weights_only=True))
    actors['evader'].load_state_dict(torch.load(os.path.join(save_dir, f'a_evader_{model_name}.pth'), weights_only=True))

def load_models(actors, critics, model_name):
    ''' Load actor and critic models from disk. '''
    save_dir = os.path.join("GNN-DDPG", "Models")

    actors['pursuer'].load_state_dict(torch.load(os.path.join(save_dir, f'a_pursuer_{model_name}.pth'), weights_only=True))
    actors['evader'].load_state_dict(torch.load(os.path.join(save_dir, f'a_evader_{model_name}.pth'), weights_only=True))
    critics['pursuer'].load_state_dict(torch.load(os.path.join(save_dir, f'c_pursuer_{model_name}.pth'), weights_only=True))
    critics['evader'].load_state_dict(torch.load(os.path.join(save_dir, f'c_evader_{model_name}.pth'), weights_only=True))

####################################
# Visualize Rewards
####################################
def visualize_rewards(reward_path = "GNN-DDPG/rewards.npy", window=10):
    ''' Plot rewards using matplotlib. '''
    rewards = np.load(reward_path)
    
    def running_avg(data):
        ''' Return x_vals, y_vals to plot the running average. '''
        n = window - 1
        running_avg = [np.mean(data[i-n:i+1]) for i in range(n, len(data))]
        return list(range(n, len(rewards))), running_avg

    running_x, running_y = running_avg(rewards)

    plt.plot(list(range(len(rewards))), rewards, alpha=0.5, label="Total Pursuer Reward") # Plot raw rewards
    plt.plot(running_x, running_y, color="orangered", label=f"Running Avg (last {window})") # Plot running average
    plt.title("Pursuer Training")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    # Vertical line
    plt.axvline(x=100, color='orange', linestyle='--', linewidth=2)
    plt.text(100 + 1, max(rewards)*0.8, 'Evader stop training', rotation=90, verticalalignment='center', color='orange')

    plt.show()

def visualize_policy(env, actors):
    ''' Perform one rollout of environment. '''
    observations, _ = env.reset()

    # Constants
    p_names = [agent for agent in env.agents if 'adversary' in agent]
    e_names = [agent for agent in env.agents if 'agent' in agent]

    graphs = GraphEncoder(p_names, e_names, observations)

    total_reward = 0
    while True:
        env.render()
        # 1.1: Convert observations to tensors
        p_obs = torch.tensor(np.array([observations[p].copy() for p in p_names]))
        e_obs = torch.tensor(np.array([observations[e].copy() for e in e_names]))

        # 1.2: Get actions (don't need to be clipped)
        p_actions = actors['pursuer'](p_obs, graphs.p_graph).detach().numpy() # Get pursuer actions. Shape (num_pursuer, 2)
        e_actions = actors['evader'](e_obs, graphs.p_graph).detach().numpy() # Get evader actions. Shape (num_evader, 2)

        # Convert action list to dict
        env_actions = convert_actions_to_env(p_names, e_names, p_actions, e_actions)

        # Take action
        next_observations, rewards, terminations, truncations, _ = env.step(env_actions)
        done = terminations[p_names[0]] or truncations[p_names[0]]
        total_reward += rewards[p_names[0]]

        if done:
            break
        observations = next_observations
        graphs.update_graphs(next_observations)

    print(f"Total Reward: {total_reward:.0f}")
    env.close()

def evaluate_policy(env, actors, num_rollouts = 100):
    ''' Average over rollouts '''
    observations, _ = env.reset()

    # Constants
    p_names = [agent for agent in env.agents if 'adversary' in agent]
    e_names = [agent for agent in env.agents if 'agent' in agent]

    total_reward = 0
    for _ in range(num_rollouts):
        observations, _ = env.reset() # Reset env
        graphs = GraphEncoder(p_names, e_names, observations)

        while True:
            # 1.1: Convert observations to tensors
            p_obs = torch.tensor(np.array([observations[p].copy() for p in p_names]))
            e_obs = torch.tensor(np.array([observations[e].copy() for e in e_names]))

            # 1.2: Get actions (don't need to be clipped)
            p_actions = actors['pursuer'](p_obs, graphs.p_graph).detach().numpy() # Get pursuer actions. Shape (num_pursuer, 2)
            e_actions = actors['evader'](e_obs, graphs.e_graph).detach().numpy() # Get evader actions. Shape (num_evader, 2)

            # Convert action list to dict
            env_actions = convert_actions_to_env(p_names, e_names, p_actions, e_actions)

            # Take action
            next_observations, rewards, terminations, truncations, _ = env.step(env_actions)
            done = terminations[p_names[0]] or truncations[p_names[0]]
            total_reward += rewards[p_names[0]]

            if done:
                break
            observations = next_observations
            graphs.update_graphs(next_observations)

    mean_reward = total_reward / num_rollouts
    print(f"Avg Reward over {num_rollouts} trials: {mean_reward:.2f}")
    env.close()