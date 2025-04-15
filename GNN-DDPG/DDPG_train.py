####################################
# Imports
####################################
import torch
from torch_geometric.data import Data

from DDPG_graph import GraphEncoder, visualize_graph
from DDPG_utils import get_Gaussian_noise, convert_actions_to_env, save_models

import numpy as np
import time

teams = ['pursuer', 'evader']
teams_training = ['pursuer', 'evader']

####################################
# Training Loop
####################################
def train(env, MAX_CYCLES,
          actors, target_actors, critics, target_critics, 
          actor_optims, critic_optims, buffers,
          NUM_EPISODES, NUM_SAMPLES_COLLECT, BATCH_SIZE, NUM_OPTIM_STEPS, GAMMA, TAU,
          model_name):
    env.reset() # Needed to get attributes
    
    # Constants
    p_names = [agent for agent in env.agents if 'adversary' in agent]
    e_names = [agent for agent in env.agents if 'agent' in agent]
    N_PURSUERS, N_EVADERS = len(p_names), len(e_names)

    # Track rewards
    reward_history = []
    time_accum = 0

    for episode in range(NUM_EPISODES):
        start_time = time.perf_counter()
        num_samples_collected = total_reward = 0

        # Stage 1: Collect experience
        while num_samples_collected < NUM_SAMPLES_COLLECT:
            observations, _ = env.reset() # Reset env
            graphs = GraphEncoder(p_names, e_names, observations)

            while True:
                # 1.1: Convert observations to tensors
                p_obs = torch.tensor(np.array([observations[p].copy() for p in p_names]))
                e_obs = torch.tensor(np.array([observations[e].copy() for e in e_names]))

                # 1.2: Get actions from actor networks
                p_actions = actors['pursuer'](p_obs, graphs.p_graph).detach().numpy() # Get pursuer actions. Shape (num_pursuer, 2)
                e_actions = actors['evader'](e_obs, graphs.e_graph).detach().numpy() # Get evader actions. Shape (num_evader, 2)

                # Add Gaussian noise to actions
                p_actions += get_Gaussian_noise(episode, NUM_EPISODES, N_PURSUERS)
                e_actions += get_Gaussian_noise(episode, NUM_EPISODES, N_EVADERS)

                # Clip actions to valid range [-1, 1]
                p_actions = torch.clip(torch.tensor(p_actions), -1, 1)
                e_actions = torch.clip(torch.tensor(e_actions), -1, 1)

                # Convert action list to dict
                env_actions = convert_actions_to_env(p_names, e_names, p_actions, e_actions)

                # 1.3: Take step in environment and observe result 
                next_observations, rewards, terminations, truncations, _ = env.step(env_actions)
                done = terminations[p_names[0]] or truncations[p_names[0]]
                total_reward += rewards[p_names[0]] # Track total reward in this rollout. 

                # Convert next observations to tensors
                p_obs_next = torch.tensor(np.array([next_observations[p].copy() for p in p_names]))
                e_obs_next = torch.tensor(np.array([next_observations[e].copy() for e in e_names]))

                p_rewards = torch.tensor(np.array([rewards[p_names[0]]] * N_PURSUERS), dtype=torch.float)
                e_rewards = torch.tensor(np.array([rewards[e_names[0]]] * N_EVADERS), dtype=torch.float)
                p_done = torch.tensor(np.array([done] * N_PURSUERS), dtype=torch.int)
                e_done = torch.tensor(np.array([done] * N_EVADERS), dtype=torch.int)

                # Update graph
                p_old_graph = graphs.p_graph.clone()
                e_old_graph = graphs.e_graph.clone()
                graphs.update_graphs(next_observations)
                p_next_graph = graphs.p_graph.clone()
                e_next_graph = graphs.e_graph.clone()

                # 1.3: store transition in RB
                buffers['pursuer'].push(p_obs, p_obs_next, p_actions, p_rewards, p_done, p_old_graph, p_next_graph) 
                buffers['evader'].push(e_obs, e_obs_next, e_actions, e_rewards, e_done, e_old_graph, e_next_graph) # TODO: evaders have their own graph
                num_samples_collected += 1

                # Check 'done', otherwise update observations
                if done:
                    break
                observations = next_observations

        mean_episode_reward = total_reward / (NUM_SAMPLES_COLLECT // MAX_CYCLES)

        # Stage 2: Learn from collected samples.
        for _ in range(NUM_OPTIM_STEPS):
            for team in teams_training: 
                # 2.1: Sample minibatch RB
                obs_batch, next_obs_batch, actions_batch, reward_batch, done_batch, graph_batch, next_graph_batch = buffers[team].sample_vectorized()
                
                # 2.2: Run DDPG
                # --- Compute target Q-val (vectorized) ---
                with torch.no_grad():
                    next_actions_batch = target_actors[team](next_obs_batch, next_graph_batch, batch_size=BATCH_SIZE)
                    target_q_batch = reward_batch + (1 - done_batch) * GAMMA * target_critics[team](next_obs_batch, next_graph_batch, next_actions_batch, BATCH_SIZE)

                curr_q_batch = critics[team](obs_batch, graph_batch, actions_batch, BATCH_SIZE)
                critic_loss = torch.nn.functional.mse_loss(curr_q_batch, target_q_batch.detach())
                critic_optims[team].zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critics[team].parameters(), max_norm=1.0)
                critic_optims[team].step()

                # --- Update Actor (vectorized) ---
                actor_actions = actors[team](obs_batch, graph_batch, batch_size=BATCH_SIZE)
                actor_loss = -critics[team](obs_batch, graph_batch, actor_actions, BATCH_SIZE).mean()
                actor_optims[team].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actors[team].parameters(), max_norm=1.0)
                actor_optims[team].step()

                # --- Soft Updates ---
                # Actors
                for target_param, source_param in zip(target_actors[team].parameters(), actors[team].parameters()):
                    target_param.data.copy_(TAU * source_param.data + (1.0 - TAU) * target_param.data)
                # Critics
                for target_param, source_param in zip(target_critics[team].parameters(), critics[team].parameters()):
                    target_param.data.copy_(TAU * source_param.data + (1.0 - TAU) * target_param.data)
        
        # Remove evader from training
        if episode == (NUM_EPISODES // 3) and len(teams_training)>1:
            teams_training.remove('evader')


        # Report statistics
        episode_time = time.perf_counter() - start_time
        time_accum += episode_time
        print(f"Episode {episode+1} | Avg Reward: {mean_episode_reward:.0f} | Time: {episode_time:.2f} | Avg Time: {(time_accum/(episode+1)):.2f}")
        reward_history.append(mean_episode_reward)
        np.save("GNN-DDPG/rewards", reward_history)
        
        # Save models every 10 episodes. 
        if (episode+1) % 10 == 0:
            save_models(actors, critics, model_name)  