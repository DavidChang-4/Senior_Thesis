####################################
# Imports
####################################
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from PPO_graph import create_graph, update_graph, visualize_graph
from PPO_utils import convert_action_to_env, compute_returns, save_models

import numpy as np
import time

teams = ['pursuer', 'evader']
teams_training = ['pursuer']

####################################
# Training Loop
####################################

def distance_reward(agent_pos, target_pos):
    '''
    Parameters: agent_pos: (x, y), target_pos (x, y)
    Returns: reward where closer distances yield a higher reward (up to 3).
    '''
    distance = np.linalg.norm(np.array(agent_pos) - np.array(target_pos))
    
    # Use the function 3/(1+d) which gives 5 at d=0 and decays as d increases.
    reward = 3.0 / (1.0 + distance)
    return reward

# Training loop
def train(env, MAX_CYCLES, 
          actorcritics, buffers,
          NUM_EPISODES, NUM_SAMPLES_COLLECT, NUM_OPTIM_STEPS, BATCH_SIZE,
          GAMMA, EPS_CLIP, LAMBDA, ENTROPY_COEFF,
          model_name):
    env.reset() # Needed to get attributes
    
    # Constants
    pursuer_names = [agent for agent in env.agents if 'adversary' in agent]
    evader_names = [agent for agent in env.agents if 'agent' in agent]
    N_PURSUERS, N_EVADERS = len(pursuer_names), len(evader_names)

    # Track rewards, time
    reward_history = []
    time_accum = 0

    for episode in range(NUM_EPISODES):
        start_time = time.perf_counter()
        
        for team in teams:
            buffers[team].clear()
        # Stage 1: Collect experience
        num_samples_collected = reward_accum = new_reward_accum = 0
        while num_samples_collected < NUM_SAMPLES_COLLECT:
            observations, _ = env.reset() # Reset env
            graph = create_graph(observations, pursuer_names, evader_names)

            while True:
                # 1.1: Select actions
                p_obs = torch.tensor(np.array([observations[p].copy() for p in pursuer_names]))
                e_obs = torch.tensor(np.array([observations[e].copy() for e in evader_names]))

                p_actions, p_logprobs, p_values = actorcritics['pursuer'].get_action(p_obs, graph) # Get pursuer actions. Shape (num_pursuer, 2)
                e_actions, e_logprobs, e_values = actorcritics['evader'].get_action(e_obs, graph) # Get evader actions. Shape (num_evader, 2)

                # Convert action list to dict
                env_actions = convert_action_to_env(pursuer_names, evader_names, p_actions, e_actions)

                # 1.2: Take action and observe result 
                next_observations, rewards, terminations, truncations, _ = env.step(env_actions)
                done = terminations[pursuer_names[0]] or truncations[pursuer_names[0]]
                reward_accum += rewards[pursuer_names[0]]

                # Get distance reward
                p_rewards_list = []
                for i, pursuer in enumerate(pursuer_names):
                    dist_reward = distance_reward(p_obs[i][2:4], e_obs[0][2:4])
                    p_rewards_list.append(rewards[pursuer]+dist_reward)

                # 1.3: Store transition in RB
                p_rewards = torch.tensor(p_rewards_list).unsqueeze(1)
                e_rewards = torch.full((N_EVADERS,), rewards[evader_names[0]], dtype=torch.float).unsqueeze(1)

                mask = 1 - int(done)
                p_done = torch.full((N_PURSUERS,), mask, dtype=torch.int).unsqueeze(1)
                e_done = torch.full((N_EVADERS,), mask, dtype=torch.int).unsqueeze(1)

                # Update graph
                old_graph = Data.clone(graph)
                update_graph(graph, next_observations, pursuer_names, evader_names)

                buffers['pursuer'].push(p_obs, p_actions, p_logprobs.unsqueeze(1), p_values, p_rewards, p_done, old_graph) 
                buffers['evader'].push(e_obs, e_actions, e_logprobs.unsqueeze(1), e_values, e_rewards, e_done, old_graph)
                num_samples_collected += 1

                # Check 'done', otherwise update observations
                if done:
                    break
                observations = next_observations

        mean_episode_reward = reward_accum / (NUM_SAMPLES_COLLECT // MAX_CYCLES)

        # Stage 2: Compute returns --> advantages
        for team in teams_training:
            pursuer_obs = torch.tensor(np.array([next_observations[p].copy() for p in pursuer_names]))
            evader_obs = torch.tensor(np.array([next_observations[e].copy() for e in evader_names]))
            with torch.no_grad():
                if team == teams[0]:
                    _, next_value = actorcritics[team](pursuer_obs, graph)
                else:
                    _, next_value = actorcritics[team](evader_obs, graph)

            returns = compute_returns(next_value, buffers[team].rewards, buffers[team].values, buffers[team].done, GAMMA, LAMBDA)
            
            advantages = [a - b for a, b in zip(returns, buffers[team].values)]
            advantages = torch.stack(advantages)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Normalize
            buffers[team].push_advantages(advantages, returns)

        # Stage 3: Learn from collected samples
        for _ in range(NUM_OPTIM_STEPS):
            for team in teams_training: 

                # Shuffle indices
                indices = np.arange(buffers[team].__len__())
                np.random.shuffle(indices)

                for start_i in range(0, NUM_SAMPLES_COLLECT, BATCH_SIZE):
                    end_i = start_i + BATCH_SIZE
                    idx = indices[start_i:end_i] 

                    # 1. Sample from RB (batch)
                    obs_batch, action_batch, old_logprob_batch, advantage_batch, return_batch, old_graph_batch  = buffers[team].sample_idx(idx)
                    # 2. Actor loss
                    dist, state_val = actorcritics[team](obs_batch, old_graph_batch, batch_size = BATCH_SIZE)
                    new_log_probs = dist.log_prob(action_batch).sum(dim=-1, keepdim=True)
                    
                    # Ratio for clipping.
                    ratio = torch.exp(new_log_probs - old_logprob_batch)
                    surr1 = ratio * advantage_batch
                    surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantage_batch
                    
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # 3. Critic loss
                    critic_loss = (return_batch - state_val).pow(2).mean()
                    
                    # 4. Total loss
                    entropy_loss = dist.entropy().mean()
                    loss = actor_loss + 0.5 * critic_loss - ENTROPY_COEFF * entropy_loss
                    
                    actorcritics[team].optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actorcritics[team].parameters(), max_norm=0.5)
                    actorcritics[team].optimizer.step()
        
        if episode == (NUM_EPISODES // 3) and len(teams_training)>1:
            teams_training.remove('evader')

        episode_time = time.perf_counter() - start_time
        time_accum += episode_time
        print(f"Episode {episode+1} | Avg Reward: {mean_episode_reward:.2f} | Time: {episode_time:.2f} | Avg Time: {(time_accum/(episode+1)):.2f}")
        reward_history.append(mean_episode_reward)
        np.save("GNN-PPO/rewards", reward_history)

        # Save actor and critic models
        if (episode+1) % 10 == 0:
            save_models(actorcritics, model_name)  
        
