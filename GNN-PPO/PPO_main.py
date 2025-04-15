####################################
# Imports
####################################
from pettingzoo.mpe import simple_tag_v3
import torch

from PPO_train import teams, train
from PPO_models import ActorCritic, ActorCritic_GNN
from PPO_utils import ReplayBuffer, visualize_reward_graph, load_models, visualize_policy

import random

####################################
# Environment Creation
####################################
SEED = random.randint(0, 1_000_000)
torch.manual_seed(SEED)

# Hyperparams: environment
RENDER_MODE = None # None or "human"
N_PURSUERS = 2 # OpenAI 'adversaries'
N_EVADERS =  1 # OpenAI 'agents'
N_OBSTACLES = 0
MAX_CYCLES = 100 # Default 25
CONTINUOUS_ACTIONS = True

# Create environment
# Adversary Obs: 4 + 2(adversary-1) + 4(agent)
# Agent Obs: 4 + 2(adversary) + 4(agent-1)
env = simple_tag_v3.parallel_env(num_adversaries = N_PURSUERS, num_good = N_EVADERS,
                                 num_obstacles = N_OBSTACLES, max_cycles = MAX_CYCLES,
                                 continuous_actions = CONTINUOUS_ACTIONS, render_mode = RENDER_MODE)

visualize_env = simple_tag_v3.parallel_env(num_adversaries = N_PURSUERS, num_good = N_EVADERS,
                                 num_obstacles = N_OBSTACLES, max_cycles = MAX_CYCLES,
                                 continuous_actions = CONTINUOUS_ACTIONS, render_mode = "human") # Rendering on

####################################
# Initialize ActorCritic, Memory
####################################
BUFFER_SIZE = 6_000 # 6_000
BATCH_SIZE = 1_000 # 400

actorcritics = {}
for team in teams:
    if team == 'pursuer':
        in_dim = 4 + 2*(N_PURSUERS-1) + 4*(N_EVADERS)
    else:
        in_dim = 4 + 2*(N_PURSUERS) + 4*(N_EVADERS-1)
    # actorcritics[team] = ActorCritic(in_dim)
    actorcritics[team] = ActorCritic_GNN(team, N_PURSUERS, N_EVADERS, in_dim) 

buffers = {}
for team in teams:
    buffers[team] = ReplayBuffer(capacity=BUFFER_SIZE, batch_size=BATCH_SIZE)

####################################
# Training
####################################
# Hyperparams: train
NUM_EPISODES = 100
NUM_SAMPLES_COLLECT = BUFFER_SIZE # we use all the experience available. 
NUM_OPTIM_STEPS = 30
GAMMA = 0.99
LAMBDA = 0.90 # for GAE
EPS_CLIP = 0.2 
ENTROPY_COEFF = 1e-4
# Hyperparams: save
model_name = "GNN_3depth_512cell_XIter"

# Start training!
train(env, MAX_CYCLES, 
      actorcritics, buffers,
      NUM_EPISODES, NUM_SAMPLES_COLLECT, NUM_OPTIM_STEPS, BATCH_SIZE,
      GAMMA, EPS_CLIP, LAMBDA, ENTROPY_COEFF,
      model_name)


####################################
# Visualize Results
####################################
# visualize_reward_graph() # Show reward graph

# load_models(actorcritics, model_name) # Load models to visualize policy.
# visualize_policy(visualize_env, actorcritics) # Show one rollout