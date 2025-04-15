####################################
# Imports
####################################
from pettingzoo.mpe import simple_tag_v3
import torch

from DDPG_train import teams, train
from DDPG_graph import NODE_DIM
from DDPG_models import Actor, Critic
from DDPG_utils import ReplayBuffer, load_models, load_actors, visualize_rewards, visualize_policy, evaluate_policy

import random

####################################
# Environment Creation
####################################
SEED = random.randint(0, 1_000_000)
torch.manual_seed(SEED)

# Hyperparams: environment
RENDER_MODE = None # None or "human"
N_PURSUERS = 6 # PettingZoo 'adversaries'
N_EVADERS =  1 # PettingZoo 'agents'
N_OBSTACLES = 0
MAX_CYCLES = 100 # Default 25
CONTINUOUS_ACTIONS = True

# Create environment
env = simple_tag_v3.parallel_env(num_adversaries = N_PURSUERS, num_good = N_EVADERS,
                                 num_obstacles = N_OBSTACLES, max_cycles = MAX_CYCLES,
                                 continuous_actions = CONTINUOUS_ACTIONS,
                                 render_mode = RENDER_MODE)

visualize_env = simple_tag_v3.parallel_env(num_adversaries = N_PURSUERS, num_good = N_EVADERS,
                                 num_obstacles = N_OBSTACLES, max_cycles = MAX_CYCLES,
                                 continuous_actions = CONTINUOUS_ACTIONS,
                                 render_mode = "human") # Rendering on

####################################
# Initialize Networks, Optimizers, Buffers
####################################
# Hyperparams: ReplayBuffer
BUFFER_SIZE = 1_000_000
BATCH_SIZE = 128

actors, target_actors, critics, target_critics = {}, {}, {}, {}
for team in teams:
    if team == 'pursuer':
        in_dim = 4 + 2*(N_PURSUERS-1) + 4*(N_EVADERS)
    else:
        in_dim = 4 + 2*(N_PURSUERS) + 4*(N_EVADERS-1)

    actors[team] = Actor(team, N_PURSUERS, N_EVADERS, in_dim)
    target_actors[team] = Actor(team, N_PURSUERS, N_EVADERS, in_dim)
    critics[team] = Critic(team, N_PURSUERS, N_EVADERS, in_dim)
    target_critics[team] = Critic(team, N_PURSUERS, N_EVADERS, in_dim)
    
    # Copy params
    target_actors[team].load_state_dict(actors[team].state_dict())
    target_critics[team].load_state_dict(critics[team].state_dict())

actor_optims, critic_optims = {}, {}
for team in teams:
    actor_optims[team] = torch.optim.Adam(actors[team].parameters(), lr=1e-4)
    critic_optims[team] = torch.optim.Adam(critics[team].parameters(), lr=3e-4)

buffers = {}
for team in teams:
    buffers[team] = ReplayBuffer(capacity=BUFFER_SIZE, batch_size=BATCH_SIZE)

####################################
# Training
####################################
# Hyperparams: train
NUM_EPISODES = 300 # 200
NUM_SAMPLES_COLLECT = 1_000
NUM_OPTIM_STEPS = 100
GAMMA = 0.99
TAU = 0.005
# Hyperparams: save
model_name = "GNN_3depth_512cell_XIter"

# Start training!
# train(env, MAX_CYCLES,
#       actors, target_actors, critics, target_critics, 
#       actor_optims, critic_optims, buffers,
#       NUM_EPISODES, NUM_SAMPLES_COLLECT, BATCH_SIZE, NUM_OPTIM_STEPS, GAMMA, TAU,
#       model_name)

####################################
# Visualize Results
####################################
# visualize_rewards() # Show reward graph
load_actors(actors, model_name) # Load models to visualize policy.
evaluate_policy(env, actors)

# visualize_policy(visualize_env, actors) # Show one rollout