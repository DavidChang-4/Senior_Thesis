##################
# Imports
##################
from pettingzoo.mpe import simple_tag_v3
import numpy as np

import time

##################
# Constants and Setup
##################

# Environment Constants
RENDER_MODE = None # None or "human"
N_PURSUERS = 1 # OpenAI 'adversaries'
N_EVADERS =  1 # OpenAI 'agents'
N_OBSTACLES = 0
MAX_CYCLES = 1000 # Default 25
CONTINUOUS_ACTIONS = True

# Observations
# Adversary: 8 + 2(adversary-1) + 4(agent-1)
# Agent: 6 + 2(adversary-1) + 4(agent-1)



# Create environment
env = simple_tag_v3.parallel_env(num_good = N_EVADERS,
                                 num_adversaries = N_PURSUERS,
                                 num_obstacles = N_OBSTACLES,
                                 max_cycles = MAX_CYCLES,
                                 render_mode = RENDER_MODE,
                                 continuous_actions = CONTINUOUS_ACTIONS)

observations, infos = env.reset()

reward_adversary_accum = 0
rewared_agent_accum = 0
##################
# Simulation Loop
##################
while env.agents:
    # Shape Discrete(5). E.g. actions['agent_0']: 4
    # 0 = no movement. 1 = left, 2 = right, 3 = down, 4 = up (positive)
    actions = {agent: env.action_space(agent).sample() for agent in env.agents} # this is where you would insert your policy
    actions['adversary_0'][0] = 0 # none
    actions['adversary_0'][1] = 0 # left
    actions['adversary_0'][2] = 0 # right
    actions['adversary_0'][3] = 0 # down
    actions['adversary_0'][4] = 0 # up

    actions['agent_0'][0] = 0 # none
    actions['agent_0'][1] = 0 # left
    actions['agent_0'][2] = 1 # right
    actions['agent_0'][3] = 0 # down
    actions['agent_0'][4] = 0 # up

    print(observations['agent_0'])
    # observations: (x, y) pairs: [self_vel, self_pos, landmark_rel, other_adv_rel, other_agent_rel, other_agent_vel]
    # rewards: floats | terminations: boolean | truncations: boolean | infos: debugging
    observations, rewards, terminations, truncations, infos = env.step(actions)
    # print(observations['adversary_0'][0:4])
    reward_adversary_accum += rewards['adversary_0']
    rewared_agent_accum += rewards['agent_0']

print(reward_adversary_accum)
print(rewared_agent_accum)
env.close()