####################################
# Imports
####################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch_geometric.nn import GCNConv, GATConv, TransformerConv

from egnn_layers import GCL, E_GCL, E_GCL_vel, GCL_rf_vel

import numpy as np

####################################
# Actor and Critic: MLP
####################################
class ActorCritic(nn.Module):
    def __init__(self, in_dim, action_dim=2, mlp_hidden_dim=512, lr=3e-4):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden_dim), # Hidden Layer 1
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim), # Hidden Layer 2
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, action_dim * 2) # Output action mean + stdev
        )
        
        self.critic = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden_dim), # Hidden Layer 1
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim), # Hidden Layer 2
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1), # Output 1 value
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)

    def forward(self, observation, graph, batch_size=1):
        actor_out = self.actor(observation)
        mean, log_std = torch.chunk(actor_out, 2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2) # Clamp log_std
        std = torch.exp(log_std)
        return Normal(mean, std), self.critic(observation) # Return distribution, state-value
    
    def get_action(self, observation, graph, batch_size=1):
        # Get distrib
        dist, state_val = self.forward(observation, graph, batch_size=batch_size)

        # Get dist, action
        action = torch.tanh(dist.rsample()) # Ensure in [-1, 1]
        return action.detach(), dist.log_prob(action).sum(dim=-1).detach(), state_val.detach() # Return action, log-prob, state-value
    
####################################
# Actor and Critic: GNN
####################################
class ActorCritic_GNN(nn.Module):
    def __init__(self, team, N_PURSUERS, N_EVADERS, in_dim, action_dim=2, mlp_hidden_dim=512, lr=3e-4):
        super(ActorCritic_GNN, self).__init__()

        self.team = team
        self.p, self.e = N_PURSUERS, N_EVADERS

        self.actor_gnn = GATConv(5, 128, heads=4, concat=True)
        self.actor = nn.Sequential(
            nn.Linear(128*4, mlp_hidden_dim), # Hidden Layer 1
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim), # Hidden Layer 2
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, action_dim * 2) # Output action mean + stdev
        )
        
        self.critic_gnn = GATConv(5, 128, heads=3, concat=False)
        self.critic = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden_dim), # Hidden Layer 1
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim), # Hidden Layer 2
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1), # Output 1 value
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)

    def forward(self, observation, graph, batch_size=1):
        idx = np.arange(len(graph.x))
        pursuer_idx, evader_idx = idx[idx % (self.p + self.e) < self.p], idx[idx % (self.p + self.e) >= self.p] # Team indices
        team_size = self.p if self.team=="pursuer" else self.e # Number of agents in team
        team_idx = pursuer_idx if self.team=="pursuer" else evader_idx # Indices for agents in team

        # Actor
        # actor_out = self.actor(observation)
        # mean, log_std = torch.chunk(actor_out, 2, dim=-1)
        # log_std = torch.clamp(log_std, min=-20, max=2) # Clamp log_std
        # std = torch.exp(log_std)

        # Actor
        actor_node_embeddings = self.actor_gnn(graph.x, graph.edge_index, graph.edge_attr) # GNN Stage
        actor_out = self.actor(actor_node_embeddings[team_idx]) # MLP Stage
        mean, log_std = torch.chunk(actor_out, 2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2) # Clamp log_std
        std = torch.exp(log_std)
        if batch_size>1:
            mean = mean.reshape(batch_size, team_size, 2)
            std = std.reshape(batch_size, team_size, 2)

        # Critic
        # critic_node_embeddings = self.critic_gnn(graph.x, graph.edge_index, graph.edge_attr) # GNN Stage
        # critic_vals = self.critic(critic_node_embeddings[team_idx]) # MLP Stage
        # if batch_size>1: # Batch reshape
        #     critic_vals = critic_vals.reshape(batch_size, team_size, 1)
        
        return Normal(mean, std), self.critic(observation)
        # return Normal(mean, std), critic_vals # Return distribution, state-value
    
    def get_action(self, observation, graph, batch_size=1):
        # Get distrib
        dist, state_val = self.forward(observation, graph, batch_size=batch_size)

        # Get dist, action
        action = torch.tanh(dist.rsample()) # Ensure in [-1, 1]
        return action.detach(), dist.log_prob(action).sum(dim=-1).detach(), state_val.detach() # Return action, log-prob, state-value