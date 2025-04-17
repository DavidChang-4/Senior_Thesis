####################################
# Imports
####################################
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.data import Data

import numpy as np

ACTION_DIM = 2 # [vertical, horizontal]

####################################
# Actor
####################################
class Actor_GNN(nn.Module):
    def __init__(self, team, N_PURSUERS, N_EVADERS, in_dim, mlp_hidden_dim=512):
        super(Actor_GNN, self).__init__()
        self.team = team
        self.p, self.e = N_PURSUERS, N_EVADERS
        
        # GNN Approach
        self.gnn1 = GATConv(5, 128, heads=4, concat=True, aggr="max")
        self.gnn2 = GATConv(128*4, 128, heads=4, concat=True, aggr="max")
        self.mlp1 = nn.Sequential(
            nn.Linear(128*4, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, ACTION_DIM), # Output 1 value
            nn.Tanh()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(5, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, ACTION_DIM), # Output 1 value
            nn.Tanh()
        )

    def forward(self, observation, graph, batch_size=1):
        # GNN Team Setup
        idx = np.arange(len(graph.x))
        pursuer_idx, evader_idx = idx[idx % (self.p + self.e) < self.p], idx[idx % (self.p + self.e) >= self.p] # Team indices
        team_size = self.p if self.team=="pursuer" else self.e # Number of agents in team
        team_idx = pursuer_idx if self.team=="pursuer" else evader_idx # Indices for agents in team

        # GNN Stage
        if len(graph.edge_attr > 0):
            node_embeddings = self.gnn1(graph.x, graph.edge_index, graph.edge_attr) # GNN Stage
            node_embeddings = self.gnn2(node_embeddings, graph.edge_index, graph.edge_attr)
            mlp_out = self.mlp1(node_embeddings[team_idx])
        else:
            mlp_out = self.mlp2(graph.x[team_idx])

        if batch_size>1: 
            mlp_out = mlp_out.view(batch_size, team_size, 2)
        
        return mlp_out

        # MLP Implementation
        return self.mlp(observation)
    
class Actor_MLP(nn.Module):
    def __init__(self, team, N_PURSUERS, N_EVADERS, in_dim, mlp_hidden_dim=512):
        super(Actor_MLP, self).__init__()

        # MLP Approach
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden_dim), # changed from 128
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim), # remove layer when doing GNN
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, ACTION_DIM), # Output action
            nn.Tanh() # map to [-1, 1]
        )

    def forward(self, observation, graph, batch_size=1):
        # MLP Implementation
        return self.mlp(observation)

####################################
# Critic
####################################
class Critic(nn.Module):
    def __init__(self, team, N_PURSUERS, N_EVADERS, in_dim, mlp_hidden_dim=512, gnn_hidden_dim=64):
        super(Critic, self).__init__()
        self.team = team
        self.p, self.e = N_PURSUERS, N_EVADERS

        self.mlp1_out_dim = 32

        # GNN Approach
        self.gnn = GATConv(5, 64, heads=4, concat=True)
        self.mlp1 = nn.Sequential(
            nn.Linear(64*4, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, self.mlp1_out_dim), # Output 1 value
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(self.mlp1_out_dim + ACTION_DIM, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            # nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            # nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1), # Output 1 value
        )
        # MLP Approach
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + ACTION_DIM, mlp_hidden_dim), # changed from 128
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim), # remove layer when doing GNN
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1), # Output action
        )


    def forward(self, observation, graph, action, BATCH_SIZE): # Critic is batched by default
        # GNN Implementation
        # idx = np.arange(len(graph.x))
        # pursuer_idx, evader_idx = idx[idx % (self.p + self.e) < self.p], idx[idx % (self.p + self.e) >= self.p] # Team indices
        # team_size = self.p if self.team=="pursuer" else self.e # Number of agents in team
        # team_idx = pursuer_idx if self.team=="pursuer" else evader_idx # Indices for agents in team

        # # Critic
        # node_embeddings = self.gnn(graph.x, graph.edge_index, graph.edge_attr) # GNN Stage
        # mlp1_out = self.mlp1(node_embeddings[team_idx])
        # mlp1_out = mlp1_out.view(BATCH_SIZE, team_size, self.mlp1_out_dim)

        # values2 = torch.cat((mlp1_out, action), dim=2) # Shape: [batch_size, num_agents, in_dim+action]
        # return self.mlp2(values2)

        # MLP Implementation
        input_tensor = torch.cat((observation, action), dim=2) # Shape: [batch_size, num_agents, in_dim+action]
        return self.mlp(input_tensor)
