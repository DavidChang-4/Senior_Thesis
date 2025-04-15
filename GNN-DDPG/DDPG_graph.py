####################################
# Imports
####################################
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx

import numpy as np
from matplotlib import pyplot as plt

NODE_DIM = 5 # Node: [team id, self vel, self pos]

####################################
# Graph Processing
####################################
class GraphEncoder:
    def __init__(self, p_names, e_names, observations):
        ''' Initialize a graph. '''
        self.p_names, self.e_names = p_names, e_names

        self.p_graph = self.create_graph(observations, "pursuer")
        self.e_graph = self.create_graph(observations, "evader")

    def create_graph(self, observations, team) -> Data:
        ''' Creates intial graph. '''
        N_PURSUERS, N_EVADERS = len(self.p_names), len(self.e_names)
        node_attr, edge_index, edge_attr = [], [], []

        # Create node features in graph
        node_attr = ([np.concatenate(([1], observations[p_id][:4])) for p_id in self.p_names] +
                    [np.concatenate(([0], observations[e_id][:4])) for e_id in self.e_names])
        node_attr = np.array(node_attr)
        
        # Create edges 
        same_team, diff_team = 0, 1
        if team == "pursuer": # Pursuer
            for i in range(N_PURSUERS): # 'i' is always pursuer
                for j in range(i+1, N_PURSUERS + N_EVADERS): 
                    # dist = 1/np.linalg.norm(node_attr[i, 3:5] - node_attr[j, 3:5])
                    if j<N_PURSUERS: # 'j' is pursuer
                        edge_index.append([i, j])
                        edge_attr.append([same_team])
                        edge_index.append([j, i])
                        edge_attr.append([same_team])
                    else: # 'j' is evader
                        edge_index.append([j, i])
                        edge_attr.append([diff_team])
        else: # Evader/Critic graph: fully connected
            # 1. Evader <-> evader
            for i in range(N_PURSUERS + N_EVADERS):
                for j in range(N_PURSUERS + N_EVADERS):
                    if i==j: # No self loops
                        continue
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    if (i<N_PURSUERS and j<N_PURSUERS) or (i>=N_PURSUERS and j>=N_PURSUERS): # Same team
                        edge_attr.append([same_team])
                        edge_attr.append([same_team])
                    else:
                        edge_attr.append([diff_team])
                        edge_attr.append([diff_team])
        
        # Convert to tensors
        node_attr = torch.tensor(node_attr, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)

    def update_graphs(self, next_observations) -> None:
        ''' Update nodes with new (self_vel, self_pos).'''
        N_PURSUERS = len(self.p_names)

        # Update pursuers
        for i, pursuer_id in enumerate(self.p_names):
            self.p_graph.x[i, 1:5] = torch.tensor(next_observations[pursuer_id][0:4], dtype=torch.float)
            self.e_graph.x[i, 1:5] = torch.tensor(next_observations[pursuer_id][0:4], dtype=torch.float)
        
        # Update evaders 
        for i, evader_id in enumerate(self.e_names):
            self.p_graph.x[N_PURSUERS + i, 1:5] = torch.tensor(next_observations[evader_id][0:4], dtype=torch.float)
            self.e_graph.x[N_PURSUERS + i, 1:5] = torch.tensor(next_observations[evader_id][0:4], dtype=torch.float)

        return None

def check_graph(graph) -> None:
    ''' Check graph has no self-loops. '''
    assert(not graph.has_self_loops()) 
    print("Graph check: Passed")
    return None

def visualize_graph(graph, to_undirected=False) -> None:
    ''' Visualize the graph. '''
    G = to_networkx(graph, to_undirected=to_undirected)
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False, node_color=graph.y, cmap="Set1")
    plt.show()
    return None