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
def create_graph(observations, pursuer_names, evader_names) -> Data:
    '''
    Input: environment observations.
    Output: returns a graph. Encodes agents->nodes and distances->edges.
    '''
    N_PURSUERS, N_EVADERS = len(pursuer_names), len(evader_names)
    node_attr, edge_index, edge_attr = [], [], []

    # Include pursuers in graph
    for pursuer_id in pursuer_names:
        feature = np.append([1], observations[pursuer_id][0:4])
        node_attr.append(feature)
    
    # Include evaders in graph
    for evader_id in evader_names:
        feature = np.append([0], observations[evader_id][0:4])
        node_attr.append(feature)
    node_attr = np.array(node_attr)
    
    # Create edges
    for i in range(N_PURSUERS):
        for j in range(i+1, N_PURSUERS + N_EVADERS):
            # dist = 1/np.linalg.norm(node_attr[i, 3:5] - node_attr[j, 3:5])

            if j<N_PURSUERS: # Pursuer <-> pursuer connection
                edge_index.append([i, j])
                edge_attr.append([0])
                edge_index.append([j, i])
                edge_attr.append([0])
            else: # Evader -> Pursuer connection
                edge_index.append([j, i])
                edge_attr.append([1])
    
    # Convert to tensors
    node_attr = torch.tensor(node_attr, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # return Data(x=node_attr, edge_index=edge_index)
    return Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)

def update_graph(graph, observations, pursuer_names, evader_names) -> None:
    '''
    Input: graph, new observations. 
    Output: update nodes (self_vel, self_pos), and recompute edge_attr distances.
    '''
    N_PURSUERS, N_EVADERS = len(pursuer_names), len(evader_names)

    # Update pursuers
    for i, pursuer_id in enumerate(pursuer_names):
        graph.x[i, 1:5] = torch.tensor(observations[pursuer_id][0:4], dtype=torch.float)
    
    # Update evaders 
    for i, evader_id in enumerate(evader_names):
        graph.x[N_PURSUERS + i, 1:5] = torch.tensor(observations[evader_id][0:4], dtype=torch.float)

    # Update edges: 
    # edge_i = 0
    # for i in range(N_PURSUERS):
    #     for j in range(i+1, N_PURSUERS + N_EVADERS):
    #         dist = 1 / np.linalg.norm(graph.x[i, 3:5] - graph.x[j, 3:5]) # self_pos

    #         if j<N_PURSUERS:
    #             graph.edge_attr[edge_i] = torch.tensor(dist, dtype=torch.float)
    #             edge_i += 1
    #             graph.edge_attr[edge_i] = torch.tensor(dist, dtype=torch.float)
    #             edge_i += 1
    #         else:
    #             graph.edge_attr[edge_i] = torch.tensor(dist, dtype=torch.float)
    #             edge_i += 1

    return None

def check_graph(graph) -> None:
    ''' 
    Input: graph.
    Output: check graph has no self-loops. 
    '''
    assert(not graph.has_self_loops()) 
    print("Graph check: Passed")

    return None

def visualize_graph(graph, to_undirected=False) -> None:
    '''
    Input: graph.
    Output: plots the graph. 
    '''
    G = to_networkx(graph, to_undirected=to_undirected)
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False, node_color=graph.y, cmap="Set1")
    plt.show()

    return None