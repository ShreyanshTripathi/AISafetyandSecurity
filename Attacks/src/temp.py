import torch
from torch_geometric.data import InMemoryDataset, Data
import pandas as pd
import os
import pdb
import re
import numpy as np
import os.path as osp
from sklearn.model_selection import train_test_split
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx


def convert_ordering_to_edges(ordering, name_to_node):
    pdb.set_trace()
    nodes_p = re.findall(r"[\w'-]+", ordering)
    nodes = re.findall(r"[\w'-]+|=|>", ordering)
    edges = [[i, j] for i in range(len(nodes_p)) for j in range(len(nodes_p)) if i != j]
    edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
    edge_label = [0] * len(edges)
    prev_sign = None
    for i in range(0, len(nodes) - 2, 2):
        if [name_to_node[nodes[i]], name_to_node[nodes[i + 2]]] in edges:
            if name_to_node[nodes[i]] > name_to_node[nodes[i + 2]]:
                if nodes[i + 1] == ">":
                    edge_label[
                        edges.index(
                            [name_to_node[nodes[i]], name_to_node[nodes[i + 2]]]
                        )
                    ] = 1
                    if prev_sign == "=":
                        if name_to_node[nodes[i - 2]] > name_to_node[nodes[i + 2]]:
                            edge_label[
                                edges.index(
                                    [
                                        name_to_node[nodes[i - 2]],
                                        name_to_node[nodes[i + 2]],
                                    ]
                                )
                            ] = 1
                        else:
                            edge_label[
                                edges.index(
                                    [
                                        name_to_node[nodes[i - 2]],
                                        name_to_node[nodes[i + 2]],
                                    ]
                                )
                            ] = -1

                        try:
                            if name_to_node[nodes[i - 4]] > name_to_node[nodes[i + 2]]:
                                edge_label[
                                    edges.index(
                                        [
                                            name_to_node[nodes[i - 4]],
                                            name_to_node[nodes[i + 2]],
                                        ]
                                    )
                                ] = 1
                            else:
                                edge_label[
                                    edges.index(
                                        [
                                            name_to_node[nodes[i - 4]],
                                            name_to_node[nodes[i + 2]],
                                        ]
                                    )
                                ] = -1
                        except:
                            pass
                    prev_sign = ">"
                else:
                    if i >= 2:
                        if name_to_node[nodes[i - 2]] > name_to_node[nodes[i + 2]]:
                            edge_label[
                                edges.index(
                                    [
                                        name_to_node[nodes[i - 2]],
                                        name_to_node[nodes[i + 2]],
                                    ]
                                )
                            ] = 1
                        else:
                            edge_label[
                                edges.index(
                                    [
                                        name_to_node[nodes[i - 2]],
                                        name_to_node[nodes[i + 2]],
                                    ]
                                )
                            ] = -1
                    prev_sign = "="
            else:
                if nodes[i + 1] == ">":
                    edge_label[
                        edges.index(
                            [name_to_node[nodes[i]], name_to_node[nodes[i + 2]]]
                        )
                    ] = -1
                    if prev_sign == "=":
                        if name_to_node[nodes[i - 2]] > name_to_node[nodes[i + 2]]:
                            edge_label[
                                edges.index(
                                    [
                                        name_to_node[nodes[i - 2]],
                                        name_to_node[nodes[i + 2]],
                                    ]
                                )
                            ] = 1
                        else:
                            edge_label[
                                edges.index(
                                    [
                                        name_to_node[nodes[i - 2]],
                                        name_to_node[nodes[i + 2]],
                                    ]
                                )
                            ] = -1
                        try:
                            if name_to_node[nodes[i - 4]] > name_to_node[nodes[i + 2]]:
                                edge_label[
                                    edges.index(
                                        [
                                            name_to_node[nodes[i - 4]],
                                            name_to_node[nodes[i + 2]],
                                        ]
                                    )
                                ] = 1
                            else:
                                edge_label[
                                    edges.index(
                                        [
                                            name_to_node[nodes[i - 4]],
                                            name_to_node[nodes[i + 2]],
                                        ]
                                    )
                                ] = -1
                        except:
                            pass
                    prev_sign = ">"
                else:
                    if i >= 2:
                        if name_to_node[nodes[i - 2]] > name_to_node[nodes[i + 2]]:
                            edge_label[
                                edges.index(
                                    [
                                        name_to_node[nodes[i - 2]],
                                        name_to_node[nodes[i + 2]],
                                    ]
                                )
                            ] = 1
                        else:
                            edge_label[
                                edges.index(
                                    [
                                        name_to_node[nodes[i - 2]],
                                        name_to_node[nodes[i + 2]],
                                    ]
                                )
                            ] = -1
                    prev_sign = "="

    # Set feature vectors for nodes
    # feature_matrix = []
    # for node_name in nodes_p:
    #     feats = x[x["job-name"] == node_name]
    #     feats = feature_matrix.append(list(feats[self.feature_columns]))

    data = Data(edge_index=edge_index, edge_y=edge_label)
    return data


def display_graph(data):
    G = to_networkx(data, to_undirected=False)
    edge_list = list(G.edges())
    edge_labels = {}
    for i, edge in enumerate(edge_list):
        label = (
            data.edge_y[i].item() if hasattr(data.edge_y[i], "item") else data.edge_y[i]
        )
        if label != 0:
            edge_labels[edge] = label

    # Create a subgraph with only edges that have labels
    edges_with_labels = list(edge_labels.keys())
    G_sub = G.edge_subgraph(edges_with_labels).copy()

    # Draw the subgraph
    pos = nx.spring_layout(G_sub)
    nx.draw(
        G_sub,
        pos,
        with_labels=True,
        node_color="skyblue",
        node_size=700,
        font_weight="bold",
    )
    nx.draw_networkx_edge_labels(G_sub, pos, edge_labels=edge_labels, font_color="red")
    plt.show()


if __name__ == "__main__":
    name_to_node = {"genai4-512": 0, "genai5-256": 1, "genai8-128": 2, "genai9-128": 3}
    data = convert_ordering_to_edges(
        "genai8-128=genai9-128>genai5-256>genai4-512", name_to_node
    )
    print(data.edge_index)
    print(data.edge_y)
    display_graph(data)
