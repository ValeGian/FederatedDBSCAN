from utils import *
from debug import *
import Local as node
import numpy as np
from collections import OrderedDict

MIN_PTS = 4


def get_all_neighbor(cell):
    (x, y) = cell
    return [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1), (x - 1, y + 1), (x + 1, y + 1), (x - 1, y - 1), (x + 1, y - 1)]


def compute_clusters(contribution_map):
    key_list = list(contribution_map.keys())
    value_list = list(contribution_map.values())

    n_dense_cell = len(key_list)
    visited = np.zeros(n_dense_cell)
    clusters = []
    while visited.__contains__(0):
        for curr_cell in key_list:
            curr_index = key_list.index(curr_cell)
            if visited[curr_index] == 0:
                visited[curr_index] = 1
                num_point = value_list[curr_index]
                if num_point < MIN_PTS:
                    continue
                else:
                    cluster = [curr_cell]
                    print(f"cella buona: {curr_cell} con elementi: {value_list[curr_index]}")
                    list_of_cell_to_check = get_all_neighbor(curr_cell)

                    while len(list_of_cell_to_check) > 0:
                        neighbor = list_of_cell_to_check.pop(0)

                        if neighbor in key_list and visited[key_list.index(neighbor)] == 0:

                            #print(f"aggiunto al cluster un nuovo elemento {neighbor}, ora il cluster contiene {len(cluster)} elementi: {cluster}")
                            neighbor_index = key_list.index(neighbor)
                            visited[neighbor_index] = 1
                            if value_list[neighbor_index] >= MIN_PTS:
                                cluster.append(neighbor)
                                list_of_cell_to_check += get_all_neighbor(neighbor)

                    clusters.append(cluster)
                    print(f"aggiunto nuovo cluster con {len(cluster)} elementi: {cluster}")

    print(len(clusters))
    return clusters
