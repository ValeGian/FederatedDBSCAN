from partition import *
from plot import *
import local as node
import numpy as np
from collections import OrderedDict


def get_all_neighbor(cell):
    (x, y) = cell
    return [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1), (x - 1, y + 1), (x + 1, y + 1), (x - 1, y - 1), (x + 1, y - 1)]


def compute_clusters(contribution_map, MIN_PTS) -> (np.ndarray, np.ndarray):
    key_list = list(contribution_map.keys())
    value_list = list(contribution_map.values())

    n_dense_cell = len(key_list)
    visited = np.zeros(n_dense_cell)
    cells = []
    labels = []
    cluster_ID = 0
    while 0 in visited:
        for curr_cell in key_list:
            curr_index = key_list.index(curr_cell)
            if visited[curr_index] == 0:
                visited[curr_index] = 1
                num_point = value_list[curr_index]
                if num_point < MIN_PTS:
                    continue
                else:
                    cells.append(curr_cell)
                    labels.append(cluster_ID)

                    list_of_cell_to_check = get_all_neighbor(curr_cell)

                    while len(list_of_cell_to_check) > 0:
                        neighbor = list_of_cell_to_check.pop(0)

                        if neighbor in key_list and visited[key_list.index(neighbor)] == 0:
                            neighbor_index = key_list.index(neighbor)
                            visited[neighbor_index] = 1
                            if value_list[neighbor_index] >= MIN_PTS:
                                cells.append(neighbor)
                                labels.append(cluster_ID)
                                list_of_cell_to_check += get_all_neighbor(neighbor)

                    cluster_ID += 1

    return np.array(cells), np.array(labels)
