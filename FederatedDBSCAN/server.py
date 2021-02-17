import numpy as np
from collections import OrderedDict


def get_all_neighbor(cell):
    (x, y) = cell
    return [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1), (x - 1, y + 1), (x + 1, y + 1), (x - 1, y - 1), (x + 1, y - 1)]


def compute_clusters(contribution_map, MIN_PTS) -> (np.ndarray, np.ndarray):
    """ Computes clusters exploting a variant of DBSCAN

    :param contribution_map: dict. Map containing couples (cell coords) -> # of points in cell
    :param MIN_PTS: int
    :return:
    """
    key_list = list(contribution_map.keys())
    value_list = list(contribution_map.values())

    n_dense_cell = len(key_list)
    visited = np.zeros(n_dense_cell)
    clustered = np.zeros(n_dense_cell)
    noise = []
    cells = []
    labels = []
    cluster_ID = 0
    while 0 in visited:
        curr_index = np.random.choice(np.where(np.array(visited) == 0)[0])
        curr_cell = key_list[curr_index]
        visited[curr_index] = 1

        num_point = value_list[curr_index]
        if num_point >= MIN_PTS:
            cells.append(curr_cell)
            labels.append(cluster_ID)
            clustered[curr_index] = 1

            list_of_cell_to_check = get_all_neighbor(curr_cell)
            while len(list_of_cell_to_check) > 0:
                neighbor = list_of_cell_to_check.pop(0)
                neighbor_index = key_list.index(neighbor) if neighbor in key_list else ""
                if neighbor in key_list and visited[neighbor_index] == 0:
                    visited[neighbor_index] = 1
                    if value_list[neighbor_index] >= MIN_PTS:
                        list_of_cell_to_check += get_all_neighbor(neighbor)
                    if clustered[neighbor_index] == 0:
                        cells.append(neighbor)
                        labels.append(cluster_ID)
                        clustered[neighbor_index] = 1
            cluster_ID += 1
        else:
            noise.append(curr_index)

    for noise_cell_index in noise:
        if clustered[noise_cell_index] == 0:
            unvisited_cell = key_list[noise_cell_index]
            cells.append(unvisited_cell)
            labels.append(-1) # -1 for outliers
            
    return np.array(cells), np.array(labels)
