import numpy as np
from collections import OrderedDict


def get_all_neighbor(cell):
    diag_coord = [(x - 1, x, x + 1) for x in cell]

    cartesian_product = [[]]
    for pool in diag_coord:
        cartesian_product = [(x + [y]) for x in cartesian_product for y in pool]

    result = []
    for prod in cartesian_product:
        result.append(tuple(prod))

    result.remove(cell)
    return result


def compute_clusters(contribution_map, MIN_PTS) -> (np.ndarray, np.ndarray):
    """ Computes clusters exploting a variant of DBSCAN

    :param contribution_map: dict. Map containing couples (cell coords) -> # of points in cell
    :param MIN_PTS: int.
    :return: cells, labels.
                cells: numpy.ndarray. Array of cells which have been clustered
                labels: numpy.ndarray. Array of class labels, each associated with the corresponding cell (the cell in
                        cells which has the same index)
    """
    key_list = list(contribution_map.keys())
    value_list = list(contribution_map.values())

    n_dense_cell = len(key_list)
    visited = np.zeros(n_dense_cell)
    clustered = np.zeros(n_dense_cell)
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
            
    return np.array(cells), np.array(labels)
