import numpy as np
from scipy.spatial import distance
import math
import arffutils as arff

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


def compute_local_update(my_index, L):
    """ Computes the local updates with a specific grid granularity and returns a mapping of the computed cells
        associated with the number of points in each cell

    :param my_index: int. Index of the partition file to read from
    :param L: float. Grid's granularity
    :return: dict. Map containing couples (cell coords) -> # of points in cell
                Example of return: {(9, 27): 2.0, (9, 29): 3.0, (9, 30): 1.0}
    """
    arr_points = np.array(get_points(my_index, L, floor=True))

    max_x = np.amax(arr_points[:, 0])
    min_x = np.amin(arr_points[:, 0])
    max_y = np.amax(arr_points[:, 1])
    min_y = np.amin(arr_points[:, 1])

    x_shift = 0
    if min_x < 0:
        x_shift = -1 * min_x

    y_shift = 0
    if min_y < 0:
        y_shift = -1 * min_y

    count_matrix = np.zeros((max_x + 1 + x_shift, max_y + 1 + y_shift))

    for x, y in arr_points:
        count_matrix[x + x_shift][y + y_shift] += 1

    #dict_to_return = OrderedDict()
    dict_to_return = {}
    for x in range(max_x + 1 + x_shift):
        for y in range(max_y + 1 + y_shift):
            if count_matrix[x][y] > 0:
                dict_to_return[(x - x_shift, y - y_shift)] = count_matrix[x][y]

    return dict_to_return


def get_points(partition_index, L, floor=False):
    data, meta = arff.loadpartition(partition_index)
    dimension = len(data[0]) - 1
    points = []
    for row in data:
        if floor:
            points.append(tuple(math.floor(row[i] / L) for i in range(dimension)))
        else:
            points.append(tuple(row[i] for i in range(dimension)))

    return points


def assign_points_to_cluster(my_index, array_cells, labels, L):
    """ Assigns to each point of the dataset the class label associated with the cell which contains such point
        or sets it as an outlier if no such cell is been clustered.

    :param my_index: int. index indicating the dataset file
    :param array_cells: numpy.ndarray. Array of cells which have been clustered
    :param labels: numpy.ndarray. Array of class labels, each associated with the corresponding cell (the cell in cells
                    which has the same index)
    :param L: int. Granularity of the grid
    :return: points_to_return, labels_to_return.
                points_to_return: numpy.ndarray. Array of points contained in the dataset
                labels_to_return: numpy.ndarray. Array of labels, each associated to the point having the same index
    """
    points = get_points(my_index, L, floor=False)

    dense_cells = []
    for row in array_cells:
        dense_cells.append(tuple(row))

    points_to_return = []
    labels_to_return = []

    while len(points) > 0:
        actual_point = points.pop(0)
        actual_cell = tuple(math.floor(actual_point[i] / L) for i in range(len(actual_point)))
        outlier = True

        if actual_cell in dense_cells:
            points_to_return.append(actual_point)
            labels_to_return.append(labels[dense_cells.index(actual_cell)])
        else:
            min_dist = float('inf')
            cluster_to_assign = -1
            check_list = get_all_neighbor(actual_cell)
            for check_cell in check_list:
                if check_cell in dense_cells:
                    cell_mid_point = tuple(cell_coord * L + L/2 for cell_coord in check_cell)
                    actual_dist = distance.euclidean(actual_point, cell_mid_point)
                    if actual_dist < min_dist:
                        min_dist = actual_dist
                        cluster_to_assign = labels[dense_cells.index(check_cell)]

                    outlier = False

            if outlier:
                points_to_return.append(actual_point)
                labels_to_return.append(-1)
            else:
                points_to_return.append(actual_point)
                labels_to_return.append(cluster_to_assign)

    return np.array(points_to_return), np.array(labels_to_return)


if __name__ == '__main__':
    print(compute_local_update(0))
