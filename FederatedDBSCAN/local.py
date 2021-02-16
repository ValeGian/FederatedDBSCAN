import numpy as np
from scipy.spatial import distance
import math
import arffutils as arff

from collections import OrderedDict


def get_all_neighbor(cell):
    (x, y) = cell
    return [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1), (x - 1, y + 1), (x + 1, y + 1), (x - 1, y - 1), (x + 1, y - 1)]


def get_local_points(my_index, L):
    arr_points = np.array(get_points(my_index, L))

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

    dict_to_return = OrderedDict()
    for x in range(max_x + 1 + x_shift):
        for y in range(max_y + 1 + y_shift):
            if count_matrix[x][y] >= 1:
                dict_to_return[(x - x_shift, y - y_shift)] = count_matrix[x][y]

    return dict_to_return


def get_points(partition_index, L, floor=True):
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

    points = get_points(my_index, L, floor=False)
    cells = get_points(my_index, L)

    dense_cells = []
    for row in array_cells:
        dense_cells.append(tuple(row))

    points_to_return = []
    labels_to_return = []

    while len(cells) > 0:
        actual_cell = cells.pop(0)
        actual_point = points.pop(0)
        assigned = False
        outlier = True

        if actual_cell in dense_cells:
            points_to_return.append(actual_point)
            labels_to_return.append(labels[dense_cells.index(actual_cell)])
            assigned = True

        if not assigned:
            min_dist = float('inf')
            cluster_to_assign = -1
            check_list = get_all_neighbor(actual_cell)
            for check_cell in check_list:
                if check_cell in dense_cells:
                    cell_mid_point = ((check_cell[0] * L) + L/2, (check_cell[1] * L) + L/2)
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

    #print(f'len: {len(points_to_return)}, points: {points_to_return}')
    #print(f'len: {len(labels_to_return)}, points: {labels_to_return}')

    return np.array(points_to_return), np.array(labels_to_return)


if __name__ == '__main__':
    print(get_local_points(0))
