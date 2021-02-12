from scipy.io import arff
import numpy as np
from scipy.spatial import distance
import math

from collections import OrderedDict

L = 0.02
PARTITIONS_PATH = "./partitions/partition"


def get_all_neighbor(cell):
    (x, y) = cell
    return [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1), (x - 1, y + 1), (x + 1, y + 1), (x - 1, y - 1), (x + 1, y - 1)]


def compute_local_update(my_index):
    arr_points = np.array(get_points(my_index))

    max_x = np.amax(arr_points[:, 0])
    min_x = np.amin(arr_points[:, 0])
    max_y = np.amax(arr_points[:, 1])
    min_y = np.amin(arr_points[:, 1])

    x_shift = 0
    if min_x < 0:
        x_shift = -1 * min_x

    y_shift = 0
    if min_x < 0:
        y_shift = -1 * min_x

    count_matrix = np.zeros((max_x + 1 + x_shift, max_y + 1 + y_shift))

    for x, y in arr_points:
        count_matrix[x + x_shift][y + y_shift] += 1

    dict_to_return = OrderedDict()
    for x in range(max_x + 1 + x_shift):
        for y in range(max_y + 1 + y_shift):
            if count_matrix[x][y] >= 1:
                dict_to_return[(x - x_shift, y - y_shift)] = count_matrix[x][y]

    return dict_to_return


def get_points(my_index, floor=1):
    #data, meta = arff.loadarff(f'{PARTITIONS_PATH}{my_index}.arff')
    data, meta = arff.loadarff(f'datasets/banana.arff')

    dimension = len(data[0]) - 1
    points = []
    only_10 = 10
    for row in data:


        if floor:
            points.append(tuple(math.floor(row[i] / L) for i in range(dimension)))
        else:
            points.append(tuple(row[i] for i in range(dimension)))

    return points


def assign_points_to_cluster(my_index, clusters):

    points = get_points(my_index, 0)

    mapping_dict = {}
    cells = get_points(my_index)
    outlier_list = []
    no_assigned = []

    while len(cells) > 0:
        actual_cell = cells.pop(0)
        actual_point = points.pop(0)
        assigned = False
        outlier = True

        for cluster in clusters:
            if actual_cell in cluster:
                mapping_dict[actual_point] = clusters.index(cluster)
                assigned = True
                break

        if not assigned:
            min_dist = float('inf')
            cluster_to_assign = -1
            no_assigned.append(actual_point)
            check_list = get_all_neighbor(actual_cell)
            for check_cell in check_list:
                for cluster in clusters:
                    if check_cell in cluster:
                        a = ((check_cell[0] * L) + L/2, (check_cell[1] * L) + L/2)
                        actual_dist = distance.euclidean(actual_point, a)
                        if actual_dist < min_dist:
                            min_dist = actual_dist
                            cluster_to_assign = clusters.index(cluster)
                            print(f"check cell: {check_cell}, a: {a}, actual point: {actual_point}, min_dist: {min_dist}, cluster: {cluster_to_assign}")

                        outlier = False
                        break

            if outlier:
                outlier_list.append(actual_point)
            else:
                mapping_dict[actual_point] = cluster_to_assign

    

    print(f'len: {len(outlier_list)}, outlier: {outlier_list}')
    print(f'len: {len(no_assigned)}, no_assigned: {no_assigned}')
    print(f'len: {len(mapping_dict.keys())}, output: {mapping_dict}')


if __name__ == '__main__':
    print(compute_local_update(0))
