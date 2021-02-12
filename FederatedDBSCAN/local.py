from scipy.io import arff
import numpy as np
import math

from collections import OrderedDict

L = 0.02
PARTITIONS_PATH = "./partitions/partition"

def compute_local_update(my_index):
    data, meta = arff.loadarff(f'{PARTITIONS_PATH}{my_index}.arff')

    dimension = len(data[0]) - 1
    points = []

    for row in data:
        points.append(tuple(math.floor(row[i] / L) for i in range(dimension)))

    arr_points = np.array(points)

    max_x = np.amax(arr_points[:, 0])
    min_x = np.amin(arr_points[:, 0])
    max_y = np.amax(arr_points[:, 1])
    min_y = np.amin(arr_points[:, 1])
    print(f'X: [{min_x}, {max_x}]')
    print(f'Y: [{min_y}, {max_y}]')

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


if __name__ == '__main__':
    print(compute_local_update(0))
