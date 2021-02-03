from scipy.io import arff
import numpy as np
import math

L = 0.01
MIN_PTS = 4

if __name__ == "__main__":
    data, meta = arff.loadarff('banana.arff')

    dimension = len(data[0]) - 1

    points = []

    for row in data:
        points.append(tuple(math.floor(row[i] / L) for i in range(dimension)))

    arrPoints = np.array(points)

    maxX = np.amax(arrPoints[:, 0])
    maxY = np.amax(arrPoints[:, 1])

    count_matrix = np.zeros((maxX + 1, maxY + 1))
    print(count_matrix.shape)

    for x, y in arrPoints:
        count_matrix[x][y] += 1
        print(f'elemento [{x}][{y}] : {count_matrix[x][y]}')

    list_to_return = []
    for x in range(maxX):
        for y in range(maxY):
            if count_matrix[x][y] >= MIN_PTS:
                list_to_return.append((x, y))

    for i in list_to_return:
        print(f'elemento {i} contiene: {count_matrix[i[0]][i[1]]} elementi')

    print(f'celle dense totali: {len(list_to_return)}')
