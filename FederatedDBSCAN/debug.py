import matplotlib.pyplot as plt
import numpy as np

def mapPlotDebug(contributionMap):
    print("---PLOTTING---")
    maxX = 0
    minX = 0
    maxY = 0
    minY = 0
    for key in contributionMap.keys():
        if key[0] > maxX:
            maxX = key[0]
        if key[0] < minX:
            minX = key[0]
        if key[1] > maxY:
            maxY = key[1]
        if key[1] < minY:
            minY = key[1]

    print(f'X: [{minX}, {maxX}]')
    print(f'Y: [{minY}, {maxY}]')

    x_shift = -1 * minX
    y_shift = -1 * minY

    matrix = np.zeros((maxY + 1 + y_shift, maxX + 1 + x_shift))

    for key, value in contributionMap.items():
        #matrix[key[0]][key[1]] += value
        matrix[key[1] + y_shift][key[0] + x_shift] += value

    plt.imshow(matrix, cmap='gray_r', origin='lower')
    plt.show()
