import matplotlib.pyplot as plt
import numpy as np

def mapPlotDebug(contribuitionMap):
    maxX = 0
    maxY = 0
    for key in contribuitionMap.keys():
        if key[0] > maxX:
            maxX = key[0]
        if key[1] > maxY:
            maxY = key[1]

    matrix = np.zeros((maxX + 1, maxY + 1))

    for key, value in contribuitionMap.items():
        matrix[key[0]][key[1]] += value

    print(matrix)
    plt.imshow(matrix, cmap='gray_r', origin='lower')
    plt.show()
