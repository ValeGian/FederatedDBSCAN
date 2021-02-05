import matplotlib.pyplot as plt
import numpy as np

def mapPlotDebug(contributionMap):
    print("---PLOTTING---")
    maxX = 0
    maxY = 0
    for key in contributionMap.keys():
        if key[0] > maxX:
            maxX = key[0]
        if key[1] > maxY:
            maxY = key[1]

    print(f'MAX_X: {maxX}')
    print(f'MAX_Y: {maxY}')
    #matrix = np.zeros((maxX + 1, maxY + 1))
    matrix = np.zeros((maxY + 1, maxX + 1))

    for key, value in contributionMap.items():
        #matrix[key[0]][key[1]] += value
        matrix[key[1]][key[0]] += value

    plt.imshow(matrix, cmap='gray_r', origin='lower')
    plt.show()
