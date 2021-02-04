from utils import *
import Local as node
import numpy as np

def mapGraphicalDebug(contribuitionMap):
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

    with open("matrix.txt", "w") as txt_file:
        txt_file.write(f'{maxY}|\t')
        for x in range(maxX + 1):
            txt_file.write(f'{matrix[x][maxY]},\t')
        txt_file.write("\n")

        for y in range(maxY):
            txt_file.write(f'{maxY - y - 1}|\t')
            for x in range(maxX + 1):
                txt_file.write(f'{matrix[x][maxY - y - 1]},\t')
            txt_file.write("\n")

        txt_file.write("\t")
        for x in range(maxX + 1):
            txt_file.write("_\t\t")
        txt_file.write("\n\t")
        for x in range(maxX + 1):
            txt_file.write(f'{x}\t\t')


if __name__ == '__main__':
    M = int(input("Insert the number of nodes: "))
    #partitionDataset(M)

    contribuitionMap = {}
    for i in range(M):
        #localUpdate = node.computeLocalUpdate(i)
        localUpdate = {(1, 1): 5, (3, 5): 2, (0, 1):1, (0, 1):3, (2, 5):1, (5, 6):9, (0, 0): 20}
        for key, value in localUpdate.items():
            if key in contribuitionMap:
                contribuitionMap[key] += value
            else:
                contribuitionMap[key] = value
    print(contribuitionMap)
    #clusters = computeClusters(localUpdates)
    mapGraphicalDebug(contribuitionMap)
