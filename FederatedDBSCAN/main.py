from utils import *
import Local as node
from debug import *

if __name__ == '__main__':
    #M = int(input("Insert the number of nodes: "))
    M = 2
    partitionDataset(M)

    contribuitionMap = {}
    for i in range(M):
        localUpdate = node.compute_local_update(i)
        #localUpdate = {(1, 2): 5, (3, 6): 3, (0, 0): 12}
        for key, value in localUpdate.items():
            if key in contribuitionMap:
                contribuitionMap[key] += value
            else:
                contribuitionMap[key] = value
    print(contribuitionMap)
    #clusters = computeClusters(localUpdates)
    #mapGraphicalDebug(contribuitionMap)
    mapPlotDebug(contribuitionMap)
