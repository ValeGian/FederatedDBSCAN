from utils import *
import Local as node

if __name__ == '__main__':
    M = int(input("Insert the number of nodes: "))
    #partitionDataset(M)

    contribuitionMap = {}
    for i in range(M):
        #localUpdate = node.computeLocalUpdate(i)
        localUpdate = {(20, 55): 5, (20, 56): 2, (20, 57):1, (20, 62):3, (20, 66):1, (21, 52):9}
        for key, value in localUpdate.items():
            if key in contribuitionMap:
                contribuitionMap[key] += value
            else:
                contribuitionMap[key] = value
    print(contribuitionMap)
    #clusters = computeClusters(localUpdates)