import math
import os
import numpy as np
import pandas as pd

import arffutils as arff

PARTITIONS_PATH = "./partitions/"
PARTITIONING_METHODS = ["stratified", "separated", "partially_separated"]

def partitionDataset(M = 2):
    removePartitions()
    #print(f'Choose the dataset to be partitioned:\n{os.listdir(DATASETS_PATH)}')
    #file = DATASETS_PATH + input()
    #print()
    file = "banana.arff"
    data, meta = arff.loadarff(file)

    #print('Choose the partitioning method:')
    #for (i, item) in enumerate(PARTITIONING_METHODS):
        #print(f'{i} -> {item}')
    #partitioningMethod = input();
    partitioningMethod = 1
    #print()

    if partitioningMethod == 0:
        rng = np.random.default_rng()
        rng.shuffle(data)
        for i in range(M):
            lowerB = i * data.size // M
            upperB = (i+1) * data.size // M
            df = pd.DataFrame(data[lowerB:upperB])
            arff.dumpArff(df, i)
    elif partitioningMethod == 1:
        classes = np.unique(data[meta.names()[-1]]).tolist()
        N = len(classes)

        separatedL = [[] for n in range(N)]
        for elem in data:
            separatedL[classes.index(elem[-1])].append(elem)

        if M >= N:
            sub = obtainSubdivision(N, M)
            count = 0
            for i in range(N):
                for j in range(sub[i]):
                    lowerB = j * len(separatedL[i]) // sub[i]
                    upperB = (j+1) * len(separatedL[i]) // sub[i]
                    arr = np.array(separatedL[i][lowerB:upperB])
                    df = pd.DataFrame(arr)
                    arff.dumpArff(df, count)
                    count += 1
        else:
            rng = np.random.default_rng()
            rng.shuffle(data)
            for i in range(M):
                lowerB = i * data.size // M
                upperB = (i + 1) * data.size // M
                df = pd.DataFrame(data[lowerB:upperB])
                arff.dumpArff(df, i)
    #elif partitioningIndex == 2:
    #    i = 2
    #else:
    #    i = 3

def removePartitions():
    for file in os.listdir(PARTITIONS_PATH):
        os.remove(f'{PARTITIONS_PATH}{file}')

def obtainSubdivision(num_class, num_node):
    arr_to_return = np.full(num_class, math.floor(num_node/num_class))
    for i in range(num_node % num_class):
        arr_to_return[i] += 1
    return arr_to_return
