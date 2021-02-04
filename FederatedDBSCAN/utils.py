import math

from scipy.io import arff as arffsc
import arff
import os
import numpy as np
import pandas as pd
from datetime import datetime

DATASETS_PATH = "./datasets/"
PARTITIONS_PATH = "./partitions/"
PARTITIONING_METHODS = ["stratified", "separated", "partially_separated"]

def partitionDataset(M = 2):
    #print(f'Choose the dataset to be partitioned:\n{os.listdir(DATASETS_PATH)}')
    #file = DATASETS_PATH + input()
    #print()
    file = DATASETS_PATH + "banana.arff"
    #file = PARTITIONS_PATH + "partition0.arff"
    data, meta = arffsc.loadarff(file)

    #print('Choose the partitioning method:')
    #for (i, item) in enumerate(PARTITIONING_METHODS):
        #print(f'{i} -> {item}')
    #partitioningMethod = input();
    partitioningMethod = 1
    #print()

    if partitioningMethod == 0:
        start = datetime.now()
        rng = np.random.default_rng()
        rng.shuffle(data)
        for i in range(M):
            lowerB = i * data.size // M
            upperB = (i+1) * data.size // M
            df = pd.DataFrame(data[lowerB:upperB])
            dumpArff(df, i)
        print(datetime.now() - start)
    elif partitioningMethod == 1:
        start = datetime.now()
        classes = np.unique(data[meta.names()[-1]]).tolist()
        N = len(classes)

        separatedL = [[] for n in range(N)]
        for elem in data:
            separatedL[classes.index(elem[-1])].append(elem)

        if M >= N:
            for i in range(N):
                for j in range(M//N):
                    lowerB = j * len(separatedL[i]) // (M//N)
                    upperB = (j+1) * len(separatedL[i]) // (M//N)
                    arr = np.array(separatedL[i][lowerB:upperB])
                    df = pd.DataFrame(arr)
                    dumpArff(df, i * M // N + j)
        print(datetime.now() - start)
    #elif partitioningIndex == 2:
    #    i = 2
    #else:
    #    i = 3

def dumpArff(df, partitionIndex):
    attributes = [(c, 'NUMERIC') for c in df.columns.values[:-1]]
    t = df.columns[-1]
    attributes += [('class', df[t].unique().astype(str).tolist())]
    partitionData = [df.loc[j].values[:-1].tolist() + [str(df[t].loc[j], 'utf-8')] for j in range(df.shape[0])]
    arff_dic = {
        'attributes': attributes,
        'data': partitionData,
        'relation': f'partition{partitionIndex}',
        'description': ''
    }

    with open(f'{PARTITIONS_PATH}partition{partitionIndex}.arff', "w", encoding="utf8") as f:
        arff.dump(arff_dic, f)

def removePartitions():
    for file in os.listdir(PARTITIONS_PATH):
        os.remove(f'{PARTITIONS_PATH}{file}')


def obtainSuddivision(num_class , num_node ):
    arr_to_return = np.full(num_class, math.floor(num_node/num_class))
    for i in range(num_node % num_class):
        arr_to_return[i] += 1

    return arr_to_return