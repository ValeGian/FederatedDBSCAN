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
    #partitioningIndex = input();
    partitioningIndex = 0
    #print()

    if partitioningIndex == 0:
        rng = np.random.default_rng()
        rng.shuffle(data)
        for i in range(M):
            lowerB = i * data.size // M
            upperB = (i+1) * data.size // M
            df = pd.DataFrame(data[lowerB:upperB])

            start = datetime.now()
            attributes = [(c, 'NUMERIC') for c in df.columns.values[:-1]]
            t = df.columns[-1]
            attributes += [('class', df[t].unique().astype(str).tolist())]
            partitionData = [df.loc[j].values[:-1].tolist() + [str(df[t].loc[j], 'utf-8')] for j in range(df.shape[0])]
            arff_dic = {
                'attributes': attributes,
                'data': partitionData,
                'relation': 'banana1',
                'description': ''
            }
            print(datetime.now() - start)

            with open(f'{PARTITIONS_PATH}partition{i}.arff', "w", encoding="utf8") as f:
                arff.dump(arff_dic, f)
    #elif partitioningIndex == 1:
    #    i = 1
    #elif partitioningIndex == 2:
    #    i = 2
    #else:
    #    i = 3



