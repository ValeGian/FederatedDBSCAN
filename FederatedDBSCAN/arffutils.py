from scipy.io import arff as arffsc
import arff
import numpy as np

DATASETS_PATH = "./datasets/"
PARTITIONS_PATH = "./partitions/"

def loadarff(file):
    path = DATASETS_PATH + file
    return arffsc.loadarff(path)

def loadarffNDArray(file):
    arff = loadarff(file)
    return arffToNDArray(arff)

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

def arffToNDArray(arff) -> (np.ndarray, np.ndarray):
    data = arff[0]
    meta = arff[1]
    classes = np.unique(data[meta.names()[-1]]).tolist()
    dimension = len(data[0]) - 1

    points = []
    labels = []
    for row in data:
        point = [row[i] for i in range(dimension)]
        points.append(point)
        labels.append(classes.index(row[dimension]))

    return np.array(points), np.array(labels)