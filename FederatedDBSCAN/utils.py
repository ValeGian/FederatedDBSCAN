from scipy.io import arff
import os

DATASETS_PATH = "./datasets/"

def getDatasetPartitions():
    #print("Choose the dataset to be partitioned:")
    #print(os.listdir(DATASETS_PATH))
    #file = DATASETS_PATH + input()
    file = DATASETS_PATH + "banana.arff"
    data, meta = arff.loadarff(file)
