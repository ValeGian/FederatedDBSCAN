from collections import OrderedDict

import clustering as cltr
import arffutils as arff
import plot as plt
import partition as prt
import local as lcl
import server as srv

import numpy as np
import os

DATASETS_PATH = "./datasets/"

if __name__ == '__main__':
    #print(f'Choose the dataset to be partitioned:\n{os.listdir(DATASETS_PATH)}')
    #file = DATASETS_PATH + input()
    #print()
    file = "banana.arff"

    #M = int(input("Insert the number of nodes: "))
    # print()
    M = 3

    # print('Choose the partitioning method:')
    # for (i, item) in enumerate(PARTITIONING_METHODS):
    # print(f'{i} -> {item}')
    # partitioningMethod = input();
    # print()
    partitioning_method = 1

    L = 0.02
    MIN_PTS = 4

    arf = prt.partitionDataset(file, M, partitioning_method)

    #for i in range(M):
    #    points, labels = arff.loadpartitionNDArray(i)
    #    plt.plot2Dcluster(points, labels)

    #contribution_map = OrderedDict()
    contribution_map = {}
    for i in range(M):
        localUpdate = lcl.compute_local_update(i, L)
        for key, value in localUpdate.items():
            if key in contribution_map:
                contribution_map[key] += value
            else:
                contribution_map[key] = value

    #plt.plotGridMap(contribution_map)
    cells, cell_labels = srv.compute_clusters(contribution_map, MIN_PTS)
    #plt.plot2Dcluster(cells, cell_labels)

    elaborated_points = []
    elaborated_labels = []
    for i in range(M):
        local_points, local_labels = lcl.assign_points_to_cluster(i, cells, cell_labels, L)
        if len(elaborated_points) == 0:
            elaborated_points = local_points
            elaborated_labels = local_labels
        else:
            elaborated_points = np.concatenate((elaborated_points, local_points), axis=0)
            elaborated_labels = np.concatenate((elaborated_labels, local_labels), axis=0)
        #plt.plot2Dcluster(local_points, local_labels)
    plt.plot2Dcluster(elaborated_points, elaborated_labels, message="Federated")

    '''EVALUATION'''
    ### FEDERATED ###
    Tpoints, Tlabels = arff.arffToNDArray(arf)
    print(f'Federated\t>>\t'
          f'PURITY: {cltr.PURITY_score(Tlabels, elaborated_labels):.4f}\t-\t'
          f'ARI: {cltr.ARI_score(Tlabels, elaborated_labels):.4f}\t-\t'
          f'AMI: {cltr.AMI_score(Tlabels, elaborated_labels):.4f}')

    ### DBSCAN ###
    Tpoints, Tlabels = arff.loadarffNDArray(file)
    predicted_labels = cltr.dbscan(Tpoints, eps=L / 2, min_pts=MIN_PTS)
    plt.plot2Dcluster(Tpoints, predicted_labels, message="DBSCAN")

    plt.plot2Dcluster(Tpoints, Tlabels, message="Original")

    print(f'DBSCAN\t\t>>\t'
          f'PURITY: {cltr.PURITY_score(Tlabels, predicted_labels):.4f}\t-\t'
          f'ARI: {cltr.ARI_score(Tlabels, predicted_labels):.4f}\t-\t'
          f'AMI: {cltr.AMI_score(Tlabels, predicted_labels):.4f}')
'''
    for L in np.arange(1, 8, 0.5)/100:
        MinPts = 4
        file = "banana.arff"
        points, labels = arff.loadarffNDArray(file)

        predicted_labels = cltr.dbscan(points, eps=L/2, min_pts=MinPts)
        plt.plot2Dcluster(points, predicted_labels)

        print(f'L: {L:.3f}:\t-\t'
              f'AMI: {cltr.PURITY_score(labels, predicted_labels):.4f}\t-\t'
              f'ARI: {cltr.ARI_score(labels, predicted_labels):.4f}\t-\t'
              f'AMI: {cltr.AMI_score(labels, predicted_labels):.4f}')

    for MinPts in range(2, 10):
        L = 0.2
        file = "banana.arff"
        points, labels = arff.loadarffNDArray(file)

        predicted_labels = cltr.dbscan(points, eps=L/2, min_pts=MinPts)
        plt.plot2Dcluster(points, predicted_labels)

        print(f'MinPts: {MinPts}:\t-\t'
              f'PURITY: {cltr.PURITY_score(labels, predicted_labels):.4f}\t-\t'
              f'ARI: {cltr.ARI_score(labels, predicted_labels):.4f}\t-\t'
              f'AMI: {cltr.AMI_score(labels, predicted_labels):.4f}')
'''