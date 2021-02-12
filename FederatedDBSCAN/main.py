from collections import OrderedDict

import clustering as cltr
import arffutils as arff
import plot as plt
import partition as prt
import local as lcl
import server as srv

import numpy as np

if __name__ == '__main__':
    #M = int(input("Insert the number of nodes: "))
    #M = 2
    #prt.partitionDataset(M)
#
    #contribution_map = OrderedDict()
    #for i in range(M):
    #    localUpdate = lcl.compute_local_update(i)
    #    for key, value in localUpdate.items():
    #        if key in contribution_map:
    #            contribution_map[key] += value
    #        else:
    #            contribution_map[key] = value
#
    #points, labels = srv.compute_clusters(contribution_map)
    #plt.plot2Dcluster(points, labels)

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

    #for MinPts in range(2, 10):
    #    L = 0.2
    #    file = "banana.arff"
    #    points, labels = arff.loadarffNDArray(file)
#
    #    predicted_labels = cltr.dbscan(points, eps=L/2, min_pts=MinPts)
    #    plt.plot2Dcluster(points, predicted_labels)
#
    #    print(f'MinPts: {MinPts}:\t-\t'
    #          f'AMI: {cltr.PURITY_score(labels, predicted_labels):.4f}\t-\t'
    #          f'ARI: {cltr.ARI_score(labels, predicted_labels):.4f}\t-\t'
    #          f'AMI: {cltr.AMI_score(labels, predicted_labels):.4f}')
