import clustering as cltr
import arffutils as arff
import plot as plt
import partition as prt
import local as lcl
import server as srv

import numpy as np

DATASETS_PATH = "./datasets/"

if __name__ == '__main__':
    file = "banana.arff"

    M = 3

    partitioning_method = 0

    L = 0.05
    MIN_PTS = 2

    arf = prt.partitionDataset(file, M, partitioning_method)
    dimensions = len(arf[0][0]) - 1

    #for i in range(M):
    #    points, labels = arff.loadpartitionNDArray(i)
    #    plt.plotCluster(points, labels, message=f'Partition {i}')

    contribution_map = {}
    for i in range(M):
        local_update = lcl.compute_local_update(i, L)
        for key, value in local_update.items():
            if key in contribution_map:
                contribution_map[key] += value
            else:
                contribution_map[key] = value

    cells, cell_labels = srv.compute_clusters(contribution_map, MIN_PTS)
    plt.plotCluster(cells, cell_labels, message="Cells Plot", marker="s")

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
    plt.plotCluster(elaborated_points, elaborated_labels, message="Federated")

    '''EVALUATION'''
    ### FEDERATED ###
    Tpoints, Tlabels = arff.arffToNDArray(arf)
    print(f'Federated\t>>\t'
          f'PURITY: {cltr.PURITY_score(Tlabels, elaborated_labels):.4f}\t-\t'
          f'ARI: {cltr.ARI_score(Tlabels, elaborated_labels):.4f}\t-\t'
          f'AMI: {cltr.AMI_score(Tlabels, elaborated_labels):.4f}')
          #f'BCubed PRECISION: {cltr.BCubed_Precision_score(Tlabels, elaborated_labels):.4f}\t-\t'
          #f'BCubed RECALL: {cltr.BCubed_Recall_score(Tlabels, elaborated_labels):.4f}')

    ### DBSCAN ###
    Tpoints, Tlabels = arff.loadarffNDArray(file)
    predicted_labels = cltr.dbscan(Tpoints, eps=L / 2, min_pts=MIN_PTS)
    plt.plotCluster(Tpoints, predicted_labels, message="DBSCAN")

    plt.plotCluster(Tpoints, Tlabels, message="Original")

    print(f'DBSCAN\t\t>>\t'
          f'PURITY: {cltr.PURITY_score(Tlabels, predicted_labels):.4f}\t-\t'
          f'ARI: {cltr.ARI_score(Tlabels, predicted_labels):.4f}\t-\t'
          f'AMI: {cltr.AMI_score(Tlabels, predicted_labels):.4f}\t-\t')
          #f'BCubed PRECISION: {cltr.BCubed_Precision_score(Tlabels, predicted_labels):.4f}\t-\t'
          #f'BCubed RECALL: {cltr.BCubed_Recall_score(Tlabels, predicted_labels):.4f}')
'''
    for L in np.arange(1, 8, 0.5)/100:
        MinPts = 4
        file = "banana.arff"
        points, labels = arff.loadarffNDArray(file)

        predicted_labels = cltr.dbscan(points, eps=L/2, min_pts=MinPts)
        plt.plot2Dcluster(points, predicted_labels)

        print(f'L: {L:.3f}:\t-\t'
              f'PURITY: {cltr.PURITY_score(labels, predicted_labels):.4f}\t-\t'
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