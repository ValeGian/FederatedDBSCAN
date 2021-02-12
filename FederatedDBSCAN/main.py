import clustering as cl
import arffutils as arff
import plot as plt

import numpy as np
import sklearn.metrics as mtr

if __name__ == '__main__':
    #M = int(input("Insert the number of nodes: "))
    #M = 2
    #partitionDataset(M)

    #contribution_map = OrderedDict()
    #for i in range(M):
    #    localUpdate = node.compute_local_update(i)
    #    for key, value in localUpdate.items():
    #        if key in contribution_map:
    #            contribution_map[key] += value
    #        else:
    #            contribution_map[key] = value
    #print(contribution_map)
    #clusters = compute_clusters(contribution_map)
    #mapPlot2Dcluster(clusters)

    for L in np.arange(1, 10, 0.5)/100:
        MinPts = 4
        file = "banana.arff"
        points, labels = arff.loadarffNDArray(file)
        
        clusters, outlier_index, predicted_labels = cl.dbscan(points, eps=L/2, min_pts=MinPts)
        plt.plot2Dcluster(clusters, outlier_index)
        print(f'L: {L}\t-\t'
              f'ARI: {mtr.adjusted_rand_score(labels, predicted_labels):.4f}\t-\t'
              f'AMI: {mtr.adjusted_mutual_info_score(labels, predicted_labels):.4f}')
