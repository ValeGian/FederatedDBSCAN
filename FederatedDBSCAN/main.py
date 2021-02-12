from server import compute_clusters
from utils import partitionDataset
from debug import mapPlotDebug, map_plot_cluster
import Local as node
import numpy as np
from collections import OrderedDict

if __name__ == '__main__':
    #M = int(input("Insert the number of nodes: "))
    M = 1
    #partitionDataset(M)

    contribution_map = OrderedDict()
    for i in range(M):
        localUpdate = node.compute_local_update(i)
        for key, value in localUpdate.items():
            if key in contribution_map:
                contribution_map[key] += value
            else:
                contribution_map[key] = value
    print(contribution_map)
    clusters = compute_clusters(contribution_map)
    #mapPlotDebug(contribution_map)
    #map_plot_cluster(clusters)

    for i in range(M):
        node.assign_points_to_cluster(i, clusters)


