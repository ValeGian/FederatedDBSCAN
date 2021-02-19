from collections import OrderedDict

import clustering as cltr
import arffutils as arff
from random import random
from tabulate import tabulate
from pandas.plotting import table
import plot as plt
import matplotlib.pyplot as plott
import partition as prt
import local as lcl
import server as srv
from IPython.display import display
import pandas as pd

import numpy as np
import os

DATASETS_PATH = "./datasets/"


def create_metric_table(metric_values, column_list, name, file):

    df = pd.DataFrame(data=metric_values,
                      columns=column_list)

    fig, ax = plott.subplots()
    ax.axis('off')
    ax.axis('tight')
    t = ax.table(cellText=df.values, colWidths=[0.1] * len(df.columns), colLabels=df.columns, loc='center')
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    fig.tight_layout()
    plott.savefig(f'table_{name}_{file}.png')
    #plott.show()


def truncate(f, n):
    """Truncates/pads a float f to n decimal places without rounding"""
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


def execute_federated(M, L, MIN_PTS):
    contribution_map = {}
    for i in range(M):
        local_update = lcl.compute_local_update(i, L)
        for key, value in local_update.items():
            if key in contribution_map:
                contribution_map[key] += value
            else:
                contribution_map[key] = value

    cells, cell_labels = srv.compute_clusters(contribution_map, MIN_PTS)

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

    return elaborated_labels


    '''print(f'Federated\t>>\t'
          f'PURITY: {cltr.PURITY_score(Tlabels, elaborated_labels):.4f}\t-\t'
          f'ARI: {cltr.ARI_score(Tlabels, elaborated_labels):.4f}\t-\t'
          f'AMI: {cltr.AMI_score(Tlabels, elaborated_labels):.4f}')

    ### DBSCAN ###
    Tpoints, Tlabels = arff.loadarffNDArray(file)
    predicted_labels = cltr.dbscan(Tpoints, eps=L / 2, min_pts=MIN_PTS)
    plt.plotCluster(Tpoints, predicted_labels, message="DBSCAN")

    plt.plotCluster(Tpoints, Tlabels, message="Original")

    print(f'DBSCAN\t\t>>\t'
          f'PURITY: {cltr.PURITY_score(Tlabels, predicted_labels):.4f}\t-\t'
          f'ARI: {cltr.ARI_score(Tlabels, predicted_labels):.4f}\t-\t'
          f'AMI: {cltr.AMI_score(Tlabels, predicted_labels):.4f}')
'''


if __name__ == '__main__':

    file = "banana.arff"
    M = 2
    partitioning_method = 1

    arf = prt.partitionDataset(file, M, partitioning_method)
    dimensions = len(arf[0][0]) - 1

    range_L = (2, 5, 0.5)
    range_minPts = (2, 6)
    MIN_PTS_list = ["L\MinPTs"]

    rows = int(((range_L[1] - range_L[0]) * range_L[2] ** -1))
    cols = range_minPts[1] - range_minPts[0] + 1

    print(rows, cols)

    PURITY_values = np.zeros((rows, cols))
    PURITY_values_db = np.zeros((rows, cols))
    AMI_values = np.zeros((rows, cols))
    AMI_values_db = np.zeros((rows, cols))
    ARI_values = np.zeros((rows, cols))
    ARI_values_db = np.zeros((rows, cols))

    Tpoints, Tlabels = arff.arffToNDArray(arf)
    Tpoints_db, Tlabels_db = arff.loadarffNDArray(file)

    first_iteration = True
    i, j = 0, 1
    for L in np.arange(range_L[0], range_L[1], range_L[2]) / 100:
        PURITY_values[i][0] = PURITY_values_db[i][0] = AMI_values[i][0] = AMI_values_db[i][0] = ARI_values[i][0] = ARI_values_db[i][0] = L
        for MinPts in range(range_minPts[0], range_minPts[1]):
            if first_iteration:
                MIN_PTS_list.append(MinPts)

            federated_labels = execute_federated(M, L, MinPts)

            PURITY_values[i][j] = truncate(cltr.PURITY_score(Tlabels, federated_labels), 4)
            AMI_values[i][j] = truncate(cltr.AMI_score(Tlabels, federated_labels), 4)
            ARI_values[i][j] = truncate(cltr.ARI_score(Tlabels, federated_labels), 4)

            dbscan_labels = cltr.dbscan(Tpoints_db, eps=L / 2, min_pts=MinPts)

            PURITY_values_db[i][j] = truncate(cltr.PURITY_score(Tlabels_db, dbscan_labels), 4)
            AMI_values_db[i][j] = truncate(cltr.AMI_score(Tlabels_db, dbscan_labels), 4)
            ARI_values_db[i][j] = truncate(cltr.ARI_score(Tlabels_db, dbscan_labels), 4)
            j += 1

        j = 1
        i += 1
        if first_iteration:
            first_iteration = False

    print(PURITY_values_db)
    create_metric_table(PURITY_values, MIN_PTS_list, "PURITY_values", file)
    create_metric_table(AMI_values, MIN_PTS_list, "AMI_values", file)
    create_metric_table(ARI_values, MIN_PTS_list, "ARI_values", file)
    create_metric_table(PURITY_values_db, MIN_PTS_list, "PURITY_values_db", file)
    create_metric_table(AMI_values_db, MIN_PTS_list, "AMI_values_db", file)
    create_metric_table(ARI_values_db, MIN_PTS_list, "ARI_values_db", file)
