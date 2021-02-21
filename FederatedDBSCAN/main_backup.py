import clustering as cltr
import arffutils as arff
import plot as plt
import matplotlib.pyplot as plott
import partition as prt
import local as lcl
import server as srv
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
    t = ax.table(cellText=df.values, colLabels=df.columns, loc='center') #colWidths=[0.1] * len(df.columns)
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.auto_set_column_width(col=list(range(len(df.columns))))
    fig.tight_layout()

    if not os.path.exists(file[:-5]):
        os.makedirs(file[:-5])

    plott.savefig(f'{file[:-5]}/table_{name}.png')
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

    L_list = ["L\MinPTs"]
    range_L = (2, 40, 2)
    range_minPts = (2, 8)

    cols = int(((range_L[1] - range_L[0]) * range_L[2] ** -1)) + 1
    rows = range_minPts[1] - range_minPts[0]

    PURITY_values = np.zeros((rows, cols))
    PURITY_values_db = np.zeros((rows, cols))
    AMI_values = np.zeros((rows, cols))
    AMI_values_db = np.zeros((rows, cols))
    ARI_values = np.zeros((rows, cols))
    ARI_values_db = np.zeros((rows, cols))
    num_clusters = np.zeros((rows, cols))
    num_outliers = np.zeros((rows, cols))
    num_clusters_db = np.zeros((rows, cols))
    num_outliers_db = np.zeros((rows, cols))

    Tpoints, Tlabels = arff.arffToNDArray(arf)
    Tpoints_db, Tlabels_db = arff.loadarffNDArray(file)

    first_iteration = True
    i, j = 0, 1

    min_pts_values = []

    fd_c_purities = []
    fd_c_amis = []
    fd_c_aris = []
    fd_c_num_clusters = []

    db_c_purities = []
    db_c_amis = []
    db_c_aris = []
    db_c_num_clusters = []

    for MinPts in range(range_minPts[0], range_minPts[1]):

        min_pts_values.append(MinPts)
        l_values = []

        fd_purities = []
        fd_amis = []
        fd_aris = []
        fd_num_clusters = []

        db_purities = []
        db_amis = []
        db_aris = []
        db_num_clusters = []

        for L in np.arange(range_L[0], range_L[1], range_L[2]) / 1000:
            l_values.append(L)

            PURITY_values[i][0] = PURITY_values_db[i][0] = AMI_values[i][0] = AMI_values_db[i][0] = ARI_values[i][0] = ARI_values_db[i][0] = num_outliers[i][0] = num_clusters[i][0] = num_outliers_db[i][0] = num_clusters_db[i][0] = MinPts
            if first_iteration:
                L_list.append(L)

            federated_labels = execute_federated(M, L, MinPts)

            purity = cltr.PURITY_score(Tlabels, federated_labels)
            ami = cltr.AMI_score(Tlabels, federated_labels)
            ari = cltr.ARI_score(Tlabels, federated_labels)

            fd_purities.append(purity)
            fd_amis.append(ami)
            fd_aris.append(ari)
            fd_num_clusters.append(cltr.compute_clusters(federated_labels))

            PURITY_values[i][j] = truncate(purity, 4)
            AMI_values[i][j] = truncate(ami, 4)
            ARI_values[i][j] = truncate(ari, 4)
            num_clusters[i][j] = cltr.compute_clusters(federated_labels)
            num_outliers[i][j] = cltr.num_outliers(federated_labels)

            dbscan_labels = cltr.dbscan(Tpoints_db, eps=L / 2, min_pts=MinPts)

            purity = cltr.PURITY_score(Tlabels_db, dbscan_labels)
            ami = cltr.AMI_score(Tlabels_db, dbscan_labels)
            ari = cltr.ARI_score(Tlabels_db, dbscan_labels)

            db_purities.append(purity)
            db_amis.append(ami)
            db_aris.append(ari)
            db_num_clusters.append(cltr.compute_clusters(dbscan_labels))

            PURITY_values_db[i][j] = truncate(purity, 4)
            AMI_values_db[i][j] = truncate(ami, 4)
            ARI_values_db[i][j] = truncate(ari, 4)
            num_clusters_db[i][j] = cltr.compute_clusters(dbscan_labels)
            num_outliers_db[i][j] = cltr.num_outliers(dbscan_labels)

            j += 1

        j = 1
        i += 1
        if first_iteration:
            first_iteration = False

        fd_c_purities.append([l_values, fd_purities])
        fd_c_amis.append([l_values, fd_amis])
        fd_c_aris.append([l_values, fd_aris])
        fd_c_num_clusters.append([l_values, fd_num_clusters])

        l_values = [l_value/2 for l_value in l_values]
        db_c_purities.append([l_values, db_purities])
        db_c_amis.append([l_values, db_amis])
        db_c_aris.append([l_values, db_aris])
        db_c_num_clusters.append([l_values, db_num_clusters])

    plt.plot_curves(fd_c_purities, min_pts_values, "MinPts", "L", "PURITY", file)
    plt.plot_curves(fd_c_amis, min_pts_values, "MinPts", "L", "AMI", file)
    plt.plot_curves(fd_c_aris, min_pts_values, "MinPts", "L", "ARI", file)
    plt.plot_curves(fd_c_num_clusters, min_pts_values, "MinPts", "L", "#Clusters", file)

    plt.plot_curves(db_c_purities, min_pts_values, "MinPts", "Eps = L/2", "PURITY", file)
    plt.plot_curves(db_c_amis, min_pts_values, "MinPts", "Eps = L/2", "AMI", file)
    plt.plot_curves(db_c_aris, min_pts_values, "MinPts", "Eps = L/2", "ARI", file)
    plt.plot_curves(db_c_num_clusters, min_pts_values, "MinPts", "Eps = L/2", "#Clusters", file)

    create_metric_table(PURITY_values, L_list, "PURITY_values", file)
    create_metric_table(AMI_values, L_list, "AMI_values", file)
    create_metric_table(ARI_values, L_list, "ARI_values", file)
    create_metric_table(num_clusters, L_list, "clusters_values", file)
    create_metric_table(num_outliers, L_list, "outliers_values", file)

    create_metric_table(PURITY_values_db, L_list, "PURITY_values_db", file)
    create_metric_table(AMI_values_db, L_list, "AMI_values_db", file)
    create_metric_table(ARI_values_db, L_list, "ARI_values_db", file)
    create_metric_table(num_clusters_db, L_list, "clusters_values_db", file)
    create_metric_table(num_outliers_db, L_list, "outliers_values_db", file)
