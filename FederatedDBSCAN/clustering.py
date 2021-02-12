from sklearn.cluster import DBSCAN
import sklearn.metrics as mtr
import numpy as np


def dbscan(points: np.ndarray, eps, min_pts) -> np.ndarray:
    clustering = DBSCAN(eps=eps, min_samples=min_pts)
    labels = clustering.fit_predict(points)
    return labels


def ARI_score(true_labels, predicted_labels):
    return mtr.adjusted_rand_score(true_labels, predicted_labels)


def AMI_score(true_labels, predicted_labels):
    return mtr.adjusted_mutual_info_score(true_labels, predicted_labels)


def PURITY_score(true_labels, predicted_labels):
    contingency_matrix = mtr.cluster.contingency_matrix(true_labels, predicted_labels)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
