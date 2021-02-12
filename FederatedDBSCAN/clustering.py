from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np

def dbscan(points: np.ndarray, eps, min_pts) -> (np.ndarray, int):
    clustering = DBSCAN(eps=eps, min_samples=min_pts)

    labels = clustering.fit_predict(points)
    unique_labels = set(labels)

    clusters = [[] for i in range(len(unique_labels))]
    outlier_index = len(unique_labels) - 1 if -1 in labels else -1
    for label, point in zip(labels, points):
        # outliers are put at the end of the clusters array (they have labels = -1)
        clusters[label].append(point)

    clusters_array = np.array([np.array(x) for x in clusters], dtype=object)
    return clusters_array, outlier_index, labels