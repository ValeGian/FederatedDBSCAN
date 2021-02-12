from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np

def dbscan(points: np.ndarray, eps, min_pts) -> np.ndarray:
    clustering = DBSCAN(eps=eps, min_samples=min_pts)
    labels = clustering.fit_predict(points)
    return labels