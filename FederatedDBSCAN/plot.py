from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plotGridMap(contributionMap):
    maxX = max([cord[0] for cord in contributionMap.keys()])
    minX = min([cord[0] for cord in contributionMap.keys()])
    maxY = max([cord[1] for cord in contributionMap.keys()])
    minY = min([cord[1] for cord in contributionMap.keys()])

    x_shift = -1 * minX
    y_shift = -1 * minY

    matrix = np.zeros((maxY + 1 + y_shift, maxX + 1 + x_shift))

    for key, value in contributionMap.items():
        matrix[key[1] + y_shift][key[0] + x_shift] += value

    plt.imshow(matrix, cmap='gray_r', origin='lower')
    plt.show()


def plot2Dcluster(clusters: np.ndarray, outlierIndex = -1):
    colors = cm.rainbow(np.linspace(0, 1, len(clusters)))
    count = 0
    count_outliers = 0
    for cluster, col in zip(clusters, colors):
        if count == outlierIndex:
            count_outliers += len(cluster)
            col = [0, 0, 0, 1]  # Black used for outliers
        X1 = [cord[0] for cord in cluster]
        X2 = [cord[1] for cord in cluster]
        plt.scatter(X1, X2, color=col)
        count += 1

    if outlierIndex >= 0:
        count -= 1

    plt.title(f'{count} Clusters - {count_outliers} Outliers')
    plt.show()


if __name__ == '__main__':
    test_dict = {(0, 1): 4, (2, 2): 6, (3, 1): 7, (1, 2): 10, (3, 2): 11, (-3, 1): 7, (2, -1): 2}

    clusters = [[(0, 1), (0, 2), (0, 3), (0, 4)], [(1, 1), (1, 5), (1, 3), (1, 2)], [(2, 1), (2, 13)], [(3, 10), (3, 11)]]
    a = np.array([np.array(x) for x in clusters], dtype=object)
    plot2Dcluster(a)