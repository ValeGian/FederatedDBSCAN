from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import numpy as np
import warnings


markers = ["o", "s", "v", "^", "<", ">", "p", "P", "*", "H", "X", "D", "1", "+"]


def plotGridMap(contributionMap):
    maxX = max([cord[0] for cord in contributionMap.keys()])
    minX = min([cord[0] for cord in contributionMap.keys()])
    maxY = max([cord[1] for cord in contributionMap.keys()])
    minY = min([cord[1] for cord in contributionMap.keys()])

    x_shift = 0
    if minX < 0:
        x_shift = -1 * minX

    y_shift = 0
    if minY < 0:
        y_shift = -1 * minY

    matrix = np.zeros((maxY + 1 + y_shift, maxX + 1 + x_shift))

    for key, value in contributionMap.items():
        matrix[key[1] + y_shift][key[0] + x_shift] += value

    plt.imshow(matrix, cmap='gray_r', origin='lower')
    plt.show()


def plotCluster(points: np.ndarray, labels: np.ndarray, message="", marker="o"):
    dimensions = len(points[0])
    if dimensions == 2:
        plot2Dcluster(points, labels, message, marker)
    elif dimensions == 3:
        plot3Dcluster(points, labels, message, marker)


def plot2Dcluster(points: np.ndarray, labels:np.ndarray, message="", marker="o"):
    color_range = cm.rainbow(np.linspace(0, 1, np.max(np.unique(labels))+1))
    colors = []
    count_outliers = 0

    for label in labels:
        if label == -1:
            count_outliers += 1
            colors.append([0, 0, 0, 1])  # Black used for outliers
        else:
            colors.append(color_range[label])

    plt.scatter(points[:, 0], points[:, 1], color=colors, marker=marker)
    plt.title(f'{message} - '
              f'{len(labels)} Points - '
              f'{len(np.unique(labels)) - (1 if count_outliers > 0 else 0)} Clusters - '
              f'{count_outliers} Outliers')
    plt.show()


def plot3Dcluster(points: np.ndarray, labels:np.ndarray, message="", marker="o"):
    warnings.filterwarnings("ignore")
    color_range = cm.rainbow(np.linspace(0, 1, np.max(np.unique(labels)) + 1))
    colors = []
    count_outliers = 0

    for label in labels:
        if label == -1:
            count_outliers += 1
            colors.append([0, 0, 0, 1])  # Black used for outliers
        else:
            colors.append(color_range[label])

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, marker=marker)

    print(f'{message} - '
          f'{len(labels)} Points - '
          f'{len(np.unique(labels)) - (1 if count_outliers > 0 else 0)} Clusters - '
          f'{count_outliers} Outliers')
    plt.show()


def plot_curves(curves, MValues, x_message, y_message):
    for i in range(len(curves)):
        plt.plot(curves[i][0], curves[i][1], label=f'MinPts = {MValues[i]}', marker=markers[i])

    plt.xlabel(x_message)
    plt.ylabel(y_message)

    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.show()


if __name__ == '__main__':
    x1 = [10, 20, 30]
    y1 = [20, 40, 10]
    curve1 = []
    curve1.append(x1)
    curve1.append(y1)

    x2 = [10, 20, 30]
    y2 = [40, 10, 30]
    curve2 = []
    curve2.append(x2)
    curve2.append(y2)

    x3 = [10, 20, 30]
    y3 = [10, 20, 30]
    curve3 = []
    curve3.append(x3)
    curve3.append(y3)

    metric1_curves = []
    metric1_curves.append(curve1)
    metric1_curves.append(curve2)
    metric1_curves.append(curve3)

    x1 = [10, 20, 30]
    y1 = [20, 40, 10]
    curve1 = []
    curve1.append(x1)
    curve1.append(y1)

    x2 = [10, 20, 30]
    y2 = [40, 10, 30]
    curve2 = []
    curve2.append(x2)
    curve2.append(y2)

    x3 = [10, 20, 30]
    y3 = [10, 20, 30]
    curve3 = []
    curve3.append(x3)
    curve3.append(y3)

    metric2_curves = []
    metric2_curves.append(curve1)
    metric2_curves.append(curve2)
    metric2_curves.append(curve3)

    x1 = [10, 20, 30]
    y1 = [20, 40, 10]
    curve1 = []
    curve1.append(x1)
    curve1.append(y1)

    x2 = [10, 20, 30]
    y2 = [40, 10, 30]
    curve2 = []
    curve2.append(x2)
    curve2.append(y2)

    x3 = [10, 20, 30]
    y3 = [10, 20, 30]
    curve3 = []
    curve3.append(x3)
    curve3.append(y3)

    metric3_curves = []
    metric3_curves.append(curve1)
    metric3_curves.append(curve2)
    metric3_curves.append(curve3)

    plot_curves(metric1_curves, [3, 4, 6], "L", "PURITY")
    plot_curves(metric2_curves, [3, 4, 6], "L", "AMI")
    plot_curves(metric3_curves, [3, 4, 6], "EPSI", "ARI")