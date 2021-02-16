from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np

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


def plot2Dcluster(points: np.ndarray, labels:np.ndarray):
    color_range = cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))
    colors = []
    count_outliers = 0
    for label in labels:
        if label == -1:
            count_outliers += 1
            colors.append([0, 0, 0, 1])  # Black used for outliers
        else:
            colors.append(color_range[label])
            #print(f'{label} - {color_range[label]}')

    plt.scatter(points[:, 0], points[:, 1], color=colors)
    plt.title(f'{len(labels)} Points - {len(np.unique(labels)) - (1 if count_outliers > 0 else 0)} Clusters - {count_outliers} Outliers')
    plt.show()


if __name__ == '__main__':
    points = np.array([[1, 1], [2, 4], [3, 5], [2, 2]])
    labels = np.array([1, 0, 0, -1])
    plot2Dcluster(points, labels)