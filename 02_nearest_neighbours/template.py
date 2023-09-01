# Author: Bethany Bronkema
# Date: 01 September 2023
# Project: HW1 kNN
# Acknowledgements: 
#

import numpy as np
import matplotlib.pyplot as plt

from tools import load_iris, split_train_test, plot_points

d, t, classes = load_iris()
#plot_points(d, t)

def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    d = 0
    for i in range(len(x)):
        d += np.square(y[i]-x[i])
    return np.sqrt(d)

d, t, classes = load_iris()
x, points = d[0,:], d[1:, :]
x_target, point_targets = t[0], t[1:]

def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and and many
    points
    '''
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        distances[i] = euclidian_distance(x, points[i])
    return distances

def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    '''
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points
    '''
    dist_list = euclidian_distances(x, points)
    ind_sort = np.argsort(dist_list)
    return ind_sort[0:k]

def vote(targets, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    count = np.zeros(len(classes))
    for i in range(len(classes)):
        count[i] = np.count_nonzero(targets == classes[i])
    return np.argmax(count)

def knn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    near = k_nearest(x, points, k)
    return vote(point_targets[near], classes)

def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    check = np.zeros(len(points))
    for i in range(len(points)):
        new_points = np.delete(points, i, 0)
        new_point_targets = np.delete(point_targets, i, 0)
        check[i] = knn(points[i,:], new_points, new_point_targets, classes, k)
    return check    

d, t, classes = load_iris()
(d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio=0.8)

def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    predictions = knn_predict(points, point_targets, classes, k)
    check = predictions - point_targets
    count = np.count_nonzero(check)
    return (1 - (count/len(predictions)))

def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    con_mat = np.zeros((len(classes), len(classes)))
    guess = knn_predict(points, point_targets, classes, k)
    targ = point_targets
    for i in range(len(targ)):
        con_mat[int(guess[i]),targ[i]] += 1
    return con_mat

def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
) -> int:
    score = 0
    N = len(points)
    for k in range(1,(N-1)):
        accur = knn_accuracy(points, point_targets, classes, k)
        if accur > score:
            score = accur
            best = k
    return best

def knn_plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
):
    predictions = knn_predict(points, point_targets, classes, k)
        
    colors = ['yellow', 'purple', 'blue']
    for i in range(points.shape[0]):
        if predictions[i] == point_targets[i]:
            [x, y] = points[i,:2]
            plt.scatter(x, y, c=colors[point_targets[i]], edgecolors='green',
                linewidths=2)
        else:
            [x, y] = points[i,:2]
            plt.scatter(x, y, c=colors[point_targets[i]], edgecolors='red',
                linewidths=2)
    plt.title('Yellow=0, Purple=1, Blue=2')
    plt.show()
