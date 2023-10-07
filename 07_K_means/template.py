# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.



import numpy as np
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from typing import Union

from tools import load_iris, image_to_numpy, plot_gmm_results


def distance_matrix(
    X: np.ndarray,
    Mu: np.ndarray
) -> np.ndarray:
    '''
    Returns a matrix of euclidian distances between points in
    X and Mu.

    Input arguments:
    * X (np.ndarray): A [n x f] array of samples
    * Mu (np.ndarray): A [k x f] array of prototypes

    Returns:
    out (np.ndarray): A [n x k] array of euclidian distances
    where out[i, j] is the euclidian distance between X[i, :]
    and Mu[j, :]
    '''
    
    dist = np.zeros([X.shape[0], Mu.shape[0]])
    
    for j in range(Mu.shape[0]):
        for i in range(X.shape[0]):
            sum = X[i]-Mu[j]
            new_sum = 0
            for f in range(X.shape[1]):
                new_sum += np.square(sum[f])
                dist[i, j] = np.sqrt(new_sum)
    return dist

def determine_r(dist: np.ndarray) -> np.ndarray:
    '''
    Returns a matrix of binary indicators, determining
    assignment of samples to prototypes.

    Input arguments:
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    out (np.ndarray): A [n x k] array where out[i, j] is
    1 if sample i is closest to prototype j and 0 otherwise.
    '''
    matrix = np.zeros([dist.shape[0], dist.shape[1]])
    for i in range(dist.shape[0]):
        cluster = np.argmin(dist[i])
        matrix[i, cluster] = 1
    return matrix
            
def determine_j(R: np.ndarray, dist: np.ndarray) -> float:
    '''
    Calculates the value of the objective function given
    arrays of indicators and distances.

    Input arguments:
    * R (np.ndarray): A [n x k] array where out[i, j] is
        1 if sample i is closest to prototype j and 0
        otherwise.
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    * out (float): The value of the objective function
    '''
    cost = 0
    for n in range(R.shape[0]):
        for k in range(R.shape[1]):
            cost += R[n, k] * dist[n, k]
        
    return cost/R.shape[0]

def update_Mu(
    Mu: np.ndarray,
    X: np.ndarray,
    R: np.ndarray
) -> np.ndarray:
    '''
    Updates the prototypes, given arrays of current
    prototypes, samples and indicators.

    Input arguments:
    Mu (np.ndarray): A [k x f] array of current prototypes.
    X (np.ndarray): A [n x f] array of samples.
    R (np.ndarray): A [n x k] array of indicators.

    Returns:
    out (np.ndarray): A [k x f] array of updated prototypes.
    '''
    for k in range(R.shape[1]):
        num = 0
        den = 0
        for n in range(R.shape[0]):
            num = num + R[n, k] * X[n]
            den = den + R[n, k]
        Mu[k] = num/den
            
    return Mu

def k_means(
    X: np.ndarray,
    k: int,
    num_its: int
) -> Union[list, np.ndarray, np.ndarray]:
    # We first have to standardize the samples
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standard = (X-X_mean)/X_std
    # run the k_means algorithm on X_st, not X.

    # we pick K random samples from X as prototypes
    nn = sk.utils.shuffle(range(X_standard.shape[0]))
    Mu = X_standard[nn[0: k], :]

    j_hat = np.zeros(num_its)
    for i in range(num_its):
        dist = distance_matrix(X_standard, Mu)
        resps = determine_r(dist)
        j_hat[i] = determine_j(resps, dist)
        Mu = update_Mu(Mu, X_standard, resps)

    # Then we have to "de-standardize" the prototypes
    for i in range(k):
        Mu[i, :] = Mu[i, :] * X_std + X_mean

    return Mu, resps, j_hat

def _plot_j():
    X, y, c = load_iris()
    Mu, resps, j_hat = k_means(X, 4, 10)
    plt.plot(j_hat)
    plt.show()

def _plot_multi_j():
    K = [2, 3, 5, 10]
    
    X, y, c = load_iris()
    for k in K:
        Mu, resps, j_hat = k_means(X, k, 10)
        plt.plot(j_hat)
    plt.legend('2' '3' '5' '10')
    plt.show()

#f = open('1_8.txt', 'w+')
#f.write('According to the plot, the best value of k is 10 clusters. Setting k = n would be an example of overfitting, because each point would just be assigned to its own cluster and the cost function would be zero. This sort of classifier would not be able to provide useful information.')

def k_means_predict(
    X: np.ndarray,
    t: np.ndarray,
    classes: list,
    num_its: int
) -> np.ndarray:
    '''
    Determine the accuracy and confusion matrix
    of predictions made by k_means on a dataset
    [X, t] where we assume the most common cluster
    for a class label corresponds to that label.

    Input arguments:
    * X (np.ndarray): A [n x f] array of input features
    * t (np.ndarray): A [n] array of target labels
    * classes (list): A list of possible target labels
    * num_its (int): Number of k_means iterations

    Returns:
    * the predictions (list)
    '''
    Mu, resps, j_hat = k_means(X, len(classes), num_its)
    predictions = np.zeros(X.shape[0])
    collect = []
    for k in range(len(classes)):
        for n in range(X.shape[0]):
            if t[n] == k:
                collect = np.append(collect, np.nonzero(resps[n]))
        values, counts = np.unique(collect, return_counts = True)
        assign = values[counts.argmax()]
        for n in range(X.shape[0]):
            if t[n] == k:
                predictions[n] = assign
    return predictions   

def _iris_kmeans_accuracy():
    X, y, c = load_iris()
    predictions = k_means_predict(X, y, c, 5)
    accuracy = accuracy_score(y, predictions)
    con_mat = confusion_matrix(y, predictions)
    print(accuracy)
    print(con_mat)

_iris_kmeans_accuracy()
#f = open('1_10.txt', 'w+')
#f.write('The accuracy is 0.6 or 60 percent and the confusion matrix is [50 0 0], [10 40 0], [33 17 0]')

def _my_kmeans_on_image():
    image, (w, h) = image_to_numpy('07_K_means/images/buoys.png')
    cluster = k_means(image, 7, 5)
    return cluster

def plot_image_clusters(n_clusters: int):
    '''
    Plot the clusters found using sklearn k-means.
    '''
    image, (w, h) = image_to_numpy('07_K_means/images/buoys.png')
    kmeans = KMeans(n_clusters=n_clusters).fit(image)
    plt.subplot(121)
    plt.imshow(image.reshape(w, h, 3))
    plt.subplot(122)
    # uncomment the following line to run
    plt.imshow(kmeans.labels_.reshape(w, h), cmap="plasma")
    plt.show()
