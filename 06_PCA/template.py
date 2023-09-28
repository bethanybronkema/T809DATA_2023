# Author: Bethany Bronkema
# Date: 19 September 2023
# Project: PCA
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.



import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from tools import load_cancer


def standardize(X: np.ndarray) -> np.ndarray:
    '''
    Standardize an array of shape [N x 1]

    Input arguments:
    * X (np.ndarray): An array of shape [N x 1]

    Returns:
    (np.ndarray): A standardized version of X, also
    of shape [N x 1]
    '''
    x_hat = np.empty((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_hat[i,j] = (X[i,j] - np.mean(X[:,j]))/np.std(X[:,j])
    return x_hat

def scatter_standardized_dims(
    X: np.ndarray,
    i: int,
    j: int,
):
    '''
    Plots a scatter plot of N points where the n-th point
    has the coordinate (X_ni, X_nj)

    Input arguments:
    * X (np.ndarray): A [N x f] array
    * i (int): The first index
    * j (int): The second index
    '''
    x_hat = standardize(X)
    plt.scatter(x_hat[:,i], x_hat[:,j], s = 1/60)

def _scatter_cancer():
    X, y = load_cancer()
    for j in range(X.shape[1]):
        plt.subplot(5, 6, j+1)
        scatter_standardized_dims(X, 0, j)
    plt.show()

f = open("1_4.txt", "w+")
f.write("Judging from the plots, the ones that display approximately a straight line are the ones that correlate best with dimension 1. The dimensions 3 and 4 seem to display the most significant correlations. According to the chart at the start of the assignment, these features correspond to perimeter mean and area mean. This makes sense as dimension 1 is radius mean, and the perimeter and area of a sample will be related to its radius.")

X, y = load_cancer()
pca = PCA()
n_properties = X.shape[1]
x_standard = standardize(X)
pca.fit_transform(x_standard)
components = pca.components_

def _plot_pca_components():
    X, y = load_cancer()
    for i in range(X.shape[1]):
        plt.subplot(5, 6, i+1)
        plt.plot(components[i])
        ax = plt.gca()
        # Hide X and Y axes label marks
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        # Hide X and Y axes tick marks
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title('PCA %i' %(i+1))
        plt.tight_layout()
    plt.show()

def _plot_eigen_values():
    eig_values = pca.explained_variance_
    i = np.linspace(1, eig_values.shape[0], eig_values.shape[0])
    plt.scatter(i, eig_values)
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()

def _plot_log_eigen_values():
    eig_values = pca.explained_variance_
    i = np.linspace(1, eig_values.shape[0], eig_values.shape[0])
    log_values = np.log10(eig_values)
    plt.scatter(i, log_values)
    plt.xlabel('Eigenvalue index')
    plt.ylabel('$\log_{10}$ Eigenvalue')
    plt.grid()
    plt.show()

def _plot_cum_variance():
    eig_values = pca.explained_variance_
    i = np.linspace(1, eig_values.shape[0], eig_values.shape[0])
    tot_sum = np.cumsum(eig_values)
    y = tot_sum/tot_sum[-1]
    plt.scatter(i, y)
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Percentage variance')
    plt.grid()
    plt.show()

f = open("3_4.txt", "w+")
f.write("For the eigenvalue plot, the trend is that the eigen values sharply decrease until the values level off between values 10-15. This happens because all 30 dimensions are not required to contain most of the information in the dataset, showing that the data could be reduced from 30 dimensions to somewhere around 7 while still retaining most of the information. The cumulative variance shows the same information, but as a percentage of variance retained at each additional eigenvalue. For example, it shows that if we reduce the dimensions of the data to 7, we retain approximately 90percent of the information. The remaining dimensions provide increasingly less information as they are added, as seen by the shape of the plot.")