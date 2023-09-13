# Author: Bethany Bronkema
# Date: 6 September
# Project: Linear Regression
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.

import numpy as np
import matplotlib.pyplot as plt

from tools import load_regression_iris
from scipy.stats import multivariate_normal

def mvn_basis(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float
) -> np.ndarray:
    '''
    Multivariate Normal Basis Function
    The function transforms (possibly many) data vectors <features>
    to an output basis function output <fi>
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional
    data vectors.
    * mu: [MxD] matrix of M D-dimensional mean vectors defining
    the multivariate normal distributions.
    * sigma: All normal distributions are isotropic with sigma*I covariance
    matrices (where I is the MxM identity matrix)
    Output:
    * fi - [NxM] is the basis function vectors containing a basis function
    output fi for each data vector x in features
    '''
    pass

    phi_guess = np.empty((features.shape[0],mu.shape[0]))
    for i in range(features.shape[0]):
        for k in range(mu.shape[0]):
            phi_guess[i,k] = multivariate_normal.pdf(features[i], mu[k], sigma); phi_guess
    return phi_guess

X, t = load_regression_iris()
N, D = X.shape
M, sigma = 10, 10
mu = np.zeros((M, D))
for i in range(D):
    mmin = np.min(X[i, :])
    mmax = np.max(X[i, :])
    mu[:, i] = np.linspace(mmin, mmax, M)
fi = mvn_basis(X, mu, sigma)

def _plot_mvn():
    for i in range(fi.shape[1]):
        plt.plot(np.linspace(0, fi.shape[0], fi.shape[0]), fi[:,i])
    plt.show()

def max_likelihood_linreg(
    fi: np.ndarray,
    targets: np.ndarray,
    lamda: float
) -> np.ndarray:
    '''
    Estimate the maximum likelihood values for the linear model

    Inputs :
    * Fi: [NxM] is the array of basis function vectors
    * t: [Nx1] is the target value for each input in Fi
    * lamda: The regularization constant

    Output: [Mx1], the maximum likelihood estimate of w for the linear model
    '''
    I = np.identity(fi.shape[1])
    omegas = np.linalg.inv(lamda * I + (np.transpose(fi) @ fi)) @ np.transpose(fi) @ targets
    return omegas

def linear_model(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    w: np.ndarray
) -> np.ndarray:
    '''
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * sigma: All normal distributions are isotropic with s*I covariance
    matrices (where I is the MxM identity matrix).
    * w: [Mx1] the weights, e.g. the output from the max_likelihood_linreg
    function.

    Output: [Nx1] The prediction for each data vector in features
    '''
    fi = mvn_basis(features, mu, sigma) # same as before
    y = fi @ w
    return y

lamda = 0.001
wml = max_likelihood_linreg(fi, t, lamda) # as before
prediction = linear_model(X, mu, sigma, wml)

f = open("1_5_1.txt", "w+")
f.write("As can be clearly seen from the plot of the square error, these predictions are reasonably accurate for class 0 of irises, less accurate for class 1, and even less accuate for class 2.")

x = np.linspace(0, len(t), len(t))
sqr_err = np.square(t-prediction)
plt.plot(x, sqr_err, label="square error")
leg = plt.legend(loc='upper center')
plt.show()
