# Author: Bethany Bronkema
# Date: 05 September 2023
# Project: Sequential Estimation
# Acknowledgements: 
#


from tools import scatter_3d_data, bar_per_axis

import matplotlib.pyplot as plt
import numpy as np


def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    I = np.identity(k)
    X = np.random.multivariate_normal(mean, np.square(var)*I, n)
    return X

np.random.seed(1234)

X = gen_data(300, 3, [0, 1, -1], np.sqrt(3))

def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''
    Performs the mean sequence estimation update
    '''
    mu = mu + (1/(n)) * (x - mu)
    return mu

def _plot_sequence_estimate():
    data = gen_data(100, 3, np.array([0, 0, 0]), 1)
    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        estimates = np.append(estimates, [update_sequence_mean(estimates[i], data[i], i)], axis = 0)
    #plt.plot([e[0] for e in estimates], label='First dimension')
    #plt.plot([e[1] for e in estimates], label='Second dimension')
    #plt.plot([e[2] for e in estimates], label='Third dimension')
    #plt.legend(loc='upper center')
    #plt.show()
    return estimates

def _square_error(y, y_hat):
    sqr_err = np.square(y-y_hat)
    return np.mean(sqr_err)


def _plot_mean_square_error():
    data = gen_data(100, 3, np.array([0, 0, 0]), 1)
    error = np.zeros(len(data))
    estimates = _plot_sequence_estimate()
    for i in range(len(data)):
        error[i] = _square_error(np.mean(data), estimates[i+1,:])
    plt.plot(error)
    plt.show()

# Naive solution to the independent question.