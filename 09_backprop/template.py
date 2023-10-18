from typing import Union
import numpy as np

from tools import load_iris, split_train_test


def sigmoid(x: float) -> float:
    '''
    Calculate the sigmoid of x
    '''
    if x < -100:
        return 0
    else:
        return 1/(1+np.exp(-x))


def d_sigmoid(x: float) -> float:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    return sigmoid(x)*(1-sigmoid(x))


def perceptron(
    x: np.ndarray,
    w: np.ndarray
) -> Union[float, float]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    a = 0
    for i in range(x.shape[0]):
        a += w[i]*x[i]
    return a, sigmoid(a)


def ffnn(
    x: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    '''
    
    z0 = np.append(1.0, x)
    a1 = np.zeros(M)
    z1 = np.zeros(M+1)
    z1[0] = 1
    a2 = np.zeros(K)
    y = np.zeros(K)
    for m in range(M):
        a1[m], z1[m+1] = perceptron(z0, W1[:, m])
    for k in range(K):
        a2[k], y[k] = perceptron(z1, W2[:, k])
    return y, z0, z1, a1, a2

np.random.seed(123)

features, targets, classes = load_iris()
(train_features, train_targets), (test_features, test_targets) = \
    split_train_test(features, targets)

def backprop(
    x: np.ndarray,
    target_y: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    '''
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    delta_k = y - target_y
    delta_j = np.zeros(M)
    for j in range(M):
        for k in range(K):
            delta_j[j] = delta_j[j] + d_sigmoid(a1[j])*W2[j+1, k]*delta_k[k]
    dE1 = np.zeros([W1.shape[0], W1.shape[1]])
    dE2 = np.zeros([W2.shape[0], W2.shape[1]])
    for i in range(len(z0)):
        for j in range(len(delta_j)):
            dE1[i, j] = delta_j[j]*z0[i]
    for j in range(len(z1)):
        for k in range(len(delta_k)):
            dE2[j, k] = delta_k[k]*z1[j]

    return y, dE1, dE2

# initialize random generator to get predictable results
np.random.seed(42)

K = 3  # number of classes
M = 6
D = train_features.shape[1]

x = features[0, :]

# create one-hot target for the feature
target_y = np.zeros(K)
target_y[targets[0]] = 1.0

# Initialize two random weight matrices
W1 = 2 * np.random.rand(D + 1, M) - 1
W2 = 2 * np.random.rand(M + 1, K) - 1

y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)
print('y: ', y, '\ndE1: ', dE1, '\ndE2: ', dE2)

def train_nn(
    X_train: np.ndarray,
    t_train: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
    iterations: int,
    eta: float
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network
    made and the actual target
    3. Backpropagating the error through the network to adjust
    the weights.
    '''
    ...


def test_nn(
    X: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> np.ndarray:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''
    ...


if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    """
    pass