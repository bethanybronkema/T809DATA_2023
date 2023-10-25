from typing import Union
import numpy as np
import matplotlib.pyplot as plt

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
    E_total = np.zeros(iterations)
    misclassification_rate = np.zeros(iterations)
    last_guesses = np.zeros(X_train.shape[0])
    N = X_train.shape[0]

    for i in range(iterations):
        dE1_total = np.zeros(W1.shape)
        dE2_total = np.zeros(W2.shape)
        for j in range(X_train.shape[0]):
            target_y = np.zeros(K)
            target_y[t_train[j]] = 1.0
            y, dE1, dE2 = backprop(X_train[j], target_y, M, K, W1, W2)
            dE1_total += dE1
            dE2_total += dE2
            last_guesses[j] = np.argmax(y)
            E_total[i] -= (target_y@np.log(y)) + ((1-target_y)@np.log(1-y))
            misclassification_rate[i] += np.count_nonzero(t_train[j] - last_guesses[j])
        W1 = W1 - eta * dE1_total/N
        W2 = W2 - eta * dE2_total/N
    return W1, W2, E_total/N, misclassification_rate/N, last_guesses

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
    con_mat = np.zeros((K, K))
    guesses = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        y, z0, z1, a1, a2 = ffnn(X[i], M, K, W1, W2)
        guesses[i] = np.argmax(y)
        con_mat[int(guesses[i]),int(test_targets[i])] += 1
    accuracy = 1-((np.count_nonzero(test_targets - guesses))/X.shape[0])
    #print('Accuracy:', accuracy)
    #print('Confusion Matrix:\n', con_mat)
    plt.plot(E_total)
    plt.plot(misclassification_rate)
    plt.legend(['Total Error', 'Misclassification Rate'])
    #plt.show()

np.random.seed(123)

features, targets, classes = load_iris()
(train_features, train_targets), (test_features, test_targets) = \
    split_train_test(features, targets)
    
# initialize the random seed to get predictable results
np.random.seed(1234)
K = 3  # number of classes
M = 6
D = train_features.shape[1]

# Initialize two random weight matrices
W1 = 2 * np.random.rand(D + 1, M) - 1
W2 = 2 * np.random.rand(M + 1, K) - 1
W1tr, W2tr, E_total, misclassification_rate, last_guesses = train_nn(
    train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1)

if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    """
    pass