# Author: Bethany Bronkema
# Date: 30 August 2023
# Project: Decision Trees
# Acknowledgements: 
#


from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

from tools import load_iris, split_train_test


def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''
    sum = np.zeros(len(classes))
    prob_class = np.zeros(len(classes))
    for i in range(len(classes)):
        for j in range(len(targets)):
            if targets[j] == classes[i]:
                sum[i] += 1
    prob_class = sum/len(targets)

    return prob_class

def split_data(
    features: np.ndarray,
    targets: np.ndarray,
    split_feature_index: int,
    theta: float
) -> Union[tuple, tuple]:
    '''
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    '''
    #create the lists
    features_1 = []
    targets_1 = []

    features_2 = []
    targets_2 = []

    #split the data
    for i in range(len(features)):
        if features[i,split_feature_index] < theta:
            features_1.append(features[i])
            targets_1.append(targets[i])
        else:
            features_2.append(features[i])
            targets_2.append(targets[i])    

    return (features_1, targets_1), (features_2, targets_2)

features, targets, classes = load_iris()
(f_1, t_1), (f_2, t_2) = split_data(features, targets, 2, 4.65)

def gini_impurity(targets: np.ndarray, classes: list) -> float:
    '''
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    '''
    sum = 0
    prior_prob = prior(targets, classes)
    for i in range(len(prior_prob)):
        sum += np.square(prior_prob[i])        
    impurity = 1/2 * (1-sum)
    return impurity

def weighted_impurity(
    t1: np.ndarray,
    t2: np.ndarray,
    classes: list
) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    g1 = gini_impurity(t1, classes)
    g2 = gini_impurity(t2, classes)
    n = len(t1) + len(t2)
    impurity = (len(t1)*g1)/n + (len(t2)*g2)/n
    return impurity

def total_gini_impurity(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    split_feature_index: int,
    theta: float
) -> float:
    '''
    Calculate the gini impurity for a split on split_feature_index
    for a given dataset of features and targets.
    '''
    (f_1, t_1), (f_2, t_2) = split_data(features, targets, split_feature_index, theta)
    impurity = weighted_impurity(t_1, t_2, classes)
    return impurity

def brute_best_split(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    num_tries: int
) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    '''
    best_gini, best_dim, best_theta = float("inf"), None, None
    # iterate feature dimensions
    store = 1
    for i in range(features.shape[1]):
        # create the thresholds
        thetas = np.linspace(min(features[:,i]),  max(features[:,i]), num_tries + 2)
        # iterate thresholds
        for theta in thetas:
            test = total_gini_impurity(features, targets, classes, i, theta)
            if test < store:
                store = test
                best_gini, best_dim, best_theta = store, i, theta

    return best_gini, best_dim, best_theta

class IrisTreeTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = [0, 1, 2],
        train_ratio: float = 0.8
    ):
        '''
        train_ratio: The ratio of the Iris dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets),\
            (self.test_features, self.test_targets) =\
            split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()

    def train(self):
        return self.tree.fit(self.train_features, self.train_targets)

    def accuracy(self):
        return self.tree.score(self.test_features, self.test_targets)
    
    def plot(self):
        plot_tree(self.tree)
        plt.show()

    def guess(self):
        return self.tree.predict(self.test_features)

    def confusion_matrix(self):
        con_mat = np.zeros((3, 3))
        guess = self.guess()
        targ = self.test_targets
        for i in range(len(targ)):
            con_mat[guess[i],targ[i]] += 1
        return con_mat
            
