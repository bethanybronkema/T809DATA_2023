# Author: Bethany Bronkema
# Date: 18 September 2023
# Project: Classification
# Acknowledgements: 
#


from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    collect_mu = np.empty((0,features.shape[1]))
    for i in range(features.shape[0]):
        if targets[i] == selected_class:
            collect_mu = np.append(collect_mu, [features[i]], axis = 0)
    return np.mean(collect_mu, axis = 0)

def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    collect_cov = np.empty((0, features.shape[1]))
    for i in range(features.shape[0]):
        if targets[i] == selected_class:
            collect_cov = np.append(collect_cov, [features[i]], axis = 0)
    return np.cov(collect_cov, rowvar = False)

features, targets, classes = load_iris()
(train_features, train_targets), (test_features, test_targets)\
    = split_train_test(features, targets, train_ratio=0.6)   

def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    prob = multivariate_normal.pdf(feature, class_mean, class_covar)
    return prob

class_mean = mean_of_class(train_features, train_targets, 0)
class_cov = covar_of_class(train_features, train_targets, 0)

def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = np.empty((0, test_features.shape[1])), np.empty((0, test_features.shape[1], test_features.shape[1]))
    for class_label in classes:
        means = np.append(means, [mean_of_class(train_features, train_targets, class_label)], axis = 0) 
        covs = np.append(covs, [covar_of_class(train_features, train_targets, class_label)], axis = 0)
    likelihoods = np.empty((0, len(classes)))
    likelihood_row = np.zeros(len(classes))
    for i in range(test_features.shape[0]):
        for j in classes:
            likelihood_row[j] = likelihood_of_class(test_features[i], means[j], covs[j])
        likelihoods = np.append(likelihoods, [likelihood_row], axis = 0)
    return np.array(likelihoods)

def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    pred = np.zeros(likelihoods.shape[0])
    for i in range(likelihoods.shape[0]):
        pred[i] = np.argmax(likelihoods[i])
    return pred

likelihoods_max = maximum_likelihood(train_features, train_targets, test_features, classes)
predictions_max = predict(likelihoods_max)

def maximum_aposteriori(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum a posteriori for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = np.empty((0, test_features.shape[1])), np.empty((0, test_features.shape[1], test_features.shape[1]))
    for class_label in classes:
        means = np.append(means, [mean_of_class(train_features, train_targets, class_label)], axis = 0) 
        covs = np.append(covs, [covar_of_class(train_features, train_targets, class_label)], axis = 0)
    likelihoods = np.empty((0, len(classes)))
    likelihood_row = np.zeros(len(classes))
    sort, count = np.unique(test_targets, return_counts = True)
    for i in range(test_features.shape[0]):
        for j in classes:
            likelihood_row[j] = likelihood_of_class(test_features[i], means[j], covs[j]) * (count[j]/test_features.shape[0])
        likelihoods = np.append(likelihoods, [likelihood_row], axis = 0)
    return np.array(likelihoods)

#f = open("2_2.txt", "w+")
#f.write("For this situation, there is not a difference between either the accuracy or the confusion matrix for the two methods. Each produce an accuracy of 0.983, corresponding to 1 error in the confusion matrix (got class 2, actual class 1). This similarity makes sense since the sample provided to analyze had an approximately even split between the classes, so accounting for the aposteriori probability did not change the result.")

likelihoods_apost = maximum_aposteriori(train_features, train_targets, test_features, classes)
predictions_apost = predict(likelihoods_apost)

def accuracy(
    predictions: np.ndarray,
    test_targets: np.ndarray,
) -> float:
    
    difference = test_targets - predictions
    count = np.count_nonzero(difference)
    return (1 - (count/len(predictions)))

max_accuracy = accuracy(predictions_max, test_targets)
apost_accuracy = accuracy(predictions_apost, test_targets)
#print(max_accuracy)
#print (apost_accuracy)

def confusion_matrix(
    predictions: np.ndarray,
    test_targets: np.ndarray,
    classes: list,
) -> np.ndarray:
    con_mat = np.zeros((len(classes), len(classes)))
    for i in range(len(test_targets)):
        con_mat[int(predictions[i]),test_targets[i]] += 1
    return con_mat

max_conmat = confusion_matrix(predictions_max, test_targets, classes)
apost_conmat = confusion_matrix(predictions_apost, test_targets, classes)
#print(max_conmat)
#print(apost_conmat)