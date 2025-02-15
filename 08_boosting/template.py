# Author: Bethany Bronkema
# Date: 3 October 2023
# Project: Boosting
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, RandomizedSearchCV)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score, precision_score)

from tools import get_titanic, build_kaggle_submission

def get_better_titanic():
    '''
    Loads the cleaned titanic dataset but change
    how we handle the age column.
    '''
    # Load in the raw data
    # check if data directory exists for Mimir submissions
    # DO NOT REMOVE
    if os.path.exists('./data/train.csv'):
        train = pd.read_csv('./data/train.csv')
        test = pd.read_csv('./data/test.csv')
    else:
        train = pd.read_csv('08_boosting/data/train.csv')
        test = pd.read_csv('08_boosting/data/test.csv')

    # Concatenate the train and test set into a single dataframe
    # we drop the `Survived` column from the train set
    X_full = pd.concat([train.drop('Survived', axis=1), test], axis=0)

    # The cabin category consist of a letter and a number.
    # We can divide the cabin category by extracting the first
    # letter and use that to create a new category. So before we
    # drop the `Cabin` column we extract these values
    X_full['Cabin_mapped'] = X_full['Cabin'].astype(str).str[0]
    # Then we transform the letters into numbers
    cabin_dict = {k: i for i, k in enumerate(X_full.Cabin_mapped.unique())}
    X_full.loc[:, 'Cabin_mapped'] =\
        X_full.loc[:, 'Cabin_mapped'].map(cabin_dict)

    # We drop multiple columns that contain a lot of NaN values
    # in this assignment
    # Maybe we should
    X_full.drop(
        ['PassengerId', 'Cabin', 'Name', 'Ticket'],
        inplace=True, axis=1)
    # Instead of dropping the Embarked column we replace NaN values
    # with `S` denoting Southampton, the most common embarking
    # location
    age_mode = X_full.Age.mode()[0]
    X_full['Age'].fillna(age_mode, inplace=True)
    # Instead of dropping the fare column we replace NaN values
    # with the 3rd class passenger fare mean.
    fare_mean = X_full[X_full.Pclass == 3].Fare.mean()
    X_full['Fare'].fillna(fare_mean, inplace=True)
    # Instead of dropping the Embarked column we replace NaN values
    # with `S` denoting Southampton, the most common embarking
    # location
    X_full['Embarked'].fillna('S', inplace=True)

    # We then use the get_dummies function to transform text
    # and non-numerical values into binary categories.
    X_dummies = pd.get_dummies(
        X_full,
        columns=['Sex', 'Cabin_mapped', 'Embarked'],
        drop_first=True)

    # We now have the cleaned data we can use in the assignment
    X = X_dummies[:len(train)]
    submission_X = X_dummies[len(train):]
    y = train.Survived
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=5, stratify=y)

    return (X_train, y_train), (X_test, y_test), submission_X

(tr_X, tr_y), (tst_X, tst_y), submission_X = get_better_titanic()

#f = open('1.2_txt', 'w+')
#f.write('Instead of deleting the column, I chose to use the most common value that appeared in the age column. I got this by using the .mode function and replaced the NaN values using the same method as described in the original function for the Fare and Embarked columns')

def rfc_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a random forest classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test)
    '''
    forest = RandomForestClassifier()
    forest.fit(X_train, t_train)
    predictions = forest.predict(X_test)

    return accuracy_score(t_test, predictions), precision_score(t_test, predictions), recall_score(t_test, predictions)

#f = open('2_2.txt', 'w+')
#f.write('For fitting the classifier, I used all the default parameters. This means that 100 decision trees were used and there were no weights added to any of the classes. Gini impurity is the default impurity measure, so I used this as well. I did not impose a maximum or minimum node count, so allowed the trees to grow fully since random forest are fairly robust to overfitting. I retured a value of 0.8134 for accuracy, 0.7732 for precision, and 0.7282 for recall.')

def gb_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a Gradient boosting classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test)
    '''
    gradient = GradientBoostingClassifier()
    gradient.fit(X_train, t_train)
    predictions = gradient.predict(X_test)

    return accuracy_score(t_test, predictions), precision_score(t_test, predictions), recall_score(t_test, predictions)

#f = open('2_4.txt', 'w+')
#f.write('For the Gradient boosting classifier, the accuracy was 0.813, the precision was 0.785, and the recall was 0.709. In comparison to the random forest, the accuracy is the same, the precision is higher, and the recall is lower.')

def param_search(X, y):
    '''
    Perform randomized parameter search on the
    gradient boosting classifier on the dataset (X, y)
    '''
    # Create the parameter grid
    gb_param_grid = {
        'n_estimators': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
        'max_depth': [5, 10, 15, 20, 25, 30, 35, 40, 45],
        'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    # Instantiate the regressor
    gb = GradientBoostingClassifier()
    # Perform random search
    gb_random = RandomizedSearchCV(
        param_distributions=gb_param_grid,
        estimator=gb,
        scoring="accuracy",
        verbose=0,
        n_iter=50,
        cv=4)
    # Fit randomized_mse to the data
    gb_random.fit(X, y)
    # Print the best parameters and lowest RMSE
    return gb_random.best_params_

#f = open('2_5.txt', 'w+')
#f.write('Using the param_search function, it was found that the best values of parameters were n_estimators = 95, max_depth = 5, and learning_rate = 0.1.')

def gb_optimized_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a gradient boosting classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test) with
    your own optimized parameters
    '''
    gradient = GradientBoostingClassifier(n_estimators=95, max_depth=5, learning_rate=0.1)
    gradient.fit(X_train, t_train)
    predictions = gradient.predict(X_test)

    return accuracy_score(t_test, predictions), precision_score(t_test, predictions), recall_score(t_test, predictions)

def _create_submission():
    '''Create your kaggle submission
    '''
    pass
    prediction = None # !!! Your prediction here !!!
    build_kaggle_submission(prediction)
