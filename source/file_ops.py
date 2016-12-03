# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 18:19:10 2016

@author: Eric
"""
import os
import csv
import numpy as np
import constants


def load_news_dict():
    """Load all the data from file into a dictionary."""

    # Retrieve the names of the data.
    prediction_name, feature_names = constants.data_labels()

    # Get the path to the data file.
    filepath = constants.data_path()

    # Create a dictionary that will map the names to the lists of values.
    features_dict = {}

    # Populate the dictionary with keys and corresponding empty lists.
    for name in feature_names:
        features_dict[name] = []

    # Read the data from the file and fill the dictionary lists.
    with open(filepath) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for name in feature_names:
                # The raw data has a space before each column header name.
                features_dict[name].append(row[" " + name])

    # Determine the number of samples.
    N = len(features_dict[prediction_name])

    return features_dict, N


def load_news_matrix():
    """Convert the data from dictionary into a matrix."""

    # Retrieve the names of the data.
    prediction_name, feature_names = constants.data_labels()

    # Get the dictionary.
    features_dict, N = load_news_dict()
    # Create a list and populate it with lists in the order of feature names.
    features_list = []
    for name in feature_names:
        features_list.append(features_dict[name])

    # Use Numpy functions to convert the list of lists to matrix.
    X = np.transpose(np.stack(features_list)).astype(float)

    return X, N


def write_array(A, name='temp_array.csv'):
    cwd = os.getcwd()
    def_path = os.path.join(cwd, name)
    array = np.expand_dims(A, 1)
    with open(def_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(array)


def write_matrix(M, name='temp_matrix.csv'):
    cwd = os.getcwd()
    def_path = os.path.join(cwd, name)
    with open(def_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(M)
