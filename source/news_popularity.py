# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 19:34:56 2016

@author: Eric
"""

import file_ops as fo
import pca_calcs
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.graph_objs as go
from sklearn.decomposition import PCA
import classifiers as cls
import math

#####################################################################
# Set the variables that will be used for the calculations.

threshold = 1400  # Define the number of shares that indicate popular.
B = 20  # Define the number of histogram bins to use.
plotVariance = 0  # Turns the variance plot on/off.
plot2D = 1  # Turns the 2D plot on/off.
plot3D = 0  # Turns the 3D plot on/off.
plothistogram = 0  # Turns this histogram plot on/off.
rescale = 1  # Turn rescaling on or off.
manual_pca = 1  # If equal to 0 then use sklearn for PCA.
calc_histogram = 1  # Turn histogram calculation on/off.

#####################################################################

# Import the features from the raw data file.
X, T, N = fo.load_feature_matrix(shuffle=True)
print("The number of samples is ", N)

# Divide the data into training and testing data.
N_train = int(0.75 * N)
N_test = N - N_train
X_train = X[0:N_train, :]
X_test = X[N_train:, :]
T_train = T[0:N_train]
T_test = T[N_train:]

# Calculate the P matrix.
if manual_pca == 1:
    P, V, μ, λ, sigma = pca_calcs.pca_manual(X_train, rescale)
else:
    if rescale == 1:
        μ = np.mean(X_train, axis=0)  # this is the mean vector
        sigma = np.std(X_train, axis=0)
    else:
        sigma = 1
    X_rescale = (X_train - μ) / sigma
    pca = PCA(n_components=2)
    P = pca.fit_transform(X_rescale)

# Get the range of coordinates from the first two principal components.
data_range = [[np.min(P[:, 0]), np.max(P[:, 0])],
              [np.min(P[:, 1]), np.max(P[:, 1])]]

# Divide the data based on a threshold for the number of shares.
P1_pop = np.array([j for (i, j) in zip(T_train, P[:, 0:2]) if i >= threshold])
P1_unpop = np.array([j for (i, j) in zip(T_train, P[:, 0:2]) if i < threshold])
T1_pop = np.array([i for i in T_train if i >= threshold])
T1_unpop = np.array([i for i in T_train if i < threshold])

if calc_histogram == 1:
    # Calculate the bin edges for the histograms.
    bin_edges1_x = cls.bin_edges(P[:, 0], B)
    bin_edges1_y = cls.bin_edges(P[:, 1], B)

    # Create histograms.
    hist1_pop = cls.create_hist_np(B, P1_pop[:, 0], P1_pop[:, 1], data_range)
    hist1_unpop = cls.create_hist_np(B, P1_unpop[:, 0],
                                     P1_unpop[:, 1], data_range)

    hist_TP = 0  # true positive count
    hist_FP = 0  # false positive count

    p_list = []
    # Iterate over the test samples and check if they match expected.
    for i, x in enumerate(X_test):
        t = T_test[i]  # This is the corresponding target
        z = (x - μ) / sigma  # mean subtracted features
        p = np.dot(z, V[0:2, :].T)

        # Calculate the probabilities using histogram classifier group 1
        pop_count = cls.count(p[0], p[1], bin_edges1_x, bin_edges1_y,
                              hist1_pop)
        unpop_count = cls.count(p[0], p[1], bin_edges1_x, bin_edges1_y,
                                hist1_unpop)

        pop_prediction = pop_count / (pop_count + unpop_count)

        p_list.append(pop_count)
        if math.isnan(pop_prediction):
            pop_prediction = 0
        if pop_prediction >= 0.5 and t >= threshold:
            hist_TP += 1
        else:
            hist_FP += 1
    p_list = np.array(p_list)

#####################################################################
# Below is the code that generates the plots if enabled.

# Calculate cumalitive variance and plot it.
if plotVariance == 1:
    cum_variance = pca.calculate_variance(λ)
    plt.figure(1)
    plt.plot(cum_variance)

if plot2D == 1:
    plt.figure(2)
    plt.plot(P1_unpop[:, 0], P1_unpop[:, 1], 'b.')
    plt.plot(P1_pop[:, 0], P1_pop[:, 1], 'r.')

# Create 3D plot with Plotly.
if plot3D == 1:
    trace1 = go.Scatter3d(x=P1_unpop[:, 0], y=P1_unpop[:, 1], z=T1_unpop,
                          mode='markers',
                          marker=dict(size=4, color='rgba(0,0,200,.7)'))
    trace2 = go.Scatter3d(x=P1_pop[:, 0], y=P1_pop[:, 1], z=T1_pop,
                          mode='markers',
                          marker=dict(size=4, color='rgba(200,0,0,.7)'))
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    plotly.offline.plot(fig, filename='simple-3d-scatter.html')

# Create histogram with Plotly.
if plothistogram == 1:
    trace1 = go.Histogram(x=P1_pop[:, 0], name='popular', opacity=0.75,
                          autobinx=True,
                          marker=dict(color='rgba(200,0,0,.7)'))
    trace2 = go.Histogram(x=P1_unpop[:, 0], name='unpopular', opacity=0.75,
                          autobinx=True,
                          marker=dict(color='rgba(0,0,200,.7)'))
    data = [trace1, trace2]
    layout = go.Layout(barmode='overlay')
    fig2 = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig2, filename='histogram.html')
