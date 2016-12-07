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
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

#####################################################################
# Set the variables that will be used for the calculations.
threshold = 1400  # Define the number of shares that indicate popular.
B = 40  # Define the number of histogram bins to use.

# Determine which calculations will be used/performed.
rescale = True  # Turn rescaling on or off.
manual_pca = True  # If equal to 0 then use sklearn for PCA.
calc_histogram = True  # Turn histogram calculation on/off.

# Choose which plots to display
plotVariance = False  # Turns the variance plot on/off.
plot2D = True  # Turns the 2D plot on/off.
calc_Kmeans = True  # Turn the Kmeans calculation on/off.
plot3D = False  # Turns the 3D plot on/off.
plot2Dhistogram = False  # Turns this histogram plot on/off.
plot3Dhistogram = False

#####################################################################

plt.close('all')

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
if manual_pca:
    P, V, μ, λ, sigma = pca_calcs.pca_manual(X_train, rescale)
else:
    if rescale:
        μ = np.mean(X_train, axis=0)  # this is the mean vector
        sigma = np.std(X_train, axis=0)
    else:
        sigma = 1
    X_rescale = (X_train - μ) / sigma
    pca = PCA(n_components=2)
    P = pca.fit_transform(X_rescale)

# Get the range of coordinates from the first two principal components.
data_range = [[np.amin(P[:, 0]), np.amax(P[:, 0])],
              [np.amin(P[:, 1]), np.amax(P[:, 1])]]

# Divide the data based on a threshold for the number of shares.
P_pop = np.array([j for (i, j) in zip(T_train, P[:, 0:2]) if i >= threshold])
P_unpop = np.array([j for (i, j) in zip(T_train, P[:, 0:2]) if i < threshold])
T_pop = np.array([i for i in T_train if i >= threshold])
T_unpop = np.array([i for i in T_train if i < threshold])

if calc_histogram:
    # Calculate the bin edges for the histograms.
    bin_edges1_x = cls.bin_edges(P[:, 0], B)
    bin_edges1_y = cls.bin_edges(P[:, 1], B)

    # Create histograms.
    hist_pop = cls.create_hist_np(B, P_pop[:, 0], P_pop[:, 1], data_range)
    hist_unpop = cls.create_hist_np(B, P_unpop[:, 0],
                                    P_unpop[:, 1], data_range)

    true_postive = 0  # true positive count
    true_negative = 0  # true negative count
    correct = 0  # correct predictions
    incorrect = 0  # incorrect predictions

    # Iterate over the test samples and check if they match expected.
    for i, x in enumerate(X_test):
        t = T_test[i]  # This is the corresponding target
        z = (x - μ) / sigma  # mean subtracted features
        p = np.dot(z, V[0:2, :].T)

        # Calculate the probabilities using histogram classifier group 1
        pop_count = cls.count(p[0], p[1], bin_edges1_x, bin_edges1_y,
                              hist_pop)
        unpop_count = cls.count(p[0], p[1], bin_edges1_x, bin_edges1_y,
                                hist_unpop)

        pop_prediction = pop_count / (pop_count + unpop_count)

        if math.isnan(pop_prediction):
            pop_prediction = 0
        if pop_prediction >= 0.5 and t >= threshold:
            true_postive += 1
            correct += 1
        elif pop_prediction < 0.5 and t < threshold:
            true_negative += 1
            correct += 1
        else:
            incorrect += 1
    print('Number of correct predictions ', correct)
    print('Number of incorrect predictions ', incorrect)
    print('Percentage correct ', (correct / (correct + incorrect)))

#####################################################################
# Below is the code that generates the plots if enabled.
#####################################################################

# Calculate cumalitive variance and plot it.
if plotVariance:
    cum_variance = pca.calculate_variance(λ)
    plt.figure(1)
    plt.plot(cum_variance)

# Attempt to cluster the data if the Kmeans calculation is turned on.
if calc_Kmeans:
    km = KMeans(n_clusters=2, init='random', n_init=10, max_iter=1000,
                tol=1e-05)
    y_km = km.fit_predict(P)

# Show a 2D scatter plot with matplotlib.
if plot2D:
    plt.figure(2)
    if calc_Kmeans:
        plt.scatter(P[y_km == 0, 0], P[y_km == 0, 1], s=50, c='lightgreen',
                    marker='o', label='cluster 1')
        plt.scatter(P[y_km == 1, 0], P[y_km == 1, 1], s=50, c='orange',
                    marker='o', label='cluster 2')
        plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                    s=250, marker='*', c='red', label='centroids')
    else:
        plt.scatter(P_unpop[:, 0], P_unpop[:, 1], s=50, c='blue', marker='.',
                    linewidths=0, label='unpopular')
        plt.scatter(P_pop[:, 0], P_pop[:, 1], s=50, c='red', marker='.',
                    linewidths=0, alpha=.4, label='popular')
    plt.legend()
    plt.grid()
    plt.show()

# Create 3D plot with Plotly.
if plot3D:
    trace1 = go.Scatter3d(x=P_unpop[:, 0], y=P_unpop[:, 1], z=T_unpop,
                          mode='markers',
                          marker=dict(size=4, color='rgba(0,0,200,.7)'))
    trace2 = go.Scatter3d(x=P_pop[:, 0], y=P_pop[:, 1], z=T_pop,
                          mode='markers',
                          marker=dict(size=4, color='rgba(200,0,0,.7)'))
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    plotly.offline.plot(fig, filename='simple-3d-scatter.html')

# Create histogram with Plotly.
if plot2Dhistogram:
    trace1 = go.Histogram(x=P_pop[:, 0], name='popular', opacity=0.75,
                          autobinx=True,
                          marker=dict(color='rgba(200,0,0,.7)'))
    trace2 = go.Histogram(x=P_unpop[:, 0], name='unpopular', opacity=0.75,
                          autobinx=True,
                          marker=dict(color='rgba(0,0,200,.7)'))
    data = [trace1, trace2]
    layout = go.Layout(barmode='overlay')
    fig2 = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig2, filename='histogram.html')

if plot3Dhistogram:
    fig = plt.figure(3)
    ax = Axes3D(fig)

    # unpopular
    x1_pos, y1_pos = np.meshgrid(np.arange(hist_unpop.shape[1]),
                                 np.arange(hist_unpop.shape[0]))
    x1_pos = x1_pos.flatten()
    y1_pos = y1_pos.flatten()
    z1_pos = hist_unpop.flatten()
    ax.bar3d(x1_pos, y1_pos, np.zeros(len(z1_pos)), 1, 1,
             z1_pos, color='b', alpha=0.70)

    # popular
    x2_pos, y2_pos = np.meshgrid(np.arange(hist_pop.shape[1]),
                                 np.arange(hist_pop.shape[0]))
    x2_pos = x2_pos.flatten()
    y2_pos = y2_pos.flatten()
    z2_pos = hist_pop.flatten()
    ax.bar3d(x2_pos, y2_pos, np.zeros(len(z2_pos)), 1, 1,
             z2_pos, color='r', alpha=0.40)
    ax.view_init(elev=15, azim=0)
    ax.dist = 12
    plt.show()
