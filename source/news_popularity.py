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
threshold = 6200  # Define the number of shares that indicate popular.
B = 40  # Define the number of histogram bins to use.

# Determine which calculations will be used/performed.
rescale = True  # Turn rescaling on or off.
manual_pca = True  # If equal to 0 then use sklearn for PCA.
histogram_binary = True  # Predicts binary classification based on threshold
histogram_shares = False  # Predicts number of shares - not classification.
mse = False  # Perform mean-square error linear classifier
calc_Kmeans = False  # Turn the Kmeans calculation on/off.

# Choose which plots to display
shares_histogram = False  # Show a histogram of the number of shares.
plotVariance = False  # Turns the variance plot on/off.
plot2D = False  # Turns the 2D plot on/off.
plot3D = False  # Turns the 3D plot on/off.
plot2Dhistogram = True  # Turns this histogram plot on/off.
plot3Dhistogram_binary = False
plot3Dhistogram = False  # Histogram shares calculation must be on.

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

# Separate the data into two classes based on a threshold.
# Above a certain number of shares will be considered popular.
P_hi = np.array([j for (i, j) in zip(T_train, P[:, 0:2]) if i >= threshold])
P_lo = np.array([j for (i, j) in zip(T_train, P[:, 0:2]) if i < threshold])
T_hi = np.array([i for i in T_train if i >= threshold])
T_lo = np.array([i for i in T_train if i < threshold])


#####################################################################
# Histogram based predictions for binary classification.
#####################################################################
if histogram_binary:
    # Calculate the bin edges for the histograms.
    bin_edges1_x = cls.bin_edges(P[:, 0], B)
    bin_edges1_y = cls.bin_edges(P[:, 1], B)

    # Create histograms.
    hist_hi = cls.histogram_np(B, P_hi[:, 0], P_hi[:, 1], data_range)
    hist_lo = cls.histogram_np(B, P_lo[:, 0], P_lo[:, 1], data_range)

    tp_hist = 0  # true positive count
    tn_hist = 0  # true negative count
    fp_hist = 0  # false positive count
    fn_hist = 0  # false negative count
    correct_hist = 0  # correct predictions
    incorrect_hist = 0  # incorrect predictions

    # Iterate over the test samples and check if they match expected.
    for i, x in enumerate(X_test):
        t = T_test[i]  # This is the corresponding target
        z = (x - μ) / sigma  # mean subtracted features
        p = np.dot(z, V[0:2, :].T)

        # Calculate the probabilities
        hi_count = cls.count(p[0], p[1], bin_edges1_x, bin_edges1_y, hist_hi)
        lo_count = cls.count(p[0], p[1], bin_edges1_x, bin_edges1_y, hist_lo)

        hi_prediction = hi_count / (hi_count + lo_count)

        if math.isnan(hi_prediction):
            hi_prediction = 0

        if hi_prediction >= 0.5 and t >= threshold:
            tp_hist += 1
            correct_hist += 1
        elif hi_prediction < 0.5 and t < threshold:
            tn_hist += 1
            correct_hist += 1
        elif hi_prediction >= 0.5 and t < threshold:
            fp_hist += 1
            incorrect_hist += 1
        elif hi_prediction <= 0.5 and t >= threshold:
            fn_hist += 1
            incorrect_hist += 1

    print('\n### HISTOGRAM RESULTS ###')
    print('Number of correct predictions ', correct_hist)
    print('Number of incorrect predictions ', incorrect_hist)
    print('Accuracy ', (correct_hist / (correct_hist + incorrect_hist)))
    print('Sensitivity ', (tp_hist / (tp_hist + fn_hist)))
    print('Specificity ', (tn_hist / (fp_hist + tn_hist)))
    print('PPV ', (tp_hist / (fp_hist + tp_hist)))


#####################################################################
# Histogram based predictions for regression prediction.
#####################################################################
if histogram_shares:
    # Remove the outliers since it adversely affects the prediction.
    P_clean = np.array([j for (i, j) in zip(T_train, P[:, 0:2]) if i < 100000])
    T_clean = np.array([i for i in T_train if i < 100000])
    X_test_clean = np.array([j for (i, j) in zip(T_test, X_test)
                             if i < 100000])
    T_test_clean = np.array([i for i in T_test if i < 100000])
    bin_edges1_x = cls.bin_edges(P_clean[:, 0], B)
    bin_edges1_y = cls.bin_edges(P_clean[:, 1], B)

    d_range = [[np.amin(P_clean[:, 0]), np.amax(P_clean[:, 0])],
               [np.amin(P_clean[:, 1]), np.amax(P_clean[:, 1])]]

    # Create histograms
    hist_counts, hist_values = cls.hist_weighted(B, P_clean[:, 0],
                                                 P_clean[:, 1],
                                                 T_clean, d_range)
    # Iterate over the test samples and check if they match expected.
    error_list = []
    for i, x in enumerate(X_test_clean):
        t = T_test_clean[i]  # This is the corresponding target
        z = (x - μ) / sigma  # mean subtracted features
        p = np.dot(z, V[0:2, :].T)

        # Predict number of shares
        avg_value = cls.count(p[0], p[1], bin_edges1_x, bin_edges1_y,
                              hist_values)
        # Determine the error.
        error = abs(t - avg_value) / t
        error_list.append(error)
    error_list = np.array(error_list)


#####################################################################
# Linear classifier based predictions for binary classification.
#####################################################################
if mse:
    Xa = np.insert(X_train, 0, 0, axis=1)
    Xa_pinv = np.linalg.pinv(Xa)
    T_train_binary = np.full(len(T_train), -1, dtype=np.int8)

    for i, e in enumerate(T_train):
        if e >= threshold:
            T_train_binary[i] = 1

    W_binary = np.dot(Xa_pinv, T_train_binary)

    Xa_test = np.insert(X_test, 0, 0, axis=1)

    correct_lin = 0
    incorrect_lin = 0
    for i, x in enumerate(Xa_test):
        t = T_test[i]  # This is the target
        prediction = (np.dot(x, W_binary))
        if prediction >= 0 and t >= threshold:
            correct_lin += 1
        elif prediction < 0 and t < threshold:
            correct_lin += 1
        else:
            incorrect_lin += 1
    print('\n### LINEAR CLASSIFIER RESULTS ###')
    print('Number of correct predictions ', correct_lin)
    print('Number of incorrect predictions ', incorrect_lin)
    print('Accuracy ', (correct_lin / (correct_lin + incorrect_lin)))


#####################################################################
# Below is the code that generates the plots if they are enabled.
#####################################################################

# Calculate cumalitive variance and plot it.
if plotVariance:
    cum_variance = pca_calcs.calculate_variance(λ)
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
        plt.scatter(P_lo[:, 0], P_lo[:, 1], s=50, c='blue', marker='.',
                    linewidths=0, label='low')
        plt.scatter(P_hi[:, 0], P_hi[:, 1], s=50, c='red', marker='.',
                    linewidths=0, alpha=.4, label='high')
    plt.legend()
    plt.grid()
    plt.show()

# Create 3D plot with Plotly.
if plot3D:
    trace1 = go.Scatter3d(x=P_lo[:, 0], y=P_lo[:, 1], z=T_lo,
                          mode='markers',
                          marker=dict(size=4, color='rgba(0,0,200,.7)'))
    trace2 = go.Scatter3d(x=P_hi[:, 0], y=P_hi[:, 1], z=T_hi,
                          mode='markers',
                          marker=dict(size=4, color='rgba(200,0,0,.7)'))
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    plotly.offline.plot(fig, filename='simple-3d-scatter.html')

# Create histogram with Plotly.
if plot2Dhistogram:
    trace1 = go.Histogram(x=P_hi[:, 0], name='high', opacity=0.75,
                          autobinx=True,
                          marker=dict(color='rgba(200,0,0,.7)'))
    trace2 = go.Histogram(x=P_lo[:, 0], name='low', opacity=0.75,
                          autobinx=True,
                          marker=dict(color='rgba(0,0,200,.7)'))
    data = [trace1, trace2]
    layout = go.Layout(barmode='overlay')
    fig2 = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig2, filename='histogram.html')

if shares_histogram:
    trace1 = go.Histogram(x=T_lo, name='high', opacity=0.75,
                          autobinx=True,
                          marker=dict(color='rgba(0,200,0,.7)'))
    data = [trace1]
    layout = go.Layout(barmode='overlay')
    fig3 = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig3, filename='shares_histogram.html')

if plot3Dhistogram_binary:
    fig = plt.figure(3)
    ax = Axes3D(fig)

    # lo
    x1_pos, y1_pos = np.meshgrid(np.arange(hist_lo.shape[1]),
                                 np.arange(hist_lo.shape[0]))
    x1_pos = x1_pos.flatten()
    y1_pos = y1_pos.flatten()
    z1_pos = hist_lo.flatten()
    ax.bar3d(x1_pos, y1_pos, np.zeros(len(z1_pos)), 1, 1,
             z1_pos, color='b', alpha=0.50)

    # popular
    x2_pos, y2_pos = np.meshgrid(np.arange(hist_hi.shape[1]),
                                 np.arange(hist_hi.shape[0]))
    x2_pos = x2_pos.flatten()
    y2_pos = y2_pos.flatten()
    z2_pos = hist_hi.flatten()
    ax.bar3d(x2_pos, y2_pos, np.zeros(len(z2_pos)), 1, 1,
             z2_pos, color='r', alpha=0.40)
    ax.view_init(elev=15, azim=0)
    ax.dist = 12
    plt.show()

if plot3Dhistogram:
    fig = plt.figure(3)
    ax = Axes3D(fig)
    x1_pos, y1_pos = np.meshgrid(np.arange(hist_values.shape[1]),
                                 np.arange(hist_values.shape[0]))
    x1_pos = x1_pos.flatten()
    y1_pos = y1_pos.flatten()
    z1_pos = hist_values.flatten()
    ax.bar3d(x1_pos, y1_pos, np.zeros(len(z1_pos)), 1, 1,
             z1_pos, color='b', alpha=0.50)

    ax.view_init(elev=15, azim=0)
    ax.dist = 12
    plt.show()
