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

#####################################################################
# Set the parameters that will be used for the calculations.

plotVariance = 0  # Turns the variance plot on/off.
plot2D = 1  # Turns the 2D plot on/off.
plot3D = 0  # Turns the 3D plot on/off.
rescale = 1  # Turn rescaling on or off.
threshold = 12000  # Define the number of shares that indicate popular.
manual_pca = 0  # If equal to 0 then use sklearn for PCA.

#####################################################################


# Import the features from the raw data file.
X, T, N = fo.load_feature_matrix()
print("The number of samples is ", N)

# Calculate the P matrix.
if manual_pca == 1:
    P, V, μ, λ = pca_calcs.pca_manual(X, rescale)
else:
    if rescale == 1:
        μ = np.mean(X, axis=0)  # this is the mean vector
        sigma = np.std(X, axis=0)
    else:
        sigma = 1
    X_rescale = (X - μ) / sigma
    pca = PCA(n_components=2)
    P = pca.fit_transform(X_rescale)

# Calculate cumalitive variance and plot it.
if plotVariance == 1:
    cum_variance = pca.calculate_variance(λ)
    plt.figure(1)
    plt.plot(cum_variance)

# Get the records of the first and second principal components.
P_pop = np.array([j for (i, j) in zip(T, P[:, 0:2]) if i >= threshold])
P_unpop = np.array([j for (i, j) in zip(T, P[:, 0:2]) if i < threshold])
T_pop = np.array([i for i in T if i >= threshold])
T_unpop = np.array([i for i in T if i < threshold])

# Create a 2D point cloud.
if plot2D == 1:
    plt.figure(2)
    plt.plot(P_unpop[:, 0], P_unpop[:, 1], 'b.')
    plt.plot(P_pop[:, 0], P_pop[:, 1], 'r.')
#    plt.plot([-6.7, -6.7], [-30, 10], color='k', linestyle='--', linewidth=1)

# Create 3D plot with Plotly.
if plot3D == 1:
    trace1 = go.Scatter3d(x=P_unpop[:, 0], y=P_unpop[:, 1], z=T_unpop,
                          mode='markers',
                          marker=dict(size=4, color='rgba(0,0,200,.7)'))
    trace2 = go.Scatter3d(x=P_pop[:, 0], y=P_pop[:, 1], z=T_pop,
                          mode='markers',
                          marker=dict(size=4, color='rgba(200,0,0,.7)'))
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    plotly.offline.plot(fig, filename='simple-3d-scatter.html')
