# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 19:34:56 2016

@author: Eric
"""

import file_ops as fo
import pca
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.graph_objs as go

# Turn plots on or off.
plotVariance = 1
plot2D = 1
plot3D = 0

# Turn rescaling on or off.
rescale = 1

# Define the number of shares that indicate popular.
threshold = 9000

# Import the features from the raw data file.
X, T, N = fo.load_feature_matrix()
print("The number of samples is ", N)

P, V, μ, λ = pca.PCA(X, rescale)
Xrec = pca.Xrec(P, V, μ, 100)
diff = Xrec - X
print("The max diff is ", np.amax(np.abs(diff)))

# Calculate cumalitive variance and plot it.
if plotVariance == 1:
    cum_variance = pca.calculate_variance(λ)
    plt.figure(1)
    plt.plot(cum_variance)

# Get the records of the first and second principal components.
P_pop = np.array([j for (i, j) in zip(T, P[:, 0:2]) if i >= threshold])
P_unpop = np.array([j for (i, j) in zip(T, P[:, 0:2]) if i < threshold])

# Create a 2D point cloud.
if plot2D == 1:
    plt.figure(2)
    plt.plot(P_unpop[:, 0:1], P_unpop[:, 1:2], 'b.')
    plt.plot(P_pop[:, 0:1], P_pop[:, 1:2], 'r.')

# Create 3D plot with Plotly.
if plot3D == 1:
    trace1 = go.Scatter3d(x=P[:, 0], y=P[:, 1], z=T,
                          mode='markers',
                          marker=dict(size=4))
    #                      line=dict(color='rgba(217, 217, 217, 0.14)', width=0.1),
    #                      opacity=0.9)
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
    fig = go.Figure(data=[trace1], layout=layout)
    plotly.offline.plot(fig, filename='simple-3d-scatter')
