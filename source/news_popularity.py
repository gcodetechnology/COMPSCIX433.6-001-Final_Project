# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 19:34:56 2016

@author: Eric
"""

import file_ops as fo
import pca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import plotly
import plotly.graph_objs as go


# Import the features from the raw data file.
X, T, N = fo.load_feature_matrix()
print("The number of samples is ", N)


P, V, μ, λ = pca.PCA(X)
Xrec = pca.Xrec(P, V, μ, 100)
diff = Xrec - X
print("The max diff is ", np.amax(np.abs(diff)))

# Calculate cumalitive variance and plot it.
cum_variance = pca.create_var_plot(λ)
plt.plot(cum_variance)

# Create a 2D plot
#plt.plot(P[:, 0:1], P[:, 1:2], 'r.')

# Create a 3D plot
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(P[:, 0], P[:, 1], T)

# Create 3D plot with Plotly.
trace1 = go.Scatter3d(x=P[:, 0], y=P[:, 1], z=T,
                      mode='markers',
                      marker=dict(size=4))
#                      line=dict(color='rgba(217, 217, 217, 0.14)', width=0.1),
#                      opacity=0.9)
layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))

fig = go.Figure(data=[trace1], layout=layout)
plotly.offline.plot(fig, filename='simple-3d-scatter')
