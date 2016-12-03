# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 19:34:56 2016

@author: Eric
"""

import file_ops as fo
import pca
import matplotlib.pyplot as plt
import numpy as np


# Import the features from the raw data file.
X, N = fo.load_feature_matrix()
print("The number of samples is ", N)


P, V, μ, λ = pca.PCA(X)
Xrec = pca.Xrec(P, V, μ, 100)
diff = Xrec - X
print("The max diff is ", np.amax(np.abs(diff)))

Px2 = P[:, 0:2]

plt.figure(1)
plt.plot(Px2[:, 0:1], Px2[:, 1:2], 'r.')
