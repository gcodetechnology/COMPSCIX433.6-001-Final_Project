# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 11:30:17 2016

@author: Eric
"""
import numpy as np
import numpy.linalg as LA


def mean_vector(X):
    """Given a matrix, return the mean adjusted matrix and mean vector."""
    μ = np.mean(X, axis=0)  # this is the mean vector
    Z = X - μ
    return(Z, μ)


def PCA(X):
    Z, μ = mean_vector(X)
    C = np.cov(Z, rowvar=False)
    [λ, V] = LA.eigh(C)
    λ = np.flipud(λ)
    V = np.flipud(np.transpose(V))
    P = np.dot(Z, V.T)
    return (P, V, μ, λ)


def Xrec(P, V, μ, n=1):
    R = (np.dot(P[:, 0: n], V[0: n, :]))
    Xrec = R + μ
    return Xrec
