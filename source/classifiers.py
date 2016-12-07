# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 12:26:17 2016

@author: Eric
"""

import numpy as np
from math import exp


def pdf_gauss(x, y, μ, P, n_components=2):
    """Gaussian probability distribution function."""
    covdet = np.linalg.det(np.cov(P[:, 0:n_components], rowvar=False))
    covinv = np.linalg.inv(np.cov(P[:, 0:n_components], rowvar=False))
    data_point = [x, y]
    xminusμ = np.subtract(data_point, μ)
    xminusμ_T = np.transpose(xminusμ)
    a = (1 / (2 * np.pi * np.sqrt(covdet)))
    b = np.dot(xminusμ, covinv)
    c = np.dot(b, xminusμ_T)
    return a * exp(-0.5 * c)


def bin_edges(single_axis, B):
    '''Define the bin edges for the histogram.'''
    x_min = min(single_axis)
    x_max = max(single_axis)
    x_range = x_max - x_min
    edge = x_min
    bin_edges = []
    bin_edges.append(x_min)
    while (edge < x_max - (.001 * x_max)):
        edge = edge + (x_range / B)
        bin_edges.append(edge)
    return bin_edges


def histogram(B, x_coordinates, y_coordinates, data_range):
    """Manually calculate the histogram."""
    histogram = np.zeros((B, B))
    x1_min = data_range[0][0]
    x1_max = data_range[0][1]
    x2_min = data_range[1][0]
    x2_max = data_range[1][1]
    for i, x in enumerate(x_coordinates):
        x_histogram_pos = (x - x1_min) / (x1_max - x1_min)
        y_histogram_pos = (y_coordinates[i] - x2_min) / (x2_max - x2_min)
        if x_histogram_pos >= 1:
            ix = B - 1
        else:
            ix = int(B * x_histogram_pos)
        if y_histogram_pos >= 1:
            iy = B - 1
        else:
            iy = int(B * y_histogram_pos)
        histogram[ix, iy] = histogram[ix, iy] + 1
    return histogram


def hist_weighted(B, x_coords, y_coords, Z_values, data_range):
    """Manually calculate a histogram and return the average Z value of all
    points within the histogram."""
    histogram = np.zeros((B, B))
    values_total = np.zeros((B, B))
    x1_min = data_range[0][0]
    x1_max = data_range[0][1]
    x2_min = data_range[1][0]
    x2_max = data_range[1][1]
    for i, x in enumerate(x_coords):
        x_histogram_pos = (x - x1_min) / (x1_max - x1_min)
        y_histogram_pos = (y_coords[i] - x2_min) / (x2_max - x2_min)
        if x_histogram_pos >= 1:
            ix = B - 1
        else:
            ix = int(B * x_histogram_pos)
        if y_histogram_pos >= 1:
            iy = B - 1
        else:
            iy = int(B * y_histogram_pos)
        histogram[ix, iy] = histogram[ix, iy] + 1
        values_total[ix, iy] = values_total[ix, iy] + Z_values[i]
    values_avg = np.nan_to_num(values_total / histogram)
    return histogram, values_avg


def histogram_np(B, x_coordinates, y_coordinates, data_range):
    """Use Numpy to calculate the histogram."""
#    data_range = [[x1_min, x1_max], [x2_min, x2_max]]
    histogram = np.histogram2d(x_coordinates, y_coordinates, B, data_range)[0]
    return histogram


# Function to determine which bin a value resides in.
def hist_index(x, bin_edges):
    for i, be in enumerate(bin_edges):
        if x < min(bin_edges):
            bin_index = 0
        elif x > max(bin_edges):
            bin_index = len(bin_edges) - 2
        elif be <= x <= bin_edges[i + 1]:
            bin_index = i
            break
    return bin_index


# Return the class conditional probability
def count(x1, x2, bin_edges1, bin_edges2, histogram):
    x1_index = hist_index(x1, bin_edges1)
    x2_index = hist_index(x2, bin_edges2)
    return histogram[x1_index, x2_index]


# Calculate the posterior probability of being female.
def P_Hist(count_p, count_n):
    return count_p / (count_n + count_p)


def P_Bayes(px1, px2):
    return (px1 / (px1 + px2))
