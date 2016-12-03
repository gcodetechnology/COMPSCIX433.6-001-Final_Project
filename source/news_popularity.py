# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 19:34:56 2016

@author: Eric
"""

import file_ops as fo


# Import the features from the raw data file.
X, N = fo.load_news_matrix()
print("The number of samples is ", N)
