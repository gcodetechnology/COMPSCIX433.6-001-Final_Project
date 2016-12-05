import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
from pylab import *
from numpy import *
import scipy.sparse as sparse
import scipy.linalg as linalg


csvfile = open('OnlineNewsPopularity.csv')
data = pd.read_csv(csvfile, header = 0)

target = data[' shares']
print(target.describe())
#count     39644.000000
#mean       3395.380184
#std       11626.950749
#min           1.000000
#25%         946.000000
#50%        1400.000000
#75%        2800.000000
#max      843300.000000

# we can see that there are huge outliers (max = 843300 vs mode of 1400)

# plt.hist(target, bins = 400)
# plt.show

# determining popular and viral videos
popular = np.percentile(target, 95) # 10800 shares
viral = np.percentile(target, 99.5) # 50679 shares

#creating classification
data['normal'] = -1
data['popular'] = -1
data['viral'] = -1

for article in range (len(data)):
    if data.loc[article,' shares'] <popular:
        data.loc[article, 'normal'] += 2
    elif data.loc[article,' shares'] >= popular and data.loc[article,' shares'] < viral:
        data.loc[article, 'popular'] += 2
    elif data.loc[article, ' shares'] >= viral:
        data.loc[article, 'viral'] += 2

# create augmented X
X = data.iloc[:,2:59]
a = np.ones((len(data),1))
Xa = np.hstack((a,X))
Xap = np.linalg.pinv(Xa)
print(Xap.shape)
# create target
target = data[['normal','popular','viral']]
W = np.dot(Xap, target)

# model assessment
# T_p = np.dot(Xap, W)
# T_c = np.zeros((len(T_p),))
# for i in range(0,len(T_p[:,1])):
#     T_c[i] = np.argmax(T_p[i,:])
#
# comparisons = np.zeros((3, 3))
# for i in range(0, len(T_c)):
#     actual = target[i]
#     predicted = T_c[i] # if you don't use the 0 at the end, this is an array
#     comparisons[actual, predicted] += 1
#
# print(comparisons)
