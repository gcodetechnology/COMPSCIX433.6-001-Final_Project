import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
from pylab import *
from numpy import *
import scipy.sparse as sparse
import scipy.linalg as linalg

###############################
### download & explore data ###
###############################

csvfile = open('OnlineNewsPopularity.csv')
data = pd.read_csv(csvfile, header = 0)

#just getting general information on 'shares'
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

############################################
### determining popular and viral vidoes ###
############################################
target = data[' shares']
popular = np.percentile(target, 50)
viral = np.percentile(target, 50.005)
print(popular)
print(viral)

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

# create training and test data
msk = np.random.rand(len(data)) < 0.75
train = data[msk]
test = data[~msk]

######################
### building model ###
######################

# create augmented X
X_train = train.iloc[:,2:60]
a = np.ones((len(train),1))
Xa_train = np.hstack((a,X_train))
# find pseudoinverse
Xap_train = np.linalg.pinv(Xa_train)
# create target
target_train = train[['normal','popular','viral']]
target_train = np.asarray(target_train)
# finding W
W = np.dot(Xap_train, target_train)

########################
### model assessment ###
########################
# building Xa for test
X_test = test.iloc[:,2:60]
a = np.ones((len(test),1))
Xa_test = np.hstack((a,X_test))
# making classifier
T_p = np.dot(Xa_test, W)
T_classify = np.zeros((len(T_p),))
for i in range(0,len(T_p[:,1])):
    T_classify[i] = np.argmax(T_p[i,:])

# create target
target_test = test[['normal','popular','viral']]
target_test = np.asarray(target_test)
# convert target into 0,1,2 within one column
T_data = np.zeros((len(T_p),))
for i in range(0,len(T_p[:,1])):
    T_data[i] = np.argmax(target_test[i,:])

# making comparisons
comparisons = np.zeros((3, 3))
for i in range(0, len(T_p)):
    actual = T_data[i]
    predicted = T_classify[i] # if you don't use the 0 at the end, this is an array
    comparisons[actual, predicted] += 1

print(comparisons)
