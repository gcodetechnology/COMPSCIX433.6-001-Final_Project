
# coding: utf-8

# # Final Project: Online News Popularity

# # Data Set: https://archive.ics.uci.edu/ml/machine-learning-databases/00332/

# In[89]:

#example: http://yilinwei.com/project/Online-news-popularity-classification-with-R.html


# # Load Library

# In[61]:

import os, struct
import matplotlib as plt
import numpy as np
import pandas as pd 
import random


# # Load Data

# In[45]:

data = pd.read_csv('C:/Users/fbeker/Desktop/notes/Final_Project/raw_data/OnlineNewsPopularity.csv', sep=',') 
#print(data.describe)


# # Summarize Data

# In[47]:

print(data.shape)
data.describe()


# # Convert Data Frame To An Array

# In[42]:

data=np.array(data)


# # Target Column (# of Shares)

# In[48]:

target=np.array(data)[:,60]
target[0:5]


# # Divide into Train (75%) and Test (25%) Set

# In[88]:

random.seed(1001)
msk = np.random.rand(len(data)) < 0.75
train = data[msk]
test = data[~msk]


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



