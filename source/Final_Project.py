import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

data['rating'] = 'normal'

for article in range (len(data)):
    if data.iloc[article,' shares'] >= popular:
        data.iloc[article, 'rating'] = 'popular'
    elif data.iloc[article, ' shares'] >= viral:
        data.iloc[article, 'rating'] = 'viral'
