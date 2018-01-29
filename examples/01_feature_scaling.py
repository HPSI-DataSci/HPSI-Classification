#######################################################
########### Human Performance Systems, Inc. ###########
#######################################################

##############################
## Classification in Python ##
## Day 1 -- Feature Scaling ##
##############################


## NOTE: To run individual code cells, press Shift + enter for PCs & Macs
# ========================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

#%% Standard Scaler

from sklearn.preprocessing import StandardScaler

# set random seed 
np.random.seed(1)

# create dummy data
df = pd.DataFrame({
    'x1': np.random.normal(0, 2, 10000),  # normal dist w/ mean 0, std 2
    'x2': np.random.normal(5, 3, 10000),  # normal dist w/ mean 5, std 3
    'x3': np.random.normal(-5, 5, 10000)  # normal dist w/ mean -5, std 5
})

# instantiate standard scaler
std_scaler = StandardScaler()
# fit the scaler
std_scaler.fit(df)
# extract parameters
print('mean: {}'.format(std_scaler.mean_))
print('standard deviation: {}'.format(std_scaler.scale_))

# scale the data with the learned parameters
std_df = std_scaler.transform(df)
std_df = pd.DataFrame(std_df, columns=['x1', 'x2', 'x3'])

# plot to see the results of the transformation
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 6))

ax1.set_title('Before Scaling')
sns.kdeplot(df['x1'], ax=ax1)
sns.kdeplot(df['x2'], ax=ax1)
sns.kdeplot(df['x3'], ax=ax1)
ax2.set_title('After Standard Scaler')
sns.kdeplot(std_df['x1'], ax=ax2)
sns.kdeplot(std_df['x2'], ax=ax2)
sns.kdeplot(std_df['x3'], ax=ax2)
plt.show()

"""
Note how all features are standardized and brought onto the same scale.
"""

#%% Robust Scaler

from sklearn.preprocessing import RobustScaler

# set random seed 
np.random.seed(1)

# create dummy data
df = pd.DataFrame({
    # Distribution with low-end outliers
    'x1': np.concatenate([np.random.normal(20, 1, 1000), 
                          np.random.normal(1, 1, 25)]),  # add outliers
    # Distribution with high-end outliers
    'x2': np.concatenate([np.random.normal(30, 1, 1000), 
                          np.random.normal(50, 1, 25)]),  # add outliers
})

# instantiate scaler 
rbst_scaler = RobustScaler()
# fit the scaler
rbst_scaler.fit(df)
# extract parameters
print('median: {}'.format(rbst_scaler.center_))
print('IQR: {}'.format(rbst_scaler.scale_))
rbst_df = rbst_scaler.transform(df)

rbst_df = pd.DataFrame(rbst_df, columns=['x1', 'x2'])

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 6))
ax1.set_title('Before Scaling')
sns.kdeplot(df['x1'], ax=ax1)
sns.kdeplot(df['x2'], ax=ax1)
ax2.set_title('After Robust Scaling')
sns.kdeplot(rbst_df['x1'], ax=ax2)
sns.kdeplot(rbst_df['x2'], ax=ax2)
plt.show()

"""
Notice that after robust scaling, the distributions are now on the same 
scale and overlap, but the outliers remain outside of the bulk of the 
new distributions.
"""

#%% Min-Max Scaler

# set random seed 
np.random.seed(1)

from sklearn.preprocessing import MinMaxScaler

df = pd.DataFrame({
    # positive skew
    'x1': np.random.chisquare(8, 1000),
    # negative skew 
    'x2': np.random.beta(8, 2, 1000) * 40,
    # no skew
    'x3': np.random.normal(50, 3, 1000)
})

min_max = MinMaxScaler()
min_max.fit(df)

print('max: {}'.format(min_max.data_max_))
print('min: {}'.format(min_max.data_min_))
print('range: {}'.format(min_max.data_range_))
print('range to transform to: {}'.format(min_max.feature_range))

min_max_df = min_max.transform(df)
min_max_df = pd.DataFrame(min_max_df, columns=['x1', 'x2', 'x3'])

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 6))
ax1.set_title('Before Scaling')
sns.kdeplot(df['x1'], ax=ax1)
sns.kdeplot(df['x2'], ax=ax1)
sns.kdeplot(df['x3'], ax=ax1)
ax2.set_title('After Min-Max Scaling')
sns.kdeplot(min_max_df['x1'], ax=ax2)
sns.kdeplot(min_max_df['x2'], ax=ax2)
sns.kdeplot(min_max_df['x3'], ax=ax2)
plt.show()

"""
Notice that the skewness of the distribution is maintained but the 
3 distributions are brought onto the same scale so that they overlap.
"""

#%% Normalizer

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import Normalizer

# set random seed 
np.random.seed(1)

df = pd.DataFrame({
    'x1': np.random.randint(-100, 100, 1000).astype(float),
    'y1': np.random.randint(-80, 80, 1000).astype(float),
    'z1': np.random.randint(-150, 150, 1000).astype(float),
})

normalizer = Normalizer()
# fit_transform at once since there are no learned parameters
norm_df = normalizer.fit_transform(df)
norm_df = pd.DataFrame(norm_df, columns=df.columns)

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
ax1.scatter(df['x1'], df['y1'], df['z1'])
ax2.scatter(norm_df['x1'], norm_df['y1'], norm_df['z1'])
plt.show()

"""
Note that the points are all scaled within the unit sphere. Additionally, the 
axes have all been put on the same scale. 
"""
