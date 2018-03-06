#######################################################
########### Human Performance Systems, Inc. ###########
#######################################################

#####################################################
## Classification in Python                        ##
## Day 2 -- kNN Example   How kNN sees data        ##
#####################################################


## NOTE: To run individual code cells, press Shift + enter for PCs 
#==============================================================================

#%%
# fake data
import pandas as pd
train = pd.DataFrame({'id':[0,1,2], 'length':[0.9,0.3,0.6], 
                      'mass':[0.1,0.2,0.8], 'rings':[40,50,60]})
test_point = pd.DataFrame({'length':[0.59], 'mass':[0.79], 'rings':[54]})

#%%
# Show training data
train

#%%
# Show test point
test_point

#%%
# define X_train, X_test and y_train
X_train = train[['length', 'mass', 'rings']]
y_train = train.id

#%%
# KNN with K=1
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

#%%
# what "should" it predict?
knn.predict(test_point)

#%% How we interpret the data
import matplotlib.pyplot as plt

# create a "colors" array for plotting
import numpy as np
colors = np.array(['red', 'green', 'blue'])
plt.figure(1)
# scatter plot of training data, colored by id (0=red, 1=green, 2=blue)
plt.scatter(train.mass, train.rings, c=colors[train.id], s=50)

# testing data
plt.scatter(test_point.mass, test_point.rings, c='k', s=50)

# add labels
plt.xlabel('mass')
plt.ylabel('rings')
plt.title('How we interpret the data')

#%%How We kNN interprets the data

# adjust the x-limits
plt.figure(2)
plt.scatter(train.mass, train.rings, c=colors[train.id], s=50)
plt.scatter(test_point.mass, test_point.rings, c='k', s=50)
plt.xlabel('mass')
plt.ylabel('rings')
plt.title('How KNN interprets the data')
plt.xlim(0, 30)
plt.show()

#%%
# Does StandardScaler solve the problem?
# StandardScaler is used for the "standardization" of features, 
# also known as "center and scale" or "z-score normalization".

# standardize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

#%%
# original values
X_train.values

#%%
# standardized values
X_train_scaled

#%%
# figure out how it standardized
print(scaler.mean_)
print(scaler.scale_)

#%%
# manually standardize, to make sure we understand what is going on
(X_train.values - scaler.mean_) / scaler.scale_

#%% Exercise

# Applying the StandardScaler to a real dataset:
# Wine dataset 
# from the UCI Machine Learning Repository

# Goal: Predict the origin of wine using chemical analysis


# read three columns from the dataset into a DataFrame
import pandas as pd
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
col_names = ['label', 'color', 'proline']
wine = pd.read_csv(url, header=None, names=col_names, usecols=[0, 10, 13])
#%%
wine.head()
#%%
wine.describe()

#%%
# define X and y
feature_cols = ['color', 'proline']
X = wine[feature_cols]
y = wine.label

#%%
# split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    random_state=4,stratify=y)
#%%
# standardize X_train
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
#%%
# check that it standardized properly
print(X_train_scaled[:, 0].mean())
print (X_train_scaled[:, 0].std())
print(X_train_scaled[:, 1].mean())
print(X_train_scaled[:, 1].std())
#%%
# standardize X_test
X_test_scaled = scaler.transform(X_test)


#%%

# KNN accuracy on original (unscaled) data
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))

#%%
# KNN accuracy on scaled data
knn.fit(X_train_scaled, y_train)
y_pred_class = knn.predict(X_test_scaled)
print(metrics.accuracy_score(y_test, y_pred_class))

#%% 

#But what were we trying to do again?
# define X and y
feature_cols = ['color', 'proline']
X = wine[feature_cols]
y = wine.label

#%%
# proper cross-validation on the original (unscaled) data
knn = KNeighborsClassifier(n_neighbors=3)
from sklearn.cross_validation import cross_val_score
cross_val_score(knn, X, y, cv=5, scoring='accuracy').mean()

#%%
# why is this improper cross-validation on the scaled data?
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy').mean()

#%% Use a sklearn pipeline
# How does pipeline solve this problem?

# fix the cross-validation process using Pipeline
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()

#%%
# search for an optimal n_neighbors value using GridSearchCV
neighbors_range = range(1, 21)
param_grid = dict(kneighborsclassifier__n_neighbors=neighbors_range)
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)
print(grid.best_score_)
print(grid.best_params_)

