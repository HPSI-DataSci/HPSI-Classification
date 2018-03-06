
#######################################################
########### Human Performance Systems, Inc. ###########
#######################################################

#####################################################
## Classification in Python                        ##
## Day 2 -- kNN From Scratch Exa
mple               ##
#####################################################


## NOTE: To run individual code cells, press Shift + enter for PCs 
#==============================================================================
#%%
# make our X
import numpy as np
from sklearn.neighbors import DistanceMetric

import pandas as pd

# Make our data
df = pd.DataFrame([[0,3,0,'Red'],
     [2,0,0,'Red'],
     [0,1,3,'Red'],
     [0,1,2,'Green'],
     [-1,0,1,'Green'],
     [1,1,1,'Red']])
df.columns = ['X1','X2','X3','Color']
df
#%%
X_neighbors = df.iloc[:,:3]

#%%
# make our test point
X_test = np.array([[0,0,0]])
X_test

#%%
# get euclidean distance
euc_dist = DistanceMetric.get_metric('euclidean')

# calculate euclidean distance between our test point and our training points
test_euc_distance = euc_dist.pairwise(X_neighbors,X_test)
test_euc_distance
#%%
# add euclidean distance to our DataFrame
df['euclidean_dist']=test_euc_distance
df

#%%
manh_dist = DistanceMetric.get_metric('manhattan')

# calculte the manhattan distance between our test point and our training points
test_man_distance = manh_dist.pairwise(X_neighbors,X_test)
test_man_distance
#%%
# add manhattan distance to our DataFrame
df['manhattan_dist'] = test_man_distance
df
#%%
# sort by euclidean
df.sort_values(by='euclidean_dist')

#%%
# sort by manhattan
df.sort_values(by='manhattan_dist')
#%%
from scipy.spatial import distance

distance.euclidean([0,0,0],[3,3,2])
#%%
distance.cityblock([0,0,0],[3,3,2])  

#%%


#%%
distance.minkowski([3,-4,5,6],[0,0,0,0],1)

#%%
distance.hamming('brent','mike')

#%%
distance.hamming('brent','brent')

#%% Re-load the data
# import load_iris function from datasets module
from sklearn.datasets import load_iris

# save "bunch" object containing iris dataset and its attributes
iris = load_iris()

# store feature matrix in "X"
X = iris.data

# store response vector in "y"
y = iris.target

#split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=4)

#%%

import numpy as np
from collections import Counter
class simple_knn():
    "a simple kNN with L2 distance"

    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=3):
        
        dists = self.compute_distances(X)
        
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            k_closest_y = []
            labels = self.y_train[np.argsort(dists[i,:])].flatten()
            # find k nearest lables
            k_closest_y = labels[:k]
            c = Counter(k_closest_y)
            y_pred[i] = c.most_common(1)[0][0]

        return(y_pred)

    def compute_distances(self, X):
        
        dot_pro = np.dot(X, self.X_train.T)
        sum_square_test = np.square(X).sum(axis = 1)
        sum_square_train = np.square(self.X_train).sum(axis = 1)
        dists = np.sqrt(-2 * dot_pro + sum_square_train + np.matrix(sum_square_test).T)
        
        return(dists)
        
#%%
classifier = simple_knn()
classifier.train(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


