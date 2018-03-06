#######################################################
########### Human Performance Systems, Inc. ###########
#######################################################

#####################################################
## Classification in Python                        ##
## Day 2 -- kNN Example   Decision Boundary Plot   ##
#####################################################


## NOTE: To run individual code cells, press Shift + enter for PCs 
#==============================================================================
#%% Decision Boundary Example
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import numpy as np
import seaborn as sns
sns.set()
 
# import some data to play with
iris = datasets.load_iris()
 
# prepare data
X = iris.data[:, 2:4]  
y = iris.target
h = .01 
 
# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00','#00FFFF'])
 



#%%  Plot kNN Decision Boundaries
n_neighbors = 17    
                    
# we create an instance of Neighbours Classifier and fit the data.
clf = neighbors.KNeighborsClassifier(n_neighbors)
clf.fit(X, y)
 
# calculate min, max and limits
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
 
# predict class using data and kNN classifier
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
 
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
 
# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolors='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = {})".format(n_neighbors))
plt.show()
plt.xlabel('Iris Sepal Length (cm)')
plt.ylabel('Iris Sepal Width (cm)')
#%%
# Plot just the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolors='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Training Points")
plt.xlabel('Iris Sepal Length (cm)')
plt.ylabel('Iris Sepal Width (cm)')
plt.show()

