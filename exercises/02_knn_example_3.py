#######################################################
########### Human Performance Systems, Inc. ###########
#######################################################

#####################################################
## Classification in Python                        ##
## Day 2 -- kNN Example   Choosing k               ##
#####################################################


## NOTE: To run individual code cells, press Shift + enter for PCs 
#==============================================================================
#%% Re-load the data
# import load_iris function from datasets module
from sklearn.datasets import load_iris

# save "bunch" object containing iris dataset and its attributes
iris = load_iris()

# store feature matrix in "X"
X = iris.data

# store response vector in "y"
y = iris.target
#%% Split data into TRAINING/TESTING sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=42)


#%% Cross Validation
'''
Question: What if we created a bunch of train/test splits, calculated the 
testing accuracy for each, and averaged the results together?

Answer: That's the essense of cross-validation...

STEPS FOR CROSS VALIDATION:
1. Split the dataset into K equal partitions (or "folds").
2. Use fold 1 as the testing set and the union of the other folds as the 
training set.
3. Calculate testing accuracy.
4. Repeat steps 2 and 3 K times, using a different fold as the testing set 
each time.
5. Use the average testing accuracy as the estimate of out-of-sample accuracy.

'''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
print(scores)

# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())
#%% Visualize the effect of changing k
# search for an optimal value of K for KNN
# try K=1 through K=30, do 5-fold cv for each k, and record avg. test acc.
k_range = list(range(1, 31,2))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.xticks(range(1, 31,2))


#%%
from sklearn.model_selection import GridSearchCV
import numpy as np

# Create our 'param_grid' 
# The key is the name of the hyperparameter we wish to tune
# The values are a list of values which we wish to tune the hyperparameter over
# If we specify multiple parameters, ALL POSSIBLE parameter combinations will
# be tried
param_grid = {'n_neighbors' : np.arange(1,31,2)}

# Instantiate our classifier
knn = KNeighborsClassifier()

# This returns a grid-search object which you can then fit to the data
knn_cv = GridSearchCV(knn, param_grid, cv=5)

# Fit the gridsearch, this performs the actual grid search in-place
knn_cv.fit(X,y)

# We can then extract the optimal parameters
print('optimal k: ')
print(knn_cv.best_params_)

# 
print('best accuracy score with optimal k: ')
print(knn_cv.best_score_)

#%% Use the optimal k=7 to get the confusion matrix and classification_report

from sklearn.metrics import confusion_matrix, classification_report

# Instantiate the model with our optimal k
knn7 = KNeighborsClassifier(n_neighbors=7)

# Fit classifier on training set
knn7.fit(X_train,y_train)

# Get predictions for test set
y_pred = knn7.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

