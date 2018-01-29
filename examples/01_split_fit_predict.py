#######################################################
########### Human Performance Systems, Inc. ###########
#######################################################

############################################
## Classification in Python               ##
## Day 1 -- Train Test Split/Fit/Predict  ##
############################################


## NOTE: To run individual code cells, press Shift + enter for PCs 
# ==================================================================================


# import sklearn methods and estimator
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#%% Create Feature Matrix & Target Vector

# load the iris data 
iris = load_iris()

# feature matrix and target vector
X = iris.data
y = iris.target

#%% Split into Training & Test Set

# split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#%% Fit & Predict

# instantiate estimator
knn = KNeighborsClassifier(n_neighbors=1)

# fit on training set to 'learn' parameters
knn.fit(X_train, y_train)

# predict on the test set
y_pred = knn.predict(X_test)

#%% Compute the Accuracy 

# print the accuracy score
print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
