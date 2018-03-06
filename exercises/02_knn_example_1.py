#######################################################
########### Human Performance Systems, Inc. ###########
#######################################################

#####################################################
## Classification in Python                        ##
## Day 2 -- kNN Example                            ##
#####################################################


## NOTE: To run individual code cells, press Shift + enter for PCs 
#==============================================================================
#%%
'''
K-nearest neighbors (KNN) classification
1. 
Pick a value for K.

2. 
Search for the K observations in the training data that are "nearest" to 
the measurements of the unknown iris.

3. 
Use the most popular response value from the K nearest neighbors as the 
predicted response value for the unknown iris.
'''

#%%

# import load_iris function from datasets module
from sklearn.datasets import load_iris

# save "bunch" object containing iris dataset and its attributes
iris = load_iris()

# store feature matrix in "X"
X = iris.data

# store response vector in "y"
y = iris.target

# print the shapes of X and y
print(X.shape)
print(y.shape)

#%% Scikit-learn 4-step modeling pattern

# STEP 1: Import the class you plan to use
from sklearn.neighbors import KNeighborsClassifier

#%% STEP 2: "Instantiate" the "estimator"

# "Estimator" is scikit-learn's term for model
# "Instantiate" means "make an instance of"

knn1 = KNeighborsClassifier(n_neighbors=1)

# Name of the object does not matter
# Can specify tuning parameters (aka "hyperparameters") during this step
# All parameters not specified are set to their defaults
print(knn1)

#%% STEP 3: Fit the model with data (aka "model training")

# Model is learning the relationship between X and y
# Occurs in-place
knn1.fit(X, y)
#%% STEP 4: Predict the response for a new observation

# new observations are called "out-of-sample" data
# uses the information it learned during the model training process
knn1.predict([[3, 5, 4, 2]])

#%%
# returns a NumPy array
# can predict for multiple observations at once
X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
knn1.predict(X_new)

# notice that the model predicts values of "2", and "1" which means that the prediction for the first 
# unknown iris was a "2", and for the second unknown iris was a "1"
#%% Using a different value for K

# this is considered "model tuning" - adjusting hyperparameters
# re-instantiate the model (using the value K=5)
knn5 = KNeighborsClassifier(n_neighbors=5)

# fit the model with data
knn5.fit(X, y)
#%%

# use the new model (knn,n_neighbors=5) to predict the response for the same pre-defined X_new
knn5.predict(X_new)

# this time, the model predicts the value "1" for BOTH unknown iris'
#%% Trying a different classifier

# Because sklearn has a uniform estimator interface, it is easy to try a new model
# import the Logistic Regression classifier
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X, y)

#%%

# Use the new classifier to predict the response for new observations (X_new)
logreg.predict(X_new)

# notice that this model (logistic regression) predicts values of "2", and "0" 

#%% Comparing Model Predictions

print("knn, 1 neighbor, predicts: ",knn1.predict(X_new))
print("knn, 5 neighbors, predicts: ",knn5.predict(X_new))
print("logistic regression, predicts: ",logreg.predict(X_new))

# Which model is right? Which predictions are correct?

#%%
# the most common metric for classification model eval: classification accuracy
# lets compute classification accuracy for each of our models

from sklearn.metrics import accuracy_score

knn1_acc = accuracy_score(y,knn1.predict(X)) 
knn5_acc = accuracy_score(y,knn5.predict(X))
logreg_acc = accuracy_score(y,logreg.predict(X))

print("kNN, 1-Neighbor Accuracy: {}".format(knn1_acc))
print("kNN, 5-Neighbor Accuracy: {}".format(knn5_acc))
print("Logreg Accuracy: {}".format(logreg_acc))

# Clearly, kNN with n_neighbors = 1 is the most accurate, so it's the best

# What Accuracy is this? Training/Testing?
#%% TAKEAWAYS 
'''
PROBLEMS WITH TRAINING AND TESTING ON THE SAME DATA
1.
Remember our goal (with model evaluation) is to estimate how well our model
will perform on **Out-of-Sample data

2.
We just saw that maximizing training accuracy can reward overly complex
models that won't necessarily generalize well

3.
Unnecessarily complex models overfit to the training data

"models that overfit have learned the noise in the data rather than the signal"

In the case of k-NN, with k=1, we create an overly complex model because it
follows the noise in the data
'''
