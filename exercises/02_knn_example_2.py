#######################################################
########### Human Performance Systems, Inc. ###########
#######################################################

#####################################################
## Classification in Python                        ##
## Day 2 -- kNN Example   2                        ##
#####################################################


## NOTE: To run individual code cells, press Shift + enter for PCs 
#==============================================================================

#%% TRAIN_TEST_SPLIT

'''
1. Split the dataset into two pieces: 
    TRAINING_SET (X_train, y_train)
    TESTING_SET (X_test, y_test)
2. Train the model on the TRAINING_SET
3. Test the model on the TESTING_SET, and evaluate how well we did.

What does this accomplish?

- Model can be trained and tested on different data
- Response values are known for the TESTING_SET, and thus predictions 
  can be evaluated
- Testing accuracy is a better estimate than training accuracy of 
  out-of-sample performance
'''
#%% Re-load the data
# import load_iris function from datasets module
from sklearn.datasets import load_iris

# save "bunch" object containing iris dataset and its attributes
iris = load_iris()

# store feature matrix in "X"
X = iris.data

# store response vector in "y"
y = iris.target

#%% STEP 1: split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42,stratify=y)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#%% Step 2: Instantiate and Fit our models on the TRAINING_SET
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Instantiate KNeighborsClassifier with n_neighbors = 1
knn1 = KNeighborsClassifier(n_neighbors=1)

# Instantiate KNeighborsClassifier with n_neighbors = 5
knn5 = KNeighborsClassifier(n_neighbors=5)

# Instantiate LogisticRegression
logreg = LogisticRegression()

# First with kNN with n_neighbors = 1
knn1.fit(X_train, y_train)

# Finally, kNN with n_neighbors = 5
knn5.fit(X_train, y_train)

# Next with Logistic Regression
logreg.fit(X_train, y_train)
#%% Step 3: Make Predictions on our TEST_SET

# Pass X_test set through our fitted knn1 model, assign the results
# to knn1_y_pred
knn1_y_pred = knn1.predict(X_test)

# Pass X_test through our fitted knn5 model, assign the results
# to knn5_y_pred
knn5_y_pred = knn5.predict(X_test)

# Pass the X_test through our fitted logreg model, assign the results
# to logreg_y_pred
logreg_y_pred = logreg.predict(X_test)

#%% Step 4: Compare actual response values (y_test) with these predicted values
from sklearn.metrics import accuracy_score

knn1_acc = accuracy_score(y_test, knn1_y_pred) 
knn5_acc = accuracy_score(y_test, knn5_y_pred)
logreg_acc = accuracy_score(y_test, logreg_y_pred)

print("kNN, 1-Neighbor Accuracy: {0:.6f}".format(knn1_acc))
print("kNN, 5-Neighbor Accuracy: {0:.6f}".format(knn5_acc))
print("Logreg Accuracy: {0:.6f}".format(logreg_acc))

#%% How to choose k?

#OK, so k=5 is better, why not choose k=50?
knn50 = KNeighborsClassifier(n_neighbors=50)
knn50.fit(X_train,y_train)
print("kNN, 50-Neighbor Accuracy: {0:.6f}".format(knn50.score(X_test,y_test)))

