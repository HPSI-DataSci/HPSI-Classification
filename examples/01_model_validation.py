#######################################################
########### Human Performance Systems, Inc. ###########
#######################################################

############################################
## Classification in Python               ##
## Day 1 -- Model Validation              ##
############################################


## NOTE: To run individual code cells, press Shift + enter for PCs 
# ==================================================================================


from sklearn.datasets import load_iris

# load the iris data 
iris = load_iris()

# feature matrix and target vector
X = iris.data
y = iris.target

#%% Hold Out Cross-Validation (1)

from sklearn.model_selection import train_test_split

# split into training (85 %) and test set (15 %)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    random_state=0, 
                                                    test_size=0.15)

#%% Hold Out Cross-Validation (2)

print('% of Data in Training Set: {:03.2f}'.format(X_train.shape[0]/X.shape[0]))
print('% of Data in Test Set: {:03.2f}'.format(X_test.shape[0]/X.shape[0]))

#%% Hold Out Cross-Validation (3)

# further split into training (70 % of total) and validation set (15 % of total)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                    random_state=0, 
                                                    test_size=X_test.shape[0]/X_train.shape[0])

#%% Hold Out Cross-Validation (4)

print('% of Data in Training Set: {:03.2f}'.format(X_train.shape[0]/X.shape[0]))
print('% of Data in Validation Set: {:03.2f}'.format(X_val.shape[0]/X.shape[0]))
print('% of Data in Test Set: {:03.2f}'.format(X_test.shape[0]/X.shape[0]))

#%% k-Fold Cross-Validation Example

# simulate splitting a dataset of 25 observations into 5 folds
from sklearn.cross_validation import KFold

# in practice, we would  shuffle but it is easier to make a point if we don't
kf = KFold(25, n_folds=5, shuffle=False)

# print the contents of each training and testing set
print('{} {:^61} {}'.format('Iteration', 'Training set observations', 'Testing set observations'))
for i, v in enumerate(kf, start=1):
    print('{:^9} {} {}'.format(i, v[0], v[1]))

#%% k-Fold CV (1)
    
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

# split into training (70 %) and test set (30 %)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    random_state=0, 
                                                    test_size=0.30)

# k-fold CV in sklearn (defaults to 3 folds)
scores = cross_val_score(estimator=logreg, X=X_train, y=y_train)

print("CV Scores: {}".format(scores))
# we average the CV scores to summarize the test performance
print('Average CV Score: {}'.format(scores.mean()))

#%% k-Fold CV (2)

# specify a number of folds (i.e. k=5)
scores = cross_val_score(estimator=logreg, X=X_train, y=y_train, cv=5)

# k=10 is the most-widely used

print("CV Scores: {}".format(scores))
# we average the CV scores to summarize the test performance
print('Average CV Score: {}'.format(scores.mean()))

#%% k-Fold CV (3)

from sklearn.model_selection import KFold

# we can make a KFold iterator first
kfold = KFold(n_splits=10, shuffle=True, random_state=0) 

# use our KFold iterator in cross_val_score
scores = cross_val_score(estimator=logreg, X=X_train, y=y_train, cv=kfold)

print("CV Scores: {}".format(scores))
# we average the CV scores to summarize the test performance
print('Average CV Score: {}'.format(scores.mean()))

#%% k-Fold CV (4)

# specify scoring metric in cross_val_score
scores = cross_val_score(logreg, X_train, y_train, 
                         cv=kfold, scoring='accuracy')

print("CV Scores: {}".format(scores))
# we average the CV scores to summarize the test performance
print('Average CV Score: {}'.format(scores.mean()))

# list of possible scoring metrics 
# http://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules 
metrics = ['accuracy', 'precision', 'recall', 'roc_auc']

#%% k-Fold CV (5)

# CV is easily scaled and parallelized 
scores = cross_val_score(logreg, X_train, y_train, 
                         cv=kfold, scoring='accuracy', n_jobs=-1)
# n_jobs=-1 means use all CPU cores

print("CV Scores: {}".format(scores))
# we average the CV scores to summarize the test performance
print('Average CV Score: {}'.format(scores.mean()))

#%% k-Fold CV (6)

from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()

# specify we want to use cv=loo
scores = cross_val_score(logreg, X_train, y_train, 
                         cv=loo, n_jobs=-1)

print("Number of CV Iterations: {}".format(len(scores)))
print('Average CV Score: {}'.format(scores.mean()))

#%% k-Fold CV (7) 

# it is helpful to provide the variability of our CV estimates
scores = cross_val_score(logreg, X_train, y_train, cv=10)
print("CV Scores: {}".format(scores))
# we average the CV scores to summarize the test performance
print('Average CV Score: {:4.3f} +/- {:4.3f}'.format(scores.mean(), scores.std()))
