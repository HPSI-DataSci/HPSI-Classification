#######################################################
########### Human Performance Systems, Inc. ###########
#######################################################

############################################
## Classification in Python               ##
## Day 1 -- Performance Metrics           ##
############################################


## NOTE: To run individual code cells, press Shift + enter for PCs 
# ==================================================================================


import numpy as np
 
#%% Accuracy 

from sklearn.metrics import accuracy_score 
y_pred = [0, 2, 1, 3] 
y_true = [0, 1, 2, 3] 

# fraction of classified samples
accuracy_score(y_true, y_pred)

# number of correctly classified samples
accuracy_score(y_true, y_pred, normalize=False) 

#%% Confusion Matrix

from sklearn.metrics import confusion_matrix 
y_true = [0, 1, 0, 1]
y_pred = [1, 1, 1, 0] 
confusion_matrix(y_true, y_pred) 

# we can extract the following info in the binary case
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print((tn, fp, fn, tp)) 

#%% Misclassification Rate, Precision, & Recall

from sklearn.metrics import accuracy_score, precision_score, recall_score

miss_rate = 1 - accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true,y_pred)

print("Misclassification Rate: {:03.2f}".format(miss_rate))
print("Precision: {:03.2f}".format(precision))
print("Recall: {:03.2f}".format(recall))

#%% f1-score and Classification Report

from sklearn.metrics import f1_score
print("f1-score: {:03.2f}".format(f1_score(y_true, y_pred)))

# can do precision, recall, and f1 all together
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))

#%% Example Prep

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = [8, 8]

# read the data into a Pandas DataFrame
import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(url, header=None, names=col_names)

# define X and y
feature_cols = ['pregnant', 'insulin', 'bmi', 'age']
X = pima[feature_cols]
y = pima.label

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# make class predictions for the testing set
y_pred_class = logreg.predict(X_test)

#%% Thresholding (1)

# print out the first 10 predicted probabilities for each class
print(logreg.predict_proba(X_test)[0:10, :])

#%% Thresholding (2)

# print the first 10 predicted probabilities for class 1
print(logreg.predict_proba(X_test)[0:10, 1])

#%% Thresholding (3)

# store the predicted probabilities for class 1
y_pred_prob = logreg.predict_proba(X_test)[:, 1]

#%% Thresholding (4)

plt.figure()
plt.hist(y_pred_prob)
plt.xlim(0, 1)
plt.title('Histogram of Predicted Probabilities')
plt.xlabel('Predicted Probability of Diabetes')
plt.ylabel('Frequency')
plt.show()

#%% Thresholding (5)

from sklearn.preprocessing import binarize

# predict diabetes if the predicted probability is greater than 0.35
y_pred_class_adj = binarize([y_pred_prob], 0.35)[0]

#%% Thresholding (6)

print('Original Confusion Matrix: ')
print('-'*50)
print(confusion_matrix(y_test, y_pred_class))
print('Confusion Matrix after Threshold Adjustment: ')
print('-'*50)
print(confusion_matrix(y_test, y_pred_class_adj))

#%% Thresholding (7)

print('Original Classification Report: ')
print('-'*50)
print(classification_report(y_test, y_pred_class))
print('Classification Report After Threshold Adjustment: ')
print('-'*50)
print(classification_report(y_test, y_pred_class_adj))

# here we see the effect of decreasing the threshold makes our model more 
# sensitive because we need to be less sure of a prediction to say someone has 
# diabetes

#%% Precision-Recall Curve (1)

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test,
                                                       logreg.decision_function(X_test))

#%% Precision-Recall Curve (2)

# find threshold closest to zero
close_zero = np.argmin(np.abs(thresholds))

#%% Precision-Recall Curve (3)

plt.figure()
# plot point for decision threshold of zero
plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10, 
             label="threshold zero", fillstyle="none", c='k', mew=2)
# plot precision-recall curve
plt.plot(precision, recall, label="precision-recall curve")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.title("Precision-Recall Curve")
plt.legend(loc="best")
plt.show()

#%% Precision-Recall Curve (4)

precision, recall, thresholds = precision_recall_curve(y_test,
                                                       logreg.predict_proba(X_test)[:,1])

#%% Precision-Recall Curve (5)

# find threshold closest to predicted probabilities of 50%
prob_50 = np.argmin(np.abs(thresholds - 0.5))

#%% Precision-Recall Curve (6)

plt.figure()
# plot threshold for 50% predicted probability threshold
plt.plot(precision[prob_50], recall[prob_50], 'o', markersize=10, 
             label="predicted prob 50%", fillstyle="none", c='k', mew=2)
# plot precision-recall curve
plt.plot(precision, recall, label="precision-recall curve")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.title("Precision-Recall Curve")
plt.legend(loc="best")
plt.show()

#%% ROC Curve (1)

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, 
                                 logreg.decision_function(X_test))

#%% ROC Curve (2)

# find threshold closest to zero
close_zero = np.argmin(np.abs(thresholds))

#%% ROC Curve (3)

plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10, 
         label="threshold zero", fillstyle="none", c='k', mew=2)
plt.legend(loc=4)
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR (recall)")
plt.title("ROC Curve")
plt.show()

#%% ROC Curve (4)

fpr, tpr, thresholds = roc_curve(y_test, 
                                 logreg.predict_proba(X_test)[:,1])

#%% ROC Curve (5)

# find threshold closest to predicted probabilities of 50%
prob_50 = np.argmin(np.abs(thresholds - 0.5))

#%% ROC Curve (6)

plt.plot(fpr[prob_50], tpr[prob_50], 'o', markersize=10, 
         label="predicted prob 50%", fillstyle="none", c='k', mew=2)
plt.legend(loc=4)
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR (recall)")
plt.title("ROC Curve")
plt.show()

#%% ROC Curve (7)

plt.plot(fpr[prob_50], tpr[prob_50], 'o', markersize=10, 
         label="predicted prob 50%", fillstyle="none", c='k', mew=2)
# add a random guess line for reference
plt.plot(np.linspace(0,1), np.linspace(0,1), '--k', label='random guess')
plt.legend(loc=4)
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR (recall)")
plt.title("ROC Curve")
plt.show()

#%% AUC 

from sklearn.metrics import roc_auc_score
# use predicted probability for positive class
auc = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])

print('Area Under the ROC Curve: {}'.format(auc))


