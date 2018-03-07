#==================================================================================
#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import math

#==================================================================================
#%%

#    Exercise 1:
#    Build the simple tennis table we just reviewed, in python as a dataframe. Label the columns.
#    We are going to calculate entropy manually, but in python.
#
#    Make sure to enter all variables as binary vs. the actual categorical names
#
#    Name the dataframe tennis_ex.

#==================================================================================
#%%

#    Exercise 2:
#
#    Build a function that will calculate entropy. Calculate entropy for the table we just went over
#    in the example, but in python
#
#    This is for the first split.

#==================================================================================
#%%

#    Dataset: Pima Indians Diabetes Data
#
#    Sources:
#       (a) Original owners: National Institute of Diabetes and Digestive and
#                            Kidney Diseases
#       (b) Donor of database: Vincent Sigillito (vgs@aplcen.apl.jhu.edu)
#                              Research Center, RMI Group Leader
#                              Applied Physics Laboratory
#                              The Johns Hopkins University
#                              Johns Hopkins Road
#                              Laurel, MD 20707
#                              (301) 953-6231
#       (c) Date received: 9 May 1990
#
#    For Each Attribute: (all numeric-valued)
#       1. Number of times pregnant
#       2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
#       3. Diastolic blood pressure (mm Hg)
#       4. Triceps skin fold thickness (mm)
#       5. 2-Hour serum insulin (mu U/ml)
#       6. Body mass index (weight in kg/(height in m)^2)
#       7. Diabetes pedigree function
#       8. Age (years)
#       9. Class variable (0 or 1)  0 = sign of diabetes / 1 = no sign of diabetes
#
#==================================================================================
#%%

#    Exercise 3:
#    What are we interested in predicting? What is the target value?
#
#    Read in the data 'diabetes.csv' and build a table of counts of the target variable.

#==================================================================================
#%%

#     Exercise 4:
#
#     Clean the data. Make sure you go through all the steps we did with the pregnancy data set
#
#     1- make sure all data is numeric
#     2- check for NAs
#     3- separate the predictor array from the remaining dataset

#==================================================================================
#%%


#    Exercise 5:
#
#    Split the data into 80/20 train/test sets. Set random state to 1.

#==================================================================================
#%%

#    Exercise 6:
#
#    Build the decision tree, fit the model and then predict and save the predictions
#    as a variable.

#==================================================================================
#%%

#    Exercise 7:
#
#    Model evaluation
#    Look at the metrics to evaluate the model.
#
#    First build a confusion matrix.
#
#    Now use the console to calculate the following metrics by hand.
#
#    Accuracy
#    Misclassification Rate
#    TPR
#    FPR

#==================================================================================
#%%


#    Exercise 8:
#
#    Check the accuracy you calculated vs the metric via sklearn.
#    Save the score as dec_acc.

#==================================================================================
#%%


#    Exercise 9:
#
#    Build a random forest now, keeping in mind the metrics from the decision tree.
#    Save the model as a variable, fit. Use 100 trees.
#
#    Random forest should be more precise because of the concept of bagging.


#==================================================================================
#%%


#    Exercise 10:
#
#    Evaluate the random forest. Look at the confusion matrix and accuracy score.
#    Save the score as rf_acc.

#==================================================================================
#%%

#    Exercise 11:
#
#    Build an ROC curve. What is this showing? What is the value we want to take away from this
#    and what is it telling us?


#==================================================================================
#%%

#    Exercise 12:
#
#    Let's look finally at the variable importance, and build a plot. This is something
#    we could show clients and non-data people to explain the findings from the model.
#    What are some conclusions you can draw?












































