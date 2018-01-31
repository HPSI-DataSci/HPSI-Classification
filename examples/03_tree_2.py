# Import all packages beforehand

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

#==================================================================================
#%%
tennis = pd.read_csv('tennis.csv')

# convert the boolean variable to a string
tennis[["windy"]] = tennis[["windy"]].astype(str)


# Transform factors to numeric
cleanup_cols = {"outlook":     {"sunny": 1, "overcast": 2, "rainy": 0},
                "temp": {"hot": 1, "mild": 2, "cool": 0},
                "humidity": {"high": 0, "normal": 1},
                "windy": {"False": 0, "True": 1},
                "play": {"no": 0, "yes": 1}}

tennis.replace(cleanup_cols, inplace = True)

#==================================================================================
#%%

from sklearn import tree

# Separate the predictor array from the remaining dataset

X_tennis = tennis.drop('play', axis = 1)
y_tennis = tennis['play']


model = tree.DecisionTreeClassifier()

# Look at the model's attributes
model

#==================================================================================
#%%

# Split the data into a train and test set

from sklearn.model_selection import train_test_split


# Split the training and test set, use a 70 test - 30 train split
Tennis_train, Tennis_test, play_train, play_test = train_test_split(X_tennis, 
                                                                    y_tennis, 
                                                                    test_size = .3, 
                                                                    random_state = 1)

model.fit(Tennis_train, play_train)
# ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
import os     
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
clf = model.fit(Tennis_train, play_train)
tree.export_graphviz(clf,  out_file='tree.dot')

# http://webgraphviz.com/
# ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
data_feature_names = ['outlook', 'temp', 'humidity','windy']
dot_data = tree.export_graphviz(clf,
                                feature_names=data_feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
import pydotplus
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')

import collections
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')
graph.write_svg('tree.svg')
# ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
#==================================================================================
#%%

y_predict_tennis = model.predict(Tennis_test)

from sklearn.metrics import accuracy_score

accuracy_score(play_test, y_predict_tennis)

# .8433 - this is significantly better than 50/50 guessing

'''
Now we look at the confusion matrix
'''

from sklearn.metrics import confusion_matrix

confusion = pd.DataFrame(
    confusion_matrix(play_test, y_predict_tennis),
    columns = ['Predicted no play', 'Predicted play'],
    index = ['True No play', 'True play']
)

confusion


#==================================================================================
#%%

pregnancy = pd.read_csv('pregnancy.csv')

# Check how many people in the data ser at pregnant vs. not pregnant

pregnancy.PREGNANT.value_counts()


#==================================================================================
#%%

###############################
##    GO TO EXERCISE 1 & 2   ##
###############################

#==================================================================================
#%%

# Clean up the table to run a decision tree on it

# Transform factors to numeric
cleanup_cols = {"Implied Gender":     {"M": 1, "F": 2,"U": 0},
                "Home/Apt/P.O. Box": {"A": 1, "H": 2, "P": 0}}

pregnancy.replace(cleanup_cols, inplace = True)

# Remove all NAs

pregnancy = pregnancy.dropna()

# Separate the predictor array from the remaining dataset
X = pregnancy.drop('PREGNANT', axis = 1)
y = pregnancy['PREGNANT']

#==================================================================================
#%%

###############################
##    GO TO EXERCISE 3 & 4   ##
###############################

#==================================================================================
#%%

from sklearn.model_selection import train_test_split


# Split the training and test set, use a 70 test - 30 train split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = .3, 
                                                    random_state = 1)

#==================================================================================
#%%

from sklearn import tree

model = tree.DecisionTreeClassifier()

# look at the model's attributes
model


fit = model.fit(X_train, y_train)

# ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
import os     
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
tree.export_graphviz(fit,  out_file='tree.dot')

# http://webgraphviz.com/
# ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
data_feature_names = list(pregnancy.columns)
data_feature_names.remove(data_feature_names[-1])
dot_data = tree.export_graphviz(fit,
                                feature_names=data_feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
import pydotplus
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')

import collections
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree1.png')
graph.write_svg('tree1.svg')
# https://cloudconvert.com/svg-to-pdf
# ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo




a = fit.feature_importances_
a.tolist().index(a.max())


var_imp = pd.DataFrame(a, columns = ['Importance'])
var_imp['variable_names'] = X.columns
var_imp
#==================================================================================
#%%

###############################
##    GO TO EXERCISE 5 & 6   ##
###############################

#==================================================================================
#%%

y_predict = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_predict)

# .8433 - this is significantly better than 50/50 guessing

# Now we look at the confusion matrix

from sklearn.metrics import confusion_matrix

confusion = pd.DataFrame(
    confusion_matrix(y_test, y_predict),
    columns=['Predicted Not pregnant', 'Predicted pregnant'],
    index=['True Not pregnant', 'True pregnant']
)

confusion

#==================================================================================
#%%

###############################
##    GO TO EXERCISE 7 & 8   ##
###############################

#==================================================================================
#%%
# Random forest in scikit-learn

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion = 'gini',
                                n_estimators = 100, 
                                random_state = 1)
forest.fit(X_train, y_train)

fit = forest.fit(X_train, y_train)

#==================================================================================
#%%

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# using the model built on the training data to predict on the test set
y_predict = forest.predict(X_test)

# Let's build the confusion matrix using the actual from the 'y_test' 
# compared to the predictions in y_predict
pd.DataFrame(
    confusion_matrix(y_test, y_predict),
    columns = ['Predicted Not pregnant', 'Predicted pregnant'],
    index = ['True Not pregnant', 'True pregnant']
)

#==================================================================================
#%%

################################
##    GO TO EXERCISE 9 & 10   ##
################################

#==================================================================================
#%%
# Visualize - ROC


accuracy_score(y_test, y_predict)

#.855, increased from the decision tree model
# conda install -c conda-forge ggplot 
from sklearn import metrics
import pandas as pd
from ggplot import *

fpr, tpr, _ = metrics.roc_curve(y_test, y_predict)
auc = metrics.auc(fpr, tpr)

df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
ggplot(df, aes(x = 'fpr', y = 'tpr')) +\
    geom_line() +\
    geom_abline(linetype = 'dashed')

#==================================================================================
#%%
# Visualizing the model
    

# conda install tabulate
from tabulate import tabulate

headers = ["name", "score"]
values = sorted(zip(X_train.columns,fit.feature_importances_), 
                key = lambda x: x[1] * -1)
print(tabulate(values, headers, tablefmt = "plain"))

vimp = fit.feature_importances_

var_imp = pd.DataFrame(vimp, columns = ['Importance'])
var_imp['variable_names'] = X.columns

#==================================================================================
#%%
#Visualizing the model

var_imp.plot.bar(x = 'variable_names')

#==================================================================================
#%%

############################
##    GO TO EXERCISE 11   ##
############################


















