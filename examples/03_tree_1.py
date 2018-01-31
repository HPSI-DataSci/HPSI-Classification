#==================================================================================
#%%
#Here we are going to generate a basic decision tree using SKLearn Slide 22

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn import tree
import sklearn as sk
import pydotplus
import graphviz
import os
import sys
import collections
#==================================================================================
#%%
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
#==================================================================================
#%%
#Set up all our data in a couple of data frames.
customers = pd.DataFrame()
customers['purchases_amount'] = [105, 65, 89, 99, 149, 102, 34, 120, 129, 39,
                                 20, 30, 109, 40, 55, 100, 23, 20, 70, 10]

customers['purchases_items'] = [1, 4, 5, 4, 7, 1, 2, 10, 6, 5,
                                1, 3, 2, 1, 5, 10, 3, 3, 1, 1]

customers['promo'] = [1, 1, 0, 1, 0, 0, 0, 0, 0, 1,
                      1, 1, 1, 0, 1, 1, 1, 0, 1, 1]

customers['email_list'] = [1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
                           0, 1, 1, 0, 1, 0, 1, 1, 0, 0]

customers['checkouts'] = [1, 5, 3, 3, 1, 2, 4, 4, 1, 1,
                          1, 1, 2, 4, 1, 1, 2, 1, 1, 1]

repeat_customers = pd.DataFrame()

repeat_customers['repeat'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
customers.head()
repeat_customers.head()
customers.info()
repeat_customers.info()
#==================================================================================
#%%
#Need to install graphviz and pydotplus
# To install pydotplus use !pip install pydotplus in the console, let's you use the console vice a command window
# Alternatively you can also use the anaconda command prompt
## from IPython.display import Image
# The package above makes things easier to visualize 
#Below we are calling the packages we just installed
#==================================================================================
#%%
# Initialize and train our tree.
clf_1 = tree.DecisionTreeClassifier(
    criterion='entropy',
    max_features=1,
    max_depth=4,
    random_state = 1000
)

clf_1.fit(customers, repeat_customers)
l = customers.columns
dot_data = tree.export_graphviz(clf_1,
                                feature_names=list(l),
                                out_file=None,
                                filled=True,
                                rounded=True)
#==================================================================================
#%%
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('brown', 'forestgreen')
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
#==================================================================================
#%%
# Slide 22 another example
#Iris Dataset building a decision tree
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()

clf_2 = tree.DecisionTreeClassifier()
#Creates the decision tree
clf = clf_2.fit(iris.data, iris.target)

dot_data = tree.export_graphviz(clf_2,
                                feature_names=iris.feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('brown', 'forestgreen')
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

#==================================================================================
#%%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

#The IRIS dataset is currently a numpy arrary 
iris = load_iris()
#To get a better look at the data let's construct a pandas dataframe, just for reference purposes
iris_pd=pd.DataFrame(iris.data, columns=iris['feature_names'])
iris_pd.info()
iris_pd.head()

#Also reducing the number of features 
X = iris.data[:, 2:] # petal length and width, serve as the training dataset
y = iris.target # serve as the test dataset or contains the class labels 

#Also controlling the depth 
clf_3 = DecisionTreeRegressor(max_depth=2, random_state=42)
clf_3.fit(X, y)

dot_data = tree.export_graphviz(clf_3,
                                feature_names=iris.feature_names[2:],
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('brown', 'forestgreen')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree2.png')
graph.write_svg('tree2.svg')
#==================================================================================
#%%
#Replicate the model with all the features so we can compare to the classification model
X = iris.data #using the entire dataset 
y = iris.target

print(X)
print(y)

clf_4 = DecisionTreeRegressor(max_depth=6,  random_state=0)
clf_4.fit(X, y)
dot_data = tree.export_graphviz(clf_4,
                                feature_names=iris.feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('brown', 'forestgreen')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree3.png')
graph.write_svg('tree3.svg')
#==================================================================================
#%% Ok now that we actually have a model lets use it to predict something and then compare results from our models

clf_2.predict([[5,1.5,2,2]])
clf_4.predict([[5,1.5,2,2]])
#==================================================================================
#Seems we have different results
#%% # Slide 39 confusion matrix and ROC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

clfpred = clf.predict(iris.data)
sk.metrics.confusion_matrix(y,clfpred)

regpred = clf_4.predict(iris.data)
sk.metrics.confusion_matrix(y,regpred)
#==================================================================================
#%% Slide 41
#The ROC is best used in a 2-demensional format so we will pull forward a previous example focused on customer prediction

customerpred = clf_1.predict(customers)

print(repeat_customers)
customerpred

#Here we see the confusion matrix
sk.metrics.confusion_matrix(repeat_customers, customerpred)

#The roc_curve has three outputs fpr:false positive rate, tpr: true positive rate and thresholds
false_positive_rate, true_positive_rate, thresholds = roc_curve(repeat_customers, customerpred) #Here we are developing variables for each output

roc_auc = auc(false_positive_rate, true_positive_rate)


#The code below plots these outputs
plt.figure(1)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#==================================================================================
sk.metrics.roc_curve(repeat_customers,customerpred) # Stand alone outside of the above process

#%% Probability for ensemble methods slide 48
import numpy as np
import os

heads_proba = 0.51
coin_tosses = (np.random.rand(10000, 10) < heads_proba).astype(np.int32)
cumulative_heads_ratio = np.cumsum(coin_tosses, axis=0) / np.arange(1, 10001).reshape(-1, 1)

#Defining where print outs will be run 
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


PROJECT_ROOT_DIR = "."
CHAPTER_ID = "ensembles"

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)

def save_fig(fig_id, tight_layout=True):
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id) + ".png", format='png', dpi=300)

plt.figure(figsize=(8,3.5))
plt.plot(cumulative_heads_ratio)
plt.plot([0, 10000], [0.51, 0.51], "k--", linewidth=2, label="51%")
plt.plot([0, 10000], [0.5, 0.5], "k-", label="50%")
plt.xlabel("Number of coin tosses")
plt.ylabel("Heads ratio")
plt.legend(loc="lower right")
plt.axis([0, 10000, 0.42, 0.58])

plt.show()
#==================================================================================
#%% Ensemble example
#Get our data and the function we will use to develop train test datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

#Load in the models we will be using
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier


#Use the train test function on the moon dataset
X, y = make_moons(n_samples=500, noise=0.30, random_state=1)
plt.figure(3)
plt.scatter(X[:,0],X[:,1], c=y)
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#Next we are establishing the variables for each of our models
log_clf = LogisticRegression(random_state=1)
rnd_clf = RandomForestClassifier(random_state=1)
svm_clf = SVC(random_state=1)

#Next we are just developing a voting classifier as we discussed that uses hard voting to develop an ensemble method
voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='hard')
voting_clf.fit(X_train, y_train)

#Now let's see how we did
from sklearn.metrics import accuracy_score # This metric calculates the error rate for each of our models

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
    
    
#==================================================================================    
 #%% Let's run some simple code to take a look at the bagging process
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

#This is for classification if we want to run the same process on regression it's BaggingRegressor
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,max_samples=100, bootstrap=True)
#n_estimators = number of tree classifiers
#max_samples= number of times each of those classifiers is trained using random sampling with replacement
#bootstrap = tells the baggingclassifier to use with replacement 
#n_jobs = tells the classifier the number of CPUs to use for the classifier, -1 tells to use all available
bag_clf.fit(X_train, y_train) #No we run the data from the moon dataset

y_pred=bag_clf.predict(X_test) #Generate predictions using the test dataset
print(accuracy_score(y_test, y_pred)) #Calculate the accuracy score   
    
    
#==================================================================================
#%%
#    #%% Let's run some simple code to take a look at the bagging process
#    from sklearn.ensemble import BaggingClassifier
#    from sklearn.tree import DecisionTreeClassifier
#
#    #This is for classification if we want to run the same process on regression it's BaggingRegressor
#    #bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,max_samples=100, bootstrap=True,n_jobs=-1)
#    bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=2,max_samples=10, bootstrap=True,n_jobs=-1)
#    #n_estimators = number of tree classifiers
#    #max_samples= number of times each of those classifiers is trained using random sampling with replacement
#    #bootstrap = tells the baggingclassifier to use with replacement
#    #n_jobs = tells the classifier the number of CPUs to use for the classifier, -1 tells to use all available
#    bag_clf.fit(X_train, y_train) #No we run the data from the moon dataset
#    y_pred=bag_clf.predict(X_test) #Generate predictions using the test dataset
#    print(accuracy_score(y_test, y_pred)) #Calculate the accuracy score
#    #%% Let's use the Random Forrest Classifier

#==================================================================================
#%%
from sklearn.ensemble import RandomForestClassifier

rnd_clf=RandomForestClassifier(n_estimators=500, max_leaf_nodes=16)
#n_estimators = number of tree classifiers
#max_leaf_nodes = complexity of the model, limits the terminal leaf nodes

rnd_clf.fit(X_train, y_train)

y_pred_rf=rnd_clf.predict(X_test)
accuracy_score(y_test,y_pred_rf)
#let's take a look at feature importance
print(rnd_clf.feature_importances_)
#X_train


from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

#==================================================================================
#%%
# load the iris datasets
iris = datasets.load_iris()
# Fit the model
ext = ExtraTreesClassifier()
ext.fit(iris.data, iris.target)
# display the relative importance of each attribute
print(ext.feature_importances_)

# Below will align the feature importance with the variable being used and print the result
for name, importance in zip(iris["feature_names"], ext.feature_importances_):
    print(name, "=", importance)