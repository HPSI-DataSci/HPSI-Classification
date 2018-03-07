#######################################################
########### Human Performance Systems, Inc. ###########
#######################################################

#####################################################
## Classification in Python                        ##
## Day 2 -- Naive Bayes Example 2                  ##
#####################################################


## NOTE: To run individual code cells, press Shift + enter for PCs 
#==============================================================================


#%%

# start with a simple example
train_simple = ['A machine learning class is a fun class',
                'Who said he likes machine learning MORE THAN Mike?',
                'Amir likes machine learning the more than Mike likes it']



# Corpus

# Documents

#%%
# COUNTVECTORIZER: 'convert text into a matrix of token counts'
# http://scikit-learn.org/stable/modules/generated/##sklearn.feature_extraction.text.CountVectorizer.html

from sklearn.feature_extraction.text import CountVectorizer
# learn the 'vocabulary' of the training data
# instantiate our "vectorizer"
vect = CountVectorizer(stop_words='english')

#%%
# fit our vectorizer to the training data (train_simple)
# look at those 3 strings, and learn what words we have
vect.fit(train_simple)

#%%
#get_feature_names() a method that returns a list of strings
vect.get_feature_names()

#%%
# Transform training data into a 'document-term matrix'
# Also known as a "Bag of Words" technique
# Also known as tokenization/vectorization
train_simple_dtm = vect.transform(train_simple)
train_simple_dtm

# Things we are not considering (NLP is a huge field)
# Word Embeddings
# n-grams
# POS tagging
# stemming/lemmatizing
# lots more

#%%
type(train_simple_dtm)
# Why is it a 3x8 sparse matrix?

#%%
# examine the sparse document term matrix
print(train_simple_dtm)


#%%
# we can convert a sparse matrix to "dense" representation with .toarray()
train_simple_dtm.toarray()

#%%
import pandas as pd
# examine the vocabulary and document-term matrix together
pd.DataFrame(train_simple_dtm.toarray(),columns=vect.get_feature_names())

#%%
# transform testing data into a document-term matrix (using existing vocabulary)
test_simple = ["mike said nah, this class is definitely not fun anymore"]
test_simple_dtm = vect.transform(test_simple)
test_simple_dtm.toarray()
pd.DataFrame(test_simple_dtm.toarray(), columns=vect.get_feature_names())

#What happened here? Why don't we have "definitely"?
#%%
# We skipped the fit, we only did a transform, so what happened to "definitely"?
# Since "definitely" wasn't seen during model training, it wasn't in our vocabulary


# So, taking a step back, we are pretending that we trained a model on our 3x12 sparse dtm
# We are ready to make predictions on whether our new message was "sent during class" or "sent after class"

# Recall, when we pass in a new test point (think iris, [3,5,4,6]) it needs to be the same shape as our training records, ()

#%%
'''
CLASS: Naive Bayes SMS spam classifier
DATA SOURCE: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
'''
#%%
import os
#os.getcwd()
#os.listdir()
#os.chdir()

#%%
## READING IN THE DATA
# read tab-separated file using pandas

# SMSSpamCollection.txt is a .tsv
df = pd.read_table('SMSSpamCollection.txt',
                   sep='\t', header=None, names=['label', 'msg'])

#%%
# examine the data
df.head(20)

#%%
df.label.value_counts()

#%%
df.msg.describe()

#%%
# convert label to a binary variable
df['label'] = df.label.map({'ham':0, 'spam':1})
df.head()

#%% REPEAT PATTERN WITH SMS DATA
# split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.msg, df.label, random_state=42)

# Check shapes
print(X_train.shape)
print(X_test.shape)

# But why are we doing train_test_split before CountVectorizer?

#%% Vectorizing our dataset

# We want to simulate the real world, where our texts will have words that are
# not in our training set...


# instantiate the vectorizer
vect = CountVectorizer()


#%%

# learn vocabulary and create document-term matrix in a single step
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm

#%% Let's look at the X_train_dtm
print(X_train_dtm)
#%%
# Check out the vocabulary of our SMS training texts
vect.get_feature_names()
#%%
# transform testing data into a document-term matrix
# remember, we are using the vocabulary learned from TRAINING
X_test_dtm = vect.transform(X_test)
X_test_dtm

#%% BUILDING AND EVALUATING A MODEL

# MULTINOMIAL NAIVE BAYES
# http://scikit-learn.org/stable/modules/naive_bayes.html

# The multinomial Naive Bayes Classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts.

#%%
# train a Naive Bayes model using train_dtm
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
%time nb.fit(X_train_dtm, y_train)

#%%
# make predictions on test data using test_dtm
%time y_pred = nb.predict(X_test_dtm)
y_pred
#%% Guesses for our accuracy?
# DUMMY CLASSIFIER
# http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
from sklearn.dummy import DummyClassifier
clf = DummyClassifier(strategy='most_frequent',random_state=0)
clf.fit(X_train_dtm,y_train)
clf.score(X_test_dtm,y_test)
#%%
# what is our null accuracy?
df.label.value_counts()[0]/len(df.label)

#%%
# compare predictions to true labels
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
#%%
# predict probabilities and calculate AUC
y_prob = nb.predict_proba(X_test_dtm)[:, 1]
y_prob

print(metrics.roc_auc_score(y_test, y_prob))


#%%
# calculate the fpr and tpr for all thresholds of the classification
probs = nb.predict_proba(X_test_dtm)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#%%
# exercise: show the message text for the false positives
X_test[(y_test==0) & (y_pred==1)]

#%%
# exercise: show the message text for the false negatives
X_test[(y_test==1) & (y_pred==0)]

#%% EXAMINING OUR MODEL:
#Remember the learned parameters?
nb.feature_count_

#what is this?
#%%
nb.feature_count_.shape
# for each and every token, it calculates the conditional probability of each 
# token, given each class (spam/ham)
# So, given spam, what's the probability of "Click"

# to make a prediction, it calculates the conditional probability of a class, 
# given the tokens in that message

# the bottom line, is that it learns the "spamminess" of each token
# or the "1-classiness" of each token, and visaversa

#%%
# number of times each token appears across all HAM messages
ham_token_count = nb.feature_count_[0,:]
ham_token_count

#%%
# number of times each token appears across all SPAM messages
spam_token_count = nb.feature_count_[1,:]
spam_token_count

#%%
#Get the learned vocab from the training data into a list
X_train_tokens = vect.get_feature_names()
#%%
# create a DataFrame of tokens with their separate ham anbd spam counts
tokens = pd.DataFrame({'token':X_train_tokens, 'ham':ham_token_count, 'spam':spam_token_count})
tokens['ham'] = tokens.ham + 1
tokens['spam'] = tokens.spam + 1

#%%
#what are the most spammy words then?
tokens.sort_values(by='spam',ascending=False,inplace=True)
tokens.head(200)

#%% Look up a word
tokens[tokens.token=='respond']

#%%
tokens['spam_ratio'] = (tokens['spam'])/(tokens['ham'] + tokens['spam'])
tokens.sort_values(by='spam_ratio',ascending=False,inplace=True)
tokens
#%%
#Comparing Models with logistic regression

# A couple points, recall how fast naive bayes was, we are now going to fit a logreg model and compare the speed
# Logistic Regression is quite a bit slower than Naive Bayes, however it is slightly more robust in terms of what it requires for preprocessing its inputs. Where Multinomial Naive Bayes is going to fail on negative values (it can't take a negative number) Logistic Regression doesn't care, it will work just fine (although "slowly")

#%%
#Let's try Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
#%%
# train the model using X_train_dtm
%time logreg.fit(X_train_dtm,y_train)
#%%
# make class predictions for X_test_dtm
%time y_pred_class = logreg.predict(X_test_dtm)
#%%
# calculate predicted probabilities for X_test_dtm
y_pred_prob = logreg.predict_proba(X_test_dtm)[:,1]
y_pred_prob
#%%
# calculate accuracy
metrics.accuracy_score(y_test,y_pred_class)
#%%
# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)

#%%
probs = logreg.predict_proba(X_test_dtm)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

