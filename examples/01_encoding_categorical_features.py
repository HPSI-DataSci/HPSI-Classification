#######################################################
########### Human Performance Systems, Inc. ###########
#######################################################

############################################
## Classification in Python               ##
## Day 1 -- Encoding Categorical Features ##
############################################


## NOTE: To run individual code cells, press Shift + enter for PCs 
# ==================================================================================


#%% Create Dummy Data

import pandas as pd

# create a DataFrame of dummy data
df = pd.DataFrame([
        ['green', 'M', 10.1, 'class1'],
        ['red', 'L', 13.5, 'class2'], 
        ['blue', 'XL', 15.3, 'class1']])

# specify the column names of the DataFrame
df.columns = ['color', 'size', 'price', 'classlabel']

print(df)

#%% Mapping Ordinal Features

# define a mapping dictionary for shirt size
size_mapping = {
        'XL': 3,
        'L': 2,
        'M': 1}

# apply the mapping dictionary with map()
df['size'] = df['size'].map(size_mapping)

print(df)

#%% Recovering Labels by Inverse Mapping

# create an inverse mapping by swapping the keys & values in the mapper
inv_size_mapping = {v: k for k, v in size_mapping.items()}

# recover the class labels
df['size'].map(inv_size_mapping)

#%% Encoding Class Labels with the LabelEncoder

from sklearn.preprocessing import LabelEncoder

# instantiate the label encoder
class_le = LabelEncoder()

# encode the target labels
y = class_le.fit_transform(df['classlabel'].values)

print(y)

#%%% Inverse Mapping with the Label Encoder

# recover the original target labels 
class_le.inverse_transform(y)

#%% One-Hot Encoding Categorical Features (1)

# create a feature matrix
X = df[['color', 'size', 'price']].values

# instantiate the label encoder 
color_le = LabelEncoder()

# transform the first column in the feature matrix (color)
X[:, 0] = color_le.fit_transform(X[:, 0])

print(X)

#%% One-Hot Encoding Categorical Features (2)

from sklearn.preprocessing import OneHotEncoder

# instantiate the one-hot encoder and specify the first column to transform
ohe = OneHotEncoder(categorical_features=[0])

# one-hot encode color variable & return array instead of sparse matrix
ohe.fit_transform(X).toarray()

#%% One-Hot Encoding Categorical Features (3)

# we can also one-hot encdode with the get_dummies() method
pd.get_dummies(df[['price', 'color', 'size']])   

#%% One-Hot Encoding Categorical Features (4)

# we can drop the reference/baseline with the drop_first parameter
pd.get_dummies(df[['price', 'color', 'size']],
               drop_first=True)   
