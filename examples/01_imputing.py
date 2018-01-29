#######################################################
########### Human Performance Systems, Inc. ###########
#######################################################

################################
## Classification in Python   ##
## Day 1 -- Imputation Review ##
################################


## NOTE: To run individual code cells, press Shift + enter for PCs 
# ==================================================================================

#%% Create Dummy Data

import pandas as pd
from numpy import NaN

# create a DataFrame of dummy data
df = pd.DataFrame([
        ['Andre' , NaN, 70, 175, 'Russia'],
        ['Mike' ,NaN ,NaN ,NaN , 'USA'], 
        ['Marcela' ,45 ,63 ,NaN , 'Colombia'],
        ['Brent' ,NaN ,NaN ,172.5 , 'USA'], 
        [NaN , 24, 74, 180, 'Croatia'],
        ['Adam' , 35, 56, 170, 'England'],
        ['Francesca', 62, 52, 120, NaN],
        [NaN, NaN, NaN, NaN, NaN], 
        ['Jessica', 50, 67, 122.5, 'USA']])

# specify the column names of the DataFrame
df.columns = ['name', 'age', 'height_in', 'weight_lb', 'country_origin']

print(df)

#%% Checking for Missing Data (1)

# how many missing values are there in each column?
df.isnull().sum()

#%% Checking for Missing Data (2)

# how many missing values are there in each row?
df.isnull().sum(axis=1)

#%% Dropping Missing Values (1)

# drop all rows with missing values
df.dropna(axis=0)  # axis = 0 for rows (default)

#%% Dropping Missing Values (2)

# drop all columns with missing values 
df.dropna(axis=1)  # axis = 1 for columns

# note that all columns have a missing value so they were all dropped

#%% Dropping Missing Values (3)

# drop rows where all columns are missing and save results inplace
df.dropna(how = 'all', inplace=True)

print(df)

#%% Dropping Missing Values (4)

# drop rows that have 3 or more missing values
df.dropna(thresh = 3, inplace=True)

print(df)

#%% Dropping Missing Values (5)

# drop rows where NaN appear in the name column
df.dropna(subset = ['name'], inplace=True)

print(df)

#%% Imputing Missing Values (1)

from sklearn.preprocessing import Imputer

# instantiate imputer for replacing with mean
mean_imputer = Imputer(missing_values='NaN',
			strategy='mean',
			axis=0)

# instantiate imputer for replacing with median
median_imputer = Imputer(missing_values='NaN',
                         strategy='median')

#%% Imputing Missing Values (2)

# fitting the imputer learns the mean of the columns we pass it
mean_imputer.fit(df.age.values.reshape(-1,1))

# display the mean
print(mean_imputer.statistics_)

#%% Imputing Missing Values (3)

# impute the missing values with the transform() method
imputed_ages = mean_imputer.transform(df.age.values.reshape(-1,1))

# replace the age column with the imputed values
df.age = imputed_ages

print(df)


#%% Imputing Missing Values (4)

# impute the missing values by calling the fit_transform() method
df.height_in = median_imputer.fit_transform(df.height_in.values.reshape(-1,1))

print(df)

#%% Imputing Missing Values (5)

# return the most common class label
most_common = df.country_origin.value_counts().index[0]

# fill the NA with the most common class label
df.country_origin.fillna(value=most_common, inplace=True)
