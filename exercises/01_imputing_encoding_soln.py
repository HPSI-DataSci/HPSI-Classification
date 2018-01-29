#######################################################
########### Human Performance Systems, Inc. ###########
#######################################################

#####################################################
## Classification in Python                        ##
## Day 1 -- Imputing Exercise Solutions            ##
#####################################################


## NOTE: To run individual code cells, press Shift + enter for PCs 
# ==================================================================================
#%%
# Import Required Packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
pd.set_option('max_columns', 500)
pd.set_option('max_rows', 500)
pd.set_option('max_colwidth', 500)

#%%
# check current working directory:
# notice the filepath in the top right also shows this directory
print("starting directory: ",os.getcwd())

# if working directory isn't correct, set it:
os.chdir('/Users/brentskoumal/Desktop/HPSI/HPSI_exercises')

# confirm working directory (you can also check in the top right)
#print("changed directory: ",os.getcwd())
#%% Load in the filtered loans dataset 
'''
Here's an outline of what we'll be doing in this exercise:

 - Handling Missing Values
 - Investigating Categorical Columns
 - Converting Categorical Columns To Numeric Features
 - Mapping Ordinal Values To Integers
 - Encoding Nominal Values As Dummy Variables

First though, let's load in the data from last section's final output:
Load "filtered_loans.csv" into a Pandas DataFrame called "filtered_loans" and investigate the size/shape/dtypes
'''
filtered_loans = pd.read_csv('filtered_loans.csv')

print("\nShape of DataFrame: ")
print("ROWS: {0}\nCOLUMNS: {1}\n".format(filtered_loans.shape[0],filtered_loans.shape[1]))
print("\nDataFrame dtypes: ")
print(filtered_loans.dtypes)
print("\ndtype Counts: ")
print(filtered_loans.dtypes.value_counts())
#%% Drop the following columns that were determined to have redundant information: last_credit_pull_d, addr_state, pymnt_plan, title, eaerliest_cr_line
drop_cols = ['last_credit_pull_d','addr_state','pymnt_plan','title','earliest_cr_line']
filtered_loans = filtered_loans.drop(drop_cols,axis=1)

#%% Use the Pandas DataFrame method isnull() to return a DataFrame containing Boolean values: Then, use the Pandas DataFrame method sum() to calculate the number of null values in each column.

null_counts = filtered_loans.isnull().sum()
print("\n\nNumber of null values in each column:\n{}".format(null_counts))

#%% Notice while most of the columns have 0 missing values, revol_util has 50, and pub_rec_bankruptcies contains 697 rows with missing values. Fill in these missing values with the most frequently occuring value from each column. This can be done individually be reassigning each column with the first value from value_counts().index and passing it to .fillna()

filtered_loans['revol_util'] = filtered_loans['revol_util'].fillna(filtered_loans['revol_util'].value_counts().index[0])
filtered_loans['pub_rec_bankruptcies'] = filtered_loans['pub_rec_bankruptcies'].fillna(filtered_loans['pub_rec_bankruptcies'].value_counts().index[0])

#this can also be done with a lambda function to apply over each column in the entire dataframe
#filtered_loans = filtered_loans.apply(lambda x:x.fillna(x.value_counts().index[0]))

null_counts = filtered_loans.isnull().sum()
print("\n\nNumber of null values in each column:\n{}".format(null_counts))

#%% Investigate Categorical Columns, the goal in this section is to have all the columns as numeric columns (int or float data type), and containing no missing values. We just dealt with the missing values, so let's now find out the number of columns that are of the object data type and then move on to process them into numeric form. Check the dtypes.value_counts() to see how many "object" types we have to deal with

print("\n\nData types and their frequency\n{}".format(filtered_loans.dtypes.value_counts()))

#%% We have 7 object columns that contain text which need to be converted into numeric features. Let's select just the object columns using the DataFrame method .select_dtypes(include=['object']), then display the head to get a better sense of how the values in each column are formatted.

object_columns_df = filtered_loans.select_dtypes(include=['object'])
print(object_columns_df.head())

#%% Notice that revol_util column contains numeric values, but is formatted as object. We need to format revol_util as numeric values. Here's what we should do: Use the str.rstrip() string method to strip the right trailing percent sign (%). On the resulting Series object, use the astype() method to convert to the type float. Assign the new Series of float values back to the revol_util column in the filtered_loans.
print(filtered_loans['revol_util'].value_counts())
filtered_loans['revol_util'] = filtered_loans['revol_util'].str.rstrip('%').astype('float')
print(filtered_loans['revol_util'].value_counts())

#%% Notice the interest rate is stored as an object, we need to convert it to a float. Do the same thing that we just did to revol_util - use the str.rstrip() string method to strip the right trailing percent sign (%). On the resulting Series object, use the astype() method to convert to the type float. Assign the new Series of float values back to the revol_util column in the filtered_loans.

filtered_loans.int_rate = filtered_loans.int_rate.str.rstrip("%").astype('float')
#%% emp_length is an ordinal categorical variable and should be mapped to integers accordingly. To map these ordinal values to integers, we can use the pandas DataFrame method replace() to map both grade and emp_length to appropriate numeric values at the same time. Create a dictionary, with key-value pairs for "emp_length" following this pattern "10+ years": 10, "9 years": 9, ... "< 1 year": 0, "n/a":, 0  

emp_dict = {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0 }

filtered_loans.emp_length = filtered_loans.emp_length.replace(emp_dict)
filtered_loans['emp_length'].head()

#%% For nominal categorical variables home_ownership, verification_status, purpose, and term we can simply get the pandas .get_dummies() method. Pass the column names that you want to get_dummies for

filtered_loans = pd.get_dummies(filtered_loans, columns=['home_ownership','verification_status','purpose','term'])

#%% Inspect your hard work. Use pandas .info() method to inspect the filtered_loans DataFrame to make sure all the features are of the same length, contain no null value, and are numericals.
filtered_loans.info()

#%% It is a good practice to store the final output of each section or stage of your workflow in a separate csv file. One of the benefits of this practice is that it helps us to make changes in our data processing flow without having to recalculate everything save the filtered_loans DataFrame to a .csv using the Pandas .to_csv() method

filtered_loans.to_csv("preprocessed_loans.csv",index=False)