#######################################################
########### Human Performance Systems, Inc. ###########
#######################################################

#####################################################
## Classification in Python                        ##
## Day 1 -- Preprocessing Exercise Solutions       ##
#####################################################


## NOTE: To run individual code cells, press Shift + enter for PCs 
# ==================================================================================
#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
pd.set_option('max_columns', 500)
pd.set_option('max_rows', 500)
pd.set_option('max_colwidth', 500)
#%% Set your working directory
# notice the filepath in the top right also shows this directory
print("\nstarting directory: ",os.getcwd())

# if working directory isn't correct, set it:
#os.chdir('/Users/brentskoumal/Desktop/HPSI/HPSI_exercises')

# confirm working directory (you can also check inthe top right)
#print("\nchanged directory: ",os.getcwd())
#%%
'''
The Challenge

Suppose an investor has approached us and has asked us to build a machine learning model that can reliably predict if a loan will be paid off or not. In other words, he wants us to build a binary classifier. This investor described himself/herself as a conservative, who only wants to invest in loans that have a good chance of being paid off on time. Thus, this client is more interested in a machine learning model which does a good job of filtering out a high percentage of loan defaulters.

Can we build a machine learning model that can accurately predict if a borrower will pay off their loan on time or not?

Before we can start building a model, we need to define what features we want to use and which column repesents the target column we want to predict. Let's start by reading in the dataset and exploring it.

Raw data can be downloaded from 

 - https://query.data.world/s/GqUrSahIonBqjqupkDORXVkT2NyL7t 
 
A data dictionary is also provided and can be useful to help understand what a column represents in the dataset.

Download the Data Dictionary here:

 - https://query.data.world/s/hdF7Ur3Bfw4-UUI94sEae06XAVvpJG
'''

#%% Read-in the loans data from the following url into a Pandas DataFrame called "loans"
# https://query.data.world/s/6Vgh0LcibNhMVb-o_0eH97tLqaj6nI
# you will need to skip the first row


#%% Read-in loans data column explanations from the following url into another Pandas DataFrame called "loan_dict"
# https://query.data.world/s/hdF7Ur3Bfw4-UUI94sEae06XAVvpJG
# This data dictionary explains each of the 115 columns represents


#%% Drop columns from the loans DataFrame that have more than half of their values missing
# set a threshold, half of the length of the DataFrame, and apply that to the Pandas .dropna() method


#%% We have reduced our column count from 115 to 58, but we can do better
'''
Many of the remaining columns are redundant, or "leak data" from the future. Create a drop_list containing the columns listed below, and drop them from the loans DataFrame.

id, desc, url, member_id, funded_amnt, funded_amnt_inv, grade, sub_grade, emp_title, issue_d, zip_code, out_prncp, out_prncp_inv, total_pymnt, total_pymnt_inv, total_rec_prncp, total_rec_int, total_rec_late_fee, recoveries, collection_recovery_fee, last_pymnt_d, last_pymnt_amnt, acc_now_delinq, delinq_amnt, tax_liens

For a more detailed explanation of why we chose to drop these columns see ___
'''


#%% Remove columns that only have one unique value loans.apply(pd.Series.nunique) will return a list of the counts of unique values in each column of our loans DataFrame apply the result of this command as a Boolean Mask to subset the loans DataFrame 

# execute the following command to create a count of unique values in each column


# We want to filter the dataframe such that it includes all rows and only columns where (unique_count != 1)
# use .loc to make this selection, and re-assign loans


#%% Create a horizontal bar plot of loan_status' value_counts() to visualize/understand the class frequency breakdown of our target column

#%% Since we want to build a model that predicts whether or not a customer will default, we ultimately want a binary target vector. Let's subset the DataFrame even further by selecting only those rows corresponding to a loan_status of "Fully Paid" or "Charged Off"

# Construct a Boolean Mask to subset the loans DataFrame, we are only interested in loan_status=="Fully Paid" OR loan_status=="Charged Off"


#%% Create a mapping dictionary to Binarize our target vector, "Fully Paid" should be replace with "1" and "Charged Off" should be replaced with "0" 


#use .replace() to map the values in the loan_status column

#%% Re-visualize the distribution of loan_status with another horizontal bar plot, make sure we only have values of "1" or "0"

#%% Write the filtered_loans dataframe to a file called "filtered_loans.csv"
