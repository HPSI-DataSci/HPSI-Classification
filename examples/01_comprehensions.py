#######################################################
########### Human Performance Systems, Inc. ###########
#######################################################

#####################################################
## Classification in Python                        ##
## Day 1 -- List/Dict Comprehensions               ##
#####################################################


## NOTE: To run individual code cells, press Shift + enter for PCs 
#==============================================================================
#%%

import numpy as np

nums = np.arange(1, 11, 1)

print("original list: \n", nums)
print("")

squares=[]

for num in nums:
    squares.append(num**2)

print("squared numbers: \n", squares)
#%%

# list comprehension does this in one line of code
comp_squares = [num ** 2 for num in nums]

print("comprehension squares: \n",comp_squares)

#%%

letters = ['a', 'b', 'c', 'd']
numbers = [1, 2, 3, 4]

# create dictionary with letters as keys and numbers as values
my_dict = dict(zip(letters, numbers))

print(my_dict)

#%%

# for loop to switch the keys with the values
inverse_dict = {}

for key, value in my_dict.items():
    inverse_dict.update({value: key})
    
print(inverse_dict)

#%%

# dictionary comprehension can do this in 1 line of code
dict_comp = {value: key for key, value in my_dict.items()}

print(dict_comp)
