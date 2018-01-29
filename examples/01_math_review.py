#######################################################
########### Human Performance Systems, Inc. ###########
#######################################################

##############################
## Classification in Python ##
## Day 1 -- Feature Scaling ##
##############################


## NOTE: To run individual code cells, press Shift + enter for PCs & Macs
# ========================================================================

import numpy as np

#%% Scalar Addition

a = 2
b = 4
a + b

#%% Vector Basics

# vector is 1-D array
a = np.arange(4)
print(a)

# check number of dimensions
print(a.ndim)

# check the number of rows and columns
print(a.shape)

#%% Matrix Basics

# matrix is 2-D array
a = np.arange(4).reshape(2,2)
print(a)

# check number of dimensions
print(a.ndim)

# check the number of rows and columns
print(a.shape)

#%% Tensor Basics

# a tensor is a higher dimensional array
print(a.reshape(1, 2, 2))

print(a.reshape(1, 2, 2).shape)

print(a.reshape(1, 2, 2).ndim)

#%% Transposing Vectors

a = np.arange(5)
print(a)

print(a.shape)

# method to transpose
print(a.T)

print(a.T.shape)

# function to transpose
print(np.transpose(a))
print(np.transpose(a).shape)

# transposing vectors continued
print(a.reshape(-1,1))

print(a.reshape(-1,1).shape)

print(a.reshape(-1,1).T)

print(a.reshape(-1,1).T.shape)

#%% Transposing Matrices

b = np.arange(9).reshape(3,3)
print(b)

print(b.T)

print(b.transpose())

#%% Vector Addition

a = np.arange(6)
b = a + 4

print(a)
print(b)

print(a + b)

#%% Matrix Addition

a = np.arange(4).reshape(2,2)
b = a + 4

print(a)
print(b)

print(a + b)

#%% Vector Dot-Product

a = np.arange(6)
b = a + 4

print(a)
print(b)

# element-wise multiplication
print(a*b)

# dot product
print(a.dot(b))

#%% Vector Norm

a = np.arange(6)
print(a)

# euclidean norm
print(np.linalg.norm(a))

# taxicab norm
print(np.linalg.norm(a, ord=1))

#%% Matrix Multiplication

a = np.arange(4).reshape(2,2)
b = a + 4

print(a)
print(b)

# element-wise multiplication
print(a*b)

# matrix multiplication
print(a.dot(b))
print(a@b)
print(np.matmul(a,b))
