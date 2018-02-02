# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Probability -------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Q-1: Import packages that you can generate random numbers.
import random
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------

# Q-2: a) Generate 100 samples that has uniform distribution. Plot your results and verify it.

# 100 samples from uniform distribution
uniform_dist = np.random.uniform(1, 10, 100)
# plot
plt.figure(1)
plt.hist(uniform_dist, histtype='bar')
plt.title('100 Samples from the Uniform Distribution')
plt.show()

# ----------------------------------------------------------------------------------------------------------------------

# Q-2: b)Generate 1000 samples that has normal distribution (mean zero and std 1). Plot your results and verify it.

# 100 samples from standard normal distribution
normal_dist = np.random.normal(0,1, 1000)
# plot
plt.figure(2)
plt.hist(normal_dist, histtype='bar')
plt.title('100 Samples from the Standard Normal Distribution')
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Q-3: a)Write a list of numbers and convert it to array.
a = [i for i in range(10)]
type(a)
b = np.array(a)
type(b)

# ----------------------------------------------------------------------------------------------------------------------
# Q-3: b) Create 1x3, 3x3 and 5x5 matrix. Use different methods that you know. It can be random or manually enter numbers.
a = np.array([1,2,3])
b = np.arange(9).reshape(3,3)
c = np.arange(25).reshape(5,5)

# ----------------------------------------------------------------------------------------------------------------------
# Q-3: c) Multiply each one of those matrix by each other element wise and matrix multiply. Check your results by hand.
#         just the first two matrix.

# multiply element-wise
np.multiply(a,b)
# matrix multiplication
a.dot(b)

# ----------------------------------------------------------------------------------------------------------------------
# Q-4: a) Create a matrix 3x3. Add the first and second column then sum the the results.

a = np.arange(9).reshape(3,3)
sum(a[:,0] * a[:,1])

# ----------------------------------------------------------------------------------------------------------------------
# Q-4: b) Create a vector of 100 samples (any method you want). Plot the vector. Calculate the mean and std.

a = np.array(np.random.normal(0, 1, 100))
a.mean()
a.std()
plt.figure(3)
plt.hist(a)
plt.axvline(a.mean(), color='black', linestyle='--', label='mean')
plt.axvline(a.mean()+a.std(), color='red', linestyle='--', label='1 Standard Dev')
plt.axvline(a.mean()-a.std(), color='red', linestyle='--')
plt.axvline(a.mean()+2*a.std(), color='purple', linestyle='--', label='1 Standard Dev')
plt.axvline(a.mean()-2*a.std(), color='purple', linestyle='--')
plt.legend()
plt.title('Distribution of 100 samples drawn from Standard Normal Distribution')
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Q-4: c) Create a matrix 3x3 and 3x1. Multiply the first matrix by the second matrix. Define a logsig transfer function.
#         pass the results of the the multiplication to the transfer function. Check does it make sense. Explain your results

a = np.arange(9).reshape(3,3)
b = np.arange(3).reshape(3,1)
n = np.dot(a, b)

def logsig(n):
    return 1 / (1 + np.exp(-n))

results = logsig(n)

vec = np.linspace(-20, 20, 100)
check = logsig(vec)
plt.figure(4)
plt.plot(vec, check)
plt.axhline(0.5, color='black', linestyle='--')
plt.title('Logistic Sigmoid (logsig) Transfer Function')
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Q-5: a) Calculate the expanded form of (x+y)^6.
from sympy import *
x = Symbol('x')
y = Symbol('y')
expand((x+y) ** 6)

# ----------------------------------------------------------------------------------------------------------------------
# Q-5: b) Simplify the trigonometric expression sin(x) / cos(x)

simplify(sin(x)/cos(x))

# ----------------------------------------------------------------------------------------------------------------------
# Q-5: c) Calculate the derivative of log(x) for x

diff(log(x),x)

# ----------------------------------------------------------------------------------------------------------------------
# Q-5: d) Solve the system of equations x + y = 2, 2x + y = 0

solve([x + y - 2, 2*x + y], [x, y])

# ----------------------------------------------------------------------------------------------------------------------
# Q-6:  Estimating Pi using the Monte Carlo Method
#      1- To estimate a value of Pi using the Monte Carlo method - generate a large number of random points and see
#         how many fall in the circle enclosed by the unit square.
#      2- Check the following link for instruction
#      3- There are variety of codes available in the net please write your own code.
from math import sqrt

# counter for number of points in circle
num_circle = 0
# sample size
n = 1000000
for i in range(n):
    # randomly generate x between 0 and 1
    x = random.random()
    # randomly generate x between 0 and 1
    y = random.random()
    # check if the point is in the circle
    if sqrt(x ** 2 + y **2) <= 1:
        # add to counter
        num_circle += 1

# ratio of points in circle to total multiplied by 4 quadrants
pi = 4 * num_circle/n
print('The value of pi is approximately {0}'.format(pi))