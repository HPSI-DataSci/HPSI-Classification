#######################################################
########### Human Performance Systems, Inc. ###########
#######################################################

###############################
## Classification in Python  ##
## Day 1 -- Environment Test ##
###############################

# list of packages installed in conda virtual env
installed_packages = ['numpy', 'pandas', 'scipy', 'tensorflow', 'keras',
                      'graphviz', 'pydot', 'pydotplus', 'seaborn', 'sklearn',
                      'skimage', 'sympy']
# dynamically import all packages
packages = list(map(__import__, installed_packages))

# list all imported packages and version numbers
for i in range(len(packages)):
    try:
        print("{}={}".format(packages[i].__name__, packages[i].__version__))
    except:
        print("{}".format(packages[i].__name__))
print('-' * 54)
print("| Congrats! Your virtual environment is ready to go! |")
print('-' * 54)
