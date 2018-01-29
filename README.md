# HPSI-Classification

This repository contains all the course material for the classification course 

## Steps to Create the Conda Environment

This guide will walk you through the steps to set up the virtual environment for the course. 

### Windows Instructions

Open up a terminal (Command Prompt/Anaconda Prompt/Git Bash/Powershell/etc.) in the same directory as 
the course materials. Then, run the following command in your terminal to create the virtual environment: 
```commandline
conda env create -f windows_environment.yml
```

When the environment has been created, you should see something like: 
```
done
#
# To activate this environment, use:
# > source activate HPSI-Classification
#
# To deactivate an active environment, use:
# > source deactivate
#
```

Depending on your version of `conda`, this prompt could say `source activate` or `conda activate`. Run this command 
in the terminal to activate the virtual environment. Once the virtual environment has been activated, we need to 
run the test script to ensure all the packages we need for the course have been installed and configured successfully. 
You can run the script by entering the following command in your terminal: 
```commandline
python import_test.py
```
You should see the following output from running the script: 
```
Using TensorFlow backend.
numpy=1.12.1
pandas=0.22.0
scipy=1.0.0
tensorflow=1.1.0
keras=2.1.2
graphviz=0.5.2
pydot=1.2.3
pydotplus
seaborn=0.8.1
sklearn=0.19.1
skimage=0.13.1
sympy=1.1.1
------------------------------------------------------
| Congrats! Your virtual environment is ready to go! |
------------------------------------------------------
```

If your environment is set up and running, you can deactivate it with either `source deactivate` or 
`conda deactivate` (the matching command for whichever `activate` command you used before).

### Mac Instructions

Open up a terminal (Terminal/Anaconda Prompt/iTerm 2/etc.) in the same directory as 
the course materials. Then, run the following command in your terminal to create the virtual environment: 
```commandline
conda env create -f mac_environment.yml
```

When the environment has been created, you should see something like: 
```
done
#
# To activate this environment, use:
# > source activate HPSI-Classification
#
# To deactivate an active environment, use:
# > source deactivate
#
```

Depending on your version of `conda`, this prompt could say `source activate` or `conda activate`. Run this command 
in the terminal to activate the virtual environment. Once the virtual environment has been activated, we need to 
run the test script to ensure all the packages we need for the course have been installed and configured successfully. 
You can run the script by entering the following command in your terminal: 
```commandline
python import_test.py
```
You should see the following output from running the script: 
```
Using TensorFlow backend.
numpy=1.12.1
pandas=0.22.0
scipy=1.0.0
tensorflow=1.1.0
keras=2.1.2
graphviz=0.5.2
pydot=1.2.3
pydotplus
seaborn=0.8.1
sklearn=0.19.1
skimage=0.13.1
sympy=1.1.1
------------------------------------------------------
| Congrats! Your virtual environment is ready to go! |
------------------------------------------------------
```

If your environment is set up and running, you can deactivate it with either `source deactivate` or 
`conda deactivate` (the matching command for whichever `activate` command you used before).
