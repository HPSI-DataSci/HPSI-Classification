# HPSI-Classification

This repository contains all the course material for the classification course 

## Steps to Create the Conda Environment

This guide will walk you through the steps to set up the virtual environment for the course. 

The first step is to clone the repo either by clicking the green button in the top-right corner labeled `Clone or download` and selecting the `Download ZIP` option or by running `git clone https://github.com/HPSI-DataSci/HPSI-Classification.git` in your terminal. If you chose to download a ZIP file, then be sure to extract the contents by unzipping it. Next, move the directory for the repository to your desktop. 

*****To get Started******

### Windows Instructions

Open up a terminal (Command Prompt/Anaconda Prompt/Git Bash/Powershell/etc.). Navigate to the directory for the repository: 
`
cd C:\Users\User\Desktop\HPSI-Classification-Master
`
Note that you may have saved the directory with a different name, so you may have to change the last name in the path. Then, run the following command in your terminal to create the virtual environment: 
```commandline
conda env create -f lmi_env.yml
```

When the environment has been created, you should see something like: 
```
done
#
# To activate this environment, use:
# > activate HPSI-Classification
#
# To deactivate an active environment, use:
# > deactivate
#
```

Depending on your version of `conda`, this prompt could say `activate` or `conda activate`. Run this command 
in the terminal to activate the virtual environment. Once the virtual environment has been activated, we need to 
run the test script to ensure all the packages we need for the course have been installed and configured successfully. 
You can run the script by entering the following command in your terminal: 
```commandline
python import_test.py
```
You should see the following output from running the script: 
```
Using TensorFlow backend.
[OK] numpy 1.12.1 is installed
[OK] pandas 0.22.0 is installed
[OK] scipy 1.0.0 is installed
[OK] tensorflow 1.1.0 is installed
[OK] keras 2.1.2 is installed
[OK] graphviz 0.5.2 is installed
[OK] pydot 1.2.3 is installed
[OK] pydotplus 0.0 is installed
[OK] seaborn 0.8.1 is installed
[OK] sklearn 0.19.1 is installed
[OK] skimage 0.13.1 is installed
[OK] sympy 1.1.1 is installed
[SUCCESS] Your virtual environment is ready to go!
```

If your environment is set up and running, you can deactivate it with either `source deactivate` or 
`conda deactivate` (the matching command for whichever `activate` command you used before).

### Mac Instructions

Open up a terminal (Command Prompt/Anaconda Prompt/Git Bash/Powershell/etc.). Navigate to the directory for the repository: 
`
cd ~/Users/$USERNAME/Desktop/HPSI-Classification
`
where `$USERNAME` is the username for your user profile.
Then, run the following command in your terminal to create the virtual environment: 
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
[OK] numpy 1.12.1 is installed
[OK] pandas 0.22.0 is installed
[OK] scipy 1.0.0 is installed
[OK] tensorflow 1.1.0 is installed
[OK] keras 2.1.2 is installed
[OK] graphviz 0.5.2 is installed
[OK] pydot 1.2.3 is installed
[OK] pydotplus 0.0 is installed
[OK] seaborn 0.8.1 is installed
[OK] sklearn 0.19.1 is installed
[OK] skimage 0.13.1 is installed
[OK] sympy 1.1.1 is installed
[SUCCESS] Your virtual environment is ready to go!
```

If your environment is set up and running, you can deactivate it with either `source deactivate` or 
`conda deactivate` (the matching command for whichever `activate` command you used before).
