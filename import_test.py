#######################################################
########### Human Performance Systems, Inc. ###########
#######################################################

###############################
## Classification in Python  ##
## Day 1 -- Environment Test ##
###############################

def test_packages(pkgs):
    versions = []
    for p in pkgs:
        try:
            imported = __import__(p)
            try:
                versions.append(imported.__version__)
            except AttributeError:
                try:
                    versions.append(imported.version)
                except AttributeError:
                    try:
                        versions.append(imported.version_info)
                    except AttributeError:
                        versions.append('0.0')
        except ImportError:
            versions.append('Fail')
    return versions

packages = ['numpy', 'pandas', 'scipy', 'tensorflow', 'keras',
            'graphviz', 'pydot', 'pydotplus', 'seaborn', 'sklearn',
            'skimage', 'sympy']
versions = test_packages(packages)

for p, v in zip(packages, versions):
    success = True
    if v is not 'Fail':
        print("[OK] {} {} is installed".format(p, v))
    else:
        print('[FAIL]: {} is not installed'.format(p))
        success = False
if success:
    print('[SUCCESS] Your virtual environment is ready to go!')
else:
    print('[FAILURE] Please install the missing required packages')