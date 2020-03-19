import os
import configparser

# Get installed methods
def get_methods():
    os.path.join('..',recipes)
    files = os.listdir()
    methods = []
    for f in files:
        if '.cfg' in files:
            methods.append(f[0:-4])

    return methods

# List installed methods
def list_methods():
    methods = get_methods()
    for i in methods:
        print(i)

# Print out details of a method
def print_method(method):
    methods = get_methods()
    if method in methods:
        # Print out method configuration
        print(method + ' configuration')
        methods_path = os.path.join('..',recipes)
        method = method + '.cfg'
        f = open(os.path.join(methods_path,method))
        for line in f:
            print(line)

        # Print out method recipe
        config = configparser.ConfigParser()
        config.read(os.path.join(methods_path,method))
        f = open(os.path.join(methods_path,config['recipe']))
        for line in f:
            print(line)
