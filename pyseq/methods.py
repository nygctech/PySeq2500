#!/usr/bin/python
'''Get details about installed methods

Kunal Pandit 3/15/2020
'''

import configparser
from os.path import expanduser, join, isfile, isdir
from os import mkdir
import time

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from . import recipes
from . import resources


def return_method(method):
    '''Return the paths to the method configuration and recipe.

       Parameters:
       method (str): Name of the method.

       Returns.
       path, path: Path to method configuration, and path to the recipe
    '''
    config_path = None
    recipe_path = None
    contents = pkg_resources.contents(recipes)
    for i in contents:
        if '.cfg' in i and method == i[0:-4]:
            with pkg_resources.path(recipes, method+'.cfg') as path:
                config_path = str(path)
            config = configparser.ConfigParser()
            config.read(config_path)
            recipe = config[method]['recipe']
            with pkg_resources.path(recipes, recipe) as path:
                recipe_path = str(path)

    return config_path, recipe_path


def get_methods():
    '''Return list of installed methods'''

    contents = pkg_resources.contents(recipes)
    methods = []
    for i in contents:
        if '.cfg' in i:
            methods.append(i[0:-4])
    return methods


def list_methods():
    '''Print list of methods.'''

    methods = get_methods()
    methods.remove('settings')
    for i in methods:
        print(i)


def print_method(method):
    '''Print configuration and recipe of method'''

    methods = get_methods()
    if method in methods:
        # Print out method configuration
        print()
        print(method + ' configuration:')
        print()
        f = pkg_resources.open_text(recipes, method+'.cfg')
        for line in f:
            print(line[0:-1])

        # Print out method recipe
        print()
        print(method + ' recipe:')
        print()
        config = configparser.ConfigParser()
        with pkg_resources.path(recipes, method+'.cfg') as config_path:
            config.read(config_path)
        recipe = config[method]['recipe']
        with pkg_resources.path(recipes, recipe) as recipe_path:
            f = open(recipe_path)
        for line in f:
            print(line[0:-1])



def list_settings(instrument = 'HiSeq2500'):
    """Print all possible setting keywords and descriptions."""

    settings = configparser.ConfigParser()
    with pkg_resources.path(resources, 'settings.cfg') as settings_path:
        settings.read(settings_path)

    settings = settings[instrument]
    for s in settings:
        print(s,':', settings[s])
        print()



def get_settings(instrument = 'HiSeq2500'):
    """Return all possible setting keywords."""

    settings = configparser.ConfigParser()
    with pkg_resources.path(resources, 'settings.cfg') as settings_path:
        settings.read(settings_path)

    settings = settings[instrument]

    return settings



def check_settings(input_settings, instrument = 'HiSeq2500'):
    """Check setting keywords in config file are valid."""

    settings = get_settings(instrument)

    all_clear = True
    for s in input_settings:
        if s not in [*settings.keys()]:
            print(s, 'is not a valid setting')
            all_clear = False

    return all_clear



def list_com_ports(instrument = 'HiSeq2500'):
    """Print and return COM Ports of instruments."""

    com_ports = configparser.ConfigParser()
    with pkg_resources.path(resources, 'com_ports.cfg') as config_path:
        com_ports.read(config_path)

    com_ports = com_ports[instrument]
    for port_name in com_ports:
        print(port_name,':', com_ports[port_name])

    return com_ports



def assign_com_ports(instrument = False, machine = 'HiSeq2500'):
    """Change COM ports of instruments."""

    com_ports = list_com_ports()

    if userYN('Assign new COM ports?'):
        keep_assigning = True
    else:
        keep_assigning = False


    while keep_assigning:
        if not instrument:
            instrument = input('Which instrument? ')

        if instrument in com_ports:
            print('Old port =', com_ports[instrument])
            port = input('New port? ').strip()
            if userYN('Confirm new port for', instrument, 'is', port):
                com_ports[instrument] = port
        else:
            instrument = False
            print('Instrument is not valid. Enter one of the following:')
            print(com_ports.options('HiSeq2500'))

        if not userYN('Assign another new COM port?'):
            keep_assigning = False

            # Read default COM ports
            default_com_ports = configparser.ConfigParser()
            with pkg_resources.path(resources, 'com_ports.cfg') as config_path:
                default_com_ports.read(config_path)

            # Overide default COM port
            updated_com_ports = {machine:com_ports}
            default_com_ports.read_dict(updated_com_ports)

            # write new config file
            with pkg_resources.path(resources, 'com_ports.cfg') as config_path:
                f = open(config_path,mode='w')
                default_com_ports.write(f)
        else:
            instrument = False



def get_machine_info(virtual=False):
    """Specify machine model and name."""

    # Open machine_info.cfg save in USERHOME/PySeq2500
    homedir = expanduser('~')
    if not isdir(join(homedir,'PySeq2500')):
        mkdir(join(homedir,'PySeq2500'))

    config_path = join(homedir,'PySeq2500','machine_info.cfg')
    config = configparser.ConfigParser()
    NAME_EXISTS = isfile(config_path)
    if NAME_EXISTS:
        with open(config_path,'r') as f:
            config.read_file(f)
        model = config['DEFAULT']['model']
        name = config['DEFAULT']['name']
    else:
        model = None
        name = None


    # Get machine model from user
    while model is None:
        if userYN('Is this a HiSeq2500'):
            model = 'HiSeq2500'
            if model not in ['HiSeq2500']:
                model = None

    # Get machine name from user
    while name is None and not virtual:
        name = input('Name of '+model+' = ')
        if not userYN('Name this '+model+' '+name):
            name = None

    if virtual:
        name = 'virtual'


    # Check if background and registration data exists
    # Open machine_settings.cfg saved in USERHOME/PySeq2500
    machine_settings = configparser.ConfigParser()
    ms_path = join(homedir,'PySeq2500','machine_settings.cfg')
    if isfile(ms_path):
        with open(ms_path,'r') as f:
            machine_settings.read_file(f)

    if not machine_settings.has_section(name+'background'):
        if not userYN('Continue experiment without background data for',name):
            model = None
    if not machine_settings.has_section(name+'registration') and model is not None:
        if not userYN('Continue experiment without registration data for',name):
            model = None

    if not NAME_EXISTS and model is not None and name is not None:
        # Save machine info
        config.read_dict({'DEFAULT':{'model':model,'name':name}})
        with open(config_path,'w') as f:
            config.write(f)
        #Add to list in machine settings
        if not machine_settings.has_section('machines'):
            machine_settings.add_section('machines')
        machine_settings.set('machines', name, time.strftime('%m %d %y'))
        with open(ms_path,'w') as f:
            machine_settings.write(f)

    return model, name



def userYN(*args):
    """Ask a user a Yes/No question and return True if Yes, False if No."""

    question = ''
    for a in args:
        question += str(a) + ' '

    response = True
    while response:
        answer = input(question + '? Y/N = ')
        answer = answer.upper().strip()
        if answer in ['Y','YES','TRUE']:
            response = False
            answer = True
        elif answer in ['N', 'NO','FALSE']:
            response = False
            answer = False

    return answer
