#!/usr/bin/python
'''Get details about installed methods

Kunal Pandit 3/15/2020
'''

import configparser

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from . import recipes


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
    settings = configparser.ConfigParser()
    with pkg_resources.path(recipes, 'settings.cfg') as settings_path:
        settings.read(settings_path)

    settings = settings[instrument]
    for s in settings:
        print(s,':', settings[s])
        print()

def get_settings(instrument = 'HiSeq2500'):
    settings = configparser.ConfigParser()
    with pkg_resources.path(recipes, 'settings.cfg') as settings_path:
        settings.read(settings_path)

    settings = settings[instrument]

    return settings

def check_settings(input_settings, instrument = 'HiSeq2500'):
    settings = configparser.ConfigParser()
    with pkg_resources.path(recipes, 'settings.cfg') as settings_path:
        settings.read(settings_path)

    all_clear = True
    for s in input_settings:
        if s not in [*settings[instrument].keys()]:
            print(s, 'is not a valid setting')
            all_clear = False

    return all_clear
