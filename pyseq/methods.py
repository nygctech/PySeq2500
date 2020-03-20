import configparser

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from . import recipes

# Return method config and recipe
def return_method(method):
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

# Get installed methods
def get_methods():
    contents = pkg_resources.contents(recipes)
    methods = []
    for i in contents:
        if '.cfg' in i:
            methods.append(i[0:-4])
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
