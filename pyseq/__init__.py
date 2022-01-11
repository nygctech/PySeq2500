import logging
import configparser
import time
from os.path import expanduser, join, isfile, isdir
import os

from .methods import userYN

# Functions common to all HiSeq models

def get_instrument(virtual=False, logger=None):

    # Get instrument model and name
    model, name = get_machine_info(virtual)

    # Create HiSeq Object
    if model == 'HiSeq2500' and name is not None:
        if virtual:
            from . import virtualHiSeq
            hs = virtualHiSeq.HiSeq2500(name, logger)
        else:
            from . import hiseq2500
            hs = hiseq2500.HiSeq2500(name, logger)
    else:
        hs = None

    return hs



def get_machine_info(virtual=False):
    """Specify machine model and name."""

    # Open machine_info.cfg save in USERHOME/.pyseq2500
    homedir = expanduser('~')
    if not isdir(join(homedir,'.pyseq2500')):
        mkdir(join(homedir,'.pyseq2500'))

    config_path = join(homedir,'.pyseq2500','machine_info.cfg')
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
    # Open machine_settings.cfg saved in USERHOME/.pyseq2500
    machine_settings = configparser.ConfigParser()
    ms_path = join(homedir,'.pyseq2500','machine_settings.cfg')
    if isfile(ms_path):
        with open(ms_path,'r') as f:
            machine_settings.read_file(f)

    if not machine_settings.has_section(name+'background'):
        if not userYN('Continue experiment without background data for',name):
            model = None
    # if not machine_settings.has_section(name+'registration') and model is not None:
    #     if not userYN('Continue experiment without registration data for',name):
    #         model = None

    if not NAME_EXISTS and model is not None and name not in [None,'virtual']:
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



def setup_logger(log_name=None, log_path = None, config=None):
    """Create a logger and return the handle."""


    if config is not None:
        log_path = config.get('experiment','log_path')
    elif log_path is None:
        log_path = os.getcwd()

    if config is not None:
        log_name = config.get('experiment','experiment name')
    elif log_name is None:
        log_name = time.strftime('%Y%m%d_%H%M%S')

    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(10)

    # Create console handler
    c_handler = logging.StreamHandler()
    c_handler.setLevel(21)
    # Create file handler
    f_log_name = join(log_path,log_name + '.log')
    f_handler = logging.FileHandler(f_log_name)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(asctime)s - %(message)s', datefmt = '%Y-%m-%d %H:%M')
    f_format = logging.Formatter('%(asctime)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger
