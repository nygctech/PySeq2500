#!/usr/bin/python
"""Arguments for Pyseq

usage: pyseq [-h] [-config PATH] [-name NAME] [-output PATH] [-list]
             [-method METHOD] [-diagnostics]

optional arguments:
  -h, --help      show this help message and exit
  -config PATH    path to config file, default = ./config.cfg
  -name NAME      experiment name, default = timestap(YYYYMMDD_HHMMSS)
  -output PATH    directory to save data, default = current directory
  -list           list installed methods
  -method METHOD  print method details
  -diagnostics    diagnostic run

Kunal Pandit 3/15/2020
"""
import argparse
import sys
import os
from os.path import join
import time
from . import methods

# Create argument parser
parser = argparse.ArgumentParser(prog='pyseq')
# Optional Configuration Path
parser.add_argument('-config',
                    help='path to config file',
                    metavar = 'PATH',
                    default = join(os.getcwd(),'config.cfg'),
                    )
# Optional Experiment Name
parser.add_argument('-name',
                    help='experiment name',
                    default= time.strftime('%Y%m%d_%H%M%S'),
                    )
# Optional Output Path
parser.add_argument('-output',
                    help='directory to save data',
                    metavar = 'PATH',
                    default = os.getcwd()
                    )
# Flag to print out installed methods
parser.add_argument('-list',
                    help='list installed methods',
                    action = 'store_true'
                    #nargs = 0,
                    )
# Flag to print out installed methods
parser.add_argument('-method',
                    help='print method details',
                    choices = methods.get_methods(),
                    metavar = 'METHOD'
                    )

# Flag to use virtual HiSeq
parser.add_argument('-virtual',
                    help='use virtual HiSeq',
                    action = 'store_true',
                    )

# Flag to print HiSeq settings
parser.add_argument('-settings',
                    help='print optional HiSeq settings',
                    action = 'store_true',
                    )

# Flag to perform a diagnostic run
parser.add_argument('-ports',
                    help='view com ports',
                    action = 'store_true',
                    )

# Flag to perform a diagnostic run
parser.add_argument('-diagnostics',
                    help='perform a diagnostics run',
                    action = 'store_true',
                    )

def get_arguments():
    """Return arguments from command line"""

    args = parser.parse_args()
    args = vars(args)

    if args['list'] is True:
        methods.list_methods()
        sys.exit()

    if args['method'] in methods.get_methods():
        methods.print_method(args['method'])
        sys.exit()

    if args['settings']:
        settings = methods.get_settings()
        for s in settings:
            print(s, '=', settings[s])
            print()
        sys.exit()

    if args['diagnostics']:
        from . import check_system
        sys.exit()

    if args['ports'] is True:
        methods.list_com_ports()

        sys.exit()

    return args
