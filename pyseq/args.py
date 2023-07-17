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

# Flag to perform a diagnostic run
parser.add_argument('-gmail',
                    help='add gmail account and app key',
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

    if args['gmail'] is True:
        from cryptography.fernet import Fernet
        import base64
        from getpass import getpass
        import smtplib, ssl


        not_in = True; tries = 0
        while not_in:
            # Enter gmail account only USERNAME not @gmail.com
            loop = True
            while loop:
                username = input('Gmail account (skip @gmail.com): ')
                username = username.strip()
                loop = not methods.userYN(f'Confirm Gmail account is {username}')

            # Enter appkey that was created in Google Account Security Settings
            loop = True
            while loop:
                appkey = input('App key: ')
                loop = not methods.userYN(f'Confirm appkey is {appkey}')

            # Test appkey works
            try:
                port = 465
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
                    reponse = server.login(f'{username}@gmail.com', appkey)
                    not_in = False
            except:
                tries += 1
                if tries > 3:
                    not_in = False
                print(f'Username/appkey did not work, attempt {i}/3')
        if tries >= 3:
            raise ValueError('Double check username and app keys')

        # Enter PySeq Password, that is password to login to machine
        loop = True
        tries = 0
        while loop:
            print('Enter PySeq password')
            password = getpass()
            print('Re-enter PySeq password')
            password_ = getpass()
            MATCH = password == password_

            if not MATCH:
                print('Password did not match')
            else:
                loop = False


        encoded_pw = base64.b64encode(password.encode("utf-8"))
        salt = Fernet.generate_key()
        key = salt + encoded_pw
        f = Fernet(key)
        token = f.encrypt(appkey.encode('utf-8'))
        config, config_path = methods.get_config(config_type='machine_info')
        config.read_dict({'DEFAULT':{'token':token.decode('utf-8'),
                                     'salt':salt.decode('utf-8'),
                                     'username': username}})
        with open(config_path,'w') as f:
            config.write(f)
        sys.exit()

    return args
