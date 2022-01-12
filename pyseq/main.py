"""
TODO:

"""

import logging
import time
import os
from os.path import join
import sys
import configparser
import threading
import argparse
import pyseq


from . methods import *
from . import args
from . import focus
from . import flowcell

                                                                                # Global int to track # of errors during start up
def error(*args):
    """Keep count of errors and print to logger and/or console."""

    global n_errors

    i = 0
    if isinstance(args[0], logging.Logger):
        logger = args[0]
        i = 1

    msg = 'ERROR::'
    for a in args[i:]:
        msg = msg + str(a) + ' '
    if i is 0:
        print(msg)
    else:
        logger.log(21, msg)

    n_errors += 1

    return n_errors



def setup_flowcells(first_line, IMAG_counter):
    """Read configuration file and create flowcells.

       **Parameters:**
       - first_line (int): Line number for the recipe to start from on the
         initial cycle.

       **Returns:**
       - dict: Dictionary of flowcell position keys with flowcell object values.

    """
    err_msg = 'ConfigFile::sections::'
    experiment = config['experiment']
    method = experiment['method']
    method = config[method]

    flowcells = {}
    for sect_name in config['sections']:
        #f_sect_name = sect_name.replace('_','')                                 #remove underscores
        position = config['sections'][sect_name]
        AorB, coord  = position.split(':')
        AorB = AorB.strip()
        # Create flowcell if it doesn't exist
        if AorB not in flowcells.keys():
            fc = flowcell.Flowcell(AorB)
            fc.recipe_path = experiment['recipe path']
            fc.first_line = first_line
            fc.volume['main'] = int(method.get('main prime volume', fallback=500))
            fc.volume['side'] = int(method.get('side prime volume', fallback=350))
            fc.volume['sample'] = int(method.get('sample prime volume', fallback=250))
            fc.volume['flush'] = int(method.get('flush volume', fallback=1000))
            fs = int(method.get('flush flowrate',fallback=700))
            fc.pump_speed['flush'] = fs
            ps = int(method.get('prime flowrate',fallback=100))
            fc.pump_speed['prime'] = ps
            rs = int(method.get('reagent flowrate', fallback=40))
            fc.pump_speed['reagent'] = rs
            fc.total_cycles = int(config.get('experiment','cycles'))
            fc.temp_interval = float(method.get('temperature interval', fallback=5))*60
            z_planes = int(method.get('z planes', fallback=0))
            if z_planes > 0:
                fc.z_planes = z_planes
            if IMAG_counter > 1:
                fc.IMAG_counter = 0
            fc.prerecipe_path = method.get('pre recipe', fallback = None)
            flowcells[AorB] = fc


    for sect_name in config['sections']:
        position = config['sections'][sect_name]
        f_sect_name = sect_name.replace('_','')                                 #remove underscores
        AorB, coord  = position.split(':')
        err_msgs = flowcells[AorB].addSection(f_sect_name, coord)
        for err in err_msgs:
            error('ConfigFile::'+err)


    # if runnning multiple flowcells...
    # Define first flowcell
    # Define prior flowcell signals to next flowcell
    if len(flowcells) > 1:
        flowcell_list = [*flowcells]
        for fc in flowcells.keys():
            flowcells[fc].waits_for = flowcell_list[flowcell_list.index(fc)-1]
        if experiment['first flowcell'] not in flowcells:
            error('ConfigFile::First flowcell does not exist')
        if isinstance(IMAG_counter, int):
            error('Recipe::Need WAIT before IMAG with 2 flowcells.')


    return flowcells


##########################################################
## Parse lines from recipe ###############################
##########################################################
def parse_line(line):
    """Parse line and return event (str) and command (str).

       If line starts with the comment character, #, then None is return for
       both event and command.
    """


    comment_character = '#'
    #delimiter = '\t'
    no_comment = line.split(comment_character)[0]                               # remove comment
    sections = no_comment.split(':')
    if len(sections) == 2:
        event = sections[0].strip()                                             # first section is event
        event = event[0:4]                                                      # event identified by first 4 characters
        command = sections[1]                                                   # second section is command
        command = command.strip()                                               # remove space
    else:
        event = None
        command = None

    return event, command


def configure_instrument(IMAG_counter, port_dict, flowcells):
    """Configure and check HiSeq settings."""

    global n_errors

    hs = pyseq.get_instrument(args_['virtual'], logger)

    if hs is not None:
        config['experiment']['machine'] = hs.model+'::'+hs.name
    else:
        sys.exit()

    experiment = config['experiment']
    method = experiment['method']
    method = config[method]

    try:
        total_cycles = int(experiment.get('cycles'))
    except:
        error('ConfigFile:: Cycles not specified')

    # Check side ports
    try:
        side_ports = method.get('side ports', fallback = '9,21,22,23,24')
        side_ports = side_ports.split(',')
        side_ports = list(map(int, side_ports))
    except:
        error('ConfigFile:: Side ports not valid')
    # Check sample port
    try:
        sample_port = int(method.get('sample port', fallback = 20))
    except:
        error('ConfigFile:: Sample port not valid')
    # Check barrels per lane make sense:
    n_barrels = int(method.get('barrels per lane', fallback = 1))               # Get method specific pump barrels per lane, fallback to 1
    if n_barrels not in [1,2,4,8]:
        error('ConfigFile:: Barrels per lane must be 1, 2, 4 or 8')
    # Check inlet ports, note switch inlet ports in initialize_hs
    inlet_ports = int(method.get('inlet ports', fallback = 2))
    if inlet_ports not in [2,8]:
        error('ConfigFile:: inlet ports must be 2 or 8.')
    # Check rinse port
    rinse_port = method.get('rinse', fallback = None)
    if rinse_port not in port_dict.keys():
        rinse_port = None

    variable_ports = method.get('variable reagents', fallback = None)
    hs.z.image_step = int(method.get('z position', fallback = 21500))
    hs.overlap = abs(int(method.get('overlap', fallback = 0)))
    hs.overlap_dir = method.get('overlap direction', fallback = 'left').lower()
    if hs.overlap_dir not in ['left', 'right']:
        error('ConfigFile:: overlap direction must be left or right')

    # Add flowcells
    for fc in flowcells.values():
        AorB = fc.position
        hs.flowcells[AorB] = flowcells[AorB]
        hs.v24[AorB].side_ports = side_ports
        hs.v24[AorB].sample_port = sample_port
        hs.v24[AorB].port_dict = port_dict                                      # Assign ports on HiSeq
        hs.v24[AorB].rinse_port = rinse_port                                    # Assign port for rinsing out lines and flowcells
        if variable_ports is not None:
            v_ports = variable_ports.split(',')
            for v in v_ports:                                                   # Assign variable ports
                hs.v24[AorB].variable_ports.append(v.strip())
        hs.p[AorB].update_limits(n_barrels)                                     # Assign barrels per lane to pump
        for section in fc.sections:                                             # Convert coordinate sections on flowcell to stage info
            pos = hs.position(AorB, fc.sections[section])
            fc.stage[section] = pos
            fc.stage[section]['z_pos'] = [hs.z.image_step]*3
    # Remove unused flowcell slots
    if hs.flowcells['A'] is None: hs.flowcells.pop('A')
    if hs.flowcells['B'] is None: hs.flowcells.pop('B')

    ## TODO: Changing laser color unecessary for now, revist if upgrading HiSeq
    # Configure laser color & filters
    # colors = [method.get('laser color 1', fallback = 'green'),
    #           method.get('laser color 2', fallback = 'red')]
    # for i, color in enumerate(default_colors):
    #     if color is not colors[i]:
    #         laser = hs.lasers.pop(color)                                        # Remove default laser color
    #         hs.lasers[colors[i]] = laser                                        # Add new laser
    #         hs.lasers[colors[i]].color = colors[i]                              # Update laser color
    #         hs.optics.colors[i] = colors[i]                                     # Update laser line color

    # Check laser power
    for color in hs.lasers.keys():
        lp = int(method.get(color+' laser power', fallback = 10))
        if hs.lasers[color].min_power <= lp <= hs.lasers[color].max_power:
            hs.lasers[color].set_point = lp
        else:
            error('ConfigFile:: Invalid '+color+' laser power')

    #Check filters for laser at each cycle are valid
    hs.optics.cycle_dict = check_filters(hs.optics.cycle_dict, hs.optics.ex_dict)
    focus_filters = [method.get('green focus filter', fallback = 2.0),
                     method.get('red focus filter', fallback = 2.4)]
    for i, f in enumerate(focus_filters):
        try:
            f = float(f)
        except:
            pass
        if f not in hs.optics.ex_dict[hs.optics.colors[i]]:
            error('ConfigFile:: Focus filter not valid.')
        else:
            hs.optics.focus_filters[i] = f

    # Check Autofocus Settings
    hs.AF = method.get('autofocus', fallback = 'partial once')
    if hs.AF.lower() in ['','none']: hs.AF = None
    if hs.AF not in ['partial', 'partial once', 'full', 'full once', 'manual', None]:
        # Skip autofocus and set objective position in config file
        try:
            if hs.obj.min_z <= int(hs.AF) <= hs.obj.max_z:
                hs.AF = int(hs.AF)
        except:
            error('ConfigFile:: Auto focus method not valid.')
    #Enable/Disable z stage
    hs.z.active = method.getboolean('enable z stage', fallback = True)
    # Get focus Tolerance
    hs.focus_tol = float(method.get('focus tolerance', fallback = 0))
    # Get focus range
    range = float(method.get('focus range', fallback = 90))
    spacing = float(method.get('focus spacing', fallback = 4.1))
    hs.obj.update_focus_limits(range=range, spacing=spacing)                    # estimate, get actual value in hs.obj_stack()
    hs.stack_split = float(method.get('stack split', fallback = 2/3))
    hs.bundle_height = int(method.get('bundle height', fallback = 128))

    hs.image_path = experiment['image path']
    with open(join(hs.image_path,'machine_name.txt'),'w') as file:
        file.write(hs.name)
    hs.log_path = experiment['log path']

    return hs

def make_directories():

    experiment = config['experiment']
    method = experiment['method']
    method = config[method]

    # Assign output directory
    save_path = experiment['save path']
    experiment_name = experiment['experiment name']
    save_path = join(experiment['save path'], experiment['experiment name'])
    if not os.path.exists(save_path):
        try:
            os.mkdir(save_path)
        except:
            print('ConfigFile:: Save path not valid.')

    # Make image directory
    image_path = experiment['image path']
    if not os.path.exists(image_path):
        os.mkdir(image_path)

    # Make log directory
    log_path = experiment['log path']
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    return log_path

def confirm_settings(recipe_z_planes = []):
    """Have user confirm the HiSeq settings before experiment."""

    experiment = config['experiment']
    method = experiment['method']
    method = config[method]
    total_cycles = int(experiment['cycles'])
    # Print settings to screen
    try:
        import tabulate
        print_table = True
    except:
        print_table = False

    if n_errors > 0:
        print()
        if not userYN('Continue checking experiment before exiting'):
            sys.exit()

    # Experiment summary
    print()
    print('-'*80)
    print()
    print(experiment['experiment name'], 'summary')
    print()
    print('method:', experiment['method'])
    print('recipe:', method['recipe'])
    print('cycles:', experiment['cycles'])
    pre_recipe = method.get('pre recipe', fallback = None)
    if pre_recipe is not None:
        print('pre recipe:', pre_recipe)
    first_port = method.get('first port', fallback = None)
    if first_port is not  None:
        print('first_port:', first_port)
    print('save path:', experiment['save path'])
    print('enable z stage:', hs.z.active)
    print('machine:', experiment['machine'])
    print()
    if not userYN('Confirm experiment'):
        sys.exit()
    print()

    # Flowcell summary
    table = {}
    for fc in hs.flowcells.keys():
        table[fc] = hs.flowcells[fc].sections.keys()
    print('-'*80)
    print()
    print('Flowcells:')
    print()
    if print_table:
        print(tabulate.tabulate(table, headers = 'keys', tablefmt = 'presto'))
    else:
        print(table)
    print()
    if not userYN('Confirm flowcells'):
        sys.exit()
    print()

    # Valve summary:
    table = []
    ports = []
    for port in port_dict:
        if not isinstance(port_dict[port], dict):
            ports.append(int(port_dict[port]))
            table.append([port_dict[port], port])
    print('-'*80)
    print()
    print('Valve:')
    print()
    if print_table:
        print(tabulate.tabulate(table, headers=['port', 'reagent'], tablefmt = 'presto'))
    else:
        print(table)
    print()
    if not userYN('Confirm valve assignment'):
        sys.exit()
    print()

    # Pump summary:
    AorB = [*hs.flowcells.keys()][0]
    fc = hs.flowcells[AorB]
    print('-'*80)
    print()
    print('Pump Settings:')
    print()
    inlet_ports = int(method.get('inlet ports', fallback = 2))
    print('Reagents pumped through row with ', inlet_ports, 'inlet ports')
    print(hs.p[AorB].n_barrels, 'syringe pump barrels per lane')
    print('Flush volume:',fc.volume['flush'], 'μL')
    if any([True for port in ports if port in [*range(1,9),*range(10,20)]]):
        print('Main prime volume:', fc.volume['main'], 'μL')
    if any([True for port in ports if port in [9,21,22,23,24]]):
        print('Side prime volume:', fc.volume['side'], 'μL')
    if 20 in ports:
        print('Sample prime volume:', fc.volume['sample'], 'μL')
    print('Flush flowrate:',fc.pump_speed['flush'], 'μL/min')
    print('Prime flowrate:',fc.pump_speed['prime'], 'μL/min')
    print('Reagent flowrate:',fc.pump_speed['reagent'], 'μL/min')
    print('Max volume:', hs.p[AorB].max_volume, 'μL')
    print('Min flow:', hs.p[AorB].min_flow, 'μL/min')
    print()
    if not userYN('Confirm pump settings'):
        sys.exit()

    # Cycle summary:

    variable_ports = hs.v24[AorB].variable_ports
    start_cycle = 1
    if method.get('pre recipe', fallback = None) is not None:
        start_cycle = 0
    table = []
    for cycle in range(start_cycle,total_cycles+1):
        row = []
        row.append(cycle)
        if len(variable_ports) > 0:
            for vp in variable_ports:
                if cycle > 0:
                    row.append(port_dict[vp][cycle])
                else:
                    row.append(None)
        if IMAG_counter > 0:
            colors = [*hs.optics.cycle_dict.keys()]
            for color in colors:
                row.append(hs.optics.cycle_dict[color][cycle])
        else:
            colors = []
        table.append(row)



    print('-'*80)
    print()
    print('Cycles:')
    print()
    if len(variable_ports) + len(colors) > 0:
        headers = ['cycle', *variable_ports, *colors]
        if print_table:
            print(tabulate.tabulate(table, headers, tablefmt='presto'))
        else:
            print(headers)
            print(table)
        print()
        stop_experiment = not userYN('Confirm cycles')
    else:
        if total_cycles == 1:
            stop_experiment = not userYN('Confirm only 1 cycle')
        else:
            stop_experiment = not userYN('Confirm all', total_cycles, 'cycles are the same')
    if stop_experiment:
        sys.exit()
    print()


    if IMAG_counter > 0:
        print('-'*80)
        print()
        print('Imaging settings:')
        print()
        laser_power = [hs.lasers['green'].set_point,
                       hs.lasers['red'].set_point]
        print('green laser power:', laser_power[0], 'mW')
        print('red laser power:',laser_power[1], 'mW')
        print('autofocus:', hs.AF)
        if hs.AF is not None:
            print('focus spacing', hs.obj.focus_spacing,'um')
            print('focus range', hs.obj.focus_range, '%')
            if hs.focus_tol > 0 and hs.AF != 'manual':
                print('focus tolerance:', hs.focus_tol, 'um')
            elif hs.AF != 'manual':
                print('focus tolerance: None')
                print('WARNING::Out of focus image risk increased')
            for i, filter in enumerate(hs.optics.focus_filters):
                if filter == 'home':
                    focus_laser_power = 0
                elif filter == 'open':
                    focus_laser_power = laser_power[i]
                else:
                    focus_laser_power = laser_power[i]*10**(-float(filter))
                print(colors[i+1], 'focus laser power ~', focus_laser_power, 'mW')
        print('z position when imaging:', hs.z.image_step)
        if hs.overlap > 0:
            print('pixel overlap:', hs.overlap)
            print('overlap direction:', hs.overlap_dir)
        z_planes = int(method.get('z planes', fallback = 0))
        if z_planes > 0:
            print('z planes:', z_planes)
        else:
            print('z planes:', *recipe_z_planes)
        if z_planes > 1 or any(recipe_z_planes):
            print('stack split:', hs.stack_split)


        if not userYN('Confirm imaging settings'):
            sys.exit()

    # Check if previous focus positions have been found, and confirm to use
    if os.path.exists(join(hs.log_path, 'focus_config.cfg')):

        focus_config = configparser.ConfigParser()
        focus_config.read(join(hs.log_path, 'focus_config.cfg'))
        cycles = 0
        sections = []
        for section in config.options('sections'):
            if focus_config.has_section(section):
                sections.append(section)
                n_focus_cycles = len(focus_config.options(section))
                if n_focus_cycles > cycles:
                    cycles = n_focus_cycles

        table = []
        for section in sections:
            row = []
            row.append(section)
            for c in range(1,cycles+1):
                if focus_config.has_option(section, str(c)):
                    row.append(focus_config[section][str(c)])
                else:
                    row.append(None)
            table.append(row)

        if len(sections) > 0 and cycles > 0:
            print('-'*80)
            print()
            print('Previous Autofocus Objective Positions:')
            print()
            headers = ['section', *['cycle'+str(c) for c in range(1,cycles+1)]]
            if print_table:
                print(tabulate.tabulate(table, headers, tablefmt='presto'))
            else:
                print(headers)
                print(table)
            print()
            if not userYN('Confirm using previous autofocus positions'):
                sys.exit()
            print()

##########################################################
## Setup HiSeq ###########################################
##########################################################
def initialize_hs(IMAG_counter):
    """Initialize the HiSeq and return the handle."""

    global n_errors

    experiment = config['experiment']
    method = experiment['method']
    method = config[method]

    if n_errors is 0:

        if not userYN('Initialize HiSeq'):
            sys.exit()

        hs.initializeCams(hs.logger)
        x_homed = hs.initializeInstruments()
        if not x_homed:
            error('HiSeq:: X-Stage did not home correctly')

        # HiSeq Settings
        inlet_ports = int(method.get('inlet ports', fallback = 2))
        hs.move_inlet(inlet_ports)                                              # Move to 2 or 8 port inlet

        # Set laser power
        for color in hs.lasers.keys():
            laser_power = int(method.get(color+' laser power', fallback = 10))
            hs.lasers[color].set_power(laser_power)
            if IMAG_counter > 0:
                if not hs.lasers[color].on:
                    error('HiSeq:: Lasers did not turn on.')

        hs.f.LED('A', 'off')
        hs.f.LED('B', 'off')
        hs.LED('all', 'startup')

        hs.move_stage_out()

    return hs


##########################################################
## Check Instructions ####################################
##########################################################
def check_instructions():
    """Check the instructions for errors.

       **Returns:**
       - first_line (int): Line number for the recipe to start from on the
       initial cycle.
       - IMAG_counter (int): The number of imaging steps.

    """

    method = config.get('experiment', 'method')
    method = config[method]

    first_port = method.get('first port', fallback = None)                      # Get first reagent to use in recipe
    # Backdoor to input line number for first step in recipe
    try:
        first_port = int(first_port)
        first_line = first_port
        first_port = None
    except:
        first_line = 0

    variable_ports = method.get('variable reagents', fallback  = None)


    valid_wait = []
    ports = []
    for port in config['reagents'].items():
        ports.append(port[1])
    if variable_ports is not None:
        variable_ports = variable_ports.split(',')
        for port in variable_ports:
            ports.append(port.strip())
    valid_wait = ports
    valid_wait.append('IMAG')
    valid_wait.append('STOP')
    valid_wait.append('TEMP')

    recipes = {}
    recipes['Recipe'] = config['experiment']['recipe path']
    pre_recipe = method.get('pre recipe',fallback= None)
    if pre_recipe is not None:
        recipes['Pre Recipe'] = pre_recipe

    for recipe in sorted([*recipes.keys()]):
        f = recipes[recipe]
        try:
            f = open(recipes[recipe])
        except:
            error(recipe,'::Unable to open', recipes[recipe])
        #Remove blank lines
        f_ = [line for line in f if line.strip()]
        f.close()

        IMAG_counter = 0.0
        wait_counter = 0
        z_planes = []

        for line_num, line in enumerate(f_):
            instrument, command = parse_line(line)

            if instrument == 'PORT':
                # Make sure ports in instruction files exist in port dictionary in config file
                if command not in ports:
                    error(recipe,'::', command, 'on line', line_num,
                          'is not listed as a reagent')

                #Find line to start at for first cycle
                if first_line == 0 and first_port is not None and recipe is 'Recipe':
                    if command.find(first_port) != -1:
                        first_line = line_num

            # Make sure pump volume is a number
            elif instrument == 'PUMP':
                if command.isdigit() == False:
                    error(recipe,'::Invalid volume on line', line_num)

            # Make sure wait command is valid
            elif instrument == 'WAIT':
                wait_counter += 1
                if command not in valid_wait:
                    error(recipe,'::Invalid wait command on line', line_num)

            # Make sure z planes is a number
            elif instrument == 'IMAG':
                IMAG_counter = int(IMAG_counter + 1)
                # Flag to make check WAIT is used before IMAG for 2 flowcells
                if wait_counter >= IMAG_counter:
                    IMAG_counter = float(IMAG_counter)
                if command.isdigit() == False:
                    error(recipe,'::Invalid number of z planes on line', line_num)
                else:
                    z_planes.append(command)

            # Make sure hold time (minutes) is a number
            elif instrument == 'HOLD':
                if command.isdigit() == False:
                    if command != 'STOP':
                        error(recipe,'::Invalid time on line', line_num)
                    else:
                        print(recipe,'::WARNING::HiSeq will stop until user input at line',
                               line_num)
            elif instrument == 'TEMP':
                if not command.isdigit():
                    error(recipe,'::Invalid temperature on line', line_num)
            # # Warn user that HiSeq will completely stop with this command
            # elif instrument == 'STOP':
            #     print('WARNING::HiSeq will stop until user input at line',
            #            line_num)
            # Make sure the instrument name is valid
            else:
                error(recipe,'::Bad instrument name on line',line_num)
                print(line)

    return first_line, IMAG_counter, z_planes

##########################################################
## Check Ports ###########################################
##########################################################
def check_ports():
    """Check for port errors and return a port dictionary.

    """

    method = config.get('experiment', 'method')
    method = config[method]
    total_cycles = int(config.get('experiment', 'cycles'))

    # Get cycle and port information from configuration file
    valve = config['reagents']                                                   # Get dictionary of port number of valve : name of reagent
    cycle_variables = method.get('variable reagents', fallback = None )         # Get list of port names in recipe that change every cycle
    cycle_reagents = config['cycles'].items()                                   # Get variable reagents that change with each cycle

    port_dict = {}

    # Make sure there are no duplicated names in the valve
    if len(valve.values()) != len(set(valve.values())):
        error('ConfigFile: Reagent names are not unique')
        #TODO: PRINT DUPLICATES

    if len(valve) > 0:
        # Create port dictionary
        for port in valve.keys():
            try:
                port_dict[valve[port]] = int(port)
            except:
                error('ConfigFile:List reagents as n (int) = name (str) ')

        # Add cycle variable port dictionary
        if cycle_variables is not None:
            cycle_variables = cycle_variables.split(',')
            for variable in cycle_variables:
                variable = variable.replace(' ','')
                if variable in port_dict:
                    error('ConfigFile::Variable', variable, 'can not be a reagent')
                else:
                    port_dict[variable] = {}

            # Fill cycle variable port dictionary with cycle: reagent name
            for cycle in cycle_reagents:
                reagent = cycle[1]
                variable, cyc_number = cycle[0].split(' ')
                if reagent in valve.values():
                    if variable in port_dict:
                        port_dict[variable][int(cyc_number)] = reagent
                    else:
                        error('ConfigFile::', variable, 'not listed as variable reagent')
                else:
                    error('ConfigFiles::Cycle reagent:', reagent, 'does not exist on valve')

            # Check number of reagents in variable reagents matches number of total cycles
            for variable in cycle_variables:
                variable = variable.replace(' ','')
                if len(port_dict[variable]) != total_cycles:
                    error('ConfigFile::Number of', variable, 'reagents does not match experiment cycles')

    else:
        print('WARNING::No ports are specified')

    # table = []
    # for port in port_dict:
    #     if not isinstance(port_dict[port], dict):
    #         table.append([port_dict[port], port])
    # print('Valve summary')
    # print(tabulate.tabulate(table, headers=['port', 'reagent'], tablefmt = 'presto'))

    return port_dict



def check_filters(cycle_dict, ex_dict):
    """Check filter section of config file.

       **Errors:**
       - Invalid Filter: System exits when a listed filter does not match
       configured filters on the HiSeq.
       - Duplicate Cycle: System exists when a filter for a laser is listed for
         the same cycle more than once.
       - Invalid laser: System exits when a listed laser color does not match
       configured laser colors on the HiSeq.

    """

    colors = [*cycle_dict.keys()]

    # Check laser, cycle, and filter are valid
    cycle_filters = config['filters'].items()
    for item in cycle_filters:
        # Get laser cycle = filter
        filter = item[1]

        # filters are floats, except for home and open,
        # and emission (True/False)
        if filter.lower() in ['true', 'yes', '1', 't', 'y']:
            filter = True
        elif filter.lower() in ['false', 'no', '0', 'f', 'n']:
            filter = False
        elif filter not in ['home','open']:
            filter = float(filter)
        laser, cycle = item[0].split()
        cycle = int(cycle)

        # Check if laser is valid, can use partial match ie, g or G for green
        if laser in colors:
            laser = [laser]
        else:
            laser = [colors[i] for i, c in enumerate(colors) if laser.lower() in c[0]]

        if len(laser) > 0:
            laser = laser[0]
            if laser in ex_dict.keys():
                if filter in ex_dict[laser]:
                    if cycle not in cycle_dict[laser]:
                        cycle_dict[laser][cycle] = filter
                    else:
                        error('ConfigFile::Duplicated cycle for', laser, 'laser')
            elif laser == 'em':
                if isinstance(filter, bool):
                    if cycle not in cycle_dict[laser]:
                        cycle_dict[laser][cycle] = filter
                    else:
                        error('ConfigFile::Duplicated emission filter cycle')
            else:
                error('ConfigFile::Invalid filter for', laser, 'laser')
        else:
            error('ConfigFile:Invalid laser')

    # Add default/home to cycles with out filters specified
    method = config.get('experiment', 'method')
    method = config[method]
    start_cycle = 1
    if method.get('pre recipe', fallback = None):
        start_cycle = 0
    last_cycle = int(config.get('experiment','cycles'))+1
    # Get/check default filters
    default_filters = {}
    fallbacks = {'red':'home', 'green':'home', 'em':'True'}
    for laser in colors:
        filter =  method.get('default '+laser+' filter', fallback = fallbacks[laser])
        try:
            filter = float(filter)
        except:
            pass
        if laser in ex_dict.keys():
            if filter in ex_dict[laser].keys():
                default_filters[laser] = filter
        elif laser == 'em':
            if filter in ['True', 'False']:
                default_filters[laser] = filter
    # Assign default filters to missing cycles
    for cycle in range(start_cycle,last_cycle):
        for laser in colors:
            if cycle not in cycle_dict[laser]:
                 cycle_dict[laser][cycle] = default_filters[laser]

    return cycle_dict




def endHOLD(fc):
    """Print end hold message for flowcell fc, returns False"""

    msg = 'PySeq::'+fc.position+'::cycle'+str(fc.cycle)+'::Hold stopped'
    hs.message(msg)

    return False

##########################################################
def do_nothing():
    """Do nothing."""
    pass


##########################################################
## iterate over lines, send to pump, and print response ##
##########################################################
def do_recipe(fc):
    """Do the next event in the recipe.

       **Parameters:**
       - fc (flowcell): The current flowcell.

    """

    AorB = fc.position
    fc.thread = None

    # Skip to first line of recipe on initial cycle
    if fc.cycle == 1 and fc.first_line is not None:
        for i in range(fc.first_line):
            line = fc.recipe.readline()
        fc.first_line = None


    #get instrument and command
    instrument = None
    while instrument is None:
        line = fc.recipe.readline()
        if line:
            instrument, command = parse_line(line)
        else:
            break

    if line:

        # Move reagent valve
        if instrument == 'PORT':
            #Move to cycle specific reagent if it is variable a reagent
            if fc.cycle <= fc.total_cycles:
                if command in hs.v24[AorB].variable_ports:
                    command = hs.v24[AorB].port_dict[command][fc.cycle]

            log_message = 'Move to ' + command
            fc.thread = threading.Thread(target = hs.v24[AorB].move,
                args = (command,))
            if fc.cycle <= fc.total_cycles:
                hs.LED(AorB, 'awake')

        # Pump reagent into flowcell
        elif instrument == 'PUMP':
            volume = int(command)
            speed = fc.pump_speed['reagent']
            log_message = 'Pumping ' + str(volume) + ' uL'
            fc.thread = threading.Thread(target = hs.p[AorB].pump,
                args = (volume, speed,))
            if fc.cycle <= fc.total_cycles:
                hs.LED(AorB, 'awake')
        # Incubate flowcell in reagent for set time
        elif instrument == 'HOLD':
            if command.isdigit():
                holdTime = float(command)*60
                log_message = 'Flowcell holding for ' + str(command) + ' min.'
                if hs.virtual:
                    fc.thread = threading.Timer(holdTime/hs.speed_up, endHOLD, args=(fc,))
                else:
                    fc.thread = threading.Timer(holdTime, endHOLD, args=(fc,))
            elif command == 'STOP':
                hs.message('PySeq::Paused')
                hs.LED(AorB, 'user')
                input("Press enter to continue...")
                log_message = ('Continuing...')
                fc.thread = threading.Thread(target = do_nothing)
            if fc.cycle <= fc.total_cycles:
                hs.LED(AorB, 'sleep')
        # Wait for other flowcell to finish event before continuing with current flowcell
        elif instrument == 'WAIT':
            if command == 'TEMP':
                fc.thread = threading.Thread(target = hs.T.wait_fc_T,
                                             args=(AorB, fc.temperature,))
                log_message = ('Waiting to reach '+str(fc.temperature)+'°C')
            elif fc.waits_for is not None:
                if command in flowcells[fc.waits_for].events_since_IMAG:
                    log_message = command + ' has occurred, skipping WAIT'
                    fc.thread = threading.Thread(target = do_nothing)
                else:
                    log_message = 'Waiting for ' + command
                    fc.thread = threading.Thread(target = WAIT,
                        args = (AorB, command,))
            else:
                log_message = 'Skip waiting for ' + command
                fc.thread = threading.Thread(target = do_nothing)
            if fc.cycle <= fc.total_cycles:
                hs.LED(AorB, 'sleep')
        # Image the flowcell
        elif instrument == 'IMAG':
            if hs.scan_flag and fc.cycle <= fc.total_cycles:
                hs.message('PySeq::'+AorB+'::Waiting for camera')
                while hs.scan_flag:
                    pass
            #hs.scan_flag = True
            fc.events_since_IMAG = []
            log_message = 'Imaging flowcell'
            fc.thread = threading.Thread(target = IMAG,
                args = (fc,int(command),))
            if fc.cycle <= fc.total_cycles:
                hs.LED(AorB, 'imaging')
        elif instrument == 'TEMP':
            log_message = 'Setting temperature to ' + command + ' °C'
            command  = float(command)
            fc.thread = threading.Thread(target = hs.T.set_fc_T,
                args = (AorB,command,))
            fc.temperature = command
        # Block all further processes until user input
        # elif instrument == 'STOP':
        #     hs.message('PySeq::Paused')
        #     hs.LED(AorB, 'user')
        #     input("Press enter to continue...")
        #     hs.message('PySeq::Continuing...')


        #Signal to other flowcell that current flowcell reached signal event
        if fc.signal_event == instrument or fc.signal_event == command:
            fc.wait_thread.set()
            fc.signal_event = None

        # Start new action on current flowcell
        if fc.thread is not None and fc.cycle <= fc.total_cycles:
            fc.addEvent(instrument, command)
            hs.message('PySeq::'+AorB+'::cycle'+str(fc.cycle)+'::'+log_message)
            thread_id = fc.thread.start()
        elif fc.thread is not None and fc.cycle > fc.total_cycles:
            fc.thread =  threading.Thread(target = time.sleep, args = (10,))

    else:
        # End of recipe
        fc.restart_recipe(hs)


##########################################################
## Image flowcell ########################################
##########################################################
def IMAG(fc, n_Zplanes):
    """Image the flowcell at a number of z planes.

       For each section on the flowcell, the stage is first positioned
       to the center of the section to find the optimal focus. Then if no
       optical settings are listed, the optimal filter sets are found.
       Next, the stage is repositioned to scan the entire section and
       image the specified number of z planes.

       **Parameters:**
       fc: The flowcell to image.
       n_Zplanes: The number of z planes to image.

       **Returns:**
       int: Time in seconds to scan the entire section.

    """

    hs.scan_flag = True
    AorB = fc.position
    cycle = str(fc.cycle)
    start = time.time()

    # Manual focus ALL sections across flowcells
    if hs.AF == 'manual':
        focus.manual_focus(hs, flowcells)
        hs.AF = 'partial once'


    #Image sections on flowcell
    for section in fc.sections:
        pos = fc.stage[section]
        hs.y.move(pos['y_initial'])
        hs.x.move(pos['x_initial'])
        hs.z.move(pos['z_pos'])
        hs.obj.move(hs.obj.focus_rough)

        # Autofocus
        msg = 'PySeq::' + AorB + '::cycle' + cycle+ '::' + str(section) + '::'
        if hs.AF and not isinstance(hs.AF, int):
            obj_pos = focus.get_obj_pos(hs, section, cycle)
            if obj_pos is None:
                # Move to focus filters
                for i, color in enumerate(hs.optics.colors):
                    hs.optics.move_ex(color,hs.optics.focus_filters[i])
                hs.message(msg + 'Start Autofocus')
                try:
                    if hs.autofocus(pos):                                       # Moves to optimal objective position
                        hs.message(msg + 'Autofocus complete')
                        pos['obj_pos'] = hs.obj.position
                    else:                                                       # Moves to rough focus objective position
                        hs.message(msg + 'Autofocus failed')
                        pos['obj_pos'] = None
                except:
                    hs.message(msg + 'Autofocus failed')
                    print(sys.exc_info()[0])
                    pos['obj_pos'] = None
            else:
                hs.obj.move(obj_pos)
                pos['obj_pos'] = hs.obj.position
            focus.write_obj_pos(hs, section, cycle)

        #Override recipe number of z planes
        if fc.z_planes is not None: n_Zplanes = fc.z_planes

        # Calculate objective positions to image
        if n_Zplanes > 1 and not isinstance(hs.AF, int):
            obj_start = int(hs.obj.position - hs.nyquist_obj*n_Zplanes*hs.stack_split)       # (Default) 2/3 of planes below opt_ob_pos and 1/3 of planes above
        elif isinstance(hs.AF, int):
            obj_start = hs.AF
        else:
            obj_start = hs.obj.position

        image_name = AorB
        image_name += '_s' + str(section)
        image_name += '_r' + cycle
        if fc.IMAG_counter is not None:
            image_name += '_' + str(fc.IMAG_counter)

        # Scan section on flowcell
        hs.y.move(pos['y_initial'])
        hs.x.move(pos['x_initial'])
        hs.obj.move(obj_start)
        n_tiles = pos['n_tiles']
        n_frames = pos['n_frames']

        # Set filters
        for color in hs.optics.cycle_dict.keys():
            filter = hs.optics.cycle_dict[color][fc.cycle]
            if color is 'em':
                hs.optics.move_em_in(filter)
            else:
                hs.optics.move_ex(color, filter)

        hs.message(msg + 'Start Imaging')

        try:
            scan_time = hs.scan(n_tiles, n_Zplanes, n_frames, image_name)
            scan_time = str(int(scan_time/60))
            hs.message(msg + 'Imaging completed in', scan_time, 'minutes')
        except:
            error('Imaging failed.')

    # Reset filters
    for color in hs.optics.cycle_dict.keys():
        if color is 'em':
            hs.optics.move_em_in(True)
        else:
            hs.optics.move_ex(color, 'home')

    if fc.IMAG_counter is not None:
        fc.IMAG_counter += 1

    hs.scan_flag = False



def WAIT(AorB, event):
    """Hold the flowcell *AorB* until the specfied event in the other flowell.

       **Parameters:**
       AorB (str): Flowcell position, A or B, to be held.
       event: Event in the other flowcell that releases the held flowcell.

       **Returns:**
       int: Time in seconds the current flowcell was held.

    """
    signaling_fc = flowcells[AorB].waits_for
    cycle = str(flowcells[AorB].cycle)
    start = time.time()
    flowcells[signaling_fc].signal_event = event                                # Set the signal event in the signal flowcell
    flowcells[signaling_fc].wait_thread.wait()                                  # Block until signal event in signal flowcell
    hs.message('PySeq::'+AorB+'::cycle'+cycle+'::Flowcell ready to continue')
    flowcells[signaling_fc].wait_thread.clear()                                 # Reset wait event
    stop = time.time()
    return stop-start


def flush_lines():
    """Flush all, some, or none of lines.

       If flush_ports are supplied then no user prompts asking for which
       ports to flush are given. The default volume is 1000 uL and the
       default flowrate is 700 uL/min.

    """

    AorB = [*hs.flowcells.keys()][0]
    port_dict = hs.v24[AorB].port_dict

    hs.LED('all', 'user')

    # Select lines to flush
    confirm = False
    while not confirm:
        flush_ports = input("Flush all, some, or none of the lines? ")
        if flush_ports.strip().lower() == 'all':
            flush_ports = list(port_dict.keys())
            for vp in hs.v24[AorB].variable_ports:
                if vp in flush_ports:
                    flush_ports.remove(vp)
            confirm = userYN('Confirm flush all lines')
        elif flush_ports.strip().lower() in ['none', 'n', '']:
            flush_ports = []
            confirm = userYN('Confirm skip flushing lines')
        else:
            good =[]
            bad = []
            for fp in flush_ports.split(','):
                fp = fp.strip()
                if fp in port_dict.keys():
                    good.append(fp)
                else:
                    try:
                        fp = int(fp)
                        if fp in range(1,hs.v24[AorB].n_ports+1):
                            good.append(fp)
                        else:
                            bad.append(fp)
                    except:
                        bad.append(fp)
            if len(bad) > 0:
                print('Valid ports:', *good)
                print('Invalid ports:', *bad)
                confirm = not userYN('Re-enter lines to flush')
            else:
                confirm = userYN('Confirm only flushing',*good)

            if confirm:
                flush_ports = hs.flowcells

    if len(flush_ports) > 0:
        while not userYN('Temporary flowcell(s) locked on to stage'): pass
        while not userYN('All valve input lines in water'): pass
        while not userYN('Ready to flush'): pass

        # Flush ports
        flowrate = hs.flowcells[AorB].pump_speed['flush']
        volume = hs.flowcells[AorB].volume['flush']
        hs.flush_lines(flush_ports = flush_ports, volume = volume, flowrate = flowrate)


    return confirm

def prime_lines(self, flush_YorN = True):
    """Prime lines with all reagents in valve.

       Prime all reagent lines listed in the 24 port valve port dictionary
       The default volumes for ports 1-8 & 10-19 (in the chiller) is 500 uL
       port 20 (sample) is 250 uL, and ports 9 & 22-24 is 350 uL (all
       volumes stored in self.flowcell.[AorB].volume dictionary). After
       priming, the lines will be rinsed with the rinse port reagent, if
       supplied.

       **Parameters:**
        - flush_YorN (bool): Flag for user prompts in automated control

       **Returns:**
        - string/int: Last port that was used

    """

    hs.LED('all', 'user')

    confirm = False
    while not confirm:
        prime_YorN = userYN("Prime lines")
        if prime_YorN:
            confirm = userYN("Confirm prime lines")
        else:
            confirm = userYN("Confirm skip priming lines")

    if prime_YorN:
        if not flush_YorN:
            while not userYN('Temporary flowcell(s) locked on to stage'): pass
        while not userYN('Valve input lines in reagents'): pass
        while not userYN('Ready to prime lines'): pass

        AorB_ = [*hs.flowcells.keys()][0]
        flowrate = hs.flowcells[AorB_].pump_speed['prime']
        last_port = hs.flush_lines(flowrate = flowrate)

        rinse_lines(last_port = last_port)

        hs.LED('all', 'user')
        while not userYN('Temporary flowcell(s) removed'): pass


    while not userYN('Experiment flowcell(s) locked on to stage'): pass
    if not prime_YorN:
        while not userYN('Valve input lines in reagents'): pass
    while not userYN('Door closed'): pass

    return prime_YorN

def rinse_lines(flowcells='AB', last_port=None):


    # Get default rinse port
    AorB = flowcells[0]
    rinse_port = hs.v24[AorB].rinse_port
    ask_rinse = rinse_port is not None
    if rinse_port == last_port:                                                 # Option to skip rinse if last reagent pump was rinse reagent
        ask_rinse = False

    # Ask for rinse reagent if not supplied
    if ask_rinse:
        hs.LED('all', 'user')
        print('Last reagent pumped was', str(last_port))
        if userYN('Rinse lines'):
            while not ask_rinse:
                rinse_port = input('Specify rinse reagent: ')
                ask_rinse = rinse_port in hs.v24[AorB].port_dict or rinse_port is None
                if not ask_rinse:
                    print('ERROR::Invalid rinse reagent')
                    print('Choose from:', *list(hs.v24[AorB].port_dict.keys()))
                if rinse_port is None:
                    ask_rinse = userYN('Skip rinsing')

    # Rinse linse
    if rinse_port:
        flowrate = hs.flowcells[AorB].pump_speed['prime']
        last_port = hs.flush_lines(flowcells=flowcells, flush_ports = [rinse_port], flowrate=flowrate)


    fc.thread.start()


##########################################################
## Shut down system ######################################
##########################################################
def do_shutdown():
    """Shutdown the HiSeq and flush all reagent lines if prompted."""

    for fc in flowcells.values():
        while fc.thread.is_alive():
            fc.wait_thread.set()
            time.sleep(10)

    hs.LED('all', 'startup')
    hs.message('PySeq::Shutting down...')


    hs.z.move([0, 0, 0])
    hs.move_stage_out()
    flush_lines()
    ##Flush all lines##
    # hs.LED('all', 'user')
    #
    # # flush_YorN = userYN("Flush lines")
    # if flush_YorN:
    #     hs.message('Lock temporary flowcell on  stage')
    #     hs.message('Place all valve input lines in PBS/water')
    #     input('Press enter to continue...')
    #
    #     hs.LED('all', 'startup')
    #     for fc in flowcells.keys():
    #         volume = flowcells[fc].volume['main']
    #         speed = flowcells[fc].pump_speed['flush']
    #         for port in hs.v24[fc].port_dict.keys():
    #             if isinstance(port_dict[port], int):
    #                 hs.v24[fc].move(port)
    #                 hs.p[fc].pump(volume, speed)
    #         ##Return pump to top and NO port##
    #         hs.p[fc].command('OA0R')
    #         hs.p[fc].command('IR')
    # else:
    #     hs.LED('all', 'user')


    hs.message('Retrieve experiment flowcells')
    input('Press any key to finish shutting down')

    for fc in flowcells.values():
        AorB = fc.position
        fc_log_path = join(hs.log_path, 'Flowcell'+AorB+'.log')
        with open(fc_log_path, 'w') as fc_file:
            for i in range(len(fc.history[0])):
                fc_file.write(str(fc.history[0][i])+' '+
                              str(fc.history[1][i])+' '+
                              str(fc.history[2][i])+'\n')

    # Turn off y stage motor
    hs.y.move(0)
    hs.y.command('OFF')
    hs.LED('all', 'off')



##########################################################
## Free Flowcells ########################################
##########################################################
def free_fc():
    """Release the first flowcell if flowcells are waiting on each other."""

    # Get which flowcell is to be first
    experiment = config['experiment']
    cycles = int(experiment.get('first flowcell', fallback = 'A'))
    first_fc = experiment.get('first flowcell', fallback = 'A')

    if len(flowcells) == 1:
        fc = flowcells[[*flowcells][0]]
        try:
            fc.wait_thread.set()
        except:
            pass
        fc.signal_event = None
    else:
        flowcells_ = [fc.position for fc in flowcells.values() if fc.total_cycles <= cycles]
        if len(flowcells_) == 1:
            fc = flowcells_[0]
        else:
            fc = flowcells[first_fc]
        flowcells[fc.waits_for].wait_thread.set()
        flowcells[fc.waits_for].signal_event = None

    hs.message('PySeq::Flowcells are waiting on each other starting flowcell',
                fc.position)

    return fc.position



def get_config(args):
    """Return the experiment config appended with the method config.

       **Parameters:**
       - args (dict): Dictionary with the config path, the experiment name and
         the output path to store images and logs.

       **Returns:**
       - config: The experiment config appended with the method config.

    """

    # Create config parser
    config = configparser.ConfigParser()

    # Open config file
    if os.path.isfile(args['config']):
         config_path = args['config']
         config.read(config_path)
    elif args['config'] in methods.get_methods():
        config_path, recipe_path = methods.return_method(args['config'])
        config.read(config_path)
    else:
        error('ConfigFile::Does not exist')
        sys.exit()
    # Set output path
    output_path = args['output']
    config['experiment']['save path'] = output_path
    # Set experiment name
    experiment_name = args['name']
    config['experiment']['experiment name'] = experiment_name

    # set log and image path
    save_dir = join(output_path, experiment_name)
    config['experiment']['log path'] = join(save_dir, 'logs')
    config['experiment']['image path'] = join(save_dir, 'images')

    # save user valve
    USERVALVE = False
    if config.has_section('reagents'):
        valve = config['reagents'].items()
        if len(valve) > 0:
            USERVALVE = True

    # Get method specific configuration
    method = config['experiment']['method']
    if method in get_methods():
        config_path, recipe_path = return_method(method)
        config.read(config_path)
    elif os.path.isfile(method):
        config.read(method)
        recipe_path = None
    elif config.has_section(method):
        recipe_path = None
    else:
        error('ConfigFile::Error reading method configuration')
        sys.exit()


    # Check method keys
    if not check_settings(config[method]):
        go = userYN('Proceed with experiment')
        if not go:
            sys.exit()

    # Get recipe
    recipe_name = config[method]['recipe']
    if recipe_path is not None:
        pass
    elif os.path.isfile(recipe_name):
        recipe_path = recipe_name
    else:
        error('ConfigFile::Error reading recipe')

    config['experiment']['recipe path'] = recipe_path


    # Don't override user defined valve
    user_config = configparser.ConfigParser()
    user_config.read(config_path)
    if USERVALVE:
        config.read_dict({'reagents':dict(user_config['reagents'])})
    if user_config.has_section(method):
        config.read_dict({method:dict(user_config[method])})

    return config

def check_fc_temp(fc):
    """Check temperature of flowcell."""

    if fc.temperature is not None:
        if fc.temp_timer is None:
            fc.temp_timer = threading.Timer(fc.temp_interval, do_nothing)
            fc.temp_timer.start()
        if not fc.temp_timer.is_alive():
            #print('checking temp')
            T = hs.T.get_fc_T(fc.position)
            hs.message(False, 'PySeq::'+fc.position+'::Temperature::',T,'°C')
            fc.temp_timer = None

            if abs(fc.temperature - T) > 5:
                msg =  'PySeq::'+fc.position+'::WARNING::Set Temperature '
                msg += str(fc.temperature) + ' C'
                hs.message(msg)
                msg =  'PySeq::'+fc.position+'::WARNING::Actual Temperature '
                msg += str(T) + ' C'
                hs.message(msg)

            return T

###################################
## Run System #####################
###################################
args_ = args.get_arguments()                                                    # Get config path, experiment name, & output path
if __name__ == 'pyseq.main':
    n_errors = 0
    config = get_config(args_)                                                  # Get config file
    log_path = make_directories()                                               # create exp, image, and log directories
    logger = pyseq.setup_logger(log_path=log_path, config=config)                               # Create logfiles
    port_dict = check_ports()                                                   # Check ports in configuration file
    first_line, IMAG_counter, z_planes = check_instructions()                   # Checks instruction file is correct and makes sense
    flowcells = setup_flowcells(first_line, IMAG_counter)                       # Create flowcells
    hs = configure_instrument(IMAG_counter, port_dict, flowcells)
    confirm_settings(z_planes)
    hs = initialize_hs(IMAG_counter)                                            # Initialize HiSeq, takes a few minutes

    if n_errors is 0:
        flush_YorN = flush_lines()                                              # Flush lines
        prime_YorN = prime_lines(flush_YorN)                                    # Ask to prime lines
        if not userYN('Start experiment'):
            sys.exit()

        # Do prerecipe or Initialize Flowcells
        for fc in hs.flowcells.values():
            if fc.prerecipe_path:
                fc.pre_recipe(hs)
            else:
                fc.restart_recipe(hs)

        cycles_complete = False
        while not cycles_complete:
            stuck = 0
            complete = 0

            for fc in hs.flowcells.values():
                if not fc.thread.is_alive():                                    # flowcell not busy, do next step in recipe
                    do_recipe(fc)

                if fc.signal_event:                                             # check if flowcells are waiting on each other
                    stuck += 1

                if fc.cycle > fc.total_cycles:                                  # check if all cycles are complete on flowcell
                    complete += 1

                check_fc_temp(fc)

            if stuck == len(flowcells):                                         # Start the first flowcell if they are waiting on each other
                free_fc()

            if complete == len(flowcells):                                      # Exit while loop
                cycles_complete = True

            if hs.current_view is not None:                                     # Show latest images in napari, WILL BLOCK
                hs.current_view.show()
                hs.current_view = None

        do_shutdown()                                                           # Shutdown HiSeq
    else:
        error('Total number of errors =', n_errors)

def main():
    pass
