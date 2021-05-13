"""
TODO:

"""

import time
import logging
import os
from os.path import join
import sys
import configparser
import threading
import argparse

from . import methods
from . import args
from . import focus

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

##########################################################
## Flowcell Class ########################################
##########################################################
class Flowcell():
    """HiSeq 2500 System :: Flowcell

       **Attributes:**
       - position (str): Flowcell is at either position A (left slot )
         or B (right slot).
       - recipe_path (path): Path to the recipe.
       - recipe (file): File handle for the recipe.
       - first_line (int): Line number for the recipe to start from on the
         initial cycle.
       - cycle (int): The current cycle.
       - total_cycles (int): Total number of the cycles for the experiment.
       - history ([[int,],[str,],[str,]]): Timeline of flowcells events, the
         1st column is the timestamp, the 2nd column is the event, and the
         3rd column is an event specific detail.
       - sections (dict): Dictionary of section names keys and coordinate
         positions of the sections on the flowcell values.
       - stage (dict): Dictionary of section names keys and stage positioning
         and imaging details of the sections on the flowcell values.
       - thread (int): Thread id of the current event on the flowcell.
       - signal_event (str): Event that signals the other flowcell to continue
       - wait_thread (threading.Event()): Blocks other flowcell until current
         flowcell reaches signal event.
       - waits_for (str): Flowcell A waits for flowcell B and vice versa.
       - pump_speed (dict): Dictionary of pump scenario keys and pump speed
         values.
       - volume (dict): Keys are events/situations and values are volumes
         in uL to use at the event/situation.
       - filters (dict): Dictionary of filter set at each cycle, c: em, ex1, ex2.
       - IMAG_counter (None/int): Counter for multiple images per cycle.
       - events_since_IMAG (list): Record events since last IMAG step.
       - temp_timer: Timer to check temperature of flowcell.
       - temperature (float): Set temperature of flowcell in °C.
       - temp_interval (float): Interval in seconds to check flowcell temperature.
       - z_planes (int): Override number of z planes to image in recipe.
       - pre_recipe_path (path): Recipe to run before actually starting experiment
       - pre_recipe (file): File handle for the pre recipe.

    """

    def __init__(self, position):
        """Constructor for flowcells

           **Parameters:**
           - position (str): Flowcell is at either position A (left slot) or
             B (right slot).

        """

        self.recipe_path = None
        self.recipe = None
        self.first_line = None
        self.cycle = 0                                                          # Current cycle
        self.total_cycles = 0                                                   # Total number of cycles for experiment
        self.history = [[],[],[]]                                               # summary of events in flowcell history
        self.sections = {}                                                      # coordinates of flowcell of sections to image
        self.stage = {}                                                         # stage positioning info for each section
        self.thread = None                                                      # threading to do parallel actions on flowcells
        self.signal_event = None                                                # defines event that signals the next flowcell to continue
        self.wait_thread = threading.Event()                                    # blocks next flowcell until current flowcell reaches signal event
        self.waits_for = None                                                   # position of the flowcell that signals current flowcell to continue
        self.pump_speed = {}
        self.volume = {'main':None,'side':None,'sample':None,'flush':None}      # Flush volume
        self.filters = {}                                                       # Dictionary of filter set at each cycle, c: em, ex1, ex2
        self.IMAG_counter = None                                                # Counter for multiple images per cycle
        self.events_since_IMAG = []                                             # List events since last IMAG step
        self.temp_timer = None                                                  # Timer to check temperature of flowcell
        self.temperature = None                                                 # Set temperature of flowcell
        self.temp_interval = None                                               # Interval in minutes to check flowcell temperature
        self.z_planes = None                                                    # Override number of z planes to image in recipe.
        self.pre_recipe_path = None                                              # Recipe to run before actually starting experiment

        while position not in ['A', 'B']:
            print('Flowcell must be at position A or B')
            position = input('Enter A or B for ' + str(position) + ' : ')

        self.position = position


    def addEvent(self, event, command):
        """Record history of events on flow cell.

           **Parameters:**
           - instrument (str): Type of event can be valv, pump, hold, wait, or
             imag.
           - command (str): Details specific to each event such as hold time,
             buffer, event to wait for, z planes to image, or pump volume.

           **Returns:**
           - int: A time stamp of the last event.

        """

        self.history[0].append(time.time())                                     # time stamp
        self.history[1].append(event)                                           # event (valv, pump, hold, wait, imag)
        self.history[2].append(command)                                         # details such hold time, buffer, event to wait for
        self.events_since_IMAG.append(event)
        if event is 'PORT':
            self.events_since_IMAG.append(command)
        if event in ['IMAG', 'STOP']:
            self.events_since_IMAG.append(event)

        return self.history[0][-1]                                              # return time stamp of last event


    def restart_recipe(self):
        """Restarts the recipe and returns the number of completed cycles."""

        # Restart recipe
        if self.recipe is not None:
            self.recipe.close()
        self.recipe = open(self.recipe_path)
        # Reset image counter (if mulitple images per cycle)
        if self.IMAG_counter is not None:
            self.IMAG_counter = 0

        msg = 'PySeq::'+self.position+'::'
        if self.cycle == self.total_cycles:
            # Increase cycle counter
            self.cycle += 1
            # Flowcell completed all cycles
            hs.message(msg+'Completed '+ str(self.total_cycles) + ' cycles')
            hs.T.fc_off(fc.position)
            self.temperature = None
            do_rinse(self)
            if self.temp_timer is not None:
                self.temp_timer.cancel()
                self.temp_timer = None
            self.thread = threading.Thread(target = time.sleep, args = (10,))
        elif self.cycle < self.total_cycles:
            # Increase cycle counter
            self.cycle += 1
            # Start new cycle
            restart_message = msg+'Starting cycle '+str(self.cycle)
            self.thread = threading.Thread(target = hs.message,
                                           args = (restart_message,))
        else:
            self.thread = threading.Thread(target = time.sleep, args = (10,))

        thread_id = self.thread.start()

        return self.cycle

    def pre_recipe(self):
        """Initializes pre recipe before starting experiment."""
        prerecipe_message = 'PySeq::'+self.position+'::'+'Starting pre recipe'
        self.recipe = open(self.prerecipe_path)
        self.thread = threading.Thread(target = hs.message,
                                       args = (prerecipe_message,))
        thread_id = self.thread.start()

        return thread_id

    def endHOLD(self):
        """Ends hold for incubations in buffer, returns False."""

        msg = 'PySeq::'+self.position+'::cycle'+str(self.cycle)+'::Hold stopped'
        hs.message(msg)

        return False



##########################################################
## Setup Flowcells #######################################
##########################################################

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
        f_sect_name = sect_name.replace('_','')                                 #remove underscores
        position = config['sections'][sect_name]
        AorB, coord  = position.split(':')
        # Create flowcell if it doesn't exist
        if AorB not in flowcells.keys():
            fc = Flowcell(AorB)
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


        # Add section to flowcell
        if sect_name in flowcells[AorB].sections:
            error(err_msg, sect_name, 'duplicated on flowcell', AorB)
        else:
            coord = coord.split(',')
            flowcells[AorB].sections[f_sect_name] = []                          # List to store coordinates of section on flowcell
            flowcells[AorB].stage[f_sect_name] = {}                             # Dictionary to store stage position of section on flowcell
            if float(coord[0]) <  float(coord[2]):
                error(err_msg,'Invalid x coordinates for', sect_name)
            if float(coord[1]) <  float(coord[3]):
                error(err_msg, 'Invalid y coordinates for', sect_name)
            for i in range(4):
                try:
                    flowcells[AorB].sections[f_sect_name].append(float(coord[i]))
                except:
                    error(err_msg,' No position for', sect_name)

        # if runnning mulitiple flowcells...
        # Define first flowcell
        # Define prior flowcell signals to next flowcell
        if len(flowcells) > 1:
            flowcell_list = [*flowcells]
            for fc in flowcells.keys():
                flowcells[fc].waits_for = flowcell_list[
                    flowcell_list.index(fc)-1]
            if experiment['first flowcell'] not in flowcells:
                error('ConfigFile::First flowcell does not exist')
            if isinstance(IMAG_counter, int):
                error('Recipe::Need WAIT before IMAG with 2 flowcells.')

    # table = {}
    # for fc in flowcells:
    #     table[fc] = flowcells[fc].sections.keys()
    # print('Flowcell section summary')
    # print(tabulate.tabulate(table, headers = 'keys', tablefmt = 'presto'))
    #
    # userYN('Confirm flowcell(s)')

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


##########################################################
## Setup Logging #########################################
##########################################################
def setup_logger():
    """Create a logger and return the handle."""

    # Get experiment info from config file
    experiment = config['experiment']
    experiment_name = experiment['experiment name']
    # Make directory to save data
    save_path = join(experiment['save path'],experiment_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # Make directory to save logs
    log_path = join(save_path, experiment['log path'])
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(10)

    # Create console handler
    c_handler = logging.StreamHandler()
    c_handler.setLevel(21)
    # Create file handler
    f_log_name = join(log_path,experiment_name + '.log')
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

    # Save copy of config with log
    config_path = join(log_path,'config.cfg')
    with open(config_path, 'w') as configfile:
        config.write(configfile)

    return logger


def configure_instrument(IMAG_counter, port_dict):
    """Configure and check HiSeq settings."""

    global n_errors


    model, name = methods.get_machine_info(args_['virtual'])
    if model is not None:
        config['experiment']['machine'] = model+'::'+name
    experiment = config['experiment']
    method = experiment['method']
    method = config[method]

    try:
        total_cycles = int(experiment.get('cycles'))
    except:
        error('ConfigFile:: Cycles not specified')

    # Creat HiSeq Object
    if model == 'HiSeq2500':
        if args_['virtual']:
            from . import virtualHiSeq
            hs = virtualHiSeq.HiSeq(name, logger)
            hs.speed_up = int(method.get('speed up', fallback = 5000))
        else:
            import pyseq
            com_ports = pyseq.get_com_ports()
            hs = pyseq.HiSeq(name, logger)
    else:
        sys.exit()

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
        error('MethodFile:: inlet ports must be 2 or 8.')

    variable_ports = method.get('variable reagents', fallback = None)
    hs.z.image_step = int(method.get('z position', fallback = 21500))
    hs.overlap = abs(int(method.get('overlap', fallback = 0)))
    hs.overlap_dir = method.get('overlap direction', fallback = 'left').lower()
    if hs.overlap_dir not in ['left', 'right']:
        error('MethodFile:: overlap direction must be left or right')

    for fc in flowcells.values():
        AorB = fc.position
        hs.v24[AorB].side_ports = side_ports
        hs.v24[AorB].sample_port = sample_port
        hs.v24[AorB].port_dict = port_dict                                      # Assign ports on HiSeq
        if variable_ports is not None:
            v_ports = variable_ports.split(',')
            for v in v_ports:                                                   # Assign variable ports
                hs.v24[AorB].variable_ports.append(v.strip())
        hs.p[AorB].update_limits(n_barrels)                                     # Assign barrels per lane to pump
        for section in fc.sections:                                             # Convert coordinate sections on flowcell to stage info
            pos = hs.position(AorB, fc.sections[section])
            fc.stage[section] = pos
            fc.stage[section]['z_pos'] = [hs.z.image_step]*3

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
            error('MethodFile:: Invalid '+color+' laser power')

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
    if hs.AF not in ['partial', 'partial once', 'full', 'full once', 'manual', None]:
        error('ConfigFile:: Auto focus method not valid.')
    #Enable/Disable z stage
    hs.z.active = method.getboolean('enable z stage', fallback = True)
    # Get focus Tolerance
    hs.focus_tol = float(method.get('focus tolerance', fallback = 0))
    # Get focus range
    range = float(method.get('focus range', fallback = 90))
    spacing = float(method.get('focus spacing', fallback = 4.1))
    hs.obj.update_focus_limits(range=range, spacing=spacing)                    # estimate, get actual value in hs.obj_stack()

    hs.bundle_height = int(method.get('bundle height', fallback = 128))

    # Assign output directory
    save_path = experiment['save path']
    experiment_name = experiment['experiment name']
    save_path = join(experiment['save path'], experiment['experiment name'])
    if not os.path.exists(save_path):
        try:
            os.mkdir(save_path)
        except:
            error('ConfigFile:: Save path not valid.')
    # Assign image directory
    image_path = join(save_path, experiment['image path'])
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    with open(join(image_path,'machine_name.txt'),'w') as file:
        file.write(hs.name)
    hs.image_path = image_path
    # Assign log directory
    log_path = join(save_path, experiment['log path'])
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    hs.log_path = log_path

    return hs

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
    for fc in flowcells:
        table[fc] = flowcells[fc].sections.keys()
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
    AorB = [*flowcells.keys()][0]
    fc = flowcells[AorB]
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

        hs.initializeCams(logger)
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
        LED('all', 'startup')

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



def LED(AorB, indicate):
    """Control front LEDs to communicate what the HiSeq is doing.

       **Parameters:**
       - AorB (str): Flowcell position (A or B), or all.
       - indicate (str): Current action of the HiSeq or state of the flowcell.

        ===========  ===========  =============================
        LED MODE      indicator   HiSeq Action / Flowcell State
        ===========  ===========  ===================================================
        off              off      The flowcell is not in use.
        yellow          error     There is an error with the flowcell.
        green          startup    The HiSeq is starting up or shutting down
        pulse green     user      The HiSeq requires user input
        blue            sleep     The flowcell is holding or waiting.
        pulse blue      awake     HiSeq valve, pump, or temperature action on the flowcell.
        sweep blue     imaging    HiSeq is imaging the flowcell.
        ===========  ===========  ========================================

    """

    fc = []
    if AorB in flowcells.keys():
        fc = [AorB]
    elif AorB == 'all':
        fc = [*flowcells.keys()]

    for AorB in fc:
        if indicate == 'startup':
            hs.f.LED(AorB, 'green')
        elif indicate == 'user':
            hs.f.LED(AorB, 'pulse green')
        elif indicate == 'error':
            hs.f.LED(AorB, 'yellow')
        elif indicate == 'sleep':
            hs.f.LED(AorB, 'blue')
        elif indicate == 'awake':
            hs.f.LED(AorB, 'pulse blue')
        elif indicate == 'imaging':
            hs.f.LED(AorB, 'sweep blue')
        elif indicate == 'off':
            hs.f.LED(AorB, 'off')

    return True

def userYN(*args):
    """Ask a user a Yes/No question and return True if Yes, False if No."""

    question = ''
    for a in args:
        question += str(a) + ' '

    response = True
    while response:
        answer = input(question + '? Y/N = ')
        answer = answer.upper().strip()
        if answer == 'Y':
            response = False
            answer = True
        elif answer == 'N':
            response = False
            answer = False

    return answer



def do_flush():
    """Flush all, some, or none of lines."""

    AorB_ = [*flowcells.keys()][0]
    port_dict = hs.v24[AorB_].port_dict

    # Select lines to flush
    LED('all', 'user')
    confirm = False
    while not confirm:
        flush_ports = input("Flush all, some, or none of the lines? ")
        if flush_ports.strip().lower() == 'all':
            flush_all = True
            flush_ports = [*port_dict.keys()]
            for vp in hs.v24[AorB_].variable_ports:
                if vp in flush_ports:
                    flush_ports.remove(vp)
            confirm = userYN('Confirm flush all lines')
        elif flush_ports.strip().lower() in ['none', 'N', 'n', '']:
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
                        if fp in range(1,hs.v24[AorB_].n_ports+1):
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
                flush_ports = good

    if len(flush_ports) > 0:
        while not userYN('Temporary flowcell(s) locked on to stage'): pass
        while not userYN('All valve input lines in water'): pass
        while not userYN('Ready to flush'): pass

        LED('all', 'startup')

        # Flush ports
        speed = flowcells[AorB_].pump_speed['flush']
        volume = flowcells[AorB_].volume['flush']
        for port in flush_ports:
            if port in hs.v24[AorB_].variable_ports:
                flush_ports.append(*hs.v24[AorB_].port_dict[port].values())
            else:
                hs.message('Flushing ' + str(port))
                for fc in flowcells.values():
                    AorB = fc.position
                    fc.thread = threading.Thread(target=hs.v24[AorB].move,
                                                 args=(port,))
                    fc.thread.start()
                alive = True
                while alive:
                    alive_ = []
                    for fc in flowcells.values():
                        alive_.append(fc.thread.is_alive())
                        alive = any(alive_)
                for fc in flowcells.values():
                    AorB = fc.position
                    fc.thread = threading.Thread(target=hs.p[AorB].pump,
                                                 args=(volume, speed,))
                    fc.thread.start()
                alive = True
                while alive:
                    alive_ = []
                    for fc in flowcells.values():
                        alive_.append(fc.thread.is_alive())
                        alive = any(alive_)

##########################################################
## Flush Lines ###########################################
##########################################################
def do_prime(flush_YorN):
    """Prime lines with all reagents in config if prompted."""

    LED('all', 'user')

    ## Prime lines
    confirm = False
    while not confirm:
        prime_YorN = userYN("Prime lines")
        if prime_YorN:
            confirm = userYN("Confirm prime lines")
        else:
            confirm = userYN("Confirm skip priming lines")
    # LED('all', 'startup')
    # hs.z.move([0,0,0])
    # hs.move_stage_out()
    #LED('all', 'user')

    if prime_YorN:
        if flush_YorN:
            while not userYN('Temporary flowcell(s) locked on to stage'): pass
        while not userYN('Valve input lines in reagents'): pass
        while not userYN('Ready to prime lines'): pass

        #Flush all lines
        LED('all', 'startup')
        while True:
            AorB_ = [*flowcells.keys()][0]
            port_dict = hs.v24[AorB_].port_dict
            speed = flowcells[AorB_].pump_speed['prime']

            for port in port_dict.keys():
                if isinstance(port_dict[port], int):
                    hs.message('Priming ' + str(port))
                    for fc in flowcells.values():
                        port_num = port_dict[port]
                        AorB = fc.position
                        fc.thread = threading.Thread(target=hs.v24[AorB].move,
                                                     args=(port,))
                        fc.thread.start()
                    alive = True
                    while alive:
                        alive_ = []
                        for fc in flowcells.values():
                            alive_.append(fc.thread.is_alive())
                            alive = any(alive_)
                    for fc in flowcells.values():
                        if port_num in hs.v24[AorB].side_ports:
                            volume = fc.volume['side']
                        elif port_num == hs.v24[AorB].sample_port:
                            volume = fc.volume['sample']
                        else:
                            volume = fc.volume['main']
                        AorB = fc.position
                        fc.thread = threading.Thread(target=hs.p[AorB].pump,
                                                     args=(volume, speed,))
                        fc.thread.start()
                    alive = True
                    while alive:
                        alive_ = []
                        for fc in flowcells.values():
                            alive_.append(fc.thread.is_alive())
                            alive = any(alive_)
            break

        # Rinse flowcells
        method = config.get('experiment', 'method')                             # Read method specific info
        method = config[method]
        rinse_port = method.get('rinse', fallback = None)
        rinse = rinse_port in hs.v24[AorB].port_dict
        if rinse_port == port:                                                  # Option to skip rinse if last reagent pump was rinse reagent
            rinse = False

        # Get rinse reagents
        if not rinse:
            LED('all', 'user')
            print('Last reagent pumped was', port)
            if userYN('Rinse flowcell'):
                while not rinse:
                    if rinse_port not in hs.v24[AorB].port_dict:
                        rinse_port = input('Specify rinse reagent: ')
                    rinse = rinse_port in hs.v24[AorB].port_dict
                    if not rinse:
                        print('ERROR::Invalid rinse reagent')
                        print('Choose from:', *list(hs.v24[AorB].port_dict.keys()))
        if rinse:
            # Simultaneously Rinse Flowcells
            for fc in flowcells.values():
                fc.thread = threading.Thread(target=do_rinse,
                                             args=(fc,rinse_port,))
                fc.thread.start()
                alive = True
            # Wait for rinsing to complete
            while alive:
                alive_ = []
                for fc in flowcells.values():
                    alive_.append(fc.thread.is_alive())
                    alive = any(alive_)

        LED('all', 'user')
        while not userYN('Temporary flowcell(s) removed'): pass

    while not userYN('Experiment flowcell(s) locked on to stage'): pass
    if not prime_YorN:
        while not userYN('Valve input lines in reagents'): pass
    while not userYN('Door closed'): pass

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
            LED(AorB, 'awake')

        # Pump reagent into flowcell
        elif instrument == 'PUMP':
            volume = int(command)
            speed = fc.pump_speed['reagent']
            log_message = 'Pumping ' + str(volume) + ' uL'
            fc.thread = threading.Thread(target = hs.p[AorB].pump,
                args = (volume, speed,))
            LED(AorB, 'awake')
        # Incubate flowcell in reagent for set time
        elif instrument == 'HOLD':
            if command.isdigit():
                holdTime = float(command)*60
                log_message = 'Flowcell holding for ' + str(command) + ' min.'
                if hs.virtual:
                    fc.thread = threading.Timer(holdTime/hs.speed_up, fc.endHOLD)
                    #fc.thread = threading.Timer(holdTime, fc.endHOLD)
                else:
                    fc.thread = threading.Timer(holdTime, fc.endHOLD)
            elif command == 'STOP':
                hs.message('PySeq::Paused')
                LED(AorB, 'user')
                input("Press enter to continue...")
                log_message = ('Continuing...')
                fc.thread = threading.Thread(target = do_nothing)

            LED(AorB, 'sleep')
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
            LED(AorB, 'sleep')
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
            LED(AorB, 'imaging')
        elif instrument == 'TEMP':
            log_message = 'Setting temperature to ' + command + ' °C'
            command  = float(command)
            fc.thread = threading.Thread(target = hs.T.set_fc_T,
                args = (AorB,command,))
            fc.temperature = command
        # Block all further processes until user input
        # elif instrument == 'STOP':
        #     hs.message('PySeq::Paused')
        #     LED(AorB, 'user')
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
        fc.restart_recipe()


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
        if hs.AF:
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
        if n_Zplanes > 1:
            obj_start = int(hs.obj.position - hs.nyquist_obj*n_Zplanes*2/3)       # Want 2/3 of planes below opt_ob_pos and 1/3 of planes above
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
            hs.scan_flag = False
            error('Imaging failed.')


    # Reset filters
    for color in hs.optics.cycle_dict.keys():
        if color is 'em':
            hs.optics.move_em_in(True)
        else:
            hs.optics.move_ex(color, 'home')

    if fc.IMAG_counter is not None:
        fc.IMAG_counter += 1



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

def do_rinse(fc, port=None):
    """Rinse flowcell with reagent specified in config file.

       **Parameters:**
       fc (flowcell): The flowcell to rinse.

    """

    method = config.get('experiment', 'method')                                 # Read method specific info
    method = config[method]
    if port is None:
        port = method.get('rinse', fallback = None)
    AorB  = fc.position
    rinse = port in hs.v24[AorB].port_dict

    if rinse:
        LED(fc.position, 'awake')
        # Move valve
        hs.message('PySeq::'+AorB+'::Rinsing flowcell with', port)
        fc.thread = threading.Thread(target = hs.v24[AorB].move, args = (port,))
        fc.thread.start()

        # Pump
        port_num = hs.v24[AorB].port_dict[port]
        if port_num in hs.v24[AorB].side_ports:
            volume = fc.volume['side']
        elif port_num == hs.v24[AorB].sample_port:
            volume = fc.volume['sample']
        else:
            volume = fc.volume['main']
        speed = fc.pump_speed['reagent']
        while fc.thread.is_alive():                                             # Wait till valve has moved
            pass
        fc.thread = threading.Thread(target = hs.p[AorB].pump,
                                       args = (volume, speed,))
    else:
        fc.thread = threading.Thread(target = do_nothing)

##########################################################
## Shut down system ######################################
##########################################################
def do_shutdown():
    """Shutdown the HiSeq and flush all reagent lines if prompted."""

    for fc in flowcells.values():
        while fc.thread.is_alive():
            fc.wait_thread.set()
            time.sleep(10)

    LED('all', 'startup')
    hs.message('PySeq::Shutting down...')


    hs.z.move([0, 0, 0])
    hs.move_stage_out()
    do_flush()
    ##Flush all lines##
    # LED('all', 'user')
    #
    # # flush_YorN = userYN("Flush lines")
    # if flush_YorN:
    #     hs.message('Lock temporary flowcell on  stage')
    #     hs.message('Place all valve input lines in PBS/water')
    #     input('Press enter to continue...')
    #
    #     LED('all', 'startup')
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
    #     LED('all', 'user')


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
    LED('all', 'off')



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

    # Defaults that can be overided
    config.read_dict({'experiment' : {'log path': 'logs',
                                      'image path': 'images'}
                      })
    # Open config file
    if os.path.isfile(args['config']):
         config.read(args['config'])
    else:
        error('ConfigFile::Does not exist')
        sys.exit()
    # Set output path
    config['experiment']['save path'] = args['output']
    # Set experiment name
    config['experiment']['experiment name'] = args['name']
    # save user valve
    USERVALVE = False
    if config.has_section('reagents'):
        valve = config['reagents'].items()
        if len(valve) > 0:
            USERVALVE = True

    # Get method specific configuration
    method = config['experiment']['method']
    if method in methods.get_methods():
        config_path, recipe_path = methods.return_method(method)
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
    if not methods.check_settings(config[method]):
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
    user_config.read(args['config'])
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
    logger = setup_logger()                                                     # Create logfiles
    port_dict = check_ports()                                                   # Check ports in configuration file
    first_line, IMAG_counter, z_planes = check_instructions()                   # Checks instruction file is correct and makes sense
    flowcells = setup_flowcells(first_line, IMAG_counter)                       # Create flowcells
    hs = configure_instrument(IMAG_counter, port_dict)
    confirm_settings(z_planes)
    hs = initialize_hs(IMAG_counter)                                            # Initialize HiSeq, takes a few minutes

    if n_errors is 0:
        flush_YorN = do_flush()                                                 # Ask to flush out lines
        do_prime(flush_YorN)                                                    # Ask to prime lines
        if not userYN('Start experiment'):
            sys.exit()

        # Do prerecipe or Initialize Flowcells
        for fc in flowcells.values():
            if fc.prerecipe_path:
                fc.pre_recipe()
            else:
                fc.restart_recipe()

        cycles_complete = False
        while not cycles_complete:
            stuck = 0
            complete = 0

            for fc in flowcells.values():
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
