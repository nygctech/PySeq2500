#!/usr/bin/python
# TO DO: NEW LOG FILE

import time
import logging
import os
import sys
import configparser
import threading

import pyseq

##########################################################
## Default Values for Configuration File #################
##########################################################
config = configparser.ConfigParser()
config.read_dict({'experiment' : {'save path': 'C:\\Users\\hisequser\\Documents\\',
                                 'log path': 'logs',
                                 'image path': 'images'}
                 })
                 
                                 
     

##########################################################
## Flowcell Class ########################################
##########################################################
class flowcell():

    def __init__(self, position):

        self.recipe_path = None
        self.recipe = None
        self.first_line = None
        self.cycle = 0                          # Current cycle
        self.total_cycles = 0                   # Total number of cycles for experiment
        self.history = [[],[],[]]               # summary of events in flowcell history
        self.imaging = False
        self.sections = {}                      # coordinates of flowcell of sections to image
        self.stage = {}                         # stage positioning info for each section
        self.thread = None                      # threading to do parallel actions on flowcells         
        self.signal_event = None                # defines event that signals the next flowcell to continue
        self.wait_thread = threading.Event()    # blocks next flowcell until current flowcell reaches signal event
        self.waits_for = None                   # position of the flowcell that signals current flowcell to continue
        self.dead_volume = None
        self.pump_speed = {}
        self.flush_volume = None
        
        while position not in ['A', 'B']:
            print(self.name + ' must be at position A or B')
            position = input('Enter position of ' + self.name + ' : ')
            
        self.position = position

    # Record history of events on flow cell, returns time stamp of last event
    def addEvent(self, instrument, command):
        self.history[0].append(time.time())     # time stamp
        self.history[1].append(instrument)      # instrument (valv, pump, hold, wait, imag)
        self.history[2].append(command)         # details such hold time, buffer, event to wait for

        
        return self.history[0][-1]              # return time stamp of last event

    # restart the recipe for the flowcell, returns number of cycles flowcell completed
    def restart_recipe(self):
        if self.recipe is not None:
            self.recipe.close()
        self.recipe = open(self.recipe_path)
        self.cycle += 1
        if self.cycle > self.total_cycles:
            end_message = 'Completed ' + self.total_cycles + ' on flowcell ' + self.position
            self.thread = threading.Thread(target = logger.log, args = (21, end_message,))
            thread_id = self.thread.start()            
        else:
            restart_message = 'Starting cycle ' + str(self.cycle) + ' on flowcell ' + self.position
            self.thread = threading.Thread(target = logger.log, args = (21, restart_message,))
            thread_id = self.thread.start()

        return self.cycle

    # Ends hold for incubations of flowcell in buffer, returns boolean of if flowcell is held
    def endHOLD(self):
        self.hold = False
        logger.log(21,'Flowcell ' + self.position + ' HOLD stopped at ' + time.asctime())

        return self.hold


        
        
##########################################################
## Setup Flowcells #######################################
##########################################################
        
def setup_flowcells(config, first_line):
    method = config.get('experiment', 'method')
    method = config[method]
    
    flowcells = {}
    for sect_name in config['sections']:
        AorB = config['sections'][sect_name]
        # Create flowcell if it doesn't exist
        if AorB not in flowcells.keys():
            flowcells[AorB] = flowcell(AorB)
            flowcells[AorB].recipe_path = method['recipe']
            flowcells[AorB].dead_volume = int(method.get('dead volume', 100))
            flowcells[AorB].flush_volume = int(method.get('flush volume', 10))
            flowcells[AorB].pump_speed['flush'] = int(method.get('flush speed', 700))
            flowcells[AorB].pump_speed['reagent'] = int(method.get('reagent speed', 40))
            flowcells[AorB].first_line = first_line
            flowcells[AorB].total_cycles = int(config.get('experiment','cycles'))

        # Add section to flowcell
        # Make sure section name : flowcell and section name: position match
        if sect_name in config['section position']:
            #  Add new section to flowcell
            if sect_name in flowcells[AorB].sections:
                print(sect_name + ' already on flowcell ' + AorB)
                print('check config file for section name duplications')
                sys.exit()
            if sect_name:
                sect_position = config['section position'][sect_name]
                sect_position = sect_position.split(',')
                flowcells[AorB].sections[sect_name] = []
                flowcells[AorB].stage[sect_name] = {}
                for pos in sect_position:
                    flowcells[AorB].sections[sect_name].append(int(pos))
        else:
            print(sect_name + ' does not have a position, check config file')
            sys.exit()

        # if runnning mulitiple flowcells, define prior flowcell signals to next flowcell
        if len(flowcells) > 1:
            flowcell_list = [*flowcells]
            for fc in flowcells.keys():
                flowcells[fc].waits_for = flowcell_list[flowcell_list.index(fc)-1]
                
    return flowcells


                   
##########################################################
## Parse lines from recipe ###############################
##########################################################
def parse_line(line):
        comment_character = '#'
        delimiter = '\t'
        no_comment = line.split(comment_character)[0]   # remove comment
        sections = no_comment.split(delimiter)
        instrument = sections[0]                        # first section is port name
        instrument = instrument[0:4]                    # instrument identified by first 4 characters
        command = sections[1]                           # second section is port name
        command = command.replace(' ','')               # remove space
        
        return instrument, command


##########################################################
## Setup Logging #########################################
##########################################################
def setup_logger(experiment):
    # Get experiment info from config file
    experiment_name = experiment.get('experiment name', )
    save_path = experiment['save path'] + '\\' + experiment_name
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    log_path = experiment['log path']
    log_path = save_path + '\\' + log_path
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(10)

    # Create console handler
    c_handler = logging.StreamHandler()
    c_handler.setLevel(21)
    # Create file handler
    f_log_name = log_path + '\\' + experiment_name + '.log'
    f_handler = logging.FileHandler(f_log_name)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(asctime)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

##########################################################
## Setup HiSeq ###########################################
##########################################################
def initialize_hs(experiment):
    
    hs = pyseq.HiSeq(logger)
    hs.initializeCams()
    hs.initializeInstruments()

    save_path = experiment['save path']
    experiment_name = experiment['experiment name']
    save_path = save_path + '\\' + experiment_name
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    image_path = experiment['image path']
    image_path = save_path + '\\' + image_path
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    
    hs.image_path = image_path + '\\'

    return hs


##########################################################
## Check Instructions ####################################
##########################################################
def check_instructions(config):
    method = config.get('experiment', 'method')
    method = config[method]
    
    first_port = method.get('first port', None)                   # Get first reagent to use in recipe
    variable_ports = method.get('variable reagents')
    variable_ports = variable_ports.split(',')
    
    valid_wait = []
    ports = []
    for port in config['valve24'].items():
        ports.append(port[1])
    for port in variable_ports:
        ports.append(port.replace(' ',''))
    valid_wait = ports
    valid_wait.append('IMAG')
    valid_wait.append('STOP')

    f = open(method['recipe'])
    line_num = 1
    error = 0
    first_line = 0

    
    def message(text, error):
        try:
            logger(21,text)
        except:
            print(text)
        error += 1
        return error     

    for line in f:
            instrument, command = parse_line(line)
            
            if instrument == 'PORT':
                # Make sure ports in instruction files exist in port dictionary in config file
                if command not in ports:
                    error = message(command + ' port on line ' + str(line_num) + ' does not exist.\n', error)
                    
                #Find line to start at for first cycle
                if first_line == 0 and first_port is not None:
                    if command.find(first_port) != -1:
                        first_line = line_num
                        
            # Make sure pump volume is a number
            elif instrument == 'PUMP':
                if command.isdigit() == False:
                    error = message('Invalid volume on line ' + str(line_num) + '\n', error)
                    
            # Make sure wait command is valid
            elif instrument == 'WAIT':
                if command not in valid_wait:
                    error = message('Invalid wait command on line ' + str(line_num) + '\n', error)

            # Make sure z planes is a number
            elif instrument == 'IMAG':
                 if command.isdigit() == False:
                    error = message('Invalid number of z planes on line ' + str(line_num) + '\n', error)

            # Make sure hold time (minutes) is a number
            elif instrument == 'HOLD':
                if command.isdigit() == False:
                    error = message('Invalid time on line ' + str(line_num) + '\n', error)

            # Warn user that HiSeq will completely stop with this command        
            elif instrument == 'STOP':
                print('HiSeq will complete stop until user input at line ' + str(line_num) + '\n')

            # Make sure the instrument name is valid
            else:
                error = message('Bad instrument name on line ' + str(line_num) + '\n', error)

            line_num += 1

    if error > 0:
            print(str(error) + " errors in instruction file")
            f.close() #close instruction file
            sys.exit()
    else:
            print("Good instruction file")
            f.close() #close instruction file
            return first_line
        
##########################################################
## Check Ports ###########################################
##########################################################
def check_ports(config):
    method = config.get('experiment', 'method')
    method = config[method]
    total_cycles = int(config.get('experiment', 'cycles'))
    
    # Get cycle and port information from configuration file
    valve = config['valve24']                           # Get dictionary of port number of valve : name of reagent
    cycle_variables = method['variable reagents']       # Get list of port names in recipe that change every cycle
    cycle_reagents = config['cycles'].items()           # Get variable reagents that change with each cycle

    error = 0
    port_dict = {}

    # Make sure there are no duplicated names in the valve                
    if len(valve.values()) != len(set(valve.values())):
        print('Port names are not unique in configuration file. Rename or remove duplications.')
        error += 1
        
    if len(valve) > 0:
        # Create port dictionary
        for port in valve.keys():
            port_dict[valve[port]] = int(port)              

        # Add cycle variable port dictionary
        cycle_variables = cycle_variables.split(',')
        for variable in cycle_variables:
            variable = variable.replace(' ','')
            port_dict[variable] = {}

        # Fill cycle variable port dictionary with cycle: reagent name 
        for cycle in cycle_reagents:
            if cycle[1] in valve.values():
                variable = cycle[1].split(':')[0]
                if variable in cycle_variables:
                    if variable not in port_dict:
                        port_dict[variable] = {}
                    port_dict[variable][int(cycle[0])] = cycle[1]
                else:
                    print(variable + ' does not exist in recipe')
                    error += 1
            else:
                print('Cycle reagent: ' + cycle_reagents[cycle] + ' does not exist on valve')
                error += 1

        # Check number of reagents in variable reagents matches nunmber of total cycles
        for variable in cycle_variables:
            variable = variable.replace(' ','')
            if len(port_dict[variable]) != total_cycles:
                print('Number of ' + variable + ' reagents does not match experiment cycles')
                error += 1
            
    else:
        print('Dictionary of port number : reagent name does not exist under valve24 of configuration file')
        error += 1
            
    if error > 0:
        print(str(error) + ' errors in configuration file')
        sys.exit()
    else:
        print('Ports checked without errors')
        return port_dict


##########################################################
## Flush Lines ###########################################
##########################################################
def do_flush(flowcells, hs):
    
    ## Flush lines
    flush_YorN = input("Flush lines? Y/N = ")
    if flush_YorN == 'Y':
        print("Place temporary flowcell on  stage")
        print("Place all valve input lines in PBS/water")
        input("Press enter to continue...")
        #Flush all lines
        for fc in flowcells.keys():
            volume = fc.dead_volume*fc.flush_volume
            speed = fc.speed['flush']
            for port in hs.v24[fc].port_dict.keys():
                hs.v24[fc].move(port)
                hs.p[fc].pump(volume, speed)
                
        print("Replace temporary flowcell with experiment flowcell on stage")
        print("Place all valve input lines in correct reagent")
        input("Press enter to continue...")
    else:
        print("Place experiment flowcells on stage")
        input("Press enter to continue...")

##########################################################
## iterate over lines, send to pump, and print response ##
##########################################################
def do_recipe(fc, hs):

    AorB = fc.position
    fc.thread = None

    # Skip to first line of recipe on initial cycle
    if fc.cycle == 1 and fc.first_line is not None:
        for i in range(fc.first_line-1):
            line = fc.recipe.readline()
        fc.first_line = None
        
            
    # get instrument and command
    line = fc.recipe.readline()
    if line:
        instrument, command = parse_line(line)                     
        
        # Move reagent valve
        if instrument == 'PORT':
            if command in hs.v24[AorB].variable_ports:
                command = hs.v24[AorB].port_dict[command][fc.cycle]
            logger.log(21, 'Move to ' + command + ' on ' + hs.v24[AorB].name)
            fc.thread = threading.Thread(target = hs.v24[AorB].move, args = (command,))
            print(fc.thread)
        # Pump reagent into flowcell
        elif instrument == 'PUMP':
            volume = int(command)*fc.dead_volume
            speed = fc.pump_speed['reagent']
            logger.log(21, 'Pumping ' + str(volume) + ' uL to flowcell ' + AorB)
            fc.thread = threading.Thread(target = hs.p[AorB].pump, args = (volume, speed,))
        # Incubate flowcell in reagent for set time
        elif instrument == 'HOLD':
            holdTime = float(command)*60
            logger.log(21, 'Flowcell ' + AorB + ' holding for ' + str(command) + ' min.')
            fc.thread = threading.Timer(holdTime, fc.endHOLD)
        # Wait for other flowcell to finish event before continuing with current flowcell
        elif instrument == 'WAIT':
            logger.log(21, 'Flowcell ' + fc.position + ' waiting for ' + command + ' in other flowcell')
            fc.thread = threading.Thread(target = WAIT, args = (AorB, command,))
        # Image the flowcell
        elif instrument == 'IMAG':
            logger.log(21, 'Imaging flowcell ' + AorB)
            fc.thread = threading.Thread(target = IMAG, args = (fc, hs, int(command),))
        # Block all further processes until user input
        elif instrument == 'STOP':
            logger.log(21,'Paused')
            input("press enter to continue...")
            logger.log(21,'Continuing...')

            
        #Signal to other flowcell that current flowcell reached signal event
        if fc.signal_event == instrument or fc.signal_event == command:
            fc.wait_thread.set()
            fc.signal_event = None
            
        # Start new action on current flowcell   
        if fc.thread is not None and fc.cycle <= fc.total_cycles:
            fc.addEvent(instrument, command)
            thread_id = fc.thread.start()
        elif fc.thread is not None and fc.cycle > fc.total_cycles:
            fc.thread = threading.Thread(target = WAIT, args = (AorB, None,))

    # End of recipe 
    else:
        fc.restart_recipe()
        

##########################################################
## Image flowcell ########################################
##########################################################
def IMAG(fc, hs, n_Zplanes):

    fc.imaging = True
    start = time.time()
    for section in fc.sections:
        # Find/Move to focal z stage position
        if fc.stage[section]['z pos'] is None:
            logger.log(21, 'Finding rough focus of ' +
                   str(section) + ' on flowcell ' + fc.position)
                   
            hs.y.move(fc.stage[section]['y center'])
            hs.x.move(fc.stage[section]['x center'])
            Z,C = hs.rough_focus()
            fc.stage[section]['z pos'] = hs.z.position
        else:
            hs.z.move(fc.stage[section]['z pos'])

        # Find/Move to focal obj stage position    
        if fc.stage[section]['obj pos'] is None:
            logger.log(21, 'Finding fine focus of ' +
                   str(section) + ' on flowcell ' + fc.position)
                   
            hs.y.move(fc.stage[section]['y center'])
            hs.x.move(fc.stage[section]['x center'])
            Z,C = hs.fine_focus()
            fc.stage[section]['obj pos'] = hs.obj.position
        else:
            hs.obj.move(fc.stage[section]['obj pos'])

        x_pos = fc.stage[section]['x initial']
        y_pos = fc.stage[section]['y initial']
        n_scans = fc.stage[section]['n scans']
        n_frames = fc.stage[section]['n frames']
        if n_Zplanes > 1:
            obj_start = int(hs.obj.position - hs.nyquist_obj*n_Zplanes/2)
            obj_step = hs.nyquist_obj
            obj_stop = int(hs.obj.position + hs.nyquist_obj*n_Zplanes/2)
        else:
            obj_start = hs.obj.position
            obj_step = 1000
            obj_stop = hs.obj.position + 10

        image_name = fc.position
        image_name = image_name + '_' + str(section)
        image_name = image_name + '_' + 'c' + str(fc.cycle)

        # Scan section on flowcell
        logger.log(21, 'Imaging ' + str(section) + ' on flowcell ' + fc.position)
        scan_time = hs.scan(x_pos, y_pos,
                            obj_start, obj_stop, obj_step,
                            n_scans, n_frames, image_name)
        scan_time = str(int(scan_time/60))
        logger.log(21, 'Took ' + scan_time + ' minutes ' + 'imaging ' +
                   str(section) + ' on flowcell ' + fc.position)

    fc.imaging = False
    stop = time.time()
                   
    return stop-start

# holds current flowcell until an event in the signal flowcell, returns time held
def WAIT(AorB, event):
    signaling_fc = flowcells[AorB].waits_for
    
    start = time.time()
    flowcells[signaling_fc].signal_event = event                        # Set the signal event in the signal flowcell
    flowcells[signaling_fc].wait_thread.wait()                          # Block until signal event in signal flowcell
    logger.log(21, 'Flowcell ' + AorB + ' ready to continue')
    flowcells[signaling_fc].wait_thread.clear()                         # Reset wait event
    stop = time.time()
    
    return stop-start
    
##########################################################
## Shut down system ######################################
##########################################################
def do_shutdown(flowcells, hs):
    
    logger.log(21,'Shutting down...')
    for fc in flowcells.values():
        fc.wait_thread.set()
    
    ##Flush all lines##
    flush_YorN = input("Flush lines? Y/N = ")
    if flush_YorN == 'Y':
        print("Place temporary flowcell on  stage")
        print("Place all valve input lines in PBS/water")
        input("Press enter to continue...")
        volume = fc.dead_volume*fc.flush_volume
        speed = fc.speed['flush']
        for fc in flowcells.keys():
            for port in hs.v24[fc].port_dict.keys():
                hs.v24[fc].move(port)
                hs.p[fc].pump(volume, speed)
            ##Return pump to top and NO port##
            hs.p[fc].command('OA0R')
            hs.p[fc].command('IR')
    

##########################################################
## Free Flowcells ########################################
##########################################################
def free_fc(flowcells, experiment):

        first_fc = experiment.get('first flowcell', 'A')       # Get which flowcell is to be first
        
        if len(flowcells) == 1:
            fc = flowcells[[*flowcells][0]]
            fc.wait_thread.set()
            fc.signal_event = None
        else:           
            if first_fc in flowcells:
                fc = flowcells[first_fc]
                flowcells[fc.waits_for].wait_thread.set()
                flowcells[fc.waits_for].signal_event = None

        logger.log(21, 'Flowcells are waiting on each other, starting flowcell ' + fc.position)

        return fc.position
##########################################################
## Initialize Flowcells ##################################
##########################################################
def integrate_fc_and_hs(flowcells, hs, config):
    
    method = config.get('experiment', 'method')         # Read method specific info
    method = config[method]
    variable_ports = method['variable reagents']
    variable_ports = variable_ports.split(',')
    
    n_barrels = int(method.get('barrels per lane', 8))  # Get method specific pump barrels per lane, fallback to 8     

    for fc in flowcells.values():
        AorB = fc.position
        hs.v24[AorB].port_dict = port_dict              # Assign ports on HiSeq
        for variable in variable_ports:                 # Assign variable ports
            variable = variable.replace(' ','')
            hs.v24[AorB].variable_ports.append(variable)
        hs.p[AorB].n_barrels = n_barrels                # Assign barrels per lane to pump
        for section in fc.sections:                     # Convert coordinate sections on flowcell to stage info
            stage = hs.position(fc.position, fc.sections[section])
            fc.stage[section]['x center'] = stage[0]
            fc.stage[section]['y center'] = stage[1]
            fc.stage[section]['x initial'] = stage[2]
            fc.stage[section]['y initial'] = stage[3]
            fc.stage[section]['n scans'] = stage[4]
            fc.stage[section]['n frames'] = stage[5]
            fc.stage[section]['z pos'] = None
            fc.stage[section]['obj pos'] = None
        fc.restart_recipe()                             # Open recipe and increase cycle counter to 1
        time.sleep(0.1)
        

    
###################################
## Run System #####################
###################################
config.read('config.cfg')                           # Open config file
experiment = config['experiment']                   # Read experiment specific info


logger = setup_logger(experiment)                   # Create logfiles
port_dict = check_ports(config)                     # Check ports in configuration file
first_line = check_instructions(config)             # Checks instruction file is correct and makes sense                  
flowcells = setup_flowcells(config, first_line)     # Create flowcells
hs = initialize_hs(experiment)                      # Initialize HiSeq, takes a few minutes 
integrate_fc_and_hs(flowcells, hs, config)          # Integrate flowcell info with hs 
                               
do_flush(flowcells, hs)                             # Flush out lines

cycles_complete = False

while not cycles_complete:
    stuck = 0
    complete = 0
    
    for fc in flowcells.values():                       
        if not fc.thread.is_alive():                # flowcell not busy, do next step in recipe 
            do_recipe(fc, hs)                                               
        
        if fc.signal_event:                         # check if flowcells are waiting on each other                             
            stuck += 1
        
        if fc.cycle == fc.total_cycles:             # check if all cycles are complete on flowcell
            complete += 1
                                             
    if stuck == len(flowcells):                     # Start the first flowcell if they are waiting on each other
        free_fc(flowcells, experiment)
            
    if complete == len(flowcells):                  # Exit while loop
        cycles_complete = True

    
do_shutdown(flowcells, hs)                          # Shutdown  HiSeq

                 

