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
# Defaults that can be overided
config.read_dict({'experiment' : {'save path': 'C:\\Users\\Public\\Documents\\',
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
            end_message = self.position + '::Completed ' + str(self.total_cycles) + ' cycles'
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
        logger.log(21, fc.position+'::cycle'+str(fc.cycle)+'::Hold stopped')

        return self.hold


        
        
##########################################################
## Setup Flowcells #######################################
##########################################################
        
def setup_flowcells(first_line):
    experiment = config['experiment']
    method = experiment['method']
    method = config[method]
    
    flowcells = {}
    for sect_name in config['sections']:
        AorB = config['sections'][sect_name]
        # Create flowcell if it doesn't exist
        if AorB not in flowcells.keys():
            flowcells[AorB] = flowcell(AorB)
            flowcells[AorB].recipe_path = method['recipe']
            flowcells[AorB].flush_volume = int(method.get('flush volume', fallback=2000))
            flowcells[AorB].pump_speed['flush'] = int(method.get('flush speed', fallback=700))
            flowcells[AorB].pump_speed['reagent'] = int(method.get('reagent speed', fallback=40))
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
                    flowcells[AorB].sections[sect_name].append(float(pos))
        else:
            print(sect_name + ' does not have a position, check config file')
            sys.exit()

        # if runnning mulitiple flowcells...
        # Define first flowcell
        # Define prior flowcell signals to next flowcell
        if len(flowcells) > 1:
            flowcell_list = [*flowcells]
            for fc in flowcells.keys():
                flowcells[fc].waits_for = flowcell_list[flowcell_list.index(fc)-1]
            if experiment['first flowcell'] not in flowcells:
                print('First flowcell does not exist')
                sys.exit()
                
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
def setup_logger():
    
    # Get experiment info from config file
    experiment = config['experiment']
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
    
    # Save copy of config with log
    with open(log_path+'\\config.cfg', 'w') as configfile:
        config.write(configfile)
        
    return logger

##########################################################
## Setup HiSeq ###########################################
##########################################################
def initialize_hs():

    hs = pyseq.HiSeq(logger)
    hs.initializeCams(logger)
    hs.initializeInstruments()
    
    experiment = config['experiment']
    method = config[experiment['method']]
    
    hs.l1.set_power(int(method.get('laser power', fallback = 100)))
    hs.l2.set_power(int(method.get('laser power', fallback = 100)))
    
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
def check_instructions():
    method = config.get('experiment', 'method')
    method = config[method]
    
    first_port = method.get('first port', fallback = None)              # Get first reagent to use in recipe
    try:
        first_port = int(first_port)
        first_line = first_port
        first_port = None
    except:
        first_line = 0
        
    variable_ports = method.get('variable reagents', fallback  = None)
    
    
    valid_wait = []
    ports = []
    for port in config['valve24'].items():
        ports.append(port[1])
    if variable_ports is not None:
        variable_ports = variable_ports.split(',')
        for port in variable_ports:
            ports.append(port.replace(' ',''))
    valid_wait = ports
    valid_wait.append('IMAG')
    valid_wait.append('STOP')

    f = open(method['recipe'])
    line_num = 1
    error = 0

    
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
def check_ports():
    method = config.get('experiment', 'method')
    method = config[method]
    total_cycles = int(config.get('experiment', 'cycles'))
    
    # Get cycle and port information from configuration file
    valve = config['valve24']                                               # Get dictionary of port number of valve : name of reagent
    cycle_variables = method.get('variable reagents', fallback = None )     # Get list of port names in recipe that change every cycle
    cycle_reagents = config['cycles'].items()                               # Get variable reagents that change with each cycle

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
        if cycle_variables is not None:
            cycle_variables = cycle_variables.split(',')
            for variable in cycle_variables:
                variable = variable.replace(' ','')
                if variable in port_dict:
                    print('Variable ' + variable + ' can not be a reagent!')
                    error = error+1
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
                        print(variable + ' not listed as variable reagent in config')
                        error += 1
                else:
                    print('Cycle reagent: ' + reagent + ' does not exist on valve')
                    error += 1
                    
            # Check number of reagents in variable reagents matches nunmber of total cycles
            for variable in cycle_variables:
                variable = variable.replace(' ','')
                if len(port_dict[variable]) != total_cycles:
                    print('Number of ' + variable + ' reagents does not match experiment cycles')
                    error += 1
                                  
        else:
            response = True
            while response:
                response = input('Are all ports the same for every cycle? Y/N: ')
                if response == 'Y':
                    response = False
                elif response == 'N':
                    sys.exit()
                    
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
def do_flush():
    
    ## Flush lines
    flush_YorN = input("Prime lines? Y/N = ")
    hs.z.move([0,0,0])
    hs.move_stage_out()
    if flush_YorN == 'Y':
        print("Lock temporary flowcell(s) on to stage")
        print("Place all valve input lines in PBS/water")
        input("Press enter to continue...")
        #Flush all lines
        for fc in flowcells.keys():
            volume = flowcells[fc].flush_volume
            speed = flowcells[fc].pump_speed['flush']
            for port in hs.v24[fc].port_dict.keys():
                if isinstance(port_dict[port], int):
                    print('Priming ' + str(port))
                    hs.v24[fc].move(port)
                    hs.p[fc].pump(volume, speed)
                
        print("Replace temporary flowcell with experiment flowcell and lock on to stage")
        print("Place all valve input lines in correct reagent")
        input("Press enter to continue...")
    else:
        print("Lock experiment flowcells on to stage")
        input("Press enter to continue...")

    for fc in flowcells.values():
        fc.restart_recipe()

#######
def do_nothing():
    pass


##########################################################
## iterate over lines, send to pump, and print response ##
##########################################################
def do_recipe(fc):

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
            if command in hs.v24[AorB].variable_ports and fc.cycle <= fc.total_cycles:
                command = hs.v24[AorB].port_dict[command][fc.cycle]
            log_message = 'Move to ' + command
            fc.thread = threading.Thread(target = hs.v24[AorB].move, args = (command,))
        # Pump reagent into flowcell
        elif instrument == 'PUMP':
            volume = int(command)
            speed = fc.pump_speed['reagent']
            log_message = 'Pumping ' + str(volume) + ' uL'
            fc.thread = threading.Thread(target = hs.p[AorB].pump, args = (volume, speed,))
        # Incubate flowcell in reagent for set time
        elif instrument == 'HOLD':
            holdTime = float(command)*60
            log_message = 'Flowcell holding for ' + str(command) + ' min.'
            fc.thread = threading.Timer(holdTime, fc.endHOLD)
        # Wait for other flowcell to finish event before continuing with current flowcell
        elif instrument == 'WAIT':
            if fc.waits_for is not None:
                log_message = 'Flowcell waiting for ' + command
                fc.thread = threading.Thread(target = WAIT, args = (AorB, command,))
            else:
                log_message = 'Skipping waiting for ' + command
                fc.thread = threading.Thread(target = do_nothing)
        # Image the flowcell
        elif instrument == 'IMAG':
            log_message = 'Imaging flowcell'
            fc.thread = threading.Thread(target = IMAG, args = (fc,int( command),))
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
            logger.log(21, AorB+'::cycle'+str(fc.cycle)+'::'+log_message)
            thread_id = fc.thread.start()
        elif fc.thread is not None and fc.cycle > fc.total_cycles:
            fc.thread = threading.Thread(target = WAIT, args = (AorB, None,))

    # End of recipe 
    elif fc.cycle <= fc.total_cycles:
        fc.restart_recipe()
    elif fc.cycle > fc.total_cycles:
        fc.thread =  threading.Thread(target = time.sleep, args = (10,))
        fc.thread.start()
        
##########################################################
## Image flowcell ########################################
##########################################################
def IMAG(fc, n_Zplanes):
    AorB = fc.position
    cycle = str(fc.cycle)
    fc.imaging = True
    start = time.time()

    
    for section in fc.sections:
        x_center = fc.stage[section]['x center']
        y_center = fc.stage[section]['y center']
        x_pos = fc.stage[section]['x initial']
        y_pos = fc.stage[section]['y initial']
        n_scans = fc.stage[section]['n scans']
        n_frames = fc.stage[section]['n frames']
        
        # Find/Move to focal z stage position
        if fc.stage[section]['z pos'] is None:
            logger.log(21, AorB+'::Finding rough focus of ' + str(section))
                   
            hs.y.move(y_center)
            hs.x.move(x_center)
            hs.optics.move_ex(1, 0.6)
            hs.optics.move_ex(2, 0.9)
            hs.optics.move_em_in(True)
            Z,C = hs.rough_focus()
            fc.stage[section]['z pos'] = hs.z.position[:]
        else:
            hs.z.move(fc.stage[section]['z pos'])

        # Find/Move to focal obj stage position,
        # Edited to find focus every cycle change -1 to None if only want initial cycle
        if fc.stage[section]['obj pos'] is not -1:
            logger.log(21, AorB+'::Finding fine focus of ' + str(section))
                
            hs.y.move(y_center)
            hs.x.move(x_center)
            hs.optics.move_ex(1, 0.6)
            hs.optics.move_ex(2, 0.9)
            hs.optics.move_em_in(True)
            Z,C = hs.fine_focus()
            fc.stage[section]['obj pos'] = hs.obj.position
        else:
            hs.obj.move(fc.stage[section]['obj pos'])

        # Optimize filter
        logger.log(21, AorB+'::Finding optimal filter')
        hs.y.move(y_pos)
        hs.x.move(x_center)
        opt_filter = hs.optimize_filter(32)                                         #Find optimal filter set on 32 frames on image
        hs.optics.move_ex(1, opt_filter[0])
        hs.optics.move_ex(2, opt_filter[1])
        hs.optics.move_em_in(True)
        fc.ex_filter1 = opt_filter[0]
        fc.ex_filter2 = opt_filter[1]


        
        if n_Zplanes > 1:
            obj_start = int(hs.obj.position - hs.nyquist_obj*n_Zplanes/2)
            obj_step = hs.nyquist_obj
            obj_stop = int(hs.obj.position + hs.nyquist_obj*n_Zplanes/2)
        else:
            obj_start = hs.obj.position
            obj_step = 1000
            obj_stop = hs.obj.position + 10
        
        image_name = AorB
        image_name = image_name + '_' + str(section)
        image_name = image_name + '_' + 'c' + cycle

        # Scan section on flowcell
        logger.log(21, AorB + '::cycle'+cycle+'::Imaging ' + str(section))
        scan_time = hs.scan(x_pos, y_pos,
                            obj_start, obj_stop, obj_step,
                            n_scans, n_frames, image_name)
        scan_time = str(int(scan_time/60))
        logger.log(21, AorB+'::cycle'+cycle+'::Took ' + scan_time + ' minutes ' + 'imaging ' + str(section))

    fc.imaging = False
    stop = time.time()
    hs.z.move([0,0,0])
                   
    return stop-start

# holds current flowcell until an event in the signal flowcell, returns time held
def WAIT(AorB, event):
    signaling_fc = flowcells[AorB].waits_for
    cycle = str(flowcells[AorB].cycle)
    start = time.time()
    flowcells[signaling_fc].signal_event = event                        # Set the signal event in the signal flowcell
    flowcells[signaling_fc].wait_thread.wait()                          # Block until signal event in signal flowcell
    logger.log(21, AorB+ '::cycle'+cycle+'::Flowcell ready to continue')
    flowcells[signaling_fc].wait_thread.clear()                         # Reset wait event
    stop = time.time()
    
    return stop-start
    
##########################################################
## Shut down system ######################################
##########################################################
def do_shutdown():
    
    logger.log(21,'Shutting down...')
    for fc in flowcells.values():
        fc.wait_thread.set()
        
    hs.z.move([0, 0, 0])
    hs.move_stage_out()
    ##Flush all lines##
    flush_YorN = input("Flush lines? Y/N = ")
    if flush_YorN == 'Y':
        print("Lock temporary flowcell on  stage")
        print("Place all valve input lines in PBS/water")
        input("Press enter to continue...")

        for fc in flowcells.keys():
            volume = flowcells[fc].flush_volume
            speed = flowcells[fc].pump_speed['flush']
            for port in hs.v24[fc].port_dict.keys():
                if isinstance(port_dict[port], int):
                    hs.v24[fc].move(port)
                    hs.p[fc].pump(volume, speed)
            ##Return pump to top and NO port##
            hs.p[fc].command('OA0R')
            hs.p[fc].command('IR')
    else:
        print('Retrieve experiment flowcells')
        input('Press any key to finish shutting down')

    # Write flowcell histories to file
    experiment_name = config.get('experiment','experiment name')
    save_path = config.get('experiment','save path') + experiment_name
    log_path = save_path + '\\' + config.get('experiment','log path')
    for fc in flowcells.values():
        AorB = fc.position
        with open(log_path+'\\Flowcell'+AorB+'.log', 'w') as fc_file:
            for i in range(len(fc.history[0])):
                fc_file.write(str(fc.history[0][i])+' '+
                              str(fc.history[1][i])+' '+
                              str(fc.history[2][i])+'\n')

    # Turn off y stage motor   
    hs.y.command('OFF')                                     

    

##########################################################
## Free Flowcells ########################################
##########################################################
def free_fc():

        # Get which flowcell is to be first
        experiment = config['experiment']
        first_fc = experiment.get('first flowcell', fallback = 'A')       
        
        if len(flowcells) == 1:
            fc = flowcells[[*flowcells][0]]
            fc.wait_thread.set()
            fc.signal_event = None
        else:           
            fc = flowcells[first_fc]
            flowcells[fc.waits_for].wait_thread.set()
            flowcells[fc.waits_for].signal_event = None

        logger.log(21, 'Flowcells are waiting on each other, starting flowcell ' + fc.position)

        return fc.position
##########################################################
## Initialize Flowcells ##################################
##########################################################
def integrate_fc_and_hs(port_dict):
    
    method = config.get('experiment', 'method')         # Read method specific info
    method = config[method]
    variable_ports = method.get('variable reagents', fallback = None)
    z_pos = config['z position']
    obj_pos = config['obj position']
    
    n_barrels = int(method.get('barrels per lane', 8))  # Get method specific pump barrels per lane, fallback to 8     

    for fc in flowcells.values():
        AorB = fc.position
        hs.v24[AorB].port_dict = port_dict              # Assign ports on HiSeq
        if variable_ports is not None:
            variable_ports = variable_ports.split(',')
            for variable in variable_ports:             # Assign variable ports
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
            fc.stage[section]['z pos'] = z_pos.get(section,fallback=None)
            fc.stage[section]['obj pos'] = obj_pos.get(section,fallback=None)

    
###################################
## Run System #####################
###################################
config.read('config.cfg')                           # Open config file               
logger = setup_logger()                             # Create logfiles
port_dict = check_ports()                           # Check ports in configuration file
first_line = check_instructions()                   # Checks instruction file is correct and makes sense                  
flowcells = setup_flowcells(first_line)             # Create flowcells
hs = initialize_hs()                                # Initialize HiSeq, takes a few minutes 
integrate_fc_and_hs(port_dict)                      # Integrate flowcell info with hs 
                               
do_flush()                                          # Flush out lines

cycles_complete = False

while not cycles_complete:
    stuck = 0
    complete = 0
    
    for fc in flowcells.values():                       
        if not fc.thread.is_alive():                # flowcell not busy, do next step in recipe 
            do_recipe(fc)                                               
        
        if fc.signal_event:                         # check if flowcells are waiting on each other                             
            stuck += 1
        
        if fc.cycle > fc.total_cycles:              # check if all cycles are complete on flowcell
            complete += 1
                                             
    if stuck == len(flowcells):                     # Start the first flowcell if they are waiting on each other
        free_fc()
            
    if complete == len(flowcells):                  # Exit while loop
        cycles_complete = True

    
do_shutdown()                                       # Shutdown  HiSeq

                 

