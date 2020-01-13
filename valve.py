#!/usr/bin/python
#
## @file
#
# Kunal Pandit 9/19
#
# Illumina HiSeq2500 Valve
# Uses command set from Vici Technical Note 415
#


import serial
import io
import time

# Valve object

class Valve():
    
    #
    # Make pump object
    #
    def __init__(self, com_port, name = None, logger = None, port_dict = dict()):
        
        baudrate = 9600

        # Open Serial Port
        s = serial.Serial(com_port, baudrate, timeout = 1)

        # Text wrapper around serial port                
        self.serial_port = io.TextIOWrapper(io.BufferedRWPair(s,s,),
                                            encoding = 'ascii',
                                            errors = 'ignore')                
        self.n_ports = 10
        self.port_dict = port_dict
        self.variable_ports = []
        self.prefix = ''
        self.suffix = '\r'
        self.logger = logger
        self.log_flag = False
        self.name = name

        

    
    #
    # Initialize valve
    #
    def initialize(self):
        #Get ID of valve
        prefix = None                                    
        while prefix == None:
            prefix = self.command('ID')                                         # Query ID number
            try:
                prefix = prefix.split('=')[1]
                prefix = prefix.replace(' ','')                                 # remove whitespace
                prefix = prefix.replace('\n','')                                # remove newline
                if prefix == 'notused':
                    prefix = ''
                self.prefix = prefix
                
            except:
                self.write_log('error: could not parse ID')           		# Write error to log
                prefix = None
            
        #Get number of ports on valve
        n_ports = None
        while n_ports == None:
            n_ports = self.command('NP')                                    # Query Port number
            try:
                n_ports = n_ports.split('=')[1]
                n_ports = n_ports.replace(' ','')                               # remove whitespace
                n_ports = n_ports.replace('\n','')                              # remove newline
                self.n_ports = int(n_ports)
            except:
                self.write_log('error: could not get number of ports')           		# Write error to log
                n_ports = None
                
        #If port dictionary empty map 1:1
        if not self.port_dict:
            for i in range(1,self.n_ports+1):
                self.port_dict[i] = i
                

    #
    # Send generic serial commands to pump and return response 
    #
    def command(self, text):
        text = self.prefix + text + self.suffix                         # Format the command
        self.serial_port.write(text)                                    # Write to serial port
        self.serial_port.flush()                                        # Flush serial port
        response = self.serial_port.readline()
        
        if self.logger is not None:                                     # Log sent command
            self.logger.info(self.name + '::txmt::'+text)
        else:
            print(text)

        blank = response 
        while blank is not '':                                          # Log received commands
            if self.logger is not None:
                self.logger.info(self.name + '::rcvd::'+blank)
            else:
                print(blank)
            blank = self.serial_port.readline()
        
        return  response                    
                                                 

    #
    # Move to port 
    #
    def move(self, port_name):
        position = self.port_dict[port_name]
        while position != self.check_valve():
            response = self.command('GO' + str(position))
            time.sleep(1)


    #
    # Query valve position
    #
    def check_valve(self):
        
        position = ''
        while not position:
            position = self.command('CP')
            
            try:    
                position = position.split('=')[1]
                position = position.replace(' ','')         # remove whitespace
                position = position.replace('\n','')        # remove newline
            except:
                self.write_log('error: could not parse position')           		# Write error to log
                position = None
     
        return int(position)

    def write_log(self, text):
        if self.logger is not None:
            self.logger.info(self.name + ' ' + text)
