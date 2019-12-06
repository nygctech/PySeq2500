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
    def __init__(self, com_port, port_dict = dict(), baudrate = 9600):

        # Open Serial Port
        s = serial.open(serial.Serial(com_port, baudrate, timeout = 1)

        # Text wrapper around serial port                
        self.serial_port = io.TextIOWrapper(io.BufferedRWPair(s,s,),
                                            encoding = 'ascii',
                                            errors = 'ignore')                
        self.n_ports = 10
        self.port_dict = port_dict
        self.prefix = '*'
        self.suffix = '\r'

    
    #
    # Initialize valve
    #
    def initialize(self):
        #Get ID of valve
        prefix = self.command('ID')                                     # Query ID number
        prefix = prefix.split('=')[1]
        prefix = prefix.replace(' ','')                                 # remove whitespace
        prefix = prefix.replace('\n','')                                # remove newline
        self.prefix = prefix
        #Get number of ports on valve                
        n_ports = self.command('NP')                                    # Query Port number
        n_ports = n_ports.split('=')[1]
        n_ports = n_ports.replace(' ','')                               # remove whitespace
        n_ports = n_ports.replace('\n','')                              # remove newline
        self.n_ports = int(n_ports)


    #
    # Send generic serial commands to pump and return response 
    #
    def command(self, text):                        
        self.serial_port.write(text + self.suffix)                      # Write to serial port
        self.serial_port.flush()                                        # Flush serial port
        return self.serial_port.readline()                              # Return response


    #
    # Move to port 
    #
    def move(self, port_name):
        position = self.port_dict[port_name]
        while position != check_valve:
            self.command('GO' + str(position))


    #
    # Check valve position
    #
    def check_valve(self):
        position = self.command('CP')
        position = position.split('=')[1]
        position = position.replace(' ','')         # remove whitespace
        position = position.replace('\n','')        # remove newline
        return int(position)
