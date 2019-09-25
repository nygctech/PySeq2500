#!/usr/bin/python
#
## @file
#
# Kunal Pandit 9/19
#
# Illumina HiSeq2500 FPGA
# Just creates fpga object, sends commands to serial port, and returns response
#


import serial
import io


# FPGA object

class FPGA():

    #
    # Make FPGA object
    #
    def __init__(self, com_port_command, com_port_response, baudrate):

        # Open Serial Port
        s_command = serial.Serial(com_port_command, baudrate, timeout = 1)
        s_response = serial.Serial(com_port_response, baudrate, timeout = 1)

        # Text wrapper around serial port                
        self.serial_port = io.TextIOWrapper(io.BufferedRWPair(s_command,s_response,),
                                            encoding = 'ascii',
                                            errors = 'ignore')                
        self.suffix = '\n'

    #
    # Initialize FPGA
    #
    def initialize(self):
    
        response = self.command('RESET')                                # Initialize FPGA
        print('ystage: ' + response)     
     
     
    #
    # Send commands to FPGA and return response
    #
    def command(self, text):
    
        self.serial_port.write(text + self.suffix)                      # Write to serial port
        self.serial_port.flush()                                        # Flush serial port
        return self.serial_port.readline()                              # Return response
