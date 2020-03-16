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
import time


# FPGA object

class FPGA():

    #
    # Make FPGA object
    #
    def __init__(self, com_port_command, com_port_response, baudrate = 115200, logger = None):

        
        # Open Serial Port
        s_command = serial.Serial(com_port_command, baudrate, timeout = 1)
        s_response = serial.Serial(com_port_response, baudrate, timeout = 1)

        # Text wrapper around serial port                
        self.serial_port = io.TextIOWrapper(io.BufferedRWPair(s_response,s_command),
                                            encoding = 'ascii',
                                            errors = 'ignore')                
        self.suffix = '\n'
        self.y_offset = 7000000
        self.logger = logger

    #
    # Initialize FPGA
    #
    def initialize(self):
    
        response = self.command('RESET')                                # Initialize FPGA
        self.command('EX1HM')                                           # Home excitation filter 
        self.command('EX2HM')                                           # Home excitation filter 
        self.command('EM2I')                                            # Move emission filter into light path 
        self.command('SWLSRSHUT 0')                                     # Shutter lasers
     
     
    #
    # Send commands to FPGA and return response
    #
    def command(self, text):
        text = text + self.suffix
        self.serial_port.write(text)                                    # Write to serial port
        self.serial_port.flush()                                        # Flush serial port
        response = self.serial_port.readline()
        if self.logger is not None:
            self.logger.info('FPGA::txmt::'+text)
            self.logger.info('FPGA::rcvd::'+response)
        
        return  response                    

    #
    # Read encoder position
    #
    def read_position(self):
        tdi_pos = self.command('TDIYERD')
        tdi_pos = tdi_pos.split(' ')[1]
        tdi_pos = int(tdi_pos[0:-1]) - self.y_offset
        return tdi_pos

    #
    # Write encoder position
    #
    def write_position(self, position):
        position = position + self.y_offset
        while abs(self.read_position() + self.y_offset - position) > 5:
            self.command('TDIYEWR ' + str(position))
            time.sleep(1)
    #
    # Set TDIYPOS
    #
    def TDIYPOS(self, y_pos):
        self.command('TDIYPOS ' + str(y_pos+self.y_offset-80000))

    #
    # Arm stage
    #
    def TDIYARM3(self, n_triggers, y_pos):
        self.command('TDIYARM3 ' + str(n_triggers) + ' ' +
                  str(y_pos + self.y_offset-10000) + ' 1')

            
