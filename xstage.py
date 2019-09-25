#!/usr/bin/python
#
## @file
#
# Kunal Pandit 9/19
#
# Illumina HiSeq2500 X-STAGE
# Uses command set from Schneider Electric MCode
#


import serial
import io
import time


# XSTAGE object

class Xstage():
    #
    # Make Xstage object
    #
    def __init__(self, com_port, baudrate):

        # Open Serial Port
        s = serial.open(serial.Serial(com_port, baudrate, timeout = 1)

        # Text wrapper around serial port                
        self.serial_port = io.TextIOWrapper(io.BufferedRWPair(s,s,),
                                            encoding = 'ascii',
                                            errors = 'ignore')                
        self.min_x = 0
        self.max_x = 60000
        self.suffix = '\r'
        self.position = 0
                        
                        
    #
    # Initialize Xstage 
    #
    def initialize(self):
        response = self.command('\x03')                                 # Initialize Stage
        print('xstage: ' + response) 
                        
                        
    #
    # Send generic serial commands to Xstage and return response 
    #
    def command(self, text):                        
        self.serial_port.write(text + self.suffix)                      # Write to serial port
        self.serial_port.flush()                                        # Flush serial port
        return self.serial_port.readline()                              # Return response
                        
                        
    #   
    # Move Xstage to absolute position    
    #
    def move(self, position):
        if position <= self.max_x and position >= self.min_x:
            self.command('MA' + str(position))                          # Move Absolute
            return self.check_position()                                # Check position
        else:
            print("XSTAGE can only move between " + str(self.min_y) + ' and ' + str(self.max_y))
                        
                        
    #                    
    # Check if Xstage is at a position
    #
    def check_position(self, position):
        moving = 1
        while moving != 0:
            moving = self.command('PR(MV)')                             # Check if moving, 1 = yes, 0 = no
            time.sleep(2)

        self.position = self.command('R(P)')                            # Set position
                        
        return position == self.position                                # Return TRUE if in position or False if not
