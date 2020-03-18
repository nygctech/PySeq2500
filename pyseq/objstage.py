#!/usr/bin/python
#
## @file
#
# Kunal Pandit 9/19
#
# Illumina HiSeq2500 OBJ-STAGE
# Uses commands found on  https://www.hackteria.org/wiki/HiSeq2000_-_Next_Level_Hacking#Control_Software
#

import time


# OBJSTAGE object

class OBJstage():
    #
    # Make OBJstage object
    #
    def __init__(self, fpga, logger = None):
    
        self.serial_port = fpga              
        self.min_z = 0
        self.max_z = 65535
        self.spum = 262         #steps per um
        self.max_v = 5 #mm/s
        self.min_v = 0 #mm/s
        self.v = None
        self.suffix = '\n'
        self.position = None
        self.logger = logger


    #
    # Initialize OBJstage 
    #
    def initialize(self): 
        self.position = self.check_position()                           # Update position
        self.set_velocity(5)                                            # Set velocity



    #
    # Send generic serial commands to OBJstage and return response 
    #
    def command(self, text):
        text = text + self.suffix
        self.serial_port.write(text)                                    # Write to serial port
        self.serial_port.flush()                                        # Flush serial port
        response = self.serial_port.readline()
        if self.logger is not None:
            self.logger.info('OBJstage::txmt::'+text)
            self.logger.info('OBJstage::rcvd::'+response)
        
        return  response  
    #   
    # Move OBJstage to absolute position   
    #
    def move(self, position):
        if position >= self.min_z and position <= self.max_z:
            try:
                while self.check_position() != position:
                    self.command('ZMV ' + str(position))                        # Move Objective
                
            except:
                print('Error moving objective stage')
        else:
            print('Objective position out of range')

    
    #   
    # Check position of objective   
    #
    def check_position(self):
        try:
            position = self.command('ZDACR')                                # Read position
            position = position.split(' ')[1]
            position = int(position[0:-1])
            self.position = position
            return position
        except:
            print('Error reading position of objective')
            return None
            
    #   
    # Set velocity 
    #
    def set_velocity(self, v):
        if v > self.min_v and v <= self.max_v:
            self.v = v
            # convert mm/s to steps/s
            v = v * 1288471 #steps/mm
            self.command('ZSTEP ' + str(v))                              # Set velocity
        else:
            print('Objective velocity out of range')
    
                        
                        
    
        
                        
                        
