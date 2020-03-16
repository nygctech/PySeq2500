#!/usr/bin/python
#
## @file
#
# Kunal Pandit 9/19
#
# Illumina HiSeq2500 Z-STAGE
# Uses commands found on  https://www.hackteria.org/wiki/HiSeq2000_-_Next_Level_Hacking#Control_Software
#

import time


# ZSTAGE object

class Zstage():
    #
    # Make Zstage object
    #
    def __init__(self, fpga, logger = None):
    
        self.serial_port = fpga              
        self.min_z = 0
        self.max_z = 25000
        self.spum = 0.656           #steps per um
        self.suffix = '\n'
        self.position = [0, 0, 0]
        self.motors = ['1','2','3']
        self.logger = logger
                        
                        
    #
    # Initialize Zstage 
    #
    def initialize(self):

        #Home Motors
        for i in range(3):
            response = self.command('T' + self.motors[i] + 'HM')        

        #Wait till they stop
        response = self.check_position()

        # Clear motor count registers
        for i in range(3):
            response = self.command('T' + self.motors[i] + 'CR')        

        # Update position
        for i in range(3):
            self.position[i] = int(self.command('T' + self.motors[i] + 'RD')[5:])                          # Set position

        
                        
                        
    #
    # Send generic serial commands to Zstage and return response 
    #
    def command(self, text):
        text = text + self.suffix
        self.serial_port.write(text)                                    # Write to serial port
        self.serial_port.flush()                                        # Flush serial port
        response = self.serial_port.readline()
        if self.logger is not None:
            self.logger.info('Zstage::txmt::'+text)
            self.logger.info('Zstage::rcvd::'+response)
        
        return  response                    
        
                        
    #   
    # Move Zstage to absolute position (position is a 3 element array)    
    #
    def move(self, position):
        for i in range(3):
            if position[i] <= self.max_z and position[i] >= self.min_z:
                self.command('T' + self.motors[i] + 'MOVETO ' + str(position[i]))                        # Move Absolute
            else:
                print("ZSTAGE can only move between " + str(self.min_z) + ' and ' + str(self.max_z))
                
        return self.check_position()                                                                    # Check position
                        
                        
    #                    
    # Check if Zstage motors are stopped and return their position
    #
    def check_position(self):
        # Get Current position
        old_position = [0,0,0]
        for i in range(3):
            successful = True
            while successful:
                try:
                    old_position[i] = int(self.command('T' + self.motors[i] + 'RD')[5:])
                    successful = False
                except:
                    time.sleep(2)

        
        all_stopped = 0
        while all_stopped != 3:
            all_stopped = 0
            for i in range(3):
                successful = True
                while successful:
                    try:
                        new_position = int(self.command('T' + self.motors[i] + 'RD')[5:])       # Get current position
                        stopped = new_position == old_position[i]                               # Compare old position to new position
                        all_stopped = all_stopped + stopped                                     # all_stopped will = 3 if all 3 motors are in position
                        old_position[i] = new_position                                          # Save new position
                        successful = False
                    except:
                        time.sleep(2)
            
        for i in range(3):
            self.position[i] = old_position[i]                                      # Set position
                             
        return self.position                                                        # Return position
        
