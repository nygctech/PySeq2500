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

    def __init__(self, com_port, baudrate):

        # Open Serial Port
        s = serial.open(serial.Serial(com_port, baudrate, timeout = 1)

        # Text wrapper around serial port                
        self.serial_port = io.TextIOWrapper(io.BufferedRWPair(s,s,),
                                            encoding = 'ascii',
                                            errors = 'ignore')                
        self.min_x = 0
        self.max_x = 7500000
        self.prefix = '1'
        self.suffix = '\r\n'
        self.on = False
        self.position = int(self.command('R(PA)'))                      # Set position

        response = self.command('Z')                                    # Initialize Stage
        print('ystage: ' + response)
                        
        response = self.command('GAINS(5,2,5,10,0)')                    # Set gains
        print('ystage: ' + response)
                        
        response = self.command('ON')                                   # Turn Motor ON
        print('ystage: ' + response)
        self.on = True
                        
        response = self.command('GH')                                   # Home Stage
        print('ystage: ' + response)                
        response = self.check_postion()                                 
        print('ystage: ' + response)
        response = self.command(W(PA,0))
        print('ystage: ' + response)
        self.position = int(self.command('R(PA)'))                      

    
        
    def command(self, text):
                        
        self.serial_port.write(self.prefix + text + self.suffix)        # Write to serial port
        self.serial_port.flush()                                        # Flush serial port
        return self.serial_port.readline()                              # Return response
        
        
    def move(self, position):
        self.command('D' + str(position)                                # Set distance
        self.command('G')                                               # Go           
        return self.check_position()                                    # Check position

    def check_position(self):
        moving = 1
        while moving != 0:
            moving = int(self.command(R(MV)))                           # Check if moving, 1 = yes, 0 = no
            time.sleep(2)

        self.position = self.command('R(PA)')                           # Set position

        return int(self.command('R(IP)'))                               # Check if in position, 1 = yes, 0 = no
