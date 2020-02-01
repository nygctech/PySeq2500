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
    def __init__(self, com_port, baudrate = 9600, logger = None):

        # Open Serial Port
        s = serial.Serial(com_port, baudrate, timeout = 1)

        # Text wrapper around serial port                
        self.serial_port = io.TextIOWrapper(io.BufferedRWPair(s,s,),
                                            encoding = 'ascii',
                                            errors = 'ignore')                
        self.min_x = 1000
        self.max_x = 50000
        self.home = 30000
        self.spum = 100/244     #steps per um
        self.suffix = '\r'
        self.position = 0
        self.logger = logger
                        
                        
    #
    # Initialize Xstage 
    #
    def initialize(self):
        response = self.command('\x03')                                 # Initialize Stage
        
        #Change echo mode to respond only to print and list commands 
        response = self.command('EM=2')

        #Enable Encoder
        response = self.command('EE=1')
        #Set Initial Velocity
        response = self.command('VI=40')
        #Set Max Velocity
        response = self.command('VM=1000')
        #Set Acceleration
        response = self.command('A=4000')
        #Set Deceleration
        response = self.command('D=4000')
        #Set Home
        response = self.command('S1=1,0,0')
        #Set Neg. Limit
        response = self.command('S2=3,1,0')
        #Set Pos. Limit
        response = self.command('S3=2,1,0')
        #Set Stall Mode = stop motor
        response = self.command('SM=0')
        # limit mode = stop if sensed
        response = self.command('LM=1')
        #Encoder Deadband
        response = self.command('DB=8')
        #Debounce home
        response = self.command('D1=5')
        # Set hold current
        response = self.command('HC=20')
        # Set run current
        response = self.command('RC=100')


        # Home stage
        self.serial_port.write('PG 1\r')
        self.serial_port.write('HM 1\r')
        self.serial_port.write('H\r')
        self.serial_port.write('P = 30000\r')
        self.serial_port.write('E\r')
        self.serial_port.write('PG\r')
        self.serial_port.flush()
        self.serial_port.write('EX 1\r')
        self.serial_port.flush()
        self.position = 30000
        self.check_position(self.position)
        
    #
    # Send generic serial commands to Xstage and return response 
    #
    def command(self, text):
        text = text + self.suffix
        self.serial_port.write(text)                                    # Write to serial port
        self.serial_port.flush()                                        # Flush serial port
        response = self.serial_port.readline()
        if self.logger is not None:
            self.logger.info('Xstage::txmt::'+text)
            self.logger.info('Xstage::rcvd::'+response)
        
        return  response                    

                        
                        
    #   
    # Move Xstage to absolute position    
    #
    def move(self, position):
        if position <= self.max_x and position >= self.min_x:
            self.command('MA ' + str(position))                         # Move Absolute
            return self.check_position(position)                        # Check position
        else:
            print("XSTAGE can only move between " + str(self.min_x) + ' and ' + str(self.max_x))
                        
                        
    #                    
    # Check if Xstage is at a position
    #
    def check_position(self, position):
        moving = 1
        while moving != 0:
            moving = int(self.command('PR MV'))                             # Check if moving, 1 = yes, 0 = no
            time.sleep(1)

        self.position = int(self.command('PR P'))                           # Set position
                        
        return position == self.position                                # Return TRUE if in position or False if not
