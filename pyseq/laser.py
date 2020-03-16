#!/usr/bin/python
#
## @file
#
# Kunal Pandit 11/19
#
# Illumina HiSeq2500 Laser
#


import serial
import io
import time

# Laser object

class Laser():    
    #
    # Make laser object
    #
    def __init__(self, com_port, baudrate = 9600, color = None, logger = None):

        # Open Serial Port
        s = serial.Serial(com_port, baudrate, timeout = 1)

        # Text wrapper around serial port                
        self.serial_port = io.TextIOWrapper(io.BufferedRWPair(s,s,),
                                            encoding = 'ascii',
                                            errors = 'ignore')                
        self.on = False
        self.power = 0
        self.max_power = 500
        self.min_power = 0
        self.suffix = '\r'
        self.logger = logger
        self.color = color
        self.version = self.command('VERSION?')[0:-1]
        


    #
    # Initialize laser
    #
    def initialize(self):
        self.turn_on(True)
        self.set_power(10)


    #
    # Send generic serial commands to laser and return response 
    #
    def command(self, text):
        text = text + self.suffix
        self.serial_port.write(text)                                    # Write to serial port
        self.serial_port.flush()                                        # Flush serial port
        response = self.serial_port.readline()
        if self.logger is not None:
            self.logger.info(self.color+'Laser::txmt::'+text)
            self.logger.info(self.color+'Laser::rcvd::'+response)
        
        return  response  

    #
    # Turn laser on/off
    #
    def turn_on(self, state):
        if state:
            while not self.get_status():
                self.command('ON')

            self.on = True
        else:
            while self.get_status():
                self.command('OFF')

            self.on = False
                
        return self.on

    #
    # Get power level
    #
    def get_power(self):
        self.power = int(self.command('POWER?').split('mW')[0])

        return self.power

    #
    # Set power level
    #
    def set_power(self, power):
        
        if power >= self.min_power and power <= self.max_power:
                self.command('POWER='+str(power))
                self.power = self.get_power()
        else:
            print('Power must be between ' +
                  str(self.min_power) +
                  ' and ' +
                  str(self.max_power))

        return self.get_status()
            
    #
    # Get Status
    #
    def get_status(self):
        
        self.status = self.command('STAT?')[0:-1]

        if self.status == 'ENABLED':
            return True
        else:
            return False
