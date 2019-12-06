#!/usr/bin/python
#
## @file
#
# Kunal Pandit 9/19
#
# Illumina HiSeq2500 Pump
# Uses command set from Kloehn VersaPump3
#


import serial
import io
import time

# Pump object

class Pump():    
    #
    # Make pump object
    #
    def __init__(self, com_port, baudrate = 9600):

        # Open Serial Port
        s = serial.open(serial.Serial(com_port, baudrate, timeout = 1)

        # Text wrapper around serial port                
        self.serial_port = io.TextIOWrapper(io.BufferedRWPair(s,s,),
                                            encoding = 'ascii',
                                            errors = 'ignore')                
        self.n_barrels = 8
        self.barrel_volume = 250 # uL
        self.steps = 48000
        self.max_volume = self.n_barrels*self.barrel_volume #uL
        self.min_volume = self.max_volume/self.steps #uL
        self.min_speed = 40 # steps per second (sps)
        self.max_speed = 8000 # steps per second (sps)
        self.dispense_speed = 7000 # speed to dispense (sps)
        self.prefix = '/1'
        self.suffix = '\r'


    #
    # Initialize pump
    #
    def initialize(self):
        response = self.command('W4R')                                  # Initialize Stage
                    


    #
    # Send generic serial commands to pump and return response 
    #
    def command(self, text):                        
        self.serial_port.write(self.prefix + text + self.suffix)        # Write to serial port
        self.serial_port.flush()                                        # Flush serial port
        return self.serial_port.readline()                              # Return response

    #
    # Pump desired volume at desired speed then waste
    #
    def pump(self, volume, speed):
        position = self.vol_to_position(volume)                         # Convert volume (uL) to position (steps)

        #Aspirate                
        while position != self.check_position():
            self.command('IV' + str(speed) + 'A' + str(position) + 'R') # Pull syringe down to position
            self.check_pump()                               
        self.command('OR')                                              # Switch valve to waste

        #Dispense
        position = 0
        while position != self.check_position():
            self.command('OV' + str(self.dispense_speed) + 'A0R')       # Dispense, Push syringe to top at dispense speed
            self.check_pump()
        self.command('IR')                                              # Switch valve to input

    
    #
    # Check Pump status
    #
    def check_pump(self):
        busy = '@'
        ready = '`'
        status_code = busy
        
        while status_code[0] != ready :
            status_code = self.command('')                              # Ping pump for status
            status_code = str(status_code.split('0')[1]) 
            
            if status_code[0] == busy :
                    time.sleep(2)
            elif status_code[0] == ready:
                    check_pos(hardware,command_pos,f)
            elif status_code[0] != ready :
                    print('pump error')
                    sys.exit()
                        

    #
    # Pump desired volume at desired speed then waste
    #
    def check_position(self, position):
        pump_position = self.command('?')
        pump_position = pump_position.split('`')[1]
        pump_position = int(pump_position.split('\x03')[0])

        return pump_position


    #
    # Convert volume in uL to pump position
    #
    def vol_to_pos(self, volume):
        position = round(volume / self.max_volume * self.steps)
        return position
        
        

    
