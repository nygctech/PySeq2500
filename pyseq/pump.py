#!/usr/bin/python
'''
Kunal Pandit 9/19

Illumina HiSeq 2500 System Pump
Uses command set from Kloehn VersaPump3
'''


import serial
import io
import time

# Pump object

class Pump():
    '''Pump object'''

    # Make pump object
    def __init__(self, com_port, name=None, logger=None):
        '''Create a pump object.'''

        baudrate = 9600

        # Open Serial Port
        s = serial.Serial(com_port, baudrate, timeout = 1)

        # Text wrapper around serial port
        self.serial_port = io.TextIOWrapper(io.BufferedRWPair(s,s,),
                                            encoding = 'ascii',
                                            errors = 'ignore')
        self.n_barrels = 8
        self.barrel_volume = 250.0 # uL
        self.steps = 48000.0
        self.max_volume = self.n_barrels*self.barrel_volume #uL
        self.min_volume = self.max_volume/self.steps #uL
        self.min_speed = int(self.min_volume*40*60) # uL per min (upm)
        self.max_speed = int(self.min_volume*8000*60) # uL per min (upm)
        self.dispense_speed = 7000 # speed to dispense (sps)
        self.prefix = '/1'
        self.suffix = '\r'
        self.logger = logger
        self.name = name


    # Initialize pump
    def initialize(self):
        '''Initialize the pump.'''

        response = self.command('W4R')                                          # Initialize Stage


    def command(self, text):
        '''Send a serial command to the pump and return the response'''

        text = self.prefix + text + self.suffix                                 # Format text
        self.serial_port.write(text)                                            # Write to serial port
        self.serial_port.flush()                                                # Flush serial port
        response = self.serial_port.readline()                                  # Get the response
        if self.logger is not None:                                             # Write command and response to log
            self.logger.info(self.name+'::txmt::'+text)
            self.logger.info(self.name+'::rcvd::'+response)

        return  response



    def pump(self, volume, speed = 0):
        '''Pump desired volume at desired speed then send liquid to waste'''

        if speed == 0:
            speed = self.min_speed                                              # Default to min speed

        position = self.vol_to_pos(volume)                                      # Convert volume (uL) to position (steps)
        sps = self.uLperMin_to_sps(speed)                                       # Convert flowrate(uLperMin) to steps per second

        self.check_pump()                                                       # Make sure pump is ready

        #Aspirate
        while position != self.check_position():
            self.command('IV' + str(sps) + 'A' + str(position) + 'R')           # Pull syringe down to position
            self.check_pump()
        self.command('OR')                                                      # Switch valve to waste

        #Dispense
        position = 0
        while position != self.check_position():
            self.command('OV' + str(self.dispense_speed) + 'A0R')               # Dispense, Push syringe to top at dispense speed
            self.check_pump()
        self.command('IR')                                                      # Switch valve to input


    def check_pump(self):
        '''Check the pump status and return True WHEN ready.'''

        busy = '@'
        ready = '`'
        status_code = ''

        while status_code != ready :

            while not status_code:
                status_code = self.command('')                                  # Ping pump for status

                if status_code.find(busy) > -1:
                    status_code = ''
                    time.sleep(2)
                elif status_code.find(ready) > -1:
                    status_code = ready
                    return True
                else:
                    self.write_log('pump error')
                    return False



    def check_position(self):
        '''Query and return the pump position'''

        pump_position = self.command('?')

        while not isinstance(pump_position, int):
            pump_position = self.command('?')
            try:
                pump_position = pump_position.split('`')[1]
                pump_position = int(pump_position.split('\x03')[0])
            except:
                self.write_log('error: could not parse position')
                pump_position = None

        return pump_position


    def vol_to_pos(self, volume):
        '''Convert volume from uL to pump position

           If volume is too big or too small, return the max or min volume.
        '''

        if volume > self.max_volume:
            write_log('Volume is too large, only pumping ' +
                       str(self.max_volume))
            volume = self.max_volume
        elif volume < self.min_volume:
            write_log('Volume is too small, pumping ' +
                       str(self.min_volume))
            volume = self.min_volume

        position = round(volume / self.max_volume * self.steps)
        return int(position)


    def uLperMin_to_sps(self, speed):
        '''Convert flowrate from uL per min. to steps per second'''

        sps = round(speed / self.min_volume / 60)
        return int(sps)


    def write_log(self, text):
        '''Write messages to log'''

        if self.logger is not None:
            self.logger.info(self.name + ' ' + text)
