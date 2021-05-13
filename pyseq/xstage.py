#!/usr/bin/python
"""Illumina HiSeq 2500 Systems :: X-STAGE

   Uses command set from Schneider Electric MCode

   The xstage can be moved from step positions 1000 to 50000. Initially, the
   xstage is homed to step position 30000. Lower step positions are to the right,
   and higher step positions are to the left. Each xstage step is 0.375 microns.

   **Example:**

.. code-block:: python

   #Create xstage
   import pyseq
   xstage = pyseq.xstage.Xstage('COM9')
   #Initialize xstage
   xstage.initialize()
   #Move xstage to step position 10000
   xstage.move(10000)
   10000

TODO:
   * Change initialization to be aware position of flags.

"""


import serial
import io
import time


class Xstage():
    """Illumina HiSeq 2500 Systems :: X-STAGE

       **Attributes:**
        - spum (float): Number of xstage steps per micron.
        - position (int): The absolution position of the xstage in steps.
        - min_x (int): Minimum safe xstage step position.
        - max_x (int): Maximum safe xstage step position.
        - home (int): Step position to move xstage out.

    """

    # Make Xstage object
    def __init__(self, com_port, baudrate = 9600, logger = None):
        """The constructor for the xstage.

           **Parameters:**
            - com_port (str): Communication port for the xstage.
            - baudrate (int, optional): The communication speed in symbols per
              second.
            - logger (log, optional): The log file to write communication with
              the xstage to.

           **Returns:**
            - xstage object: A xstage object to control the xstage.

        """

        if isinstance(com_port, int):
            com_port = 'COM'+str(com_port)

        try:
            # Open Serial Port
            s  = serial.Serial(com_port, baudrate, timeout = 1)
            # Text wrapper around serial port
            self.serial_port = io.TextIOWrapper(io.BufferedRWPair(s,s,),
                                                encoding = 'ascii',
                                                errors = 'ignore')
        except:
            print('ERROR::Check X Stage Port')
            self.serial_port = None


        self.min_x = 1000
        # Need to update too large
        self.max_x = 50000
        self.home = 30000
        self.spum = 0.4096     #steps per um
        self.suffix = '\r'
        self.position = 0
        self.logger = logger


    def initialize(self):
        """Initialize the xstage."""

        # Initialize Stage
        response = self.command('\x03')

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


        # Home stage program
        self.serial_port.write('PG 1\r')
        self.serial_port.write('HM 1\r')
        self.serial_port.write('H\r')
        self.serial_port.write('P = 30000\r')
        self.serial_port.write('E\r')
        self.serial_port.write('PG\r')
        self.serial_port.flush()
        #self.serial_port.write('EX 1\r')
        #self.serial_port.flush()
        #self.position = 30000
        #self.check_position(self.position)

        # Check if stage is homed correctly
        self.command('EX 1')                                                #Execute home stage program
        self.position = 30000
        self.check_position(self.position)
        homed = self.check_home()
        if not homed:
            self.move(40000)
            self.command('EX 1')
            self.position = 30000
            self.check_position(self.position)
            homed = self.check_home()

        return homed

    def check_home(self):

        homed = False
        self.move(29000)
        response = self.command('PR I1')
        if int(response.strip()):
            self.move(31000)
            response = self.command('PR I1')
            if not int(response.strip()):
                homed = True

        self.move(30000)

        return homed

    def command(self, text):
        """Send a serial command to the xstage and return the response.

           **Parameters:**
            - text (str): A command to send to the xstage.

           **Returns:**
            - str: The response from the xstage.

        """
        text = text + self.suffix
        self.serial_port.write(text)                                            # Write to serial port
        self.serial_port.flush()                                                # Flush serial port
        response = self.serial_port.readline()
        if self.logger is not None:
            self.logger.info('Xstage::txmt::'+text)
            self.logger.info('Xstage::rcvd::'+response)
        else:
            print(response)

        return  response




    def move(self, position):
        """Move xstage to absolute step position.

           **Parameters:**
            - position (int): Absolute step position must be between 1000 - 50000.

           **Returns:**
            - int: Absolute step position after move.

        """
        if self.min_x <= position <= self.max_x:
            self.command('MA ' + str(position))                                 # Move Absolute
            return self.check_position(position)                                # Check position
        else:
            print('XSTAGE can only move between ' + str(self.min_x) +
                  ' and ' + str(self.max_x))


    # Check if Xstage is at a positio
    def check_position(self, position):
        """Check if xstage is in position.

           **Parameters:**
            - position (int): Absolute step position must be between 1000 - 50000.

           **Returns:**
            - bool: True if xstage is in position, False if it is not in position.

        """
        moving = 1
        while moving != 0:
            moving = int(self.command('PR MV'))                                 # Check if moving, 1 = yes, 0 = no
            time.sleep(1)

        self.position = int(self.command('PR P'))                               # Set position

        return position == self.position                                        # Return TRUE if in position or False if not
