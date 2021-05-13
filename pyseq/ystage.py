#!/usr/bin/python
"""Illumina HiSeq 2500 System :: Y-STAGE

   Uses command set from Parker ViX 250IH & ViX500 IH

   The ystage can be moved from step positions -7000000 to 7500000. Initially,
   the ystage is homed to step position 0. Negative step positions are to the
   front, and positive step positions are to the back. Each ystage step = 10 nm.

   **Example:**

.. code-block:: python

    #Create ystage
    import pyseq
    xstage = pyseq.ystage.Ystage('COM10')
    #Initialize ystage
    ystage.initialize()
    #Move ystage to step position 3000000
    ystage.move(3000000)
    True

"""


import serial
import io
import time


class Ystage():
    """Illumina HiSeq 2500 System :: Y-STAGE

       **Attributes:**
        - spum (float): Number of ystage steps per micron.
        - position (int): The absolution position of the ystage in steps.
        - min_y (int): Minimum safe ystage step position.
        - max_y (int): Maximum safe ystage step position.
        - home (int): Step position to move ystage out.

    """


    def __init__(self, com_port, baudrate = 9600, logger = None):
        """The constructor for the ystage.

           **Parameters:**
            - com_port (str): Communication port for the ystage.
            - baudrate (int, optional): The communication speed in symbols per
              second.
            - logger (log, optional): The log file to write communication with the
              ystage to.

           **Returns:**
            - ystage object: A ystage object to control the ystage.

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
            print('ERROR::Check Y Stage Port')
            self.serial_port = None

        self.min_y = -7000000
        self.max_y = 7500000
        self.spum = 100     # steps per um
        self.prefix = '1'
        self.suffix = '\r\n'
        self.on = False
        self.position = 0
        self.home = 0
        self.mode = None
        self.velocity = None
        self.gains = None
        self.configurations = {'imaging':{'g':'5,10,5,2,0'  ,'v':0.15400},
                               'moving': {'g':'5,10,7,1.5,0','v':1}
                               }
        self.logger = logger

    def initialize(self):
        """Initialize the ystage."""

        response = self.command('Z')                                            # Initialize Stage
        time.sleep(2)
        response = self.command('W(EX,0)')                                      # Turn off echo
        self.set_mode('moving')
        #response = self.command('GAINS(5,10,7,1.5,0)')                          # Set gains
        response = self.command('MA')                                           # Set to absolute position mode
        response = self.command('ON')                                           # Turn Motor ON
        self.on = True
        response = self.command('GH')                                           # Home Stage



    def command(self, text):
        """Send a serial command to the ystage and return the response.

           **Parameters:**
            - text (str): A command to send to the ystage.

           **Returns:**
            - str: The response from the ystage.
        """

        text = self.prefix + text + self.suffix
        self.serial_port.write(text)                                            # Write to serial port
        self.serial_port.flush()                                                # Flush serial port
        response = self.serial_port.readline()
        if self.logger is not None:
            self.logger.info('Ystage::txmt::'+text)
            self.logger.info('Ystage::rcvd::'+response)

        return  response


    def move(self, position):
        """Move ystage to absolute step position.

           **Parameters:**
            - position (int): Absolute step position must be between -7000000
              and 7500000.

           **Returns:**
            - bool: True when stage is in position.

        """

        if self.min_y <= position <= self.max_y:
            while self.position != position:
                self.command('D' + str(position))                               # Set distance
                self.command('G')                                               # Go
                while not self.check_position():                                # Wait till y stage is in position
                    time.sleep(1)
                self.read_position()                                            # Update stage position
            return True                                                         # Return True that stage is in position
        else:
            print("YSTAGE can only between " + str(self.min_y) + ' and ' +
                str(self.max_y))


    def check_position(self):
        """Check if ystage is in position.

           **Returns:**
            - int: 1 if ystage is in position, 0 if it is not in position.

        """
        
        try:
            ip = int(self.command('R(IP)')[1:])
        except:
            ip = 0

        return


    def read_position(self):
        """Return the absolute step position of the ystage (int)."""

        try:
            self.position = int(self.command('R(PA)')[1:])                      # Read and store position
        except:
            pass

        return self.position

    def set_mode(self, mode):
        "Change between imaging and moving configurations."

        mode_changed = True
        if self.mode != mode:
            if mode in self.configurations.keys():
                gains = str(self.configurations[mode]['g'])
                _gains = [float(g) for g in gains.split(',')]
                velocity = self.configurations[mode]['v']
                all_true = False
                while not all_true:
                    self.command('GAINS('+gains+')')
                    time.sleep(1)
                    try:
                        gains_ = self.command('GAINS').strip()[1:].split(' ')       # format reponse
                        all_true = all([float(g[2:]) == _gains[i] for i, g in enumerate(gains_)])
                    except:
                        all_true = False
                velocity_ = None
                while velocity_ != float(velocity):
                    self.command('V'+str(velocity))
                    time.sleep(1)
                    try:
                        velocity_ = float(self.command('V').strip()[1:])
                    except:
                        velocity_ = False
                self.mode = mode
                self.velocity = velocity_
                self.gains = gains
            else:
                mode_change = False
                message = 'Ystage::ERROR::Invalid configuration::'+str(mode)
                if self.logger is not None:
                    self.logger.info(message)
                else:
                    print(message)

        return mode_changed
