#!/usr/bin/python
"""Illumina HiSeq 2500 System :: Z-STAGE

   Uses commands found on `hackteria
   <www.hackteria.org/wiki/HiSeq2000_-_Next_Level_Hacking>`_

   The zstage can be moved up and down by 3 independent tilt motors. Each tilt
   motor can from step positions 0 to 25000. Initially, all of the tilt motors
   in the zstage are homed to step position 0. Lower step positions are down,
   and higher step positions are up. Each tilt motor step is about 1.5 microns.
   These motors are not precise and not repeatable. That is they are not
   expected to go to the exact step position and they are not expected to go to
   the same position over and over again.

**Example:**

.. code-block:: python

    #Create zstage
    import pyseq
    zstage = pyseq.zstage.Zstage('COM10')
    #Initialize zstage
    zstage.initialize()
    #Move all tilt motors on zstage to absolute step position 21000
    zstage.move([21000, 21000, 21000])
    [21000, 21000, 21000]

"""


import time


class Zstage():
    """Illumina HiSeq 2500 System :: Z-STAGE

       **Attributes:**
        - spum (float): Number of zstage steps per micron.
        - position ([int, int, int]): A list with absolute positions of each tilt
          motor in steps.
        - motors ([int, int, int]): Motor ids.
        - tolerance (int): Maximum step error allowed, default is 2.
        - min_z (int): Minimum zstage step position.
        - max_z (int): Maximum safe zstage step position.
        - xstep ([int, int, int]): Xstage position of the respective motors.
        - ystep ([int, int, int]): Ystage position of the respective motors.
        - image_step (int): Initial step position of motors for imaging
        - logger (logger): Logger for messaging.
        - focus_pos (int): Step used used for imaging.
        - active (bool): Flag to enable/disable z stage movements.

    """


    def __init__(self, fpga, logger = None):
        """The constructor for the zstage.

           **Parameters:**
           - fpga (fpga object): The Illumina HiSeq 2500 System :: FPGA.
           - logger (log, optional): The log file to write communication with the
             zstage to.

           **Returns:**
           zstage object: A zstage object to control the zstage.

        """

        self.serial_port = fpga
        self.min_z = 0
        self.max_z = 25000
        self.spum = 0.656                                                       #steps per um
        self.suffix = '\n'
        self.position = [0, 0, 0]
        self.motors = ['1','2','3']
        self.logger = logger
        self.tolerance = 2
        #self.xstep = [-10060, -10060, 44990]                                    # x step position of motors
        #self.ystep = [-2580000, 5695000, 4070000]                               # y step position of motors
        self.xstep = [-447290,   16770, -179390]
        self.ystep = [-10362000, -61867000, 73152000]
        self.focus_pos = 21500
        self.active = True                                                  # rough focus position


    def initialize(self):
        """Initialize the zstage."""

        #Home Motors
        if self.active:
            for i in range(3):
                response = self.command('T' + self.motors[i] + 'HM')

        #Wait till they stop
        response = self.check_position()

        # Clear motor count registers
        for i in range(3):
            response = self.command('T' + self.motors[i] + 'CR')

        # Update position
        for i in range(3):
            self.position[i] = int(self.command('T' + self.motors[i]
                + 'RD')[5:])                                                    # Set position


    def command(self, text):
        """Send a serial command to the zstage and return the response.

           **Parameters:**
            - text (str): A command to send to the zstage.

           **Returns:**
            - str: The response from the zstage.

        """

        text = text + self.suffix
        self.serial_port.write(text)                                            # Write to serial port
        self.serial_port.flush()                                                # Flush serial port
        response = self.serial_port.readline()
        if self.logger is not None:
            self.logger.info('Zstage::txmt::'+text)
            self.logger.info('Zstage::rcvd::'+response)

        return  response


    def move(self, position):
        """Move all tilt motors to specified absolute step positions.

           **Parameters:**
            - position ([int, int, int]): List of absolute positions for each tilt
              motor.

           **Returns:**
            - [int, int, int]: List with absolute positions of each tilt motor
              after the move.

        """
        for i in range(3):
            if self.min_z <= position[i] <= self.max_z:
                if self.active:
                    self.command('T' + self.motors[i] + 'MOVETO ' +
                        str(position[i]))                                       # Move Absolute
            else:
                print("ZSTAGE can only move between " + str(self.min_z) +
                    ' and ' + str(self.max_z))

        return self.check_position()                                            # Check position

    # Check if Zstage motors are stopped and return their position
    def check_position(self):
        """Return a list with absolute positions of each tilt motor.

           **Returns:**
            - [int, int ,int]: List of absolution positions.

        """

        # Get Current position
        old_position = [0,0,0]
        for i in range(3):
            successful = True
            while successful:
                try:
                    old_position[i] = int(self.command('T' + self.motors[i] +
                        'RD')[5:])
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
                        new_position = int(self.command('T' + self.motors[i] +
                            'RD')[5:])                                          # Get current position
                        stopped = new_position == old_position[i]               # Compare old position to new position
                        all_stopped = all_stopped + stopped                     # all_stopped will = 3 if all 3 motors are in position
                        old_position[i] = new_position                          # Save new position
                        successful = False
                    except:
                        time.sleep(2)

        self.position = old_position                                            # Set position

        return self.position                                                    # Return position

    def in_position(self, position):
        """Return True if all motors are in position, False if not.

           **Parameters:**
            - position ([int,int,int]): List of motor positions to test.

           **Returns:**
            - bool: True if all motors are in position, False if not.

        """

        for i in range(3):
            if abs(position[i]-self.position[i]) <= self.tolerance:
                in_pos = True
            else:
                in_pos = False

        return in_pos

    def get_motor_points(self):
        """Return stage step coordinates tilt motors."""

        points = [[self.xstep[0], self.ystep[0], self.position[0]],
                  [self.xstep[1], self.ystep[1], self.position[1]],
                  [self.xstep[2], self.ystep[2], self.position[2]]]

        return points
