#!/usr/bin/python
"""Illumina HiSeq2500 :: Objective Stage
Uses commands found on www.hackteria.org/wiki/HiSeq2000_-_Next_Level_Hacking

The objective can move between steps 0 and 65535, where step 0 is
the closest to the stage. Each objective stage step is about 4 nm.

Examples:
    #Create an objective stage objective
    >>>import pyseq
    >>>fpga = pyseq.fpga.FPGA('COM12','COM15')
    >>>fpga.initialize()
    >>>obj = pyseq.objstage.OBJstage(fpga)
    #Initialize the objective stage
    >>>obj.initialize()
    # Change objective velocity to 1 mm/s and move to step 5000
    >>>obj.set_velocity(1)
    >>>obj.move(5000)

Kunal Pandit 9/19
"""


import time


class OBJstage():
    """HiSeq 2500 System :: Objective Stage

       Attributes:
       spum (int): The number of objective steps per micron.
       v (float): The velocity the objective will move at in mm/s.
       position (int): The absolute position of the objective in steps.
    """


    def __init__(self, fpga, logger = None):
        """The constructor for the objective stage.

           Parameters:
           fpga (fpga object): The Illumina HiSeq 2500 System :: FPGA.
           logger (log, optional): The log file to write communication with the
                objective stage to.

           Returns:
           objective stage object: A objective stage object to control the
                position of the objective.
        """

        self.serial_port = fpga
        self.min_z = 0
        self.max_z = 65535
        self.spum = 262                                                         #steps per um
        self.max_v = 5                                                          #mm/s
        self.min_v = 0                                                          #mm/s
        self.v = None
        self.suffix = '\n'
        self.position = None
        self.logger = logger


    def initialize(self):
        """Initialize the objective stage."""

        # Update the position of the objective
        self.position = self.check_position()
        #Set velocity to 5 mm/s
        self.set_velocity(5)


    def command(self, text):
        """Send a command to the objective stage and return the response.

           Parameters:
           text (str): A command to send to the objective stage.

           Returns:
           str: The response from the objective stage.
        """

        text = text + self.suffix
        self.serial_port.write(text)
        self.serial_port.flush()
        response = self.serial_port.readline()
        if self.logger is not None:
            self.logger.info('OBJstage::txmt::'+text)
            self.logger.info('OBJstage::rcvd::'+response)

        return  response


    def move(self, position):
        """Move the objective to an absolute step position.

           The objective can move between steps 0 and 65535, where step 0 is
           the closest to the stage. If the position is out of range, the
           objective will not move and a warning message is printed.

           Parameters:
           position (int): The step position to move the objective to.
        """

        if position >= self.min_z and position <= self.max_z:
            try:
                while self.check_position() != position:
                    self.command('ZMV ' + str(position))                        # Move Objective

            except:
                print('Error moving objective stage')
        else:
            print('Objective position out of range')


    def check_position(self):
        """Return the absolute step position of the objective.

           The objective can move between steps 0 and 65535, where step 0 is
           the closest to the stage. If the position of the objective can't be
           read, None is returned.

           Returns:
           int: The absolution position of the objective steps.
        """

        try:
            position = self.command('ZDACR')                                    # Read position
            position = position.split(' ')[1]
            position = int(position[0:-1])
            self.position = position
            return position
        except:
            print('Error reading position of objective')
            return None


    def set_velocity(self, v):
        """Set the velocity of the objective.

           The maximum objective velocity is 5 mm/s. If the objective velocity
           is not in range, the velocity is not set and an error message is
           printed.

           Parameters:
           v (float): The velocity for the objective to move at in mm/s.
        """

        if v > self.min_v and v <= self.max_v:
            self.v = v
            # convert mm/s to steps/s
            v = int(v * 1288471)                                                #steps/mm
            self.command('ZSTEP ' + str(v))                                     # Set velocity
        else:
            print('Objective velocity out of range')

    def set_focus_trigger(self, position):
        """Set trigger for an objective stack to determine focus position.

           Parameters:
           position (int): Step position to start imaging.

           Returns:
           int: Current step position of the objective.

        """

        self.command('ZTRG ' + str(position))
        self.command('ZYT 0 3')

        return self.check_position

    def move_and_image(self, position):
        """Start imaging an objective stack.

           Parameters:
           position (int): Step position to move the objective to.

           Returns:
           int: Initial step position of objective.

        """
        initial = self.check_position

        self.command('ZMV ' + str(position))

        return initial
