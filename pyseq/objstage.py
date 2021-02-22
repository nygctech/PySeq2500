#!/usr/bin/python
"""Illumina HiSeq2500 :: Objective Stage

   Uses commands found on `hackteria
   <www.hackteria.org/wiki/HiSeq2000_-_Next_Level_Hacking>`_

   The objective can move between steps 0 and 65535, where step 0 is
   the closest to the stage. Each objective stage step is about 4 nm.

   **Examples:**

.. code-block:: python

    #Create an objective stage objective
    import pyseq
    fpga = pyseq.fpga.FPGA('COM12','COM15')
    fpga.initialize()
    obj = pyseq.objstage.OBJstage(fpga)
    #Initialize the objective stage
    obj.initialize()
    # Change objective velocity to 1 mm/s and move to step 5000
    obj.set_velocity(1)
    obj.move(5000)

"""


import time
from math import ceil, floor


class OBJstage():
    """HiSeq 2500 System :: Objective Stage

       **Attributes:**
        - spum (int): The number of objective steps per micron.
        - v (float): The velocity the objective will move at in mm/s.
        - position (int): The absolute position of the objective in steps.
        - min_z (int): Minimum obj stage step position.
        - max_z (int): Maximum obj stage step position.
        - min_v (int): Minimum velocity in mm/s.
        - max_v (int): Maximum velocity in mm/s.
        - focus_spacing: Distance in microns between frames in an objective stack
        - focus_velocity (float): Velocity used for objective stack
        - focus_frames (int): Number of camera frames used for objective stack
        - focus_range (float): Percent of total objective range used for objective stack
        - focus_start (int): Initial step for objective stack.
        - focus_stop (int): Final step for objective stack.
        - focus_rough (int): Position used for imaging when focus position is
          not known.
        - logger (logger): Logger used for messaging.


    """


    def __init__(self, fpga, logger = None):
        """The constructor for the objective stage.

           **Parameters:**
            - fpga (fpga object): The Illumina HiSeq 2500 System :: FPGA.
            - logger (log, optional): The log file to write communication with
                                      the objective stage to.

           **Returns:**
            - objective stage object: A objective stage object to control the
              position of the objective.
        """

        self.serial_port = fpga
        self.min_z = 0
        self.max_z = 65535
        self.spum = 262                                                         #steps per um
        self.max_v = 5                                                          #mm/s
        self.min_v = 0.1                                                          #mm/s
        self.v = None                                                           #mm/s
        self.suffix = '\n'
        self.position = None
        self.logger = logger
        self.focus_spacing = 0.5                                                # distance in microns between frames in obj stack
        self.focus_velocity = 0.1                                               #mm/s
        self.focus_frames = 200                                                 # number of total camera frames for obj stack
        self.focus_range = 90                                                   #%
        self.focus_start =  2000                                                # focus start step
        self.focus_stop = 62000                                                 # focus stop step
        self.focus_rough = int((self.max_z - self.min_z)/2 + self.min_z)
        self.timeout = 100

    def initialize(self):
        """Initialize the objective stage."""

        # Update the position of the objective
        self.position = self.check_position()
        #Set velocity to 5 mm/s
        self.set_velocity(5)


    def command(self, text):
        """Send a command to the objective stage and return the response.

           **Parameters:**
            - text (str): A command to send to the objective stage.

           **Returns:**
            - str: The response from the objective stage.

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

           **Parameters:**
            - position (int): The step position to move the objective to.

        """

        if self.min_z <= position <= self.max_z:
            try:
                position = int(position)
                start = time.time()
                while self.check_position() != position:
                    response = self.command('ZMV ' + str(position))                 # Move Objective
                    if (time.time() - start) > self.timeout:
                        self.check_position()
                        break
            except:
                self.check_position()
                self.write_log('ERROR::Could not move objective')
        else:
            self.write_log('ERROR::Objective position out of range')


    def check_position(self):
        """Return the absolute step position of the objective.

           The objective can move between steps 0 and 65535, where step 0 is
           the closest to the stage. If the position of the objective can't be
           read, None is returned.

           **Returns:**
            - int: The absolution position of the objective steps.
        """

        try:
            position = self.command('ZDACR')                                    # Read position
            position = position.split(' ')[1]
            position = int(position[0:-1])
            self.position = position
            return position
        except:
            self.write_log('WARNING:: Could not read objective position')
            return None


    def set_velocity(self, v):
        """Set the velocity of the objective.

           The maximum objective velocity is 5 mm/s. If the objective velocity
           is not in range, the velocity is not set and an error message is
           printed.

           **Parameters:**
            - v (float): The velocity for the objective to move at in mm/s.
        """

        if self.min_v <= v <= self.max_v:
            self.v = v
            # convert mm/s to steps/s
            v = int(v * 1288471)                                                #steps/mm
            self.command('ZSTEP ' + str(v))                                     # Set velocity
        else:
            self.write_log('ERROR::Objective velocity out of range')

    def set_focus_trigger(self, position):
        """Set trigger for an objective stack to determine focus position.

           **Parameters:**
            - position (int): Step position to start imaging.

           **Returns:**
            - int: Current step position of the objective.

        """

        self.command('ZTRG ' + str(position))
        self.command('ZYT 0 3')

        return self.check_position()

    def update_focus_limits(self, cam_interval=0.040202, range=90, spacing=4.1):
        """Update objective velocity and start/stop positions for focusing.

           **Parameters:**
            - cam_interval (float): Camera frame interval in seconds per frame
            - range(float): Percent of total objective range to use for focusing
            - spacing (float): Distance between objective stack frames in microns.

           **Returns:**
            - bool: True if all values are acceptable.
        """


        # Calculate velocity needed to space out frames
        velocity = spacing/cam_interval/1000                               #mm/s
        if self.min_v > velocity:
            spacing = self.min_v*1000*cam_interval
            velocity = self.min_v
            print('Spacing too small, changing to ', spacing)
        elif self.max_v < velocity:
            spacing = self.max_v*1000*cam_interval
            velocity = self.max_v
            print('Spacing too large, changing to ', spacing)

        self.focus_spacing = spacing
        self.focus_velocity = velocity
        spf = spacing*self.spum                                                 # steps per frame

        # Update focus range, ie start and stop step positions
        if 1 <= range <= 100:
            self.focus_range = range
            range_step = int(range/100*(self.max_z-self.min_z)/2)
            self.focus_stop = self.focus_rough+range_step
            self.focus_start = self.focus_rough-range_step
            self.focus_frames = ceil((self.focus_stop-self.focus_start)/spf)
            self.focus_frames += 100
            acceptable = True
        else:
            acceptable = False

        return acceptable


    def write_log(self, text):
        """Write messages to the log."""

        if self.logger is None:
            print('OBJstage::'+text)
        else:
            self.logger.info('OBJstage::'+text)
