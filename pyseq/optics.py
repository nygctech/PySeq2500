#!/usr/bin/python
"""Illumina HiSeq 2500 System :: Optics
Uses commands found on www.hackteria.org/wiki/HiSeq2000_-_Next_Level_Hacking

Controls the excitation and emission filters on the Illumina HiSeq 2500
System. The excitation filters are optical density filters that block a
portion of the light to quickly change between laser intensities. The
percent of light passed through is 10**-OD*100 where OD is the optical
density of the filter. All of the light is blocked at the home "filter".
The names and OD of available filters are listed in the following table.

===========  ===========  ==================================
laser color  laser index  filters
===========  ===========  ==================================
green        1            0.2, 0.6, 1.4, 1.6, 2.0, 4.0, home
red          2            0.2, 0.9, 1.0, 2.0, 3.0, 4.5, home
===========  ===========  ==================================

The emission filter has 2 states, in the light path or out of the light
path.

Examples:
    #Create optics object
    >>>import pyseq
    >>>fpga = pyseq.fpga.FPGA('COM12','COM15')
    >>>fpga.initialize()
    >>>optics = pyseq.optics.Optics(fpga)
    #Initialize optics
    >>>optics.initialize()
    # Move to green line OD1.6 filter and red line OD1.0 filter
    >>>optics.move_ex(1,'1.6')
    >>>optics.move_ex(2,'1.0')
    # Move the emission filter out of the light path
    >>>optics.move_em_in(False)

Kunal Pandit 9/19
"""

import time


class Optics():
    """Illumina HiSeq 2500 System :: Optics

    Attributes:
    ex ([str,str]): The current excitation filter for each laser line. The
        first filter is for the green laser and the second filter for the
        red laser.
    em_in (bool): True if the emission filter is in the light path or False
        if the emission filter is out of the light path.
    """


    def __init__(self, fpga, logger = None):
        """Constructor for the optics.

           Parameters:
           fpga (fpga object): The Illumina HiSeq 2500 System :: FPGA.
           logger (log, optional): The log file to write communication with the
                optics to.
        """

        self.serial_port = fpga
        self.logger = logger
        self.ex = [None, None]
        self.em_in = None
        self.suffix = '\n'
        self.ex_dict = [
                        # EX1
                        {'home' : 0,
                         0.2 : -36,
                         0.6 : -71,
                         1.4 : -107,
                         'open'  : 143,
                         1.6 : 107,
                         2.0 : 71,
                         4.0 : 36},
                        # EX
                        {'home' : 0,
                         4.5 : 36,
                         3.0 : 71,
                         0.2 : -107,
                         'open' : 143,
                         2.0 : 107,
                         1.0 : -36,
                         0.9: -71}
                        ]


    def initialize(self):
        """Initialize the optics.

           The default position for the excitation filters is home
           which blocks excitation light. The default position for
           the emission filter is in the light path.
        """

        #Home Excitation Filters
        self.move_ex(1, 'home')
        self.move_ex(2, 'home')

        # Move emission filter into light path
        self.move_em_in(True)


    def command(self, text):
        """Send a command to the optics and return the response.

           Parameters:
           text (str): A command to send to the optics.

           Returns:
           str: The response from the optics.
        """

        text = text + self.suffix
        self.serial_port.write(text)                                            # Write to serial port
        self.serial_port.flush()                                                # Flush serial port
        response = self.serial_port.readline()
        if self.logger is not None:
            self.logger.info('optics::txmt::'+text)
            self.logger.info('optics::rcvd::'+response)

        return  response


    def move_ex(self, wheel, position):
        """Move the excitation wheel to the specified position.

           The excitation filters are optical density filters that block a
           portion of the light to quickly change between laser intensities.
           The percent of light passed through is 10**-OD*100 where OD is
           the optical density of the filter. All of the light is blocked with
           the home "filter". The names and OD of available filters are listed
           in the following table.

           ===========  ===========  ==================================
           laser color  laser index  filters
           ===========  ===========  ==================================
           green        1            0.2, 0.6, 1.4, 1.6, 2.0, 4.0, home
           red          2            0.2, 0.9, 1.0, 2.0, 3.0, 4.5, home
           ===========  ===========  ==================================

           Parameters:
           wheel (int): The index of laser line where 1 = green laser or
                2 = red laser.
           position (str): The name of the filter to change to.
        """

        if wheel != 1 and wheel != 2:
            print('Choose excitation filter 1 or 2')
        elif position in self.ex_dict[wheel-1].keys():
            self.command('EX' + str(wheel)+ 'HM')                               # Home Filter
            self.ex[wheel-1] = position
            if position != 'home':
                position = str(self.ex_dict[wheel-1][position])                 # get step position
                self.command('EX' + str(wheel) + 'MV ' + position)              # Move Filter relative to home
        elif position not in self.ex_dict[wheel-1].keys():
            print(position + ' filter does not exist in excitation ' +
                  str(wheel))


    def move_em_in(self, INorOUT):
        """Move the emission filter in to or out of the light path.

           Parameters:
           INorOUT (bool): True for the emission in the light path or
                False for the emission filter out of the light path.
        """

        # Move emission filter into path
        if INorOUT:

            self.command('EM2I')
            self.em_in = True
        # Move emission filter out of path
        else:
            self.command('EM2O')
            self.em_in = False
