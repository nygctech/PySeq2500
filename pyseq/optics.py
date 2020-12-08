#!/usr/bin/python
"""Illumina HiSeq 2500 System :: Optics

   Uses commands found on `hackteria
   <www.hackteria.org/wiki/HiSeq2000_-_Next_Level_Hacking>`_

   Controls the excitation and emission filters on the Illumina HiSeq 2500
   System. The excitation filters are optical density filters that block a
   portion of the light to quickly change between laser intensities. The
   percent of light passed through is 10**-OD*100 where OD is the optical
   density of the filter. All of the light is blocked, laser intensity = 0
   mW at the *home* "filter". None of the light is blocked, laser intensity
   = the set power of the laser, at the *open* "filter". The names and OD of
   available filters are listed in the following table.

    ===========  ===========  ========================================
    laser color  laser index  filters (Optical Density)
    ===========  ===========  ========================================
    green        1            open, 1.0, 2.0, 3.5, 3.8, 4.0, 4.5, home
    red          2            open, 0.2, 0.5, 0.6, 1.0, 2.4, 4.0, home
    ===========  ===========  ========================================

   The emission filter has 2 states, in the light path or out of the light
   path.

**Examples:**

.. code-block:: python

    #Create optics object
    import pyseq
    fpga = pyseq.fpga.FPGA('COM12','COM15')
    fpga.initialize()
    optics = pyseq.optics.Optics(fpga)
    #Initialize optics
    optics.initialize()
    # Move to green line OD1.6 filter and red line OD1.0 filter
    optics.move_ex('green','1.6')
    optics.move_ex('red','1.0')
    # Move the emission filter out of the light path
    optics.move_em_in(False)


"""

import time
import warnings


class Optics():
    """Illumina HiSeq 2500 System :: Optics

    **Attributes:**
     - ex ([str,str]): The current excitation filter for each laser line. The
       first filter is for the green laser and the second filter for the
       red laser.
     - em_in (bool): True if the emission filter is in the light path or False
       if the emission filter is out of the light path.
     - colors (dict): Laser dictionary of color keys and index values.
     - cycle_dict[dict,dict]: Dictionaries of filters to use for each laser line
       at different cycles. The first dictionary is for the green laser and the
       second dictionary is for the red laser.
     - focus_filters (list): Filter to use for each laser line during autofocus.

    """



    def __init__(self, fpga, logger = None, colors = ['green','red']):
        """Constructor for the optics.

           **Parameters:**
            - fpga (fpga object): The Illumina HiSeq 2500 System :: FPGA.
            - logger (log, optional): The log file to write communication with
              the optics to.
            - colors ([str,str], optional): The color of the laser lines.


           **Returns:**
            - optics object: An optics object to control the optical filters.

        """

        self.serial_port = fpga
        self.logger = logger
        self.ex = [None, None]
        self.em_in = None
        self.suffix = '\n'
        self.colors = [colors[0], colors[1]]
        self.cycle_dict = {'em':{}, self.colors[0]:{}, self.colors[1]:{}}
        self.focus_filters = [None, None]
        self.ex_dict = {
                        # EX1
                        colors[0]:
                        {'home' : 0,
                         4.5 : -36,
                         3.8 : -71,
                         3.5 : -107,
                         'open'  : 143,
                         1.0 : 107,
                         2.0 : 71,
                         4.0 : 36},
                        # EX
                        colors[1]:
                        {'home' : 0,
                         4.0 : 36,
                         2.4 : 71,
                         0.2 : -107,
                         'open' : 143,
                         1.0 : 107,
                         0.6 : -36,
                         0.5: -71}
                        }

        
    def initialize(self):
        """Initialize the optics.

           The default position for the excitation filters is home
           which blocks excitation light. The default position for
           the emission filter is in the light path.

        """

        #Home Excitation Filters
        for color in self.colors:
            self.move_ex(color, 'home')

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


    def move_ex(self, color, position):
        """Move the excitation wheel to the specified position.

           The excitation filters are optical density filters that block a
           portion of the light to quickly change between laser intensities.
           The percent of light passed through is 10**-OD*100 where OD is
           the optical density of the filter. All of the light is blocked with
           the *home* "filter". The names and OD of available filters are listed
           in the following table.

           ===========  ===========  ========================================
           laser color  laser index  filters
           ===========  ===========  ========================================
           green        1            open, 0.2, 0.6, 1.4, 1.6, 2.0, 4.0, home
           red          2            open, 0.2, 0.9, 1.0, 2.0, 3.0, 4.5, home
           ===========  ===========  ========================================

           **Parameters:**
            - color (str): The color of laser line.
            - position (str): The name of the filter to change to.

        """


        if color not in self.colors:
            warnings.warn('Laser color is invalid.')
        elif position in self.ex_dict[color].keys():
            index = self.colors.index(color)
            self.command('EX' + str(index+1)+ 'HM')                               # Home Filter
            self.ex[index] = position
            if position != 'home':
                time.sleep(2)
                position = str(self.ex_dict[color][position])                   # get step position
                self.command('EX' + str(index+1) + 'MV ' + position)              # Move Filter relative to home
        elif position not in self.ex_dict[color].keys():
            print(str(position) + ' excitation filter does not exist for ' +
                  color + ' laser.')


    def move_em_in(self, INorOUT):
        """Move the emission filter in to or out of the light path.

           **Parameters:**
            - INorOUT (bool): True for the emission in the light path or
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
