#!/usr/bin/python
"""Illumina HiSeq 2500 System :: FPGA

   The FPGA arms triggers for the TDI cameras.
   The FPGA also controls the z stage, objective stage, optics, LEDS, and stage
   temperature.

   Commands from `hackteria
   <www.hackteria.org/wiki/HiSeq2000_-_Next_Level_Hacking>`_

   **Example:**

   .. code-block:: python

        #Create FPGA object
        import pyseq
        fpga = pyseq.fpga.FPGA('COM12','COM15')
        #Initialize FPGA
        fpga.initialize()
        # Read write encoder position (to sync with y stage).
        fpga.read_position()
        0
        fpga.write_position(0)
        # Arm y stage triggers for TDI imgaging.
        fpga.TDIYPOS(3000000)
        fpga.TDIYPOS3(4096,3000000)
        # Set LEDs to nightride mode (blue sweep)
        fpga.led('A', 'sweep blue')
        fpga.led('B', 'sweep blue')


"""


import serial
import io
import time


# FPGA object

class FPGA():
    """HiSeq 2500 System :: FPGA

       **Attributes**
        - led_dict (dict): Dictionary of command options for different LED modes.
        - y_offset (int): FPGA offset of ystage position
        - logger (logger): Logger used for messaging

    """



    def __init__(self, com_port_command, com_port_response, baudrate = 115200, logger = None):
        """The constructor for the FPGA.

           **Parameters:**
            - com_port_command (str): The communication port to send FPGA
              commands.
            - com_port_response (str): The communication port to receive FPGA
              responses.
            - baudrate (int, optional): The communication speed in symbols per
              second.
            - logger (log, optional): The log file to write communication with
              the FPGA.


           **Returns:**
            - fpga object: A fpga object to control the FPGA.

        """
        if isinstance(com_port_command, int):
            com_port_command = 'COM'+str(com_port)
        if isinstance(com_port_response, int):
            com_port_response = 'COM'+str(com_port)

        try:
            # Open Serial Port
            s_command = serial.Serial(com_port_command, baudrate, timeout = 1)
            s_response = serial.Serial(com_port_response, baudrate, timeout = 1)

            # Text wrapper around serial port
            self.serial_port = io.TextIOWrapper(io.BufferedRWPair(s_response,s_command),
                                                encoding = 'ascii',
                                                errors = 'ignore')
        except:
            print('ERROR::Check FPGA Ports')
            self.serial_port = None

        self.suffix = '\n'
        self.y_offset = 7000000
        self.logger = logger
        self.busy = False
        self.led_dict = {'off':'0', 'yellow':'1', 'green':'3', 'pulse green':'4',
                         'blue':'5', 'pulse blue':'6', 'sweep blue': '7'}


    def initialize(self):
        """Initialize the FPGA."""

        response = self.command('RESET')                                        # Initialize FPGA
        self.command('EX1HM')                                                   # Home excitation filter on laser line 1
        self.command('EX2HM')                                                   # Home excitation filter on laser line 2
        self.command('EM2I')                                                    # Move emission filter into light path
        self.command('SWLSRSHUT 0')                                             # Shutter lasers
        self.LED(1,'off')
        self.LED(2,'off')

    def command(self, text):
        """Send commands to the FPGA and return the response.

           **Parameters:**
            - text (str): A command to send to the FPGA.

           **Returns:**
            - str: The response from the FPGA.

        """
        # Block communication if busy
        while self.busy:
            pass

        self.busy = True
        text = text + self.suffix
        self.serial_port.write(text)                                    # Write to serial port
        self.serial_port.flush()                                        # Flush serial port
        response = self.serial_port.readline()
        if self.logger is not None:
            self.logger.info('FPGA::txmt::'+text)
            self.logger.info('FPGA::rcvd::'+response)
        else:
            print(response)

        self.busy = False

        return  response


    def read_position(self):
        """Read the y position of the encoder for TDI imaging.

           **Returns:**
            - int: The y position of the encoder.

        """
        tdi_pos = None
        while not isinstance(tdi_pos, int):
            try:
                tdi_pos = self.command('TDIYERD')
                tdi_pos = tdi_pos.split(' ')[1]
                tdi_pos = int(tdi_pos[0:-1]) - self.y_offset
            except:
                tdi_pos = None
        return tdi_pos


    def write_position(self, position):
        """Write the position of the y stage to the encoder.

           Allows for a 5 step (50 nm) error.

           **Parameters:**
            - position (int) = The position of the y stage.

        """
        position = position + self.y_offset
        while abs(self.read_position() + self.y_offset - position) > 5:
            self.command('TDIYEWR ' + str(position))
            time.sleep(1)


    def TDIYPOS(self, y_pos):
        """Set the y position for TDI imaging.

           **Parameters:**
            - y_pos (int): The initial y position of the image.

        """
        self.command('TDIYPOS ' + str(y_pos+self.y_offset-80000))


    def TDIYARM3(self, n_triggers, y_pos):
        """Arm the y stage triggers for TDI imaging.

           **Parameters:**
            - n_triggers (int): Number of triggers to send to the cameras.
            - y_pos (int): The initial y position of the image.

        """

        self.command('TDIYARM3 ' + str(n_triggers) + ' ' +
                  str(y_pos + self.y_offset-10000) + ' 1')



    def LED(self, AorB, mode, **kwargs):
        """Set front LEDs.

           **Parameters:**
            - AorB (int/str): A or 1 for the left LED, B or 2 for the right LED.
            - mode (str): Color / mode to set the LED to, see list below.
            - kwargs: sweep (1-255): sweep rate
                      pulse (1-255): pulse rate

           **Available Colors/Modes:**
            - off
            - yellow
            - green
            - pulse green
            - blue
            - pulse blue
            - sweep blue

           **Returns:**
            - bool: True if AorB and mode are valid, False if not.

        """

        s = None
        if type(AorB) is str:
            if AorB.upper() == 'A':
                s = '1'
            elif AorB.upper() == 'B':
                s = '2'
        elif type(AorB) is int:
            if AorB == 1 or AorB == 2:
                s = str(AorB)

        m = None
        if mode in self.led_dict:
            m = self.led_dict[mode]

        if s is not None and m is not None:
            response = self.command('LEDMODE' + s + ' ' + m)
            worked = True
        else:
            worked =  False

        for key, value in kwargs.items():
            if 1 <= value <= 255:
                value = str(int(value))

                if key == 'sweep':
                    response = self.command('LEDSWPRATE ' + value)
                elif key == 'pulse':
                    response = self.command('LEDPULSRATE ' + value)

        return worked
