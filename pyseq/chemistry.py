"""Illumina HiSeq 2500 System :: ARM9 CHEM

   Uses commands found on `hackteria
   <www.hackteria.org/wiki/HiSeq2000_-_Next_Level_Hacking>`_

"""


class CHEM():
    """HiSeq 2500 System :: ARM9 CHEM

       **Attributes**
        - logger (logger): Logger used for messaging

    """
    def __init__(self, com_port, baudrate = 115200, logger = None):
        """The constructor for the ARM9 CHEM.

           **Parameters:**
            - com_port (str): The communication port for the ARM9 CHEM.
            - baudrate (int, optional): The communication speed in symbols per
              second.
            - logger (log, optional): The log file to write communication with
              the ARM9 CHEM.
            - version (str): Controller version.
            - T_fc [float, float]: Set temperature of flowcell in degrees C.
            - T_chiller (float): Set temperature of chiller in degrees C.


           **Returns:**
            - chem object: A chem object to control the ARM9 CHEM.

        """

        # Open Serial Port
        s = serial.Serial(com_port_command, baudrate, timeout = 1)

        # Text wrapper around serial port
        self.serial_port = io.TextIOWrapper(io.BufferedRWPair(s,s),
                                            encoding = 'ascii',
                                            errors = 'ignore')
        self.suffix = '\r'
        self.logger = logger
        self.version = None
        self.T_fc = [None, None]
        self.T_chiller = None
        self.min_fc_T = 20
        self.max_fc_T = 50
        self.min_chiller_T = 4
        self.max_chiller_T = 20


    def initialize(self):
        """Initialize the ARM9 CHEM"""

        response = self.command('INIT')                                         # Initialize ARM9 CHEM
        response = self.command('?IDN')                                         # Get Controller Version
        self.version = response

    def command(self, text):
        """Send a serial command to the ARM9 CHEM and return the response.

           **Parameters:**
            - text (str): A command to send to the ystage.

           **Returns:**
            - str: The response from the ystage.
        """

        text = text + self.suffix
        self.serial_port.write(text)                                            # Write to serial port
        self.serial_port.flush()                                                # Flush serial port
        response = self.serial_port.readline()
        if self.logger is not None:
            self.logger.info('CHEM::txmt::'+text)
            self.logger.info('CHEM::rcvd::'+response)
        else:
            print(response)

        return  response

    def get_fc_T(self, fc):
        """Return temperature of flowcell in degrees C."""

        if fc == 'A':
            fc = 0
        elif fc == 'B':
            fc = 1
        elif fc not in (0,1):
            write_log('get_fc_T::Invalid flowcell')

        response = self.command('?FCTEMP:'+str(fc))

        return float(response)

    def get_chiller_T(self, fc):
        """Return temperature of chiller in degrees C."""

        response = self.command('?RETEMP')

        return float(response)

    def set_fc_T(self, fc, T):
        """Return temperature of flowcell in degrees C."""

        if fc == 'A':
            fc = 0
        elif fc == 'B':
            fc = 1
        elif fc not in (0,1):
            write_log('set_fc_T::Invalid flowcell')

        if self.min_fc_T >= T:
            T = self.min_fc_T
            write_log('set_fc_T::Set temperature too cold, setting to ' + str(T))
        elif self.max_fc_T <= T:
            T = self.max_fc_T
            write_log('set_fc_T::Set temperature too hot, setting to ' + str(T))

        response = self.command('?FCTEMP:'+str(fc)+':'+float(T))

        return float(response)

    def set_chiller_T(self, T):
        """Return temperature of chiller in degrees C."""

        if self.min_chiller_T >= T:
            T = self.min_chiller_T
            write_log('set_chiller_T::Set temperature too cold, setting to ' + str(T))
        elif self.max_chiller_T <= T:
            T = self.max_chiller_T
            write_log('set_chiller_T::Set temperature too hot, setting to ' + str(T))

        response = self.command('?RETEMP:'+str(T))

        return float(response)

    def write_log(self, text):
        """Write messages to the log."""

        msg = 'CHEM::' + text
        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)
