"""Illumina HiSeq 2500 System :: ARM9 CHEM

   Uses commands found on `hackteria
   <www.hackteria.org/wiki/HiSeq2000_-_Next_Level_Hacking>`_

"""

import serial
import io

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
        s = serial.Serial(com_port, baudrate, timeout = 1)

        # Text wrapper around serial port
        self.serial_port = io.TextIOWrapper(io.BufferedRWPair(s,s),
                                            encoding = 'ascii',
                                            errors = 'ignore')
        self.suffix = '\r'
        self.logger = logger
        self.version = None
        self.T_fc = [None, None]
        self.T_chiller = [None, None, None]
        self.min_fc_T = 20.0
        self.max_fc_T = 60.0
        self.min_chiller_T = 0.1
        self.max_chiller_T = 20.0
        #Temperature servo-loop parameters
        #Servo_Proportional, Servo_Integral, Servo_Derivative, Feed_Frw_StepSize, Feed_frw_Threshold
        self.fc_PIDSF0 = (0.2, 0.1, 0.0, 1.875, 6.0)        # flowcell A
        self.fc_PIDSF1 = (0.2, 0.1, 0.0, 1.875, 6.0)        # flowecell B
        self.tec_PIDSF0 = (0.8, 0.2, 0.0, 1.875, 6.0)       # TEC0
        self.tec_PIDSF1 = (0.8, 0.2, 0.0, 1.875, 6.0)       # TEC1
        self.tec_PIDSF2 = (1.7, 1.1, 0.0)                   # TEC2
        self.p_ = 'PIDSF'


    def initialize(self):
        """Initialize the ARM9 CHEM"""

        response = self.command('INIT')                                         # Initialize ARM9 CHEM
        response = self.command('?IDN')                                         # Get Controller Version
        self.version = response
        # Set flowcell/chiller temperature control parameters
        for i, p in enumerate(self.fc_PIDSF0):
            reponse = self.command('FCTEMP:0:'+self.p_[i]+':'+str(p))
        for i, p in enumerate(self.fc_PIDSF1):
            reponse = self.command('FCTEMP:1:'+self.p_[i]+':'+str(p))
        for i, p in enumerate(self.tec_PIDSF0):
            reponse = self.command('RETEC:0:'+self.p_[i]+':'+str(p))
        for i, p in enumerate(self.tec_PIDSF1):
            reponse = self.command('RETEC:1:'+self.p_[i]+':'+str(p))    
        for i, p in enumerate(self.tec_PIDSF2):
            reponse = self.command('RETEC:2:'+self.p_[i]+':'+str(p))

              
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
        r = self.serial_port.readline()
        response = None
        while '' != r:
            if 'A1' in r:
                response = r
            r = self.serial_port.readline()
            
        if self.logger is not None:
            self.logger.info('ARM9CHEM::txmt::'+text)
            self.logger.info('ARM9CHEM::rcvd::'+response)
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
        T = float(response.split(':')[0][:-1])

        return T

    def get_chiller_T(self):
        """Return temperature of chiller in degrees C.

            There are 2 TEC blocks cooling the chiller
        """

        response = self.command('?RETEMP:3')
        response = response.split(':')
        T = [None,None,None]
        for i, t in enumerate(T):
              T[i] = float(response[i])

        return T
    
    def set_fc_T(self, fc, T):
        """Return temperature of flowcell in degrees C."""

        if fc == 'A':
            fc = 0
        elif fc == 'B':
            fc = 1
        elif fc not in (0,1):
            self.write_log('set_fc_T::Invalid flowcell')

        if self.min_fc_T > T:
            T = self.min_fc_T
            self.write_log('set_fc_T::Set temperature too cold, setting to ' + str(T))
        elif self.max_fc_T < T:
            T = self.max_fc_T
            self.write_log('set_fc_T::Set temperature too hot, setting to ' + str(T))

        response = self.command('FCTEMP:'+str(fc)+':'+str(T))
        response = self.command('FCTEC:'+str(fc)+':1')
        self.T_fc[fc] = T

        return response

    def set_chiller_T(self, T, i):
        """Return temperature of chiller in degrees C."""

        T = float(T)
        if self.min_chiller_T >= T:
            T = self.min_chiller_T
            self.write_log('set_chiller_T::Set temperature too cold, setting to ' + str(T))
        elif self.max_chiller_T <= T:
            T = self.max_chiller_T
            self.write_log('set_chiller_T::Set temperature too hot, setting to ' + str(T))

        if i not in (0,1,2):
            self.write_log('set_chiller_T::Chiller index must 0, 1, or 2')
            response = None
        else:
            response = self.command('RETEMP:'+str(i)+':'+str(T))
            self.T_chiller[i] = T

        return response

    def write_log(self, text):
        """Write messages to the log."""

        msg = 'ARM9CHEM::' + text
        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)
