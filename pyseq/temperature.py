"""Illumina HiSeq 2500 System :: ARM9 CHEM

   Uses commands found on `hackteria
   <www.hackteria.org/wiki/HiSeq2000_-_Next_Level_Hacking>`_

   **Example:**

    .. code-block:: python

        #Create ARM9 Chemistry object
        import pyseq
        chem = pyseq.chemistry.CHEM('COM8')
        #Initialize ARM Chemistry
        chem.initialize()
        # Get temperature of flowcell A
        chem.get_fc_T('A')
        # Set temperature of flowcell A
        chem.set_fc_T('A', 55.0)
        # Set temperature of flowcell A and block until temperature is reached
        chem.wait_fc_T('A'), 55.0

    Temperatures of flowcells A and B can be independently controlled. The
    min temperature is 20 °C and the max temperature is 60 °C.

"""

import serial
import io
import time

class Temperature():
    """HiSeq 2500 System :: Stage & Chiller Temperature Control (ARM9 CHEM)

       **Attributes**
        - serial_port: IO wrapper around serial port
        - suffix: Suffix to send commands
        - logger (logger): Logger used for messaging
        - version (str): Controller version.
        - T_fc [float, float]: Set temperature of flowcell in °C.
        - T_chiller [float, float, float]: Set temperature of chillers in °C.
        - min_fc_T (float): Minimum flowcell temperature in  °C.
        - max_fc_T (float): Maximum flowcell temperature in  °C.
        - min_chiller_T (float): Minimum chiller temperature in °C.
        - max_chiller_T (float): Maximum flowcell temperature in °C.
        - fc_PIDSF0: Flowcell A Temperature servo-loop parameters
        - fc_PIDSF1: Flowcell B Temperature servo-loop parameters
        - tec_PIDSF0: Chiller 0 Temperature servo-loop parameters
        - tec_PIDSF1: Chiller 1 Temperature servo-loop parameters
        - tec_PIDSF2: Chiller 2 Temperature servo-loop parameters
        - p_ = servo-loop parameters: Servo_Proportional, Servo_Integral, Servo_Derivative, Feed_Frw_StepSize, Feed_frw_Threshold
        - delay (int): Delay time in querying temperature.

    """
    def __init__(self, com_port, baudrate = 115200, logger = None):
        """The constructor for the ARM9 CHEM.

           **Parameters:**
            - com_port (str): The communication port for the ARM9 CHEM.
            - baudrate (int, optional): The communication speed in symbols per
              second.
            - logger (log, optional): The log file to write communication with
              the ARM9 CHEM.


           **Returns:**
            - chem object: A chem object to control the ARM9 CHEM.

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
            print('ERROR::Check ARM9 CHEM Port')
            self.serial_port = None

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
        self.fc_PIDSF1 = (0.2, 0.1, 0.0, 1.875, 6.0)        # flowcell B
        self.tec_PIDSF0 = (0.8, 0.2, 0.0, 1.875, 6.0)       # TEC0
        self.tec_PIDSF1 = (0.8, 0.2, 0.0, 1.875, 6.0)       # TEC1
        self.tec_PIDSF2 = (1.7, 1.1, 0.0)                   # TEC2
        self.p_ = 'PIDSF'
        self.delay = 60                                     # delay time in s
        self.busy = False


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

        self.fc_off(0)
        self.fc_off(1)


    def command(self, text):
        """Send a serial command to the ARM9 CHEM and return the response.

           **Parameters:**
            - text (str): A command to send to the ystage.

           **Returns:**
            - str: The response from the ystage.
        """

        # Block communication if busy
        while self.busy:
            pass

        self.busy = True
        text = text + self.suffix
        self.serial_port.write(text)                                            # Write to serial port
        self.serial_port.flush()                                                # Flush serial port
        r = 1
        response = 'No response'
        while r != '':
            r = self.serial_port.readline()
            if 'A1' in r:
                response = r

        if self.logger is not None:
            self.logger.info('ARM9CHEM::txmt::'+text)
            self.logger.info('ARM9CHEM::rcvd::'+response)
        else:
            print(response)

        self.busy = False

        return response

    def get_fc_index(self, fc):
        """Return flowcell index."""

        if fc == 'A':
            fc = 0
        elif fc == 'B':
            fc = 1
        elif fc not in (0,1):
            self.write_log('get_fc_index::Invalid flowcell')
            fc = None

        return fc

    def get_fc_T(self, fc):
        """Return temperature of flowcell in °C."""

        fc = self.get_fc_index(fc)

        T = None
        if fc is not None:
            while T is None:
                try:
                    response = self.command('?FCTEMP:'+str(fc))
                    T = float(response.split(':')[0][:-1])
                except:
                    T = None

        return T

    def get_chiller_T(self):
        """Return temperature of chiller in °C.

           NOTE: There are 3 TEC blocks cooling the chiller. I'm not sure if all
           3 cool chiller. I think the 3rd one cools something else because the
           control parameters are different. - Kunal

        """

        response = self.command('?RETEMP:3')
        response = response.split(':')
        T = [None,None,None]
        for i, t in enumerate(T):
              T[i] = float(response[i])

        return T

    def set_fc_T(self, fc, T):
        """Set temperature of flowcell in °C.

           **Parameters:**
            - fc (str or int): Flowcell position either A or 0, or B or 1
            - T (float): Temperature in °C.

           **Returns:**
            - (float): Current temperature in °C.

        """

        fc = self.get_fc_index(fc)

        if type(T) not in [int, float]:
            self.write_log('set_fc_T::ERROR::Temperature must be a number')
            T = None

        if fc is not None and T is not None:
            if self.min_fc_T > T:
                T = self.min_fc_T
                self.write_log('set_fc_T::Set temperature too cold, ' +
                               'setting to ' + str(T))
            elif self.max_fc_T < T:
                T = self.max_fc_T
                self.write_log('set_fc_T::Set temperature too hot, ' +
                               'setting to ' + str(T))

            if self.T_fc[fc] is None:
                self.fc_on(fc)

            response = self.command('FCTEMP:'+str(fc)+':'+str(T))
            self.T_fc[fc] = T
            current_temp = self.get_fc_T(fc)
        else:
            current_temp = None

        return current_temp

    def wait_fc_T(self, fc, T):
        """Set and wait for flowcell to reach temperature in °C.

           **Parameters:**
            - fc (str or int): Flowcell position either A or 0, or B or 1
            - T (float): Temperature in °C.

           **Returns:**
            - (float): Current temperature in °C.

        """


        try:
            error = abs(T - self.set_fc_T(fc,T))
            while error > 1:
                time.sleep(self.delay)
                error = abs(T - self.get_fc_T(fc))
        except:
            self.write_log('Unable to wait for flowcell', fc, 'to reach', T, '°C')

        T = self.get_fc_T(fc)

        return T

    def fc_off(self, fc):
        """Turn off temperature control for flowcell fc."""

        fc = self.get_fc_index(fc)
        response = self.command('FCTEC:'+str(fc)+':0')
        self.T_fc[fc] = None

        return False

    def fc_on(self, fc):
        """Turn on temperature control for flowcell fc."""


        fc = self.get_fc_index(fc)
        response = self.command('FCTEC:'+str(fc)+':1')
        self.T_fc[fc] = self.get_fc_T(fc)

        return True

    def set_chiller_T(self, T, i):
        """Return temperature of chiller in °C."""

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

    def write_log(self, *args):
        """Write messages to the log."""


        msg = 'ARM9CHEM::'
        for a in args:
            msg += ' ' + str(a)

        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)
