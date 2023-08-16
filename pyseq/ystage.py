#!/usr/bin/python
"""Illumina HiSeq 2500 and HiSeq X System :: Y-STAGE
   
   For HiSeq2500: Uses command set from Parker ViX 250IH & ViX500 IH
   For HISeqX: Uses command set from Copley Controls

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

        return ip


    def read_position(self):
        """Return the absolute step position of the ystage (int)."""

        try:
            self.position = int(self.command('R(PA)')[1:])                      # Read and store position
        except:
            pass

        return self.position

    def set_mode(self, mode):
        """Change between imaging and moving configurations."""

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


class YStageX():
    def __init__(self, com_port, baudrate = 9600, logger = None):
        if isinstance(com_port, int):
            com_port = 'COM'+str(com_port) 

                    # Open Serial Port
        s  = serial.Serial(com_port, baudrate, timeout = 1)
        # Text wrapper around serial port
        self.serial_port = io.TextIOWrapper(io.BufferedRWPair(s,s,),
                                            encoding = 'ascii',
                                            errors = 'ignore')
            
        self.min_y = -7000000
        self.max_y = 7500000
        self.spum = 100     # steps per um
        self.suffix = '\r\n'
        self.prefix = ''
        self.on = False
        self.home = 0
        self.configurations = {'imaging':{'g':'5,10,5,2,0'  ,'v':0.15400},
                               'moving': {'g':'7,10,5,1.5,0','v':1} #GF, GV, GP, GI, FT
                               }
        self.logger = logger
        #self.register = register
        
        self._position = None
        #self.error_list = [False for i in range(6)]
        self._velocity = 1000000
        self._accel = 10000
        self._is_moving = False
        self.PV = 1000 #position loop proportional gain 
        self.VF = 16384 #velocity feed forward
        self.AF = 0 #acceleration feed forward
        self.GM = 100 #position loop gain multiplier
        self._gains = ''
        self._mode_changed = True
        #homing values
        self.homed = False
        self.slow_velocity = 1
        self.fast_velocity = 25
        self.home_accel = 2500
        self.home_offset = 1000
        self.gen_gains = ''
        

        
        self.register_dict = {'Status Register': {0:'Short circuit detected', 1:'Drive over temperature', 2: 'Over voltage', 3: 'Under voltage', 4: ' Motor temperature sensor active', 5: ' Feedback error.', 6:' Motor phasing error.', 7: ' Current output limited.', 8: ' Voltage output limited.', 9: ' Positive limit switch active', 10: ' Negative limit switch active.', 11: ' Enable input not active.', 12: 'Drive is disabled by software', 13: ' Trying to stop motor', 14: 'Motor brake activated', 15: 'PWM outputs disabled.', 16: ' Positive software limit condition', 17: 'Negative software limit condition.', 18: 'Tracking error.', 19: 'Tracking warning', 20: ' Drive has been reset.', 21: ' Position has wrapped. The Position parameter cannot increase indefinitely. After reaching a certain value the parameter rolls back. This type of counting is called position wrapping or modulo count. Note that this bit is only active as the position wraps.', 22: ' Drive fault. A drive fault that was configured as latching has occurred. For information on latching faults, see the CME 2 User Guide.', 23: ' Velocity limit has been reached.', 24: ' Acceleration limit has been reached.', 25: 'Position outside of tracking window.', 26: ' Home switch is active', 27: 'Trajectory is stil running, motor has not yet settled into position.', 28: ' Velocity window. Set if the absolute velocity error exceeds the velocity window value.', 29: 'Phase not yet initialized. If the drive is phasing with no Halls, this bit is set until the drive has initialized its phase.', 30: ' Command fault. PWM or other command signal not present', 31: 'Error not defined'}, 'Trajectory Register' : {0: 'Reserved for future use.', 1: 'Reserved for future use.', 2: 'Reserved for future use.', 3: 'Reserved for future use.', 4: 'Reserved for future use.', 5: 'Reserved for future use.', 6:'Reserved for future use.', 7:'Reserved for future use.', 8:'Reserved for future use.', 9: 'Cam table underflow.', 10:'Reserved for future use', 11: 'Homing error. If set, an error occurred in the last home attempt. Cleared by a home command.', 12: 'Referenced. Set when a homing command has been successfully executed. Cleared by a home command.', 13: 'Homing. If set, the drive is running a home command', 14: 'Set when a move is aborted. Cleared at the start of the next move.', 15: 'In-Motion Bit. If set, the trajectory generator is presently generating a profile.'}, 'Latching Fault Status Register' : {0: 'Data flash CRC failure. This fault is considered fatal and cannot be cleared.', 1: 'Drive internal error. This fault is considered fatal and cannot be cleared.', 2: 'Short circuit.', 3: 'Drive over temperature.', 4: 'Motor over temperature.', 5: 'Over voltage.', 6: 'Under voltage. ', 7: 'Feedback fault.' , 8: 'Phasing error', 9: 'Following error', 10: 'Over Current (Latched)', 11: 'FPGA failure. This fault is considered fatal and cannot usually be cleared. If this fault occurred after a firmware download, repeating the download may clear this fault.', 12: 'Command input lost', 13: 'Reserved', 14: 'Safety circuit consistency check failure.', 15: 'Unable to control motor current', 16: 'Motor wiring disconnected', 17: 'Reserved.', 18: 'Safe torque off active.'}, 'Error codes' : {1: 'Too much data passed with command', 3: 'Unknown command code', 4: 'Not enough data was supplied with the command', 5: 'Too much data was supplied with the command', 9: 'Unknown parameter ID', 10: 'Data value out of range', 11: 'Attempt to modify read-only parameter', 14: 'Unknown axis state', 15: 'Parameter doesn’t exist on requested page', 16: 'Illegal serial port forwarding', 18: 'Illegal attempt to start a move while currently moving', 19: 'Illegal velocity limit for move', 20: 'Illegal acceleration limit for move', 21: 'Illegal deceleration limit for move', 22: 'Illegal jerk limit for move', 25: 'Invalid trajectory mode', 27: 'Command is not allowed while CVM is running', 31: 'Invalid node ID for serial port forwarding', 32: 'CAN Network communications failure', 33: 'ASCII command parsing error', 36: 'Bad axis letter specified', 46: 'Error sending command to encoder', 48: 'Unable to calculate filter'} }

    
    def initialize(self):
        """Initialize the ystage."""
        response = self.command('r')                                            # Initialize Stage

        #set Set the trajectory generator to absolute move, trapezoidal profile
        response = self.command('s r0xc8 0') 
        self.command('s r0xcb 1000000') 
        self.command('s r0xcc 10000') 
        self.command('s r0xcd 10000') 
        response = self.command('s r0x24 21')
        #set position properties
        
        #sets homing method 
        #self.command('s r0xc3 ' + str(self._fast_velocity) )        
        self.command('s r0xc4 ' + str(self.slow_velocity) )
        self.command('s r0xc5 ' + str(self.home_accel) )
        self.command('s r0xc6 ' + str(self.home_offset)) 
        
        self.command('s r0xc2 513')
        self.command('t 2')
            
    
    def parse_registers(self, response, register):
        binary = bin(int(response.split(' ')[0].strip()))
        bits = []
        for i, bit in enumerate(reversed(binary)):
            if bit == '1':
                bits.append(i)
                logger_message = self.register_dict[register][i]
                if self.logger is not None:
                    self.logger.info(f'Ystage::{register}::{i}::{logger_message}')  
                else:
                    print(f'Ystage::{register}::{i}::{logger_message}')
        return bits     

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
        else:
            print('Ystage::txmt::'+text)
            print(f'Ystage::rcvd::'+response)
            
        try: 
            accepted, value = response.split(' ')
            
            if accepted == 'e':
                self.parse_error(value)
                return False
            elif accepted == 'v':
                return value
        except: 
            return response
        
        
    def command_data(self, text):
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

        return response
        
        
    def parse_error(self, i):
        register = 'Error codes'
        i = int(i)
        logger_message = self.register_dict[register][i]
        if self.logger is not None:
            self.logger.info(f'Ystage::{register}::{i}::{logger_message}')  
        else:
            print(f'Ystage::{register}::{i}::{logger_message}')

        return i
    
    @property
    def homing(self):
        fast_vel = self.command('g r0xc3')
        self._fast_velocity = print(f'fast velocity::{int(fast_vel.strip())}')
        slow_vel = self.command('g r0xc4')
        self._slow_velocity = print(f'slow velocity::{int(slow_vel.strip())}')
        hom_acc = self.command('g r0xc5')
        self._home_accel = print(f'homing acceleration::{int(hom_acc.strip())}')
        hom_off = self.command('g r0xc6')
        self._home_offset = print(f'home offset::{int(hom_off.strip())}')
        


        
    
    @property
    def position(self):
        # get current position of stage
        response =  self.command('g r0x32')
        # formatting
        self._position = int(response.strip())

        return self._position

    @property
    def velocity(self):
        print('getting velocity')
        response = self.command('g r0x18')
        #self._velocity = int(vel.split(' ')[1].strip()
        self._velocity = int(response.strip())
        
        return self._velocity
          
    @velocity.setter
    def velocity(self, value):
        print('setting velocity')
        self._velocity = self.command('s r0xcb ' + str(value))
        
    
    @property
    def is_moving(self):
        #if self._in_position= pos, then self._in_position = True
        status = self.command('g r0xa0')
        bit = self.parse_registers(status, 'Status Register')
        if 27 in bit:
            self._is_moving = True
        else:
            self._is_moving = False
                                      
        return self._is_moving
                                      
        
    def move(self, pos):
        
        if pos != self.position: #if position of stage is not equal to the assigned position 
            status = self.command('g r0xa1')
            bits = self.parse_registers(status, 'Status Register')
            if len(bits) > 1:
                self.command('s r0xa1 0xFFFFFFFF') # clear status register
                #print('test')
            
            #set position of stage
            self.command('s r0xca ' + str(pos)) 
            #enable move
            self.command('t 1')

            time.sleep(1)
            while self.is_moving:
                time.sleep(1)
                #print(f'velocity:{self.velocity}')
            
        return self.position
   
        
    @property
    def gains(self):
        self._gain_vel_feedforward = self.command('g r0x33').strip()
        self._gain_acc_feedforward = self.command('g r0x34').strip()
        self._gain_proportional = self.command('g r0x30').strip()
  
        self._gains = self._gain_vel_feedforward + ',' + self._gain_acc_feedforward + ',' + self._gain_proportional 
    
        return self._gains
    
        #get values of individual components 
        #switch from imaging to moving mode
        
        
    @gains.setter
    def gains(self, value):
        self._gains = value
        

        
    def set_mode(self, mode):        
            self._gains = str(self.configurations[mode]['g'])
            print(self._gains)
            return self._gains
       

