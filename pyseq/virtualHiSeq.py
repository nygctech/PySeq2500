#!/usr/bin/python
from os.path import join
import importlib.resources as pkg_resources
from math import floor, ceil
from . import image_analysis as IA


# HiSeq Simulater
class Temperature():
    def __init__(self):
        self.T_fc = [None, None]
        self.T_chiller = [None, None, None]
        self.min_fc_T = 20.0
        self.max_fc_T = 60.0
        self.min_chiller_T = 0.1
        self.max_chiller_T = 20.0
        self.delay = 5*60

    def initialize(self):
        self.T_fc = [20, 20]
        self.T_chiller = [20, 20, 20]

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
                    T = self.T_fc[fc]
                except:
                    T = None

        return T

    def get_chiller_T(self):
        """Return temperature of chiller in °C.

           NOTE: There are 3 TEC blocks cooling the chiller. I'm not sure if all
           3 cool chiller. I think the 3rd one cools something else because the
           control parameters are different. - Kunal

        """

        T = [None,None,None]
        for i, t in enumerate(T):
              T[i] = float(self.T_chiller[i])

        return T

    def set_fc_T(self, fc, T):
        """Set temperature of flowcell in °C.

           **Parameters:**
            - fc (str or int): Flowcell position either A or 0, or B or 1
            - T (float): Temperature in °C.

           **Returns:**
            - int: Flowcell index, 0 for A and 1 or B.

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

            direction = T - self.get_fc_T(fc)

            self.T_fc[fc] = T
        else:
            direction = None

        return direction

    def wait_fc_T(self, fc, T):
        """Set and wait for flowcell to reach temperature in °C.

           **Parameters:**
            - fc (str or int): Flowcell position either A or 0, or B or 1
            - T (float): Temperature in °C.

        """

        direction = self.set_fc_T(fc,T)
        if direction is None:
            self.message('Unable to wait for flowcell', fc, 'to reach', T, '°C')
        if direction < 0:
            while self.get_fc_T(fc) >= T:
                time.sleep(self.delay)
        elif direction > 0:
            while self.get_fc_T(fc) <= T:
                time.sleep(self.delay)

        return self.get_fc_T(fc)

    def fc_off(self, fc):
        """Turn off temperature control for flowcell fc."""

        fc = self.get_fc_index(fc)

        return False

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



class Xstage():
    def __init__(self):
        self.spum = 100/244
        self.min_x = 1000
        self.max_x = 50000
        self.home = 30000
        self.spum = 0.4096     #steps per um
        self.position = 0

    def move(self, position):
        """Move xstage to absolute step position.

           Parameters:
           position (int): Absolute step position must be between 1000 - 50000.

           Returns:
           int: Absolute step position after move.
        """
        position = int(position)
        if position <= self.max_x and position >= self.min_x:
            self.position = position                                            # Move Absolute
            return position                                                     # Check position
        else:
            print('XSTAGE can only move between ' + str(self.min_x) +
                  ' and ' + str(self.max_x))

        return self.position



class Ystage():
    def __init__(self):
        self.min_y = -7000000
        self.max_y = 7500000
        self.spum = 100     # steps per um
        self.mode = None
        self.velocity = None
        self.gains = None
        self.position = 0

    def move(self, position):
        """Move ystage to absolute step position.

           Parameters:
           position (int): Absolute step position must be between -7000000 and
                7500000.

           Returns:
           bool: True when stage is in position.
        """
        try:
            position = int(position)
            if position <= self.max_y and position >= self.min_y:
                self.position = position
                return True                                                         # Return True that stage is in position
            else:
                print("YSTAGE can only between " + str(self.min_y) + ' and ' +
                    str(self.max_y))
        except:
            print('Position is not an integer')

        return self.position

    def command(self, text):
        return text

    def set_mode(self, mode):
        "Change between imaging and moving configurations."

        if self.mode != mode:
            if mode in self.configurations.keys():
                gains = str(self.configurations[mode]['g'])
                velocity = str(self.configurations[mode]['v'])
                self.mode = mode
                self.velocity = v
                self.gains = gains
            else:
                gains = None
                message = 'Ystage::ERROR::Invalid configuration::'+str(mode)
                if self.logger is not None:
                    self.logger.info(message)
                else:
                    print(message)

        return True


class Zstage():
    def __init__(self):
        self.spum = 0.656
        self.max_z = 25000
        self.min_z = 0
        self.position = [0, 0, 0]
        #self.xstep = [-10060, -10060, 44990]                                     # x step position of motors
        #self.ystep = [-2580000, 5695000, 4070000]                                # y step position of motors
        self.xstep = [60720,   -8930, -8930]
        self.ystep = [2950000, 7950000, -4050000]
        self.active = True

    def get_motor_points(self):
        points = [[self.xstep[0], self.ystep[0], self.position[0]],
                  [self.xstep[1], self.ystep[1], self.position[1]],
                  [self.xstep[2], self.ystep[2], self.position[2]]]

        return points

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
            position[i] = int(position[i])
            if position[i] <= self.max_z and position[i] >= self.min_z:
                if self.active:
                    self.position[i] = position[i]
            else:
                print("ZSTAGE can only move between " + str(self.min_z) +
                    ' and ' + str(self.max_z))

        return self.position

class OBJstage():
    def __init__(self):
        self.min_z = 0
        self.max_z = 65535
        self.spum = 262                                                         #steps per um
        self.max_v = 5                                                          #mm/s
        self.min_v = 0.1                                                        #mm/s
        self.v = None                                                           #mm/s
        self.position = None
        self.focus_spacing = 0.5                                                # distance in microns between frames in obj stack
        self.focus_velocity = 0.1                                               #mm/s
        self.focus_frames = 450                                                 # number of total camera frames for obj stack
        self.focus_range = 90                                                   #%
        self.focus_start =  2000                                                # focus start step
        self.focus_stop = 62000                                                 # focus stop step
        self.focus_rough = int((self.max_z - self.min_z)/2 + self.min_z)

    def move(self, position):
        """Move the objective to an absolute step position.

           The objective can move between steps 0 and 65535, where step 0 is
           the closest to the stage. If the position is out of range, the
           objective will not move and a warning message is printed.

           Parameters:
           - position (int): The step position to move the objective to.

        """

        position = int(position)
        if position >= self.min_z and position <= self.max_z:
            self.position = position

            return position

        else:
            print('Objective position out of range')

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
        else:
            print('Objective velocity out of range')

    def set_focus_trigger(self, position):
        """Set trigger for an objective stack to determine focus position.

           **Parameters:**
           - position (int): Step position to start imaging.

           **Returns:**
           - int: Current step position of the objective.

        """

        position = int(position)

        return self.position

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

class FPGA():
    def __init__(self, ystage):
        self.y = ystage
        self.led_dict = {'off':'0', 'yellow':'1', 'green':'3', 'pulse green':'4',
                         'blue':'5', 'pulse blue':'6', 'sweep blue': '7'}

    def read_position(self):
        """Read the y position of the encoder for TDI imaging.

           ****Returns:****
           - int: The y position of the encoder.

        """
        tdi_pos = self.y.position

        return tdi_pos

    def write_position(self, position):
        """Write the position of the y stage to the encoder.

           Allows for a 5 step (50 nm) error.

           **Parameters:**
           - position (int) = The position of the y stage.

        """
        position = int(position)
        while abs(self.read_position() - position) > 5:
            self.y.move(position)

    def TDIYPOS(self, y_pos):
        """Set the y position for TDI imaging.

           **Parameters:**
           - y_pos (int): The initial y position of the image.

        """
        y_pos = int(y_pos)

    def TDIYARM3(self, n_triggers, y_pos):
        """Arm the y stage triggers for TDI imaging.

           **Parameters:**
           - n_triggers: Number of triggers to send to the cameras.
           - y_pos (int): The initial y position of the image.

        """
        n_triggers = int(n_triggers)
        y_pos = int(y_pos)

    def command(self, text):
        return text

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
                s = '0'
            elif AorB.upper() == 'B':
                s = '1'
        elif type(AorB) is int:
            if AorB == 0 or AorB == 1:
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


class Laser():
    def __init__(self, color):
        self.on = False
        self.power = 0
        self.max_power = 500
        self.min_power = 0
        self.color = color

    def initialize(self):
        """Turn the laser on and set the power to the default 10 mW level."""
        self.turn_on(True)
        self.set_power(10)

    def turn_on(self, state):
        """Turn the laser on or off.

           Parameters:
           state (bool): True to turn on, False to turn off.

           Returns:
           bool: True if laser is on, False if laser is off.
        """
        if state:
            self.on = True
        else:
            self.on = False

        return self.on

    def get_power(self):
        """Return the power level of the laser in mW (int)."""
        return self.power

    def set_power(self, power):
        """Set the power level of the laser.

        Parameters:
        power (int): Power level to set the laser to.

        Returns:
        bool: True if the laser is on, False if the laser is off.
        """
        power = int(power)
        if power >= self.min_power and power <= self.max_power:
                self.power = power
        else:
            print('Power must be between ' +
                  str(self.min_power) +
                  ' and ' +
                  str(self.max_power))

        return self.on

class Optics():
    def __init__(self, colors = ['green', 'red']):
        self.ex = [None, None]
        self.em_in = None
        self.colors = [colors[0], colors[1]]
        self.focus_filters = [None, None]
        self.cycle_dict = {'em':{}, self.colors[0]:{}, self.colors[1]:{}}
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


    def move_ex(self, color, position):
        """Move the excitation wheel to the specified position.

           The excitation filters are optical density filters that block a
           portion of the light to quickly change between laser intensities.
           The percent of light passed through is 10**-OD*100 where OD is
           the optical density of the filter. All of the light is blocked with
           the home "filter". The names and OD of available filters are listed
           in the following table.

           ===========  ===========  ========================================
           laser color  laser index  filters
           ===========  ===========  ========================================
           green        1            open, 0.2, 0.6, 1.4, 1.6, 2.0, 4.0, home
           red          2            open, 0.2, 0.9, 1.0, 2.0, 3.0, 4.5, home
           ===========  ===========  ========================================

           Parameters:
           color (str): The color of laser line.
           position (str): The name of the filter to change to.

        """


        if color not in self.colors:
            warnings.warn('Laser color is invalid.')
        elif position in self.ex_dict[color].keys():
            index = self.colors.index(color)
            self.ex[index] = position
            if position != 'home':
                position = str(self.ex_dict[color][position])                   # get step position
        elif position not in self.ex_dict[color].keys():
            print(str(position) + ' excitation filter does not exist for ' +
                  color + ' laser.')

    def move_em_in(self, INorOUT):
        """Move the emission filter in to or out of the light path.

           Parameters:
           INorOUT (bool): True for the emission in the light path or
                False for the emission filter out of the light path.
        """

        # Move emission filter into path
        if INorOUT:
            self.em_in = True
        # Move emission filter out of path
        else:
            self.em_in = False

class Pump():
    def __init__(self, name = 'pump'):
        self.n_barrels = 1
        self.barrel_volume = 250.0 # uL
        self.steps = 48000.0
        self.max_volume = self.n_barrels*self.barrel_volume #uL
        self.min_volume = self.max_volume/self.steps #uL
        self.min_flow = int(self.min_volume*40*60) # uL per min (upm)
        self.max_flow = int(self.min_volume*8000*60) # uL per min (upm)
        self.dispense_speed = 7000 # speed to dispense (sps)
        self.name = name

    def update_limits(self, n_barrels):
        self.n_barrels = n_barrels
        self.max_volume = self.n_barrels*self.barrel_volume #uL
        self.min_volume = self.max_volume/self.steps #uL
        self.min_flow = int(self.min_volume*40*60) # uL per min (upm)
        self.max_flow = int(self.min_volume*8000*60) # uL per min (upm)

    def command(self, text):
        return text

    def pump(self, volume, flow = 0):
        """Pump desired volume at desired flowrate then send liquid to waste.

           **Parameters:**

           - volume (float): The volume to be pumped in uL.
           - flow (float): The flowrate to pump at in uL/min.

        """

        if flow == 0:
            flow = self.min_flow                                              # Default to min flow

        position = self.vol_to_pos(volume)                                      # Convert volume (uL) to position (steps)
        sps = self.uLperMin_to_sps(flow)                                       # Convert flowrate(uLperMin) to steps per second

        self.check_pump()                                                       # Make sure pump is ready

        #Aspirate
        self.check_pump()

        #Dispense
        position = 0
        self.check_pump()


    def check_pump(self):
        """Wait until pump is ready and then return True.

           **Returns:**

           - bool: True when the pump is ready. False, if the pump has an error.

        """

        busy = '@'
        ready = '`'
        status_code = '`'

        while status_code != ready :

            while not status_code:
                status_code = '`'                                               # Ping pump for status

                if status_code.find(busy) > -1:
                    status_code = ''
                    time.sleep(2)
                elif status_code.find(ready) > -1:
                    status_code = ready
                    return True
                else:
                    return False

    def vol_to_pos(self, volume):
        """Convert volume from uL (float) to pump position (int, 0-48000).

           If the volume is too big or too small, returns the max or min volume.

        """

        if volume > self.max_volume:
            print('Volume is too large, only pumping ' +
                       str(self.max_volume))
            volume = self.max_volume
        elif volume < self.min_volume:
            print('Volume is too small, pumping ' +
                       str(self.min_volume))
            volume = self.min_volume

        position = round(volume / self.max_volume * self.steps)
        return int(position)

    def uLperMin_to_sps(self, flow):
        """Convert flowrate from uL per min. (float) to steps per second (int).

        """
        if self.min_flow > flow:
            flow = self.min_flow
            print('Flowrate too slow, increased to ', str(flow), 'uL/min')
        elif flow > self.max_flow:
            flow = self.max_flow
            print('Flowrate too fast, decreased to ', str(flow), 'uL/min')

        sps = round(flow / self.min_volume / 60)

        return int(sps)

class Valve():
    def __init__(self, name = 'valve', n_ports = 10, port_dict = dict()):
        self.n_ports = n_ports
        self.port_dict = port_dict
        self.variable_ports = []
        self.name = name

    def initialize(self):
        #If port dictionary empty map 1:1
        if not self.port_dict:
            for i in range(1,self.n_ports+1):
                self.port_dict[i] = i

    def move(self, port_name):
        """Move valve to the specified port_name (str)."""

        if isinstance(port_name, int):
            if port_name in range(1,self.n_ports+1):
                position = port_name
        else:
            position = self.port_dict[port_name]

        return position

class CHEM():
    def __init__(self, logger = None):
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
        self.delay = 2                                      # delay time in s
        self.logger = logger

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

        if self.T_fc[fc] is None:
            self.T_fc[fc] = 20

        T = self.T_fc[fc]

        return T

    def set_fc_T(self, fc, T):
        """Set temperature of flowcell in °C.

           **Parameters:**
            - fc (str or int): Flowcell position either A or 0, or B or 1
            - T (float): Temperature in °C.

           **Returns:**
            - int: Flowcell index, 0 for A and 1 or B.

        """

        fc = self.get_fc_index(fc)

        print('Flowcell index is', fc)

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
                self.get_fc_T(fc)

            direction = T - self.T_fc[fc]
            print('direction', direction)
            self.T_fc[fc] = T

            return direction

    def fc_off(self, fc):
        """Turn off temperature control for flowcell fc."""

        fc = self.get_fc_index(fc)

        return False

    def write_log(self, text):
        """Write messages to the log."""

        msg = 'ARM9CHEM::' + text
        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)


import imageio
import numpy as np
class Camera():
    def __init__(self, camera_id):
        self.frames = 0
        self.bundle_height = 128
        self.sensor_mode = 'TDI'
        self.camera_id = camera_id
        self.left_emission = None
        self.right_emission = None
        self.frame_interval = 0.040202


    def stopAcquisition(self):
        return 1

    def startAcquisition(self):
        return 1

    def freeFrames(self):
        return 1

    def captureSetup(self):
        return True

    def setPropertyValue(self, property, value):
        value = int(value)
        if property is 'sensor_mode_line_bundle_height':
            self.bundle_height = int(value)
        return int(value)

    def allocFrame(self, frames):
        self.frames = frames
        return 1

    def get_status(self):
        return 3

    def setTDI(self):
        self.sensor_mode = 'TDI'
        return True

    def saveImage(self, image_name, image_path):
        # save text info as image data
        n_bytes = self.frames*self.bundle_height*2048*2*12
        im1 = np.random.randint(2, size=(256,256), dtype='uint8')
        im2 = np.random.randint(2, size=(256,256), dtype='uint8')

        left_name = 'c' + str(self.left_emission)+'_'+image_name+'.tiff'
        right_name = 'c' + str(self.right_emission)+'_'+image_name+'.tiff'

        imageio.imwrite(join(image_path,left_name), im1)
        imageio.imwrite(join(image_path,right_name), im2)

        return n_bytes

    def getFrameCount(self):
        return int(self.frames)

    def getFrameInterval(self):
        return self.frame_interval

from . import focus
from os import getcwd
from os import listdir
from os.path import getsize
from os.path import dirname

from math import ceil
import time
import warnings
import pandas as pd

class HiSeq():
    def __init__(self, name = 'HiSeq2500', Logger = None):
        self.x = Xstage()
        self.y = Ystage()
        self.z = Zstage()
        self.obj = OBJstage()
        self.f = FPGA(self.y)
        self.lasers = {'green': Laser(color = 'green'),
                       'red'  : Laser(color = 'red')}
        self.p = {'A': Pump('pumpA'),
                  'B': Pump('pumpB')}
        self.v10 = {'A': Valve('valveA10', 10),
                    'B': Valve('valveB10', 10)}
        self.v24 = {'A': Valve('valveA24', 24),
                    'B': Valve('valveB24', 24)}
        self.T = Temperature()
        self.optics = Optics()
        self.cam1 = None
        self.cam2 = None

        self.im_obj_pos = 30000
        self.image_path = getcwd()                                              # path to save images in
        self.log_path = getcwd()                                                # path to save logs in
        self.fc_origin = {'A':[17571,-180000],
                          'B':[43310,-180000]}
        self.tile_width = 0.769                                                 #mm
        self.resolution = 0.375                                                 #um/px
        self.overlap = 0
        self.bundle_height = 128
        self.nyquist_obj = 235                                                  # 0.9 um (235 obj steps) is nyquist sampling distance in z plane
        self.logger = Logger
        self.channels = None
        self.virtual = True
        self.focus_data = join(dirname(__file__), 'focus_data')
        self.focus_path = join(dirname(__file__), 'focus_data','Objective Stack.zarr')
        self.AF = 'partial'
        self.focus_tol = 0
        self.scan_flag = False
        self.speed_up = 10
        self.current_view = None
        self.name = name

    def initializeCams(self, Logger=None):
        """Initialize all cameras."""

        self.message('HiSeq::Initializing cameras')

        self.cam1 = Camera(0)
        self.cam2 = Camera(1)

        #Set emission labels, wavelengths in  nm
        self.cam1.left_emission = 687
        self.cam1.right_emission = 558
        self.cam2.left_emission = 610
        self.cam2.right_emission = 740

        # Initialize camera 1
        self.cam1.setTDI()
        self.cam1.captureSetup()
        self.cam1.get_status()

        # Initialize Camera 2
        self.cam2.setTDI()
        self.cam2.captureSetup()
        self.cam2.get_status()

        self.channels =[str(self.cam1.left_emission),
                        str(self.cam1.right_emission),
                        str(self.cam2.left_emission),
                        str(self.cam2.right_emission)]


    def initializeInstruments(self):
        """Initialize x,y,z, & obj stages, pumps, valves, optics, and FPGA."""
        msg = 'HiSeq::'

        #Initialize X Stage before Y Stage!
        self.message(msg+'Initializing X & Y stages')
        #self.x.initialize()
        #TODO, make sure x stage is in correct place.
        self.x.move(30000)
        self.y.move(0)
        self.message(msg+'Initializing lasers')
        self.lasers['green'].initialize()
        self.lasers['red'].initialize()
        self.message(msg+'Initializing pumps and valves')
        self.v10['A'].initialize()
        self.v10['B'].initialize()
        self.v24['A'].initialize()
        self.v24['B'].initialize()
        self.message(msg+'Initializing FPGA')


        # Initialize Z, objective stage, and optics after FPGA
        self.message(msg+'Initializing optics and Z stages')
        self.z.move([0,0,0])
        self.obj.move(30000)
        self.obj.set_velocity(5)
        self.optics.initialize()
        self.f.write_position(0)

        # Initialize ARM9 Temperature control
        self.T.initialize()

        return True

    def move_inlet(self, n_ports):
        """Move 10 port valves to 2 inlet row or 8 inlet row ports."""

        if n_ports == 2:
            self.v10['A'].move(2)
            self.v10['B'].move(4)
            return True
        elif n_ports == 8:
            self.v10['A'].move(3)
            self.v10['B'].move(5)
            return True
        else:
            return False

    def obj_stack(self, n_frames = None, velocity = None):

        if velocity is None:
            velocity= self.obj.focus_velocity
        if n_frames is None:
            n_frames = self.obj.focus_frames
        self.obj.v = velocity
        self.cam1.allocFrame(n_frames)
        self.cam2.allocFrame(n_frames)

        focus_stack = IA.HiSeqImages(image_path = self.focus_path, logger=self.logger)
        focus_stack.im = focus_stack.im[0]

        #focus_data = np.loadtxt(join(self.focus_data,'obj_stack_data.txt'))

        return focus_stack



    def take_picture(self, n_frames, image_name = None):
        """Take a picture using all the cameras and save as a tiff.

           The section to be imaged should already be in position and
           optical settings should already be set.

           The final size of the image is 2048 px wide and n_frames *
           self.bundle_height px long. The images and metadata are stored in the
           self.image_path directory.

           **Parameters:**
           - n_frames (int): Number of frames in the images.
           - image_name (str, optional): Common name of the images, the default
             is a time stamp.

           **Returns:**
           - bool: True if all of the frames of the image were taken, False if
             there were incomplete frames.

        """

        msg = 'HiSeq::TakePicture::'

        y = self.y
        x = self.x
        obj = self.obj
        f = self.f
        op = self.optics
        cam1 = self.cam1
        cam2 = self.cam2

        if image_name is None:
            image_name = time.strftime('%Y%m%d_%H%M%S')


        #Make sure TDI is synced with Ystage
        y_pos = y.position
        if abs(y_pos - f.read_position()) > 10:
            self.message(msg, 'Attempting to sync TDI and stage')
            f.write_position(y.position)
        else:
            self.message(False, msg+'TDI synced with stage')

        #TO DO, double check gains and velocity are set
        #Set gains and velocity of image scanning for ystage
        y.set_mode('imaging')
        # response = y.command('GAINS(5,10,5,2,0)')
        # response = y.command('V0.15400')


        # Make sure cameras are ready (status = 3)
        if cam1.sensor_mode != 'TDI':
            cam1.setTDI()
        if cam2.sensor_mode != 'TDI':
            cam2.setTDI()
        while cam1.get_status() != 3:
            cam1.stopAcquisition()
            cam1.freeFrames()
            cam1.captureSetup()
        while cam2.get_status() != 3:
            cam2.stopAcquisition()
            cam2.freeFrames()
            cam2.captureSetup()
        # Set bundle height
        cam1.setPropertyValue("sensor_mode_line_bundle_height",
                               self.bundle_height)
        cam2.setPropertyValue("sensor_mode_line_bundle_height",
                               self.bundle_height)
        cam1.captureSetup()
        cam2.captureSetup()
        # Allocate memory for image data
        cam1.allocFrame(n_frames)
        cam2.allocFrame(n_frames)


        #
        #Arm stage triggers
        #
        #TODO check trigger y values are reasonable
        n_triggers = n_frames * self.bundle_height
        end_y_pos = int(y_pos - n_triggers*self.resolution*y.spum - 300000)
        f.TDIYPOS(y_pos)
        f.TDIYARM3(n_triggers, y_pos)
        #print('Trigger armed, Imaging starting')

        meta_f = self.write_metadata(n_frames, image_name)

        ################################
        ### Start Imaging ##############
        ################################

        # Start cameras
        cam1.startAcquisition()
        cam2.startAcquisition()
        # Open laser shutter
        f.command('SWLSRSHUT 1')
        # move ystage (blocking)
        y.move(end_y_pos)



        ################################
        ### Stop Imaging ###############
        ################################
        # Close laser shutter
        f.command('SWLSRSHUT 0')

        # Stop Cameras
        cam1.stopAcquisition()
        cam2.stopAcquisition()

        # Check if all frames were taken from camera 1 then save images
        if cam1.getFrameCount() != n_frames:
            self.message('Cam1 frames: ', cam1.getFrameCount())
            self.message('Cam1 image not taken')
            image_complete = False
        else:
            cam1.saveImage(image_name, self.image_path)
            image_complete = True
        # Check if all frames were taken from camera 2 then save images
        if cam2.getFrameCount() != n_frames:
            self.message('Cam2 frames: ', cam2.getFrameCount())
            self.message('Cam2 image not taken')
            image_complete += False
        else:
            cam2.saveImage(image_name, self.image_path)
            image_complete += True
        # Print out info pulses = triggers, not sure with CLINES is
        if image_complete:
            response  = self.cam1.getFrameCount()
            meta_f.write('frame count 1 ' + str(response) +'\n')
            response  = self.cam2.getFrameCount()
            meta_f.write('frame count 2 ' + str(response) +'\n')
            response = f.command('TDICLINES')
            meta_f.write('clines ' + str(response) + '\n')
            response = f.command('TDIPULSES')
            meta_f.write('pulses ' + str(response) +'\n')

        # Free up frames/memory
        cam1.freeFrames()
        cam2.freeFrames()

        # Reset gains & velocity for ystage
        y.set_mode('moving')
        # y.command('GAINS(5,10,7,1.5,0)')
        # y.command('V1')

        meta_f.close()

        return image_complete == 2

    def message(self, *args):
        """Print output text to logger or console"""

        i = 0
        if isinstance(args[0], bool):
            screen = args[0]
            i = 1
        else:
            screen = True

        msg = ''
        for a in args[i:]:
            msg += str(a) + ' '

        if self.logger is None:
            print(msg)
        else:
            if screen:
                self.logger.log(21,msg)
            else:
                self.logger.info(msg)

    def write_metadata(self, n_frames, image_name):
        """Write image metadata to file.

           Parameters:
           n_frames (int): Number of frames in the images.
           bundle (int): Line bundle height of the images.
           image_name (int): Common name of the images.

           Returns:
           file: Metadata file to write info about images to.
        """

        date = time.strftime('%Y%m%d_%H%M%S')
        meta_path = join(self.image_path, 'meta_'+image_name+'.txt')
        meta_f = open(meta_path, 'w+')
        meta_f.write('time ' + date + '\n' +
                     'y ' + str(self.y.position) + '\n' +
                     'x ' + str(self.x.position) + '\n' +
                     'z ' + str(self.z.position) + '\n' +
                     'obj ' + str(self.obj.position) + '\n' +
                     'frames ' + str(n_frames) + '\n' +
                     'bundle ' + str(self.bundle_height) + '\n' +
                     'TDIY ' + str(self.f.read_position()) +  '\n' +
                     'laser1 ' + str(self.lasers['green'].get_power()) + '\n' +
                     'laser2 ' + str(self.lasers['red'].get_power()) + '\n' +
                     'ex filters ' + str(self.optics.ex) + '\n' +
                     'em filter in ' + str(self.optics.em_in) + '\n' +
                     'interval 1 ' + str(self.cam1.getFrameInterval()) + '\n' +
                     'interval 2 ' + str(self.cam2.getFrameInterval()) + '\n'
                     )

        return meta_f


    def zstack(self, n_Zplanes, n_frames, image_name=None):
        """Take a zstack/tile of images.

           Takes images from all channels at incremental z planes at the same
           x&y position.

           **Parameters:**
           - n_Zplanes (int): Number of Z planes to image.
           - n_frames (int): Number of frames to image.
           - image_name (str): Common name for images, the default is a time
             stamp.

           **Returns:**
           - int: Time it took to do zstack in seconds.

        """

        if image_name is None:
            image_name = time.strftime('%Y%m%d_%H%M%S')

        y_pos = self.y.position
        obj_pos = self.obj.position

        start = time.time()

        for n in range(n_Zplanes):
            im_name = image_name + '_o' + str(self.obj.position)
            image_complete = False

            while not image_complete:
                image_complete = self.take_picture(n_frames, im_name)
                if image_complete:
                    self.obj.move(self.obj.position + self.nyquist_obj)
                    self.y.move(y_pos)
                else:
                    warnings.warn('Image not taken')
                    # Reset stage and FPGA
                    self.reset_stage()
                    self.y.move(y_pos)

        self.obj.move(obj_pos)
        stop = time.time()

        return stop-start


    def scan(self, n_tiles, n_Zplanes, n_frames, image_name=None, overlap=0):
        """Image a volume.

           Images a zstack at incremental x positions.
           The length of the image (y dimension) remains constant.

           **Parameters:**
           - n_tiles (int): Number of x positions to image.
           - n_Zplanes (int): Number of Z planes to image.
           - n_frames (int): Number of frames to image.
           - image_name (str): Common name for images, the default is a time
             stamp.
           - overlap (int): Number of column pixels to overlap between tiles

           **Returns:**
           - int: Time it took to do scan in seconds.

        """

        self.scan_flag = True
        dx = self.tile_width*1000-self.resolution*overlap                  # x stage delta in in microns
        dx = round(dx*self.x.spum)

        if image_name is None:
            image_name = time.strftime('%Y%m%d_%H%M%S')

        start = time.time()


        for tile in range(n_tiles):
            self.message('HiSeq::Scan::Tile '+str(tile+1)+'/'+str(n_tiles))
            im_name = image_name + '_x' + str(self.x.position)
            stack_time = self.zstack(n_Zplanes, n_frames, im_name)              # Take a zstack
            self.x.move(self.x.position + dx)                                   # Move to next x position

        stop = time.time()
        self.scan_flag = False

        return stop - start

    def autofocus(self, pos_dict):
        """Find optimal objective position for imaging, True if found."""

        opt_obj_pos = focus.autofocus(self, pos_dict)
        if opt_obj_pos:
            self.obj.move(opt_obj_pos)
            return True
        else:
            self.obj.move(self.obj.focus_rough)
            return False

    def optimize_filter(self, pos_dict, init_filter, n_filters):
        """Image a section with different filters.

           Images a section with all possible excitation filter set
           combinations. The highest OD filters (lowest light intensity) are
           imaged first. Lower OD filters are sequentially used to image the
           section. The laser is blocked with the last filter. Upon completion
           of imaging, users can inspect the images to ascertain which filter
           set is optimal.

           **Parameters:**
           - pos_dict (dict): Dictionary of stage position information
           - init_filter (int): Descending order position of highest OD filter
           - n_filters (int): Number of filters to use for imaging

        """

        msg = 'HiSeq::OptimizeFilter::'

        # position stage
        self.y.move(pos_dict['y_initial'])
        self.x.move(pos_dict['x_initial'])
        self.z.move([21500, 21500, 21500])
        self.obj.move(self.obj.focus_rough)

        #Order of filters to loop through
        colors = self.optics.colors
        f_order = [[],[]]
        for i, color in enumerate(colors):
            filters = self.optics.ex_dict[color].keys()
            f_order[i] = [f for f in filters if isinstance(f,float)]
            f_order[i] = sorted(f_order[i], reverse = True)
            f_order[i] = f_order[i][init_filter:init_filter+n_filters]
            f_order[i].append('home')

        print(f_order)
        # Set optical filters
        for i, color in enumerate(colors):
            self.optics.move_ex(color,f_order[i][0])

        # Focus on section
        self.message(msg, 'Starting Autofocus')
        if self.autofocus(pos_dict):
            self.message(msg, 'Autofocus completed')
        else:
            self.message(msg, 'Autofocus failed')

        # Loop through filters and image section
        for f in range(n_filters+1):
            self.optics.move_ex(colors[0], f_order[0][f])
            for f in range(n_filters+1):
                self.optics.move_ex(colors[1], f_order[1][f])

                image_name = colors[0][0].upper()+str(self.optics.ex[0])+'_'
                image_name += colors[1][0].upper()+str(self.optics.ex[1])

                self.y.move(pos_dict['y_initial'])
                self.x.move(pos_dict['x_initial'])
                self.message(msg, colors[0], 'filter set to', self.optics.ex[0])
                self.message(msg, colors[1], 'filter set to', self.optics.ex[1])
                self.message(msg, 'Starting imaging')
                img_time = self.scan(pos_dict['n_tiles'],1,
                                   pos_dict['n_frames'], image_name)
                img_time /= 60
                self.message(msg, 'Imaging complete in ', img_time, 'min.')

    def move_stage_out(self):
        """Move stage out for loading/unloading flowcells."""

        self.z.move([0,0,0])
        self.x.move(self.x.home)
        self.y.move(self.y.min_y)



    def position(self, AorB, box):
        """Returns stage position information.

           The center of the image is used to bring the section into focus
           and optimize laser intensities. Image scans of sections start on
           the upper right corner of the section. The section is imaged in
           strips 0.760 mm wide by length of the section long until the entire
           section has been imaged. The box region of interest surrounding the
           section is converted into stage and imaging details to scan the
           entire section.

           ========   ===========
             key      description
           ========   ===========
           x_center   The xstage center position of the section.
           y_center   The ystage center position of the section.
           x_initial  Initial xstage position to scan the section.
           y_initial  Initial ystage position to scan the section.
           x_final    Last xstage position of the section scan
           y_final    Last ystage position of the section scan
           n_tiles    Number of tiles to scan the entire section.
           n_frames   Number of frames to scan the entire section.

           Parameters:
           AorB (str): Flowcell A or B.
           box ([float, float, float, float]) = The region of interest as
                x&y position of the corners of a box surrounding the section
                to be imaged defined as [LLx, LLy, URx, URy] where LL=Lower
                Left and UR=Upper Right corner using the slide ruler.

           Returns:
           dict: Dictionary of stage positioning and
                 imaging details to scan the entire section. See table
                 above for details.
        """

        pos = {}

        LLx = box[0]
        LLy = box[1]
        URx = box[2]
        URy = box[3]

        # Number of scans

        dx = self.tile_width-self.resolution*self.overlap/1000                  # x stage delta in in mm
        n_tiles = ceil((LLx - URx)/dx)

        pos['n_tiles'] = n_tiles

        # X center of scan
        x_center = self.fc_origin[AorB][0]
        x_center -= LLx*1000*self.x.spum
        x_center += (LLx-URx)*1000/2*self.x.spum
        x_center = int(x_center)

        # initial X of scan
        x_initial = int(x_center - n_tiles*dx*1000*self.x.spum/2)
        pos['x_initial'] = x_initial

        # initial Y of scan
        y_initial = int(self.fc_origin[AorB][1] + LLy*1000*self.y.spum)
        pos['y_initial'] = y_initial

        # Y center of scan
        y_length = (LLy - URy)*1000
        y_center = y_initial - y_length/2*self.y.spum
        y_center = int(y_center)

        # Number of frames
        n_frames = y_length/self.bundle_height/self.resolution
        pos['n_frames'] = ceil(n_frames + 10)

        # Adjust x and y center so focus will image (32 frames, 128 bundle) in center of section
        x_center -= int(self.tile_width*1000*self.x.spum/2)
        pos['x_center'] = x_center
        y_center += int(32*self.bundle_height/2*self.resolution*self.y.spum)
        pos['y_center'] = y_center

        # Calculate final x & y stage positions of scan
        pos['y_final'] = int(y_initial - y_length*self.y.spum)
        pos['x_final'] = int(x_initial +(LLx - URx)*1000*self.x.spum)

        pos['obj_pos'] = None

        return pos

    def px_to_step(self, row, col, pos_dict, scale):
        '''Convert pixel coordinates in image to stage step position.

           Parameters:
           row_col ([int,int]): Row and column pixel position in image.
           pos_dict (dict): Dictionary of position data
           scale (int): Scale factor of imaged

           Returns:
           [int, int]: X-stage and Y-stage step position respectively.

        '''
        #print(row_col)
        #row = row_col[0]
        #col = row_col[1]
        scale = scale*self.resolution
        x_init = pos_dict['x_initial']
        y_init = pos_dict['y_initial']

        x_step = col*scale*self.x.spum
        x_step = int(x_init + x_step - 315/2)

        trigger_offset = -80000
        frame_offset = 64/2*self.resolution*self.y.spum
        y_step = row*scale*self.y.spum
        y_step = int(y_init + trigger_offset - y_step - frame_offset)

        return [x_step, y_step]
