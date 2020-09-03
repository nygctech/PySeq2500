#!/usr/bin/python
from os.path import join

# HiSeq Simulater
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


class Zstage():
    def __init__(self):
        self.spum = 0.656
        self.max_z = 25000
        self.min_z = 0
        self.position = [21500, 21500, 21500]
        #self.xstep = [-10060, -10060, 44990]                                     # x step position of motors
        #self.ystep = [-2580000, 5695000, 4070000]                                # y step position of motors
        self.xstep = [60720,   -8930, -8930]
        self.ystep = [2950000, 7950000, -4050000]

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
        self.min_v = 0                                                          #mm/s
        self.v = None                                                           #mm/s
        self.position = None
        self.focus_start =  2000                                                # focus start step
        self.focus_stop = 62000                                                 # focus stop step
        self.focus_rough = int((self.focus_stop - self.focus_start)/2 +
                                self.focus_start)

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

           Parameters:
           - v (float): The velocity for the objective to move at in mm/s.
        """

        if v > self.min_v and v <= self.max_v:
            self.v = v
            # convert mm/s to steps/s
            v = int(v * 1288471)                                                #steps/mm
        else:
            print('Objective velocity out of range')

    def set_focus_trigger(self, position):
        """Set trigger for an objective stack to determine focus position.

           Parameters:
           - position (int): Step position to start imaging.

           Returns:
           - int: Current step position of the objective.

        """

        position = int(position)

        return self.position

class FPGA():
    def __init__(self, ystage):
        self.y = ystage

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
        self.cycle_dict = {self.colors[0]:{}, self.colors[1]:{}}
        self.ex_dict = {
                        # EX1
                        self.colors[0]:
                        {'home' : 0,
                         0.2 : -36,
                         0.6 : -71,
                         1.4 : -107,
                         'open'  : 143,
                         1.6 : 107,
                         2.0 : 71,
                         4.0 : 36},
                        # EX
                        self.colors[1]:
                        {'home' : 0,
                         4.5 : 36,
                         3.0 : 71,
                         0.2 : -107,
                         'open' : 143,
                         2.0 : 107,
                         1.0 : -36,
                         0.9: -71}
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
        elif position not in self.ex_dict[wheel-1].keys():
            print(position + ' excitation filter does not exist for ' + color +
                  ' laser.')

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
        self.n_barrels = 8
        self.barrel_volume = 250.0 # uL
        self.steps = 48000.0
        self.max_volume = self.n_barrels*self.barrel_volume #uL
        self.min_volume = self.max_volume/self.steps #uL
        self.min_speed = int(self.min_volume*40*60) # uL per min (upm)
        self.max_speed = int(self.min_volume*8000*60) # uL per min (upm)
        self.dispense_speed = 7000 # speed to dispense (sps)
        self.name = name

    def pump(self, volume, speed = 0):
        """Pump desired volume at desired speed then send liquid to waste.

           **Parameters:**

           - volume (float): The volume to be pumped in uL.
           - speed (float): The flowrate to pump at in uL/min.

        """

        if speed == 0:
            speed = self.min_speed                                              # Default to min speed

        position = self.vol_to_pos(volume)                                      # Convert volume (uL) to position (steps)
        sps = self.uLperMin_to_sps(speed)                                       # Convert flowrate(uLperMin) to steps per second

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
            write_log('Volume is too large, only pumping ' +
                       str(self.max_volume))
            volume = self.max_volume
        elif volume < self.min_volume:
            write_log('Volume is too small, pumping ' +
                       str(self.min_volume))
            volume = self.min_volume

        position = round(volume / self.max_volume * self.steps)
        return int(position)

    def uLperMin_to_sps(self, speed):
        """Convert flowrate from uL per min. (float) to steps per second (int).

        """

        sps = round(speed / self.min_volume / 60)
        return int(sps)

class Valve():
    def __init__(self, name = 'valve', n_ports = 10, port_dict = dict()):
        self.n_ports = n_ports
        self.port_dict = port_dict
        self.variable_ports = []
        self.log_flag = False
        self.name = name

    def initialize(self):
        #If port dictionary empty map 1:1
        if not self.port_dict:
            for i in range(1,self.n_ports+1):
                self.port_dict[i] = i

    def move(self, port_name):
        """Move valve to the specified port_name (str)."""

        position = self.port_dict[port_name]

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
        self.frame_interval = 0.005444266666666667


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
        im1 = np.random.randint(2, size=(2048,2048))
        im2 = np.random.randint(2, size=(2048,2048))

        left_name = 'c' + str(self.left_emission)+'_'+image_name+'.tiff'
        right_name = 'c' + str(self.right_emission)+'_'+image_name+'.tiff'

        imageio.imwrite(join(image_path,left_name), im1)
        imageio.imwrite(join(image_path,right_name), im2)

        return n_bytes

    def getFrameCount(self):
        return int(self.frames)

    def getFrameInterval(self):
        return self.frame_interval

from os import getcwd
from math import ceil
import time
import warnings
class HiSeq():
    def __init__(self, Logger = None):
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
        self.bundle_height = 128
        self.nyquist_obj = 235                                                  # 0.9 um (235 obj steps) is nyquist sampling distance in z plane
        self.logger = Logger
        self.channels = None


    def initializeCams(self, Logger=None):
        """Initialize all cameras."""

        self.cam1 = Camera(0)
        self.cam2 = Camera(1)

        #Set emission labels, wavelengths in  nm
        self.cam1.left_emission = 687
        self.cam1.right_emission = 558
        self.cam2.left_emission = 610
        self.cam2.right_emission = 740

        # Initialize camera 1
        self.message('Initializing camera 1...')
        self.cam1.setTDI()
        self.cam1.captureSetup()
        self.cam1.get_status()

        # Initialize Camera 2
        self.message('Initializing camera 2...')
        self.cam2.setTDI()
        self.cam2.captureSetup()
        self.cam2.get_status()

        self.channels =[str(self.cam1.left_emission),
                        str(self.cam1.right_emission),
                        str(self.cam2.left_emission),
                        str(self.cam2.right_emission)]


    def initializeInstruments(self):
        """Initialize x,y,z, & obj stages, pumps, valves, optics, and FPGA."""

        #Initialize X Stage before Y Stage!
        self.message('Initializing X & Y stages')
        #self.x.initialize()
        #TODO, make sure x stage is in correct place.
        self.x.move(30000)
        self.y.move(0)
        self.message('Initializing lasers')
        self.lasers['green'].initialize()
        self.lasers['red'].initialize()
        self.message('Initializing pumps and valves')
        self.v10['A'].initialize()
        self.v10['B'].initialize()
        self.v24['A'].initialize()
        self.v24['B'].initialize()
        self.message('Initializing FPGA')


        # Initialize Z, objective stage, and optics after FPGA
        self.message('Initializing optics and Z stages')
        self.z.move([0,0,0])
        self.obj.move(30000)
        self.obj.set_velocity(5)
        self.optics.initialize()
        self.f.write_position(0)

        self.message('HiSeq initialized!')


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
            self.message('Attempting to sync TDI and stage')
            f.write_position(y.position)
        else:
            self.message('TDI and stage are synced')

        #TO DO, double check gains and velocity are set
        #Set gains and velocity of image scanning for ystage
        response = y.command('GAINS(5,10,5,2,0)')
        response = y.command('V0.15400')


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
            print('Cam1 frames: ', cam1.getFrameCount())
            print('Cam1 image not taken')
            image_complete = False
        else:
            cam1.saveImage(image_name, self.image_path)
            image_complete = True
        # Check if all frames were taken from camera 2 then save images
        if cam2.getFrameCount() != n_frames:
            print('Cam2 frames: ', cam2.getFrameCount())
            print('Cam2 image not taken')
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
        y.command('GAINS(5,10,7,1.5,0)')
        y.command('V1')

        meta_f.close()

        return image_complete == 2

    def message(self, text):
        """Print output text to logger or console"""

        if self.logger is None:
            print(str(text))
        else:
            self.logger.info(str(text))

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


    def scan(self, n_tiles, n_Zplanes, n_frames, image_name=None):
        """Image a volume.

           Images a zstack at incremental x positions.
           The length of the image (y dimension) remains constant.

           **Parameters:**
           - n_tiles (int): Number of x positions to image.
           - n_Zplanes (int): Number of Z planes to image.
           - n_frames (int): Number of frames to image.
           - image_name (str): Common name for images, the default is a time
             stamp.

           **Returns:**
           - int: Time it took to do scan in seconds.

        """

        if image_name is None:
            image_name = time.strftime('%Y%m%d_%H%M%S')

        start = time.time()

        for tile in range(n_tiles):
            im_name = image_name + '_x' + str(self.x.position)
            stack_time = self.zstack(n_Zplanes, n_frames, im_name)           # Take a zstack
            self.x.move(self.x.position + 315)                                  # Move to next x position

        stop = time.time()

        return stop - start


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
        n_tiles = ceil((LLx - URx)/self.tile_width)
        pos['n_tiles'] = n_tiles

        # X center of scan
        x_center = self.fc_origin[AorB][0]
        x_center -= LLx*1000*self.x.spum
        x_center += (LLx-URx)*1000/2*self.x.spum
        x_center = int(x_center)

        # initial X of scan
        x_initial = int(x_center - n_tiles*self.tile_width*1000*self.x.spum/2)
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
        pos['x_final'] = int(x_initial + 315*self.tile_width)

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
