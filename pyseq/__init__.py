#!/usr/bin/python
"""Illumina HiSeq 2500 System

Examples:

.. code-block:: python

    #Create HiSeq object
    import pyseq
    hs = pyseq.HiSeq()
    #Initialize cameras
    hs.initializeCams()
    #Initialize Instruments, will take a few minutes
    hs.initializeInstruments()
    #Specify directory to save images in
    hs.image_path = 'C:\\Users\\Public\\HiSeqImages\\'
    #Load stage
    hs.move_stage_out()
    #Get stage positioning and imaging details for section on flowcell A
    pos = hs.position('A', [15.5, 45, 10.5, 35])
    #Move stage to imaging position.
    hs.z.move([21500, 21500, 21500])
    # Set laser intensity to 100 mW
    hs.lasers['green'].set_power(100)
    hs.lasers['red'].set_power(100)
    #Move green excitation filter to optical density 1.4
    hs.move_ex('green', 1.4)
    #Move red excitation filter to optical density 1.0
    hs.move_ex('red', 1.0)
    #Find focus
    hs.AF = 'partial once'
    hs.autofocus(pos)
    True
    #Move to center of section
    hs.x.move(pos['x_center'])
    12000
    hs.y.move(pos['y_center'])
    True
    # Take a 32 frame picture, creates image for each channel 2048 x 4096 px
    hs.take_picture(32, image_name='FirstHiSeqImage')
    #Move stage to the initial image scan position and scan image at 1 obj plane
    hs.x.move(pos['x_initial'])
    10000
    hs.y.move(pos['y_initial'])
    True
    hs.scan(pos['n_scans'], 1, pos['n_frames'], image_name='FirstHiSeqScan')


TODO:
    - Double check gains and velocity are set in take_picture

"""


#import instruments
from . import fpga
from . import laser
from . import objstage
from . import optics
from . import pump
from . import valve
from . import xstage
from . import ystage
from . import zstage
from . import focus
from . import temperature
from .image_analysis import get_machine_config

import time
from os import getcwd
import threading
import numpy as np
from math import ceil
import configparser
from serial.tools.list_ports import comports
from pathlib import Path
import logging
import traceback

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources
from . import resources


class HiSeq():
    """Illumina HiSeq 2500 System

       **Attributes:**
        - x (xstage): Illumina HiSeq 2500 :: Xstage.
        - y (ystage): Illumina HiSeq 2500 :: Ystage.
        - z (zstage): Illumina HiSeq 2500 :: Zstage.
        - obj (objstage): Illumina HiSeq 2500 :: Objective stage.
        - p['A'] (pump): Illumina HiSeq 2500 :: Pump for flowcell A.
        - p['B'] (pump): Illumina HiSeq 2500 :: Pump for flowcell B.
        - v10['A'] (valve): Illumina HiSeq 2500 :: Valve with 10 ports
          for flowcell A.
        - v10['B'] (valve): Illumina HiSeq 2500 :: Valve with 10 ports
          for flowcell B.
        - v24['A'] (valve): Illumina HiSeq 2500 :: Valve with 24 ports
          for flowcell A.
        - v24['B'] (valve): Illumina HiSeq 2500 :: Valve with 24 ports
          for flowcell B.
        - lasers['green'] (laser): Illumina HiSeq 2500 :: Laser for 532 nm line.
        - lasers['red'] (laser): Illumina HiSeq 2500 :: Laser for 660 nm line.
        - f (fpga): Illumina HiSeq 2500 :: FPGA.
        - optics (optics): Illumina HiSeq 2500 :: Optics.
        - cam1 (camera): Camera for 558 nm and 687 nm emissions.
        - cam2 (camera): Camera for 610 nm and 740 nm emissions.
        - T (temperature): ARM9 CHEM for stage and temperature control.
        - logger (logger): Logger object to log communication with HiSeq.
        - image_path (path): Directory to store images in.
        - preview_path (path): Directory to store preview images ni
        - log_path (path): Directory to write log files in.
        - tile_width (float): Width of field of view in mm.
        - resolution (float): Scale of pixels in microns per pixel.
        - bundle_height: Line bundle height for TDI imaging.
        - nyquist_obj: Nyquist sampling distance of z plane in objective steps.
        - channels: List of imaging channel names.
        - AF: Autofocus routine, options are full, partial, full once,
          partial once, or None, the default is partial once.
        - focus_tol: Focus tolerance, distance in microns.
        - stack_split: = Portion of image stack below optimal focus object step, default is 2/3
        - overlap: Pixel overlap, the default is 0.
        - overlap_dir: Pixel overlap direction (left/right), the default is left.
        - virtual: Flag for using virtual HiSeq
        - fc_origin: Upper right X and Y stage step position for flowcell slots.
        - scan_flag: True if HiSeq is currently scanning
        - current_view: Block run to show latest images, otherwise is None
        - email_to: Email recipients to send notifications to
        - email_password: Password to login to email account

    """


    def __init__(self, name = 'HiSeq2500', Logger = None, com_ports = None):
        """Constructor for the HiSeq."""

        if com_ports is None:
            com_ports = get_com_ports('HiSeq2500')

        self.config, self._config_path = get_machine_config()
        
        if Logger is None:
            # Create a custom logger
            Logger = logging.getLogger(__name__)
            Logger.setLevel(logging.DEBUG)
            # Create console handler
            c_handler = logging.StreamHandler()
            c_handler.setLevel(logging.INFO)
            # Create formatters and add it to handlers
            c_format = logging.Formatter('%(asctime)s - %(message)s', datefmt = '%Y-%m-%d %H:%M')
            c_handler.setFormatter(c_format)
            Logger.addHandler(c_handler)

        self.y = ystage.Ystage(com_ports['ystage'], logger = Logger)
        self.f = fpga.FPGA(com_ports['fpgacommand'], com_ports['fpgaresponse'], logger = Logger)
        self.x = xstage.Xstage(com_ports['xstage'], logger = Logger)
        self.lasers = {'green': laser.Laser(com_ports['laser1'], color = 'green',
                                            logger = Logger),
                      'red': laser.Laser(com_ports['laser2'], color = 'red',
                                         logger = Logger)}
        self.z = zstage.Zstage(self.f, logger = Logger)
        self.obj = objstage.OBJstage(self.f, logger = Logger)
        self.optics = optics.Optics(self.f, logger = Logger, config = self.config)
        self.cams = {0: None, 1:None}
        self._cam_frames_sentinel = {0: False, 1: False}
        self.p = {'A': pump.Pump(com_ports['pumpa'], 'pumpA', logger = Logger),
                  'B': pump.Pump(com_ports['pumpb'], 'pumpB', logger = Logger)
                  }
        self.v10 = {'A': valve.Valve(com_ports['valvea10'], 'valveA10', logger = Logger),
                    'B': valve.Valve(com_ports['valveb10'], 'valveB10', logger = Logger)
                    }
        self.v24 = {'A': valve.Valve(com_ports['valvea24'], 'valveA24', logger = Logger),
                    'B': valve.Valve(com_ports['valveb24'], 'valveB24', logger = Logger)
                    }
        self.T = temperature.Temperature(com_ports['arm9chem'], logger = Logger)
        self.image_path = Path(getcwd())                                                # path to save images in
        self.log_path = Path(getcwd())                                                  # path to save logs in
        self.focus_path = Path(getcwd())                                                # path to save focus data in
        self.preview_path = Path(getcwd())                                              # path to save preview images in
        self.fc_origin = {'A':[17571,-180000],
                          'B':[43310,-180000]}
        self.tile_width = 0.769                                                 #mm
        self.resolution = 0.375                                                 #um/px
        self.bundle_height = 128
        self.nyquist_obj = 235                                                  # 0.9 um (235 obj steps) is nyquist sampling distance in z plane
        self.logger = Logger
        self.channels = None
        self.AF = 'partial'                                                     # autofocus routine
        self.focus_tol = 0                                                      # um, focus tolerance
        self.stack_split = 2/3                                                  # portion of image stack below optimal focus object step
        self.overlap = 0
        self.overlap_dir = 'left'
        self.virtual = False                                                    # virtual flag
        self.scan_flag = False                                                  # imaging/scanning flag
        self.current_view = None                                                # latest images
        self.name = name
        self.email_to = False
        self.email_password = None
        self.check_COM()


    def check_COM(self):
        """Check to see COM Ports setup sucessfully."""

        com_ports = ['X Stage', 'Y Stage', 'FPGA', 'Laser1', 'Laser2',
                     'Kloehn A', 'Kloehn B', 'VICI A1', 'VICI B1', 'VICI A2',
                     'VICI B2', 'ARM9 CHEM']
        success = []
        success.append(False) if self.x.serial_port is None else success.append(True)
        success.append(False) if self.y.serial_port is None else success.append(True)
        success.append(False) if self.f.serial_port is None else success.append(True)
        success.append(False) if self.lasers['green'].serial_port is None else success.append(True)
        success.append(False) if self.lasers['red'].serial_port is None else success.append(True)
        success.append(False) if self.p['A'].serial_port is None else success.append(True)
        success.append(False) if self.p['B'].serial_port is None else success.append(True)
        success.append(False) if self.v10['A'].serial_port is None else success.append(True)
        success.append(False) if self.v10['B'].serial_port is None else success.append(True)
        success.append(False) if self.v24['A'].serial_port is None else success.append(True)
        success.append(False) if self.v24['B'].serial_port is None else success.append(True)
        success.append(False) if self.T.serial_port is None else success.append(True)

        if not all(success):
            for i, go in enumerate(success):
                if not go:
                    print(com_ports[i],'Offline')
            response = None
            while response is None:
                response = input('Proceed with out all instruments? ').lower()
                if response not in ['y', 'yes', 'proceed', 'true']:
                    raise ValueError
                else:
                    return True


    def initializeCams(self):
        """Initialize all cameras."""

        self.message('HiSeq::Initializing cameras')
        from . import dcam
        
        for i in range(2):

            self.cams[i] = dcam.HamamatsuCamera(i, logger = self.logger)

            #Set emission labels, wavelengths in  nm
            # TODO, pull from config
            if i == 0:
                self.cams[i].left_emission = 687
                self.cams[i].right_emission = 558
            else:
                self.cams[i].left_emission = 610
                self.cams[i].right_emission = 740

            # Initialize camera 1
            # self.cam1.setPropertyValue("exposure_time", 40.0)
            # self.cam1.setPropertyValue("binning", 1)
            # self.cam1.setPropertyValue("sensor_mode", 4)                            #1=AREA, 2=LINE, 4=TDI, 6=PARTIAL AREA
            # self.cam1.setPropertyValue("trigger_mode", 1)                           #Normal
            # self.cam1.setPropertyValue("trigger_polarity", 1)                       #Negative
            # self.cam1.setPropertyValue("trigger_connector", 1)                      #Interface
            # self.cam1.setPropertyValue("trigger_source", 2)                         #1 = internal, 2=external
            # self.cam1.setPropertyValue("contrast_gain", 0)
            # self.cam1.setPropertyValue("subarray_mode", 1)                          #1 = OFF, 2 = ON
            self.cams[i].setTDI()
            self.cams[i].captureSetup()
            self.cams[i].get_status()

        self.channels =[str(self.cams[0].left_emission),
                        str(self.cams[0].right_emission),
                        str(self.cams[1].left_emission),
                        str(self.cams[1].right_emission)]
        

    def initializeInstruments(self):
        """Initialize x,y,z, & obj stages, pumps, valves, optics, and FPGA."""
        msg = 'HiSeq::'

        self.message(msg+'Initializing FPGA')
        self.f.initialize()
        self.f.LED(1, 'green')
        self.f.LED(2, 'green')

        #Initialize X Stage before Y Stage!
        self.message(msg+'Initializing X & Y stages')
        self.y.command('OFF')
        homed = self.x.initialize()
        self.y.initialize()
        self.message(msg+'Initializing lasers')
        self.lasers['green'].initialize()
        self.lasers['red'].initialize()
        self.message(msg+'Initializing pumps and valves')
        self.p['A'].initialize()
        self.p['B'].initialize()
        self.v10['A'].initialize()
        self.v10['B'].initialize()
        self.v24['A'].initialize()
        self.v24['B'].initialize()

        # Initialize Z, objective stage, and optics after FPGA
        self.message(msg+'Initializing optics and Z stages')
        self.z.initialize()
        self.obj.initialize()
        self.optics.initialize()

        #Initialize ARM9 CHEM for temperature control
        self.T.initialize()

        #Sync TDI encoder with YStage
        self.message(msg+'Syncing Y stage')
        while not self.y.check_position():
            time.sleep(1)
        self.y.position = self.y.read_position()
        self.f.write_position(0)

        self.logger.info(msg+'Initialized!')

        return homed

    def write_metadata(self, n_frames, meta_path, lock = None):
        """Write image metadata to file.

           **Parameters:**
            - n_frames (int): Number of frames in the images.
            - meta_path (path): Path object to write metadata file
            - lock (ThreadLock): Metdata file lock

           **Returns:**
            - path: Path object to write metadata file.
        """

        date = time.strftime('%Y%m%d_%H%M%S')

        if lock is None:
            lock = threading.Lock()

        with lock:
            with open(meta_path, 'w') as meta_f:
                meta_f.write(f'''
time {date}
y {self.y.position}
x {self.x.position}
z {self.z.position}
obj {self.obj.position}
frames {n_frames}
bundle {self.bundle_height}
TDIY {self.f.read_position()}
laser1 {self.lasers['green'].get_power()}
laser2 {self.lasers['red'].get_power()}
ex filters {self.optics.ex}
em filter {self.optics.em_in}
interval 1 {self.cams[0].getFrameInterval()}
interval 2 {self.cams[1].getFrameInterval()}
flowcell A {self.T.T_fc[0]} °C
flowcell B {self.T.T_fc[1]} °C
''')

        return meta_path

    def ready_cam(self, index, n_frames):
        """Ready cameras for imaging.""" 
        
        # Make sure cameras are ready (status = 3)
        
        cam = self.cams[index]
        
        while cam.get_status() != 3:
            cam.stopAcquisition()
            cam.freeFrames()
            cam.captureSetup()

        # Set mode and bundle height
        if cam.sensor_mode != 'TDI':
            cam.setTDI()
            cam.setPropertyValue("sensor_mode_line_bundle_height", self.bundle_height)


        # Allocate memory for image data
        cam.captureSetup()
        cam.allocFrame(n_frames)
        self._cam_frames_sentinel[index] = False

        return cam.sensor_mode



    def save_cam(self, index, n_frames, image_name, meta_path = None, lock = None):
        """Save images from camera and write metadata."""
        
        cam = self.cams[index]
        
        # Stop Cameras
        cam.stopAcquisition()

        # Check if all frames were taken from camera 1 then save images
        actual_frames = cam.getFrameCount()
        self.logger.debug(f'Cam {index} frames = {actual_frames}')
        self._cam_frames_sentinel[index] = n_frames == actual_frames
        if self._cam_frames_sentinel[index]:
            cam.saveImage(image_name, self.image_path)

        # Free up frames/memory
        cam.freeFrames()

        if meta_path is not None:
            if lock is None:
                lock = threading.Lock()
            with lock:
                with open(meta_path, 'a') as meta_f:
                    meta_f.write(f'frame count {index} {actual_frames} \n')

        return actual_frames


    def save_TDI_meta(self, meta_path, lock):
        """Save TDI FPGA data to meta data file."""
        with lock:
            with open(meta_path, 'a') as meta_f:
                response = self.f.command('TDICLINES')
                meta_f.write('clines ' + str(response) + '\n')
                response = self.f.command('TDIPULSES')
                meta_f.write('pulses ' + str(response) +'\n')
            
            
            
    def sync_stage(self):
        """Sync Ystage with FPGA."""
        

        #Make sure TDI is synced with Ystage
        y_pos = self.y.position
        if abs(y_pos - self.f.read_position()) > 10:
            self.logger.debug('HiSeq::Attempting to sync TDI and stage')
            self.f.write_position(self.y.position)
        else:
            self.logger.debug('HiSeq::TDI and stage are synced')
        self.y.set_mode('imaging')
        
        return self.y.mode
   
    
    def y_move(self, y_pos):
        """Fast y_move for imaging.
        
           TODO: update hs.y.move
        """

        mode = 'moving'

        gains = self.y.configurations[mode]['g']
        velocity = str(self.y.configurations[mode]['v'])


        # Set Gains
        command = f'{self.y.prefix}GAINS({gains}){self.y.suffix}'
        self.y.logger.debug(f'HiSeqYstage::txmt::{command}')
        self.y.serial_port.write(command)                                            # Write to serial port
        self.y.serial_port.flush()  
        g_response = self.y.command('GAINS')

        # Set Velocity
        command = f'{self.y.prefix}V{velocity}{self.y.suffix}'
        self.y.logger.debug(f'HiSeqYstage::txmt::{command}')
        self.y.serial_port.write(command)                                            # Write to serial port
        self.y.serial_port.flush()  
        v_response = self.y.command('V')
        
         # Assert mode changed correctly
        try: 

            assert g_response == self.y.configurations[mode]['gcheck']
            assert v_response == self.y.configurations[mode]['vcheck']
            self.y.mode = mode
            self.y.gains = gains
            self.y.velocity = float(v_response.strip()[1:])

        except:
            self.y.logger.debug('Ystage::FAILED to change to moving mode')

        # Move Y Stage within 100 steps of position within 3 attempts.
        # Don't need to be precise because triggering camera based oh ROI
        # NOT on initial y position
        attempts = 0
        while abs(self.y.position - y_pos) > 100 and attempts <= 3:
            
            # Move Y Stage
            command = f'{self.y.prefix}D{y_pos}{self.y.suffix}'
            self.y.logger.debug(f'HiSeqYstage::txmt::{command}')
            self.y.serial_port.write(command) 
            self.y.serial_port.flush()
            command = f'{self.y.prefix}G{self.y.suffix}'
            self.y.logger.debug(f'HiSeqYstage::txmt::{command}')
            self.y.serial_port.write(command)
            self.y.serial_port.flush()
 
            # Wait till Y Stage is in position
            time.sleep(1)
            while not self.y.check_position():
                time.sleep(1)
            # Update Y Stage position
            attempts += 1
            self.y.read_position() 
                
        self.y.logger.debug(f'Ystage::Yposition::{self.y.position}')
        
        return self.y.position

    
    def x_move(self, x_pos):
        """Fast x move for imaging.
        
           TODO update hs.x.move
        """
        
        command = f'MA {x_pos}{self.x.suffix}'
        self.x.logger.debug(f'Xstage::txmt::{command}')
        self.x.serial_port.write(command)                                           
        self.x.serial_port.flush()       

        moving = 1
        while moving != 0:
            moving = int(self.x.command('PR MV'))                                 # Check if moving, 1 = yes, 0 = no

        self.x.position = int(self.x.command('PR P'))                             # Set position

        return x_pos == self.x.position                                        # Return TRUE if in position or False if not
        

    def take_picture(self, n_frames=None, image_name=None, pos_dict=None):
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
                - pos_dict (dict, optional) Position dictionary from position(AorB, [LLx, LLy, URx, URy])

               **Returns:**
                - bool: True if all of the frames of the image were taken, False if
                  there were incomplete frames.

            """

            y = self.y
            f = self.f
            cam1 = self.cams[0]
            cam2 = self.cams[1]

            if image_name is None:
                image_name = time.strftime('%Y%m%d_%H%M%S')

            msg = 'HiSeq::TakePicture::'
            meta_path = self.image_path / f'meta_{image_name}.txt'
            
            try: 
                y_pos = pos_dict.get('y_initial', self.y.position)
                if n_frames is None:
                    n_frames = pos_dict.get('n_frames', n_frames)
            except:
                self.logger.debug('Could not get y_initial from position dictionary')
                y_pos = self.y.position

            setup_threads = []
            meta_file_lock = threading.Lock()
            setup_threads.append(threading.Thread(target = self.sync_stage))
            setup_threads.append(threading.Thread(target = self.ready_cam, args = (0, n_frames)))
            setup_threads.append(threading.Thread(target = self.ready_cam, args = (1, n_frames)))
            setup_threads.append(threading.Thread(target = self.write_metadata, args = (n_frames, meta_path, meta_file_lock)))

            for t in setup_threads:
                t.start()
            for t in setup_threads:
                t.join()

            #Arm stage triggers
            #
            #TODO check trigger y values are reasonable
            n_triggers = n_frames * self.bundle_height 
            end_y_pos = int(y_pos - n_triggers*self.resolution*y.spum - 300000)
            self.logger.debug(f'{msg}end y pos = {end_y_pos}')
            self.logger.debug(f'{msg}n triggers = {n_triggers}')
            f.TDIYPOS(y_pos)
            f.TDIYARM3(n_triggers, y_pos)

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

            y_thread = threading.Thread(target = self.y_move, args = (y_pos,))
            y_thread.start()

            save_threads = []
            save_threads.append(threading.Thread(target = self.save_cam, args = (0, n_frames, image_name, meta_path, meta_file_lock)))
            save_threads.append(threading.Thread(target = self.save_cam, args = (1, n_frames, image_name, meta_path, meta_file_lock)))
            save_threads.append(threading.Thread(target = self.save_TDI_meta, args = (meta_path, meta_file_lock)))

            for t in save_threads:
                t.start()
            for t in save_threads:
                t.join()

            # If expected number of frames acquired sentinel = True
            good_image = self._cam_frames_sentinel[0] + self._cam_frames_sentinel[1] == 2

            if not good_image:
                self.logger.debug(f'{msg} Missing Images, resetting FPGA')
                f.reset()
            
            return good_image, y_thread
    


    def obj_stack(self, n_frames = None, velocity = None):
        """Take an objective stack of images.

           The start and stop position of the objective is set by
           **hs.obj.focus_start** and **hs.obj.focus_stop**.

           **Parameters:**
            - n_frames (int): Number of images in the stack.
            - velocity (float): Speed in mm/s to move objective.

           **Returns:**
            - array: N x 2 array where the column 1 is the objective step the
              frame was taken and column 2 is the file size of the frame summed
              over all channels

        """
        msg = 'HiSeq::ObjectiveStack::'

        f = self.f
        obj = self.obj
        z = self.z
        cam1 = self.cams[0]
        cam2 = self.cams[1]


        if cam1.sensor_mode != 'AREA':
            cam1.setAREA()
        if cam2.sensor_mode != 'AREA':
            cam2.setAREA()
        # Make sure cameras are ready (status = 3)
        while cam1.get_status() != 3:
            cam1.stopAcquisition()
            cam1.freeFrames()
            cam1.captureSetup()
        while cam2.get_status() != 3:
            cam2.stopAcquisition()
            cam2.freeFrames()
            cam2.captureSetup()


        #Set line bundle height to 8
        cam1.setPropertyValue("sensor_mode_line_bundle_height", 64)
        cam2.setPropertyValue("sensor_mode_line_bundle_height", 64)

        cam1.captureSetup()
        cam2.captureSetup()

        # Update limits that were previously based on estimates
        obj.update_focus_limits(cam_interval = cam1.getFrameInterval(),
                                range = obj.focus_range,
                                spacing = obj.focus_spacing)
        if n_frames is None:
            n_frames = obj.focus_frames
        if velocity is None:
            velocity = obj.focus_velocity

        response = cam1.allocFrame(n_frames)
        response = cam2.allocFrame(n_frames)


        # Position objective stage
        obj.set_velocity(5) # mm/s
        obj.move(obj.focus_start)

        # Set up objective to trigger as soon as it moves
        obj.set_velocity(velocity) #mm/s
        obj.set_focus_trigger(obj.focus_start)

        # Open laser shutters
        f.command('SWLSRSHUT 1')

        # Prepare move objective to move
        text = 'ZMV ' + str(obj.focus_stop) + obj.suffix
        obj.fpga.serial_port.write(text)

        # Start Cameras
        cam1.startAcquisition()
        cam2.startAcquisition()

        # Move objective
        obj.fpga.serial_port.flush()
        response = obj.fpga.serial_port.readline()

        # Wait for objective
        start_time = time.time()
        stack_time = (obj.focus_stop - obj.focus_start)/obj.spum/1000/velocity

        while obj.check_position() != obj.focus_stop:
            now = time.time()

            if now - start_time > stack_time*10:
                self.message(msg,'Objective took too long to move.')
                break

        # Wait for imaging
        start_time = time.time()
        while cam1.getFrameCount() + cam2.getFrameCount() != 2*n_frames:
            now = time.time()
            if now - start_time > stack_time*20:
                self.message(msg, 'Imaging took too long.')
                break

        # Close laser shutters
        f.command('SWLSRSHUT 0')

        # Stop cameras
        cam1.stopAcquisition()
        cam2.stopAcquisition()

        # Check if received correct number of frames
        if cam1.getFrameCount() != n_frames:
            self.message(True, msg,'Cam1::Images not taken')
            self.message(False,msg,'Cam1::',cam1.getFrameCount(),'of',n_frames)
            image_complete = False
        else:
            cam1_stack = cam1.getFocusStack()
            image_complete = True
        # Check if all frames were taken from camera 2 then save images
        if cam2.getFrameCount() != n_frames:
            self.message(True, msg, 'Cam2::Images not taken')
            self.message(False,msg,'Cam2::',cam2.getFrameCount(),'of',n_frames)
            image_complete = False
        else:
            cam2_stack = cam2.getFocusStack()
            image_complete = True

        cam1.freeFrames()
        cam2.freeFrames()

        # if image_complete:
        #     f_filesize = np.concatenate((cam1_filesize,cam2_filesize), axis = 1)
        # else:
        #     f_filesize = 0

        return np.hstack((cam1_stack, cam2_stack))



    def reset_stage(self):
        """Home ystage and sync with TDI through the FPGA."""

        self.logger.debug('HiSeq::Resetting FPGA and syncing with stage')
        self.y.move(self.y.home)
        self.f.initialize()
        self.f.write_position(self.y.home)


    def move_stage_out(self):
        """Move stage out for loading/unloading flowcells."""

        self.message('HiSeq::Moving stage out')
        self.z.move([0,0,0])
        self.x.move(self.x.home)
        self.y.set_mode('moving')
        #self.y.command('GAINS(5,10,7,1.5,0)')
        #self.y.command('V1')
        self.y.move(self.y.min_y)

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

    def expose(self, pos_dict, repeat=1, power=200, OD = 'open'):
        """Expose ROI to green light.

        TODO: Add support for multiple lasers

           **Parameters:**
            - pos_dict (dict): Dictionary of stage position information.
            - repeat (int): Number of times to expose ROI, defeault = 1.
            - power (int): Green laser power in mW, default = 100 mW.
            - OD (int, str): Green emission filter OD, default = open.

           **Returns:**
            - float: Time in minutes to expose ROI.

        """

        self.message(False, f'Begin exposing tissue {repeat} times')

        #x spacing
        dx = self.tile_width*1000-self.resolution*self.overlap                       # x stage delta in in microns
        dx = round(dx*self.x.spum)

        # set laser power
        self.lasers['green'].set_power(power)

        # initialize stage
        self.y.move(pos_dict['y_initial'])
        self.x.move(pos_dict['x_initial'])
        x_pos = pos_dict['x_initial']

        # set optical filter
        self.optics.move_ex('green', OD)
        self.message(False, f'Moving green excitation filter to {OD} position')
        laser_power = self.lasers['green'].get_power()
        self.message(False, f'Green laser power is {laser_power} mW')

        start_time = time.time()
        direction = 1
        for xi in range(pos_dict['n_tiles']):
            for yi in range(repeat):

                # Open laser shutter
                self.f.command('SWLSRSHUT 1')

                if direction > 0:
                    self.y.move(pos_dict['y_final'])
                else:
                    self.y.move(pos_dict['y_initial'])

                direction *= -1

                # Close laser shutter
                self.f.command('SWLSRSHUT 0')

            self.x.move(self.x.position + dx)

        stop_time = time.time()
        total_time = round((stop_time - start_time)/60,2)
        self.message(False, f'Finished exposing tissue in {total_time} minutes')

        return total_time



    def autofocus(self, pos_dict):
        """Find optimal objective position for imaging, True if found."""

        image_path = self.image_path
        try:
            opt_obj_pos = focus.autofocus(self, pos_dict)
        except Exception as error:
            print(traceback.format_exc())
            self.message('HiSeq::Autofocus::',error)
            opt_obj_pos = False

        if opt_obj_pos:
            self.obj.move(opt_obj_pos)
            self.message('HiSeq::Autofocus complete')
            success =  True
        else:
            self.obj.move(self.obj.focus_rough)
            self.message('HiSeq::Autofocus failed')
            success =  False

        self.image_path = image_path

        return success

#     def autolevel(self, focal_points, obj_focus):
#         """Tilt the stage motors so the focal points are on a level plane.

#             # TODO: Improve autolevel, only makes miniscule improvement

#            Parameters:
#            - focal_points [int, int, int]: List of focus points.
#            - obj_focus: Objective step of the level plane.

#            Returns:
#            - [int, int, int]: Z stage positions for a level plane.

#         """

#         # Find point closest to midpoint of imaging plane
#         obj_step_distance = abs(obj_focus - focal_points[:,2])
#         min_obj_step_distance = np.min(obj_step_distance)
#         p_ip = np.where(obj_step_distance == min_obj_step_distance)
#         offset_distance = obj_focus - focal_points[p_ip,2]

#         # Convert stage step position microns
#         focal_points[:,0] = focal_points[:,0]/self.x.spum
#         focal_points[:,1] = focal_points[:,1]/self.y.spum
#         focal_points[:,2] = focal_points[:,2]/self.obj.spum
#         offset_distance = offset_distance/self.obj.spum

#         # Find normal vector of imaging plane
#         u_ip = focal_points[1,:] - focal_points[0,:]
#         v_ip = focal_points[2,:] - focal_points[0,:]
#         n_ip = np.cross(u_ip, v_ip)

#         # Imaging plane correction
#         correction = [0, 0, n_ip[2]] - n_ip

#         # Find normal vector of stage plane
#         mp = np.array(self.z.get_motor_points())
#         mp[:,0] = mp[:,0] / self.x.spum
#         mp[:,1] = mp[:,1] / self.y.spum
#         mp[:,2] = mp[:,2] / self.z.spum
#         u_mp = mp[1,:] - mp[0,:]
#         v_mp = mp[2,:] - mp[0,:]
#         n_mp = np.cross(u_sp, v_sp)

#         # Pick reference motor
#         if correction[0] >= 0:
#             p_mp = 0 # right motor
#         elif correction[1] >= 0:
#             p_mp = 2 # left back motors
#         else:
#             p_mp = 1 # left front motor

#         # Target normal of level stage plane
#         n_tp = n_mp + correction

#         # Solve system equations for level plane
#         # A = np.array([[v_mp[1]-u_mp[1], -v_mp[1], u_mp[1]],
#         #               [u_mp[0]-v_mp[0], v_mp[0], u_mp[0]],
#         #               [0, 0, 0]])
#         A = np.array([[mp[2,1]-mp[1,1], mp[0,1]-mp[2,1], mp[1,1]-mp[0,1]],
#                       [mp[1,0]-mp[2,0], mp[2,0]-mp[0,0], mp[0,0]-mp[1,0]],
#                       [0,0,0]])
#         A[2, p_sp] = 1
#         #offset_distance = int(offset_distance*self.z.spum)
#         offset_distance += self.z.position[p_sp]/self.z.spum
#         B = np.array([n_tp[0], n_tp[1], offset_distance])
#         z_pos = np.linalg.solve(A,B)
#         z_pos = int(z_pos * self.z.spum)
#         z_pos = z_pos.astype('int')

#         self.z.move(z_pos)

#         return z_pos
    
    

    def zstack(self, n_Zplanes, n_frames=None, image_name=None, obj_direction=1, pos_dict=None, x_thread=None):
        """Take a zstack/tile of images.

           Takes images from all channels at incremental z planes at the same
           x&y position.

           **Parameters:**
            - n_Zplanes (int): Number of Z planes to image.
            - n_frames (int): Number of frames to image.
            - image_name (str): Common name for images, the default is a time stamp.
            - obj_direction (int, optional): 1 to stack from bottom to top (default). -1 to stack from top to bottom. 
            - pos_dict (dict, optional) Position dictionary from position(AorB, [LLx, LLy, URx, URy]).
            - x_thread (Thread, optional) Thread to move x stage.

           **Returns:**
            - int: Time it took to do zstack in seconds.

        """
        

        if image_name is None:
            image_name = time.strftime('%Y%m%d_%H%M%S')

        
        start = time.time()
        
        # Thread to move y stage, created in take_picture()
        y_thread = None 
        
        self.logger.debug(f'ZSTACK::obj_direction = {obj_direction}')
        for n in range(n_Zplanes):
            im_name = f'{image_name}_o{self.obj.position}'
            image_complete = False
            
            if x_thread is not None:
                x_thread.join()
            if y_thread is not None:
                y_thread.join()
                
            while not image_complete:
                image_complete, y_thread = self.take_picture(image_name=im_name, n_frames=n_frames, pos_dict=pos_dict)
                if image_complete and n < n_Zplanes-1:
                    obj_pos = self.obj.position + self.nyquist_obj*obj_direction
                    self.logger.debug(f'ZSTACK::obj move = {obj_pos}')
                    self.obj.move(obj_pos)
                elif not image_complete:
                    self.logger.warning('HiSeq::ZStack::WARNING::Image not taken')      

        y_thread.join()
        stop = time.time()

        return stop-start
    


    def scan(self, n_Zplanes, n_frames=None, n_tiles=None, image_name=None, obj_direction = 1, pos_dict=None):
        """Image a volume.

           Images a zstack at incremental x positions.
           The length of the image (y dimension) remains constant.
           Need a minimum overlap of 4 pixels for a significant x increment.

           **Parameters:**
            - n_tiles (int): Number of x positions to image.
            - n_Zplanes (int): Number of Z planes to image.
            - n_frames (int): Number of frames to image.
            - image_name (str): Common name for images, the default is a time stamp.
            - obj_direction (int, optional): 1 to stack from bottom to top (default). -1 to stack from top to bottom. 
            - pos_dict (dict, optional) Position dictionary from position(AorB, [LLx, LLy, URx, URy]).

           **Returns:**
            - int: Time it took to do scan in seconds.

        """


        if isinstance(pos_dict, dict) and n_tiles is None:
            n_tiles = pos_dict.get('n_tiles', n_tiles)

        
        dx = self.tile_width*1000-self.resolution*self.overlap                                            # x stage delta in in microns
        dx = round(dx*self.x.spum)                                                                        # x stage delta in steps

        if image_name is None:
            image_name = time.strftime('%Y%m%d_%H%M%S')

        start = time.time()

        for tile in range(n_tiles):
            if tile > 0:
                x_pos = self.x.position + dx
                x_thread = threading.Thread(target=self.x_move, args=(x_pos, ))                       # Move to next x position
                x_thread.start()
            else:
                x_thread = None
                x_pos = self.x.position
            self.message(f'HiSeq::Scan::Tile {tile+1} / {n_tiles}')
            im_name = f'{image_name}_x{x_pos}'
            stack_time = self.zstack(n_Zplanes,
                                     n_frames=n_frames,
                                     image_name=im_name,
                                     obj_direction=obj_direction,
                                     pos_dict=pos_dict,
                                     x_thread=x_thread)    
            obj_direction *= -1                                                                           # switch stack direction
            
        stop = time.time()

        return stop - start



    def twoscan(self, n):
        """Takes n (int) images at 2 different positions.

           For validation of positioning.
        """

        for i in range(n):
            self.take_picture(50, 128, y_pos = 6500000, x_pos = 11900,
                obj_pos = 45000)
            self.take_picture(50, 128, y_pos = 6250000, x_pos = 11500,
                obj_pos = 45000)



    def position(self, AorB, box):
        """Returns stage position information.

           The center of the image is used to bring the section into focus
           and optimize laser intensities. Image scans of sections start on
           the upper right corner of the section. The section is imaged in
           strips 0.760 mm wide by length of the section long until the entire
           section has been imaged. The box region of interest surrounding the
           section is converted into stage and imaging details to scan the
           entire section.

           =========  ==============================================
             key      description
           =========  ==============================================
           x_center   The xstage center position of the section.
           y_center   The ystage center position of the section.
           x_initial  Initial xstage position to scan the section.
           y_initial  Initial ystage position to scan the section.
           x_final    Last xstage position of the section scan
           y_final    Last ystage position of the section scan
           n_tiles    Number of tiles to scan the entire section.
           n_frames   Number of frames to scan the entire section.
           =========  ==============================================

           **Parameters:**
            - AorB (str): Flowcell A or B.
            - box ([float, float, float, float]) = The region of interest as
              x&y position of the corners of a box surrounding the section
              to be imaged defined as [LLx, LLy, URx, URy] where LL=Lower
              Left and UR=Upper Right corner using the slide ruler.

           **Returns:**
            - dict: Dictionary of stage positioning and imaging details to scan
              the entire section. See table above for details.
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
        x_initial = n_tiles*dx*1000/2                                           #1/2 fov width in microns
        if self.overlap_dir == 'left':
            x_initial -= self.resolution*self.overlap                           #Move stage to compensate for discarded initial px
        x_initial = int(x_center - x_initial*self.x.spum)
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
        pos['y_final'] = int(y_initial - y_length*self.y.spum - 300000)
        pos['x_final'] = int(x_initial +(LLx - URx)*1000*self.x.spum)
        pos['obj_pos'] = None

        return pos

    def px_to_step(self, row, col, pos_dict, scale):
        """Convert pixel coordinates in image to stage step position.

           **Parameters:**
            - row_col ([int,int]): Row and column pixel position in image.
            - pos_dict (dict): Dictionary of position data
            - scale (int): Scale factor of imaged

           **Returns:**
            - [int, int]: X-stage and Y-stage step position respectively.

        """
        #print(row_col)
        #row = row_col[0]
        #col = row_col[1]
        scale = scale*self.resolution
        x_init = pos_dict['x_initial']
        y_init = pos_dict['y_initial']

        x_step = col*scale*self.x.spum
        if self.overlap_dir == 'left':
            x_step += self.overlap*scale
        x_step = int(x_init + x_step - 315/2)

        trigger_offset = -80000
        frame_offset = 64/2*self.resolution*self.y.spum
        y_step = row*scale*self.y.spum
        y_step = int(y_init + trigger_offset - y_step - frame_offset)

        return [x_step, y_step]


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
        for color in colors:
            self.optics.move_ex(color,f_order[0][0])

        # Focus on section
        self.message('HiSeq::OptimizeFilter::Starting Autofocus')
        if self.autofocus(pos_dict):
            self.message('HiSeq::OptimizeFilter::Autofocus completed')
        else:
            self.message('HiSeq::OptimizeFilter::Autofocus failed')

        # Loop through filters and image section
        for f in range(n_filters+1):
            self.optics.move_ex(colors[0], f_order[0][f])
            for f in range(n_filters+1):
                self.optics.move_ex(colors[1], f_order[1][f])

                image_name = colors[0][0].upper()+str(self.optics.ex[0])+'_'
                image_name += colors[1][0].upper()+str(self.optics.ex[1])

                self.y.move(pos_dict['y_initial'])
                self.x.move(pos_dict['x_initial'])
                msg = 'HiSeq::OptimizeFilter::Excitation filter'
                self.message(msg, colors[0], self.optics.ex[0])
                self.message(msg, colors[1], self.optics.ex[1])
                self.message(msg, 'Starting imaging')
                img_time = self.scan(pos_dict['n_tiles'],1,
                                   pos_dict['n_frames'], image_name)
                img_time /= 60
                self.message(msg, 'Imaging complete in ', img_time, 'min.')



    def message(self, *args):
        """Print output text to logger or console.

           If there is no logger, text is printed to the console.
           If a logger is assigned, and the first argument is False, text is
           printed only the log, otherwise text is printed to the log & console.

        """

        i = 0
        if isinstance(args[0], bool):
            screen = args[0]
            i = 1
        else:
            screen = True

        msg = ''
        for a in args[i:]:
            msg = msg + str(a) + ' '


        if self.logger is None:
            print(msg)
        else:
            if screen:
                self.logger.info(msg)
            else:
                self.logger.debug(msg)

def _1gaussian(x, amp1,cen1,sigma1):
    """Gaussian function for curve fitting."""
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2)))

def get_com_ports(machine = 'HiSeq2500'):

    # Read cosmetic instrument names : and COM serial_number
    com_names = configparser.ConfigParser()
    with pkg_resources.path(resources, 'com_ports.cfg') as config_path:
        com_names.read(config_path)

    # Get dictionary of COM ports : and their serial_number
    devices = {dev.serial_number: dev.device for dev in comports()}

    # Match instruments to ports
    matched_ports = {}
    for instrument, sn in com_names.items(machine):
        try:
            matched_ports[instrument] = devices[sn]
        except ValueError:
            print('Could not find port for', instrument)

    # ids = []
    # com_ports = []
    # for d in devices:
    #     if 'USB Serial Port' in d.caption:
    #         try:
    #             ids.append(d.deviceid)
    #             caption = d.caption
    #             id_start = caption.find('(')+1
    #             id_end = caption.find(')')
    #             caption = caption[id_start:id_end]
    #             com_ports.append(caption)
    #         except:
    #             pass
    #
    # # Match instruments to ports
    # matched_ports = {}
    # for instrument, com_name in com_names.items(machine):
    #     try:
    #         ind = [i for i, id in enumerate(ids) if com_name in id]
    #         if len(ind) == 1:
    #             ind = ind[0]
    #         else:
    #             print('Multiple COM Port matches for', instrument)
    #             raise ValueError
    #         matched_ports[instrument] = com_ports[ind]
    #     except ValueError:
    #         matched_ports[instrument] = None
    #         print('Could not find port for', instrument)

    return matched_ports
