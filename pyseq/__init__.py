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

import time
from os.path import getsize, join, isfile
from os import getcwd
import threading
import numpy as np
import imageio
from scipy.optimize import curve_fit
from math import ceil
import configparser

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
        - log_path (path): Directory to write log files in.
        - tile_width (float): Width of field of view in mm.
        - resolution (float): Scale of pixels in microns per pixel.
        - bundle_height: Line bundle height for TDI imaging.
        - nyquist_obj: Nyquist sampling distance of z plane in objective steps.
        - channels: List of imaging channel names.
        - AF: Autofocus routine, options are full, partial, full once,
          partial once, or None, the default is partial once.
        - focus_tol: Focus tolerance, distance in microns.
        - overlap: Pixel overlap, the default is 0.
        - overlap_dir: Pixel overlap direction (left/right), the default is left.
        - virtual: Flag for using virtual HiSeq
        - fc_origin: Upper right X and Y stage step position for flowcell slots.
        - scan_flag: True if HiSeq is currently scanning
        - current_view: Block run to show latest images, otherwise is None

    """


    def __init__(self, name = 'HiSeq2500', Logger = None):
        """Constructor for the HiSeq."""

        com_ports = get_com_ports('HiSeq2500')

        self.y = ystage.Ystage(com_ports['ystage'], logger = Logger)
        self.f = fpga.FPGA(com_ports['fpgacommand'], com_ports['fpgaresponse'], logger = Logger)
        self.x = xstage.Xstage(com_ports['xstage'], logger = Logger)
        self.lasers = {'green': laser.Laser(com_ports['laser1'], color = 'green',
                                            logger = Logger),
                      'red': laser.Laser(com_ports['laser2'], color = 'red',
                                         logger = Logger)}
        self.z = zstage.Zstage(self.f.serial_port, logger = Logger)
        self.obj = objstage.OBJstage(self.f.serial_port, logger = Logger)
        self.optics = optics.Optics(self.f.serial_port, logger = Logger)
        self.cam1 = None
        self.cam2 = None
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
        self.image_path = getcwd()                                                  # path to save images in
        self.log_path = getcwd()                                                  # path to save logs in
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
        self.overlap = 0
        self.overlap_dir = 'left'
        self.virtual = False                                                    # virtual flag
        self.scan_flag = False                                                  # imaging/scanning flag
        self.current_view = None                                                # latest images
        self.name = name
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


    def initializeCams(self, Logger=None):
        """Initialize all cameras."""

        self.message('HiSeq::Initializing cameras')
        from . import dcam

        self.cam1 = dcam.HamamatsuCamera(0, logger = Logger)
        self.cam2 = dcam.HamamatsuCamera(1, logger = Logger)

        #Set emission labels, wavelengths in  nm
        self.cam1.left_emission = 687
        self.cam1.right_emission = 558
        self.cam2.left_emission = 610
        self.cam2.right_emission = 740

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
        self.cam1.setTDI()
        self.cam1.captureSetup()
        self.cam1.get_status()

        # Initialize Camera 2
        # self.cam2.setPropertyValue("exposure_time", 40.0)
        # self.cam2.setPropertyValue("binning", 1)
        # self.cam2.setPropertyValue("sensor_mode", 4)                            #1=AREA, 2=LINE, 4=TDI, 6=PARTIAL AREA
        # self.cam2.setPropertyValue("trigger_mode", 1)                           #Normal
        # self.cam2.setPropertyValue("trigger_polarity", 1)                       #Negative
        # self.cam2.setPropertyValue("trigger_connector", 1)                      #Interface
        # self.cam2.setPropertyValue("trigger_source", 2)                         #1 = internal, 2=external
        # self.cam2.setPropertyValue("contrast_gain", 0)
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

        self.message(msg+'Initialized!')

        return homed

    def write_metadata(self, n_frames, image_name):
        """Write image metadata to file.

           **Parameters:**
            - n_frames (int): Number of frames in the images.
            - bundle (int): Line bundle height of the images.
            - image_name (int): Common name of the images.

           **Returns:**
            - file: Metadata file to write info about images to.
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
                     'interval 2 ' + str(self.cam2.getFrameInterval()) + '\n' +
                     'flowcell A ' + str(self.T.T_fc[0]) + ' °C' + '\n' +
                     'flowcell B ' + str(self.T.T_fc[1]) + ' °C' + '\n'
                     )

        return meta_f


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

        msg = 'HiSeq::TakePicture::'
        #Make sure TDI is synced with Ystage
        y_pos = y.position
        if abs(y_pos - f.read_position()) > 10:
            self.message(msg+'Attempting to sync TDI and stage')
            f.write_position(y.position)
        else:
            self.message(False, msg+'TDI and stage are synced')

        #TO DO, double check gains and velocity are set
        #Set gains and velocity of image scanning for ystage
        y.set_mode('imaging')
        # response = y.command('GAINS(5,10,5,2,0)')
        # response = y.command('V0.15400')


        # Make sure cameras are ready (status = 3)
        while cam1.get_status() != 3:
            cam1.stopAcquisition()
            cam1.freeFrames()
            cam1.captureSetup()
        while cam2.get_status() != 3:
            cam2.stopAcquisition()
            cam2.freeFrames()
            cam2.captureSetup()

        if cam1.sensor_mode != 'TDI':
            cam1.setTDI()
        if cam2.sensor_mode != 'TDI':
            cam2.setTDI()

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
            self.message(False, msg, 'Cam1 frames = ', cam1.getFrameCount())
            self.message(True, msg, 'Cam1 image not taken')
            image_complete = False
        else:
            cam1.saveImage(image_name, self.image_path)
            image_complete = True
        # Check if all frames were taken from camera 2 then save images
        if cam2.getFrameCount() != n_frames:
            self.message(False, msg, 'Cam2 frames = ', cam2.getFrameCount())
            self.message(True, msg,  'Cam2 image not taken')
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
        cam1 = self.cam1
        cam2 = self.cam2


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
        obj.serial_port.write(text)

        # Start Cameras
        cam1.startAcquisition()
        cam2.startAcquisition()

        # Move objective
        obj.serial_port.flush()
        response = obj.serial_port.readline()

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

        self.message('HiSeq::Resetting FPGA and syncing with stage')
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

    def autofocus(self, pos_dict):
        """Find optimal objective position for imaging, True if found."""

        opt_obj_pos = focus.autofocus(self, pos_dict)
        if opt_obj_pos:
            self.obj.move(opt_obj_pos)
            self.message('HiSeq::Autofocus complete')
            return True
        else:
            self.obj.move(self.obj.focus_rough)
            self.message('HiSeq::Autofocus failed')
            return False

    def autolevel(self, focal_points, obj_focus):
        """Tilt the stage motors so the focal points are on a level plane.

            # TODO: Improve autolevel, only makes miniscule improvement

           Parameters:
           - focal_points [int, int, int]: List of focus points.
           - obj_focus: Objective step of the level plane.

           Returns:
           - [int, int, int]: Z stage positions for a level plane.

        """

        # Find point closest to midpoint of imaging plane
        obj_step_distance = abs(obj_focus - focal_points[:,2])
        min_obj_step_distance = np.min(obj_step_distance)
        p_ip = np.where(obj_step_distance == min_obj_step_distance)
        offset_distance = obj_focus - focal_points[p_ip,2]

        # Convert stage step position microns
        focal_points[:,0] = focal_points[:,0]/self.x.spum
        focal_points[:,1] = focal_points[:,1]/self.y.spum
        focal_points[:,2] = focal_points[:,2]/self.obj.spum
        offset_distance = offset_distance/self.obj.spum

        # Find normal vector of imaging plane
        u_ip = focal_points[1,:] - focal_points[0,:]
        v_ip = focal_points[2,:] - focal_points[0,:]
        n_ip = np.cross(u_ip, v_ip)

        # Imaging plane correction
        correction = [0, 0, n_ip[2]] - n_ip

        # Find normal vector of stage plane
        mp = np.array(self.z.get_motor_points())
        mp[:,0] = mp[:,0] / self.x.spum
        mp[:,1] = mp[:,1] / self.y.spum
        mp[:,2] = mp[:,2] / self.z.spum
        u_mp = mp[1,:] - mp[0,:]
        v_mp = mp[2,:] - mp[0,:]
        n_mp = np.cross(u_sp, v_sp)

        # Pick reference motor
        if correction[0] >= 0:
            p_mp = 0 # right motor
        elif correction[1] >= 0:
            p_mp = 2 # left back motors
        else:
            p_mp = 1 # left front motor

        # Target normal of level stage plane
        n_tp = n_mp + correction

        # Solve system equations for level plane
        # A = np.array([[v_mp[1]-u_mp[1], -v_mp[1], u_mp[1]],
        #               [u_mp[0]-v_mp[0], v_mp[0], u_mp[0]],
        #               [0, 0, 0]])
        A = np.array([[mp[2,1]-mp[1,1], mp[0,1]-mp[2,1], mp[1,1]-mp[0,1]],
                      [mp[1,0]-mp[2,0], mp[2,0]-mp[0,0], mp[0,0]-mp[1,0]],
                      [0,0,0]])
        A[2, p_sp] = 1
        #offset_distance = int(offset_distance*self.z.spum)
        offset_distance += self.z.position[p_sp]/self.z.spum
        B = np.array([n_tp[0], n_tp[1], offset_distance])
        z_pos = np.linalg.solve(A,B)
        z_pos = int(z_pos * self.z.spum)
        z_pos = z_pos.astype('int')

        self.z.move(z_pos)

        return z_pos

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
                    self.message('HiSeq::ZStack::WARNING::Image not taken')
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
           Need a minimum overlap of 4 pixels for a significant x increment.

           **Parameters:**
            - n_tiles (int): Number of x positions to image.
            - n_Zplanes (int): Number of Z planes to image.
            - n_frames (int): Number of frames to image.
            - image_name (str): Common name for images, the default is a time
              stamp.

           **Returns:**
            - int: Time it took to do scan in seconds.

        """

        self.scan_flag = True
        dx = self.tile_width*1000-self.resolution*self.overlap                       # x stage delta in in microns
        dx = round(dx*self.x.spum)                                              # x stage delta in steps

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


    def twoscan(self, n):
        """Takes n (int) images at 2 different positions.

           For validation of positioning.
        """

        for i in range(n):
            hs.take_picture(50, 128, y_pos = 6500000, x_pos = 11900,
                obj_pos = 45000)
            hs.take_picture(50, 128, y_pos = 6250000, x_pos = 11500,
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
        pos['y_final'] = int(y_initial - y_length*self.y.spum)
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
                self.logger.log(21,msg)
            else:
                self.logger.info(msg)

def _1gaussian(x, amp1,cen1,sigma1):
    """Gaussian function for curve fitting."""
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2)))

def get_com_ports(machine = 'HiSeq2500'):

    # Read COM Names
    com_names = configparser.ConfigParser()
    with pkg_resources.path(resources, 'com_ports.cfg') as config_path:
        com_names.read(config_path)

    # Get list of connected devices
    import wmi
    conn = wmi.WMI()
    devices = conn.CIM_LogicalDevice()
    # Get lists of valid COM ports
    ids = []
    com_ports = []
    for d in devices:
        if 'USB Serial Port' in d.caption:
            try:
                ids.append(d.deviceid)
                caption = d.caption
                id_start = caption.find('(')+1
                id_end = caption.find(')')
                caption = caption[id_start:id_end]
                com_ports.append(caption)
            except:
                pass

    # Match instruments to ports
    matched_ports = {}
    for instrument, com_name in com_names.items(machine):
        try:
            ind = [i for i, id in enumerate(ids) if com_name in id]
            if len(ind) == 1:
                ind = ind[0]
            else:
                print('Multiple COM Port matches for', instrument)
                raise ValueError
            matched_ports[instrument] = com_ports[ind]
        except ValueError:
            matched_ports[instrument] = None
            print('Could not find port for', instrument)

    return matched_ports
