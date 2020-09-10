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
    #Get stage positioning and imaging details for section on flowcell A
    [xi, yi, xc, yx, n_tiles, n_frames] = hs.position['A', 15.5, 45, 10.5, 35]
    #Move to center of section
    hs.x.move(xc)
    12000
    hs.y.move(yc)
    True
    #Move stage within focusing range.
    hs.z.move([21500, 21500, 21500])
    #Find focus
    hs.fine_focus()
    [10000, 15000, 20000, 25000, 30000, 35000, 40000]
    [5, 24, 1245, 1593, 1353, 54, 9]
    #Get optimal filters
    [l1_filter, l2_filter] = hs.optimize_filters()
    # Take a 32 frame picture using the optimal filters
    hs.move_ex(1, l1_filter)
    hs.move_ex(2, l2_filter)
    hs.take_picture(32, image_name='FirstHiSeqImage')
    #Move stage to the initial image scan position and scan image
    hs.x.move(xi)
    10000
    hs.y.move(yi)
    True
    hs.scan(n_scans, 1, n_frames, image_name='FirstHiSeqScan')


TODO:
    - Double check gains and velocity are set in take_picture
    - Filter score without background substraction
    - New fine focus routine
    - Fix positioning details in example

Kunal Pandit 9/19
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

import time
from os.path import getsize
from os.path import join
from os import getcwd
import threading
import numpy as np
import imageio
from scipy.optimize import curve_fit
from math import ceil


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
       - logger (logger): Logger object to log communication with HiSeq.
       - image_path (path): Directory to store images in.
       - log_path (path): Directory to write log files in.
       - bg_path (path): Directory to background calibration images.
       - tile_width (float): Width of field of view in mm.
       - resolution (float): Scale of pixels in microns per pixel.
       - bundle_height: Line bundle height for TDI imaging.
       - nyquist_obj: Nyquist sampling distance of z plane in objective steps.
    """


    def __init__(self, Logger = None,
                       yCOM = 'COM10',
                       xCOM = 'COM9',
                       pumpACOM = 'COM20',
                       pumpBCOM = 'COM21',
                       valveA24COM = 'COM22',
                       valveB24COM = 'COM23',
                       valveA10COM = 'COM18',
                       valveB10COM = 'COM19',
                       fpgaCOM = ['COM12','COM15'],
                       laser1COM = 'COM13',
                       laser2COM = 'COM14'):
        """Constructor for the HiSeq."""

        self.y = ystage.Ystage(yCOM, logger = Logger)
        self.f = fpga.FPGA(fpgaCOM[0], fpgaCOM[1], logger = Logger)
        self.x = xstage.Xstage(xCOM, logger = Logger)
        self.lasers = {'green': laser.Laser(laser1COM, color = 'green',
                                            logger = Logger),
                      'red': laser.Laser(laser2COM, color = 'red',
                                         logger = Logger)}
        self.z = zstage.Zstage(self.f.serial_port, logger = Logger)
        self.obj = objstage.OBJstage(self.f.serial_port, logger = Logger)
        self.optics = optics.Optics(self.f.serial_port, logger = Logger)
        self.cam1 = None
        self.cam2 = None
        self.p = {'A': pump.Pump(pumpACOM, 'pumpA', logger = Logger),
                  'B': pump.Pump(pumpBCOM, 'pumpB', logger = Logger)
                  }
        self.v10 = {'A': valve.Valve(valveA10COM, 'valveA10', logger = Logger),
                    'B': valve.Valve(valveB10COM, 'valveB10', logger = Logger)
                    }
        self.v24 = {'A': valve.Valve(valveA24COM, 'valveA24', logger = Logger),
                    'B': valve.Valve(valveB24COM, 'valveB24', logger = Logger)
                    }
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
        self.af = True                                                          # autofocus flag

    def initializeCams(self, Logger=None):
        """Initialize all cameras."""

        from . import dcam

        self.cam1 = dcam.HamamatsuCamera(0, logger = Logger)
        self.cam2 = dcam.HamamatsuCamera(1, logger = Logger)

        #Set emission labels, wavelengths in  nm
        self.cam1.left_emission = 687
        self.cam1.right_emission = 558
        self.cam2.left_emission = 610
        self.cam2.right_emission = 740

        # Initialize camera 1
        self.message('Initializing camera 1...')
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
        self.message('Initializing camera 2...')
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

        #Initialize X Stage before Y Stage!
        self.message('Initializing X & Y stages')
        #self.x.initialize()
        #TODO, make sure x stage is in correct place.
        self.x.initialize() # Do it twice to make sure its centered!
        self.y.initialize()
        self.message('Initializing lasers')
        self.lasers['green'].initialize()
        self.lasers['red'].initialize()
        self.message('Initializing pumps and valves')
##        self.p['A'].initialize()
##        self.p['B'].initialize()
##        self.v10['A'].initialize()
##        self.v10['B'].initialize()
##        self.v24['A'].initialize()
##        self.v24['B'].initialize()
        self.message('Initializing FPGA')
        self.f.initialize()

        # Initialize Z, objective stage, and optics after FPGA
        self.message('Initializing optics and Z stages')
        self.z.initialize()
        self.obj.initialize()
        self.optics.initialize()

        #Sync TDI encoder with YStage
        self.message('Syncing Y stage')
        while not self.y.check_position():
            time.sleep(1)
        self.y.position = self.y.read_position()
        self.f.write_position(0)

        self.message('Initialized!')


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
            self.message(True, 'Attempting to sync TDI and stage')
            f.write_position(y.position)

        #TO DO, double check gains and velocity are set
        #Set gains and velocity of image scanning for ystage
        response = y.command('GAINS(5,10,5,2,0)')
        response = y.command('V0.15400')


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
            self.message('Cam1 frames = ', cam1.getFrameCount())
            self.message(True, 'Cam1 image not taken')
            image_complete = False
        else:
            cam1.saveImage(image_name, self.image_path)
            image_complete = True
        # Check if all frames were taken from camera 2 then save images
        if cam2.getFrameCount() != n_frames:
            self.message('Cam2 frames = ', cam2.getFrameCount())
            self.message(True, 'Cam2 image not taken')
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


    def obj_stack(self, n_frames = 232, velocity = 0.1):
        """Take an objective stack of images.

           Parameters:
           - n_frames (int): Number of images in the stack.
           - start (int): Objective step to start imaging.
           - stop (int): Objective step to stop imaging

           Returns:
           - array: N x 2 array where the column 1 is the objective step the
                    frame was taken and column 2 is the file size of the frame
                    summed over all channels

        """

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

        cam1.allocFrame(n_frames)
        cam2.allocFrame(n_frames)


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
        while obj.check_position() != obj.focus_stop:
           now = time.time()
           if now - start_time > 10:
               self.message(True, 'Objective took too long to move.')
               break

        # Wait for imaging
        start_time = time.time()
        while cam1.getFrameCount() + cam2.getFrameCount() != 2*n_frames:
           now = time.time()
           if now - start_time > 10:
               self.message(True, 'Imaging took too long.')
               break

        # Close laser shutters
        f.command('SWLSRSHUT 0')

        # Stop cameras
        cam1.stopAcquisition()
        cam2.stopAcquisition()

        # Check if received correct number of frames
        if cam1.getFrameCount() != n_frames:
            self.message(True, 'Cam1::Images not taken')
            self.message('Cam1::',cam1.getFrameCount(),'of',n_frames)
            image_complete = False
        else:
            cam1_filesize = cam1.saveFocus(self.image_path)
            image_complete = True
        # Check if all frames were taken from camera 2 then save images
        if cam2.getFrameCount() != n_frames:
            self.message(True, 'Cam2::Images not taken')
            self.message('Cam2::',cam2.getFrameCount(),'of',n_frames)
            image_complete = False
        else:
            cam2_filesize = cam2.saveFocus(self.image_path)
            image_complete = True

        cam1.freeFrames()
        cam2.freeFrames()

        if image_complete:
            f_filesize = np.concatenate((cam1_filesize,cam2_filesize), axis = 1)
        else:
            f_filesize = 0

        return f_filesize



    def reset_stage(self):
        """Home ystage and sync with TDI through the FPGA."""

        self.message('Resetting FPGA and syncing with stage')
        self.y.move(self.y.home)
        self.f.initialize()
        self.f.write_position(self.y.home)


    def move_stage_out(self):
        """Move stage out for loading/unloading flowcells."""

        self.z.move([0,0,0])
        self.x.move(self.x.home)
        self.y.move(self.y.min_y)

    def autofocus(self, pos_dict):
        """Find optimal objective position for imaging, True if found."""

        opt_obj_pos = focus.autofocus(self, pos_dict)
        if opt_obj_pos:
            self.obj.move(opt_obj_pos)
            self.message('Autofocus complete')
            return True
        else:
            self.obj.move(self.obj.focus_rough)
            self.message('Autofocus failed')
            return False

    def autolevel(self, focal_points, obj_focus):
        """Tilt the stage motors so the focal points are on a level plane.

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
                    self.message('WARNING::Image not taken')
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


    def twoscan(self, n):
        """Takes n (int) images at 2 different positions.

           For validation of positioning.
        """

        for i in range(n):
            hs.take_picture(50, 128, y_pos = 6500000, x_pos = 11900,
                obj_pos = 45000)
            hs.take_picture(50, 128, y_pos = 6250000, x_pos = 11500,
                obj_pos = 45000)


    def jpeg(self, filename):
        """Saves image from all channels as jpegs.

           Convert all tiff images with the common filename to jpegs and
           add the file size from all 4 emission channels together as a
           measure of how sharp and focused the image is. Images that are
           more in focus will have a larger file size after compressed into
           a jpeg.

           Parameters:
           filename (str): Common filename for all emmission channels.

           Returns:
           int: Sum of the jpeg size from all emmission channels.
        """

        image_prefix=[self.cam1.left_emission,
                      self.cam1.right_emission,
                      self.cam2.left_emission,
                      self.cam2.right_emission]

        C = 0
        for image in image_prefix:
            im_path = join(self.image_path, str(image)+'_'+filename+'.tiff')
            im = imageio.imread(image_path)                                     #read picture
            im = im[64:,:]                                                      #Remove bright band artifact
            im_path = join(self.image_path, str(image)+'_'+filename+'.jpeg')
            imageio.imwrite(im_path, im)
            C += getsize(image_path)

        return C


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
            for f in self.optics.ex_dict[color].keys():
                if isinstance(f,float):
                    f_order.append(f)
            f_order[i] = sorted(f_order[i], reverse = True)
            f_order[i] = f_order[i][init_filter:init_filter+n_filters]
            f_order[i].append('home')

        # Set optical filters
        for color in colors:
            self.optics.move_ex(color,f_order[0][0])

        # Focus on section
        self.message(True, 'Starting Autofocus')
        if self.autofocus(pos_dict):
            self.message(True, 'Autofocus completed')
        else:
            self.message(True, 'Autofocus failed')

        # Loop through filters and image section
        for f in range(n_filters+1):
            self.optics.move_ex(colors[0], f_order[0][f])
            for f in range(n_filters+1):
                self.optics.move_ex(colors[1], f_order[1][f])

                image_name = colors[0][0].upper()+str(self.optics.ex[0])+'_'
                image_name += colors[1][0].upper()+str(self.optics.ex[1])

                self.y.move(pos_dict['y_initial'])
                self.x.move(pos_dict['x_initial'])
                msg = ['Excitation filters =']
                msg.append(colors[0], self.optics.ex[0])
                msg.append(colors[1],self.optics.ex[1])
                self.message(True, *msg)
                self.message(True, 'Starting imaging')
                img_time = self.scan(pos_dict['n_tiles'],1,
                                   pos_dict['n_frames'], image_name)
                img_time /= 60
                self.message(True, 'Imaging complete in ', img_time, 'min.')




    def optimize_filter_old(self, nframes, color, nbins = 256,
                        sat_threshold = 0.0005, signal_threshold = 20):
        """Finds and returns the best filter settings.

           Images a portion of the section with one laser at increasing laser
           intensity. Images are analyzed from all emission channels to
           calculate a filter score. Images are scored to maximize the image
           histogram without saturating too many pixels and minimizes
           crosstalk in non specific channels.

           **Parameters:**
           - nframes (int): Number of frames for portion of section.
           - color (str): Laser line color to optimize filter.
           - nbins (int, optional): Number of bins to define the image histogram,
             default is 256 bins.
           - sat_threshold (float): Maximum fraction of pixels allowed to be
             saturated, default is 0.0005.
           - signal_threshold (int): Maximum signal intensity allowed in
             nonspecific channels, default is 20.

           **Returns:**
           - float: Optimal filter for specified laser.

         """

        # Save y position
        y_pos = self.y.position

        image_prefix=[self.cam1.left_emission,                                  # 687
                      self.cam1.right_emission,                                 # 558
                      self.cam2.left_emission,                                  # 610
                      self.cam2.right_emission]                                 # 740

        #Order of filters to loop through
        f_order = []
        for key in hs.optics.ex_dict[color].keys():
            if isinstance(key,float):
                f_order.append(key)
        f_order = sorted(f_order, reverse = True)
        # f_order = [[4.0, 2.0, 1.6, 1.4, 0.6, 0.2],
        #         [4.5, 3.0, 2.0, 1.0, 0.9, 0.2]]

        # Empty list of optimal filters
        opt_filter = None
        #opt_filter = [None, None]
        # list of lasers, Green = 1, Red = 2
        #lasers = [1,2]

        # Loop through filters until optimized
        #for li in lasers:
        fi = 0                                                                  # filter index
        self.optics.move_ex(1, 'home')                                          # Home and block lasers
        self.optics.move_ex(2, 'home')
        self.optics.move_em_in(True)
        new_f_score = 0
        while opt_filter is None:
            self.optics.move_ex(color,f_order[fi])                              # Move excitation filter
            image_name = 'L'+color+'_filter'+str(f_order[fi])
            image_complete = False
            # Take Picture
            while not image_complete:
                image_complete = self.take_picture(nframes, 128, image_name)
                self.y.move(y_pos)

            contrast = np.array([])
            saturation = np.array([])
            #read picture
            for image in image_prefix:
                im_name = self.image_path+str(image)+'_'+image_name+'.tiff'
                #Calculate background subtracted contrast and %saturation
                C, S = contrast2(im_name, image, self.bg_path, nbins)

                contrast = np.append(contrast,C)
                saturation = np.append(saturation,S)

            old_f_score = new_f_score
            # make sure image is not too saturated
            if any(saturation > sat_threshold):
                new_f_score = 0
            # Green laser, 558+610-687-740 emissions
            elif self.optics.colors[color] == 1 :
                signal = contrast[1] + contrast[2]
                new_f_score = contrast[1] + contrast[2] - contrast[0] - contrast[3]
            # Red laser, 687+740-558-610- emissions
            elif self.optics.colors[color] == 2:
                signal = contrast[0] + contrast[3]
                new_f_score = contrast[0] + contrast[3] - contrast[1] - contrast[2]

            # Increase laser until you at least see some signal
            if signal > signal_threshold:
                # Too much crosstalk / saturation
                if old_f_score >= new_f_score and fi > 0:
                    opt_filter = f_order[fi-1]
                else:
                    fi += 1
            # Last filter
            elif fi == (len(f_order) - 1):
                opt_filter = f_order[fi]
            # Move to next filter
            else:
                fi += 1

            self.message('Laser : ' + str(li))
            self.message('Filter: ' + str(f_order[li-1][fi]))
            self.message('Contrast: ' + str(contrast))
            self.message('Saturation: ' + str(saturation))
            self.message('Filter Score ' + str(new_f_score))

        return opt_filter


    def message(self, *args):
        """Print output text to logger or console"""

        i = 0
        if isinstance(args[0], bool):
            if args[0]:
                i = 1

        msg = 'HiSeq::'
        for a in args[i:]:
            msg = msg + str(a) + ' '

        if self.logger is None:
            print(msg)
        else:
            if i is 1:
                self.logger.log(21,msg)
            else:
                self.logger.info(msg)


def contrast2(filename, channel, path, nbins=256):
    """Return the image contrast and fraction of pixels saturated.

       filename = name of image
       channel = color channel of image
       path = path of image
       nbins = number of bins in histogram

       Uses a precalibrated background file to subtract the background from the
       image, then fit the remaining signal to a gaussian curve to find the mean
       image contrast.
    """
    im = imageio.imread(filename)    #read picture
    im = im[64:,:]                                                              #Remove bright band artifact
    bg_path = join(path, str(channel) + 'background.txt')
    bg = np.loadtxt(bg_path)                                                    #Load background for sensor
    im = im - bg                                                                #Remove background
    im[im<0] = 0                                                                #Convert negative px values to 0

    contrast = np.max(im) - np.min(im)                                          #Calculate image contrast

    # Histogram of image
    xrange = np.array(range(4096))
    hist, bin_edges = np.histogram(im, bins = 256, range=(0,4095-np.min(bg)), density= True)
    saturation = hist[-1]

    return contrast, saturation
