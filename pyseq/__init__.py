# Control an Illumina HiSeq 2500 System


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

import time
from os.path import getsize
from os.path import join
import threading
import numpy as np
import imageio
from scipy.optimize import curve_fit
from math import ceil



class HiSeq():

    #
    # Make HiSeq Object
    #
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

        self.y = ystage.Ystage(yCOM, logger = Logger)
        self.f = fpga.FPGA(fpgaCOM[0], fpgaCOM[1], logger = Logger)
        self.x = xstage.Xstage(xCOM, logger = Logger)
        self.l1 = laser.Laser(laser1COM, color = 'green', logger = Logger)
        self.l2 = laser.Laser(laser2COM, color = 'red', logger = Logger)
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
        self.image_path = None
        self.log_path = None
        self.bg_path = 'C:\\Users\\Public\\Documents\\PySeq2500\\PySeq2500V2\\calibration\\'
        self.distance = None
        self.distance_offset = 0
        self.fc_height = 1200   # height of flow cell in microns
        self.max_distance = self.z.max_z/self.z.spum + self.obj.max_z/self.obj.spum - self.fc_height
        self.fc_origin = {'A':[17571,-180000],
                          'B':[43310,-180000]}
        self.scan_width = 0.769 #mm
        self.resolution = 0.375 # um/px
        self.bundle_height = 128.0
        self.nyquist_obj = 235 # 0.9 um (235 obj steps) is nyquist sampling distance in z plane
        self.logger = Logger

    def initializeCams(self, Logger=None):

        from . import dcam

        self.cam1 = dcam.HamamatsuCamera(0, logger = Logger)
        self.cam2 = dcam.HamamatsuCamera(1, logger = Logger)

        #Set emission labels, wavelengths in  nm
        self.cam1.left_emission = 687
        self.cam1.right_emission = 558
        self.cam2.left_emission = 610
        self.cam2.right_emission = 740

        # Initialize camera 1
        print('Initializing camera 1...')
        self.cam1.setPropertyValue("exposure_time", 40.0)
        self.cam1.setPropertyValue("binning", 1)
        self.cam1.setPropertyValue("sensor_mode", 4) #1=AREA, 2=LINE, 4=TDI, 6=PARTIAL AREA
        self.cam1.setPropertyValue("trigger_mode", 1) #Normal
        self.cam1.setPropertyValue("trigger_polarity", 1) #Negative
        self.cam1.setPropertyValue("trigger_connector", 1) #Interface
        self.cam1.setPropertyValue("trigger_source", 2) #1 = internal, 2=external
        self.cam1.setPropertyValue("contrast_gain", 0)
        self.cam1.setPropertyValue("subarray_mode", 1) #1 = OFF, 2 = ON

        self.cam1.captureSetup()
        self.cam1.get_status()

        # Initialize Camera 2
        print('Initializing camera 2...')
        self.cam2.setPropertyValue("exposure_time", 40.0)
        self.cam2.setPropertyValue("binning", 1)
        self.cam2.setPropertyValue("sensor_mode", 4) #1=AREA, 2=LINE, 4=TDI, 6=PARTIAL AREA
        self.cam2.setPropertyValue("trigger_mode", 1) #Normal
        self.cam2.setPropertyValue("trigger_polarity", 1) #Negative
        self.cam2.setPropertyValue("trigger_connector", 1) #Interface
        self.cam2.setPropertyValue("trigger_source", 2) #1 = internal, 2=external
        self.cam2.setPropertyValue("contrast_gain", 0)

        self.cam2.captureSetup()
        self.cam2.get_status()

    def initializeInstruments(self):
        ########################################
        ### Set Stage, Lasers, & Optics ########
        ########################################

        #Initialize X Stage before Y Stage!
        print('Initializing X & Y stages')
        self.x.initialize()
        #TODO, make sure x stage is in correct place.
        self.x.initialize() # Do it twice to make sure its centered!
        self.y.initialize()
        print('Initializing lasers')
        self.l1.initialize()
        self.l2.initialize()
        print('Initializing pumps and valves')
        self.p['A'].initialize()
        self.p['B'].initialize()
        self.v10['A'].initialize()
        self.v10['B'].initialize()
        self.v24['A'].initialize()
        self.v24['B'].initialize()
        print('Initializing FPGA')
        self.f.initialize()

        # Initialize Z, objective stage, and optics after FPGA
        print('Initializing optics and Z stages')
        self.z.initialize()
        self.obj.initialize()
        self.optics.initialize()

        #Sync TDI encoder with YStage
        print('Syncing Y stage')
        while not self.y.check_position():
            time.sleep(1)
        self.y.position = self.y.read_position()
        self.f.write_position(0)

        self.distance = self.get_distance()
        print('HiSeq initialized!')

    # Write metadata file
    def write_metadata(self, n_frames, bundle, image_name):

        date = time.strftime('%Y%m%d_%H%M%S')
        meta_path = join(self.image_path, 'meta_'+image_name+'.txt')
        meta_f = open(image_path, 'w+')
        meta_f.write('time ' + date + '\n' +
                     'y ' + str(self.y.position) + '\n' +
                     'x ' + str(self.x.position) + '\n' +
                     'z ' + str(self.z.position) + '\n' +
                     'obj ' + str(self.obj.position) + '\n' +
                     'frames ' + str(n_frames) + '\n' +
                     'bundle ' + str(bundle) + '\n' +
                     'TDIY ' + str(self.f.read_position()) +  '\n' +
                     'laser1 ' + str(self.l1.get_power()) + '\n' +
                     'laser2 ' + str(self.l2.get_power()) + '\n' +
                     'ex filters ' + str(self.optics.ex) + '\n' +
                     'em filter in ' + str(self.optics.em_in)
                     )
                     #TODO write number of frames actually take

        return meta_f

    def take_picture(self, n_frames, bundle, image_name = None):

        y = self.y
        x = self.x
        obj = self.obj
        f = self.f
        l1 = self.l1
        l2 = self.l2
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

        #
        # Setup Cameras
        #
        # Make sure cameras are ready (status = 3)
        while cam1.get_status() != 3:
            cam1.stopAcquisition()
            cam1.freeFrames()
        while cam2.get_status() != 3:
            cam2.stopAcquisition()
            cam2.freeFrames()
        # Set bundle height
        cam1.setPropertyValue("sensor_mode_line_bundle_height", bundle)
        cam2.setPropertyValue("sensor_mode_line_bundle_height", bundle)
        cam1.captureSetup()
        cam2.captureSetup()
        # Allocate memory for image data
        cam1.allocFrame(n_frames)
        cam2.allocFrame(n_frames)


        #
        #Arm stage triggers
        #
        #TODO check trigger y values are reasonable
        n_triggers = n_frames * bundle
        end_y_pos = y_pos - n_triggers*75
        f.TDIYPOS(y_pos)
        f.TDIYARM3(n_triggers, y_pos)
        #print('Trigger armed, Imaging starting')



        meta_f = self.write_metadata(n_frames, bundle, image_name)

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

        # Stop Cameras
        cam1.stopAcquisition()
        cam2.stopAcquisition()
        # Close laser shutter
        f.command('SWLSRSHUT 0')
        # Check if all frames were taken from camera 1 then save images
        if cam1.getFrameCount() != n_frames:
            print('Cam1 image not taken')
            image_complete = False
        else:
            cam1.saveImage(image_name, self.image_path)
            image_complete = True
        # Check if all frames were taken from camera 2 then save images
        if cam2.getFrameCount() != n_frames:
            print('Cam2 image not taken')
            image_complete = False
        else:
            cam2.saveImage(image_name, self.image_path)
            image_complete = True
        # Print out info pulses = triggers, not sure with CLINES is
        if image_complete:
            response = f.command('TDICLINES')
            response = f.command('TDIPULSES')
        # Free up frames/memory
        cam1.freeFrames()
        cam2.freeFrames()

        # Reset gains & velocity for ystage
        y.command('GAINS(5,10,7,1.5,0)')
        y.command('V1')

        meta_f.close()

        return image_complete

#############################################################
### Take picture with TDIscan ###############################
#############################################################

    def TDIscan(self, n_frames, n_exposures, y_pos = None):
        ################################
        ### ARM STAGE and FPGA  ########
        ##################################
        y = self.y
        x = self.x
        f = self.f
        l1 = self.l1
        l2 = self.l2

        if y_pos == None:
            y_pos = y.position


        n_triggers = n_frames * n_exposures

        y.move(y_pos)
        response = f.command('TDIYERD')
        print(response)
        response = y.command('GAINS(5,10,5,2,0)')
        response = y.command('V0.15400')
        response = f.command('TDIYPOS ' + str(y_pos-100000))
        print(response)
        response = f.command('TDIYARM3 ' +
                             str(n_triggers) +
                             ' ' +
                             str(y_pos-500000) +
                             ' 1')
        print(response)
        response = f.command('TDICLINES')
        print(response)
        response = f.command('TDIPULSES')
        print(response)
        #response = f.command('TDIYWAIT')
        #print(response)
        l1.command('ON')
        l2.command('ON')
        l1.command('POWER=10')
        l2.command('POWER=10')

        # Start Imaging on TDIscan
        print('In TDIscan set sensor mode to TDI')
        print('In TDIscan set repeat to bundle number')
        print('In TDIscan set frames to frame number')
        print('Get ready to hit enter quickly on command line...')
        print('After hitting Sequential in TDIscan to start imaging')
        # Set frames to number of frames
        input('Press enter to start cameras')
        # Move emission filter out of path
        f.command('EM2O')
        # Move excitation filter wheel
        f.command('SWLSRSHUT 1')
        f.command('EX1MV 71')
        f.command('EX2MV 71')
        y.move(0)

        ################################
        ### Read Out ###################
        ################################


        response = f.command('TDICLINES')
        print(response)
        response = f.command('TDIPULSES')
        print(response)

        # Reset gains for ystage
        y.command('GAINS(5,10,7,1.5,0)')
        y.command('V1')

        # Move emission filter out of path
        f.command('EM2I')
        f.command('EX1HM2')
        f.command('EX2HM2')
        f.command('SWLSRSHUT 0')


##########################################
#### AUTOFOCUS WORK IN PROGRESS ##########
##########################################
##
##    def zStackAutoFocus(self, n_frames):
##        f = self.f
##        obj = self.obj
##        z = self.z
##        cam1 = self.cam1
##        cam2 = self.cam2
##
##        ##########################
##        ## Setup Cameras #########
##        ##########################
##
##        #Switch Camera to Area mode
##        cam1.setPropertyValue("sensor_mode", 1) #1=AREA, 2=LINE, 4=TDI, 6=PARTIAL AREA
##        cam2.setPropertyValue("sensor_mode", 1) #1=AREA, 2=LINE, 4=TDI, 6=PARTIAL AREA
##        #Set line bundle height to 8
##        cam1.setPropertyValue("sensor_mode_line_bundle_height", 8)
##        cam2.setPropertyValue("sensor_mode_line_bundle_height", 8)
##
##        cam1.captureSetup()
##        cam2.captureSetup()
##
##        cam1.allocFrame(n_frames)
##        cam2.allocFrame(n_frames)
##
##        cam1.status()
##        cam2.status()
##
##        # Position zstage
##        z.move([21061 21061 21061])
##
##        # Position objective stage
##        obj.set_velocity(5) # mm/s
##        start_position = 60292
##        obj.move(start_position)
##
##
##        f.command('SWYZ_POS 1')
##
##
##        # Set up objective to move and trigger
##        obj.set_velocity(0.42) #mm/s
##        obj.command('ZTRG ' + str(start_position))
##        obj.command('ZYT 0 3')
##
##        # Start Camera
##        cam1.startAcquisition()
##        cam2.startAcquisition()
##
##        # Move stage
##        stop_position = 2621
##        obj.command('ZMV ' + str(stop_position))
##        start_time = time.time()
##        while obj.check_position() != stop_position:
##            now = time.time()
##            if now - start_time > 10:
##                print('Objective stage took too long to move')
##                break
##
##        # Save Images and reset
##        cam1.stopAcquisition()
##        date = time.strftime('%m-%d-%Y')
##        cam1.saveImage('Z_AF'+'cam1_8exp_' +
##                                              str(n_frames) + 'f_' +
##                                              date)
##
##
##        cam2.stopAcquisition()
##        cam2.saveImage('Z_AF'+'cam2_8exp_' +
##                                              str(n_frames) + 'f_' +
##                                              date)
##
##        cam1.freeFrames()
##        cam2.freeFrames()
##
##        #Switch Camera back to TDI
##        cam1.setPropertyValue("sensor_mode", 4) #1=AREA, 2=LINE, 4=TDI, 6=PARTIAL AREA
##        cam2.setPropertyValue("sensor_mode", 4) #1=AREA, 2=LINE, 4=TDI, 6=PARTIAL AREA
##
##        cam1.captureSetup()
##        cam2.captureSetup()
    def reset_stage(self):
        print('Resetting FPGA and syncing with stage')
        self.y.move(self.y.home)
        self.f.initialize()
        self.f.write_position(self.y.home)

    def move_stage_out(self):
        self.x.move(self.x.home)
        self.y.move(self.y.min_y)

    def zstack(self, obj_start, obj_stop, obj_step, n_frames, y_pos):
        exp_date = time.strftime('%Y%m%d_%H%M%S')

        for obj_pos in range(obj_start, obj_stop+1, obj_step):
            image_name = exp_date + '_' + str(obj_pos)
            image_complete = False
            self.y.move(y_pos)
            while not image_complete:
                image_complete = self.take_picture(n_frames, bundle, image_name)
                if not image_complete:
                    print('Image not take')
                    # Reset stage and FPGA
                    self.reset_stage()
                    self.y.move(y_pos)

    def scan(self, x_pos, y_pos, obj_start, obj_stop, obj_step, n_scans, n_frames, image_name=None):

        if image_name is None:
            image_name = time.strftime('%Y%m%d_%H%M%S')

        start = time.time()
        self.y.move(y_pos)
        for n in range(n_scans):
            self.x.move(x_pos)
            for obj_pos in range(obj_start, obj_stop+1, obj_step):
                self.obj.move(obj_pos)
                f_img_name = image_name + '_x' + str(x_pos) + '_o' + str(obj_pos)
                image_complete = False

                while not image_complete:
                    image_complete = self.take_picture(n_frames, 128, f_img_name)
                    self.y.move(y_pos)
                    if not image_complete:
                        print('Image not taken')
                        self.reset_stage()
                        self.y.move(y_pos)

            x_pos = self.x.position + 315

        stop = time.time()

        return stop - start

##    def autofocus(self):
##        # move stage to initial distance
##        z_pos = 20600
##        obj_pos = 30000
##        self.z.move([z_pos, z_pos, z_pos])
##        self.obj.move(obj_pos)
##
##        #Initialize parameters
##        DC = []                                                             # list of distance [0] and contrast [1]
##        n_moves = 0                                                          # of focus movements
##        dD = 100                                                            # delta distance
##        y_pos = self.y.position
##
##        #Initial image
##        D = self.get_distance()                                             # Calculate distance
##        image_name = 'AF_'+str(n_moves)
##        image_complete = False
##        while not image_complete:
##            image_complete = self.take_picture(32, 128, image_name)         # take picture
##        self.y.move(y_pos)                                                  # reset stage
##        C = self.contrast(image_name)                                       # calculate contrast
##        DC.append([D,C])
##
##        # Move stage for next step
##        z_pos = int(z_pos + dD*4)
##        self.z.move([z_pos, z_pos, z_pos])
##
##
##        # Take image, and move stage according to change in contrast to dial in focus
##        while abs(dD) > 1 and n_moves < 30:
##
##            n_moves = n_moves + 1                                           # increase move counter
##            print('Auto Focus step ' + str(n_moves))
##
##            image_name = 'AF_'+str(n_moves)
##            image_complete = False
##            while not image_complete:
##                image_complete = self.take_picture(32, 128, image_name)     # take picture
##            self.y.move(y_pos)                                              # reset stage
##            C = self.contrast(image_name)                                   # Calculate contrast
##            D = self.get_distance()                                         # Calculate distance
##            DC.append([D,C])
##            dC = DC[-1][1] - DC[-2][1]                                      # change in contrast
##            print('Change in contrast is ' + str(dC))
##            if dC < 0:                                                      # update delta distance
##                dD = -(DC[-1][0] - DC[-2][0])*0.6
##
##            if abs(dD) > 25:                                                # move z-stage
##                z_pos = int(sum(self.z.position)/len(self.z.position) - dD*4)
##                self.z.move([z_pos, z_pos, z_pos])
##            else:                                                           # move objective
##                obj_pos = int(self.obj.position - dD*262)
##                self.obj.move(obj_pos)
##
##            D = self.get_distance()                                         # Calculate distance
##            print('Changed distance by ' + str(D) + ' microns')
##
##        if n_move >= 30:
##            print('Did not find optimal focal plane')
##            return False, DC
##        else:
##            return True, DC

    # Get distance between object and stage in microns
    def get_distance(self):
        z_distance = (sum(self.z.position)/len(self.z.position))/self.z.spum
        obj_distance = self.obj.position/self.obj.spum
        self.distance = self.max_distance - obj_distance - z_distance + self.distance_offset

        return self.distance


    def twoscan(hs, n):
        for i in range(n):
            hs.take_picture(50, 128, y_pos = 6500000, x_pos = 11900, obj_pos = 45000)
            hs.take_picture(50, 128, y_pos = 6250000, x_pos = 11500, obj_pos = 45000)

##    def rough_focus(self):
##        self.obj.move(30000)
##        y_pos = self.y.position
##
##        exp_date = time.strftime('%Y%m%d_%H%M%S')
##
##        for z_pos in range(18000, 22000, 200):
##            self.z.move([z_pos, z_pos, z_pos])
##            image_name = exp_date + '_' + str(z_pos)
##            image_complete = False
##            self.y.move(y_pos)
##            while not image_complete:
##                image_complete = self.take_picture(32, 128, image_name)
##                if not image_complete:
##                    print('Image not taken... Restarting FPGA')
##                    # Reset stage and FPGA
##                    self.reset_state()
##                    self.y.move(y_pos)
    def jpeg(self, filename):
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

    # def contrast(self, filename):
    #     image_prefix=[self.cam1.left_emission,
    #                   self.cam1.right_emission,
    #                   self.cam2.left_emission,
    #                   self.cam2.right_emission]
    #     C = 0
    #     for image in image_prefix:
    #         im_path = join(self.image_path, str(image)+'_'+filename+'.tiff')
    #         im = imageio.imread(im_path)                                        #read picture
    #         im = im[64:,:]                                                      #Remove bright band artifact
    #         bg = np.loadtxt(self.bg_path+str(image)+'background.txt')           #Open background file for sensor
    #         im = im - bg                                                        #Subtract background
    #         im[im <0] = 0                                                       #Make negative values 0
    #         max_px = np.amax(im)                                                #Find max pixel value
    #
    #         # weight max pixels if they are saturated
    #         if max_px >= 4096 - np.max(bg):
    #             n_max_px = np.sum(im >= max_px)                                     #count number of pixels with max value
    #         else:
    #             n_max_px = 1
    #
    #         C = C + (max_px*n_max_px - np.amin(im))                                 #Calculate contrast
    #
    #     return C


    def rough_focus(self, z_start = 21200, z_interval = 200, n_images = 6):

        # move stage to initial distance
        y_pos = self.y.position
        obj_pos = 17500
        self.obj.move(obj_pos)
        z_pos = z_start
        self.z.move([z_pos, z_pos, z_pos])

        Z = []                                                             # list of distance [0] and contrast [1]
        C = []                                                              # list of contrasts
        for i in range(n_images):
            image_name = 'AF_rough_'+str(i)
            image_complete = False
            while not image_complete:
                image_complete = self.take_picture(32, 128, image_name)         # take picture
                self.y.move(y_pos)                                              # reset stage


            C.append(self.jpeg(image_name))                                 # calculate compression
            Z.append(z_pos)

            # Move stage for next step
            z_pos = int(z_pos + z_interval)
            self.z.move([z_pos, z_pos, z_pos])

        # find best z stage position
        self.message('Compressions: ' + str(C))
        self.message('Z pos: ' + str(Z))
        z_opt = find_focus(Z, C)
        self.message('Optimal Z pos = ' + str(z_opt))
        if z_opt is None:
            self.message('Could not find rough focus')
            z_opt = 21500
        # move z stage to optimal contrast position
        self.z.move([z_opt, z_opt, z_opt])

        return Z,C

    def fine_focus(self, obj_start = 10000, obj_interval = 5000, n_images = 7):

        #Initialize
        y_pos = self.y.position
        obj_pos = obj_start
        self.obj.move(obj_pos)

        # Sweep across objective positions
        Z = []                                                              # list of distance
        C = []                                                              # list of contrasts
        for i in range(n_images):
            image_name = 'AF_fine_'+str(i)
            image_complete = False
            while not image_complete:
                image_complete = self.take_picture(32, 128, image_name)         # take picture
                self.y.move(y_pos)                                              # reset stage


            C.append(self.jpeg(image_name))                                      # calculate compression
            Z.append(self.obj.position)

            # Move stage for next step
            obj_pos = int(obj_pos + obj_interval)
            self.obj.move(obj_pos)

        # find best obj stage position
        obj_pos = find_focus(Z, C)
        self.message('Compressions: ' + str(C))
        self.message('OBJ pos: ' + str(Z))
        self.message('Optimal OBJ pos = ' + str(obj_pos))

        if obj_pos is None:
            self.message('Could not find fine focus')
            i = np.argmax(C)
            obj_pos = Z[i]
            if i != 0 and i != len(C)-1:
                if C[i-1] > C[i+1]:
                    obj_pos -= int(obj_interval/2)
                else:
                    obj_pos += int(obj_interval/2)
            elif i == 0:
                obj_pos += int(obj_interval/2)
            elif i == len(C)-1:
                obj_pos -= int(obj_interval/2)

        # Home in on objective position for optimal contrast
        while abs(obj_pos-self.obj.position) >= self.obj.spum/2:
            self.message('Moving objective by ' + str((obj_pos-self.obj.position)/self.obj.spum) +  ' microns')
            self.obj.move(obj_pos)                                                # move objective
            i = i + 1
            image_name = 'AF_fine_'+str(i)
            image_complete = False
            while not image_complete:
                image_complete = self.take_picture(32, 128, image_name)           # take picture

            self.y.move(y_pos)                                                    # reset stage

            C.append(self.jpeg(image_name))                                   # calculate contrast
            Z.append(self.obj.position)

            obj_pos = find_focus(Z, C)                                              #find best obj stage position
            self.message('Contrasts: ' + str(C))
            self.message('OBJ pos: ' + str(Z))
            self.message('Optimal OBJ pos = ' + str(obj_pos))


            if obj_pos is None:
                self.message('Could not find fine focus')
                obj_pos = Z[np.argmax(C)]

        self.message('Contrasts: ' + str(C))
        self.message('OBJ pos: ' + str(Z))
        self.message('Optimal OBJ pos = ' + str(obj_pos))

        return Z, C

    def position(self, AorB, box):
        LLx = box[0]; LLy = box[1]; URx = box[2]; URy = box[3]

        # Number of scans
        n_scans = ceil((LLx - URx)/self.scan_width)

        # X center of scan
        x_center = self.fc_origin[AorB][0]
        x_center -= LLx*1000*self.x.spum
        x_center += (LLx-URx)*1000/2*self.x.spum
        x_center = int(x_center)

        # initial X of scan
        x_initial = x_center - n_scans*self.scan_width*1000*self.x.spum/2
        x_initial = int(x_initial)

        # initial Y of scan
        y_initial = int(self.fc_origin[AorB][1] + LLy*1000*self.y.spum)

        # Y center of scan
        y_length = (LLy - URy)*1000
        y_center = y_initial - y_length/2*self.y.spum
        y_center = int(y_center)

        # Number of frames
        n_frames = y_length/self.bundle_height/self.resolution
        n_frames = ceil(n_frames + 10)

        # Adjust x and y center so focus will image (32 frames, 128 bundle) in center of section
        x_center -= int(self.scan_width*1000*self.x.spum/2)
        y_center += int(32*128/2*self.resolution*self.y.spum)


        return [x_center, y_center, x_initial, y_initial, n_scans, n_frames]

    def optimize_filter(self, nframes, nbins = 256, sat_threshold = 0.0005, signal_threshold = 20):
        # Save y position
        y_pos = self.y.position

        image_prefix=[self.cam1.left_emission,                                      # 687
                      self.cam1.right_emission,                                     # 558
                      self.cam2.left_emission,                                      # 610
                      self.cam2.right_emission]                                     # 740

        f_order = [[4.0, 2.0, 1.6, 1.4, 0.6, 0.2],[4.5, 3.0, 2.0, 1.0, 0.9, 0.2]]   # Order of filters to loop through
        opt_filter = [None, None]                                                   # Empty list of optimal filters
        lasers = [1,2]                                                              # list of lasers, Green = 1, Red = 2

        for li in lasers:
            fi = 0                                                                  # Loop through filters until optimized
            self.optics.move_ex(1, 'home')                                          # Home and block lasers
            self.optics.move_ex(2, 'home')
            self.optics.move_em_in(True)
            new_f_score = 0
            while opt_filter[li-1] is None:
                self.optics.move_ex(li,f_order[li-1][fi])                           # Move excitation filter
                image_name = 'L'+str(li)+'_filter'+str(f_order[li-1][fi])
                image_complete = False
                while not image_complete:
                    image_complete = self.take_picture(nframes, 128, image_name)         # Take Picture
                    self.y.move(y_pos)

                contrast = np.array([])
                saturation = np.array([])
                for image in image_prefix:
                    im_name = self.image_path+str(image)+'_'+image_name+'.tiff'    #read picture

                    C, S = contrast2(im_name, image, self.bg_path, nbins)            #Calculate background subtracted contrast and %saturatio

                    contrast = np.append(contrast,C)
                    saturation = np.append(saturation,S)

                old_f_score = new_f_score
                if any(saturation > sat_threshold):                                         # make sure image is not too saturated
                    new_f_score = 0
                elif li == 1:
                    signal = contrast[1] + contrast[2]
                    new_f_score = contrast[1] + contrast[2] - contrast[0] - contrast[3]             # Green laser, 558+610-687-740 emissions
                elif li == 2:
                    signal = contrast[0] + contrast[3]
                    new_f_score = contrast[0] + contrast[3] - contrast[1] - contrast[2]             # Red laser, 687+740-558-610- emissions

                if signal > signal_threshold:                                                       # Increase laser until you at least see some signal
                    if old_f_score >= new_f_score and fi > 0:                                       # Too much crosstalk / saturation
                        opt_filter[li-1] = f_order[li-1][fi-1]
                    else:
                        fi += 1
                elif fi == 5:                                                               # Last filter
                    opt_filter[li-1] = f_order[li-1][fi]
                else:                                                                       # Move to next filter
                    fi += 1
                self.message('Laser : ' + str(li))
                self.message('Filter: ' + str(f_order[li-1][fi]))
                self.message('Contrast: ' + str(contrast))
                self.message('Saturation: ' + str(saturation))
                self.message('Filter Score ' + str(new_f_score))

        return opt_filter

    def message(self, text):
        if self.logger is None:
            print(str(text))
        else:
            self.logger.info(str(text))

# Fit Contrast(Z) to gaussian curve and find Z that maximizes C
# C = Y
# Z = X
def find_focus(X, Y):
    Ynorm = Y - np.min(Y)
    Ynorm = Ynorm/np.max(Y)

    amp = 1
    cen = X[np.argmax(Y)]
    sigma = 1

    # Optimize amplitude, center, std

    # Calculate error
    try:
        popt_gauss, pcov_gauss = curve_fit(_1gaussian, X, Ynorm, p0 = [amp, cen, sigma])
        #perr_gauss = np.sqrt(np.diag(pcov_gauss))
        center  = int(popt_guass[1])
        #error  = perr_gauss[1]
    except:
        center = None
        #perr_gauss = None

    # Return center and error
    return center


# Find mean intensity of signal and % of pixels saturated
def signal_and_saturation(img, nbins = 256):

    # Histogram of image
    hist, bin_edges = np.histogram(img, bins = nbins, range = (0,4095), density = True)

    # Find background signal intensity
    amp1 = 0.5
    cen1 = 100
    sigma1 = 10
    x_range = range(0,4095,int(4096/nbins))
    pback, pcov = curve_fit(_1gaussian, x_range[0:-1], hist[0:-1], p0=[amp1, cen1, sigma1])

    # Mask background
    masked_img = img[img > pback[1] + 6*pback[2]]

    if len(masked_img):
        # Histogram of image with background masked
        masked_hist, bin_edges = np.histogram(masked_img, bins = nbins, range = (0,4095), density = True)

        # Find signal intensity
        amp1 = 0.1
        cen1 = 2048
        sigma1 = 20
        psignal, pcov = curve_fit(_1gaussian, x_range[0:-1], masked_hist[0:-1], p0=[amp1, cen1, sigma1])
        mean_intensity = psignal[1]
    else:
        mean_intensity = 0

    return mean_intensity, hist[-1]

def contrast2(filename, channel, path, nbins=256):

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

# Gaussian function for curve fitting
def _1gaussian(x, amp1,cen1,sigma1):
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2)))
