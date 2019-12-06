#import instruments
import ystage
import xstage
import fpga
import zstage
import laser
import objstage
import optics

import time
import threading
import numpy as np



class HiSeq():

    #
    # Make HiSeq Object
    #
    def __init__(self, yCOM = 'COM10',
                       xCOM = 'COM9',
                       objCOM = None,
                       pumpACOM = None,
                       pumpBCOM = None,
                       fpgaCOM = ['COM12','COM15'],
                       laser1COM = 'COM13',
                       laser2COM = 'COM14'):
    
        self.y = ystage.Ystage(yCOM)
        self.f = fpga.FPGA(fpgaCOM[0], fpgaCOM[1])
        self.x = xstage.Xstage(xCOM)
        self.l1 = laser.Laser(laser1COM)
        self.l2 = laser.Laser(laser2COM)
        self.z = zstage.Zstage(self.f.serial_port)
        self.obj = objstage.OBJstage(self.f.serial_port)
        self.optics = optics.Optics(self.f.serial_port)
        self.cam1 = None
        self.cam2 = None
        self.image_path = 'C:\\Users\\Public\\Documents\\PySeq2500\\Images\\'
    
                       
                    


    def initializeCams(self):

        import dcam

        self.cam1 = dcam.HamamatsuCamera(0)
        self.cam2 = dcam.HamamatsuCamera(1)

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
        self.x.initialize()
        #TODO, make sure x stage is in correct place. 
        self.x.initialize() # Do it twice to make sure its centered!
        self.y.initialize()
        self.l1.initialize()
        self.l2.initialize()
        self.f.initialize()

        # Initialize Z, objective stage, and optics after FPGA
        self.z.initialize()
        self.obj.initialize()
        self.optics.initialize()

        #Sync TDI encoder with YStage
        while not self.y.check_position():
            time.sleep(1)
        self.y.position = self.y.read_position()
        self.f.write_position(0)

    # Write metadata file    
    def write_metadata(self, n_frames, bundle, image_name):
                
        date = time.strftime('%Y%m%d_%H%M%S')
        meta_f = open(self.image_path + 'meta_' + image_name + '.txt', 'w+')
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
            print('Attempting to sync TDI and stage')
            f.write_position(y.position)  
        else:
            print('TDI and stage are synced')

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
        print('Trigger armed, Imaging starting')


        
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
            print(response)
            response = f.command('TDIPULSES')
            print(response)
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
                    
    def scan(self, x_start, x_stop, obj_start, obj_stop, obj_step, n_frames, y_pos):
        exp_date = time.strftime('%Y%m%d_%H%M%S')

        for x_pos in range(x_start, x_stop+1, 315):
            for obj_pos in range(obj_start, obj_stop+1, obj_step):
                image_name = exp_date + '_x' + str(x_pos) + '_o' + str(obj_pos)
                image_complete = False
                self.y.move(y_pos)
                while not image_complete:
                    image_complete = self.take_picture(n_frames, 128, image_name)
                    if not image_complete:
                        print('Image not taken')
                        self.reset_stage()
                        self.y.move(y_pos)
            
            
            

    def twoscan(hs, n):
        for i in range(n):
            hs.take_picture(50, 128, y_pos = 6500000, x_pos = 11900, obj_pos = 45000)
            hs.take_picture(50, 128, y_pos = 6250000, x_pos = 11500, obj_pos = 45000)

    def rough_focus(self):
        self.obj.move(30000)
        y_pos = self.y.position

        exp_date = time.strftime('%Y%m%d_%H%M%S')
        
        for z_pos in range(18000, 22000, 200):
            self.z.move([z_pos, z_pos, z_pos])
            image_name = exp_date + '_' + str(z_pos)
            image_complete = False
            self.y.move(y_pos)
            while not image_complete:
                image_complete = self.take_picture(32, 128, image_name)
                if not image_complete:
                    print('Image not take... Restarting FPGA')
                    # Reset stage and FPGA
                    self.reset_state()
                    self.y.move(y_pos)
                    
