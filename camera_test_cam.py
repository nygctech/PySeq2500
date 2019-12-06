import dcam
import ystage
import xstage
import fpga
import time
import threading
import laser

################################
### Set Up cam1 ##############
################################
def initialize():
    cam1 = dcam.HamamatsuCamera(0)
    cam2 = dcam.HamamatsuCamera(1)


    ###Guessed properties
    cam1.setPropertyValue("exposure_time", 40.0)
    cam1.setPropertyValue("binning", 1)
    cam1.setPropertyValue("sensor_mode", 4) #1=AREA, 2=LINE, 4=TDI, 6=PARTIAL AREA
    n_exposures = 128
    cam1.setPropertyValue("sensor_mode_line_bundle_height", n_exposures)
    cam1.setPropertyValue("trigger_mode", 1) #Normal
    cam1.setPropertyValue("trigger_polarity", 1) #Negative
    cam1.setPropertyValue("trigger_connector", 1) #Interface
    cam1.setPropertyValue("trigger_source", 2) #1 = internal, 2=external
    cam1.setPropertyValue("contrast_gain", 0)
    #From log file
    cam1.setPropertyValue("subarray_mode", 1) #1 = OFF, 2 = ON
    #cam1.setPropertyValue("subarray_hpos", 0)
    #cam1.setPropertyValue("subarray_vpos", 0)
    #cam1.setPropertyValue("subarray_hsize", 2048)
    #cam1.setPropertyValue("subarray_vsize", 64)
    #From Hackteria

    #Trying random things
    #cam1.setPropertyValue("cc2_on_framegrabber", 2) #1 = OFF, 2=ON
    #cam1.setPropertyValue("primary_buffer_mode", 1) #2 = direct, 1=AUTO



    #cam1.setTriggerMode('TDI') #options=internal, TDI, TDI internal

    cam1.captureSetup()
    ##cam1.startSequence(1)
    cam1.status()

    # set up camera 2
    ###Guessed properties
    cam2.setPropertyValue("exposure_time", 40.0)
    cam2.setPropertyValue("binning", 1)
    cam2.setPropertyValue("sensor_mode", 4) #1=AREA, 2=LINE, 4=TDI, 6=PARTIAL AREA
    n_exposures = 128
    cam2.setPropertyValue("sensor_mode_line_bundle_height", n_exposures)
    cam2.setPropertyValue("trigger_mode", 1) #Normal
    cam2.setPropertyValue("trigger_polarity", 1) #Negative
    cam2.setPropertyValue("trigger_connector", 1) #Interface
    cam2.setPropertyValue("trigger_source", 2) #1 = internal, 2=external
    cam2.setPropertyValue("contrast_gain", 0)
    #From log file
    cam2.setPropertyValue("subarray_mode", 1) #1 = OFF, 2 = ON
    #cam1.setPropertyValue("subarray_hpos", 0)
    #cam1.setPropertyValue("subarray_vpos", 0)
    #cam1.setPropertyValue("subarray_hsize", 2048)
    #cam1.setPropertyValue("subarray_vsize", 64)
    #From Hackteria

    #Trying random things
    #cam1.setPropertyValue("cc2_on_framegrabber", 2) #1 = OFF, 2=ON
    #cam1.setPropertyValue("primary_buffer_mode", 1) #2 = direct, 1=AUTO



    #cam1.setTriggerMode('TDI') #options=internal, TDI, TDI internal

    cam2.captureSetup()
    ##cam1.startSequence(1)
    cam2.status()


    ################################
    ### Set Ystage and FPGA ########
    ################################

    y = ystage.Ystage('COM10')
    f = fpga.FPGA('COM12', 'COM15')
    x = xstage.Xstage('COM9')
    l1 = laser.Laser('COM13')
    l2 = laser.Laser('COM14')

    y.initialize()
    x.initialize()
    l1.initialize()
    l2.initialize()
    response = f.command('TDIYEWR 0')
    print(response)

    f.initialize()
    #Home Excitation filters
    f.command('EX1HM2') 
    f.command('EX2HM2')
    

    return cam1, cam2, y, f, x

def take_picture(f, y, cam1, cam2, l1, l2, n_frames, n_exposures, y_pos = None):
    ################################
    ### ARM STAGE and FPGA  ########
    ##################################

    
    if y_pos == None:
        y_pos = y.position

    
    cam1.setPropertyValue("sensor_mode_line_bundle_height", n_exposures)
    cam2.setPropertyValue("sensor_mode_line_bundle_height", n_exposures)
    
    n_triggers = n_frames * n_exposures
    
    y.move(y_pos)
    response = f.command('TDIYERD')
    print(response)
    response = y.command('W(GP,6)')
    print(response)
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

    


    ################################
    ### ARM cam1 & Move Stage#####
    ################################

    def start_cam1(n_frames):
        cam1.allocFrame(n_frames)
        cam1.startAcquisition()
        cam1.wait()

    def start_cam2(n_frames):
        cam2.allocFrame(n_frames)
        cam2.startAcquisition()
        cam2.wait()
        
    def move_stage(y_position):
        y.move(y_position)

    cam1_thread = threading.Thread(target = start_cam1, args = (n_frames,))
    cam2_thread = threading.Thread(target = start_cam2, args = (n_frames,))
    ystage_thread = threading.Thread(target = move_stage, args = (0,))

    #input('Press enter to start cameras')
    # Move emission filter out of path
    f.command('EM2O')
    # Move excitation filter wheel
    f.command('EX1MV 71')
    f.command('EX2MV 71')
    time.sleep(1)



    cam1_thread.start()
    cam2_thread.start()
    time.sleep(1)
    cam1.status()
    cam2.status()
    ystage_thread.start()

    ystage_thread.join()
    cam1_thread.join()
    cam2_thread.join()



    ##time.sleep(1)
    ##input('Press enter to move stage')

    ################################
    ### Trigger ####################
    ################################
    ##y.command('D0')
    ##y.command('G')
    #cam1.status()
    #time.sleep(10)

    ################################
    ### Read Out ###################
    ################################
    cam1.stopAcquisition()
    date = time.strftime('%m-%d-%Y')
    [frame_x, frame_y] = cam1.saveImage('LED_'+'cam1_' + 
                                          str(n_exposures) + 'exp_' +
                                          str(n_frames) + 'f_' +
                                          date)
    print(frame_x, frame_y)

    cam2.stopAcquisition()
    [frame_x, frame_y] = cam2.saveImage('LED_'+'cam2_' + 
                                          str(n_exposures) + 'exp_' +
                                          str(n_frames) + 'f_' +
                                          date)
    print(frame_x, frame_y)
    
    response = f.command('TDICLINES')
    print(response)
    response = f.command('TDIPULSES')
    print(response)
    cam1.freeFrames()
    cam2.freeFrames()

    # Reset gains for ystage
    y.command('W(GP,5)')

    # Move emission filter out of path
    f.command('EM2I')
    f.command('EX1HM2')
    f.command('EX2HM2')
    time.sleep(2)
