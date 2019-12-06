##import dcam
import ystage
import xstage
import fpga
import time
import threading

################################
### Set Up Camera ##############
################################

##camera = dcam.HamamatsuCamera(0)
##
##
#####Guessed properties
##camera.setPropertyValue("exposure_time", 40.0)
##camera.setPropertyValue("binning", 1)
##camera.setPropertyValue("sensor_mode", 4) #1=AREA, 2=LINE, 4=TDI, 6=PARTIAL AREA
n_exposures = 128
##camera.setPropertyValue("sensor_mode_line_bundle_height", n_exposures)
##camera.setPropertyValue("trigger_mode", 1) #Normal
##camera.setPropertyValue("trigger_polarity", 1) #Negative
##camera.setPropertyValue("trigger_connector", 1) #Interface
##camera.setPropertyValue("trigger_source", 2) #1 = internal, 2=external
##camera.setPropertyValue("contrast_gain", 0)
###From log file


##camera.setPropertyValue("subarray_mode", 1) #1 = OFF, 2 = ON
###camera.setPropertyValue("subarray_hpos", 0)
###camera.setPropertyValue("subarray_vpos", 0)
###camera.setPropertyValue("subarray_hsize", 2048)
###camera.setPropertyValue("subarray_vsize", 64)
###From Hackteria
##
###Trying random things
###camera.setPropertyValue("cc2_on_framegrabber", 2) #1 = OFF, 2=ON
###camera.setPropertyValue("primary_buffer_mode", 1) #2 = direct, 1=AUTO

##
##
###camera.setTriggerMode('TDI') #options=internal, TDI, TDI internal

##camera.captureSetup()
####camera.startSequence(1)
##camera.status()

def initialize():
    ################################
    ### Set Ystage and FPGA ########
    ################################

    y = ystage.Ystage('COM31')
    f = fpga.FPGA('COM33', 'COM34')
    x = xstage.Xstage('COM30')

    y.initialize()
    f.initialize()
    x.initialize()
    print(y.position)
    response = f.command('TDIYEWR 0')
    print(response)
    response = x.move(31250)
    print(response)
    return y, f, x

def take_picture(y, f, n_frames, n_exposures, y_pos = None):
    ################################
    ### ARM STAGE and FPGA  #1######
    ##################################

    if y_pos is None:
        y_pos = y.position
        
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
                         str(y_pos-10000) +
                         ' 1')
    print(response)
    response = f.command('TDICLINES')
    print(response)
    response = f.command('TDIPULSES')
    print(response)
    #response = f.command('TDIYWAIT')
    #print(response)


    ################################
    ### ARM Camera & Move Stage#####
    ################################

    ##def start_camera(n_frames):
    ##    camera.allocFrame(n_frames)
    ##    camera.startAcquisition()
    ##    camera.wait()

    def move_stage(y_position):
        y.move(y_position)

    ##camera_thread = threading.Thread(target = start_camera, args = (n_frames,))
    ystage_thread = threading.Thread(target = move_stage, args = (0,))

    input('Press enter to start camera')
    # Move emission filter out of path
    f.command('EM2O')
    time.sleep(1)



    ##camera_thread.start()
    ##time.sleep(1)
    ##camera.status()
    ystage_thread.start()

    ystage_thread.join()
    ##camera_thread.join()



    ##time.sleep(1)
    ##input('Press enter to move stage')

    ################################
    ### Trigger ####################
    ################################
    ##y.command('D0')
    ##y.command('G')
    #camera.status()
    #time.sleep(10)

    ################################
    ### Read Out ###################
    ################################
    ##camera.stopAcquisition()
    ##date = time.strftime('%m-%d-%Y')
    ##[frame_x, frame_y] = camera.saveImage('LED_'+
    ##                                      str(n_exposures) + 'exp_' +
    ##                                      str(n_frames) + 'f_' +
    ##                                      date)
    ##print(frame_x, frame_y)
    response = f.command('TDICLINES')
    print(response)
    response = f.command('TDIPULSES')
    print(response)
    ##camera.freeFrames()

    # Reset gains for ystage
    y.command('W(GP,5)')

    # Move emission filter out of path
    f.command('EM2I')
    time.sleep(2)


