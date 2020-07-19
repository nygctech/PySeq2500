import time
from multiprocessing import Process
import numpy as np

def obj_stack(hs, start, stop, n_frames):
    frames_per_step = 1
    step = int((stop-start)/n_frames)
    for pos in range(start, stop, step):

        hs.obj.move(int(pos))
        cam1 = hs.cam1
        cam2 = hs.cam2
        f = hs.f

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

        #Switch Camera to Area mode
        cam1.setPropertyValue("sensor_mode", 1) #1=AREA, 2=LINE, 4=TDI, 6=PARTIAL AREA
        cam2.setPropertyValue("sensor_mode", 1) #1=AREA, 2=LINE, 4=TDI, 6=PARTIAL AREA
        #Set line bundle height to 8
        cam1.setPropertyValue("sensor_mode_line_bundle_height", 64)
        cam2.setPropertyValue("sensor_mode_line_bundle_height", 64)

        cam1.captureSetup()
        cam2.captureSetup()

        cam1.allocFrame(frames_per_step)
        cam2.allocFrame(frames_per_step)

        #hs.obj.set_focus_trigger(int(pos))

        # Open laser shutters
        f.command('SWLSRSHUT 1')

        # Start Cameras
        cam1.startAcquisition()
        cam2.startAcquisition()

        #f.command('ZCLK 1')

        # Wait for imaging
        start_time = time.time()
        while cam1.getFrameCount() + cam2.getFrameCount() != 2*frames_per_step:
           now = time.time()
           if now - start_time > 10:
               print('Imaging took too long.')
               break

        # Close laser shutters
        f.command('SWLSRSHUT 0')

        # Stop cameras
        cam1.stopAcquisition()
        cam2.stopAcquisition()

        # Check if received correct number of frames
        if cam1.getFrameCount() != frames_per_step:
            print('Cam1 images not taken')
            image_complete = False
        else:
            cam1.saveImage('o'+str(hs.obj.position),hs.image_path)
            image_complete = True
        # Check if all frames were taken from camera 2 then save images
        if cam2.getFrameCount() != frames_per_step:
            print('Cam2 images not taken')
            image_complete += False
        else:
            cam2.saveImage('o'+str(hs.obj.position),hs.image_path)
            image_complete += True

        cam1.freeFrames()
        cam2.freeFrames()

    return image_complete

def scan(hs, n_frames, px_step):
    start = hs.y.position
    step = int(px_step*hs.resolution*hs.y.spum)
    stop = int(start + n_frames*step)

    for pos in range(start, stop, step):
        hs.y.move(int(pos))

        cam1 = hs.cam1
        cam2 = hs.cam2
        f = hs.f

        # Make sure cameras are ready (status = 3)
        if cam1.sensor_mode != 'AREA':
            cam1.setAREA()
        if cam2.sensor_mode != 'AREA':
            cam2.setAREA()
        while cam1.get_status() != 3:
            cam1.stopAcquisition()
            cam1.freeFrames()
            cam1.captureSetup()
        while cam2.get_status() != 3:
            cam2.stopAcquisition()
            cam2.freeFrames()
            cam2.captureSetup()

        #Switch Camera to Area mode
        cam1.setPropertyValue("sensor_mode", 1) #1=AREA, 2=LINE, 4=TDI, 6=PARTIAL AREA
        cam2.setPropertyValue("sensor_mode", 1) #1=AREA, 2=LINE, 4=TDI, 6=PARTIAL AREA
        #Set line bundle height to 8
        cam1.setPropertyValue("sensor_mode_line_bundle_height", 8)
        cam2.setPropertyValue("sensor_mode_line_bundle_height", 8)

        cam1.captureSetup()
        cam2.captureSetup()

        cam1.allocFrame(1)
        cam2.allocFrame(1)

        # Open laser shutters
        f.command('SWLSRSHUT 1')

        # Start Cameras
        cam1.startAcquisition()
        cam2.startAcquisition()

        f.command('ZCLK 1')

        # Wait for imaging
        start_time = time.time()
        frame_time = []
        while cam1.getFrameCount() + cam2.getFrameCount() != 2*n_frames:
           now = time.time()
           if now - start_time > 10:
               print('Imaging took too long.')
               break

        # Close laser shutters
        f.command('SWLSRSHUT 0')

        # Stop cameras
        cam1.stopAcquisition()
        cam2.stopAcquisition()

        # Check if received correct number of frames
        if cam1.getFrameCount() != 1:
            print('Cam1 images not taken')
            image_complete = False
        else:
            cam1.saveImage('y'+str(hs.y.position),hs.image_path)
            image_complete = True
        # Check if all frames were taken from camera 2 then save images
        if cam2.getFrameCount() != 1:
            print('Cam2 images not taken')
            image_complete += False
        else:
            cam2.saveImage('y'+str(hs.y.position),hs.image_path)
            image_complete += True

        cam1.freeFrames()
        cam2.freeFrames()

        return image_complete

def objstack(hs, n_frames = 232, velocity = 0.1):
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

        f = hs.f
        obj = hs.obj
        z = hs.z
        cam1 = hs.cam1
        cam2 = hs.cam2

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

        #Switch Camera to Area mode
        #cam1.setPropertyValue("sensor_mode", 1) #1=AREA, 2=LINE, 4=TDI, 6=PARTIAL AREA
        #cam2.setPropertyValue("sensor_mode", 1) #1=AREA, 2=LINE, 4=TDI, 6=PARTIAL AREA
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

        text = 'ZMV ' + str(obj.focus_stop) + obj.suffix
        obj.serial_port.write(text)

        # Start Cameras
        cam1.startAcquisition()
        cam2.startAcquisition()

        obj.serial_port.flush()
        response = obj.serial_port.readline()


        # Wait for imaging
        start_time = time.time()
        while hs.obj.check_position() != obj.focus_stop:
           now = time.time()
           if now - start_time > 10:
               print('Imaging took too long.')
               break

        start_time = time.time()
        while cam1.getFrameCount() + cam2.getFrameCount() != 2*n_frames:
           now = time.time()
           if now - start_time > 10:
               print('Imaging took too long.')
               break

        # Close laser shutters
        f.command('SWLSRSHUT 0')

        # Stop cameras
        cam1.stopAcquisition()
        cam2.stopAcquisition()

        # Check if received correct number of frames
        if cam1.getFrameCount() != n_frames:
            print('Cam1 only took ' + str(cam1.getFrameCount()) + ' out of ' + str(n_frames))
            image_complete = False
        else:
            cam1_filesize = cam1.saveFocus(hs.image_path)
            image_complete = True
        # Check if all frames were taken from camera 2 then save images
        if cam2.getFrameCount() != n_frames:
            print('Cam2 images not taken')
            image_complete = False
        else:
            cam2_filesize = cam2.saveFocus(hs.image_path)
            image_complete = True

        cam1.freeFrames()
        cam2.freeFrames()

        if image_complete:
            f_filesize = np.concatenate((cam1_filesize,cam2_filesize), axis = 1)
        else:
            f_filesize = 0

        return f_filesize
