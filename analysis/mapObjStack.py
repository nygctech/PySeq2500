def obj_stack(hs, n_frames, px_step):
    step = int(px_step*hs.resolution*hs.y.spum)
    start = hs.y.position
    stop = start + n_frames*step
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
           frame_time.append([time.time()-start_time,  cam1.getFrameCount(), obj.check_position()])
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
            cam1.saveImage('y'+str(hs.y.position),self.image_path)
            image_complete = True
        # Check if all frames were taken from camera 2 then save images
        if cam2.getFrameCount() != 1:
            print('Cam2 images not taken')
            image_complete = False
        else:
            cam2.saveImage('y'+str(hs.y.position),self.image_path)
            image_complete = True

        cam1.freeFrames()
        cam2.freeFrames()

def scan(start, stop, n_frames:
    hs.y.move(start)
    step = (stop-start)/n_frames
    for pos in range(start, stop, step):
        hs.obj.move(int(pos))

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
           frame_time.append([time.time()-start_time,  cam1.getFrameCount(), obj.check_position()])
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
            cam1.saveImage('o'+str(hs.obj.position),self.image_path)
            image_complete = True
        # Check if all frames were taken from camera 2 then save images
        if cam2.getFrameCount() != 1:
            print('Cam2 images not taken')
            image_complete = False
        else:
            cam2.saveImage('o'+str(hs.obj.position),self.image_path)
            image_complete = True

        cam1.freeFrames()
        cam2.freeFrames()
