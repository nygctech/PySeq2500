#######################
Interactive Quick Start
#######################

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
    [xi, yi, xc, yx, n_scans, n_frames] = hs.position['A', 15.5, 45, 10.5, 35]
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
    hs.scan(xi, yi, hs.obj.position, hs.obj.position+1, 10, n_scans, n_frames, image_name='FirstHiSeqScan')
