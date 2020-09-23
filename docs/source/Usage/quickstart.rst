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
    pos = hs.position('A', [15.5, 45, 10.5, 35])
    #Move to center of section
    hs.x.move(pos['x_center'])
    12000
    hs.y.move(pos['y_center'])
    True
    #Move stage to imaging position.
    hs.z.move([21500, 21500, 21500])
    #Find focus
    hs.autofocus()
    True
    #Move green excitation filter to optical density 1.4
    hs.move_ex('green', 1.4)
    #Move red excitation filter to optical density 1.0
    hs.move_ex('red', 1.0)
    # Take a 32 frame picture, creates image for each channel 2048 x 4096 px
    hs.take_picture(32, image_name='FirstHiSeqImage')
    #Move stage to the initial image scan position and scan image at 1 obj plane
    hs.x.move(pos['x_initial'])
    10000
    hs.y.move(pos['y_initial'])
    True
    hs.scan(pos['n_scans'], 1, pos['n_frames'], image_name='FirstHiSeqScan')
