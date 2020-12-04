###################
Interactive Control
###################

Initializing
============

The HiSeq instruments and cameras must first be initialized by using
:meth:`~pyseq.__init__.HiSeq.initializeInstruments` and
:meth:`~pyseq.__init__.HiSeq.initializeCams` respectively. DCAM-API from
Hamamatsu needs to be installed to use the cameras. Initializing the cameras
may take up to 2 mins. Initializing the instruments including the stages, FPGA,
pumps, valves and lasers may take up to 10 minutes.

.. code-block:: python

  import pyseq
  hs = pyseq.HiSeq()
  hs.initializeCams()
  hs.initializeInstruments()

Pumping Reagents
================
Flowcell specific :ref:`pumps<pump>` and :ref:`valves<valve>` are accessed by
keys 'A' or 'B'. Reagents are selected using the 24 port valve, **hs.v24**.
Reagents are assigned to a port on the valve with it's **port_dict**. Use
:meth:`~pyseq.valve.Valve.move` to select a reagent in the **port_dict**.
If the **port_dict** is not specified, the port index may be used instead.

.. code-block:: python

  hs.v24['A'].move(1)                             # Move to port 1
  hs.v24['A'].port_dict = {'water':1, 'PBS':2}    # Assign reagents to ports
  hs.v24['A'].move('PBS')                         # Move to PBS at port 2

Use :meth:`~pyseq.pump.Pump.pump` to pull reagents through the flowcell with
negative pressure on a flowcell lane basis. For example, with an 8 lane flowcell,
:code:`pump(250, 100)`, will pump 250 uL at 100 uL/min through each flowcell
lane for a total volume of 2000 uL pumped at a total rate of 800 uL/min. The
default number of syringe barrels dedicated to a flowcell lane is 1 and can be
changed with :meth:`~pyseq.pump.Pump.update_limits`. Use larger volumes at
slower flowrates for optimal pumping.

.. code-block:: python

  n_barrels_per_lane = 8
  hs.v24['A'].update_limits(n_barrels_per_lane)       # Tied 8 outlets together
  volume = 1000                                       # uL
  flowrate = 100                                      # uL/min
  hs.p['A'].pump(volume, flowrate)

Reagents can be pulled through either the 2 inlet row or 8 inlet row
by switching the port on the 10 port valve, **hs.v10**. Alternatively,
:meth:`~pyseq.__init__.HiSeq.move_inlet` can be used to change between the inlet
rows.

.. code-block:: python

  # Pump through 2 inlet port row
  hs.v10['A'].move(2)
  hs.v10['B'].move(4)
  # Pump through 8 inlet port row
  hs.v10['A'].move(3)
  hs.v10['B'].move(5)
  # Or change between inlet rows with move_inlet
  # Change to 2 inlet row
  hs.move_inlet(2)
  # Change to 8 inlet row
  hs.move_inlet(8)

Positioning
===========
Use :meth:`~pyseq.__init__.HiSeq.position` to return all the stage information
needed to image a section in dictionary.

.. code-block:: python

  pos_dict = hs.position('A', [15, 45, 10, 35])

Move the :ref:`ystage` to 'y_initial', the :ref:`xstage` to 'x_initial' and
the :ref:`zstage` to [21500, 21500, 21500] before imaging a section. In general,
move the :ref:`ystage` before moving the :ref:`xstage`.

.. code-block:: python

  hs.y.move(pos_dict['y_initial'])
  hs.x.move(pos_dict['x_initial'])
  hs.z.move([21500, 21500, 21500])

The in focus position of the :ref:`objective<objstage>` can be found with and
moved to with :meth:`~pyseq.__init__.HiSeq.autofocus`. The default
:ref:`Autofocus` routine set in **HiSeq.AF** is **partial once**. Other
:ref:`Autofocus` routines include **full**, **partial**, and **full once**.

.. code-block:: python

  hs.AF = 'full once'
  hs.autofocus(pos_dict)

Setup Optics
============
The :ref:`excitaton filters<optics>` need to be moved in place before taking an
image or autofocusing. In general the emission filter should always be set in
the path of the light when focusing and out of the light path when imaging in
the lowest 558 nm channel.

.. code-block:: python

  hs.optics.move_ex('green','open')
  hs.optics.move_ex('red','open')
  hs.optics.move_em_in(True)

Take Images
===========
All the following imaging commands take and save images from all 4 emission
channels. The images are saved as 16 bit TIFFs (the actual pixel depth of the
camera is only 12 bit) and stored in **hs.image_path**. The default common name
of the images is a time stamp, but can be changed by specifying a name. The
exact name of the images depend on the imaging command but are always prefixed
with the emission channel wavelength, for example c610. The size of the images
are 2048 columns of pixels and 128 (the default sensor mode line bundle height)
x **n_frames** rows of pixels. The area imaged is 0.769 mm wide. The length of
the area is .048 x **n_frames** mm, from close to the initial position of the
ystage to the back end of the ystage. The stage, objective, and filters
should be positioned before the imaging command. A text file with the
imaging settings is also saved with the same name as the image prefixed with
metadata.

Use :meth:`~pyseq.__init__.HiSeq.take_picture` to image a small area. The exact
image names will be the common name if specified or a time stamp if not,
prefixed with 'c' and the emission channel, for example
'c610_FirstHiSeqPicture.tiff'.

.. code-block:: python

  n_frames = 16
  im_name = 'FirstHiSeqPicture'
  hs.take_picture(n_frames,im_name) # 2048x2048 px images

Use :meth:`~pyseq.__init__.HiSeq.zstack` to image a tile, or an area at a series
of objective positions starting from the current object position, and increasing
further in distance from the stage. The spacing of objective positions is
controlled by the **hs.nyquist_obj** attribute. The exact image names will be
the emission channel, followed by the common name, and finally the objective
step position the images were taken at, for example
'c610_FirstHiSeqZStack_o30000.tiff'.


.. code-block:: python

  hs.nyquist_obj = 235                      # 235 obj step spacing = 0.9 um
  n_obj_planes = 10
  n_frames = 16
  im_name = 'FirstHiSeqZStack'
  hs.zstack(n_obj_planes, n_frames, im_name) # 10 obj planes


Use :meth:`~pyseq.__init__.HiSeq.scan` to image across an entire section,
or image a volume of a sample. After a tile has been imaged, the stage moves in
the x direction, the distance of which is controlled by **hs.overlap**. The
default overlap is 0 pixels. The minimum significant overlap is 4 pixels. The
exact image names will in the following order: emission channel, common name,
xstage step position, and finally objective step position, for example
'c610_FirstHiSeqScan_x10000_o30000.tiff'.

.. code-block:: python

  hs.y.move(pos_dict['y_initial'])
  hs.autofocus(pos_dict)
  hs.overlap = 0                                      # non-overlapping images
  n_obj_planes = 1
  n_frames = pos_dict['n_frames']
  n_tiles = pos_dict['n_tiles']
  im_name = 'FirstHiSeqScan'
  hs.zstack(n_tiles, n_obj_planes, n_frames, im_name) # Scan area defined in pos_dict

HiSeq
=====
.. currentmodule:: pyseq

.. automodule:: pyseq.__init__
   :members:

   .. rubric:: Classes

   .. autosummary::
