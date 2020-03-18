# PySeq2500
Control an Illumina HiSeq 2500 System

# Initializing HiSeq

```python
import pyseq

hs = pyseq.HiSeq()                  
hs.initializeCams()                
hs.initializeInstruments()          # Initialize x,y,z & objective stages. Initialize lasers and optics (filters)
```

Note that the `pyseq.HiSeq()` constructor accepts serial ports assignments in case your setup is different, i.e:

```python
hs = pyseq.HiSeq(xCOM='COM67', yCOM='COM68', fpgaCOM=['COM10', 'COM11'], laser1COM='COM12', laser2COM='COM13')
```

# Basic setup of HiSeq

```python
hs.l1.set_power(100)                #Set green laser power to 100 mW
hs.l2.set_power(100)                #Set red laser power to 100 mW

hs.y.move(-180000)                  #Move stage to top right corner of Flow Cell A
hs.x.move(17571)
hs.z.move([21250, 21250, 21250])    #Raise z stage

hs.obj.move(30000)                  #Move objective to middle-ish

hs.move_ex(1,'open')                #Move excitation filter 1 to open position
hs.move_ex(2,'open')                #Move excitation filter 2 to open position

hs.l1.get_power()                   #Get green laser power (mW i think)
hs.l2.get_power()                   #Get red laser power   (mW i think)
```

# Image acquisition

The following code takes a picture from each of the cameras, splits each image into 2, saves all 4 images, and writes a metafile. 
Images and metafile are saved in the directory set in `hs.image_path`.

```python
# Image destination path
hs.image_path = 'C:\\Users\\Public\\Documents\\PySeq2500\\Images\\'

# Take an image
hs.take_picture(32, 128) # take_picture(# frames, bundle height, image_name)
```

Names of the images are `hs.cam1.left_emission + image_name`. The name of the metafile is just `image_name`. The `image_name` 
argument is optional, if not used it defaults to a time stamp.

Currently all of the image prefixes (`camN.L/R_emission`) are set to the emission wavelength in `hs.InitializeCams()`

The images are # frames x bundle height pixels long in the y dimension and 2048 pixels in the x dimension.
Changing the # frames is probably the best way to change the length of the scan.
Only certain values are acceptable for the bundle height, I've just been using 128 as Illumina does.

The metafile contains info like time, stage position, laser power, filter settings. 

# Moving the stage

```python
# Positioning the stage
# Currently all of the stages move to absolute positions that are defined in steps
hs.y.move(Y)         # Y should be a number between -7000000 and 7500000
hs.x.move(X)         # X should be between 1000 and 50000
hs.z.move([Z, Z, Z]) # Z should be between 0 and 25000
 
hs.obj.move(O)       # O should be between 0 and 65000
```
 
The safest way to move the stage out to load slides onto it is `hs.move_stage_out()`. 

Also I would first move the stage in the y direction, into the hiseq before moving it in the x direction because there are some knobs at the front of the hiseq that the stage can run into.
 
During `hs.intializeInstruments()`, the staged is homed to **Y=0, X=30000, Z=0, and O=30000** (although there is no homing for the objective).
  
# Setting up optics
Before taking a picture, the laser power should be set, the excitation filters should be set, and the emission filter should be in the light path. 
 
## Lasers

```python
hs.l1.set_power(100) # sets laser 1 (green) to 100 mW
hs.l2.set_power(100) # sets laser 2 (red) to 100 mW
 
hs.l1.get_power() # returns the power of laser 1 and stores it in hs.l1.power
hs.l2.get_power() # returns the power of laser 2 and stores it in hs.l2.power
```

During `hs.initializeInstruments()`, both lasers are set to 10 mW
 
## Filters

During `hs.initializeInstruments()`, the excitation filters are homed to the block position and the emission filter is moved into the light path. 

```python
hs.optics.move_ex(N, filter)		#  moves the excitation filter wheel in the N (1 or 2) light path to the filter.
hs.optics.ex_dict 					# stores the positions and names of the filters in a dictionary
hs.optics.move_em_in(True/False) 	# "True" moves the emission filter into the light path, False moves it out.
```