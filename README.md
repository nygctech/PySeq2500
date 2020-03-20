# PySeq2500
Control an Illumina HiSeq 2500 System Interactively

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

# Automate a method/recipe on a HiSeq2500
The following files are needed to automate a method on a HiSeq 2500 System.
 1. experiment config
 1. method config
 1. method recipe
 
Start a method on a HiSeq2500 System.
```
pyseq -c experiment config -n experiment name -o output path
```
See usage of pyseq.
```
pyseq -h, --help
```
See installed methods.
```
pyseq -l
```
See a method config and method recipe. The example here is to see the config and recipe for a method called 4i.
```
pyseq -m 4i
```
## Experiment Config
The experiment config has 4 sections.
```
[experiment]
[sections]
[valve24]
[cycles]
```
### [experiment]
experiment details (required, unless noted)
- method: name of installed method or path to method config file (string)
- cycle: number of cycles to run (integer)
- first flowcell: which flowcell to start first if running 2, optional (A or B)
```
[experiment]
method = 4i            
cycles = 2              
first flowcell = A
```
### [sections]
position of sections on flowcell (required). `section name = F: LLx, LLy, URx, URy`
- section name: name/id of section to image (string)
- F: flowcell section is on (A or B)
- LLx: lower left x coordinate of section, use slide ruler (float)
- LLy: lower left y coordinate of section, use slide ruler (float)
- URx: upper right x coordinate of section, use slide ruler (float)
- URy: upper right y coordinate of section, use slide ruler (float)
```
[sections]
section1 = A: 15.5, 45, 10.5, 35
```
### [valve24]
Specify additional ports (optional). `N = name`
- N: port number (integer)
- Name: name of reagent (string)
```
[valve24]
6 = GFAP
7 = IBA1
8 = AF547 + Cy5
```
### [cycles]
Specify cycle specific reagents (optional). `variablereagent N = name`
- variablereagent: cycle dependent reagent in recipe, specified in method config (string)
- N: cycle (integer)
- name: reagent used for variablereagent at cycle N (string)
```
[cycles]
1stab 1 = GFAP
1stab 2 = IBA1
2ndab 1 = AF547 + Cy5
2ndab 2 = AF547 + Cy5
```
## Method Config
The method config has 2 sections.
```
[method]
[valve24]
```
### [method]
The name of this section should match the name of the method.
The only required key in this section is **recipe** that has the path to the method recipe as its value.
```
[4i]
recipe = 4i_recipe.txt
```
The other keys are optional.
- **flush speed**: flowrate to flush lines with in uL/min (integer)
- **flush volume**:  volume to flush line with in uL (integer)
- **reagent speed**: flowrate to pump reagents during recipe in uL/min (integer)
- **variable reagents**: name of variable ports in recipes that are cycle dependent (string)
- **first port**: port to start recipe at on first cycle (string)
- **barrels per lane**: number of syringe barrels that are used per lane on flowcell (integer)
- **laser power**: set power of laser in mW (integer)
```
[4i]
recipe = 4i_recipe.txt
flush speed = 500
flush volume = 1000
reagent speed = 200
variable reagents = 1stab, 2ndab
first port = blocking
barrels per lane = 8
laser power = 400
```
### [valve24]
Specify method required ports (optional). `N = name`
- N: port number (integer)
- Name: name of reagent (string)
```
[valve24]
1 = PBS
2 = water
3 = elution
4 = blocking
5 = imaging
```
## Method Recipe
There are 5 basic actions to build a recipe.
1. **PORT**: *port name* (string)
>Valve switches to specified port.
```
PORT: water
```
2. **PUMP**: *pump volume in uL* (integer)
>Syringe pump draws specified volume through flowcell lane.
```
PUMP: 2000
```
3. **HOLD**: *hold time in min.* (integer)
>Recipe pauses for specified time.
```
HOLD: 10
```
4. **WAIT**: ***IMAG** or port name* (string)
>Recipe waits to continue until the other flowcell is imaging (**IMAG**) or switches to *port name*. If there is only one flowcell, **WAIT** is ignored.
```
WAIT: water
```
5. **IMAG**: *z focal planes* (integer)
>The flowcell is imaged at the specified number of z focal planes at the sections listed in the experiment config.
```
IMAG: 15
```

