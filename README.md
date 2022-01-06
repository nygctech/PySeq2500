# PySeq2500
Control an Illumina HiSeq 2500 System Interactively

[Read the Docs](https://pyseq2500.readthedocs.io/en/latest/)

# Installation

## Requirements
PySeq2500 has only been tested on PCs that were previously used to control the HiSeq2500s with the Illumina Control Software.

### PC Specs
- Windows 7 64 bit
- Dual Intel Xeon CPU 2.00 GHx
- 64 GB RAM
- Active Silicon Phoenix Camera Link Frame Grabbers (D48CL PE4)

Newer operating systems have not been tried yet because the [drivers](https://dcam-api.com/downloads/#archive) for the cameras/frame grabber are not compatible.

### Software Requirements
- Windows 7
- Python 3.7
- compiler such as Build Tools for Visual Studio 2019 (version 16.11)

## PySeq2500 Installation
```
pip install pyseq2500

pip remove qtpy

pip install qtpy==1.11.2

pip install pyqt5==5.15.4
```

## Verify Installation
```
pyseq -h

usage: pyseq [-h] [-config PATH] [-name NAME] [-output PATH] [-list]
             [-method METHOD] [-virtual] [-settings] [-ports] [-diagnostics]

optional arguments:
  -h, --help      show this help message and exit
  -config PATH    path to config file
  -name NAME      experiment name
  -output PATH    directory to save data
  -list           list installed methods
  -method METHOD  print method details
  -virtual        use virtual HiSeq
  -settings       print optional HiSeq settings
  -ports          view com ports
  -diagnostics    perform a diagnostics run
```

## Issues
PySeq2500 relies on napari for manual focusing and displaying images with the
image analysis module. However the dependencies for napari do not get installed
correctly so there are some additional pip install commands.

# HiSeq Modifications
See [Wiki on HiSeq Modification](https://github.com/nygctech/PySeq2500/wiki/PySeq2500-Wiki)

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
hs.lasers['green'].set_power(100)   #Set green laser power to 100 mW
hs.lasers['red'].set_power(100)     #Set red laser power to 100 mW

hs.y.move(-180000)                  #Move stage to top right corner of Flow Cell A
hs.x.move(17571)
hs.z.move([21250, 21250, 21250])    #Raise z stage

hs.obj.move(30000)                  #Move objective to middle-ish

hs.move_ex('green','open')                #Move excitation filter 1 to open position
hs.move_ex('red','open')                #Move excitation filter 2 to open position

hs.lasers['green'].get_power()      #Get green laser power (mW i think)
hs.lasers['red'].get_power()        #Get red laser power   (mW i think)
```

# Image acquisition

The following code takes a picture from each of the cameras, splits each image into 2, saves all 4 images as tiffs, and writes a metadata textfile.
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

The images are `# frames` x `bundle height` rows of pixels (length of scan) and 2048 columns of pixels.
Changing the `# frames` is the best way to change the length of the scan.
Only certain values are acceptable for the bundle height, the default, which Illumina uses, is 128.

The metadata textfile contains info like time, stage position, laser power, filter settings.

# Moving the stage

```python
# Positioning the stage
# Currently all of the stages move to absolute positions that are defined in steps
hs.y.move(Y)         # Y should be between -7000000 and 7500000
hs.x.move(X)         # X should be between 1000 and 50000
hs.z.move([Z, Z, Z]) # Z should be between 0 and 25000

hs.obj.move(31000)   # Objective should be between 0 and 65000
```

To move the stage out to load slides onto it is `hs.move_stage_out()`.

Generally, when moving the stage, position the stage in the y direction first, into the hiseq, and then position it in the x direction because there are some knobs at the front of the hiseq that the stage can run into.

During `hs.intializeInstruments()`, the staged is homed to **Y=0, X=30000, Z=0, and O=30000** (although there is no homing for the objective).

# Setting up optics
Before taking a picture, the laser power should be set, the excitation filters should be set, and the emission filter should be in the light path.

## Lasers

```python
hs.lasers['green'].set_power(100)  # sets laser 1 (green) to 100 mW
hs.lasers['red'].set_power(100)    # sets laser 2 (red) to 100 mW

hs.lasers['green'].get_power()     # returns the power of laser 1 and stores it in hs.lasers['green'].power
hs.lasers['red'].get_power()       # returns the power of laser 2 and stores it in hs.lasers['red'].power
```

During `hs.initializeInstruments()`, both lasers are set to 10 mW

## Filters

During `hs.initializeInstruments()`, the excitation filters are homed to the block position and the emission filter is moved into the light path.

```python
hs.optics.move_ex(color, filter)		 #  moves the excitation filter wheel in the color ('green' or 'red') light path to the filter.
hs.optics.ex_dict 					      # stores the positions and names of the filters in a dictionary
hs.optics.move_em_in(True/False) 	# "True" moves the emission filter into the light path, False moves it out.
```

# Automate a method/recipe on a HiSeq2500
The following files are necessary to automate a method on a HiSeq 2500 System.
 1. experiment config
 1. method recipe
 1. method config (optional)

## Experiment Config
The experiment config has 4 sections.
```
[experiment]
[sections]
[reagents]
[cycles]
[filters]
[method] #optional
```

### [experiment]
experiment details (required, unless noted)
- method: name of installed method, path to method config file, or section name in config file
- cycle: number of cycles to run (integer)
- first flowcell: which flowcell to start first if running 2, optional (A or B)
```
[experiment]
method = 4i            
cycles = 2              
first flowcell = A
```

### [sections]
position of sections on flowcell as measured with slide ruler (required for imaging).

`section name = F: LLx, LLy, URx, URy`

- section name: unique name/id of section to image (string)
- F: flowcell section is on (A or B)
- LLx: lower left x coordinate of section, use slide ruler (float)
- LLy: lower left y coordinate of section, use slide ruler (float)
- URx: upper right x coordinate of section, use slide ruler (float)
- URy: upper right y coordinate of section, use slide ruler (float)
```
[sections]
section1 = A: 15.5, 45, 10.5, 35
```

### [reagents]
Specify ports (optional).
It is possible to also to specify ports in a seperate method config file.

`N = name`

- N: port number (integer)
- Name: name of reagent (string)
```
[reagents]
6 = GFAP
7 = IBA1
8 = AF547 + Cy5
```

### [cycles]
Specify cycle specific reagents (optional).

`variablereagent N = name`

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

### [filters]
Specify cycle specific optical filters (optional).
If a filter is not specified for a cycle, **default focus filter 1** is used for the green laser and **default focus filter 2** is used for the red laser.

`lasercolor N = name`

The HiSeq uses neutral density filters of various optical densities to reduce the intensity of light.
The `open` filter allows the laser to pass without reduction.
The `'home` filter completely blocks the laser.
- lasercolor: color of laser line
- N: cycle (integer)
- name: optical density of filter to use (float/string), see table below.

laser color | filters (Optical Density)
-----------:|  ----------------------------------------
green (g/G) | open, 0.2, 0.6, 1.4, 1.6, 2.0, 4.0, home
red (r,R)   | open, 0.2, 0.9, 1.0, 2.0, 3.0, 4.5, home


```
[filters]
green 1 = 1.6
g 2 = 1.4
G 3 = 0.6
red 1 = 1.0
r 1 = 0.9
R 1 = 2.0
```

### [`method name`]
Method specific HiSeq settings (optional)  
Must match the `method` item in the **[experiment]** section.
Instead of a method specific section in the experiment config file, a seperate method config file may be used.

See **Method Config** below for details.

## Method Config
The method config file has 2 possible sections.
Instead of a seperate method config file, a method specific section in the experiment config file may be used (in which case only use 1 **[reagent]** section).
```
['method name'] #required
[reagents] #optional,
```
### [`method name`]
HiSeq settings specific to the method.
The name of this section should match the name of the method.
The only required key in this section is **recipe** that has the path to the method recipe as its value.
```
[4i]
recipe = 4i_recipe.txt
```
The other keys are optional.
- **flush speed**: flowrate to flush lines with in uL/min (integer), default is 700
- **flush volume**:  volume to flush line with in uL (integer), default is 2000
- **reagent speed**: flowrate to pump reagents during recipe in uL/min (integer), default is 40
- **variable reagents**: name of variable ports in recipes that are cycle dependent (string)
- **first port**: port to start recipe at on first cycle (string)
- **barrels per lane**: number of syringe barrels that are used per lane on flowcell (integer), default is 8
- **laser power**: set power of laser in mW (integer), default is 10
- **z position**: step of tilt motors when imaging (integer), default is 21500
- **focus filter 1**: filter for green laser for autofocus routine, default is 2.0
- **focus filter 2**: filter for red laser for autofocus routine, default is 2.0
- **default em filter**: emission filter used for imaging, True for in path, False for out of path (bool), default is True
- **default filter 1**: filter for green laser if not specified in **[filter]** section of experiment config file (float/string), default is `home`
- **default filter 2**: filter for red laser if not specified in **[filter]** section of experiment config file (float/string), default is `home`
- **rinse**: reagent to rinse the flowcell with between completion of the experiment and flushing of the lines during shutdown (string), default is `None`
- **autofocus**: routine used for autofocusing (string), see **Autofocus** for more info, default is `partial once`
- **bundle height:** sensor bundle height of cameras (integer), only certain values are valid, default is 128
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

### [reagents]
Specify method required ports (optional).

`N = name`

Useful if running the same method repeatedly and only some of the ports change from each experiment.
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
There are 6 basic actions to build a recipe.
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
4. **WAIT**: **IMAG** or port name* (string)
>Recipe waits to continue until the other flowcell is imaging (**IMAG**) or switches to *port name*. If there is only one flowcell, **WAIT** is ignored.
```
WAIT: water
```
5. **IMAG**: *z focal planes* (integer)
>The flowcell is imaged at the specified number of z focal planes at the sections listed in the experiment config.
```
IMAG: 15
```
6. **TEMP**: *temperature in degrees Celsius* (integer/float)
>The temperature of the stage is change to the specified temperature.
```
TEMP: 55
```


# Run an automated experiment
Start a method on a HiSeq2500 System from the command line.
All arguments are optional.
```
usage: pyseq [-h] [-config PATH] [-name NAME] [-output PATH] [-list]
             [-method METHOD] [-virtual]
```
- **h**: **See usage of pyseq**
- **config**: path to the configuration file, default is config.cfg in the current working directory
- **name**: name of the experiment, default is a YYYYMMDD_HHMMSS time stamp.
- **output**: path to the output directory to save images and logs, default is the current working directory
- **list**: **See installed methods**
- **method**: **See an installed method config and method recipe**
- **virtual**: **Run a virtual experiment**
- **settings**: **See available configuration options**
- **ports**: **List COM port identifier of instruments**
- **diagnostics**: **Perform a simple diagnostics run**

## Run an experiment
Assumes an experiment file, config.cfg, is in the current working directory.
```
pyseq -n MyFirstHiSeqExperiment
```

## Run a virtual experiment
Assumes an experiment file, config.cfg, is in the current working directory.
Useful for building and testing new methods.
```
pyseq -n TestCustomExperiment -v
```

## See usage of pyseq
```
pyseq -h, --help
```

## See installed methods
```
pyseq -l
```

## See an installed method config and method recipe
The example here is to see the config and recipe for a method called 4i.
```
pyseq -m 4i
```
