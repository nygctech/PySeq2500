#################
Automated Control
#################

***************************************
Automate a method/recipe on a HiSeq2500
***************************************

The following files are needed to automate a method on a HiSeq 2500 System.

 - Method Config
 - Experiment Config
 - Method Recipe

Start a method on a HiSeq2500 System.
=====================================

::

   pyseq -c experiment_config -n experiment_name -o output_path

- experiment_config = path to experiment config, default = ./config.cfg
- experiment_name = name of experiment, default = YYYYMMDD_hhmmss timestamp
- output_path = directory to save images and log, default = current directory

See usage of pyseq.
===================

::

   pyseq -h, --help

See installed methods.
======================

::

   pyseq -l

See a method config and method recipe.
======================================
The example here is to see the config and recipe for a method called 4i.

::

   pyseq -m 4i

*************
Method Config
*************

1. [method]
===========

::

   [4i]
   recipe = 4i_recipe.txt
   flush speed = 500
   flush volume = 1000
   reagent speed = 200
   variable reagents = 1stab, 2ndab
   first port = blocking
   barrels per lane = 8
   laser power = 400

The name of this section should match the name of the method.

The only required key in this section is **recipe**.

- **recipe**: path to the method recipe as its value (path)

The other keys are optional.

- **flush speed**: flowrate to flush lines with in uL/min (integer)
- **flush volume**:  volume to flush line with in uL (integer)
- **reagent speed**: flowrate to pump reagents during recipe in uL/min (integer)
- **variable reagents**: name of variable ports in recipes that are cycle dependent (string)
- **first port**: port to start recipe at on first cycle (string)
- **barrels per lane**: number of syringe barrels that are used per lane on flowcell (integer)
- **laser power**: set power of laser in mW (integer)



2. [valve24]
============

::

   [valve24]
   1 = PBS
   2 = water
   3 = elution
   4 = blocking
   5 = imaging

Specify method required ports (optional). `N = name`

- N: port number (integer)
- Name: name of reagent (string)



*****************
Experiment Config
*****************

1. [experiment]
===============

::

   [experiment]
   method = 4i
   cycles = 2
   first flowcell = A

experiment details (required, unless noted)

- method: name of installed method or path to method config file (string)
- cycle: number of cycles to run (integer)
- first flowcell: which flowcell to start first if running 2, optional (A or B)

2. [sections]
=============

::

   [sections]
   section1 = A: 15.5, 45, 10.5, 35

position of sections on flowcell (required). `section name = F: LLx, LLy, URx, URy`

- section name: name/id of section to image (string)
- F: flowcell section is on (A or B)
- LLx: lower left x coordinate of section, use slide ruler (float)
- LLy: lower left y coordinate of section, use slide ruler (float)
- URx: upper right x coordinate of section, use slide ruler (float)
- URy: upper right y coordinate of section, use slide ruler (float)



3. [valve24]
============

::

   [valve24]
   6 = GFAP
   7 = IBA1
   8 = AF547 + Cy5

Specify additional ports (optional). `N = name`

- N: port number (integer)
- Name: name of reagent (string)

4. [cycles]
===========

::

   [cycles]
   1stab 1 = GFAP
   1stab 2 = IBA1
   2ndab 1 = AF547 + Cy5
   2ndab 2 = AF547 + Cy5

Specify cycle specific reagents (optional). `variablereagent N = name`

- variablereagent: cycle dependent reagent in recipe, specified in method config (string)
- N: cycle (integer)
- name: reagent used for variablereagent at cycle N (string)


*************
Method Recipe
*************

There are 5 basic actions to build a recipe.

1. **PORT**: *port name* (string)
=================================

Valve switches to specified port.
::

   PORT: water

2. **PUMP**: *pump volume in uL* (integer)
==========================================

Syringe pump draws specified volume through flowcell lane.
::

   PUMP: 2000


3. **HOLD**: *hold time in min.* (integer)
==========================================

Recipe pauses for specified time.
::

   HOLD: 10

4. **WAIT**: **IMAG** or `port name` (string)
=============================================

Recipe waits to continue until the other flowcell is imaging (**IMAG**) or
switches to *port name*. If there is only one flowcell, **WAIT** is ignored.
::

   WAIT: water

5. **IMAG**: *z focal planes* (integer)
=======================================

The flowcell is imaged at the specified number of z focal planes at the
sections listed in the experiment config.
::

   IMAG: 15

Example Recipe
==============

This recipe automates a method called 4i. In the 4i **method config** the
**variable reagents** and **first port** are set as follows:
::

   variable reagents = 1stab, 2ndab
   first port = blocking

The 4i method stains tissue sections by first blocking sections for 1 hr, then
staining sections with 1stab, followed by 2ndab for 2 hrs. After the sections
are imaged in imaging buffer, antibodies are eluted off the tissue sections, and
are then ready for subsequent rounds of staining.
::

   PORT:	water		#Move valve to water wash (port 2)
   PUMP:	2000		#Pump 2000 uL
   PORT:	elution    	#Move valve to elution (port 3)
   PUMP:	500		#Pump 500 uL
   HOLD:	10		#Hold for 10 minutes
   PUMP:	500		#Pump 500 uL
   HOLD:	10		#Hold for 10 minutes
   PUMP:	500		#Pump 500 uL
   HOLD:	10		#Hold for 10 minutes
   PUMP:	500		#Pump 500 uL
   HOLD:	10		#Hold for 10 minutes
   PUMP:	500		#Pump 500 uL
   HOLD:	10		#Hold for 10 minutes
   PUMP:	500		#Pump 500 uL
   HOLD:	10		#Hold for 10 minutes
   PORT:	blocking	#Move valve to blocking buffer (port 4)
   PUMP:	800		#Pump 800 uL
   HOLD:	60		#Hold for 60 min
   PORT:	PBS		#Move valve to PBS wash (port 1)
   PUMP:	2000		#Pump 2000 uL
   PORT:	1stab		#Move valve to primary antibody (variable)
   PUMP:	500		#Pump 500 uL
   HOLD:	120		#Hold for 120 min
   PORT:	PBS		#Move valve to PBS wash (port 1)
   PUMP:	2000		#Pump 2000 uL
   PORT:	2ndab		#Move valve to secondary antibody (variable)
   PUMP:	500		#Pump 500 uL
   HOLD:	120		#Hold for 120 min
   PORT:	PBS		#Move valve to PBS wash (port 1)
   PUMP:	2000		#Pump 2000 uL
   WAIT:	water		#Wait till other flowcell is washing with water
   PORT:	imaging		#Move valve to imaging buffer (port 5)
   PUMP:	750		#Pump 750 uL
   IMAG:	15		#image 15 z sections
