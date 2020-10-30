*************
Method Recipe
*************
There are 6 basic actions to build a recipe.

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

2. **TEMP**: *temperature in degrees Celsius* (float)
=====================================================

 Set temperature of flowcell. 
 ::

    TEMP: 55.0

3. **HOLD**: *hold time in min.* (integer)
==========================================

 Recipe pauses for specified time. The recipe can also be paused until user
 input by using `STOP` instead of an integer.
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

 This recipe automates a method called 4i. In the 4i :ref:`method config<Method
 Config>` the :ref:`variable reagents<[cycles]>` and :ref:`first port<[method]>`
 are set as follows:

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
