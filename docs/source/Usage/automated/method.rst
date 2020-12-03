*************
Method Config
*************
The method config file has 2 possible sections.
Instead of a separate method config file, a method specific section in the
experiment config file may be used (in which case only use 1 **[reagent]**
section). See :ref:`[method] in experiment config`

[method]
========
HiSeq settings specific to the method.

The name of this section should match the name of the method. The only required
key in this section is **recipe**. The other keys are optional.

 - **recipe**: path to the method recipe as its value (path), required
 - **flush volume**: volume to flush line with in uL (integer), default is 1000
 - **main prime volume**: volume to prime main lines (ports  1-8 & 10-19) in uL (integer), default is 500
 - **side prime volume**: volume to prime side lines (ports 9 & 22-24) in uL (integer), default is 350
 - **sample prime volume**: volume to prime samples lines (port 20) in uL (integer), default is 250
 - **flush speed**: flowrate to flush lines with in uL/min (integer), default is 700
 - **reagent speed**: flowrate to pump reagents during recipe in uL/min (integer), default is the :ref:`minimum flow rate<pump>`
 - **variable reagents**: name of variable ports in recipes that are cycle dependent (string)
 - **first port**: port to start recipe at on first cycle (string)
 - **barrels per lane**: number of syringe barrels that are used per lane on flowcell (integer), default is 8
 - **inlet ports**: 2 inlet port row or 8 inlet port row (integer), default is 2
 - **laser power**: set power of laser in mW (integer), default is 10
 - **z position**: step of tilt motors when imaging (integer), default is 21500
 - **focus filter 1**: filter for green laser for autofocus routine, default is 2.0
 - **focus filter 2**: filter for red laser for autofocus routine, default is 2.0
 - **default em filter**: emission filter used for imaging, True for in path, False for out of path (bool), default is True
 - **default filter 1**: filter for green laser if not specified in **[filter]** section of experiment config file (float/string), default is `home`
 - **default filter 2**: filter for red laser if not specified in **[filter]** section of experiment config file (float/string), default is `home`
 - **rinse**: reagent to rinse the flowcell with between completion of the experiment and flushing of the lines during shutdown (string), default is `None`
 - **autofocus**: routine used for autofocusing (string), see **Autofocus** for more info, default is `partial once`
 - **focus tolerance**: distance in microns for acceptable focus error, default is 0 um which allows for maximum error
 - **enable z stage**: Enable/disable z stage movements (bool), default is True.
 - **bundle height:** sensor bundle height of cameras (integer), only certain values are valid, default is 128
 - **temperature interval:** time between checking temperature of flowcell (integer), default is 5
 
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


[reagents] in method config
===========================
Specify ports for reagent specific to the method. This is useful if running a
method repeatedly and none or some of the ports change from each experiment.
Alternatively, all the ports may be specified in the experiment config.

See :ref:`[reagents]` for details.
