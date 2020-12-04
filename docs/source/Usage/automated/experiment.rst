*****************
Experiment Config
*****************

 The experiment config file has the following sections:
   #. :ref:`[experiment]`
   #. :ref:`[sections]`
   #. :ref:`[reagents]`
   #. :ref:`[cycles]`
   #. :ref:`[filters]`
   #. :ref:`[method]`

[experiment]
============
 experiment details (required, unless noted)
  - **method**: name of installed method, path to method config file, or section name in config file
  - **cycle**: number of cycles to run (integer)
  - **first flowcell**: which flowcell to start first if running 2, optional (A or B)

  ::

     [experiment]
     method = 4i
     cycles = 2
     first flowcell = A

[sections]
==========
position of sections on flowcell (required).
  `section name = AorB: LLx, LLy, URx, URy`

  - **section name**: name/id of section to image (string)
  - **AorB**: flowcell section is on (A or B)
  - **LLx**: lower left x coordinate of section, use slide ruler (float)
  - **LLy**: lower left y coordinate of section, use slide ruler (float)
  - **URx**: upper right x coordinate of section, use slide ruler (float)
  - **URy**: upper right y coordinate of section, use slide ruler (float)

  ::

     [sections]
     section1 = A: 15.5, 45, 10.5, 35



[reagents]
==========
Specify reagent ports (optional).
 `N = name`

 - **N**: port number (integer)
 - **Name**: name of reagent (string)

 ::

    [reagent]
    6 = GFAP
    7 = IBA1
    8 = AF547 + Cy5

[cycles]
========
Specify cycle specific reagents (optional).
 `variablereagent N = name`

- **variablereagent**: cycle dependent reagent in recipe, specified in method config (string)
- **N**: cycle (integer)
- **name**: reagent used for **variablereagent** at cycle **N** (string)

 ::

    [cycles]
    1stab 1 = GFAP
    1stab 2 = IBA1
    2ndab 1 = AF547 + Cy5
    2ndab 2 = AF547 + Cy5


[filters]
=========
Specify cycle specific optical filters (optional).
 `lasercolor N = name`

 - **lasercolor**: color of laser line, can use lower or uppercase initial (string)
 - **N**: cycle (integer)
 - **name**: optical density of filter to use (float/string), see table below.


  ===========  ===========  ========================================
  laser color  laser index  filters (Optical Density)
  ===========  ===========  ========================================
  green        1            open, 1.0, 2.0, 3.5, 3.8, 4.0, 4.5, home
  red          2            open, 0.2, 0.5, 0.6, 1.0, 2.4, 4.0, home
  ===========  ===========  ========================================

  ::

     [filters]
     green 1 = 1.6
     g 2 = 1.4
     G 3 = 0.6
     red 1 = 2.0
     r 2 = 1.0
     R 3 = 0.9

[method] in experiment config
=============================
Method specific HiSeq settings (optional)

Must match the **method** item in the :ref:`[experiment]` section.
Instead of a method specific section in the experiment config file,
a separate method config file may be used.

See :ref:`[method]` for details.

   ::

      [4i]
      recipe = 4i_recipe.txt
