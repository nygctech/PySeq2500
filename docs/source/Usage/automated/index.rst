#################
Automated Control
#################

The following files are used to automate a method on a HiSeq 2500 System.
 - :ref:`Experiment Config` (required)
 - :ref:`Method Config` (optional)
 - :ref:`Method Recipe` required

Start a method on a HiSeq2500 System.
=====================================

   ::

      pyseq -c experiment_config -n experiment_name -o output_path

   - experiment_config = path to experiment config, default = ./config.cfg
   - experiment_name = name of experiment, default = YYYYMMDD_hhmmss timestamp
   - output_path = directory to save images and log, default = current directory

Run a virtual experiment
========================
   Use a virtual HiSeq, can be used for building and testing new methods.

   ::

       pyseq -n TestCustomExperiment -v

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


Configs & Recipe
================
.. toctree::
   :maxdepth: 2

   experiment
   method
   recipe
