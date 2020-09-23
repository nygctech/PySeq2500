###################
Interactive Control
###################

Initializing
============
The HiSeq instruments and cameras must first be initialized by using
:meth:`~pyseq.__init__.HiSeq.initializeInstruments` and
:meth:`~pyseq.__init__.HiSeq.initializeCams` respectively.


Positioning
===========
Use :meth:`~pyseq.__init__.HiSeq.position` to get all the stage information
needed to image a section in **pos_dict**. Move to the :ref:`xstage` to
'x_initial', the :ref:`ystage` to 'y_initial', and the :ref:`zstage` to
[21500, 21500, 21500] before imaging a section. The in focus position of the
:ref:`objective<objstage>` can be found with and moved to with
:meth:`~pyseq.__init__.HiSeq.autofocus`. The default :ref:`Autofocus` routine
set in **HiSeq.AF** is **partial once**. Other :ref:`Autofocus` include **full**,
**partial**, and **full once**.

Set Optics
==========
The :ref:`excitaton filters<optics>` need be moved in place before taking an
image or autofocusing. In general the emission filter should always be set in
the path of the light. The filters will be returned to the home (all laser light
blocked) position after an image has been taken.

Take Images
===========
Use :meth:`~pyseq.__init__.HiSeq.take_picture` to image the current stage
position. Use :meth:`~pyseq.__init__.HiSeq.zstack` to image the current stage
position at a series of objective positions increasing further in distance from
the stage. The spacing of objective positions is controlled by HiSeq.nyquist_obj
attribute. Use :meth:`~pyseq.__init__.HiSeq.scan` to image an entire section at
a series of objective positions increasing further in distance from the stage.


HiSeq
=====
.. currentmodule:: pyseq

.. automodule:: pyseq.__init__
   :members:

   .. rubric:: Classes

   .. autosummary::
