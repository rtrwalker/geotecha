*******
speccon
*******

Contents

.. toctree:: 


What is speccon?
================

'speccon' is the common name for a number of programs that use the
spectral Galerkin method to solve multilayer consolidation problems.

Speccon programs currently available in *geotecha* are:

 - **speccon1d_vr** - Multilayer consolidation with vertical drains.
  - See :class:`geotecha.speccon.speccon1d_vr.Speccon1dVR` for details and 
    input variables.

 - **speccon1d_vrw** - Multilayer consolidation with vertical drains
   including well resistance.
  - See :class:`geotecha.speccon.speccon1d_vrw.Speccon1dVRW` for details and 
    input variables.

 - **speccon1d_vrc** - Multilayer consolidation with stone columns 
   including vertical and radial drainage in the column.
  - See :class:`geotecha.speccon.speccon1d_vrc.Speccon1dVRC` for details and 
    input variables.

 - **speccon1d_unsat** - Multilayer consolidation of unsaturated soil.
  - See :class:`geotecha.speccon.speccon1d_unsat.Speccon1dUnsat` for details and 
    input variables.



How to use speccon programs?
============================

There are two main ways to use speccon programs:

 - Use a speccon.exe script

  - Create an input file containing python syntax initializing relevant 
    variables.  Only a subset of python syntax is allowed.
  - Run the relevant speccon.exe script with the input file.
  - Output such as csv data files, parsed input files, and plots 
    are controlled by various input variables.
  - This is the best option for working with one set of input at a time.
  - The speccon.exe scripts should have been installed with geotecha.
    They can usually be found in the /python27/Scripts/ or similar 
    folder.  
    Alternately run them directly from a command prompt (assuming their
    location is in your path environment variable).
 - Use the Speccon classes in your own python programs/scripts.

  - Instead of passing an input file to a script you can initialize a
    speccon object with a multi-line string.
  - Create a multi-line string containing python syntax to initialize
    relevant variables.  Remember that the string itself cannot contain
    all python syntax.  This is not as restrictive as it first appears
    as you can easily pass calculated values into the string using 
    the format method essentially giving you access to all python 
    functionality.
  - After initializing a speccon object perform the analysis with the
    make_all method.
    a subset of teh python language
  - This is the most flexible approach especially for parametric studies
    and custom plots.


The relevant variables for use in each speccon program can be found in 
the docstring of the relevant file which can be found in the api section
of the docs.



Examples
--------

Specific Examples can be found in the :ref:`examples-index` section.
Further examples can be found by digging around in the testing 
routines in the source code.


Using speccon1d_vr below are three different ways to run a speccon 
analysis.

**Script**

Create a text file containing the text below.  Locate and run the 
speccon1d_vr.exe file.  Choose the just created input file.  A folder of
output should be created in the same directory as the input file.

.. literalinclude:: ..\\geotecha_examples\\speccon\\example_1d_vr_001_schiffmanandstein1970.py
   :language: python


The speccon.exe scripts accept different arguments when run from 
a command line.  To view the command line options use::

    speccon1d_vr.exe -h


**Input file as template**

Put the following in a .py file and run it with python.  A figure with 
three subplots should appear.


.. plot:: geotecha_examples\\speccon\\example_1d_vr_001a_schiffmanandstein1970.py
   :include-source:



**Multi-line string**

Put the following in a .py file and run it with python.  A figure with 
three subplots should appear.


.. plot:: geotecha_examples\\speccon\\example_1d_vr_001b_schiffmanandstein1970.py
   :include-source:


