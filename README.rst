geotecha: A software suite for geotechnical engineering
=======================================================


.. image:: https://raw.githubusercontent.com/rtrwalker/geotecha/master/docs/_static/logo.png



Writen by Dr Rohan Walker

*geotecha* is a a GPLv3_-licensed Python package for geotechncial 
engineering.

Many components of *geotecha* derive from geotechnical research 
conducted by Dr. Rohan Walker, Prof. Buddhima Indraratna, and others 
at the University of Wollongong, NSW, Australia.  

Primarily a repository of programs, tools, and code used by 
Dr Rohan Walker, the content reflects his primary interest in soft soil 
consolidation with and without vertical drains.  In particular the
`speccon` programs solve one-dimensional partial differential equations
associated with multi-layer problems using the spectral Galerkin 
method.  Material properties are constant with time but piecewsie-linear
with depth.  Loads and boundary conditions are piecewise linear with 
time (plus a sinusoidal component).  A number of other analytical 
solutions to soil consolidatoin problems are avialable in the 
`consolidation` sub-package.


Documentation
-------------
*geotecha* documentation is currently stored at http://pythonhosted.org//geotecha/ .


Installation
------------
*geotecha* was developed and tested using python 2.7 on a 
windows 7 32-bit computer.  No attempt (as yet) has been made to 
develop a linux or mac distribution.  Major requirements are
numpy, matplotlib, and scipy.  Setting up your python environment 
to succesfully run all these packages can be cumbersome so pre-built
python stacks such as the freely available `Python(x,y)`_ are highly 
recommended (I use `Python(x,y)`_). Note it is best to unistall any 
existing python distributions before installing `Python(x,y)`_.  Also
make sure that you check the sympy option when specifying what 
packages in Python(x,y) to install (I just choose the "full" version.

Windows binaries
++++++++++++++++
The easiest, hassle-free way to install *geotecha* on a windows 
machine is to download one of the pre-built binaries available  
at https://pypi.python.org/pypi/geotecha .  Once downloaded 
double click the .exe file to install.  I am not sure but you 
may need to install the dependency packages separately.


pip
+++
To install *geotecha* from the Python Package Index (PyPI) using pip:

.. code-block::

   pip install geotecha

This will essentially download the source files and build and install
the package.  As such you may have difficultly in building the 
external extensions (see `Building from source`_ below.


Note that due to anomalies in handling dashes in required package 
names, the required packages `pkg_resources` and `mpl_toolkits` 
will not automatically be installed.  Just install these using pip if
they are not already present:

.. code-block::

   pip install pkg_resources
   pip install mpl_toolkits



Building from source
++++++++++++++++++++
You can download the *geotecha* source files from pypi_ or from the 
Github repository https://github.com/rtrwalker/geotecha .  
*geotecha* uses some external extensions written in fortran, so 
you will need to have a fortran compiler present.  Then it is a 
matter of building and installing:

.. code-block::

   python setup.py build
   python setup.py install --record install.record

The "--record install.record" will make a file containing a list
of all the files installed.  It is possible to skip the build step
(it will be included in the install step).  But I find it more
informative to use two steps.  

Once installed you can test the package using:

.. code-block::

   nosetests geotecha -v -w C:\Python27\Lib\site-packages\ --with-doctest --doctest-options=+ELLIPSIS

The '-w' working directory tag is so that nose runs tests on the 
installed version of *geotecha* rather than the source code version 
(the source version will not have the external extensions).  Change 
the working directory to match your python location.


Issues with building/installing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At times I have had issues with 
the build step and have had to explicitly specify the compiler to
use, for example:

.. code-block::
   
   python setup.py build --compiler=mingw32

You can see other build options using:

.. code-block::

   python setup.py build --help

Another problem is getting errors such as:

.. code-block::

   gcc is not recognized as an internal or external command


I had to make sure that the 'MinGW\\bin\\' directory was in my *PATH* 
environment variable. Note you may have to install MinGW.



Removing geotecha
+++++++++++++++++
The cleanest method for removing *geotecha* is simply to use pip:

.. code-block::

   pip uninstall geotecha

You can also manually delete all files in the 'install.record' file.




.. _GPLv3: http://choosealicense.com/licenses/gpl-3.0/
.. _`Python(x,y)`: https://code.google.com/p/pythonxy/
.. _pypi: https://pypi.python.org/pypi






