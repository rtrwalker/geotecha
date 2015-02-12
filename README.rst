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
*geotecha* was developed on a Windows platform.  While I have tried
to use cross platform programming idioms, I have (at the time of 
writing) made no attempt to build/use *geotecha* on any platform other
than windows.  That does not mean that it will not work.  I simply
do not know if it does or does not.


Requirements
++++++++++++
*geotecha* uses a number of other python packages such as 
numpy, matplotlib, and scipy.  Setting up your python environment 
to succesfully run all these packages can be cumbersome so pre-built
python stacks such as the readily available Anaconda_ or 
`Python(x,y)`_ are highly recommended . Note generally better to 
unistall any existing python distributions before installing a new 
one.

Package requirements:

 - numpy
 - matplotlib
 - scipy
 - sympy
 - brewer2mpl
 - testfixtures
 - numpydoc
 - IPython
 - pandas
 - pkg_resources
 - mpl_toolkits
 - nose
 - one of wx (i.e. wxPython) or PyQt (PyQt4 or PyQt5)
 - Sphinx-PyPI-upload

Before worrying about if your system has the required packages just
try one of the installation methods below (first try the 
`Windows binaries`_ option).  Hopefully you will already have  
all the packages or else the requirements section of the setup.py
file will get them from pypi_.  However, issues can arise.
Due to anomalies in handling dashes in required package 
names, the required packages `pkg_resources` and `mpl_toolkits` 
will not automatically be installed.  Just install these using pip if
they are not already present:

.. code-block::

   pip install pkg_resources
   pip install mpl_toolkits

wxPython and PyQt are not always avialable through pypi_ so may 
have to be installed manually (there are usually windows binaries 
available.)

Windows binaries
++++++++++++++++
The easiest, hassle-free way to install *geotecha* on a windows 
machine is to download one of the pre-built binaries available  
at https://pypi.python.org/pypi/geotecha .  Once downloaded 
double click the .exe file to install.  I am not sure but you 
may need to install the dependency packages separately.
Binaries are avialable for 32 and 64 bit python-2.7 and python-3.4. 

See the `Building from source`_ section below for instructions
on how to to test your *geotecha* installation.
  

pip
+++
To install *geotecha* from the Python Package Index (PyPI) using pip:

.. code-block::

   pip install geotecha

This will essentially download the source files and build and install
the package.  *geotecha* has extension modules written in fortran 
which can cause issues if your python environent is not set up to 
handle them ( I think you need a fortran and a c compiler).
.As such you may have difficultly in building the 
external extensions (see `Building from source`_ below.

See the `Building from source`_ section below for instructions
on how to test your *geotecha* installation.

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


I had to modify my *PATH* environment variable to include the path 
to a gcc command (You shouldn't have trouble when using Anaconda_ 
becuase it comes packaged with MinGW, but occaisionally 
with `Python(x,y)`_ I've had to install MinGW).



Removing geotecha
+++++++++++++++++
The cleanest method for removing *geotecha* is simply to use pip:

.. code-block::

   pip uninstall geotecha

You can also manually delete all files in the 'install.record' file.




.. _GPLv3: http://choosealicense.com/licenses/gpl-3.0/
.. _`Python(x,y)`: https://code.google.com/p/pythonxy/
.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _pypi: https://pypi.python.org/pypi






