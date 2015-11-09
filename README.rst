geotecha: A software suite for geotechnical engineering
=======================================================


.. image:: https://raw.githubusercontent.com/rtrwalker/geotecha/master/docs/_static/logo.png



Writen by Dr Rohan Walker

*geotecha* is a a GPLv3_-licensed Python package for geotechnical
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
solutions to soil consolidation problems are available in the
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
to successfully run all these packages can be cumbersome so pre-built
python stacks such as the readily available Anaconda_ or
`Python(x,y)`_ are highly recommended . Note generally better to
uninstall any existing python distributions before installing a new
one.

Package requirements:

 - numpy
 - matplotlib
 - scipy
 - pandas
 - sympy
 - brewer2mpl
 - testfixtures
 - numpydoc
 - IPython
 - pkg_resources
 - mpl_toolkits
 - nose
 - one of wx (i.e. wxPython) or PyQt (PyQt4 or PyQt5)

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

wxPython and PyQt are not always available through pypi_ so may
have to be installed manually (there are usually windows binaries
available).

Windows binaries
++++++++++++++++
The easiest, hassle-free way to install *geotecha* on a windows
machine is to download one of the pre-built binaries available
at https://pypi.python.org/pypi/geotecha .  Once downloaded
double click the .exe file to install.  Note that the installer
will display the raw text of this file and it may look odd. This
does not matter.
You may need to install the dependency packages separately.
Binaries are available for 32 and 64 bit python-2.7 and python-3.4.

Test you installation by opening a command prompt (Windows+R cmd) and
enter the following command:

.. code-block::

   nosetests geotecha -v --with-doctest --doctest-options=+ELLIPSIS --doctest-options=+IGNORE_EXCEPTION_DETAIL


It is common to get an error such as:

.. code-block::

   ImportError: No module named 'brewer2mpl'

which usually means one of the dependencies is not installed.  Simply
rerun the tests after installing the missing package with:

.. code-block::

   pip install brewer2mpl


pip
+++
To install *geotecha* from the Python Package Index (PyPI) using pip:

.. code-block::

   pip install geotecha

This will essentially download the source files and build and install
the package.  *geotecha* has extension modules written in Fortran
which can cause issues if your python environment is not set up to
handle them ( I think you need a Fortran and a c compiler).
.As such you may have difficultly in building the
external extensions (see `Building from source`_ below.

See the `Windows binaries`_ section above for instructions
on how to to test your *geotecha* installation.


Building from source
++++++++++++++++++++
You can download the *geotecha* source files from pypi_ or from the
Github repository https://github.com/rtrwalker/geotecha .
*geotecha* uses some external extensions written in Fortran, so
you will need to have a Fortran compiler present.  Then it is a
matter of building and installing:

.. code-block::

   python setup.py build
   python setup.py install --record install.record

The "--record install.record" will make a file containing a list
of all the files installed.  It is possible to skip the build step
(it will be included in the install step).  But I find it more
informative to use two steps.

See the `Windows binaries`_ section above for instructions
on how to to test your *geotecha* installation.  When testing
you may wish to use the
'-w' working directory tag is so that nose runs tests on the
installed version of *geotecha* rather than the source code version
(the source version will not have the external extensions).  Change
the working directory to match your python location, for example:

.. code-block::

   nosetests geotecha -v -w C:\Python27\Lib\site-packages\ --with-doctest --doctest-options=+ELLIPSIS


Building the docs
^^^^^^^^^^^^^^^^^
The *geotecha* docs can be build by running the following in the
docs directory:

.. code-block::

   make html

The build requires a symlink to the examples directory.  See the
README.txt in the docs for instructions.


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
because it comes packaged with MinGW, but occasionally
with `Python(x,y)`_ I've had to install MinGW).


When trying to build *geotecha* from source on 64-bit windows you may
get the following error:

.. code-block::

   File "C:\Anaconda3\envs\py27\lib\site-packages\numpy\distutils\fcompiler\gnu.p
   y", line 337, in get_libraries
    raise NotImplementedError("Only MS compiler supported with gfortran on win64
   ")

According to http://scientificcomputingco.blogspot.com.au/2013/02/f2py-on-64bit-windows-python27.html
the error can be fixed by changing the source code to pass the exception
(i.e. add "pass #" before the "raise").


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






