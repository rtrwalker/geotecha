geotecha: A software suite for geotechnical engineering
=======================================================


.. image:: https://github.com/rtrwalker/geotecha/blob/master/docs/_static/logo.png

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


Installation
------------
*geotecha* was developed and tested using python 2.7 on a 
windows 7 32-bit computer.  No attempt (as yet) has been made to 
develop a linux or mac distribution.  Major requirements are
numpy, matplotlib, and scipy.  Setting up your python environment 
to succesfully run all these packages can be cumbersome so pre-built
python stacks such as the freely available `Python(x,y)`_ are highly 
recommended (I use `Python(x,y)`_). Note it is best to unistall any 
existing python distributions before installing `Python(x,y)`_.

pip
+++
To install *geotecha* from the Python Package Index (PyPI) using pip:

.. code-block::

   pip install geotecha

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
informative to use two steps.  At times I have have issues with 
the build step and have had to explicitly specify the compiler to
use, for example:

.. code-block::
   
   python setup.py build --compiler=mingw32

You can see other build options using:

.. code-block::

   python setup.py build --help

Once installed you can test the package using:

.. code-block::

   nosetests geotecha -v -w C:\Python27\Lib\site-packages\ --with-doctest --doctest-options=+ELLIPSIS

The '-w' working directory tag is so that nose runs tests on the 
installed version of *geotecha* rather than the source code version 
(the source version will not have the external extensions).  Change 
the working directory to match your python location.

Removing geotecha
+++++++++++++++++
The cleanest method for removing *geotecha* is simply to use pip:

.. code-block::

   pip uninstall geotecha

You can also manually delete all files in the 'install.record' file.






.. _GPLv3: http://choosealicense.com/licenses/gpl-3.0/
.. _`Python(x,y)`: https://code.google.com/p/pythonxy/
.. _pypi: https://pypi.python.org/pypi






