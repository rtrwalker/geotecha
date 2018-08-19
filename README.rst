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
Dr. Rohan Walker, the content reflects his primary interest in soft soil
consolidation with and without vertical drains.  In particular:

 - the `speccon` programs solve one-dimensional partial differential equations
   associated with multi-layer problems using the spectral Galerkin
   method.  Material properties are constant with time but piecewise-linear
   with depth.  Loads and boundary conditions are piecewise linear with
   time (plus a sinusoidal component).
 - `specbeam` models a finite elastic Euler-Bernoulli beam resting on
   viscoelastic foundation subjected to a moving load(s), with piecewise-linear
   spatially varying properties.  It uses the spectral Galerkin method to solve the
   dynamic system deflections over time.
 - A number of other analytical solutions to soil consolidation problems are
   available in the
   `consolidation` sub-package.


Documentation
-------------
*geotecha* documentation is currently stored at http://pythonhosted.org//geotecha/ .
There you can find more descriptions of speccon and specbeam.  Peruse the
api docs for a listing of all the modules, classes and code (make sure you
scroll up to the top of each api_doc page to see the summary listing of
each module - frustratingly a clicked hyperlink doesn't
take you to the top of the page!)


GitHub Repository
-----------------
The *geotecha* codebase is on GitHub, https://github.com/rtrwalker/geotecha
here you will find the development version of the code set.


Installation
------------
*geotecha* was developed on a windows platform.  While I have tried
to use cross platform programming idioms, I have (at the time of
writing) made no attempt to build/use *geotecha* on any platform other
than windows.  That does not mean that it will not work.  I simply
do not know if it does or does not.


Requirements
++++++++++++
*geotecha* uses a number of other python packages such as
numpy, matplotlib, and scipy.  Setting up your python environment
to successfully run all these packages can be cumbersome so pre-built
python stacks such as the readily available `Anaconda`_
are highly recommended . Note it is generally better to
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
 - sphinx
 - mock

Before worrying about if your system has the required packages just
try one of the installation methods below (first try the
`Windows binaries`_ option).  Hopefully you will already have
all the packages or else the requirements section of the setup.py
file will get them from `pypi`_.  However, issues can arise.
Due to anomalies in handling dashes in required package
names, the required packages `pkg_resources` and `mpl_toolkits`
will not automatically be installed.  Just install these using pip if
they are not already present:

.. code-block::

   pip install pkg_resources
   pip install mpl_toolkits

wxPython and PyQt are not always available through `pypi`_ so may
have to be installed manually (there are usually windows binaries
available).

Windows Binaries
++++++++++++++++
The easiest, hassle-free way to install *geotecha* on a windows
machine is to download one of the pre-built binaries available
at https://pypi.python.org/pypi/geotecha .  Once downloaded
double click the .exe file to install.  Note that the installer
will display the raw text of this file and it may look odd. This
does not matter.
You may need to install the dependency packages separately.
If your setup doesn't match the binaries then you will have to try
`Building from source`_.

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
You can download the *geotecha* source files from pypi_ (release version) 
or from the Github repository (development version) https://github.com/rtrwalker/geotecha .
*geotecha* uses some external extensions written in Fortran, so
you will need to have a Fortran compiler present.  Building from source on
Windows can be troublesome at the best of times, so see the
`Issues with building/installing`_ section below if you are trying to build
on windows.  For other systems it 'should' be as easy as:

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
'-w' working directory tag so that nose runs tests on the
installed version of *geotecha* rather than the source code version
(the source version will not have the external extensions).  Change
the working directory to match your python location, for example:

.. code-block::

   nosetests geotecha -v -w C:\Python36\Lib\site-packages\ --with-doctest --doctest-options=+ELLIPSIS

or if you are using an Anaconda env (like me) use something like:

.. code-block::

   nosetests geotecha -v -w C:\Anaconda3\envs\py36\Lib\site-packages\ --with-doctest --doctest-options=+ELLIPSIS --verbosity=3

You might get two test failures about importing ext_integrals and ext_epus.
This indicates that the fortran extensions are not working.  Don't worry
python/numpy (slower) versions of relevant functions will be used instead.

If you have a numpy version less than 1.14 then the tests will probably throw
many failures associated with spaces and string representations of numpy
arrays.  This is due to changes in numpy
https://docs.scipy.org/doc/numpy-1.14.0/release.html
Don't worry I've just updated things for python3.6, you will eventually
upgrade and the test failures will disappear. Check which numpy version you
have with :

.. code-block::

   python -c "import numpy; print(numpy.version.version)"

I have also had some odd behaviour where I run tests and get a couple
of test failures.  Then run the same tests and they all pass.

Building the docs
^^^^^^^^^^^^^^^^^
The *geotecha* docs can be built by running the following in the
geotecha directory:

.. code-block::

   python setup.py build_sphinx --source-dir=docs/ --build-dir=docs/_build --all-files

The build requires a symlink to the examples directory.  See the
README.txt in the docs for instructions.


Issues with building/installing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
At times (every time?) I have had issues with building from source on windows.
So here are some hints to point you in the right direction.
In python2.7 and up to python 3.4 it was relatively easy because
there was a Mingwpy package ( https://anaconda.org/carlkl/mingwpy ), however,
that very useful project has been abandoned
( https://groups.google.com/forum/#!topic/mingwpy/1k_BLFPLmBI ).
So here is what works for me on Windows 10, 64 bit with python3.6

Based on the helpful blog post from Michael Hirsch ( https://www.scivision.co/python-windows-visual-c++-14-required/ )
install the relevant version of Microsoft Build Tools for Visual C++
(2017 for me) from https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017
(look in the "Tools for Visual Studio 2017" section).
Note it is a large install taking up ~6GB.

Now install m2w64-toolchain hosted by Anaconda:

.. code-block::

   conda install -c msys2 m2w64-toolchain

Clean up previous builds:

.. code-block::

   python setup.py clean --all

Now try and build the thing explicitly specifying the compiler:

.. code-block::

   python setup.py build --compiler=mingw32
   python setup.py install --record install.record


Test the install as above.  No test failures will indicate that the
Fortran extension modules have been successfully built and installed.


Removing geotecha
+++++++++++++++++
The cleanest method for removing *geotecha* is simply to use pip:

.. code-block::

   pip uninstall geotecha

You can also manually delete all files in the 'install.record' file.


Setting up an Anaconda env on Windows
+++++++++++++++++++++++++++++++++++++
After downloading and installing Anaconda make sure "C:\Anaconda3\Scripts" is
in your PATH environment variable (otherwise conda command will not be found).
Open the Anaconda prompt (start menu).  Create a full anaconda env named py36
with a specified python version using (note it will download large files):

.. code-block::

   conda create -n py36 python=3.6 anaconda

If you need to start again remove the env with:

.. code-block::

   conda env remove --name py36

Close the anaconda prompt and then open the py36 anaconda prompt (start menu).
Your py36 env is now ready to install *geotecha* and other python packages.


.. _GPLv3: http://choosealicense.com/licenses/gpl-3.0/
.. _Anaconda: https://www.anaconda.com/download/
.. _pypi: https://pypi.python.org/pypi

