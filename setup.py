# geotecha - A software suite for geotechncial engineering
# Copyright (C) 2013  Rohan T. Walker (rtrwalker@gmail.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses/gpl.html.
"""geotecha: a software suite for geotechncial engineering

Hey look at me I'm a long description
But how long am I?

"""

from __future__ import division, print_function

#ideas for setup/f2py came from:
#    -numpy setup.py: https://github.com/numpy/numpy/blob/master/setup.py 2013-11-07
#    -winpython setup.py: http://code.google.com/p/winpython/source/browse/setup.py 2013-11-07
#    -needing to use
#        import setuptools; from numpy.distutils.core import setup, Extension:
#        http://comments.gmane.org/gmane.comp.python.f2py.user/707 2013-11-07
#    -wrapping FORTRAN code with f2py: http://www2-pcmdi.llnl.gov/cdat/tutorials/f2py-wrapping-fortran-code 2013-11-07
#    -numpy disutils: http://docs.scipy.org/doc/numpy/reference/distutils.html 2013-11-07
#    -manifest files in disutils:
#        'distutils doesn't properly update MANIFEST. when the contents of directories change.'
#        https://github.com/numpy/numpy/blob/master/setup.py
#    -if things are not woring try deleting build, sdist, egg directories  and try again:
#        http://stackoverflow.com/a/9982133/2530083 2013-11-07
#    -getting fortran extensions to be installed in their appropriate sub package
#        i.e. "my_ext = Extension(name = 'my_pack._fortran', sources = ['my_pack/code.f90'])"
#        Note that sources is a list even if one file:
#        http://numpy-discussion.10968.n7.nabble.com/f2py-and-setup-py-how-can-I-specify-where-the-so-file-goes-tp34490p34497.html 2013-11-07
#    -install fortran source files into their appropriate sub-package
#        i.e. "package_data={'': ['*.f95','*.f90']}# Note it's a dict and list":
#        http://stackoverflow.com/a/19373744/2530083 2013-11-07
#    -Chapter 9 Fortran Programming with NumPy Arrays:
#        Langtangen, Hans Petter. 2013. Python Scripting for Computational Science. 3rd edition. Springer.
#    -Hitchhikers guide to packaging :
#        http://guide.python-distribute.org/
#    -Python Packaging: Hate, hate, hate everywhere :
#        http://lucumr.pocoo.org/2012/6/22/hate-hate-hate-everywhere/
#    -How To Package Your Python Code:
#        http://www.scotttorborg.com/python-packaging/
#    -install testing requirements:
#        http://stackoverflow.com/a/7747140/2530083 2013-11-07
#    - 'python setup.py develop' :
#        http://stackoverflow.com/a/19048754/2530083

import setuptools
from numpy.distutils.core import setup, Extension
import os
import os.path as osp

def readme(filename='README.rst'):
    with open('README.rst') as f:
        text=f.read()
    f.close()
    return text

def get_package_data(name, extlist):
    """Return data files for package *name* with extensions in *extlist*"""
    #modified slightly from taken from http://code.google.com/p/winpython/source/browse/setup.py 2013-11-7
    flist = []
    # Workaround to replace os.path.relpath (not available until Python 2.6):
    offset = len(name)+len(os.pathsep)
    for dirpath, _dirnames, filenames in os.walk(name):
        for fname in filenames:
            if not fname.startswith('.') and osp.splitext(fname)[1] in extlist:
#                flist.append(osp.join(dirpath, fname[offset:]))
                flist.append(osp.join(dirpath, fname))
    return flist

def get_folder(name, foldernames):
    flist = []
    for dirpath, _dirnames, filenames in os.walk(name):
#        print(dirpath, _dirnames)
        for dname in _dirnames:
            if dname in foldernames:

                flist.append(osp.join(dirpath, dname))
    return flist


DOCLINES = __doc__.split("\n")
CLASSIFIERS = """\
Development Status :: 1 - Planning
License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)
Programming Language :: Python :: 2.7
Topic :: Scientific/Engineering :: Geotechnical Engineering
Topic :: Scientific/Engineering
"""

NAME = 'geotecha'
MAINTAINER = "Dr Rohan Walker"
MAINTAINER_EMAIL = "rtrwalker@gmail.com"
DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = "\n".join(DOCLINES[2:])#readme('readme.rst')
URL = "https://github.com/rtrwalker/geotecha.git"
DOWNLOAD_URL = "https://github.com/rtrwalker/geotecha.git"
LICENSE = 'GNU General Public License v3 or later (GPLv3+)'
CLASSIFIERS = [_f for _f in CLASSIFIERS.split('\n') if _f]
KEYWORDS=''
AUTHOR = "Dr Rohan Walker"
AUTHOR_EMAIL = "rtrwalker@gmail.com"
PLATFORMS = ["Windows"]#, "Linux", "Solaris", "Mac OS-X", "Unix"]
MAJOR = 0
MINOR = 1
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

INSTALL_REQUIRES=[
    "numpy>=1.8",
    "matplotlib>=1.3",
    "scipy>=0.13.3",
    "sympy>=0.2.7"]#note my speccon integrals only work on sympy<=0.2.7.
ZIP_SAFE=False
TEST_SUITE='nose.collector'
TESTS_REQUIRE=['nose', 'testfixtures']

DATA_FILES = [(NAME, ['LICENSE.txt','README.rst'])]
PACKAGES=setuptools.find_packages()
PACKAGES.remove('tools')

PACKAGE_DATA={'': ['*.f95','*f90'],} #'geotecha.plotting.test': ['geotecha\\plotting\\test\\baseline_images\\test_one_d\\spines_axes_positions.png'] }
ext_files = get_package_data(NAME,['.f90', '.f95','.F90', '.F95'])
ext_module_names = ['.'.join(osp.splitext(v)[0].split(osp.sep)) for v in ext_files]
EXT_MODULES = [Extension(name=x,sources=[y]) for x, y in zip(ext_module_names, ext_files)]


#add baseline_images for matplotlib image comparison
baseline_folders = get_folder(NAME, ['baseline_images'])
baseline_module_names = [osp.split(v)[0].replace(osp.sep,'.') for v in baseline_folders]
for v in baseline_module_names:
    if PACKAGE_DATA.has_key(v):
        PACKAGE_DATA[v].append(osp.join('baseline_images','*','*.*'))
    else:
        PACKAGE_DATA[v]=[osp.join('baseline_images','*','*.*')]
#[PACKAGE_DATA[v] = osp.join('baseline_images','*','*.*') for v in baseline_module_names]
#baseline_files = [osp.join('baseline_images','*','*.*')]
#png_files = get_package_data(NAME,['.png'])
#
#png_files =[v for v in png_files if 'baseline_images' in v]
##png_files=[v.split(osp.sep)[v.split()]]
#png_module_names = ['.'.join(osp.split(v)[0].split(osp.sep)) for v in png_files]

ENTRY_POINTS = {
        'console_scripts': [
            'speccon1d_vr = geotecha.speccon.speccon1d_vr:main',
            'nogamiandli2003 = geotecha.consolidation.nogamiandli2003:main',
            'schiffmanandstein1970 = '
                    'geotecha.consolidation.schiffmanandstein1970:main'
        ]}

setup(
    name=NAME,
    version=VERSION,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url=URL,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    platforms=PLATFORMS,
    packages=PACKAGES,
    data_files=DATA_FILES,
    install_requires=INSTALL_REQUIRES,
    zip_safe=ZIP_SAFE,
    test_suite=TEST_SUITE,
    tests_require=TESTS_REQUIRE,
    package_data=PACKAGE_DATA,
    ext_modules=EXT_MODULES,
    entry_points=ENTRY_POINTS,
    )
########################################################

#maybe use:
#from distutils.core import setup
# why different? http://stackoverflow.com/questions/6344076/differences-between-distribute-distutils-setuptools-and-distutils2






#find all .f90 files
#files = get_package_data('geotecha',['.f90', '.f95','.F90', '.F95'])


#ext_modules = [Extension(osp.splitext(osp.basename(v))[0], v) for v in get_package_data('geotecha',['.f90', '.f95','.F90', '.F95'])]















#!/usr/bin/env python
## setup.py copied from https://github.com/numpy/numpy/blob/master/setup.py
#
#from __future__ import division, print_function
#
#DOCLINES = __doc__.split("\n")
#
#import os
#import shutil
#import sys
#import re
#import subprocess
#
#if sys.version_info[:2] < (2, 6) or (3, 0) <= sys.version_info[0:2] < (3, 2):
#    raise RuntimeError("Python version 2.6, 2.7 or >= 3.2 required.")
#
#if sys.version_info[0] >= 3:
#    import builtins
#else:
#    import __builtin__ as builtins
#
#CLASSIFIERS = """\
#Development Status :: 1 - Planning
#License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)
#Programming Language :: Python :: 2.7
#Topic :: Scientific/Engineering :: Geotechnical Engineering
#Topic :: Scientific/Engineering
#"""
#
#NAME = 'geotecha'
#MAINTAINER = "Dr Rohan Walker"
#MAINTAINER_EMAIL = "rtrwalker@gmail.com"
#DESCRIPTION = DOCLINES[0]
#LONG_DESCRIPTION = "\n".join(DOCLINES[2:])
#URL = "https://github.com/rtrwalker/geotecha.git"
#DOWNLOAD_URL = "https://github.com/rtrwalker/geotecha.git"
#LICENSE = 'GNU General Public License v3 or later (GPLv3+)'
#CLASSIFIERS = [_f for _f in CLASSIFIERS.split('\n') if _f]
#AUTHOR = "Dr Rohan Walker"
#AUTHOR_EMAIL = "rtrwalker@gmail.com"
#PLATFORMS = ["Windows"]#, "Linux", "Solaris", "Mac OS-X", "Unix"]
#MAJOR = 0
#MINOR = 1
#MICRO = 0
#ISRELEASED = False
#VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
#
## Return the git revision as a string
#def git_version():
#    def _minimal_ext_cmd(cmd):
#        # construct minimal environment
#        env = {}
#        for k in ['SYSTEMROOT', 'PATH']:
#            v = os.environ.get(k)
#            if v is not None:
#                env[k] = v
#        # LANGUAGE is used on win32
#        env['LANGUAGE'] = 'C'
#        env['LANG'] = 'C'
#        env['LC_ALL'] = 'C'
#        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
#        return out
#
#    try:
#        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
#        GIT_REVISION = out.strip().decode('ascii')
#    except OSError:
#        GIT_REVISION = "Unknown"
#
#    return GIT_REVISION
#
## BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
## update it when the contents of directories change.
#if os.path.exists('MANIFEST'): os.remove('MANIFEST')
#
## This is a bit hackish: we are setting a global variable so that the main
## numpy __init__ can detect if it is being loaded by the setup routine, to
## avoid attempting to load components that aren't built yet. While ugly, it's
## a lot more robust than what was previously being used.
######builtins.__NUMPY_SETUP__ = True
#
#
#def write_version_py(filename='{pkg}/version.py'.format(pkg=NAME)):
#    cnt = """
## THIS FILE IS GENERATED FROM {pkg} SETUP.PY
#short_version = '%(version)s'
#version = '%(version)s'
#full_version = '%(full_version)s'
#git_revision = '%(git_revision)s'
#release = %(isrelease)s
#
#if not release:
#version = full_version
#""".format(pkg=NAME.upper())
#    # Adding the git rev number needs to be done inside write_version_py(),
#    # otherwise the import of numpy.version messes up the build under Python 3.
#    FULLVERSION = VERSION
#    if os.path.exists('.git'):
#        GIT_REVISION = git_version()
#    elif os.path.exists('{pkg}/version.py'.format(pkg=NAME)):
#        # must be a source distribution, use existing version file
#        try:
##            from numpy.version import git_revision as GIT_REVISION
#            GIT_REVISION =__import__('{pkg}.version'.format(pkg=NAME), globals(), locals(), ['git_revision'], -1).git_revision
#
#        except ImportError:
#            raise ImportError("Unable to import git_revision. Try removing " \
#                              "{pkg}/version.py and the build directory " \
#                              "before building.".format(pkg=NAME))
#    else:
#        GIT_REVISION = "Unknown"
#
#    if not ISRELEASED:
#        FULLVERSION += '.dev-' + GIT_REVISION[:7]
#
#    a = open(filename, 'w')
#    try:
#        a.write(cnt % {'version': VERSION,
#                       'full_version' : FULLVERSION,
#                       'git_revision' : GIT_REVISION,
#                       'isrelease': str(ISRELEASED)})
#    finally:
#        a.close()
#
#def configuration(parent_package='',top_path=None):
#    from numpy.distutils.misc_util import Configuration
#
#    config = Configuration(None, parent_package, top_path)
#    config.set_options(ignore_setup_xxx_py=True,
#                       assume_default_configuration=True,
#                       delegate_options_to_subpackages=True,
#                       quiet=True)
#
#    config.add_subpackage(NAME)
#
#    config.get_version('{pkg}/version.py'.format(pkg=NAME)) # sets config.version
#
#    return config
#
#def setup_package():
#
#    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
#    old_path = os.getcwd()
#    os.chdir(src_path)
#    sys.path.insert(0, src_path)
#
#    # Rewrite the version file everytime
#    write_version_py()
#
#    # Run build
#    from numpy.distutils.core import setup
#
#    try:
#        setup(
#            name=NAME,
#            maintainer=MAINTAINER,
#            maintainer_email=MAINTAINER_EMAIL,
#            description=DESCRIPTION,
#            long_description=LONG_DESCRIPTION,
#            url=URL,
#            download_url=DOWNLOAD_URL,
#            license=LICENSE,
#            classifiers=CLASSIFIERS,
#            author=AUTHOR,
#            author_email=AUTHOR_EMAIL,
#            platforms=PLATFORMS,
#            configuration=configuration())
#    finally:
#        del sys.path[0]
#        os.chdir(old_path)
#    return
#
#if __name__ == '__main__':
#    setup_package()















