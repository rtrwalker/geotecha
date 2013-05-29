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

from setuptools import setup, find_packages
#maybe use:
#from distutils.core import setup
# why different? http://stackoverflow.com/questions/6344076/differences-between-distribute-distutils-setuptools-and-distutils2

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='geotecha',
      version='0.1',
      description='A software suite for geotechncial engineering',
      long_description=readme(),
      classifiers=[
        'Development Status :: 1 - Planning',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Geotechnical Engineering',
        'Topic :: Scientific/Engineering'
      ],
      keywords='',
      url='https://github.com/rtrwalker/geotecha.git',
      author='Rohan Walker',
      author_email='rtrwalker@gmail.com',
      license='GNU General Public License v3 or later (GPLv3+)',
      packages=find_packages(),#packages=['spec1d','tools', 'spec1d.test'],
      data_files=[('', ['LICENSE.txt','README.rst'])],
      install_requires=[],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'])