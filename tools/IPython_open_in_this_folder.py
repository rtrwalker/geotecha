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

"""Script to start Ipython Notebook Server in current working directory

IPython will open with --NotebookManager.save_script=True which means a .py 
version of all notebooks will be saved along with the default .ipynb notebook 
version.

Basically a glorified shortcut.

Assumes ipython.exe is on the path somewhere.

"""


import subprocess


cmd = 'ipython.exe notebook --pylab=inline --NotebookManager.save_script=True'
title = "IPython notebook server"

"""Note if you just use a subprocess.call approach then the program will be 
run entirely in python which is not want you always want.  If you begin your 
command line with 'start' then a new stand alone window will be created.  

'start' expects the first argument to be the title of the window.  Remember 
this as it can stuff you up.. use an empty "" string if in doubt

"""

c = subprocess.Popen('cmd', stdin = subprocess.PIPE)
#c.stdin.write(s + '\n')
#for wierd 1st argument to start see http://stackoverflow.com/questions/154075/using-the-dos-start-command-with-parameters-passed-to-the-started-program
c.stdin.write('start ' + '"' + title + '" ' + cmd + "\n")
