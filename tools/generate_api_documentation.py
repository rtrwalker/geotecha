# geotecha - A software suite for geotechncial engineering
# Copyright (C) 2018  Rohan T. Walker (rtrwalker@gmail.com)
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
"""
Script/Function to auto api document geotecha python package.

for usage see :func:`main`.

"""


import os
import shutil
import subprocess
import re
import shlex
import sys

#these vars will be passed to sphinx-quickstart and conf.py
project_name = "geotehca"
authors = "Rohan Walker"
project_version = "0.1"
project_release = "0.1.1"

extensions = ['numpydoc', 'sphinx.ext.autosummary']
exclude_patterns = ['setup.py','test*','**/*test*']


def main():
    """Script/Function to auto api document geotecha python package.

    Generate documentation for a python package using sphinx. Documentation
    will be created in a folder called geotecha\docs\auto.  Must be run in
    the geotecha/tools directory.

    1. runs sphinx-quickstart using seleced module level variables
    2. modifies the sphinx conf.py file
    3. runs make html to created the docs
    4. opens resulting html file

    """

    cwd = os.getcwd()
    dirname, basename = os.path.split(cwd)
    if not (basename == 'tools' and
            os.path.basename(dirname) == 'geotecha'):
        print('script will only work from .../geotecha/tools. Script aborted.')
        sys.exit(1)

    os.chdir(os.path.join(os.path.pardir,'docs'))

    cwd = os.path.join(os.getcwd(), 'auto')
    if os.path.isdir(cwd):
        shutil.rmtree(cwd)
    os.mkdir(cwd)
    os.chdir(cwd)



    c = subprocess.Popen('sphinx-quickstart', stdin = subprocess.PIPE)
    c.stdin.write('\n') #Name prefix for templates and static dir [_]:
    c.stdin.write('y\n')#Separate source and build directories (y/N) [n]:
    c.stdin.write('\n')#Name prefix for templates and static dir [_]:
    c.stdin.write(project_name + '\n')#Project name:
    c.stdin.write(authors + '\n')#Author name(s):

    c.stdin.write(project_version + '\n')#Project version:
    c.stdin.write(project_release + '\n')#Project release [0.1]:
    c.stdin.write('\n')#Source file suffix [.rst]:
    c.stdin.write('\n')#Name of your master document (without suffix) [index]:
    c.stdin.write('y\n')#Do you want to use the epub builder (y/N) [n]:
    c.stdin.write('y\n')#autodoc: automatically insert docstrings from modules (y/N) [n]:
    c.stdin.write('\n')#doctest: automatically test code snippets in doctest blocks (y/N) [n]:
    c.stdin.write('\n')##intersphinx: link between Sphinx documentation of different projects (y/N) [n]:
    c.stdin.write('y\n')#todo: write "todo" entries that can be shown or hidden on build (y/N) [n]:
    c.stdin.write('y\n')#coverage: checks for documentation coverage (y/N) [n]:
    c.stdin.write('\n')#pngmath: include math, rendered as PNG images (y/N) [n]:
    c.stdin.write('y\n')#mathjax: include math, rendered in the browser by MathJax (y/N) [n]:
    c.stdin.write('\n')#ifconfig: conditional inclusion of content based on config values (y/N) [n]:
    c.stdin.write('\n')#viewcode: include links to the source code of documented Python objects (y/N) [n]:
    c.stdin.write('y\n')#Create Makefile? (Y/n) [y]:
    c.stdin.write('y\n')#Create Windows command file? (Y/n) [y]:
    c.communicate()  #wiat untill subprocess finsihes

    os.chdir(os.path.join(os.getcwd(),'source'))


    #add numpy
    file = 'conf.py'

    try: #ensure file exists before trying to exit
        f = open(file, 'r')
    except IOError:
        print('{:s} doesnt exist'.format(file)); sys.exit(1)

    filestr = f.read()
    f.close()

    #autodoc stuff
    pattern = "#sys.path.insert(0, os.path.abspath('.'))"
    repl = 'sys.path.insert(0, os.path.abspath(os.pardir))'
    repl = 'sys.path.insert(0, os.path.abspath(os.path.join(os.pardir, os.pardir)))'
    pattern = '(' + re.escape(pattern) + '\n)' #re.escape adds all the escape characters to a string
    repl = '\g<1>' + repl + '\n'
    filestr = re.sub(pattern, repl, filestr)


    #extensions
    pattern = "(extensions\s=\s\[.*)(\]\n)"
    repl = "\g<1>,'numpydoc', 'sphinx.ext.autosummary'\g<2>"
    repl = "\g<1>, " + ", ".join(["'" + v + "'" for v in extensions]) + "\g<2>"
    filestr = re.sub(pattern, repl, filestr)

    #exclusions:
    pattern = '(exclude_patterns\s=\s\[)(\]\s*)'
    repl = "\g<1>'setup.py','test*'\g<2>"
    repl = "\g<1>" + ", ".join(["'" + v + "'" for v in exclude_patterns]) + "\g<2>"
    filestr = re.sub(pattern, repl, filestr)

    #autosummary
    filestr += "\nautosummary_generate = True\n"

    f = open(file, 'w')
    f.write(filestr)
    f.close()

    os.chdir(os.pardir)#auto folder


    #command_line ='sphinx-apidoc -A "'+ authors + r'" -o \source ..\'
    command_line ='sphinx-apidoc -A "Rohan Walker" -f -o \source ' + '"' + os.path.dirname(os.path.dirname(os.getcwd())) + '"'
    print(command_line)
    print(shlex.split(command_line))
    subprocess.call(shlex.split(command_line))

    command_line ='make html'
    subprocess.call(shlex.split(command_line))


    command_line = os.path.join(os.getcwd(),'build', 'html', 'index' + os.extsep + 'html')
    if sys.platform.startswith('win'):
        os.startfile(command_line)


if __name__ == '__main__':
    main()
