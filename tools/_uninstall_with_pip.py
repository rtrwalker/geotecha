"""Uninstall package using pip"""

from __future__ import print_function, division
import os
import subprocess
import shlex

def main():
    """run "pip uninstall <package-name>" where <package-name is ".." directory."""
    path = os.path.abspath(os.path.join('..'))

    package_name = os.path.basename(path)

    command_line ='pip uninstall {}'.format(package_name)

    print('running "{}" in "{}".'.format(command_line, os.getcwd()))

    subprocess.call(shlex.split(command_line))


if __name__ == '__main__':
    main()