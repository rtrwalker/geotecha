"""clean up the install from the setup.py file."""


from __future__ import print_function, division
import os
import subprocess
import shlex


def main():
    """run "python setup.py clean --all" in ".." directory"""

    cwd = os.getcwd()
    path = os.path.join('..')
    os.chdir(path)

    command_line = 'python setup.py clean --all'

    print('running "{}" in "{}".'.format(command_line, os.getcwd()))

    subprocess.call(shlex.split(command_line))

    os.chdir(cwd)

if __name__ == '__main__':
    main()
