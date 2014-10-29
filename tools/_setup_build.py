"""build from the setup.py file"""


from __future__ import print_function, division
import os
import subprocess
import shlex


def main():
    """run "python setup.py build --compiler=mingw32" in ".." directory"""
    cwd = os.getcwd()
    path = os.path.join('..')
    os.chdir(path)

    command_line = 'python setup.py build --compiler=mingw32'

    print('running "{}" in "{}".'.format(command_line, os.getcwd()))

    subprocess.call(shlex.split(command_line))

    os.chdir(cwd)

if __name__ == '__main__':
    main()
