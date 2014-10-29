"""Build html docs"""


from __future__ import print_function, division
import os
import subprocess
import shlex


def main():
    r"""run "make html" in "..\docs\" directory"""

    cwd = os.getcwd()
    path = os.path.join('..', 'docs')
    os.chdir(path)

    command_line = 'make html'

    print('running "{}" in "{}".'.format(command_line, os.getcwd()))

    subprocess.call(shlex.split(command_line))

    os.chdir(cwd)


if __name__ == '__main__':
    main()
