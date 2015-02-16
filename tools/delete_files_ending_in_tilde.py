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
"""
Script/Function to delete all files in current directory tree ending in '~'.

for usage see :func:`main` or at the command line enter::
    python delete_files_ending_in_tilde.py -h

"""

#see http://pythonhosted.org/sphinxcontrib-programoutput/
#for sphinx documentation stuff


def main(argv=None):
    """Delete all files in current working directory tree ending in '~'.

    When I use Vim on windows, e.g. I open "house.rst", then a file
    "house.rst~" is created.  As far as I can see these files are useless and
    annoying.  This function/script gets rid of them.

    Parameters
    ----------
    argv : list of str, optional
        list of command line options (see usage below) (default=None,
        files will be deleted from the current working directory and its sub
        directories).  Example is ['-v','dirs','path/to/dir']

    Notes
    -----

    usage: delete_files_ending_in_tilde.py [-h] [-v] [-n]
                                       [-d FOLDER_PATH [FOLDER_PATH ...]]

    optional arguments:
      -h, --help            show this help message and exit
      -v, --verbose         print list of filenames [default: don't print
                            filenames]
      -n, --no_recursion    only search current directory [default: search sub-
                            directories]
      -d FOLDER_PATH [FOLDER_PATH ...], --dirs FOLDER_PATH [FOLDER_PATH ...]
                            Directories to search [default: current working
                            directory]

    """
    import os, re, sys, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="print list of filenames "
                        "[default: don't print filenames]")

    parser.add_argument('-n', '--no_recursion', action='store_true',
                        help="only search current directory "
                        "[default: search sub-directories]")

    parser.add_argument('-d', '--dirs', metavar='FOLDER_PATH', type=str,
                        nargs='+', default=[os.getcwd()],
                        help="directories to search "
                        "[default: current working directory")

    if argv == None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)

   # print args
    pattern = ".+~$"
    matching_files = set()

    for mypath in args.dirs:

        if args.no_recursion:
            for file in [x for x in os.listdir(mypath) if re.match(pattern, x)]:
                matching_files.add(os.path.join(mypath, file))
        else:
            for root, dirs, files in os.walk(mypath):
                for file in [x for x in files if re.match(pattern, x)]:
                    matching_files.add(os.path.join(root, file))

    if args.verbose:
        print('{:d} filenames matching "{:s} found and removed:'.format((len(matching_files), pattern)))
        for file in matching_files:
            print(file)

    for file in matching_files:
        print(("name: " + file))
        #os.remove(os.path.join(root, file))


if __name__ == '__main__':
    main()#