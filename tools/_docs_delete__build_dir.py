"""Delete the docs/_build directory"""


from __future__ import print_function, division
import shutil
import os


def main():
    r"""Delete '..\docs\_build directory' if it exists"""

    path = os.path.abspath(os.path.join('..', 'docs', '_build'))
    if os.path.isdir(path):
        print('"{}" directory exists and will be deleted'.format(path))
        shutil.rmtree(path)
    else:
        print('"{}" directory does not exists.  '
              'No need for deletion.'.format(path))


if __name__ == '__main__':
    main()
