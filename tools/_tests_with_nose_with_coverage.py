"""run nose tests on <package-name> with coverage"""


from __future__ import print_function, division
import os
import nose
import importlib


def main():
    """run nose tests with coverage on "<package-name>" where <package-name is ".." dir"""

    path = os.path.abspath(os.path.join('..'))

    package_name = os.path.basename(path)

    try:
        my_module = importlib.import_module(package_name)
    except ImportError:
        raise ImportError('Cound not import {} so cannot '
                          'run nose tests'.format(package_name))

    # need to change the working directory to the installed package
    # otherwise nose will just find <package-name> based on the current
    # directory
    cwd = os.getcwd()
    package_path = os.path.dirname(my_module.__file__)
    os.chdir(package_path)

#    print(os.path.join(cwd, 'cover'))
    print('nose tests with coverage on "{}" package.'.format(package_name))
    #'nose ignores 1st argument http://stackoverflow.com/a/7070571/2530083'
    nose.main(argv=['nose_ignores_1st_arg',
                    '-v',
                    '--with-doctest',
                    '--doctest-options=+ELLIPSIS',
                    '--doctest-options=+IGNORE_EXCEPTION_DETAIL'
                    '--with-coverage',
                    '--cover-erase',
                    '--cover-package=geotecha',
                    '--cover-tests',
                    '--cover-html',
                    '--cover-html-dir={}'.format(os.path.join(cwd, 'cover'))])


    os.chdir(cwd)


if __name__ == '__main__':
    main()
