# Copyright (C) 2013  Rohan T. Walker (rtrwalker@gmail.com)
# geotecha - A software suite for geotechncial engineering
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
module for rootfinding

"""

from __future__ import print_function, division

import numpy as np
import scipy.optimize



def find_n_roots(func, args=(), n=1, x0=0.001, dx=0.001, p=1.0, max_iter=2000,
                 atol=1e-8, rtol=1e-5, debug=False, fsolve_kwargs={}):
    """find the first n roots of a function

    Root/zero finding can be tempramental.  Small `dx`, `p`=1,  and large
    `max_iter` will probably find the roots/zeros but it may take some time.

    Parameters
    ----------
    func : callable f(x, *args)
        A function that takes at least one argument
    args : tuple, optional
        any extra arguments to `func`
    n : float, optional
        number of roots to find, default n=1
    x0 : float, optional
        An x value less than the first root. This is NOT the initial guess of
        fsolve! default x0 = 0.001
    dx : float, optional
        initial interval lenght to check for root in. default dx = 0.001
    p : float, optional
        factor to increase dx by up untill first root is found. default p=1.0
    max_iter: int, optional
        maximum iterations when searching for an interval containing a root.
        .default = 2000.  Note this is not the same as fsolve maxfev keword
        argument.
    atol, rtol: float, optional
        numpy.allclose parameters.  Used for checking if found the
        same root. default atol = 1e-8, rtol=1e-5
    debug: True/False, optional
        print calculations to stdout. default=False
    fsolve_kwargs: dict, optional
        dict of kwargs to pass to scipy.optimize.fsolve.
        Default fsolve_kwargs={}


    Returns
    -------
    roots: 1d ndarray
        array of len n containing the first n roots of `func`.

    Notes
    -----

    Here is approximately what happens:

     - `func` is evaluated with `x0` and `args` to give y0
     - x is incremented by dx to give x1
     - `func` is evaluated with `x1` and `args` to give y1
     - If y0 and y1 have the same sign then there is no root between x0 and x1.
     - `dx` is multiplied by `p` and the x is incremented again.
     - successive intevals are checked until y0 and y1 are of different sign
       indicating that there is a root somewhere between x0 and x1.
     - `scipy.optimize.fsolve` is used to find the root. If it is the first
       root then the the left end of the interval, the current x0, is used as
       the initial guess for fsolve.  If it is a later root then the guess
       for fsolve is the x-axis intercept of the straight line joining
       (x0, y0) and (x1, y1).
     - Once a root is found, the search begins for the next root which starts
       between the root and the current x1at the current x1 or 0.1. The
       current dx value, the one that contained the root is reduced by a fifth
       and off we go again.

    There are a few checks:

     - At least five intervals must be checked before a possible root
         interval is found.  If less than five occur then we go back to the
         starting point of that roots search and use a reduced dx.
     - There is a maximum limit to the number of iterations used for each root
     - It is possible that the algorithm will, when searching for succesive
       roots, find identical roots.  This is wrong.  The current interval size
       along with the current dx will be reduced and the search will continue.
       If the same root is found five times then the root finding parameters
       are inappropriate and an error message will be raised.



    """
#    debug=True

    dx_zero = dx #starting increment from input
    x0_zero = x0 #starting point from input

    roots = np.zeros(n, dtype=float)

    root_number = 0
    iter_count = 0

    x00 = x0_zero #starting point for current root
    dx0 = dx_zero #starting increment for current root
    x0 = x0
    y0 = func(x0, *args)
    if debug:
        print('Trying to find {0} roots of "{1}" starting at x0={2} '
             'with dx={3}, p={4}.'.format(n, func.__name__, x0, dx, p))
        print('root_number, iter_count, x, y, comment')
        print(root_number, 0, x0, y0, 'start')
    while root_number < n:
        if iter_count > max_iter:
            raise ValueError("Exceeded {0} iterations in trying to find "
                            "root #{1} of {2} for the function '{3}'. "
                            "Consider changing x0,dx, and to better capture "
                            "the root.  Currently x0={4:.4g}, dx={5:.4g}, "
                            "p={6}. Roots already found are: {7}"

                            "".format(max_iter,
                                      root_number,
                                      n,
                                      func.__name__,
                                      x0_zero,
                                      dx_zero,
                                      p,
                                      [roots[v] for v in range(root_number)]))
        iter_count += 1
        x1 = x0 + dx
        y1 = func(x1, *args)

        if debug:
            print(root_number, iter_count, x1, y1,"")
        if y0 * y1 <= 0.0: #root found

            if debug:
                print(root_number, iter_count, x1, y1, 'possible root interval')
            if iter_count < 5:
                #too few steps to reach possible root; go back to start and
                # use a smaller increment
                if debug:
                    print(root_number, iter_count, x1, y1,
                          'too few steps to reach possible root. '
                          'Trying again with smaller dx')
                x0 = x00
                y0 = func(x0, *args)
                iter_count = 0
                dx *= 0.2
                continue
            if root_number==0:
                dx /= p
                xguess = x0
            else:
                xguess = x0 - y0 * (x1-x0)/(y1-y0)

            xr, = scipy.optimize.fsolve(func, xguess, args, **fsolve_kwargs)

            if root_number==0:
                if np.allclose(xr, roots[root_number-1], atol=atol, rtol=rtol):
                    #found same root
                    if debug:
                        print(root_number, iter_count, xr, func(xr, *args), 'found same root. reducing dx')
                    iter_same +=1
                    if iter_same > 5:
                        raise ValueError(
                            "Found same root {0} times while trying to find "
                            "root #{1} of {2} for the function '{3}'. "
                            "Consider changing x0, dx, and to better capture "
                            "the root.  Currently x0={4:.4g}, dx={5:.4g}, "
                            "p={6}. Roots already found are: {7}"
                                "".format(iter_same,
                                          root_number + 1,
                                          n,
                                          func.__name__,
                                          x0_zero,
                                          dx_zero,
                                          p,
                                          [roots[v] for v in range(root_number)]))
                    iter_count = iter_count-2
                    dx*=0.2
                    continue
            if debug:
                print(root_number, iter_count, xr, func(xr, *args), 'found root')

            roots[root_number] = xr

            #reset dx
            dx*=0.5
            dx0 = dx
            #move right
            if not np.allclose(xr, x1, atol=atol, rtol=rtol):
                x1 = xr+(x1-xr) * 0.1
                y1 = func(x1, *args)
            x00 = x1

            #reset iters for next root
            iter_count=0
            iter_same=0


#            if root_number >= 2:
#                dx = (roots[root_number] - roots[root_number-1]) / 20
            root_number += 1
            if debug:
                if root_number < n:
                    print(root_number, iter_count, x1, y1,
                          'start root#{}'.format(root_number))
        x0 = x1
        y0 = y1
#        if root_number == 0:
        dx *= p


    return roots


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])
