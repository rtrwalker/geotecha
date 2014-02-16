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



def find_n_roots(func, args=(), n=1, x0=0.001, dx=0.001, p=1.0, fsolve_kwargs={}):
    """find the first n roots of a function

    Starting at x0, step along x values by `dx` until sign of `func` changes
    indicating an interval containing a root is found.  Then use
    scipy.optimize.fsolve to determine the root.  The interval `dx` is
    increased by a factor of `p` each time up untill the first root is found;
    subseuquently `dx` not changed.


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
        initial invterval lenght to check for root in. default dx = 0.001
    p : float, optional
        factor to increase dx by up untill first root is found. default p=1.0
    fsolve_kwargs: dict, optional
        dict of kwargs to pass to scipy.optimize.fsolve.
        Default fsolve_kwargs={}


    Returns
    -------
    roots: 1d ndarray
        array of len n containing the first n roots of `func`.


    """

    x0 = x0
    y0 = func(x0, *args)

    roots = np.zeros(n, dtype=float)

    nroots = 0

    while nroots < n:
        x1 = x0 + dx
        y1 = func(x1, *args)

        if y0 * y1 <= 0.0: #root found
            if nroots==0:
                dx /= p
            xguess = x0 - y0 * (x1-x0)/(y1-y0)

            xr, = scipy.optimize.fsolve(func, xguess, args, **fsolve_kwargs)
            roots[nroots]=xr
            nroots += 1

        x0 = x1
        y0 = y1
        if nroots == 0:
            dx *= p

    return roots


