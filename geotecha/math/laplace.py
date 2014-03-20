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

"""this module implements numerical inverse laplace transform"""

from __future__ import print_function, division

import numpy as np

def cot(phi):
    return 1/np.tan(phi)

def csc(phi):
    return 1.0/np.sin(phi)

class Talbot(object):
    """numerical inverse laplace transform using mpmath

    Attributes
    ----------
    f : function or method
        function to perform inverse Laplace transform on. Function should be
        vectorised.
    n : even int, optional
        number of integration points. if n is even it will be rounded up to
        nearest even number default n = 24
    shift : float
        Shift contour to the right in case there is a pole on the positive
        real axis. default shift=0.0

    Notes
    -----
    Talbot suggested that the Bromwich line be deformed into a contour that
    begins and ends in the left half plane, i.e., z infinity at both ends.
    Due to the exponential factor the integrand decays rapidly
    on such a contour. In such situations the trapezoidal rule converge
    extraordinarily rapidly.

    Shift contour to the right in case there is a pole on the positive real axis : Note the contour will
    not be optimal since it was originally devoloped for function with
    singularities on the negative real axis
    For example take F(s) = 1/(s-1), it has a pole at s = 1, the contour needs to be shifted with one
    unit, i.e shift  = 1.

    References
    ----------
    Code adapted (vectorised, args added) from [1]_ and [2]_ (including much
    of the text taken verbatim). Algorithm from [3]_:

    .. [1] Created by Fernando Damian Nieuwveldt, 25 October 2009,
           fdnieuwveldt@gmail.com, http://code.activestate.com/recipes/576934-numerical-inversion-of-the-laplace-transform-using/
    .. [2] Adapted to mpmath and classes by Dieter Kadelka, 27 October 2009,
           Dieter.Kadelka@kit.edu, http://code.activestate.com/recipes/578799-numerical-inversion-of-the-laplace-transform-with-/
    .. [3] L.N.Trefethen, J.A.C.Weideman, and T.Schmelzer. Talbot quadratures
           and rational approximations. BIT. Numerical Mathematics,
           46(3):653 670, 2006.

    See Also
    --------
    `geotecha.math.mp_laplace.Talbot` : higher precision numerical inverse laplace

    """

    def __init__(self, f, n=24, shift=0.0):

        self.f = f
        self.n = n + n % 2
        self.shift = shift

    def __call__(self, t, args=()):
        """Numerical inverse laplace transform of F at various time t.

        Parameters
        ----------
        t : single value or np.array of float
            time values to evaluate inverse laplace at
        args : tuple, optional
            additional arguments to pass to F. default args=()

        Returns
        -------
        inv_laplace : float or 1d array of float
            numerical inverse laplace transform at time t

        """

        t = np.atleast_1d(t)[np.newaxis, :]

        if np.any(t==0):
            raise ValueError('Inverse transform can not be calculated for t=0')

        #   Initiate the stepsize
        h = 2*np.pi/self.n;

        theta = (-np.pi + (np.arange(self.n)+1./2)*h)[:, np.newaxis]
        z = self.shift + self.n/t*(0.5017*theta*cot(0.6407*theta) - 0.6122 + 0.2645j*theta)
        dz = self.n/t*(-0.5017*0.6407*theta*(csc(0.6407*theta)**2)+0.5017*cot(0.6407*theta)+0.2645j)
        inv_laplace = (np.exp(z * t) * self.f(z, *args) * dz).sum(axis=0)
        inv_laplace *= h / (2j * np.pi)

        if len(inv_laplace)==1:
            return inv_laplace[0].real
        else:
            return inv_laplace.real

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])

