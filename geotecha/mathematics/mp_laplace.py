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

"""this module implements numerical inverse laplace transform using mpmath"""

from __future__ import print_function, division

import numpy as np

try:
    import mpmath
except ImportError:
    try:
        import sympy.mpmath as mpmath
    except ImportError:
        raise ImportError("No mpmath module can be found."
                          "Checked mpmath and sympy.mpmath")

tan = np.frompyfunc(mpmath.tan, 1, 1)
sin = np.frompyfunc(mpmath.sin, 1, 1)
exp = np.frompyfunc(mpmath.exp, 1, 1)

class Talbot(object):
    """numerical inverse laplace transform using mpmath for high precision

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
    dps : int, optional
        mpmath.mp.dps.  default dps=None i.e. use what exists usually 15.
        note  that this changes the global dps value

    Notes
    -----
    laplace.py with mpmath
        appropriate for high precision

    Talbot suggested that the Bromwich line be deformed into a contour that
    begins and ends in the left half plane, i.e., z infinity at both ends.
    Due to the exponential factor the integrand decays rapidly
    on such a contour. In such situations the trapezoidal rule converge
    extraordinarily rapidly.

    Shift contour to the right in case there is a pole on the positive real
    axis : Note the contour will not be optimal since it was originally
    devoloped for function with singularities on the negative real axis
    For example take F(s) = 1/(s-1), it has a pole at s = 1, the contour needs
    to be shifted with one unit, i.e shift  = 1.

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
    geotecha.mathematics.laplace.Talbot : numerical inverse laplace without mpmath;
        less precision

    """
    def __init__(self, f, n=24, shift=0.0, dps = None):

        self.f = f
        self.n = n + n % 2
        self.shift = shift
        if not dps is None:
            mpmath.mp.dps=dps

    @property
    def dps(self):
        """Get the dps property."""
        return mpmath.mp.dps

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
        inv_laplace : mpmath.mpf or np.array of mpmath.mpf
            numerical inverse laplace transform at time t

        """

        t = np.atleast_1d(t)[np.newaxis, :]

        if np.any(t==0):
            raise ValueError('Inverse transform can not be calculated for t=0')

        #   Initiate the stepsize
        h = 2*mpmath.pi/self.n;

        c1 = mpmath.mpf('0.5017')
        c2 = mpmath.mpf('0.6407')
        c3 = mpmath.mpf('0.6122')
        c4 = mpmath.mpc('0','0.2645')
        half = mpmath.mpc('0.5')
        one = mpmath.mpc('1.0')

        theta = (-mpmath.pi + (np.arange(self.n)*one + half)*h)[:, np.newaxis]
        z = self.shift + self.n/t*(c1*theta/tan(c2*theta) - c3 + c4*theta)
        dz = self.n/t * (-c1*c2*theta/sin(c2*theta)**2 + c1/tan(c2*theta)+c4)
        inv_laplace = (exp(z * t) * self.f(z, *args) * dz).sum(axis=0)
        inv_laplace *= h / (2j * mpmath.pi)

        if len(inv_laplace)==1:
            return inv_laplace[0].real
        else:
            return np.array([getattr(v, 'real') for v in inv_laplace])


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])

