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
"""Testing routines for the root_finding module."""


from __future__ import division, print_function

from nose import with_setup
from nose.tools.trivial import assert_almost_equal
from nose.tools.trivial import assert_raises
from nose.tools.trivial import ok_
from numpy.testing import assert_allclose


import math
import numpy as np
from geotecha.piecewise.piecewise_linear_1d import PolyLine

from geotecha.mathematics.root_finding import find_n_roots

from scipy.special import j0



def test_find_n_roots():
    """test for find_n_roots"""
    #find_n_roots(func, args=(), n=1, x0=0.001, dx=0.001, p=1.0, fsolve_kwargs={}):

    assert_allclose(find_n_roots(math.sin, n=3, x0=0.1, dx=0.1, p=1.01),
                    np.arange(1,4)*np.pi, atol=1e-5)

    #bessel roots from  http://mathworld.wolfram.com/BesselFunctionZeros.html
    assert_allclose(find_n_roots(j0, n=3, x0=0.1, dx=0.1, p=1.01),
                    np.array([2.4048, 5.5201, 8.6537]), atol=1e-4)

if __name__ == '__main__':

    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])