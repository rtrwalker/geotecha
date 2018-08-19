# geotecha - A software suite for geotechncial engineering
# Copyright (C) 2018  Rohan T. Walker (rtrwalker@gmail.com)
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
"""Testing routines for the transformations module."""


from __future__ import division, print_function

from nose import with_setup
from nose.tools.trivial import assert_almost_equal
from nose.tools.trivial import assert_raises
from nose.tools.trivial import ok_
from numpy.testing import assert_allclose


from math import pi
import numpy as np
import textwrap
import matplotlib.pyplot as plt
from geotecha.piecewise.piecewise_linear_1d import PolyLine

from geotecha.mathematics.transformations import depth_to_reduced_level


def test_depth_to_reduced_level():
    """test for depth_to_reduced_level"""
    #depth_to_reduced_level(z, H = 1.0, rlzero=None)

    assert_allclose(depth_to_reduced_level(z=1, H = 2.0, rlzero=None), 2,
                    atol = 1e-4)
    assert_allclose(depth_to_reduced_level(z=1, H = 2.0, rlzero=5), 3,
                    atol = 1e-4)

    #numpy
    assert_allclose(depth_to_reduced_level(z=np.array([0.5, 1.0]), H = 2.0,
                                           rlzero=None), np.array([1,2]),
                                           atol = 1e-4)
    assert_allclose(depth_to_reduced_level(z=np.array([0.5, 1.0]), H = 2.0,
                                           rlzero=5.0), np.array([4,3]),
                                           atol = 1e-4)



if __name__ == '__main__':

    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])