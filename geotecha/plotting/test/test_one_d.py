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
"""Some test routines for the one_d module



"""
from __future__ import division, print_function

from nose import with_setup
from nose.tools.trivial import assert_almost_equal
from nose.tools.trivial import assert_equal
from nose.tools.trivial import assert_raises
from nose.tools.trivial import ok_

from math import pi
import numpy as np
from geotecha.piecewise.piecewise_linear_1d import PolyLine

from geotecha.plotting.one_d import rgb_shade
from geotecha.plotting.one_d import rgb_tint

def test_rgb_shade():
    """some tests for rgb_shade"""
#    rgb_shade(rgb, factor=1, scaled=True)
    ok_(np.allclose(rgb_shade((0.4, 0.8, 0.2)), (0.4, 0.8, 0.2)))
    
    ok_(np.allclose(rgb_shade((0.4, 0.8, 0.2), factor=0.5),
                        (0.2, 0.4, 0.1)))
    
#    assert_equal(rgb_shade((0.4, 0.8, 0.2)), (0.4, 0.8, 0.2))                    

    assert_raises(ValueError, rgb_shade,(0.5,0.5,0.5),factor=1.5)
    assert_raises(ValueError, rgb_shade,(0.5,0.5,0.5),factor=-0.5)

def test_rgb_tint():
    """some tests for rgb_tint"""
#    rgb_tint(rgb, factor=1, scaled=True)
    ok_(np.allclose(rgb_tint((0.4, 0.8, 0.2)), (0.4, 0.8, 0.2)))
    
    ok_(np.allclose(rgb_tint((0.4, 0.8, 0.2), factor=0.5),
                        (0.7, 0.9, 0.6)))
    
#    assert_equal(rgb_tint((0.4, 0.8, 0.2)), (0.4, 0.8, 0.2))                    

    assert_raises(ValueError, rgb_tint,(0.5,0.5,0.5),factor=1.5)
    assert_raises(ValueError, rgb_tint,(0.5,0.5,0.5),factor=-0.5)

    ok_(np.allclose(rgb_tint((155, 205, 55), factor=0.5, scaled=False),
                        (205, 230, 155)))