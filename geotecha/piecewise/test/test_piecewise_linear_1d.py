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
module for piecewise 1d linear relationships

"""
from __future__ import division, print_function

from nose import with_setup
from nose.tools.trivial import assert_almost_equal
from nose.tools.trivial import assert_raises
from nose.tools.trivial import ok_
from nose.tools.trivial import assert_false
from nose.tools.trivial import assert_equal
#from nose.tools.trivial import assertSequenceEqual
import unittest

from numpy.testing import assert_allclose

from math import pi
import numpy as np

from geotecha.piecewise.piecewise_linear_1d import has_steps
from geotecha.piecewise.piecewise_linear_1d import is_initially_increasing
from geotecha.piecewise.piecewise_linear_1d import non_decreasing
from geotecha.piecewise.piecewise_linear_1d import non_increasing
from geotecha.piecewise.piecewise_linear_1d import strictly_decreasing
from geotecha.piecewise.piecewise_linear_1d import strictly_increasing

from geotecha.piecewise.piecewise_linear_1d import start_index_of_steps
from geotecha.piecewise.piecewise_linear_1d import start_index_of_ramps
from geotecha.piecewise.piecewise_linear_1d import start_index_of_constants
from geotecha.piecewise.piecewise_linear_1d import ramps_constants_steps
from geotecha.piecewise.piecewise_linear_1d import segments_less_than_xi
from geotecha.piecewise.piecewise_linear_1d import segment_containing_xi
from geotecha.piecewise.piecewise_linear_1d import segment_containing_also_segments_less_than_xi
from geotecha.piecewise.piecewise_linear_1d import segment_containing_xi_also_containing_xj
from geotecha.piecewise.piecewise_linear_1d import segments_between_xi_and_xj

from geotecha.piecewise.piecewise_linear_1d import force_strictly_increasing
from geotecha.piecewise.piecewise_linear_1d import force_non_decreasing
from geotecha.piecewise.piecewise_linear_1d import non_increasing_and_non_decreasing_parts
from geotecha.piecewise.piecewise_linear_1d import convert_x1_x2_y1_y2_to_x_y
from geotecha.piecewise.piecewise_linear_1d import interp_x1_x2_y1_y2
from geotecha.piecewise.piecewise_linear_1d import interp_x_y
from geotecha.piecewise.piecewise_linear_1d import remove_superfluous_from_x_y
from geotecha.piecewise.piecewise_linear_1d import interp_xa_ya_multipy_x1b_x2b_y1b_y2b
from geotecha.piecewise.piecewise_linear_1d import avg_x_y_between_xi_xj
from geotecha.piecewise.piecewise_linear_1d import integrate_x_y_between_xi_xj
from geotecha.piecewise.piecewise_linear_1d import avg_x1_x2_y1_y2_between_xi_xj
from geotecha.piecewise.piecewise_linear_1d import integrate_x1_x2_y1_y2_between_xi_xj
from geotecha.piecewise.piecewise_linear_1d import xa_ya_multipy_avg_x1b_x2b_y1b_y2b_between
from geotecha.piecewise.piecewise_linear_1d import integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between
from geotecha.piecewise.piecewise_linear_1d import xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between

from geotecha.piecewise.piecewise_linear_1d import convert_x_y_to_x1_x2_y1_y2
from geotecha.piecewise.piecewise_linear_1d import PolyLine
from geotecha.piecewise.piecewise_linear_1d import polyline_make_x_common

from geotecha.piecewise.piecewise_linear_1d import pinterp_x1_x2_y1_y2
from geotecha.piecewise.piecewise_linear_1d import pinterp_x_y
from geotecha.piecewise.piecewise_linear_1d import pinterp_xa_ya_multipy_x1b_x2b_y1b_y2b
from geotecha.piecewise.piecewise_linear_1d import pavg_x_y_between_xi_xj
from geotecha.piecewise.piecewise_linear_1d import pintegrate_x_y_between_xi_xj
from geotecha.piecewise.piecewise_linear_1d import pavg_x1_x2_y1_y2_between_xi_xj
from geotecha.piecewise.piecewise_linear_1d import pintegrate_x1_x2_y1_y2_between_xi_xj
from geotecha.piecewise.piecewise_linear_1d import pxa_ya_multipy_avg_x1b_x2b_y1b_y2b_between
from geotecha.piecewise.piecewise_linear_1d import pintegrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between
from geotecha.piecewise.piecewise_linear_1d import pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between

from geotecha.piecewise.piecewise_linear_1d import pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between_super

from geotecha.piecewise.piecewise_linear_1d import subdivide_x_y_into_segments
from geotecha.piecewise.piecewise_linear_1d import subdivide_x_into_segments



class test_linear_piecewise(object):
    """Some piecewise distributions for testing"""

    def __init__(self):
        self.two_steps = {'x': [0,  0,  1,  1,  2],
                          'y': [0, 10, 10, 30, 30]}
        self.two_steps_reverse = {'x': [0,  0,  -1,  -1,  -2],
                                  'y': [0, 10,  10,  30,  30]}
        self.two_ramps = {'x': [0,  0.5,  1,  1.5,  2],
                          'y': [0, 10, 10, 30, 30]}
        self.two_ramps_reverse = {'x': [0,  -0.5,  -1,  -1.5,  -2],
                                  'y': [0,  10.0,  10,  30.0,  30]}
        self.two_ramps_two_steps = {'x': [0,  0.4,   0.4,  1,  2.5,  3,  3],
                                    'y': [0,  10.0, 20.0, 20, 30.0, 30, 40]}
        self.two_ramps_two_steps_reverse = {'x': [0,  -0.4,   -0.4,  -1,  -2.5,  -3,  -3],
                                            'y': [0,  10.0, 20.0, 20, 30.0, 30, 40]}
        self.switch_back = {'x': [0, 0.5, 1, 0.75, 1.5, 2],
                            'y': [0, 1.2, 2, 2.25, 3.5, 3]}
        self.switch_back_steps = {'x': [0, 0, 1, 0.75, 0.75, 2],
                            'y': [0, 1.2, 2, 2.25, 3.5, 3]}

    def test_has_steps(self):
        """test some has_steps examples"""

        ok_(has_steps(self.two_steps['x']))
        ok_(has_steps(self.two_steps_reverse['x']))
        assert_false(has_steps(self.two_ramps['x']))
        assert_false(has_steps(self.two_ramps_reverse['x']))
        ok_(has_steps(self.two_ramps_two_steps['x']))
        ok_(has_steps(self.two_ramps_two_steps_reverse['x']))

    def test_strictly_increasing(self):
        """test some strictly_increasing examples"""

        assert_false(strictly_increasing(self.two_steps['x']))
        assert_false(strictly_increasing(self.two_steps_reverse['x']))
        ok_(strictly_increasing(self.two_ramps['x']))
        assert_false(strictly_increasing(self.two_ramps_reverse['x']))
        assert_false(strictly_increasing(self.two_ramps_two_steps['x']))
        assert_false(strictly_increasing(self.two_ramps_two_steps_reverse['x']))
        assert_false(strictly_increasing(self.switch_back['x']))
        assert_false(strictly_increasing(self.switch_back_steps['x']))

    def test_strictly_decreasing(self):
        """test some strictly_decreasing examples"""

        assert_false(strictly_decreasing(self.two_steps['x']))
        assert_false(strictly_decreasing(self.two_steps_reverse['x']))
        assert_false(strictly_decreasing(self.two_ramps['x']))
        ok_(strictly_decreasing(self.two_ramps_reverse['x']))
        assert_false(strictly_decreasing(self.two_ramps_two_steps['x']))
        assert_false(strictly_decreasing(self.two_ramps_two_steps_reverse['x']))
        assert_false(strictly_decreasing(self.switch_back['x']))
        assert_false(strictly_decreasing(self.switch_back_steps['x']))

    def test_non_decreasing(self):
        """test some non_decreasing examples"""

        ok_(non_decreasing(self.two_steps['x']))
        assert_false(non_decreasing(self.two_steps_reverse['x']))
        ok_(non_decreasing(self.two_ramps['x']))
        assert_false(non_decreasing(self.two_ramps_reverse['x']))
        ok_(non_decreasing(self.two_ramps_two_steps['x']))
        assert_false(non_decreasing(self.two_ramps_two_steps_reverse['x']))
        assert_false(non_decreasing(self.switch_back['x']))
        assert_false(non_decreasing(self.switch_back_steps['x']))

    def test_non_increasing(self):
        """test some non_increasing examples"""

        assert_false(non_increasing(self.two_steps['x']))
        ok_(non_increasing(self.two_steps_reverse['x']))
        assert_false(non_increasing(self.two_ramps['x']))
        ok_(non_increasing(self.two_ramps_reverse['x']))
        assert_false(non_increasing(self.two_ramps_two_steps['x']))
        ok_(non_increasing(self.two_ramps_two_steps_reverse['x']))
        assert_false(non_increasing(self.switch_back['x']))
        assert_false(non_increasing(self.switch_back_steps['x']))

    def test_non_increasing_and_non_decreasing_parts(self):
        """test some non_increasing_and_non_decreasing_parts examples"""
        assert_equal(non_increasing_and_non_decreasing_parts(self.two_steps['x']), [range(len(self.two_steps['x'])-1)])
        assert_equal(non_increasing_and_non_decreasing_parts(self.two_ramps_reverse['x']), [range(len(self.two_ramps_reverse['x'])-1)])
        assert_equal(non_increasing_and_non_decreasing_parts(self.switch_back['x']), [[0,1],[2],[3,4]])
        assert_equal(non_increasing_and_non_decreasing_parts(self.switch_back_steps['x']), [[0,1],[2,3],[4]])

        assert_equal(non_increasing_and_non_decreasing_parts(self.two_steps['x'],include_end_point=True), [range(len(self.two_steps['x']))])
        assert_equal(non_increasing_and_non_decreasing_parts(self.two_ramps_reverse['x'],include_end_point=True), [range(len(self.two_ramps_reverse['x']))])
        assert_equal(non_increasing_and_non_decreasing_parts(self.switch_back['x'],include_end_point=True), [[0,1,2],[2,3],[3,4,5]])
        assert_equal(non_increasing_and_non_decreasing_parts(self.switch_back_steps['x'],include_end_point=True), [[0,1,2],[2,3,4],[4,5]])

    def test_force_strictly_increasing(self):
        """test force_strictly_increasing"""
        x, y = force_strictly_increasing(self.two_ramps['x'], eps=0.01)
        ok_(np.all(x==np.array(self.two_ramps['x'])))

        assert_raises(ValueError, force_strictly_increasing, self.switch_back['x'])

        x, y = force_strictly_increasing(self.two_ramps_two_steps['x'], self.two_ramps_two_steps['y'], keep_end_points = True, eps=0.01)
        ok_(np.allclose(x, np.array([0,  0.38,   0.4,  1,  2.5,  2.99,  3])))
        ok_(np.allclose(y, np.array(self.two_ramps_two_steps['y'])))

        x, y = force_strictly_increasing(self.two_ramps_two_steps['x'], self.two_ramps_two_steps['y'], keep_end_points = False, eps=0.01)
        ok_(np.allclose(x, np.array([0,  0.4,   0.41,  1,  2.5,  3,  3.02])))

        x, y = force_strictly_increasing(self.two_ramps_two_steps_reverse['x'], self.two_ramps_two_steps_reverse['y'], keep_end_points = False, eps=0.01)
        ok_(np.allclose(x, np.array([-3, -2.99, -2.5, -1, -0.4, -0.38, 0])))
        ok_(np.allclose(y, np.array(self.two_ramps_two_steps['y'][::-1])))

    def test_force_non_decreasing(self):
        """test force_non_decreasing"""
        x, y = force_non_decreasing(self.two_ramps_two_steps['x'], self.two_ramps_two_steps['y'])
        ok_(np.all(x==np.array(self.two_ramps_two_steps['x'])))
        ok_(np.all(y==np.array(self.two_ramps_two_steps['y'])))

        assert_raises(ValueError, force_non_decreasing, self.switch_back['x'])

        x, y = force_non_decreasing(self.two_ramps_two_steps_reverse['x'], self.two_ramps_two_steps_reverse['y'])
        ok_(np.all(x==np.array([-3, -3, -2.5, -1, -0.4, -0.4, 0])))
        ok_(np.all(y==np.array(self.two_ramps_two_steps['y'][::-1])))


    def test_ramps_constants_steps(self):
        """test_ramps_constants_steps"""

        ramps, constants, steps = ramps_constants_steps(self.two_steps['x'], self.two_steps['y'])
        ok_(np.all(ramps==np.array([])))
        ok_(np.all(constants==np.array([1,3])))
        ok_(np.all(steps==np.array([0,2])))

        ramps, constants, steps = ramps_constants_steps(self.two_ramps['x'], self.two_ramps['y'])
        ok_(np.all(ramps==np.array([0,2])))
        ok_(np.all(constants==np.array([1,3])))
        ok_(np.all(steps==np.array([])))

        ramps, constants, steps = ramps_constants_steps(self.two_ramps_two_steps['x'], self.two_ramps_two_steps['y'])
        ok_(np.all(ramps==np.array([0,3])))
        ok_(np.all(constants==np.array([2,4])))
        ok_(np.all(steps==np.array([1,5])))

#               0      1      2     3     4     5    6
#        {'x': [0,  -0.4,   -0.4,  -1,  -2.5,  -3,  -3],
#         'y': [0,  10.0,   20.0,  20,  30.0, 30, 40]}

    def test_start_index_of_steps(self):
        """test_start_index_of_steps"""
        ok_(np.allclose(start_index_of_steps(**self.two_steps),np.array([0,2])))
        ok_(np.allclose(start_index_of_steps(**self.two_ramps),np.array([])))
        ok_(np.allclose(start_index_of_steps(**self.two_ramps_two_steps),np.array([1,5])))

    def test_start_index_of_ramps(self):
        """test_start_index_of_ramps"""
        ok_(np.allclose(start_index_of_ramps(**self.two_steps),np.array([])))
        ok_(np.allclose(start_index_of_ramps(**self.two_ramps),np.array([0,2])))
        ok_(np.allclose(start_index_of_ramps(**self.two_ramps_two_steps),np.array([0,3])))

    def test_start_index_of_constants(self):
        """test_start_index_of_constants"""
        ok_(np.allclose(start_index_of_constants(**self.two_steps),np.array([1,3])))
        ok_(np.allclose(start_index_of_constants(**self.two_ramps),np.array([1,3])))
        ok_(np.allclose(start_index_of_constants(**self.two_ramps_two_steps),np.array([2,4])))

    def test_segment_containing_xi(self):
        """test_segment_containing_xi"""
        #segment_containing_xi(x, xi, subset = None, choose_max = False)
        ok_(np.allclose(segment_containing_xi(x=self.two_steps['x'], xi=-1, subset=None), np.array([[]])))
        ok_(np.allclose(segment_containing_xi(x=self.two_steps['x'], xi=8, subset=None), np.array([[]])))
        ok_(np.allclose(segment_containing_xi(x=self.two_steps['x'], xi= 0, subset=None), np.array([[]])))
        ok_(np.allclose(segment_containing_xi(x=self.two_steps['x'], xi= 0.2, subset=None), np.array([[]])))

        ok_(all(map(np.allclose,
                    segment_containing_xi(x=self.two_steps['x'], xi=[0,0.2, 1, 2], subset=None),
                    [[],[],[],[]]
                    )))

        ok_(np.allclose(segment_containing_xi(x=self.two_ramps_two_steps['x'], xi=0.2, subset=[0,3]), np.array([[0]])))
        ok_(np.allclose(segment_containing_xi(x=self.two_ramps_two_steps['x'], xi=[0.2,1.2], subset=[0,3]), np.array([[0],[3]])))
        ok_(np.allclose(segment_containing_xi(x=self.two_ramps_two_steps['x'], xi=[0.2,1.2], subset=None), np.array([[0],[3]])))
        ok_(np.allclose(segment_containing_xi(x=self.two_ramps_two_steps['x'], xi=[0.4], subset=None, choose_max=False), np.array([[0]])))
        ok_(np.allclose(segment_containing_xi(x=self.two_ramps_two_steps['x'], xi=[0.4], subset=None, choose_max=True), np.array([[2]])))

        ok_(all(map(np.allclose,
                    segment_containing_xi(x=self.two_ramps_two_steps['x'], xi=[0, 0.2, 0.4], subset=None),
                    [[0],[0],[0]]
                    )))

        ok_(all(map(np.allclose,
                    segment_containing_xi(x=self.two_ramps_two_steps['x'], xi=[0, 0.2, 1.2], subset=None, choose_max=True),
                    [[0],[0],[3]]
                    )))

    def test_segments_less_than_xi(self):
        """test_segments_less_than_xi"""
        #segments_less_than_xi(x, xi, subset = None, or_equal_to = False):
        ok_(np.allclose(segments_less_than_xi(x=self.two_steps['x'], xi=-1, subset=None, or_equal_to=False), np.array([[]])))
        ok_(np.allclose(segments_less_than_xi(x=self.two_steps['x'], xi= 0, subset=None, or_equal_to=False), np.array([[]])))
        ok_(np.allclose(segments_less_than_xi(x=self.two_steps['x'], xi= 0, subset=None, or_equal_to=True), np.array([[0]])))
        ok_(np.allclose(segments_less_than_xi(x=self.two_steps['x'], xi= 8, subset=None, or_equal_to=False), np.array([[0,1,2,3]])))
        ok_(np.allclose(segments_less_than_xi(x=self.two_steps['x'], xi= 1.1, subset=None, or_equal_to=False), np.array([[0,1,2]])))

        ok_(all(map(np.allclose,
                    segments_less_than_xi(x=self.two_steps['x'], xi=[-1,0,8,1.1], subset=None, or_equal_to=False),
                    [[],[],[0,1,2,3],[0,1,2]]
                    )))

    def test_ramps_constants_steps(self):
        """test_ramps_constants_steps"""
        ramps, constants, steps = ramps_constants_steps(**self.two_ramps_two_steps)
        ok_(all(map(np.allclose,ramps, [0,3])))
        ok_(all(map(np.allclose,constants, [2,4])))
        ok_(all(map(np.allclose,steps, [1,5])))


    def test_segment_containing_also_segments_less_than_xi(self):
        """test_segment_containing_also_segments_less_than_xi"""
        #segment_containing_also_segments_less_than_xi(x, y, xi, steps_or_equal_to = True, ramp_const_or_equal_to = False, choose_max = False):
        #TODO: some smaller tests of this with just a single load
        (ramps_less_than_xi, constants_less_than_xi, steps_less_than_xi,
            ramps_containing_xi, constants_containing_xi) = segment_containing_also_segments_less_than_xi(xi = 0.42, **self.two_ramps_two_steps)
        ok_(all(map(np.allclose, ramps_less_than_xi, [[0]])))
        ok_(all(map(np.allclose, constants_less_than_xi, [[]])))
        ok_(all(map(np.allclose, steps_less_than_xi, [1])))
        ok_(all(map(np.allclose, ramps_containing_xi, [[]])))
        ok_(all(map(np.allclose, constants_containing_xi, [2])))

        (ramps_less_than_xi, constants_less_than_xi, steps_less_than_xi,
            ramps_containing_xi, constants_containing_xi) = segment_containing_also_segments_less_than_xi(**{'x': np.array([0,0,10,20]), 'y': np.array([0, -100,-100,-50]), 'xi': np.array([-1,0,1,10,15])})
#        {'x': np.array([0,0,10,20]),
#         'y': np.array([0, -100,-100,-50]),
#         'xi': np.array([-1,0,1,10,15])}
        ok_(all(map(np.allclose, ramps_less_than_xi, [[],[],[],[],[]])))
        ok_(all(map(np.allclose, constants_less_than_xi, [[],[],[],[],[1]])))
        ok_(all(map(np.allclose, steps_less_than_xi, [[],[0],[0],[0],[0]])))
        ok_(all(map(np.allclose, ramps_containing_xi, [[],[],[],[],[2]])))
        ok_(all(map(np.allclose, constants_containing_xi, [[],[],[1],[1],[]])))

    def test_segment_containing_xi_also_containing_xj(self):
        """test_segment_containing_xi_also_containing_xj"""
        #segments_containing_pair(x,pair, subset=None)
        ix1, ix2 = segment_containing_xi_also_containing_xj(x=[0,0.2,0.4,1], xi=0.1, xj=0.2, subset=None)
        ok_(all(map(np.allclose, ix1, [[0]])))
        ok_(all(map(np.allclose, ix2, [[0]])))

        ix1, ix2 = segment_containing_xi_also_containing_xj(x=[0,0.2,0.4,1], xi=[0.1], xj=[0.2], subset=None)
        ok_(all(map(np.allclose, ix1, [[0]])))
        ok_(all(map(np.allclose, ix2, [[0]])))

        ix1, ix2 = segment_containing_xi_also_containing_xj(x=[0,0.2,0.4,1], xi=[0.1, 0.2], xj=[0.2,0.3], subset=None)
        ok_(all(map(np.allclose, ix1, [[0],[1]])))
        ok_(all(map(np.allclose, ix2, [[0],[1]])))

        ix1, ix2 = segment_containing_xi_also_containing_xj(x=[0,0.2,0.4,1], xi=4, xj=9, subset=None)
        ok_(all(map(np.allclose, ix1, [[]])))
        ok_(all(map(np.allclose, ix2, [[]])))

        ix1, ix2 = segment_containing_xi_also_containing_xj(x=[0,0.2,0.4,1], xi=[0.1, 0.2], xj=[0.2,0.3], subset=[])
        ok_(all(map(np.allclose, ix1, [[],[]])))
        ok_(all(map(np.allclose, ix2, [[],[]])))

        ix1, ix2 = segment_containing_xi_also_containing_xj(x=[0,0.2,0.4,1], xi=[0.1, 0.2], xj=[0.2,0.3], subset=[1])
        ok_(all(map(np.allclose, ix1, [[],[1]])))
        ok_(all(map(np.allclose, ix2, [[],[1]])))


    def test_segments_between_xi_and_xj(self):
        """test_segments_between_xi_and_xj"""
        #segments_between_xi_and_xj(x,xi,xj)

        (ix_both, ix1_only, ix2_only, between
        ) = segments_between_xi_and_xj(x=[0,0.2,0.4,1], xi = 0.1, xj = 0.2)
        ok_(all(map(np.allclose, ix_both, [[0]])))
        ok_(all(map(np.allclose, ix1_only, [[]])))
        ok_(all(map(np.allclose, ix2_only, [[]])))
        ok_(all(map(np.allclose, between, [[]])))

        (ix_both, ix1_only, ix2_only, between
        ) = segments_between_xi_and_xj(x=[0,0.2,0.4,1], xi = 0.1, xj = 0.3)
        ok_(all(map(np.allclose, ix_both, [[]])))
        ok_(all(map(np.allclose, ix1_only, [[0]])))
        ok_(all(map(np.allclose, ix2_only, [[1]])))
        ok_(all(map(np.allclose, between, [[]])))

        (ix_both, ix1_only, ix2_only, between
        ) = segments_between_xi_and_xj(x=[0,0.2,0.4,1], xi = 0.1, xj = 0.6)
        ok_(all(map(np.allclose, ix_both, [[]])))
        ok_(all(map(np.allclose, ix1_only, [[0]])))
        ok_(all(map(np.allclose, ix2_only, [[2]])))
        ok_(all(map(np.allclose, between, [[1]])))

        (ix_both, ix1_only, ix2_only, between
        ) = segments_between_xi_and_xj(x=[0,0.2,0.4,1,1.4], xi = [0.1,0.1,0.1], xj = [0.2,0.3,1.2])
        ok_(all(map(np.allclose, ix_both, [[0],[],[]])))
        ok_(all(map(np.allclose, ix1_only, [[],[0],[0]])))
        ok_(all(map(np.allclose, ix2_only, [[],[1],[3]])))
        ok_(all(map(np.allclose, between, [[],[],[1,2]])))

    def test_convert_x1_x2_y1_y2_to_x_y(self):
        """test_convert_x1_x2_y1_y2_to_x_y"""
        #convert_x1_x2_y1_y2_to_x_y(x1, x2, y1, y2)

        #ok_(np.allclose(convert_x1_x2_y1_y2_to_x_y(**{}), ))

        assert_raises(ValueError, convert_x1_x2_y1_y2_to_x_y,
                       **{'x1': [0.0, 0.3, 0.7], 'y1': [1, 1], 'x2':[0.3, 0.7, 1.0], 'y2': [3, 1, 4]})
        assert_raises(ValueError, convert_x1_x2_y1_y2_to_x_y,
                       **{'x1': [0.0, 0.3, 0.7], 'y1': [1, 1, 2], 'x2':[0.3, 0.8, 1.0], 'y2': [3, 1, 4]})

#        ok_(all(map(np.allclose,
#                    convert_x1_x2_y1_y2_to_x_y(**{'x1': [0.0, 0.3, 0.7], 'y1': [1, 1, 2], 'x2':[0.3, 0.7, 1.0], 'y2': [3, 1, 4]}),
#                    ([0,0.3,0.3,0.7,0.7,1], [1,3,1,1,2,4])
#                    )))

        ok_(all(map(np.allclose,
                    convert_x1_x2_y1_y2_to_x_y(**{'x1': [0.0], 'y1': [1], 'x2':[1], 'y2': [3]}),
                    ([0,1], [1,3])
                    )))

        ok_(all(map(np.allclose,
                    convert_x1_x2_y1_y2_to_x_y(**{'x1': [0.0, 0.3], 'y1': [1,1], 'x2':[0.3, 1], 'y2': [1,1]}),
                    ([0,0.3,1], [1,1,1])
                    )))

        ok_(all(map(np.allclose,
                    convert_x1_x2_y1_y2_to_x_y(**{'x1': [0.0, 0.3], 'y1': [1,1], 'x2':[0.3, 1], 'y2': [1,2]}),
                    ([0,0.3,1], [1,1,2])
                    )))
        ok_(all(map(np.allclose,
                    convert_x1_x2_y1_y2_to_x_y(**{'x1': [0.0, 0.3], 'y1': [1,1], 'x2':[0.3, 1], 'y2': [5,2]}),
                    ([0,0.3, 0.3, 1], [1,5,1,2])
                    )))

    def test_convert_x_y_to_x1_x2_y1_y2(self):
        """test_convert_x_y_to_x1_x2_y1_y2"""
        #convert_x_y_to_x1_x2_y1_y2(x,y):
        assert_raises(ValueError, convert_x_y_to_x1_x2_y1_y2,
                       **{'x': [0.0, 0.3, 0.7], 'y': [1, 1]})
        ok_(all(map(np.allclose,
                    convert_x_y_to_x1_x2_y1_y2(**{'x':[0,1], 'y':[1,3]}),
                    ([0.0],[1],[1],[3])
                    )))

        ok_(all(map(np.allclose,
                    convert_x_y_to_x1_x2_y1_y2(**{'x':[0,0.3,1], 'y':[1,1,1]}),
                    ([0.0, 0.3], [0.3, 1], [1,1], [1,1])
                    )))

        ok_(all(map(np.allclose,
                    convert_x_y_to_x1_x2_y1_y2(**{'x':[0,0.3,1], 'y':[1,1,2]}),
                    ([0.0, 0.3], [0.3, 1],[1,1],[1,2])
                    )))
        ok_(all(map(np.allclose,
                    convert_x_y_to_x1_x2_y1_y2(**{'x':[0,0.3, 0.3, 1], 'y':[1,5,1,2]}),
                    ([0.0, 0.3],[0.3, 1],[1,1],[5,2])
                    )))

    def test_interp_x1_x2_y1_y2(self):
        "test_interp_x1_x2_y1_y2"
        ok_(np.allclose(interp_x1_x2_y1_y2(**{'x1': [0.0], 'y1': [10], 'x2':[1], 'y2': [20], 'xi': 0.5}),
                        [15]))
        ok_(np.allclose(interp_x1_x2_y1_y2(**{'x1': [0.0], 'y1': [10], 'x2':[1], 'y2': [20], 'xi': [0.5]}),
                        [15]))

        ok_(np.allclose(interp_x1_x2_y1_y2(**{'x1': [0.0], 'y1': [10], 'x2':[1], 'y2': [20], 'xi': 2}),
                        [20]))
        ok_(np.allclose(interp_x1_x2_y1_y2(**{'x1': [0.0], 'y1': [10], 'x2':[1], 'y2': [20], 'xi': -5}),
                        [10]))

        ok_(np.allclose(interp_x1_x2_y1_y2(**{'x1': [0.0], 'y1': [10], 'x2':[1], 'y2': [20], 'xi': [0.5, 2 ,-5]}),
                        [15,20,10]))

        ok_(np.allclose(interp_x1_x2_y1_y2(**{'x1': [0.0,0.5], 'y1': [10,20], 'x2':[0.5,1], 'y2': [50,60], 'xi': [0.25, 0.75]}),
                        [30, 40]))


        ok_(np.allclose(interp_x1_x2_y1_y2(**{'x1': [1], 'y1': [10], 'x2':[0], 'y2': [20], 'xi': 0.75}),
                        [12.5]))
        ok_(np.allclose(interp_x1_x2_y1_y2(**{'x1': [1], 'y1': [10], 'x2':[0], 'y2': [20], 'xi': 2}),
                        [10]))
        ok_(np.allclose(interp_x1_x2_y1_y2(**{'x1': [1], 'y1': [10], 'x2':[0], 'y2': [20], 'xi': -5}),
                        [20]))
    def test_pinterp_x1_x2_y1_y2(self):
        "test_pinterp_x1_x2_y1_y2"
        ok_(np.allclose(pinterp_x1_x2_y1_y2(**{'a': PolyLine([0.0], [1],[10],[20]), 'xi': [0.5, 2 ,-5]}),
                        [15,20,10]))

    def test_interp_x_y(self):
        """test_interp_x_y"""
        #interp_x_y(x,y,xi, choose_max = False)

        ok_(np.allclose(interp_x_y(**{'x': [0,1], 'y': [10,20], 'xi': [-5]}),
                        [10]))
        ok_(np.allclose(interp_x_y(**{'x': [0,1], 'y': [10,20], 'xi': [0.5]}),
                        [15]))
        ok_(np.allclose(interp_x_y(**{'x': [0,1], 'y': [10,20], 'xi': [8]}),
                        [20]))

        ok_(np.allclose(interp_x_y(**{'x': [0,0.5,0.5,1], 'y': [10,30,40,100], 'xi': 0.5, 'choose_max':False}),
                        [30]))
        ok_(np.allclose(interp_x_y(**{'x': [0,0.5,0.5,1], 'y': [10,30,40,100], 'xi': 0.5, 'choose_max':True}),
                        [40]))
        ok_(np.allclose(interp_x_y(**{'x': [0,0.5,0.5,1], 'y': [10,30,40,100], 'xi': [0.25,0.5,0.75], 'choose_max':False}),
                        [20,30,70]))

        ok_(np.allclose(interp_x_y(**{'x': [1,0], 'y': [20,10], 'xi': [-5]}),
                        [10]))
        ok_(np.allclose(interp_x_y(**{'x': [1,0], 'y': [20,10], 'xi': [0.5]}),
                        [15]))
        ok_(np.allclose(interp_x_y(**{'x': [1,0], 'y': [20,10], 'xi': [8]}),
                        [20]))
        ok_(np.allclose(interp_x_y(**{'x': [1,0.5,0.5,0], 'y': [100,40,30,10], 'xi': [0.25,0.5,0.75]}),
                        [20,40,70]))
    def test_pinterp_x_y(self):
        """test_pinterp_x_y"""
        #interp_x_y(x,y,xi, choose_max = False)

        ok_(np.allclose(pinterp_x_y(**{'a': PolyLine([0,0.5,0.5,1],[10,30,40,100]), 'xi': 0.5, 'choose_max':False}),
                        [30]))

    def test_remove_superfluous_from_x_y(self):
        """test_remove_superfluous_from_x_y"""
        ok_(all(map(np.allclose,
                    remove_superfluous_from_x_y(**{'x': [0.0, 0.5, 1.0], 'y': [0,1,0]}),
                    ([0.0, 0.5, 1.0], [0,1,0])
                    )))
        ok_(all(map(np.allclose,
                    remove_superfluous_from_x_y(**{'x': [0.0, 0.5, 1.0], 'y': [0,1,2]}),
                    ([0.0, 1.0], [0,2])
                    )))

        ok_(all(map(np.allclose,
                    remove_superfluous_from_x_y(**{'x': [0.0, 0.5, 1.0], 'y': [0,1.01,2], 'atol':0.1}),
                    ([0.0, 1.0], [0,2])
                    )))

        ok_(all(map(np.allclose,
                    remove_superfluous_from_x_y(**{'x': [0.0, 0.5, 1.0], 'y': [0,1.01,2], 'atol':0.001}),
                    ([0.0, 0.5, 1.0], [0,1.01,2])
                    )))

        ok_(all(map(np.allclose,
                    remove_superfluous_from_x_y(**{'x': [0.0, 0.5, 1.0, 1.5, 2, 2.5], 'y': [0,1,2,3,2,1], }),
                    ([0.0, 1.5, 2.5], [0,3,1])
                    )))

        ok_(all(map(np.allclose,
                    remove_superfluous_from_x_y(**{'x': [0.0, 0.0, 1.0], 'y': [0,0,1]}),
                    ([0.0, 1.0], [0,1])
                    )))

        ok_(all(map(np.allclose,
                    remove_superfluous_from_x_y(**{'x': [0.0, 0.0, 0, 0.0], 'y': [0,1,2,4]}),
                    ([0.0,0.0], [0,4])
                    )))

        ok_(all(map(np.allclose,
                    remove_superfluous_from_x_y(**{'x': [0.0, 0.0, 0, 0.0], 'y': [0,1,5,3]}),
                    ([0.0,0.0], [0,3])
                    )))

    def test_interp_xa_ya_multipy_x1b_x2b_y1b_y2b(self):
        """test_interp_xa_ya_multipy_x1b_x2b_y1b_y2b"""
        #interp_xa_ya_multipy_x1b_x2b_y1b_y2b(xa, ya, x1b, x2b, y1b, y2b, xai, xbi, achoose_max=False, bchoose_max=True):
        ok_(all(map(np.allclose,
                    interp_xa_ya_multipy_x1b_x2b_y1b_y2b(
                        **{'xa':[0,1] , 'ya':[1,2],
                           'x1b':[4], 'x2b':[5], 'y1b':[2], 'y2b':[4],
                           'xai':0.5, 'xbi':4.5}),
                    [[1.5*3]]
                    )))

        ok_(all(map(np.allclose,
                    interp_xa_ya_multipy_x1b_x2b_y1b_y2b(
                        **{'xa':[0,1] , 'ya':[1,2],
                           'x1b':[4], 'x2b':[5], 'y1b':[2], 'y2b':[4],
                           'xai':-1, 'xbi':8}),
                    [[4]]
                    )))
        ok_(all(map(np.allclose,
                    interp_xa_ya_multipy_x1b_x2b_y1b_y2b(
                        **{'xa':[0,1] , 'ya':[1,2],
                           'x1b':[4], 'x2b':[5], 'y1b':[2], 'y2b':[4],
                           'xai':4, 'xbi':-1}),
                    [[4]]
                    )))

        ok_(np.allclose(
                    interp_xa_ya_multipy_x1b_x2b_y1b_y2b(
                        **{'xa':[0,1] , 'ya':[1,2],
                           'x1b':[4], 'x2b':[5], 'y1b':[2], 'y2b':[4],
                           'xai':[0,0.5,1], 'xbi':[4, 4.5]}),
                    [[2,3,4],[3,4.5,6]]#[[2,3],[3,4.5],[4,6]]
                    ))
        #TODO: do some tests with the keyword arguments, check others.
    def test_pinterp_xa_ya_multipy_x1b_x2b_y1b_y2b(self):
        """test_pinterp_xa_ya_multipy_x1b_x2b_y1b_y2b"""
        #pinterp_xa_ya_multipy_x1b_x2b_y1b_y2b(a,,xai, xbi, achoose_max=False, bchoose_max=True):

        ok_(np.allclose(
                    pinterp_xa_ya_multipy_x1b_x2b_y1b_y2b(
                        **{'a': PolyLine([0,1],[1,2]),
                           'b': PolyLine([4],[5],[2],[4]),
                           'xai':[0,0.5,1], 'xbi':[4, 4.5]}),
                    [[2,3,4],[3,4.5,6]]#[[2,3],[3,4.5],[4,6]]
                    ))

    def test_integrate_x_y_between_xi_xj(self):
        """test_integrate_x_y_between_xi_xj"""
        #integrate_x_y_between_xi_xj(x, y, xi, xj)
        ok_(all(map(np.allclose,
                    integrate_x_y_between_xi_xj(
                        **{'x':[0,1] , 'y':[1,2],
                           'xi':0, 'xj':1}),
                    [1.5]
                    )))
        ok_(all(map(np.allclose,
                    integrate_x_y_between_xi_xj(
                        **{'x':[0,1,2] , 'y':[2,3,2],
                           'xi':0, 'xj':2}),
                    [5]
                    )))
        ok_(all(map(np.allclose,
                    integrate_x_y_between_xi_xj(
                        **{'x':[0,1,2,3] , 'y':[3,4,3,4],
                           'xi':0, 'xj':3}),
                    [10.5]
                    )))
        ok_(all(map(np.allclose,
                    integrate_x_y_between_xi_xj(
                        **{'x':[0,1,2,3] , 'y':[3,4,3,4],
                           'xi':[0,0,0], 'xj':[1,2,3]}),
                    [3.5,7,10.5]
                    )))
        ok_(all(map(np.allclose,
                    integrate_x_y_between_xi_xj(
                        **{'x':[0,1,1,2] , 'y':[5,5,6,6],
                           'xi':0, 'xj':1}),
                    [5.0]
                    )))
        ok_(all(map(np.allclose,
                    integrate_x_y_between_xi_xj(
                        **{'x':[0,1,1,2] , 'y':[5,5,6,6],
                           'xi':1, 'xj':2}),
                    [6.0]
                    )))
        ok_(all(map(np.allclose,
                    integrate_x_y_between_xi_xj(
                        **{'x':[0,1,1,2] , 'y':[5,5,6,6],
                           'xi':0, 'xj':2}),
                    [11.0]
                    )))
        ok_(all(map(np.allclose,
                    integrate_x_y_between_xi_xj(
                        **{'x':[0,1.5,2,4] , 'y':[0,1,2,3],
                           'xi':1, 'xj':3}),
                    [3.416666666667]
                    )))

    def test_pintegrate_x_y_between_xi_xj(self):
        """test_pintegrate_x_y_between_xi_xj"""
        #pintegrate_x_y_between_xi_xj(a, xi, xj)

        ok_(all(map(np.allclose,
                    pintegrate_x_y_between_xi_xj(
                        **{'a': PolyLine([0,1.5,2,4] ,[0,1,2,3]),
                           'xi':1, 'xj':3}),
                    [3.416666666667]
                    )))

    def test_avg_x_y_between_xi_xj(self):
        """test_avg_x_y_between_xi_xj"""
        #avg_x_y_between_xi_xj(x, y, xi, xj)
        ok_(all(map(np.allclose,
                    avg_x_y_between_xi_xj(
                        **{'x':[0,1] , 'y':[1,2],
                           'xi':0, 'xj':1}),
                    [1.5]
                    )))
        ok_(all(map(np.allclose,
                    avg_x_y_between_xi_xj(
                        **{'x':[0,1,2] , 'y':[2,3,2],
                           'xi':0, 'xj':2}),
                    [2.5]
                    )))
        ok_(all(map(np.allclose,
                    avg_x_y_between_xi_xj(
                        **{'x':[0,1,2,3] , 'y':[3,4,3,4],
                           'xi':0, 'xj':3}),
                    [3.5]
                    )))
        ok_(all(map(np.allclose,
                    avg_x_y_between_xi_xj(
                        **{'x':[0,1,2,3] , 'y':[3,4,3,4],
                           'xi':[0,0,0], 'xj':[1,2,3]}),
                    [3.5,3.5,3.5]
                    )))
        ok_(all(map(np.allclose,
                    avg_x_y_between_xi_xj(
                        **{'x':[0,1,1,2] , 'y':[5,5,6,6],
                           'xi':0, 'xj':1}),
                    [5.0]
                    )))
        ok_(all(map(np.allclose,
                    avg_x_y_between_xi_xj(
                        **{'x':[0,1,1,2] , 'y':[5,5,6,6],
                           'xi':1, 'xj':2}),
                    [6.0]
                    )))
        ok_(all(map(np.allclose,
                    avg_x_y_between_xi_xj(
                        **{'x':[0,1,1,2] , 'y':[5,5,6,6],
                           'xi':0, 'xj':2}),
                    [5.5]
                    )))
        ok_(all(map(np.allclose,
                    avg_x_y_between_xi_xj(
                        **{'x':[0,0.75,1,2] , 'y':[0,1,2,3],
                           'xi':0.5, 'xj':1.5}),
                    [1.70833333333]
                    )))
    def test_pavg_x_y_between_xi_xj(self):
        """test_pavg_x_y_between_xi_xj"""
        #pavg_x_y_between_xi_xj(a, xi, xj)

        ok_(all(map(np.allclose,
                    pavg_x_y_between_xi_xj(
                        **{'a': PolyLine([0,0.75,1,2] , [0,1,2,3]),
                           'xi':0.5, 'xj':1.5}),
                    [1.70833333333]
                    )))
    def test_integrate_x1_x2_y1_y2_between_xi_xj(self):
        """test_integrate_x1_x2_y1_y2_between_xi_xj"""
        #integrate_x1_x2_y1_y2_between_xi_xj(x1,x2,y1,y1, xi, xj)
        ok_(all(map(np.allclose,
                    integrate_x1_x2_y1_y2_between_xi_xj(
                        **{'x1':[0],'x2':[1],'y1':[1], 'y2':[2],
                           'xi':0, 'xj':1}),
                    [1.5]
                    )))
        ok_(all(map(np.allclose,
                    integrate_x1_x2_y1_y2_between_xi_xj(
                        **{'x1':[0,1],'x2':[1,2],'y1':[2,3], 'y2':[3,2],
                           'xi':0, 'xj':2}),
                    [5]
                    )))
        ok_(all(map(np.allclose,
                    integrate_x1_x2_y1_y2_between_xi_xj(
                        **{'x1':[0,1,2],'x2':[1,2,3],'y1':[3,4,3], 'y2':[4,3,4],
                           'xi':0, 'xj':3}),
                    [10.5]
                    )))
        ok_(all(map(np.allclose,
                    integrate_x1_x2_y1_y2_between_xi_xj(
                        **{'x1':[0,1,2],'x2':[1,2,3],'y1':[3,4,3], 'y2':[4,3,4],
                           'xi':[0,0,0], 'xj':[1,2,3]}),
                    [3.5,7,10.5]
                    )))
        ok_(all(map(np.allclose,
                    integrate_x1_x2_y1_y2_between_xi_xj(
                        **{'x1':[0,1],'x2':[1,2],'y1':[5,6], 'y2':[5,6],
                           'xi':0, 'xj':1}),
                    [5.0]
                    )))
        ok_(all(map(np.allclose,
                    integrate_x1_x2_y1_y2_between_xi_xj(
                        **{'x1':[0,1],'x2':[1,2],'y1':[5,6], 'y2':[5,6],
                           'xi':1, 'xj':2}),
                    [6.0]
                    )))
        ok_(all(map(np.allclose,
                    integrate_x1_x2_y1_y2_between_xi_xj(
                        **{'x1':[0,1],'x2':[1,2],'y1':[5,6], 'y2':[5,6],
                           'xi':0, 'xj':2}),
                    [11.0]
                    )))
        ok_(all(map(np.allclose,
                    integrate_x1_x2_y1_y2_between_xi_xj(
                        **{'x1':[0,1.5,2],'x2':[1.5,2,4],'y1':[0,1,2], 'y2':[1,2,3],
                           'xi':1, 'xj':3}),
                    [3.416666666667]
                    )))

    def test_pintegrate_x1_x2_y1_y2_between_xi_xj(self):
        """test_pintegrate_x1_x2_y1_y2_between_xi_xj"""
        #pintegrate_x1_x2_y1_y2_between_xi_xj(a, xi, xj)

        ok_(all(map(np.allclose,
                    pintegrate_x1_x2_y1_y2_between_xi_xj(
                        **{'a': PolyLine([0,1.5,2],[1.5,2,4],[0,1,2],[1,2,3]),
                           'xi':1, 'xj':3}),
                    [3.416666666667]
                    )))
    def test_avg_x1_x2_y1_y2_between_xi_xj(self):
        """test_avg_x1_x2_y1_y2_between_xi_xj"""
        #avg_x1_x2_y1_y2_between_xi_xj(x1,x2,y1,y1, xi, xj)
        ok_(all(map(np.allclose,
                    avg_x1_x2_y1_y2_between_xi_xj(
                        **{'x1':[0],'x2':[1],'y1':[1], 'y2':[2],
                           'xi':0, 'xj':1}),
                    [1.5]
                    )))
        ok_(all(map(np.allclose,
                    avg_x1_x2_y1_y2_between_xi_xj(
                        **{'x1':[0,1],'x2':[1,2],'y1':[2,3], 'y2':[3,2],
                           'xi':0, 'xj':2}),
                    [2.5]
                    )))
        ok_(all(map(np.allclose,
                    avg_x1_x2_y1_y2_between_xi_xj(
                        **{'x1':[0,1,2],'x2':[1,2,3],'y1':[3,4,3], 'y2':[4,3,4],
                           'xi':0, 'xj':3}),
                    [3.5]
                    )))
        ok_(all(map(np.allclose,
                    avg_x1_x2_y1_y2_between_xi_xj(
                        **{'x1':[0,1,2],'x2':[1,2,3],'y1':[3,4,3], 'y2':[4,3,4],
                           'xi':[0,0,0], 'xj':[1,2,3]}),
                    [3.5,3.5,3.5]
                    )))
        ok_(all(map(np.allclose,
                    avg_x1_x2_y1_y2_between_xi_xj(
                        **{'x1':[0,1],'x2':[1,2],'y1':[5,6], 'y2':[5,6],
                           'xi':0, 'xj':1}),
                    [5.0]
                    )))
        ok_(all(map(np.allclose,
                    avg_x1_x2_y1_y2_between_xi_xj(
                        **{'x1':[0,1],'x2':[1,2],'y1':[5,6], 'y2':[5,6],
                           'xi':1, 'xj':2}),
                    [6.0]
                    )))
        ok_(all(map(np.allclose,
                    avg_x1_x2_y1_y2_between_xi_xj(
                        **{'x1':[0,1],'x2':[1,2],'y1':[5,6], 'y2':[5,6],
                           'xi':0, 'xj':2}),
                    [5.5]
                    )))
        ok_(all(map(np.allclose,
                    avg_x1_x2_y1_y2_between_xi_xj(
                        **{'x1':[0,1.5,2],'x2':[1.5,2,4],'y1':[0,1,2], 'y2':[1,2,3],
                           'xi':1, 'xj':3}),
                    [1.70833333333]
                    )))


    def test_xa_ya_multipy_avg_x1b_x2b_y1b_y2b_between(self):
        """test_xa_ya_multipy_avg_x1b_x2b_y1b_y2b_between"""
        #xa_ya_multipy_avg_x1b_x2b_y1b_y2b_between(xa, ya, x1b, x2b, y1b, y2b, xai, xbi, xbj, achoose_max=False):
        ok_(all(map(np.allclose,
                    xa_ya_multipy_avg_x1b_x2b_y1b_y2b_between(
                        **{'xa':[0,1,1,2],'ya':[0,1,2,3],
                           'x1b':[0,1,2],'x2b':[1,2,3],'y1b':[3,4,3], 'y2b':[4,3,4],
                           'xai':[0.5, 1, 1.5, 2],
                           'xbi':[0,0,0], 'xbj':[1,2,3]
                           }),
                    [[1.75, 3.5, 8.75, 10.5],
                     [1.75, 3.5, 8.75, 10.5],
                     [1.75, 3.5, 8.75, 10.5]]
                    )))
        ok_(all(map(np.allclose,
                    xa_ya_multipy_avg_x1b_x2b_y1b_y2b_between(
                        **{'xa':[0,1,1,2],'ya':[0,1,2,3],
                           'x1b':[0,1,2],'x2b':[1,2,3],'y1b':[3,4,3], 'y2b':[4,3,4],
                           'xai':[0.5, 1, 1.5, 2],
                           'xbi':[0,0,0], 'xbj':[1,2,3], 'achoose_max':True
                           }),
                     [[1.75, 7, 8.75, 10.5],
                     [1.75, 7, 8.75, 10.5],
                     [1.75, 7, 8.75, 10.5]]
                    )))

    def test_pxa_ya_multipy_avg_x1b_x2b_y1b_y2b_between(self):
        """test_pxa_ya_multipy_avg_x1b_x2b_y1b_y2b_between"""
        #pxa_ya_multipy_avg_x1b_x2b_y1b_y2b_between(a, b, xai, xbi, xbj, achoose_max=False):
        ok_(all(map(np.allclose,
                    pxa_ya_multipy_avg_x1b_x2b_y1b_y2b_between(
                        **{'a': PolyLine([0,1,1,2],[0,1,2,3]),
                           'b': PolyLine([0,1,2],[1,2,3],[3,4,3],[4,3,4]),
                           'xai':[0.5, 1, 1.5, 2],
                           'xbi':[0,0,0], 'xbj':[1,2,3], 'achoose_max':True
                           }),
                     [[1.75, 7, 8.75, 10.5],
                     [1.75, 7, 8.75, 10.5],
                     [1.75, 7, 8.75, 10.5]]
                    )))

    def test_integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between(self):
        """test_integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between"""
        ok_(np.allclose(
                    integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between(
                        **{'x1a':[0],'x2a':[0.5],'y1a':[1], 'y2a':[1],
                           'x1b':[0],'x2b':[0.5],'y1b':[1], 'y2b':[1],
                           'xi':[0], 'xj':[0.5]
                           }),
                    [0.5]))
        ok_(np.allclose(
                    integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between(
                        **{'x1a':[0],'x2a':[0.5],'y1a':[1], 'y2a':[1],
                           'x1b':[0],'x2b':[0.5],'y1b':[2], 'y2b':[2],
                           'xi':[0], 'xj':[0.5]
                           }),
                    [1.0]))
        ok_(np.allclose(
                    integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between(
                        **{'x1a':[0],'x2a':[0.5],'y1a':[1], 'y2a':[1],
                           'x1b':[0],'x2b':[0.5],'y1b':[2], 'y2b':[2],
                           'xi':[0,0], 'xj':[0.5, 0.25]
                           }),
                    [1.0, 0.5]))
        ok_(np.allclose(
                    integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between(
                        **{'x1a':[0],'x2a':[1],'y1a':[1], 'y2a':[1],
                           'x1b':[0],'x2b':[1],'y1b':[1], 'y2b':[2],
                           'xi':[0], 'xj':[1]
                           }),
                    [1.5]))
        ok_(np.allclose(
                    integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between(
                        **{'x1a':[0],'x2a':[1],'y1a':[1], 'y2a':[1],
                           'x1b':[0],'x2b':[1],'y1b':[1], 'y2b':[2],
                           'xi':[0], 'xj':[1]
                           }),
                    [1.5]))
        ok_(np.allclose(
                    integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between(
                        **{'x1a':[0],'x2a':[1],'y1a':[1], 'y2a':[1],
                           'x1b':[0],'x2b':[1],'y1b':[1], 'y2b':[2],
                           'xi':[0.25], 'xj':[0.75]
                           }),
                    [0.75]))

        assert_raises(ValueError,integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between,
                          **{'x1a':[0],'x2a':[1],'y1a':[1], 'y2a':[1],
                           'x1b':[0],'x2b':[0.5],'y1b':[1], 'y2b':[2],
                           'xi':[0.25], 'xj':[0.75]})

        #[0,0.4] a(x) = 1, b(x)=2, [0.4,1] a(x)=1, b(x)=4
        ok_(np.allclose(
                    integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between(
                        **{'x1a':[0,0.4],'x2a':[0.4,1],'y1a':[1,1], 'y2a':[1,1],
                           'x1b':[0,0.4],'x2b':[0.4,1],'y1b':[2,4], 'y2b':[2,4],
                           'xi':[0], 'xj':[1.0]
                           }),
                    [3.2]))
        #[0,1] a(x) = x, b(x)=1-x
        ok_(np.allclose(
                    integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between(
                        **{'x1a':[0],'x2a':[1],'y1a':[0], 'y2a':[1],
                           'x1b':[0],'x2b':[1],'y1b':[1], 'y2b':[0],
                           'xi':[0.2], 'xj':[0.4]
                           }),
                    [0.041333333]))

        #[0,0.4] a(x) = 1+x, b(x)=2-x, [0.4,1] a(x)=x, b(x)=1-x
        ok_(np.allclose(
                    integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between(
                        **{'x1a':[0,0.4],'x2a':[0.4,1],'y1a':[1,0.4], 'y2a':[1.4,1],
                           'x1b':[0,0.4],'x2b':[0.4,1],'y1b':[2,0.6], 'y2b':[1.6,0],
                           'xi':[0], 'xj':[1]
                           }),
                    [0.9666666667]))
    def test_pintegrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between(self):
        """test_pintegrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between"""

        ok_(np.allclose(
                    pintegrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between(
                        **{'a': PolyLine([0],[1],[1],[1]),
                           'b': PolyLine([0],[1],[1],[2]),
                           'xi':[0.25], 'xj':[0.75]
                           }),
                    [0.75]))
        ok_(np.allclose(
                    pintegrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between(
                        **{'a': PolyLine([0],[1],[1],[1]),
                           'b': PolyLine([0],[0.5],[1],[2]),
                           'xi':[0.25], 'xj':[0.75]
                           }),
                    [0.9375]))

        ok_(np.allclose(
                    pintegrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between(
                        **{'a': PolyLine([0,0.5],[0.5,1],[1,1],[1,1]),
                           'b': PolyLine([0],[1],[1],[2]),
                           'xi':[0.25], 'xj':[0.75]
                           }),
                    [0.75]))

    def test_xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(self):
        """test_xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between"""
        #xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(xa,ya,x1b,x2b,y1b,y2b, x1c, x2c, y1c, y2c, xai,xbi,xbj, achoose_max=False)
        ok_(np.allclose(
                    xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(
                        **{'xa':[0,1,1,2],'ya':[0,1,2,3],
                           'x1b':[0],'x2b':[0.5],'y1b':[1], 'y2b':[1],
                           'x1c':[0],'x2c':[0.5],'y1c':[2], 'y2c':[2],
                           'xai':[0.5, 1, 1.5, 2],
                           'xbi':[0,0], 'xbj':[0.5, 0.25],
                            'achoose_max': False
                           }),
                    [[1*0.5,1*1,1*2.5,1*3],
                     [0.5*0.5,0.5*1,0.5*2.5,0.5*3]]))
        ok_(np.allclose(
                    xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(
                        **{'xa':[0,1,1,2],'ya':[0,1,2,3],
                           'x1b':[0],'x2b':[0.5],'y1b':[1], 'y2b':[1],
                           'x1c':[0],'x2c':[0.5],'y1c':[2], 'y2c':[2],
                           'xai':[0.5, 1, 1.5, 2],
                           'xbi':[0,0], 'xbj':[0.5, 0.25],
                            'achoose_max': True
                           }),
                    [[1*0.5,1*2,1*2.5,1*3],
                     [0.5*0.5,0.5*2,0.5*2.5,0.5*3]]))
        assert_raises(ValueError, xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between,
                      **{'xa':[0,1,1,2],'ya':[0,1,2,3],
                           'x1b':[0],'x2b':[1],'y1b':[1], 'y2b':[1],
                           'x1c':[0],'x2c':[9],'y1c':[2], 'y2c':[2],
                           'xai':[0.5, 1, 1.5, 2],
                           'xbi':[0,0], 'xbj':[0.5, 0.25],
                            'achoose_max': True
                           })

    def test_pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(self):
        """test_pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between"""
        #pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(a, b, c, xai,xbi,xbj, achoose_max=False)
        ok_(np.allclose(
                    pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(
                        **{'a': PolyLine([0,1,1,2],[0,1,2,3]),
                           'b': PolyLine([0],[0.5],[1],[1]),
                           'c': PolyLine([0],[0.5],[2],[2]),
                           'xai':[0.5, 1, 1.5, 2],
                           'xbi':[0,0], 'xbj':[0.5, 0.25],
                            'achoose_max': False
                           }),
                    [[1*0.5,1*1,1*2.5,1*3],
                     [0.5*0.5,0.5*1,0.5*2.5,0.5*3]]))

    def test_pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between_super(self):
        """test_pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between_super"""
        #pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(a, b, c, xai,xbi,xbj, achoose_max=False)
        ok_(np.allclose(
                    pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between_super(
                        **{'a': [PolyLine([0,1,1,2],[0,1,2,3]), PolyLine([0,1,1,2],[0,1,2,3])],
                           'b': [PolyLine([0],[0.5],[0.5],[0.5]),PolyLine([0,0.2],[0.2,0.5],[0.5,0.5],[0.5,0.5])],
                           'c': PolyLine([0],[0.5],[2],[2]),
                           'xai':[0.5, 1, 1.5, 2],
                           'xbi':[0,0], 'xbj':[0.5, 0.25],
                            'achoose_max': False
                           }),
                    [[1*0.5,1*1,1*2.5,1*3],
                     [0.5*0.5,0.5*1,0.5*2.5,0.5*3]]))


def test_PolyLine():
    #define with x and y
    ok_(np.allclose(PolyLine([0,1],[3,4]).xy,
                    [[0,3],[1,4]]
                    ))
    ok_(np.allclose(PolyLine([0,1],[3,4]).x,
                    [0,1]
                    ))
    ok_(np.allclose(PolyLine([0,1],[3,4]).y,
                    [3,4]
                    ))
    ok_(all(map(np.allclose, PolyLine([0,1],[3,4]).x1_x2_y1_y2,
                    [[0],[1],[3],[4]]
                    )))

    #define with xy
    ok_(np.allclose(PolyLine([[0, 3],[1, 4]]).xy,
                    [[0,3],[1,4]]
                    ))
    ok_(np.allclose(PolyLine([[0, 3],[1, 4]]).x,
                    [0,1]
                    ))
    ok_(np.allclose(PolyLine([[0, 3],[1, 4]]).y,
                    [3,4]
                    ))
    ok_(all(map(np.allclose, PolyLine([[0, 3],[1, 4]]).x1_x2_y1_y2,
                    [[0],[1],[3],[4]]
                    )))

    #define with x1, x2, y1, y2
    ok_(np.allclose(PolyLine([0],[1],[3],[4]).xy,
                    [[0,3],[1,4]]
                    ))
    ok_(np.allclose(PolyLine([0],[1],[3],[4]).x,
                    [0,1]
                    ))
    ok_(np.allclose(PolyLine([0],[1],[3],[4]).y,
                    [3,4]
                    ))
    ok_(all(map(np.allclose, PolyLine([0],[1],[3],[4]).x1_x2_y1_y2,
                    [[0],[1],[3],[4]]
                    )))

    #scalar addtion
    ok_(np.allclose((0 + PolyLine([0,1],[3,4])).xy,
                    [[0,3],[1,4]]
                    ))
    ok_(np.allclose((2.5 + PolyLine([0,1],[3,4])).xy,
                    [[0,5.5],[1,6.5]]
                    ))
    ok_(np.allclose((PolyLine([0,1],[3,4]) + 2.5).xy,
                    [[0,5.5],[1,6.5]]
                    ))
    #scalar subtraction
    ok_(np.allclose((2.5 - PolyLine([0,1],[3,4])).xy,
                    [[0,-0.5],[1, -1.5]]
                    ))
    ok_(np.allclose((PolyLine([0,1],[3,4])-2.5).xy,
                    [[0,0.5],[1, 1.5]]
                    ))
    #scalar multiplication
    ok_(np.allclose((2 * PolyLine([0,1],[3,4])).xy,
                    [[0,6],[1, 8]]
                    ))
    ok_(np.allclose((2 * PolyLine([0,1],[3,4])).y,
                    [6, 8]
                    ))
    ok_(np.allclose((PolyLine([0,1],[3,4]) * 2).xy,
                    [[0,6],[1, 8]]
                    ))
    ok_(np.allclose((PolyLine([0,1],[3,4]) * 2).y,
                    [6, 8]
                    ))
    #scalar_division
    ok_(np.allclose((2 / PolyLine([0,1],[3,4])).xy,
                    [[0,2/3],[1, 1/2]]
                    ))
    ok_(np.allclose((2 / PolyLine([0,1],[3,4])).y,
                    [2/3,1/2]
                    ))
    ok_(np.allclose((PolyLine([0,1],[3,4]) /2).xy,
                    [[0,1.5],[1, 2]]
                    ))
    ok_(np.allclose((PolyLine([0,1],[3,4]) /2).y,
                    [1.5, 2]
                    ))

    #addition of PolyLines
    ok_(np.allclose((PolyLine([0,1],[3,4]) + PolyLine([0,1],[1,2])).xy,
                    [[0,4],[1, 6]]
                    ))

    ok_(np.allclose((PolyLine([0,1], [2,2]) + PolyLine([0,0.5,0.5,0.6,0.6,1], [0,0,1,1,0,0])).xy,
                    [[0,2],
                     [0.5,2],
                     [0.5,3],
                     [0.6,3],
                     [0.6,2],
                     [1, 2]]
                    ))
    ok_(np.allclose((PolyLine([0,1,2], [0,1,0]) + PolyLine([0,1,2], [1,0,1])).xy,
                    [[0,1],
                     [1,1],
                     [2,1]]
                    ))
    ok_(np.allclose((PolyLine([0,1,2], [0,1,0]) + PolyLine([0,1+1e-9,2], [1,0,1])).xy,
                    [[0,1],
                     [1,1],
                     [2,1]]
                    ))

    ok_(np.allclose((PolyLine([0,1,2], [0,1,0]) + PolyLine([0,1+1e-8,2], [1,0,1])).xy,
                    [[0,1],
                     [1,1],
                     [2,1]]
                    ))
    ok_(np.allclose((PolyLine([0,1,2], [0,1,0]) + PolyLine([0,1-1e-8,2], [1,0,1])).xy,
                    [[0,1],
                     [1,1],
                     [2,1]]
                    ))
    ok_(np.allclose((PolyLine([0,1,2], [0,1,0]) + PolyLine([0,1+1e-3,2], [1,0,1])).xy,
                    [[0,1],
                     [1,1+1e-3],
                     [1+1e-3,1-1e-3] ,
                     [2,1]]
                    ))


    ok_(np.allclose((PolyLine([0,1,1,2], [0,0,1,1]) + PolyLine([0,1,1,2], [0,0,2,3])).xy,
                    [[0,0],
                     [1,0],
                     [1,3],
                     [2,4]]
                    ))
    #dirty
    ok_(np.allclose((PolyLine([0,1,1+1e-6,2], [0,0,1,1]) + PolyLine([0,1,1,2], [0,0,2,3])).xy,
                    [[0,0],
                     [1,0],
                     [1,3],
                     [2,4]]
                    ))
    ok_(np.allclose((PolyLine([0,1,1+1e-3,2], [0,0,1,1]) + PolyLine([0,1,1,2], [0,0,2,3])).xy,
                    [[0,0],
                     [1,0],
                     [1,2],
                     [1.001,3.001],
                     [2,4]]
                    ))
    #reverse
    ok_(np.allclose((PolyLine([0,1,1+1e-6,2][::-1], [0,0,1,1][::-1]) + PolyLine([0,1,1,2][::-1], [0,0,2,3][::-1])).xy,
                    [[0,0],
                     [1,0],
                     [1,3],
                     [2,4]]
                    ))
    ok_(np.allclose((PolyLine([0,1,1+1e-3,2][::-1], [0,0,1,1][::-1]) + PolyLine([0,1,1,2][::-1], [0,0,2,3][::-1])).xy,
                    [[0,0],
                     [1,0],
                     [1,2],
                     [1.001,3.001],
                     [2,4]]
                    ))

    ok_(np.allclose((PolyLine([0,1,1+1e-6,2], [0,0,1,1]) - PolyLine([0,1,1,2], [0,0,2,3])).xy,
                    [[0,0],
                     [1,0],
                     [1,-1],
                     [2,-2]]
                    ))
    ok_(np.allclose((PolyLine([0,1,1+1e-3,2], [0,0,1,1]) - PolyLine([0,1,1,2], [0,0,2,3])).xy,
                    [[0,0],
                     [1,0],
                     [1,-2],
                     [1.001,-1.001],
                     [2,-2]]
                    ))

    #equality
    ok_(PolyLine([0,1],[3,4])==PolyLine([0,1],[3,4]))
    ok_(PolyLine([0,1],[3,4])==PolyLine([0,1],[3+1e-10,4]))
    assert_false(PolyLine([0,1],[3,4])==PolyLine([0,1],[3,8]))


#    #subdivide_into_linear_segments
#    ok_(PolyLine([0,1], [6,8]).subdivide_into_linear_segments(500,2)==
#        PolyLine([[ 0. ,  6. ],[ 0.5,  7. ],[ 1.,  8. ]]))
#    ok_(PolyLine([1,0], [6,8]).subdivide_into_linear_segments(500,2)==
#        PolyLine([[ 1. ,  6. ],[ 0.5,  7. ],[ 0.,  8. ]]))
#    ok_(PolyLine([0,3,9], [10,13,19]).subdivide_into_linear_segments(500,3)==
#        PolyLine([[ 0 ,  10 ],[ 1,  11 ],[ 2,  12 ],
#                  [3, 13], [5, 15], [7,17], [9,19]]))
#    ok_(PolyLine([0,3,9], [10,13,19]).subdivide_into_linear_segments(2,2)==
#        PolyLine([[ 0 ,  10 ],[ 1.5,  11.5 ],[ 3,  13 ],
#                  [5, 15], [7,17], [9,19]]))
#    ok_(PolyLine([0,1], [6,8]).subdivide_into_linear_segments(500,2, 0.2)==
#        PolyLine([[ 0. ,  6. ],[ 0.4,  6.8 ],[0.8, 7.6],[ 1.,  8. ]]))
#
##    assert_allclose(PolyLine([0,1], [6,8]).subdivide_into_linear_segments(500,2, 0.2).xy,
##                    [[ 0. ,  6. ],[ 0.4,  6.8 ],[0.8, 7.6],[ 1.,  8. ]])










class test_polyline_make_x_common(unittest.TestCase):

    def test_two_in_two_out(self):
        self.assertSequenceEqual(
                 polyline_make_x_common(PolyLine([[0,1],[3,4]]), PolyLine([[0,1],[3,4]])),
                 (PolyLine([[0,1],[3,4]]), PolyLine([[0,1],[3,4]]))
                 )

    def test_1(self):
        self.assertSequenceEqual([1,2,3],[1,2,3])

    def test_one_in_one_out(self):
        assert_equal(polyline_make_x_common(PolyLine([[0,1],[3,4]])),
                 PolyLine([[0,1],[3,4]]))

    def test_one_in_one_out(self):
        assert_equal(polyline_make_x_common(PolyLine([[0,1],[3,4]])),
                 PolyLine([[0,1],[3,4]]))

    def test_one_in_one_out_rtol(self):
        assert_equal(polyline_make_x_common(PolyLine([[0,1+1e-10],[3,4]])),
                 PolyLine([[0,1],[3,4]]))

    def test_many(self):
        self.assertSequenceEqual(
                 polyline_make_x_common(PolyLine([0,1],[3,4]), PolyLine([0,0.5],[5,5])),
                 (PolyLine([0,0.5,1],[3,3.5,4]), PolyLine([0,0.5,1],[5,5,5]))
                 )
        self.assertSequenceEqual(
                 polyline_make_x_common(PolyLine([0,0,2],[3,4,5]),
                                        PolyLine([0,0.5],[5,5])),
                 (PolyLine([ 0.,   0.,   0.5,  2. ],[ 3.,    4.  ,  4.25 , 5.  ]),
                  PolyLine([0,0.5,2],[5,5,5]))
                 )

        self.assertSequenceEqual(
                 polyline_make_x_common(PolyLine([0,1], [1,2]),
                                        PolyLine([0,0.5,0.5,0.6,0.6,1], [0,0,1,1,0,0])),
                         (PolyLine([0.0, 0.5, 0.6, 1.0],[1.0, 1.5, 1.6, 2.0]),
                          PolyLine([0.0, 0.5, 0.5, 0.6, 0.6, 1.0],[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]))
                 )

        self.assertSequenceEqual(
                 polyline_make_x_common(PolyLine([0,1,2], [0,1,0]),PolyLine([0,1+1e-3,2], [1,0,1])),
                         (PolyLine([0.0, 1.0, 1.001, 2.0],[0.0, 1.0, 0.999, 0.0]),
                          PolyLine([0.0, 1.0, 1.001, 2.0],[1.0, 0.001, 0.0, 1.0]))
                 )

        self.assertSequenceEqual(
                 polyline_make_x_common(PolyLine([0,1], [1,2]),
                                        PolyLine([0,1,2,4], [0,6,5,7]),
                                        PolyLine([0.5,2], [1,7])),
                         (PolyLine([0.0, 0.5, 1.0, 2.0, 4.0],[1.0, 1.5, 2.0, 2.0, 2.0]),
                             PolyLine([0.0, 0.5, 1.0, 2.0, 4.0],[0.0, 3.0, 6.0, 5.0, 7.0]),
                             PolyLine([0.0, 0.5, 1.0, 2.0, 4.0],[1.0, 1.0, 3.0, 7.0, 7.0]))
                 )

        self.assertSequenceEqual(
                 polyline_make_x_common(PolyLine([0,1][::-1], [1,2][::-1]),
                                        PolyLine([0,1,2,4][::-1], [0,6,5,7][::-1]),
                                        PolyLine([0.5,2][::-1], [1,7][::-1])),
                         (PolyLine([0.0, 0.5, 1.0, 2.0, 4.0],[1.0, 1.5, 2.0, 2.0, 2.0]),
                             PolyLine([0.0, 0.5, 1.0, 2.0, 4.0],[0.0, 3.0, 6.0, 5.0, 7.0]),
                             PolyLine([0.0, 0.5, 1.0, 2.0, 4.0],[1.0, 1.0, 3.0, 7.0, 7.0]))
                 )

class test_subdivide_x_y_into_segments(unittest.TestCase):
    """test for subdivide_x_y_into_segments"""

    def test_single_segment(self):
        x, y = subdivide_x_y_into_segments([0,1], [6,8],
                            dx=None,min_segments=2)
        assert_allclose(x, [0.0, 0.5, 1.0])
        assert_allclose(y, [6.0, 7.0, 8.0])

    def test_single_segment_reverse_order(self):
        x, y = subdivide_x_y_into_segments([1,0], [8,6],
                            dx=None,min_segments=2)
        assert_allclose(x, [1.0, 0.5, 0.0])
        assert_allclose(y, [8.0, 7.0, 6.0])

    def test_two_segments_of_different_length(self):
        x, y = subdivide_x_y_into_segments([0,3,9], [10,13,19],
                            dx=None,min_segments=3)
        assert_allclose(x, [0,1,2,3,5,7,9])
        assert_allclose(y, [10,11,12,13,15,17,19])

    def test_two_segments_of_different_length_dx_governs(self):
        x, y = subdivide_x_y_into_segments([0,3,9], [10,13,19],
                            dx=2,min_segments=2)
        assert_allclose(x, [0,1.5,3, 5, 7, 9])
        assert_allclose(y, [10,11.5,13,15,17,19])

    def test_just_before(self):
        x, y = subdivide_x_y_into_segments([0,1], [6,8],
                            dx=None,min_segments=2, just_before=0.2)
        assert_allclose(x, [0.0, 0.4, 0.8, 1.0])
        assert_allclose(y, [6.0, 6.8, 7.6, 8.0])


    def test_single_segment_logx_only(self):
        x, y = subdivide_x_y_into_segments([1,10], [6,8],
                            dx=None,min_segments=2, logx=True)
        assert_allclose(x, [1, 3.16227766, 10])
        assert_allclose(y, [6.0, 7.0, 8.0])

    def test_single_segment_logy_only(self):
        x, y = subdivide_x_y_into_segments([1,10], [6,8],
                            dx=None,min_segments=2, logy=True)
        assert_allclose(x, [1, 5.5, 10])
        assert_allclose(y, [6.0, 6.9282032302755, 8.0])

    def test_single_segment_logx_logy(self):
        x, y = subdivide_x_y_into_segments([1,10], [6,8],
                            dx=None,min_segments=2, logx=True, logy=True)
        assert_allclose(x, [1, 3.16227766, 10])
        assert_allclose(y, [6.0, 6.9282032302755, 8.0])

    def test_single_segment_logx_zero(self):
        x, y = subdivide_x_y_into_segments([0,10], [6,8],
                            dx=None, min_segments=2, logx=True, logxzero=0.1)
        assert_allclose(x, [0.1, 1, 10])
        assert_allclose(y, [6.0, 7.0, 8.0])
    def test_single_segment_logy_zero(self):
        x, y = subdivide_x_y_into_segments([0,10], [0,10],
                            dx=None, min_segments=2, logy=True, logyzero=0.1)
        assert_allclose(x, [0, 5, 10])
        assert_allclose(y, [0.1, 1, 10])


class test_subdivide_x_into_segments(unittest.TestCase):
    """test for subdivide_x_into_segments"""

    def test_single_segment(self):
        x = subdivide_x_into_segments([0,1],
                            dx=None,min_segments=2)
        assert_allclose(x, [0.0, 0.5, 1.0])


    def test_single_segment_reverse_order(self):
        x= subdivide_x_into_segments([1,0],
                            dx=None,min_segments=2)
        assert_allclose(x, [1.0, 0.5, 0.0])


    def test_two_segments_of_different_length(self):
        x= subdivide_x_into_segments([0,3,9],
                            dx=None,min_segments=3)
        assert_allclose(x, [0,1,2,3,5,7,9])


    def test_two_segments_of_different_length_dx_governs(self):
        x = subdivide_x_into_segments([0,3,9],
                            dx=2,min_segments=2)
        assert_allclose(x, [0,1.5,3, 5, 7, 9])


    def test_just_before(self):
        x = subdivide_x_into_segments([0,1],
                            dx=None,min_segments=2, just_before=0.2)
        assert_allclose(x, [0.0, 0.4, 0.8, 1.0])


    def test_single_segment_logx(self):
        x = subdivide_x_into_segments([1,10],
                            dx=None,min_segments=2, logx=True)
        assert_allclose(x, [1, 3.16227766, 10])


    def test_single_segment_logx_zero(self):
        x= subdivide_x_into_segments([0,10],
                            dx=None, min_segments=2, logx=True, logxzero=0.1)
        assert_allclose(x, [0.1, 1, 10])




#    ok_(np.allclose((PolyLine([0,1],[3,4]) + PolyLine([0,1],[1,2])).xy,
#                    [[0,4],[1, 6]]
#                    ))
        #xc_yc_multiply_integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between(xc,yc,x1a,x2a,y1a,y2a, x1b, x2b, y1b, y2b, xci,xai,xaj, cchoose_max=False)
        #integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between(x1a,x2a,y1a,y2a,x1b,x2b,y1b,y2b,xi,xj)


        #interp_xa_ya_multipy_x1b_x2b_y1b_y2b(xa, ya, x1b, x2b, y1b, y2b, xai, xbi, achoose_max=False, bchoose_max=True):
#        self.two_steps = {'x': [0,  0,  1,  1,  2] ,
#                          'y': [0, 10, 10, 30, 30]}
#        self.two_steps_reverse = {'x': [0,  0,  -1,  -1,  -2],
#                                  'y': [0, 10,  10,  30,  30]}
#        self.two_ramps = {'x': [0,  0.5,  1,  1.5,  2],
#                          'y': [0, 10, 10, 30, 30]}
#        self.two_ramps_reverse = {'x': [0,  -0.5,  -1,  -1.5,  -2],
#                                  'y': [0,  10.0,  10,  30.0,  30]}
#        self.two_ramps_two_steps = {'x': [0,  0.4,   0.4,  1,  2.5,  3,  3],
#                                    'y': [0,  10.0, 20.0, 20, 30.0, 30, 40]}
#        self.two_ramps_two_steps_reverse = {'x': [0,  -0.4,   -0.4,  -1,  -2.5,  -3,  -3],
#                                            'y': [0,  10.0, 20.0, 20, 30.0, 30, 40]}
#        self.switch_back = {'x': [0, 0.5, 1, 0.75, 1.5, 2],
#                            'y': [0, 1.2, 2, 2.25, 3.5, 3]}
#        self.switch_back_steps = {'x': [0, 0, 1, 0.75, 0.75, 2],
#                            'y': [0, 1.2, 2, 2.25, 3.5, 3]}
#class test_has_steps(sample_linear_piecewise):
#    """test some has_steps examples"""
#    def __init__(self):
#        sample_linear_piecewise.__init__(self)
#
#    def test_(self):
#        """test some has_steps examples"""
#
#        assert_false(has_steps(self.two_steps['x'], self.two_steps['y']))
#        ok_(has_steps(self.two_steps_reverse['x'], self.two_steps_reverse['y']))
#        assert_false(has_steps(self.two_ramps['x'], self.two_ramps['y']))
#        assert_false(has_steps(self.two_ramps_reverse['x'], self.two_ramps_reverse['y']))
#        ok_(has_steps(self.two_ramps_two_steps['x'], self.two_ramps_two_steps['y']))
#        ok_(has_steps(self.two_ramps_two_steps_reverse['x'], self.two_ramps_two_steps_reverse['y']))


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])