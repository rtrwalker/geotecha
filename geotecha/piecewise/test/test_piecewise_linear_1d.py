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
                    [[2,3],[3,4.5],[4,6]]
                    ))                
                    
                    
                    
        #interp_xa_ya_multipy_x1b_x2b_y1b_y2b(xa, ya, x1b, x2b, y1b, y2b, xai, xbi, achoose_max=False, bchoose_max=True):                    
#        self.two_steps = {'x': [0,  0,  1,  1,  2],
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