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

from geotecha.piecewise.geometry import xyz_from_pts
from geotecha.piecewise.geometry import eqn_of_plane


class test_geometry(object):
    """Some shapes for testing"""
        
    def __init__(self):
        
            
        self.shp = dict()
        self.shp['unit square'] = [[0,0],[1,0],[1,1],[0,1]]
        self.shp['right tri'] = [[0,0],[1,0],[0,1]]
        self.shp['octahedral tri'] = [[1,0,0],[0,1,0],[0,0,1]]
        self.shp['3D tri'] = [[1,-2,0],[3,1,4],[0,-1,2]]
                                    
    def test_xyz_from_pts(self):
        """test some xyz_from_pts"""        
        ok_(all(map(np.allclose,
                    xyz_from_pts(self.shp['unit square'],False),
                    [[0,1,1,0],[0,0,1,1],[0,0,0,0]],
                    )))
        ok_(all(map(np.allclose,
                    xyz_from_pts(self.shp['unit square'],True),
                    [[0,1,1,0,0],[0,0,1,1,0],[0,0,0,0,0]],
                    )))
        ok_(all(map(np.allclose,
                    xyz_from_pts(self.shp['octahedral tri'],False),
                    [[1,0,0],[0,1,0],[0,0,1]],
                    )))                    
        ok_(all(map(np.allclose,
                    xyz_from_pts(self.shp['octahedral tri'],True),
                    [[1,0,0,1],[0,1,0,0],[0,0,1,0]],
                    )))                                        
    def test_eqn_of_plane(self):
        """test some eqn_of_plane"""        
        ok_(all(map(np.allclose,
                    eqn_of_plane(self.shp['unit square']),
                    [[0,0,1],0],
                    )))
                    
        ok_(all(map(np.allclose,
                    eqn_of_plane(self.shp['right tri']),
                    [[0,0,1],0],
                    ))) 
        ok_(all(map(np.allclose,
                    eqn_of_plane(self.shp['octahedral tri']),
                    [[0.57735,0.57735,0.57735],-0.57735],
                    )))  
        ok_(all(map(np.allclose,
                    eqn_of_plane(self.shp['3D tri']),
                    [[0.20739,-0.82956,0.51848],-1.8665],
                    ))) #http://tutorial.math.lamar.edu/Classes/CalcIII/EqnsOfPlanes.aspx                       
        