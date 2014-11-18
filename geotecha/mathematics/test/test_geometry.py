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

"""Testing routines for the geometry module"""

from __future__ import division, print_function

from nose import with_setup
from nose.tools.trivial import assert_almost_equal
from nose.tools.trivial import assert_raises
from nose.tools.trivial import ok_
from nose.tools.trivial import assert_false
from nose.tools.trivial import assert_equal
from numpy.testing import assert_allclose

from math import pi
import numpy as np
import unittest

from geotecha.mathematics.geometry import xyz_from_pts
from geotecha.mathematics.geometry import eqn_of_plane
from geotecha.mathematics.geometry import replace_x0_and_x1_with_vect
from geotecha.mathematics.geometry import polygon_area
from geotecha.mathematics.geometry import polygon_centroid
from geotecha.mathematics.geometry import integrate_f_over_polygon_code
from geotecha.mathematics.geometry import integrate_f_over_polyhedra_code
from geotecha.mathematics.geometry import make_hexahedron
from geotecha.mathematics.geometry import polyhedron_volume
from geotecha.mathematics.geometry import polygon_2nd_moment_of_area

def test_replace_x0_and_x1_with_vect():
    """test for replace_x0_and_x1_with_vect"""

    assert_equal(replace_x0_and_x1_with_vect('x0'),'x[:-1]')
    assert_equal(replace_x0_and_x1_with_vect('x0 + x1 + y0'),'x[:-1] + x[1:] + y[:-1]')
    assert_equal(replace_x0_and_x1_with_vect('f0', ['f']), 'f[:-1]')


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
    def test_polygon_area(self):
        """test some polygon_area"""

        assert_almost_equal(polygon_area(self.shp['unit square']),1)
        assert_almost_equal(polygon_area(self.shp['right tri']),0.5)
        assert_almost_equal(polygon_area(self.shp['octahedral tri']),0.8660254)
        assert_almost_equal(polygon_area(self.shp['3D tri']),4.82182538)

    def test_polygon_centroid(self):
        """test some polygon_centroid"""

        ok_(np.allclose(polygon_centroid(self.shp['unit square']),[0.5,0.5,0]))
        ok_(np.allclose(polygon_centroid(self.shp['right tri']),[1/3.0,1/3.0,0]))
        ok_(np.allclose(polygon_centroid(self.shp['octahedral tri']),[1/3.0,1/3.0,1/3.0]))


class test_integrate_f_over_polygon_code(unittest.TestCase):
    """tests for integrate_f_over_polygon_code"""

    #integrate_f_over_polyhedra_code(f)

    def test_area(self):

        assert_equal(integrate_f_over_polygon_code(1).splitlines(),
                     'def ifxy(pts):\n    "Integrate f = 1 over '
                     'polygon"\n\n    x, y, z = xyz_from_pts(pts, True)'
                     '\n\n    return np.sum((x[:-1]/2 + x[1:]/2)*(-y[:-1] '
                     '+ y[1:]))'.splitlines())

class test_integrate_f_over_polyhedra_code(unittest.TestCase):
    """tests for integrate_f_over_polyhedra_code"""

    #integrate_f_over_polyhedra_code(f)

    def test_volume(self):

        assert_equal(integrate_f_over_polyhedra_code(1).splitlines(),
                     'def ifxyz(faces):\n    "Integrate f = 1 over '
                     'polyhedron"\n\n    x, y, z = xyz_from_pts(pts,True)'
                     '\n    igral = 0\n    for pts in faces:\n        '
                     '(n1,n2,n3),d = eqn_of_plane(pts)\n        '
                     'if n1==0:\n            continue\n        '
                     'igral += np.sum((-z[:-1] + z[1:])*((-2*d*y[:-1] - '
                     'n2*y[:-1]**2 - 2*n3*y[:-1]*z[:-1])/(2*n1) + (d*y[:-1] '
                     '- d*y[1:] + n2*y[:-1]**2 - n2*y[:-1]*y[1:] + '
                     '2*n3*y[:-1]*z[:-1] - n3*y[:-1]*z[1:] - '
                     'n3*y[1:]*z[:-1])/(2*n1) + (-n2*y[:-1]**2 + '
                     '2*n2*y[:-1]*y[1:] - n2*y[1:]**2 - 2*n3*y[:-1]*z[:-1] + '
                     '2*n3*y[:-1]*z[1:] + 2*n3*y[1:]*z[:-1] - '
                     '2*n3*y[1:]*z[1:])/(6*n1)))\n\n    '
                     'return igral'.splitlines())


class test_make_hexahedron(unittest.TestCase):
    """tests for make_hexahedron"""

    #make_hexahedron(coords)


    twounitcube = [
                    [-1,-1,-1],
                    [-1,-1, 1],
                    [1,-1, 1],
                    [1,-1,-1],
                    [-1,1,-1],
                    [-1,1,1],
                    [1,1,1],
                    [1,1,-1]]


    def test_twounitcube(self):
        assert_allclose(make_hexahedron(self.twounitcube),
                        [np.array([[-1, -1, -1],
       [ 1, -1, -1],
       [ 1, -1,  1],
       [-1, -1,  1]]), np.array([[ 1, -1,  1],
       [ 1, -1, -1],
       [ 1,  1, -1],
       [ 1,  1,  1]]), np.array([[ 1,  1, -1],
       [-1,  1, -1],
       [-1,  1,  1],
       [ 1,  1,  1]]), np.array([[-1, -1,  1],
       [-1,  1,  1],
       [-1,  1, -1],
       [-1, -1, -1]]), np.array([[-1,  1,  1],
       [-1, -1,  1],
       [ 1, -1,  1],
       [ 1,  1,  1]]), np.array([[-1,  1, -1],
       [ 1,  1, -1],
       [ 1, -1, -1],
       [-1, -1, -1]])])

class test_polyhedron_volume(unittest.TestCase):
    """tests for polyhedron_volume"""

    #polyhedron_volume(faces)


    twounitcube = [np.array([
       [-1, -1, -1],
       [ 1, -1, -1],
       [ 1, -1,  1],
       [-1, -1,  1]]), np.array([
       [ 1, -1,  1],
       [ 1, -1, -1],
       [ 1,  1, -1],
       [ 1,  1,  1]]), np.array([
       [ 1,  1, -1],
       [-1,  1, -1],
       [-1,  1,  1],
       [ 1,  1,  1]]), np.array([
       [-1, -1,  1],
       [-1,  1,  1],
       [-1,  1, -1],
       [-1, -1, -1]]), np.array([
       [-1,  1,  1],
       [-1, -1,  1],
       [ 1, -1,  1],
       [ 1,  1,  1]]), np.array([
       [-1,  1, -1],
       [ 1,  1, -1],
       [ 1, -1, -1],
       [-1, -1, -1]])]


    righttetra = [np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0]]),
                  np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]),
                  np.array([[0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0]]),
                  np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0]])]

    def test_twounitcube(self):
        assert_allclose(polyhedron_volume(self.twounitcube), 8)

    def test_twounitcube_translated(self):
        assert_allclose(polyhedron_volume([np.array([4,5,6]) + v for v in
                                            self.twounitcube]), 8)

    def test_righttetra(self):
        assert_allclose(polyhedron_volume(self.righttetra), 0.166666666667)

    def test_righttetra_translated(self):
        assert_allclose(polyhedron_volume([np.array([4,5,6]) + v for v in
                                        self.righttetra]), 0.166666666667)



class test_polygon_2nd_moment_of_area(unittest.TestCase):
    """tests for polygon_2nd_moment_of_area"""

    shp = dict()
    shp['unit square'] = [[0,0],[1,0],[1,1],[0,1]]
    shp['right tri'] = [[0,0],[1,0],[0,1]]
    shp['3D tri'] = [[1,-2,0],[3,1,4],[0,-1,2]]
    shp['2D tri'] = [[1,0],[3,4],[0,2]]
    shp['octahedral tri'] = [[1,0,0],[0,1,0],[0,0,1]]


    def test_unit_square(self):
        assert_allclose(polygon_2nd_moment_of_area(self.shp['unit square']),
                        [ 0.08333333, 0.08333333, 0.])

    def test_3D_tri(self):
        assert_allclose(polygon_2nd_moment_of_area(self.shp['3D tri']),
                        np.array([ 0.38888889,  0.38888889,  0.66666667]))

    def test_right_tri(self):
        assert_allclose(polygon_2nd_moment_of_area(self.shp['right tri']),
                        np.array([ 0.05555556,  0.05555556,  0.        ]))

    def test_octahedral_tri(self):
        assert_allclose(polygon_2nd_moment_of_area(self.shp['octahedral tri']),
                        np.array([ 0.05555556,  0.05555556,  0.05555556]))

    def test_2D_tri(self):
        assert_allclose(polygon_2nd_moment_of_area(self.shp['2D tri']),
                        np.array([ 0.38888889,  0.66666667,  0.        ]))



if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])