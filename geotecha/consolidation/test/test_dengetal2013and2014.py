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
"""Some test routines for the dengetal2013and2014 module

"""
from __future__ import division, print_function

from nose import with_setup
from nose.tools.trivial import assert_almost_equal
from nose.tools.trivial import assert_raises
from nose.tools.trivial import ok_
from numpy.testing import assert_allclose
import unittest

from math import pi
import numpy as np
import textwrap
import matplotlib.pyplot as plt
from geotecha.consolidation.dengetal2013and2014 import dengetal2013

class test_dengetal2013(unittest.TestCase):

    def test_figure3a_A2_equals_point_one(self):
        assert_allclose(
            dengetal2013(z=np.array([0.05, 0.1, 0.2, 0.5, 0.8, 1.0])*20,
                         t=[11025.,  110250.],
                         rw=0.035, re=0.525,
                         A1=1, A2=0.1, A3=0,
                         H=20,
                         rs=0.175,
                         ks=2e-8/1.8,
                         kw0=1e-3,
                         kh=2e-8,
                         mv=0.2e-3,
                         gamw=10,
                         ui=1),
            np.array([[ 0.81103472,  0.12313867],
                       [ 0.83472101,  0.16421523],
                       [ 0.86588137,  0.23691031],
                       [ 0.90695675,  0.3765874 ],
                       [ 0.92048265,  0.43667275],
                       [ 0.92267961,  0.44720765]]))

    def test_figure3a_A2_equals_point_nine_nine(self):
        assert_allclose(
            dengetal2013(z=np.array([0.05, 0.1, 0.2, 0.5, 0.8, 1.0])*20,
                         t=[11025.,  110250.],
                         rw=0.035, re=0.525,
                         A1=1, A2=0.99, A3=0,
                         H=20,
                         rs=0.175,
                         ks=2e-8/1.8,
                         kw0=1e-3,
                         kh=2e-8,
                         mv=0.2e-3,
                         gamw=10,
                         ui=1),
            np.array([[ 0.81168066,  0.12412292],
                       [ 0.83668852,  0.16812724],
                       [ 0.87096244,  0.25118534],
                       [ 0.92073203,  0.43785721],
                       [ 0.94257216,  0.55353625],
                       [ 0.95050775,  0.60194473]]))


    def test_figure3b_a3_equals_point_one(self):
        assert_allclose(
            dengetal2013(z=np.array([0.05, 0.1, 0.2, 0.5, 0.8, 1.0])*20,
                         t=[11025.,  110250.],
                         rw=0.035, re=0.525,
                         A1=1, A2=0, A3=9.07029478e-07,
                         H=20,
                         rs=0.175,
                         ks=2e-8/1.8,
                         kw0=1e-3,
                         kh=2e-8,
                         mv=0.2e-3,
                         gamw=10,
                         ui=1),
            np.array([[ 0.81110629,  0.12524912],
                       [ 0.83472394,  0.16807639],
                       [ 0.86563054,  0.24288244],
                       [ 0.90599734,  0.38274085],
                       [ 0.91909172,  0.44098945],
                       [ 0.92118321,  0.45096744]]))
    def test_figure3b_a3_equals_point_five(self):
        assert_allclose(
            dengetal2013(z=np.array([0.05, 0.1, 0.2, 0.5, 0.8, 1.0])*20,
                         t=[11025.,  110250.],
                         rw=0.035, re=0.525,
                         A1=1, A2=0, A3=4.53514739e-06,
                         H=20,
                         rs=0.175,
                         ks=2e-8/1.8,
                         kw0=1e-3,
                         kh=2e-8,
                         mv=0.2e-3,
                         gamw=10,
                         ui=1),
            np.array([[ 0.81168093,  0.13560054],
                       [ 0.8355792 ,  0.18756625],
                       [ 0.86669814,  0.27523356],
                       [ 0.90708096,  0.42885195],
                       [ 0.92011769,  0.48939465],
                       [ 0.92219713,  0.49958604]]))

    def test_figure3b_a3_equals_one(self):
        assert_allclose(
            dengetal2013(z=np.array([0.05, 0.1, 0.2, 0.5, 0.8, 1.0])*20,
                         t=[11025.,  110250.],
                         rw=0.035, re=0.525,
                         A1=1, A2=0, A3=9.07029478e-06,
                         H=20,
                         rs=0.175,
                         ks=2e-8/1.8,
                         kw0=1e-3,
                         kh=2e-8,
                         mv=0.2e-3,
                         gamw=10,
                         ui=1),
            np.array([[ 0.81241413,  0.15242832],
                       [ 0.83666321,  0.21757866],
                       [ 0.86803975,  0.32108789],
                       [ 0.90842794,  0.4868681 ],
                       [ 0.92138857,  0.54790671],
                       [ 0.92345241,  0.55797917]]))
    def test_figure3c_A2_equals_point_nin_nine_a3_equals_one(self):
        assert_allclose(
            dengetal2013(z=np.array([0.05, 0.1, 0.2, 0.5, 0.8, 1.0])*20,
                         t=[11025.,  110250.],
                         rw=0.035, re=0.525,
                         A1=1, A2=0.99, A3=9.07029478e-06,
                         H=20,
                         rs=0.175,
                         ks=2e-8/1.8,
                         kw0=1e-3,
                         kh=2e-8,
                         mv=0.2e-3,
                         gamw=10,
                         ui=1),
            np.array([[ 0.81315577,  0.15423227],
                   [ 0.83889148,  0.22420948],
                   [ 0.87368554,  0.34204751],
                   [ 0.92325897,  0.55703101],
                   [ 0.9446685 ,  0.66714553],
                   [ 0.95239683,  0.70945769]]))

if __name__ == '__main__':

    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])

