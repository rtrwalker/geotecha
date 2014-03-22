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
"""Some test routines for the speccon_1d_vert_radial_boundary module

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
from geotecha.piecewise.piecewise_linear_1d import PolyLine

from geotecha.speccon.speccon1d import dim1sin_f
from geotecha.speccon.speccon1d import dim1sin_avgf

class test_dim1sin_f(unittest.TestCase):
    """tests for dim1sin_f"""
    #dim1sin_f(m, outz, tvals, v_E_Igamv_the, drn, top_vs_time, bot_vs_time, top_omega_phase, bot_omega_phase)

#    artificallially create 3 eigs, 2 zs and 2 ts

    outz = np.array([0.2, 0.4])
    m = np.array([1.0,2.0, 3.0]) #1x2
    v_E_Igamv_the = np.ones((3,2), dtype=float)
    tvals = np.array([1.0, 3])
    top_vs_time = PolyLine([0,2,4],[0,2,2])
    bot_vs_time = PolyLine([0,2,4],[0,2,2])
    omega_phase = (1,2)
    def test_no_BC(self):
        #expected is from:
        #np.dot(np.sin(outz[:,np.newaxis] * m[np.newaxis,:]), v_E_Igamv_the)

        assert_allclose(dim1sin_f(self.m,
                                  self.outz,
                                  self.tvals,
                                  self.v_E_Igamv_the,
                                  drn=0),

                                  np.array([[ 1.15273015,  1.15273015],
                                            [ 2.03881352,  2.03881352]]))
    def test_top_vs_time_drn_0(self):
        #expected is from:
        #test no_bc + top_vs time interplolated at tvals and depth z (mag_vs_depth reduces to zero at bottom)

        assert_allclose(dim1sin_f(self.m,
                                  self.outz,
                                  self.tvals,
                                  self.v_E_Igamv_the,
                                  drn=0,
                                  top_vs_time = [self.top_vs_time]),

                                  np.array([[ 1*0.8+1.15273015,  2*0.8+1.15273015],
                                            [ 1*0.6+2.03881352,  2*0.6+2.03881352]]))

    def test_top_vs_time_drn_1(self):
        #expected is from:
        #test no_bc + top_vs time interplolated at tvals and depth z (mag vs depth is uniform top to bottom)

        assert_allclose(dim1sin_f(self.m,
                                  self.outz,
                                  self.tvals,
                                  self.v_E_Igamv_the,
                                  drn=1,
                                  top_vs_time = [self.top_vs_time]),

                                  np.array([[ 1+1.15273015,  2+1.15273015],
                                            [ 1+2.03881352,  2+2.03881352]]))

    def test_bot_vs_time_drn_0(self):
        #expected is from:
        #test no_bc + bot_vs time interplolated at tvals and depth z (mag_vs_depth reduces to zero at top)

        assert_allclose(dim1sin_f(self.m,
                                  self.outz,
                                  self.tvals,
                                  self.v_E_Igamv_the,
                                  drn=0,
                                  bot_vs_time = [self.bot_vs_time]),

                                  np.array([[ 1*0.2+1.15273015,  2*0.2+1.15273015],
                                            [ 1*0.4+2.03881352,  2*0.4+2.03881352]]))

    def test_bot_vs_time_top_vs_time_drn_1(self):
        #expected is from:
        #test no_bc + top_vs_time and bot_vs time interplolated at tvals and depth z (mag_vs_depth is uniform from top plus reduces to zero at top)

        assert_allclose(dim1sin_f(self.m,
                                  self.outz,
                                  self.tvals,
                                  self.v_E_Igamv_the,
                                  drn=1,
                                  top_vs_time = [self.top_vs_time],
                                  bot_vs_time = [self.bot_vs_time]),

                                  np.array([[ 1+1*0.2+1.15273015,  2+2*0.2+1.15273015],
                                            [ 1+1*0.4+2.03881352,  2+2*0.4+2.03881352]]))

    def test_top_vs_time_drn_0_omega_phase(self):
        #expected is from:
        #test no_bc + cos(omega*t+phase)  multiplied top_vs time interplolated at tvals and depth z (mag_vs_depth reduces to zero at bottom)

        assert_allclose(dim1sin_f(self.m,
                                  self.outz,
                                  self.tvals,
                                  self.v_E_Igamv_the,
                                  drn=0,
                                  top_vs_time = [self.top_vs_time],
                                  top_omega_phase=[self.omega_phase]),

                                  np.array([[ 1*0.8*np.cos(1*1+2)+1.15273015,  2*0.8*np.cos(1*3+2)+1.15273015],
                                            [ 1*0.6*np.cos(1*1+2)+2.03881352,  2*0.6*np.cos(1*3+2)+2.03881352]]))
    def test_bot_vs_time_drn_0(self):
        #expected is from:
        #test no_bc + bot_vs time interplolated at tvals and depth z (mag_vs_depth reduces to zero at top)

        assert_allclose(dim1sin_f(self.m,
                                  self.outz,
                                  self.tvals,
                                  self.v_E_Igamv_the,
                                  drn=0,
                                  bot_vs_time = [self.bot_vs_time],
                                  bot_omega_phase=[self.omega_phase]),

                                  np.array([[ 1*0.2*np.cos(1*1+2)+1.15273015,  2*0.2*np.cos(1*3+2)+1.15273015],
                                            [ 1*0.4*np.cos(1*1+2)+2.03881352,  2*0.4*np.cos(1*3+2)+2.03881352]]))

    def test_bot_vs_time_top_vs_time_drn_1_double_loads(self):
        #expected is from:
        #test no_bc + top_vs_time and bot_vs time interplolated at tvals and depth z (mag_vs_depth is uniform from top plus reduces to zero at top)

        assert_allclose(dim1sin_f(self.m,
                                  self.outz,
                                  self.tvals,
                                  self.v_E_Igamv_the,
                                  drn=1,
                                  top_vs_time = [self.top_vs_time, self.top_vs_time],
                                  bot_vs_time = [self.bot_vs_time, self.bot_vs_time]),

                                  np.array([[ 2+2*0.2+1.15273015,  4+4*0.2+1.15273015],
                                            [ 2+2*0.4+2.03881352,  4+4*0.4+2.03881352]]))


class test_dim1sin_avgf(unittest.TestCase):
    """tests for dim1sin_avgf"""
    #dim1sin_avgf(m, z, tvals, v_E_Igamv_the, drn, top_vs_time = None, bot_vs_time=None, top_omega_phase=None, bot_omega_phase=None)

#    artificallially create 3 eigs, 2 zs and 2 ts

    outz = np.array([[0.2, 0.4], [0.4, 0.6]])
    z1 = outz[:, 0]
    z2 = outz[:, 1]
    m = np.array([1.0,2.0, 3.0]) #1x2
    v_E_Igamv_the = np.ones((3,2), dtype=float)
    tvals = np.array([1.0, 3])
    top_vs_time = PolyLine([0,2,4],[0,2,2])
    bot_vs_time = PolyLine([0,2,4],[0,2,2])
    omega_phase = (1,2)

    def test_no_BC(self):
        #expected is from:
        #np.dot((np.cos(z1[:,np.newaxis] * m[np.newaxis,:])-np.cos(z2[:,np.newaxis] * m[np.newaxis,:]))/m[np.newaxis,:]/(z2[:,np.newaxis]-z1[:,np.newaxis]), v_E_Igamv_the)
        assert_allclose(dim1sin_avgf(self.m,
                                  self.outz,
                                  self.tvals,
                                  self.v_E_Igamv_the,
                                  drn=0),

                                  np.array([[ 1.6275434 ,  1.6275434 ],
                                            [ 2.29709903,  2.29709903]]))

    def test_top_vs_time_drn_0(self):
        #expected is from:
        #test no_bc + top_vs time interplolated at tvals and depth z (mag_vs_depth reduces to zero at bottom)

        assert_allclose(dim1sin_avgf(self.m,
                                  self.outz,
                                  self.tvals,
                                  self.v_E_Igamv_the,
                                  drn=0,
                                  top_vs_time = [self.top_vs_time]),

                                  np.array([[ 1*0.7+1.6275434 ,  2*0.7+1.6275434 ],
                                            [ 1*0.5+2.29709903,  2*0.5+2.29709903]]))

    def test_top_vs_time_drn_1(self):
        #expected is from:
        #test no_bc + top_vs time interplolated at tvals and depth z (mag vs depth is uniform top to bottom)

        assert_allclose(dim1sin_avgf(self.m,
                                  self.outz,
                                  self.tvals,
                                  self.v_E_Igamv_the,
                                  drn=1,
                                  top_vs_time = [self.top_vs_time]),

                                  np.array([[ 1+1.6275434 ,  2+1.6275434 ],
                                            [ 1+2.29709903,  2+2.29709903]]))
#
    def test_bot_vs_time_drn_0(self):
        #expected is from:
        #test no_bc + bot_vs time interplolated at tvals and depth z (mag_vs_depth reduces to zero at top)

        assert_allclose(dim1sin_avgf(self.m,
                                  self.outz,
                                  self.tvals,
                                  self.v_E_Igamv_the,
                                  drn=0,
                                  bot_vs_time = [self.bot_vs_time]),

                                  np.array([[ 1*0.3+1.6275434 ,  2*0.3+1.6275434 ],
                                            [ 1*0.5+2.29709903,  2*0.5+2.29709903]]))

    def test_bot_vs_time_top_vs_time_drn_1(self):
        #expected is from:
        #test no_bc + top_vs_time and bot_vs time interplolated at tvals and depth z (mag_vs_depth is uniform from top plus reduces to zero at top)

        assert_allclose(dim1sin_avgf(self.m,
                                  self.outz,
                                  self.tvals,
                                  self.v_E_Igamv_the,
                                  drn=1,
                                  top_vs_time = [self.top_vs_time],
                                  bot_vs_time = [self.bot_vs_time]),

                                  np.array([[ 1+1*0.3+1.6275434 ,  2+2*0.3+1.6275434 ],
                                            [ 1+1*0.5+2.29709903,  2+2*0.5+2.29709903]]))

    def test_top_vs_time_drn_0_omega_phase(self):
        #expected is from:
        #test no_bc + cos(omega*t+phase)  multiplied top_vs time interplolated at tvals and depth z (mag_vs_depth reduces to zero at bottom)

        assert_allclose(dim1sin_avgf(self.m,
                                  self.outz,
                                  self.tvals,
                                  self.v_E_Igamv_the,
                                  drn=0,
                                  top_vs_time = [self.top_vs_time],
                                  top_omega_phase=[self.omega_phase]),

                                  np.array([[ 1*0.7*np.cos(1*1+2)+1.6275434,  2*0.7*np.cos(1*3+2)+1.6275434],
                                            [ 1*0.5*np.cos(1*1+2)+2.29709903,  2*0.5*np.cos(1*3+2)+2.29709903]]))

    def test_bot_vs_time_drn_0(self):
        #expected is from:
        #test no_bc + bot_vs time interplolated at tvals and depth z (mag_vs_depth reduces to zero at top)

        assert_allclose(dim1sin_avgf(self.m,
                                  self.outz,
                                  self.tvals,
                                  self.v_E_Igamv_the,
                                  drn=0,
                                  bot_vs_time = [self.bot_vs_time],
                                  bot_omega_phase=[self.omega_phase]),

                                  np.array([[ 1*0.3*np.cos(1*1+2)+1.6275434,  2*0.3*np.cos(1*3+2)+1.6275434],
                                            [ 1*0.5*np.cos(1*1+2)+2.29709903,  2*0.5*np.cos(1*3+2)+2.29709903]]))
#
    def test_bot_vs_time_top_vs_time_drn_1_double_loads(self):
        #expected is from:
        #test no_bc + top_vs_time and bot_vs time interplolated at tvals and depth z (mag_vs_depth is uniform from top plus reduces to zero at top)

        assert_allclose(dim1sin_avgf(self.m,
                                  self.outz,
                                  self.tvals,
                                  self.v_E_Igamv_the,
                                  drn=1,
                                  top_vs_time = [self.top_vs_time, self.top_vs_time],
                                  bot_vs_time = [self.bot_vs_time, self.bot_vs_time]),

                                  np.array([[ 2+2*0.3+1.6275434 ,  4+4*0.3+1.6275434 ],
                                            [ 2+2*0.5+2.29709903,  4+4*0.5+2.29709903]]))


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])