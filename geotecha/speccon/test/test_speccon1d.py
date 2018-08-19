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
from geotecha.speccon.speccon1d import dim1sin_integrate_af
from geotecha.speccon.speccon1d import dim1sin_E_Igamv_the_BC_abf_linear
from geotecha.speccon.speccon1d import dim1sin_E_Igamv_the_BC_D_aDf_linear
from geotecha.speccon.speccon1d import dim1sin_E_Igamv_the_BC_aDfDt_linear
from geotecha.speccon.speccon1d import dim1sin_E_Igamv_the_BC_deltaf_linear
from geotecha.speccon.speccon1d import dim1sin_E_Igamv_the_abmag_bilinear
from geotecha.speccon.speccon1d import dim1sin_E_Igamv_the_aDmagDt_bilinear
from geotecha.speccon.speccon1d import dim1sin_E_Igamv_the_deltamag_linear

class test_dim1sin_f(unittest.TestCase):
    """tests for dim1sin_f

    These are not exhasutive tests.

     - I've used top_vs_time=bot_vs_time, top_omega_phase=bot_omega_phase,
       so I can't tell if the code uses each correctly of if I've typed
       bot_vs_time instead of top_vs_time

    """
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
    def test_bot_vs_time_drn_0_omega_phase(self):
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
    """tests for dim1sin_avgf

    These are not exhasutive tests.

     - I've used top_vs_time=bot_vs_time, top_omega_phase=bot_omega_phase,
       so I can't tell if the code uses each correctly of if I've typed
       bot_vs_time instead of top_vs_time
       """
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

    def test_bot_vs_time_drn_0_omega_phase(self):
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


class test_dim1sin_integrate_af(unittest.TestCase):
    """tests for dim1sin_integrate_af

    see geotecha.speccon.test.speccon1d_test_data_gen.py
    for test case data generation

    These are not exhasutive tests.

     - I've used top_vs_time=bot_vs_time, top_omega_phase=bot_omega_phase,
       so I can't tell if the code uses each correctly of if I've typed
       bot_vs_time instead of top_vs_time

    """
    #dim1sin_integrate_af(m, z, tvals, v_E_Igamv_the, drn, a, top_vs_time = None, bot_vs_time=None, top_omega_phase=None, bot_omega_phase=None)

#    artificallially create 3 eigs, 2 zs and 2 ts

    outz = np.array([[0.2, 0.4], [0.4, 0.6]])
    z1 = outz[:, 0]
    z2 = outz[:, 1]
    m = np.array([1.0,2.0, 3.0])
    v_E_Igamv_the = np.ones((3,2), dtype=float)
    tvals = np.array([1.0, 3])
    top_vs_time = PolyLine([0,2,4],[0,2,2])
    bot_vs_time = PolyLine([0,2,4],[0,2,2])
    omega_phase = (1,2)
    a = PolyLine([0, 1], [1, 2])# y = 1 + z
    g = np.array([1.0,2.0])# this is interpolated from top_vs_time at t = 1, 3

    def test_no_BC(self):
        #expected is from:

        assert_allclose(dim1sin_integrate_af(self.m,
                                  self.outz,
                                  self.tvals,
                                  self.v_E_Igamv_the,
                                  drn=0,
                                  a = self.a),

                                  np.array([[ 0.42612566,  0.42612566],
                                            [ 0.69057191,  0.69057191]]))

    def test_top_vs_time_drn_0(self):

        assert_allclose(dim1sin_integrate_af(self.m,
                                  self.outz,
                                  self.tvals,
                                  self.v_E_Igamv_the,
                                  drn=0,
                                  a=self.a,
                                  top_vs_time = [self.top_vs_time]),

                                  np.array([[ 0.60745899,  0.78879232],
                                            [ 0.83990524,  0.98923858]]))
#
    def test_top_vs_time_drn_1(self):
        assert_allclose(dim1sin_integrate_af(self.m,
                                  self.outz,
                                  self.tvals,
                                  self.v_E_Igamv_the,
                                  drn=1,
                                  a = self.a,
                                  top_vs_time = [self.top_vs_time]),

                                  np.array([[ 0.68612566,  0.94612566],
                                            [ 0.99057191,  1.29057191]]))
#
    def test_bot_vs_time_drn_0(self):


        assert_allclose(dim1sin_integrate_af(self.m,
                                  self.outz,
                                  self.tvals,
                                  self.v_E_Igamv_the,
                                  drn=0,
                                  a=self.a,
                                  bot_vs_time = [self.bot_vs_time]),

                                  np.array([[ 0.50479232,  0.58345899],
                                            [ 0.84123858,  0.99190524]]))

    def test_bot_vs_time_top_vs_time_drn_1(self):


        assert_allclose(dim1sin_integrate_af(self.m,
                                  self.outz,
                                  self.tvals,
                                  self.v_E_Igamv_the,
                                  drn=1,
                                  a=self.a,
                                  top_vs_time = [self.top_vs_time],
                                  bot_vs_time = [self.bot_vs_time]),

                                  np.array([[ 0.76479232,  1.10345899],
                                            [ 1.14123858,  1.59190524]]))

    def test_top_vs_time_drn_0_omega_phase(self):


        assert_allclose(dim1sin_integrate_af(self.m,
                                  self.outz,
                                  self.tvals,
                                  self.v_E_Igamv_the,
                                  drn=0,
                                  a=self.a,
                                  top_vs_time = [self.top_vs_time],
                                  top_omega_phase=[self.omega_phase]),

                                  np.array([[ 0.24660702,  0.52900048],
                                            [ 0.54273303,  0.77529235]]))

    def test_bot_vs_time_drn_0_omega_phase(self):


        assert_allclose(dim1sin_integrate_af(self.m,
                                  self.outz,
                                  self.tvals,
                                  self.v_E_Igamv_the,
                                  drn=0,
                                  a=self.a,
                                  bot_vs_time = [self.bot_vs_time],
                                  bot_omega_phase=[self.omega_phase]),

                                  np.array([[ 0.34824625,  0.47075517],
                                            [ 0.54141304,  0.77604878]]))

    def test_bot_vs_time_top_vs_time_drn_1_double_loads(self):

        assert_allclose(dim1sin_integrate_af(self.m,
                                  self.outz,
                                  self.tvals,
                                  self.v_E_Igamv_the,
                                  drn=1,
                                  a=self.a,
                                  top_vs_time = [self.top_vs_time, self.top_vs_time],
                                  bot_vs_time = [self.bot_vs_time, self.bot_vs_time]),

                                  np.array([[ 1.10345899,  1.78079232],
                                            [ 1.59190524,  2.49323858]]))

class test_dim1sin_E_Igamv_the_BC_abf_linear(unittest.TestCase):
    """tests for dim1sin_E_Igamv_the_BC_abf_linear

    see geotecha.speccon.test.speccon1d_test_data_gen.py
    for test case data generation

    These are not exhasutive tests.

     - I've used top_vs_time=bot_vs_time, top_omega_phase=bot_omega_phase,
       so I can't tell if the code uses each correctly of if I've typed
       bot_vs_time instead of top_vs_time
     - I've the test case data was generated by evaluating the function
       i.e. it's not an independant test.  I think it is OK because
       the Speccon1dVR testing whcih uses all these fns uses independant data
    """
    #dim1sin_E_Igamv_the_BC_abf_linear(drn, m, eigs, tvals, Igamv, a, b, top_vs_time, bot_vs_time, top_omega_phase=None, bot_omega_phase=None, dT=1.0):

    outz = np.array([[0.2, 0.4], [0.4, 0.6]])
    z1 = outz[:, 0]
    z2 = outz[:, 1]
    m = np.array([1.0,2.0, 3.0])
    v_E_Igamv_the = np.ones((3,2), dtype=float)
    tvals = np.array([1.0, 3])
    top_vs_time = PolyLine([0,2,4],[0,2,2])
    bot_vs_time = PolyLine([0,2,4],[0,2,2])
    omega_phase = (1,2)
    a = PolyLine([0, 1], [1, 2])# y = 1 + z
    b = PolyLine([0, 1], [1, 2])# y = 1 + z
    g = np.array([1.0,2.0])# this is interpolated from top_vs_time at t = 1, 3
    Igamv = np.identity(3)
    eigs = np.ones(3)


    def test_no_BC(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_abf_linear(
            **{'drn': 0, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'a': self.a, 'b':self.b, 'top_vs_time': None, 'bot_vs_time':None}),

                                  np.array([[ 0.,  0.],
                                            [ 0.,  0.],
                                            [ 0.,  0.]]))

    def test_top_vs_time_drn_0(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_abf_linear(
            **{'drn': 0, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'a': self.a, 'b':self.b, 'top_vs_time': [self.top_vs_time], 'bot_vs_time':None}),

                                  np.array([[ 0.13262919,  0.60636726],
                                       [ 0.21993155,  1.00550484],
                                       [ 0.23855946,  1.09066975]]))


    def test_top_vs_time_drn_1(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_abf_linear(
            **{'drn': 1, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'a': self.a, 'b':self.b, 'top_vs_time': [self.top_vs_time], 'bot_vs_time':None}),

                                  np.array([[ 0.47282784,  2.1617211 ],
                                       [ 0.69439245,  3.17469213],
                                       [ 0.57706911,  2.63830166]]))


    def test_bot_vs_time_drn_0(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_abf_linear(
            **{'drn': 0, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'a': self.a, 'b':self.b, 'top_vs_time': None, 'bot_vs_time':[self.bot_vs_time]}),

                                  np.array([[ 0.34019865,  1.55535384],
                                   [ 0.4744609 ,  2.16918729],
                                   [ 0.33850965,  1.54763191]]))

    def test_bot_vs_time_top_vs_time_drn_1(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_abf_linear(
            **{'drn': 1, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'a': self.a, 'b':self.b, 'top_vs_time': [self.top_vs_time], 'bot_vs_time':[self.bot_vs_time]}),

                                  np.array([[ 0.81302649,  3.71707495],
                                       [ 1.16885336,  5.34387942],
                                       [ 0.91557876,  4.18593358]]))

    def test_top_vs_time_drn_0_omega_phase(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_abf_linear(
            **{'drn': 0, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'a': self.a, 'b':self.b,
            'top_vs_time': [self.top_vs_time],
            'bot_vs_time':None,
            'top_omega_phase': [self.omega_phase]}),

                                  np.array([[-0.1181581 , -0.18510014],
                                   [-0.19593495, -0.30694118],
                                   [-0.21253038, -0.33293869]]))

    def test_bot_vs_time_drn_0_omega_phase(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_abf_linear(
            **{'drn': 0, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'a': self.a, 'b':self.b,
            'top_vs_time': None,
            'bot_vs_time':[self.bot_vs_time],
            'bot_omega_phase': [self.omega_phase]}),

                                  np.array([[-0.30307978, -0.47478852],
                                   [-0.42269275, -0.66216779],
                                   [-0.30157506, -0.47243132]]))

    def test_bot_vs_time_top_vs_time_drn_1_double_loads(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_abf_linear(
            **{'drn': 1, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'a': self.a, 'b':self.b, 'top_vs_time': [self.top_vs_time,self.top_vs_time], 'bot_vs_time':[self.bot_vs_time,self.bot_vs_time]}),

                                  np.array([[  1.62605298,   7.43414989],
                                   [  2.33770671,  10.68775884],
                                   [  1.83115753,   8.37186715]]))


class test_dim1sin_E_Igamv_the_BC_D_aDf_linear(unittest.TestCase):
    """tests for dim1sin_E_Igamv_the_BC_D_aDf_linear

    see geotecha.speccon.test.speccon1d_test_data_gen.py
    for test case data generation

    These are not exhasutive tests.

     - I've used top_vs_time=bot_vs_time, top_omega_phase=bot_omega_phase,
       so I can't tell if the code uses each correctly of if I've typed
       bot_vs_time instead of top_vs_time
     - I've not included layers in  the a part
     - I've the test case data was generated by evaluating the function
       i.e. it's not an independant test.  I think it is OK because
       the Speccon1dVR testing whcih uses all these fns uses independant data
    """
    #dim1sin_E_Igamv_the_BC_D_aDf_linear(drn, m, eigs, tvals, Igamv, a, top_vs_time, bot_vs_time, top_omega_phase=None, bot_omega_phase=None, dT=1.0):

    outz = np.array([[0.2, 0.4], [0.4, 0.6]])
    z1 = outz[:, 0]
    z2 = outz[:, 1]
    m = np.array([1.0,2.0, 3.0])
    v_E_Igamv_the = np.ones((3,2), dtype=float)
    tvals = np.array([1.0, 3])
    top_vs_time = PolyLine([0,2,4],[0,2,2])
    bot_vs_time = PolyLine([0,2,4],[0,2,2])
    omega_phase = (1,2)
    a = PolyLine([0, 1], [1, 2])# y = 1 + z
    b = PolyLine([0, 1], [1, 2])# y = 1 + z
    g = np.array([1.0,2.0])# this is interpolated from top_vs_time at t = 1, 3
    Igamv = np.identity(3)
    eigs = np.ones(3)


    def test_no_BC(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_D_aDf_linear(
            **{'drn': 0, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'a': self.a,  'top_vs_time': None, 'bot_vs_time':None}),

                                  np.array([[ 0.,  0.],
                                            [ 0.,  0.],
                                            [ 0.,  0.]]))

    def test_top_vs_time_drn_0(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_D_aDf_linear(
            **{'drn': 0, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'a': self.a,  'top_vs_time': [self.top_vs_time], 'bot_vs_time':None}),

                                  np.array([[-0.16911333, -0.77316906],
       [-0.26048565, -1.19091408],
       [-0.24402578, -1.11566119]]))


    def test_top_vs_time_drn_1(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_D_aDf_linear(
            **{'drn': 1, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'a': self.a,  'top_vs_time': [self.top_vs_time], 'bot_vs_time':None}),

                                  np.array([[ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.]]))


    def test_bot_vs_time_drn_0(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_D_aDf_linear(
            **{'drn': 0, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'a': self.a,  'top_vs_time': None, 'bot_vs_time':[self.bot_vs_time]}),

                                  np.array([[ 0.16911333,  0.77316906],
                               [ 0.26048565,  1.19091408],
                               [ 0.24402578,  1.11566119]]))

    def test_bot_vs_time_top_vs_time_drn_1(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_D_aDf_linear(
            **{'drn': 1, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'a': self.a,  'top_vs_time': [self.top_vs_time], 'bot_vs_time':[self.bot_vs_time]}),

                                  np.array([[ 0.16911333,  0.77316906],
                                   [ 0.26048565,  1.19091408],
                                   [ 0.24402578,  1.11566119]]))

    def test_top_vs_time_drn_0_omega_phase(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_D_aDf_linear(
            **{'drn': 0, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'a': self.a,
            'top_vs_time': [self.top_vs_time],
            'bot_vs_time':None,
            'top_omega_phase': [self.omega_phase]}),

                                  np.array([[ 0.15066148,  0.23601818],
                                   [ 0.23206422,  0.36353935],
                                   [ 0.21740027,  0.3405676 ]]))

    def test_bot_vs_time_drn_0_omega_phase(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_D_aDf_linear(
            **{'drn': 0, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'a': self.a,
            'top_vs_time': None,
            'bot_vs_time':[self.bot_vs_time],
            'bot_omega_phase': [self.omega_phase]}),

                                  np.array([[-0.15066148, -0.23601818],
                                   [-0.23206422, -0.36353935],
                                   [-0.21740027, -0.3405676 ]]))

    def test_bot_vs_time_top_vs_time_drn_1_double_loads(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_D_aDf_linear(
            **{'drn': 1, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'a': self.a,  'top_vs_time': [self.top_vs_time,self.top_vs_time], 'bot_vs_time':[self.bot_vs_time,self.bot_vs_time]}),

                                  np.array([[ 0.33822666,  1.54633812],
                                   [ 0.52097131,  2.38182817],
                                   [ 0.48805155,  2.23132237]]))


class test_dim1sin_E_Igamv_the_BC_aDfDt_linear(unittest.TestCase):
    """tests for dim1sin_E_Igamv_the_BC_aDfDt_linear

    see geotecha.speccon.test.speccon1d_test_data_gen.py
    for test case data generation

    These are not exhasutive tests.

     - I've used top_vs_time=bot_vs_time, top_omega_phase=bot_omega_phase,
       so I can't tell if the code uses each correctly of if I've typed
       bot_vs_time instead of top_vs_time
     - I've not included layers in  the a part
     - I've the test case data was generated by evaluating the function
       i.e. it's not an independant test.  I think it is OK because
       the Speccon1dVR testing whcih uses all these fns uses independant data
    """
    #dim1sin_E_Igamv_the_BC_aDfDt_linear(drn, m, eigs, tvals, Igamv, a, top_vs_time, bot_vs_time, top_omega_phase=None, bot_omega_phase=None, dT=1.0):

    outz = np.array([[0.2, 0.4], [0.4, 0.6]])
    z1 = outz[:, 0]
    z2 = outz[:, 1]
    m = np.array([1.0,2.0, 3.0])
    v_E_Igamv_the = np.ones((3,2), dtype=float)
    tvals = np.array([1.0, 3])
    top_vs_time = PolyLine([0,2,4],[0,2,2])
    bot_vs_time = PolyLine([0,2,4],[0,2,2])
    omega_phase = (1,2)
    a = PolyLine([0, 1], [1, 2])# y = 1 + z
    b = PolyLine([0, 1], [1, 2])# y = 1 + z
    g = np.array([1.0,2.0])# this is interpolated from top_vs_time at t = 1, 3
    Igamv = np.identity(3)
    eigs = np.ones(3)


    def test_no_BC(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_aDfDt_linear(
            **{'drn': 0, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'a': self.a,  'top_vs_time': None, 'bot_vs_time':None}),

                                  np.array([[ 0.,  0.],
                                            [ 0.,  0.],
                                            [ 0.,  0.]]))

    def test_top_vs_time_drn_0(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_aDfDt_linear(
            **{'drn': 0, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'a': self.a,  'top_vs_time': [self.top_vs_time], 'bot_vs_time':None}),

                                  np.array([[ 0.14946707,  0.07521403],
       [ 0.25246136,  0.12704228],
       [ 0.28406245,  0.14294441]]))


    def test_top_vs_time_drn_1(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_aDfDt_linear(
            **{'drn': 1, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'a': self.a,  'top_vs_time': [self.top_vs_time], 'bot_vs_time':None}),

                                  np.array([[ 0.48095928,  0.24202579],
       [ 0.72281165,  0.36372947],
       [ 0.63781491,  0.32095785]]))


    def test_bot_vs_time_drn_0(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_aDfDt_linear(
            **{'drn': 0, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'a': self.a,  'top_vs_time': None, 'bot_vs_time':[self.bot_vs_time]}),

                                  np.array([[ 0.33149221,  0.16681176],
       [ 0.47035029,  0.23668719],
       [ 0.35375246,  0.17801345]]))

    def test_bot_vs_time_top_vs_time_drn_1(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_aDfDt_linear(
            **{'drn': 1, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'a': self.a,  'top_vs_time': [self.top_vs_time], 'bot_vs_time':[self.bot_vs_time]}),

                                  np.array([[ 0.81245149,  0.40883755],
       [ 1.19316194,  0.60041665],
       [ 0.99156737,  0.4989713 ]]))

    def test_top_vs_time_drn_0_omega_phase(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_aDfDt_linear(
            **{'drn': 0, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'a': self.a,
            'top_vs_time': [self.top_vs_time],
            'bot_vs_time':None,
            'top_omega_phase': [self.omega_phase]}),

                                  np.array([[-0.15659179,  0.2555458 ],
       [-0.26449556,  0.4316365 ],
       [-0.297603  ,  0.48566529]]))

    def test_bot_vs_time_drn_0_omega_phase(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_aDfDt_linear(
            **{'drn': 0, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'a': self.a,
            'top_vs_time': None,
            'bot_vs_time':[self.bot_vs_time],
            'bot_omega_phase': [self.omega_phase]}),

                                  np.array([[-0.34729361,  0.56675657],
       [-0.49277071,  0.80416404],
       [-0.37061495,  0.60481521]]))

    def test_bot_vs_time_top_vs_time_drn_1_double_loads(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_aDfDt_linear(
            **{'drn': 1, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'a': self.a,  'top_vs_time': [self.top_vs_time,self.top_vs_time], 'bot_vs_time':[self.bot_vs_time,self.bot_vs_time]}),

                                  np.array([[ 1.62490297,  0.8176751 ],
                                   [ 2.38632387,  1.20083331],
                                   [ 1.98313474,  0.9979426 ]]))

class test_dim1sin_E_Igamv_the_BC_deltaf_linear(unittest.TestCase):
    """tests for dim1sin_E_Igamv_the_BC_deltaf_linear

    see geotecha.speccon.test.speccon1d_test_data_gen.py
    for test case data generation

    These are not exhasutive tests.

     - I've used top_vs_time=bot_vs_time, top_omega_phase=bot_omega_phase,
       so I can't tell if the code uses each correctly of if I've typed
       bot_vs_time instead of top_vs_time
     - I've not included layers in  the a part
     - I've the test case data was generated by evaluating the function
       i.e. it's not an independant test.  I think it is OK because
       the Speccon1dVR testing whcih uses all these fns uses independant data
    """
    #dim1sin_E_Igamv_the_BC_deltaf_linear(drn, m, eigs, tvals, Igamv, zvals, pseudo_k, top_vs_time, bot_vs_time, top_omega_phase=None, bot_omega_phase=None, dT=1.0)

    outz = np.array([[0.2, 0.4], [0.4, 0.6]])
    z1 = outz[:, 0]
    z2 = outz[:, 1]
    m = np.array([1.0,2.0, 3.0])
    v_E_Igamv_the = np.ones((3,2), dtype=float)
    tvals = np.array([1.0, 3])
    top_vs_time = PolyLine([0,2,4],[0,2,2])
    bot_vs_time = PolyLine([0,2,4],[0,2,2])
    omega_phase = (1,2)
    a = PolyLine([0, 1], [1, 2])# y = 1 + z
    b = PolyLine([0, 1], [1, 2])# y = 1 + z
    g = np.array([1.0,2.0])# this is interpolated from top_vs_time at t = 1, 3
    Igamv = np.identity(3)
    eigs = np.ones(3)
    zvals=[0.2]
    pseudo_k=[1000]

    def test_no_BC(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_deltaf_linear(
            **{'drn': 0, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'zvals': self.zvals, 'pseudo_k':self.pseudo_k,  'top_vs_time': None, 'bot_vs_time':None}),

                                  np.array([[ 0.,  0.],
                                            [ 0.,  0.],
                                            [ 0.,  0.]]))

    def test_top_vs_time_drn_0(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_deltaf_linear(
            **{'drn': 0, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'zvals': self.zvals, 'pseudo_k':self.pseudo_k,  'top_vs_time': [self.top_vs_time], 'bot_vs_time':None}),

                                  np.array([[  58.46908991,  267.3147702 ],
       [ 114.60720172,  523.97254408],
       [ 166.17628606,  759.74118611]]))


    def test_top_vs_time_drn_1(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_deltaf_linear(
            **{'drn': 1, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'zvals': self.zvals, 'pseudo_k':self.pseudo_k,  'top_vs_time': [self.top_vs_time], 'bot_vs_time':None}),

                                  np.array([[  73.08636239,  334.14346275],
       [ 143.25900215,  654.9656801 ],
       [ 207.72035757,  949.67648264]]))


    def test_bot_vs_time_drn_0(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_deltaf_linear(
            **{'drn': 0, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'zvals': self.zvals, 'pseudo_k':self.pseudo_k,  'top_vs_time': None, 'bot_vs_time':[self.bot_vs_time]}),

                                  np.array([[  14.61727248,   66.82869255],
       [  28.65180043,  130.99313602],
       [  41.54407151,  189.93529653]]))

    def test_bot_vs_time_top_vs_time_drn_1(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_deltaf_linear(
            **{'drn': 1, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'zvals': self.zvals, 'pseudo_k':self.pseudo_k,  'top_vs_time': [self.top_vs_time], 'bot_vs_time':[self.bot_vs_time]}),

                                  np.array([[   87.70363487,   400.97215531],
       [  171.91080258,   785.95881612],
       [  249.26442909,  1139.61177917]]))

    def test_top_vs_time_drn_0_omega_phase(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_deltaf_linear(
            **{'drn': 0, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'zvals': self.zvals, 'pseudo_k':self.pseudo_k,
            'top_vs_time': [self.top_vs_time],
            'bot_vs_time':None,
            'top_omega_phase': [self.omega_phase]}),

                                  np.array([[ -52.08956221,  -81.6007135 ],
       [-102.10247796, -159.94826406],
       [-148.04489011, -231.91918207]]))

    def test_bot_vs_time_drn_0_omega_phase(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_deltaf_linear(
            **{'drn': 0, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'zvals': self.zvals, 'pseudo_k':self.pseudo_k,
            'top_vs_time': None,
            'bot_vs_time':[self.bot_vs_time],
            'bot_omega_phase': [self.omega_phase]}),

                                  np.array([[-13.02239055, -20.40017837],
       [-25.52561949, -39.98706601],
       [-37.01122253, -57.97979552]]))

    def test_bot_vs_time_top_vs_time_drn_1_double_loads(self):
        assert_allclose(dim1sin_E_Igamv_the_BC_deltaf_linear(
            **{'drn': 1, 'm': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
            'zvals': self.zvals, 'pseudo_k':self.pseudo_k,  'top_vs_time': [self.top_vs_time,self.top_vs_time], 'bot_vs_time':[self.bot_vs_time,self.bot_vs_time]}),

                                  np.array([[  175.40726974,   801.94431061],
       [  343.82160516,  1571.91763224],
       [  498.52885818,  2279.22355834]]))




class test_dim1sin_E_Igamv_the_abmag_bilinear(unittest.TestCase):
    """tests for dim1sin_E_Igamv_the_abmag_bilinear

    see geotecha.speccon.test.speccon1d_test_data_gen.py
    for test case data generation

    because this fn is basically a glorified wrapper we really just need to
    test if all the args are passed properly.

    """
    #dim1sin_E_Igamv_the_abmag_bilinear(m, eigs, tvals, Igamv, a, b, mag_vs_depth, mag_vs_time, omega_phase=None, dT=1.0):


    m = np.array([1.0,2.0, 3.0])
    v_E_Igamv_the = np.ones((3,2), dtype=float)
    tvals = np.array([1.0, 3])
    mag_vs_time = PolyLine([0,2,4],[0,2,2])
    mag_vs_depth = PolyLine([0, 1], [1, 2])#y = (1+z)

    omega_phase = (1,2)
    a = PolyLine([0, 1], [1, 2])# y = 1 + z
    b = PolyLine([0, 1], [1, 2])# y = 1 + z
    g = np.array([1.0,2.0])# this is interpolated from top_vs_time at t = 1, 3
    Igamv = np.identity(3)
    eigs = np.ones(3)





    def test_single_load(self):
        assert_allclose(dim1sin_E_Igamv_the_abmag_bilinear(
            **{'m': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
               'a':self.a, 'b':self.b,
                'mag_vs_depth': [self.mag_vs_depth], 'mag_vs_time': [self.mag_vs_time]}),

                                  np.array([[ 0.81302649,  3.71707495],
                                            [ 1.16885336,  5.34387942],
                                            [ 0.91557876,  4.18593358]]))

    def test_double_load(self):
        assert_allclose(dim1sin_E_Igamv_the_abmag_bilinear(
            **{'m': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
               'a':self.a, 'b':self.b,
                'mag_vs_depth': [self.mag_vs_depth]*2, 'mag_vs_time': [self.mag_vs_time]*2}),

                                  2*np.array([[ 0.81302649,  3.71707495],
                                            [ 1.16885336,  5.34387942],
                                            [ 0.91557876,  4.18593358]]))
    def test_omega_phase(self):
        assert_allclose(dim1sin_E_Igamv_the_abmag_bilinear(
            **{'m': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
               'a':self.a, 'b':self.b,
                'mag_vs_depth': [self.mag_vs_depth], 'mag_vs_time': [self.mag_vs_time], 'omega_phase': [self.omega_phase]}),

                                  np.array([[-0.72431765, -1.13467717],
                                           [-1.04132046, -1.63127676],
                                           [-0.81568051, -1.27780132]]))




class test_dim1sin_E_Igamv_the_aDmagDt_bilinear(unittest.TestCase):
    """tests for dim1sin_E_Igamv_the_aDmagDt_bilinear

    see geotecha.speccon.test.speccon1d_test_data_gen.py
    for test case data generation

    because this fn is basically a glorified wrapper we really just need to
    test if all the args are passed properly.

    """
    #dim1sin_E_Igamv_the_aDmagDt_bilinear(m, eigs, tvals, Igamv, a, mag_vs_depth, mag_vs_time, omega_phase = None, dT=1.0):


    m = np.array([1.0,2.0, 3.0])
    v_E_Igamv_the = np.ones((3,2), dtype=float)
    tvals = np.array([1.0, 3])
    mag_vs_time = PolyLine([0,2,4],[0,2,2])
    mag_vs_depth = PolyLine([0, 1], [1, 2])#y = (1+z)

    omega_phase = (1,2)
    a = PolyLine([0, 1], [1, 2])# y = 1 + z
    b = PolyLine([0, 1], [1, 2])# y = 1 + z
    g = np.array([1.0,2.0])# this is interpolated from top_vs_time at t = 1, 3
    Igamv = np.identity(3)
    eigs = np.ones(3)





    def test_single_load(self):
        assert_allclose(dim1sin_E_Igamv_the_aDmagDt_bilinear(
            **{'m': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
               'a':self.a,
                'mag_vs_depth': [self.mag_vs_depth], 'mag_vs_time': [self.mag_vs_time]}),

                                  np.array([[ 0.81245149,  0.40883755],
                                       [ 1.19316194,  0.60041665],
                                       [ 0.99156737,  0.4989713 ]]))

    def test_double_load(self):
        assert_allclose(dim1sin_E_Igamv_the_aDmagDt_bilinear(
            **{'m': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
               'a':self.a,
                'mag_vs_depth': [self.mag_vs_depth]*2, 'mag_vs_time': [self.mag_vs_time]*2}),

                                  2*np.array([[ 0.81245149,  0.40883755],
                                       [ 1.19316194,  0.60041665],
                                       [ 0.99156737,  0.4989713 ]]))
    def test_omega_phase(self):
        assert_allclose(dim1sin_E_Igamv_the_aDmagDt_bilinear(
            **{'m': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
               'a':self.a,
                'mag_vs_depth': [self.mag_vs_depth], 'mag_vs_time': [self.mag_vs_time], 'omega_phase': [self.omega_phase]}),

                                  np.array([[-0.85117901,  1.38905894],
                                           [-1.25003698,  2.03996458],
                                           [-1.0388329 ,  1.69529571]]))

class test_dim1sin_E_Igamv_the_deltamag_linear(unittest.TestCase):
    """tests for dim1sin_E_Igamv_the_deltamag_linear

    see geotecha.speccon.test.speccon1d_test_data_gen.py
    for test case data generation

    because this fn is basically a glorified wrapper we really just need to
    test if all the args are passed properly.

    """
    #dim1sin_E_Igamv_the_deltamag_linear(m, eigs, tvals, Igamv, zvals, a, mag_vs_time, omega_phase=None, dT=1.0):


    m = np.array([1.0,2.0, 3.0])
    v_E_Igamv_the = np.ones((3,2), dtype=float)
    tvals = np.array([1.0, 3])
    mag_vs_time = PolyLine([0,2,4],[0,2,2])
    mag_vs_depth = PolyLine([0, 1], [1, 2])#y = (1+z)

    omega_phase = (1,2)
    a = PolyLine([0, 1], [1, 2])# y = 1 + z
    b = PolyLine([0, 1], [1, 2])# y = 1 + z
    g = np.array([1.0,2.0])# this is interpolated from top_vs_time at t = 1, 3
    Igamv = np.identity(3)
    eigs = np.ones(3)
    zvals=[0.2]
    pseudo_k=[1000]



    def test_single_load(self):
        assert_allclose(dim1sin_E_Igamv_the_deltamag_linear(
            **{'m': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
               'zvals':self.zvals, 'pseudo_k':self.pseudo_k,
                'mag_vs_time': [self.mag_vs_time]}),

                                  np.array([[  73.08636239,  334.14346275],
                                       [ 143.25900215,  654.9656801 ],
                                       [ 207.72035757,  949.67648264]]))

    def test_double_load(self):
        assert_allclose(dim1sin_E_Igamv_the_deltamag_linear(
            **{'m': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
               'zvals':self.zvals*2, 'pseudo_k':self.pseudo_k*2,
                 'mag_vs_time': [self.mag_vs_time]*2}),

                                  2*np.array([[  73.08636239,  334.14346275],
                                               [ 143.25900215,  654.9656801 ],
                                               [ 207.72035757,  949.67648264]]))
    def test_omega_phase(self):
        assert_allclose(dim1sin_E_Igamv_the_deltamag_linear(
            **{'m': self.m, 'eigs':self.eigs, 'tvals':self.tvals,'Igamv':self.Igamv,
               'zvals':self.zvals, 'pseudo_k':self.pseudo_k,
                'mag_vs_time': [self.mag_vs_time], 'omega_phase': [self.omega_phase]}),

                                  np.array([[ -65.11195277, -102.00089187],
                                           [-127.62809745, -199.93533007],
                                           [-185.05611264, -289.89897759]]))


if __name__ == '__main__':

    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])