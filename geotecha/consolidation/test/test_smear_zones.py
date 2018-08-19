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
"""Some test routines for the smear_zones module

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
from geotecha.consolidation.smear_zones import mu_ideal
from geotecha.consolidation.smear_zones import mu_constant
from geotecha.consolidation.smear_zones import mu_linear
from geotecha.consolidation.smear_zones import mu_overlapping_linear
from geotecha.consolidation.smear_zones import mu_parabolic
from geotecha.consolidation.smear_zones import mu_piecewise_constant
from geotecha.consolidation.smear_zones import mu_piecewise_linear
from geotecha.consolidation.smear_zones import mu_well_resistance

from geotecha.consolidation.smear_zones import u_ideal
from geotecha.consolidation.smear_zones import u_constant
from geotecha.consolidation.smear_zones import u_linear
#from geotecha.consolidation.smear_zones import mu_overlapping_constant
from geotecha.consolidation.smear_zones import u_parabolic
from geotecha.consolidation.smear_zones import u_piecewise_constant
from geotecha.consolidation.smear_zones import u_piecewise_linear

from geotecha.consolidation.smear_zones import k_linear
from geotecha.consolidation.smear_zones import k_overlapping_linear
from geotecha.consolidation.smear_zones import k_parabolic
from geotecha.consolidation.smear_zones import re_from_drain_spacing
from geotecha.consolidation.smear_zones import back_calc_drain_spacing_from_eta

class test_mu_ideal(unittest.TestCase):
    """tests for mu_ideal"""

    def test_ideal(self):
        assert_allclose(mu_ideal(np.array([5,10,20,50,100])),
                        [0.936497825, 1.578343528,
                         2.253865374, 3.163688441,
                         3.855655749])

    def test_n_less_than_one(self):
        assert_raises(ValueError, mu_ideal, 0.5)

class test_mu_constant(unittest.TestCase):
    """tests for mu_constant"""

    def test_ideal(self):
        assert_allclose(mu_constant(np.array([5,10,20,50,100]),
                                    np.array([1,1,1,1,1]),
                                    np.array([5,10,20,50,100])),
                        [0.936497825, 1.578343528,
                         2.253865374, 3.163688441,
                         3.855655749])

    def test_constant(self):
        assert_allclose(mu_constant(np.array([5,10,20,50,100]),
                                    np.array([1.5, 2, 1, 4, 8]),
                                    np.array([1.6, 1, 5, 0.4, 4])),
                        [1.159679143, 1.578343528,
                         2.253865374, 2.335174298,
                         10.07573309])

    def test_n_less_than_one(self):
        assert_raises(ValueError, mu_constant, 0.5, 2, 5)

    def test_s_less_than_one(self):
        assert_raises(ValueError, mu_constant, 50, 0.5, 5)

    def test_kap_less_than_zero(self):
        assert_raises(ValueError, mu_constant, 50, 0.5, -5)

    def test_s_greater_than_n(self):
        assert_raises(ValueError, mu_constant, 50, 100, 5)

class test_mu_well_resistance(unittest.TestCase):
    """tests for mu_well_resistance"""
    #kh, qw, n, H, z=None
    def test_z(self):
        assert_allclose(mu_well_resistance(kh=2,qw=4,n=5,H=6,z=4),
                        48.25486315913922)

    def test_z_two_values(self):
        assert_allclose(mu_well_resistance(kh=2,qw=4,n=5,H=6,z=np.array([4,5])),
                        [48.25486315913922, 52.77875658030852])

    def test_z(self):
        assert_allclose(mu_well_resistance(kh=2,qw=4,n=5,H=6),
                        36.19114736935442)

    def test_n_less_than_one(self):
        assert_raises(ValueError, mu_well_resistance, 2, 4, 0.5, 6)




class test_mu_linear(unittest.TestCase):
    """tests for mu_linear"""

    def test_ideal(self):
        assert_allclose(mu_linear(np.array([5,10,20,50,100]),
                                    np.array([1,1,1,1,1]),
                                    np.array([5,10,20,50,100])),
                        [0.936497825, 1.578343528,
                         2.253865374, 3.163688441,
                         3.855655749])

    def test_linear(self):
        assert_allclose(mu_linear(np.array([5,10,20,50,100]),
                                    np.array([1.5, 2, 1, 4, 8]),
                                    np.array([1.6, 1, 5, 4, 4])),
                        [1.040117086, 1.578343528,
                         2.253865374, 4.774441621,
                         6.625207688])

    def test_n_less_than_one(self):
        assert_raises(ValueError, mu_linear, 0.5, 2, 5)

    def test_s_less_than_one(self):
        assert_raises(ValueError, mu_linear, 50, 0.5, 5)

    def test_kap_less_than_zero(self):
        assert_raises(ValueError, mu_linear, 50, 0.5, -5)

    def test_s_greater_than_n(self):
        assert_raises(ValueError, mu_linear, 50, 100, 5)

    def test_s_n_unequal_len(self):
        assert_raises(ValueError, mu_linear, [50,40], 100, 5)

class test_mu_overlapping_linear(unittest.TestCase):
    """tests for mu_overlapping_linear"""

    def test_ideal(self):
        assert_allclose(mu_overlapping_linear(np.array([5,10,20,50,100]),
                                    np.array([1,1,1,2,1]),
                                    np.array([5,10,20,1,100])),
                        [0.936497825, 1.578343528,
                         2.253865374, 3.163688441,
                         3.855655749])

    def test_linear(self):
        assert_allclose(mu_overlapping_linear(np.array([5,10,20,50,100]),
                                    np.array([1.5, 2, 1, 4, 8]),
                                    np.array([1.6, 1, 5, 4, 4])),
                        [1.040117086, 1.578343528,
                         2.253865374, 4.774441621,
                         6.625207688])

    def test_all_disturbed(self):
        assert_allclose(mu_overlapping_linear(np.array([5,10]),
                                    np.array([20, 30]),
                                    np.array([1.6, 1.5,])),
                        [1.498396521, 2.367515292])

    def test_intersecting(self):
        assert_allclose(mu_overlapping_linear(np.array([5,10]),
                                    np.array([7, 12]),
                                    np.array([1.6, 1.5,])),
                        [1.387620117, 2.200268994])

    def test_all_at_once(self):
        assert_allclose(mu_overlapping_linear(
                                    np.array([5,  10, 10, 10, 100]),
                                    np.array([1.5, 1, 30, 12, 8]),
                                    np.array([1.6, 1,1.5,1.5, 1])),
                        [1.040117086, 1.578343528,
                         2.367515292, 2.200268994,
                         3.855655749])

    def test_n_less_than_one(self):
        assert_raises(ValueError, mu_overlapping_linear, 0.5, 2, 5)

    def test_s_less_than_one(self):
        assert_raises(ValueError, mu_overlapping_linear, 50, 0.5, 5)

    def test_kap_less_than_zero(self):
        assert_raises(ValueError, mu_overlapping_linear, 50, 0.5, -5)


    def test_s_n_unequal_len(self):
        assert_raises(ValueError, mu_overlapping_linear, [50,40], 100, 5)


class test_mu_parabolic(unittest.TestCase):
    """tests for mu_parabolic"""

    def test_ideal(self):
        assert_allclose(mu_parabolic(np.array([5,10,20,50,100]),
                                    np.array([1,1,1,1,1]),
                                    np.array([5,10,20,50,100])),
                        [0.936497825, 1.578343528,
                         2.253865374, 3.163688441,
                         3.855655749])

    def test_parabolic(self):
        assert_allclose(mu_parabolic(np.array([5,10,20,50,100]),
                                    np.array([1.5, 2, 1, 4, 8]),
                                    np.array([1.6, 1, 5, 4, 4])),
                        [1.006231891, 1.578343528,
                         2.253865374, 4.258523315,
                         5.834098317])

    def test_n_less_than_one(self):
        assert_raises(ValueError, mu_parabolic, 0.5, 2, 5)

    def test_s_less_than_one(self):
        assert_raises(ValueError, mu_parabolic, 50, 0.5, 5)

    def test_kap_less_than_zero(self):
        assert_raises(ValueError, mu_parabolic, 50, 0.5, -5)

    def test_s_greater_than_n(self):
        assert_raises(ValueError, mu_parabolic, 50, 100, 5)

    def test_s_n_unequal_len(self):
        assert_raises(ValueError, mu_parabolic, [50,40], 100, 5)

class test_mu_piecewise_constant(unittest.TestCase):
    """tests for mu_piecewise_constant"""

    def test_ideal(self):
        assert_allclose(mu_piecewise_constant(5,
                                              1),
                        0.936497825)
    def test_ideal_multi(self):
        assert_allclose(mu_piecewise_constant([3,5],
                                              [1,1]),
                        0.936497825)

    def test_const(self):
        assert_allclose(mu_piecewise_constant([1.5, 5],
                                              [1.6, 1]),
                        1.159679143)

    def test_const_multi(self):
        assert_allclose(mu_piecewise_constant([1.3, 1.4, 1.5, 5],
                                              [1.6, 1.6, 1.6, 1]),
                        1.159679143)

    def test_const_two_smear_zones(self):
        assert_allclose(mu_piecewise_constant([1.5, 3, 5,],
                                              [2, 3, 1.0]),
                        2.253304564)

    def test_const_two_smear_zones_n_kap_m(self):
        assert_allclose(mu_piecewise_constant([1.5, 3],
                                              [2, 3], n=5, kap_m=1),
                        2.253304564)
    def test_const_two_smear_zones_n(self):
        assert_allclose(mu_piecewise_constant([1.5, 3, 4],
                                              [2, 3, 1], n=5),
                        2.253304564)

    def test_parabolic(self):
        """piecewise constant approximation of parabolic with n = 30, s=5, kap=2"""


        x = np.array(
        [   1.06779661,  1.13559322,  1.20338983,  1.27118644,
        1.33898305,  1.40677966,  1.47457627,  1.54237288,  1.61016949,
        1.6779661 ,  1.74576271,  1.81355932,  1.88135593,  1.94915254,
        2.01694915,  2.08474576,  2.15254237,  2.22033898,  2.28813559,
        2.3559322 ,  2.42372881,  2.49152542,  2.55932203,  2.62711864,
        2.69491525,  2.76271186,  2.83050847,  2.89830508,  2.96610169,
        3.03389831,  3.10169492,  3.16949153,  3.23728814,  3.30508475,
        3.37288136,  3.44067797,  3.50847458,  3.57627119,  3.6440678 ,
        3.71186441,  3.77966102,  3.84745763,  3.91525424,  3.98305085,
        4.05084746,  4.11864407,  4.18644068,  4.25423729,  4.3220339 ,
        4.38983051,  4.45762712,  4.52542373,  4.59322034,  4.66101695,
        4.72881356,  4.79661017,  4.86440678,  4.93220339,  5., 30       ])

        y = 1.0/np.array(
        [ 0.5       ,  0.51680552,  0.53332376,  0.54955473,  0.56549842,
        0.58115484,  0.59652399,  0.61160586,  0.62640046,  0.64090779,
        0.65512784,  0.66906061,  0.68270612,  0.69606435,  0.70913531,
        0.72191899,  0.7344154 ,  0.74662453,  0.75854639,  0.77018098,
        0.7815283 ,  0.79258834,  0.8033611 ,  0.8138466 ,  0.82404481,
        0.83395576,  0.84357943,  0.85291583,  0.86196495,  0.8707268 ,
        0.87920138,  0.88738868,  0.89528871,  0.90290147,  0.91022695,
        0.91726515,  0.92401609,  0.93047975,  0.93665613,  0.94254525,
        0.94814708,  0.95346165,  0.95848894,  0.96322896,  0.9676817 ,
        0.97184717,  0.97572537,  0.97931629,  0.98261994,  0.98563631,
        0.98836541,  0.99080724,  0.99296179,  0.99482907,  0.99640908,
        0.99770181,  0.99870727,  0.99942545,  0.99985636,  1.        ])

        assert_allclose(mu_piecewise_constant(x,y),
                        3.2542191564, atol=0.03)

    def test_linear(self):
        """piecewise constant approximation of linear with n = 30, s=5, kap=2"""


        x = np.array(
        [   1.06779661,  1.13559322,  1.20338983,  1.27118644,
        1.33898305,  1.40677966,  1.47457627,  1.54237288,  1.61016949,
        1.6779661 ,  1.74576271,  1.81355932,  1.88135593,  1.94915254,
        2.01694915,  2.08474576,  2.15254237,  2.22033898,  2.28813559,
        2.3559322 ,  2.42372881,  2.49152542,  2.55932203,  2.62711864,
        2.69491525,  2.76271186,  2.83050847,  2.89830508,  2.96610169,
        3.03389831,  3.10169492,  3.16949153,  3.23728814,  3.30508475,
        3.37288136,  3.44067797,  3.50847458,  3.57627119,  3.6440678 ,
        3.71186441,  3.77966102,  3.84745763,  3.91525424,  3.98305085,
        4.05084746,  4.11864407,  4.18644068,  4.25423729,  4.3220339 ,
        4.38983051,  4.45762712,  4.52542373,  4.59322034,  4.66101695,
        4.72881356,  4.79661017,  4.86440678,  4.93220339,  5., 30       ])

        y = 1.0/np.array(
        [ 0.5       ,  0.50847458,  0.51694915,  0.52542373,  0.53389831,
        0.54237288,  0.55084746,  0.55932203,  0.56779661,  0.57627119,
        0.58474576,  0.59322034,  0.60169492,  0.61016949,  0.61864407,
        0.62711864,  0.63559322,  0.6440678 ,  0.65254237,  0.66101695,
        0.66949153,  0.6779661 ,  0.68644068,  0.69491525,  0.70338983,
        0.71186441,  0.72033898,  0.72881356,  0.73728814,  0.74576271,
        0.75423729,  0.76271186,  0.77118644,  0.77966102,  0.78813559,
        0.79661017,  0.80508475,  0.81355932,  0.8220339 ,  0.83050847,
        0.83898305,  0.84745763,  0.8559322 ,  0.86440678,  0.87288136,
        0.88135593,  0.88983051,  0.89830508,  0.90677966,  0.91525424,
        0.92372881,  0.93220339,  0.94067797,  0.94915254,  0.95762712,
        0.96610169,  0.97457627,  0.98305085,  0.99152542,  1.        ])

        assert_allclose(mu_piecewise_constant(x,y),
                        3.482736134, atol=0.03)


    def test_s_increasing(self):
        assert_raises(ValueError, mu_piecewise_constant,[1.5,1,2], [1,1,1] )

    def test_s_less_than_one(self):
        assert_raises(ValueError, mu_piecewise_constant,[0.5,1,2], [1,1,1] )

    def test_kap_less_than_zero(self):
        assert_raises(ValueError, mu_piecewise_constant,[1.5,1.6,2], [-2,1,1] )

    def test_s_n_unequal_len(self):
        assert_raises(ValueError, mu_piecewise_constant, [2,4], [1])

class test_mu_piecewise_linear(unittest.TestCase):
    """tests for mu_piecewise_linear"""

    def test_ideal(self):
        assert_allclose(mu_piecewise_linear([1, 5],
                                              [1,1]),
                        0.936497825)
    def test_ideal_multi(self):
        assert_allclose(mu_piecewise_linear([1,3,5],
                                              [1,1,1]),
                        0.936497825)

    def test_const(self):
        assert_allclose(mu_piecewise_linear([1, 1.5, 1.5, 5],
                                              [1.6, 1.6, 1, 1]),
                        1.159679143)

    def test_const_multi(self):
        assert_allclose(mu_piecewise_linear([1.0, 1.3, 1.4, 1.5, 1.5, 5.0],
                                            [1.6, 1.6, 1.6, 1.6, 1.0, 1.0]),
                        1.159679143)

    def test_const_two_smear_zones(self):
        assert_allclose(mu_piecewise_linear([1.0, 1.5, 1.5, 3.0, 3.0, 5.0],
                                            [2.0, 2.0, 3.0, 3.0, 1.0, 1.0]),
                        2.253304564)
    def test_const_two_smear_zones_n_kap_m(self):
        assert_allclose(mu_piecewise_linear([1.0, 1.5, 1.5, 3.0, 3.0],
                                            [2.0, 2.0, 3.0, 3.0, 1.0], n=5, kap_m=1),
                        2.253304564)

    def test_const_two_smear_zones_n(self):
        assert_allclose(mu_piecewise_linear([1.0, 1.5, 1.5, 3.0, 3.0, 4.0],
                                            [2.0, 2.0, 3.0, 3.0, 1.0, 1.0], n=5),
                        2.253304564)

    def test_parabolic(self):
        """piecewise constant approximation of parabolic with n = 30, s=5, kap=2"""


        x = np.array(
        [1.,    1.06779661,  1.13559322,  1.20338983,  1.27118644,
        1.33898305,  1.40677966,  1.47457627,  1.54237288,  1.61016949,
        1.6779661 ,  1.74576271,  1.81355932,  1.88135593,  1.94915254,
        2.01694915,  2.08474576,  2.15254237,  2.22033898,  2.28813559,
        2.3559322 ,  2.42372881,  2.49152542,  2.55932203,  2.62711864,
        2.69491525,  2.76271186,  2.83050847,  2.89830508,  2.96610169,
        3.03389831,  3.10169492,  3.16949153,  3.23728814,  3.30508475,
        3.37288136,  3.44067797,  3.50847458,  3.57627119,  3.6440678 ,
        3.71186441,  3.77966102,  3.84745763,  3.91525424,  3.98305085,
        4.05084746,  4.11864407,  4.18644068,  4.25423729,  4.3220339 ,
        4.38983051,  4.45762712,  4.52542373,  4.59322034,  4.66101695,
        4.72881356,  4.79661017,  4.86440678,  4.93220339,  5., 30       ])

        y = 1.0/np.array(
        [ 0.5       ,  0.51680552,  0.53332376,  0.54955473,  0.56549842,
        0.58115484,  0.59652399,  0.61160586,  0.62640046,  0.64090779,
        0.65512784,  0.66906061,  0.68270612,  0.69606435,  0.70913531,
        0.72191899,  0.7344154 ,  0.74662453,  0.75854639,  0.77018098,
        0.7815283 ,  0.79258834,  0.8033611 ,  0.8138466 ,  0.82404481,
        0.83395576,  0.84357943,  0.85291583,  0.86196495,  0.8707268 ,
        0.87920138,  0.88738868,  0.89528871,  0.90290147,  0.91022695,
        0.91726515,  0.92401609,  0.93047975,  0.93665613,  0.94254525,
        0.94814708,  0.95346165,  0.95848894,  0.96322896,  0.9676817 ,
        0.97184717,  0.97572537,  0.97931629,  0.98261994,  0.98563631,
        0.98836541,  0.99080724,  0.99296179,  0.99482907,  0.99640908,
        0.99770181,  0.99870727,  0.99942545,  0.99985636,  1., 1.        ])

        assert_allclose(mu_piecewise_linear(x,y),
                        3.2542191564, atol=1e-4)

    def test_linear(self):
        """piecewise linear approximation of linear with n = 30, s=5, kap=2"""


        x = np.array(
        [1.,   1.06779661,  1.13559322,  1.20338983,  1.27118644,
        1.33898305,  1.40677966,  1.47457627,  1.54237288,  1.61016949,
        1.6779661 ,  1.74576271,  1.81355932,  1.88135593,  1.94915254,
        2.01694915,  2.08474576,  2.15254237,  2.22033898,  2.28813559,
        2.3559322 ,  2.42372881,  2.49152542,  2.55932203,  2.62711864,
        2.69491525,  2.76271186,  2.83050847,  2.89830508,  2.96610169,
        3.03389831,  3.10169492,  3.16949153,  3.23728814,  3.30508475,
        3.37288136,  3.44067797,  3.50847458,  3.57627119,  3.6440678 ,
        3.71186441,  3.77966102,  3.84745763,  3.91525424,  3.98305085,
        4.05084746,  4.11864407,  4.18644068,  4.25423729,  4.3220339 ,
        4.38983051,  4.45762712,  4.52542373,  4.59322034,  4.66101695,
        4.72881356,  4.79661017,  4.86440678,  4.93220339,  5., 30       ])

        y = 1.0/np.array(
        [ 0.5       ,  0.50847458,  0.51694915,  0.52542373,  0.53389831,
        0.54237288,  0.55084746,  0.55932203,  0.56779661,  0.57627119,
        0.58474576,  0.59322034,  0.60169492,  0.61016949,  0.61864407,
        0.62711864,  0.63559322,  0.6440678 ,  0.65254237,  0.66101695,
        0.66949153,  0.6779661 ,  0.68644068,  0.69491525,  0.70338983,
        0.71186441,  0.72033898,  0.72881356,  0.73728814,  0.74576271,
        0.75423729,  0.76271186,  0.77118644,  0.77966102,  0.78813559,
        0.79661017,  0.80508475,  0.81355932,  0.8220339 ,  0.83050847,
        0.83898305,  0.84745763,  0.8559322 ,  0.86440678,  0.87288136,
        0.88135593,  0.88983051,  0.89830508,  0.90677966,  0.91525424,
        0.92372881,  0.93220339,  0.94067797,  0.94915254,  0.95762712,
        0.96610169,  0.97457627,  0.98305085,  0.99152542,  1., 1.        ])

        assert_allclose(mu_piecewise_linear(x,y),
                        3.482736134, atol=1e-8)


    def test_s_increasing(self):
        assert_raises(ValueError, mu_piecewise_linear,[1.5,1,2], [1,1,1] )

    def test_s_less_than_one(self):
        assert_raises(ValueError, mu_piecewise_linear,[0.5,1,2], [1,1,1] )

    def test_kap_less_than_zero(self):
        assert_raises(ValueError, mu_piecewise_linear,[1.5,1.6,2], [-2,1,1] )

    def test_s_n_unequal_len(self):
        assert_raises(ValueError, mu_piecewise_linear, [2,4], [1])

    def test_first_s_not_one(self):
        assert_raises(ValueError, mu_piecewise_linear,[1.1,1.6,2], [1,1,1])

class test_k_parabolic(unittest.TestCase):
    """tests for k_parabolic"""

    def test_ideal(self):
        assert_allclose(k_parabolic(20,1,2,[4,6,7]),
                        [1,1,1])
    def test_ideal2(self):
        assert_allclose(k_parabolic(20,2,1,[4,6,7]),
                        [1,1,1])
    def test_within_smear_zone(self):
        assert_allclose(k_parabolic(30, 5, 2, [1, 1.13559322]),
                        [0.5, 0.53332376])
    def test_outside_smear_zone(self):
        assert_allclose(k_parabolic(30, 5, 2, [5, 8]),
                        [1, 1])
    def test_n_less_than_one(self):
        assert_raises(ValueError, k_parabolic, 0.5, 2, 5, 1)

    def test_s_less_than_one(self):
        assert_raises(ValueError, k_parabolic, 50, 0.5, 5, 1)

    def test_kap_less_than_zero(self):
        assert_raises(ValueError, k_parabolic, 50, 0.5, -5,1)

    def test_s_greater_than_n(self):
        assert_raises(ValueError, k_parabolic, 50, 100, 5, 1)

    def test_si_greater_than_n(self):
        assert_raises(ValueError, k_parabolic, 50, 5, 5, 100)

    def test_si_less_than_1(self):
        assert_raises(ValueError, k_parabolic, 50, 5, 5, 0.5)

class test_k_linear(unittest.TestCase):
    """tests for k_linear"""

    def test_ideal(self):
        assert_allclose(k_linear(20,1,2,[4,6,7]),
                        [1,1,1])
    def test_ideal2(self):
        assert_allclose(k_linear(20,2,1,[4,6,7]),
                        [1,1,1])
    def test_within_smear_zone(self):
        assert_allclose(k_linear(30, 5, 2, [1, 1.13559322]),
                        [0.5, 0.51694915])
    def test_outside_smear_zone(self):
        assert_allclose(k_linear(30, 5, 2, [5, 8]),
                        [1, 1])
    def test_n_less_than_one(self):
        assert_raises(ValueError, k_linear, 0.5, 2, 5, 1)

    def test_s_less_than_one(self):
        assert_raises(ValueError, k_linear, 50, 0.5, 5, 1)

    def test_kap_less_than_zero(self):
        assert_raises(ValueError, k_linear, 50, 0.5, -5,1)

    def test_s_greater_than_n(self):
        assert_raises(ValueError, k_linear, 50, 100, 5, 1)

    def test_si_greater_than_n(self):
        assert_raises(ValueError, k_linear, 50, 5, 5, 100)

    def test_si_less_than_1(self):
        assert_raises(ValueError, k_linear, 50, 5, 5, 0.5)

class test_k_overlapping_linear(unittest.TestCase):
    """tests for k_overlapping_linear"""

    def test_ideal(self):
        assert_allclose(k_overlapping_linear(20,1,2,[4,6,7]),
                        [1,1,1])
    def test_ideal2(self):
        assert_allclose(k_overlapping_linear(20,2,1,[4,6,7]),
                        [1,1,1])
    def test_within_smear_zone(self):
        assert_allclose(k_overlapping_linear(30, 5, 2, [1, 1.13559322]),
                        [0.5, 0.51694915])
    def test_outside_smear_zone(self):
        assert_allclose(k_overlapping_linear(30, 5, 2, [5, 8]),
                        [1, 1])

    def test_within_smear_zone_overlap(self):
        assert_allclose(k_overlapping_linear(30, 50, 8, [1, 1.13559322]),
                        [0.125, 0.12742131])
    def test_outside_smear_zone(self):
        assert_allclose(k_overlapping_linear(30, 50, 8, [25, 28]),
                        [0.28571429, 0.28571429])

    def test_n_less_than_one(self):
        assert_raises(ValueError, k_overlapping_linear, 0.5, 2, 5, 1)

    def test_s_less_than_one(self):
        assert_raises(ValueError, k_overlapping_linear, 50, 0.5, 5, 1)

    def test_kap_less_than_zero(self):
        assert_raises(ValueError, k_overlapping_linear, 50, 0.5, -5,1)

#    def test_s_greater_than_n(self):
#        assert_raises(ValueError, k_overlapping_linear, 50, 100, 5, 1)

    def test_si_greater_than_n(self):
        assert_raises(ValueError, k_overlapping_linear, 50, 5, 5, 100)

    def test_si_less_than_1(self):
        assert_raises(ValueError, k_overlapping_linear, 50, 5, 5, 0.5)


class test_u_ideal(unittest.TestCase):
    """tests for u_ideal"""

    def test_ideal(self):
        assert_allclose(u_ideal(10,[1,5,10]),
                        [ 0.        ,  0.94367157,  1.14524187])

    def test_n_less_than_one(self):
        assert_raises(ValueError, u_ideal, 0.5, 2)


    def test_si_greater_than_n(self):
        assert_raises(ValueError, u_ideal, 50, 100)

    def test_si_less_than_1(self):
        assert_raises(ValueError, u_ideal, 50, 0.5)

class test_u_constant(unittest.TestCase):
    """tests for u_constant"""

    def test_ideal1(self):
        assert_allclose(u_constant(10,1,5,[1,5,10]),
                        [ 0.        ,  0.94367157,  1.14524187])
    def test_ideal2(self):
        assert_allclose(u_constant(10,5,1,[1,5,10]),
                        [ 0.        ,  0.94367157,  1.14524187])

    def test_within_smear_zone(self):
        assert_allclose(u_constant(10, 5, 2, [1, 2]),
                        [0.       ,  0.4555341])

    def test_outside_smear_zone(self):
        assert_allclose(u_constant(10, 5, 2, [6, 10]),
                        [1.04326825,  1.10736022])

    def test_in_and_out_smear_zone(self):
        assert_allclose(u_constant(10, 5, 2, [2, 10]),
                        [0.4555341, 1.10736022])

    def test_in_and_out_smear_zone_uavg_uw_muw(self):
        assert_allclose(u_constant(30, 5, 2, [2, 10], uavg=10, uw=-3, muw=3),
                        [4.869976  ,  9.28842172])

    def test_n_less_than_one(self):
        assert_raises(ValueError, u_constant, 0.5, 3, 2,2)

    def test_si_greater_than_n(self):
        assert_raises(ValueError, u_constant, 50, 5, 2, 100)

    def test_si_less_than_1(self):
        assert_raises(ValueError, u_constant, 50, 5, 2, 0.5)

    def test_s_less_than_one(self):
        assert_raises(ValueError, u_constant, 50, 0.5, 2, 2)

    def test_kap_less_than_zero(self):
        assert_raises(ValueError, u_constant, 50, 5, -0.5, 2)

    def test_s_greater_than_n(self):
        assert_raises(ValueError, u_constant, 50, 100, 2, 2)

class test_u_linear(unittest.TestCase):
    """tests for u_linear"""

    def test_ideal1(self):
        assert_allclose(u_linear(10,1,5,[1,5,10]),
                        [ 0.        ,  0.94367157,  1.14524187])
    def test_ideal2(self):
        assert_allclose(u_linear(10,5,1,[1,5,10]),
                        [ 0.        ,  0.94367157,  1.14524187])

    def test_within_smear_zone(self):
        assert_allclose(u_linear(30, 5, 2, [1, 2]),
                        [ 0.        ,  0.35902939])

    def test_outside_smear_zone(self):
        assert_allclose(u_linear(30, 5, 2, [6, 10]),
                        [0.74728049,  0.88374505])

    def test_in_and_out_smear_zone(self):
        assert_allclose(u_linear(30, 5, 2, [2, 10]),
                        [0.35902939,  0.88374505])

    def test_in_and_out_smear_zone_s_eq_kap(self):
        assert_allclose(u_linear(30, 5, 5, [2, 10]),
                        [0.49583999,  0.92019343])
    def test_in_and_out_smear_zone_uavg_uw_muw(self):
        assert_allclose(u_linear(30, 5, 2, [2, 10], uavg=10, uw=-3, muw=3),
                        [5.52344733,  9.18807294])

    def test_n_less_than_one(self):
        assert_raises(ValueError, u_linear, 0.5, 3, 2,2)

    def test_si_greater_than_n(self):
        assert_raises(ValueError, u_linear, 50, 5, 2, 100)

    def test_si_less_than_1(self):
        assert_raises(ValueError, u_linear, 50, 5, 2, 0.5)

    def test_s_less_than_one(self):
        assert_raises(ValueError, u_linear, 50, 0.5, 2, 2)

    def test_kap_less_than_zero(self):
        assert_raises(ValueError, u_linear, 50, 5, -0.5, 2)

    def test_s_greater_than_n(self):
        assert_raises(ValueError, u_linear, 50, 100, 2, 2)

class test_u_parabolic(unittest.TestCase):
    """tests for u_parabolic"""

    def test_ideal1(self):
        assert_allclose(u_parabolic(10,1,5,[1,5,10]),
                        [ 0.        ,  0.94367157,  1.14524187])
    def test_ideal2(self):
        assert_allclose(u_parabolic(10,5,1,[1,5,10]),
                        [ 0.        ,  0.94367157,  1.14524187])

    def test_within_smear_zone(self):
        assert_allclose(u_parabolic(30, 5, 2, [1, 2]),
                        [ 0.        ,  0.35695287])

    def test_outside_smear_zone(self):
        assert_allclose(u_parabolic(30, 5, 2, [6, 10]),
                        [0.72909691,  0.87514425])

    def test_in_and_out_smear_zone(self):
        assert_allclose(u_parabolic(30, 5, 2, [2, 10]),
                        [0.35695287,  0.87514425])

    def test_in_and_out_smear_zone_uavg_uw_muw(self):
        assert_allclose(u_parabolic(30, 5, 2, [2, 10], uavg=10, uw=-3, muw=3),
                        [5.65029444,  9.15544955])

    def test_n_less_than_one(self):
        assert_raises(ValueError, u_parabolic, 0.5, 3, 2,2)

    def test_si_greater_than_n(self):
        assert_raises(ValueError, u_parabolic, 50, 5, 2, 100)

    def test_si_less_than_1(self):
        assert_raises(ValueError, u_parabolic, 50, 5, 2, 0.5)

    def test_s_less_than_one(self):
        assert_raises(ValueError, u_parabolic, 50, 0.5, 2, 2)

    def test_kap_less_than_zero(self):
        assert_raises(ValueError, u_parabolic, 50, 5, -0.5, 2)

    def test_s_greater_than_n(self):
        assert_raises(ValueError, u_parabolic, 50, 100, 2, 2)

class test_u_piecewise_constant(unittest.TestCase):
    """tests for u_piecewise_constant"""

    def test_ideal1(self):
        assert_allclose(u_piecewise_constant(10, 1,[1,5,10]),
                        [ 0.        ,  0.94367157,  1.14524187])

    def test_ideal_multi(self):
        assert_allclose(u_piecewise_constant([5, 10],
                                              [1, 1],[1,5,10]),
                        [ 0.        ,  0.94367157,  1.14524187])


    def test_const(self):
        assert_allclose(u_piecewise_constant([5,10], [2,1], [2, 10]),
                        [0.4555341, 1.10736022])

    def test_const_multi(self):
        assert_allclose(u_piecewise_constant([2,5,8,10], [2,2,1,1], [2, 10]),
                        [0.4555341, 1.10736022])

    def test_const_two_smear_zones(self):
        assert_allclose(u_piecewise_constant([1.5, 3, 5,],
                                              [2, 3, 1.0], 1.6),
                         0.41536586)

    def test_const_two_smear_zones_n_kap_m(self):
        assert_allclose(u_piecewise_constant([1.5, 3,],
                                              [2, 3], 1.6, n=5, kap_m=1),
                         0.41536586)

    def test_const_two_smear_zones_n(self):
        assert_allclose(u_piecewise_constant([1.5, 3,4],
                                              [2, 3, 1], 1.6, n=5),

                         0.41536586)
    def test_const_uavg_uw_muw(self):
        assert_allclose(u_piecewise_constant([5,30], [2,1], [2, 10], uavg=10, uw=-3, muw=3),
                        [4.869976  ,  9.28842172])

    def test_parabolic(self):
        """piecewise constant approximation of parabolic with n = 30, s=5, kap=2"""


        x = np.array(
        [   1.06779661,  1.13559322,  1.20338983,  1.27118644,
        1.33898305,  1.40677966,  1.47457627,  1.54237288,  1.61016949,
        1.6779661 ,  1.74576271,  1.81355932,  1.88135593,  1.94915254,
        2.01694915,  2.08474576,  2.15254237,  2.22033898,  2.28813559,
        2.3559322 ,  2.42372881,  2.49152542,  2.55932203,  2.62711864,
        2.69491525,  2.76271186,  2.83050847,  2.89830508,  2.96610169,
        3.03389831,  3.10169492,  3.16949153,  3.23728814,  3.30508475,
        3.37288136,  3.44067797,  3.50847458,  3.57627119,  3.6440678 ,
        3.71186441,  3.77966102,  3.84745763,  3.91525424,  3.98305085,
        4.05084746,  4.11864407,  4.18644068,  4.25423729,  4.3220339 ,
        4.38983051,  4.45762712,  4.52542373,  4.59322034,  4.66101695,
        4.72881356,  4.79661017,  4.86440678,  4.93220339,  5., 30       ])

        y = 1.0/np.array(
        [ 0.5       ,  0.51680552,  0.53332376,  0.54955473,  0.56549842,
        0.58115484,  0.59652399,  0.61160586,  0.62640046,  0.64090779,
        0.65512784,  0.66906061,  0.68270612,  0.69606435,  0.70913531,
        0.72191899,  0.7344154 ,  0.74662453,  0.75854639,  0.77018098,
        0.7815283 ,  0.79258834,  0.8033611 ,  0.8138466 ,  0.82404481,
        0.83395576,  0.84357943,  0.85291583,  0.86196495,  0.8707268 ,
        0.87920138,  0.88738868,  0.89528871,  0.90290147,  0.91022695,
        0.91726515,  0.92401609,  0.93047975,  0.93665613,  0.94254525,
        0.94814708,  0.95346165,  0.95848894,  0.96322896,  0.9676817 ,
        0.97184717,  0.97572537,  0.97931629,  0.98261994,  0.98563631,
        0.98836541,  0.99080724,  0.99296179,  0.99482907,  0.99640908,
        0.99770181,  0.99870727,  0.99942545,  0.99985636,  1.        ])

        assert_allclose(u_piecewise_constant(x, y,[2,10]),
                        [0.35695287,  0.87514425], atol=0.003)

    def test_linear(self):
        """piecewise constant approximation of linear with n = 30, s=5, kap=2"""


        x = np.array(
        [   1.06779661,  1.13559322,  1.20338983,  1.27118644,
        1.33898305,  1.40677966,  1.47457627,  1.54237288,  1.61016949,
        1.6779661 ,  1.74576271,  1.81355932,  1.88135593,  1.94915254,
        2.01694915,  2.08474576,  2.15254237,  2.22033898,  2.28813559,
        2.3559322 ,  2.42372881,  2.49152542,  2.55932203,  2.62711864,
        2.69491525,  2.76271186,  2.83050847,  2.89830508,  2.96610169,
        3.03389831,  3.10169492,  3.16949153,  3.23728814,  3.30508475,
        3.37288136,  3.44067797,  3.50847458,  3.57627119,  3.6440678 ,
        3.71186441,  3.77966102,  3.84745763,  3.91525424,  3.98305085,
        4.05084746,  4.11864407,  4.18644068,  4.25423729,  4.3220339 ,
        4.38983051,  4.45762712,  4.52542373,  4.59322034,  4.66101695,
        4.72881356,  4.79661017,  4.86440678,  4.93220339,  5., 30       ])

        y = 1.0/np.array(
        [ 0.5       ,  0.50847458,  0.51694915,  0.52542373,  0.53389831,
        0.54237288,  0.55084746,  0.55932203,  0.56779661,  0.57627119,
        0.58474576,  0.59322034,  0.60169492,  0.61016949,  0.61864407,
        0.62711864,  0.63559322,  0.6440678 ,  0.65254237,  0.66101695,
        0.66949153,  0.6779661 ,  0.68644068,  0.69491525,  0.70338983,
        0.71186441,  0.72033898,  0.72881356,  0.73728814,  0.74576271,
        0.75423729,  0.76271186,  0.77118644,  0.77966102,  0.78813559,
        0.79661017,  0.80508475,  0.81355932,  0.8220339 ,  0.83050847,
        0.83898305,  0.84745763,  0.8559322 ,  0.86440678,  0.87288136,
        0.88135593,  0.88983051,  0.89830508,  0.90677966,  0.91525424,
        0.92372881,  0.93220339,  0.94067797,  0.94915254,  0.95762712,
        0.96610169,  0.97457627,  0.98305085,  0.99152542,  1.        ])

        assert_allclose(u_piecewise_constant(x,y,[2,10]),
                        [0.35902939,  0.88374505], atol=0.003)


    def test_n_less_than_one(self):
        assert_raises(ValueError, u_piecewise_constant, [2, 0.5],[2,1], 1.5)

    def test_si_greater_than_n(self):
        assert_raises(ValueError, u_piecewise_constant, [2, 4],[2,1], 5)

    def test_si_less_than_1(self):
        assert_raises(ValueError, u_piecewise_constant, [2, 4],[2,1], 0.5)

    def test_s_less_than_one(self):
        assert_raises(ValueError, u_piecewise_constant, [0.5, 4],[2,1], 1.5)

    def test_kap_less_than_zero(self):
        assert_raises(ValueError, u_piecewise_constant, [2, 4],[2,-1], 3)

    def test_s_increasing(self):
        assert_raises(ValueError, u_piecewise_constant,[1.5,1,2], [1,1,1],1 )

    def test_s_less_than_one(self):
        assert_raises(ValueError, u_piecewise_constant,[0.5,1,2], [1,1,1],1 )

    def test_kap_less_than_zero(self):
        assert_raises(ValueError, u_piecewise_constant,[1.5,1.6,2], [-2,1,1],1)

    def test_s_n_unequal_len(self):
        assert_raises(ValueError, u_piecewise_constant, [2,4], [1],1)

class test_u_piecewise_linear(unittest.TestCase):
    """tests for u_piecewise_linear"""

    def test_ideal1(self):
        assert_allclose(u_piecewise_linear([1,10], [1,1], [1,5,10]),
                        [ 0.        ,  0.94367157,  1.14524187])

    def test_ideal_multi(self):
        assert_allclose(u_piecewise_linear([1,5, 10],
                                              [1,1, 1],[1,5,10]),
                        [ 0.        ,  0.94367157,  1.14524187])


    def test_const(self):
        assert_allclose(u_piecewise_linear([1,5,5,10], [2,2,1,1], [2, 10]),
                        [0.4555341, 1.10736022])

    def test_const_multi(self):
        assert_allclose(u_piecewise_linear([1,2,5,5,10], [2,2,2,1,1], [2, 10]),
                        [0.4555341, 1.10736022])

    def test_const_two_smear_zones(self):
        assert_allclose(u_piecewise_linear([1,1.5,1.5, 3,3, 5],
                                              [2,2,3, 3,1, 1.0], 1.6),
                         0.41536586)
    def test_const_two_smear_zones_n_kap_m(self):
        assert_allclose(u_piecewise_linear([1,1.5,1.5, 3,3,],
                                              [2,2,3, 3,1,], 1.6,n=5, kap_m=1),
                         0.41536586)
    def test_const_two_smear_zones_n(self):
        assert_allclose(u_piecewise_linear([1,1.5,1.5, 3,3, 4],
                                              [2,2,3, 3,1, 1.0], 1.6,n=5),
                         0.41536586)

    def test_const_uavg_uw_muw(self):
        assert_allclose(u_piecewise_linear([1,5,5,30], [2,2,1,1], [2, 10], uavg=10, uw=-3, muw=3),
                        [4.869976  ,  9.28842172])

    def test_single_linear_s_neq_kap(self):
        assert_allclose(u_piecewise_linear([1,5,30], [2,1,1], [2, 10]),
                        [0.35902939,  0.88374505])

    def test_single_linear_s_eq_kap(self):
        assert_allclose(u_piecewise_linear([1,5,30], [5,1,1], [2, 10]),
                        [0.49583999,  0.92019343])

    def test_parabolic(self):
        """piecewise constant approximation of parabolic with n = 30, s=5, kap=2"""


        x = np.array(
        [1.,    1.06779661,  1.13559322,  1.20338983,  1.27118644,
        1.33898305,  1.40677966,  1.47457627,  1.54237288,  1.61016949,
        1.6779661 ,  1.74576271,  1.81355932,  1.88135593,  1.94915254,
        2.01694915,  2.08474576,  2.15254237,  2.22033898,  2.28813559,
        2.3559322 ,  2.42372881,  2.49152542,  2.55932203,  2.62711864,
        2.69491525,  2.76271186,  2.83050847,  2.89830508,  2.96610169,
        3.03389831,  3.10169492,  3.16949153,  3.23728814,  3.30508475,
        3.37288136,  3.44067797,  3.50847458,  3.57627119,  3.6440678 ,
        3.71186441,  3.77966102,  3.84745763,  3.91525424,  3.98305085,
        4.05084746,  4.11864407,  4.18644068,  4.25423729,  4.3220339 ,
        4.38983051,  4.45762712,  4.52542373,  4.59322034,  4.66101695,
        4.72881356,  4.79661017,  4.86440678,  4.93220339,  5., 30       ])

        y = 1.0/np.array(
        [ 0.5       ,  0.51680552,  0.53332376,  0.54955473,  0.56549842,
        0.58115484,  0.59652399,  0.61160586,  0.62640046,  0.64090779,
        0.65512784,  0.66906061,  0.68270612,  0.69606435,  0.70913531,
        0.72191899,  0.7344154 ,  0.74662453,  0.75854639,  0.77018098,
        0.7815283 ,  0.79258834,  0.8033611 ,  0.8138466 ,  0.82404481,
        0.83395576,  0.84357943,  0.85291583,  0.86196495,  0.8707268 ,
        0.87920138,  0.88738868,  0.89528871,  0.90290147,  0.91022695,
        0.91726515,  0.92401609,  0.93047975,  0.93665613,  0.94254525,
        0.94814708,  0.95346165,  0.95848894,  0.96322896,  0.9676817 ,
        0.97184717,  0.97572537,  0.97931629,  0.98261994,  0.98563631,
        0.98836541,  0.99080724,  0.99296179,  0.99482907,  0.99640908,
        0.99770181,  0.99870727,  0.99942545,  0.99985636,  1., 1.        ])

        assert_allclose(u_piecewise_linear(x,y,[2,10]),
                        [0.35695287,  0.87514425], atol=1e-5)


    def test_linear(self):
        """piecewise linear approximation of linear with n = 30, s=5, kap=2"""


        x = np.array(
        [1.,   1.06779661,  1.13559322,  1.20338983,  1.27118644,
        1.33898305,  1.40677966,  1.47457627,  1.54237288,  1.61016949,
        1.6779661 ,  1.74576271,  1.81355932,  1.88135593,  1.94915254,
        2.01694915,  2.08474576,  2.15254237,  2.22033898,  2.28813559,
        2.3559322 ,  2.42372881,  2.49152542,  2.55932203,  2.62711864,
        2.69491525,  2.76271186,  2.83050847,  2.89830508,  2.96610169,
        3.03389831,  3.10169492,  3.16949153,  3.23728814,  3.30508475,
        3.37288136,  3.44067797,  3.50847458,  3.57627119,  3.6440678 ,
        3.71186441,  3.77966102,  3.84745763,  3.91525424,  3.98305085,
        4.05084746,  4.11864407,  4.18644068,  4.25423729,  4.3220339 ,
        4.38983051,  4.45762712,  4.52542373,  4.59322034,  4.66101695,
        4.72881356,  4.79661017,  4.86440678,  4.93220339,  5., 30       ])

        y = 1.0/np.array(
        [ 0.5       ,  0.50847458,  0.51694915,  0.52542373,  0.53389831,
        0.54237288,  0.55084746,  0.55932203,  0.56779661,  0.57627119,
        0.58474576,  0.59322034,  0.60169492,  0.61016949,  0.61864407,
        0.62711864,  0.63559322,  0.6440678 ,  0.65254237,  0.66101695,
        0.66949153,  0.6779661 ,  0.68644068,  0.69491525,  0.70338983,
        0.71186441,  0.72033898,  0.72881356,  0.73728814,  0.74576271,
        0.75423729,  0.76271186,  0.77118644,  0.77966102,  0.78813559,
        0.79661017,  0.80508475,  0.81355932,  0.8220339 ,  0.83050847,
        0.83898305,  0.84745763,  0.8559322 ,  0.86440678,  0.87288136,
        0.88135593,  0.88983051,  0.89830508,  0.90677966,  0.91525424,
        0.92372881,  0.93220339,  0.94067797,  0.94915254,  0.95762712,
        0.96610169,  0.97457627,  0.98305085,  0.99152542,  1., 1.        ])

        assert_allclose(u_piecewise_linear(x,y,[2,10]),
                        [0.35902939,  0.88374505], atol=1e-8)


    def test_n_less_than_one(self):
        assert_raises(ValueError, u_piecewise_linear, [1, 2, 0.5],[2,1], 1.5)

    def test_si_greater_than_n(self):
        assert_raises(ValueError, u_piecewise_linear, [1, 2, 4],[2,1], 5)

    def test_si_less_than_1(self):
        assert_raises(ValueError, u_piecewise_linear, [1, 2, 4],[2,1], 0.5)

    def test_s_less_than_one(self):
        assert_raises(ValueError, u_piecewise_linear, [0.5, 4],[2,1], 1.5)

    def test_kap_less_than_zero(self):
        assert_raises(ValueError, u_piecewise_linear, [1,2, 4],[2,-1], 3)

    def test_s_increasing(self):
        assert_raises(ValueError, u_piecewise_linear,[1,1.5,1,2], [1,1,1],1 )

    def test_s_less_than_one(self):
        assert_raises(ValueError, u_piecewise_linear,[1,0.5,1,2], [1,1,1],1 )

    def test_kap_less_than_zero(self):
        assert_raises(ValueError, u_piecewise_linear,[1,1.5,1.6,2], [-2,1,1],1)

    def test_s_n_unequal_len(self):
        assert_raises(ValueError, u_piecewise_linear, [1,2,4], [1],1)

class test_re_from_drain_spacing(unittest.TestCase):
    """tests for re_from_drain_spacing"""

    def test_sp_less_than_zero(self):
        assert_raises(ValueError, re_from_drain_spacing, -1)

    def wrong_pattern(self):
        assert_raises(ValueError, re_from_drain_spacing, 2.0, 'calamari')

    def test_triangle(self):
        assert_allclose(re_from_drain_spacing(3),
                        1.575112704)
    def test_triangle2(self):
        assert_allclose(re_from_drain_spacing(3, 'tri'),
                        1.575112704)
    def test_square(self):
        assert_allclose(re_from_drain_spacing(3, 'Square'),
                        1.692568751)
    def test_square2(self):
        assert_allclose(re_from_drain_spacing(3, 'sqr'),
                        1.692568751)

class test_back_calc_drain_spacing_from_eta(unittest.TestCase):
    """tests for back_calc_drain_spacing_from_eta"""


    def test_ideal(self):
        assert_allclose(back_calc_drain_spacing_from_eta(1.0680524125462512, 't', mu_ideal,
                            0.05, 5, 2, muw=1),
                            [  1.5       ,   0.78755635,  15.75112704],
                            atol=1e-4)
    def test_constant(self):
        assert_allclose(back_calc_drain_spacing_from_eta(0.71017973670799939, 't', mu_constant,
                            0.05, 5, 2, muw=1),
                            [  1.5       ,   0.78755635,  15.75112704],
                            atol=1e-4)
    def test_piecewise_constant(self):
        assert_allclose(back_calc_drain_spacing_from_eta(0.71017973670799939, 't',
                            mu_piecewise_constant,
                            0.05, [5, 6], [2, 1], muw=1),
                            [  1.5       ,   0.78755635,  15.75112704],
                            atol=1e-4)

    def test_piecewise_linear(self):
        assert_allclose(back_calc_drain_spacing_from_eta(0.71017973670799939, 't',
                            mu_piecewise_linear,
                            0.05, [1,5,5], [2,2, 1], muw=1),
                            [  1.5       ,   0.78755635,  15.75112704],
                            atol=1e-4)

    def test_piecewise_constant_square(self):
        assert_allclose(back_calc_drain_spacing_from_eta(0.60411247160628478, 's',
                            mu_piecewise_constant,
                            0.05, [5, 6], [2, 1], muw=1),
                            [  1.5       ,   0.84628438,  16.92568751],
                            atol=1e-4)


    def test_overlapping_linear(self):
        assert_allclose(back_calc_drain_spacing_from_eta(0.71017973670799939, 't',
                            mu_overlapping_linear,
                            0.05, 5, 2, muw=1),
                            [  1.61711464,   0.84904594,  16.9809188 ],
                            atol=1e-4)

    def test_n_falls_below_s(self):
        assert_raises(ValueError, back_calc_drain_spacing_from_eta, 5, 't',
                            mu_constant,
                            0.05, 5, 2, muw=1)

    def test_multipe_s(self):
        assert_raises(ValueError, back_calc_drain_spacing_from_eta, 1, 't',
                            mu_constant,
                            0.05, [5,6], 2, muw=1)
    def test_multipe_kap(self):
        assert_raises(ValueError, back_calc_drain_spacing_from_eta, 1, 't',
                            mu_constant,
                            0.05, [5,6], 2, muw=1)


if __name__ == '__main__':

    import nose
#    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
    nose.runmodule(argv=['nose', 'test_smear_zones:test_mu_parabolic.test_s_n_unequal_len', '--verbosity=3', '--with-doctest', '--doctest-options=+ELLIPSIS'])
#    nose.runmodule(argv=['nosetests', 'test_foxcon:test_foxcon.test__update_Hlayers', '--verbosity=3', '--with-doctest', '--doctest-options=+ELLIPSIS'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])

#test_mu_parabolic