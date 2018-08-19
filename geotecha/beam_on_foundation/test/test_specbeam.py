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

"""
Finite elastic Euler-Bernoulli beam resting on viscoelastic foundation
with piecewise-linear properties, subjected to multiple moving point loads.

"""

from __future__ import division, print_function


import numpy as np
from numpy.testing import assert_allclose

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.style
import matplotlib as mpl

import time
from datetime  import timedelta
import datetime
from collections import OrderedDict
import os

import geotecha.piecewise.piecewise_linear_1d as pwise
from geotecha.piecewise.piecewise_linear_1d import PolyLine

from geotecha.beam_on_foundation.specbeam import SpecBeam
#from geotecha.beam_on_foundation.specbeam import MovingPointLoads
#from geotecha.beam_on_foundation.dingetal2012 import DingEtAl2012


DEBUG=False # if True a plot might be produced or a table may be printed




def test_SpecBeam_const_mat_midpoint_defl():
    """Test SpecBeam for constant mat: Figure 8, displacement vs time at beam midpoint"""

    # expected values are digitised from Ding et al 2012.
    expected_50terms_t = np.array(
      [ 3.511,  3.553,  3.576,  3.605,  3.621,  3.637,  3.651,  3.661,
        3.676,  3.693,  3.701,  3.717,  3.733,  3.743,  3.761,  3.777,
        3.79 ,  3.804,  3.822,  3.833,  3.848,  3.862,  3.877,  3.899,
        3.919,  3.94 ,  3.956,  3.97 ,  3.978,  3.99 ,  4.001,  4.014,
        4.025,  4.041,  4.052,  4.064,  4.076,  4.086,  4.094,  4.104,
        4.112,  4.123,  4.134,  4.146,  4.159,  4.172,  4.18 ,  4.192,
        4.212,  4.221,  4.234,  4.255,  4.273,  4.292,  4.324,  4.357,
        4.373,  4.399,  4.418,  4.445])

    expected_50terms_displacement = np.array(
      [ -1.30900000e-04,   8.85900000e-05,   2.63900000e-04,
         3.30200000e-04,   3.30500000e-04,   2.87100000e-04,
         2.00000000e-04,   9.09500000e-05,  -8.36000000e-05,
        -3.01800000e-04,  -4.32800000e-04,  -6.07300000e-04,
        -7.59900000e-04,  -8.90900000e-04,  -9.99800000e-04,
        -1.02100000e-03,  -9.99100000e-04,  -8.67700000e-04,
        -5.61300000e-04,  -2.11300000e-04,   2.04300000e-04,
         7.29200000e-04,   1.42900000e-03,   2.41300000e-03,
         3.46300000e-03,   4.29400000e-03,   5.08100000e-03,
         5.62800000e-03,   5.86800000e-03,   6.19600000e-03,
         6.39300000e-03,   6.52500000e-03,   6.59100000e-03,
         6.48200000e-03,   6.32900000e-03,   6.02300000e-03,
         5.69600000e-03,   5.45500000e-03,   5.06200000e-03,
         4.75600000e-03,   4.40700000e-03,   3.90400000e-03,
         3.35800000e-03,   2.87800000e-03,   2.35300000e-03,
         1.82900000e-03,   1.52300000e-03,   9.98700000e-04,
         5.18300000e-04,   2.12500000e-04,  -4.95200000e-05,
        -2.67600000e-04,  -3.76500000e-04,  -3.76100000e-04,
        -2.00600000e-04,   1.87300000e-05,   1.06500000e-04,
         1.94500000e-04,   1.94900000e-04,   1.30000000e-04])

    expected_200terms_t = np.array(
      [ 3.503,  3.519,  3.539,  3.556,  3.576,  3.595,  3.613,  3.632,
        3.651,  3.667,  3.69 ,  3.708,  3.727,  3.746,  3.766,  3.782,
        3.801,  3.822,  3.84 ,  3.856,  3.878,  3.896,  3.915,  3.933,
        3.951,  3.97 ,  3.99 ,  4.009,  4.027,  4.046,  4.067,  4.085,
        4.102,  4.122,  4.141,  4.159,  4.178,  4.197,  4.217,  4.234,
        4.252,  4.273,  4.291,  4.31 ,  4.329,  4.347,  4.368,  4.386,
        4.405,  4.423,  4.442,  4.46 ,  4.479,  4.495])

    expected_200terms_displacement = np.array(
      [  7.04000000e-08,   2.22800000e-05,   2.27000000e-05,
         1.23200000e-06,   2.35100000e-05,   2.39300000e-05,
         2.46400000e-06,  -1.89700000e-05,  -6.22700000e-05,
        -1.27500000e-04,  -1.70700000e-04,  -2.57800000e-04,
        -3.44800000e-04,  -4.53600000e-04,  -5.40600000e-04,
        -5.84000000e-04,  -5.61700000e-04,  -4.73800000e-04,
        -2.54900000e-04,   1.17100000e-04,   6.85900000e-04,
         1.51700000e-03,   2.50100000e-03,   3.70300000e-03,
         5.01500000e-03,   6.26200000e-03,   7.20200000e-03,
         7.61800000e-03,   7.39900000e-03,   6.74400000e-03,
         5.91400000e-03,   5.01800000e-03,   4.10100000e-03,
         3.22700000e-03,   2.50600000e-03,   1.85000000e-03,
         1.32600000e-03,   9.11400000e-04,   5.84000000e-04,
         3.00200000e-04,   1.47600000e-04,   3.87500000e-05,
        -4.82900000e-05,  -1.13400000e-04,  -1.13000000e-04,
        -1.34500000e-04,  -1.34000000e-04,  -1.33600000e-04,
        -1.11400000e-04,  -6.72600000e-05,  -6.68400000e-05,
        -6.64500000e-05,  -4.41700000e-05,  -4.38200000e-05])

    t = np.linspace(0, 4.5, 400)

    #see Dingetal2012 Table 2 for material properties
    pdict = OrderedDict(
            E = 6.998*1e9, #Pa
            rho = 2373, #kg/m3
            L = 160, #m
            #v_norm=0.01165,
            kf=5.41e-4,
            #Fz_norm=1.013e-4,
            mu_norm=39.263,
            k1_norm=97.552,
            k3_norm=2.497e6,
            nterms=50,
            BC="SS",
            nquad=20,
            k1bar=PolyLine([0,0.5],[0.5,1],[1,1],[1,1]),
            moving_loads_x_norm=[[0]],
            moving_loads_Fz_norm=[[1.013e-4]],
            moving_loads_v_norm=[0.01165,],
            tvals=t,
            xvals_norm=0.5)



    a = SpecBeam(**pdict)
    a.calulate_qk(t=t)

    yall = a.wofx(x_norm=0.5, normalise_w=False)
    ycompare = np.interp(expected_50terms_t, t, yall)



    if DEBUG:


        title ='SpecBeam, const mat prop vs Ding et al (2012) \nFigure 8 Displacement at Midpoint of beam'
        print(title)
        print(' '.join(['{:>9s}']*4).format('t', 'expect', 'calc', 'diff'))
        for i, j, k in zip(expected_50terms_t, expected_50terms_displacement, ycompare):
            print(' '.join(['{:9.4f}']*4).format(i, j, k, j-k))
        print()




        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot("111")
        ax.set_xlabel("time, s")
        ax.set_ylabel("w, m")
        ax.set_xlim(3.5, 4.5)
        ax.set_title(title)

        ax.plot(expected_50terms_t, expected_50terms_displacement,
                label="expect n=50", color='green', marker='o', ls='-')
        ax.plot(expected_200terms_t, expected_200terms_displacement,
                label="expect n=200", color='black', marker=None, ls='-')


        ax.plot(t, yall, label="calc n=50")
        ax.plot(expected_50terms_t, ycompare, label="calc, n=50 (compare)",
                color='red', marker='s', ms=4, ls=':')

        leg = ax.legend(loc='upper left')
        leg.draggable()

        plt.show()

    assert_allclose(expected_50terms_displacement, ycompare, atol=2.4e-4)


def test_SpecBeam_const_mat_deflection_shape():
    """Test SpecBeam for constant mat: Figure 7, deflection with load at midpoint"""

    # expected values are digitised from Ding et al 2012.
    expected_50terms_x = np.array(
      [ 70.68 ,  71.275,  71.785,  72.189,  72.529,  72.848,  73.209,
        73.677,  74.06 ,  74.378,  74.74 ,  75.122,  75.505,  75.781,
        76.079,  76.312,  76.525,  76.716,  76.95 ,  77.12 ,  77.418,
        77.715,  77.907,  78.204,  78.459,  78.778,  79.224,  79.522,
        79.883,  80.181,  80.436,  80.606,  80.776,  81.01 ,  81.137,
        81.307,  81.477,  81.626,  81.902,  82.179,  82.497,  82.774,
        82.986,  83.263,  83.539,  84.028,  84.368,  84.623,  84.878,
        85.239,  85.728,  86.153,  86.684,  87.386,  87.853,  88.363,
        88.81 ,  89.341,  89.788])

    expected_50terms_displacement = np.array(
      [  3.23100000e-06,   1.48600000e-04,   1.80900000e-04,
         1.64800000e-04,   1.00200000e-04,   1.93900000e-05,
        -1.09900000e-04,  -2.55300000e-04,  -3.52200000e-04,
        -3.68300000e-04,  -3.19900000e-04,  -1.26000000e-04,
         1.64800000e-04,   5.04000000e-04,   9.24100000e-04,
         1.27900000e-03,   1.66700000e-03,   2.10300000e-03,
         2.58800000e-03,   3.00800000e-03,   3.59000000e-03,
         4.17100000e-03,   4.70400000e-03,   5.27000000e-03,
         5.70600000e-03,   6.17400000e-03,   6.49800000e-03,
         6.57800000e-03,   6.44900000e-03,   6.15800000e-03,
         5.91600000e-03,   5.64100000e-03,   5.30200000e-03,
         4.88200000e-03,   4.51100000e-03,   4.17100000e-03,
         3.81600000e-03,   3.38000000e-03,   2.79800000e-03,
         2.05500000e-03,   1.42500000e-03,   7.78700000e-04,
         3.26300000e-04,  -9.37000000e-05,  -5.29900000e-04,
        -9.01500000e-04,  -9.98400000e-04,  -1.01500000e-03,
        -9.66100000e-04,  -8.04500000e-04,  -5.29900000e-04,
        -2.22900000e-04,   8.40100000e-05,   3.42500000e-04,
         3.74800000e-04,   3.10200000e-04,   1.80900000e-04,
         1.93900000e-05,  -1.26000000e-04])

    expected_200terms_x = np.array(
      [ 70.064,  70.404,  70.723,  71.02 ,  71.36 ,  71.679,  71.955,
        72.317,  72.614,  72.976,  73.273,  73.571,  73.911,  74.23 ,
        74.527,  74.867,  75.165,  75.505,  75.866,  76.164,  76.483,
        76.801,  77.12 ,  77.418,  77.779,  78.098,  78.438,  78.693,
        79.033,  79.352,  79.671,  79.989,  80.308,  80.648,  80.967,
        81.286,  81.583,  81.924,  82.264,  82.582,  82.88 ,  83.199,
        83.539,  83.815,  84.134,  84.495,  84.793,  85.112,  85.43 ,
        85.749,  86.047,  86.429,  86.727,  87.046,  87.365,  87.683,
        87.981,  88.342,  88.64 ,  88.959,  89.277,  89.617,  89.958])

    expected_200terms_displacement = np.array(
      [ -2.90800000e-05,  -2.90800000e-05,  -6.13900000e-05,
        -7.75400000e-05,  -7.75400000e-05,  -1.09900000e-04,
        -1.26000000e-04,  -1.26000000e-04,  -1.26000000e-04,
        -1.42200000e-04,  -1.09900000e-04,  -1.09900000e-04,
        -9.37000000e-05,  -2.90800000e-05,   3.55400000e-05,
         1.64800000e-04,   2.94000000e-04,   4.55600000e-04,
         7.30200000e-04,   1.02100000e-03,   1.40900000e-03,
         1.86100000e-03,   2.37800000e-03,   2.99200000e-03,
         3.68700000e-03,   4.43000000e-03,   5.22100000e-03,
         5.96400000e-03,   6.67500000e-03,   7.27300000e-03,
         7.58000000e-03,   7.48300000e-03,   6.95000000e-03,
         6.06100000e-03,   4.99500000e-03,   3.88000000e-03,
         2.83000000e-03,   1.94200000e-03,   1.16600000e-03,
         5.20200000e-04,   5.17000000e-05,  -2.55300000e-04,
        -4.33000000e-04,  -5.46000000e-04,  -5.78400000e-04,
        -5.62200000e-04,  -4.81400000e-04,  -4.33000000e-04,
        -3.19900000e-04,  -2.39100000e-04,  -1.74500000e-04,
        -1.26000000e-04,  -4.52300000e-05,  -1.29200000e-05,
         1.93900000e-05,   1.93900000e-05,   3.55400000e-05,
         6.78500000e-05,   3.55400000e-05,   5.17000000e-05,
         3.55400000e-05,   1.93900000e-05,   1.93900000e-05])


    t = np.linspace(0, 160/2/20, 400)

    xvals = np.linspace(0, 160, 200)
    #see Dingetal2012 Table 2 for material properties
    pdict = OrderedDict(
            E = 6.998*1e9, #Pa
            rho = 2373, #kg/m3
            L = 160, #m
            #v_norm=0.01165,
            kf=5.41e-4,
            #Fz_norm=1.013e-4,
            mu_norm=39.263,
            k1_norm=97.552,
            k3_norm=2.497e6,
            nterms=50,
            BC="SS",
            nquad=20,
            moving_loads_x_norm=[[0]],
            moving_loads_Fz_norm=[[1.013e-4]],
            moving_loads_v_norm=[0.01165,],
            tvals=t,
            xvals=xvals)



    a = SpecBeam(**pdict)

    a.calulate_qk(t=t)

    yall = a.wofx(x=xvals, tslice=slice(-1, None, None), normalise_w=False)[:,0]
    ycompare = np.interp(expected_50terms_x, xvals, yall)



    if DEBUG:


        title ='SpecBeam, const mat prop vs Ding et al (2012) \nFigure 7 deflection when load at Midpoint of beam'
        print(title)
        print(' '.join(['{:>9s}']*4).format('t', 'expect', 'calc', 'diff'))
        for i, j, k in zip(expected_50terms_x, expected_50terms_displacement, ycompare):
            print(' '.join(['{:9.4f}']*4).format(i, j, k, j-k))
        print()




        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot("111")
        ax.set_xlabel("x, m")
        ax.set_ylabel("w, m")
        ax.set_xlim(70, 90)
        ax.set_title(title)

        ax.plot(expected_50terms_x, expected_50terms_displacement,
                label="expect n=50", color='green', marker='o', ls='-')
        ax.plot(expected_200terms_x, expected_200terms_displacement,
                label="expect n=200", color='black', marker=None, ls='-')


        ax.plot(xvals, yall, label="calc n=50")
        ax.plot(expected_50terms_x, ycompare, label="calc, n=50 (compare)",
                color='red', marker='s', ms=4, ls=':')

        leg = ax.legend(loc='upper left')
        leg.draggable()

        plt.show()

    assert_allclose(expected_50terms_displacement, ycompare, atol=2.4e-4)



def test_SpecBeam_const_mat_midpoint_defl_runme():
    """Test SpecBeam for constant mat: Figure 8, displacement vs time at
    beam midpoint (using the runme method)"""

    # expected values are digitised from Ding et al 2012.
    expected_50terms_t = np.array(
      [ 3.511,  3.553,  3.576,  3.605,  3.621,  3.637,  3.651,  3.661,
        3.676,  3.693,  3.701,  3.717,  3.733,  3.743,  3.761,  3.777,
        3.79 ,  3.804,  3.822,  3.833,  3.848,  3.862,  3.877,  3.899,
        3.919,  3.94 ,  3.956,  3.97 ,  3.978,  3.99 ,  4.001,  4.014,
        4.025,  4.041,  4.052,  4.064,  4.076,  4.086,  4.094,  4.104,
        4.112,  4.123,  4.134,  4.146,  4.159,  4.172,  4.18 ,  4.192,
        4.212,  4.221,  4.234,  4.255,  4.273,  4.292,  4.324,  4.357,
        4.373,  4.399,  4.418,  4.445])

    expected_50terms_displacement = np.array(
      [ -1.30900000e-04,   8.85900000e-05,   2.63900000e-04,
         3.30200000e-04,   3.30500000e-04,   2.87100000e-04,
         2.00000000e-04,   9.09500000e-05,  -8.36000000e-05,
        -3.01800000e-04,  -4.32800000e-04,  -6.07300000e-04,
        -7.59900000e-04,  -8.90900000e-04,  -9.99800000e-04,
        -1.02100000e-03,  -9.99100000e-04,  -8.67700000e-04,
        -5.61300000e-04,  -2.11300000e-04,   2.04300000e-04,
         7.29200000e-04,   1.42900000e-03,   2.41300000e-03,
         3.46300000e-03,   4.29400000e-03,   5.08100000e-03,
         5.62800000e-03,   5.86800000e-03,   6.19600000e-03,
         6.39300000e-03,   6.52500000e-03,   6.59100000e-03,
         6.48200000e-03,   6.32900000e-03,   6.02300000e-03,
         5.69600000e-03,   5.45500000e-03,   5.06200000e-03,
         4.75600000e-03,   4.40700000e-03,   3.90400000e-03,
         3.35800000e-03,   2.87800000e-03,   2.35300000e-03,
         1.82900000e-03,   1.52300000e-03,   9.98700000e-04,
         5.18300000e-04,   2.12500000e-04,  -4.95200000e-05,
        -2.67600000e-04,  -3.76500000e-04,  -3.76100000e-04,
        -2.00600000e-04,   1.87300000e-05,   1.06500000e-04,
         1.94500000e-04,   1.94900000e-04,   1.30000000e-04])

    expected_200terms_t = np.array(
      [ 3.503,  3.519,  3.539,  3.556,  3.576,  3.595,  3.613,  3.632,
        3.651,  3.667,  3.69 ,  3.708,  3.727,  3.746,  3.766,  3.782,
        3.801,  3.822,  3.84 ,  3.856,  3.878,  3.896,  3.915,  3.933,
        3.951,  3.97 ,  3.99 ,  4.009,  4.027,  4.046,  4.067,  4.085,
        4.102,  4.122,  4.141,  4.159,  4.178,  4.197,  4.217,  4.234,
        4.252,  4.273,  4.291,  4.31 ,  4.329,  4.347,  4.368,  4.386,
        4.405,  4.423,  4.442,  4.46 ,  4.479,  4.495])

    expected_200terms_displacement = np.array(
      [  7.04000000e-08,   2.22800000e-05,   2.27000000e-05,
         1.23200000e-06,   2.35100000e-05,   2.39300000e-05,
         2.46400000e-06,  -1.89700000e-05,  -6.22700000e-05,
        -1.27500000e-04,  -1.70700000e-04,  -2.57800000e-04,
        -3.44800000e-04,  -4.53600000e-04,  -5.40600000e-04,
        -5.84000000e-04,  -5.61700000e-04,  -4.73800000e-04,
        -2.54900000e-04,   1.17100000e-04,   6.85900000e-04,
         1.51700000e-03,   2.50100000e-03,   3.70300000e-03,
         5.01500000e-03,   6.26200000e-03,   7.20200000e-03,
         7.61800000e-03,   7.39900000e-03,   6.74400000e-03,
         5.91400000e-03,   5.01800000e-03,   4.10100000e-03,
         3.22700000e-03,   2.50600000e-03,   1.85000000e-03,
         1.32600000e-03,   9.11400000e-04,   5.84000000e-04,
         3.00200000e-04,   1.47600000e-04,   3.87500000e-05,
        -4.82900000e-05,  -1.13400000e-04,  -1.13000000e-04,
        -1.34500000e-04,  -1.34000000e-04,  -1.33600000e-04,
        -1.11400000e-04,  -6.72600000e-05,  -6.68400000e-05,
        -6.64500000e-05,  -4.41700000e-05,  -4.38200000e-05])

    t = np.linspace(0, 4.5, 400)

    #see Dingetal2012 Table 2 for material properties
    pdict = OrderedDict(
            E = 6.998*1e9, #Pa
            rho = 2373, #kg/m3
            L = 160, #m
            #v_norm=0.01165,
            kf=5.41e-4,
            #Fz_norm=1.013e-4,
            mu_norm=39.263,
            k1_norm=97.552,
            k3_norm=2.497e6,
            nterms=50,
            BC="SS",
            nquad=20,
            k1bar=PolyLine([0,0.5],[0.5,1],[1,1],[1,1]),
            moving_loads_x_norm=[[0]],
            moving_loads_Fz_norm=[[1.013e-4]],
            moving_loads_v_norm=[0.01165,],
            tvals=t,
            xvals_norm=0.5)



    a = SpecBeam(**pdict)
    a.runme()

#    a.calulate_qk(t=t)
#
#    yall = a.wofx(x_norm=0.5, normalise_w=False)
    ycompare = np.interp(expected_50terms_t, t, a.defl)



    if DEBUG:


        title ='SpecBeam, const mat prop vs Ding et al (2012) \nFigure 8 Displacement at Midpoint of beam'
        print(title)
        print(' '.join(['{:>9s}']*4).format('t', 'expect', 'calc', 'diff'))
        for i, j, k in zip(expected_50terms_t, expected_50terms_displacement, ycompare):
            print(' '.join(['{:9.4f}']*4).format(i, j, k, j-k))
        print()




        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot("111")
        ax.set_xlabel("time, s")
        ax.set_ylabel("w, m")
        ax.set_xlim(3.5, 4.5)
        ax.set_title(title)

        ax.plot(expected_50terms_t, expected_50terms_displacement,
                label="expect n=50", color='green', marker='o', ls='-')
        ax.plot(expected_200terms_t, expected_200terms_displacement,
                label="expect n=200", color='black', marker=None, ls='-')


        ax.plot(t, a.defl, label="calc n=50")
        ax.plot(expected_50terms_t, ycompare, label="calc, n=50 (compare)",
                color='red', marker='s', ms=4, ls=':')

        leg = ax.legend(loc='upper left')
        leg.draggable()

        plt.show()

    assert_allclose(expected_50terms_displacement, ycompare, atol=2.4e-4)

def test_SpecBeam_const_mat_deflection_shape_runme():
    """Test SpecBeam for constant mat: Figure 7, deflection with load at midpoint"""

    # expected values are digitised from Ding et al 2012.
    expected_50terms_x = np.array(
      [ 70.68 ,  71.275,  71.785,  72.189,  72.529,  72.848,  73.209,
        73.677,  74.06 ,  74.378,  74.74 ,  75.122,  75.505,  75.781,
        76.079,  76.312,  76.525,  76.716,  76.95 ,  77.12 ,  77.418,
        77.715,  77.907,  78.204,  78.459,  78.778,  79.224,  79.522,
        79.883,  80.181,  80.436,  80.606,  80.776,  81.01 ,  81.137,
        81.307,  81.477,  81.626,  81.902,  82.179,  82.497,  82.774,
        82.986,  83.263,  83.539,  84.028,  84.368,  84.623,  84.878,
        85.239,  85.728,  86.153,  86.684,  87.386,  87.853,  88.363,
        88.81 ,  89.341,  89.788])

    expected_50terms_displacement = np.array(
      [  3.23100000e-06,   1.48600000e-04,   1.80900000e-04,
         1.64800000e-04,   1.00200000e-04,   1.93900000e-05,
        -1.09900000e-04,  -2.55300000e-04,  -3.52200000e-04,
        -3.68300000e-04,  -3.19900000e-04,  -1.26000000e-04,
         1.64800000e-04,   5.04000000e-04,   9.24100000e-04,
         1.27900000e-03,   1.66700000e-03,   2.10300000e-03,
         2.58800000e-03,   3.00800000e-03,   3.59000000e-03,
         4.17100000e-03,   4.70400000e-03,   5.27000000e-03,
         5.70600000e-03,   6.17400000e-03,   6.49800000e-03,
         6.57800000e-03,   6.44900000e-03,   6.15800000e-03,
         5.91600000e-03,   5.64100000e-03,   5.30200000e-03,
         4.88200000e-03,   4.51100000e-03,   4.17100000e-03,
         3.81600000e-03,   3.38000000e-03,   2.79800000e-03,
         2.05500000e-03,   1.42500000e-03,   7.78700000e-04,
         3.26300000e-04,  -9.37000000e-05,  -5.29900000e-04,
        -9.01500000e-04,  -9.98400000e-04,  -1.01500000e-03,
        -9.66100000e-04,  -8.04500000e-04,  -5.29900000e-04,
        -2.22900000e-04,   8.40100000e-05,   3.42500000e-04,
         3.74800000e-04,   3.10200000e-04,   1.80900000e-04,
         1.93900000e-05,  -1.26000000e-04])

    expected_200terms_x = np.array(
      [ 70.064,  70.404,  70.723,  71.02 ,  71.36 ,  71.679,  71.955,
        72.317,  72.614,  72.976,  73.273,  73.571,  73.911,  74.23 ,
        74.527,  74.867,  75.165,  75.505,  75.866,  76.164,  76.483,
        76.801,  77.12 ,  77.418,  77.779,  78.098,  78.438,  78.693,
        79.033,  79.352,  79.671,  79.989,  80.308,  80.648,  80.967,
        81.286,  81.583,  81.924,  82.264,  82.582,  82.88 ,  83.199,
        83.539,  83.815,  84.134,  84.495,  84.793,  85.112,  85.43 ,
        85.749,  86.047,  86.429,  86.727,  87.046,  87.365,  87.683,
        87.981,  88.342,  88.64 ,  88.959,  89.277,  89.617,  89.958])

    expected_200terms_displacement = np.array(
      [ -2.90800000e-05,  -2.90800000e-05,  -6.13900000e-05,
        -7.75400000e-05,  -7.75400000e-05,  -1.09900000e-04,
        -1.26000000e-04,  -1.26000000e-04,  -1.26000000e-04,
        -1.42200000e-04,  -1.09900000e-04,  -1.09900000e-04,
        -9.37000000e-05,  -2.90800000e-05,   3.55400000e-05,
         1.64800000e-04,   2.94000000e-04,   4.55600000e-04,
         7.30200000e-04,   1.02100000e-03,   1.40900000e-03,
         1.86100000e-03,   2.37800000e-03,   2.99200000e-03,
         3.68700000e-03,   4.43000000e-03,   5.22100000e-03,
         5.96400000e-03,   6.67500000e-03,   7.27300000e-03,
         7.58000000e-03,   7.48300000e-03,   6.95000000e-03,
         6.06100000e-03,   4.99500000e-03,   3.88000000e-03,
         2.83000000e-03,   1.94200000e-03,   1.16600000e-03,
         5.20200000e-04,   5.17000000e-05,  -2.55300000e-04,
        -4.33000000e-04,  -5.46000000e-04,  -5.78400000e-04,
        -5.62200000e-04,  -4.81400000e-04,  -4.33000000e-04,
        -3.19900000e-04,  -2.39100000e-04,  -1.74500000e-04,
        -1.26000000e-04,  -4.52300000e-05,  -1.29200000e-05,
         1.93900000e-05,   1.93900000e-05,   3.55400000e-05,
         6.78500000e-05,   3.55400000e-05,   5.17000000e-05,
         3.55400000e-05,   1.93900000e-05,   1.93900000e-05])


    t = np.linspace(0, 160/2/20, 400)

    xvals = np.linspace(0, 160, 200)
    #see Dingetal2012 Table 2 for material properties
    pdict = OrderedDict(
            E = 6.998*1e9, #Pa
            rho = 2373, #kg/m3
            L = 160, #m
            #v_norm=0.01165,
            kf=5.41e-4,
            #Fz_norm=1.013e-4,
            mu_norm=39.263,
            k1_norm=97.552,
            k3_norm=2.497e6,
            nterms=50,
            BC="SS",
            nquad=20,
            moving_loads_x_norm=[[0]],
            moving_loads_Fz_norm=[[1.013e-4]],
            moving_loads_v_norm=[0.01165,],
            tvals=t,
            xvals=xvals)

    a = SpecBeam(**pdict)
    a.runme()

#    a = SpecBeam(**pdict)
#
#    a.calulate_qk(t=t)

    yall = a.defl[:,-1]#a.wofx(x=xvals, tslice=slice(-1, None, None), normalise_w=False)[:,0]
    ycompare = np.interp(expected_50terms_x, xvals, yall)



    if DEBUG:


        title ='SpecBeam, const mat prop vs Ding et al (2012) \nFigure 7 deflection when load at Midpoint of beam'
        print(title)
        print(' '.join(['{:>9s}']*4).format('t', 'expect', 'calc', 'diff'))
        for i, j, k in zip(expected_50terms_x, expected_50terms_displacement, ycompare):
            print(' '.join(['{:9.4f}']*4).format(i, j, k, j-k))
        print()




        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot("111")
        ax.set_xlabel("x, m")
        ax.set_ylabel("w, m")
        ax.set_xlim(70, 90)
        ax.set_title(title)

        ax.plot(expected_50terms_x, expected_50terms_displacement,
                label="expect n=50", color='green', marker='o', ls='-')
        ax.plot(expected_200terms_x, expected_200terms_displacement,
                label="expect n=200", color='black', marker=None, ls='-')


        ax.plot(xvals, yall, label="calc n=50")
        ax.plot(expected_50terms_x, ycompare, label="calc, n=50 (compare)",
                color='red', marker='s', ms=4, ls=':')

        leg = ax.legend(loc='upper left')
        leg.draggable()

        plt.show()

    assert_allclose(expected_50terms_displacement, ycompare, atol=2.4e-4)


def test_SpecBeam_const_mat_midpoint_defl_runme_analytical_50():
    """Test SpecBeam for constant mat: close to Ding et al Figure 8, displacement vs time at
    beam midpoint (using the runme method) but with k3=0"""

    start_time0 = time.time()
    ftime = datetime.datetime.fromtimestamp(start_time0).strftime('%Y-%m-%d %H%M%S')

    # expected values are digitised from Ding et al 2012.
    expected_50terms_t = np.array(
      [ 3.511,  3.553,  3.576,  3.605,  3.621,  3.637,  3.651,  3.661,
        3.676,  3.693,  3.701,  3.717,  3.733,  3.743,  3.761,  3.777,
        3.79 ,  3.804,  3.822,  3.833,  3.848,  3.862,  3.877,  3.899,
        3.919,  3.94 ,  3.956,  3.97 ,  3.978,  3.99 ,  4.001,  4.014,
        4.025,  4.041,  4.052,  4.064,  4.076,  4.086,  4.094,  4.104,
        4.112,  4.123,  4.134,  4.146,  4.159,  4.172,  4.18 ,  4.192,
        4.212,  4.221,  4.234,  4.255,  4.273,  4.292,  4.324,  4.357,
        4.373,  4.399,  4.418,  4.445])

    #old values for k3~=0
#    expected_50terms_displacement = np.array(
#      [ -1.30900000e-04,   8.85900000e-05,   2.63900000e-04,
#         3.30200000e-04,   3.30500000e-04,   2.87100000e-04,
#         2.00000000e-04,   9.09500000e-05,  -8.36000000e-05,
#        -3.01800000e-04,  -4.32800000e-04,  -6.07300000e-04,
#        -7.59900000e-04,  -8.90900000e-04,  -9.99800000e-04,
#        -1.02100000e-03,  -9.99100000e-04,  -8.67700000e-04,
#        -5.61300000e-04,  -2.11300000e-04,   2.04300000e-04,
#         7.29200000e-04,   1.42900000e-03,   2.41300000e-03,
#         3.46300000e-03,   4.29400000e-03,   5.08100000e-03,
#         5.62800000e-03,   5.86800000e-03,   6.19600000e-03,
#         6.39300000e-03,   6.52500000e-03,   6.59100000e-03,
#         6.48200000e-03,   6.32900000e-03,   6.02300000e-03,
#         5.69600000e-03,   5.45500000e-03,   5.06200000e-03,
#         4.75600000e-03,   4.40700000e-03,   3.90400000e-03,
#         3.35800000e-03,   2.87800000e-03,   2.35300000e-03,
#         1.82900000e-03,   1.52300000e-03,   9.98700000e-04,
#         5.18300000e-04,   2.12500000e-04,  -4.95200000e-05,
#        -2.67600000e-04,  -3.76500000e-04,  -3.76100000e-04,
#        -2.00600000e-04,   1.87300000e-05,   1.06500000e-04,
#         1.94500000e-04,   1.94900000e-04,   1.30000000e-04])


    expected_50terms_displacement = np.array(
      [ -1.74400116e-05,   2.56595529e-04,   3.64963938e-04,
         3.96163257e-04,   3.52239419e-04,   2.55908617e-04,
         1.36296573e-04,   3.12193756e-05,  -1.48035583e-04,
        -3.73050108e-04,  -4.80427236e-04,  -6.84847774e-04,
        -8.61705766e-04,  -9.44268811e-04,  -1.02333177e-03,
        -1.00753661e-03,  -9.18192599e-04,  -7.32771117e-04,
        -3.72382730e-04,  -7.79760498e-05,   4.06998900e-04,
         9.37679667e-04,   1.57072235e-03,   2.58323632e-03,
         3.52928135e-03,   4.47726082e-03,   5.12406706e-03,
         5.61329890e-03,   5.84231170e-03,   6.13309619e-03,
         6.32505191e-03,   6.46203546e-03,   6.49032925e-03,
         6.39329339e-03,   6.23960924e-03,   5.99242067e-03,
         5.67177967e-03,   5.35598822e-03,   5.07919623e-03,
         4.69552191e-03,   4.37030991e-03,   3.90678952e-03,
         3.43034780e-03,   2.90774639e-03,   2.35231514e-03,
         1.82145021e-03,   1.51724930e-03,   1.09014092e-03,
         4.86434682e-04,   2.61200867e-04,  -5.02752907e-06,
        -2.99906282e-04,  -4.26188536e-04,  -4.46910518e-04,
        -3.09328219e-04,  -6.53887601e-05,   4.77786782e-05,
         1.84483472e-04,   2.30390682e-04,   2.18090273e-04])

    expected_200terms_t = np.array(
      [ 3.503,  3.519,  3.539,  3.556,  3.576,  3.595,  3.613,  3.632,
        3.651,  3.667,  3.69 ,  3.708,  3.727,  3.746,  3.766,  3.782,
        3.801,  3.822,  3.84 ,  3.856,  3.878,  3.896,  3.915,  3.933,
        3.951,  3.97 ,  3.99 ,  4.009,  4.027,  4.046,  4.067,  4.085,
        4.102,  4.122,  4.141,  4.159,  4.178,  4.197,  4.217,  4.234,
        4.252,  4.273,  4.291,  4.31 ,  4.329,  4.347,  4.368,  4.386,
        4.405,  4.423,  4.442,  4.46 ,  4.479,  4.495])

    #wrong : displ are with k3~=0
    expected_200terms_displacement = np.array(
      [  7.04000000e-08,   2.22800000e-05,   2.27000000e-05,
         1.23200000e-06,   2.35100000e-05,   2.39300000e-05,
         2.46400000e-06,  -1.89700000e-05,  -6.22700000e-05,
        -1.27500000e-04,  -1.70700000e-04,  -2.57800000e-04,
        -3.44800000e-04,  -4.53600000e-04,  -5.40600000e-04,
        -5.84000000e-04,  -5.61700000e-04,  -4.73800000e-04,
        -2.54900000e-04,   1.17100000e-04,   6.85900000e-04,
         1.51700000e-03,   2.50100000e-03,   3.70300000e-03,
         5.01500000e-03,   6.26200000e-03,   7.20200000e-03,
         7.61800000e-03,   7.39900000e-03,   6.74400000e-03,
         5.91400000e-03,   5.01800000e-03,   4.10100000e-03,
         3.22700000e-03,   2.50600000e-03,   1.85000000e-03,
         1.32600000e-03,   9.11400000e-04,   5.84000000e-04,
         3.00200000e-04,   1.47600000e-04,   3.87500000e-05,
        -4.82900000e-05,  -1.13400000e-04,  -1.13000000e-04,
        -1.34500000e-04,  -1.34000000e-04,  -1.33600000e-04,
        -1.11400000e-04,  -6.72600000e-05,  -6.68400000e-05,
        -6.64500000e-05,  -4.41700000e-05,  -4.38200000e-05])

    t = np.linspace(0, 4.5, 400)

    #see Dingetal2012 Table 2 for material properties
    pdict = OrderedDict(
            E = 6.998*1e9, #Pa
            rho = 2373, #kg/m3
            L = 160, #m
            #v_norm=0.01165,
            kf=5.41e-4,
            #Fz_norm=1.013e-4,
            mu_norm=39.263,
            k1_norm=97.552,
#            k3_norm=2.497e6,
            nterms=50,
            BC="SS",
            nquad=20,
            k1bar=PolyLine([0,0.5],[0.5,1],[1,1],[1,1]),
#            k1bar=PolyLine([0,0.5],[0.5,1],[2,1],[2,1]),
            moving_loads_x_norm=[[0]],
            moving_loads_Fz_norm=[[1.013e-4]],
            moving_loads_v_norm=[0.01165,],
            tvals=t,
#            tvals=np.array([4.0]),
            xvals_norm=np.array([0.5]),

            use_analytical=True,
            implementation="vectorized",
#            implementation="fortran",
#            file_stem="specbeam50",
            force_calc=True,
            )



    a = SpecBeam(**pdict)
    a.runme()

#    a.calulate_qk(t=t)
#
#    yall = a.wofx(x_norm=0.5, normalise_w=False)
    ycompare = np.interp(expected_50terms_t, t, a.defl[0,:])
#    a.saveme()

    print("n=", pdict['nterms'])
    end_time0 = time.time()
    elapsed_time = (end_time0 - start_time0); print("Total run time={}".format(str(timedelta(seconds=elapsed_time))))

    if DEBUG:


        title ='SpecBeam, const mat prop vs Ding et al (2012) \nFigure 8 Displacement at Midpoint of beam but with k3=0'
        print(title)
        print(' '.join(['{:>9s}']*4).format('t', 'expect', 'calc', 'diff'))
        for i, j, k in zip(expected_50terms_t, expected_50terms_displacement, ycompare):
            print(' '.join(['{:9.4f}']*4).format(i, j, k, j-k))
        print()




        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot("111")
        ax.set_xlabel("time, s")
        ax.set_ylabel("w, m")
        ax.set_xlim(3.5, 4.5)
        ax.set_title(title)

        ax.plot(expected_50terms_t, expected_50terms_displacement,
                label="expect n=50", color='green', marker='o', ls='-')
        ax.plot(expected_200terms_t, expected_200terms_displacement,
                label="expect n=200", color='black', marker=None, ls='-')


        ax.plot(t, a.defl, label="calc n=50")
        ax.plot(expected_50terms_t, ycompare, label="calc, n=50 (compare)",
                color='red', marker='s', ms=4, ls=':')

        leg = ax.legend(loc='upper left')
        leg.draggable()

        plt.show()

    assert_allclose(expected_50terms_displacement, ycompare, atol=2.4e-4)




#    f.write("Total run time={}".format(str(timedelta(seconds=elapsed_time))) + os.linesep)

def test_SpecBeam_const_mat_midpoint_defl_runme_analytical_200():
    """Test SpecBeam for constant mat: close to Ding et al Figure 8, displacement vs time at
    beam midpoint (using the runme method) but with k3=0"""

    start_time0 = time.time()
    ftime = datetime.datetime.fromtimestamp(start_time0).strftime('%Y-%m-%d %H%M%S')

    # expected values are digitised from Ding et al 2012.
    expected_50terms_t = np.array(
      [ 3.511,  3.553,  3.576,  3.605,  3.621,  3.637,  3.651,  3.661,
        3.676,  3.693,  3.701,  3.717,  3.733,  3.743,  3.761,  3.777,
        3.79 ,  3.804,  3.822,  3.833,  3.848,  3.862,  3.877,  3.899,
        3.919,  3.94 ,  3.956,  3.97 ,  3.978,  3.99 ,  4.001,  4.014,
        4.025,  4.041,  4.052,  4.064,  4.076,  4.086,  4.094,  4.104,
        4.112,  4.123,  4.134,  4.146,  4.159,  4.172,  4.18 ,  4.192,
        4.212,  4.221,  4.234,  4.255,  4.273,  4.292,  4.324,  4.357,
        4.373,  4.399,  4.418,  4.445])

    #old values for k3~=0
#    expected_50terms_displacement = np.array(
#      [ -1.30900000e-04,   8.85900000e-05,   2.63900000e-04,
#         3.30200000e-04,   3.30500000e-04,   2.87100000e-04,
#         2.00000000e-04,   9.09500000e-05,  -8.36000000e-05,
#        -3.01800000e-04,  -4.32800000e-04,  -6.07300000e-04,
#        -7.59900000e-04,  -8.90900000e-04,  -9.99800000e-04,
#        -1.02100000e-03,  -9.99100000e-04,  -8.67700000e-04,
#        -5.61300000e-04,  -2.11300000e-04,   2.04300000e-04,
#         7.29200000e-04,   1.42900000e-03,   2.41300000e-03,
#         3.46300000e-03,   4.29400000e-03,   5.08100000e-03,
#         5.62800000e-03,   5.86800000e-03,   6.19600000e-03,
#         6.39300000e-03,   6.52500000e-03,   6.59100000e-03,
#         6.48200000e-03,   6.32900000e-03,   6.02300000e-03,
#         5.69600000e-03,   5.45500000e-03,   5.06200000e-03,
#         4.75600000e-03,   4.40700000e-03,   3.90400000e-03,
#         3.35800000e-03,   2.87800000e-03,   2.35300000e-03,
#         1.82900000e-03,   1.52300000e-03,   9.98700000e-04,
#         5.18300000e-04,   2.12500000e-04,  -4.95200000e-05,
#        -2.67600000e-04,  -3.76500000e-04,  -3.76100000e-04,
#        -2.00600000e-04,   1.87300000e-05,   1.06500000e-04,
#         1.94500000e-04,   1.94900000e-04,   1.30000000e-04])


    expected_50terms_displacement = np.array(
      [ -1.74400116e-05,   2.56595529e-04,   3.64963938e-04,
         3.96163257e-04,   3.52239419e-04,   2.55908617e-04,
         1.36296573e-04,   3.12193756e-05,  -1.48035583e-04,
        -3.73050108e-04,  -4.80427236e-04,  -6.84847774e-04,
        -8.61705766e-04,  -9.44268811e-04,  -1.02333177e-03,
        -1.00753661e-03,  -9.18192599e-04,  -7.32771117e-04,
        -3.72382730e-04,  -7.79760498e-05,   4.06998900e-04,
         9.37679667e-04,   1.57072235e-03,   2.58323632e-03,
         3.52928135e-03,   4.47726082e-03,   5.12406706e-03,
         5.61329890e-03,   5.84231170e-03,   6.13309619e-03,
         6.32505191e-03,   6.46203546e-03,   6.49032925e-03,
         6.39329339e-03,   6.23960924e-03,   5.99242067e-03,
         5.67177967e-03,   5.35598822e-03,   5.07919623e-03,
         4.69552191e-03,   4.37030991e-03,   3.90678952e-03,
         3.43034780e-03,   2.90774639e-03,   2.35231514e-03,
         1.82145021e-03,   1.51724930e-03,   1.09014092e-03,
         4.86434682e-04,   2.61200867e-04,  -5.02752907e-06,
        -2.99906282e-04,  -4.26188536e-04,  -4.46910518e-04,
        -3.09328219e-04,  -6.53887601e-05,   4.77786782e-05,
         1.84483472e-04,   2.30390682e-04,   2.18090273e-04])

    expected_200terms_t = np.array(
      [ 3.503,  3.519,  3.539,  3.556,  3.576,  3.595,  3.613,  3.632,
        3.651,  3.667,  3.69 ,  3.708,  3.727,  3.746,  3.766,  3.782,
        3.801,  3.822,  3.84 ,  3.856,  3.878,  3.896,  3.915,  3.933,
        3.951,  3.97 ,  3.99 ,  4.009,  4.027,  4.046,  4.067,  4.085,
        4.102,  4.122,  4.141,  4.159,  4.178,  4.197,  4.217,  4.234,
        4.252,  4.273,  4.291,  4.31 ,  4.329,  4.347,  4.368,  4.386,
        4.405,  4.423,  4.442,  4.46 ,  4.479,  4.495])


    expected_200terms_displacement = np.array(
      [  7.04000000e-08,   2.22800000e-05,   2.27000000e-05,
         1.23200000e-06,   2.35100000e-05,   2.39300000e-05,
         2.46400000e-06,  -1.89700000e-05,  -6.22700000e-05,
        -1.27500000e-04,  -1.70700000e-04,  -2.57800000e-04,
        -3.44800000e-04,  -4.53600000e-04,  -5.40600000e-04,
        -5.84000000e-04,  -5.61700000e-04,  -4.73800000e-04,
        -2.54900000e-04,   1.17100000e-04,   6.85900000e-04,
         1.51700000e-03,   2.50100000e-03,   3.70300000e-03,
         5.01500000e-03,   6.26200000e-03,   7.20200000e-03,
         7.61800000e-03,   7.39900000e-03,   6.74400000e-03,
         5.91400000e-03,   5.01800000e-03,   4.10100000e-03,
         3.22700000e-03,   2.50600000e-03,   1.85000000e-03,
         1.32600000e-03,   9.11400000e-04,   5.84000000e-04,
         3.00200000e-04,   1.47600000e-04,   3.87500000e-05,
        -4.82900000e-05,  -1.13400000e-04,  -1.13000000e-04,
        -1.34500000e-04,  -1.34000000e-04,  -1.33600000e-04,
        -1.11400000e-04,  -6.72600000e-05,  -6.68400000e-05,
        -6.64500000e-05,  -4.41700000e-05,  -4.38200000e-05])

    t = np.linspace(0, 4.5, 400)

    #see Dingetal2012 Table 2 for material properties
    pdict = OrderedDict(
            E = 6.998*1e9, #Pa
            rho = 2373, #kg/m3
            L = 160, #m
            #v_norm=0.01165,
            kf=5.41e-4,
            #Fz_norm=1.013e-4,
            mu_norm=39.263,
            k1_norm=97.552,
#            k3_norm=2.497e6,
            nterms=200,
            BC="SS",
            nquad=20,
            k1bar=PolyLine([0,0.5],[0.5,1],[1,1],[1,1]),
#            k1bar=PolyLine([0,0.5],[0.5,1],[2,1],[2,1]),
            moving_loads_x_norm=[[0]],
            moving_loads_Fz_norm=[[1.013e-4]],
            moving_loads_v_norm=[0.01165,],
            tvals=t,
#            tvals=np.array([4.0]),
            xvals_norm=np.array([0.5]),

            use_analytical=True,
            implementation="vectorized",
#            implementation="fortran",
            force_calc=True,
            )



    a = SpecBeam(**pdict)
    a.runme()

#    a.calulate_qk(t=t)
#
#    yall = a.wofx(x_norm=0.5, normalise_w=False)
    ycompare = np.interp(expected_200terms_t, t, a.defl[0,:])
#    print(repr(ycompare))

    print("n=", pdict['nterms'])
    end_time0 = time.time()
    elapsed_time = (end_time0 - start_time0); print("Total run time={}".format(str(timedelta(seconds=elapsed_time))))

    if DEBUG:


        title ='SpecBeam, const mat prop vs Ding et al (2012) \nFigure 8 Displacement at Midpoint of beam but with k3=0'
        print(title)
        print(' '.join(['{:>9s}']*4).format('t', 'expect', 'calc', 'diff'))
        for i, j, k in zip(expected_200terms_t, expected_200terms_displacement, ycompare):
            print(' '.join(['{:9.4f}']*4).format(i, j, k, j-k))
        print()




        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot("111")
        ax.set_xlabel("time, s")
        ax.set_ylabel("w, m")
        ax.set_xlim(3.5, 4.5)
        ax.set_title(title)

        ax.plot(expected_50terms_t, expected_50terms_displacement,
                label="expect n=50", color='green', marker='o', ls='-')
        ax.plot(expected_200terms_t, expected_200terms_displacement,
                label="expect n=200", color='black', marker=None, ls='-')


        ax.plot(t, a.defl, label="calc n=200")
        ax.plot(expected_200terms_t, ycompare, label="calc, n=200 (compare)",
                color='red', marker='s', ms=4, ls=':')

        leg = ax.legend(loc='upper left')
        leg.draggable()

        plt.show()

    assert_allclose(expected_200terms_displacement, ycompare, atol=2.4e-4)




    #f.write("Total run time={}".format(str(timedelta(seconds=elapsed_time))) + os.linesep)

def test_SpecBeam_const_mat_deflection_shape_stationary_load():
    """static point load at beam centre, analytical vs numerical
    properties from moving load case (but with k3=0) in Figure 7 of ding et al"""

    # expected values are digitised from Ding et al 2012.
    expected_50terms_x = np.array(
      [ 70.68 ,  71.275,  71.785,  72.189,  72.529,  72.848,  73.209,
        73.677,  74.06 ,  74.378,  74.74 ,  75.122,  75.505,  75.781,
        76.079,  76.312,  76.525,  76.716,  76.95 ,  77.12 ,  77.418,
        77.715,  77.907,  78.204,  78.459,  78.778,  79.224,  79.522,
        79.883,  80.181,  80.436,  80.606,  80.776,  81.01 ,  81.137,
        81.307,  81.477,  81.626,  81.902,  82.179,  82.497,  82.774,
        82.986,  83.263,  83.539,  84.028,  84.368,  84.623,  84.878,
        85.239,  85.728,  86.153,  86.684,  87.386,  87.853,  88.363,
        88.81 ,  89.341,  89.788])

    # this was generated with use_analytical=False
    expected_50terms_displacement = np.array(
      [  1.65264432e-04,   2.63028987e-04,   2.72285077e-04,
         2.14305202e-04,   1.15107602e-04,  -2.68417019e-05,
        -1.91982014e-04,  -4.49035633e-04,  -6.36376878e-04,
        -7.29821557e-04,  -8.36195689e-04,  -7.26860083e-04,
        -5.95623829e-04,  -3.27010071e-04,   3.04509524e-05,
         3.09942290e-04,   6.76583319e-04,   1.05404663e-03,
         1.51648858e-03,   1.85245069e-03,   2.52742529e-03,
         3.22449986e-03,   3.67513393e-03,   4.31941644e-03,
         4.85502315e-03,   5.52505664e-03,   6.09193410e-03,
         6.46152111e-03,   6.55576573e-03,   6.55576573e-03,
         6.51361055e-03,   6.30277232e-03,   6.09193410e-03,
         5.80172148e-03,   5.64421292e-03,   5.34652107e-03,
         4.98944993e-03,   4.67648758e-03,   4.09677209e-03,
         3.47328742e-03,   2.72692475e-03,   2.07679122e-03,
         1.64296844e-03,   1.09554783e-03,   5.50103467e-04,
        -9.78991467e-05,  -5.05740583e-04,  -6.39483464e-04,
        -7.26860083e-04,  -8.42366564e-04,  -6.98673331e-04,
        -5.42409810e-04,  -2.50752820e-04,   7.72841199e-05,
         2.20332813e-04,   2.93525229e-04,   2.50445556e-04,
         1.59399796e-04,   3.45668416e-05])

    #no longer correct:
    expected_200terms_x = np.array(
      [ 70.064,  70.404,  70.723,  71.02 ,  71.36 ,  71.679,  71.955,
        72.317,  72.614,  72.976,  73.273,  73.571,  73.911,  74.23 ,
        74.527,  74.867,  75.165,  75.505,  75.866,  76.164,  76.483,
        76.801,  77.12 ,  77.418,  77.779,  78.098,  78.438,  78.693,
        79.033,  79.352,  79.671,  79.989,  80.308,  80.648,  80.967,
        81.286,  81.583,  81.924,  82.264,  82.582,  82.88 ,  83.199,
        83.539,  83.815,  84.134,  84.495,  84.793,  85.112,  85.43 ,
        85.749,  86.047,  86.429,  86.727,  87.046,  87.365,  87.683,
        87.981,  88.342,  88.64 ,  88.959,  89.277,  89.617,  89.958])

    expected_200terms_displacement = np.array(
      [ -2.90800000e-05,  -2.90800000e-05,  -6.13900000e-05,
        -7.75400000e-05,  -7.75400000e-05,  -1.09900000e-04,
        -1.26000000e-04,  -1.26000000e-04,  -1.26000000e-04,
        -1.42200000e-04,  -1.09900000e-04,  -1.09900000e-04,
        -9.37000000e-05,  -2.90800000e-05,   3.55400000e-05,
         1.64800000e-04,   2.94000000e-04,   4.55600000e-04,
         7.30200000e-04,   1.02100000e-03,   1.40900000e-03,
         1.86100000e-03,   2.37800000e-03,   2.99200000e-03,
         3.68700000e-03,   4.43000000e-03,   5.22100000e-03,
         5.96400000e-03,   6.67500000e-03,   7.27300000e-03,
         7.58000000e-03,   7.48300000e-03,   6.95000000e-03,
         6.06100000e-03,   4.99500000e-03,   3.88000000e-03,
         2.83000000e-03,   1.94200000e-03,   1.16600000e-03,
         5.20200000e-04,   5.17000000e-05,  -2.55300000e-04,
        -4.33000000e-04,  -5.46000000e-04,  -5.78400000e-04,
        -5.62200000e-04,  -4.81400000e-04,  -4.33000000e-04,
        -3.19900000e-04,  -2.39100000e-04,  -1.74500000e-04,
        -1.26000000e-04,  -4.52300000e-05,  -1.29200000e-05,
         1.93900000e-05,   1.93900000e-05,   3.55400000e-05,
         6.78500000e-05,   3.55400000e-05,   5.17000000e-05,
         3.55400000e-05,   1.93900000e-05,   1.93900000e-05])


    t = np.linspace(0, 160/2/20, 400)

    xvals = np.linspace(0, 160, 200)
    #see Dingetal2012 Table 2 for material properties
    pdict = OrderedDict(
            E = 6.998*1e9, #Pa
            rho = 2373, #kg/m3
            L = 160, #m
            #v_norm=0.01165,
            kf=5.41e-4,
            #Fz_norm=1.013e-4,
            mu_norm=39.263,
            k1_norm=97.552,
            #k3_norm=2.497e6,
            nterms=50,
            BC="SS",
            nquad=20,
#            moving_loads_x_norm=[[0]],
#            moving_loads_Fz_norm=[[1.013e-4]],
#            moving_loads_v_norm=[0.01165,],
#            stationary_loads_x=None,
#            stationary_loads_vs_t=None,
#            stationary_loads_omega_phase=None,
            stationary_loads_x_norm=[0.5],
            stationary_loads_vs_t_norm=[PolyLine([0,t[-1]/160*np.sqrt(6.998*1e9/2373.)],[1.013e-4,1.013e-4])],
                                        #self.L * np.sqrt(self.E / self.rho)
#            stationary_loads_omega_phase_norm=None,

            tvals=t,
            xvals=xvals,
            use_analytical=False,
            implementation="vectorized",
#            implementation="fortran",
            )

    a = SpecBeam(**pdict)
    a.runme()

#    a = SpecBeam(**pdict)
#
#    a.calulate_qk(t=t)

    yall = a.defl[:,-1]#a.wofx(x=xvals, tslice=slice(-1, None, None), normalise_w=False)[:,0]
    ycompare = np.interp(expected_50terms_x, xvals, yall)
    print(repr(ycompare))


    if DEBUG:


        title ='SpecBeam, stationary load, analytical vs numerical, deflection shape for point load applied at midpoint\n prop from moving load Figure 7 of Ding et al (2012)'
        print(title)
        print(' '.join(['{:>9s}']*4).format('t', 'expect', 'calc', 'diff'))
        for i, j, k in zip(expected_50terms_x, expected_50terms_displacement, ycompare):
            print(' '.join(['{:9.4f}']*4).format(i, j, k, j-k))
        print()




        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot("111")
        ax.set_xlabel("x, m")
        ax.set_ylabel("w, m")
        ax.set_xlim(70, 90)
        ax.set_title(title)

        ax.plot(expected_50terms_x, expected_50terms_displacement,
                label="expect n=50", color='green', marker='o', ls='-')
        ax.plot(expected_200terms_x, expected_200terms_displacement,
                label="expect n=200", color='black', marker=None, ls='-')


        ax.plot(xvals, yall, label="calc n=50")
        ax.plot(expected_50terms_x, ycompare, label="calc, n=50 (compare)",
                color='red', marker='s', ms=4, ls=':')
        ax.grid()
        leg = ax.legend(loc='upper left')
        leg.draggable()

        plt.show()

    assert_allclose(expected_50terms_displacement, ycompare, atol=2.4e-4)


if __name__ == "__main__":
    mpl.style.use('classic')
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest', '--doctest-options=+ELLIPSIS'])
    #test_DingEtAl2012()

#    test_DingEtAl2012()
#    test_DingEtAl2012_deflection_shape()
#    test_SpecBeam_const_mat_midpoint_defl()
#    test_SpecBeam_const_mat_deflection_shape()
#    test_SpecBeam_const_mat_midpoint_defl_runme()
#    test_SpecBeam_const_mat_deflection_shape_runme()
#    test_SpecBeam_const_mat_midpoint_defl_runme_analytical_50()
#    test_SpecBeam_const_mat_midpoint_defl_runme_analytical_200()
#    SpecBeam_const_mat_midpoint_defl_runme_analytical_play()
#    test_SpecBeam_const_mat_deflection_shape_stationary_load()




