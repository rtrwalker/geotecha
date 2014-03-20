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

from geotecha.speccon.speccon1d_vr import Speccon1dVR

import geotecha.math.transformations as transformations

TERZ1D_Z = np.array([0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,
                      0.7,  0.8,  0.9,  1. ])


TERZ1D_T = np.array([0.008, 0.018, 0.031, 0.049, 0.071, 0.096, 0.126,
                      0.159, 0.197, 0.239, 0.286, 0.34, 0.403, 0.477, 0.567,
                      0.684, 0.848, 1.129, 1.781])

TERZ1D_POR = np.array(
      [[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.5708047 ,  0.40183855,  0.31202868,  0.25060581,  0.209277  ,
         0.18051017,  0.15777238,  0.1401947 ,  0.12492869,  0.11139703,
         0.09868545,  0.08618205,  0.07371295,  0.0613951 ,  0.04916581,
         0.03683692,  0.02457785,  0.01228656,  0.00245901],
       [ 0.8861537 ,  0.70815945,  0.57815202,  0.47709676,  0.40440265,
         0.35188372,  0.30934721,  0.27584089,  0.24631156,  0.21986645,
         0.19487593,  0.17022241,  0.145606  ,  0.12127752,  0.09712088,
         0.07276678,  0.04855051,  0.02427058,  0.00485748],
       [ 0.98229393,  0.8861537 ,  0.77173068,  0.66209592,  0.57402972,
         0.50633278,  0.44919934,  0.40274312,  0.36079264,  0.322593  ,
         0.28615206,  0.25003642,  0.21390512,  0.17817202,  0.14268427,
         0.10690487,  0.07132769,  0.03565698,  0.00713634],
       [ 0.9984346 ,  0.96498502,  0.89182244,  0.79866319,  0.71151086,
         0.63842889,  0.57300943,  0.5173496 ,  0.46536864,  0.41697458,
         0.37024076,  0.32365106,  0.27692667,  0.23067729,  0.18473404,
         0.13841059,  0.09234855,  0.04616539,  0.00923947],
       [ 0.99992277,  0.99159201,  0.95536184,  0.8897753 ,  0.81537699,
         0.74554825,  0.67795464,  0.61693194,  0.55750293,  0.50070214,
         0.44507671,  0.38925529,  0.33311924,  0.27750057,  0.22223477,
         0.16650815,  0.11109548,  0.05553705,  0.0111151 ],
       [ 0.9999979 ,  0.9984346 ,  0.9840325 ,  0.94470726,  0.88846498,
         0.82769841,  0.76271322,  0.69962982,  0.63517948,  0.57181325,
         0.50885214,  0.44524423,  0.38110176,  0.3174894 ,  0.25426314,
         0.19050572,  0.12710688,  0.0635412 ,  0.01271704],
       [ 0.99999997,  0.99977515,  0.99506515,  0.97461982,  0.93621426,
         0.88684221,  0.82720628,  0.76436582,  0.69689722,  0.62871883,
         0.5600537 ,  0.49025645,  0.41969701,  0.34965996,  0.28003063,
         0.2098124 ,  0.13998847,  0.06998076,  0.01400585],
       [ 1.        ,  0.99997517,  0.99868444,  0.9892702 ,  0.96479424,
         0.92594095,  0.87215551,  0.81066205,  0.74161724,  0.67020692,
         0.59748729,  0.52320368,  0.44795959,  0.37322105,  0.29890288,
         0.2239528 ,  0.14942309,  0.07469716,  0.01494978],
       [ 1.        ,  0.99999789,  0.99968908,  0.99551731,  0.97956541,
         0.94796078,  0.89856843,  0.83840947,  0.76868357,  0.69543129,
         0.62029292,  0.54329327,  0.46519818,  0.3875934 ,  0.31041531,
         0.23257876,  0.15517842,  0.07757427,  0.0155256 ],
       [ 1.        ,  0.99999973,  0.99988166,  0.9971974 ,  0.98407824,
         0.95504225,  0.90726835,  0.8476479 ,  0.77774256,  0.70389411,
         0.62795246,  0.55004364,  0.47099154,  0.39242376,  0.31428453,
         0.23547787,  0.15711273,  0.07854125,  0.01571913]])



TERZ1D_AVP = np.array(
        [[ 0.8990747 ,  0.84861205,  0.80132835,  0.75022262,  0.69933407,
        0.65038539,  0.59948052,  0.55017049,  0.49966188,  0.44989787,
        0.40039553,  0.35035814,  0.2998893 ,  0.24983377,  0.20008097,
        0.14990996,  0.10002108,  0.05000091,  0.01000711]])


def test_terzaghi_1d_PTIB():
    """test for terzaghi 1d PTIB

    dTv turns out to be 1.0
    Pervious top impervious bottom
    instant surcharge of 100

    """
    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np
    H = 1
    drn = 1
    dTv = 0.1
    neig = 20

    mvref = 2.0
    mv = PolyLine([0,1], [0.5,0.5])
    kv = PolyLine([0,1], [5,5])

    #note: combo of dTv, mv, kv essentially gives dTv = 1

    surcharge_vs_depth = PolyLine([0,1], [100,100])
    surcharge_vs_time = PolyLine([0,0.0,8], [0,1,1])


    ppress_z = np.%s
    avg_ppress_z_pairs = [[0,1]]
    settlement_z_pairs = [[0,1]]

    tvals = np.%s

    """ % (repr(TERZ1D_Z), repr(TERZ1D_T)))


    por = 100 * TERZ1D_POR
    avp = 100 * TERZ1D_AVP
    settle = 100 * (1 - TERZ1D_AVP)



    for impl in ["scalar", "vectorized", "fortran"]:
        for dT in [0.1, 1, 10]:
            a = Speccon1dVR(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()

            assert_allclose(a.avp, avp, atol=1e-2,
                            err_msg = ("Fail. test_terzaghi_1d_PTIB, avp, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.por, por, atol=1e-2,
                            err_msg = ("Fail. test_terzaghi_1d_PTIB, por, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=1e-2,
                            err_msg = ("Fail. test_terzaghi_1d_PTIB, settle, "
                                "implementation='%s', dT=%s" % (impl, dT)))


def test_terzaghi_1d_PTPB():
    """test for terzaghi 1d PTPB

    dTv turns out to be 1.0
    Pervious top pervious bottom
    instant surcharge of 100

    """

    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np
    H = 1
    drn = 0
    dTv = 0.1 * 0.25
    neig = 20

    mvref = 2.0
    mv = PolyLine([0,1], [0.5,0.5])
    kv = PolyLine([0,1], [5,5])

    #note: combo of dTv, mv, kv essentially gives dTv = 1

    surcharge_vs_depth = PolyLine([0,1], [100,100])
    surcharge_vs_time = PolyLine([0,0.0,8], [0,1,1])

    ppress_z = np.%s
    avg_ppress_z_pairs = [[0,1]]
    settlement_z_pairs = [[0,1]]

    tvals = np.%s

    """ % (repr(np.append(0.5*TERZ1D_Z, 1 - 0.5*TERZ1D_Z[::-1])),
           repr(TERZ1D_T)))


    por = 100 * np.vstack((TERZ1D_POR, TERZ1D_POR[::-1,:]))
    avp = 100 * TERZ1D_AVP
    settle = 100 * (1 - TERZ1D_AVP)



    for impl in ["scalar", "vectorized", "fortran"]:
        for dT in [0.1]:
            a = Speccon1dVR(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()
#            print(a.por)
            assert_allclose(a.avp, avp, atol=1e-2,
                            err_msg = ("Fail. test_terzaghi_1d_PTPB, avp, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.por, por, atol=1e-2,
                            err_msg = ("Fail. test_terzaghi_1d_PTPB, por, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=1e-2,
                            err_msg = ("Fail. test_terzaghi_1d_PTPB, settle, "
                                "implementation='%s', dT=%s" % (impl, dT)))


def test_BC_terzaghi_1d_PTIB():
    """test for boundary condition immitation of terzaghi 1d, PTIB.

    dTv turns out to be 1.0
    Pervious top impervious bottom
    imitates surcharge of 100. i.e. top BC reduces instantly to -100


    """
    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np
    H = 1
    drn = 1
    dTv = 0.1
    neig = 20

    mvref = 2.0
    mv = PolyLine([0,1], [0.5,0.5])
    kv = PolyLine([0,1], [5,5])

    #note: combo of dTv, mv, kv essentially gives dTv = 1

    top_vs_time = PolyLine([0, 0.0, 5], [0,-100,-100])


    ppress_z = np.%s
    avg_ppress_z_pairs = [[0,1]]
    settlement_z_pairs = [[0,1]]

    tvals = np.%s

    """ % (repr(TERZ1D_Z), repr(TERZ1D_T)))


    por = 100 * TERZ1D_POR - 100
    avp = 100 * TERZ1D_AVP - 100
    settle = 100 * (1 - TERZ1D_AVP)



    for impl in ["scalar", "vectorized", "fortran"]:
        for dT in [0.1, 1, 10]:
            a = Speccon1dVR(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()

            assert_allclose(a.avp, avp, atol=1e-2,
                            err_msg = ("Fail. test_BC_terzaghi_1d_PTIB, avp, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.por, por, atol=1e-2,
                            err_msg = ("Fail. test_BC_terzaghi_1d_PTIB, por, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=1e-2,
                            err_msg = ("Fail. test_BC_terzaghi_1d_PTIB, settle, "
                                "implementation='%s', dT=%s" % (impl, dT)))

def test_BC_terzaghi_1d_PTPB():
    """test for boundary condition imitation of terzaghi 1d, PTPB.

    dTv turns out to be 1.0
    Pervious top pervious bottom
    imitates surcharge of 100. i.e. top and bot BC reduces instantly to -100

    """

    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np
    H = 1
    drn = 0
    dTv = 0.1 * 0.25
    neig = 20

    mvref = 2.0
    mv = PolyLine([0,1], [0.5,0.5])
    kv = PolyLine([0,1], [5,5])

    #note: combo of dTv, mv, kv essentially gives dTv = 1

    top_vs_time = PolyLine([0, 0.0, 5], [0,-100,-100])
    bot_vs_time = PolyLine([0, 0.0, 5], [0,-100,-100])

    ppress_z = np.%s
    avg_ppress_z_pairs = [[0,1]]
    settlement_z_pairs = [[0,1]]

    tvals = np.%s

    """ % (repr(np.append(0.5*TERZ1D_Z, 1 - 0.5*TERZ1D_Z[::-1])),
           repr(TERZ1D_T)))


    por = 100 * np.vstack((TERZ1D_POR, TERZ1D_POR[::-1,:])) - 100
    avp = 100 * TERZ1D_AVP - 100
    settle = 100 * (1 - TERZ1D_AVP)



    for impl in ["scalar", "vectorized", "fortran"]:
        for dT in [0.1]:
            a = Speccon1dVR(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()
#            print(a.por)
            assert_allclose(a.avp, avp, atol=1e-2,
                            err_msg = ("Fail. test_BC_terzaghi_1d_PTPB, avp, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.por, por, atol=1e-2,
                            err_msg = ("Fail. test_BC_terzaghi_1d_PTPB, por, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=1e-2,
                            err_msg = ("Fail. test_BC_terzaghi_1d_PTPB, settle, "
                                "implementation='%s', dT=%s" % (impl, dT)))

def test_schiffman_and_stein_1970():
    """test for multilayer vertical consolidation

    example as per Schiffman and stein 1970

    """


    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np

    #<start params from Schiffman and stein
    h = np.array([10, 20, 30, 20])
    cv = np.array([0.0411, 0.1918, 0.0548, 0.0686])
    mv = np.array([3.07e-3, 1.95e-3, 9.74e-4, 1.95e-3])
    #kv = np.array([7.89e-6, 2.34e-5, 3.33e-6, 8.35e-6])
    kv = cv*mv

    bctop = 0
    #htop = None
    #ktop = None
    bcbot = 0
    #hbot = None
    #kbot = None

    n = 25
    surcharge_vs_time = [PolyLine([0,0,10], [0,100,100])]
    #end params from Schiffman and stein>

    H = np.sum(h)
    z2 = np.cumsum(h) / H
    z1 = (np.cumsum(h) - h) / H
    mvref = mv[0]
    kvref = kv[0]

    drn = 0

    dTv = 1 / H**2 * kvref / mvref
    neig = 60


    mv = PolyLine(z1, z2, mv/mvref, mv/mvref)

    kv = PolyLine(z1, z2, kv/kvref, kv/kvref)


    surcharge_vs_time = PolyLine([0,0,30000], [0,100,100])
    surcharge_vs_depth = PolyLine([0,1], [1,1])


    ppress_z = np.array(
        [  0. ,   0.4,   0.8,   1.2,   1.6,   2. ,   2.4,   2.8,   3.2,
         3.6,   4. ,   4.4,   4.8,   5.2,   5.6,   6. ,   6.4,   6.8,
         7.2,   7.6,   8. ,   8.4,   8.8,   9.2,   9.6,  10. ,  10.8,
        11.6,  12.4,  13.2,  14. ,  14.8,  15.6,  16.4,  17.2,  18. ,
        18.8,  19.6,  20.4,  21.2,  22. ,  22.8,  23.6,  24.4,  25.2,
        26. ,  26.8,  27.6,  28.4,  29.2,  30. ,  31.2,  32.4,  33.6,
        34.8,  36. ,  37.2,  38.4,  39.6,  40.8,  42. ,  43.2,  44.4,
        45.6,  46.8,  48. ,  49.2,  50.4,  51.6,  52.8,  54. ,  55.2,
        56.4,  57.6,  58.8,  60. ,  80. ])/H

    tvals=np.array(
        [1.21957046e+02,   1.61026203e+02,   2.12611233e+02,
         2.80721620e+02,   3.70651291e+02,   4.89390092e+02,
         740.0,   8.53167852e+02,   1.12648169e+03,
         1.48735211e+03,   1.96382800e+03,   2930.0,
         3.42359796e+03,   4.52035366e+03,   5.96845700e+03,
         7195.0,   1.04049831e+04,   1.37382380e+04,
         1.81393069e+04,   2.39502662e+04,   3.16227766e+04])

    ppress_z_tval_indexes=[6, 11, 15]

    avg_ppress_z_pairs = [[0,1]]
    settlement_z_pairs = [[0,1]]

    """)

    t = np.array(
        [1.21957046e+02,   1.61026203e+02,   2.12611233e+02,
         2.80721620e+02,   3.70651291e+02,   4.89390092e+02,
         740.0,   8.53167852e+02,   1.12648169e+03,
         1.48735211e+03,   1.96382800e+03,   2930.0,
         3.42359796e+03,   4.52035366e+03,   5.96845700e+03,
         7195.0,   1.04049831e+04,   1.37382380e+04,
         1.81393069e+04,   2.39502662e+04,   3.16227766e+04])

    z = np.array(
        [  0. ,   0.4,   0.8,   1.2,   1.6,   2. ,   2.4,   2.8,   3.2,
         3.6,   4. ,   4.4,   4.8,   5.2,   5.6,   6. ,   6.4,   6.8,
         7.2,   7.6,   8. ,   8.4,   8.8,   9.2,   9.6,  10. ,  10.8,
        11.6,  12.4,  13.2,  14. ,  14.8,  15.6,  16.4,  17.2,  18. ,
        18.8,  19.6,  20.4,  21.2,  22. ,  22.8,  23.6,  24.4,  25.2,
        26. ,  26.8,  27.6,  28.4,  29.2,  30. ,  31.2,  32.4,  33.6,
        34.8,  36. ,  37.2,  38.4,  39.6,  40.8,  42. ,  43.2,  44.4,
        45.6,  46.8,  48. ,  49.2,  50.4,  51.6,  52.8,  54. ,  55.2,
        56.4,  57.6,  58.8,  60. ,  80. ])


    avp = np.array([
      [               92.76130612,  91.67809644,  90.42431165,  88.96340161,
        87.24639479,  85.20740028,  81.3788705 ,  79.77961148,
        76.12942592,  71.64722094,  66.18237931,  56.39858151,
        51.98404777,  43.386903  ,  34.191069  ,  27.95321684,
        16.51473866,   9.56939964,   4.65849181,   1.80156111,   0.51403147]])
    settle = np.array([
      [                1.41200092,   1.6224817 ,   1.86433805,   2.14224976,
         2.4616197 ,   2.82878775,   3.48058021,   3.73910018,
         4.30338166,   4.95707122,   5.71168747,   6.98645472,
         7.54031351,   8.59560031,   9.70441392,  10.45107157,
        11.81714408,  12.64703246,  13.23440044,  13.57631953,  13.73045647]])
    por = np.array(
      [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
       [  4.13834977e+00,   2.23230302e+00,   1.09205628e+00],
       [  8.26665907e+00,   4.46299216e+00,   2.18342511e+00],
       [  1.23749742e+01,   6.69045542e+00,   3.27341951e+00],
       [  1.64535135e+01,   8.91308460e+00,   4.36135334e+00],
       [  2.04927512e+01,   1.11292772e+01,   5.44654181e+00],
       [  2.44834962e+01,   1.33374381e+01,   6.52830185e+00],
       [  2.84169681e+01,   1.55359819e+01,   7.60595261e+00],
       [  3.22848665e+01,   1.77233340e+01,   8.67881585e+00],
       [  3.60794342e+01,   1.98979331e+01,   9.74621637e+00],
       [  3.97935136e+01,   2.20582323e+01,   1.08074825e+01],
       [  4.34205949e+01,   2.42027016e+01,   1.18619464e+01],
       [  4.69548559e+01,   2.63298287e+01,   1.29089447e+01],
       [  5.03911937e+01,   2.84381213e+01,   1.39478186e+01],
       [  5.37252464e+01,   3.05261082e+01,   1.49779147e+01],
       [  5.69534067e+01,   3.25923412e+01,   1.59985851e+01],
       [  6.00728261e+01,   3.46353962e+01,   1.70091878e+01],
       [  6.30814103e+01,   3.66538748e+01,   1.80090873e+01],
       [  6.59778056e+01,   3.86464056e+01,   1.89976549e+01],
       [  6.87613785e+01,   4.06116454e+01,   1.99742692e+01],
       [  7.14321862e+01,   4.25482803e+01,   2.09383162e+01],
       [  7.39909409e+01,   4.44550270e+01,   2.18891901e+01],
       [  7.64389684e+01,   4.63306339e+01,   2.28262933e+01],
       [  7.87781601e+01,   4.81738816e+01,   2.37490370e+01],
       [  8.10109212e+01,   4.99835844e+01,   2.46568415e+01],
       [  8.31401151e+01,   5.17585907e+01,   2.55491366e+01],
       [  8.44998523e+01,   5.29288192e+01,   2.61388910e+01],
       [  8.57757214e+01,   5.40677278e+01,   2.67145538e+01],
       [  8.69700510e+01,   5.51747208e+01,   2.72758152e+01],
       [  8.80853569e+01,   5.62492286e+01,   2.78223730e+01],
       [  8.91243107e+01,   5.72907091e+01,   2.83539334e+01],
       [  9.00897093e+01,   5.82986469e+01,   2.88702105e+01],
       [  9.09844442e+01,   5.92725537e+01,   2.93709265e+01],
       [  9.18114723e+01,   6.02119687e+01,   2.98558123e+01],
       [  9.25737875e+01,   6.11164580e+01,   3.03246072e+01],
       [  9.32743935e+01,   6.19856150e+01,   3.07770593e+01],
       [  9.39162783e+01,   6.28190603e+01,   3.12129254e+01],
       [  9.45023898e+01,   6.36164415e+01,   3.16319713e+01],
       [  9.50356133e+01,   6.43774329e+01,   3.20339718e+01],
       [  9.55187511e+01,   6.51017356e+01,   3.24187110e+01],
       [  9.59545034e+01,   6.57890773e+01,   3.27859823e+01],
       [  9.63454509e+01,   6.64392117e+01,   3.31355883e+01],
       [  9.66940400e+01,   6.70519185e+01,   3.34673412e+01],
       [  9.70025690e+01,   6.76270032e+01,   3.37810629e+01],
       [  9.72731763e+01,   6.81642963e+01,   3.40765848e+01],
       [  9.75078308e+01,   6.86636534e+01,   3.43537482e+01],
       [  9.77083235e+01,   6.91249547e+01,   3.46124041e+01],
       [  9.78762608e+01,   6.95481043e+01,   3.48524135e+01],
       [  9.80130591e+01,   6.99330301e+01,   3.50736473e+01],
       [  9.81199421e+01,   7.02796833e+01,   3.52759864e+01],
       [  9.81979376e+01,   7.05880378e+01,   3.54593221e+01],
       [  9.87682210e+01,   7.34771068e+01,   3.72096111e+01],
       [  9.91715754e+01,   7.60658313e+01,   3.88020800e+01],
       [  9.94518248e+01,   7.83564228e+01,   4.02299509e+01],
       [  9.96430709e+01,   8.03517886e+01,   4.14871268e+01],
       [  9.97711949e+01,   8.20552178e+01,   4.25682125e+01],
       [  9.98553518e+01,   8.34700864e+01,   4.34685333e+01],
       [  9.99093466e+01,   8.45995889e+01,   4.41841507e+01],
       [  9.99428210e+01,   8.54465027e+01,   4.47118771e+01],
       [  9.99622150e+01,   8.60129914e+01,   4.50492866e+01],
       [  9.99714904e+01,   8.63004510e+01,   4.51947246e+01],
       [  9.99726177e+01,   8.63094030e+01,   4.51473151e+01],
       [  9.99658317e+01,   8.60394373e+01,   4.49069657e+01],
       [  9.99496594e+01,   8.54892052e+01,   4.44743699e+01],
       [  9.99207219e+01,   8.46564639e+01,   4.38510083e+01],
       [  9.98733033e+01,   8.35381707e+01,   4.30391462e+01],
       [  9.97986801e+01,   8.21306261e+01,   4.20418304e+01],
       [  9.96842063e+01,   8.04296612e+01,   4.08628818e+01],
       [  9.95121587e+01,   7.84308668e+01,   3.95068872e+01],
       [  9.92583711e+01,   7.61298579e+01,   3.79791876e+01],
       [  9.88907136e+01,   7.35225677e+01,   3.62858635e+01],
       [  9.83675182e+01,   7.06055651e+01,   3.44337187e+01],
       [  9.76360994e+01,   6.73763874e+01,   3.24302595e+01],
       [  9.66315712e+01,   6.38338795e+01,   3.02836723e+01],
       [  9.52762065e+01,   5.99785320e+01,   2.80027970e+01],
       [  9.34796129e+01,   5.58128098e+01,   2.55970977e+01],
       [ -1.25983583e-11,  -7.37138821e-12,  -3.67147534e-12]])

    for impl in ["vectorized"]:
        for dT in [0.1]:
            a = Speccon1dVR(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()
#            plt.figure()
#            plt.plot(por, z,'b-*')
#            plt.plot(a.por, z, 'r-+')
#
#
#            plt.figure()
#            plt.plot(t,settle[0],'b-*')
#            plt.plot(t, a.set[0], 'r-+')
#            plt.figure()
#            plt.plot(t, avp[0],'b-*')
#            plt.plot(t, a.avp[0], 'r-+')
#            plt.show()

            #atol is quite high for these but looking at comparative plots
            #they are ok.
            assert_allclose(a.por, por, atol=1,
                            err_msg = ("Fail. test_schiffman_and_stein_1970, por, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.avp, avp, atol=1,
                            err_msg = ("Fail. test_schiffman_and_stein_1970, avp, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=1,
                            err_msg = ("Fail. test_schiffman_and_stein_1970, settle, "
                                "implementation='%s', dT=%s" % (impl, dT)))


def test_fixed_ppress_terzaghi_PTPB():
    """test for fixed_ppress

    fixed pore pressure is zero at 0.5, each half is equivalent to terzaghi_1d
    PTPB

    instant surcharge of 100

    close to the fixed ppress zero is not perfectly accurate but it is reasonable
    """


    tslice = slice(5,None) #restrict times
    zslice = slice(2,None) # restrict zvals
    t = TERZ1D_T[tslice]
    z = np.append(0.25*TERZ1D_Z[zslice], [0.5 - 0.25*TERZ1D_Z[zslice][::-1], 0.5 + 0.25*TERZ1D_Z[zslice], 1 - 0.25 * TERZ1D_Z[zslice][::-1]])



    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np
    H = 1
    drn = 0
    dTv = 0.1 /16
    neig = 40

    mvref = 2.0
    mv = PolyLine([0,1], [0.5,0.5])
    kv = PolyLine([0,1], [5,5])

    #note: combo of dTv, mv, kv essentially gives dTv = 1

    surcharge_vs_depth = PolyLine([0,1], [100,100])
    surcharge_vs_time = PolyLine([0,0.0,8], [0,1,1])


    fixed_ppress = [(0.5, 10000, None)]

    ppress_z = np.%s
    avg_ppress_z_pairs = [[0,1]]
    settlement_z_pairs = [[0,1]]

    tvals = np.%s

    """ % (repr(z),
           repr(t)))




    por = 100 * np.vstack((TERZ1D_POR[zslice, tslice], TERZ1D_POR[zslice, tslice][::-1,:], TERZ1D_POR[zslice, tslice], TERZ1D_POR[zslice, tslice][::-1,:]))
    avp = 100 * TERZ1D_AVP[:, tslice]
    settle = 100 * (1 - TERZ1D_AVP[:,tslice])



    for impl in ["vectorized"]:
        for dT in [0.1, 1, 10]:
            a = Speccon1dVR(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()

#            plt.clf()
#            plt.figure()
#            plt.plot(por, z,'b-*')
#            plt.plot(a.por, z, 'r-+')
#
#
#            plt.figure()
#            plt.plot(t,settle[0],'b-*')
#            plt.plot(t, a.set[0], 'r-+')
#            plt.figure()
#            plt.plot(t, avp[0],'b-*')
#            plt.plot(t, a.avp[0], 'r-+')
#            plt.show()

            assert_allclose(a.avp, avp, atol=2,
                            err_msg = ("Fail. test_fixed_ppress_terzaghi_PTPB, avp, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.por, por, atol=5,
                            err_msg = ("Fail. test_fixed_ppress_terzaghi_PTPB, por, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=2,
                            err_msg = ("Fail. test_fixed_ppress_terzaghi_PTPB, settle, "
                                "implementation='%s', dT=%s" % (impl, dT)))



def test_fixed_ppress_BC_terzaghi_PTPB():
    """test for fixed_ppress

    fixed pore pressure is -100 at 0.5.  fixed boundary conditions are
    instantly -100.  each half is equivalent to terzaghi_1d PTPB -100.

    instant surcharge of 100

    close to the fixed ppress zero is not perfectly accurate but it is reasonable
    """


    tslice = slice(5,None) #restrict times
    zslice = slice(2,None) # restrict zvals
    t = TERZ1D_T[tslice]
    z = np.append(0.25*TERZ1D_Z[zslice], [0.5 - 0.25*TERZ1D_Z[zslice][::-1], 0.5 + 0.25*TERZ1D_Z[zslice], 1 - 0.25 * TERZ1D_Z[zslice][::-1]])



    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np
    H = 1
    drn = 0
    dTv = 0.1 /16
    neig = 40

    mvref = 2.0
    mv = PolyLine([0,1], [0.5,0.5])
    kv = PolyLine([0,1], [5,5])

    #note: combo of dTv, mv, kv essentially gives dTv = 1

    #surcharge_vs_depth = PolyLine([0,1], [100,100])
    #surcharge_vs_time = PolyLine([0,0.0,8], [0,1,1])

    top_vs_time = PolyLine([0, 0.0, 5], [0,-100,-100])
    bot_vs_time = PolyLine([0, 0.0, 5], [0,-100,-100])

    fixed_ppress = [(0.5, 10000, PolyLine([0,0,10],[0,-100,-100]))]

    ppress_z = np.%s
    avg_ppress_z_pairs = [[0,1]]
    settlement_z_pairs = [[0,1]]

    tvals = np.%s

    """ % (repr(z),
           repr(t)))




    por = -100 + 100 * np.vstack((TERZ1D_POR[zslice, tslice], TERZ1D_POR[zslice, tslice][::-1,:], TERZ1D_POR[zslice, tslice], TERZ1D_POR[zslice, tslice][::-1,:]))
    avp = -100 + 100 * TERZ1D_AVP[:, tslice]
    settle = 100 * (1 - TERZ1D_AVP[:,tslice])



    for impl in ["vectorized"]:
        for dT in [0.1, 1, 10]:
            a = Speccon1dVR(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()

#            plt.clf()
#            plt.figure()
#            plt.plot(por, z,'b-*')
#            plt.plot(a.por, z, 'r-+')
#
#
#            plt.figure()
#            plt.plot(t,settle[0],'b-*')
#            plt.plot(t, a.set[0], 'r-+')
#            plt.figure()
#            plt.plot(t, avp[0],'b-*')
#            plt.plot(t, a.avp[0], 'r-+')
#            plt.show()

            assert_allclose(a.avp, avp, atol=2,
                            err_msg = ("Fail. test_fixed_ppress_BC_terzaghi_PTPB, avp, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.por, por, atol=5,
                            err_msg = ("Fail. test_fixed_ppress_BC_terzaghi_PTPB, por, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=2,
                            err_msg = ("Fail. test_fixed_ppress_BC_terzaghi_PTPB, settle, "
                                "implementation='%s', dT=%s" % (impl, dT)))



def test_hansbo_avp():
    """test for average hansbo radial consolidation

    instant surcharge of 100
    compare with 100*exp(t)

    tolerance is quite large because method is not great when no vertical
    drainage is present.

    """


    t = np.array(
      [ 0.05,  0.06,  0.08,  0.1 ,  0.13,  0.17,  0.21,  0.27,  0.35,
        0.44,  0.57,  0.72,  0.92,  1.17,  1.49,  1.9 ,  2.42,  3.09,
        3.93,  5.01])

    hansbo_avp = np.array(
      [[ 0.95122942,  0.94176453,  0.92311635,  0.90483742,  0.87809543,
        0.84366482,  0.81058425,  0.76337949,  0.70468809,  0.64403642,
        0.56552544,  0.48675226,  0.39851904,  0.31036694,  0.22537266,
        0.14956862,  0.08892162,  0.04550195,  0.01964367,  0.0066709 ]])

    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np
    H = 1
    drn = 1
    dTh = 0.1
    neig = 60

    mvref = 2.0
    mv = PolyLine([0, 1], [0.5, 0.5])
    kh = PolyLine([0, 1], [5, 5])
    et = PolyLine([0,1], [1, 1])
    #note: combo of dTv, mv, kv essentially gives dTv = 1

    surcharge_vs_depth = PolyLine([0,1], [100,100])
    surcharge_vs_time = PolyLine([0,0.0,8], [0,1,1])


    avg_ppress_z_pairs = [[0,1]]
    settlement_z_pairs = [[0,1]]

    tvals = np.%s

    """ % (repr(t)))

    avp = 100 * hansbo_avp
    settle = 100 - 100 * hansbo_avp

    for impl in ["vectorized"]:
        for dT in [0.1, 1, 10]:
            a = Speccon1dVR(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()

#            plt.clf()
#            plt.figure()
#            plt.plot(por, z,'b-*', label='expected')
#            plt.plot(a.por, z, 'r-+', label='calculated')
#            plt.legend()
#
#
#            plt.figure()
#            plt.plot(t,settle[0],'b-*', label='expected')
#            plt.plot(t, a.set[0], 'r-+', label='calculated')
#            plt.legend()
#            plt.figure()
#            plt.plot(t, avp[0],'b-*',  label='expected')
#            plt.plot(t, a.avp[0], 'r-+', label='calculated')
#            plt.legend()
#            plt.show()

            assert_allclose(a.avp, avp, atol=1,
                            err_msg = ("Fail. test_hansbo_avp, avp, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=1,
                            err_msg = ("Fail. test_hansbo_avp, settle, "
                                "implementation='%s', dT=%s" % (impl, dT)))

def test_hansbo_avp_vacuum():
    """test for average hansbo radial consolidation

    BC and vacuum drop instantly to -100
    compare with 100*exp(t)-100

    tolerance is quite large because method is not great when no vertical
    drainage is present.

    """


    t = np.array(
      [ 0.05,  0.06,  0.08,  0.1 ,  0.13,  0.17,  0.21,  0.27,  0.35,
        0.44,  0.57,  0.72,  0.92,  1.17,  1.49,  1.9 ,  2.42,  3.09,
        3.93,  5.01])

    hansbo_avp = np.array(
      [[ 0.95122942,  0.94176453,  0.92311635,  0.90483742,  0.87809543,
        0.84366482,  0.81058425,  0.76337949,  0.70468809,  0.64403642,
        0.56552544,  0.48675226,  0.39851904,  0.31036694,  0.22537266,
        0.14956862,  0.08892162,  0.04550195,  0.01964367,  0.0066709 ]])

    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np
    H = 1
    drn = 1
    dTh = 0.1
    neig = 60

    mvref = 2.0
    mv = PolyLine([0, 1], [0.5, 0.5])
    kh = PolyLine([0, 1], [5, 5])
    et = PolyLine([0,1], [1, 1])
    #note: combo of dTv, mv, kv essentially gives dTv = 1

    vacuum_vs_depth = PolyLine([0,1], [1,1])
    vacuum_vs_time = PolyLine([0,0.0,8], [0,-100,-100])

    top_vs_time = PolyLine([0,0.0,8], [0,-100,-100])

    avg_ppress_z_pairs = [[0,1]]
    settlement_z_pairs = [[0,1]]

    tvals = np.%s

    """ % (repr(t)))

    avp = 100 * hansbo_avp - 100
    settle = 100 - 100 * hansbo_avp

    for impl in ["vectorized"]:
        for dT in [0.1,1,10]:
            a = Speccon1dVR(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()

#            plt.clf()
#            plt.figure()
#            plt.plot(por, z,'b-*', label='expected')
#            plt.plot(a.por, z, 'r-+', label='calculated')
#            plt.legend()
#
#
#            plt.figure()
#            plt.plot(t,settle[0],'b-*', label='expected')
#            plt.plot(t, a.set[0], 'r-+', label='calculated')
#            plt.legend()
#            plt.figure()
#            plt.plot(t, avp[0],'b-*',  label='expected')
#            plt.plot(t, a.avp[0], 'r-+', label='calculated')
#            plt.legend()
#            plt.show()

            assert_allclose(a.avp, avp, atol=1,
                            err_msg = ("Fail. test_hansbo_avp_vacuum, avp, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=1,
                            err_msg = ("Fail. test_hansbo_avp_vacuum, settle, "
                                "implementation='%s', dT=%s" % (impl, dT)))

def test_terzaghi_1d_PTPB_bot_BC_gradient():
    """test for terzaghi 1d PTPB simulated by specifying pore pressure gradient at bottom

    top BC drops to -100 instantly
    gradient at bot BC is prescribed

    should be same as terzaghi PTPB - 100

    """



    flow_t = np.array([  0, 0.00000000e+00,   1.00000000e-05,   1.32571137e-05,
         1.75751062e-05,   2.32995181e-05,   3.08884360e-05,
         4.09491506e-05,   5.42867544e-05,   7.19685673e-05,
         9.54095476e-05,   1.26485522e-04,   1.67683294e-04,
         2.22299648e-04,   2.94705170e-04,   3.90693994e-04,
         5.17947468e-04,   6.86648845e-04,   9.10298178e-04,
         1.20679264e-03,   1.59985872e-03,   2.12095089e-03,
         2.81176870e-03,   3.72759372e-03,   4.94171336e-03,
         6.55128557e-03,   8.68511374e-03,   1.15139540e-02,
         1.52641797e-02,   2.02358965e-02,   2.68269580e-02,
         3.55648031e-02,   4.71486636e-02,   6.25055193e-02,
         8.28642773e-02,   1.09854114e-01,   1.45634848e-01,
         1.93069773e-01,   2.55954792e-01,   3.39322177e-01,
         4.49843267e-01,   5.96362332e-01,   7.90604321e-01,
         1.04811313e+00,   1.38949549e+00,   1.84206997e+00,
         2.44205309e+00,   3.23745754e+00,   4.29193426e+00,
         5.68986603e+00,   7.54312006e+00,   1.00000000e+01])

    # flow_v comes from terzaghi_1d_flowrate(z=np.array([0.0]), t=flow_t[tslice], kv=10, mv=1, gamw=10, ui=100, nterms=500)
    flow_v = -np.array([  0.00000000e+00,   1.00000000e+05,   1.78412412e+04,
         1.54953209e+04,   1.34578624e+04,   1.16883065e+04,
         1.01514272e+04,   8.81663000e+03,   7.65734340e+03,
         6.65048985e+03,   5.77602610e+03,   5.01654435e+03,
         4.35692582e+03,   3.78403963e+03,   3.28648146e+03,
         2.85434652e+03,   2.47903242e+03,   2.15306785e+03,
         1.86996392e+03,   1.62408493e+03,   1.41053624e+03,
         1.22506677e+03,   1.06398442e+03,   9.24082570e+02,
         8.02576220e+02,   6.97046575e+02,   6.05392880e+02,
         5.25790600e+02,   4.56655118e+02,   3.96610163e+02,
         3.44460438e+02,   2.99167808e+02,   2.59830644e+02,
         2.25665819e+02,   1.95991124e+02,   1.70184572e+02,
         1.47532018e+02,   1.26954815e+02,   1.07034205e+02,
         8.66871910e+01,   6.59246745e+01,   4.59181293e+01,
         2.84338280e+01,   1.50624045e+01,   6.48748315e+00,
         2.12376806e+00,   4.83256782e-01,   6.78952680e-02,
         5.03366995e-03,   1.59915607e-04,   1.65189842e-06,
         3.84807183e-09])

#    flow_t =np.array([  0.0, 0.00000000e+00,   1.00000000e-04,   2.03503287e-04,
#         4.14135879e-04,   8.42780126e-04,   1.71508526e-03,
#         3.49025488e-03,   7.10278341e-03,   1.44543977e-02,
#         2.94151745e-02,   5.98608469e-02,   1.21818791e-01,
#         2.47905244e-01,   5.04495321e-01,   1.02666456e+00,
#         2.08929613e+00])
#    flow_v = -np.array([  0, 2.00000000e+05,   1.12837917e+04,   7.90986998e+03,
#         5.54477121e+03,   3.88685121e+03,   2.72465929e+03,
#         1.90996975e+03,   1.33887729e+03,   9.38544911e+02,
#         6.57914329e+02,   4.61193930e+02,   3.23118191e+02,
#         2.18601587e+02,   1.15205739e+02,   3.17620214e+01,
#         2.30788836e+00])
    z = np.append(0.5*TERZ1D_Z, 1 - 0.5*TERZ1D_Z[::-1])
    t = TERZ1D_T
    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np
    H = 1
    drn = 1
    dTv = 1 * 0.25
    neig = 15

    mvref = 1.0
    mv = PolyLine([0,1], [1,1])
    kv = PolyLine([0,1], [1,1])

    #note: combo of dTv, mv, kv essentially gives dTv = 1

    top_vs_time = PolyLine([0, 0.0, 5], [0,-100,-100])
    bot_vs_time = PolyLine([0, 0.0, 5], [0,-100,-100])
    bot_vs_time = PolyLine(np.%s, np.%s)

    ppress_z = np.%s
    avg_ppress_z_pairs = [[0,1]]
    settlement_z_pairs = [[0,1]]

    tvals = np.%s

    """ % (repr(flow_t), repr(flow_v*2), repr(z),repr(t)))

    # we use flow_v*2 because flow_v on it's own is for flowrate of
    # terzaghi PTIB where h=H = 1.  for this test we have basically have 2 layers
    # each of h=0.5.  Thus we divide dTv by 4.  The flow_v data is du/dz.
    # because H was one du/dz = du/Dz.  when h=0.5 we need to multiply flow_v
    # 2 to get the same gradient at the base

    por = 100 * np.vstack((TERZ1D_POR, TERZ1D_POR[::-1,:])) - 100
    avp = 100 * TERZ1D_AVP - 100
    settle = 100 * (1 - TERZ1D_AVP)



    for impl in ["vectorized"]:
        for dT in [0.1]:
            a = Speccon1dVR(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()


#            slope = (a.por[-1,:]-a.por[-2,:]) / (a.ppress_z[-1]-a.ppress_z[-2])
#            print(repr(t))
#            print(repr(slope))
#            print(a.por)
#            plt.clf()
#            plt.figure()
#            plt.plot(por, z,'b-*', label='expected')
#            plt.plot(a.por, z,lw=2)
##            plt.plot(a.por, z, 'r-+', label='calculated')
#            plt.gca().invert_yaxis()
#            plt.legend()


#            plt.figure()
#            plt.plot(t,settle[0],'b-*', label='expected')
#            plt.plot(t, a.set[0], 'r-+', label='calculated')
#            plt.legend()
#            plt.figure()
#            plt.plot(t, avp[0],'b-*',  label='expected')
#            plt.plot(t, a.avp[0], 'r-+', label='calculated')
#            plt.legend()
#            plt.show()
            assert_allclose(a.avp, avp, atol=1,
                            err_msg = ("Fail. test_terzaghi_1d_PTPB_bot_BC_gradient, avp, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.por, por, atol=2,
                            err_msg = ("Fail. test_terzaghi_1d_PTPB_bot_BC_gradient, por, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=1,
                            err_msg = ("Fail. test_terzaghi_1d_PTPB_bot_BC_gradient, settle, "
                                "implementation='%s', dT=%s" % (impl, dT)))


def test_terzaghi_1d_pumping():
    """test for terzaghi 1d PTPB simulated by pumping at mid depth

    surcharge of 100
    pumping at mid depth such that pore press at mid depth is zero

    top half should be same as terzaghi 1d PTPB, bottom half should be same
    as terzaghi 1d PTPB.  but H is now 1/4 of terzaghi H


    """



    flow_t = np.array([  0, 0.00000000e+00,   1.00000000e-05,   1.32571137e-05,
         1.75751062e-05,   2.32995181e-05,   3.08884360e-05,
         4.09491506e-05,   5.42867544e-05,   7.19685673e-05,
         9.54095476e-05,   1.26485522e-04,   1.67683294e-04,
         2.22299648e-04,   2.94705170e-04,   3.90693994e-04,
         5.17947468e-04,   6.86648845e-04,   9.10298178e-04,
         1.20679264e-03,   1.59985872e-03,   2.12095089e-03,
         2.81176870e-03,   3.72759372e-03,   4.94171336e-03,
         6.55128557e-03,   8.68511374e-03,   1.15139540e-02,
         1.52641797e-02,   2.02358965e-02,   2.68269580e-02,
         3.55648031e-02,   4.71486636e-02,   6.25055193e-02,
         8.28642773e-02,   1.09854114e-01,   1.45634848e-01,
         1.93069773e-01,   2.55954792e-01,   3.39322177e-01,
         4.49843267e-01,   5.96362332e-01,   7.90604321e-01,
         1.04811313e+00,   1.38949549e+00,   1.84206997e+00,
         2.44205309e+00,   3.23745754e+00,   4.29193426e+00,
         5.68986603e+00,   7.54312006e+00,   1.00000000e+01])
    # flow_v comes from terzaghi_1d_flowrate(z=np.array([0.0]), t=flow_t[tslice], kv=10, mv=1, gamw=10, ui=100, nterms=500)
    flow_v = -np.array([  0.00000000e+00,   1.00000000e+05,   1.78412412e+04,
         1.54953209e+04,   1.34578624e+04,   1.16883065e+04,
         1.01514272e+04,   8.81663000e+03,   7.65734340e+03,
         6.65048985e+03,   5.77602610e+03,   5.01654435e+03,
         4.35692582e+03,   3.78403963e+03,   3.28648146e+03,
         2.85434652e+03,   2.47903242e+03,   2.15306785e+03,
         1.86996392e+03,   1.62408493e+03,   1.41053624e+03,
         1.22506677e+03,   1.06398442e+03,   9.24082570e+02,
         8.02576220e+02,   6.97046575e+02,   6.05392880e+02,
         5.25790600e+02,   4.56655118e+02,   3.96610163e+02,
         3.44460438e+02,   2.99167808e+02,   2.59830644e+02,
         2.25665819e+02,   1.95991124e+02,   1.70184572e+02,
         1.47532018e+02,   1.26954815e+02,   1.07034205e+02,
         8.66871910e+01,   6.59246745e+01,   4.59181293e+01,
         2.84338280e+01,   1.50624045e+01,   6.48748315e+00,
         2.12376806e+00,   4.83256782e-01,   6.78952680e-02,
         5.03366995e-03,   1.59915607e-04,   1.65189842e-06,
         3.84807183e-09])


    tslice = slice(5,-2) #restrict times
    zslice = slice(1,None) # restrict zvals
    t = TERZ1D_T[tslice]
    z = np.append(0.25*TERZ1D_Z[zslice], [0.5 - 0.25*TERZ1D_Z[zslice][::-1], 0.5 + 0.25*TERZ1D_Z[zslice], 1 - 0.25 * TERZ1D_Z[zslice][::-1]])


#    z = np.append(0.5*TERZ1D_Z, 1 - 0.5*TERZ1D_Z[::-1])
#    t = TERZ1D_T
    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np
    H = 1
    drn = 0
    dTv = 0.1 /16
    neig = 40

    mvref = 2.0
    mv = PolyLine([0,1], [0.5,0.5])
    kv = PolyLine([0,1], [5,5])

    #dTv = 1/16
    #mvref = 1.0
    #mv = PolyLine([0,1], [1,1])
    #kv = PolyLine([0,1], [1,1])

    #note: combo of dTv, mv, kv essentially gives dTv = 1

    surcharge_vs_time = PolyLine([0, 0.0, 10], [0,100,100])
    surcharge_vs_depth = PolyLine([0, 1], [1,1])

    pumping = (0.5, PolyLine(np.%s, np.%s))

    ppress_z = np.%s
    avg_ppress_z_pairs = [[0,1]]
    settlement_z_pairs = [[0,1]]

    tvals = np.%s

    """ % (repr(flow_t), repr(2*flow_v/4), repr(z),repr(t)))

    # we use 2*flow_v/4 because flow_v on it's own is for flowrate of
    # terzaghi PTIB where H = 1.  for this test we have basically have 4 layers
    # each of H=0.25.  Thus we divide dTv by 16.  because our pump is
    # extracting for a quarter of the height we divide the original flow_v
    # by 4.  But because we are using a single pump to drain both the top and
    # bottom halves we then multiply by 2.  This gives us our 2*flow_v/4


    por = 100 * np.vstack((TERZ1D_POR[zslice, tslice], TERZ1D_POR[zslice, tslice][::-1,:], TERZ1D_POR[zslice, tslice], TERZ1D_POR[zslice, tslice][::-1,:]))
    avp = 100 * TERZ1D_AVP[:, tslice]
    settle = 100 * (1 - TERZ1D_AVP[:,tslice])
#    por = 100 * np.vstack((TERZ1D_POR, TERZ1D_POR[::-1,:])) - 100
#    avp = 100 * TERZ1D_AVP - 100
#    settle = 100 * (1 - TERZ1D_AVP)


    #Note here that the pore pressure at z = 0.5 is slightly off.
    for impl in ["vectorized"]:
        for dT in [0.1]:
            a = Speccon1dVR(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()


#            slope = (a.por[-1,:]-a.por[-2,:]) / (a.ppress_z[-1]-a.ppress_z[-2])
#            print(repr(t))
#            print(repr(slope))
#            print(a.por)
#            plt.clf()
#            plt.figure()
#            plt.plot(por, z,'b-*', label='expected')
#            plt.plot(a.por, z,lw=2)
##            plt.plot(a.por, z, 'r-+', label='calculated')
#            plt.gca().invert_yaxis()
#            plt.legend()


#            plt.figure()
#            plt.plot(t,settle[0],'b-*', label='expected')
#            plt.plot(t, a.set[0], 'r-+', label='calculated')
#            plt.legend()
#            plt.figure()
#            plt.plot(t, avp[0],'b-*',  label='expected')
#            plt.plot(t, a.avp[0], 'r-+', label='calculated')
#            plt.legend()
#            plt.show()
            assert_allclose(a.avp, avp, atol=1,
                            err_msg = ("Fail. test_terzaghi_1d_PTPB_bot_BC_gradient, avp, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.por, por, atol=2,
                            err_msg = ("Fail. test_terzaghi_1d_PTPB_bot_BC_gradient, por, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=1,
                            err_msg = ("Fail. test_terzaghi_1d_PTPB_bot_BC_gradient, settle, "
                                "implementation='%s', dT=%s" % (impl, dT)))



def test_tang_and_onitsuka_vert_and_radial():
    """tang_and_onitsuka_vert_and_radial

    vertical and radial consolidation
    compare average pore pressure of whole layer and settlement

    H=1
    kv=kh=10, mv=1, gamw=10
    dTv=kvref/mvref/gamw/H**2 = 1
    #re=0.5, rw = 0.03, n = 16.6667, mu = 2.074475589, etref = 3.856396307
    #dTh = 2*khref/mvref/gamw/mu = 3.856396307

    """

    t = np.array([  1.00000000e-03,   2.00000000e-03,   3.00000000e-03,
         4.00000000e-03,   5.00000000e-03,   6.00000000e-03,
         7.00000000e-03,   8.00000000e-03,   9.00000000e-03,
         1.00000000e-02,   2.00000000e-02,   3.00000000e-02,
         4.00000000e-02,   5.00000000e-02,   6.00000000e-02,
         7.00000000e-02,   8.00000000e-02,   9.00000000e-02,
         1.00000000e-01,   1.10000000e-01,   1.20000000e-01,
         1.30000000e-01,   1.40000000e-01,   1.50000000e-01,
         1.60000000e-01,   1.70000000e-01,   1.80000000e-01,
         1.90000000e-01,   2.00000000e-01,   2.10000000e-01,
         2.20000000e-01,   2.30000000e-01,   2.40000000e-01,
         2.50000000e-01,   2.60000000e-01,   2.70000000e-01,
         2.80000000e-01,   2.90000000e-01,   3.00000000e-01,
         3.10000000e-01,   3.20000000e-01,   3.30000000e-01,
         3.40000000e-01,   3.50000000e-01,   3.60000000e-01,
         3.70000000e-01,   3.80000000e-01,   3.90000000e-01,
         4.00000000e-01,   4.10000000e-01,   4.20000000e-01,
         4.30000000e-01,   4.40000000e-01,   4.50000000e-01,
         4.60000000e-01,   4.70000000e-01,   4.80000000e-01,
         4.90000000e-01,   5.00000000e-01,   5.10000000e-01,
         5.20000000e-01,   5.30000000e-01,   5.40000000e-01,
         5.50000000e-01,   5.60000000e-01,   5.70000000e-01,
         5.80000000e-01,   5.90000000e-01,   6.00000000e-01,
         6.10000000e-01,   6.20000000e-01,   6.30000000e-01,
         6.40000000e-01,   6.50000000e-01,   6.60000000e-01,
         6.70000000e-01,   6.80000000e-01,   6.90000000e-01,
         7.00000000e-01,   7.10000000e-01,   7.20000000e-01,
         7.30000000e-01,   7.40000000e-01,   7.50000000e-01,
         7.60000000e-01,   7.70000000e-01,   7.80000000e-01,
         7.90000000e-01,   8.00000000e-01,   8.10000000e-01,
         8.20000000e-01,   8.30000000e-01,   8.40000000e-01,
         8.50000000e-01,   8.60000000e-01,   8.70000000e-01,
         8.80000000e-01,   8.90000000e-01,   9.00000000e-01,
         9.10000000e-01,   9.20000000e-01,   9.30000000e-01,
         9.40000000e-01,   9.50000000e-01,   9.60000000e-01,
         9.70000000e-01,   9.80000000e-01,   9.90000000e-01,
         1.00000000e+00,   1.01000000e+00])



    avp = 100*np.array([[ 0.00324696,  0.00641694,  0.00953238,  0.0126017 ,  0.01562987,
        0.01862029,  0.02157548,  0.02449743,  0.02738778,  0.03024788,
        0.05738761,  0.0822719 ,  0.10525907,  0.12658293,  0.1464181 ,
        0.16490438,  0.18215844,  0.19828034,  0.21335753,  0.22746753,
        0.24067983,  0.25305715,  0.26465659,  0.27553032,  0.25547838,
        0.23790104,  0.22198642,  0.2074141 ,  0.19398549,  0.18155873,
        0.17002455,  0.15929482,  0.14929611,  0.13996587,  0.13124986,
        0.12310046,  0.11547534,  0.10833658,  0.10164995,  0.12563221,
        0.14689894,  0.16627677,  0.18410003,  0.20058033,  0.21587175,
        0.23009504,  0.2433491 ,  0.25571746,  0.26727216,  0.27807632,
        0.28818593,  0.29765116,  0.30651729,  0.31482546,  0.29236538,
        0.27252759,  0.25449115,  0.2379271 ,  0.22262886,  0.20844708,
        0.19526545,  0.18298923,  0.1715388 ,  0.1608458 ,  0.15085055,
        0.14150028,  0.13274787,  0.1245509 ,  0.11687089,  0.10967276,
        0.10292438,  0.09659617,  0.09066084,  0.08509314,  0.07986962,
        0.0749685 ,  0.07036946,  0.06605359,  0.06200322,  0.05820182,
        0.05463397,  0.05128519,  0.04814195,  0.04519158,  0.04242218,
        0.03982263,  0.03738247,  0.03509191,  0.03294176,  0.0309234 ,
        0.02902874,  0.02725019,  0.02558063,  0.02401338,  0.02254216,
        0.02116109,  0.01986464,  0.01864762,  0.01750516,  0.01643271,
        0.01542596,  0.01448089,  0.01359372,  0.0127609 ,  0.01197911,
        0.01124522,  0.01055628,  0.00990956,  0.00930246,  0.00873255]])


    z = np.array([1.0])
#    por = np.array([[  0.283431  ,   0.56778819,   0.85299595,   1.13898094,
#         1.42567202,   1.71300021,   2.00089862,   2.2893024 ,
#         2.5781487 ,   2.86737658,   5.76851863,   8.65330671,
#        11.48571634,  14.2404251 ,  16.90020798,  19.45393937,
#        21.89506213,  24.22041593,  26.42934195,  28.52300016,
#        30.50384982,  32.37525553,  34.14118942,  35.80600729,
#        34.50690467,  33.08215696,  31.58654901,  30.06070598,
#        28.53443942,  27.02931974,  25.56065285,  24.13899804,
#        22.77133268,  21.461945  ,  20.21311782,  19.02565116,
#        17.89926127,  16.83288438,  15.82490746,  17.74071942,
#        19.74447843,  21.78368984,  23.81988286,  25.82527171,
#        27.78019028,  29.67112233,  31.48919014,  33.22899615,
#        34.88773599,  36.4645206 ,  37.95985881,  39.37526352,
#        40.71295263,  41.97562286,  40.2989028 ,  38.51958958,
#        36.69107475,  34.85267114,  33.03295504,  31.25233471,
#        29.5250231 ,  27.86055198,  26.26493296,  24.7415468 ,
#        23.29182368,  21.91576275,  20.61232799,  19.37974914,
#        18.21574971,  17.11771906,  16.08284145,  15.10819246,
#        14.19081006,  13.32774674,  12.51610685,  11.75307301,
#        11.03592401,  10.36204651,   9.72894187,   9.13422959,
#         8.57564807,   8.05105349,   7.5584174 ,   7.09582328,
#         6.66146252,   6.25363001,   5.87071945,   5.51121863,
#         5.17370459,   4.856839  ,   4.55936353,   4.2800954 ,
#         4.01792312,   3.77180238,   3.54075214,   3.32385094,
#         3.12023333,   2.9290866 ,   2.74964755,   2.58119956,
#         2.42306976,   2.2746264 ,   2.13527633,   2.00446268,
#         1.88166266,   1.76638544,   1.65817025,   1.55658452,
#         1.46122217,   1.37170196]])

    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np
    H = 1
    drn = 1
    dTv = 1 #dTv=kvref/mvref/gamw/H**2
    #re=0.5, rw = 0.03, n = 16.6667, mu = 2.074475589,
    #dTh = 2*khref/mvref/gamw/mu


    dTh = 3.856396307

    neig = 20

    mvref = 1.0
    kvref = 10.0
    khref = 10.0
    etref = 3.856396307 #2/mu/re**2

    mv = PolyLine([0,1], [1,1])
    kv = PolyLine([0,1], [1,1])
    kh = PolyLine([0,1], [1,1])
    et = PolyLine([0,1], [1,1])


    surcharge_vs_depth = PolyLine([0,1], [1,1])
    surcharge_vs_time = PolyLine([0,0.15,0.3,0.45,4],[0.0,50,50,100,100])


    #ppress_z = np.%s
    avg_ppress_z_pairs = [[0,1]]
    settlement_z_pairs = [[0,1]]

    tvals = np.%s

    """ % (repr(z), repr(t)))


#    por = 100 * TERZ1D_POR
#    avp = 100 * TERZ1D_AVP
    settle = (np.interp(t,[0,0.15,0.3,0.45,4], [0.0,50,50,100,100]) - avp)



    for impl in ["vectorized"]:
        for dT in [0.1]:
            a = Speccon1dVR(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()



#            #plt.figure()
#            #plt.plot(por, z,'b-*', label='expected')
#            #plt.plot(a.por, z, 'r-+', label='calculated')
#            #plt.legend()
#
#
#            plt.figure()
#            plt.plot(t,settle[0],'b-*', label='expected')
#            plt.plot(t, a.set[0], 'r-+', label='calculated')
#            plt.legend()
#            plt.figure()
#            plt.plot(t, avp[0],'b-*',  label='expected')
#            plt.plot(t, a.avp[0], 'r-+', label='calculated')
#            plt.legend()
#            plt.show()
            assert_allclose(a.avp, avp, atol=1e-2,
                            err_msg = ("Fail. test_terzaghi_1d_PTIB, avp, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.por, por, atol=1e-2,
#                            err_msg = ("Fail. test_terzaghi_1d_PTIB, por, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=1e-2,
                            err_msg = ("Fail. test_terzaghi_1d_PTIB, settle, "
                                "implementation='%s', dT=%s" % (impl, dT)))


class test_omega_phase(unittest.TestCase):
    """compare omega_phase loads to equivalent piecewise"""

    ##To get the piecewise approximation of a mag_vs_time_PolyLIne use:
    ##
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np
    #from geotecha.inputoutput.inputoutput import PrefixNumpyArrayString
    #import geotecha.piecewise.piecewise_linear_1d as pwise
    #
    #PrefixNumpyArrayString().turn_on()
    #
    #vs_depth = PolyLine([0,1], [1,1])
    #vs_time = PolyLine([0,1,2.0], [0,10,10])
    #omega_phase = (2*np.pi*0.1, 0)
    #
    #omega, phase = omega_phase
    #x, y = pwise.subdivide_x_y_into_segments(vs_time.x, vs_time.y, dx=0.1)
    #
    #y = y * np.cos(omega * x + phase)
    #v_time = PolyLine(x, y)
    #print(v_time)


    #reader is generic input file with named parameters:
    #   drn, use_actual, load_to_test.
    reader = textwrap.dedent("""\
        #from geotecha.piecewise.piecewise_linear_1d import PolyLine
        #import numpy as np
        H = 1
        drn = %(drn)d
        dT = 1
        dTh = 1
        dTv = 0.1 * 0.25
        neig = 10


        mvref = 2.0
        kvref = 1.0
        khref = 1.0
        etref = 1.0

        mv = PolyLine([0,1], [0.5,0.5])
        kh = PolyLine([0,1], [1,1])
        kv = PolyLine([0,1], [5,5])

        et = PolyLine([0,1], [1,1])



        vs_depth = PolyLine([0,1], [1,1])
        vs_time = PolyLine([0,1,2.0], [0,10,10])
        omega_phase = (2*np.pi*0.1, 0)

        use_actual = %(use_actual)s
        load_to_test = '%(load_to_test)s'


        if use_actual:
            if load_to_test=='surcharge':
                surcharge_vs_depth = vs_depth
                surcharge_vs_time = vs_time
                surcharge_omega_phase = omega_phase
            if load_to_test=='vacuum':
                vacuum_vs_depth = vs_depth
                vacuum_vs_time = vs_time
                vacuum_omega_phase = omega_phase
            if load_to_test=='top':
                top_vs_time = vs_time
                top_omega_phase = omega_phase
            if load_to_test=='bot':
                bot_vs_time = vs_time
                bot_omega_phase = omega_phase
            if load_to_test=='fixed_ppress':
                fixed_ppress = (0.2, 1000, vs_time)
                fixed_ppress_omega_phase = omega_phase
            if load_to_test=='pumping':
                pumping = (0.4, vs_time)
                pumping_omega_phase = omega_phase

        else: #approximate the sinusoidal loading with a piecewise load
            v_time = PolyLine(np.array([[ 0.        ,  0.        ],
                           [ 0.1       ,  0.99802673],
                           [ 0.2       ,  1.9842294 ],
                           [ 0.3       ,  2.94686175],
                           [ 0.4       ,  3.87433264],
                           [ 0.5       ,  4.75528258],
                           [ 0.6       ,  5.57865892],
                           [ 0.7       ,  6.33378937],
                           [ 0.8       ,  7.01045344],
                           [ 0.9       ,  7.59895133],
                           [ 1.        ,  8.09016994],
                           [ 1.1       ,  7.70513243],
                           [ 1.2       ,  7.28968627],
                           [ 1.3       ,  6.84547106],
                           [ 1.4       ,  6.3742399 ],
                           [ 1.5       ,  5.87785252],
                           [ 1.6       ,  5.35826795],
                           [ 1.7       ,  4.81753674],
                           [ 1.8       ,  4.25779292],
                           [ 1.9       ,  3.68124553],
                           [ 2.        ,  3.09016994]]))


            if load_to_test=='surcharge':
                surcharge_vs_depth = vs_depth
                surcharge_vs_time = v_time
                surcharge_omega_phase = None
            if load_to_test=='vacuum':
                vacuum_vs_depth = vs_depth
                vacuum_vs_time = v_time
                vacuum_omega_phase = None
            if load_to_test=='top':
                top_vs_time = v_time
                top_omega_phase = None
            if load_to_test=='bot':
                bot_vs_time = v_time
                bot_omega_phase = None
            if load_to_test=='fixed_ppress':
                fixed_ppress = (0.2, 1000, v_time)
                fixed_ppress_omega_phase = None
            if load_to_test=='pumping':
                pumping = (0.4, v_time)
                pumping_omega_phase = None


        ppress_z = np.linspace(0,1,20)
        avg_ppress_z_pairs = [[0,1],[0.4, 0.5]]
        settlement_z_pairs = [[0,1],[0.4, 0.5]]

        #tvals = np.logspace(-2, 0.3,50)
        tvals = np.linspace(0.01, 2, 50)
        ppress_z_tval_indexes = np.arange(len(tvals))[::len(tvals)//7]
        #avg_ppress_z_pairs_tval_indexes = slice(None,None)#[0,4,6]
        #settlement_z_pairs_tval_indexes = slice(None, None)#[0,4,6]

        implementation='vectorized'

        #RLzero = -12.0
        #plot_properties={}
            """)


    def test_surcharge(self):
        """test surcharge"""
        drn=0
        load_to_test='surcharge'

        a = Speccon1dVR(self.reader %
            {'drn': drn, 'use_actual': True, 'load_to_test': load_to_test})
        b = Speccon1dVR(self.reader %
            {'drn': drn, 'use_actual': False, 'load_to_test': load_to_test})
        a.make_all()
        b.make_all()
        assert_allclose(a.por, b.por, atol=1e-2)
        assert_allclose(a.avp, b.avp, atol=1e-2)
        assert_allclose(a.set, b.set, atol=1e-2)

    def test_vacuum(self):
        """test vacuum"""
        drn=0
        load_to_test='vacuum'

        a = Speccon1dVR(self.reader %
            {'drn': drn, 'use_actual': True, 'load_to_test': load_to_test})
        b = Speccon1dVR(self.reader %
            {'drn': drn, 'use_actual': False, 'load_to_test': load_to_test})
        a.make_all()
        b.make_all()
        assert_allclose(a.por, b.por, atol=1e-2)
        assert_allclose(a.avp, b.avp, atol=1e-2)
        assert_allclose(a.set, b.set, atol=1e-2)

    def test_top(self):
        """test top"""
        drn=0
        load_to_test='top'

        a = Speccon1dVR(self.reader %
            {'drn': drn, 'use_actual': True, 'load_to_test': load_to_test})
        b = Speccon1dVR(self.reader %
            {'drn': drn, 'use_actual': False, 'load_to_test': load_to_test})
        a.make_all()
        b.make_all()
        assert_allclose(a.por, b.por, atol=1e-1)
        assert_allclose(a.avp, b.avp, atol=1e-2)
        assert_allclose(a.set, b.set, atol=1e-2)

    def test_bot(self):
        """test bot"""
        drn=0
        load_to_test='bot'

        a = Speccon1dVR(self.reader %
            {'drn': drn, 'use_actual': True, 'load_to_test': load_to_test})
        b = Speccon1dVR(self.reader %
            {'drn': drn, 'use_actual': False, 'load_to_test': load_to_test})
        a.make_all()
        b.make_all()
        assert_allclose(a.por, b.por, atol=1e-1)
        assert_allclose(a.avp, b.avp, atol=1e-2)
        assert_allclose(a.set, b.set, atol=1e-2)

    def test_bot_gradient(self):
        """test bot gradient"""
        drn=1
        load_to_test='bot'

        a = Speccon1dVR(self.reader %
            {'drn': drn, 'use_actual': True, 'load_to_test': load_to_test})
        b = Speccon1dVR(self.reader %
            {'drn': drn, 'use_actual': False, 'load_to_test': load_to_test})
        a.make_all()
        b.make_all()
        assert_allclose(a.por, b.por, atol=1e-2)
        assert_allclose(a.avp, b.avp, atol=1e-2)
        assert_allclose(a.set, b.set, atol=1e-2)


#    def test_fixed_ppress(self):
#        """test fixed_ppress"""
#        drn=0
#        load_to_test='fixed_ppress'
#
#        a = Speccon1dVR(self.reader %
#            {'drn': drn, 'use_actual': True, 'load_to_test': load_to_test})
#        b = Speccon1dVR(self.reader %
#            {'drn': drn, 'use_actual': False, 'load_to_test': load_to_test})
#        a.make_all()
#        b.make_all()
#        assert_allclose(a.por, b.por, atol=1e-2)
#        assert_allclose(a.avp, b.avp, atol=1e-2)
#        assert_allclose(a.set, b.set, atol=1e-2)

    def test_pumping(self):
        """test pumping"""
        drn=0
        load_to_test='pumping'

        a = Speccon1dVR(self.reader %
            {'drn': drn, 'use_actual': True, 'load_to_test': load_to_test})
        b = Speccon1dVR(self.reader %
            {'drn': drn, 'use_actual': False, 'load_to_test': load_to_test})
        a.make_all()
        b.make_all()
        assert_allclose(a.por, b.por, atol=1e-2)
        assert_allclose(a.avp, b.avp, atol=1e-2)
        assert_allclose(a.set, b.set, atol=1e-2)
if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])

#    test_terzaghi_1d_PTPB()
#    test_schiffman_and_stein_1970()

#    print(np.append(0.5*TERZ1D_Z, 1-0.5*TERZ1D_Z[::-1]))
#    test_terzaghi_1d()

#    test_fixed_ppress_BC_terzaghi_PTPB()

#    test_hansbo_avp_vacuum()
#    test_terzaghi_1d_PTPB_bot_BC_gradient()
#    test_terzaghi_1d_pumping()
#    test_tang_and_onitsuka_vert_and_radial()