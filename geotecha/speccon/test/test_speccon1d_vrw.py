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
"""Some test routines for the speccon_1d_vert_radial_boundary_well
resistance module

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

from geotecha.speccon.speccon1d_vrw import Speccon1dVRW

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

    dTh=0.1
    dTw=0
    khref = 1
    etref = 1
    kwref=1
    kw = PolyLine([0,1], [1,1])
    kh = PolyLine([0,1], [1,1])
    et = PolyLine([0,1], [1,1])




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



    for impl in ["vectorized", "fortran"]:
        for dT in [0.1, 1, 10]:
            a = Speccon1dVRW(reader + "\n" +
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

    dTh=0.1
    dTw=0
    khref = 1
    etref = 1
    kwref=1
    kw = PolyLine([0,1], [1,1])
    kh = PolyLine([0,1], [1,1])
    et = PolyLine([0,1], [1,1])


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



    for impl in ["vectorized", "fortran"]:
        for dT in [0.1]:
            a = Speccon1dVRW(reader + "\n" +
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

    dTh=0.1
    dTw=0
    khref = 1
    etref = 1
    kwref=1
    kw = PolyLine([0,1], [1,1])
    kh = PolyLine([0,1], [1,1])
    et = PolyLine([0,1], [1,1])

    top_vs_time = PolyLine([0, 0.0, 5], [0,-100,-100])


    ppress_z = np.%s
    avg_ppress_z_pairs = [[0,1]]
    settlement_z_pairs = [[0,1]]

    tvals = np.%s

    """ % (repr(TERZ1D_Z), repr(TERZ1D_T)))


    por = 100 * TERZ1D_POR - 100
    avp = 100 * TERZ1D_AVP - 100
    settle = 100 * (1 - TERZ1D_AVP)



    for impl in ["vectorized", "fortran"]:
        for dT in [0.1, 1, 10]:
            a = Speccon1dVRW(reader + "\n" +
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

    dTh=0.1
    dTw=0
    khref = 1
    etref = 1
    kwref=1
    kw = PolyLine([0,1], [1,1])
    kh = PolyLine([0,1], [1,1])
    et = PolyLine([0,1], [1,1])


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



    for impl in [ "vectorized", "fortran"]:
        for dT in [0.1]:
            a = Speccon1dVRW(reader + "\n" +
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
#
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


    dTh=1
    dTw=0
    khref = 1
    etref = 1
    kwref=1
    kw = PolyLine([0,1], [1,1])
    kh = PolyLine([0,1], [1,1])
    et = PolyLine([0,1], [1,1])

    surcharge_vs_time = PolyLine([0,0,30000], [0,100,100])
    surcharge_vs_depth = PolyLine([0,1], [1,1])


    ppress_z = np.array(
      [  0.        ,   1.        ,   2.        ,   3.        ,
         4.        ,   5.        ,   6.        ,   7.        ,
         8.        ,   9.        ,  10.        ,  12.        ,
        14.        ,  16.        ,  18.        ,  20.        ,
        22.        ,  24.        ,  26.        ,  28.        ,
        30.        ,  33.        ,  36.        ,  39.        ,
        42.        ,  45.        ,  48.        ,  51.        ,
        54.        ,  57.        ,  60.        ,  62.22222222,
        64.44444444,  66.66666667,  68.88888889,  71.11111111,
        73.33333333,  75.55555556,  77.77777778,  80.        ])/H

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
       [  1.03239323e+01,   5.57722764e+00,   2.72863702e+00],
       [  2.04927512e+01,   1.11292772e+01,   5.44654181e+00],
       [  3.03596123e+01,   1.66311548e+01,   8.14302495e+00],
       [  3.97935136e+01,   2.20582323e+01,   1.08074825e+01],
       [  4.86855554e+01,   2.73864218e+01,   1.34294382e+01],
       [  5.69534067e+01,   3.25923412e+01,   1.59985851e+01],
       [  6.45436877e+01,   3.76534683e+01,   1.85048266e+01],
       [  7.14321862e+01,   4.25482803e+01,   2.09383162e+01],
       [  7.76220298e+01,   4.72563776e+01,   2.32894965e+01],
       [  8.31401151e+01,   5.17585907e+01,   2.55491366e+01],
       [  8.63829254e+01,   5.46252500e+01,   2.69970037e+01],
       [  8.91243107e+01,   5.72907091e+01,   2.83539334e+01],
       [  9.14062350e+01,   5.97466005e+01,   2.96153647e+01],
       [  9.32743935e+01,   6.19856150e+01,   3.07770593e+01],
       [  9.47754352e+01,   6.40015052e+01,   3.18351159e+01],
       [  9.59545034e+01,   6.57890773e+01,   3.27859823e+01],
       [  9.68531745e+01,   6.73441748e+01,   3.36264667e+01],
       [  9.75078308e+01,   6.86636534e+01,   3.43537482e+01],
       [  9.79484713e+01,   6.97453488e+01,   3.49653850e+01],
       [  9.81979376e+01,   7.05880378e+01,   3.54593221e+01],
       [  9.93247463e+01,   7.72482246e+01,   3.95369875e+01],
       [  9.97711949e+01,   8.20552178e+01,   4.25682125e+01],
       [  9.99281809e+01,   8.50582178e+01,   4.44716699e+01],
       [  9.99714904e+01,   8.63004510e+01,   4.51947246e+01],
       [  9.99590815e+01,   8.57994747e+01,   4.47146245e+01],
       [  9.98733033e+01,   8.35381707e+01,   4.30391462e+01],
       [  9.96067213e+01,   7.94677707e+01,   4.02066990e+01],
       [  9.88907136e+01,   7.35225677e+01,   3.62858635e+01],
       [  9.71725283e+01,   6.56443116e+01,   3.13743194e+01],
       [  9.34796129e+01,   5.58128098e+01,   2.55970977e+01],
       [  9.11831418e+01,   5.22050000e+01,   2.36222413e+01],
       [  8.71521621e+01,   4.77829652e+01,   2.13561648e+01],
       [  8.11139375e+01,   4.25889537e+01,   1.88257516e+01],
       [  7.28265056e+01,   3.66856767e+01,   1.60614922e+01],
       [  6.21545873e+01,   3.01556450e+01,   1.30971084e+01],
       [  4.91434216e+01,   2.30996016e+01,   9.96911691e+00],
       [  3.40694327e+01,   1.56340897e+01,   6.71633983e+00],
       [  1.74495002e+01,   7.88825146e+00,   3.37936978e+00],
       [  7.44980896e-12,   2.91898853e-12,   7.07416666e-13]])

    for impl in ["vectorized"]:
        for dT in [0.1]:
            a = Speccon1dVRW(reader + "\n" +
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

    dTh=1
    dTw=0
    khref = 1
    etref = 1
    kwref=1
    kw = PolyLine([0,1], [1,1])
    kh = PolyLine([0,1], [1,1])
    et = PolyLine([0,1], [1,1])

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
            a = Speccon1dVRW(reader + "\n" +
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


#
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

    dTh=1
    dTw=0
    khref = 1
    etref = 1
    kwref=1
    kw = PolyLine([0,1], [1,1])
    kh = PolyLine([0,1], [1,1])
    et = PolyLine([0,1], [1,1])


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
            a = Speccon1dVRW(reader + "\n" +
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


#
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

    dTw=1000
    kwref=1
    kw = PolyLine([0,1], [1,1])

    avg_ppress_z_pairs = [[0,1]]
    settlement_z_pairs = [[0,1]]

    tvals = np.%s

    """ % (repr(t)))

    avp = 100 * hansbo_avp
    settle = 100 - 100 * hansbo_avp

    for impl in ["vectorized"]:
        for dT in [0.1, 1, 10]:
            a = Speccon1dVRW(reader + "\n" +
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

    #vacuum_vs_depth = PolyLine([0,1], [1,1])
    #vacuum_vs_time = PolyLine([0,0.0,8], [0,-100,-100])

    dTw=1000
    kwref=1
    kw = PolyLine([0,1], [1,1])

    top_vs_time = PolyLine([0,0.0,8], [0,-100,-100])

    avg_ppress_z_pairs = [[0,1]]
    settlement_z_pairs = [[0,1]]

    tvals = np.%s

    """ % (repr(t)))

    avp = 100 * hansbo_avp - 100
    settle = 100 - 100 * hansbo_avp

    for impl in ["vectorized"]:
        for dT in [0.1,1,10]:
            a = Speccon1dVRW(reader + "\n" +
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

    dTh=1
    dTw=0
    khref = 1
    etref = 1
    kwref=1
    kw = PolyLine([0,1], [1,1])
    kh = PolyLine([0,1], [1,1])
    et = PolyLine([0,1], [1,1])

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
            a = Speccon1dVRW(reader + "\n" +
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

    dTh=1
    dTw=0
    khref = 1
    etref = 1
    kwref=1
    kw = PolyLine([0,1], [1,1])
    kh = PolyLine([0,1], [1,1])
    et = PolyLine([0,1], [1,1])

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
            a = Speccon1dVRW(reader + "\n" +
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


#


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

        dTw = 0.1
        kwref=1
        kw = PolyLine([0,1], [1,1])


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

        a = Speccon1dVRW(self.reader %
            {'drn': drn, 'use_actual': True, 'load_to_test': load_to_test})
        b = Speccon1dVRW(self.reader %
            {'drn': drn, 'use_actual': False, 'load_to_test': load_to_test})
        a.make_all()
        b.make_all()
        assert_allclose(a.por, b.por, atol=1e-2)
        assert_allclose(a.avp, b.avp, atol=1e-2)
        assert_allclose(a.set, b.set, atol=1e-2)

#    def test_vacuum(self):
#        """test vacuum"""
#        drn=0
#        load_to_test='vacuum'
#
#        a = Speccon1dVRW(self.reader %
#            {'drn': drn, 'use_actual': True, 'load_to_test': load_to_test})
#        b = Speccon1dVRW(self.reader %
#            {'drn': drn, 'use_actual': False, 'load_to_test': load_to_test})
#        a.make_all()
#        b.make_all()
#        assert_allclose(a.por, b.por, atol=1e-2)
#        assert_allclose(a.avp, b.avp, atol=1e-2)
#        assert_allclose(a.set, b.set, atol=1e-2)

    def test_top(self):
        """test top"""
        drn=0
        load_to_test='top'

        a = Speccon1dVRW(self.reader %
            {'drn': drn, 'use_actual': True, 'load_to_test': load_to_test})
        b = Speccon1dVRW(self.reader %
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

        a = Speccon1dVRW(self.reader %
            {'drn': drn, 'use_actual': True, 'load_to_test': load_to_test})
        b = Speccon1dVRW(self.reader %
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

        a = Speccon1dVRW(self.reader %
            {'drn': drn, 'use_actual': True, 'load_to_test': load_to_test})
        b = Speccon1dVRW(self.reader %
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
#        a = Speccon1dVRW(self.reader %
#            {'drn': drn, 'use_actual': True, 'load_to_test': load_to_test})
#        b = Speccon1dVRW(self.reader %
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

        a = Speccon1dVRW(self.reader %
            {'drn': drn, 'use_actual': True, 'load_to_test': load_to_test})
        b = Speccon1dVRW(self.reader %
            {'drn': drn, 'use_actual': False, 'load_to_test': load_to_test})
        a.make_all()
        b.make_all()
        assert_allclose(a.por, b.por, atol=1e-2)
        assert_allclose(a.avp, b.avp, atol=1e-2)
        assert_allclose(a.set, b.set, atol=1e-2)


def test_nogamiandli2003_lam_5():
    """test for nogami and li 2003 lambda = 5
     nogami and li use rigorous formulation, speccon uses equal strain so

     expect differences

    """


    t = np.array([ 0.01      ,  0.01603286,  0.02570525,  0.04121285,  0.06607597,
        0.1, 0.10593866,  0.16984993,  0.27231794,  0.4, 0.43660343,  0.7 ])

    z = np.array(
      [ 0.    ,  0.0625,  0.125 ,  0.1875,  0.25  ,  0.3125,  0.375 ,
        0.4375,  0.5   ,  0.5625,  0.625 ,  0.6875,  0.75  ,  0.8125,
        0.875 ,  0.9375,  1.    ,  1.025 ,  1.05  ,  1.075 ,  1.1   ,
        1.1625,  1.225 ,  1.2875,  1.35  ,  1.4125,  1.475 ,  1.5375,
        1.6   ,  1.6625,  1.725 ,  1.7875,  1.85  ,  1.9125,  1.975 ,
        2.0375,  2.1   ,  2.125 ,  2.15  ,  2.175 ,  2.2   ,  2.2625,
        2.325 ,  2.3875,  2.45  ,  2.5125,  2.575 ,  2.6375,  2.6999   ])




    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np

    ################################################
    #nogami and li parameters
    surcharge_vs_time = PolyLine([0,0,10], [0,100,100])
    hs=0.05
    h = np.array([1, hs, hs, 1, hs, hs, 0.5])
    lam = 5
    kv = np.array([1,lam/hs, lam/hs, 1, lam/hs, lam/hs, 1])
    mv = np.array([1.0, 1, 1, 1, 1, 1, 1])
    kh = kv

    r0 = 0.05
    r1 = 20 * r0
    #z = layer_coords(h, 45,2)

    bctop = 0
    bcbot = 1
    nv = 15
    nh = 5

    tpor = np.array([0.01,0.1, 0.4])

    z = np.array(
      [ 0.    ,  0.0625,  0.125 ,  0.1875,  0.25  ,  0.3125,  0.375 ,
        0.4375,  0.5   ,  0.5625,  0.625 ,  0.6875,  0.75  ,  0.8125,
        0.875 ,  0.9375,  1.    ,  1.025 ,  1.05  ,  1.075 ,  1.1   ,
        1.1625,  1.225 ,  1.2875,  1.35  ,  1.4125,  1.475 ,  1.5375,
        1.6   ,  1.6625,  1.725 ,  1.7875,  1.85  ,  1.9125,  1.975 ,
        2.0375,  2.1   ,  2.125 ,  2.15  ,  2.175 ,  2.2   ,  2.2625,
        2.325 ,  2.3875,  2.45  ,  2.5125,  2.575 ,  2.6375,  2.6999   ])


    t = np.array(
       [ 0.01      ,  0.01603286,  0.02570525,  0.04121285,  0.06607597,
        0.1, 0.10593866,  0.16984993,  0.27231794,  0.4, 0.43660343,  0.7 ])

    max_iter=20000
    vertical_roots_x0 = 2.2
    vertical_roots_dx = 1e-3
    vertical_roots_p = 1.01
    ################################################

    z2 = np.cumsum(h)
    z1 = z2-h
    H = np.sum(h)

    z1/=H
    z2/=H

    kv = PolyLine(z1, z2, kv, kv)
    mv = PolyLine(z1, z2, mv, mv)
    kh = kv


    drn = 1
    neig=50

    mvref=1.0

    surcharge_vs_depth = mv

    #rw=0.05, re = 20*rw = 1.0, n=20, no smear zone
    #Therfore muI=2.253865374, eta = 2/mu/re**2 = 0.887364446
    etref = 0.887364446
    et = PolyLine(z1, z2, np.ones_like(z1), np.ones_like(z1))

    dTv = 1/H**2
    dTh = etref

    dTw=1000
    kwref=1
    kw = PolyLine([0,1], [1,1])
    ppress_z = np.array(
       [ 0.    ,  0.0625,  0.125 ,  0.1875,  0.25  ,  0.3125,  0.375 ,
        0.4375,  0.5   ,  0.5625,  0.625 ,  0.6875,  0.75  ,  0.8125,
        0.875 ,  0.9375,  1.    ,  1.025 ,  1.05  ,  1.075 ,  1.1   ,
        1.1625,  1.225 ,  1.2875,  1.35  ,  1.4125,  1.475 ,  1.5375,
        1.6   ,  1.6625,  1.725 ,  1.7875,  1.85  ,  1.9125,  1.975 ,
        2.0375,  2.1   ,  2.125 ,  2.15  ,  2.175 ,  2.2   ,  2.2625,
        2.325 ,  2.3875,  2.45  ,  2.5125,  2.575 ,  2.6375,  2.6999   ])

    ppress_z/=H

    avg_ppress_z_pairs = [[0,1]]
    settlement_z_pairs = [[0,1]]

    tvals = np.array(
      [ 0.01      ,  0.01603286,  0.02570525,  0.04121285,  0.06607597,
        0.1, 0.10593866,  0.16984993,  0.27231794,  0.4, 0.43660343,  0.7 ])

    ppress_z_tval_indexes = [0, 5, 9] #0.01, 0.1, 0.4
    """)

    por = np.array(
      [[  0.        ,   0.        ,   0.        ],
       [ 32.94991031,   9.30846577,   1.06072593],
       [ 60.50469133,  18.35862945,   2.09682932],
       [ 79.52571707,  26.90168552,   3.08429341],
       [ 90.06180426,  34.70710083,   4.00029715],
       [ 94.55282914,  41.57006481,   4.82377368],
       [ 96.04098984,  47.31721713,   5.5359229 ],
       [ 96.63719346,  51.81051011,   6.12066479],
       [ 97.07580625,  54.94931228,   6.56502187],
       [ 97.27065296,  56.67105285,   6.85942107],
       [ 97.1057093 ,  56.95080968,   6.99790774],
       [ 96.72063876,  55.80024278,   6.9782666 ],
       [ 96.14739652,  53.26618066,   6.8020474 ],
       [ 94.76324343,  49.42901062,   6.47449533],
       [ 91.15886034,  44.40084879,   6.00438888],
       [ 83.63522467,  38.3233171 ,   5.40379048],
       [ 71.02182882,  31.36466998,   4.68771724],
       [ 70.97596237,  31.34317255,   4.68582079],
       [ 70.96055716,  31.33737197,   4.68627617],
       [ 70.9756061 ,  31.34726572,   4.68908363],
       [ 71.02113267,  31.37285944,   4.6942446 ],
       [ 83.61246044,  39.36666485,   6.22480907],
       [ 91.38126902,  46.54039404,   7.6281786 ],
       [ 95.44742589,  52.78332249,   8.87779047],
       [ 97.18636227,  58.00520529,   9.9502025 ],
       [ 97.77106952,  62.13581189,  10.8255135 ],
       [ 97.95566599,  65.12364104,  11.48771358],
       [ 98.07498151,  66.93424716,  11.92495889],
       [ 98.17356755,  67.54865883,  12.12976682],
       [ 98.18150818,  66.96232237,  12.09912898],
       [ 98.05090064,  65.18486623,  11.83454088],
       [ 97.75273273,  62.24078238,  11.34194855],
       [ 97.08501041,  58.17089632,  10.63161376],
       [ 95.39424089,  53.03429149,   9.71790087],
       [ 91.45890546,  46.91020634,   8.61898956],
       [ 83.7556734 ,  39.89936258,   7.35651899],
       [ 71.08436042,  32.12422375,   5.95517002],
       [ 71.03838507,  32.09989551,   5.95082665],
       [ 71.0230861 ,  32.09179329,   5.94949046],
       [ 71.03844883,  32.09991289,   5.95116079],
       [ 71.08449479,  32.12425873,   5.9558385 ],
       [ 83.78763806,  39.90383793,   7.44085772],
       [ 91.57940299,  46.91911482,   8.78729168],
       [ 95.64743846,  53.04766406,   9.97071357],
       [ 97.44492495,  58.18889925,  10.9696657 ],
       [ 98.14320956,  62.26387924,  11.76604301],
       [ 98.4208958 ,  65.21406514,  12.34541665],
       [ 98.54986983,  66.99953518,  12.69729194],
       [ 98.59179084,  67.59720313,  12.81529574]])
    avp = np.array(
      [[ 88.32365031,  84.02918765,  78.13336593,  70.18369459,
         59.71491547,  48.2730443 ,  46.5276473 ,  31.41366172,
         16.85941031,   7.82517078,   6.2871533 ,   1.31568525]])
    settle = np.array(
      [[  31.52614416,   43.12119333,   59.03991198,   80.50402461,
         108.76972824,  139.66278038,  144.37535228,  185.18311334,
         224.47959217,  248.87203889,  253.02468609,  266.44764984]])

    for impl in ["vectorized"]:
        for dT in [10]:
            a = Speccon1dVRW(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()

#            plt.clf()
#            plt.figure()
#            plt.plot(por, z,'b-*', label='expected')
#            plt.plot(a.por, z, 'r-+', label='calculated')
#            plt.legend()
##
##
#            plt.figure()
#            plt.plot(t, settle[0],'b-*', label='expected')
#            plt.plot(t, a.set[0], 'r-+', label='calculated')
#            plt.legend()
#            plt.figure()
#            plt.plot(t, avp[0],'b-*',  label='expected')
#            plt.plot(t, a.avp[0], 'r-+', label='calculated')
#            plt.legend()
#            plt.show()

            assert_allclose(a.por, por, atol=5,
                            err_msg = ("Fail. test_nogamiandli2003_lam_5, por, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.avp, avp, atol=2,
                            err_msg = ("Fail. test_nogamiandli2003_lam_5, avp, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=4,
                            err_msg = ("Fail. test_nogamiandli2003_lam_5, set, "
                                "implementation='%s', dT=%s" % (impl, dT)))

#
def test_nogamiandli2003_lam_100():
    """test for nogami and li 2003 lambda = 100
     nogami and li use rigorous formulation, speccon uses equal strain so

     expect differences

    """


    t = np.array([ 0.01      ,  0.01603286,  0.02570525,  0.04121285,  0.06607597,
        0.1, 0.10593866,  0.16984993,  0.27231794,  0.4, 0.43660343,  0.7 ])

    z = np.array(
      [ 0.    ,  0.0625,  0.125 ,  0.1875,  0.25  ,  0.3125,  0.375 ,
        0.4375,  0.5   ,  0.5625,  0.625 ,  0.6875,  0.75  ,  0.8125,
        0.875 ,  0.9375,  1.    ,  1.025 ,  1.05  ,  1.075 ,  1.1   ,
        1.1625,  1.225 ,  1.2875,  1.35  ,  1.4125,  1.475 ,  1.5375,
        1.6   ,  1.6625,  1.725 ,  1.7875,  1.85  ,  1.9125,  1.975 ,
        2.0375,  2.1   ,  2.125 ,  2.15  ,  2.175 ,  2.2   ,  2.2625,
        2.325 ,  2.3875,  2.45  ,  2.5125,  2.575 ,  2.6375,  2.6999   ])




    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np

    ################################################
    #nogami and li parameters
    surcharge_vs_time = PolyLine([0,0,10], [0,100,100])
    hs=0.05
    h = np.array([1, hs, hs, 1, hs, hs, 0.5])
    lam = 100
    kv = np.array([1,lam/hs, lam/hs, 1, lam/hs, lam/hs, 1])
    mv = np.array([1.0, 1, 1, 1, 1, 1, 1])
    kh = kv

    r0 = 0.05
    r1 = 20 * r0
    #z = layer_coords(h, 45,2)

    bctop = 0
    bcbot = 1
    nv = 15
    nh = 5

    tpor = np.array([0.01,0.1, 0.4])

    z = np.array(
      [ 0.    ,  0.0625,  0.125 ,  0.1875,  0.25  ,  0.3125,  0.375 ,
        0.4375,  0.5   ,  0.5625,  0.625 ,  0.6875,  0.75  ,  0.8125,
        0.875 ,  0.9375,  1.    ,  1.025 ,  1.05  ,  1.075 ,  1.1   ,
        1.1625,  1.225 ,  1.2875,  1.35  ,  1.4125,  1.475 ,  1.5375,
        1.6   ,  1.6625,  1.725 ,  1.7875,  1.85  ,  1.9125,  1.975 ,
        2.0375,  2.1   ,  2.125 ,  2.15  ,  2.175 ,  2.2   ,  2.2625,
        2.325 ,  2.3875,  2.45  ,  2.5125,  2.575 ,  2.6375,  2.6999   ])


    t = np.array(
       [ 0.01      ,  0.01603286,  0.02570525,  0.04121285,  0.06607597,
        0.1, 0.10593866,  0.16984993,  0.27231794,  0.4, 0.43660343,  0.7 ])

    max_iter=20000
    vertical_roots_x0 = 2.2
    vertical_roots_dx = 1e-3
    vertical_roots_p = 1.01
    ################################################

    z2 = np.cumsum(h)
    z1 = z2-h
    H = np.sum(h)

    z1/=H
    z2/=H

    kv = PolyLine(z1, z2, kv, kv)
    mv = PolyLine(z1, z2, mv, mv)
    kh = kv


    drn = 1
    neig=80

    mvref=1.0

    surcharge_vs_depth = mv

    #rw=0.05, re = 20*rw = 1.0, n=20, no smear zone
    #Therfore muI=2.253865374, eta = 2/mu/re**2 = 0.887364446
    etref = 0.887364446
    et = PolyLine(z1, z2, np.ones_like(z1), np.ones_like(z1))

    dTv = 1/H**2
    dTh = etref

    dTw=1000
    kwref=1
    kw = PolyLine([0,1], [1,1])

    ppress_z = np.array(
       [ 0.    ,  0.0625,  0.125 ,  0.1875,  0.25  ,  0.3125,  0.375 ,
        0.4375,  0.5   ,  0.5625,  0.625 ,  0.6875,  0.75  ,  0.8125,
        0.875 ,  0.9375,  1.    ,  1.025 ,  1.05  ,  1.075 ,  1.1   ,
        1.1625,  1.225 ,  1.2875,  1.35  ,  1.4125,  1.475 ,  1.5375,
        1.6   ,  1.6625,  1.725 ,  1.7875,  1.85  ,  1.9125,  1.975 ,
        2.0375,  2.1   ,  2.125 ,  2.15  ,  2.175 ,  2.2   ,  2.2625,
        2.325 ,  2.3875,  2.45  ,  2.5125,  2.575 ,  2.6375,  2.6999   ])

    ppress_z/=H

    avg_ppress_z_pairs = [[0,1]]
    settlement_z_pairs = [[0,1]]

    tvals = np.array(
      [ 0.01      ,  0.01603286,  0.02570525,  0.04121285,  0.06607597,
        0.1, 0.10593866,  0.16984993,  0.27231794,  0.4, 0.43660343,  0.7 ])

    ppress_z_tval_indexes = [0, 5, 9] #0.01, 0.1, 0.4
    """)

    por = np.array(
      [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
       [  3.30085802e+01,   8.39070407e+00,   3.58707855e-01],
       [  6.05040116e+01,   1.64657476e+01,   7.03954600e-01],
       [  7.93733840e+01,   2.39216010e+01,   1.02278441e+00],
       [  8.98276000e+01,   3.04784367e+01,   1.30323304e+00],
       [  9.44368466e+01,   3.58906380e+01,   1.53477694e+00],
       [  9.61739263e+01,   3.99558431e+01,   1.70872820e+00],
       [  9.69096134e+01,   4.25222425e+01,   1.81856068e+00],
       [  9.72060228e+01,   4.34939559e+01,   1.86015483e+00],
       [  9.70563632e+01,   4.28344044e+01,   1.83195234e+00],
       [  9.64675854e+01,   4.05676349e+01,   1.73501446e+00],
       [  9.51338553e+01,   3.67775792e+01,   1.57298210e+00],
       [  9.15678072e+01,   3.16052348e+01,   1.35193903e+00],
       [  8.28664682e+01,   2.52437753e+01,   1.08018340e+00],
       [  6.59357468e+01,   1.79316479e+01,   7.67916132e-01],
       [  3.96855349e+01,   9.94380297e+00,   4.26857907e-01],
       [  6.59655749e+00,   1.58132211e+00,   6.98091328e-02],
       [  6.59134099e+00,   1.58005846e+00,   6.97557684e-02],
       [  6.58961224e+00,   1.57964519e+00,   6.97399004e-02],
       [  6.59137041e+00,   1.58008208e+00,   6.97615203e-02],
       [  6.59661708e+00,   1.58136940e+00,   6.98206398e-02],
       [  3.98547617e+01,   1.00628681e+01,   4.55528866e-01],
       [  6.63377268e+01,   1.81728261e+01,   8.24478652e-01],
       [  8.35306351e+01,   2.56128437e+01,   1.16311926e+00],
       [  9.24242085e+01,   3.21098488e+01,   1.45901394e+00],
       [  9.60705684e+01,   3.74261238e+01,   1.70129661e+00],
       [  9.74392519e+01,   4.13677993e+01,   1.88107072e+00],
       [  9.81005003e+01,   4.37915763e+01,   1.99173594e+00],
       [  9.83479591e+01,   4.46095565e+01,   2.02923036e+00],
       [  9.81079539e+01,   4.37921401e+01,   1.99217957e+00],
       [  9.74436874e+01,   4.13689901e+01,   1.88194701e+00],
       [  9.60657415e+01,   3.74280676e+01,   1.70258384e+00],
       [  9.24170918e+01,   3.21127333e+01,   1.46068005e+00],
       [  8.35319659e+01,   2.56169153e+01,   1.16512240e+00],
       [  6.63469036e+01,   1.81783823e+01,   8.26768047e-01],
       [  3.98602199e+01,   1.00702443e+01,   4.58045763e-01],
       [  6.59182174e+00,   1.59091774e+00,   7.24995316e-02],
       [  6.58658179e+00,   1.58963382e+00,   7.24411588e-02],
       [  6.58484730e+00,   1.58920618e+00,   7.24217283e-02],
       [  6.58661723e+00,   1.58963460e+00,   7.24412296e-02],
       [  6.59189335e+00,   1.59091932e+00,   7.24996732e-02],
       [  4.00379871e+01,   1.00742370e+01,   4.58399695e-01],
       [  6.66669810e+01,   1.81862162e+01,   8.27472042e-01],
       [  8.39318565e+01,   2.56282977e+01,   1.16616892e+00],
       [  9.28276558e+01,   3.21272424e+01,   1.46205756e+00],
       [  9.64383871e+01,   3.74451724e+01,   1.70427658e+00],
       [  9.77664761e+01,   4.13880757e+01,   1.88393471e+00],
       [  9.83983815e+01,   4.38125372e+01,   1.99443713e+00],
       [  9.86319025e+01,   4.46305720e+01,   2.03172745e+00]])
    avp = np.array(
      [[  7.25979052e+01,   6.65166314e+01,   5.89096834e+01,
          4.94554633e+01,   3.79564622e+01,   2.66323138e+01,
          2.50358034e+01,   1.28862133e+01,   4.44927613e+00,
          1.18311566e+00,   8.09339892e-01,   5.26895921e-02]])
    settle = np.array(
      [[  73.98565603,   90.40509513,  110.94385476,  136.47024905,
         167.51755212,  198.09275262,  202.40333078,  235.20722417,
         257.98695445,  266.80558772,  267.81478229,  269.8577381 ]])

    for impl in ["vectorized"]:
        for dT in [10]:
            a = Speccon1dVRW(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()

#            plt.clf()
#            plt.figure()
#            plt.plot(por, z,'b-*', label='expected')
#            plt.plot(a.por, z, 'r-+', label='calculated')
#            plt.legend()
##
##
#            plt.figure()
#            plt.plot(t, settle[0],'b-*', label='expected')
#            plt.plot(t, a.set[0], 'r-+', label='calculated')
#            plt.legend()
#            plt.figure()
#            plt.plot(t, avp[0],'b-*',  label='expected')
#            plt.plot(t, a.avp[0], 'r-+', label='calculated')
#            plt.legend()
#            plt.show()

            assert_allclose(a.por, por, atol=13,
                            err_msg = ("Fail. test_nogamiandli2003_lam_100, por, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.avp, avp, atol=5,
                            err_msg = ("Fail. test_nogamiandli2003_lam_100, avp, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=7,
                            err_msg = ("Fail. test_nogamiandli2003_lam_100, set, "
                                "implementation='%s', dT=%s" % (impl, dT)))

def test_zhuandyin2012_drn0_kv_linear_mv_const():
    """test for zhu and yin 2012

     vertical drainage, depth dependent properties, instant load
     generally:
     mv = mv0*(1+alpha*z/H)**q
     kv = kv0* (1+alpha*z/H)**p
     for this case
     mv=mv0
     PTPB
    """

    t = np.array([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
        11.,  12.,  13.,  14.,  15.])

    z = np.array([ 0.        ,  0.13157895,  0.26315789,  0.39473684,  0.52631579,
        0.65789474,  0.78947368,  0.92105263,  1.05263158,  1.18421053,
        1.31578947,  1.44736842,  1.57894737,  1.71052632,  1.84210526,
        1.97368421,  2.10526316,  2.23684211,  2.36842105,  2.5       ])

    tpor=t[np.array([2,4,9,13])]




    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np


    ####################################
    #zhuandyin2012 properties
    #ui = 100
    #drn = 0
    #nterms = 50
    #mv0 = 1.2
    #kv0 = 1.6
    #H = 2.5
    #alpha = 1
    #q = 0
    #p = 1
    #z = np.linspace(0,H,20)
    #t = np.linspace(0,15,16)
    #tpor=t[np.array([2,4,9,13])]
    #plot_eigs=False
    #
    #por, doc, settle = zhuandyin2012(
    #    z=z, t=t, alpha=alpha, p=p, q=q, drn=drn, tpor=tpor, H = H, kv0 = kv0, mv0 = mv0, gamw = 10,
    #        ui = 100, nterms = nterms, plot_eigs=plot_eigs)

    ####################################

    neig=40

    H = 2.5
    drn = 0

    mvref = 1.2
    kvref = 1.6 / 10

    kv = PolyLine([0,1], [1,2])
    mv = PolyLine([0,1], [1,1])

    dTv = kvref/mvref/H**2


    dTh=0.1
    dTw=0
    khref = 1
    etref = 1
    kwref=1
    kw = PolyLine([0,1], [1,1])
    kh = PolyLine([0,1], [1,1])
    et = PolyLine([0,1], [1,1])


    surcharge_vs_time = PolyLine([0,0,10], [0,100,100])
    surcharge_vs_depth = PolyLine([0,1], [1,1])


    ppress_z = np.array(
        [ 0.        ,  0.13157895,  0.26315789,  0.39473684,  0.52631579,
        0.65789474,  0.78947368,  0.92105263,  1.05263158,  1.18421053,
        1.31578947,  1.44736842,  1.57894737,  1.71052632,  1.84210526,
        1.97368421,  2.10526316,  2.23684211,  2.36842105,  2.5       ])

    ppress_z/=H

    settlement_z_pairs = [[0,1]]

    tvals = np.array(
      [  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
        11.,  12.,  13.,  14.,  15.])

    ppress_z_tval_indexes = [2,4,9,13]
    """)

    por = np.array(
     [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00],
       [  1.45637289e+01,   7.93407071e+00,   1.72266349e+00,
          5.06825357e-01],
       [  2.78786258e+01,   1.51937492e+01,   3.29853310e+00,
          9.70461303e-01],
       [  3.95929298e+01,   2.15886617e+01,   4.68603058e+00,
          1.37867434e+00],
       [  4.94637465e+01,   2.69822591e+01,   5.85540859e+00,
          1.72271221e+00],
       [  5.73451924e+01,   3.12869817e+01,   6.78769539e+00,
          1.99699341e+00],
       [  6.31732721e+01,   3.44589624e+01,   7.47352321e+00,
          2.19876233e+00],
       [  6.69501080e+01,   3.64925266e+01,   7.91190221e+00,
          2.32772855e+00],
       [  6.87296115e+01,   3.74146824e+01,   8.10898721e+00,
          2.38570374e+00],
       [  6.86058287e+01,   3.72797427e+01,   8.07687246e+00,
          2.37624707e+00],
       [  6.67043109e+01,   3.61641803e+01,   7.83244041e+00,
          2.30432633e+00],
       [  6.31761472e+01,   3.41617884e+01,   7.39628296e+00,
          2.17600049e+00],
       [  5.81938563e+01,   3.13791946e+01,   6.79170747e+00,
          1.99812723e+00],
       [  5.19481922e+01,   2.79317585e+01,   6.04383529e+00,
          1.77809769e+00],
       [  4.46450073e+01,   2.39398742e+01,   5.17879661e+00,
          1.52359956e+00],
       [  3.65015772e+01,   1.95256817e+01,   4.22302247e+00,
          1.24240884e+00],
       [  2.77421116e+01,   1.48101921e+01,   3.20263266e+00,
          9.42209774e-01],
       [  1.85924877e+01,   9.91081790e+00,   2.14291617e+00,
          6.30442069e-01],
       [  9.27449220e+00,   4.93929752e+00,   1.06789988e+00,
          3.14174014e-01],
       [  4.17631424e-12,   2.26493642e-12,   4.90759409e-13,
          1.44383480e-13]])
#
#    avp = np.array(
#      [[  7.25979052e+01,   6.65166314e+01,   5.89096834e+01,
#          4.94554633e+01,   3.79564622e+01,   2.66323138e+01,
#          2.50358034e+01,   1.28862133e+01,   4.44927613e+00,
#          1.18311566e+00,   8.09339892e-01,   5.26895921e-02]])
    settle = np.array(
        [[   2.43103297,  119.32510052,  168.14191428,  202.97841422,
        228.55495738,  247.38377244,  261.24954126,  271.46110823,
        278.98165371,  284.52037457,  288.59953574,  291.60376277,
        293.81632162,  295.44583147,  296.64593629,  297.52979203]])

    for impl in ["vectorized"]:
        for dT in [10]:
            a = Speccon1dVRW(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()

#            plt.clf()
#            plt.figure()
#            plt.plot(por, z,'b-*', label='expected')
#            plt.plot(a.por, z, 'r-+', label='calculated')
#            plt.legend()
##
##
#            plt.figure()
#            plt.plot(t, settle[0],'b-*', label='expected')
#            plt.plot(t, a.set[0], 'r-+', label='calculated')
#            legend=plt.legend()
#            legend.draggable()
##            plt.legend.DraggableLegend()
##            plt.figure()
##            plt.plot(t, avp[0],'b-*',  label='expected')
##            plt.plot(t, a.avp[0], 'r-+', label='calculated')
##            plt.legend()
#            plt.show()

            assert_allclose(a.por, por, atol=1e-2,
                            err_msg = ("Fail. test_zhuandyin2012_drn0_kv_linear_mv_const, por, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_zhuandyin2012_drn0_kv_linear_mv_const, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=1,
                            err_msg = ("Fail. test_zhuandyin2012_drn0_kv_linear_mv_const, set, "
                                "implementation='%s', dT=%s" % (impl, dT)))

def test_zhuandyin2012_drn1_kv_linear_mv_const():
    """test for zhu and yin 2012

     vertical drainage, depth dependent properties, instant load
     generally:
     mv = mv0*(1+alpha*z/H)**q
     kv = kv0* (1+alpha*z/H)**p
     for this case
     mv=mv0
     PTIB

    """

    t = np.array([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
        11.,  12.,  13.,  14.,  15.])

    z = np.array([ 0.        ,  0.13157895,  0.26315789,  0.39473684,  0.52631579,
        0.65789474,  0.78947368,  0.92105263,  1.05263158,  1.18421053,
        1.31578947,  1.44736842,  1.57894737,  1.71052632,  1.84210526,
        1.97368421,  2.10526316,  2.23684211,  2.36842105,  2.5       ])

    tpor=t[np.array([2,4,9,13])]




    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np


    ####################################
    #zhuandyin2012 properties
    #ui = 100
    #drn = 1
    #nterms = 50
    #mv0 = 1.2
    #kv0 = 1.6
    #H = 2.5
    #alpha = 1
    #q = 0
    #p = 1
    #z = np.linspace(0,H,20)
    #t = np.linspace(0,15,16)
    #tpor=t[np.array([2,4,9,13])]
    #plot_eigs=False
    #
    #por, doc, settle = zhuandyin2012(
    #    z=z, t=t, alpha=alpha, p=p, q=q, drn=drn, tpor=tpor, H = H, kv0 = kv0, mv0 = mv0, gamw = 10,
    #        ui = 100, nterms = nterms, plot_eigs=plot_eigs)

    ####################################

    neig=40

    H = 2.5
    drn = 1

    mvref = 1.2
    kvref = 1.6 / 10

    kv = PolyLine([0,1], [1,2])
    mv = PolyLine([0,1], [1,1])

    dTv = kvref/mvref/H**2

    dTh=0.1
    dTw=0
    khref = 1
    etref = 1
    kwref=1
    kw = PolyLine([0,1], [1,1])
    kh = PolyLine([0,1], [1,1])
    et = PolyLine([0,1], [1,1])

    surcharge_vs_time = PolyLine([0,0,10], [0,100,100])
    surcharge_vs_depth = PolyLine([0,1], [1,1])


    ppress_z = np.array(
        [ 0.        ,  0.13157895,  0.26315789,  0.39473684,  0.52631579,
        0.65789474,  0.78947368,  0.92105263,  1.05263158,  1.18421053,
        1.31578947,  1.44736842,  1.57894737,  1.71052632,  1.84210526,
        1.97368421,  2.10526316,  2.23684211,  2.36842105,  2.5       ])

    ppress_z/=H

    settlement_z_pairs = [[0,1]]

    tvals = np.array(
      [  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
        11.,  12.,  13.,  14.,  15.])

    ppress_z_tval_indexes = [2,4,9,13]
    """)

    por = np.array(
      [[  0.        ,   0.        ,   0.        ,   0.        ],
       [ 15.18444088,  11.10894698,   7.40806725,   5.65866747],
       [ 29.21825868,  21.52741854,  14.393854  ,  10.99614641],
       [ 41.86741966,  31.18979916,  20.94166598,  16.00146099],
       [ 53.00073922,  40.05518511,  27.04142859,  20.66754327],
       [ 62.58176447,  48.10531202,  32.68786111,  24.99062132],
       [ 70.65301836,  55.34166198,  37.87975403,  28.96970795],
       [ 77.31639526,  61.78209413,  42.61933358,  32.60616995],
       [ 82.71294612,  67.45729525,  46.91170217,  35.90336248],
       [ 87.00441979,  72.40728918,  50.76434639,  38.86631703],
       [ 90.35798578,  76.67818396,  54.18670618,  41.50147367],
       [ 92.93472805,  80.3192787 ,  57.18979959,  43.81645046],
       [ 94.88186164,  83.38060239,  59.78589829,  45.81984409],
       [ 96.32821071,  85.9109158 ,  61.98824926,  47.52105729],
       [ 97.38227721,  87.95617524,  63.81083812,  48.93014883],
       [ 98.1321791 ,  89.55843377,  65.26819006,  50.05770314],
       [ 98.64679433,  90.75513955,  66.37520386,  50.91471679],
       [ 98.97756401,  91.57878173,  67.14701531,  51.51249953],
       [ 99.16054817,  92.05682986,  67.59888601,  51.862588  ],
       [ 99.21846412,  92.21191229,  67.74611382,  51.97667046]])
#
#    avp = np.array(
#      [[  7.25979052e+01,   6.65166314e+01,   5.89096834e+01,
#          4.94554633e+01,   3.79564622e+01,   2.66323138e+01,
#          2.50358034e+01,   1.28862133e+01,   4.44927613e+00,
#          1.18311566e+00,   8.09339892e-01,   5.26895921e-02]])
    settle = np.array(
    [[   1.00721992,   51.02212219,   73.064841  ,   90.33221466,
        105.12195388,  118.31183726,  130.33109045,  141.41484463,
        151.70343029,  161.28848225,  170.23575019,  178.59664384,
        186.4141468 ,  193.72588488,  200.56574999,  206.96478826]])

    for impl in ["vectorized"]:
        for dT in [10]:
            a = Speccon1dVRW(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()

#            plt.clf()
#            plt.figure()
#            plt.plot(por, z,'b-*', label='expected')
#            plt.plot(a.por, z, 'r-+', label='calculated')
#            plt.legend()
##
##
#            plt.figure()
#            plt.plot(t, settle[0],'b-*', label='expected')
#            plt.plot(t, a.set[0], 'r-+', label='calculated')
#            legend=plt.legend()
#            legend.draggable()
##            plt.legend.DraggableLegend()
##            plt.figure()
##            plt.plot(t, avp[0],'b-*',  label='expected')
##            plt.plot(t, a.avp[0], 'r-+', label='calculated')
##            plt.legend()
#            plt.show()

            assert_allclose(a.por, por, atol=1e-2,
                            err_msg = ("Fail. test_zhuandyin2012_drn1_kv_linear_mv_const, por, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_zhuandyin2012_drn1_kv_linear_mv_const, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=1,
                            err_msg = ("Fail. test_zhuandyin2012_drn1_kv_linear_mv_const, set, "
                                "implementation='%s', dT=%s" % (impl, dT)))

def test_zhuandyin2012_drn0_kv_const_mv_linear():
    """test for zhu and yin 2012

     vertical drainage, depth dependent properties, instant load
     generally:
     mv = mv0*(1+alpha*z/H)**q
     kv = kv0* (1+alpha*z/H)**p
     for this case
     kv=kv0
     PTPB
    """

    t = np.array([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
        11.,  12.,  13.,  14.,  15.])

    z = np.array([ 0.        ,  0.13157895,  0.26315789,  0.39473684,  0.52631579,
        0.65789474,  0.78947368,  0.92105263,  1.05263158,  1.18421053,
        1.31578947,  1.44736842,  1.57894737,  1.71052632,  1.84210526,
        1.97368421,  2.10526316,  2.23684211,  2.36842105,  2.5       ])

    tpor=t[np.array([2,4,9,13])]




    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np


    ####################################
    #zhuandyin2012 properties
    #ui = 100
    #drn = 0
    #nterms = 50
    #mv0 = 1.2
    #kv0 = 1.6
    #H = 2.5
    #alpha = 1
    #q = 1
    #p = 0
    #z = np.linspace(0,H,20)
    #t = np.linspace(0,15,16)
    #tpor=t[np.array([2,4,9,13])]
    #plot_eigs=False
    #
    #por, doc, settle = zhuandyin2012(
    #    z=z, t=t, alpha=alpha, p=p, q=q, drn=drn, tpor=tpor, H = H, kv0 = kv0, mv0 = mv0, gamw = 10,
    #        ui = 100, nterms = nterms, plot_eigs=plot_eigs)

    ####################################

    neig=40

    H = 2.5
    drn = 0

    mvref = 1.2
    kvref = 1.6 / 10

    kv = PolyLine([0,1], [1,1])
    mv = PolyLine([0,1], [1,2])

    dTv = kvref/mvref/H**2

    dTh=0.1
    dTw=0
    khref = 1
    etref = 1
    kwref=1
    kw = PolyLine([0,1], [1,1])
    kh = PolyLine([0,1], [1,1])
    et = PolyLine([0,1], [1,1])

    surcharge_vs_time = PolyLine([0,0,10], [0,100,100])
    surcharge_vs_depth = PolyLine([0,1], [1,1])


    ppress_z = np.array(
        [ 0.        ,  0.13157895,  0.26315789,  0.39473684,  0.52631579,
        0.65789474,  0.78947368,  0.92105263,  1.05263158,  1.18421053,
        1.31578947,  1.44736842,  1.57894737,  1.71052632,  1.84210526,
        1.97368421,  2.10526316,  2.23684211,  2.36842105,  2.5       ])

    ppress_z/=H

    settlement_z_pairs = [[0,1]]

    tvals = np.array(
      [  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
        11.,  12.,  13.,  14.,  15.])

    ppress_z_tval_indexes = [2,4,9,13]
    """)

    por = np.array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00],
       [  1.54526693e+01,   1.09324170e+01,   5.45432949e+00,
          3.12385435e+00],
       [  3.04146954e+01,   2.16470695e+01,   1.08043650e+01,
          6.18772293e+00],
       [  4.44006882e+01,   3.19132040e+01,   1.59384135e+01,
          9.12738341e+00],
       [  5.69796067e+01,   4.14931978e+01,   2.07391114e+01,
          1.18753937e+01],
       [  6.78211642e+01,   5.01510045e+01,   2.50865613e+01,
          1.43629200e+01],
       [  7.67238893e+01,   5.76602545e+01,   2.88619716e+01,
          1.65218535e+01],
       [  8.36163878e+01,   6.38112411e+01,   3.19517313e+01,
          1.82871712e+01],
       [  8.85292651e+01,   6.84163589e+01,   3.42518199e+01,
          1.95994813e+01],
       [  9.15424970e+01,   7.13140823e+01,   3.56724239e+01,
          2.04076724e+01],
       [  9.27199192e+01,   7.23721783e+01,   3.61425945e+01,
          2.06715704e+01],
       [  9.20474065e+01,   7.14913358e+01,   3.56147594e+01,
          2.03644897e+01],
       [  8.93932711e+01,   6.86105734e+01,   3.40688657e+01,
          1.94755543e+01],
       [  8.45078598e+01,   6.37154707e+01,   3.15159206e+01,
          1.80116555e+01],
       [  7.70734988e+01,   5.68494130e+01,   2.80006785e+01,
          1.59989093e+01],
       [  6.68048929e+01,   4.81267528e+01,   2.36032269e+01,
          1.34834831e+01],
       [  5.35839387e+01,   3.77453917e+01,   1.84392403e+01,
          1.05316710e+01],
       [  3.75948100e+01,   2.59952204e+01,   1.26587025e+01,
          7.22911840e+00],
       [  1.94124730e+01,   1.32586059e+01,   6.44295503e+00,
          3.67912598e+00],
       [ -3.45068593e-14,  -1.97857435e-14,  -8.69186348e-15,
         -4.90979514e-15]])
#
#    avp = np.array(
#      [[  7.25979052e+01,   6.65166314e+01,   5.89096834e+01,
#          4.94554633e+01,   3.79564622e+01,   2.66323138e+01,
#          2.50358034e+01,   1.28862133e+01,   4.44927613e+00,
#          1.18311566e+00,   8.09339892e-01,   5.26895921e-02]])
    settle = np.array(
    [[   3.57704735,  120.04921228,  170.08668445,  208.48781421,
        240.52536911,  268.00075818,  291.78168717,  312.42809025,
        330.37165232,  345.97183673,  359.53650775,  371.33191184,
        381.58908098,  390.50873519,  398.26534318,  405.01058755]])

    for impl in ["vectorized"]:
        for dT in [10]:
            a = Speccon1dVRW(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()

#            plt.clf()
#            plt.figure()
#            plt.plot(por, z,'b-*', label='expected')
#            plt.plot(a.por, z, 'r-+', label='calculated')
#            plt.legend()
##
##
#            plt.figure()
#            plt.plot(t, settle[0],'b-*', label='expected')
#            plt.plot(t, a.set[0], 'r-+', label='calculated')
#            legend=plt.legend()
#            legend.draggable()
##            plt.legend.DraggableLegend()
##            plt.figure()
##            plt.plot(t, avp[0],'b-*',  label='expected')
##            plt.plot(t, a.avp[0], 'r-+', label='calculated')
##            plt.legend()
#            plt.show()

            assert_allclose(a.por, por, atol=1e-2,
                            err_msg = ("Fail. test_zhuandyin2012_drn0_kv_const_mv_linear, por, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_zhuandyin2012_drn0_kv_const_mv_linear, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=1,
                            err_msg = ("Fail. test_zhuandyin2012_drn0_kv_const_mv_linear, set, "
                                "implementation='%s', dT=%s" % (impl, dT)))


def test_zhuandyin2012_drn1_kv_const_mv_linear():
    """test for zhu and yin 2012

     vertical drainage, depth dependent properties, instant load
     generally:
     mv = mv0*(1+alpha*z/H)**q
     kv = kv0* (1+alpha*z/H)**p
     for this case
     kv=kv0
     PTIB
    """

    t = np.array([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
        11.,  12.,  13.,  14.,  15.])

    z = np.array([ 0.        ,  0.13157895,  0.26315789,  0.39473684,  0.52631579,
        0.65789474,  0.78947368,  0.92105263,  1.05263158,  1.18421053,
        1.31578947,  1.44736842,  1.57894737,  1.71052632,  1.84210526,
        1.97368421,  2.10526316,  2.23684211,  2.36842105,  2.5       ])

    tpor=t[np.array([2,4,9,13])]




    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np


    ####################################
    #zhuandyin2012 properties
    #ui = 100
    #drn = 1
    #nterms = 50
    #mv0 = 1.2
    #kv0 = 1.6
    #H = 2.5
    #alpha = 1
    #q = 1
    #p = 0
    #z = np.linspace(0,H,20)
    #t = np.linspace(0,15,16)
    #tpor=t[np.array([2,4,9,13])]
    #plot_eigs=False
    #
    #por, doc, settle = zhuandyin2012(
    #    z=z, t=t, alpha=alpha, p=p, q=q, drn=drn, tpor=tpor, H = H, kv0 = kv0, mv0 = mv0, gamw = 10,
    #        ui = 100, nterms = nterms, plot_eigs=plot_eigs)

    ####################################

    neig=40

    H = 2.5
    drn = 1

    mvref = 1.2
    kvref = 1.6 / 10

    kv = PolyLine([0,1], [1,1])
    mv = PolyLine([0,1], [1,2])

    dTv = kvref/mvref/H**2

    dTh=0.1
    dTw=0
    khref = 1
    etref = 1
    kwref=1
    kw = PolyLine([0,1], [1,1])
    kh = PolyLine([0,1], [1,1])
    et = PolyLine([0,1], [1,1])

    surcharge_vs_time = PolyLine([0,0,10], [0,100,100])
    surcharge_vs_depth = PolyLine([0,1], [1,1])


    ppress_z = np.array(
        [ 0.        ,  0.13157895,  0.26315789,  0.39473684,  0.52631579,
        0.65789474,  0.78947368,  0.92105263,  1.05263158,  1.18421053,
        1.31578947,  1.44736842,  1.57894737,  1.71052632,  1.84210526,
        1.97368421,  2.10526316,  2.23684211,  2.36842105,  2.5       ])

    ppress_z/=H

    settlement_z_pairs = [[0,1]]

    tvals = np.array(
      [  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
        11.,  12.,  13.,  14.,  15.])

    ppress_z_tval_indexes = [2,4,9,13]
    """)

    por = np.array(
      [[  0.        ,   0.        ,   0.        ,   0.        ],
       [ 15.45888274,  11.25256128,   7.81026213,   6.59018061],
       [ 30.43068308,  22.32963838,  15.56683296,  13.14658973],
       [ 44.4354633 ,  33.04917896,  23.21234569,  19.63280111],
       [ 57.05195463,  43.23209565,  30.68678233,  26.00996498],
       [ 67.96788342,  52.71408077,  37.92918048,  32.23741176],
       [ 77.01434637,  61.35697402,  44.87949524,  38.2733285 ],
       [ 84.17673582,  69.05832424,  51.48050894,  44.07548095],
       [ 89.58036923,  75.75791659,  57.67966088,  49.60194716],
       [ 93.45534202,  81.44039358,  63.43065826,  54.81182438],
       [ 96.09001616,  86.13363155,  68.69472915,  59.66586883],
       [ 97.78432712,  89.90314892,  73.4413916 ,  64.12702968],
       [ 98.81236575,  92.84339272,  77.64864003,  68.16084386],
       [ 99.39948403,  95.0671534 ,  81.30248985,  71.73566822],
       [ 99.71432767,  96.69450875,  84.39587248,  74.82273882],
       [ 99.87247181,  97.84256704,  86.92693088,  77.39606414],
       [ 99.94666509,  98.6169071 ,  88.89682733,  79.43217778],
       [ 99.97898153,  99.1050939 ,  90.30723423,  80.90979557],
       [ 99.99163793,  99.37210731,  91.15773026,  81.80943988],
       [ 99.9948905 ,  99.45708676,  91.44336375,  82.11310762]])
#
#    avp = np.array(
#      [[  7.25979052e+01,   6.65166314e+01,   5.89096834e+01,
#          4.94554633e+01,   3.79564622e+01,   2.66323138e+01,
#          2.50358034e+01,   1.28862133e+01,   4.44927613e+00,
#          1.18311566e+00,   8.09339892e-01,   5.26895921e-02]])
    settle = np.array(
    [[   1.48203385,   50.94661452,   72.8626314 ,   89.97524417,
        104.5927801 ,  117.61098055,  129.48992505,  140.50305227,
        150.82806371,  160.58734407,  169.86851905,  178.73606076,
        187.23840576,  195.41260211,  203.28747828,  210.88586589]])

    for impl in ["vectorized"]:
        for dT in [10]:
            a = Speccon1dVRW(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()

#            plt.clf()
#            plt.figure()
#            plt.plot(por, z,'b-*', label='expected')
#            plt.plot(a.por, z, 'r-+', label='calculated')
#            plt.legend()
##
##
#            plt.figure()
#            plt.plot(t, settle[0],'b-*', label='expected')
#            plt.plot(t, a.set[0], 'r-+', label='calculated')
#            legend=plt.legend()
#            legend.draggable()
##            plt.legend.DraggableLegend()
##            plt.figure()
##            plt.plot(t, avp[0],'b-*',  label='expected')
##            plt.plot(t, a.avp[0], 'r-+', label='calculated')
##            plt.legend()
#            plt.show()

            assert_allclose(a.por, por, atol=1e-2,
                            err_msg = ("Fail. test_zhuandyin2012_drn1_kv_const_mv_linear, por, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_zhuandyin2012_drn1_kv_const_mv_linear, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=1,
                            err_msg = ("Fail. test_zhuandyin2012_drn1_kv_const_mv_linear, set, "
                                "implementation='%s', dT=%s" % (impl, dT)))

def test_zhuandyin2012_drn0_kv_linear_mv_linear():
    """test for zhu and yin 2012

     vertical drainage, depth dependent properties, instant load
     generally:
     mv = mv0*(1+alpha*z/H)**q
     kv = kv0* (1+alpha*z/H)**p
     for this case

     PTPB
    """

    t = np.array([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
        11.,  12.,  13.,  14.,  15.])

    z = np.array([ 0.        ,  0.13157895,  0.26315789,  0.39473684,  0.52631579,
        0.65789474,  0.78947368,  0.92105263,  1.05263158,  1.18421053,
        1.31578947,  1.44736842,  1.57894737,  1.71052632,  1.84210526,
        1.97368421,  2.10526316,  2.23684211,  2.36842105,  2.5       ])

    tpor=t[np.array([2,4,9,13])]




    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np


    ####################################
    #zhuandyin2012 properties
    #ui = 100
    #drn = 0
    #nterms = 50
    #mv0 = 1.2
    #kv0 = 1.6
    #H = 2.5
    #alpha = 1
    #q = 1
    #p = 1
    #z = np.linspace(0,H,20)
    #t = np.linspace(0,15,16)
    #tpor=t[np.array([2,4,9,13])]
    #plot_eigs=False
    #
    #por, doc, settle = zhuandyin2012(
    #    z=z, t=t, alpha=alpha, p=p, q=q, drn=drn, tpor=tpor, H = H, kv0 = kv0, mv0 = mv0, gamw = 10,
    #        ui = 100, nterms = nterms, plot_eigs=plot_eigs)

    ####################################

    neig=40

    H = 2.5
    drn = 0

    mvref = 1.2
    kvref = 1.6 / 10

    kv = PolyLine([0,1], [1,2])
    mv = PolyLine([0,1], [1,2])

    dTv = kvref/mvref/H**2


    dTh=0.1
    dTw=0
    khref = 1
    etref = 1
    kwref=1
    kw = PolyLine([0,1], [1,1])
    kh = PolyLine([0,1], [1,1])
    et = PolyLine([0,1], [1,1])

    surcharge_vs_time = PolyLine([0,0,10], [0,100,100])
    surcharge_vs_depth = PolyLine([0,1], [1,1])


    ppress_z = np.array(
        [ 0.        ,  0.13157895,  0.26315789,  0.39473684,  0.52631579,
        0.65789474,  0.78947368,  0.92105263,  1.05263158,  1.18421053,
        1.31578947,  1.44736842,  1.57894737,  1.71052632,  1.84210526,
        1.97368421,  2.10526316,  2.23684211,  2.36842105,  2.5       ])

    ppress_z/=H

    settlement_z_pairs = [[0,1]]

    tvals = np.array(
      [  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
        11.,  12.,  13.,  14.,  15.])

    ppress_z_tval_indexes = [2,4,9,13]
    """)

    por = np.array(
      [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00],
       [  1.62512314e+01,   1.07457907e+01,   3.84160979e+00,
          1.67218369e+00],
       [  3.12321729e+01,   2.06934529e+01,   7.39486876e+00,
          3.21878437e+00],
       [  4.46159170e+01,   2.96501455e+01,   1.05883768e+01,
          4.60866933e+00],
       [  5.61568702e+01,   3.74501715e+01,   1.33611971e+01,
          5.81528170e+00],
       [  6.56952104e+01,   4.39574427e+01,   1.56639060e+01,
          6.81710096e+00],
       [  7.31489554e+01,   4.90673778e+01,   1.74593972e+01,
          7.59799347e+00],
       [  7.84964687e+01,   5.27082371e+01,   1.87234166e+01,
          8.14744312e+00],
       [  8.17543474e+01,   5.48419309e+01,   1.94448120e+01,
          8.46065565e+00],
       [  8.29565248e+01,   5.54643442e+01,   1.96254925e+01,
          8.53853409e+00],
       [  8.21399323e+01,   5.46052032e+01,   1.92800950e+01,
          8.38752580e+00],
       [  7.93403412e+01,   5.23274722e+01,   1.84353661e+01,
          8.01934544e+00],
       [  7.45994455e+01,   4.87262199e+01,   1.71292755e+01,
          7.45058093e+00],
       [  6.79814251e+01,   4.39268564e+01,   1.54098798e+01,
          6.70219250e+00],
       [  5.95947547e+01,   3.80826216e+01,   1.33339668e+01,
          5.79891783e+00],
       [  4.96134660e+01,   3.13712176e+01,   1.09655124e+01,
          4.76859794e+00],
       [  3.82918130e+01,   2.39905233e+01,   8.37398825e+00,
          3.64144101e+00],
       [  2.59674561e+01,   1.61534102e+01,   5.63256373e+00,
          2.44924221e+00],
       [  1.30506509e+01,   8.08177394e+00,   2.81624514e+00,
          1.22457875e+00],
       [  3.63287486e-14,   4.09473739e-14,   1.63287722e-14,
          7.13624388e-15]])
#
#    avp = np.array(
#      [[  7.25979052e+01,   6.65166314e+01,   5.89096834e+01,
#          4.94554633e+01,   3.79564622e+01,   2.66323138e+01,
#          2.50358034e+01,   1.28862133e+01,   4.44927613e+00,
#          1.18311566e+00,   8.09339892e-01,   5.26895921e-02]])
    settle = np.array(
     [[   3.64504448,  148.20259183,  209.3368549 ,  255.32464212,
        292.06381677,  321.7801835 ,  345.88327216,  365.44839429,
        381.33428266,  394.23438072,  404.71050089,  413.21838104,
        420.12792413,  425.739451  ,  430.29682587,  433.99808524]])

    for impl in ["vectorized"]:
        for dT in [10]:
            a = Speccon1dVRW(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()

#            plt.clf()
#            plt.figure()
#            plt.plot(por, z,'b-*', label='expected')
#            plt.plot(a.por, z, 'r-+', label='calculated')
#            plt.legend()
##
##
#            plt.figure()
#            plt.plot(t, settle[0],'b-*', label='expected')
#            plt.plot(t, a.set[0], 'r-+', label='calculated')
#            legend=plt.legend()
#            legend.draggable()
##            plt.legend.DraggableLegend()
##            plt.figure()
##            plt.plot(t, avp[0],'b-*',  label='expected')
##            plt.plot(t, a.avp[0], 'r-+', label='calculated')
##            plt.legend()
#            plt.show()

            assert_allclose(a.por, por, atol=1e-2,
                            err_msg = ("Fail. test_zhuandyin2012_drn0_kv_linear_mv_linear, por, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_zhuandyin2012_drn0_kv_linear_mv_linear, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=1,
                            err_msg = ("Fail. test_zhuandyin2012_drn0_kv_linear_mv_linear, set, "
                                "implementation='%s', dT=%s" % (impl, dT)))

def test_zhuandyin2012_drn1_kv_linear_mv_linear():
    """test for zhu and yin 2012

     vertical drainage, depth dependent properties, instant load
     generally:
     mv = mv0*(1+alpha*z/H)**q
     kv = kv0* (1+alpha*z/H)**p
     for this case

     PTIB
    """

    t = np.array([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
        11.,  12.,  13.,  14.,  15.])

    z = np.array([ 0.        ,  0.13157895,  0.26315789,  0.39473684,  0.52631579,
        0.65789474,  0.78947368,  0.92105263,  1.05263158,  1.18421053,
        1.31578947,  1.44736842,  1.57894737,  1.71052632,  1.84210526,
        1.97368421,  2.10526316,  2.23684211,  2.36842105,  2.5       ])

    tpor=t[np.array([2,4,9,13])]




    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np


    ####################################
    #zhuandyin2012 properties
    #ui = 100
    #drn = 1
    #nterms = 50
    #mv0 = 1.2
    #kv0 = 1.6
    #H = 2.5
    #alpha = 1
    #q = 1
    #p = 1
    #z = np.linspace(0,H,20)
    #t = np.linspace(0,15,16)
    #tpor=t[np.array([2,4,9,13])]
    #plot_eigs=False
    #
    #por, doc, settle = zhuandyin2012(
    #    z=z, t=t, alpha=alpha, p=p, q=q, drn=drn, tpor=tpor, H = H, kv0 = kv0, mv0 = mv0, gamw = 10,
    #        ui = 100, nterms = nterms, plot_eigs=plot_eigs)

    ####################################

    neig=40

    H = 2.5
    drn = 1

    mvref = 1.2
    kvref = 1.6 / 10

    kv = PolyLine([0,1], [1,2])
    mv = PolyLine([0,1], [1,2])

    dTv = kvref/mvref/H**2

    dTh=0.1
    dTw=0
    khref = 1
    etref = 1
    kwref=1
    kw = PolyLine([0,1], [1,1])
    kh = PolyLine([0,1], [1,1])
    et = PolyLine([0,1], [1,1])

    surcharge_vs_time = PolyLine([0,0,10], [0,100,100])
    surcharge_vs_depth = PolyLine([0,1], [1,1])


    ppress_z = np.array(
        [ 0.        ,  0.13157895,  0.26315789,  0.39473684,  0.52631579,
        0.65789474,  0.78947368,  0.92105263,  1.05263158,  1.18421053,
        1.31578947,  1.44736842,  1.57894737,  1.71052632,  1.84210526,
        1.97368421,  2.10526316,  2.23684211,  2.36842105,  2.5       ])

    ppress_z/=H

    settlement_z_pairs = [[0,1]]

    tvals = np.array(
      [  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
        11.,  12.,  13.,  14.,  15.])

    ppress_z_tval_indexes = [2,4,9,13]
    """)

    por = np.array(
      [[  0.        ,   0.        ,   0.        ,   0.        ],
       [ 16.37146015,  12.2738666 ,   8.84546454,   7.43834356],
       [ 31.5074054 ,  23.79150445,  17.2062873 ,  14.47509718],
       [ 45.12640436,  34.47333568,  25.07537157,  21.10931735],
       [ 57.04026068,  44.25775233,  32.4458375 ,  27.33909292],
       [ 67.16521634,  53.10456242,  39.31181908,  33.16205864],
       [ 75.51945528,  60.99668435,  45.6690481 ,  38.57579977],
       [ 82.20890991,  67.94011853,  51.51524375,  43.57816742],
       [ 87.40509838,  73.96235989,  56.85032632,  48.16751987],
       [ 91.31947167,  79.10952461,  61.67647461,  52.34290194],
       [ 94.17850768,  83.44253908,  65.99804905,  56.10417281],
       [ 96.2027854 ,  87.03277091,  69.8214049 ,  59.45209115],
       [ 97.59187767,  89.9574734 ,  73.15462183,  62.38836579],
       [ 98.51550262,  92.29537114,  76.00717725,  64.91567929],
       [ 99.11026291,  94.12264673,  78.38959142,  67.03769123],
       [ 99.48062621,  95.50950908,  80.3130709 ,  68.75902746],
       [ 99.70256333,  96.5174444 ,  81.78917543,  70.08526077],
       [ 99.82836475,  97.19718251,  82.82952989,  71.02288785],
       [ 99.89146014,  97.5873598 ,  83.44559918,  71.57930625],
       [ 99.91043614,  97.71382984,  83.64853883,  71.76279462]])
#
#    avp = np.array(
#      [[  7.25979052e+01,   6.65166314e+01,   5.89096834e+01,
#          4.94554633e+01,   3.79564622e+01,   2.66323138e+01,
#          2.50358034e+01,   1.28862133e+01,   4.44927613e+00,
#          1.18311566e+00,   8.09339892e-01,   5.26895921e-02]])
    settle = np.array(
    [[   1.21583113,   52.56272601,   76.10323202,   94.84376502,
        111.09114323,  125.73965256,  139.24586843,  151.87600354,
        163.79688456,  175.11790375,  185.91346217,  196.23605561,
        206.12424666,  215.60762381,  224.70991703,  233.45096577]])

    for impl in ["vectorized"]:
        for dT in [10]:
            a = Speccon1dVRW(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()

#            plt.clf()
#            plt.figure()
#            plt.plot(por, z,'b-*', label='expected')
#            plt.plot(a.por, z, 'r-+', label='calculated')
#            plt.legend()
##
##
#            plt.figure()
#            plt.plot(t, settle[0],'b-*', label='expected')
#            plt.plot(t, a.set[0], 'r-+', label='calculated')
#            legend=plt.legend()
#            legend.draggable()
##            plt.legend.DraggableLegend()
##            plt.figure()
##            plt.plot(t, avp[0],'b-*',  label='expected')
##            plt.plot(t, a.avp[0], 'r-+', label='calculated')
##            plt.legend()
#            plt.show()

            assert_allclose(a.por, por, atol=1e-2,
                            err_msg = ("Fail. test_zhuandyin2012_drn1_kv_linear_mv_linear, por, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_zhuandyin2012_drn1_kv_linear_mv_linear, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=1,
                            err_msg = ("Fail. test_zhuandyin2012_drn1_kv_linear_mv_linear, set, "
                                "implementation='%s', dT=%s" % (impl, dT)))


#
def test_zhuandyin2012_drn0_kv_mv_non_linear():
    """test for zhu and yin 2012

     vertical drainage, depth dependent properties, instant load
     generally:
     mv = mv0*(1+alpha*z/H)**q
     kv = kv0* (1+alpha*z/H)**p
     for this case

     PTPB
    """

    t = np.array([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
        11.,  12.,  13.,  14.,  15.])

    z = np.array([ 0.        ,  0.13157895,  0.26315789,  0.39473684,  0.52631579,
        0.65789474,  0.78947368,  0.92105263,  1.05263158,  1.18421053,
        1.31578947,  1.44736842,  1.57894737,  1.71052632,  1.84210526,
        1.97368421,  2.10526316,  2.23684211,  2.36842105,  2.5       ])

    tpor=t[np.array([2,4,9,13])]




    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np


    ####################################
    #zhuandyin2012 properties
    #ui = 100
    #drn = 0
    #nterms = 50
    #mv0 = 1.2
    #kv0 = 1.6
    #H = 2.5
    #alpha = 0.5
    #q = 2
    #p = -2
    #z = np.linspace(0,H,20)
    #t = np.linspace(0,15,16)
    #tpor=t[np.array([2,4,9,13])]
    #plot_eigs=False
    #
    #por, doc, settle = zhuandyin2012(
    #    z=z, t=t, alpha=alpha, p=p, q=q, drn=drn, tpor=tpor, H = H, kv0 = kv0, mv0 = mv0, gamw = 10,
    #        ui = 100, nterms = nterms, plot_eigs=plot_eigs)

    ####################################

    neig=40

    H = 2.5
    drn = 0

    mvref = 1.2
    kvref = 1.6 / 10

    kv = PolyLine(np.array(
        [ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ]),
                  np.array(
        [ 1.        ,  0.90702948,  0.82644628,  0.75614367,  0.69444444,
        0.64      ,  0.59171598,  0.54869684,  0.51020408,  0.47562426,
        0.44444444]))
    mv = PolyLine(np.array(
        [ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ]),
                  np.array(
        [ 1.    ,  1.1025,  1.21  ,  1.3225,  1.44  ,  1.5625,  1.69  ,
        1.8225,  1.96  ,  2.1025,  2.25  ]))



    dTv = kvref/mvref/H**2

    dTh=0.1
    dTw=0
    khref = 1
    etref = 1
    kwref=1
    kw = PolyLine([0,1], [1,1])
    kh = PolyLine([0,1], [1,1])
    et = PolyLine([0,1], [1,1])

    surcharge_vs_time = PolyLine([0,0,10], [0,100,100])
    surcharge_vs_depth = PolyLine([0,1], [1,1])


    ppress_z = np.array(
        [ 0.        ,  0.13157895,  0.26315789,  0.39473684,  0.52631579,
        0.65789474,  0.78947368,  0.92105263,  1.05263158,  1.18421053,
        1.31578947,  1.44736842,  1.57894737,  1.71052632,  1.84210526,
        1.97368421,  2.10526316,  2.23684211,  2.36842105,  2.5       ])

    ppress_z/=H

    settlement_z_pairs = [[0,1]]

    tvals = np.array(
      [  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
        11.,  12.,  13.,  14.,  15.])

    ppress_z_tval_indexes = [2,4,9,13]
    """)

    por = np.array(
     [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00],
       [  1.46735407e+01,   1.03912749e+01,   6.41229682e+00,
          4.57268569e+00],
       [  2.95788346e+01,   2.11321198e+01,   1.30795985e+01,
          9.32820812e+00],
       [  4.40987535e+01,   3.19912441e+01,   1.99046557e+01,
          1.41983660e+01],
       [  5.75828248e+01,   4.26939050e+01,   2.67633017e+01,
          1.90957668e+01],
       [  6.94433436e+01,   5.29361414e+01,   3.35030492e+01,
          2.39125719e+01],
       [  7.92552649e+01,   6.24046262e+01,   3.99428658e+01,
          2.85200193e+01],
       [  8.68301825e+01,   7.07988643e+01,   4.58744872e+01,
          3.27690394e+01],
       [  9.22365664e+01,   7.78502690e+01,   5.10656237e+01,
          3.64923102e+01],
       [  9.57527634e+01,   8.33312726e+01,   5.52654037e+01,
          3.95081180e+01],
       [  9.77603095e+01,   8.70482108e+01,   5.82123900e+01,
          4.16263817e+01],
       [  9.86004778e+01,   8.88153120e+01,   5.96455150e+01,
          4.26571486e+01],
       [  9.84164540e+01,   8.84142419e+01,   5.93183109e+01,
          4.24217705e+01],
       [  9.69914128e+01,   8.55535605e+01,   5.70168132e+01,
          4.07667995e+01],
       [  9.35945553e+01,   7.98525651e+01,   5.25814092e+01,
          3.75803906e+01],
       [  8.68970325e+01,   7.08792914e+01,   4.59325274e+01,
          3.28106427e+01],
       [  7.51182570e+01,   5.82658943e+01,   3.70992525e+01,
          2.64848548e+01],
       [  5.66168535e+01,   4.18996669e+01,   2.62486163e+01,
          1.87281176e+01],
       [  3.09602450e+01,   2.21444807e+01,   1.37115752e+01,
          9.77906130e+00],
       [ -6.18241273e-14,  -2.23931281e-14,  -8.15431343e-15,
         -5.67041685e-15]])
#
#    avp = np.array(
#      [[  7.25979052e+01,   6.65166314e+01,   5.89096834e+01,
#          4.94554633e+01,   3.79564622e+01,   2.66323138e+01,
#          2.50358034e+01,   1.28862133e+01,   4.44927613e+00,
#          1.18311566e+00,   8.09339892e-01,   5.26895921e-02]])
    settle = np.array(
    [[   3.8496919 ,   98.88619572,  139.84619549,  171.27505923,
        197.75770538,  221.03013705,  241.9290118 ,  260.91094338,
        278.25424995,  294.1490242 ,  308.73922326,  322.14274325,
        334.46119179,  345.78480236,  356.19504202,  365.76611184]])

    for impl in ["vectorized"]:
        for dT in [10]:
            a = Speccon1dVRW(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()

#            plt.clf()
#            plt.figure()
#            plt.plot(por, z,'b-*', label='expected')
#            plt.plot(a.por, z, 'r-+', label='calculated')
#            plt.legend()
##
##
#            plt.figure()
#            plt.plot(t, settle[0],'b-*', label='expected')
#            plt.plot(t, a.set[0], 'r-+', label='calculated')
#            legend=plt.legend()
#            legend.draggable()
##            plt.legend.DraggableLegend()
##            plt.figure()
##            plt.plot(t, avp[0],'b-*',  label='expected')
##            plt.plot(t, a.avp[0], 'r-+', label='calculated')
##            plt.legend()
#            plt.show()

            assert_allclose(a.por, por, atol=1e-1,
                            err_msg = ("Fail. test_zhuandyin2012_drn0_kv_mv_non_linear, por, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_zhuandyin2012_drn0_kv_mv_non_linear, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=2,
                            err_msg = ("Fail. test_zhuandyin2012_drn0_kv_mv_non_linear, set, "
                                "implementation='%s', dT=%s" % (impl, dT)))

def test_zhuandyin2012_drn1_kv_mv_non_linear():
    """test for zhu and yin 2012

     vertical drainage, depth dependent properties, instant load
     generally:
     mv = mv0*(1+alpha*z/H)**q
     kv = kv0* (1+alpha*z/H)**p
     for this case

     PTIB
    """

    t = np.array([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
        11.,  12.,  13.,  14.,  15.])

    z = np.array([ 0.        ,  0.13157895,  0.26315789,  0.39473684,  0.52631579,
        0.65789474,  0.78947368,  0.92105263,  1.05263158,  1.18421053,
        1.31578947,  1.44736842,  1.57894737,  1.71052632,  1.84210526,
        1.97368421,  2.10526316,  2.23684211,  2.36842105,  2.5       ])

    tpor=t[np.array([2,4,9,13])]




    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np


    ####################################
    #zhuandyin2012 properties
    #ui = 100
    #drn = 1
    #nterms = 50
    #mv0 = 1.2
    #kv0 = 1.6
    #H = 2.5
    #alpha = 0.5
    #q = 2
    #p = -2
    #z = np.linspace(0,H,20)
    #t = np.linspace(0,15,16)
    #tpor=t[np.array([2,4,9,13])]
    #plot_eigs=False
    #
    #por, doc, settle = zhuandyin2012(
    #    z=z, t=t, alpha=alpha, p=p, q=q, drn=drn, tpor=tpor, H = H, kv0 = kv0, mv0 = mv0, gamw = 10,
    #        ui = 100, nterms = nterms, plot_eigs=plot_eigs)

    ####################################

    neig=40

    H = 2.5
    drn = 1

    mvref = 1.2
    kvref = 1.6 / 10

    kv = PolyLine(np.array(
        [ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ]),
                  np.array(
        [ 1.        ,  0.90702948,  0.82644628,  0.75614367,  0.69444444,
        0.64      ,  0.59171598,  0.54869684,  0.51020408,  0.47562426,
        0.44444444]))
    mv = PolyLine(np.array(
        [ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ]),
                  np.array(
        [ 1.    ,  1.1025,  1.21  ,  1.3225,  1.44  ,  1.5625,  1.69  ,
        1.8225,  1.96  ,  2.1025,  2.25  ]))



    dTv = kvref/mvref/H**2

    dTh=0.1
    dTw=0
    khref = 1
    etref = 1
    kwref=1
    kw = PolyLine([0,1], [1,1])
    kh = PolyLine([0,1], [1,1])
    et = PolyLine([0,1], [1,1])

    surcharge_vs_time = PolyLine([0,0,10], [0,100,100])
    surcharge_vs_depth = PolyLine([0,1], [1,1])


    ppress_z = np.array(
        [ 0.        ,  0.13157895,  0.26315789,  0.39473684,  0.52631579,
        0.65789474,  0.78947368,  0.92105263,  1.05263158,  1.18421053,
        1.31578947,  1.44736842,  1.57894737,  1.71052632,  1.84210526,
        1.97368421,  2.10526316,  2.23684211,  2.36842105,  2.5       ])

    ppress_z/=H

    settlement_z_pairs = [[0,1]]

    tvals = np.array(
      [  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
        11.,  12.,  13.,  14.,  15.])

    ppress_z_tval_indexes = [2,4,9,13]
    """)

    por = np.array(
     [[  0.        ,   0.        ,   0.        ,   0.        ],
       [ 14.67355514,  10.40528932,   6.9478055 ,   5.78180724],
       [ 29.57888041,  21.1645055 ,  14.20340783,  11.8343646 ],
       [ 44.09888584,  32.05180943,  21.70257559,  18.12252389],
       [ 57.58320674,  42.8013085 ,  29.36397652,  24.60042608],
       [ 69.44444839,  53.12388946,  37.08988223,  31.21115238],
       [ 79.25845477,  62.73135179,  44.76833267,  37.88694384],
       [ 86.83932182,  71.36544094,  52.27696814,  44.55012872],
       [ 92.26237958,  78.82739783,  59.48856982,  51.11485949],
       [ 95.82413544,  85.0022523 ,  66.27810786,  57.48969044],
       [ 97.95205743,  89.87205113,  72.53079584,  63.58092097],
       [ 99.09709856,  93.51401605,  78.15033325,  69.29648861],
       [ 99.64623111,  96.08312814,  83.06624662,  74.55003194],
       [ 99.87831495,  97.78286166,  87.23908646,  79.26457494],
       [ 99.96373069,  98.83118188,  90.66226567,  83.3751421 ],
       [ 99.99076102,  99.43000313,  93.35956014,  86.82953547],
       [ 99.99801788,  99.74450265,  95.37770132,  89.58653348],
       [ 99.99964732,  99.89475021,  96.77397903,  91.61094907],
       [ 99.99994823,  99.95773745,  97.59921715,  92.86534606],
       [ 99.99998809,  99.97464489,  97.8768125 ,  93.29878245]])
#
#    avp = np.array(
#      [[  7.25979052e+01,   6.65166314e+01,   5.89096834e+01,
#          4.94554633e+01,   3.79564622e+01,   2.66323138e+01,
#          2.50358034e+01,   1.28862133e+01,   4.44927613e+00,
#          1.18311566e+00,   8.09339892e-01,   5.26895921e-02]])
    settle = np.array(
    [ [   1.92503833,   49.44309786,   69.92309956,   85.63795758,
         98.88619572,  110.55812783,  121.11036104,  130.81414062,
        139.84619549,  148.32927168,  156.35271013,  163.98389265,
        171.27505923,  178.26759418,  184.99484499,  191.48405032]])

    for impl in ["vectorized"]:
        for dT in [10]:
            a = Speccon1dVRW(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()

#            plt.clf()
#            plt.figure()
#            plt.plot(por, z,'b-*', label='expected')
#            plt.plot(a.por, z, 'r-+', label='calculated')
#            plt.legend()
##
##
#            plt.figure()
#            plt.plot(t, settle[0],'b-*', label='expected')
#            plt.plot(t, a.set[0], 'r-+', label='calculated')
#            legend=plt.legend()
#            legend.draggable()
##            plt.legend.DraggableLegend()
##            plt.figure()
##            plt.plot(t, avp[0],'b-*',  label='expected')
##            plt.plot(t, a.avp[0], 'r-+', label='calculated')
##            plt.legend()
#            plt.show()

            assert_allclose(a.por, por, atol=1e-1,
                            err_msg = ("Fail. test_zhuandyin2012_drn1_kv_mv_non_linear, por, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_zhuandyin2012_drn1_kv_mv_non_linear, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=2,
                            err_msg = ("Fail. test_zhuandyin2012_drn1_kv_mv_non_linear, set, "
                                "implementation='%s', dT=%s" % (impl, dT)))

def test_zhuandyin2012_drn0_kv_mv_non_linear_BC():
    """test for zhu and yin 2012

     vertical drainage, depth dependent properties, instant load
     generally:
     mv = mv0*(1+alpha*z/H)**q
     kv = kv0* (1+alpha*z/H)**p
     for this case


     PTPB
     replicate with negative BC
    """

    t = np.array([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
        11.,  12.,  13.,  14.,  15.])

    z = np.array([ 0.        ,  0.13157895,  0.26315789,  0.39473684,  0.52631579,
        0.65789474,  0.78947368,  0.92105263,  1.05263158,  1.18421053,
        1.31578947,  1.44736842,  1.57894737,  1.71052632,  1.84210526,
        1.97368421,  2.10526316,  2.23684211,  2.36842105,  2.5       ])

    tpor=t[np.array([2,4,9,13])]




    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np


    ####################################
    #zhuandyin2012 properties
    #ui = 100
    #drn = 0
    #nterms = 50
    #mv0 = 1.2
    #kv0 = 1.6
    #H = 2.5
    #alpha = 0.5
    #q = 2
    #p = -2
    #z = np.linspace(0,H,20)
    #t = np.linspace(0,15,16)
    #tpor=t[np.array([2,4,9,13])]
    #plot_eigs=False
    #
    #por, doc, settle = zhuandyin2012(
    #    z=z, t=t, alpha=alpha, p=p, q=q, drn=drn, tpor=tpor, H = H, kv0 = kv0, mv0 = mv0, gamw = 10,
    #        ui = 100, nterms = nterms, plot_eigs=plot_eigs)

    ####################################

    neig=40

    H = 2.5
    drn = 0

    mvref = 1.2
    kvref = 1.6 / 10

    kv = PolyLine(np.array(
        [ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ]),
                  np.array(
        [ 1.        ,  0.90702948,  0.82644628,  0.75614367,  0.69444444,
        0.64      ,  0.59171598,  0.54869684,  0.51020408,  0.47562426,
        0.44444444]))
    mv = PolyLine(np.array(
        [ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ]),
                  np.array(
        [ 1.    ,  1.1025,  1.21  ,  1.3225,  1.44  ,  1.5625,  1.69  ,
        1.8225,  1.96  ,  2.1025,  2.25  ]))



    dTv = kvref/mvref/H**2

    dTh=0.1
    dTw=0
    khref = 1
    etref = 1
    kwref=1
    kw = PolyLine([0,1], [1,1])
    kh = PolyLine([0,1], [1,1])
    et = PolyLine([0,1], [1,1])

    #surcharge_vs_time = PolyLine([0,0,10], [0,100,100])
    #surcharge_vs_depth = PolyLine([0,1], [1,1])
    top_vs_time = PolyLine([0,0,10], [0,-100,-100])
    bot_vs_time = PolyLine([0,0,10], [0,-100,-100])

    ppress_z = np.array(
        [ 0.        ,  0.13157895,  0.26315789,  0.39473684,  0.52631579,
        0.65789474,  0.78947368,  0.92105263,  1.05263158,  1.18421053,
        1.31578947,  1.44736842,  1.57894737,  1.71052632,  1.84210526,
        1.97368421,  2.10526316,  2.23684211,  2.36842105,  2.5       ])

    ppress_z/=H

    settlement_z_pairs = [[0,1]]

    tvals = np.array(
      [  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
        11.,  12.,  13.,  14.,  15.])

    ppress_z_tval_indexes = [2,4,9,13]
    """)

    por = -100+np.array(
     [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00],
       [  1.46735407e+01,   1.03912749e+01,   6.41229682e+00,
          4.57268569e+00],
       [  2.95788346e+01,   2.11321198e+01,   1.30795985e+01,
          9.32820812e+00],
       [  4.40987535e+01,   3.19912441e+01,   1.99046557e+01,
          1.41983660e+01],
       [  5.75828248e+01,   4.26939050e+01,   2.67633017e+01,
          1.90957668e+01],
       [  6.94433436e+01,   5.29361414e+01,   3.35030492e+01,
          2.39125719e+01],
       [  7.92552649e+01,   6.24046262e+01,   3.99428658e+01,
          2.85200193e+01],
       [  8.68301825e+01,   7.07988643e+01,   4.58744872e+01,
          3.27690394e+01],
       [  9.22365664e+01,   7.78502690e+01,   5.10656237e+01,
          3.64923102e+01],
       [  9.57527634e+01,   8.33312726e+01,   5.52654037e+01,
          3.95081180e+01],
       [  9.77603095e+01,   8.70482108e+01,   5.82123900e+01,
          4.16263817e+01],
       [  9.86004778e+01,   8.88153120e+01,   5.96455150e+01,
          4.26571486e+01],
       [  9.84164540e+01,   8.84142419e+01,   5.93183109e+01,
          4.24217705e+01],
       [  9.69914128e+01,   8.55535605e+01,   5.70168132e+01,
          4.07667995e+01],
       [  9.35945553e+01,   7.98525651e+01,   5.25814092e+01,
          3.75803906e+01],
       [  8.68970325e+01,   7.08792914e+01,   4.59325274e+01,
          3.28106427e+01],
       [  7.51182570e+01,   5.82658943e+01,   3.70992525e+01,
          2.64848548e+01],
       [  5.66168535e+01,   4.18996669e+01,   2.62486163e+01,
          1.87281176e+01],
       [  3.09602450e+01,   2.21444807e+01,   1.37115752e+01,
          9.77906130e+00],
       [ -6.18241273e-14,  -2.23931281e-14,  -8.15431343e-15,
         -5.67041685e-15]])
#
#    avp = np.array(
#      [[  7.25979052e+01,   6.65166314e+01,   5.89096834e+01,
#          4.94554633e+01,   3.79564622e+01,   2.66323138e+01,
#          2.50358034e+01,   1.28862133e+01,   4.44927613e+00,
#          1.18311566e+00,   8.09339892e-01,   5.26895921e-02]])
    settle = np.array(
    [[   3.8496919 ,   98.88619572,  139.84619549,  171.27505923,
        197.75770538,  221.03013705,  241.9290118 ,  260.91094338,
        278.25424995,  294.1490242 ,  308.73922326,  322.14274325,
        334.46119179,  345.78480236,  356.19504202,  365.76611184]])

    for impl in ["vectorized"]:
        for dT in [10]:
            a = Speccon1dVRW(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()

#            plt.clf()
#            plt.figure()
#            plt.plot(por, z,'b-*', label='expected')
#            plt.plot(a.por, z, 'r-+', label='calculated')
#            plt.legend()
##
##
#            plt.figure()
#            plt.plot(t, settle[0],'b-*', label='expected')
#            plt.plot(t, a.set[0], 'r-+', label='calculated')
#            legend=plt.legend()
#            legend.draggable()
##            plt.legend.DraggableLegend()
##            plt.figure()
##            plt.plot(t, avp[0],'b-*',  label='expected')
##            plt.plot(t, a.avp[0], 'r-+', label='calculated')
##            plt.legend()
#            plt.show()

            assert_allclose(a.por, por, atol=1e-1,
                            err_msg = ("Fail. test_zhuandyin2012_drn0_kv_mv_non_linear_BC, por, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_zhuandyin2012_drn0_kv_mv_non_linear_BC, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=2,
                            err_msg = ("Fail. test_zhuandyin2012_drn0_kv_mv_non_linear_BC, set, "
                                "implementation='%s', dT=%s" % (impl, dT)))



def test_tang_and_onitsuka_vert_and_radial_ideal():
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


    z = np.array(
      [ 0.        ,  0.11111111,  0.22222222,  0.33333333,  0.44444444,
        0.55555556,  0.66666667,  0.77777778,  0.88888889,  1.        ])

    por = np.array(
    [[  0.        ,   0.        ,   0.        ],
       [ 10.62834433,   5.71540426,   0.79195768],
       [ 18.10955225,  11.14818494,   1.55981113],
       [ 23.22767635,  16.0566849 ,   2.2801995 ],
       [ 26.62643019,  20.26892536,   2.93122316],
       [ 28.80909111,  23.69187844,   3.49311208],
       [ 30.15548187,  26.3014876 ,   3.94882356],
       [ 30.93852215,  28.11965758,   4.28455203],
       [ 31.33977648,  29.18687894,   4.49013755],
       [ 31.4624026 ,  29.5381209 ,   4.55936353]])

    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np

    #########################################
    #Tang and onitsuka input
    #t = np.array([  1.00000000e-03,   2.00000000e-03,   3.00000000e-03,
    #     4.00000000e-03,   5.00000000e-03,   6.00000000e-03,
    #     7.00000000e-03,   8.00000000e-03,   9.00000000e-03,
    #     1.00000000e-02,   2.00000000e-02,   3.00000000e-02,
    #     4.00000000e-02,   5.00000000e-02,   6.00000000e-02,
    #     7.00000000e-02,   8.00000000e-02,   9.00000000e-02,
    #     1.00000000e-01,   1.10000000e-01,   1.20000000e-01,
    #     1.30000000e-01,   1.40000000e-01,   1.50000000e-01,
    #     1.60000000e-01,   1.70000000e-01,   1.80000000e-01,
    #     1.90000000e-01,   2.00000000e-01,   2.10000000e-01,
    #     2.20000000e-01,   2.30000000e-01,   2.40000000e-01,
    #     2.50000000e-01,   2.60000000e-01,   2.70000000e-01,
    #     2.80000000e-01,   2.90000000e-01,   3.00000000e-01,
    #     3.10000000e-01,   3.20000000e-01,   3.30000000e-01,
    #     3.40000000e-01,   3.50000000e-01,   3.60000000e-01,
    #     3.70000000e-01,   3.80000000e-01,   3.90000000e-01,
    #     4.00000000e-01,   4.10000000e-01,   4.20000000e-01,
    #     4.30000000e-01,   4.40000000e-01,   4.50000000e-01,
    #     4.60000000e-01,   4.70000000e-01,   4.80000000e-01,
    #     4.90000000e-01,   5.00000000e-01,   5.10000000e-01,
    #     5.20000000e-01,   5.30000000e-01,   5.40000000e-01,
    #     5.50000000e-01,   5.60000000e-01,   5.70000000e-01,
    #     5.80000000e-01,   5.90000000e-01,   6.00000000e-01,
    #     6.10000000e-01,   6.20000000e-01,   6.30000000e-01,
    #     6.40000000e-01,   6.50000000e-01,   6.60000000e-01,
    #     6.70000000e-01,   6.80000000e-01,   6.90000000e-01,
    #     7.00000000e-01,   7.10000000e-01,   7.20000000e-01,
    #     7.30000000e-01,   7.40000000e-01,   7.50000000e-01,
    #     7.60000000e-01,   7.70000000e-01,   7.80000000e-01,
    #     7.90000000e-01,   8.00000000e-01,   8.10000000e-01,
    #     8.20000000e-01,   8.30000000e-01,   8.40000000e-01,
    #     8.50000000e-01,   8.60000000e-01,   8.70000000e-01,
    #     8.80000000e-01,   8.90000000e-01,   9.00000000e-01,
    #     9.10000000e-01,   9.20000000e-01,   9.30000000e-01,
    #     9.40000000e-01,   9.50000000e-01,   9.60000000e-01,
    #     9.70000000e-01,   9.80000000e-01,   9.90000000e-01,
    #     1.00000000e+00,   1.01000000e+00])
    #
    #H = 1
    #z  = np.linspace(0, H,10)
    #kv, kh, ks, kw = (10, 10, 10, 1e7)
    #mv=1
    #gamw = 10
    #rw, rs, re = (0.03, 0.03, 0.5)
    #drn = 1
    #surcharge_vs_time = ((0,0.15, 0.3, 0.45,100.0), (0,50,50.0,100.0,100.0))
    #tpor = t[np.array([20,60,90])]
    #nterms = 20
    #
    #por, avp, settle = tangandonitsuka2000(z=z, t=t, kv=kv, kh=kh, ks=ks, kw=kw, mv=mv, gamw=gamw, rw=rw, rs=rs, re=re, H=H,
    #                   drn=drn, surcharge_vs_time=surcharge_vs_time,
    #                   tpor=tpor, nterms=nterms)
    ##################################################################


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

    dTw=5000
    kwref=1
    kw = PolyLine([0,1], [1,1])


    surcharge_vs_depth = PolyLine([0,1], [1,1])
    surcharge_vs_time = PolyLine([0,0.15,0.3,0.45,4],[0.0,50,50,100,100])


    ppress_z = np.%s
    avg_ppress_z_pairs = [[0,1]]
    settlement_z_pairs = [[0,1]]
    ppress_z_tval_indexes = [20,60,90]
    tvals = np.%s

    """ % (repr(z), repr(t)))


#    por = 100 * TERZ1D_POR
#    avp = 100 * TERZ1D_AVP
    settle = (np.interp(t,[0,0.15,0.3,0.45,4], [0.0,50,50,100,100]) - avp)



    for impl in ["vectorized"]:
        for dT in [0.1]:
            a = Speccon1dVRW(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()



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
            assert_allclose(a.avp, avp, atol=1e-2,
                            err_msg = ("Fail. test_tang_and_onitsuka_vert_and_radial, avp, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.por, por, atol=1e-2,
                            err_msg = ("Fail. test_tang_and_onitsuka_vert_and_radial, por, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=1e-2,
                            err_msg = ("Fail. test_tang_and_onitsuka_vert_and_radial, settle, "
                                "implementation='%s', dT=%s" % (impl, dT)))

def test_tang_and_onitsuka_vert_and_radial_well_resistance():
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



    avp = np.array([[  0.32511915,   0.64395023,   0.95850548,   1.26960191,
         1.57771242,   1.88315609,   2.1861672 ,   2.48692722,
         2.78558219,   3.0822529 ,   5.95702388,   8.69654086,
        11.32667999,  13.86239625,  16.31378009,  18.68823918,
        20.9915131 ,  23.22821996,  25.40218126,  27.51662947,
        29.5743482 ,  31.57777121,  33.52905472,  35.43013181,
        34.20050107,  33.13149968,  32.15237839,  31.23863626,
        30.37660619,  29.55741125,  28.77479332,  28.02410687,
        27.30177917,  26.60499098,  25.93147359,  25.27937235,
        24.6471508 ,  24.03352086,  23.43739069,  25.94007773,
        28.25103728,  30.44178874,  32.53758204,  34.5528131 ,
        36.4970684 ,  38.37729705,  40.19881778,  41.9658593 ,
        43.6818802 ,  45.34977247,  46.97199843,  48.55068709,
        50.08770434,  51.58470557,  49.96092296,  48.5074373 ,
        47.15325214,  45.8736276 ,  44.65466537,  43.48726435,
        42.36494895,  41.28286238,  40.23722658,  39.2250226 ,
        38.24378734,  37.29147691,  36.36637054,  35.46700056,
        34.59210015,  33.74056327,  32.91141376,  32.10378096,
        31.31688056,  30.54999935,  29.80248297,  29.07372621,
        28.36316519,  27.67027097,  26.99454454,  26.33551264,
        25.69272447,  25.06574893,  24.45417241,  23.857597  ,
        23.27563895,  22.70792745,  22.15410362,  21.61381963,
        21.08673795,  20.57253078,  20.07087947,  19.5814741 ,
        19.10401306,  18.6382027 ,  18.18375701,  17.74039736,
        17.3078522 ,  16.88585686,  16.47415334,  16.07249005,
        15.68062171,  15.29830908,  14.92531886,  14.5614235 ,
        14.20640105,  13.86003501,  13.5221142 ,  13.1924326 ,
        12.87078925,  12.55698809]])


    z = np.array(
      [ 0.        ,  0.11111111,  0.22222222,  0.33333333,  0.44444444,
        0.55555556,  0.66666667,  0.77777778,  0.88888889,  1.        ])

    por = np.array(
    [[  0.        ,   0.        ,   0.        ],
       [ 12.53440575,  12.14329557,   5.47538418],
       [ 21.70121334,  23.7536632 ,  10.78420485],
       [ 28.183674  ,  34.3627168 ,  15.76501214],
       [ 32.61240467,  43.61300164,  20.26641403],
       [ 35.52623848,  51.27538849,  24.15169042],
       [ 37.36067128,  57.23734454,  27.30293317],
       [ 38.44529475,  61.47089757,  29.62459012],
       [ 39.00780593,  63.99300069,  31.04631563],
       [ 39.18082801,  64.82986237,  31.52505531]])

    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np

    #########################################
    #Tang and onitsuka input
    #t = np.array([  1.00000000e-03,   2.00000000e-03,   3.00000000e-03,
    #     4.00000000e-03,   5.00000000e-03,   6.00000000e-03,
    #     7.00000000e-03,   8.00000000e-03,   9.00000000e-03,
    #     1.00000000e-02,   2.00000000e-02,   3.00000000e-02,
    #     4.00000000e-02,   5.00000000e-02,   6.00000000e-02,
    #     7.00000000e-02,   8.00000000e-02,   9.00000000e-02,
    #     1.00000000e-01,   1.10000000e-01,   1.20000000e-01,
    #     1.30000000e-01,   1.40000000e-01,   1.50000000e-01,
    #     1.60000000e-01,   1.70000000e-01,   1.80000000e-01,
    #     1.90000000e-01,   2.00000000e-01,   2.10000000e-01,
    #     2.20000000e-01,   2.30000000e-01,   2.40000000e-01,
    #     2.50000000e-01,   2.60000000e-01,   2.70000000e-01,
    #     2.80000000e-01,   2.90000000e-01,   3.00000000e-01,
    #     3.10000000e-01,   3.20000000e-01,   3.30000000e-01,
    #     3.40000000e-01,   3.50000000e-01,   3.60000000e-01,
    #     3.70000000e-01,   3.80000000e-01,   3.90000000e-01,
    #     4.00000000e-01,   4.10000000e-01,   4.20000000e-01,
    #     4.30000000e-01,   4.40000000e-01,   4.50000000e-01,
    #     4.60000000e-01,   4.70000000e-01,   4.80000000e-01,
    #     4.90000000e-01,   5.00000000e-01,   5.10000000e-01,
    #     5.20000000e-01,   5.30000000e-01,   5.40000000e-01,
    #     5.50000000e-01,   5.60000000e-01,   5.70000000e-01,
    #     5.80000000e-01,   5.90000000e-01,   6.00000000e-01,
    #     6.10000000e-01,   6.20000000e-01,   6.30000000e-01,
    #     6.40000000e-01,   6.50000000e-01,   6.60000000e-01,
    #     6.70000000e-01,   6.80000000e-01,   6.90000000e-01,
    #     7.00000000e-01,   7.10000000e-01,   7.20000000e-01,
    #     7.30000000e-01,   7.40000000e-01,   7.50000000e-01,
    #     7.60000000e-01,   7.70000000e-01,   7.80000000e-01,
    #     7.90000000e-01,   8.00000000e-01,   8.10000000e-01,
    #     8.20000000e-01,   8.30000000e-01,   8.40000000e-01,
    #     8.50000000e-01,   8.60000000e-01,   8.70000000e-01,
    #     8.80000000e-01,   8.90000000e-01,   9.00000000e-01,
    #     9.10000000e-01,   9.20000000e-01,   9.30000000e-01,
    #     9.40000000e-01,   9.50000000e-01,   9.60000000e-01,
    #     9.70000000e-01,   9.80000000e-01,   9.90000000e-01,
    #     1.00000000e+00,   1.01000000e+00])
    #
    #H = 1
    #z  = np.linspace(0, H,10)
    #kv, kh, ks, kw = (10, 10, 10, 1)
    #mv=1
    #gamw = 10
    #rw, rs, re = (0.03, 0.03, 0.5)
    #drn = 1
    #surcharge_vs_time = ((0,0.15, 0.3, 0.45,100.0), (0,50,50.0,100.0,100.0))
    #tpor = t[np.array([20,60,90])]
    #nterms = 20
    #
    #por, avp, settle = tangandonitsuka2000(z=z, t=t, kv=kv, kh=kh, ks=ks, kw=kw, mv=mv, gamw=gamw, rw=rw, rs=rs, re=re, H=H,
    #                   drn=drn, surcharge_vs_time=surcharge_vs_time,
    #                   tpor=tpor, nterms=nterms)
    ##################################################################


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

    dTw = 0.0003613006824568446 #dTv=kwref/mvref/gamw/H**2
    kwref = 1
    kw = PolyLine([0,1], [1,1])


    surcharge_vs_depth = PolyLine([0,1], [1,1])
    surcharge_vs_time = PolyLine([0,0.15,0.3,0.45,4],[0.0,50,50,100,100])


    ppress_z = np.%s
    avg_ppress_z_pairs = [[0,1]]
    settlement_z_pairs = [[0,1]]
    ppress_z_tval_indexes = [20,60,90]
    tvals = np.%s

    """ % (repr(z), repr(t)))


#    por = 100 * TERZ1D_POR
#    avp = 100 * TERZ1D_AVP
    settle = (np.interp(t,[0,0.15,0.3,0.45,4], [0.0,50,50,100,100]) - avp)



    for impl in ["vectorized"]:
        for dT in [0.1]:
            a = Speccon1dVRW(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()



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
            assert_allclose(a.avp, avp, atol=1e-2,
                            err_msg = ("Fail. test_tang_and_onitsuka_vert_and_radial, avp, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.por, por, atol=1e-2,
                            err_msg = ("Fail. test_tang_and_onitsuka_vert_and_radial, por, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=1e-2,
                            err_msg = ("Fail. test_tang_and_onitsuka_vert_and_radial, settle, "
                                "implementation='%s', dT=%s" % (impl, dT)))





def test_two_layers_no_drain_in_bottom_layer():
    """2 soil layers

    kw=0 in bottom layer should be same as 2 layers with no drain
    in bottom layer.


    """

    t = np.array(
      [ 0.01      ,  0.01114137,  0.01241302,  0.01382981,  0.01540831,
        0.01716698,  0.01912637,  0.02130941,  0.02374161,  0.02645142,
        0.02947052,  0.03283421,  0.03658182,  0.04075718,  0.0454091 ,
        0.05059197,  0.05636641,  0.06279993,  0.06996776,  0.0779537 ,
        0.08685114,  0.09676411,  0.10780851,  0.1201135 ,  0.13382295,
        0.14909717,  0.16611474,  0.18507465,  0.2061986 ,  0.22973358,
        0.25595479,  0.28516882,  0.31771727,  0.35398071,  0.39438316,
        0.43939706,  0.48954872,  0.54542456,  0.60767794,  0.67703675,
        0.75431201,  0.84040726,  0.93632921,  1.04319944,  1.16226758,
        1.29492584,  1.44272539,  1.60739439,  1.7908583 ,  1.99526231])



    avp = np.array(
     [[ 18.02593864,  17.87835341,  17.71980093,  17.54964207,
         17.36724711,  17.17200801,  16.96335316,  16.7407645 ,
         16.50379736,  16.25210253,  15.98545054,  15.70375721,
         15.40710959,  15.09579046,  14.77029944,  14.43136775,
         14.07996352,  13.7172843 ,  13.3447335 ,  12.96387856,
         12.57638968,  12.18396013,  11.7882113 ,  11.39058856,
         10.99225595,  10.59400026,  10.1961553 ,   9.79855724,
          9.40053937,   9.00097176,   8.59834691,   8.19090865,
          7.77681827,   7.35434941,   6.92210151,   6.47921886,
          6.02559885,   5.56206938,   5.09051311,   4.61391873,
          4.13634604,   3.66280228,   3.19903619,   2.7512627 ,
          2.32583262,   1.92886278,   1.56584432,   1.24125415,
          0.95820373,   0.71816733]])

    settle = np.array(
     [[  1.97406136,   2.12164659,   2.28019907,   2.45035793,
          2.63275289,   2.82799199,   3.03664684,   3.2592355 ,
          3.49620264,   3.74789747,   4.01454946,   4.29624279,
          4.59289041,   4.90420954,   5.22970056,   5.56863225,
          5.92003648,   6.2827157 ,   6.6552665 ,   7.03612144,
          7.42361032,   7.81603987,   8.2117887 ,   8.60941144,
          9.00774405,   9.40599974,   9.8038447 ,  10.20144276,
         10.59946063,  10.99902824,  11.40165309,  11.80909135,
         12.22318173,  12.64565059,  13.07789849,  13.52078114,
         13.97440115,  14.43793062,  14.90948689,  15.38608127,
         15.86365396,  16.33719772,  16.80096381,  17.2487373 ,
         17.67416738,  18.07113722,  18.43415568,  18.75874585,
         19.04179627,  19.28183267]])


    z = np.array(
      [ 0.,  0.01010101,  0.02020202,  0.03030303,  0.04040404,
        0.05050505,  0.06060606,  0.07070707,  0.08080808,  0.09090909,
        0.1010101 ,  0.11111111,  0.12121212,  0.13131313,  0.14141414,
        0.15151515,  0.16161616,  0.17171717,  0.18181818,  0.19191919,
        0.2020202 ,  0.21212121,  0.22222222,  0.23232323,  0.24242424,
        0.25252525,  0.26262626,  0.27272727,  0.28282828,  0.29292929,
        0.3030303 ,  0.31313131,  0.32323232,  0.33333333,  0.34343434,
        0.35353535,  0.36363636,  0.37373737,  0.38383838,  0.39393939,
        0.4040404 ,  0.41414141,  0.42424242,  0.43434343,  0.44444444,
        0.45454545,  0.46464646,  0.47474747,  0.48484848,  0.49494949,
        0.50505051,  0.51515152,  0.52525253,  0.53535354,  0.54545455,
        0.55555556,  0.56565657,  0.57575758,  0.58585859,  0.5959596 ,
        0.60606061,  0.61616162,  0.62626263,  0.63636364,  0.64646465,
        0.65656566,  0.66666667,  0.67676768,  0.68686869,  0.6969697 ,
        0.70707071,  0.71717172,  0.72727273,  0.73737374,  0.74747475,
        0.75757576,  0.76767677,  0.77777778,  0.78787879,  0.7979798 ,
        0.80808081,  0.81818182,  0.82828283,  0.83838384,  0.84848485,
        0.85858586,  0.86868687,  0.87878788,  0.88888889,  0.8989899 ,
        0.90909091,  0.91919192,  0.92929293,  0.93939394,  0.94949495,
        0.95959596,  0.96969697,  0.97979798,  0.98989899,  1.        ])

    por = np.array(
    [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00],
       [  2.05562052e+00,   1.25986166e+00,   6.78991414e-01,
          2.84541748e-01,   9.10626442e-02,   3.66582957e-02,
          1.69401561e-02,   3.80087999e-03],
       [  4.06978916e+00,   2.50772351e+00,   1.35494648e+00,
          5.68592909e-01,   1.82215916e-01,   7.34271212e-02,
          3.39358576e-02,   7.61423113e-03],
       [  6.00356619e+00,   3.73195163e+00,   2.02489728e+00,
          8.51696052e-01,   2.73576045e-01,   1.10434600e-01,
          5.10513206e-02,   1.14544710e-02],
       [  7.82272729e+00,   4.92156423e+00,   2.68594746e+00,
          1.13339909e+00,   3.65261297e-01,   1.47810954e-01,
          6.83518475e-02,   1.53362612e-02],
       [  9.49951294e+00,   6.06649007e+00,   3.33527234e+00,
          1.41322066e+00,   4.57359240e-01,   1.85665856e-01,
          8.58926759e-02,   1.92720039e-02],
       [  1.10138572e+01,   7.15787331e+00,   3.97020234e+00,
          1.69069012e+00,   5.49949944e-01,   2.24102244e-01,
          1.03725678e-01,   2.32733458e-02],
       [  1.23539525e+01,   8.18834767e+00,   4.58833192e+00,
          1.96541993e+00,   6.43158810e-01,   2.63251001e-01,
          1.21916356e-01,   2.73549933e-02],
       [  1.35160196e+01,   9.15213127e+00,   5.18751845e+00,
          2.23709246e+00,   7.37150779e-01,   3.03268914e-01,
          1.40542948e-01,   3.15345114e-02],
       [  1.45034034e+01,   1.00449646e+01,   5.76578610e+00,
          2.50536456e+00,   8.32060724e-01,   3.44294329e-01,
          1.59674785e-01,   3.58274664e-02],
       [  1.53253465e+01,   1.08640917e+01,   6.32131649e+00,
          2.76983967e+00,   9.27965781e-01,   3.86427657e-01,
          1.79362674e-01,   4.02452659e-02],
       [  1.59956837e+01,   1.16083632e+01,   6.85260170e+00,
          3.03018014e+00,   1.02495891e+00,   4.29776751e-01,
          1.99660976e-01,   4.48001158e-02],
       [  1.65314127e+01,   1.22782578e+01,   7.35857377e+00,
          3.28621398e+00,   1.12322631e+00,   4.74507099e-01,
          2.20652168e-01,   4.95105326e-02],
       [  1.69510760e+01,   1.28756170e+01,   7.83851004e+00,
          3.53786532e+00,   1.22300400e+00,   5.20816277e-01,
          2.42434492e-01,   5.43985725e-02],
       [  1.72732221e+01,   1.34032356e+01,   8.29182518e+00,
          3.78498779e+00,   1.32446005e+00,   5.68859447e-01,
          2.65085625e-01,   5.94816751e-02],
       [  1.75153748e+01,   1.38646581e+01,   8.71805753e+00,
          4.02735125e+00,   1.42767771e+00,   6.18736572e-01,
          2.88656320e-01,   6.47712366e-02],
       [  1.76935872e+01,   1.42642400e+01,   9.11710254e+00,
          4.26483640e+00,   1.53278641e+00,   6.70573816e-01,
          3.13210040e-01,   7.02815062e-02],
       [  1.78221743e+01,   1.46071103e+01,   9.48936081e+00,
          4.49757345e+00,   1.64006257e+00,   7.24588524e-01,
          3.38854727e-01,   7.60367172e-02],
       [  1.79132974e+01,   1.48987602e+01,   9.83553656e+00,
          4.72580202e+00,   1.74983905e+00,   7.81034321e-01,
          3.65716164e-01,   8.20651082e-02],
       [  1.79766272e+01,   1.51445247e+01,   1.01563148e+01,
          4.94962489e+00,   1.86233366e+00,   8.40092957e-01,
          3.93885235e-01,   8.83870841e-02],
       [  1.80195496e+01,   1.53494630e+01,   1.04523667e+01,
          5.16902028e+00,   1.97765035e+00,   9.01872744e-01,
          4.23417032e-01,   9.50150160e-02],
       [  1.80478503e+01,   1.55186405e+01,   1.07246977e+01,
          5.38413656e+00,   2.09597996e+00,   9.66533630e-01,
          4.54391769e-01,   1.01966913e-01],
       [  1.80662269e+01,   1.56572445e+01,   1.09748268e+01,
          5.59546291e+00,   2.21772452e+00,   1.03436719e+00,
          4.86953872e-01,   1.09275200e-01],
       [  1.80782168e+01,   1.57701702e+01,   1.12044654e+01,
          5.80359717e+00,   2.34334503e+00,   1.10570326e+00,
          5.21266616e-01,   1.16976531e-01],
       [  1.80859786e+01,   1.58615086e+01,   1.14150755e+01,
          6.00890456e+00,   2.47312536e+00,   1.18076135e+00,
          5.57439678e-01,   1.25095531e-01],
       [  1.80906357e+01,   1.59346218e+01,   1.16079283e+01,
          6.21157118e+00,   2.60720185e+00,   1.25966668e+00,
          5.95536796e-01,   1.33646512e-01],
       [  1.80930997e+01,   1.59927387e+01,   1.17846201e+01,
          6.41202956e+00,   2.74585451e+00,   1.34263190e+00,
          6.35664302e-01,   1.42653348e-01],
       [  1.80944960e+01,   1.60392359e+01,   1.19473001e+01,
          6.61116490e+00,   2.88965737e+00,   1.43005370e+00,
          6.78018321e-01,   1.52160068e-01],
       [  1.80957291e+01,   1.60771499e+01,   1.20981978e+01,
          6.80995564e+00,   3.03924030e+00,   1.52236606e+00,
          7.22813391e-01,   1.62214836e-01],
       [  1.80969034e+01,   1.61085810e+01,   1.22390365e+01,
          7.00900862e+00,   3.19496835e+00,   1.61983851e+00,
          7.70184138e-01,   1.72847877e-01],
       [  1.80975980e+01,   1.61349333e+01,   1.23711871e+01,
          7.20867828e+00,   3.35701566e+00,   1.72261964e+00,
          8.20206314e-01,   1.84076202e-01],
       [  1.80978176e+01,   1.61578205e+01,   1.24964258e+01,
          7.40967825e+00,   3.52578454e+00,   1.83099894e+00,
          8.73024317e-01,   1.95932230e-01],
       [  1.80983643e+01,   1.61794414e+01,   1.26172283e+01,
          7.61333250e+00,   3.70208829e+00,   1.94552420e+00,
          9.28908577e-01,   2.08476672e-01],
       [  1.81000339e+01,   1.62018446e+01,   1.27360680e+01,
          7.82102538e+00,   3.88678397e+00,   2.06677438e+00,
          9.88145045e-01,   2.21773722e-01],
       [  1.81027168e+01,   1.62260938e+01,   1.28546163e+01,
          8.03355595e+00,   4.08032705e+00,   2.19507974e+00,
          1.05089878e+00,   2.35860441e-01],
       [  1.81058045e+01,   1.62526620e+01,   1.29740448e+01,
          8.25136565e+00,   4.28291792e+00,   2.33061024e+00,
          1.11725681e+00,   2.50756369e-01],
       [  1.81095573e+01,   1.62827408e+01,   1.30961613e+01,
          8.47544533e+00,   4.49512325e+00,   2.47376386e+00,
          1.18741731e+00,   2.66506000e-01],
       [  1.81155659e+01,   1.63186915e+01,   1.32237945e+01,
          8.70765607e+00,   4.71810895e+00,   2.62531624e+00,
          1.26176269e+00,   2.83195183e-01],
       [  1.81254743e+01,   1.63628374e+01,   1.33596952e+01,
          8.94985732e+00,   4.95305435e+00,   2.78605752e+00,
          1.34068289e+00,   3.00911458e-01],
       [  1.81396357e+01,   1.64161739e+01,   1.35053517e+01,
          9.20294655e+00,   5.20048966e+00,   2.95637618e+00,
          1.42437252e+00,   3.19698520e-01],
       [  1.81578285e+01,   1.64790210e+01,   1.36615449e+01,
          9.46728165e+00,   5.46057213e+00,   3.13642740e+00,
          1.51291266e+00,   3.39574576e-01],
       [  1.81814579e+01,   1.65530899e+01,   1.38301894e+01,
          9.74414988e+00,   5.73409163e+00,   3.32676174e+00,
          1.60657705e+00,   3.60601086e-01],
       [  1.82142256e+01,   1.66421109e+01,   1.40149053e+01,
          1.00362427e+01,   6.02281375e+00,   3.52854518e+00,
          1.70593964e+00,   3.82906896e-01],
       [  1.82598458e+01,   1.67496569e+01,   1.42190781e+01,
          1.03461151e+01,   6.32843987e+00,   3.74291340e+00,
          1.81156028e+00,   4.06617686e-01],
       [  1.83195944e+01,   1.68767966e+01,   1.44437454e+01,
          1.06744760e+01,   6.65142810e+00,   3.97023148e+00,
          1.92362389e+00,   4.31774982e-01],
       [  1.83936222e+01,   1.70233166e+01,   1.46886736e+01,
          1.10210191e+01,   6.99153949e+00,   4.21042835e+00,
          2.04210301e+00,   4.58372631e-01],
       [  1.84854082e+01,   1.71919460e+01,   1.49561524e+01,
          1.13874555e+01,   7.34991267e+00,   4.46429367e+00,
          2.16738950e+00,   4.86498608e-01],
       [  1.86034190e+01,   1.73899685e+01,   1.52524813e+01,
          1.17787407e+01,   7.72993606e+00,   4.73403351e+00,
          2.30056585e+00,   5.16395916e-01],
       [  1.87553299e+01,   1.76237886e+01,   1.55831379e+01,
          1.21992398e+01,   8.13465336e+00,   5.02165754e+00,
          2.44262007e+00,   5.48286351e-01],
       [  1.89382521e+01,   1.78896791e+01,   1.59444713e+01,
          1.26460422e+01,   8.56216203e+00,   5.32609341e+00,
          2.59303948e+00,   5.82054847e-01],
       [  1.91345873e+01,   1.81698299e+01,   1.63200883e+01,
          1.31059554e+01,   9.00346539e+00,   5.64181399e+00,
          2.74914017e+00,   6.17098967e-01],
       [  1.93193626e+01,   1.84392866e+01,   1.66869996e+01,
          1.35603096e+01,   9.44566173e+00,   5.96079822e+00,
          2.90702027e+00,   6.52542875e-01],
       [  1.94739543e+01,   1.86790060e+01,   1.70273194e+01,
          1.39942958e+01,   9.87830343e+00,   6.27649452e+00,
          3.06348991e+00,   6.87670545e-01],
       [  1.95945372e+01,   1.88839740e+01,   1.73356247e+01,
          1.44029384e+01,   1.02975418e+01,   6.58642788e+00,
          3.21734206e+00,   7.22211032e-01],
       [  1.96887424e+01,   1.90601614e+01,   1.76163417e+01,
          1.47891267e+01,   1.07048744e+01,   6.89144450e+00,
          3.36898611e+00,   7.56256230e-01],
       [  1.97654357e+01,   1.92149177e+01,   1.78752097e+01,
          1.51568579e+01,   1.11025813e+01,   7.19287669e+00,
          3.51906778e+00,   7.89951076e-01],
       [  1.98280036e+01,   1.93506316e+01,   1.81136117e+01,
          1.55066972e+01,   1.14906182e+01,   7.49059990e+00,
          3.66752260e+00,   8.23281093e-01],
       [  1.98759361e+01,   1.94662250e+01,   1.83298971e+01,
          1.58368009e+01,   1.18672879e+01,   7.78344350e+00,
          3.81377547e+00,   8.56117177e-01],
       [  1.99101245e+01,   1.95621686e+01,   1.85238747e+01,
          1.61465006e+01,   1.22316808e+01,   8.07071284e+00,
          3.95748188e+00,   8.88381988e-01],
       [  1.99346954e+01,   1.96422202e+01,   1.86983936e+01,
          1.64375879e+01,   1.25845702e+01,   8.35275360e+00,
          4.09880309e+00,   9.20111710e-01],
       [  1.99539901e+01,   1.97105518e+01,   1.88567878e+01,
          1.67122946e+01,   1.29270486e+01,   8.63010556e+00,
          4.23799424e+00,   9.51363598e-01],
       [  1.99694135e+01,   1.97687301e+01,   1.90001708e+01,
          1.69711270e+01,   1.32590479e+01,   8.90257760e+00,
          4.37495375e+00,   9.82114837e-01],
       [  1.99801567e+01,   1.98163574e+01,   1.91279944e+01,
          1.72132985e+01,   1.35796230e+01,   9.16942145e+00,
          4.50930778e+00,   1.01228149e+00],
       [  1.99862887e+01,   1.98539620e+01,   1.92406280e+01,
          1.74387868e+01,   1.38883536e+01,   9.43020686e+00,
          4.64083644e+00,   1.04181419e+00],
       [  1.99899347e+01,   1.98840852e+01,   1.93403345e+01,
          1.76491253e+01,   1.41858944e+01,   9.68516804e+00,
          4.76964272e+00,   1.07073602e+00],
       [  1.99933051e+01,   1.99093878e+01,   1.94295728e+01,
          1.78460565e+01,   1.44730645e+01,   9.93463854e+00,
          4.89587746e+00,   1.09908082e+00],
       [  1.99965806e+01,   1.99306165e+01,   1.95091656e+01,
          1.80300647e+01,   1.47498442e+01,   1.01784245e+01,
          5.01943400e+00,   1.12682463e+00],
       [  1.99985080e+01,   1.99471402e+01,   1.95787604e+01,
          1.82007333e+01,   1.50156123e+01,   1.04159511e+01,
          5.14001953e+00,   1.15390169e+00],
       [  1.99986875e+01,   1.99590988e+01,   1.96387407e+01,
          1.83582686e+01,   1.52701838e+01,   1.06469109e+01,
          5.25747076e+00,   1.18027534e+00],
       [  1.99983823e+01,   1.99681637e+01,   1.96909009e+01,
          1.85040435e+01,   1.55141877e+01,   1.08715018e+01,
          5.37187013e+00,   1.20596404e+00],
       [  1.99989230e+01,   1.99760226e+01,   1.97370839e+01,
          1.86395136e+01,   1.57483324e+01,   1.10899707e+01,
          5.48332354e+00,   1.23099153e+00],
       [  2.00000677e+01,   1.99828219e+01,   1.97777747e+01,
          1.87650883e+01,   1.59726329e+01,   1.13021313e+01,
          5.59172561e+00,   1.25533414e+00],
       [  2.00005537e+01,   1.99876858e+01,   1.98125510e+01,
          1.88804818e+01,   1.61866455e+01,   1.15075096e+01,
          5.69683041e+00,   1.27893664e+00],
       [  1.99999608e+01,   1.99904839e+01,   1.98416561e+01,
          1.89859652e+01,   1.63903212e+01,   1.17058767e+01,
          5.79851117e+00,   1.30177053e+00],
       [  1.99992867e+01,   1.99923831e+01,   1.98664900e+01,
          1.90827602e+01,   1.65842774e+01,   1.18974209e+01,
          5.89684413e+00,   1.32385291e+00],
       [  1.99995358e+01,   1.99945183e+01,   1.98884168e+01,
          1.91720879e+01,   1.67691522e+01,   1.20823460e+01,
          5.99191295e+00,   1.34520251e+00],
       [  2.00003627e+01,   1.99967116e+01,   1.99076119e+01,
          1.92542424e+01,   1.69449702e+01,   1.22604753e+01,
          6.08361601e+00,   1.36579651e+00],
       [  2.00006244e+01,   1.99979979e+01,   1.99235256e+01,
          1.93289533e+01,   1.71113852e+01,   1.24314031e+01,
          6.17174012e+00,   1.38558701e+00],
       [  1.99999963e+01,   1.99981600e+01,   1.99362544e+01,
          1.93964768e+01,   1.72684242e+01,   1.25949591e+01,
          6.25618673e+00,   1.40455186e+00],
       [  1.99993890e+01,   1.99981307e+01,   1.99469016e+01,
          1.94578847e+01,   1.74166857e+01,   1.27513338e+01,
          6.33703331e+00,   1.42270842e+00],
       [  1.99996465e+01,   1.99987649e+01,   1.99564797e+01,
          1.95141870e+01,   1.75567431e+01,   1.29007077e+01,
          6.41435249e+00,   1.44007297e+00],
       [  2.00003842e+01,   1.99997414e+01,   1.99649264e+01,
          1.95655418e+01,   1.76886034e+01,   1.30429129e+01,
          6.48804774e+00,   1.45662381e+00],
       [  2.00005644e+01,   2.00001084e+01,   1.99715908e+01,
          1.96116375e+01,   1.78119648e+01,   1.31775942e+01,
          6.55793140e+00,   1.47231878e+00],
       [  1.99999539e+01,   1.99996671e+01,   1.99764717e+01,
          1.96526797e+01,   1.79268879e+01,   1.33046281e+01,
          6.62392921e+00,   1.48714120e+00],
       [  1.99994234e+01,   1.99992588e+01,   1.99804787e+01,
          1.96895985e+01,   1.80339391e+01,   1.34242136e+01,
          6.68612432e+00,   1.50110969e+00],
       [  1.99997093e+01,   1.99995936e+01,   1.99843859e+01,
          1.97232140e+01,   1.81336214e+01,   1.35365177e+01,
          6.74458491e+00,   1.51423953e+00],
       [  2.00003941e+01,   2.00002862e+01,   1.99879698e+01,
          1.97535442e+01,   1.82259003e+01,   1.36413798e+01,
          6.79922012e+00,   1.52651032e+00],
       [  2.00005147e+01,   2.00004305e+01,   1.99905237e+01,
          1.97802122e+01,   1.83104793e+01,   1.37384831e+01,
          6.84986370e+00,   1.53788470e+00],
       [  1.99999140e+01,   1.99998818e+01,   1.99920061e+01,
          1.98033614e+01,   1.83874225e+01,   1.38277441e+01,
          6.89646362e+00,   1.54835097e+00],
       [  1.99994438e+01,   1.99994494e+01,   1.99932167e+01,
          1.98237965e+01,   1.84572518e+01,   1.39093735e+01,
          6.93911184e+00,   1.55792975e+00],
       [  1.99997591e+01,   1.99997563e+01,   1.99947775e+01,
          1.98421700e+01,   1.85203916e+01,   1.39835304e+01,
          6.97787610e+00,   1.56663624e+00],
       [  2.00004073e+01,   2.00003757e+01,   1.99963615e+01,
          1.98583648e+01,   1.85767467e+01,   1.40500592e+01,
          7.01267151e+00,   1.57445135e+00],
       [  2.00004768e+01,   2.00004454e+01,   1.99972421e+01,
          1.98719306e+01,   1.86259978e+01,   1.41086735e+01,
          7.04335044e+00,   1.58134194e+00],
       [  1.99998769e+01,   1.99998798e+01,   1.99973793e+01,
          1.98829503e+01,   1.86681918e+01,   1.41593249e+01,
          7.06988202e+00,   1.58730105e+00],
       [  1.99994552e+01,   1.99994818e+01,   1.99975210e+01,
          1.98921215e+01,   1.87037974e+01,   1.42022384e+01,
          7.09236937e+00,   1.59235183e+00],
       [  1.99998018e+01,   1.99998107e+01,   1.99981922e+01,
          1.98999496e+01,   1.87331546e+01,   1.42375688e+01,
          7.11088248e+00,   1.59650997e+00],
       [  2.00004249e+01,   2.00004013e+01,   1.99989956e+01,
          1.99061888e+01,   1.87560932e+01,   1.42651629e+01,
          7.12534221e+00,   1.59975771e+00],
       [  2.00004476e+01,   2.00004232e+01,   1.99992023e+01,
          1.99103129e+01,   1.87722527e+01,   1.42847588e+01,
          7.13561753e+00,   1.60206561e+00],
       [  1.99998402e+01,   1.99998483e+01,   1.99987965e+01,
          1.99123488e+01,   1.87816499e+01,   1.42963407e+01,
          7.14169820e+00,   1.60343138e+00],
       [  1.99994588e+01,   1.99994872e+01,   1.99985089e+01,
          1.99129033e+01,   1.87846978e+01,   1.43001503e+01,
          7.14370042e+00,   1.60388110e+00]])

    reader = textwrap.dedent("""\

##############
##Speccon1dVR input (et=0 in 2nd layer)
#H = 1
#drn = 1
#dT = 1
#dTh = 5
#dTv = 0.1 * 0.25
#neig = 40
#
#mvref = 2.0
#kvref = 1.0
#khref = 1.0
#etref = 1.0
#
#
#mv = PolyLine([0,1], [0.5,0.5])
#kh = PolyLine([0,1], [1,1])
#kv = PolyLine([0,1], [5,5])
#et = PolyLine([0,0.5, 0.5, 1], [1,1, 0,0])
#surcharge_vs_depth = [PolyLine([0,1], [1,1]), PolyLine([0,1], [1,1])]
#surcharge_vs_time = [PolyLine([0,0,10], [0,10,10]), PolyLine([0,0,10], [0,10,10])]
#
#ppress_z = np.linspace(0,1,100)
#avg_ppress_z_pairs = [[0,1]]
#settlement_z_pairs = [[0,1]]
#
#tvals = np.logspace(-2, 0.3,50)
#ppress_z_tval_indexes = np.arange(len(tvals))[::len(tvals)//7]
###############

H = 1
drn = 1
dTh = 5
dTv = 0.1 * 0.25
dTw = 100000
neig = 200


mvref = 2.0
kvref = 1.0
khref = 1.0
etref = 1.0
kwref = 1.0

kw = PolyLine([0, 0.5, 0.5, 1.0], [1, 1, 0,0])
mv = PolyLine([0,1], [0.5,0.5])
kh = PolyLine([0,1], [1,1])
kv = PolyLine([0,1], [5,5])

et = PolyLine([0,1], [1,1])
#et = PolyLine([0, 0.5, 0.5, 1.0], [1, 1, 0,0])
surcharge_vs_depth = [PolyLine([0,1], [1,1]), PolyLine([0,1], [1,1])]
surcharge_vs_time = [PolyLine([0,0,10], [0,10,10]), PolyLine([0,0,10], [0,10,10])]


ppress_z = np.%s
avg_ppress_z_pairs = [[0,1]]
settlement_z_pairs = [[0,1]]
tvals = np.%s
ppress_z_tval_indexes = np.arange(len(tvals))[::len(tvals)//7]


    """ % (repr(z), repr(t)))

    for impl in ["vectorized"]:
        for dT in [0.1]:
            a = Speccon1dVRW(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()



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
                            err_msg = ("Fail. two_layers_no_drain_in_bottom_layer, avp, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.por, por, atol=1,
                            err_msg = ("Fail. two_layers_no_drain_in_bottom_layer, por, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=1,
                            err_msg = ("Fail. two_layers_no_drain_in_bottom_layer, settle, "
                                "implementation='%s', dT=%s" % (impl, dT)))


def test_dengetal2013():
    """
    deng et al uses an approximation for the well resistance term
    I use an 'exact method'.  The tolerance for checking is very crude
    but really it is for ball park figures.

    """

    t = np.array(
      [11025.,  110250.])

    z = np.array(
      [0.05, 0.1, 0.2, 0.5, 0.8, 1.0])

    por_99 = np.array(
    [[ 0.81168066,  0.12412292],
       [ 0.83668852,  0.16812724],
       [ 0.87096244,  0.25118534],
       [ 0.92073203,  0.43785721],
       [ 0.94257216,  0.55353625],
       [ 0.95050775,  0.60194473]])


    por_0=np.array([[ 0.81096427,  0.12303176],
       [ 0.83451179,  0.16380411],
       [ 0.86536448,  0.23549986],
       [ 0.90572564,  0.37150668],
       [ 0.91883398,  0.42891426],
       [ 0.92092841,  0.43879202]])

    por=por_99
    reader = textwrap.dedent("""\

##############
##dengetal2013 input
#dengetal2013(z=np.array([0.05, 0.1, 0.2, 0.5, 0.8, 1.0])*20,
#                             t=[11025.,  110250.],
#                             rw=0.035, re=0.525,
#                             A1=1, A2=0.99, A3=9.07029478e-06,
#                             H=20,
#                             rs=0.175,
#                             ks=2e-8/1.8,
#                             kw0=1e-3,
#                             kh=2e-8,
#                             mv=0.2e-3,
#                             gamw=10,
#                             ui=1)
###############

H = 20
drn = 1
#re=0.525, rw=0.035, rs=0.175, kh/ks=1.8, n=15, s=5, kap=1.8
#mu=3.18131104929, eta = 2/re**2/mu=2.28089479942

mvref = 0.2e-3
khref = 2e-8
etref = 2.28089479942
kwref = 1e-3

dTh=khref/mvref*etref/10
dTw=kwref/H**2/mvref/10 / (15**2-1)
#dTw = 100000
neig = 40




kw = PolyLine([0, 1], [1, 0.01])
#kw = PolyLine([0, 1], [1, 1])
mv = PolyLine([0,1], [1,1])
kh = PolyLine([0,1], [1,1])
#kv = PolyLine([0,1], [5,5])

et = PolyLine([0,1], [1,1])
surcharge_vs_depth = [PolyLine([0,1], [1,1])]
surcharge_vs_time = [PolyLine([0,0,10], [0,1,1])]


ppress_z = np.%s
tvals = np.%s


    """ % (repr(z), repr(t)))

    for impl in ["vectorized"]:
        for dT in [0.1]:
            a = Speccon1dVRW(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()



#            plt.figure()
#            plt.plot(por, z,'b-*', label='expected')
#            plt.plot(a.por, z, 'r-+', label='calculated')
#            plt.legend()


#            plt.figure()
#            plt.plot(t,settle[0],'b-*', label='expected')
#            plt.plot(t, a.set[0], 'r-+', label='calculated')
#            plt.legend()
#            plt.figure()
#            plt.plot(t, avp[0],'b-*',  label='expected')
#            plt.plot(t, a.avp[0], 'r-+', label='calculated')
#            plt.legend()
            plt.show()
#            assert_allclose(a.avp, avp, atol=1,
#                            err_msg = ("Fail. dengetal2013, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.por, por, atol=0.1,
                            err_msg = ("Fail. dengetal2013, por, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.set, settle, atol=1,
#                            err_msg = ("Fail. dengetal2013, settle, "
#                                "implementation='%s', dT=%s" % (impl, dT)))


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])

#    test_BC_terzaghi_1d_PTIB()
#    test_terzaghi_1d_PTPB()
#    test_schiffman_and_stein_1970()

#    print(np.append(0.5*TERZ1D_Z, 1-0.5*TERZ1D_Z[::-1]))
#    test_terzaghi_1d()

#    test_fixed_ppress_BC_terzaghi_PTPB()

#    test_hansbo_avp_vacuum()
#    test_terzaghi_1d_PTPB_bot_BC_gradient()
#    test_terzaghi_1d_pumping()
#    test_tang_and_onitsuka_vert_and_radial()
#    test_nogamiandli2003_lam_5()
#    test_nogamiandli2003_lam_100()
#    test_zhuandyin2012_drn0_kv_linear_mv_const()
#    test_zhuandyin2012_drn1_kv_linear_mv_const()
#    test_zhuandyin2012_drn0_kv_const_mv_linear()
#    test_zhuandyin2012_drn1_kv_const_mv_linear()
#    test_zhuandyin2012_drn0_kv_linear_mv_linear()
#    test_zhuandyin2012_drn1_kv_linear_mv_linear()
#    test_zhuandyin2012_drn0_kv_mv_non_linear()
#    test_zhuandyin2012_drn1_kv_mv_non_linear()
#    test_zhuandyin2012_drn0_kv_mv_non_linear_BC()
#    test_schiffman_and_stein_1970()
#    test_tang_and_onitsuka_vert_and_radial_well_resistance()
#    test_two_layers_no_drain_in_bottom_layer()
#    test_dengetal2013()