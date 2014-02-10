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


from math import pi
import numpy as np
import textwrap
import matplotlib.pyplot as plt
from geotecha.piecewise.piecewise_linear_1d import PolyLine

from geotecha.speccon.speccon1d_vert_radial_boundary import speccon1d_vr

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
    from geotecha.piecewise.piecewise_linear_1d import PolyLine
    import numpy as np
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
            a = speccon1d_vr(reader + "\n" +
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
    from geotecha.piecewise.piecewise_linear_1d import PolyLine
    import numpy as np
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
            a = speccon1d_vr(reader + "\n" +
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
    from geotecha.piecewise.piecewise_linear_1d import PolyLine
    import numpy as np
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
            a = speccon1d_vr(reader + "\n" +
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
    from geotecha.piecewise.piecewise_linear_1d import PolyLine
    import numpy as np
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
            a = speccon1d_vr(reader + "\n" +
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
    from geotecha.piecewise.piecewise_linear_1d import PolyLine
    import numpy as np

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
            a = speccon1d_vr(reader + "\n" +
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
    from geotecha.piecewise.piecewise_linear_1d import PolyLine
    import numpy as np
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
            a = speccon1d_vr(reader + "\n" +
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
    from geotecha.piecewise.piecewise_linear_1d import PolyLine
    import numpy as np
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
            a = speccon1d_vr(reader + "\n" +
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
    from geotecha.piecewise.piecewise_linear_1d import PolyLine
    import numpy as np
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
            a = speccon1d_vr(reader + "\n" +
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
    from geotecha.piecewise.piecewise_linear_1d import PolyLine
    import numpy as np
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
            a = speccon1d_vr(reader + "\n" +
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

if __name__ == '__main__':
#    test_terzaghi_1d_PTPB()
#    test_schiffman_and_stein_1970()

#    print(np.append(0.5*TERZ1D_Z, 1-0.5*TERZ1D_Z[::-1]))
#    test_terzaghi_1d()

#    test_fixed_ppress_BC_terzaghi_PTPB()

#    test_hansbo_avp_vacuum()
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])