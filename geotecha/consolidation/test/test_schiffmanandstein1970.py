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

from geotecha.consolidation.schiffmanandstein1970 import SchiffmanAndStein1970


def test_schiffmanandstein1970_one():
    """test for multilayer vertical consolidation

    example as per Schiffman and stein 1970

    """


    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np

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

    n = 20
    surcharge_vs_time = PolyLine([0,0,10], [0,100,100])
    z = np.concatenate((np.linspace(0, np.sum(h[:1]), 25, endpoint=False),
                        np.linspace(np.sum(h[:1]), np.sum(h[:2]), 25, endpoint=False),
                        np.linspace(np.sum(h[:2]), np.sum(h[:3]), 25, endpoint=False),
                        np.linspace(np.sum(h[:3]), np.sum(h), 25, endpoint=True)))



    tpor = np.array([740, 2930, 7195], dtype=float)

    z = np.array(
      [  0.        ,   1.        ,   2.        ,   3.        ,
         4.        ,   5.        ,   6.        ,   7.        ,
         8.        ,   9.        ,  10.        ,  12.        ,
        14.        ,  16.        ,  18.        ,  20.        ,
        22.        ,  24.        ,  26.        ,  28.        ,
        30.        ,  33.        ,  36.        ,  39.        ,
        42.        ,  45.        ,  48.        ,  51.        ,
        54.        ,  57.        ,  60.        ,  62.22222222,
        64.44444444,  66.66666667,  68.88888889,  71.11111111,
        73.33333333,  75.55555556,  77.77777778,  80.        ])

    t=np.array(
        [1.21957046e+02,   1.61026203e+02,   2.12611233e+02,
         2.80721620e+02,   3.70651291e+02,   4.89390092e+02,
         740.0,   8.53167852e+02,   1.12648169e+03,
         1.48735211e+03,   1.96382800e+03,   2930.0,
         3.42359796e+03,   4.52035366e+03,   5.96845700e+03,
         7195.0,   1.04049831e+04,   1.37382380e+04,
         1.81393069e+04,   2.39502662e+04,   3.16227766e+04])
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
      [  0.        ,   1.        ,   2.        ,   3.        ,
         4.        ,   5.        ,   6.        ,   7.        ,
         8.        ,   9.        ,  10.        ,  12.        ,
        14.        ,  16.        ,  18.        ,  20.        ,
        22.        ,  24.        ,  26.        ,  28.        ,
        30.        ,  33.        ,  36.        ,  39.        ,
        42.        ,  45.        ,  48.        ,  51.        ,
        54.        ,  57.        ,  60.        ,  62.22222222,
        64.44444444,  66.66666667,  68.88888889,  71.11111111,
        73.33333333,  75.55555556,  77.77777778,  80.        ])


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


    a = SchiffmanAndStein1970(reader)

    a.make_all()

#    plt.figure()
#    plt.plot(por, z,'b-*')
#    plt.plot(a.por, z, 'r-+')
#
#
#    plt.figure()
#    plt.plot(t,settle[0],'b-*')
#    plt.plot(t, a.set, 'r-+')
#    plt.figure()
#    plt.plot(t, avp[0],'b-*')
#    plt.plot(t, a.avp, 'r-+')
#    plt.show()


    assert_allclose(a.por, por, atol=1e-3,
                    err_msg = ("Fail. test_schiffmanandstein1970_one, por, "))
    assert_allclose(a.avp, avp[0], atol=1e-3,
                    err_msg = ("Fail. test_schiffmanandstein1970_one, avp, "))
    assert_allclose(a.set, settle[0], atol=1e-3,
                    err_msg = ("Fail. test_schiffmanandstein1970_one, set, "))

if __name__ == '__main__':
#    import nose
#    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])

    test_schiffmanandstein1970_one()