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
"""Some test routines for the xieandleo2004 module.

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
from geotecha.consolidation.xieandleo2004 import XieAndLeo2004

class test_xieandleo2004(unittest.TestCase):

    def test_figure4_Us_equals_0p5_digitized(self):


        # this one is a bit difficult. I digitized fig 4 from xie and leo for
        # the line Us=0.5.  I then used xie and leo to back calulate the a
        # values corresponding to the digitized point.  then used those a values
        # to check the corresponding pore pressure values.

        qu=100
        qp=10
        H=10
        Hw=1.0
        kv0=1e-9
        mvl=4e-3
        e00=3
        Gs=2.75
        gamw=10 #N
        drn=1
        nterms=100

        obj = XieAndLeo2004(qu=qu, qp=qp, H=H, Hw=Hw,
                        kv0=kv0, mvl=mvl, e00=e00, Gs=Gs, gamw=gamw,
                        drn=drn, nterms=nterms)



        a = np.array(
      [ 0.02383584,  0.11392357,  0.18570972,  0.27481166,  0.3635012 ,
        0.45172817,  0.52204832,  0.60948048,  0.69635416,  0.78283411,
        0.86889152,  0.95449232,  1.03972932,  1.12456789,  1.20894234,
        1.29285641,  1.39301393,  1.47614686,  1.57530062,  1.65757701,
        1.73956316,  1.83732833,  1.91850293,  2.01533737,  2.09568643,
        2.19164531,  2.28713899,  2.38216436,  2.47675735,  2.57091444,
        2.66463215,  2.75794547,  2.85079954,  2.94330538,  3.05064294,
        3.1422562 ,  3.23354725,  3.33944257,  3.4449005 ,  3.54984556,
        3.63948794,  3.74356528,  3.84724822,  3.9357537 ,  4.03859903,
        4.15562183,  4.25758871,  4.35912844,  4.46032271,  4.57547209,
        4.67584576,  4.79012932,  4.90392578,  5.00317637,  5.11617632,
        5.22877168,  5.34102057,  5.45280693,  5.56423249,  5.67529646,
        5.78598068,  5.91011626,  6.02009368,  6.12976985,  6.25277697,
        6.37539618,  6.49764812,  6.61952496,  6.74112884,  6.8489472 ,
        6.96997937,  7.10401376,  7.22435667,  7.3444893 ,  7.4643678 ,
        7.59723099,  7.71657677,  7.84896193,  7.96788913,  8.09978589,
        8.23143334,  8.34976394,  8.48109602,  8.61220892,  8.74316219,
        8.86088196,  8.99160287,  9.1221999 ,  9.25266276,  9.60452413,
        9.99508   ])

        u = np.array(
      [[  0.436047,   1.74419 ,   3.05233 ,   4.21512 ,   5.52326 ,
         6.8314  ,   7.99419 ,   9.30233 ,  10.4651  ,  11.7733  ,
        13.0814  ,  14.2442  ,  15.5523  ,  16.8605  ,  18.0233  ,
        19.186   ,  20.3488  ,  21.657   ,  22.8198  ,  23.9826  ,
        25.2907  ,  26.4535  ,  27.6163  ,  28.7791  ,  29.9419  ,
        31.1047  ,  32.2674  ,  33.4302  ,  34.593   ,  35.7558  ,
        36.9186  ,  38.0814  ,  39.0988  ,  40.2616  ,  41.4244  ,
        42.4419  ,  43.6047  ,  44.6221  ,  45.7849  ,  46.8023  ,
        47.8198  ,  48.8372  ,  50.      ,  50.8721  ,  51.8895  ,
        53.0523  ,  54.0698  ,  54.9419  ,  55.9593  ,  56.9767  ,
        57.8488  ,  58.8663  ,  59.7384  ,  60.6105  ,  61.4826  ,
        62.3547  ,  63.3721  ,  64.0988  ,  64.9709  ,  65.843   ,
        66.5698  ,  67.4419  ,  68.1686  ,  68.8953  ,  69.6221  ,
        70.3488  ,  71.0756  ,  71.657   ,  72.3837  ,  72.9651  ,
        73.6919  ,  74.2733  ,  74.7093  ,  75.2907  ,  75.8721  ,
        76.3081  ,  76.7442  ,  77.3256  ,  77.7616  ,  78.1977  ,
        78.4884  ,  78.7791  ,  79.2151  ,  79.5058  ,  79.7965  ,
        79.9419  ,  80.2326  ,  80.5233  ,  80.6686  ,  80.9593  ,  81.1047  ]])


        assert_allclose(obj.u_PTIB(a, np.array([7.87244369e+08])), u.T,
                        atol=0.5)

        #PTIB
        #Us=[ 0.    0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8   0.9   0.99]
        #t=[  4.00000000e+05   3.14230943e+07   1.25722649e+08   2.82747780e+08
        #   5.03035106e+08   7.87244369e+08   1.14671830e+09   1.61311241e+09
        #   2.26872283e+09   3.39649631e+09   7.12604069e+09]



    def test_figure5_Us_equals_0p5_digitized(self):


        # this one is a bit difficult. I digitized fig 5 from xie and leo for
        # the line Us=0.5.  I then used xie and leo to back calulate the a
        # values corresponding to the digitized point.  then used those a values
        # to check the corresponding pore pressure values.

        qu=100
        qp=10
        H=10
        Hw=1.0
        kv0=1e-9
        mvl=4e-3
        e00=3
        Gs=2.75
        gamw=10 #N
        drn=0
        nterms=100

        obj = XieAndLeo2004(qu=qu, qp=qp, H=H, Hw=Hw,
                        kv0=kv0, mvl=mvl, e00=e00, Gs=Gs, gamw=gamw,
                        drn=drn, nterms=nterms)



        a = np.array(
      [ 0.02289084,  0.08166019,  0.1592776 ,  0.21712845,  0.27446339,
        0.35034062,  0.4069592 ,  0.46294925,  0.53722803,  0.61096618,
        0.68392617,  0.73829586,  0.81042068,  0.88178646,  0.95268288,
        1.02315097,  1.09302059,  1.17965214,  1.24844496,  1.31687808,
        1.40163487,  1.46891747,  1.55252576,  1.61904599,  1.70148456,
        1.78334316,  1.86467033,  1.94548742,  2.04176209,  2.12149695,
        2.21645107,  2.31074839,  2.404423  ,  2.49750554,  2.60541511,
        2.69730391,  2.80389126,  2.90982761,  3.01515179,  3.11990136,
        3.22411273,  3.34259041,  3.46044316,  3.57775453,  3.69456873,
        3.82545626,  3.95579799,  4.08567035,  4.21517543,  4.35866226,
        4.48749166,  4.63039646,  4.75883777,  4.90142733,  5.04396904,
        5.18653616,  5.32919711,  5.45773091,  5.60080211,  5.72981954,
        5.87354154,  6.00331129,  6.13349986,  6.24963259,  6.38083102,
        6.49798246,  6.61566338,  6.73391895,  6.83791618,  6.94243466,
        7.04750993,  7.16829246,  7.25947883,  7.36644948,  7.45874141,
        7.55158307,  7.64500296,  7.73903078,  7.83370941,  7.92904715,
        8.009015  ,  8.08945757,  8.18673931,  8.2684616 ,  8.33428654,
        8.41702202,  8.50034747,  8.56751182,  8.65209349,  8.72022258,
        8.80599002,  8.87514782,  8.94487601,  9.01500734,  9.08561288,
        9.15692164,  9.22868735,  9.30092662,  9.37388661,  9.42900301,
        9.50288862,  9.55887868,  9.63397705,  9.69063019,  9.74788885,
        9.82465744])

        u = np.array(
      [[  1.24575,   3.06473,   5.086  ,   6.90498,   8.92598,  10.7452 ,
         12.7662 ,  14.5852 ,  16.4045 ,  18.4257 ,  20.245  ,  22.064  ,
         23.8832 ,  25.9045 ,  27.7237 ,  29.543  ,  31.3622 ,  33.1818 ,
         35.001  ,  36.8203 ,  38.6398 ,  40.459  ,  42.2785 ,  44.0978 ,
         45.7153 ,  47.5348 ,  49.1523 ,  50.9718 ,  52.5896 ,  54.2071 ,
         55.8248 ,  57.4426 ,  59.0604 ,  60.6781 ,  62.2961 ,  63.9139 ,
         65.3299 ,  66.7459 ,  68.1619 ,  69.5779 ,  70.7919 ,  72.2082 ,
         73.4225 ,  74.6367 ,  75.649  ,  76.6615 ,  77.4719 ,  78.2824 ,
         79.0929 ,  79.7016 ,  80.3101 ,  80.5148 ,  80.9212 ,  80.9239 ,
         81.1286 ,  80.9292 ,  80.7299 ,  80.3282 ,  79.9269 ,  79.3232 ,
         78.7198 ,  77.9141 ,  76.9064 ,  75.8985 ,  74.8908 ,  73.6808 ,
         72.4708 ,  71.2608 ,  70.0506 ,  68.8403 ,  67.428  ,  65.814  ,
         64.4015 ,  62.7872 ,  61.3746 ,  59.7601 ,  58.1455 ,  56.531  ,
         54.9164 ,  53.0998 ,  51.485  ,  49.8702 ,  48.0536 ,  46.2367 ,
         44.6216 ,  42.8048 ,  40.988  ,  39.1708 ,  37.354  ,  35.5369 ,
         33.72   ,  31.9029 ,  30.0858 ,  28.2687 ,  26.4516 ,  24.6345 ,
         22.6153 ,  20.7982 ,  18.9811 ,  17.1637 ,  15.1446 ,  13.3272 ,
         11.3081 ,   9.49068,   7.6733 ,   5.65416]])


        assert_allclose(obj.u_PTPB(a, np.array([1.96880554e+08])), u.T,
                        atol=0.9)

        # PTPB
        #Us=[ 0.    0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8   0.9   0.99]
        #t=[  4.00000000e+05   7.85909971e+06   3.14230943e+07   7.07262313e+07
        #   1.25732999e+08   1.96880554e+08   2.86531909e+08   4.03382401e+08
        #   5.68057774e+08   8.50440199e+08   1.79007642e+09]




    def test_fig6_PTIB_Us(self):

        qu=100
        qp=10
        H=10
        Hw=1.0
        kv0=1e-9
        mvl=4e-3
        e00=3
        Gs=2.75
        gamw=10 #N
        drn=1
        nterms=100

        obj = XieAndLeo2004(qu=qu, qp=qp, H=H, Hw=Hw,
                        kv0=kv0, mvl=mvl, e00=e00, Gs=Gs, gamw=gamw,
                        drn=drn, nterms=nterms)

        t = np.array([  4.00000000e+06,   9.30767268e+06,   2.16581927e+07,
         5.03968420e+07,   1.17269327e+08,   2.72876128e+08,
         6.34960421e+08,   1.47750094e+09,   3.43802378e+09,
         8.00000000e+09])

        Us = np.array([ 0.03568248,  0.05443091,  0.08303022,  0.12665628,  0.19320452,
        0.29471879,  0.44946299,  0.67415692,  0.90277857,  0.99417048])

        assert_allclose(obj.Us_PTIB(t), Us)

    def test_fig6_PTIB_Up(self):

        qu=100
        qp=10
        H=10
        Hw=1.0
        kv0=1e-9
        mvl=4e-3
        e00=3
        Gs=2.75
        gamw=10 #N
        drn=1
        nterms=100

        obj = XieAndLeo2004(qu=qu, qp=qp, H=H, Hw=Hw,
                        kv0=kv0, mvl=mvl, e00=e00, Gs=Gs, gamw=gamw,
                        drn=drn, nterms=nterms)

        t = np.array([  4.00000000e+06,   9.30767268e+06,   2.16581927e+07,
         5.03968420e+07,   1.17269327e+08,   2.72876128e+08,
         6.34960421e+08,   1.47750094e+09,   3.43802378e+09,
         8.00000000e+09])

        Up = np.array([ 0.0328077 ,  0.05004566,  0.07634084,  0.11645214,  0.17763889,
        0.27097897,  0.4141635 ,  0.63411239,  0.88384332,  0.99284489])

        assert_allclose(obj.Up_PTIB(t), Up)


    def test_fig6_PTIB_settlement(self):

        qu=100
        qp=10
        H=10
        Hw=1.0
        kv0=1e-9
        mvl=4e-3
        e00=3
        Gs=2.75
        gamw=10 #N
        drn=1
        nterms=100

        obj = XieAndLeo2004(qu=qu, qp=qp, H=H, Hw=Hw,
                        kv0=kv0, mvl=mvl, e00=e00, Gs=Gs, gamw=gamw,
                        drn=drn, nterms=nterms)

        t = np.array([  4.00000000e+06,   9.30767268e+06,   2.16581927e+07,
         5.03968420e+07,   1.17269327e+08,   2.72876128e+08,
         6.34960421e+08,   1.47750094e+09,   3.43802378e+09,
         8.00000000e+09])

        settle = np.array([[ 0.11763799,  0.17944781,  0.27373398,  0.41756036,  0.63695657,
         0.97162877,  1.48178939,  2.22256023,  2.97627998,  3.27758078]])

        assert_allclose(obj.settlement_PTIB(np.array([0.0]),t), settle)


    def test_fig6_PTPB_Us(self):

        qu=100
        qp=10
        H=10
        Hw=1.0
        kv0=1e-9
        mvl=4e-3
        e00=3
        Gs=2.75
        gamw=10 #N
        drn=0
        nterms=100

        obj = XieAndLeo2004(qu=qu, qp=qp, H=H, Hw=Hw,
                        kv0=kv0, mvl=mvl, e00=e00, Gs=Gs, gamw=gamw,
                        drn=drn, nterms=nterms)

        t = np.array([  4.00000000e+06,   9.30767268e+06,   2.16581927e+07,
         5.03968420e+07,   1.17269327e+08,   2.72876128e+08,
         6.34960421e+08,   1.47750094e+09,   3.43802378e+09,
         8.00000000e+09])

        Us = np.array([ 0.07136496,  0.10886182,  0.16606043,  0.25331256,  0.38640131,
        0.58637871,  0.83080779,  0.97883922,  0.99983224,  1.        ])

        assert_allclose(obj.Us_PTPB(t), Us)

    def test_fig6_PTIB_Up(self):

        qu=100
        qp=10
        H=10
        Hw=1.0
        kv0=1e-9
        mvl=4e-3
        e00=3
        Gs=2.75
        gamw=10 #N
        drn=0
        nterms=100

        obj = XieAndLeo2004(qu=qu, qp=qp, H=H, Hw=Hw,
                        kv0=kv0, mvl=mvl, e00=e00, Gs=Gs, gamw=gamw,
                        drn=drn, nterms=nterms)

        t = np.array([  4.00000000e+06,   9.30767268e+06,   2.16581927e+07,
         5.03968420e+07,   1.17269327e+08,   2.72876128e+08,
         6.34960421e+08,   1.47750094e+09,   3.43802378e+09,
         8.00000000e+09])

        Up = np.array([ 0.06561541,  0.10009131,  0.15268168,  0.2329045 ,  0.35547031,
        0.54565008,  0.80191869,  0.97414702,  0.99979374,  1.        ])

        assert_allclose(obj.Up_PTPB(t), Up)


    def test_fig6_PTIB_settlement(self):

        qu=100
        qp=10
        H=10
        Hw=1.0
        kv0=1e-9
        mvl=4e-3
        e00=3
        Gs=2.75
        gamw=10 #N
        drn=0
        nterms=100

        obj = XieAndLeo2004(qu=qu, qp=qp, H=H, Hw=Hw,
                        kv0=kv0, mvl=mvl, e00=e00, Gs=Gs, gamw=gamw,
                        drn=drn, nterms=nterms)

        t = np.array([  4.00000000e+06,   9.30767268e+06,   2.16581927e+07,
         5.03968420e+07,   1.17269327e+08,   2.72876128e+08,
         6.34960421e+08,   1.47750094e+09,   3.43802378e+09,
         8.00000000e+09])

        settle = np.array([[ 0.23527598,  0.35889561,  0.54746796,  0.83512073,  1.27388766,
         1.93317306,  2.73900674,  3.22703668,  3.29624648,  3.29679953]])

        assert_allclose(obj.settlement_PTPB(np.array([0.0]),t), settle)




if __name__ == '__main__':

    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])