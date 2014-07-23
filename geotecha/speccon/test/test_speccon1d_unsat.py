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

from geotecha.speccon.speccon1d_unsat import Speccon1dUnsat

import geotecha.math.transformations as transformations


def test_shanetal2012_sig_ka_divide_kw_is_100():
    """test against for shan et al 2012

    fig2a and 5a

    Shan et al (2012) 'Exact solutions for one-dimensional consolidation
    of single-layer unsaturated soil'

    test data is digitized from article

    dsig=100kPa instantly
    ka/kw=100

    other data:
    n = 0.50
    S=0.80
    kw=10^10m/s
    m1kw=-0.5x10**4 kPa-1
    h=10m
    mw2=-2.0x10**4 kPa-1
    ma1k=-2.0x10**4 kPa-1
    ma2=1.0x10**4 kPa-1

    gamw= 10000N
    ua_=uatm=101kPa,
    R=8.31432J/molK
    t0 = 20 degrees C,
    T =(t0+273.16)K,
    wa=29x10**3 kg/mol

    Note to get (dua, duw) = (0.2, 0.4) * dsig need ua_=111kPa

    Note digitized settlement data does not include immediate
    undrained deformation. data is S/S0, S0=m1ks*q0*H

    """

    tua = np.array([ 282.785, 940.375, 2644.42, 4496.89, 8551.43,
                    13372.5, 20335.4, 29242.8, 40893.1, 51137.3, 94564.5])

    tuw = np.array([3864.8, 10030.2, 18091, 37532.2, 139897,
                    599326, 5619520, 52739300, 153442000,
                    254730000, 473013000, 1577810000])

    tset = np.array([1.37958, 33.341, 423.929, 3230.37, 8513.52,
                     32049.3, 305234, 6701640, 61859100,
                     239757000, 790152000])

    z = np.array([0.5])


    reader = textwrap.dedent("""\
#from geotecha.piecewise.piecewise_linear_1d import PolyLine
#import numpy as np
# shan et al,
H = 10 #m
drn = 0
neig = 20

mvref = 1e-4 #1/kPa
kwref = 1.0e-10 #m/s

karef = kwref * 100 #m/s
Daref = karef / 10 # from equation ka=Da*g

wa = 29.0e-3 #kg / mol
R = 8.31432 #J/(mol.K)
ua_= 111 #kPa
T = 273.16 + 20
dTa = Daref /(mvref) / (wa*ua_/(R*T))/ H ** 2
dTw = kwref / mvref/ 10 / H**2
dT = max(dTw,dTa)

kw = PolyLine([0,1], [1,1])
Da = PolyLine([0,1], [1,1])
S = PolyLine([0,1], [0.8] * 2)
n = PolyLine([0,1], [0.5] * 2)

m1kw = PolyLine([0,1], [-0.5]*2)
m2w =  PolyLine([0,1], [-2.0]*2)
m1ka = PolyLine([0,1], [-2.0]*2)
m2a =  PolyLine([0,1], [1.0]*2)


surcharge_vs_depth = PolyLine([0,1], [1,1])
surcharge_vs_time = PolyLine([0,0,10000], [0,100,100])

ppress_z = [0.5]
#avg_ppress_z_pairs = [[0,1]]
settlement_z_pairs = [[0,1]]
tvals = np.logspace(1, 10, 50)

tua = np.{tua}
tuw = np.{tuw}
tset = np.{tset}
tvals = np.hstack((tua,tuw, tset))


ppress_z_tval_indexes = slice(None, len(tua)+len(tuw))
#avg_ppress_z_pairs_tval_indexes = slice(None, None)#[0,4,6]
settlement_z_pairs_tval_indexes = slice(len(tua)+len(tuw),len(tua)+len(tuw)+len(tset))

save_data_to_file= False
save_figures_to_file= False
show_figures= False

    """.format(tua=repr(tua), tuw=repr(tuw), tset=repr(tset)))

    pora = 100 * np.array(
     [[0.200018, 0.200429, 0.196143, 0.182068, 0.142173,
      0.101492, 0.0619839, 0.0349948, 0.0162211, 0.00840019,
      0.000194669]])

    porw = 100 * np.array(
     [[0.389736, 0.346022, 0.30544, 0.263294, 0.249989, 0.249159,
      0.249864, 0.234187, 0.148347, 0.0898279, 0.0289645,
      0.0000625543]])
#
#    avp = np.array(
#      [[  7.25979052e+01,   6.65166314e+01,   5.89096834e+01,
#          4.94554633e+01,   3.79564622e+01,   2.66323138e+01,
#          2.50358034e+01,   1.28862133e+01,   4.44927613e+00,
#          1.18311566e+00,   8.09339892e-01,   5.26895921e-02]])

    m1kw = -0.5e-4
    m2w =  -2e-4
    m1ka = -2e-4
    m2a =  1e-4
    uai = 20
    uwi = 40
    dsig = 100
    h=10
    set0 = (m1ka + m1kw) * (dsig-uai) + (m2a + m2w) * (uai-uwi)
    set0*= h

    settle = set0 + dsig * h * (m1ka + m1kw) * np.array(
    [[0.00108303, 0.00595668, 0.0211191, 0.0611913, 0.0990975,
      0.162455, 0.182491, 0.192238, 0.220397, 0.255596,
      0.278339]])

    for impl in ["vectorized"]:
        for dT in [10]:
            a = Speccon1dUnsat(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()


#            plt.figure()
#            plt.plot(tua, pora[0],'b-*', label='expected')
#            plt.plot(tua, a.pora[0,:len(tua)], 'r-+', label='calculated')
#            plt.gca().set_xlabel('time')
#            plt.gca().set_ylabel('pora')
#            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.figure()
#            plt.plot(tuw, porw[0],'b-*', label='expected')
#            plt.plot(tuw, a.porw[0,len(tua):len(tua)+len(tuw)], 'r-+', label='calculated')
#            plt.gca().set_xlabel('time')
#            plt.gca().set_ylabel('porw')
#            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.figure()
#            plt.plot(tset, settle[0],'b-*', label='expected')
#            plt.plot(tset, a.set[0], 'r-+', label='calculated')
#            plt.gca().set_xlabel('time')
#            plt.gca().set_ylabel('set')
#            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            legend=plt.legend()
#            legend.draggable()
#
###            plt.legend.DraggableLegend()
###            plt.figure()
###            plt.plot(t, avp[0],'b-*',  label='expected')
###            plt.plot(t, a.avp[0], 'r-+', label='calculated')
###            plt.legend()
#            plt.show()


            assert_allclose(a.pora[:,:len(tua)], pora, atol=1.0,
                            err_msg = ("Fail. test_shanetal2012_sig_ka_divide_kw_is_100, pora, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.porw[:,len(tua):len(tua)+len(tuw)], porw, atol=2.0,
                            err_msg = ("Fail. test_shanetal2012_sig_ka_divide_kw_is_100, porw, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_shanetal2012_sig_ka_divide_kw_is_100, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=5e-3,
                            err_msg = ("Fail. test_shanetal2012_sig_ka_divide_kw_is_100, set, "
                                "implementation='%s', dT=%s" % (impl, dT)))

def test_shanetal2012_sig_ka_divide_kw_is_1():
    """test against for shan et al 2012

    fig2a and 5a

    Shan et al (2012) 'Exact solutions for one-dimensional consolidation
    of single-layer unsaturated soil'

    test data is digitized from article

    dsig=100kPa instantly
    ka/kw=1

    other data:
    n = 0.50
    S=0.80
    kw=10^10m/s
    m1kw=-0.5x10**4 kPa-1
    h=10m
    mw2=-2.0x10**4 kPa-1
    ma1k=-2.0x10**4 kPa-1
    ma2=1.0x10**4 kPa-1

    gamw= 10000N
    ua_=uatm=101kPa,
    R=8.31432J/molK
    t0 = 20 degrees C,
    T =(t0+273.16)K,
    wa=29x10**3 kg/mol

    Note to get (dua, duw) = (0.2, 0.4) * dsig need ua_=111kPa

    Note digitized settlement data does not include immediate
    undrained deformation. data is S/S0, S0=m1ks*q0*H

    """

    tua = np.array([ 273460, 687668, 1640000, 2700000,
                    5590000, 10600000.0])

    tuw = np.array([369548, 881444, 1990530, 4618420,
                    23411100, 69823400, 149176000, 240773000,
                    459882000, 1159740000.0])

    tset = np.array([269.655, 5203.88, 56356.6, 653911,
                     1841400, 4700680, 88201200,
                     239743000, 1766160000])

    z = np.array([0.5])


    reader = textwrap.dedent("""\
#from geotecha.piecewise.piecewise_linear_1d import PolyLine
#import numpy as np
# shan et al,
H = 10 #m
drn = 0
neig = 20

mvref = 1e-4 #1/kPa
kwref = 1.0e-10 #m/s

karef = kwref * 1 #m/s
Daref = karef / 10 # from equation ka=Da*g

wa = 29.0e-3 #kg / mol
R = 8.31432 #J/(mol.K)
ua_= 111 #kPa
T = 273.16 + 20
dTa = Daref /(mvref) / (wa*ua_/(R*T))/ H ** 2
dTw = kwref / mvref/ 10 / H**2
dT = max(dTw,dTa)

kw = PolyLine([0,1], [1,1])
Da = PolyLine([0,1], [1,1])
S = PolyLine([0,1], [0.8] * 2)
n = PolyLine([0,1], [0.5] * 2)

m1kw = PolyLine([0,1], [-0.5]*2)
m2w =  PolyLine([0,1], [-2.0]*2)
m1ka = PolyLine([0,1], [-2.0]*2)
m2a =  PolyLine([0,1], [1.0]*2)


surcharge_vs_depth = PolyLine([0,1], [1,1])
surcharge_vs_time = PolyLine([0,0,10000], [0,100,100])

ppress_z = [0.5]
#avg_ppress_z_pairs = [[0,1]]
settlement_z_pairs = [[0,1]]


tua = np.{tua}
tuw = np.{tuw}
tset = np.{tset}
tvals = np.hstack((tua,tuw, tset))


ppress_z_tval_indexes = slice(None, len(tua)+len(tuw))
#avg_ppress_z_pairs_tval_indexes = slice(None, None)#[0,4,6]
settlement_z_pairs_tval_indexes = slice(len(tua)+len(tuw),len(tua)+len(tuw)+len(tset))

save_data_to_file= False
save_figures_to_file= False
show_figures= False

    """.format(tua=repr(tua), tuw=repr(tuw), tset=repr(tset)))

    pora = 100 * np.array(
     [[0.196222, 0.158679, 0.0831852, 0.0428964, 0.00495883,
       0.000666146]])

    porw = 100 * np.array(
     [[0.390361, 0.35523, 0.297481, 0.25611, 0.248255,
       0.219357, 0.152249, 0.0968501, 0.0320856,
       0.00163305]])
#
#    avp = np.array(
#      [[  7.25979052e+01,   6.65166314e+01,   5.89096834e+01,
#          4.94554633e+01,   3.79564622e+01,   2.66323138e+01,
#          2.50358034e+01,   1.28862133e+01,   4.44927613e+00,
#          1.18311566e+00,   8.09339892e-01,   5.26895921e-02]])

    m1kw = -0.5e-4
    m2w =  -2e-4
    m1ka = -2e-4
    m2a =  1e-4
    uai = 20
    uwi = 40
    dsig = 100
    h=10
    set0 = (m1ka + m1kw) * (dsig-uai) + (m2a + m2w) * (uai-uwi)
    set0*= h

    settle = set0 + dsig * h * (m1ka + m1kw) * np.array(
        [[0.00162455, 0.00812274, 0.0265343,
          0.0882671, 0.144043, 0.184657, 0.229061,
          0.255054, 0.279964]])

    for impl in ["vectorized"]:
        for dT in [10]:
            a = Speccon1dUnsat(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()


#            plt.figure()
#            plt.plot(tua, pora[0],'b-*', label='expected')
#            plt.plot(tua, a.pora[0,:len(tua)], 'r-+', label='calculated')
#            plt.gca().set_xlabel('time')
#            plt.gca().set_ylabel('pora')
#            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.figure()
#            plt.plot(tuw, porw[0],'b-*', label='expected')
#            plt.plot(tuw, a.porw[0,len(tua):len(tua)+len(tuw)], 'r-+', label='calculated')
#            plt.gca().set_xlabel('time')
#            plt.gca().set_ylabel('porw')
#            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.figure()
#            plt.plot(tset, settle[0],'b-*', label='expected')
#            plt.plot(tset, a.set[0], 'r-+', label='calculated')
#            plt.gca().set_xlabel('time')
#            plt.gca().set_ylabel('set')
#            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            legend=plt.legend()
#            legend.draggable()

##            plt.legend.DraggableLegend()
##            plt.figure()
##            plt.plot(t, avp[0],'b-*',  label='expected')
##            plt.plot(t, a.avp[0], 'r-+', label='calculated')
##            plt.legend()
            plt.show()


            assert_allclose(a.pora[:,:len(tua)], pora, atol=1.0,
                            err_msg = ("Fail. test_shanetal2012_sig_ka_divide_kw_is_1, pora, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.porw[:,len(tua):len(tua)+len(tuw)], porw, atol=2.0,
                            err_msg = ("Fail. test_shanetal2012_sig_ka_divide_kw_is_1, porw, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_shanetal2012_sig_ka_divide_kw_is_1, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=5e-3,
                            err_msg = ("Fail. test_shanetal2012_sig_ka_divide_kw_is_1, set, "
                                "implementation='%s', dT=%s" % (impl, dT)))

def test_shanetal2012_sig_sin_ka_divide_kw_is_100():
    """test against for shan et al 2012

    fig2c

    Shan et al (2012) 'Exact solutions for one-dimensional consolidation
    of single-layer unsaturated soil'

    test data is digitized from article

    dsig=100kPa instantly, + 100 * sin(2*pi/10e8 * t)
    ka/kw=100

    other data:
    n = 0.50
    S=0.80
    kw=10^10m/s
    m1kw=-0.5x10**4 kPa-1
    h=10m
    mw2=-2.0x10**4 kPa-1
    ma1k=-2.0x10**4 kPa-1
    ma2=1.0x10**4 kPa-1

    gamw= 10000N
    ua_=uatm=101kPa,
    R=8.31432J/molK
    t0 = 20 degrees C,
    T =(t0+273.16)K,
    wa=29x10**3 kg/mol

    Note to get (dua, duw) = (0.2, 0.4) * dsig need ua_=111kPa


    """

    tua = np.array([ 1209.27, 5028.71,
                    16722.5, 73536.8, 909364, 32518900])

    tuw = np.array([87.9258, 1759.75, 7213.86, 22435.5,
                    79084.3, 332175, 1934650, 5563950,
                    14092100, 25123700, 41707400, 58195200,
                    75041600, 100965000, 122978000,
                    151414000, 172542000, 225326000,
                    257526000])

    tset = np.array([1.0])

    z = np.array([0.5])


    reader = textwrap.dedent("""\
#from geotecha.piecewise.piecewise_linear_1d import PolyLine
#import numpy as np
# shan et al,
H = 10 #m
drn = 0
neig = 20

mvref = 1e-4 #1/kPa
kwref = 1.0e-10 #m/s

karef = kwref * 100 #m/s
Daref = karef / 10 # from equation ka=Da*g

wa = 29.0e-3 #kg / mol
R = 8.31432 #J/(mol.K)
ua_= 111 #kPa
T = 273.16 + 20
dTa = Daref /(mvref) / (wa*ua_/(R*T))/ H ** 2
dTw = kwref / mvref/ 10 / H**2
dT = max(dTw,dTa)

kw = PolyLine([0,1], [1,1])
Da = PolyLine([0,1], [1,1])
S = PolyLine([0,1], [0.8] * 2)
n = PolyLine([0,1], [0.5] * 2)

m1kw = PolyLine([0,1], [-0.5]*2)
m2w =  PolyLine([0,1], [-2.0]*2)
m1ka = PolyLine([0,1], [-2.0]*2)
m2a =  PolyLine([0,1], [1.0]*2)


surcharge_vs_depth = [PolyLine([0,1], [1,1]),
                      PolyLine([0,1], [1,1])]
surcharge_vs_time = [PolyLine([0,0,1e12], [0,100,100]),
                     PolyLine([0,0,1e12], [0,100,100])]
surcharge_omega_phase = [None,
                         (2*np.pi/1e8, -np.pi/2)]

ppress_z = [0.5]
#avg_ppress_z_pairs = [[0,1]]
settlement_z_pairs = [[0,1]]


tua = np.{tua}
tuw = np.{tuw}
tset = np.{tset}
tvals = np.hstack((tua,tuw, tset))


ppress_z_tval_indexes = slice(None, len(tua)+len(tuw))
#avg_ppress_z_pairs_tval_indexes = slice(None, None)#[0,4,6]
settlement_z_pairs_tval_indexes = slice(len(tua)+len(tuw),len(tua)+len(tuw)+len(tset))

save_data_to_file= False
save_figures_to_file= False
show_figures= False

    """.format(tua=repr(tua), tuw=repr(tuw), tset=repr(tset)))

    pora = 100 * np.array(
     [[0.201272, 0.176877, 0.082288,
       0.00172856, 0.00069029, 0.000544886]])

    porw = 100 * np.array(
     [[0.398947, 0.399218, 0.36372, 0.290698,
       0.249562, 0.255317, 0.283601, 0.339947,
       0.452531, 0.497583, 0.36, 0.09,
       -0.0498176, 0.19, 0.409602, 0.12,
       -0.12, 0.351532, -0.0140811]])
#
#    avp = np.array(
#      [[  7.25979052e+01,   6.65166314e+01,   5.89096834e+01,
#          4.94554633e+01,   3.79564622e+01,   2.66323138e+01,
#          2.50358034e+01,   1.28862133e+01,   4.44927613e+00,
#          1.18311566e+00,   8.09339892e-01,   5.26895921e-02]])

#    m1kw = -0.5e-4
#    m2w =  -2e-4
#    m1ka = -2e-4
#    m2a =  1e-4
#    uai = 20
#    uwi = 40
#    dsig = 100
#    h=10
#    set0 = (m1ka + m1kw) * (dsig-uai) + (m2a + m2w) * (uai-uwi)
#    set0*= h
#
#    settle = set0 + dsig * h * (m1ka + m1kw) * np.array(
#        [[0.00162455, 0.00812274, 0.0265343,
#          0.0882671, 0.144043, 0.184657, 0.229061,
#          0.255054, 0.279964]])

    for impl in ["vectorized"]:
        for dT in [10]:
            a = Speccon1dUnsat(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()

#            plt.figure()
#            plt.plot(tua, pora[0],'b-*', label='expected')
#            plt.plot(tua, a.pora[0,:len(tua)], 'r-+', label='calculated')
#            plt.gca().set_xlabel('time')
#            plt.gca().set_ylabel('pora')
#            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.figure()
#            plt.plot(tuw, porw[0],'b-*', label='expected')
#            plt.plot(tuw, a.porw[0,len(tua):len(tua)+len(tuw)], 'r-+', label='calculated')
#            plt.gca().set_xlabel('time')
#            plt.gca().set_ylabel('porw')
#            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()

#            plt.figure()
#            plt.plot(tset, settle[0],'b-*', label='expected')
#            plt.plot(tset, a.set[0], 'r-+', label='calculated')
#            plt.gca().set_xlabel('time')
#            plt.gca().set_ylabel('set')
#            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            legend=plt.legend()
#            legend.draggable()

##            plt.legend.DraggableLegend()
##            plt.figure()
##            plt.plot(t, avp[0],'b-*',  label='expected')
##            plt.plot(t, a.avp[0], 'r-+', label='calculated')
##            plt.legend()
            plt.show()


            assert_allclose(a.pora[:,:len(tua)], pora, atol=2.0,
                            err_msg = ("Fail. test_shanetal2012_sig_sin_ka_divide_kw_is_100, pora, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.porw[:,len(tua):len(tua)+len(tuw)], porw, atol=2.0,
                            err_msg = ("Fail. test_shanetal2012_sig_sin_ka_divide_kw_is_100, porw, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_shanetal2012_sig_sin_ka_divide_kw_is_100, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.set, settle, atol=5e-3,
#                            err_msg = ("Fail. test_shanetal2012_sig_sin_ka_divide_kw_is_100, set, "
#                                "implementation='%s', dT=%s" % (impl, dT)))

def test_shanetal2012_sig_sin_ka_divide_kw_is_1():
    """test against for shan et al 2012

    fig2c

    Shan et al (2012) 'Exact solutions for one-dimensional consolidation
    of single-layer unsaturated soil'

    test data is digitized from article

    dsig=100kPa instantly, + 100 * sin(2*pi/10e8 * t)
    ka/kw=1

    other data:
    n = 0.50
    S=0.80
    kw=10^10m/s
    m1kw=-0.5x10**4 kPa-1
    h=10m
    mw2=-2.0x10**4 kPa-1
    ma1k=-2.0x10**4 kPa-1
    ma2=1.0x10**4 kPa-1

    gamw= 10000N
    ua_=uatm=101kPa,
    R=8.31432J/molK
    t0 = 20 degrees C,
    T =(t0+273.16)K,
    wa=29x10**3 kg/mol

    Note to get (dua, duw) = (0.2, 0.4) * dsig need ua_=111kPa


    """

    tua = np.array([ 77763.7, 323377, 2044940,
                    4112230, 12931500, 27499200, 49451000,
                    71111800, 96702300, 127878000, 147054000,
                    173897000, 199973000, 250068000])

    tuw = np.array([ 127199, 385181, 2614270,
                    10166100, 23295700, 41690900, 52493100,
                    64432900, 87163500, 106021000, 122983000,
                    151241000, 176948000, 199297000, 219723000])

    tset = np.array([1.0])

    z = np.array([0.5])


    reader = textwrap.dedent("""\
#from geotecha.piecewise.piecewise_linear_1d import PolyLine
#import numpy as np
# shan et al,
H = 10 #m
drn = 0
neig = 20

mvref = 1e-4 #1/kPa
kwref = 1.0e-10 #m/s

karef = kwref * 1 #m/s
Daref = karef / 10 # from equation ka=Da*g

wa = 29.0e-3 #kg / mol
R = 8.31432 #J/(mol.K)
ua_= 111 #kPa
T = 273.16 + 20
dTa = Daref /(mvref) / (wa*ua_/(R*T))/ H ** 2
dTw = kwref / mvref/ 10 / H**2
dT = max(dTw,dTa)

kw = PolyLine([0,1], [1,1])
Da = PolyLine([0,1], [1,1])
S = PolyLine([0,1], [0.8] * 2)
n = PolyLine([0,1], [0.5] * 2)

m1kw = PolyLine([0,1], [-0.5]*2)
m2w =  PolyLine([0,1], [-2.0]*2)
m1ka = PolyLine([0,1], [-2.0]*2)
m2a =  PolyLine([0,1], [1.0]*2)


surcharge_vs_depth = [PolyLine([0,1], [1,1]),
                      PolyLine([0,1], [1,1])]
surcharge_vs_time = [PolyLine([0,0,1e12], [0,100,100]),
                     PolyLine([0,0,1e12], [0,100,100])]
surcharge_omega_phase = [None,
                         (2*np.pi/1e8, -np.pi/2)]

ppress_z = [0.5]
#avg_ppress_z_pairs = [[0,1]]
settlement_z_pairs = [[0,1]]


tua = np.{tua}
tuw = np.{tuw}
tset = np.{tset}
tvals = np.hstack((tua,tuw, tset))


ppress_z_tval_indexes = slice(None, len(tua)+len(tuw))
#avg_ppress_z_pairs_tval_indexes = slice(None, None)#[0,4,6]
settlement_z_pairs_tval_indexes = slice(len(tua)+len(tuw),len(tua)+len(tuw)+len(tset))

save_data_to_file= False
save_figures_to_file= False
show_figures= False

    """.format(tua=repr(tua), tuw=repr(tuw), tset=repr(tset)))

    pora = 100 * np.array(
     [[ 0.201102, 0.196364, 0.0802205, 0.0333903,
       0.015559, -0.00132037, -0.021001, -0.00510315,
       0.0201574, -0.00419096, -0.0201093, 0.000476728,
       0.0201278, -0.0201309]])

    porw = 100 * np.array(
     [[ 0.40148, 0.399705, 0.336129,
       0.415001, 0.497577, 0.360754, 0.180775,
       0.0139186, 0.0139459, 0.280214, 0.407727,
       0.11, -0.130365, 0.11, 0.35153]])
#
#    avp = np.array(
#      [[  7.25979052e+01,   6.65166314e+01,   5.89096834e+01,
#          4.94554633e+01,   3.79564622e+01,   2.66323138e+01,
#          2.50358034e+01,   1.28862133e+01,   4.44927613e+00,
#          1.18311566e+00,   8.09339892e-01,   5.26895921e-02]])

#    m1kw = -0.5e-4
#    m2w =  -2e-4
#    m1ka = -2e-4
#    m2a =  1e-4
#    uai = 20
#    uwi = 40
#    dsig = 100
#    h=10
#    set0 = (m1ka + m1kw) * (dsig-uai) + (m2a + m2w) * (uai-uwi)
#    set0*= h
#
#    settle = set0 + dsig * h * (m1ka + m1kw) * np.array(
#        [[0.00162455, 0.00812274, 0.0265343,
#          0.0882671, 0.144043, 0.184657, 0.229061,
#          0.255054, 0.279964]])

    for impl in ["vectorized"]:
        for dT in [10]:
            a = Speccon1dUnsat(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()

#            plt.figure()
#            plt.plot(tua, pora[0],'b-*', label='expected')
#            plt.plot(tua, a.pora[0,:len(tua)], 'r-+', label='calculated')
#            plt.gca().set_xlabel('time')
#            plt.gca().set_ylabel('pora')
#            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.figure()
#            plt.plot(tuw, porw[0],'b-*', label='expected')
#            plt.plot(tuw, a.porw[0,len(tua):len(tua)+len(tuw)], 'r-+', label='calculated')
#            plt.gca().set_xlabel('time')
#            plt.gca().set_ylabel('porw')
#            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()

#            plt.figure()
#            plt.plot(tset, settle[0],'b-*', label='expected')
#            plt.plot(tset, a.set[0], 'r-+', label='calculated')
#            plt.gca().set_xlabel('time')
#            plt.gca().set_ylabel('set')
#            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            legend=plt.legend()
#            legend.draggable()

##            plt.legend.DraggableLegend()
##            plt.figure()
##            plt.plot(t, avp[0],'b-*',  label='expected')
##            plt.plot(t, a.avp[0], 'r-+', label='calculated')
##            plt.legend()
            plt.show()


            assert_allclose(a.pora[:,:len(tua)], pora, atol=2.0,
                            err_msg = ("Fail. test_shanetal2012_sig_sin_ka_divide_kw_is_1, pora, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.porw[:,len(tua):len(tua)+len(tuw)], porw, atol=2.0,
                            err_msg = ("Fail. test_shanetal2012_sig_sin_ka_divide_kw_is_1, porw, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_shanetal2012_sig_sin_ka_divide_kw_is_1, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.set, settle, atol=5e-3,
#                            err_msg = ("Fail. test_shanetal2012_sig_sin_ka_divide_kw_is_1, set, "
#                                "implementation='%s', dT=%s" % (impl, dT)))



def test_shanetal2012_sig_ka_divide_kw_is_100_drn1():
    """test against for shan et al 2012

    fig3a and 5b

    Shan et al (2012) 'Exact solutions for one-dimensional consolidation
    of single-layer unsaturated soil'

    test data is digitized from article

    impervious bottom
    dsig=100kPa instantly
    ka/kw=100

    other data:
    n = 0.50
    S=0.80
    kw=10^10m/s
    m1kw=-0.5x10**4 kPa-1
    h=10m
    mw2=-2.0x10**4 kPa-1
    ma1k=-2.0x10**4 kPa-1
    ma2=1.0x10**4 kPa-1

    gamw= 10000N
    ua_=uatm=101kPa,
    R=8.31432J/molK
    t0 = 20 degrees C,
    T =(t0+273.16)K,
    wa=29x10**3 kg/mol

    Note to get (dua, duw) = (0.2, 0.4) * dsig need ua_=111kPa

    Note digitized settlement data does not include immediate
    undrained deformation. data is S/S0, S0=m1ks*q0*H

    """

    tua = np.array([433.207, 3735.05, 16000.4, 64813.4, 154289])

    tuw = np.array([96.6905, 4030.5, 14238.7, 70428.2, 412196,
                    19222100, 176214000, 779104000])

    tset = np.array([5.48925, 123.81, 1615.88, 10376.5, 39786.3,
                     107115, 808800, 31475000, 237605000,
                     1384160000])

    z = np.array([0.5])


    reader = textwrap.dedent("""\
#from geotecha.piecewise.piecewise_linear_1d import PolyLine
#import numpy as np
# shan et al,
H = 10 #m
drn = 1
neig = 20

mvref = 1e-4 #1/kPa
kwref = 1.0e-10 #m/s

karef = kwref * 100 #m/s
Daref = karef / 10 # from equation ka=Da*g

wa = 29.0e-3 #kg / mol
R = 8.31432 #J/(mol.K)
ua_= 111 #kPa
T = 273.16 + 20
dTa = Daref /(mvref) / (wa*ua_/(R*T))/ H ** 2
dTw = kwref / mvref/ 10 / H**2
dT = max(dTw,dTa)

kw = PolyLine([0,1], [1,1])
Da = PolyLine([0,1], [1,1])
S = PolyLine([0,1], [0.8] * 2)
n = PolyLine([0,1], [0.5] * 2)

m1kw = PolyLine([0,1], [-0.5]*2)
m2w =  PolyLine([0,1], [-2.0]*2)
m1ka = PolyLine([0,1], [-2.0]*2)
m2a =  PolyLine([0,1], [1.0]*2)


surcharge_vs_depth = PolyLine([0,1], [1,1])
surcharge_vs_time = PolyLine([0,0,10000], [0,100,100])

ppress_z = [0.5]
#avg_ppress_z_pairs = [[0,1]]
settlement_z_pairs = [[0,1]]
tvals = np.logspace(1, 10, 50)

tua = np.{tua}
tuw = np.{tuw}
tset = np.{tset}
tvals = np.hstack((tua,tuw, tset))


ppress_z_tval_indexes = slice(None, len(tua)+len(tuw))
#avg_ppress_z_pairs_tval_indexes = slice(None, None)#[0,4,6]
settlement_z_pairs_tval_indexes = slice(len(tua)+len(tuw),len(tua)+len(tuw)+len(tset))

save_data_to_file= False
save_figures_to_file= False
show_figures= False

    """.format(tua=repr(tua), tuw=repr(tuw), tset=repr(tset)))

    pora = 100 * np.array(
     [[0.19942, 0.19383, 0.141277, 0.0624325, 0.0131544]])

    porw = 100 * np.array(
     [[0.399688, 0.394084, 0.361177, 0.291483, 0.248387, 0.249039, 0.191061, 0.0845938]])
#
#    avp = np.array(
#      [[  7.25979052e+01,   6.65166314e+01,   5.89096834e+01,
#          4.94554633e+01,   3.79564622e+01,   2.66323138e+01,
#          2.50358034e+01,   1.28862133e+01,   4.44927613e+00,
#          1.18311566e+00,   8.09339892e-01,   5.26895921e-02]])

    m1kw = -0.5e-4
    m2w =  -2e-4
    m1ka = -2e-4
    m2a =  1e-4
    uai = 20
    uwi = 40
    dsig = 100
    h=10
    set0 = (m1ka + m1kw) * (dsig-uai) + (m2a + m2w) * (uai-uwi)
    set0*= h

    settle = set0 + dsig * h * (m1ka + m1kw) * np.array(
        [[0.000474773, 0.0057678, 0.019205, 0.0548722, 0.105722,
          0.15821, 0.181416, 0.193728, 0.2191, 0.264518]])

    for impl in ["vectorized"]:
        for dT in [10]:
            a = Speccon1dUnsat(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()


#            plt.figure()
#            plt.plot(tua, pora[0],'b-*', label='expected')
#            plt.plot(tua, a.pora[0,:len(tua)], 'r-+', label='calculated')
#            plt.gca().set_xlabel('time')
#            plt.gca().set_ylabel('pora')
#            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.figure()
#            plt.plot(tuw, porw[0],'b-*', label='expected')
#            plt.plot(tuw, a.porw[0,len(tua):len(tua)+len(tuw)], 'r-+', label='calculated')
#            plt.gca().set_xlabel('time')
#            plt.gca().set_ylabel('porw')
#            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.figure()
#            plt.plot(tset, settle[0],'b-*', label='expected')
#            plt.plot(tset, a.set[0], 'r-+', label='calculated')
#            plt.gca().set_xlabel('time')
#            plt.gca().set_ylabel('set')
#            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            legend=plt.legend()
#            legend.draggable()
#
###            plt.legend.DraggableLegend()
###            plt.figure()
###            plt.plot(t, avp[0],'b-*',  label='expected')
###            plt.plot(t, a.avp[0], 'r-+', label='calculated')
###            plt.legend()
#            plt.show()


            assert_allclose(a.pora[:,:len(tua)], pora, atol=1.0,
                            err_msg = ("Fail. test_shanetal2012_sig_ka_divide_kw_is_100_drn1, pora, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.porw[:,len(tua):len(tua)+len(tuw)], porw, atol=2.0,
                            err_msg = ("Fail. test_shanetal2012_sig_ka_divide_kw_is_100_drn1, porw, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_shanetal2012_sig_ka_divide_kw_is_100_drn1, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=5e-3,
                            err_msg = ("Fail. test_shanetal2012_sig_ka_divide_kw_is_100_drn1, set, "
                                "implementation='%s', dT=%s" % (impl, dT)))

def test_shanetal2012_sig_exp_ka_divide_kw_is_1():
    """test against for shan et al 2012

    fig3a and 5b

    Shan et al (2012) 'Exact solutions for one-dimensional consolidation
    of single-layer unsaturated soil'

    test data is digitized from article

    impervious bottom
    dsig=100kPa instantly
    ka/kw=1

    other data:
    n = 0.50
    S=0.80
    kw=10^10m/s
    m1kw=-0.5x10**4 kPa-1
    h=10m
    mw2=-2.0x10**4 kPa-1
    ma1k=-2.0x10**4 kPa-1
    ma2=1.0x10**4 kPa-1

    gamw= 10000N
    ua_=uatm=101kPa,
    R=8.31432J/molK
    t0 = 20 degrees C,
    T =(t0+273.16)K,
    wa=29x10**3 kg/mol

    Note to get (dua, duw) = (0.2, 0.4) * dsig need ua_=111kPa

    Note digitized settlement data does not include immediate
    undrained deformation. data is S/S0, S0=m1ks*q0*H

    """

    tua = np.array([ 98610.8, 558812, 1618050, 4430120,
                    9169230, 20639600])

    tuw = np.array([59520.3, 785687, 4227330, 14521000,
                    49880300, 233262000, 824055000])

    tset = np.array([3389.39, 48737.5, 313499, 1607330, 4061790,
                     9320600, 18839200, 67994500, 546659000,
                     1969340000])

    z = np.array([0.5])


    reader = textwrap.dedent("""\
#from geotecha.piecewise.piecewise_linear_1d import PolyLine
#import numpy as np
# shan et al,
H = 10 #m
drn = 1
neig = 20

mvref = 1e-4 #1/kPa
kwref = 1.0e-10 #m/s

karef = kwref * 1 #m/s
Daref = karef / 10 # from equation ka=Da*g

wa = 29.0e-3 #kg / mol
R = 8.31432 #J/(mol.K)
ua_= 111 #kPa
T = 273.16 + 20
dTa = Daref /(mvref) / (wa*ua_/(R*T))/ H ** 2
dTw = kwref / mvref/ 10 / H**2
dT = max(dTw,dTa)

kw = PolyLine([0,1], [1,1])
Da = PolyLine([0,1], [1,1])
S = PolyLine([0,1], [0.8] * 2)
n = PolyLine([0,1], [0.5] * 2)

m1kw = PolyLine([0,1], [-0.5]*2)
m2w =  PolyLine([0,1], [-2.0]*2)
m1ka = PolyLine([0,1], [-2.0]*2)
m2a =  PolyLine([0,1], [1.0]*2)


surcharge_vs_depth = PolyLine([0,1], [1,1])
surcharge_vs_time = PolyLine([0,0,10000], [0,100,100])

ppress_z = [0.5]
#avg_ppress_z_pairs = [[0,1]]
settlement_z_pairs = [[0,1]]


tua = np.{tua}
tuw = np.{tuw}
tset = np.{tset}
tvals = np.hstack((tua,tuw, tset))


ppress_z_tval_indexes = slice(None, len(tua)+len(tuw))
#avg_ppress_z_pairs_tval_indexes = slice(None, None)#[0,4,6]
settlement_z_pairs_tval_indexes = slice(len(tua)+len(tuw),len(tua)+len(tuw)+len(tset))

save_data_to_file= False
save_figures_to_file= False
show_figures= False

    """.format(tua=repr(tua), tuw=repr(tuw), tset=repr(tset)))

    pora = 100 * np.array(
     [[0.2, 0.185012, 0.141841, 0.0855233, 0.0404677, 0.00433402]])

    porw = 100 * np.array(
     [[0.39947, 0.381385, 0.317165, 0.259221, 0.241964,
       0.176184, 0.0822444]])
#
#    avp = np.array(
#      [[  7.25979052e+01,   6.65166314e+01,   5.89096834e+01,
#          4.94554633e+01,   3.79564622e+01,   2.66323138e+01,
#          2.50358034e+01,   1.28862133e+01,   4.44927613e+00,
#          1.18311566e+00,   8.09339892e-01,   5.26895921e-02]])

    m1kw = -0.5e-4
    m2w =  -2e-4
    m1ka = -2e-4
    m2a =  1e-4
    uai = 20
    uwi = 40
    dsig = 100
    h=10
    set0 = (m1ka + m1kw) * (dsig-uai) + (m2a + m2w) * (uai-uwi)
    set0*= h

    settle = set0 + dsig * h * (m1ka + m1kw) * np.array(
        [[0.00238898, 0.0109487, 0.0309119, 0.068754,
          0.11258, 0.156952, 0.186167, 0.20182, 0.239103, 0.272085]])

    for impl in ["vectorized"]:
        for dT in [10]:
            a = Speccon1dUnsat(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()


#            plt.figure()
#            plt.plot(tua, pora[0],'b-*', label='expected')
#            plt.plot(tua, a.pora[0,:len(tua)], 'r-+', label='calculated')
#            plt.gca().set_xlabel('time')
#            plt.gca().set_ylabel('pora')
#            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.figure()
#            plt.plot(tuw, porw[0],'b-*', label='expected')
#            plt.plot(tuw, a.porw[0,len(tua):len(tua)+len(tuw)], 'r-+', label='calculated')
#            plt.gca().set_xlabel('time')
#            plt.gca().set_ylabel('porw')
#            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.figure()
#            plt.plot(tset, settle[0],'b-*', label='expected')
#            plt.plot(tset, a.set[0], 'r-+', label='calculated')
#            plt.gca().set_xlabel('time')
#            plt.gca().set_ylabel('set')
#            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            legend=plt.legend()
#            legend.draggable()
#
##            plt.legend.DraggableLegend()
##            plt.figure()
##            plt.plot(t, avp[0],'b-*',  label='expected')
##            plt.plot(t, a.avp[0], 'r-+', label='calculated')
##            plt.legend()
#            plt.show()


            assert_allclose(a.pora[:,:len(tua)], pora, atol=1.0,
                            err_msg = ("Fail. test_shanetal2012_sig_ka_divide_kw_is_1_drn1, pora, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.porw[:,len(tua):len(tua)+len(tuw)], porw, atol=2.0,
                            err_msg = ("Fail. test_shanetal2012_sig_ka_divide_kw_is_1_drn1, porw, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_shanetal2012_sig_ka_divide_kw_is_1_drn1, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.set, settle, atol=5e-3,
                            err_msg = ("Fail. test_shanetal2012_sig_ka_divide_kw_is_1_drn1, set, "
                                "implementation='%s', dT=%s" % (impl, dT)))



def test_shanetal2012_sig_exp_ka_divide_kw_is_1():
    """test against for shan et al 2012

    fig2b

    Shan et al (2012) 'Exact solutions for one-dimensional consolidation
    of single-layer unsaturated soil'

    test data is digitized from article

    dsig = 100kPa instantly, + 100 * (1-exp(-0.00005 * t))
    ka/kw=1

    other data:
    n = 0.50
    S=0.80
    kw=10^10m/s
    m1kw=-0.5x10**4 kPa-1
    h=10m
    mw2=-2.0x10**4 kPa-1
    ma1k=-2.0x10**4 kPa-1
    ma2=1.0x10**4 kPa-1

    gamw= 10000N
    ua_=uatm=101kPa,
    R=8.31432J/molK
    t0 = 20 degrees C,
    T =(t0+273.16)K,
    wa=29x10**3 kg/mol

    Note to get (dua, duw) = (0.2, 0.4) * dsig need ua_=111kPa


    """

    tua = np.array([ 130.04, 14340.1, 57184.9, 141412, 608108,
                    1112030, 1793250, 2749920, 4434500])

    tuw = np.array([48.7307, 9074.34, 58269.9, 169467, 598895,
                    2638940, 18807200, 95600700, 214788000])

    tset = np.array([1.0])

    z = np.array([0.5])


    reader = textwrap.dedent("""\
#from geotecha.piecewise.piecewise_linear_1d import PolyLine
#import numpy as np
# shan et al,
H = 10 #m
drn = 0
neig = 20

mvref = 1e-4 #1/kPa
kwref = 1.0e-10 #m/s

karef = kwref * 1 #m/s
Daref = karef / 10 # from equation ka=Da*g

wa = 29.0e-3 #kg / mol
R = 8.31432 #J/(mol.K)
ua_= 111 #kPa
T = 273.16 + 20
dTa = Daref /(mvref) / (wa*ua_/(R*T))/ H ** 2
dTw = kwref / mvref/ 10 / H**2
dT = max(dTw,dTa)

kw = PolyLine([0,1], [1,1])
Da = PolyLine([0,1], [1,1])
S = PolyLine([0,1], [0.8] * 2)
n = PolyLine([0,1], [0.5] * 2)

m1kw = PolyLine([0,1], [-0.5]*2)
m2w =  PolyLine([0,1], [-2.0]*2)
m1ka = PolyLine([0,1], [-2.0]*2)
m2a =  PolyLine([0,1], [1.0]*2)


texp = np.logspace(0,9,50)
surexp = 100 * (1-np.exp(-0.00005 * texp))

surcharge_vs_depth = [PolyLine([0,1], [1,1]),
                      PolyLine([0,1], [1,1])]
surcharge_vs_time = [PolyLine([0,0,1e12], [0,100,100]),
                     PolyLine(texp, surexp)]

ppress_z = [0.5]
#avg_ppress_z_pairs = [[0,1]]
settlement_z_pairs = [[0,1]]


tua = np.{tua}
tuw = np.{tuw}
tset = np.{tset}
tvals = np.hstack((tua,tuw, tset))


ppress_z_tval_indexes = slice(None, len(tua)+len(tuw))
#avg_ppress_z_pairs_tval_indexes = slice(None, None)#[0,4,6]
settlement_z_pairs_tval_indexes = slice(len(tua)+len(tuw),len(tua)+len(tuw)+len(tset))

save_data_to_file= False
save_figures_to_file= False
show_figures= False

    """.format(tua=repr(tua), tuw=repr(tuw), tset=repr(tset)))

    pora = 100 * np.array(
     [[0.200685, 0.293476, 0.373111, 0.384015, 0.319116, 0.232374,
       0.143294, 0.0753091, 0.0221661]])

    porw = 100 * np.array(
     [[0.399786, 0.540823, 0.766428, 0.784975, 0.742748, 0.55719,
       0.494619, 0.397825, 0.224819]])
#
#    avp = np.array(
#      [[  7.25979052e+01,   6.65166314e+01,   5.89096834e+01,
#          4.94554633e+01,   3.79564622e+01,   2.66323138e+01,
#          2.50358034e+01,   1.28862133e+01,   4.44927613e+00,
#          1.18311566e+00,   8.09339892e-01,   5.26895921e-02]])

#    m1kw = -0.5e-4
#    m2w =  -2e-4
#    m1ka = -2e-4
#    m2a =  1e-4
#    uai = 20
#    uwi = 40
#    dsig = 100
#    h=10
#    set0 = (m1ka + m1kw) * (dsig-uai) + (m2a + m2w) * (uai-uwi)
#    set0*= h
#
#    settle = set0 + dsig * h * (m1ka + m1kw) * np.array(
#        [[0.00162455, 0.00812274, 0.0265343,
#          0.0882671, 0.144043, 0.184657, 0.229061,
#          0.255054, 0.279964]])

    for impl in ["vectorized"]:
        for dT in [10]:
            a = Speccon1dUnsat(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()

#            plt.figure()
#            plt.plot(tua, pora[0],'b-*', label='expected')
#            plt.plot(tua, a.pora[0,:len(tua)], 'r-+', label='calculated')
#            plt.gca().set_xlabel('time')
#            plt.gca().set_ylabel('pora')
#            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.figure()
#            plt.plot(tuw, porw[0],'b-*', label='expected')
#            plt.plot(tuw, a.porw[0,len(tua):len(tua)+len(tuw)], 'r-+', label='calculated')
#            plt.gca().set_xlabel('time')
#            plt.gca().set_ylabel('porw')
#            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
##            plt.figure()
##            plt.plot(tset, settle[0],'b-*', label='expected')
##            plt.plot(tset, a.set[0], 'r-+', label='calculated')
##            plt.gca().set_xlabel('time')
##            plt.gca().set_ylabel('set')
##            plt.gca().set_xscale('log')
##            plt.gca().grid()
##            legend=plt.legend()
##            legend.draggable()
#
##            plt.legend.DraggableLegend()
##            plt.figure()
##            plt.plot(t, avp[0],'b-*',  label='expected')
##            plt.plot(t, a.avp[0], 'r-+', label='calculated')
##            plt.legend()
#            plt.show()


            assert_allclose(a.pora[:,:len(tua)], pora, atol=2.0,
                            err_msg = ("Fail. test_shanetal2012_sig_exp_ka_divide_kw_is_1, pora, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.porw[:,len(tua):len(tua)+len(tuw)], porw, atol=2.0,
                            err_msg = ("Fail. test_shanetal2012_sig_exp_ka_divide_kw_is_1, porw, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_shanetal2012_sig_exp_ka_divide_kw_is_1, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.set, settle, atol=5e-3,
#                            err_msg = ("Fail. test_shanetal2012_sig_exp_ka_divide_kw_is_1, set, "
#                                "implementation='%s', dT=%s" % (impl, dT)))

def test_shanetal2012_sig_exp_ka_divide_kw_is_100_drn1():
    """test against for shan et al 2012

    fig3b

    Shan et al (2012) 'Exact solutions for one-dimensional consolidation
    of single-layer unsaturated soil'

    test data is digitized from article
    PTIB drainage
    dsig = 100kPa instantly, + 100 * (1-exp(-0.00005 * t))
    ka/kw=100

    other data:
    n = 0.50
    S=0.80
    kw=10^10m/s
    m1kw=-0.5x10**4 kPa-1
    h=10m
    mw2=-2.0x10**4 kPa-1
    ma1k=-2.0x10**4 kPa-1
    ma2=1.0x10**4 kPa-1

    gamw= 10000N
    ua_=uatm=101kPa,
    R=8.31432J/molK
    t0 = 20 degrees C,
    T =(t0+273.16)K,
    wa=29x10**3 kg/mol

    Note to get (dua, duw) = (0.2, 0.4) * dsig need ua_=111kPa


    """

    tua = np.array([ 56.9475, 1435.27, 14232.6, 52799.2, 118302, 239642])

    tuw = np.array([203.798, 8020.37, 30178.5, 86353.3, 775650,
                    36858500, 312404000, 1520470000])

    tset = np.array([1.0])

    z = np.array([0.5])


    reader = textwrap.dedent("""\
#from geotecha.piecewise.piecewise_linear_1d import PolyLine
#import numpy as np
# shan et al,
H = 10 #m
drn = 1
neig = 20

mvref = 1e-4 #1/kPa
kwref = 1.0e-10 #m/s

karef = kwref * 100 #m/s
Daref = karef / 10 # from equation ka=Da*g

wa = 29.0e-3 #kg / mol
R = 8.31432 #J/(mol.K)
ua_= 111 #kPa
T = 273.16 + 20
dTa = Daref /(mvref) / (wa*ua_/(R*T))/ H ** 2
dTw = kwref / mvref/ 10 / H**2
dT = max(dTw,dTa)

kw = PolyLine([0,1], [1,1])
Da = PolyLine([0,1], [1,1])
S = PolyLine([0,1], [0.8] * 2)
n = PolyLine([0,1], [0.5] * 2)

m1kw = PolyLine([0,1], [-0.5]*2)
m2w =  PolyLine([0,1], [-2.0]*2)
m1ka = PolyLine([0,1], [-2.0]*2)
m2a =  PolyLine([0,1], [1.0]*2)


texp = np.logspace(0,9,50)
surexp = 100 * (1-np.exp(-0.00005 * texp))

surcharge_vs_depth = [PolyLine([0,1], [1,1]),
                      PolyLine([0,1], [1,1])]
surcharge_vs_time = [PolyLine([0,0,1e12], [0,100,100]),
                     PolyLine(texp, surexp)]

ppress_z = [0.5]
#avg_ppress_z_pairs = [[0,1]]
settlement_z_pairs = [[0,1]]


tua = np.{tua}
tuw = np.{tuw}
tset = np.{tset}
tvals = np.hstack((tua,tuw, tset))


ppress_z_tval_indexes = slice(None, len(tua)+len(tuw))
#avg_ppress_z_pairs_tval_indexes = slice(None, None)#[0,4,6]
settlement_z_pairs_tval_indexes = slice(len(tua)+len(tuw),len(tua)+len(tuw)+len(tset))

save_data_to_file= False
save_figures_to_file= False
show_figures= False

    """.format(tua=repr(tua), tuw=repr(tuw), tset=repr(tset)))

    pora = 100 * np.array(
     [[0.199628, 0.212175, 0.229676, 0.162131, 0.0552287, 0.00879847]])

    porw = 100 * np.array(
     [[0.405024, 0.505495, 0.604396, 0.568289, 0.497645,
       0.494505, 0.302983, 0.0706436]])
#
#    avp = np.array(
#      [[  7.25979052e+01,   6.65166314e+01,   5.89096834e+01,
#          4.94554633e+01,   3.79564622e+01,   2.66323138e+01,
#          2.50358034e+01,   1.28862133e+01,   4.44927613e+00,
#          1.18311566e+00,   8.09339892e-01,   5.26895921e-02]])

#    m1kw = -0.5e-4
#    m2w =  -2e-4
#    m1ka = -2e-4
#    m2a =  1e-4
#    uai = 20
#    uwi = 40
#    dsig = 100
#    h=10
#    set0 = (m1ka + m1kw) * (dsig-uai) + (m2a + m2w) * (uai-uwi)
#    set0*= h
#
#    settle = set0 + dsig * h * (m1ka + m1kw) * np.array(
#        [[0.00162455, 0.00812274, 0.0265343,
#          0.0882671, 0.144043, 0.184657, 0.229061,
#          0.255054, 0.279964]])

    for impl in ["vectorized"]:
        for dT in [10]:
            a = Speccon1dUnsat(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()

            plt.figure()
            plt.plot(tua, pora[0],'b-*', label='expected')
            plt.plot(tua, a.pora[0,:len(tua)], 'r-+', label='calculated')
            plt.gca().set_xlabel('time')
            plt.gca().set_ylabel('pora')
            plt.gca().set_xscale('log')
            plt.gca().grid()
            plt.legend()
            leg = plt.legend(loc=3 )
            leg.draggable()

            plt.figure()
            plt.plot(tuw, porw[0],'b-*', label='expected')
            plt.plot(tuw, a.porw[0,len(tua):len(tua)+len(tuw)], 'r-+', label='calculated')
            plt.gca().set_xlabel('time')
            plt.gca().set_ylabel('porw')
            plt.gca().set_xscale('log')
            plt.gca().grid()
            plt.legend()
            leg = plt.legend(loc=3 )
            leg.draggable()

#            plt.figure()
#            plt.plot(tset, settle[0],'b-*', label='expected')
#            plt.plot(tset, a.set[0], 'r-+', label='calculated')
#            plt.gca().set_xlabel('time')
#            plt.gca().set_ylabel('set')
#            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            legend=plt.legend()
#            legend.draggable()

#            plt.legend.DraggableLegend()
#            plt.figure()
#            plt.plot(t, avp[0],'b-*',  label='expected')
#            plt.plot(t, a.avp[0], 'r-+', label='calculated')
#            plt.legend()
            plt.show()


            assert_allclose(a.pora[:,:len(tua)], pora, atol=2.0,
                            err_msg = ("Fail. test_shanetal2012_sig_exp_ka_divide_kw_is_100_drn1, pora, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.porw[:,len(tua):len(tua)+len(tuw)], porw, atol=2.0,
                            err_msg = ("Fail. test_shanetal2012_sig_exp_ka_divide_kw_is_100_drn1, porw, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_shanetal2012_sig_exp_ka_divide_kw_is_1, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.set, settle, atol=5e-3,
#                            err_msg = ("Fail. test_shanetal2012_sig_exp_ka_divide_kw_is_1, set, "
#                                "implementation='%s', dT=%s" % (impl, dT)))


def test_shanetal2012_uwtop_sin_ka_divide_kw_is_100_drn1():
    """test against for shan et al 2012

    fig 6

    Shan et al (2012) 'Exact solutions for one-dimensional consolidation
    of single-layer unsaturated soil'

    test data is digitized from article
    PTIB drainage
    dsig = 100kPa instantly
    uwtop = sin(2*np.pi/1e5)

    ka/kw=100

    other data:
    n = 0.50
    S=0.80
    kw=10^10m/s
    m1kw=-0.5x10**4 kPa-1
    h=10m
    mw2=-2.0x10**4 kPa-1
    ma1k=-2.0x10**4 kPa-1
    ma2=1.0x10**4 kPa-1

    gamw= 10000N
    ua_=uatm=101kPa,
    R=8.31432J/molK
    t0 = 20 degrees C,
    T =(t0+273.16)K,
    wa=29x10**3 kg/mol

    Note to get (dua, duw) = (0.2, 0.4) * dsig need ua_=111kPa


    """

    tua = np.array([5.21425, 17.7767, 88.0009, 470.683, 5628.38,
                    64115, 987508, 20046800])

    tuw = np.array([1.61839, 9.39558, 25.0597, 90.3838, 475.031,
                    1799.6, 18617.8, 65360.2, 97665.2, 161616, 453928])

    tset = np.array([1.0])

    z = np.array([0.02])


    reader = textwrap.dedent("""\
#from geotecha.piecewise.piecewise_linear_1d import PolyLine
#import numpy as np
# shan et al,
H = 10 #m
drn = 1
neig = 200

mvref = 1e-4 #1/kPa
kwref = 1.0e-10 #m/s

karef = kwref * 100 #m/s
Daref = karef / 10 # from equation ka=Da*g

wa = 29.0e-3 #kg / mol
R = 8.31432 #J/(mol.K)
ua_= 111 #kPa
T = 273.16 + 20
dTa = Daref /(mvref) / (wa*ua_/(R*T))/ H ** 2
dTw = kwref / mvref/ 10 / H**2
dT = max(dTw,dTa)

kw = PolyLine([0,1], [1,1])
Da = PolyLine([0,1], [1,1])
S = PolyLine([0,1], [0.8] * 2)
n = PolyLine([0,1], [0.5] * 2)

m1kw = PolyLine([0,1], [-0.5]*2)
m2w =  PolyLine([0,1], [-2.0]*2)
m1ka = PolyLine([0,1], [-2.0]*2)
m2a =  PolyLine([0,1], [1.0]*2)



surcharge_vs_depth = PolyLine([0,1], [1,1])
surcharge_vs_time = PolyLine([0, 0, 1e12], [0, 100, 100])


wtop_vs_time = PolyLine([0, 0.0, 1e12], [0,1,1])
wtop_omega_phase = (2*np.pi/1e5, -np.pi/2)

ppress_z = np.array([0.02])
#avg_ppress_z_pairs = [[0,1]]
settlement_z_pairs = [[0,1]]


tua = np.{tua}
tuw = np.{tuw}
tset = np.{tset}
tvals = np.hstack((tua,tuw, tset))


ppress_z_tval_indexes = slice(None, len(tua)+len(tuw))
#avg_ppress_z_pairs_tval_indexes = slice(None, None)#[0,4,6]
settlement_z_pairs_tval_indexes = slice(len(tua)+len(tuw),len(tua)+len(tuw)+len(tset))

save_data_to_file= False
save_figures_to_file= False
show_figures= False

    """.format(tua=repr(tua), tuw=repr(tuw), tset=repr(tset)))

    pora = 100 * np.array(
     [[0.196692, 0.162552, 0.0882168, 0.0391136, 0.0105529,
       0.00161958, -0.000311243, -0.00037913]])

    porw = 100 * np.array(
     [[0.400469, 0.389202, 0.356338, 0.315023, 0.280282, 0.265258,
       0.253991, 0.257746, 0.264319, 0.233333, 0.165728]])

#
#    avp = np.array(
#      [[  7.25979052e+01,   6.65166314e+01,   5.89096834e+01,
#          4.94554633e+01,   3.79564622e+01,   2.66323138e+01,
#          2.50358034e+01,   1.28862133e+01,   4.44927613e+00,
#          1.18311566e+00,   8.09339892e-01,   5.26895921e-02]])

#    m1kw = -0.5e-4
#    m2w =  -2e-4
#    m1ka = -2e-4
#    m2a =  1e-4
#    uai = 20
#    uwi = 40
#    dsig = 100
#    h=10
#    set0 = (m1ka + m1kw) * (dsig-uai) + (m2a + m2w) * (uai-uwi)
#    set0*= h
#
#    settle = set0 + dsig * h * (m1ka + m1kw) * np.array(
#        [[0.00162455, 0.00812274, 0.0265343,
#          0.0882671, 0.144043, 0.184657, 0.229061,
#          0.255054, 0.279964]])

    for impl in ["vectorized"]:
        for dT in [10]:
            a = Speccon1dUnsat(reader + "\n" +
                            "implementation = '%s'" % impl + "\n" +
                            "dT = %s" % dT)

            a.make_all()

            plt.figure()
            plt.plot(tua, pora[0],'b-*', label='expected')
            plt.plot(tua, a.pora[0,:len(tua)], 'r-+', label='calculated')
            plt.gca().set_xlabel('time')
            plt.gca().set_ylabel('pora')
            plt.gca().set_xscale('log')
            plt.gca().grid()
            plt.legend()
            leg = plt.legend(loc=3 )
            leg.draggable()

            plt.figure()
            plt.plot(tuw, porw[0],'b-*', label='expected')
            plt.plot(tuw, a.porw[0,len(tua):len(tua)+len(tuw)], 'r-+', label='calculated')
            plt.gca().set_xlabel('time')
            plt.gca().set_ylabel('porw')
            plt.gca().set_xscale('log')
            plt.gca().grid()
            plt.legend()
            leg = plt.legend(loc=3 )
            leg.draggable()

#            plt.figure()
#            plt.plot(tset, settle[0],'b-*', label='expected')
#            plt.plot(tset, a.set[0], 'r-+', label='calculated')
#            plt.gca().set_xlabel('time')
#            plt.gca().set_ylabel('set')
#            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            legend=plt.legend()
#            legend.draggable()

#            plt.legend.DraggableLegend()
#            plt.figure()
#            plt.plot(t, avp[0],'b-*',  label='expected')
#            plt.plot(t, a.avp[0], 'r-+', label='calculated')
#            plt.legend()
            plt.show()


            assert_allclose(a.pora[:,:len(tua)], pora, atol=2.0,
                            err_msg = ("Fail. test_shanetal2012_uwtop_sin_ka_divide_kw_is_100_drn1, pora, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.porw[:,len(tua):len(tua)+len(tuw)], porw, atol=2.0,
                            err_msg = ("Fail. test_shanetal2012_uwtop_sin_ka_divide_kw_is_100_drn1, porw, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_shanetal2012_uwtop_sin_ka_divide_kw_is_100_drn1, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.set, settle, atol=5e-3,
#                            err_msg = ("Fail. test_shanetal2012_uwtop_sin_ka_divide_kw_is_100_drn1, set, "
#                                "implementation='%s', dT=%s" % (impl, dT)))



if __name__ == '__main__':
#    import nose
#    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])

#    test_shanetal2012_sig_ka_divide_kw_is_100()
#    test_shanetal2012_sig_ka_divide_kw_is_1()
#    test_shanetal2012_sig_sin_ka_divide_kw_is_100()
#    test_shanetal2012_sig_sin_ka_divide_kw_is_1()
#    test_shanetal2012_sig_ka_divide_kw_is_100_drn1()
#    test_shanetal2012_sig_ka_divide_kw_is_1_drn1()
#    test_shanetal2012_sig_exp_ka_divide_kw_is_1()
#    test_shanetal2012_sig_exp_ka_divide_kw_is_100_drn1()
    test_shanetal2012_uwtop_sin_ka_divide_kw_is_100_drn1()