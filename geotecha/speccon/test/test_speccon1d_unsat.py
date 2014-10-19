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
#            plt.show()


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
#            plt.show()


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
#            plt.show()


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

#            plt.legend.DraggableLegend()
#            plt.figure()
#            plt.plot(t, avp[0],'b-*',  label='expected')
#            plt.plot(t, a.avp[0], 'r-+', label='calculated')
#            plt.legend()
#            plt.show()


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


#def test_shanetal2012_uwtop_sin_ka_divide_kw_is_100_drn1():
#    """test against for shan et al 2012
#
#    fig 6
#
#    Shan et al (2012) 'Exact solutions for one-dimensional consolidation
#    of single-layer unsaturated soil'
#
#    test data is digitized from article
#    PTIB drainage
#    dsig = 100kPa instantly
#    uwtop = sin(2*np.pi/1e5)
#
#    ka/kw=100
#
#    other data:
#    n = 0.50
#    S=0.80
#    kw=10^10m/s
#    m1kw=-0.5x10**4 kPa-1
#    h=10m
#    mw2=-2.0x10**4 kPa-1
#    ma1k=-2.0x10**4 kPa-1
#    ma2=1.0x10**4 kPa-1
#
#    gamw= 10000N
#    ua_=uatm=101kPa,
#    R=8.31432J/molK
#    t0 = 20 degrees C,
#    T =(t0+273.16)K,
#    wa=29x10**3 kg/mol
#
#    Note to get (dua, duw) = (0.2, 0.4) * dsig need ua_=111kPa
#
#
#    """
#
#    tua = np.array([5.21425, 17.7767, 88.0009, 470.683, 5628.38,
#                    64115, 987508, 20046800])
#
#    tuw = np.array([1.61839, 9.39558, 25.0597, 90.3838, 475.031,
#                    1799.6, 18617.8, 65360.2, 97665.2, 161616, 453928])
#
#    tset = np.array([1.0])
#
#    z = np.array([0.02])
#
#
#    reader = textwrap.dedent("""\
##from geotecha.piecewise.piecewise_linear_1d import PolyLine
##import numpy as np
## shan et al,
#H = 10 #m
#drn = 1
#neig = 500
#
#mvref = 1e-4 #1/kPa
#kwref = 1.0e-10 #m/s
#
#karef = kwref * 100 #m/s
#Daref = karef / 10 # from equation ka=Da*g
#
#wa = 29.0e-3 #kg / mol
#R = 8.31432 #J/(mol.K)
#ua_= 111 #kPa
#T = 273.16 + 20
#dTa = Daref /(mvref) / (wa*ua_/(R*T))/ H ** 2
#dTw = kwref / mvref / 10 / H**2
#dT = max(dTw, dTa)
#
#kw = PolyLine([0,1], [1,1])
#Da = PolyLine([0,1], [1,1])
#S = PolyLine([0,1], [0.8] * 2)
#n = PolyLine([0,1], [0.5] * 2)
#
#m1kw = PolyLine([0,1], [-0.5]*2)
#m2w =  PolyLine([0,1], [-2.0]*2)
#m1ka = PolyLine([0,1], [-2.0]*2)
#m2a =  PolyLine([0,1], [1.0]*2)
#
#
#
#surcharge_vs_depth = PolyLine([0,1], [1,1])
#surcharge_vs_time = PolyLine([0, 0, 1e12], [0, 100, 100])
#
#
#wtop_vs_time = PolyLine([0, 0.0, 1e12], [0,100,100])
#wtop_omega_phase = (2*np.pi/1e5, -np.pi/2)
#
#ppress_z = np.array([0.02])
##avg_ppress_z_pairs = [[0,1]]
#settlement_z_pairs = [[0,1]]
#
#
#tua = np.{tua}
#tuw = np.{tuw}
#tset = np.{tset}
#tvals = np.hstack((tua,tuw, tset))
#
#
#ppress_z_tval_indexes = slice(None, len(tua)+len(tuw))
##avg_ppress_z_pairs_tval_indexes = slice(None, None)#[0,4,6]
#settlement_z_pairs_tval_indexes = slice(len(tua)+len(tuw),len(tua)+len(tuw)+len(tset))
#
#save_data_to_file= False
#save_figures_to_file= False
#show_figures= False
#
#    """.format(tua=repr(tua), tuw=repr(tuw), tset=repr(tset)))
#
#    pora = 100 * np.array(
#     [[0.196692, 0.162552, 0.0882168, 0.0391136, 0.0105529,
#       0.00161958, -0.000311243, -0.00037913]])
#
#    porw = 100 * np.array(
#     [[0.400469, 0.389202, 0.356338, 0.315023, 0.280282, 0.265258,
#       0.253991, 0.257746, 0.264319, 0.233333, 0.165728]])
#
##
##    avp = np.array(
##      [[  7.25979052e+01,   6.65166314e+01,   5.89096834e+01,
##          4.94554633e+01,   3.79564622e+01,   2.66323138e+01,
##          2.50358034e+01,   1.28862133e+01,   4.44927613e+00,
##          1.18311566e+00,   8.09339892e-01,   5.26895921e-02]])
#
##    m1kw = -0.5e-4
##    m2w =  -2e-4
##    m1ka = -2e-4
##    m2a =  1e-4
##    uai = 20
##    uwi = 40
##    dsig = 100
##    h=10
##    set0 = (m1ka + m1kw) * (dsig-uai) + (m2a + m2w) * (uai-uwi)
##    set0*= h
##
##    settle = set0 + dsig * h * (m1ka + m1kw) * np.array(
##        [[0.00162455, 0.00812274, 0.0265343,
##          0.0882671, 0.144043, 0.184657, 0.229061,
##          0.255054, 0.279964]])
#
#    for impl in ["vectorized"]:
#        for dT in [10]:
#            a = Speccon1dUnsat(reader + "\n" +
#                            "implementation = '%s'" % impl + "\n" +
#                            "dT = %s" % dT)
#
#            a.make_all()
#
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
#
#
#            assert_allclose(a.pora[:,:len(tua)], pora, atol=2.0,
#                            err_msg = ("Fail. test_shanetal2012_uwtop_sin_ka_divide_kw_is_100_drn1, pora, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.porw[:,len(tua):len(tua)+len(tuw)], porw, atol=2.0,
#                            err_msg = ("Fail. test_shanetal2012_uwtop_sin_ka_divide_kw_is_100_drn1, porw, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
##            assert_allclose(a.avp, avp, atol=5,
##                            err_msg = ("Fail. test_shanetal2012_uwtop_sin_ka_divide_kw_is_100_drn1, avp, "
##                                "implementation='%s', dT=%s" % (impl, dT)))
##            assert_allclose(a.set, settle, atol=5e-3,
##                            err_msg = ("Fail. test_shanetal2012_uwtop_sin_ka_divide_kw_is_100_drn1, set, "
##                                "implementation='%s', dT=%s" % (impl, dT)))

def test_shanetal2012_uwtop_sin_ka_divide_kw_is_1_drn1():
    """test against for shan et al 2012

    As I have now implemented shanetal2012 I can do verificatinos other than
    by comparing digitised charts

    similar to fig 6 but uwtop = sin(2*np.pi/1e9) vs 1e5 in the article
    this is because the z/H=0.02 value in the article requires many many
    terms whcih takes too long.  So I've picked some z values near the middle
    that don't require so many terms.  The longer wave period also means the
    boundary conditions have time to transfer to lower depths

    Shan et al (2012) 'Exact solutions for one-dimensional consolidation
    of single-layer unsaturated soil'

    PTIB drainage
    dsig = 100kPa instantly
    uwtop = sin(2*np.pi/1e8)

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



    z = np.array([0, 3.0, 5.0, 8.0, 10.0])/10
    t = np.array([1e6,3e6, 1e8,3e8, 1e9, 2e9 ])


    reader = textwrap.dedent("""\
#############################
##shanetal2012 input
#kw = 1e-10
#ka = 1 * kw
#H=10
#Cw=-0.75
#Cvw=-5e-8
#Ca = -0.0775134
#Cva=-64504.4 * ka
#drn=1
#Csw=0.25
#Csa=0.155027
#uwi=(40, 40)
#uai=(20, 20)
#nterms=200
#f=f1=f2=f3=f4=None
#f1 = dict([('type', 'sin'), ('q0',100.0), ('omega',2*np.pi / 1e9)])
#z = np.array([0, 3.0, 5.0, 8.0, 10.0])
#t = np.array([1e6,3e6, 1e8,3e8, 1e9, 2e9 ])
#
#porw, pora = shanetal2012(z, t, H, Cw, Cvw, Ca, Cva, drn, Csw, Csa,
#             uwi, uai, nterms, f=f, f1=f1, f2=f2, f3=f3, f4=f4)
#########################


#from geotecha.piecewise.piecewise_linear_1d import PolyLine
#import numpy as np
# shan et al,
H = 10 #m
drn = 1
neig = 30

mvref = 1e-4 #1/kPa
kwref = 1.0e-10 #m/s

karef = kwref * 1 #m/s
Daref = karef / 10 # from equation ka=Da*g

wa = 29.0e-3 #kg / mol
R = 8.31432 #J/(mol.K)
ua_= 111 #kPa
T = 273.16 + 20
dTa = Daref /(mvref) / (wa*ua_/(R*T))/ H ** 2
dTw = kwref / mvref / 10 / H**2
dT = max(dTw, dTa)

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


wtop_vs_time = PolyLine([0, 0.0, 1e12], [0,100,100])
wtop_omega_phase = (2*np.pi/1e9, -np.pi/2)

ppress_z = np.{z}
#avg_ppress_z_pairs = [[0,1]]
#settlement_z_pairs = [[0,1]]

tvals = np.{t}

#ppress_z_tval_indexes = slice(None, len(tua)+len(tuw))
#avg_ppress_z_pairs_tval_indexes = slice(None, None)#[0,4,6]
#settlement_z_pairs_tval_indexes = slice(len(tua)+len(tuw),len(tua)+len(tuw)+len(tset))

save_data_to_file= False
save_figures_to_file= False
show_figures= False

    """.format(t=repr(t), z = repr(z)))

    porw = np.array(
     [[  6.28314397e-01,   1.88484397e+00,   5.87785252e+01,
          9.51056516e+01,  -2.44929360e-14,  -4.89858720e-14],
       [  3.36836720e+01,   3.02084854e+01,   2.66821009e+01,
          6.00250385e+01,  -2.37006068e+01,  -2.89247778e+01],
       [  3.73258377e+01,   3.31242502e+01,   2.43606807e+01,
          4.24445988e+01,  -1.42733983e+01,  -2.24098737e+01],
       [  3.95179974e+01,   3.58660188e+01,   2.47525077e+01,
          2.98945531e+01,   2.68168932e+00,  -8.26117630e+00],
       [  3.97911703e+01,   3.64059255e+01,   2.48530634e+01,
          2.79500925e+01,   6.68126128e+00,  -4.82458233e+00]])

    pora = np.array(
     [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
       [  1.16396722e+01,   7.03990486e+00,   1.82548313e-02,
          2.16231937e-02,   1.29313648e-02,   1.60933200e-02],
       [  1.64604603e+01,   1.08992249e+01,   1.91064304e-02,
          3.24439239e-02,   6.56045792e-03,   1.14850996e-02],
       [  1.93620127e+01,   1.45282414e+01,   1.83839252e-02,
          4.01955433e-02,  -4.28351741e-03,   2.33970574e-03],
       [  1.97235856e+01,   1.52428642e+01,   1.82305019e-02,
          4.14018795e-02,  -6.81509232e-03,   1.48876019e-04]])

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
#            plt.plot(porw, z,'b-*', label='expected')
#            plt.plot(a.porw, z, 'r-+', label='calculated')
#            plt.gca().set_xlabel('porw')
#            plt.gca().set_ylabel('depth')
##            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.figure()
#            plt.plot(pora, z,'b-*', label='expected')
#            plt.plot(a.pora, z, 'r-+', label='calculated')
#            plt.gca().set_xlabel('pora')
#            plt.gca().set_ylabel('depth')
##            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.show()


            assert_allclose(a.pora, pora, atol=0.2,
                            err_msg = ("Fail. test_shanetal2012_uwtop_sin_ka_divide_kw_is_1_drn1, pora, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.porw, porw, atol=0.2,
                            err_msg = ("Fail. test_shanetal2012_uwtop_sin_ka_divide_kw_is_1_drn1, porw, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_shanetal2012_uwtop_sin_ka_divide_kw_is_1_drn1, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.set, settle, atol=5e-3,
#                            err_msg = ("Fail. test_shanetal2012_uwtop_sin_ka_divide_kw_is_1_drn1, set, "
#                                "implementation='%s', dT=%s" % (impl, dT)))


def test_shanetal2012_uwtop_sin_ka_divide_kw_is_100_drn1():
    """test against for shan et al 2012

    As I have now implemented shanetal2012 I can do verificatinos other than
    by comparing digitised charts

    similar to fig 6 but uwtop = sin(2*np.pi/1e9) vs 1e5 in the article
    this is because the z/H=0.02 value in the article requires many many
    terms whcih takes too long.  So I've picked some z values near the middle
    that don't require so many terms.  The longer wave period also means the
    boundary conditions have time to transfer to lower depths

    Shan et al (2012) 'Exact solutions for one-dimensional consolidation
    of single-layer unsaturated soil'

    PTIB drainage
    dsig = 100kPa instantly
    uwtop = sin(2*np.pi/1e8)

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



    z = np.array([0, 3.0, 5.0, 8.0, 10.0])/10
    t = np.array([1e6,3e6, 1e8,3e8, 1e9, 2e9 ])


    reader = textwrap.dedent("""\
#############################
##shanetal2012 input
#kw = 1e-10
#ka = 100 * kw
#H=10
#Cw=-0.75
#Cvw=-5e-8
#Ca = -0.0775134
#Cva=-64504.4 * ka
#drn=1
#Csw=0.25
#Csa=0.155027
#uwi=(40, 40)
#uai=(20, 20)
#nterms=200
#f=f1=f2=f3=f4=None
#f1 = dict([('type', 'sin'), ('q0',100.0), ('omega',2*np.pi / 1e9)])
#z = np.array([0, 3.0, 5.0, 8.0, 10.0])
#t = np.array([1e6,3e6, 1e8,3e8, 1e9, 2e9 ])
#
#porw, pora = shanetal2012(z, t, H, Cw, Cvw, Ca, Cva, drn, Csw, Csa,
#             uwi, uai, nterms, f=f, f1=f1, f2=f2, f3=f3, f4=f4)
#########################


#from geotecha.piecewise.piecewise_linear_1d import PolyLine
#import numpy as np
# shan et al,
H = 10 #m
drn = 1
neig = 30

mvref = 1e-4 #1/kPa
kwref = 1.0e-10 #m/s

karef = kwref * 100 #m/s
Daref = karef / 10 # from equation ka=Da*g

wa = 29.0e-3 #kg / mol
R = 8.31432 #J/(mol.K)
ua_= 111 #kPa
T = 273.16 + 20
dTa = Daref /(mvref) / (wa*ua_/(R*T))/ H ** 2
dTw = kwref / mvref / 10 / H**2
dT = max(dTw, dTa)

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


wtop_vs_time = PolyLine([0, 0.0, 1e12], [0,100,100])
wtop_omega_phase = (2*np.pi/1e9, -np.pi/2)

ppress_z = np.{z}
#avg_ppress_z_pairs = [[0,1]]
#settlement_z_pairs = [[0,1]]

tvals = np.{t}

#ppress_z_tval_indexes = slice(None, len(tua)+len(tuw))
#avg_ppress_z_pairs_tval_indexes = slice(None, None)#[0,4,6]
#settlement_z_pairs_tval_indexes = slice(len(tua)+len(tuw),len(tua)+len(tuw)+len(tset))

save_data_to_file= False
save_figures_to_file= False
show_figures= False

    """.format(t=repr(t), z = repr(z)))

    porw = np.array(
     [[  6.28314397e-01,   1.88484397e+00,   5.87785252e+01,
          9.51056516e+01,  -2.44929360e-14,  -4.89858720e-14],
       [  2.49988064e+01,   2.49988106e+01,   2.67422339e+01,
          6.00613526e+01,  -2.36941819e+01,  -2.89329748e+01],
       [  2.49988026e+01,   2.49988076e+01,   2.44432460e+01,
          4.24971420e+01,  -1.42597875e+01,  -2.24190381e+01],
       [  2.49988009e+01,   2.49988058e+01,   2.48464705e+01,
          2.99597621e+01,   2.70588042e+00,  -8.26761967e+00],
       [  2.49988007e+01,   2.49988055e+01,   2.49481577e+01,
          2.80175191e+01,   6.70805154e+00,  -4.83000356e+00]])

    pora = np.array(
     [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
       [ -1.46048597e-04,  -1.39028204e-04,   1.92386537e-04,
          2.10618262e-04,   1.42233488e-04,   1.73712449e-04],
       [ -1.45819307e-04,  -1.39094328e-04,   2.06147206e-04,
          3.16178850e-04,   8.54777625e-05,   1.34505226e-04],
       [ -1.45592460e-04,  -1.39152190e-04,   2.03677459e-04,
          3.91531566e-04,  -1.65237578e-05,   4.94140236e-05],
       [ -1.45546361e-04,  -1.39163212e-04,   2.03057516e-04,
          4.03205556e-04,  -4.05831072e-05,   2.87469834e-05]])

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
#            plt.plot(porw, z,'b-*', label='expected')
#            plt.plot(a.porw, z, 'r-+', label='calculated')
#            plt.gca().set_xlabel('porw')
#            plt.gca().set_ylabel('depth')
##            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.figure()
#            plt.plot(pora, z,'b-*', label='expected')
#            plt.plot(a.pora, z, 'r-+', label='calculated')
#            plt.gca().set_xlabel('pora')
#            plt.gca().set_ylabel('depth')
##            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.show()


            assert_allclose(a.pora, pora, atol=0.2,
                            err_msg = ("Fail. test_shanetal2012_uwtop_sin_ka_divide_kw_is_100_drn1, pora, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.porw, porw, atol=0.2,
                            err_msg = ("Fail. test_shanetal2012_uwtop_sin_ka_divide_kw_is_100_drn1, porw, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_shanetal2012_uwtop_sin_ka_divide_kw_is_1_drn1, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.set, settle, atol=5e-3,
#                            err_msg = ("Fail. test_shanetal2012_uwtop_sin_ka_divide_kw_is_1_drn1, set, "
#                                "implementation='%s', dT=%s" % (impl, dT)))

def test_shanetal2012_uwtop_sin_ka_divide_kw_is_10_drn0():
    """test against for shan et al 2012


    PTPB drainage
    dsig = 100kPa instantly
    uwtop = sin(2*np.pi/1e8)

    ka/kw=10

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



    z = np.array([0, 3.0, 5.0, 8.0, 10.0])/10
    t = np.array([1e6,3e6, 1e8,3e8, 1e9, 2e9 ])


    reader = textwrap.dedent("""\
#############################
##shanetal2012 input
#kw = 1e-10
#ka = 10 * kw
#H=10
#Cw=-0.75
#Cvw=-5e-8
#Ca = -0.0775134
#Cva=-64504.4 * ka
#drn=0
#Csw=0.25
#Csa=0.155027
#uwi=(40, 40)
#uai=(20, 20)
#nterms=200
#f=f1=f2=f3=f4=None
#f1 = dict([('type', 'sin'), ('q0',100.0), ('omega',2*np.pi / 1e9)])
#z = np.array([0, 3.0, 5.0, 8.0, 10.0])
#t = np.array([1e6,3e6, 1e8,3e8, 1e9, 2e9 ])
#
#porw, pora = shanetal2012(z, t, H, Cw, Cvw, Ca, Cva, drn, Csw, Csa,
#             uwi, uai, nterms, f=f, f1=f1, f2=f2, f3=f3, f4=f4)
#########################


#from geotecha.piecewise.piecewise_linear_1d import PolyLine
#import numpy as np
# shan et al,
H = 10 #m
drn = 0
neig = 30

mvref = 1e-4 #1/kPa
kwref = 1.0e-10 #m/s

karef = kwref * 10 #m/s
Daref = karef / 10 # from equation ka=Da*g

wa = 29.0e-3 #kg / mol
R = 8.31432 #J/(mol.K)
ua_= 111 #kPa
T = 273.16 + 20
dTa = Daref /(mvref) / (wa*ua_/(R*T))/ H ** 2
dTw = kwref / mvref / 10 / H**2
dT = max(dTw, dTa)

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


wtop_vs_time = PolyLine([0, 0.0, 1e12], [0,100,100])
wtop_omega_phase = (2*np.pi/1e9, -np.pi/2)

ppress_z = np.{z}
#avg_ppress_z_pairs = [[0,1]]
#settlement_z_pairs = [[0,1]]

tvals = np.{t}

#ppress_z_tval_indexes = slice(None, len(tua)+len(tuw))
#avg_ppress_z_pairs_tval_indexes = slice(None, None)#[0,4,6]
#settlement_z_pairs_tval_indexes = slice(len(tua)+len(tuw),len(tua)+len(tuw)+len(tset))

save_data_to_file= False
save_figures_to_file= False
show_figures= False

    """.format(t=repr(t), z = repr(z)))

    porw = np.array(
     [[  6.28314397e-01,   1.88484397e+00,   5.87785252e+01,
          9.51056516e+01,  -2.44929360e-14,  -4.89858720e-14],
       [  2.50058742e+01,   2.49879796e+01,   2.60660524e+01,
          5.54644234e+01,  -3.29469515e+01,  -3.33094394e+01],
       [  2.50100970e+01,   2.49879587e+01,   2.15901222e+01,
          3.35472685e+01,  -2.80115278e+01,  -2.84595876e+01],
       [  2.50009489e+01,   2.49814180e+01,   1.16707860e+01,
          1.12810729e+01,  -1.13139730e+01,  -1.15773359e+01],
       [  2.59043683e-14,   1.52391468e-14,   1.06828718e-16,
         -1.56910088e-15,  -2.22743307e-15,  -2.28230456e-15]])

    pora = np.array(
     [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
       [  2.24142070e-02,  -1.42638733e-03,   9.04003658e-04,
          6.69002283e-04,   1.97774246e-03,   1.99953803e-03],
       [  2.80469792e-02,  -1.44921022e-03,   4.66153324e-04,
          8.43184232e-04,   1.68082128e-03,   1.70776209e-03],
       [  1.58629765e-02,  -1.48115057e-03,   3.67685198e-06,
          4.65935580e-04,   6.78518061e-04,   6.94353467e-04],
       [  2.05910489e-18,  -9.16650491e-19,  -6.71524095e-21,
          9.44574693e-20,   1.33569938e-19,   1.36869234e-19]])

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
#            plt.plot(porw, z,'b-*', label='expected')
#            plt.plot(a.porw, z, 'r-+', label='calculated')
#            plt.gca().set_xlabel('porw')
#            plt.gca().set_ylabel('depth')
##            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.figure()
#            plt.plot(pora, z,'b-*', label='expected')
#            plt.plot(a.pora, z, 'r-+', label='calculated')
#            plt.gca().set_xlabel('pora')
#            plt.gca().set_ylabel('depth')
##            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.show()


            assert_allclose(a.pora, pora, atol=0.2,
                            err_msg = ("Fail. test_shanetal2012_uwtop_sin_ka_divide_kw_is_10_drn0, pora, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.porw, porw, atol=0.2,
                            err_msg = ("Fail. test_shanetal2012_uwtop_sin_ka_divide_kw_is_10_drn0, porw, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_shanetal2012_uwtop_sin_ka_divide_kw_is_10_drn0, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.set, settle, atol=5e-3,
#                            err_msg = ("Fail. test_shanetal2012_uwtop_sin_ka_divide_kw_is_10_drn0, set, "
#                                "implementation='%s', dT=%s" % (impl, dT)))



def test_shanetal2012_uatop_sin_ka_divide_kw_is_10_drn1():
    """test against shan et al 2012


    PTIB drainage
    dsig = 100kPa instantly
    uatop = sin(2*np.pi/1e8)

    ka/kw=10

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



    z = np.array([0, 3.0, 5.0, 8.0, 10.0])/10
    t = np.array([1e6,3e6, 1e8,3e8, 1e9, 2e9 ])


    reader = textwrap.dedent("""\
#############################
##shanetal2012 input
#kw = 1e-10
#ka = 10 * kw
#H=10
#Cw=-0.75
#Cvw=-5e-8
#Ca = -0.0775134
#Cva=-64504.4 * ka
#drn=1
#Csw=0.25
#Csa=0.155027
#uwi=(40, 40)
#uai=(20, 20)
#nterms=200
#f=f1=f2=f3=f4=None
#f2 = dict([('type', 'sin'), ('q0',100.0), ('omega',2*np.pi / 1e9)])
#z = np.array([0, 3.0, 5.0, 8.0, 10.0])
#t = np.array([1e6,3e6, 1e8,3e8, 1e9, 2e9 ])
#
#porw, pora = shanetal2012(z, t, H, Cw, Cvw, Ca, Cva, drn, Csw, Csa,
#             uwi, uai, nterms, f=f, f1=f1, f2=f2, f3=f3, f4=f4)
#########################


#from geotecha.piecewise.piecewise_linear_1d import PolyLine
#import numpy as np
# shan et al,
H = 10 #m
drn = 1
neig = 30

mvref = 1e-4 #1/kPa
kwref = 1.0e-10 #m/s

karef = kwref * 10 #m/s
Daref = karef / 10 # from equation ka=Da*g

wa = 29.0e-3 #kg / mol
R = 8.31432 #J/(mol.K)
ua_= 111 #kPa
T = 273.16 + 20
dTa = Daref /(mvref) / (wa*ua_/(R*T))/ H ** 2
dTw = kwref / mvref / 10 / H**2
dT = max(dTw, dTa)

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


atop_vs_time = PolyLine([0, 0.0, 1e12], [0,100,100])
atop_omega_phase = (2*np.pi/1e9, -np.pi/2)

ppress_z = np.{z}
#avg_ppress_z_pairs = [[0,1]]
#settlement_z_pairs = [[0,1]]

tvals = np.{t}

#ppress_z_tval_indexes = slice(None, len(tua)+len(tuw))
#avg_ppress_z_pairs_tval_indexes = slice(None, None)#[0,4,6]
#settlement_z_pairs_tval_indexes = slice(len(tua)+len(tuw),len(tua)+len(tuw)+len(tset))

save_data_to_file= False
save_figures_to_file= False
show_figures= False

    """.format(t=repr(t), z = repr(z)))

    porw = np.array(
     [[  0.        ,   0.        ,   0.        ,   0.        ,
          0.        ,   0.        ],
       [ 26.91500646,  26.28250152,  52.65527223,  44.47861517,
         24.97238588,  23.68391274],
       [ 27.7417917 ,  26.23086184,  64.32923796,  67.23646728,
         21.9146998 ,  19.90781766],
       [ 28.54596753,  26.18842419,  68.44276332,  85.19424377,
         13.06602802,  10.36670683],
       [ 28.70804252,  26.18064377,  68.72399456,  88.26641423,
         10.84235072,   8.00409891]])

    pora = np.array(
     [[  6.28314397e-01,   1.88484397e+00,   5.87785252e+01,
          9.51056516e+01,  -2.44929360e-14,  -4.89858720e-14],
       [  2.56608995e+00,   1.72336715e+00,   5.85856281e+01,
          9.51744999e+01,  -2.35436005e-01,  -2.35358532e-01],
       [  3.66766149e+00,   1.65455907e+00,   5.84956041e+01,
          9.52067291e+01,  -3.45338714e-01,  -3.45218045e-01],
       [  4.73911132e+00,   1.59801501e+00,   5.84171806e+01,
          9.52350156e+01,  -4.41132260e-01,  -4.40969956e-01],
       [  4.95505328e+00,   1.58764844e+00,   5.84022709e+01,
          9.52404212e+01,  -4.59346272e-01,  -4.59175614e-01]])

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
#            plt.plot(porw, z,'b-*', label='expected')
#            plt.plot(a.porw, z, 'r-+', label='calculated')
#            plt.gca().set_xlabel('porw')
#            plt.gca().set_ylabel('depth')
##            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.figure()
#            plt.plot(pora, z,'b-*', label='expected')
#            plt.plot(a.pora, z, 'r-+', label='calculated')
#            plt.gca().set_xlabel('pora')
#            plt.gca().set_ylabel('depth')
##            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.show()


            assert_allclose(a.pora, pora, atol=0.2,
                            err_msg = ("Fail. test_shanetal2012_uatop_sin_ka_divide_kw_is_10_drn1, pora, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.porw, porw, atol=0.2,
                            err_msg = ("Fail. test_shanetal2012_uatop_sin_ka_divide_kw_is_10_drn1, porw, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_shanetal2012_uatop_sin_ka_divide_kw_is_10_drn1, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.set, settle, atol=5e-3,
#                            err_msg = ("Fail. test_shanetal2012_uatop_sin_ka_divide_kw_is_10_drn1, set, "
#                                "implementation='%s', dT=%s" % (impl, dT)))


def test_shanetal2012_uwbot_sin_ka_divide_kw_is_10_drn0():
    """test against for shan et al 2012


    PTPB drainage
    dsig = 100kPa instantly
    uwbot = sin(2*np.pi/1e8)

    ka/kw=10

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



    z = np.array([0, 3.0, 5.0, 8.0, 10.0])/10
    t = np.array([1e6,3e6, 1e8,3e8, 1e9, 2e9 ])


    reader = textwrap.dedent("""\
#############################
##shanetal2012 input
#kw = 1e-10
#ka = 10 * kw
#H=10
#Cw=-0.75
#Cvw=-5e-8
#Ca = -0.0775134
#Cva=-64504.4 * ka
#drn=0
#Csw=0.25
#Csa=0.155027
#uwi=(40, 40)
#uai=(20, 20)
#nterms=200
#f=f1=f2=f3=f4=None
#f3 = dict([('type', 'sin'), ('q0',100.0), ('omega',2*np.pi / 1e9)])
#z = np.array([0, 3.0, 5.0, 8.0, 10.0])
#t = np.array([1e6,3e6, 1e8,3e8, 1e9, 2e9 ])
#
#porw, pora = shanetal2012(z, t, H, Cw, Cvw, Ca, Cva, drn, Csw, Csa,
#             uwi, uai, nterms, f=f, f1=f1, f2=f2, f3=f3, f4=f4)
#########################


#from geotecha.piecewise.piecewise_linear_1d import PolyLine
#import numpy as np
# shan et al,
H = 10 #m
drn = 0
neig = 30

mvref = 1e-4 #1/kPa
kwref = 1.0e-10 #m/s

karef = kwref * 10 #m/s
Daref = karef / 10 # from equation ka=Da*g

wa = 29.0e-3 #kg / mol
R = 8.31432 #J/(mol.K)
ua_= 111 #kPa
T = 273.16 + 20
dTa = Daref /(mvref) / (wa*ua_/(R*T))/ H ** 2
dTw = kwref / mvref / 10 / H**2
dT = max(dTw, dTa)

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


wbot_vs_time = PolyLine([0, 0.0, 1e12], [0,100,100])
wbot_omega_phase = (2*np.pi/1e9, -np.pi/2)

ppress_z = np.{z}
#avg_ppress_z_pairs = [[0,1]]
#settlement_z_pairs = [[0,1]]

tvals = np.{t}

#ppress_z_tval_indexes = slice(None, len(tua)+len(tuw))
#avg_ppress_z_pairs_tval_indexes = slice(None, None)#[0,4,6]
#settlement_z_pairs_tval_indexes = slice(len(tua)+len(tuw),len(tua)+len(tuw)+len(tset))

save_data_to_file= False
save_figures_to_file= False
show_figures= False

    """.format(t=repr(t), z = repr(z)))

    porw = np.array(
     [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
       [  2.50058561e+01,   2.49879388e+01,   1.61295449e+01,
          1.76598399e+01,  -1.71174355e+01,  -1.74799234e+01],
       [  2.50100970e+01,   2.49879587e+01,   2.15901222e+01,
          3.35472685e+01,  -2.80115278e+01,  -2.84595876e+01],
       [  2.50009792e+01,   2.49815376e+01,   3.12136298e+01,
          6.86638472e+01,  -2.96382421e+01,  -2.99016050e+01],
       [  6.28314397e-01,   1.88484397e+00,   5.87785252e+01,
          9.51056516e+01,  -3.44105747e-14,  -5.89583822e-14]])

    pora = np.array(
     [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
       [  2.23998687e-02,  -1.47094529e-03,   8.84008095e-05,
          6.54472844e-04,   1.02672226e-03,   1.04851783e-03],
       [  2.80469792e-02,  -1.44921022e-03,   4.66153324e-04,
          8.43184232e-04,   1.68082128e-03,   1.70776209e-03],
       [  1.58847596e-02,  -1.41404139e-03,   9.48427047e-04,
          4.47004338e-04,   1.77943171e-03,   1.79526712e-03],
       [  2.12910817e-18,  -7.97670921e-19,   3.84688741e-19,
          4.04003392e-20,   5.95606355e-19,   5.98905652e-19]])

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
#            plt.plot(porw, z,'b-*', label='expected')
#            plt.plot(a.porw, z, 'r-+', label='calculated')
#            plt.gca().set_xlabel('porw')
#            plt.gca().set_ylabel('depth')
##            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.figure()
#            plt.plot(pora, z,'b-*', label='expected')
#            plt.plot(a.pora, z, 'r-+', label='calculated')
#            plt.gca().set_xlabel('pora')
#            plt.gca().set_ylabel('depth')
##            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.show()


            assert_allclose(a.pora, pora, atol=0.2,
                            err_msg = ("Fail. test_shanetal2012_uwbot_sin_ka_divide_kw_is_10_drn0, pora, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.porw, porw, atol=0.2,
                            err_msg = ("Fail. test_shanetal2012_uwbot_sin_ka_divide_kw_is_10_drn0, porw, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_shanetal2012_uwtop_sin_ka_divide_kw_is_10_drn0, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.set, settle, atol=5e-3,
#                            err_msg = ("Fail. test_shanetal2012_uwtop_sin_ka_divide_kw_is_10_drn0, set, "
#                                "implementation='%s', dT=%s" % (impl, dT)))

def test_shanetal2012_uabot_sin_ka_divide_kw_is_10_drn0():
    """test against for shan et al 2012


    PTPB drainage
    dsig = 100kPa instantly
    uabot = sin(2*np.pi/1e8)

    ka/kw=10

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



    z = np.array([0, 3.0, 5.0, 8.0, 10.0])/10
    t = np.array([1e6,3e6, 1e8,3e8, 1e9, 2e9 ])


    reader = textwrap.dedent("""\
#############################
##shanetal2012 input
#kw = 1e-10
#ka = 10 * kw
#H=10
#Cw=-0.75
#Cvw=-5e-8
#Ca = -0.0775134
#Cva=-64504.4 * ka
#drn=0
#Csw=0.25
#Csa=0.155027
#uwi=(40, 40)
#uai=(20, 20)
#nterms=200
#f=f1=f2=f3=f4=None
#f4 = dict([('type', 'sin'), ('q0',100.0), ('omega',2*np.pi / 1e9)])
#z = np.array([0, 3.0, 5.0, 8.0, 10.0])
#t = np.array([1e6,3e6, 1e8,3e8, 1e9, 2e9 ])
#
#porw, pora = shanetal2012(z, t, H, Cw, Cvw, Ca, Cva, drn, Csw, Csa,
#             uwi, uai, nterms, f=f, f1=f1, f2=f2, f3=f3, f4=f4)
#########################


#from geotecha.piecewise.piecewise_linear_1d import PolyLine
#import numpy as np
# shan et al,
H = 10 #m
drn = 0
neig = 30

mvref = 1e-4 #1/kPa
kwref = 1.0e-10 #m/s

karef = kwref * 10 #m/s
Daref = karef / 10 # from equation ka=Da*g

wa = 29.0e-3 #kg / mol
R = 8.31432 #J/(mol.K)
ua_= 111 #kPa
T = 273.16 + 20
dTa = Daref /(mvref) / (wa*ua_/(R*T))/ H ** 2
dTw = kwref / mvref / 10 / H**2
dT = max(dTw, dTa)

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


abot_vs_time = PolyLine([0, 0.0, 1e12], [0,100,100])
abot_omega_phase = (2*np.pi/1e9, -np.pi/2)

ppress_z = np.{z}
#avg_ppress_z_pairs = [[0,1]]
#settlement_z_pairs = [[0,1]]

tvals = np.{t}

#ppress_z_tval_indexes = slice(None, len(tua)+len(tuw))
#avg_ppress_z_pairs_tval_indexes = slice(None, None)#[0,4,6]
#settlement_z_pairs_tval_indexes = slice(len(tua)+len(tuw),len(tua)+len(tuw)+len(tset))

save_data_to_file= False
save_figures_to_file= False
show_figures= False

    """.format(t=repr(t), z = repr(z)))

    porw = np.array(
     [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
       [  2.51160269e+01,   2.53809684e+01,   2.86794389e+01,
          1.84225700e+01,   1.31402505e+01,   1.30905228e+01],
       [  2.52028754e+01,   2.56521851e+01,   3.96023735e+01,
          2.31985517e+01,   2.13815989e+01,   2.13201321e+01],
       [  2.53451599e+01,   2.60799853e+01,   3.20328797e+01,
          1.30290743e+01,   2.24473606e+01,   2.24112313e+01],
       [  2.67914232e-14,   1.68851653e-14,   9.04637595e-15,
          2.05636474e-15,   7.48375877e-15,   7.47623126e-15]])

    pora = np.array(
     [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
       [  1.69193072e-01,   5.22191992e-01,   1.75980381e+01,
          2.85434458e+01,  -4.25318313e-02,  -4.25288413e-02],
       [  2.84908997e-01,   8.83551215e-01,   2.93404622e+01,
          4.75690971e+01,  -5.86231831e-02,  -5.86194872e-02],
       [  4.74493752e-01,   1.46227901e+00,   4.69852488e+01,
          7.60973130e+01,  -4.53849520e-02,  -4.53827796e-02],
       [  6.28314397e-01,   1.88484397e+00,   5.87785252e+01,
          9.51056516e+01,  -2.45053354e-14,  -4.89982710e-14]])

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
#            plt.plot(porw, z,'b-*', label='expected')
#            plt.plot(a.porw, z, 'r-+', label='calculated')
#            plt.gca().set_xlabel('porw')
#            plt.gca().set_ylabel('depth')
##            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.figure()
#            plt.plot(pora, z,'b-*', label='expected')
#            plt.plot(a.pora, z, 'r-+', label='calculated')
#            plt.gca().set_xlabel('pora')
#            plt.gca().set_ylabel('depth')
##            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()

#            plt.show()


            assert_allclose(a.pora, pora, atol=0.2,
                            err_msg = ("Fail. test_shanetal2012_uabot_sin_ka_divide_kw_is_10_drn0, pora, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.porw, porw, atol=0.2,
                            err_msg = ("Fail. test_shanetal2012_uabot_sin_ka_divide_kw_is_10_drn0, porw, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_shanetal2012_uabot_sin_ka_divide_kw_is_10_drn0, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.set, settle, atol=5e-3,
#                            err_msg = ("Fail. test_shanetal2012_uabot_sin_ka_divide_kw_is_10_drn0, set, "
#                                "implementation='%s', dT=%s" % (impl, dT)))

def test_shanetal2012_uwbot_sin_ka_divide_kw_is_10_drn1():
    """test against for shan et al 2012


    PTIB drainage
    dsig = 100kPa instantly
    uwbot = sin(2*np.pi/1e8)

    ka/kw=10

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



    z = np.array([0, 3.0, 5.0, 8.0, 10.0])/10

    t = np.array([1e6,3e6, 1e8,3e8, 1e9, 2e9 ])


    reader = textwrap.dedent("""\
#############################
##shanetal2012 input
#kw = 1e-10
#ka = 10 * kw
#H=10
#Cw=-0.75
#Cvw=-5e-8
#Ca = -0.0775134
#Cva=-64504.4 * ka
#drn=1
#Csw=0.25
#Csa=0.155027
#uwi=(40, 40)
#uai=(20, 20)
#nterms=200
#f=f1=f2=f3=f4=None
#f3 = dict([('type', 'sin'), ('q0',100.0), ('omega',2*np.pi / 1e9)])
#z = np.array([0, 3.0, 5.0, 8.0, 10.0])
#t = np.array([1e6,3e6, 1e8,3e8, 1e9, 2e9 ])
#
#porw, pora = shanetal2012(z, t, H, Cw, Cvw, Ca, Cva, drn, Csw, Csa,
#             uwi, uai, nterms, f=f, f1=f1, f2=f2, f3=f3, f4=f4)
#########################


#from geotecha.piecewise.piecewise_linear_1d import PolyLine
#import numpy as np
# shan et al,
H = 10 #m
drn = 1
neig = 30

mvref = 1e-4 #1/kPa
kwref = 1.0e-10 #m/s

karef = kwref * 10 #m/s
Daref = karef / 10 # from equation ka=Da*g

wa = 29.0e-3 #kg / mol
R = 8.31432 #J/(mol.K)
ua_= 111 #kPa
T = 273.16 + 20
dTa = Daref /(mvref) / (wa*ua_/(R*T))/ H ** 2
dTw = kwref / mvref / 10 / H**2
dT = max(dTw, dTa)

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


wbot_vs_time = PolyLine([0, 0.0, 1e12], [0,100,100])
wbot_omega_phase = (2*np.pi/1e9, -np.pi/2)

ppress_z = np.{z}
#avg_ppress_z_pairs = [[0,1]]
#settlement_z_pairs = [[0,1]]

tvals = np.{t}

#ppress_z_tval_indexes = slice(None, len(tua)+len(tuw))
#avg_ppress_z_pairs_tval_indexes = slice(None, None)#[0,4,6]
#settlement_z_pairs_tval_indexes = slice(len(tua)+len(tuw),len(tua)+len(tuw)+len(tset))

save_data_to_file= False
save_figures_to_file= False
show_figures= False

    """.format(t=repr(t), z = repr(z)))

    porw = np.array(
     [[   0.        ,    0.        ,    0.        ,    0.        ,
           0.        ,    0.        ],
       [  26.58929795,   25.04264099,   16.78726734,   36.10203693,
         -11.36895268,  -28.70954573],
       [  27.48213509,   25.07318195,   24.68527794,   79.86610039,
         -39.8767618 ,  -66.88571939],
       [  28.34266512,   25.10271178,   52.73815028,  216.46677406,
        -115.29650468, -151.62412153],
       [  28.62104334,   25.65787256,  125.92497964,  377.55581684,
        -147.28785742, -185.48513538]])

    pora = np.array(
     [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
       [  2.13211782e+00,   7.14154844e-02,   9.52816372e-03,
          1.50092320e-02,   6.03330523e-04,   1.64598057e-03],
       [  3.32169826e+00,   1.12105603e-01,   1.60842728e-02,
          2.38288674e-02,   2.27128868e-03,   3.89527529e-03],
       [  4.46823399e+00,   1.51440990e-01,   2.49619189e-02,
          3.27852524e-02,   6.75894240e-03,   8.94323860e-03],
       [  4.69830805e+00,   1.59457415e-01,   2.76214486e-02,
          3.45394572e-02,   8.67221235e-03,   1.09689270e-02]])

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
#            plt.plot(porw, z,'b-*', label='expected')
#            plt.plot(a.porw, z, 'r-+', label='calculated')
#            plt.gca().set_xlabel('porw')
#            plt.gca().set_ylabel('depth')
##            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.figure()
#            plt.plot(pora, z,'b-*', label='expected')
#            plt.plot(a.pora, z, 'r-+', label='calculated')
#            plt.gca().set_xlabel('pora')
#            plt.gca().set_ylabel('depth')
##            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.show()


            assert_allclose(a.pora, pora, atol=0.2,
                            err_msg = ("Fail. test_shanetal2012_uwbot_sin_ka_divide_kw_is_10_drn1, pora, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.porw, porw, atol=0.2,
                            err_msg = ("Fail. test_shanetal2012_uwbot_sin_ka_divide_kw_is_10_drn1, porw, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_shanetal2012_uwbot_sin_ka_divide_kw_is_10_drn1, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.set, settle, atol=5e-3,
#                            err_msg = ("Fail. test_shanetal2012_uwbot_sin_ka_divide_kw_is_10_drn1, set, "
#                                "implementation='%s', dT=%s" % (impl, dT)))

def test_shanetal2012_uabot_sin_ka_divide_kw_is_10_drn1():
    """test against for shan et al 2012


    PTIB drainage
    dsig = 100kPa instantly
    uabot = sin(2*np.pi/1e8)

    ka/kw=10

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



    z = np.array([0, 3.0, 5.0, 8.0, 10.0])/10

    t = np.array([1e6,3e6, 1e8,3e8, 1e9, 2e9 ])


    reader = textwrap.dedent("""\
#############################
##shanetal2012 input
#kw = 1e-10
#ka = 10 * kw
#H=10
#Cw=-0.75
#Cvw=-5e-8
#Ca = -0.0775134
#Cva=-64504.4 * ka
#drn=1
#Csw=0.25
#Csa=0.155027
#uwi=(40, 40)
#uai=(20, 20)
#nterms=200
#f=f1=f2=f3=f4=None
#f4 = dict([('type', 'sin'), ('q0',100.0), ('omega',2*np.pi / 1e9)])
#z = np.array([0, 3.0, 5.0, 8.0, 10.0])
#t = np.array([1e6,3e6, 1e8,3e8, 1e9, 2e9 ])
#
#porw, pora = shanetal2012(z, t, H, Cw, Cvw, Ca, Cva, drn, Csw, Csa,
#             uwi, uai, nterms, f=f, f1=f1, f2=f2, f3=f3, f4=f4)
#########################


#from geotecha.piecewise.piecewise_linear_1d import PolyLine
#import numpy as np
# shan et al,
H = 10 #m
drn = 1
neig = 30

mvref = 1e-4 #1/kPa
kwref = 1.0e-10 #m/s

karef = kwref * 10 #m/s
Daref = karef / 10 # from equation ka=Da*g

wa = 29.0e-3 #kg / mol
R = 8.31432 #J/(mol.K)
ua_= 111 #kPa
T = 273.16 + 20
dTa = Daref /(mvref) / (wa*ua_/(R*T))/ H ** 2
dTw = kwref / mvref / 10 / H**2
dT = max(dTw, dTa)

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


abot_vs_time = PolyLine([0, 0.0, 1e12], [0,100,100])
abot_omega_phase = (2*np.pi/1e9, -np.pi/2)

ppress_z = np.{z}
#avg_ppress_z_pairs = [[0,1]]
#settlement_z_pairs = [[0,1]]

tvals = np.{t}

#ppress_z_tval_indexes = slice(None, len(tua)+len(tuw))
#avg_ppress_z_pairs_tval_indexes = slice(None, None)#[0,4,6]
#settlement_z_pairs_tval_indexes = slice(len(tua)+len(tuw),len(tua)+len(tuw)+len(tset))

save_data_to_file= False
save_figures_to_file= False
show_figures= False

    """.format(t=repr(t), z = repr(z)))

    porw = np.array(
     [[   0.        ,    0.        ,    0.        ,    0.        ,
           0.        ,    0.        ],
       [  27.19164196,   28.29102406,  147.68478678,  205.47372395,
          14.89508193,   22.69058044],
       [  28.55716017,   30.5782369 ,  239.53584802,  325.11413035,
          39.82092644,   51.96319581],
       [  30.34545405,   34.2661944 ,  354.82572495,  445.43774452,
          99.79483763,  116.12707162],
       [  31.27378709,   36.56132485,  388.39213332,  468.94479383,
         124.47109727,  141.64405217]])

    pora = np.array(
     [[   0.        ,    0.        ,    0.        ,    0.        ,
           0.        ,    0.        ],
       [   2.93465685,    4.39943096,  175.2438378 ,  285.71225215,
          -1.33569629,   -1.33616502],
       [   4.75401861,    7.44682028,  292.17273624,  476.15084674,
          -2.10474397,   -2.10547406],
       [   7.13666975,   12.36051384,  467.86519385,  761.70033099,
          -2.89271842,   -2.89370044],
       [   8.47924012,   15.96759779,  585.28135826,  951.96261034,
          -3.06544843,   -3.066481  ]])

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
#            plt.plot(porw, z,'b-*', label='expected')
#            plt.plot(a.porw, z, 'r-+', label='calculated')
#            plt.gca().set_xlabel('porw')
#            plt.gca().set_ylabel('depth')
##            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.figure()
#            plt.plot(pora, z,'b-*', label='expected')
#            plt.plot(a.pora, z, 'r-+', label='calculated')
#            plt.gca().set_xlabel('pora')
#            plt.gca().set_ylabel('depth')
##            plt.gca().set_xscale('log')
#            plt.gca().grid()
#            plt.legend()
#            leg = plt.legend(loc=3 )
#            leg.draggable()
#
#            plt.show()


            assert_allclose(a.pora, pora, atol=0.2,
                            err_msg = ("Fail. test_shanetal2012_uabot_sin_ka_divide_kw_is_10_drn1, pora, "
                                "implementation='%s', dT=%s" % (impl, dT)))
            assert_allclose(a.porw, porw, atol=0.2,
                            err_msg = ("Fail. test_shanetal2012_uabot_sin_ka_divide_kw_is_10_drn1, porw, "
                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.avp, avp, atol=5,
#                            err_msg = ("Fail. test_shanetal2012_uabot_sin_ka_divide_kw_is_10_drn0, avp, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.set, settle, atol=5e-3,
#                            err_msg = ("Fail. test_shanetal2012_uabot_sin_ka_divide_kw_is_10_drn0, set, "
#                                "implementation='%s', dT=%s" % (impl, dT)))



if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])

#    test_shanetal2012_sig_ka_divide_kw_is_100()
#    test_shanetal2012_sig_ka_divide_kw_is_1()
#    test_shanetal2012_sig_sin_ka_divide_kw_is_100()
#    test_shanetal2012_sig_sin_ka_divide_kw_is_1()
#    test_shanetal2012_sig_ka_divide_kw_is_100_drn1()
#
#    test_shanetal2012_sig_exp_ka_divide_kw_is_1()
#    test_shanetal2012_sig_exp_ka_divide_kw_is_100_drn1()
#    test_shanetal2012_uwtop_sin_ka_divide_kw_is_1_drn1()
#    test_shanetal2012_uwtop_sin_ka_divide_kw_is_100_drn1()
#    test_shanetal2012_uwtop_sin_ka_divide_kw_is_10_drn0()
#    test_shanetal2012_uatop_sin_ka_divide_kw_is_10_drn1()
#    test_shanetal2012_uwbot_sin_ka_divide_kw_is_10_drn0()
#    test_shanetal2012_uabot_sin_ka_divide_kw_is_10_drn0()
#    test_shanetal2012_uwbot_sin_ka_divide_kw_is_10_drn1()
#    test_shanetal2012_uabot_sin_ka_divide_kw_is_10_drn1()