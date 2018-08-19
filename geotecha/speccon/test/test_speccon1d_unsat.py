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

import geotecha.mathematics.transformations as transformations

DEBUG=False

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



def test_shanetal2014_3_layers_instant():
    """

    fig2a and 5a


    Shan et al (2014) 'Analytical solution for the 1D
        consolidation of unsaturated multi-layered soil'

    test data is digitized from article

    dsig=?kPa instantly


    other data:
    h = [3,4,3]
    n = [0.45, 0.5, 0.4]
    S = [0.8, 0.6, 0.7]
    kw = [1e-10, 1e-9, 1e-10] (m/s)
    ka = [1e-9, 1e-8, 1e-9]
    m1s = -2.5e-4 (1/kPa)
    m2s / m1s = 0.4
    m1w / m1s = 0.2
    m2w/m1w = 4
    #m1s=m1a+m1w
    #m2s=m2a+m2w
    gamw = 10,000 N/m3
    uatm = ua_ = 101kPa
    T = 293.16 K
    R = 8.31432J/molK
    wa=29x10**3 kg/mol



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
#    Note digitized settlement data does not include immediate
#    undrained deformation. data is S/S0, S0=m1ks*q0*H

    """

    t_for_z_vs_uw = np.array([1e1, 1e3, 1e4, 1e6, 1e8, 1e9])
    #uw is uw/q0
    uw_for_z_vs_uw = [
      [ 0.001,  0.288,  0.386,  0.401,  0.401,  0.401,  0.401,  0.38 ,
        0.354,  0.341,  0.331,  0.327,  0.326,  0.327,  0.328,  0.327,
        0.335,  0.356,  0.368,  0.368,  0.369,  0.37 ],
      [ 0.001,  0.191,  0.249,  0.279,  0.33 ,  0.38 ,  0.395,  0.401,
        0.401,  0.401,  0.396,  0.381,  0.364,  0.35 ,  0.339,  0.333,
        0.328,  0.328,  0.328,  0.331,  0.334,  0.336,  0.351,  0.363,
        0.37 ,  0.37 ,  0.37 ,  0.369],
      [ 0.001,  0.234,  0.251,  0.285,  0.319,  0.339,  0.357,  0.361,
        0.354,  0.345,  0.341,  0.34 ,  0.339,  0.337,  0.336,  0.337,
        0.338,  0.339,  0.347,  0.357,  0.361,  0.368,  0.37 ],
      [ 0.   ,  0.06 ,  0.118,  0.168,  0.214,  0.246,  0.255,  0.258,
        0.264,  0.265,  0.266,  0.267,  0.267,  0.267,  0.268,  0.269,
        0.269],
      [ 0.   ,  0.046,  0.087,  0.121,  0.159,  0.201,  0.208,  0.213,
        0.216,  0.229,  0.236,  0.24 ,  0.242],
      [ 0.003,  0.01 ,  0.018,  0.028,  0.036,  0.038,  0.039,  0.039,
        0.04 ,  0.045,  0.046,  0.047],
        ]
    #z is z/H
    z_for_z_vs_uw = [
      [ 0.   ,  0.002,  0.004,  0.013,  0.125,  0.205,  0.29 ,  0.297,
        0.299,  0.299,  0.31 ,  0.319,  0.407,  0.523,  0.655,  0.69 ,
        0.7  ,  0.702,  0.71 ,  0.755,  0.877,  0.976],
      [ 0.002,  0.002,  0.006,  0.007,  0.028,  0.058,  0.078,  0.105,
        0.147,  0.202,  0.237,  0.26 ,  0.279,  0.29 ,  0.3  ,  0.37 ,
        0.429,  0.478,  0.576,  0.625,  0.677,  0.707,  0.725,  0.745,
        0.779,  0.832,  0.919,  0.986],
      [ 0.002,  0.006,  0.007,  0.038,  0.078,  0.11 ,  0.158,  0.202,
        0.25 ,  0.289,  0.305,  0.36 ,  0.437,  0.508,  0.563,  0.625,
        0.682,  0.703,  0.743,  0.79 ,  0.832,  0.882,  0.944],
      [ 0.002,  0.009,  0.017,  0.029,  0.046,  0.064,  0.086,  0.148,
        0.257,  0.303,  0.466,  0.566,  0.67 ,  0.703,  0.81 ,  0.894,
        0.999],
      [ 0.003,  0.067,  0.121,  0.174,  0.235,  0.298,  0.435,  0.564,
        0.7  ,  0.792,  0.855,  0.947,  1.001],
      [ 0.002,  0.072,  0.149,  0.219,  0.299,  0.437,  0.572,  0.667,
        0.704,  0.815,  0.893,  0.997],
        ]


    t_for_z_vs_ua = np.array([1e1, 1e3, 1e4, 1e5, 1e6])
    ua_for_z_vs_ua = [
      [ 0.   ,  0.119,  0.183,  0.2  ,  0.201,  0.201,  0.202,  0.189,
        0.153,  0.119,  0.108,  0.102,  0.102,  0.102,  0.105,  0.113,
        0.129,  0.143,  0.154,  0.159,  0.159,  0.159,  0.159,  0.159],
      [ 0.   ,  0.027,  0.065,  0.094,  0.128,  0.157,  0.179,  0.191,
        0.197,  0.201,  0.201,  0.201,  0.202,  0.201,  0.198,  0.189,
        0.179,  0.166,  0.15 ,  0.135,  0.123,  0.117,  0.11 ,  0.107,
        0.104,  0.103,  0.102,  0.103,  0.104,  0.106,  0.108,  0.11 ,
        0.114,  0.122,  0.13 ,  0.138,  0.147,  0.154,  0.158,  0.16 ,
        0.16 ,  0.16 ],
      [ 0.   ,  0.016,  0.04 ,  0.057,  0.086,  0.114,  0.132,  0.142,
        0.147,  0.148,  0.146,  0.139,  0.127,  0.122,  0.119,  0.118,
        0.116,  0.116,  0.116,  0.117,  0.117,  0.125,  0.137,  0.144,
        0.152,  0.156,  0.158,  0.159,  0.159],
      [ 0.   ,  0.021,  0.048,  0.063,  0.075,  0.089,  0.103,  0.105,
        0.108,  0.11 ,  0.111,  0.113,  0.114,  0.118,  0.122,  0.125,
        0.129,  0.129],
      [ 0.   ,  0.006,  0.012,  0.02 ,  0.021,  0.022,  0.022,  0.022,
        0.024,  0.025,  0.025],
        ]

    z_for_z_vs_ua = [
      [ 0.   ,  0.003,  0.008,  0.012,  0.023,  0.161,  0.283,  0.295,
        0.298,  0.303,  0.31 ,  0.318,  0.396,  0.676,  0.693,  0.699,
        0.702,  0.702,  0.707,  0.716,  0.772,  0.853,  0.914,  0.969],
      [ 0.   ,  0.005,  0.016,  0.024,  0.035,  0.047,  0.061,  0.078,
        0.086,  0.107,  0.12 ,  0.154,  0.178,  0.198,  0.225,  0.246,
        0.261,  0.269,  0.283,  0.295,  0.3  ,  0.323,  0.367,  0.4  ,
        0.446,  0.491,  0.515,  0.546,  0.583,  0.617,  0.648,  0.67 ,
        0.7  ,  0.709,  0.716,  0.728,  0.74 ,  0.757,  0.779,  0.816,
        0.887,  0.976],
      [ 0.003,  0.01 ,  0.031,  0.048,  0.074,  0.103,  0.135,  0.155,
        0.181,  0.201,  0.228,  0.251,  0.284,  0.3  ,  0.369,  0.42 ,
        0.48 ,  0.542,  0.605,  0.665,  0.704,  0.731,  0.772,  0.802,
        0.853,  0.899,  0.938,  0.971,  0.998],
      [ 0.   ,  0.061,  0.136,  0.175,  0.214,  0.26 ,  0.298,  0.357,
        0.464,  0.542,  0.626,  0.702,  0.727,  0.761,  0.806,  0.858,
        0.933,  1.001],
      [ 0.002,  0.082,  0.177,  0.297,  0.418,  0.554,  0.661,  0.701,
        0.812,  0.914,  0.995],
        ]


    z_for_uw_vs_t = np.array([1.5, 4.5, 8.5])
    t_for_uw_vs_t = [
      [  1.42100000e+00,   4.12800000e+00,   1.23410000e+01,
         5.68400000e+01,   7.17913000e+02,   1.43367000e+03,
         2.27363000e+03,   3.30788000e+03,   4.81306000e+03,
         6.06455000e+03,   7.00754000e+03,   9.08780000e+03,
         1.11271000e+04,   1.66725000e+04,   2.35847000e+04,
         3.14819000e+04,   4.32394000e+04,   6.85894000e+04,
         1.08791000e+05,   1.88155000e+05,   3.25448000e+05,
         5.16347000e+05,   8.19144000e+05,   1.83676000e+06,
         3.46322000e+06,   4.89594000e+06,   6.72573000e+06,
         8.72317000e+06,   1.26974000e+07,   1.55475000e+07,
         1.90345000e+07,   2.33071000e+07,   2.93759000e+07,
         3.70178000e+07,   4.66387000e+07,   6.22643000e+07,
         8.55306000e+07,   1.14164000e+08,   1.91904000e+08,
         2.71400000e+08,   3.41985000e+08,   4.56585000e+08,
         5.58882000e+08,   7.46091000e+08,   9.96106000e+08,
         1.40839000e+09,   2.04896000e+09,   2.89590000e+09],
      [  1.23600000e+00,   3.50792000e+02,   7.20940000e+02,
         1.66261000e+03,   3.22543000e+03,   6.43960000e+03,
         1.24921000e+04,   2.42368000e+04,   3.52516000e+04,
         6.09651000e+04,   9.95393000e+04,   1.53433000e+05,
         2.04749000e+05,   2.81271000e+05,   3.54390000e+05,
         5.01146000e+05,   6.68853000e+05,   9.45692000e+05,
         1.37634000e+06,   1.94554000e+06,   3.66834000e+06,
         1.30371000e+07,   1.89620000e+07,   2.67987000e+07,
         3.78852000e+07,   5.05610000e+07,   7.35855000e+07,
         9.82439000e+07,   1.27409000e+08,   1.56007000e+08,
         2.08305000e+08,   2.55124000e+08,   3.21570000e+08,
         3.93884000e+08,   4.82343000e+08,   5.90696000e+08,
         6.82810000e+08,   8.12285000e+08,   9.66359000e+08,
         1.21822000e+09,   1.58001000e+09,   2.10846000e+09,
         2.89576000e+09],
      [  1.23300000e+00,   1.39543000e+03,   2.95200000e+03,
         5.56549000e+03,   1.24764000e+04,   2.87852000e+04,
         6.83609000e+04,   1.36602000e+05,   2.29677000e+05,
         4.59284000e+05,   7.08468000e+05,   1.26217000e+06,
         2.00252000e+06,   3.46288000e+06,   1.16165000e+07,
         3.00701000e+07,   5.35142000e+07,   8.24844000e+07,
         1.16636000e+08,   1.51268000e+08,   1.90647000e+08,
         2.47362000e+08,   3.02988000e+08,   3.71159000e+08,
         4.54668000e+08,   6.25380000e+08,   8.11854000e+08,
         1.05378000e+09,   1.29031000e+09,   1.77332000e+09,
         2.65580000e+09,   3.64678000e+09],
        ]
    #uw is uw/q0
    uw_for_uw_vs_t = [
      [ 0.401,  0.402,  0.401,  0.402,  0.4  ,  0.4  ,  0.399,  0.394,
        0.387,  0.377,  0.37 ,  0.36 ,  0.347,  0.331,  0.313,  0.302,
        0.297,  0.292,  0.288,  0.283,  0.276,  0.268,  0.261,  0.251,
        0.248,  0.241,  0.232,  0.22 ,  0.207,  0.193,  0.182,  0.169,
        0.154,  0.142,  0.134,  0.12 ,  0.112,  0.102,  0.091,  0.077,
        0.066,  0.052,  0.045,  0.032,  0.018,  0.008,  0.004,  0.002],
      [ 0.325,  0.325,  0.327,  0.329,  0.332,  0.336,  0.339,  0.34 ,
        0.34 ,  0.336,  0.33 ,  0.323,  0.317,  0.307,  0.298,  0.286,
        0.278,  0.269,  0.258,  0.253,  0.249,  0.249,  0.249,  0.247,
        0.24 ,  0.233,  0.222,  0.207,  0.197,  0.184,  0.168,  0.15 ,
        0.134,  0.115,  0.1  ,  0.084,  0.07 ,  0.057,  0.044,  0.025,
        0.013,  0.008,  0.003],
      [ 0.369,  0.369,  0.369,  0.367,  0.361,  0.356,  0.349,  0.338,
        0.323,  0.3  ,  0.28 ,  0.26 ,  0.252,  0.249,  0.25 ,  0.249,
        0.248,  0.241,  0.23 ,  0.219,  0.205,  0.186,  0.167,  0.146,
        0.125,  0.094,  0.066,  0.041,  0.028,  0.011,  0.003,  0.002],
        ]

    z_for_ua_vs_t = z_for_uw_vs_t
    t_for_ua_vs_t = [
      [  1.12200000e+00,   2.66400000e+00,   1.09350000e+01,
         3.08580000e+01,   1.59497000e+02,   6.93508000e+02,
         1.55414000e+03,   2.53662000e+03,   3.48279000e+03,
         4.51407000e+03,   5.68451000e+03,   6.95509000e+03,
         8.50966000e+03,   9.82858000e+03,   1.16838000e+04,
         1.34947000e+04,   1.69937000e+04,   2.07921000e+04,
         2.40146000e+04,   3.20355000e+04,   4.39850000e+04,
         7.17910000e+04,   1.04418000e+05,   1.47557000e+05,
         2.02597000e+05,   2.62588000e+05,   3.30673000e+05,
         4.54016000e+05,   5.39716000e+05,   6.41592000e+05,
         8.31572000e+05,   1.17513000e+06,   1.56763000e+06,
         2.21530000e+06,   3.04161000e+06],
      [  1.22400000e+00,   4.87900000e+00,   2.74980000e+01,
         2.18991000e+02,   5.35070000e+02,   9.80029000e+02,
         1.79501000e+03,   3.19434000e+03,   5.36614000e+03,
         8.50966000e+03,   1.31113000e+04,   2.02014000e+04,
         3.29722000e+04,   5.38163000e+04,   8.29179000e+04,
         1.20601000e+05,   1.56312000e+05,   2.14617000e+05,
         2.78167000e+05,   3.71075000e+05,   4.67291000e+05,
         5.55496000e+05,   7.19982000e+05,   9.88539000e+05,
         1.17513000e+06,   1.52310000e+06,   2.09122000e+06,
         2.78970000e+06],
       [  1.18900000e+00,   3.76500000e+00,   1.41720000e+01,
         7.11730000e+01,   3.47277000e+02,   1.50999000e+03,
         2.92977000e+03,   5.52303000e+03,   9.01454000e+03,
         1.51435000e+04,   2.69487000e+04,   5.38163000e+04,
         9.30488000e+04,   1.47557000e+05,   1.91250000e+05,
         2.47881000e+05,   3.21280000e+05,   4.04584000e+05,
         4.80953000e+05,   6.05658000e+05,   7.62699000e+05,
         9.06665000e+05,   1.14175000e+06,   1.39695000e+06,
         1.81060000e+06,   2.55864000e+06,   3.51303000e+06]
        ]
    #ua is ua/q0
    ua_for_ua_vs_t = [
      [ 0.201,  0.201,  0.202,  0.201,  0.201,  0.201,  0.201,  0.197,
        0.191,  0.184,  0.173,  0.163,  0.152,  0.142,  0.131,  0.12 ,
        0.108,  0.096,  0.085,  0.073,  0.063,  0.056,  0.053,  0.048,
        0.044,  0.04 ,  0.035,  0.029,  0.025,  0.02 ,  0.014,  0.009,
        0.004,  0.002,  0.   ],
      [ 0.102,  0.102,  0.102,  0.102,  0.101,  0.103,  0.105,  0.109,
        0.112,  0.116,  0.119,  0.121,  0.12 ,  0.117,  0.111,  0.104,
        0.097,  0.088,  0.077,  0.066,  0.058,  0.049,  0.036,  0.024,
        0.016,  0.009,  0.003,  0.001],
      [ 0.159,  0.159,  0.158,  0.159,  0.159,  0.159,  0.158,  0.156,
        0.153,  0.148,  0.142,  0.137,  0.127,  0.116,  0.107,  0.097,
        0.085,  0.074,  0.064,  0.052,  0.039,  0.031,  0.021,  0.013,
        0.007,  0.002,  0.001],
        ]

    z = np.linspace(0, 1, 101)

    t = np.logspace(0, 10, 101) #these should include the t for z vs u values.

    dq = 1.0
    reader = textwrap.dedent("""\
#from geotecha.piecewise.piecewise_linear_1d import PolyLine
#import numpy as np
# shan et al,
H = 10 #m
drn = 1
neig = 100

mvref = 1e-4 #1/kPa
kwref = 1.0e-9 #m/s

karef = 1e-8 #m/s
Daref = karef / 10 # from equation ka=Da*g

wa = 29.0e-3 #kg / mol
R = 8.31432 #J/(mol.K)
ua_= 101 #kPa
T = 273.16 + 20
dTa = Daref /(mvref) / (wa*ua_/(R*T))/ H ** 2
dTw = kwref / mvref/ 10 / H**2
dT = max(dTw,dTa)

kw = PolyLine([0,0.3,0.7], [0.3,0.7,1.0], [0.1,1,0.1], [0.1,1,0.1])
Da = PolyLine([0,0.3,0.7], [0.3,0.7,1.0], [0.1,1,0.1], [0.1,1,0.1])
S = PolyLine([0,0.3,0.7], [0.3,0.7,1.0], [0.8,0.6,0.7], [0.8,0.6,0.7])
n = PolyLine([0,0.3,0.7], [0.3,0.7,1.0], [0.45,0.5,0.4], [0.45,0.5,0.4])

m1s = -2.5
m2s = 0.4*m1s
m1w = 0.2*m1s
m2w = 4*m1w
m1a = m1s-m1w
m2a = m2s-m2w

#print(m1w,m2w,m1a,m2a)
m1kw = PolyLine([0,1], [m1w]*2)
m2w =  PolyLine([0,1], [m2w]*2)
m1ka = PolyLine([0,1], [m1a]*2)
m2a =  PolyLine([0,1], [m2a]*2)


surcharge_vs_depth = PolyLine([0,1], [1,1])
surcharge_vs_time = PolyLine([0,0,10000], [0, {dq}, {dq}])

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

    """.format(t=repr(t), z=repr(z),dq=dq))


    i_for_uw_vs_t = np.searchsorted(z, z_for_uw_vs_t / 10.0)
    i_for_ua_vs_t = np.searchsorted(z, z_for_ua_vs_t / 10.0)

    i_for_z_vs_uw = np.searchsorted(t, t_for_z_vs_uw)
    i_for_z_vs_ua = np.searchsorted(t, t_for_z_vs_ua)




    for impl in ["vectorized"]:
        for dT in [10]:
            a = Speccon1dUnsat(reader + "\n" +
                            "implementation = '%s'" % impl + "\n")

            a.make_all()

            #calculated values to plot
            calc_uw_vs_t = [a.porw[v,:]/dq for v in i_for_uw_vs_t]
            calc_ua_vs_t = [a.pora[v,:]/dq for v in i_for_ua_vs_t]
            calc_z_vs_uw = [a.porw[:,v]/dq for v in i_for_z_vs_uw]
            calc_z_vs_ua = [a.pora[:,v]/dq for v in i_for_z_vs_ua]

            #interplolate the calcualted ppress at digitized times
            compare_uw_vs_t = [np.interp(x, t, yp) for (yp, x) in zip(calc_uw_vs_t,  t_for_uw_vs_t)]
            compare_ua_vs_t = [np.interp(x, t, yp) for (yp, x) in zip(calc_ua_vs_t,  t_for_ua_vs_t)]
            compare_z_vs_uw = [np.interp(x, z, yp) for (yp, x) in zip(calc_z_vs_uw,  z_for_z_vs_uw)]
            compare_z_vs_ua = [np.interp(x, z, yp) for (yp, x) in zip(calc_z_vs_ua,  z_for_z_vs_ua)]

            if DEBUG:
                fig = plt.figure()

                ax = fig.add_subplot('221')
                ax.set_xlabel('time')
                ax.set_ylabel('porw')
                ax.set_xscale('log')
                ax.grid()
                for jj, zz in enumerate(z_for_uw_vs_t):
                    label_calc = "$z={}m$".format(zz)
                    if jj==0:
                        label_expect = "expect"
                    else:
                        label_expect = None

                    ax.plot(t_for_uw_vs_t[jj], uw_for_uw_vs_t[jj], 'b-', label=label_expect)
                    ax.plot(t, calc_uw_vs_t[jj], 'r-+', label=label_calc)

                leg = ax.legend(loc=3)
                plt.setp(leg.get_title(),fontsize=8)
                leg.get_frame().set_edgecolor('white')
                leg.draggable()


                ax = fig.add_subplot('222')
                ax.set_xlabel('time')
                ax.set_ylabel('pora')
                ax.set_xscale('log')
                ax.grid()
                for jj, zz in enumerate(z_for_ua_vs_t):
                    ax.plot(t_for_ua_vs_t[jj], ua_for_ua_vs_t[jj], 'b-', label='expected')
                    ax.plot(t, calc_ua_vs_t[jj], 'r-+', label="z={}m".format(zz))
    #            leg = ax.legend(loc=3 )
    #            leg.draggable()

                ax = fig.add_subplot('223')
                ax.set_xlabel('porw')
                ax.set_ylabel('z/H')
                ax.invert_yaxis()
                ax.grid()
                for jj, tt in enumerate(t_for_z_vs_uw):
                    ax.plot(uw_for_z_vs_uw[jj], z_for_z_vs_uw[jj], 'b-', label='expected')
                    ax.plot(calc_z_vs_uw[jj], z, 'r-+', label="t={}".format(tt))
    #            leg = ax.legend(loc=3 )
    #            leg.draggable()

                ax = fig.add_subplot('224')
                ax.set_xlabel('pora')
                ax.set_ylabel('z/H')
                ax.invert_yaxis()
                ax.grid()
                for jj, tt in enumerate(t_for_z_vs_ua):
                    ax.plot(ua_for_z_vs_ua[jj], z_for_z_vs_ua[jj], 'b-', label='expected')
                    ax.plot(calc_z_vs_ua[jj], z, 'r-+', label="t={}".format(tt))
    #            leg = ax.legend(loc=3 )
    #            leg.draggable()

                plt.show()

            for zz, calc, expect in zip(z_for_uw_vs_t,
                                                   compare_uw_vs_t,
                                                   uw_for_uw_vs_t):
                msg = ("Fail. test_shanetal2014_3_layers_instant."
                       "uw_vs_t, z={zz}. "
                       "implementation={impl}.").format(zz=zz, impl=impl)
                assert_allclose(calc, expect, atol=1e-2,
                            err_msg=msg)

            for zz, calc, expect in zip(z_for_ua_vs_t,
                                                   compare_ua_vs_t,
                                                   ua_for_ua_vs_t):
                msg = ("Fail. test_shanetal2014_3_layers_instant."
                       "ua_vs_t, z={zz}. "
                       "implementation={impl}.").format(zz=zz, impl=impl)
                assert_allclose(calc, expect, atol=1e-2,
                            err_msg=msg)

            for tt, calc, expect in zip(t_for_z_vs_ua[1:],
                                                   compare_z_vs_ua[1:],
                                                   ua_for_z_vs_ua[1:]):
                #miss the first t value as gibbs errors can be large
                msg = ("Fail. test_shanetal2014_3_layers_instant."
                       "z_vs_ua, t={tt}. "
                       "implementation={impl}.").format(tt=tt, impl=impl)
                assert_allclose(calc[4:], expect[4:], atol=1e-2,
                            err_msg=msg)

            for tt, calc, expect in zip(t_for_z_vs_uw[1:],
                                                   compare_z_vs_uw[1:],
                                                   uw_for_z_vs_uw[1:]):
                #miss the first two as they don't play nicely and gibbs.
                msg = ("Fail. test_shanetal2014_3_layers_instant."
                       "z_vs_uw, t={tt}. "
                       "implementation={impl}.").format(tt=tt, impl=impl)
                assert_allclose(calc[4:], expect[4:], atol=1e-2,
                            err_msg=msg)

#            assert_allclose(a.pora[:,:len(tua)], pora, atol=1.0,
#                            err_msg = ("Fail. test_shanetal2012_sig_ka_divide_kw_is_100, pora, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.porw[:,len(tua):len(tua)+len(tuw)], porw, atol=2.0,
#                            err_msg = ("Fail. test_shanetal2012_sig_ka_divide_kw_is_100, porw, "
#                                "implementation='%s', dT=%s" % (impl, dT)))
##            assert_allclose(a.avp, avp, atol=5,
##                            err_msg = ("Fail. test_shanetal2012_sig_ka_divide_kw_is_100, avp, "
##                                "implementation='%s', dT=%s" % (impl, dT)))
#            assert_allclose(a.set, settle, atol=5e-3,
#                            err_msg = ("Fail. test_shanetal2012_sig_ka_divide_kw_is_100, set, "
#                                "implementation='%s', dT=%s" % (impl, dT)))



if __name__ == '__main__':
    import nose
#    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
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
#    test_shanetal2014_3_layers_instant()