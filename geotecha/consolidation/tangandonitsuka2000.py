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
Tang and Onitsuka (2000) "Consolidation by vertical drains under
time-dependent loading".

"""
from __future__ import print_function, division

import numpy as np
from matplotlib import pyplot as plt

import math
#import textwrap
#import scipy.special
import geotecha.piecewise.piecewise_linear_1d as pwise


def tangandonitsuka2000(z, t, kv, kh, ks, kw, mv, gamw, rw, rs, re, H,
                        drn, surcharge_vs_time=((0, 0, 100.0), (0, 1.0, 1.0)),
                        tpor=None, nterms=20):
    """Consolidation by vertical drains under time-dependent loading

    An implementation of Tang and Onitsuka (2000)[1]_.

    Features:

     - Single layer.
     - Soil and drain properties constant with time.
     - Vertical and radial drainage
     - Well resistance.
     - Load is uniform with depth but piecewise linear with time.
     - Radially averaged pore pressure at depth.
     - Average pore pressure of whole layer vs time.
     - Settlement of whole layer vs time.


    Parameters
    ----------
    z : float or 1d array/list of float
        Depth values for output.
    t : float or 1d array/list of float
        Time for degree of consolidation calcs.
    kv, kh, ks, kw: float
        Vertical, horizontal, smear zone, and drain permeability.
    mv : float
        Volume compressibility.
    gamw : float
        Unit weight of water.
    rw, rs, re: float
        Drain radius, smear radius, radius of influence.
    H : float
        Drainage path length.
    drn : [0,1]
        Drainage boundary condition. drn=0 is pervious top pervious bottom.
        drn=1 is perviousbottom, impervious bottom.
    surcharge_vs_time: 2 element tuple or array/list, optional
        (time, magnitude). default = ((0,0,100.0), (0,1.0,1.0)), i.e. instant
        load of magnitude one.
    tpor : float or 1d array/list of float, optional
        Time values for pore pressure vs depth calcs.  Default tpor=None i.e.
        time values will be taken from `t`.
    nterms : int, optional
        Maximum number of series terms.  Default nterms=20


    Returns
    -------
    por : 2d array of float
        Pore pressure at depth and time.  ppress is an array of size
        (len(z), len(t)).
    avp : 1d array of float
        Average pore pressure of layer
    settlement : 1d array of float
        surface settlement


    References
    ----------
    .. [1] Tang, Xiao-Wu, and Katsutada Onitsuka. 'Consolidation by
           vertical drains under Time-Dependent Loading'. Int
           Journal for Numerical and Analytical Methods in
           Geomechanics 24, no. 9 (2000): 739-51.
           doi:10.1002/1096-9853(20000810)24:9<739::AID-NAG94>3.0.CO;2-B.


    """

    def F_func(n, s, kap):
        term1 = (math.log(n/s) + kap*math.log(s)-0.75) * n**2 / (n**2 - 1)
        term2 = s**2 / (n**2 - 1) * (1-kap) * (1-s**2/4/n**2)
        term3 = kap/ (n**2-1) * (1- 1/4/n**2)
        return term1 + term2 +term3


    def _calc_Tm(alp, t, mag_vs_time):
        """calculate the Tm expression at a given time

        Parameters
        ----------
        alp : float
            eigenvalue, note that this will be squared
        t : float
            time value
        mag_vs_time: PolyLine
            magnitude vs time

        Returns
        -------
        Tm: float
            time dependant function

        """


        loadmag = mag_vs_time.y
        loadtim = mag_vs_time.x
        (ramps_less_than_t, constants_less_than_t, steps_less_than_t,
         ramps_containing_t, constants_containing_t) = (pwise.segment_containing_also_segments_less_than_xi(loadtim, loadmag, t, steps_or_equal_to = True))

        exp = math.exp
        Tm=0
        cv = 1 # I copied the Tm function from SchiffmanAndStein1970
        i=0 #only one time value
        for k in steps_less_than_t[i]:
            sig1 = loadmag[k]
            sig2 = loadmag[k+1]

            Tm += (sig2-sig1)*exp(-cv * alp**2 * (t-loadtim[k]))
        for k in ramps_containing_t[i]:
            sig1 = loadmag[k]
            sig2 = loadmag[k+1]
            t1 = loadtim[k]
            t2 = loadtim[k+1]

#            Tm += (-sig1 + sig2)/(alp**2*cv*(-t1 + t2)) - (-sig1 + sig2)*exp(-alp**2*cv*t)*exp(alp**2*cv*t1)/(alp**2*cv*(-t1 + t2))
            Tm += (-sig1 + sig2)/(alp**2*cv*(-t1 + t2)) - (-sig1 + sig2)*exp(-alp**2*cv*(t-t1))/(alp**2*cv*(-t1 + t2))

        for k in ramps_less_than_t[i]:
            sig1 = loadmag[k]
            sig2 = loadmag[k+1]
            t1 = loadtim[k]
            t2 = loadtim[k+1]
#            Tm += -(-sig1 + sig2)*exp(-alp**2*cv*t)*exp(alp**2*cv*t1)/(alp**2*cv*(-t1 + t2)) + (-sig1 + sig2)*exp(-alp**2*cv*t)*exp(alp**2*cv*t2)/(alp**2*cv*(-t1 + t2))
            Tm += -(-sig1 + sig2)*exp(-alp**2*cv*(t-t1))/(alp**2*cv*(-t1 + t2)) + (-sig1 + sig2)*exp(-alp**2*cv*(t-t2))/(alp**2*cv*(-t1 + t2))
        return Tm



    z= np.asarray(z)
    t = np.asarray(t)

    if tpor is None:
        tpor = t
    else:
        tpor = np.asarray(tpor)



    if drn==0:
        #'PTPB'
        H_ = H/2
#        z = z / 2

        z = z.copy()
        i, = np.where(z>H_)

        z[i]= (H - z[i])

    else:
        #'PTIB'
        H_ = H

    surcharge_vs_time = pwise.PolyLine(surcharge_vs_time[0], surcharge_vs_time[1])

    kap = kh / ks
    s = rs / rw
    n = re / rw

    F = F_func(n, s, kap)

    G = kh/kw * (H_/2/rw)**2

    M = ((2 * np.arange(nterms) + 1) * np.pi/2)

    Dm = 8 / M**2 * (n**2 - 1) / n**2 * G

    bzm = kv / mv / gamw * M**2 / H_**2
    brm = kh / mv / gamw * 2 / re**2 / (F + Dm)
    bm = bzm + brm


    #por

    Tm = np.zeros((len(tpor), nterms), dtype=float)

    for j, tj in enumerate(tpor):
        for k, bk in enumerate(bm):
            Tm[j, k]= _calc_Tm(math.sqrt(bk), tj, surcharge_vs_time)
    Tm = Tm[np.newaxis, :, :]
    M_ = M[np.newaxis, np.newaxis, :]
    bm_ = bm[np.newaxis, np.newaxis, :]
    z_ = z[:, np.newaxis, np.newaxis]


    por = np.sum(2 / M_ * np.sin(M_ * z_ / H_) * Tm, axis=2)

    #avp
    Tm = np.zeros((len(t), nterms), dtype=float)

    for j, tj in enumerate(t):
        for k, bk in enumerate(bm):
            Tm[j, k]= _calc_Tm(math.sqrt(bk), tj, surcharge_vs_time)

#    Tm = Tm[:, :]
    M_ = M[np.newaxis, :]

    avp = np.sum(2 / M_**2 * Tm, axis=1)


    #settle
    load = pwise.pinterp_x_y(surcharge_vs_time, t)
    settle = H * mv * (load - avp)

    return por, avp, settle




if __name__ == '__main__':
    import geotecha.plotting.one_d
    from geotecha.plotting.one_d import plot_vs_depth
    from geotecha.plotting.one_d import plot_vs_time

#    H = 1
#    z  = np.linspace(0, H, 10)
#    t = np.linspace(0,1,100)
#    kv, kh, ks, kw = (10, 10, 3, 1)
#    mv=1
#    gamw = 10
#    rw, rs, re = (0.03, 0.06, 0.5)
#    drn = 1
#    surcharge_vs_time = ((0,0.15, 0.3, 0.45,100.0), (0,50,50.0,100.0,100.0))
#    tpor = t[np.array([20,60,90])]
#    nterms = 20
#
#    por, avp, settle = tangandonitsuka2000(z=z, t=t, kv=kv, kh=kh, ks=ks, kw=kw, mv=mv, gamw=gamw, rw=rw, rs=rs, re=re, H=H,
#                       drn=drn, surcharge_vs_time=surcharge_vs_time,
#                       tpor=tpor, nterms=nterms)
#
#
#
#
#    fig_por=plot_vs_depth(por, z, line_labels=['{0:.3g}'.format(v) for v in tpor],
#                                           prop_dict={'xlabel': 'Pore Pressure'})
#    fig_por.get_axes()[0].grid()
##    fig_por.get_axes()[0].set_xlim(0,1.3)
#
#    fig_avp=plot_vs_time(t,avp, prop_dict={'ylabel': "average pore pressure"})
#
#    fig_set=plot_vs_time(t, settle, prop_dict={'ylabel': "Settlement"})
#    fig_set.gca().invert_yaxis()
#
#    plt.show()


########################################

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

    H = 1
    z  = np.linspace(0, H,10)
    kv, kh, ks, kw = (10, 10, 10, 1)
    mv=1
    gamw = 10
    rw, rs, re = (0.03, 0.03, 0.5)
    drn = 1
    surcharge_vs_time = ((0,0.15, 0.3, 0.45,100.0), (0,50,50.0,100.0,100.0))
    tpor = t[np.array([20,60,90])]
    nterms = 20

    por, avp, settle = tangandonitsuka2000(z=z, t=t, kv=kv, kh=kh, ks=ks, kw=kw, mv=mv, gamw=gamw, rw=rw, rs=rs, re=re, H=H,
                       drn=drn, surcharge_vs_time=surcharge_vs_time,
                       tpor=tpor, nterms=nterms)
##############################################
    print('por', repr(por))
    print('avp', repr(avp))
    print('settle', repr(settle))