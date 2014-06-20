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

"""
Module implementing 'Consolidation by Vertical Drains When the Discharge
Capacity Varies with Depth and Time' as per Deng et al. (2013)[1]_.

References
----------
.. [1] Deng, Yue-Bao, Kang-He Xie, and Meng-Meng Lu. 2013. 'Consolidation by
       Vertical Drains When the Discharge Capacity Varies with Depth and
       Time'. Computers and Geotechnics 48 (March): 1-8.
       doi:10.1016/j.compgeo.2012.09.012.



"""
from __future__ import print_function, division

import numpy as np
from matplotlib import pyplot as plt
#import cmath
import math
#import scipy
#from scipy.integrate import quad
import geotecha.consolidation.smear_zones as smear_zones

def dengetal2013(z,t, rw, re, A1=1, A2=0, A3=0, H=1, rs=None, ks=None,
                 kw0=1e10, kh=1, mv=0.1, gamw=10, ui=1):
    """radial consolidation with depth and time dependent well resistance

    Average excess pore pressure at specified depth and time

    kw = kw0 * (A1 - A2 * z / H) * exp(-A3 * t)

    Parameters
    ----------
    z : float or 1d array/list of float
        depth
    t : float or 1d array/list of float
        time
    rw : float
        drain radius
    re : float
        drain influence radius
    A1 : float, optional
        parameter controlling depth dependance of well resistance. default = 1
    A2 : float, optional
        parameter controlling depth dependance of well resistance. default = 0
    A3 : float, optional
        parameter controlling time dependance of well resistance. default = 0
    H : float, optional
        drainage path length.  default H = 1
    rs : float, optional
        drain influence radius, default=None i.e. no smear zone
    ks : float, optional
        smear zone permeability, default = None, i.e. no smear zone
    kw0 : float, optional
        initial well permeability, default = 1e10 i.e. infinite
    kh : float, optional
        horizontal coefficient of permeability.  default kh = 1
    mv : float, optional
        volume compressibility. default mv = 0.1
    gamw : float, optional
        unit weight of water.  defaule gamw = 10
    ui : float, optional
        initial uniform pore water pressure.  default ui = 1


    Returns
    -------
    por : 2d array of float
        pore pressure at depth and time.  por is an array of size
        (len(z), len(t)).

    References
    ----------
    .. [1] Deng, Yue-Bao, Kang-He Xie, and Meng-Meng Lu. 2013. 'Consolidation
           by
           Vertical Drains When the Discharge Capacity Varies with Depth and
           Time'. Computers and Geotechnics 48 (March): 1-8.
           doi:10.1016/j.compgeo.2012.09.012.

    """


    if A1 < 0:
        raise ValueError("A1 must be greater than 0.  "
                            "You have A1={}".format(A1))
    if A2 > A1:
        raise ValueError("A2 must be less than A1.  "
                            "You have A1={}, A2={}".format(A1, A2))
    if A3 < 0:
        raise ValueError("A3 must be greater than or equal to 0.  "
                            "You have A3={}".format(A3))
    t = np.atleast_1d(t)
    z = np.atleast_1d(z)

    ch = kh / mv / gamw
    n = re / rw

    qw0 = kw0 * np.pi * rw**2

    Th = ch * t / 4 / re**2

    if ks is None or rs is None:
        mu0 = smear_zones.mu_ideal(n)
    else:
        s = rs / rw
        kap = kh / ks
        mu0 = smear_zones.mu_constant(n, s, kap)

    if A3==0 and A2==0:
        # qw is constant
        mus = mu0 + np.pi * kh / (qw0 * A1) * (2 * H * z - z ** 2)
        por = ui * np.exp(-8 * Th[None, :] / mus[:, None])
        return por

    if A3==0 and A2 > 0:
        # qw varies with depth
        mu1 = A2 * z / H + (A1 - A2)*np.log(1 - A2 / A1 * z / H)
        mu1 *= 2 * np.pi * kh * H**2 / (qw0 * A2**2)
        mu1 += mu0
        por = ui * np.exp(-8 * Th[None, :] / mu1[:, None])
        return por

    if A3>0 and A2==0:
        # qw varies with time
        a3 = A3 * 4 * re**2 / ch
        alp0 = qw0 * A1 * mu0 / ( np.pi * kh * (2*H*z - z**2))
        por = (ui * ((1+alp0[:, None] * np.exp(-a3 * Th[None, :])) /
                (1+alp0[:, None])) ** (8/(a3 * mu0)))
        return por

    if A3>0 and A2>0:
        # qw varies with depth and time:
        a3 = A3 * 4 * re**2 / ch
        alp = mu0 * qw0 * A2**2 / (2 * np.pi * H**2 * kh)
        alp /= A2* z / H + (A1 - A2) * np.log(1 - A2 / A1 * z / H)
        por = (ui * ((1+alp[:, None] * np.exp(-a3 * Th[None, :])) /
                (1+alp[:, None])) ** (8/(a3 * mu0)))
        return por




if __name__ == "__main__":

    print(repr(dengetal2013(z=np.array([0.05, 0.1, 0.2, 0.5, 0.8, 1.0])*20,
                             t=[11025.,  110250.],
                             rw=0.035, re=0.525,
                             A1=1, A2=0, A3=0,
                             H=20,
                             rs=0.175,
                             ks=2e-8/1.8,
                             kw0=1e-3,
                             kh=2e-8,
                             mv=0.2e-3,
                             gamw=10,
                             ui=1)))

    # Reproduce figure 3, from Deng et al 2013
    if 1:
        #reproduce deng et al 2013 fig 3a.
        # Note deng has kh/ks=5, but I can only get a match with kh/ks=1.8

        res = {}
        res['Th=0.1 A2=0.1'] = np.array([[0.777489, 0.00142817],
                    [0.811606, 0.0504537],
                    [0.836125, 0.0994905],
                    [0.864221, 0.198757],
                    [0.884466, 0.2992],
                    [0.896856, 0.398485],
                    [0.905755, 0.497774],
                    [0.912911, 0.598233],
                    [0.917452, 0.699863],
                    [0.920243, 0.799159],
                    [0.921291, 0.899625],
                    [0.92146, 0.996587]])
        res['Th=1.0 A2=0.1'] = np.array([[0.0820304, 0.00340884],
                    [0.124869, 0.0500878],
                    [0.165101, 0.102611],
                    [0.23857, 0.200656],
                    [0.295462, 0.299888],
                    [0.341883, 0.399133],
                    [0.377835, 0.499559],
                    [0.405057, 0.597658],
                    [0.424434, 0.700439],
                    [0.437696, 0.799722],
                    [0.445723, 0.899012],
                    [0.449386, 0.998307]])
        res['Th=0.1 A2=1.0'] = np.array([[0.778362, 0.00142715],
                    [0.811602, 0.0481173],
                    [0.837872, 0.100657],
                    [0.871204, 0.199917],
                    [0.894067, 0.300357],
                    [0.909074, 0.399639],
                    [0.919719, 0.498926],
                    [0.930365, 0.59938],
                    [0.936649, 0.69984],
                    [0.943803, 0.799131],
                    [0.948341, 0.899593],
                    [0.952005, 0.998888]])
        res['Th=1 A2=1.0'] = np.array([[0.0829009, 0.0022396],
                    [0.123995, 0.0489206],
                    [0.168587, 0.10027],
                    [0.252532, 0.20064],
                    [0.326005, 0.301021],
                    [0.387258, 0.39908],
                    [0.438917, 0.499487],
                    [0.48534, 0.5999],
                    [0.523035, 0.699155],
                    [0.55724, 0.798414],
                    [0.586213, 0.900016],
                    [0.610813, 0.995782]])

        z = np.linspace(0, 20, 100)
        Th = np.array([0.1, 1.0])
        rw=0.035
        rs = 0.175

        re=0.525
        gamw = 10
        mv=0.2e-3
        kh=2e-8
        ks = kh/1.8
        kw0 = 1e-3
        ch = kh / mv / gamw
        t = Th * 4 * re**2 / ch


        mu0 = smear_zones.mu_constant(re/rw, rs/rw, kh/ks)
        print('mu0', mu0)
        print('et', 2/mu0/re**2)
        for A2 in [0.1,1.0]:
            por = dengetal2013(z, t, rw=rw, re=re, A1=1, A2=A2, A3=0, H=20,
                          rs=rs, ks=ks, kw0=kw0, kh=kh, mv=mv, gamw=10, ui=1)
            for j, t_ in enumerate(t):
                uz0 = np.exp(-8*Th[j]/mu0)
                plt.plot(por[:,j], z, label="m Th={0:.3g}, A2={1:.3g}".format(Th[j], A2))
                plt.plot(uz0,0, marker='o', ms=14)

        for key, val in res.iteritems():
            x = val[:,0]
            y = val[:,1]*20
            plt.plot(x,y, label=key, marker='s', linestyle='None')

        leg = plt.legend(loc=3 )
        leg.draggable()
        plt.gca().set_xlabel('Pore pressure')
        plt.gca().set_ylabel('Depth')
        plt.gca().invert_yaxis()
        plt.gca().set_xlim(0,1)
        plt.gca().grid()
        plt.title('Deng et al. 2013, figure 3a, depth variation')

        plt.show()

    if 1:
        #reproduce deng et al 2013 fig 3b.
        # Note deng has kh/ks=5, but I can only get a match with kh/ks=1.8

        res = {}
        res['Th=1.0 a3=0.1'] = np.array([[0.0835118, 0.00577201],
                    [0.127409, 0.0519481],
                    [0.170236, 0.0995671],
                    [0.245182, 0.20202],
                    [0.301927, 0.300144],
                    [0.349036, 0.399711],
                    [0.383298, 0.499278],
                    [0.411135, 0.598846],
                    [0.430407, 0.702742],
                    [0.442184, 0.800866],
                    [0.448608, 0.89899],
                    [0.452891, 0.994228]])
        res['Th=1.0 a3=0.5'] = np.array([[0.0856531, 0.00577201],
                    [0.139186, 0.0505051],
                    [0.191649, 0.10101],
                    [0.278373, 0.199134],
                    [0.343683, 0.301587],
                    [0.391863, 0.401154],
                    [0.429336, 0.499278],
                    [0.458244, 0.601732],
                    [0.478587, 0.699856],
                    [0.491435, 0.800866],
                    [0.497859, 0.900433],
                    [0.501071, 0.992785]])
        res['Th=1.0 a3=1.0'] = np.array([[0.155246, 0.049062],
                    [0.219486, 0.0995671],
                    [0.324411, 0.200577],
                    [0.398287, 0.301587],
                    [0.448608, 0.399711],
                    [0.489293, 0.497835],
                    [0.514989, 0.598846],
                    [0.537473, 0.702742],
                    [0.54818, 0.800866],
                    [0.557816, 0.901876],
                    [0.559957, 0.992785]])

        z = np.linspace(0, 20, 100)
        Th = np.array([0.1, 1.0])
        rw=0.035
        rs = 0.175

        re=0.525
        gamw = 10
        mv=0.2e-3
        kh=2e-8
        ks = kh/1.8
        kw0 = 1e-3
        ch = kh / mv / gamw
        t = Th * 4 * re**2 / ch


        mu0 = smear_zones.mu_constant(re/rw, rs/rw, kh/ks)
        a3 = np.array([0.1, 0.5, 1.0])
        A3 = a3 * ch / re**2 /4

        for a3_, A3_ in zip(a3, A3):
            por = dengetal2013(z, t, rw=rw, re=re, A1=1, A2=0, A3=A3_, H=20,
                          rs=rs, ks=ks, kw0=kw0, kh=kh, mv=mv, gamw=10, ui=1)
            for j, t_ in enumerate(t):

                uz0 = np.exp(-8*Th[j]/mu0)
                plt.plot(por[:,j], z, label="m Th={0:.3g}, a3={1:.3g}".format(Th[j], a3_))
                plt.plot(uz0,0, marker='o', ms=14)

        for key, val in res.iteritems():
            x = val[:,0]
            y = val[:,1]*20
            plt.plot(x,y, label=key, marker='s', linestyle='None')

        leg = plt.legend(loc=3 )
        leg.draggable()
        plt.gca().set_xlabel('Pore pressure')
        plt.gca().set_ylabel('Depth')
        plt.gca().invert_yaxis()
        plt.gca().set_xlim(0,1)
        plt.gca().grid()
        plt.title('Deng et al. 2013, figure 3b, time variation')

        plt.show()


    if 1:
        #reproduce deng et al 2013 fig 3c.
        # Note deng has kh/ks=5, but I can only get a match with kh/ks=1.8

        res = {}
        res['A2=1.0 a3=1.0'] = np.array([[0.0847639, 0.00168277],
                    [0.157725, 0.0522524],
                    [0.228541, 0.101377],
                    [0.344421, 0.203869],
                    [0.435622, 0.300534],
                    [0.504292, 0.401451],
                    [0.55794, 0.500886],
                    [0.600858, 0.60029],
                    [0.639485, 0.699682],
                    [0.670601, 0.800491],
                    [0.696352, 0.899846],
                    [0.716738, 0.994868]])

        z = np.linspace(0, 20, 100)
        Th = np.array([0.1, 1.0])
        rw=0.035
        rs = 0.175

        re=0.525
        gamw = 10
        mv=0.2e-3
        kh=2e-8
        ks = kh/1.8
        kw0 = 1e-3
        ch = kh / mv / gamw
        t = Th * 4 * re**2 / ch


        mu0 = smear_zones.mu_constant(re/rw, rs/rw, kh/ks)
        a3 = np.array([1.0])
        A3 = a3 * ch / re**2 /4

        for a3_, A3_ in zip(a3, A3):
            por = dengetal2013(z, t, rw=rw, re=re, A1=1, A2=1.0, A3=A3_, H=20,
                          rs=rs, ks=ks, kw0=kw0, kh=kh, mv=mv, gamw=10, ui=1)
            for j, t_ in enumerate(t):

                uz0 = np.exp(-8*Th[j]/mu0)
                plt.plot(por[:,j], z, label="m Th={0:.3g}, a3={1:.3g}".format(Th[j], a3_))
                plt.plot(uz0,0, marker='o', ms=14)

        for key, val in res.iteritems():
            x = val[:,0]
            y = val[:,1]*20
            plt.plot(x,y, label=key, marker='s', linestyle='None')

        leg = plt.legend(loc=3 )
        leg.draggable()
        plt.gca().set_xlabel('Pore pressure')
        plt.gca().set_ylabel('Depth')
        plt.gca().invert_yaxis()
        plt.gca().set_xlim(0,1)
        plt.gca().grid()
        plt.title('Deng et al. 2013, figure 3c, depth and time variation')

        plt.show()

