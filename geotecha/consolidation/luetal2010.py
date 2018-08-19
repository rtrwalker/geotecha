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

"""Lu et al (2010) "Composite foundation considering radial and vertical flows
within the column and the variation of soil permeability within the
disturbed soil zone".

"""
from __future__ import print_function, division

import numpy as np
from matplotlib import pyplot as plt
import geotecha.consolidation.smear_zones as smear_zones

def luetal2010(z, t, rc, re, H=1, rs=None, ks=None,
               kv=1.0, kvc=1.0, kh=1.0, khc=1.0,
               mvs=0.1, mvc=0.1, gamw=10, utop=1, ubot=1, nterms=100):
    """Composite foundation considering radial and vert flows in the column

    An implementation of [1]_.

    Features:

     - Single layer, soil properties constant over time.
     - Instant load linear with depth.
     - Vertical and radial flow in in soil
     - Vertical and radial flow in drain/column
     - Different stiffness in column and soil
     - Pore pressure in soil, column and averaged at depth
     - Average pore pressure over whole profile vs time
     - Settlement of whole profile vs time


    Parameters
    ----------
    z : float or 1d array/list of float
        Depth.
    t : float or 1d array/list of float
        Time.
    rc : float
        Drain radius
    re : float
        Drain influence radius.
    H : float, optional
        Drainage path length.  Default H=1.
    rs : float, optional
        Drain influence radius.  Default rs=None i.e. no smear zone.
    ks : float, optional
        Smear zone permeability. Default ks=None, i.e. no smear zone.
    kv, kvc : float, optional
        Vertical coefficient of permeability in soil and column.
        Default kv= kvc = 1
    kh, khc : float, optional
        Horizontal coefficient of permeability in soil and column.
        default kh = khc = 1
    mvs, mvc : float, optional
        Volume compressibility in soil and column. default mvs=mvc=0.1.
    gamw : float, optional
        Unit weight of water.  Default gamw=10.
    utop, ubot : float, optional
        Initial pore water pressure at top and bottom of soil.
        Default utop=ubot=1.
    nterms : int, optional
        Number of terms to use in solution. Default=100.

    Returns
    -------
    por, pors, porc : 2d array of float
        Pore pressure at depth and time overall, in soil, in column.
        por is an array of size (len(z), len(t)).
    avp : 2d array of float
        Average overall pore pressure of whole soil profile.
    settle : 2d array of float
        Settlement of whole layer.

    References
    ----------
    .. [1] Lu, Meng-Meng, Kang-He Xie, and Biao Guo. 2010. 'Consolidation
           Theory for a Composite Foundation Considering Radial and Vertical
           Flows within the Column and the Variation of Soil Permeability
           within the Disturbed Soil Zone'. Canadian Geotechnical
           Journal 47 (2): 207-17. doi:10.1139/T09-086.

    """


    t = np.atleast_1d(t)
    z = np.atleast_1d(z)

#    ch = kh / mv / gamw
    n = re / rc
    Y = mvs / mvc


#    Th = ch * t / 4 / re**2

    if ks is None or rs is None:
        mu0 = smear_zones.mu_ideal(n)
    else:
        s = rs / rc
        kap = kh / ks
        mu0 = smear_zones.mu_constant(n, s, kap)

    M = ((2 * np.arange(nterms) + 1) / 2 * np.pi)
    betm_numer = n**2 * mu0 / 2 * kvc / kh
    betm_numer += (n**2 - 1) / 8 *kvc / khc
    betm_numer *= (rc / H) ** 2 * M**2
    betm_numer += n**2 - 1 + kvc/kv
    betm_numer *= kv / mvs * (n**2 - 1 + Y)

    betm_denom = n**2 * mu0 / 2 * kv / kh
    betm_denom += (n**2 - 1) / 8 * kv / khc
    betm_denom *= (n**2 - 1) * kvc / kv + 1
    betm_denom += n**4 / M**2 * (H / rc)**2
    betm_denom *= gamw * rc**2

    betm = betm_numer / betm_denom

    C = re**2 * mu0 / 2 / kh
    C += rc**2 * (n**2 - 1)/ 8 / khc
    C *= gamw * ((n**2 - 1)*kvc + kv) * mvs
    C /= (n**2 - 1 + Y) * (kvc - kv)

    D = re**2 * mu0 / 2 / kh
    D += rc**2 * (n**2 - 1)/ 8 / khc
    D *= kv * kvc / (kvc - kv)

    minus1 = np.ones_like(M)
    minus1[1::2] = -1

    por = (2 / M[None, None, :] *
            (utop - minus1[None, None, :] * (utop - ubot)/ M[None, None, :]) *
            np.sin(M[None, None, :] * z[:, None, None]/H) *
            np.exp(-betm[None, None, :] * t[None, :, None]))

    pors = por * (1 + (C * H**2 * betm[None, None, :] - D * M[None, None, :]**2) /
            (H**2 * (n**2 - 1)))

    porc = por * (1 - C * betm[None, None, :]
                    + D * M[None, None, :]**2 / H**2)

    por = np.sum(por, axis=2)
    pors = np.sum(pors, axis=2)
    porc = np.sum(porc, axis=2)

    avp = (2 / M[None, :]**2 *
            (utop - minus1[None, :] * (utop-ubot)/ M[None, :]) *
            np.exp(-betm[None, :] * t[:, None]))
    avp = np.sum(avp, axis=1)

    settle = (utop + ubot)/2 - avp
    settle *= H * mvs * n**2 / (n**2 - 1 + Y)

    return por, pors, porc, avp, settle


if __name__ == '__main__':


    rc=0.1
    re=0.9
    H=10
    rs=0.15
    ks=0.5
    kv=1.0
    kvc=2000
    kh=1.0
    khc=2
    mvs=0.1
    mvc=0.0005
    gamw=10
    utop=0.9
    ubot=0.2
    nterms=100
    z = np.array([0, 3.0, 6.0, 9.0, 10.0])
    t = np.array([0.1, 0.8, 2.0])

#    z= np.linspace(0,H,50)
#    t = np.logspace(-2,0.5, 30)

#    rc=0.1
#    re=1
#    H=10
#    rs=0.15
#    ks=0.5
#    kv=1.0
#    kvc=2000
#    kh=1.0
#    khc=2000
#    mvs=0.1
#    mvc=mvs/200
#    gamw=10
#    utop=1
#    ubot=1
#    nterms=100
#    z= np.linspace(0,H,50)
#    t = np.array([0.1, 0.8, 2.0])
#    t = np.logspace(-2,0.5, 30)


    por, pors, porc, avp, settle = luetal2010(
                z,t, rc, re, H, rs, ks,
                 kv, kvc, kh, khc,
                 mvs, mvc, gamw, utop, ubot, nterms)

    if 1:
        mu = smear_zones.mu_constant(re/rc, rs/rc, kh/ks)
        print('mu', mu)
        print('\npor', repr(por))
        print('\npors', repr(pors))
        print('\nporc', repr(porc))
        print('\navp', repr(avp))
        print('\nset', repr(settle))

    labels = ['Overall pore pressure', 'column pore pressure', 'Soil pore pressure']
    for p, lab in zip([por, porc, pors], labels):
        fig=plt.figure()
        ax = fig.add_subplot(1,1,1)
        for j, t_ in enumerate(t):
            ax.plot(p[:,j], z, marker='o', label="t={0:.3g}".format(t[j]))

        ax.set_xlabel(lab)
        ax.set_ylabel('Depth')
        ax.invert_yaxis()
        ax.set_xlim(0,1)
        ax.grid()
        leg = plt.legend(loc=3 )
        leg.draggable()

    fig=plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(t, avp, marker='o',)
    ax.set_xlabel('time')
    ax.set_ylabel('Overall average pore pressure')
    ax.invert_yaxis()
#    ax.set_xlim(0,1)
    ax.grid()

    fig=plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(t, settle, marker='o',)
    ax.set_xlabel('time')
    ax.set_ylabel('settlement')
    ax.invert_yaxis()
#    ax.set_xlim(0,1)
    ax.grid()

    plt.show()
