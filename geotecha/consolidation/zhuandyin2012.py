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
module for Zhu and Yin 2012

"""
from __future__ import print_function, division

import numpy as np
from matplotlib import pyplot as plt
#import geotecha.inputoutput.inputoutput as inputoutput
import math
import textwrap
import scipy.special
import geotecha.piecewise.piecewise_linear_1d as pwise

from geotecha.math.root_finding import find_n_roots


besselj = scipy.special.jv
bessely = scipy.special.yv


def zhuandyin2012(z, t, alpha, p, q, drn=0, tpor=None, H = 1, kv0 = 1, mv0 = 0.1, gamw = 10,
                ui = 1, nterms = 20, plot_eigs=False):
    """Single layer consolidation with depth dependant properties

    Parameters
    ----------
    z : float or 1d array/list of float
        depth
    t : float or 1d array/list of float
        time for degree of consolidation calcs
    tpor : float or 1d array/list of float
        time for pore pressure vs depth calcs
    alpha, p, q : float
        exponent in depth dependence of permeability and compressibility
        respectively. e.g. mv = mv0 * (1+alpha*z/H)**q. Note  p/q cannot be
        2; alpha cannot be zero.
    drn : [0,1], optional
        drainage. drn=0 is pervious top pervious bottom.  drn=1 is pervious
        bottom, impervious bottom.  default drn=0
    tpor : float or 1d array/list of float, optional
        time values for pore pressure vs depth calcs.  default = None i.e.
        time values will be taken from `t`.
    H : float, optional
        drainage path length.  default H = 1
    kv0 : float, optional
        vertical coefficient of permeability.  default kv = 1
    mv0 : float, optional
        volume compressibility. default mv = 0.1
    gamw : float, optional
        unit weight of water.  defaule gamw = 10
    ui : float, optional
        initial uniform pore water pressure.  default ui = 1
    nterms : int, optional
        maximum number of series terms. default nterms= 20
    plot_eigs : True/False, optional
        if True then a plot of the characteristic curve and associated
        eigenvalues will be created.  use plt.show after runnning the program
        to display the curve.  Use this to assess if the eigenvalues are
        correct.  Default = False
    Returns
    -------
    por: 2d array of float
        pore pressure at depth and time.  ppress is an array of size
        (len(z), len(t)).
    doc: 1d array of float
        degree of consolidation based on surface settlement.
    settlement : 1d array of float
        surface settlement

    Notes
    -----

    kv = kv0 * (1+alp*z/H)**p
    mv = mv0 * (1+alp*z/H)**q

    References
    ----------
    Code based on theroy in [1]_

    ..[1] Zhu, G., and J. Yin. 2012. 'Analysis and Mathematical Solutions
          for Consolidation of a Soil Layer with Depth-Dependent Parameters
          under Confined Compression'. International Journal of Geomechanics
          12 (4): 451-61.



    """

    def Zmu(eta, mu, nu, b):
        """Depth fn"""

        return (bessely(nu, eta) * besselj(mu, eta * b) -
                besselj(nu, eta) * bessely(mu, eta * b))

    def Zmu_(eta, mu, nu, b):
        """derivative of Zmu"""

        return ((besselj(mu - 1.0, eta * b)/2.0 -
                besselj(mu + 1.0, eta * b)/2.0)*bessely(nu, eta) -
                (bessely(mu - 1.0, eta * b)/2.0 -
            bessely(mu + 1.0, eta * b)/2.0)*besselj(nu, eta))


    def drn1root(eta, mu, nu, b, p, n):
        """eigenvalue function"""
        return (1-p)/(2-n)*Zmu(eta, nu, nu, b) + eta * b * Zmu_(eta, nu, nu, b)




    z = np.atleast_1d(z)
    t = np.atleast_1d(t)

    if tpor is None:
        tpor = t
    else:
        tpor = np.atleast_1d(tpor)

    #derived parameters
    n = p-q
    b = (1 + alpha)**(1 - n/2)

    nu = abs((1 - p) / (2 - n))

#    print('drn', drn)
#    print('p',p)
#    print('q', q)
#    print('n', n)
#    print('b', b)
#    print('nu', nu)
#    print('(1 - p) / (2 - n)',(1 - p) / (2 - n))
#    plot_eigs=True
    if drn==0:
        etam = find_n_roots(Zmu, args=(nu, nu, b), n = nterms, p=1.01)
        if plot_eigs:
            fig=plt.figure(figsize=(20,5))
            ax = fig.add_subplot('111')
            x = np.linspace(0.01,etam[-1],1000)
            y = Zmu(x, nu, nu, b)
            ax.plot(x, y, marker='.', markersize=3, ls='-')
            ax.plot(etam,np.zeros_like(etam), 'o', markersize=6, )
            ax.set_ylim(-0.3,0.3)
            ax.set_xlabel('$\eta$')
            ax.set_ylabel('characteristic curve')
            ax.grid()
            fig.tight_layout()

    elif drn==1:
        etam = find_n_roots(drn1root, args=(nu, nu, b, p, n), n = nterms, p=1.01)
        if plot_eigs:
            fig=plt.figure(figsize=(20,5))
            ax = fig.add_subplot('111')
            x = np.linspace(0.01,etam[-1],1000)
            y = drn1root(x, nu, nu, b, p, n)
            ax.plot(x, y, marker='.', markersize=3, ls='-')
            ax.plot(etam,np.zeros_like(etam), 'o', markersize=6, )
            ax.set_ylim(-0.3,0.3)
            ax.set_xlabel('$\eta$')
            ax.set_ylabel('characteristic curve')
            ax.grid()
            fig.tight_layout()

    eta0=etam[0]

    etam = etam[np.newaxis, np.newaxis, :]


    cv0 = kv0 / (mv0 * gamw)
    dTv = cv0 * (2 - n)**2 * alpha**2 * eta0**2 / (np.pi**2 * H**2)


    y = (1 + alpha * z / H)**(1-n/2)
    y = y[:, np.newaxis, np.newaxis]


    T = dTv * tpor
    T = T[np.newaxis, :, np.newaxis]
    Tm = np.exp(-np.pi**2 * etam**2 / (4 * eta0**2) * T)


#    if np.allclose(tpor, t):
#        Tm_ = Tm.copy()

    Zm = Zmu(etam, nu, nu, y)

    if drn == 0:
#        if -nu == (p - 1) / (2 - n):
        if np.allclose(-nu, (p - 1) / (2 - n)):

            numer = 2*np.pi * (2 + np.pi * etam * b**(1-nu) * Zmu(etam, nu-1, nu, b))
            denom = 4-(b * np.pi * etam * Zmu(etam, 1 + nu, nu, b))**2
            Cm = numer/denom
        else:

            numer = 2*np.pi * (2 - np.pi * etam * b**(1+nu) * Zmu(etam, 1+nu, nu, b))
            denom = 4-(b * np.pi * etam * Zmu(etam, 1 + nu, nu, b))**2
            Cm = numer/denom
    elif drn == 1:

        numer = 4 * np.pi
        denom = 4 - (b * np.pi * etam * Zmu(etam, nu, nu, b))**2
        Cm = numer / denom

    por = Cm*Zm*Tm
    por *= ui * y**((1 - p)/(2 - n))
    por = np.sum(por, axis=2)


    #degree of consolidation
    etam = etam[0, :, :]
#    if np.allclose(tpor, t):
#        Tm=Tm_[0,:,:]
#    else:
    T = dTv * t
    T = T[:, np.newaxis]

    Tm = np.exp(-np.pi**2 * etam**2 / (4 * eta0**2) * T)

    if drn == 0:
#        if -nu == (p - 1) / (2 - n):
        if np.allclose(-nu, (p - 1) / (2 - n)):

            numer = (2 + np.pi * etam * b**(1-nu) * Zmu(etam, nu-1, nu, b))**2
            denom = etam**2 * ((b * np.pi * etam * Zmu(etam, 1 + nu, nu, b))**2-4)

            Cm = numer/denom
        else:
            print('got here')
            numer = (2 - np.pi * etam * b**(1+nu) * Zmu(etam, 1 + nu, nu, b))**2
            denom = etam**2 * ((b * np.pi * etam * Zmu(etam, 1 + nu, nu, b))**2-4)

            Cm = numer/denom
    elif drn == 1:
        numer = 1.0
        denom = etam**2*((b * np.pi * etam * Zmu(etam, nu, nu, b))**2-4)

        Cm = numer / denom

    doc = np.sum(Cm*Tm,axis=1)

    if drn==0:
        numer = 4*(1+q)
#        numer_settle = 4
    elif drn==1:
        numer = 16*(1+q)
#        numer_settle = 16
    denom = (2 - n) * ((1+alpha)**(1 + q) - 1)
#    denom_settle = (2 - n)

#    settle = mv0*ui*(1 - (H/alpha*numer_settle / denom_settle) * doc)

    fr = numer / denom
    doc*= -fr
    doc += 1

    settle = doc * ui* mv0*((1+alpha)**(1 + q) - 1)/(1+q)*H/alpha

#    print(",".join(['{:.3g}']*4).format((1 - p) / (2 - n), nu, t[0], doc[0]))
    return por, doc, settle

if __name__ == '__main__':
    import geotecha.plotting.one_d
    from geotecha.plotting.one_d import plot_vs_depth
    from geotecha.plotting.one_d import plot_vs_time

    nus=np.linspace(-5,5,30)
#    nus=np.linspace(2.1,4,30)
#    print(nus)
    nus=[3.0]
    for i in nus:




        ui = 100
        drn = 1
        nterms = 50
        mv0 = 1.2
        kv0 = 1.6
        H = 2.5
        alpha = 0.5
        q = 4
#        nu = i
#        p=((2+q)*nu-1)/(nu-1)
        p = -2



        z = np.linspace(0,H,20)
#        t = np.logspace(-2,-0.5,15)
        t = np.linspace(0,15,16)
#        tpor = np.array([0.3, 0.5, 0.7])
        tpor=t[np.array([2,4,9,13])]
#        tpor=t

        plot_eigs=False
        por, doc, settle = zhuandyin2012(
            z=z, t=t, alpha=alpha, p=p, q=q, drn=drn, tpor=tpor, H = H, kv0 = kv0, mv0 = mv0, gamw = 10,
                ui = 100, nterms = nterms, plot_eigs=plot_eigs)
#            z, t, alpha = alpha, p=p, q=q, drn=drn,tpor=tpor, nterms=nterms, plot_eigs=plot_eigs)
        print('z',repr(z))
        print('t', repr(t))
        print('por',repr(por))
        print('doc',repr(doc))
        print('set',repr(settle))
        x = np.linspace(0,1,11)
        kv = (1+alpha*x)**p
        mv = (1+alpha*x)**q
#        print('x', repr(x))
#        print('kv', repr(kv))
#        print('mv', repr(mv))
        fig=plt.figure()
        ax= fig.add_subplot('111')
        ax.plot(kv, x, label='kv', ls='-')
        ax.plot(mv, x, label='mv', ls='-')
        ax.legend()
        ax.invert_yaxis()

        fig_por=plot_vs_depth(por, z, line_labels=['{0:.3g}'.format(v) for v in tpor],
                                           prop_dict={'xlabel': 'Pore Pressure'})
        fig_por.get_axes()[0].grid()
#        fig_por.get_axes()[0].set_xlim(0,1.3)
        fig_doc=plot_vs_time(t,doc, prop_dict={'ylabel': "Degree of consolidation"})
        fig_doc.gca().invert_yaxis()
        fig_set=plot_vs_time(t,settle, prop_dict={'ylabel': "Settlement"})
        fig_set.gca().invert_yaxis()
    plt.show()