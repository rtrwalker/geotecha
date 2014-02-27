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


besselj = scipy.special.jn
bessely = scipy.special.yn


def plot_one_dim_consol(z, t, por=None, doc=None, settle=None, uavg=None):

    if not por is None:
        plt.figure()
        plt.plot(por,z)
        plt.ylabel('Depth, z')
        plt.xlabel('Pore pressure')
        plt.gca().invert_yaxis()

    if not doc is None:
        plt.figure()
        plt.plot(t, doc)
        plt.xlabel('Time, t')
        plt.ylabel('Degree of consolidation')
        plt.gca().invert_yaxis()
        plt.semilogx()

    if not settle is None:
        plt.figure()
        plt.plot(t, settle)
        plt.xlabel('Time, t')
        plt.ylabel('settlement')
        plt.gca().invert_yaxis()
        plt.semilogx()

    if not uavg is None:
        plt.figure()
        plt.plot(t, uavg)
        plt.xlabel('Time, t')
        plt.ylabel('Average pore pressure')
#        plt.gca().invert_yaxis()
        plt.semilogx()

    return



def zhuandyin2012(z, t, alpha, p, q, drn=0, tpor=None, H = 1, kv0 = 1, mv0 = 0.1, gamw = 10,
                ui = 1, nterms = 20):
    """Single layer consolidation with depth dependant properties

    Parameters
    ----------
    z : float or 1d array/list of float
        depth
    t : float or 1d array/list of float
        time

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

    Returns
    -------
    por: 2d array of float
        pore pressure at depth and time.  ppress is an array of size
        (len(z), len(t)).
    avp: 1d array of float
        average pore pressure between depth H and depth Z
    settlement : 1d array of float
        surface settlement at depth z

    Notes
    -----


    References
    ----------
    Code developed based on [1]_

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

        return ((besselj(mu - 1, eta * b)/2 -
                besselj(mu + 1, eta * b)/2)*bessely(nu, eta) -
                (bessely(mu - 1, eta * b)/2 -
            bessely(mu + 1, eta * b)/2)*besselj(nu, eta))


    def drn1root(eta, mu, nu, b, p, n):
        """eigenvalue function"""

        return (1-p)/(1-n)*Zmu(eta, nu, nu, b) * eta * b * Zmu_(eta, nu, nu)



    z = np.atleast_1d(z)
    t = np.atleast_1d(t)

    if tpor is None:
        tpor=t
    else:
        tpor = np.atleast_1d(t)

    #derived parameters
    n = p/q
    b = (1 + alpha)**(1-n/2)

    nu = abs((1 - p) / (2 - n))

    if drn==0:
        etam = find_n_roots(Zmu, args=(nu, nu, b), n = nterms, p=1.01)
    elif drn==1:
        etam = find_n_roots(drn1root, args=(nu, nu, b, p, n), n = nterms, p=1.01)

    eta0=etam[0]

    etam = etam[np.newaxis, np.newaxis, :]

    cv0 = kv0 / (mv0 * gamw)
    dTv = cv0 * (2 - n)**2 * alpha**2 * eta0**2 / (np.pi**2 * H**2)


    y = (1 + alpha * z / H)**(1-n/2)
    y = y[:, np.newaxis, np.newaxis]


    T = dTv * tpor
    T = T[np.newaxis, :, np.newaxis]
    Tm = np.exp(-np.pi**2 * etam**2 / (4 * eta0**2) * T)
    if np.allclose(tpor, t):
        Tm_ = Tm.copy()

    Zm = Zmu(etam, nu, nu, y)

    if drn == 0:
        if -nu == (p - 1) / (2 - n):
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
    if np.allclose(tpor, t):
        Tm=Tm_[0,:,:]
    else:
        T = dTv * t
        T = T[:, np.newaxis]

        Tm = np.exp(-np.pi**2 * etam**2 / (4 * eta0**2) * T)

    if drn == 0:
        if -nu == (p - 1) / (2 - n):
            numer = (2 + np.pi * etam * b**(1-nu) * Zmu(etam, nu-1, nu, b))**2
            denom = etam**2 * ((b * np.pi * etam * Zmu(etam, 1 + nu, nu, b))**2-4)
            Cm = numer/denom
        else:
            numer = (2 - np.pi * etam * b**(1+nu) * Zmu(etam, 1 + nu, nu, b))**2
            denom = etam**2 * ((b * np.pi * etam * Zmu(etam, 1 + nu, nu, b))**2-4)
            Cm = numer/denom
    elif drn == 1:
        numer = 1
        denom = etam**2((b * np.pi * etam * Zmu(etam, nu, nu, b))**2-4)
        Cm = numer / denom

    doc = np.sum(Cm*Tm,axis=1)
    if drn==0:
        numer = 4*(1+q)
    elif drn==1:
        numer = 16*(1+q)

    denom = (2 - n) * ((1+alpha)**(1+ q) - 1)
    fr = numer / denom
    doc*= -fr
    doc += 1

#    doc = doc.ravel()






    return por, doc

if __name__ == '__main__':

    z = np.linspace(0,1,80)
#    t = np.linspace(0.1,3,5)
    t = np.logspace(-2,1,80)
#    tpor = np.array([0.01, 1])
    alpha=-0.95
    p= 1
    q = 1.4
    drn = 0
    por, doc = zhuandyin2012(z, t, alpha = alpha, p=p, q=q, drn=0, nterms=20)

    plot_one_dim_consol(z, t, por, doc)
    plt.show()