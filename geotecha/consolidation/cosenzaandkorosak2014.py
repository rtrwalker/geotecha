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
Cosenza and Korosak (2014) "Secondary Consolidation of Clay as
an Anomalous Diffusion Process".

"""
from __future__ import print_function, division

import numpy as np
from matplotlib import pyplot as plt
import math
import textwrap

import geotecha.piecewise.piecewise_linear_1d as pwise


#from geotecha.mathematics.mp_laplace import Talbot
from geotecha.mathematics.laplace import Talbot

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



def cosenzaandkorosak2014(z, t, theta, v, tpor=None, L = 1, kv = 1, mv = 0.1, gamw = 10,
                ui = 1, nterms = 100):
    """Secondary consolidation of Clay as an anomalous diffusion process

    An implementation of [1]_.

    Features:

     - Single layer, soil properties constant over time.
     - Instant load uniform with depth.
     - Vertical flow.
     - Similar to Terzaghi 1d consolidation equation but with additional
       fractional time derivative that retards pore pressure dissipation
       as a possible creep mechanism.
     - Uses Laplace transform in solution.


    Parameters
    ----------
    z : float or 1d array/list of float
        Depth.
    t : float or 1d array/list of float
        Time.
    theta : float or array/list of float
        Parameter in [1]_.
    v : float or array/list of float
        Parameter in [1]_.
    tpor : float or 1d array/list of float
        Time values for pore pressure vs depth calcs.
    L : float, optional
        Drainage path length.  Default H = 1.
    kv : float, optional
        Vertical coefficient of permeability.  Default kv = 1.
    mv : float, optional
        Volume compressibility.  Default mv = 0.1.
    gamw : float, optional
        Unit weight of water.  defaule gamw = 10.
    ui : float, optional
        Initial uniform pore water pressure.  Default ui = 1.
    nterms : int, optional
        Maximum number of series terms.  Default nterms= 100.

    Returns
    -------
    por: 2d array of float
        Pore pressure at depth and time.  ppress is an array of size
        (len(z), len(t)).
    avp: 1d array of float
        Average pore pressure between depth H and depth Z.
    settlement : 1d array of float
        Surface settlement at depth z.

    Notes
    -----
    The article [1]_ only has single values of `theta` and `v`.
    Here I've simply added more.

    References
    ----------
    Code developed based on [1]_

    ..[1] Cosenza, Philippe, and Dean Korosak. 2014. 'Secondary Consolidation
          of Clay as an Anomalous Diffusion Process'. International Journal
          for Numerical and Analytical Methods in Geomechanics:
          doi:10.1002/nag.2256.


    """

#    def F(s, v, theta, lam):
#        return (1+theta*s**(v-1))/(s + theta*s**v+lam)

    def F(s, v, theta, lam):
        numer = np.ones_like(s)
        denom = s+lam

        for v_, the_ in zip(v,theta):
            numer += the_*s**(v_-1)
            denom += the_*s**v_
        return  (numer/denom)

#    def Bn(r, lam, the, v, t):
#        numer = lam * the*r**(v-1)*cmath.sin(np.pi*v)
#        denom = (lam - r + the*r**v*cmath.exp(1j*np.pi*v))**2
#
#        return math.exp(-r * t)*numer/denom

    z = np.atleast_1d(z)
    t = np.atleast_1d(t)
    theta = np.atleast_1d(theta)
    v = np.atleast_1d(v)

    if tpor is None:
        tpor=t
    else:
        tpor = np.atleast_1d(t)


    M = ((2 * np.arange(nterms) + 1) * np.pi)

#    an = 2 / M

    dTv = kv / L**2 / mv / gamw
    lamda_n = M**2*dTv

    Z = (z / L)

#    por = np.zeros((len(z), len(tpor)))

#    doc = np.zeros(len(t))

    a = Talbot(F, n=24, shift=0.0)

    Bn = np.zeros((len(tpor), nterms), dtype=float)
    for j, lam in enumerate(M):
        Bn[:, j] = np.array(a(tpor, args=(v, theta, M[j])), dtype=float)

    if np.allclose(tpor, t): #reuse Bn for Doc
        Bn_ = Bn.copy()

    Bn = Bn[np.newaxis, :, :]
    An = 2/M[np.newaxis, :] * np.sin(M[np.newaxis, :] * Z[:, np.newaxis])
    An = An[:, np.newaxis, :]
    por = np.sum(An*Bn, axis=2)*ui

    #degree of consolidation
    An = 2*2**2/M[np.newaxis, :]**2
    if np.allclose(tpor, t):
        Bn = Bn_
    else:
        Bn = np.zeros((len(t), nterms), dtype=float)
        for j, lam in enumerate(M):
            Bn[:, j] = np.array(a(t, args=(v, theta, M[j])), dtype=float)

    doc = 1 - np.sum(An*Bn, axis = 1)


    return por, doc

if __name__ == '__main__':

    z = np.linspace(0,1,20)
#    t = np.linspace(0.1,3,5)
    t = np.logspace(-3,6,80)
    tpor = np.array([0.01, 1])
    theta = [0.2, 0.2]
    v = [0.1, 0.01]
#    theta = 1.5
#    v = 0.5
    por, doc = cosenzaandkorosak2014(z, t, theta, v, nterms=20)

    plot_one_dim_consol(z, t, por, doc)
    plt.show()