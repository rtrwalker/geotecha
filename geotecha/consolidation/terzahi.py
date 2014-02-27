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
module for terzaghi consolidation

"""
from __future__ import print_function, division

import numpy as np
from matplotlib import pyplot as plt
import cmath
import math
import scipy
from scipy.integrate import quad


def plot_one_dim_consol(z, t, por=None, doc=None):

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




    return




def terzaghi_1d(z, t, H = 1, kv = 1, mv = 0.1, gamw = 10,
                ui = 1, nterms = 100):
    """Terzaghi 1d consolidation

    Parameters
    ----------
    z : float or 1d array/list of float
        depth
    t : float or 1d array/list of float
        time
    H : float, optional
        drainage path length.  default H = 1
    kv : float, optional
        vertical coefficient of permeability.  default kv = 1
    mv : float, optional
        volume compressibility. default mv = 0.1
    gamw : float, optional
        unit weight of water.  defaule gamw = 10
    ui : float, optional
        initial uniform pore water pressure.  default ui = 1
    nterms : int, optional
        maximum number of series terms. default nterms= 100

    Returns
    -------
    por: 2d array of float
        pore pressure at depth and time.  ppress is an array of size
        (len(z), len(t)).
    avp: 1d array of float
        average pore pressure between depth H and depth Z
    settlement : 1d array of float
        surface settlement at depth z

    """

    z = np.atleast_1d(z)
    t = np.atleast_1d(t)

    dTv = kv / H**2 / mv / gamw
    Tv = (dTv * t)[np.newaxis, :, np.newaxis]

    M = ((2 * np.arange(nterms) + 1) / 2 * np.pi)[np.newaxis, np.newaxis, :]
    Z = (z / H)[:, np.newaxis, np.newaxis]
    por =  2 / M * np.sin(M * Z) * np.exp(-M**2 * Tv)
    por = ui * np.sum(por, axis=2)

    #the following is inefficient
    doc = 2/M**2 * np.exp(-M**2 * Tv)
    doc = np.sum(doc, axis=2)
    doc = 1 - doc[0]



    return por, doc

def one_d_ptpb_instant_triangle(z, t, H = 1, kv = 1, mv = 0.1, gamw = 10,
                ui = 1, nterms = 100):
    """
    Layer is 2H deep
    """
    z = np.atleast_1d(z)
    t = np.atleast_1d(t)

    dTv = kv / H**2 / mv / gamw
    Tv = (dTv * t)[np.newaxis, :, np.newaxis]

    M = ((2 * np.arange(nterms) + 1) / 2 * np.pi)[np.newaxis, np.newaxis, :]

    n = np.arange(1,nterms)[np.newaxis, np.newaxis, :]

    Z = (z / H)[:, np.newaxis, np.newaxis]
    por =  2 / (n*np.pi) * np.sin(n*np.pi/2 * Z) * np.exp(-n**2*np.pi**2/4 * Tv) * np.power(-1, n+1)
    por = ui * np.sum(por, axis=2)

    #the following is inefficient
    doc = 2/M**2 * np.exp(-M**2 * Tv)
    doc = np.sum(doc, axis=2)
    doc = 1 - doc[0]
    return por,  doc


def terzaghi_1d_flowrate(z, t, H = 1, kv = 1, mv = 0.1, gamw = 10,
                ui = 1, nterms = 100):
    """Terzaghi 1d consolidation, flowrate at depth

    Parameters
    ----------
    z : float or 1d array/list of float
        depth
    t : float or 1d array/list of float
        time
    H : float, optional
        drainage path length.  default H = 1
    kv : float, optional
        vertical coefficient of permeability.  default kv = 1
    mv : float, optional
        volume compressibility. default mv = 0.1
    gamw : float, optional
        unit weight of water.  defaule gamw = 10
    ui : float, optional
        initial uniform pore water pressure.  default ui = 1
    nterms : int, optional
        maximum number of series terms. default nterms= 100

    Returns
    -------
    flowrate: 2d array of float
        flowrate depth and time.  flowrate is an array of size
        (len(z), len(t)).


    """

    z = np.atleast_1d(z)
    t = np.atleast_1d(t)

    dTv = kv / H**2 / mv / gamw
    Tv = (dTv * t)[np.newaxis, :, np.newaxis]

    M = ((2 * np.arange(nterms) + 1) / 2 * np.pi)[np.newaxis, np.newaxis, :]

    Z = (z / H)[:, np.newaxis, np.newaxis]
#    por =  2 / M * np.sin(M * Z) * np.exp(-M**2 * Tv)
    flowrate =  np.cos(M * Z) * np.exp(-M**2 * Tv)
    flowrate = kv / gamw * ui / H * 2 * np.sum(flowrate, axis=2)




    return flowrate


def cosenzaandkorosak2014(z, t, theta, v, tpor=None, L = 1, kv = 1, mv = 0.1, gamw = 10,
                ui = 1, nterms = 100):
    """Terzaghi 1d consolidation

    Parameters
    ----------
    z : float or 1d array/list of float
        depth
    t : float or 1d array/list of float
        time
    theta : float
        parameter in cosenzaandkorosak2014
    v : float
        parameter in cosenzaandkorosak2014

    tpor : float or 1d array/list of float
        time values for pore pressure vs depth calcs
    L : float, optional
        drainage path length.  default H = 1
    kv : float, optional
        vertical coefficient of permeability.  default kv = 1
    mv : float, optional
        volume compressibility. default mv = 0.1
    gamw : float, optional
        unit weight of water.  defaule gamw = 10
    ui : float, optional
        initial uniform pore water pressure.  default ui = 1
    nterms : int, optional
        maximum number of series terms. default nterms= 100

    Returns
    -------
    por: 2d array of float
        pore pressure at depth and time.  ppress is an array of size
        (len(z), len(t)).
    avp: 1d array of float
        average pore pressure between depth H and depth Z
    settlement : 1d array of float
        surface settlement at depth z

    """


    def Bn(r, lam, the, v, t):
        numer = lam * the*r**(v-1)*cmath.sin(np.pi*v)
        denom = (lam - r + the*r**v*cmath.exp(1j*np.pi*v))**2

        return math.exp(-r * t)*numer/denom

    z = np.atleast_1d(z)
    t = np.atleast_1d(t)
    if tpor is None:
        tpor=t
    else:
        tpor = np.atleast_1d(t)


    M = ((2 * np.arange(nterms) + 1) * np.pi)

    an = 2 / M

    dTv = kv / L**2 / mv / gamw
    lamda_n = M**2*dTv

    Z = (z / L)

    por = np.zeros((len(z), len(tpor)))

    doc = np.zeros(len(t))



    for j, t_ in enumerate(tpor):
        for n, an_ in enumerate(an):
            Bn_ = complex_quadrature(Bn, 0, np.inf, args=(M[n], theta, v, t_))
            Bn_ = np.real(Bn_[0])/np.pi
            for i, Z_ in enumerate(Z):

                por[i,j] += an_ * np.sin(M[n] * Z_) * Bn_

    por *= ui
    for j, t_ in enumerate(t):
        for n, an_ in enumerate(an):
            Bn_ = complex_quadrature(Bn, 0, np.inf, args=(M[n], theta, v, t_))
            Bn_ = np.real(Bn_[0])/np.pi

            doc[j] +=  2*an_**2*Bn_

    doc = 1 - doc





    return por, doc


def Bnf(r, lam, the, v, t):
    return np.exp(-r*t)*Kn(r, lam, the, v)
def Bnfreal(r, lam, the, v, t):
    return np.real(np.exp(-r*t)*Kn(r, lam, the, v))
def Bnfimag(r, lam, the, v, t):
    return np.imag(np.exp(-r*t)*Kn(r, lam, the, v))

def Kn(r, lam, the, v):
    numer = lam * the*r**(v-1)*cmath.sin(np.pi*v)
    denom = (lam - r + the*r**v*cmath.exp(1j*np.pi*v))**2
    return numer / denom




def complex_quadrature(func, a, b, **kwargs):
    #http://stackoverflow.com/a/5966088/2530083
    def real_func(x, *args):
        return scipy.real(func(x, *args))
    def imag_func(x, *args):
        return scipy.imag(func(x, *args))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

#    Tv = (dTv * t)[np.newaxis, :, np.newaxis]
#
#    M = ((2 * np.arange(nterms) + 1) / 2 * np.pi)[np.newaxis, np.newaxis, :]
#    Z = (z / H)[:, np.newaxis, np.newaxis]
#    por =  2 / M * np.sin(M * Z) * np.exp(-M**2 * Tv)
#    por = ui * np.sum(por, axis=2)
#
#    #the following is inefficient
#    doc = 2/M**2 * np.exp(-M**2 * Tv)
#    doc = np.sum(doc, axis=2)
#    doc = 1 - doc[0]



    return por, doc


if __name__ == '__main__':

    z = np.linspace(0,1,20)
    t = np.linspace(0,1,5)
    tpor = np.array([0, 0.3, 0.5, 1])
    theta = 0.01
    v = 0.9
#    por, doc = cosenzaandkorosak2014(z, t, theta, v)
#
#    plot_one_dim_consol(z, t, por, doc)
#    plt.show()


    t=0.00
    n=0
    lam = (2*n+1)*np.pi

    a=3.1
    b=3.2
    x = np.linspace(a,b,200)
    y = np.zeros_like(x, dtype=complex)

    for j, x_ in enumerate(x):
        y[j] = Bnf(x[j], lam, theta, v, t)


    integ_real = np.trapz(np.real(y), x)
    integ_imag = np.trapz(np.imag(y), x)

    integ_r = quad(Bnfreal,a, b, args=(lam, theta, v, t))[0]
    integ_i = quad(Bnfimag,a, b, args=(lam, theta, v, t))[0]

    print(integ_real, integ_imag)
    print(integ_r, integ_i)

    print(y[0], y[-1])
    plt.plot(x, np.real(y))
    plt.plot(x, np.imag(y))
    plt.show()
#        print(x, Bnf(x, lam, theta, v, t))








#    z=np.linspace(0,1,11)
##    print(repr(z))
#    t=np.array([0, 0.002, 0.008, 0.018, 0.031, 0.049, 0.071, 0.096, 0.126, 0.159, 0.197, 0.239, 0.286, 0.34, 0.403, 0.477, 0.567, 0.684, 0.848, 1.129, 1.781])
#    t=np.array([0.008, 0.018, 0.031, 0.049, 0.071, 0.096, 0.126, 0.159, 0.197, 0.239, 0.286, 0.34, 0.403, 0.477, 0.567, 0.684, 0.848, 1.129, 1.781])
#    t=np.array([0.008, 0.018, 0.031, 0.049, 0.071, 0.096, 0.126,
#                      0.159, 0.197, 0.239, 0.286, 0.34, 0.403, 0.477, 0.567,
#                      0.684, 0.848, 1.129, 1.781])
##    t=np.linspace(0,1,100)
##    z = [0, 0.3, 0.5, 0.8, 1]
##    t=[0.008, 0.197, 1.129]
#    por, doc = terzaghi_1d(z, t, H = 1, kv = 1, mv = 0.1, gamw = 10,
#                ui = 1, nterms = 20)
#
#
##    por, doc = one_d_ptpb_instant_triangle(z, t, H = 1, kv = 1, mv = 0.1, gamw = 10,
##                ui = 100, nterms = 100)
#
#    t = np.linspace(0, 3, 100)
#    t = np.append(np.array([0], dtype=float), np.logspace(-5,0.32,20))
#    por = terzaghi_1d_flowrate(z, t, H = 1, kv = 10, mv = 1, gamw = 10,
#                ui = 100, nterms = 500)
#    doc = None
##    plt.plot(t, por[0], 'o')
##    plt.semilogx()
#    print (repr(t))
#    print(repr(2*por[0]))
#
#    flow_t = np.array([0,0,  1.00000000e-05,   1.32571137e-05,   1.75751062e-05,
#         2.32995181e-05,   3.08884360e-05,   4.09491506e-05,
#         5.42867544e-05,   7.19685673e-05,   9.54095476e-05,
#         1.26485522e-04,   1.67683294e-04,   2.22299648e-04,
#         2.94705170e-04,   3.90693994e-04,   5.17947468e-04,
#         6.86648845e-04,   9.10298178e-04,   1.20679264e-03,
#         1.59985872e-03,   2.12095089e-03,   2.81176870e-03,
#         3.72759372e-03,   4.94171336e-03,   6.55128557e-03,
#         8.68511374e-03,   1.15139540e-02,   1.52641797e-02,
#         2.02358965e-02,   2.68269580e-02,   3.55648031e-02,
#         4.71486636e-02,   6.25055193e-02,   8.28642773e-02,
#         1.09854114e-01,   1.45634848e-01,   1.93069773e-01,
#         2.55954792e-01,   3.39322177e-01,   4.49843267e-01,
#         5.96362332e-01,   7.90604321e-01,   1.04811313e+00,
#         1.38949549e+00,   1.84206997e+00,   2.44205309e+00,
#         3.23745754e+00,   4.29193426e+00,   5.68986603e+00,
#         7.54312006e+00,   1.00000000e+01])
#    flow_v = np.array([0,10000, -9.90047694e+03,  -9.89919282e+03,  -9.89351152e+03,
#        -9.86948623e+03,  -9.79940224e+03,  -9.64662421e+03,
#        -9.37999482e+03,  -8.98715135e+03,  -8.47820852e+03,
#        -7.88013627e+03,  -7.22737957e+03,  -6.55367320e+03,
#        -5.88712161e+03,  -5.24837850e+03,  -4.65087286e+03,
#        -4.10203104e+03,  -3.60477415e+03,  -3.15890498e+03,
#        -2.76223170e+03,  -2.41140541e+03,  -2.10250749e+03,
#        -1.83144026e+03,  -1.59417346e+03,  -1.38688959e+03,
#        -1.20606109e+03,  -1.04848326e+03,  -9.11279362e+02,
#        -7.91889189e+02,  -6.88048485e+02,  -5.97763928e+02,
#        -5.19286679e+02,  -4.51086180e+02,  -3.91821381e+02,
#        -3.40263369e+02,  -2.94993167e+02,  -2.53859632e+02,
#        -2.14030658e+02,  -1.73344998e+02,  -1.31827196e+02,
#        -9.18208453e+01,  -5.68581121e+01,  -3.01197532e+01,
#        -1.29727888e+01,  -4.24682328e+00,  -9.66351360e-01,
#        -1.35767747e-01,  -1.00656503e-02,  -3.19777540e-04,
#        -3.30324237e-06,  -7.69485207e-09])
#
#    plt.plot(t, 2*por[0], '-o')
#    plt.plot(flow_t, -flow_v, '-+')
#    plt.semilogx()
#    plt.semilogy()
##    print(1-doc)
##    for tim, p in zip(t, doc):
##        print(tim,p)
#
##    plot_one_dim_consol(z, t, por, doc)
#    plt.show()