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


if __name__ == '__main__':
    z=np.linspace(0,1,11)
#    print(repr(z))
    t=np.array([0, 0.002, 0.008, 0.018, 0.031, 0.049, 0.071, 0.096, 0.126, 0.159, 0.197, 0.239, 0.286, 0.34, 0.403, 0.477, 0.567, 0.684, 0.848, 1.129, 1.781])
    t=np.array([0.008, 0.018, 0.031, 0.049, 0.071, 0.096, 0.126, 0.159, 0.197, 0.239, 0.286, 0.34, 0.403, 0.477, 0.567, 0.684, 0.848, 1.129, 1.781])
    t=np.array([0.008, 0.018, 0.031, 0.049, 0.071, 0.096, 0.126,
                      0.159, 0.197, 0.239, 0.286, 0.34, 0.403, 0.477, 0.567,
                      0.684, 0.848, 1.129, 1.781])
#    t=np.linspace(0,1,100)
#    z = [0, 0.3, 0.5, 0.8, 1]
#    t=[0.008, 0.197, 1.129]
    por, doc = terzaghi_1d(z, t, H = 1, kv = 1, mv = 0.1, gamw = 10,
                ui = 1, nterms = 100)


#    por, doc = one_d_ptpb_instant_triangle(z, t, H = 1, kv = 1, mv = 0.1, gamw = 10,
#                ui = 100, nterms = 100)

    t = np.linspace(0, 3, 100)
    t = np.append(np.array([0], dtype=float), np.logspace(-4,0.31,50))
    por = terzaghi_1d_flowrate(z, t, H = 1, kv = 10, mv = 1, gamw = 10,
                ui = 1, nterms = 100)
    doc = None
    plt.plot(t, por[0], 'o')
    plt.semilogx()
    print (repr(t))
    print(repr(por[0]))



#    print(1-doc)
#    for tim, p in zip(t, doc):
#        print(tim,p)

#    plot_one_dim_consol(z, t, por, doc)
    plt.show()