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
Terzaghi one-dimensional consolidation

"""
from __future__ import print_function, division

import numpy as np
from matplotlib import pyplot as plt
#import cmath
import math
#import scipy
#from scipy.integrate import quad


def plot_one_dim_consol(z, t, por=None, doc=None):
    """rough plotting"""

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

    Features:

     - Single layer.
     - Vertical flow.
     - Instant load uniform with depth.
     - Soil properties constant with time.
     - Pore pressure vs depth.
     - Degree of consolidation vs time.


    Parameters
    ----------
    z : float or 1d array/list of float
        Depth to calc pore pressure at.
    t : float or 1d array/list of float
        Time to calc pore degree of consolidation at.
    H : float, optional
        Drainage path length.  Default H=1
    kv : float, optional
        Vertical coefficient of permeability.  Default kv=1.
    mv : float, optional
        Volume compressibility.  Default mv=0.1.
    gamw : float, optional
        Unit weight of water.  Default gamw=10.
    ui : float, optional
        Initial uniform pore water pressure.  Default ui = 1.
    nterms : int, optional
        Maximum number of series terms. Default nterms=100


    Returns
    -------
    por : 2d array of float
        Pore pressure at depth and time.  ppress is an array of size
        (len(z), len(t)).
    doc : 1d array of float
        degree of consolidation at time `t'


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


def terzaghi_1d_flowrate(z, t, H = 1, kv = 1, mv = 0.1, gamw = 10,
                ui = 1, nterms = 100):
    """Terzaghi 1d consolidation, flowrate at depth


    Features:

     - Single layer.
     - Vertical flow.
     - Instant load uniform with depth.
     - Soil properties constant with time.
     - flowrate at depth vs time.


    Parameters
    ----------
    z : float or 1d array/list of float
        Depth to calc pore pressure at.
    t : float or 1d array/list of float
        Time to calc pore degree of consolidation at.
    H : float, optional
        Drainage path length.  Default H=1
    kv : float, optional
        Vertical coefficient of permeability.  Default kv=1.
    mv : float, optional
        Volume compressibility.  Default mv=0.1.
    gamw : float, optional
        Unit weight of water.  Default gamw=10.
    ui : float, optional
        Initial uniform pore water pressure.  Default ui = 1.
    nterms : int, optional
        Maximum number of series terms. Default nterms=100


    Returns
    -------
    flowrate : 2d array of float
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
    pass
#    flow_t = np.array([  0, 0.00000000e+00,   1.00000000e-05,   1.32571137e-05,
#         1.75751062e-05,   2.32995181e-05,   3.08884360e-05,
#         4.09491506e-05,   5.42867544e-05,   7.19685673e-05,
#         9.54095476e-05,   1.26485522e-04,   1.67683294e-04,
#         2.22299648e-04,   2.94705170e-04,   3.90693994e-04,
#         5.17947468e-04,   6.86648845e-04,   9.10298178e-04,
#         1.20679264e-03,   1.59985872e-03,   2.12095089e-03,
#         2.81176870e-03,   3.72759372e-03,   4.94171336e-03,
#         6.55128557e-03,   8.68511374e-03,   1.15139540e-02,
#         1.52641797e-02,   2.02358965e-02,   2.68269580e-02,
#         3.55648031e-02,   4.71486636e-02,   6.25055193e-02,
#         8.28642773e-02,   1.09854114e-01,   1.45634848e-01,
#         1.93069773e-01,   2.55954792e-01,   3.39322177e-01,
#         4.49843267e-01,   5.96362332e-01,   7.90604321e-01,
#         1.04811313e+00,   1.38949549e+00,   1.84206997e+00,
#         2.44205309e+00,   3.23745754e+00,   4.29193426e+00,
#         5.68986603e+00,   7.54312006e+00,   1.00000000e+01])
#    flow_v = -np.array([0,  2.00000000e+05,   3.56824823e+04,   3.09906417e+04,
#         2.69157248e+04,   2.33766131e+04,   2.03028544e+04,
#         1.76332600e+04,   1.53146868e+04,   1.33009797e+04,
#         1.15520522e+04,   1.00330887e+04,   8.71385164e+03,
#         7.56807926e+03,   6.57296291e+03,   5.70869305e+03,
#         4.95806484e+03,   4.30613571e+03,   3.73992784e+03,
#         3.24816986e+03,   2.82107247e+03,   2.45013353e+03,
#         2.12796884e+03,   1.84816514e+03,   1.60515244e+03,
#         1.39409315e+03,   1.21078576e+03,   1.05158120e+03,
#         9.13310235e+02,   7.93220327e+02,   6.88920876e+02,
#         5.98335616e+02,   5.19661287e+02,   4.51331637e+02,
#         3.91982248e+02,   3.40369144e+02,   2.95064036e+02,
#         2.53909631e+02,   2.14068409e+02,   1.73374382e+02,
#         1.31849349e+02,   9.18362586e+01,   5.68676559e+01,
#         3.01248089e+01,   1.29749663e+01,   4.24753612e+00,
#         9.66513565e-01,   1.35790536e-01,   1.00673399e-02,
#         3.19831215e-04,   3.30379683e-06,   7.69614367e-09])
#
#    tslice = slice(1, None)
#    print(2*terzaghi_1d_flowrate(z=np.array([0.0]), t=flow_t[tslice], kv=10, mv=1, gamw=10, ui=100, nterms=500))