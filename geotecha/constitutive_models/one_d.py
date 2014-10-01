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
module for one-dimensional constitutive models
"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib



def CcCr_e_from_stresses(estress, pstress, Cc, Cr, siga, ea):
    """void ratio from from stress for CcCr soil model

    Parameters
    ----------
    estress : float
        current effective stress
    pstress : float
        current preconsolidation stress
    Cc : float
        compressibility index
    Cr : float
        recompression index
    siga, ea : float
        point on compression line fixing it in effective stress-void ratio
        space

    Returns
    -------
    e : float
        void ratio corresponding to current stress state

    Examples
    --------
    On recompression line:
    >>> CcCr_e_from_stresses(40, 50, 3, 0.5, 10, 5)
    2.95154...

    On compression line:
    >>> CcCr_e_from_stresses(60, 50, 3, 0.5, 10, 5)
    2.66554...

    Array inputs:
    >>> CcCr_e_from_stresses(np.array([40, 60]), np.array([50, 55]),
    ... 3, 0.5, 10, 5)
    array([ 2.95154499,  2.66554625])

    """


    max_past = np.maximum(pstress, estress)

    # void ratio at preconsolidation pressure
    e = ea - Cc * np.log10(max_past / siga)
    # void ratio at current effetive stress
    e += Cr * np.log10(max_past / estress)

    return e


def CcCr_estress_from_e(e, pstress, Cc, Cr, siga, ea):
    """void ratio from from stress for CcCr soil model

    Parameters
    ----------
    e : float
        current void ratio
    pstress : float
        current preconsolidation stress
    Cc : float
        compressibility index
    Cr : float
        recompression index
    siga, ea : float
        point on compression line fixing it in effective stress-void ratio
        space

    Returns
    -------
    estress : float
        effective stress corresponding to current void ratio

    Examples
    --------
    On recompression line:
    >>> CcCr_estress_from_e(2.95154499, 50, 3, 0.5, 10, 5)
    40...

    On compression line:
    >>> CcCr_estress_from_e(2.66554625, 50, 3, 0.5, 10, 5)
    59.999...

    Array inputs:
    >>> CcCr_estress_from_e(np.array([ 2.95154499,  2.66554625]), 50,
    ... 3, 0.5, 10, 5)
    array([ 40.0...,  59.99...])

    """

    # void ratio at preconsolidation pressure
    ep = ea - Cc * np.log10(pstress / siga)

    dpCc = pstress * (10.0**((ep - e) / Cc) - 1.0)
    dpCr = pstress * (10.0**((ep - e) / Cr) - 1.0)
    estress = pstress + np.minimum(dpCc, dpCr)

    return estress


def CcCr_av_from_stresses(estress, pstress, Cc, Cr, siga, ea):
    """av from from stress for CcCr soil model

    Parameters
    ----------
    estress : float
        current effective stress
    pstress : float
        current preconsolidation stress
    Cc : float
        compressibility index
    Cr : float
        recompression index
    siga, ea : float
        point on compression line fixing it in effective stress-void ratio
        space

    Returns
    -------
    av : float
        slope of void-ratio vs effective stress plot at current stress state

    Examples
    --------
    On recompression line:
    >>> CcCr_av_from_stresses(40, 50, 3, 0.5, 10, 5)
    0.00542868...

    On compression line:
    >>> CcCr_av_from_stresses(60, 50, 3, 0.5, 10, 5)
    0.02171472...

    Array inputs:
    >>> CcCr_av_from_stresses(np.array([40, 60]), np.array([50, 55]),
    ... 3, 0.5, 10, 5)
    array([ 0.00542868,  0.02171472])

    """

    chooser = np.array((Cc, Cc, Cr), dtype=float)

    Cx = chooser[np.sign(estress-pstress)]

    av = 0.43429448190325182 * Cx / estress

    return av

def av_e_from_stresses(estress, av, siga, ea):
    """void ratio from from stress for av soil model

    Parameters
    ----------
    estress : float
        current effective stress
    av : float
        slope of compression line
    siga, ea : float
        effective stress and void ratio specifying point on compression
        line fixing

    Returns
    -------
    e : float
        void ratio corresponding to current stress state

    Examples
    --------
    >>> av_e_from_stresses(20, 1.5, 19, 4)
    2.5

    Array inputs:
    >>> av_e_from_stresses(np.array([20, 21]), 1.5, 19, 4)
    array([ 2.5,  1. ])
    """

    e = ea - av * (estress - siga)

    return e


def av_estress_from_e(e, av, siga, ea):
    """effective stress from void ratio for av soil model

    Parameters
    ----------
    e : float
        current void ratio
    av : float
        slope of compression line
    siga, ea : float
        effective stress and void ratio specifying point on compression
        line fixing

    Returns
    -------
    estress : float
        effective stress corresponding to current void ratio

    Examples
    --------
    >>> av_estress_from_e(1, 1.5, 19, 4)
    21.0

    Array inputs:
    >>> av_estress_from_e(np.array([1, 2.5]), 1.5, 19, 4)
    array([ 21.,  20.])

    """

    sig = siga + (ea - e) / av

    return  sig

def ck_k_from_e(e, ck, ka, ea):
    """permeability from void ratio for ck peremability model

    Parameters
    ----------
    e : float
        current void ratio
    ck : float
        slope of semi-log void-ratio vs permeability line
    ka, ea : float
        peremability and void ratio of point specifying point on permeability
        line.

    Returns
    -------
    k : float
        permeability corresponding to current void ratio

    Examples
    --------
    >>> ck_k_from_e(1, 3, 10,3)
    2.15443...

    Array inputs
    >>> ck_k_from_e(np.array([1, 1.5]), 3, 10,3)
    array([ 2.15443469,  3.16227766])

    """


    k = ka * 10.0**((e - ea) / ck)
    return k





if __name__ == '__main__':
#    print(CcCr_estress_from_e(2.95154499, 50, 3, 0.5, 10, 5))
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest', '--doctest-options=+ELLIPSIS'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])


#    print(repr(CcCr_e_from_stresses(np.array([40, 60]), np.array([50, 60]), 3, 0.5, 10, 5)))
