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
"""integrals and eigenvalues for consolidation using spectral methods


dim1sin_af_linear: `sin(mi * z) * a(z) * sin(mj * z)` between [0, 1`sin(mi * z) * a(z) *b(z) * sin(mj * z)` between [0, 1]
dim1sin_D_aDf_linear: `sin(mi * z) * D[a(z) * D[sin(mj * z),z],z]` between [0, 1]
dim1sin_ab_linear: `sin(mi * z) * a(z) *b(z)` between [0, 1]
dim1sin_abc_linear: `sin(mi * z) * a(z) * b(z) * c(z)` between [0, 1]
dim1sin_D_aDb_linear: `sin(mi * z) * D[a(z) * D[b(z), z], z]` between [0, 1]
EDload_linear: D[load(tau), tau] * exp(dT * eig * (t-tau)) between [0, t]
Eload_linear: load(tau) * exp(dT * eig * (t-tau)) between [0, t]
dim1sin: sin(m*z) for each combination of m and z
dim1sin_avg_between: integrate(sin(m*z), (z, z1, z2))/(z2-z1) for each combination of m and [z1,z2]
dim1sin_a_linear_between: integrate(a(z) * sin(m*z), (z, z1, z2)) for each combination of m and [z1,z2]

"""

from __future__ import division, print_function
import numpy as np
import math

from geotecha.piecewise.piecewise_linear_1d import segment_containing_also_segments_less_than_xi
from geotecha.piecewise.piecewise_linear_1d import segments_between_xi_and_xj
import geotecha.piecewise.piecewise_linear_1d as pwise
from geotecha.piecewise.piecewise_linear_1d import PolyLine


def m_from_sin_mx(i, boundary=0):
    """Sine series eigenvalue of boundary value problem on [0, 1]


    `i` th eigenvalue, M,  of f(x) = sin(M*x) that satisfies:
        f(0) = 0, f(1) = 0; for `boundary` = 0 i.e. PTPB
        f(0) = 0; f'(1) = 0; for `boundary` = 1 i.e. PTIB

    Parameters
    ----------
    i : ``int``
        eigenvalue  in series to return
    boundary : {0, 1}, optional
        boundary condition.
        For 'Pervious Top Pervious Bottom (PTPB)', boundary = 0
        For 'Pervious Top Impervious Bottom (PTIB)', boundary = 1

    Returns
    -------
    out : ``float``
        returns the `i` th eigenvalue

    """
    from math import pi

    if boundary not in {0, 1}:
        raise ValueError('boundary = %s; must be 0 or 1.' % (boundary))

    return pi * (i + 1 - boundary / 2.0)

def pdim1sin_af_linear(m, a, **kwargs):
    """wrapper for dim1sin_af_linear with PolyLine input

    See also
    --------
    dim1sin_af_linear

    """

    return dim1sin_af_linear(m, a.y1, a.y2, a.x1, a.x2, **kwargs)

def dim1sin_af_linear_old(m, at, ab, zt, zb):
    """Create matrix of spectral integrations

    Performs integrations of `sin(mi * z) * a(z) * sin(mj * z)` between [0, 1]
    where a(z) is a piecewise linear function of z.  Calulation of integrals
    is performed at each element of a square symmetric matrix (size depends
    on size of `m`)

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    at : ``list`` of ``float``
        property at top of each layer
    ab : ``list`` of ``float``
        property at bottom of each layer
    zt : ``list`` of ``float``
        normalised depth or z-coordinate at top of each layer. `zt[0]` = 0
    zb : ``list`` of ``float``
        normalised depth or z-coordinate at bottom of each layer. `zt[-1]` = 1

    Returns
    -------
    A : numpy.ndarray
        returns a square symmetric matrix, size determined by size of `m`

    See Also
    --------
    m_from_sin_mx : used to generate 'm' input parameter
    geotecha.integrals_generate_code.dim1sin_af_linear : use sympy
    to perform the integrals symbolically and generate expressions for
    this function

    Notes
    -----
    The `dim1sin_af_linear` matrix, :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{{a\\left(z\\right)}{sin\\left({m_j}z\\right)}{sin\\left({m_i}z\\right)}\,dz}

    where :math:`a\\left(z\\right)` in one layer is given by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    """


    from math import sin, cos

    neig = len(m)
    nlayers = len(zt)

    A = np.zeros([neig, neig], float)
    for layer in range(nlayers):
        for i in range(neig):
            A[i, i] += (-(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*cos(m[i]*zb[layer])**2 +
                (4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*ab[layer]*cos(m[i]*zt[layer])**2 +
                (4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*cos(m[i]*zb[layer])**2 -
                (4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*cos(m[i]*zt[layer])**2 -
                2*m[i]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*
                sin(m[i]*zb[layer]) + 2*m[i]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*
                ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) +
                2*m[i]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]*cos(m[i]*zb[layer])*
                sin(m[i]*zb[layer]) - 2*m[i]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*
                zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) + m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]**2*sin(m[i]*zb[layer])**2 +
                m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]**2*
                cos(m[i]*zb[layer])**2 - m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*
                ab[layer]*zt[layer]**2*sin(m[i]*zt[layer])**2 - m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])**2 -
                m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]**2*
                sin(m[i]*zb[layer])**2 - m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*
                at[layer]*zb[layer]**2*cos(m[i]*zb[layer])**2 + m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]**2*sin(m[i]*zt[layer])**2 +
                m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]**2*
                cos(m[i]*zt[layer])**2 - 2*zb[layer]*m[i]*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) +
                2*zb[layer]*m[i]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*
                cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) + 2*zb[layer]*m[i]**2*(4*zb[layer]*
                m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]*sin(m[i]*zb[layer])**2 +
                2*zb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]*
                cos(m[i]*zb[layer])**2 - 2*zb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]*sin(m[i]*zt[layer])**2 -
                2*zb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*
                at[layer]*zt[layer]*cos(m[i]*zt[layer])**2 + 2*zt[layer]*m[i]*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) -
                2*zt[layer]*m[i]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*ab[layer]*
                cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) - 2*zt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]*sin(m[i]*zb[layer])**2 -
                2*zt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*
                ab[layer]*zb[layer]*cos(m[i]*zb[layer])**2 + 2*zt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]*sin(m[i]*zt[layer])**2 +
                2*zt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]*
                cos(m[i]*zt[layer])**2)
        for i in range(neig-1):
            for j in range(i + 1, neig):
                A[i, j] += (m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 +
                    2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                    m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                    m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                    m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                    m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                    m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                    m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                    m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                    2*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) -
                    2*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) -
                    2*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) +
                    2*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) +
                    m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                    m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) -
                    m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                    m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                    m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                    m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                    m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                    m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                    m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                    m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                    m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                    m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                    m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                    m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                    m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                    m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) -
                    zb[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                    zb[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                    zb[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                    zb[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                    zb[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                    zb[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                    zb[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                    zb[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                    zt[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                    zt[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                    zt[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                    zt[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) -
                    zt[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                    zt[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                    zt[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                    zt[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]))


    #A is symmetric
    for i in range(neig - 1):
        for j in range(i + 1, neig):
            A[j, i] = A[i, j]

    return A

def pdim1sin_abf_linear(m, a, b, **kwargs):
    """wrapper for dim1sin_abf_linear with PolyLine input

    See also
    --------
    dim1sin_abf_linear

    """
    a, b = pwise.polyline_make_x_common(a,b)
    return dim1sin_abf_linear(m, a.y1, a.y2, b.y1, b.y2, a.x1, a.x2, **kwargs)

def dim1sin_abf_linear_old(m, at, ab, bt, bb, zt, zb):
    """Create matrix of spectral integrations

    Performs integrations of `sin(mi * z) * a(z) *b(z) * sin(mj * z)`
    between [0, 1] where a(z) and b(z) are piecewise linear functions of z.
    Calulation of integrals is performed at each element of a square symmetric
    matrix (size depends on size of `m`)

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    at : ``list`` of ``float``
        property at top of each layer
    ab : ``list`` of ``float``
        property at bottom of each layer
    bt : ``list`` of ``float``
        2nd property at top of each layer
    bb : ``list`` of ``float``
        2nd property at bottom of each layer
    zt : ``list`` of ``float``
        normalised depth or z-coordinate at top of each layer. `zt[0]` = 0
    zb : ``list`` of ``float``
        normalised depth or z-coordinate at bottom of each layer. `zt[-1]` = 1

    Returns
    -------
    A : numpy.ndarray
        returns a square symmetric matrix, size determined by size of `m`

    See Also
    --------
    m_from_sin_mx : used to generate 'm' input parameter
    geotecha.integrals_generate_code.dim1sin_abf_linear : use sympy
    to perform the integrals symbolically and generate expressions for
    this function

    Notes
    -----
    The `dim1sin_abf_linear` matrix, :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{{a\\left(z\\right)}{b\\left(z\\right)}\\phi_i\\phi_j\\,dz}

    where the basis function :math:`\\phi_i` is given by:


    ..math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` and :math:`b\\left(z\\right)` are piecewise
    linear functions w.r.t. :mathL`z`, that within a layer are defined by:

    ..math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """


    from math import sin, cos

    neig = len(m)
    nlayers = len(zt)

    A = np.zeros([neig, neig], float)
    for layer in range(nlayers):
        for i in range(neig):
            A[i, i] += (3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) -
                3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) -
                3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) +
                3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) -
                3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) +
                3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) +
                3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) -
                3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) +
                3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer])**2 -
                3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])**2 -
                3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer])**2 +
                3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])**2 -
                3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer])**2 +
                3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])**2 +
                3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer])**2 -
                3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])**2 -
                3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer])**2 +
                3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])**2 +
                3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer])**2 -
                3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])**2 +
                3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer])**2 -
                3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])**2 -
                3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer])**2 +
                3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])**2 -
                6*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                + 6*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                + 6*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                - 6*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                + 6*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                - 6*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                - 6*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                + 6*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                + 2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]**3*sin(m[i]*zb[layer])**2 +
                2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]**3*cos(m[i]*zb[layer])**2 -
                2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]**3*sin(m[i]*zt[layer])**2 -
                2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]**3*cos(m[i]*zt[layer])**2 -
                2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]**3*sin(m[i]*zb[layer])**2 -
                2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]**3*cos(m[i]*zb[layer])**2 +
                2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]**3*sin(m[i]*zt[layer])**2 +
                2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]**3*cos(m[i]*zt[layer])**2 -
                2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]**3*sin(m[i]*zb[layer])**2 -
                2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]**3*cos(m[i]*zb[layer])**2 +
                2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]**3*sin(m[i]*zt[layer])**2 +
                2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]**3*cos(m[i]*zt[layer])**2 +
                2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]**3*sin(m[i]*zb[layer])**2 +
                2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]**3*cos(m[i]*zb[layer])**2 -
                2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]**3*sin(m[i]*zt[layer])**2 -
                2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]**3*cos(m[i]*zt[layer])**2 -
                3*zb[layer]*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])**2 +
                3*zb[layer]*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])**2 -
                3*zb[layer]*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])**2 +
                3*zb[layer]*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])**2 +
                6*zb[layer]*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*cos(m[i]*zb[layer])**2 -
                6*zb[layer]*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*cos(m[i]*zt[layer])**2 -
                6*zb[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                + 6*zb[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                - 6*zb[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                + 6*zb[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                + 12*zb[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                - 12*zb[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                + 3*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]**2*sin(m[i]*zb[layer])**2 +
                3*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])**2 -
                3*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]**2*sin(m[i]*zt[layer])**2 -
                3*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])**2 +
                3*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]**2*sin(m[i]*zb[layer])**2 +
                3*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])**2 -
                3*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]**2*sin(m[i]*zt[layer])**2 -
                3*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])**2 -
                6*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]**2*sin(m[i]*zb[layer])**2 -
                6*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])**2 +
                6*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]**2*sin(m[i]*zt[layer])**2 +
                6*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])**2 -
                6*zb[layer]**2*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) +
                6*zb[layer]**2*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) +
                6*zb[layer]**2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer])**2 +
                6*zb[layer]**2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])**2 -
                6*zb[layer]**2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer])**2 -
                6*zb[layer]**2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])**2 +
                6*zt[layer]*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zb[layer])**2 -
                6*zt[layer]*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zt[layer])**2 -
                3*zt[layer]*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])**2 +
                3*zt[layer]*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])**2 -
                3*zt[layer]*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])**2 +
                3*zt[layer]*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])**2 +
                12*zt[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                - 12*zt[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                - 6*zt[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                + 6*zt[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                - 6*zt[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                + 6*zt[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                - 6*zt[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]**2*sin(m[i]*zb[layer])**2 -
                6*zt[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])**2 +
                6*zt[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]**2*sin(m[i]*zt[layer])**2 +
                6*zt[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])**2 +
                3*zt[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]**2*sin(m[i]*zb[layer])**2 +
                3*zt[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])**2 -
                3*zt[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]**2*sin(m[i]*zt[layer])**2 -
                3*zt[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])**2 +
                3*zt[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]**2*sin(m[i]*zb[layer])**2 +
                3*zt[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])**2 -
                3*zt[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]**2*sin(m[i]*zt[layer])**2 -
                3*zt[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])**2 +
                6*zt[layer]*zb[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) -
                6*zt[layer]*zb[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) +
                6*zt[layer]*zb[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) -
                6*zt[layer]*zb[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) -
                6*zt[layer]*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer])**2 -
                6*zt[layer]*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])**2 +
                6*zt[layer]*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer])**2 +
                6*zt[layer]*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])**2 -
                6*zt[layer]*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer])**2 -
                6*zt[layer]*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])**2 +
                6*zt[layer]*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer])**2 +
                6*zt[layer]*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])**2 -
                6*zt[layer]**2*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) +
                6*zt[layer]**2*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) +
                6*zt[layer]**2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer])**2 +
                6*zt[layer]**2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])**2 -
                6*zt[layer]**2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer])**2 -
                6*zt[layer]**2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])**2)
        for i in range(neig-1):
            for j in range(i + 1, neig):
                A[i, j] += (2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 -
                        3*zt[layer]**2*m[j]**2*m[i]**4 + 3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        2*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 2*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 2*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 2*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 2*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 6*m[j]*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                        6*m[j]*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                        6*m[j]*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                        6*m[j]*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                        6*m[j]*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                        6*m[j]*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) -
                        6*m[j]*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                        6*m[j]*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                        4*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        - 4*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        - 4*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        + 4*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        - 4*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        + 4*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        + 4*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        - 4*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        + m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]**2*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]**2*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]**2*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]**2*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + 6*m[j]**2*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        6*m[j]**2*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        6*m[j]**2*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        6*m[j]**2*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        6*m[j]**2*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        6*m[j]**2*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        6*m[j]**2*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        6*m[j]**2*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        2*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 2*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 2*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 2*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 2*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 2*m[j]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                        2*m[j]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                        2*m[j]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                        2*m[j]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                        2*m[j]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                        2*m[j]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) -
                        2*m[j]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                        2*m[j]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) -
                        4*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        + 4*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        + 4*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        - 4*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        + 4*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        - 4*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        - 4*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        + 4*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        - 2*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 2*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]**2*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + 2*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - 2*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]**2*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + 2*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - 2*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]**2*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - 2*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 2*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]**2*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - 2*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 2*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 2*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 2*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]**2*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]**2*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]**2*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]**2*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + zb[layer]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        zb[layer]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        zb[layer]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        zb[layer]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        2*zb[layer]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        2*zb[layer]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        zb[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + zb[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - zb[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + zb[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*zb[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*zb[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*zb[layer]*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) -
                        2*zb[layer]*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) +
                        2*zb[layer]*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) -
                        2*zb[layer]*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) -
                        4*zb[layer]*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) +
                        4*zb[layer]*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) +
                        zb[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - zb[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + zb[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - zb[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - 2*zb[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 2*zb[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + 2*zb[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*zb[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*zb[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*zb[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 4*zb[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 4*zb[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 2*zb[layer]*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) +
                        2*zb[layer]*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) -
                        2*zb[layer]*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) +
                        2*zb[layer]*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) +
                        4*zb[layer]*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) -
                        4*zb[layer]*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) -
                        2*zb[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 2*zb[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - 2*zb[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 2*zb[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + 4*zb[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - 4*zb[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - zb[layer]*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        zb[layer]*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        zb[layer]*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        zb[layer]*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        2*zb[layer]*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        2*zb[layer]*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        zb[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + zb[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - zb[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + zb[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*zb[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*zb[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + zb[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - zb[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + zb[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - zb[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - 2*zb[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 2*zb[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - zb[layer]**2*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        zb[layer]**2*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        zb[layer]**2*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                        zb[layer]**2*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                        2*zb[layer]**2*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        2*zb[layer]**2*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        2*zb[layer]**2*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                        2*zb[layer]**2*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) -
                        zb[layer]**2*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        zb[layer]**2*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        zb[layer]**2*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                        zb[layer]**2*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) -
                        2*zt[layer]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        2*zt[layer]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        zt[layer]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        zt[layer]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        zt[layer]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        zt[layer]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        2*zt[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*zt[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - zt[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + zt[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - zt[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + zt[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 4*zt[layer]*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) +
                        4*zt[layer]*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) +
                        2*zt[layer]*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) -
                        2*zt[layer]*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) +
                        2*zt[layer]*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) -
                        2*zt[layer]*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) -
                        2*zt[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 2*zt[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + zt[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - zt[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + zt[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - zt[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - 4*zt[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 4*zt[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*zt[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*zt[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*zt[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*zt[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 4*zt[layer]*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) -
                        4*zt[layer]*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) -
                        2*zt[layer]*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) +
                        2*zt[layer]*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) -
                        2*zt[layer]*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) +
                        2*zt[layer]*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) +
                        4*zt[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - 4*zt[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - 2*zt[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 2*zt[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - 2*zt[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 2*zt[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + 2*zt[layer]*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        2*zt[layer]*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        zt[layer]*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        zt[layer]*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        zt[layer]*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        zt[layer]*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        2*zt[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*zt[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - zt[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + zt[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - zt[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + zt[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 2*zt[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 2*zt[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + zt[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - zt[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + zt[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - zt[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + zt[layer]*zb[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        zt[layer]*zb[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        zt[layer]*zb[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        zt[layer]*zb[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        zt[layer]*zb[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                        zt[layer]*zb[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) -
                        zt[layer]*zb[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                        zt[layer]*zb[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) -
                        2*zt[layer]*zb[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        2*zt[layer]*zb[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        2*zt[layer]*zb[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        2*zt[layer]*zb[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        2*zt[layer]*zb[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                        2*zt[layer]*zb[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                        2*zt[layer]*zb[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                        2*zt[layer]*zb[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                        zt[layer]*zb[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        zt[layer]*zb[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        zt[layer]*zb[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        zt[layer]*zb[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        zt[layer]*zb[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                        zt[layer]*zb[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) -
                        zt[layer]*zb[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                        zt[layer]*zb[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) -
                        zt[layer]**2*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        zt[layer]**2*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        zt[layer]**2*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                        zt[layer]**2*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                        2*zt[layer]**2*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        2*zt[layer]**2*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        2*zt[layer]**2*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                        2*zt[layer]**2*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) -
                        zt[layer]**2*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        zt[layer]**2*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        zt[layer]**2*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                        zt[layer]**2*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 - 6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 +
                        2*zt[layer]*zb[layer]*m[j]**6 + zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]))

    #A is symmetric
    for i in range(neig - 1):
        for j in range(i + 1, neig):
            A[j, i] = A[i, j]

    return A

def pdim1sin_D_aDf_linear(m, a, **kwargs):
    """wrapper for dim1sin_D_aDf_linear with PolyLine input

    See also
    --------
    dim1sin_D_aDf_linear

    """

    return dim1sin_D_aDf_linear(m, a.y1, a.y2, a.x1, a.x2, **kwargs)

def dim1sin_D_aDf_linear_old(m, at, ab, zt, zb):
    """Generate code to calculate spectral method integrations

    Performs integrations of `sin(mi * z) * D[a(z) * D[sin(mj * z),z],z]`
    between [0, 1] where a(z) i piecewise linear functions of z.
    Calulation of integrals is performed at each element of a square symmetric
    matrix (size depends on size of `m`)

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    at : ``list`` of ``float``
        property at top of each layer
    ab : ``list`` of ``float``
        property at bottom of each layer
    zt : ``list`` of ``float``
        normalised depth or z-coordinate at top of each layer. `zt[0]` = 0
    zb : ``list`` of ``float``
        normalised depth or z-coordinate at bottom of each layer. `zt[-1]` = 1

    Returns
    -------
    A : numpy.ndarray
        returns a square symmetric matrix, size determined by size of `m`

    See Also
    --------
    m_from_sin_mx : used to generate 'm' input parameter
    geotecha.speccon.integrals_generate_code.dim1sin_abf_linear : sympy
    integrations to produce the code to this function.

    Notes
    -----
    The `dim1sin_D_aDf_linear` matrix, :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{\\frac{d}{dz}\\left({a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\right)\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    ..math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` and :math:`b\\left(z\\right)` are piecewise
    linear functions w.r.t. :math:`z`, that within a layer are defined by:

    ..math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively."""

    from math import sin, cos

    neig = len(m)
    nlayers = len(zt)

    A = np.zeros([neig, neig], float)
    for layer in range(nlayers):
        for i in range(neig):
            A[i, i] += ((zb[layer] - zt[layer])**(-1)*(ab[layer] - at[layer])*sin(m[i]*zb[layer])**2/2 - (zb[layer] -
                zt[layer])**(-1)*(ab[layer] - at[layer])*sin(m[i]*zt[layer])**2/2 - m[i]*((zb[layer] -
                zt[layer])**(-1)*(ab[layer] - at[layer])*(zb[layer] - zt[layer]) +
                at[layer])*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) +
                m[i]*at[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) - m[i]**2*(-(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*cos(m[i]*zb[layer])**2 + (4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*cos(m[i]*zb[layer])**2 - 2*m[i]*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) +
                2*m[i]*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) +
                m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]**2*sin(m[i]*zb[layer])**2 +
                m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])**2 -
                m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]**2*sin(m[i]*zb[layer])**2 -
                m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])**2 -
                2*zb[layer]*m[i]*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) +
                2*zb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]*sin(m[i]*zb[layer])**2 +
                2*zb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]*cos(m[i]*zb[layer])**2 +
                2*zt[layer]*m[i]*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) -
                2*zt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]*sin(m[i]*zb[layer])**2 -
                2*zt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]*cos(m[i]*zb[layer])**2) +
                m[i]**2*(-(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*ab[layer]*cos(m[i]*zt[layer])**2 +
                (4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*cos(m[i]*zt[layer])**2 -
                2*m[i]*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) +
                2*m[i]*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) +
                m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]**2*sin(m[i]*zt[layer])**2 +
                m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])**2 -
                m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]**2*sin(m[i]*zt[layer])**2 -
                m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])**2 -
                2*zb[layer]*m[i]*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) +
                2*zb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]*sin(m[i]*zt[layer])**2 +
                2*zb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]*cos(m[i]*zt[layer])**2 +
                2*zt[layer]*m[i]*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) -
                2*zt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]*sin(m[i]*zt[layer])**2 -
                2*zt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]*cos(m[i]*zt[layer])**2))
        for i in range(neig-1):
            for j in range(i + 1, neig):
                A[i, j] += ((zb[layer] - zt[layer])**(-1)*m[j]*(ab[layer] - at[layer])*(-m[i]*(m[i]**2 -
                    m[j]**2)**(-1)*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) - m[j]*(m[i]**2 -
                    m[j]**2)**(-1)*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])) - (zb[layer] -
                    zt[layer])**(-1)*m[j]*(ab[layer] - at[layer])*(-m[i]*(m[i]**2 -
                    m[j]**2)**(-1)*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) - m[j]*(m[i]**2 -
                    m[j]**2)**(-1)*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])) - m[j]*((zb[layer] -
                    zt[layer])**(-1)*(ab[layer] - at[layer])*(zb[layer] - zt[layer]) +
                    at[layer])*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                    m[j]*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) - m[j]**2*(m[i]**2*(zb[layer]*m[i]**4 -
                    2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 +
                    2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                    m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                    m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                    m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                    2*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) -
                    2*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) +
                    m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                    m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                    m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                    m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                    m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                    m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                    m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                    m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                    zb[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                    zb[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                    zb[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                    zb[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                    zt[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                    zt[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                    zt[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                    zt[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])) +
                    m[j]**2*(m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                    m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                    m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                    m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                    2*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) -
                    2*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) +
                    m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) -
                    m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                    m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                    m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                    m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                    m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                    m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                    m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) -
                    zb[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                    zb[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                    zb[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                    zb[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                    zt[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                    zt[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) -
                    zt[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                    zt[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])))

    #A is symmetric
    for i in range(neig - 1):
        for j in range(i + 1, neig):
            A[j, i] = A[i, j]

    return A

def pdim1sin_D_aDf_linear_v2(m, a, **kwargs):
    """wrapper for dim1sin_D_aDf_linear_v2 with PolyLine input

    See also
    --------
    dim1sin_D_aDf_linear_v2

    """
    #a, b = pwise.polyline_make_x_common(a,b)
    return dim1sin_D_aDf_linear_v2(m, a.y1, a.y2, a.x1, a.x2, **kwargs)

def dim1sin_D_aDf_linear_v2(m, at, ab, zt, zb):
    """Generate code to calculate spectral method integrations

    Performs integrations of `sin(mi * z) * D[a(z) * D[sin(mj * z),z],z]`
    between [0, 1] where a(z) i piecewise linear functions of z.
    Calulation of integrals is performed at each element of a square symmetric
    matrix (size depends on size of `m`)

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    at : ``list`` of ``float``
        property at top of each layer
    ab : ``list`` of ``float``
        property at bottom of each layer
    zt : ``list`` of ``float``
        normalised depth or z-coordinate at top of each layer. `zt[0]` = 0
    zb : ``list`` of ``float``
        normalised depth or z-coordinate at bottom of each layer. `zt[-1]` = 1

    Returns
    -------
    A : numpy.ndarray
        returns a square symmetric matrix, size determined by size of `m`

    See Also
    --------
    m_from_sin_mx : used to generate 'm' input parameter
    geotecha.speccon.integrals_generate_code.dim1sin_abf_linear : sympy
    integrations to produce the code to this function.

    Notes
    -----
    The `dim1sin_D_aDf_linear` matrix, :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{\\frac{d}{dz}\\left({a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\right)\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    ..math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` and :math:`b\\left(z\\right)` are piecewise
    linear functions w.r.t. :math:`z`, that within a layer are defined by:

    ..math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively."""

    from math import sin, cos

    from math import sin, cos

    neig = len(m)
    nlayers = len(zt)

    A = np.zeros([neig, neig], float)
    for layer in range(nlayers):
        for i in range(neig):
            A[i, i] += (-m[i]**2*((4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*ab[layer]*cos(m[i]*zb[layer])**2 -
                (4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*cos(m[i]*zb[layer])**2 +
                2*m[i]*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) -
                2*m[i]*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) +
                m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]**2*sin(m[i]*zb[layer])**2 +
                m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])**2 -
                m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]**2*sin(m[i]*zb[layer])**2 -
                m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])**2 +
                2*zb[layer]*m[i]*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) +
                2*zb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]*sin(m[i]*zb[layer])**2 +
                2*zb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]*cos(m[i]*zb[layer])**2 -
                2*zt[layer]*m[i]*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) -
                2*zt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]*sin(m[i]*zb[layer])**2 -
                2*zt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]*cos(m[i]*zb[layer])**2) +
                m[i]**2*((4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*ab[layer]*cos(m[i]*zt[layer])**2 -
                (4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*cos(m[i]*zt[layer])**2 +
                2*m[i]*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) -
                2*m[i]*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) +
                m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]**2*sin(m[i]*zt[layer])**2 +
                m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])**2 -
                m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]**2*sin(m[i]*zt[layer])**2 -
                m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])**2 +
                2*zb[layer]*m[i]*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) +
                2*zb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]*sin(m[i]*zt[layer])**2 +
                2*zb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]*cos(m[i]*zt[layer])**2 -
                2*zt[layer]*m[i]*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) -
                2*zt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]*sin(m[i]*zt[layer])**2 -
                2*zt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]*cos(m[i]*zt[layer])**2))
        for i in range(neig-1):
            for j in range(i + 1, neig):
                A[i, j] += (-m[j]*m[i]*(m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) -
                    m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) +
                    m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                    m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                    2*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                    2*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                    m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                    m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                    m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) -
                    m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) -
                    m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                    m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                    m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                    m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                    zb[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                    zb[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                    zb[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                    zb[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                    zt[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                    zt[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                    zt[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                    zt[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])) +
                    m[j]*m[i]*(m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) -
                    m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) +
                    m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) -
                    m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                    2*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                    2*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                    m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                    m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                    m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) -
                    m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) -
                    m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                    m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                    m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                    m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4
                    + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                    zb[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) -
                    zb[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                    zb[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                    zb[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                    zt[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                    zt[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                    zt[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) -
                    zt[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 -
                    zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                    zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])))

    #A is symmetric
    for i in range(neig - 1):
        for j in range(i + 1, neig):
            A[j, i] = A[i, j]

    return A

def pdim1sin_ab_linear(m, a, b, **kwargs):
    """wrapper for dim1sin_ab_linear with PolyLine input

    See also
    --------
    dim1sin_ab_linear

    """
    a, b = pwise.polyline_make_x_common(a,b)
    return dim1sin_ab_linear(m, a.y1, a.y2, b.y1, b.y2, a.x1, a.x2, **kwargs)

def dim1sin_ab_linear_old(m, at, ab, bt, bb, zt, zb):
    """Create matrix of spectral integrations

    Performs integrations of `sin(mi * z) * a(z) *b(z)`
    between [0, 1] where a(z) and b(z) are piecewise linear functions of z.
    Calulation of integrals is performed at each element of a 1d array
    (size depends on size of `m`)

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    at : ``list`` of ``float``
        property at top of each layer
    ab : ``list`` of ``float``
        property at bottom of each layer
    bt : ``list`` of ``float``
        2nd property at top of each layer
    bb : ``list`` of ``float``
        2nd property at bottom of each layer
    zt : ``list`` of ``float``
        normalised depth or z-coordinate at top of each layer. `zt[0]` = 0
    zb : ``list`` of ``float``
        normalised depth or z-coordinate at bottom of each layer. `zt[-1]` = 1

    Returns
    -------
    A : numpy.ndarray
        returns a 1d array size determined by size of `m`. treat as column
        vector

    See Also
    --------
    m_from_sin_mx : used to generate 'm' input parameter
    geotecha.integrals_generate_code.dim1sin_ab_linear : use sympy
    to perform the integrals symbolically and generate expressions for
    this function

    Notes
    -----
    The `dim1sin_ab_linear` which should be treated as a column vector,
    :math:`A` is given by:

    .. math:: \\mathbf{A}_{i}=\\int_{0}^1{{a\\left(z\\right)}{b\\left(z\\right)}\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    ..math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` and :math:`b\\left(z\\right)` are piecewise
    linear functions w.r.t. :math:`z`, that within a layer are defined by:

    ..math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """


    from math import sin, cos

    neig = len(m)
    nlayers = len(zt)

    A = np.zeros(neig, float)
    for layer in range(nlayers):
        for i in range(neig):
            A[i] += (2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zb[layer]) - 2*(zb[layer]**2*m[i]**3 -
                2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zt[layer]) - 2*(zb[layer]**2*m[i]**3 -
                2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer]) + 2*(zb[layer]**2*m[i]**3 -
                2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer]) - 2*(zb[layer]**2*m[i]**3 -
                2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer]) + 2*(zb[layer]**2*m[i]**3 -
                2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer]) + 2*(zb[layer]**2*m[i]**3 -
                2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*cos(m[i]*zb[layer]) - 2*(zb[layer]**2*m[i]**3 -
                2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*cos(m[i]*zt[layer]) +
                2*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer]) -
                2*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer]) -
                2*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer]) +
                2*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer]) -
                2*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer]) +
                2*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer]) +
                2*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer]) -
                2*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer]) -
                m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer]) +
                m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer]) +
                m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer]) -
                m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer]) +
                m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer]) -
                m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer]) -
                m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer]) +
                m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer]) +
                zb[layer]*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*sin(m[i]*zb[layer]) -
                zb[layer]*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*sin(m[i]*zt[layer]) +
                zb[layer]*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*sin(m[i]*zb[layer]) -
                zb[layer]*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*sin(m[i]*zt[layer]) -
                2*zb[layer]*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*sin(m[i]*zb[layer]) +
                2*zb[layer]*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*sin(m[i]*zt[layer]) -
                zb[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer]) +
                zb[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer]) -
                zb[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer]) +
                zb[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer]) +
                2*zb[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer]) -
                2*zb[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer]) -
                zb[layer]**2*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*cos(m[i]*zb[layer]) +
                zb[layer]**2*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*cos(m[i]*zt[layer]) -
                2*zt[layer]*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*sin(m[i]*zb[layer]) +
                2*zt[layer]*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*sin(m[i]*zt[layer]) +
                zt[layer]*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*sin(m[i]*zb[layer]) -
                zt[layer]*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*sin(m[i]*zt[layer]) +
                zt[layer]*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*sin(m[i]*zb[layer]) -
                zt[layer]*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*sin(m[i]*zt[layer]) +
                2*zt[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer]) -
                2*zt[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer]) -
                zt[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer]) +
                zt[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer]) -
                zt[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer]) +
                zt[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer]) +
                zt[layer]*zb[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer]) -
                zt[layer]*zb[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer]) +
                zt[layer]*zb[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer]) -
                zt[layer]*zb[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer]) -
                zt[layer]**2*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zb[layer]) +
                zt[layer]**2*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zt[layer]))

    return A

def pdim1sin_abc_linear(m, a, b, c, **kwargs):
    """wrapper for dim1sin_abc_linear with PolyLine input

    See also
    --------
    dim1sin_abc_linear

    """
    a, b, c = pwise.polyline_make_x_common(a,b,c)
    return dim1sin_abc_linear(m, a.y1, a.y2, b.y1, b.y2,c.y1, c.y2, a.x1, a.x2, **kwargs)

def dim1sin_abc_linear_old(m, at, ab, bt, bb, ct, cb, zt, zb):
    """Create matrix of spectral integrations

    Performs integrations of `sin(mi * z) * a(z) * b(z) * c(z)`
    between [0, 1] where a(z), b(z), c(z) are piecewise linear functions of z.
    Calulation of integrals is performed at each element of a 1d array
    (size depends on size of `m`)

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    at : ``list`` of ``float``
        property at top of each layer
    ab : ``list`` of ``float``
        property at bottom of each layer
    bt : ``list`` of ``float``
        2nd property at top of each layer
    bb : ``list`` of ``float``
        2nd property at bottom of each layer
    ct : ``list`` of ``float``
        3rd property at top of each layer
    cb : ``list`` of ``float``
        3rd property at bottom of each layer
    zt : ``list`` of ``float``
        normalised depth or z-coordinate at top of each layer. `zt[0]` = 0
    zb : ``list`` of ``float``
        normalised depth or z-coordinate at bottom of each layer. `zt[-1]` = 1

    Returns
    -------
    A : numpy.ndarray
        returns a 1d array size determined by size of `m`. treat as column
        vector

    See Also
    --------
    m_from_sin_mx : used to generate 'm' input parameter
    geotecha.integrals_generate_code.dim1sin_abc_linear : use sympy
    to perform the integrals symbolically and generate expressions for
    this function

    Notes
    -----
    The `dim1sin_abc_linear` which should be treated as a column vector,
    :math:`A` is given by:

    .. math:: \\mathbf{A}_{i}=\\int_{0}^1{{a\\left(z\\right)}{b\\left(z\\right)}{c\\left(z\\right)}\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    ..math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)`, :math:`b\\left(z\\right)`, and
    :math:`c\\left(z\\right)` are piecewise linear functions
    w.r.t. :math:`z`, that within a layer are defined by:

    ..math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """


    from math import sin, cos

    neig = len(m)
    nlayers = len(zt)

    A = np.zeros(neig, float)
    for layer in range(nlayers):
        for i in range(neig):
            A[i] += (-6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*sin(m[i]*zb[layer]) +
                6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*sin(m[i]*zt[layer]) +
                6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*sin(m[i]*zb[layer]) -
                6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*sin(m[i]*zt[layer]) +
                6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*sin(m[i]*zb[layer]) -
                6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*sin(m[i]*zt[layer]) -
                6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*sin(m[i]*zb[layer]) +
                6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*sin(m[i]*zt[layer]) +
                6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*sin(m[i]*zb[layer]) -
                6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*sin(m[i]*zt[layer]) -
                6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*sin(m[i]*zb[layer]) +
                6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*sin(m[i]*zt[layer]) -
                6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*sin(m[i]*zb[layer]) +
                6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*sin(m[i]*zt[layer]) +
                6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*sin(m[i]*zb[layer]) -
                6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*sin(m[i]*zt[layer]) +
                6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer]) -
                6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer]) -
                6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer]) +
                6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer]) -
                6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer]) +
                6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer]) +
                6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer]) -
                6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer]) -
                6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer]) +
                6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer]) +
                6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer]) -
                6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer]) +
                6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer]) -
                6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer]) -
                6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer]) +
                6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer]) +
                3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*zb[layer]**2*sin(m[i]*zb[layer]) -
                3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*zt[layer]**2*sin(m[i]*zt[layer]) -
                3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zb[layer]**2*sin(m[i]*zb[layer]) +
                3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zt[layer]**2*sin(m[i]*zt[layer]) -
                3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zb[layer]**2*sin(m[i]*zb[layer]) +
                3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zt[layer]**2*sin(m[i]*zt[layer]) +
                3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zb[layer]**2*sin(m[i]*zb[layer]) -
                3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zt[layer]**2*sin(m[i]*zt[layer]) -
                3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zb[layer]**2*sin(m[i]*zb[layer]) +
                3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zt[layer]**2*sin(m[i]*zt[layer]) +
                3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zb[layer]**2*sin(m[i]*zb[layer]) -
                3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zt[layer]**2*sin(m[i]*zt[layer]) +
                3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zb[layer]**2*sin(m[i]*zb[layer]) -
                3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zt[layer]**2*sin(m[i]*zt[layer]) -
                3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*zb[layer]**2*sin(m[i]*zb[layer]) +
                3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*zt[layer]**2*sin(m[i]*zt[layer]) -
                m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*zb[layer]**3*cos(m[i]*zb[layer]) +
                m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*zt[layer]**3*cos(m[i]*zt[layer]) +
                m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zb[layer]**3*cos(m[i]*zb[layer]) -
                m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zt[layer]**3*cos(m[i]*zt[layer]) +
                m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zb[layer]**3*cos(m[i]*zb[layer]) -
                m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zt[layer]**3*cos(m[i]*zt[layer]) -
                m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zb[layer]**3*cos(m[i]*zb[layer]) +
                m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zt[layer]**3*cos(m[i]*zt[layer]) +
                m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zb[layer]**3*cos(m[i]*zb[layer]) -
                m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zt[layer]**3*cos(m[i]*zt[layer]) -
                m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zb[layer]**3*cos(m[i]*zb[layer]) +
                m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zt[layer]**3*cos(m[i]*zt[layer]) -
                m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zb[layer]**3*cos(m[i]*zb[layer]) +
                m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zt[layer]**3*cos(m[i]*zt[layer]) +
                m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*zb[layer]**3*cos(m[i]*zb[layer]) -
                m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*zt[layer]**3*cos(m[i]*zt[layer]) +
                2*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*cos(m[i]*zb[layer]) -
                2*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*cos(m[i]*zt[layer]) +
                2*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*cos(m[i]*zb[layer]) -
                2*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*cos(m[i]*zt[layer]) -
                4*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*cos(m[i]*zb[layer]) +
                4*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*cos(m[i]*zt[layer]) +
                2*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*cos(m[i]*zb[layer]) -
                2*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*cos(m[i]*zt[layer]) -
                4*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*cos(m[i]*zb[layer]) +
                4*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*cos(m[i]*zt[layer]) -
                4*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*cos(m[i]*zb[layer]) +
                4*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*cos(m[i]*zt[layer]) +
                6*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*cos(m[i]*zb[layer]) -
                6*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*cos(m[i]*zt[layer]) +
                2*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer]) -
                2*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer]) +
                2*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer]) -
                2*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer]) -
                4*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer]) +
                4*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer]) +
                2*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer]) -
                2*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer]) -
                4*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer]) +
                4*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer]) -
                4*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer]) +
                4*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer]) +
                6*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer]) -
                6*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer]) -
                zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer]) +
                zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer]) -
                zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer]) +
                zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer]) +
                2*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer]) -
                2*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer]) -
                zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer]) +
                zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer]) +
                2*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer]) -
                2*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer]) +
                2*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer]) -
                2*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer]) -
                3*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer]) +
                3*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer]) +
                zb[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*sin(m[i]*zb[layer]) -
                zb[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*sin(m[i]*zt[layer]) +
                zb[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*sin(m[i]*zb[layer]) -
                zb[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*sin(m[i]*zt[layer]) +
                zb[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*sin(m[i]*zb[layer]) -
                zb[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*sin(m[i]*zt[layer]) -
                3*zb[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*sin(m[i]*zb[layer]) +
                3*zb[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*sin(m[i]*zt[layer]) -
                zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer]) +
                zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer]) -
                zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer]) +
                zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer]) -
                zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer]) +
                zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer]) +
                3*zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer]) -
                3*zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer]) -
                zb[layer]**3*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*cos(m[i]*zb[layer]) +
                zb[layer]**3*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*cos(m[i]*zt[layer]) -
                6*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*cos(m[i]*zb[layer]) +
                6*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*cos(m[i]*zt[layer]) +
                4*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*cos(m[i]*zb[layer]) -
                4*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*cos(m[i]*zt[layer]) +
                4*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*cos(m[i]*zb[layer]) -
                4*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*cos(m[i]*zt[layer]) -
                2*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*cos(m[i]*zb[layer]) +
                2*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*cos(m[i]*zt[layer]) +
                4*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*cos(m[i]*zb[layer]) -
                4*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*cos(m[i]*zt[layer]) -
                2*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*cos(m[i]*zb[layer]) +
                2*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*cos(m[i]*zt[layer]) -
                2*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*cos(m[i]*zb[layer]) +
                2*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*cos(m[i]*zt[layer]) -
                6*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer]) +
                6*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer]) +
                4*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer]) -
                4*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer]) +
                4*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer]) -
                4*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer]) -
                2*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer]) +
                2*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer]) +
                4*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer]) -
                4*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer]) -
                2*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer]) +
                2*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer]) -
                2*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer]) +
                2*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer]) +
                3*zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer]) -
                3*zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer]) -
                2*zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer]) +
                2*zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer]) -
                2*zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer]) +
                2*zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer]) +
                zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer]) -
                zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer]) -
                2*zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer]) +
                2*zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer]) +
                zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer]) -
                zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer]) +
                zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer]) -
                zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer]) -
                2*zt[layer]*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*sin(m[i]*zb[layer]) +
                2*zt[layer]*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*sin(m[i]*zt[layer]) -
                2*zt[layer]*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*sin(m[i]*zb[layer]) +
                2*zt[layer]*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*sin(m[i]*zt[layer]) +
                2*zt[layer]*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*sin(m[i]*zb[layer]) -
                2*zt[layer]*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*sin(m[i]*zt[layer]) -
                2*zt[layer]*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*sin(m[i]*zb[layer]) +
                2*zt[layer]*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*sin(m[i]*zt[layer]) +
                2*zt[layer]*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*sin(m[i]*zb[layer]) -
                2*zt[layer]*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*sin(m[i]*zt[layer]) +
                2*zt[layer]*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*sin(m[i]*zb[layer]) -
                2*zt[layer]*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*sin(m[i]*zt[layer]) +
                2*zt[layer]*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer]) -
                2*zt[layer]*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer]) +
                2*zt[layer]*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer]) -
                2*zt[layer]*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer]) -
                2*zt[layer]*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer]) +
                2*zt[layer]*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer]) +
                2*zt[layer]*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer]) -
                2*zt[layer]*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer]) -
                2*zt[layer]*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer]) +
                2*zt[layer]*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer]) -
                2*zt[layer]*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer]) +
                2*zt[layer]*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer]) +
                zt[layer]*zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*cos(m[i]*zb[layer]) -
                zt[layer]*zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*cos(m[i]*zt[layer]) +
                zt[layer]*zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*cos(m[i]*zb[layer]) -
                zt[layer]*zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*cos(m[i]*zt[layer]) +
                zt[layer]*zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*cos(m[i]*zb[layer]) -
                zt[layer]*zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*cos(m[i]*zt[layer]) +
                3*zt[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*sin(m[i]*zb[layer]) -
                3*zt[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*sin(m[i]*zt[layer]) -
                zt[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*sin(m[i]*zb[layer]) +
                zt[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*sin(m[i]*zt[layer]) -
                zt[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*sin(m[i]*zb[layer]) +
                zt[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*sin(m[i]*zt[layer]) -
                zt[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*sin(m[i]*zb[layer]) +
                zt[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*sin(m[i]*zt[layer]) -
                3*zt[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer]) +
                3*zt[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer]) +
                zt[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer]) -
                zt[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer]) +
                zt[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer]) -
                zt[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer]) +
                zt[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer]) -
                zt[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer]) -
                zt[layer]**2*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*cos(m[i]*zb[layer]) +
                zt[layer]**2*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*cos(m[i]*zt[layer]) -
                zt[layer]**2*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*cos(m[i]*zb[layer]) +
                zt[layer]**2*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*cos(m[i]*zt[layer]) -
                zt[layer]**2*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*cos(m[i]*zb[layer]) +
                zt[layer]**2*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*cos(m[i]*zt[layer]) +
                zt[layer]**3*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*cos(m[i]*zb[layer]) -
                zt[layer]**3*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                3*zt[layer]**2*zb[layer]*m[i]**4 -
                zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*cos(m[i]*zt[layer]))

    return A

def pdim1sin_D_aDb_linear(m, a, b, **kwargs):
    """wrapper for dim1sin_D_aDb_linear with PolyLine input

    See also
    --------
    dim1sin_D_aDb_linear

    """
    a, b = pwise.polyline_make_x_common(a,b)
    return dim1sin_D_aDb_linear(m, a.y1, a.y2, b.y1, b.y2, a.x1, a.x2, **kwargs)

def dim1sin_D_aDb_linear_old(m, at, ab, bt, bb, zt, zb):
    """Create matrix of spectral integrations

    Performs integrations of `sin(mi * z) * D[a(z) * D[b(z), z], z]`
    between [0, 1] where a(z) and b(z) are piecewise linear functions of z.
    Calulation of integrals is performed at each element of a 1d array
    (size depends on size of `m`)

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    at : ``list`` of ``float``
        property at top of each layer
    ab : ``list`` of ``float``
        property at bottom of each layer
    bt : ``list`` of ``float``
        2nd property at top of each layer
    bb : ``list`` of ``float``
        2nd property at bottom of each layer
    zt : ``list`` of ``float``
        normalised depth or z-coordinate at top of each layer. `zt[0]` = 0
    zb : ``list`` of ``float``
        normalised depth or z-coordinate at bottom of each layer. `zt[-1]` = 1

    Returns
    -------
    A : numpy.ndarray
        returns a 1d array size determined by size of `m`. treat as column
        vector

    See Also
    --------
    m_from_sin_mx : used to generate 'm' input parameter
    geotecha.integrals_generate_code.dim1sin_D_aDb_linear : use sympy
    to perform the integrals symbolically and generate expressions for
    this function

    Notes
    -----
    The `dim1sin_D_aDb_linear` which should be treated as a column vector,
    :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{\\frac{d}{dz}\\left({a\\left(z\\right)}\\frac{d}{dz}{b\\left(z\\right)}\\right)\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    ..math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` and :math:`b\\left(z\\right)` are piecewise
    linear functions w.r.t. :math:`z`, that within a layer are defined by:

    ..math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """


    from math import sin, cos

    neig = len(m)
    nlayers = len(zt)

    A = np.zeros(neig, float)
    for layer in range(nlayers):
        for i in range(neig):
            A[i] += (-(zb[layer] - zt[layer])**(-2)*m[i]**(-1)*(bb[layer] - bt[layer])*(ab[layer] -
                at[layer])*cos(m[i]*zb[layer]) + (zb[layer] - zt[layer])**(-2)*m[i]**(-1)*(bb[layer] -
                bt[layer])*(ab[layer] - at[layer])*cos(m[i]*zt[layer]) - (zb[layer] -
                zt[layer])**(-1)*(bb[layer] - bt[layer])*((zb[layer] - zt[layer])**(-1)*(ab[layer] -
                at[layer])*(zb[layer] - zt[layer]) + at[layer])*sin(m[i]*zb[layer]) + (zb[layer] -
                zt[layer])**(-1)*(bb[layer] - bt[layer])*at[layer]*sin(m[i]*zt[layer]))

    for i in range(neig):
        A[i] += (-(zb[0] - zt[0])**(-1)*(bb[0] - bt[0])*at[0]*sin(m[i]*zt[0]) + (zb[-1] - zt[-1])**(-1)*(bb[-1] -
            bt[-1])*((zb[-1] - zt[-1])**(-1)*(ab[-1] - at[-1])*(zb[-1] - zt[-1]) + at[-1])*sin(m[i]*zb[-1]))

    return A




def pEDload_linear(a, eigs, tvals, dT=1.0, **kwargs):
    """wrapper for EDload_linear with PolyLine input

    See also
    --------
    EDload_linear

    """

    return EDload_linear(a.x, a.y, eigs, tvals, dT, **kwargs)


def EDload_linear(loadtim, loadmag, eigs, tvals, dT=1.0):
    """Generate code to perform time integration for spectral methods

    Integrates D[load(tau), tau] * exp(dT * eig * (t-tau)) between [0, t].
    Performs integrations involving time and the time derivative of a
    piecewise linear load.  A 2d array of dimesnions A[len(tvals), len(eigs)]
    is produced where the 'i'th row of A contains the diagonal elements of the
    spectral 'E' matrix calculated for the time value tvals[i]. i.e. rows of
    this matrix will be assembled into the diagonal matrix 'E' elsewhere.


    Parameters
    ----------
    loadtim : 1d numpy.ndarray
        list of times describing load application
    loadmag : 1d numpy.ndarray
        list of load magnitudes
    eigs : 1d numpy.ndarray
        list of eigenvalues
    tvals : 1d numpy.ndarray`
        list of time values to calculate integral at
    dT : ``float``, optional
        time factor multiple (default = 1.0)

    Returns
    -------
    A : numpy.ndarray
        returns a 2d array of dimesnions A[len(tvals), len(eigs)].
        The 'i'th row of A is the diagonal elements of the spectral 'E' matrix
        calculated for the time tvals[i].
        vector

    Notes
    -----

    Assuming the load are formulated as the product of separate time and depth
    dependant functions:

    .. math:: \\sigma\\left({Z,t}\\right)=\\sigma\\left({Z}\\right)\\sigma\\left({t}\\right)

    the solution to the consolidation equation using the spectral method has
    the form:

    .. math:: u\\left(Z,t\\right)=\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}

    The matrix :math:`E` is a time dependent diagonal matrix due to time
    dependant loadings.  The version of :math:`E` calculated here in
    `EDload_linear` is from loading terms in the governing equation that are
    differentiated wrt :math:`t` (hence the 'D' in the function name).
    The diagonal elements of :math:`E` are given by:

    .. math:: \\mathbf{E}_{i,i}=\\int_{0}^t{\\frac{d{\\sigma\\left(\\tau\\right)}}{d\\tau}{\exp\\left({(dT\\left(t-\\tau\\right)\\lambda_i}\\right)}\\,d\\tau}

    where

     :math:`\\lambda_i` is the `ith` eigenvalue of the problem,
     :math:`dT` is a time factor for numerical convienience,
     :math:`\\sigma\left(\\tau\\right)` is the time dependant portion of the loading function.

    When the time dependant loading term :math:`\\sigma\\left(\\tau\\right)` is
    piecewise in time. The contribution of each load segment is found by:

    .. math:: \\mathbf{E}_{i,i}=\\int_{t_s}^{t_f}{\\frac{d{\\sigma\\left(\\tau\\right)}}{d\\tau}\\exp\\left({dT\\left(t-\\tau\\right)*\\lambda_i}\\right)\\,d\\tau}

    where

    .. math:: t_s = \\min\\left(t,t_{increment\\:start}\\right)

    .. math:: t_f = \\min\\left(t,t_{increment\\:end}\\right)

    (note that this function,`EDload_linear`, rather than use :math:`t_s` and
    :math:`t_f`,
    explicitly finds increments that the current time falls in, falls after,
    and falls before and treates each case on it's own.)

    Each :math:`t` value of interest requires a separate diagonal matrix
    :math:`E`.  To use space more efficiently and to facilitate numpy
    broadcasting when using the results of the function, the diagonal elements
    of :math:`E` for each time value `t` value are stored in the rows of
    array :math:`A` returned by `EDload_linear`.  Thus:

    .. math:: \\mathbf{A}=\\left(\\begin{matrix}E_{0,0}(t_0)&E_{1,1}(t_0)& \cdots & E_{neig-1,neig-1}(t_0)\\\ E_{0,0}(t_1)&E_{1,1}(t_1)& \\cdots & E_{neig-1,neig-1}(t_1)\\\ \\vdots&\\vdots&\\ddots&\\vdots \\\ E_{0,0}(t_m)&E_{1,1}(t_m)& \cdots & E_{neig-1,neig-1}(t_m)\\end{matrix}\\right)

    """


    from math import exp

    loadtim = np.asarray(loadtim)
    loadmag = np.asarray(loadmag)
    eigs = np.asarray(eigs)
    tvals = np.asarray(tvals)


    (ramps_less_than_t, constants_less_than_t, steps_less_than_t,
            ramps_containing_t, constants_containing_t) = segment_containing_also_segments_less_than_xi(loadtim, loadmag, tvals, steps_or_equal_to = True)


    A = np.zeros([len(tvals), len(eigs)])


    for i, t in enumerate(tvals):
        for k in steps_less_than_t[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += (loadmag[k + 1] - loadmag[k])*exp(-dT*eig*(t - loadtim[k]))
        for k in ramps_containing_t[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += ((loadmag[k + 1] - loadmag[k])*(loadtim[k + 1] - loadtim[k])**(-1)/(dT*eig) - exp(-dT*eig*(t -
                    loadtim[k]))*(loadmag[k + 1] - loadmag[k])*(loadtim[k + 1] - loadtim[k])**(-1)/(dT*eig))
        for k in ramps_less_than_t[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += (exp(-dT*eig*(t - loadtim[k + 1]))*(loadmag[k + 1] - loadmag[k])*(loadtim[k + 1] -
                    loadtim[k])**(-1)/(dT*eig) - exp(-dT*eig*(t - loadtim[k]))*(loadmag[k + 1] -
                    loadmag[k])*(loadtim[k + 1] - loadtim[k])**(-1)/(dT*eig))
    return A


def pEload_linear(a, eigs, tvals, dT=1.0, **kwargs):
    """wrapper for Eload_linear with PolyLine input

    See also
    --------
    Eload_linear

    """

    return Eload_linear(a.x, a.y, eigs, tvals, dT, **kwargs)

def Eload_linear(loadtim, loadmag, eigs, tvals, dT=1.0):
    """Generate code to perform time integration for spectral methods

    Integrates load(tau) * exp(dT * eig * (t-tau)) between [0, t].
    Performs integrations involving time and the time derivative of a
    piecewise linear load.  A 2d array of dimensions A[len(tvals), len(eigs)]
    is produced where the 'i'th row of A contains the diagonal elements of the
    spectral 'E' matrix calculated for the time value tvals[i]. i.e. rows of
    this matrix will be assembled into the diagonal matrix 'E' elsewhere.


    Parameters
    ----------
    loadtim : 1d numpy.ndarray
        list of times describing load application
    loadmag : 1d numpy.ndarray
        list of load magnitudes
    eigs : 1d numpy.ndarray
        list of eigenvalues
    tvals : 1d numpy.ndarray`
        list of time values to calculate integral at
    dT : ``float``, optional
        time factor multiple (default = 1.0)

    Returns
    -------
    A : numpy.ndarray
        returns a 2d array of dimesnions A[len(tvals), len(eigs)].
        The 'i'th row of A is the diagonal elements of the spectral 'E' matrix
        calculated for the time tvals[i].
        vector

    Notes
    -----

    Assuming the load are formulated as the product of separate time and depth
    dependant functions:

    .. math:: \\sigma\\left({Z,t}\\right)=\\sigma\\left({Z}\\right)\\sigma\\left({t}\\right)

    the solution to the consolidation equation using the spectral method has
    the form:

    .. math:: u\\left(Z,t\\right)=\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}

    The matrix :math:`E` is a time dependent diagonal matrix due to time
    dependant loadings.  The version of :math:`E` calculated here in
    `Eload_linear` is from loading terms in the governing equation that are NOT
    differentiated wrt :math:`t`.
    The diagonal elements of :math:`E` are given by:

    .. math:: \\mathbf{E}_{i,i}=\\int_{0}^t{{\\sigma\\left(\\tau\\right)}{\exp\\left({(dT\\left(t-\\tau\\right)\\lambda_i}\\right)}\\,d\\tau}

    where

     :math:`\\lambda_i` is the `ith` eigenvalue of the problem,
     :math:`dT` is a time factor for numerical convienience,
     :math:`\\sigma\left(\\tau\\right)` is the time dependant portion of the loading function.

    When the time dependant loading term :math:`\\sigma\\left(\\tau\\right)` is
    piecewise in time. The contribution of each load segment is found by:

    .. math:: \\mathbf{E}_{i,i}=\\int_{t_s}^{t_f}{{\\sigma\\left(\\tau\\right)}\\exp\\left({dT\\left(t-\\tau\\right)*\\lambda_i}\\right)\\,d\\tau}

    where

    .. math:: t_s = \\min\\left(t,t_{increment\\:start}\\right)

    .. math:: t_f = \\min\\left(t,t_{increment\\:end}\\right)

    (note that this function,`EDload_linear`, rather than use :math:`t_s` and
    :math:`t_f`,
    explicitly finds increments that the current time falls in, falls after,
    and falls before and treates each case on it's own.)

    Each :math:`t` value of interest requires a separate diagonal matrix
    :math:`E`.  To use space more efficiently and to facilitate numpy
    broadcasting when using the results of the function, the diagonal elements
    of :math:`E` for each time value `t` value are stored in the rows of
    array :math:`A` returned by `EDload_linear`.  Thus:

    .. math:: \\mathbf{A}=\\left(\\begin{matrix}E_{0,0}(t_0)&E_{1,1}(t_0)& \cdots & E_{neig-1,neig-1}(t_0)\\\ E_{0,0}(t_1)&E_{1,1}(t_1)& \\cdots & E_{neig-1,neig-1}(t_1)\\\ \\vdots&\\vdots&\\ddots&\\vdots \\\ E_{0,0}(t_m)&E_{1,1}(t_m)& \cdots & E_{neig-1,neig-1}(t_m)\\end{matrix}\\right)

    """



    from math import exp

    loadtim = np.asarray(loadtim)
    loadmag = np.asarray(loadmag)
    eigs = np.asarray(eigs)
    tvals = np.asarray(tvals)

    (ramps_less_than_t, constants_less_than_t, steps_less_than_t,
            ramps_containing_t, constants_containing_t) = segment_containing_also_segments_less_than_xi(loadtim, loadmag, tvals, steps_or_equal_to = True)

    A = np.zeros([len(tvals), len(eigs)])


    for i, t in enumerate(tvals):
        for k in constants_containing_t[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += -exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT*eig) + loadmag[k]/(dT*eig)
        for k in constants_less_than_t[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += (exp(-dT*eig*(t - loadtim[k + 1]))*loadmag[k]/(dT*eig) -
                    exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT*eig) )
        for k in ramps_containing_t[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += ((-t/(dT*eig) + 1/(dT**2*eig**2))*loadmag[k]*(loadtim[k + 1] - loadtim[k])**(-1) + (t/(dT*eig) -
                    1/(dT**2*eig**2))*loadmag[k + 1]*(loadtim[k + 1] - loadtim[k])**(-1) -
                    (-loadtim[k]*exp(-dT*eig*(t - loadtim[k]))/(dT*eig) + exp(-dT*eig*(t -
                    loadtim[k]))/(dT**2*eig**2))*loadmag[k]*(loadtim[k + 1] - loadtim[k])**(-1) -
                    (loadtim[k]*exp(-dT*eig*(t - loadtim[k]))/(dT*eig) - exp(-dT*eig*(t -
                    loadtim[k]))/(dT**2*eig**2))*loadmag[k + 1]*(loadtim[k + 1] - loadtim[k])**(-1) + exp(-dT*eig*(t
                    - loadtim[k]))*loadmag[k + 1]*(loadtim[k + 1] - loadtim[k])**(-1)*loadtim[k]/(dT*eig) -
                    exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT*eig) - exp(-dT*eig*(t -
                    loadtim[k]))*loadmag[k]*(loadtim[k + 1] - loadtim[k])**(-1)*loadtim[k]/(dT*eig) - loadmag[k +
                    1]*(loadtim[k + 1] - loadtim[k])**(-1)*loadtim[k]/(dT*eig) + loadmag[k]/(dT*eig) +
                    loadmag[k]*(loadtim[k + 1] - loadtim[k])**(-1)*loadtim[k]/(dT*eig))
        for k in ramps_less_than_t[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += ((-loadtim[k + 1]*exp(-dT*eig*(t - loadtim[k + 1]))/(dT*eig) + exp(-dT*eig*(t - loadtim[k +
                    1]))/(dT**2*eig**2))*loadmag[k]*(loadtim[k + 1] - loadtim[k])**(-1) + (loadtim[k +
                    1]*exp(-dT*eig*(t - loadtim[k + 1]))/(dT*eig) - exp(-dT*eig*(t - loadtim[k +
                    1]))/(dT**2*eig**2))*loadmag[k + 1]*(loadtim[k + 1] - loadtim[k])**(-1) -
                    (-loadtim[k]*exp(-dT*eig*(t - loadtim[k]))/(dT*eig) + exp(-dT*eig*(t -
                    loadtim[k]))/(dT**2*eig**2))*loadmag[k]*(loadtim[k + 1] - loadtim[k])**(-1) -
                    (loadtim[k]*exp(-dT*eig*(t - loadtim[k]))/(dT*eig) - exp(-dT*eig*(t -
                    loadtim[k]))/(dT**2*eig**2))*loadmag[k + 1]*(loadtim[k + 1] - loadtim[k])**(-1) - exp(-dT*eig*(t
                    - loadtim[k + 1]))*loadmag[k + 1]*(loadtim[k + 1] - loadtim[k])**(-1)*loadtim[k]/(dT*eig) +
                    exp(-dT*eig*(t - loadtim[k + 1]))*loadmag[k]/(dT*eig) + exp(-dT*eig*(t - loadtim[k +
                    1]))*loadmag[k]*(loadtim[k + 1] - loadtim[k])**(-1)*loadtim[k]/(dT*eig) + exp(-dT*eig*(t -
                    loadtim[k]))*loadmag[k + 1]*(loadtim[k + 1] - loadtim[k])**(-1)*loadtim[k]/(dT*eig) -
                    exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT*eig) - exp(-dT*eig*(t -
                    loadtim[k]))*loadmag[k]*(loadtim[k + 1] - loadtim[k])**(-1)*loadtim[k]/(dT*eig))
    return A

def pEDload_coslinear(a, omega, phase, eigs, tvals, dT=1.0, **kwargs):
    """wrapper for EDload_coslinear with PolyLine input

    See also
    --------
    EDload_coslinear

    """

    return EDload_linear(a.x, a.y, omega, phase, eigs, tvals, dT, **kwargs)


def EDload_coslinear(loadtim, loadmag, omega, phase, eigs, tvals, dT=1.0):

    from math import exp

    loadtim = np.asarray(loadtim)
    loadmag = np.asarray(loadmag)
    eigs = np.asarray(eigs)
    tvals = np.asarray(tvals)

    (ramps_less_than_t, constants_less_than_t, steps_less_than_t,
            ramps_containing_t, constants_containing_t) = segment_containing_also_segments_less_than_xi(loadtim, loadmag, tvals, steps_or_equal_to = True)

    A = np.zeros([len(tvals), len(eigs)])


    for i, t in enumerate(tvals):
        for k in after_instant_loads[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += ((loadmag[k + 1] - loadmag[k])*cos(omega*loadtim[k] + phase)*exp(-dT*eig*(t - loadtim[k])))
        for k in within_constant_loads[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += (-omega*(dT*eig*sin(omega*t + phase)/(dT**2*eig**2 + omega**2) - omega*cos(omega*t +
                    phase)/(dT**2*eig**2 + omega**2))*loadmag[k] + omega*(dT*eig*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2) -
                    omega*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2
                    + omega**2))*loadmag[k])
        for k in after_constant_loads[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += (-omega*(dT*eig*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 +
                    omega**2) - omega*exp(-dT*eig*(t - loadtim[k + 1]))*cos(omega*loadtim[k + 1] +
                    phase)/(dT**2*eig**2 + omega**2))*loadmag[k] + omega*(dT*eig*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2) -
                    omega*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2
                    + omega**2))*loadmag[k])
        for k in within_ramp_loads[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += (omega*(dT*eig*sin(omega*t + phase)/(dT**2*eig**2 + omega**2) - omega*cos(omega*t +
                    phase)/(dT**2*eig**2 + omega**2))*loadmag[k + 1]*(loadtim[k + 1] -
                    loadtim[k])**(-1)*loadtim[k] - omega*(dT*eig*sin(omega*t + phase)/(dT**2*eig**2
                    + omega**2) - omega*cos(omega*t + phase)/(dT**2*eig**2 + omega**2))*loadmag[k] -
                    omega*(dT*eig*sin(omega*t + phase)/(dT**2*eig**2 + omega**2) - omega*cos(omega*t
                    + phase)/(dT**2*eig**2 + omega**2))*loadmag[k]*(loadtim[k + 1] -
                    loadtim[k])**(-1)*loadtim[k] - omega*(dT**3*eig**3*t*sin(omega*t +
                    phase)/(dT**4*eig**4 + 2*dT**2*eig**2*omega**2 + omega**4) -
                    dT**2*eig**2*omega*t*cos(omega*t + phase)/(dT**4*eig**4 +
                    2*dT**2*eig**2*omega**2 + omega**4) - dT**2*eig**2*sin(omega*t +
                    phase)/(dT**4*eig**4 + 2*dT**2*eig**2*omega**2 + omega**4) +
                    dT*eig*omega**2*t*sin(omega*t + phase)/(dT**4*eig**4 + 2*dT**2*eig**2*omega**2 +
                    omega**4) + 2*dT*eig*omega*cos(omega*t + phase)/(dT**4*eig**4 +
                    2*dT**2*eig**2*omega**2 + omega**4) - omega**3*t*cos(omega*t +
                    phase)/(dT**4*eig**4 + 2*dT**2*eig**2*omega**2 + omega**4) +
                    omega**2*sin(omega*t + phase)/(dT**4*eig**4 + 2*dT**2*eig**2*omega**2 +
                    omega**4))*loadmag[k + 1]*(loadtim[k + 1] - loadtim[k])**(-1) +
                    omega*(dT**3*eig**3*t*sin(omega*t + phase)/(dT**4*eig**4 +
                    2*dT**2*eig**2*omega**2 + omega**4) - dT**2*eig**2*omega*t*cos(omega*t +
                    phase)/(dT**4*eig**4 + 2*dT**2*eig**2*omega**2 + omega**4) -
                    dT**2*eig**2*sin(omega*t + phase)/(dT**4*eig**4 + 2*dT**2*eig**2*omega**2 +
                    omega**4) + dT*eig*omega**2*t*sin(omega*t + phase)/(dT**4*eig**4 +
                    2*dT**2*eig**2*omega**2 + omega**4) + 2*dT*eig*omega*cos(omega*t +
                    phase)/(dT**4*eig**4 + 2*dT**2*eig**2*omega**2 + omega**4) -
                    omega**3*t*cos(omega*t + phase)/(dT**4*eig**4 + 2*dT**2*eig**2*omega**2 +
                    omega**4) + omega**2*sin(omega*t + phase)/(dT**4*eig**4 +
                    2*dT**2*eig**2*omega**2 + omega**4))*loadmag[k]*(loadtim[k + 1] -
                    loadtim[k])**(-1) - omega*(dT*eig*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2) -
                    omega*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2
                    + omega**2))*loadmag[k + 1]*(loadtim[k + 1] - loadtim[k])**(-1)*loadtim[k] +
                    omega*(dT*eig*exp(-dT*eig*(t - loadtim[k]))*sin(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2) - omega*exp(-dT*eig*(t -
                    loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2))*loadmag[k]
                    + omega*(dT*eig*exp(-dT*eig*(t - loadtim[k]))*sin(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2) - omega*exp(-dT*eig*(t -
                    loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2 +
                    omega**2))*loadmag[k]*(loadtim[k + 1] - loadtim[k])**(-1)*loadtim[k] +
                    omega*(dT**3*eig**3*loadtim[k]*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2)**2 -
                    dT**2*eig**2*omega*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k]
                    + phase)/(dT**2*eig**2 + omega**2)**2 - dT**2*eig**2*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2)**2 +
                    dT*eig*omega**2*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*sin(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2)**2 + 2*dT*eig*omega*exp(-dT*eig*(t -
                    loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2)**2 -
                    omega**3*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2)**2 + omega**2*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 +
                    omega**2)**2)*loadmag[k + 1]*(loadtim[k + 1] - loadtim[k])**(-1) -
                    omega*(dT**3*eig**3*loadtim[k]*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2)**2 -
                    dT**2*eig**2*omega*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k]
                    + phase)/(dT**2*eig**2 + omega**2)**2 - dT**2*eig**2*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2)**2 +
                    dT*eig*omega**2*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*sin(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2)**2 + 2*dT*eig*omega*exp(-dT*eig*(t -
                    loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2)**2 -
                    omega**3*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2)**2 + omega**2*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 +
                    omega**2)**2)*loadmag[k]*(loadtim[k + 1] - loadtim[k])**(-1) +
                    (-dT*eig*cos(omega*t + phase)/(dT**2*eig**2 + omega**2) - omega*sin(omega*t +
                    phase)/(dT**2*eig**2 + omega**2))*loadmag[k]*(loadtim[k + 1] - loadtim[k])**(-1)
                    + (dT*eig*cos(omega*t + phase)/(dT**2*eig**2 + omega**2) + omega*sin(omega*t +
                    phase)/(dT**2*eig**2 + omega**2))*loadmag[k + 1]*(loadtim[k + 1] -
                    loadtim[k])**(-1) - (-dT*eig*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k]
                    + phase)/(dT**2*eig**2 + omega**2) - omega*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 +
                    omega**2))*loadmag[k]*(loadtim[k + 1] - loadtim[k])**(-1) -
                    (dT*eig*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2) + omega*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2))*loadmag[k
                    + 1]*(loadtim[k + 1] - loadtim[k])**(-1))
        for k in after_ramp_loads[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += (omega*(dT*eig*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 +
                    omega**2) - omega*exp(-dT*eig*(t - loadtim[k + 1]))*cos(omega*loadtim[k + 1] +
                    phase)/(dT**2*eig**2 + omega**2))*loadmag[k + 1]*(loadtim[k + 1] -
                    loadtim[k])**(-1)*loadtim[k] - omega*(dT*eig*exp(-dT*eig*(t - loadtim[k +
                    1]))*sin(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 + omega**2) -
                    omega*exp(-dT*eig*(t - loadtim[k + 1]))*cos(omega*loadtim[k + 1] +
                    phase)/(dT**2*eig**2 + omega**2))*loadmag[k] - omega*(dT*eig*exp(-dT*eig*(t -
                    loadtim[k + 1]))*sin(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 + omega**2) -
                    omega*exp(-dT*eig*(t - loadtim[k + 1]))*cos(omega*loadtim[k + 1] +
                    phase)/(dT**2*eig**2 + omega**2))*loadmag[k]*(loadtim[k + 1] -
                    loadtim[k])**(-1)*loadtim[k] - omega*(dT*eig*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2) -
                    omega*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2
                    + omega**2))*loadmag[k + 1]*(loadtim[k + 1] - loadtim[k])**(-1)*loadtim[k] +
                    omega*(dT*eig*exp(-dT*eig*(t - loadtim[k]))*sin(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2) - omega*exp(-dT*eig*(t -
                    loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2))*loadmag[k]
                    + omega*(dT*eig*exp(-dT*eig*(t - loadtim[k]))*sin(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2) - omega*exp(-dT*eig*(t -
                    loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2 +
                    omega**2))*loadmag[k]*(loadtim[k + 1] - loadtim[k])**(-1)*loadtim[k] -
                    omega*(dT**3*eig**3*loadtim[k + 1]*exp(-dT*eig*(t - loadtim[k +
                    1]))*sin(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 + omega**2)**2 -
                    dT**2*eig**2*omega*loadtim[k + 1]*exp(-dT*eig*(t - loadtim[k +
                    1]))*cos(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 + omega**2)**2 -
                    dT**2*eig**2*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*loadtim[k + 1] +
                    phase)/(dT**2*eig**2 + omega**2)**2 + dT*eig*omega**2*loadtim[k +
                    1]*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*loadtim[k + 1] +
                    phase)/(dT**2*eig**2 + omega**2)**2 + 2*dT*eig*omega*exp(-dT*eig*(t - loadtim[k
                    + 1]))*cos(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 + omega**2)**2 -
                    omega**3*loadtim[k + 1]*exp(-dT*eig*(t - loadtim[k + 1]))*cos(omega*loadtim[k +
                    1] + phase)/(dT**2*eig**2 + omega**2)**2 + omega**2*exp(-dT*eig*(t - loadtim[k +
                    1]))*sin(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 + omega**2)**2)*loadmag[k +
                    1]*(loadtim[k + 1] - loadtim[k])**(-1) + omega*(dT**3*eig**3*loadtim[k +
                    1]*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*loadtim[k + 1] +
                    phase)/(dT**2*eig**2 + omega**2)**2 - dT**2*eig**2*omega*loadtim[k +
                    1]*exp(-dT*eig*(t - loadtim[k + 1]))*cos(omega*loadtim[k + 1] +
                    phase)/(dT**2*eig**2 + omega**2)**2 - dT**2*eig**2*exp(-dT*eig*(t - loadtim[k +
                    1]))*sin(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 + omega**2)**2 +
                    dT*eig*omega**2*loadtim[k + 1]*exp(-dT*eig*(t - loadtim[k +
                    1]))*sin(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 + omega**2)**2 +
                    2*dT*eig*omega*exp(-dT*eig*(t - loadtim[k + 1]))*cos(omega*loadtim[k + 1] +
                    phase)/(dT**2*eig**2 + omega**2)**2 - omega**3*loadtim[k + 1]*exp(-dT*eig*(t -
                    loadtim[k + 1]))*cos(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 + omega**2)**2
                    + omega**2*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*loadtim[k + 1] +
                    phase)/(dT**2*eig**2 + omega**2)**2)*loadmag[k]*(loadtim[k + 1] -
                    loadtim[k])**(-1) + omega*(dT**3*eig**3*loadtim[k]*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2)**2 -
                    dT**2*eig**2*omega*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k]
                    + phase)/(dT**2*eig**2 + omega**2)**2 - dT**2*eig**2*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2)**2 +
                    dT*eig*omega**2*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*sin(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2)**2 + 2*dT*eig*omega*exp(-dT*eig*(t -
                    loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2)**2 -
                    omega**3*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2)**2 + omega**2*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 +
                    omega**2)**2)*loadmag[k + 1]*(loadtim[k + 1] - loadtim[k])**(-1) -
                    omega*(dT**3*eig**3*loadtim[k]*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2)**2 -
                    dT**2*eig**2*omega*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k]
                    + phase)/(dT**2*eig**2 + omega**2)**2 - dT**2*eig**2*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2)**2 +
                    dT*eig*omega**2*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*sin(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2)**2 + 2*dT*eig*omega*exp(-dT*eig*(t -
                    loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2)**2 -
                    omega**3*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2)**2 + omega**2*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 +
                    omega**2)**2)*loadmag[k]*(loadtim[k + 1] - loadtim[k])**(-1) +
                    (-dT*eig*exp(-dT*eig*(t - loadtim[k + 1]))*cos(omega*loadtim[k + 1] +
                    phase)/(dT**2*eig**2 + omega**2) - omega*exp(-dT*eig*(t - loadtim[k +
                    1]))*sin(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 +
                    omega**2))*loadmag[k]*(loadtim[k + 1] - loadtim[k])**(-1) +
                    (dT*eig*exp(-dT*eig*(t - loadtim[k + 1]))*cos(omega*loadtim[k + 1] +
                    phase)/(dT**2*eig**2 + omega**2) + omega*exp(-dT*eig*(t - loadtim[k +
                    1]))*sin(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 + omega**2))*loadmag[k +
                    1]*(loadtim[k + 1] - loadtim[k])**(-1) - (-dT*eig*exp(-dT*eig*(t -
                    loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2) -
                    omega*exp(-dT*eig*(t - loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2
                    + omega**2))*loadmag[k]*(loadtim[k + 1] - loadtim[k])**(-1) -
                    (dT*eig*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2) + omega*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2))*loadmag[k
                    + 1]*(loadtim[k + 1] - loadtim[k])**(-1))
    return A

def pEload_coslinear(a, omega, phase, eigs, tvals, dT=1.0, **kwargs):
    """wrapper for Eload_coslinear with PolyLine input

    See also
    --------
    Eload_coslinear

    """

    return Eload_linear(a.x, a.y, omega, phase, eigs, tvals, dT, **kwargs)


def Eload_coslinear(loadtim, loadmag, omega, phase, eigs, tvals, dT=1.0):

    from math import exp

    loadtim = np.asarray(loadtim)
    loadmag = np.asarray(loadmag)
    eigs = np.asarray(eigs)
    tvals = np.asarray(tvals)

    (ramps_less_than_t, constants_less_than_t, steps_less_than_t,
            ramps_containing_t, constants_containing_t) = segment_containing_also_segments_less_than_xi(loadtim, loadmag, tvals, steps_or_equal_to = True)

    A = np.zeros([len(tvals), len(eigs)])


    for i, t in enumerate(tvals):
        for k in within_constant_loads[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += ((dT*eig*cos(omega*t + phase)/(dT**2*eig**2 + omega**2) + omega*sin(omega*t + phase)/(dT**2*eig**2 +
                    omega**2))*loadmag[k] - (dT*eig*exp(-dT*eig*(t -
                    loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2) +
                    omega*exp(-dT*eig*(t - loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2
                    + omega**2))*loadmag[k])
        for k in after_constant_loads[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += ((dT*eig*exp(-dT*eig*(t - loadtim[k + 1]))*cos(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 +
                    omega**2) + omega*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*loadtim[k + 1] +
                    phase)/(dT**2*eig**2 + omega**2))*loadmag[k] - (dT*eig*exp(-dT*eig*(t -
                    loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2) +
                    omega*exp(-dT*eig*(t - loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2
                    + omega**2))*loadmag[k])
        for k in within_ramp_loads[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += ((-dT*eig*cos(omega*t + phase)/(dT**2*eig**2 + omega**2) - omega*sin(omega*t + phase)/(dT**2*eig**2 +
                    omega**2))*loadmag[k + 1]*(loadtim[k + 1] - loadtim[k])**(-1)*loadtim[k] +
                    (dT*eig*cos(omega*t + phase)/(dT**2*eig**2 + omega**2) + omega*sin(omega*t +
                    phase)/(dT**2*eig**2 + omega**2))*loadmag[k] + (dT*eig*cos(omega*t +
                    phase)/(dT**2*eig**2 + omega**2) + omega*sin(omega*t + phase)/(dT**2*eig**2 +
                    omega**2))*loadmag[k]*(loadtim[k + 1] - loadtim[k])**(-1)*loadtim[k] +
                    (-dT**3*eig**3*t*cos(omega*t + phase)/(dT**4*eig**4 + 2*dT**2*eig**2*omega**2 +
                    omega**4) - dT**2*eig**2*omega*t*sin(omega*t + phase)/(dT**4*eig**4 +
                    2*dT**2*eig**2*omega**2 + omega**4) + dT**2*eig**2*cos(omega*t +
                    phase)/(dT**4*eig**4 + 2*dT**2*eig**2*omega**2 + omega**4) -
                    dT*eig*omega**2*t*cos(omega*t + phase)/(dT**4*eig**4 + 2*dT**2*eig**2*omega**2 +
                    omega**4) + 2*dT*eig*omega*sin(omega*t + phase)/(dT**4*eig**4 +
                    2*dT**2*eig**2*omega**2 + omega**4) - omega**3*t*sin(omega*t +
                    phase)/(dT**4*eig**4 + 2*dT**2*eig**2*omega**2 + omega**4) -
                    omega**2*cos(omega*t + phase)/(dT**4*eig**4 + 2*dT**2*eig**2*omega**2 +
                    omega**4))*loadmag[k]*(loadtim[k + 1] - loadtim[k])**(-1) +
                    (dT**3*eig**3*t*cos(omega*t + phase)/(dT**4*eig**4 + 2*dT**2*eig**2*omega**2 +
                    omega**4) + dT**2*eig**2*omega*t*sin(omega*t + phase)/(dT**4*eig**4 +
                    2*dT**2*eig**2*omega**2 + omega**4) - dT**2*eig**2*cos(omega*t +
                    phase)/(dT**4*eig**4 + 2*dT**2*eig**2*omega**2 + omega**4) +
                    dT*eig*omega**2*t*cos(omega*t + phase)/(dT**4*eig**4 + 2*dT**2*eig**2*omega**2 +
                    omega**4) - 2*dT*eig*omega*sin(omega*t + phase)/(dT**4*eig**4 +
                    2*dT**2*eig**2*omega**2 + omega**4) + omega**3*t*sin(omega*t +
                    phase)/(dT**4*eig**4 + 2*dT**2*eig**2*omega**2 + omega**4) +
                    omega**2*cos(omega*t + phase)/(dT**4*eig**4 + 2*dT**2*eig**2*omega**2 +
                    omega**4))*loadmag[k + 1]*(loadtim[k + 1] - loadtim[k])**(-1) -
                    (-dT*eig*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2) - omega*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2))*loadmag[k
                    + 1]*(loadtim[k + 1] - loadtim[k])**(-1)*loadtim[k] - (dT*eig*exp(-dT*eig*(t -
                    loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2) +
                    omega*exp(-dT*eig*(t - loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2
                    + omega**2))*loadmag[k] - (dT*eig*exp(-dT*eig*(t -
                    loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2) +
                    omega*exp(-dT*eig*(t - loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2
                    + omega**2))*loadmag[k]*(loadtim[k + 1] - loadtim[k])**(-1)*loadtim[k] -
                    (-dT**3*eig**3*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2)**2 -
                    dT**2*eig**2*omega*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*sin(omega*loadtim[k]
                    + phase)/(dT**2*eig**2 + omega**2)**2 + dT**2*eig**2*exp(-dT*eig*(t -
                    loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2)**2 -
                    dT*eig*omega**2*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2)**2 + 2*dT*eig*omega*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2)**2 -
                    omega**3*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*sin(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2)**2 - omega**2*exp(-dT*eig*(t -
                    loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2 +
                    omega**2)**2)*loadmag[k]*(loadtim[k + 1] - loadtim[k])**(-1) -
                    (dT**3*eig**3*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2)**2 +
                    dT**2*eig**2*omega*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*sin(omega*loadtim[k]
                    + phase)/(dT**2*eig**2 + omega**2)**2 - dT**2*eig**2*exp(-dT*eig*(t -
                    loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2)**2 +
                    dT*eig*omega**2*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2)**2 - 2*dT*eig*omega*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2)**2 +
                    omega**3*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*sin(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2)**2 + omega**2*exp(-dT*eig*(t -
                    loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2 +
                    omega**2)**2)*loadmag[k + 1]*(loadtim[k + 1] - loadtim[k])**(-1))
        for k in after_ramp_loads[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += ((-dT*eig*exp(-dT*eig*(t - loadtim[k + 1]))*cos(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 +
                    omega**2) - omega*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*loadtim[k + 1] +
                    phase)/(dT**2*eig**2 + omega**2))*loadmag[k + 1]*(loadtim[k + 1] -
                    loadtim[k])**(-1)*loadtim[k] + (dT*eig*exp(-dT*eig*(t - loadtim[k +
                    1]))*cos(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 + omega**2) +
                    omega*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*loadtim[k + 1] +
                    phase)/(dT**2*eig**2 + omega**2))*loadmag[k] + (dT*eig*exp(-dT*eig*(t -
                    loadtim[k + 1]))*cos(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 + omega**2) +
                    omega*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*loadtim[k + 1] +
                    phase)/(dT**2*eig**2 + omega**2))*loadmag[k]*(loadtim[k + 1] -
                    loadtim[k])**(-1)*loadtim[k] - (-dT*eig*exp(-dT*eig*(t -
                    loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2) -
                    omega*exp(-dT*eig*(t - loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2
                    + omega**2))*loadmag[k + 1]*(loadtim[k + 1] - loadtim[k])**(-1)*loadtim[k] -
                    (dT*eig*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2) + omega*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2))*loadmag[k]
                    - (dT*eig*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2) + omega*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 +
                    omega**2))*loadmag[k]*(loadtim[k + 1] - loadtim[k])**(-1)*loadtim[k] +
                    (-dT**3*eig**3*loadtim[k + 1]*exp(-dT*eig*(t - loadtim[k +
                    1]))*cos(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 + omega**2)**2 -
                    dT**2*eig**2*omega*loadtim[k + 1]*exp(-dT*eig*(t - loadtim[k +
                    1]))*sin(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 + omega**2)**2 +
                    dT**2*eig**2*exp(-dT*eig*(t - loadtim[k + 1]))*cos(omega*loadtim[k + 1] +
                    phase)/(dT**2*eig**2 + omega**2)**2 - dT*eig*omega**2*loadtim[k +
                    1]*exp(-dT*eig*(t - loadtim[k + 1]))*cos(omega*loadtim[k + 1] +
                    phase)/(dT**2*eig**2 + omega**2)**2 + 2*dT*eig*omega*exp(-dT*eig*(t - loadtim[k
                    + 1]))*sin(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 + omega**2)**2 -
                    omega**3*loadtim[k + 1]*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*loadtim[k +
                    1] + phase)/(dT**2*eig**2 + omega**2)**2 - omega**2*exp(-dT*eig*(t - loadtim[k +
                    1]))*cos(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 +
                    omega**2)**2)*loadmag[k]*(loadtim[k + 1] - loadtim[k])**(-1) +
                    (dT**3*eig**3*loadtim[k + 1]*exp(-dT*eig*(t - loadtim[k +
                    1]))*cos(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 + omega**2)**2 +
                    dT**2*eig**2*omega*loadtim[k + 1]*exp(-dT*eig*(t - loadtim[k +
                    1]))*sin(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 + omega**2)**2 -
                    dT**2*eig**2*exp(-dT*eig*(t - loadtim[k + 1]))*cos(omega*loadtim[k + 1] +
                    phase)/(dT**2*eig**2 + omega**2)**2 + dT*eig*omega**2*loadtim[k +
                    1]*exp(-dT*eig*(t - loadtim[k + 1]))*cos(omega*loadtim[k + 1] +
                    phase)/(dT**2*eig**2 + omega**2)**2 - 2*dT*eig*omega*exp(-dT*eig*(t - loadtim[k
                    + 1]))*sin(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 + omega**2)**2 +
                    omega**3*loadtim[k + 1]*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*loadtim[k +
                    1] + phase)/(dT**2*eig**2 + omega**2)**2 + omega**2*exp(-dT*eig*(t - loadtim[k +
                    1]))*cos(omega*loadtim[k + 1] + phase)/(dT**2*eig**2 + omega**2)**2)*loadmag[k +
                    1]*(loadtim[k + 1] - loadtim[k])**(-1) -
                    (-dT**3*eig**3*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2)**2 -
                    dT**2*eig**2*omega*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*sin(omega*loadtim[k]
                    + phase)/(dT**2*eig**2 + omega**2)**2 + dT**2*eig**2*exp(-dT*eig*(t -
                    loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2)**2 -
                    dT*eig*omega**2*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2)**2 + 2*dT*eig*omega*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2)**2 -
                    omega**3*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*sin(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2)**2 - omega**2*exp(-dT*eig*(t -
                    loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2 +
                    omega**2)**2)*loadmag[k]*(loadtim[k + 1] - loadtim[k])**(-1) -
                    (dT**3*eig**3*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2)**2 +
                    dT**2*eig**2*omega*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*sin(omega*loadtim[k]
                    + phase)/(dT**2*eig**2 + omega**2)**2 - dT**2*eig**2*exp(-dT*eig*(t -
                    loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2)**2 +
                    dT*eig*omega**2*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*cos(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2)**2 - 2*dT*eig*omega*exp(-dT*eig*(t -
                    loadtim[k]))*sin(omega*loadtim[k] + phase)/(dT**2*eig**2 + omega**2)**2 +
                    omega**3*loadtim[k]*exp(-dT*eig*(t - loadtim[k]))*sin(omega*loadtim[k] +
                    phase)/(dT**2*eig**2 + omega**2)**2 + omega**2*exp(-dT*eig*(t -
                    loadtim[k]))*cos(omega*loadtim[k] + phase)/(dT**2*eig**2 +
                    omega**2)**2)*loadmag[k + 1]*(loadtim[k + 1] - loadtim[k])**(-1))
    return A


def dim1sin(m, z):
    """calc sin(m*z) for each combination of m and z

    calculates array A[len(z), len(m)]

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    z : ``list`` of ``float``
        normalised depth or z-coordinate should be between 0 and 1.

    Returns
    -------
    A : numpy.ndarray
        returns an array  1d array size A[len(z), len(m)]

    See Also
    --------
    m_from_sin_mx : used to generate 'm' input parameter

    Notes
    -----
    TODO:

    """

    m = np.asarray(m)
    z = np.asarray(z)

    return np.sin(z[:, np.newaxis]*m[np.newaxis, :])


def dim1sin_avg_between(m, z):
    """calc integrate(sin(m*z), (z, z1, z2))/(z2-z1) for each combination of m and [z1,z2]

    calculates array A[len(z), len(m)]

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    z : ``list`` of ``list`` of float``
        normalised depth or z-coordinate pair should be two values
        between 0 and 1.

    Returns
    -------
    A : numpy.ndarray
        returns an array  1d array size A[len(z), len(m)]

    See Also
    --------
    m_from_sin_mx : used to generate 'm' input parameter

    Notes
    -----
    TODO:

    """

    from numpy import sin, cos
    m = np.asarray(m)
    z = np.asarray(z)

    a = z[:,0][:,np.newaxis]
    b = z[:,1][:,np.newaxis]
    mm = m[np.newaxis,:]

    return (cos(a*mm)-cos(b*mm))/mm/(b-a)

def pdim1sin_a_linear_between(m, a, z, **kwargs):
    """wrapper for dim1sin_a_linear_between with PolyLine input

    See also
    --------
    dim1sin_a_linear_between

    """

    return dim1sin_a_linear_between(m, a.y1, a.y2, a.x1, a.x2, z, **kwargs)

def dim1sin_a_linear_between(m, at, ab, zt, zb, z):
    """calc integrate(a(z) * sin(m*z), (z, z1, z2)) for each combination of m and [z1,z2]


    Performs integrations of `sin(mi * z) * a(z) *b(z)`
    between [z1, z2] where a(z) is a piecewise linear functions of z.
    calculates array A[len(z), len(m)]

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    at : ``list`` of ``float``
        property at top of each layer
    ab : ``list`` of ``float``
        property at bottom of each layer
    zt : ``list`` of ``float``
        normalised depth or z-coordinate at top of each layer. `zt[0]` = 0
    zb : ``list`` of ``float``
        normalised depth or z-coordinate at bottom of each layer. `zt[-1]` = 1
    z : ``list`` of ``list`` of float``
        normalised depth or z-coordinate pair should be two values
        between 0 and 1.

    Returns
    -------
    A : numpy.ndarray
        returns an array  1d array size A[len(z), len(m)]

    See Also
    --------
    m_from_sin_mx : used to generate 'm' input parameter


    Notes
    -----
    TODO:

    """


    from math import sin, cos
    m = np.asarray(m)

    z = np.atleast_2d(z)

    z1 = z[:,0]
    z2 = z[:,1]

    z_for_interp = np.zeros(len(zt)+1)
    z_for_interp[:-1] = zt[:]
    z_for_interp[-1]=zb[-1]


    (segment_both, segment_z1_only, segment_z2_only, segments_between) = segments_between_xi_and_xj(z_for_interp,z1,z2)

    nz = len(z)
    neig = len(m)

    A = np.zeros([nz,neig])
    for i in range(nz):
        for layer in segment_both[i]:
            for j in range(neig):
                A[i,j] += (-(zb[layer]*m[j]**2 - zt[layer]*m[j]**2)**(-1)*ab[layer]*sin(m[j]*z1[i]) + (zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*ab[layer]*sin(m[j]*z2[i]) + (zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*at[layer]*sin(m[j]*z1[i]) - (zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*at[layer]*sin(m[j]*z2[i]) + m[j]*(zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*ab[layer]*z1[i]*cos(m[j]*z1[i]) - m[j]*(zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*ab[layer]*z2[i]*cos(m[j]*z2[i]) - m[j]*(zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*at[layer]*z1[i]*cos(m[j]*z1[i]) + m[j]*(zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*at[layer]*z2[i]*cos(m[j]*z2[i]) + zb[layer]*m[j]*(zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*at[layer]*cos(m[j]*z1[i]) - zb[layer]*m[j]*(zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*at[layer]*cos(m[j]*z2[i]) - zt[layer]*m[j]*(zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*ab[layer]*cos(m[j]*z1[i]) + zt[layer]*m[j]*(zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*ab[layer]*cos(m[j]*z2[i]))
        for layer in segment_z1_only[i]:
            for j in range(neig):
                A[i,j] += (-(zb[layer]*m[j]**2 - zt[layer]*m[j]**2)**(-1)*ab[layer]*sin(m[j]*z1[i]) + (zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*ab[layer]*sin(m[j]*zb[layer]) + (zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*at[layer]*sin(m[j]*z1[i]) - (zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*at[layer]*sin(m[j]*zb[layer]) + m[j]*(zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*ab[layer]*z1[i]*cos(m[j]*z1[i]) - m[j]*(zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*ab[layer]*zb[layer]*cos(m[j]*zb[layer]) - m[j]*(zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*at[layer]*z1[i]*cos(m[j]*z1[i]) + m[j]*(zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*at[layer]*zb[layer]*cos(m[j]*zb[layer]) +
                    zb[layer]*m[j]*(zb[layer]*m[j]**2 - zt[layer]*m[j]**2)**(-1)*at[layer]*cos(m[j]*z1[i]) -
                    zb[layer]*m[j]*(zb[layer]*m[j]**2 - zt[layer]*m[j]**2)**(-1)*at[layer]*cos(m[j]*zb[layer]) -
                    zt[layer]*m[j]*(zb[layer]*m[j]**2 - zt[layer]*m[j]**2)**(-1)*ab[layer]*cos(m[j]*z1[i]) +
                    zt[layer]*m[j]*(zb[layer]*m[j]**2 - zt[layer]*m[j]**2)**(-1)*ab[layer]*cos(m[j]*zb[layer]))
        for layer in segments_between[i]:
            for j in range(neig):
                A[i,j] += ((zb[layer]*m[j]**2 - zt[layer]*m[j]**2)**(-1)*ab[layer]*sin(m[j]*zb[layer]) - (zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*ab[layer]*sin(m[j]*zt[layer]) - (zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*at[layer]*sin(m[j]*zb[layer]) + (zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*at[layer]*sin(m[j]*zt[layer]) - m[j]*(zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*ab[layer]*zb[layer]*cos(m[j]*zb[layer]) + m[j]*(zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*ab[layer]*zt[layer]*cos(m[j]*zt[layer]) + m[j]*(zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*at[layer]*zb[layer]*cos(m[j]*zb[layer]) - m[j]*(zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*at[layer]*zt[layer]*cos(m[j]*zt[layer]) -
                    zb[layer]*m[j]*(zb[layer]*m[j]**2 - zt[layer]*m[j]**2)**(-1)*at[layer]*cos(m[j]*zb[layer]) +
                    zb[layer]*m[j]*(zb[layer]*m[j]**2 - zt[layer]*m[j]**2)**(-1)*at[layer]*cos(m[j]*zt[layer]) +
                    zt[layer]*m[j]*(zb[layer]*m[j]**2 - zt[layer]*m[j]**2)**(-1)*ab[layer]*cos(m[j]*zb[layer]) -
                    zt[layer]*m[j]*(zb[layer]*m[j]**2 - zt[layer]*m[j]**2)**(-1)*ab[layer]*cos(m[j]*zt[layer]))
        for layer in segment_z2_only[i]:
            for j in range(neig):
                A[i,j] += ((zb[layer]*m[j]**2 - zt[layer]*m[j]**2)**(-1)*ab[layer]*sin(m[j]*z2[i]) - (zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*ab[layer]*sin(m[j]*zt[layer]) - (zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*at[layer]*sin(m[j]*z2[i]) + (zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*at[layer]*sin(m[j]*zt[layer]) - m[j]*(zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*ab[layer]*z2[i]*cos(m[j]*z2[i]) + m[j]*(zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*ab[layer]*zt[layer]*cos(m[j]*zt[layer]) + m[j]*(zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*at[layer]*z2[i]*cos(m[j]*z2[i]) - m[j]*(zb[layer]*m[j]**2 -
                    zt[layer]*m[j]**2)**(-1)*at[layer]*zt[layer]*cos(m[j]*zt[layer]) -
                    zb[layer]*m[j]*(zb[layer]*m[j]**2 - zt[layer]*m[j]**2)**(-1)*at[layer]*cos(m[j]*z2[i]) +
                    zb[layer]*m[j]*(zb[layer]*m[j]**2 - zt[layer]*m[j]**2)**(-1)*at[layer]*cos(m[j]*zt[layer]) +
                    zt[layer]*m[j]*(zb[layer]*m[j]**2 - zt[layer]*m[j]**2)**(-1)*ab[layer]*cos(m[j]*z2[i]) -
                    zt[layer]*m[j]*(zb[layer]*m[j]**2 - zt[layer]*m[j]**2)**(-1)*ab[layer]*cos(m[j]*zt[layer]))
    return A

def dim1sin_af_linear(m, at, ab, zt, zb, implementation='vectorized'):
    """Create matrix of spectral integrations

    Performs integrations of `sin(mi * z) * a(z) * sin(mj * z)` between [0, 1]
    where a(z) is a piecewise linear function of z.  Calulation of integrals
    is performed at each element of a square symmetric matrix (size depends
    on size of `m`)

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    at : ``list`` of ``float``
        property at top of each layer
    ab : ``list`` of ``float``
        property at bottom of each layer
    zt : ``list`` of ``float``
        normalised depth or z-coordinate at top of each layer. `zt[0]` = 0
    zb : ``list`` of ``float``
        normalised depth or z-coordinate at bottom of each layer. `zt[-1]` = 1
    implementation: str, optional
        functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        default = 'vectorized'.  If fortran extention module cannot be imported
        then 'vectorized' version will be used.  If anything other than
        'fortran' or 'scalar' is used then default vectorized version will be
        used

    Returns
    -------
    A : numpy.ndarray
        returns a square symmetric matrix, size determined by size of `m`

    See Also
    --------
    m_from_sin_mx : used to generate 'm' input parameter
    geotecha.integrals_generate_code.dim1sin_af_linear : use sympy
    to perform the integrals symbolically and generate expressions for
    this function

    Notes
    -----
    The `dim1sin_af_linear` matrix, :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{{a\\left(z\\right)}{sin\\left({m_j}z\\right)}{sin\\left({m_i}z\\right)}\,dz}

    where :math:`a\\left(z\\right)` in one layer is given by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    """
    #import numpy as np
    #import math
    #import geotecha.speccon.ext_integrals as ext_integ


    m = np.asarray(m)
    at = np.asarray(at)
    ab = np.asarray(ab)
    zt = np.asarray(zt)
    zb = np.asarray(zb)

    neig = len(m)

    if implementation == 'scalar':
        sin = math.sin
        cos = math.cos
        A = np.zeros([neig, neig], float)
        nlayers = len(zt)
        for layer in range(nlayers):
            for i in range(neig):
                A[i, i] += (-(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*ab[layer]*cos(m[i]*zb[layer])**2 +
                    (4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*cos(m[i]*zt[layer])**2 +
                    (4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*cos(m[i]*zb[layer])**2 -
                    (4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*cos(m[i]*zt[layer])**2 -
                    2*m[i]*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                    + 2*m[i]*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                    + 2*m[i]*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                    - 2*m[i]*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                    + m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]**2*sin(m[i]*zb[layer])**2 +
                    m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])**2 -
                    m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]**2*sin(m[i]*zt[layer])**2 -
                    m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])**2 -
                    m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]**2*sin(m[i]*zb[layer])**2 -
                    m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])**2 +
                    m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]**2*sin(m[i]*zt[layer])**2 +
                    m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])**2 -
                    2*zb[layer]*m[i]*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) +
                    2*zb[layer]*m[i]*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) +
                    2*zb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]*sin(m[i]*zb[layer])**2 +
                    2*zb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]*cos(m[i]*zb[layer])**2 -
                    2*zb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]*sin(m[i]*zt[layer])**2 -
                    2*zb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]*cos(m[i]*zt[layer])**2 +
                    2*zt[layer]*m[i]*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) -
                    2*zt[layer]*m[i]*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) -
                    2*zt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]*sin(m[i]*zb[layer])**2 -
                    2*zt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]*cos(m[i]*zb[layer])**2 +
                    2*zt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]*sin(m[i]*zt[layer])**2 +
                    2*zt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]*cos(m[i]*zt[layer])**2)
            for i in range(neig-1):
                for j in range(i + 1, neig):
                    A[i, j] += (m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 +
                        2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4
                        - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4
                        - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4
                        - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4
                        - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) -
                        2*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) -
                        2*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) +
                        2*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) +
                        m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4
                        - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4
                        - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4
                        - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - zb[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        zb[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        zb[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                        zb[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                        zb[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        zb[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        zb[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                        zb[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                        zt[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        zt[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        zt[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                        zt[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) -
                        zt[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        zt[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        zt[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                        zt[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]))

        #A is symmetric
        for i in range(neig - 1):
            for j in range(i + 1, neig):
                A[j, i] = A[i, j]

    elif implementation == 'fortran':
        try:
            import geotecha.speccon.ext_integrals as ext_integ
            A = ext_integ.dim1sin_af_linear(m, at, ab, zt, zb)
        except ImportError:
            A = dim1sin_af_linear(m, at, ab, zt, zb, implementation='vectorized')

    else:#default is 'vectorized' using numpy
        sin = np.sin
        cos = np.cos
        A = np.zeros([neig, neig], float)

        diag =  np.diag_indices(neig)
        triu = np.triu_indices(neig, k = 1)
        tril = (triu[1], triu[0])

        mi = m[:, np.newaxis]
        A[diag] = np.sum(ab*mi**2*zb**2*sin(mi*zb)**2/(4*mi**2*zb - 4*mi**2*zt) + ab*mi**2*zb**2*cos(mi*zb)**2/(4*mi**2*zb -
            4*mi**2*zt) - 2*ab*mi**2*zb*zt*sin(mi*zb)**2/(4*mi**2*zb - 4*mi**2*zt) -
            2*ab*mi**2*zb*zt*cos(mi*zb)**2/(4*mi**2*zb - 4*mi**2*zt) +
            ab*mi**2*zt**2*sin(mi*zt)**2/(4*mi**2*zb - 4*mi**2*zt) +
            ab*mi**2*zt**2*cos(mi*zt)**2/(4*mi**2*zb - 4*mi**2*zt) -
            2*ab*mi*zb*sin(mi*zb)*cos(mi*zb)/(4*mi**2*zb - 4*mi**2*zt) +
            2*ab*mi*zt*sin(mi*zb)*cos(mi*zb)/(4*mi**2*zb - 4*mi**2*zt) -
            ab*cos(mi*zb)**2/(4*mi**2*zb - 4*mi**2*zt) + ab*cos(mi*zt)**2/(4*mi**2*zb - 4*mi**2*zt)
            + at*mi**2*zb**2*sin(mi*zb)**2/(4*mi**2*zb - 4*mi**2*zt) +
            at*mi**2*zb**2*cos(mi*zb)**2/(4*mi**2*zb - 4*mi**2*zt) -
            2*at*mi**2*zb*zt*sin(mi*zt)**2/(4*mi**2*zb - 4*mi**2*zt) -
            2*at*mi**2*zb*zt*cos(mi*zt)**2/(4*mi**2*zb - 4*mi**2*zt) +
            at*mi**2*zt**2*sin(mi*zt)**2/(4*mi**2*zb - 4*mi**2*zt) +
            at*mi**2*zt**2*cos(mi*zt)**2/(4*mi**2*zb - 4*mi**2*zt) +
            2*at*mi*zb*sin(mi*zt)*cos(mi*zt)/(4*mi**2*zb - 4*mi**2*zt) -
            2*at*mi*zt*sin(mi*zt)*cos(mi*zt)/(4*mi**2*zb - 4*mi**2*zt) +
            at*cos(mi*zb)**2/(4*mi**2*zb - 4*mi**2*zt) - at*cos(mi*zt)**2/(4*mi**2*zb - 4*mi**2*zt), axis=1)

        mi = m[triu[0]][:, np.newaxis]
        mj = m[triu[1]][:, np.newaxis]
        A[triu] = np.sum(-ab*mi**3*zb*sin(mj*zb)*cos(mi*zb)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt +
            mj**4*zb - mj**4*zt) + ab*mi**3*zt*sin(mj*zb)*cos(mi*zb)/(mi**4*zb - mi**4*zt -
            2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) +
            ab*mi**2*mj*zb*sin(mi*zb)*cos(mj*zb)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) - ab*mi**2*mj*zt*sin(mi*zb)*cos(mj*zb)/(mi**4*zb
            - mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) +
            ab*mi**2*sin(mi*zb)*sin(mj*zb)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) - ab*mi**2*sin(mi*zt)*sin(mj*zt)/(mi**4*zb -
            mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) +
            ab*mi*mj**2*zb*sin(mj*zb)*cos(mi*zb)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) - ab*mi*mj**2*zt*sin(mj*zb)*cos(mi*zb)/(mi**4*zb
            - mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) +
            2*ab*mi*mj*cos(mi*zb)*cos(mj*zb)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) - 2*ab*mi*mj*cos(mi*zt)*cos(mj*zt)/(mi**4*zb -
            mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) -
            ab*mj**3*zb*sin(mi*zb)*cos(mj*zb)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) + ab*mj**3*zt*sin(mi*zb)*cos(mj*zb)/(mi**4*zb -
            mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) +
            ab*mj**2*sin(mi*zb)*sin(mj*zb)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) - ab*mj**2*sin(mi*zt)*sin(mj*zt)/(mi**4*zb -
            mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) +
            at*mi**3*zb*sin(mj*zt)*cos(mi*zt)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) - at*mi**3*zt*sin(mj*zt)*cos(mi*zt)/(mi**4*zb -
            mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) -
            at*mi**2*mj*zb*sin(mi*zt)*cos(mj*zt)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) + at*mi**2*mj*zt*sin(mi*zt)*cos(mj*zt)/(mi**4*zb
            - mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) -
            at*mi**2*sin(mi*zb)*sin(mj*zb)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) + at*mi**2*sin(mi*zt)*sin(mj*zt)/(mi**4*zb -
            mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) -
            at*mi*mj**2*zb*sin(mj*zt)*cos(mi*zt)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) + at*mi*mj**2*zt*sin(mj*zt)*cos(mi*zt)/(mi**4*zb
            - mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) -
            2*at*mi*mj*cos(mi*zb)*cos(mj*zb)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) + 2*at*mi*mj*cos(mi*zt)*cos(mj*zt)/(mi**4*zb -
            mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) +
            at*mj**3*zb*sin(mi*zt)*cos(mj*zt)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) - at*mj**3*zt*sin(mi*zt)*cos(mj*zt)/(mi**4*zb -
            mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) -
            at*mj**2*sin(mi*zb)*sin(mj*zb)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) + at*mj**2*sin(mi*zt)*sin(mj*zt)/(mi**4*zb -
            mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt), axis=1)
        #A is symmetric
        A[tril] = A[triu]

    return A

def dim1sin_abf_linear(m, at, ab, bt, bb,  zt, zb, implementation='vectorized'):
    """Create matrix of spectral integrations

    Performs integrations of `sin(mi * z) * a(z) *b(z) * sin(mj * z)`
    between [0, 1] where a(z) and b(z) are piecewise linear functions of z.
    Calulation of integrals is performed at each element of a square symmetric
    matrix (size depends on size of `m`)

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    at : ``list`` of ``float``
        property at top of each layer
    ab : ``list`` of ``float``
        property at bottom of each layer
    bt : ``list`` of ``float``
        2nd property at top of each layer
    bb : ``list`` of ``float``
        2nd property at bottom of each layer
    zt : ``list`` of ``float``
        normalised depth or z-coordinate at top of each layer. `zt[0]` = 0
    zb : ``list`` of ``float``
        normalised depth or z-coordinate at bottom of each layer. `zt[-1]` = 1
    implementation: str, optional
        functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        default = 'vectorized'.  If fortran extention module cannot be imported
        then 'vectorized' version will be used.  If anything other than
        'fortran' or 'scalar' is used then default vectorized version will be
        used

    Returns
    -------
    A : numpy.ndarray
        returns a square symmetric matrix, size determined by size of `m`

    See Also
    --------
    m_from_sin_mx : used to generate 'm' input parameter
    geotecha.integrals_generate_code.dim1sin_abf_linear : use sympy
    to perform the integrals symbolically and generate expressions for
    this function

    Notes
    -----
    The `dim1sin_abf_linear` matrix, :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{{a\\left(z\\right)}{b\\left(z\\right)}\\phi_i\\phi_j\\,dz}

    where the basis function :math:`\\phi_i` is given by:


    ..math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` and :math:`b\\left(z\\right)` are piecewise
    linear functions w.r.t. :mathL`z`, that within a layer are defined by:

    ..math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """

    #import numpy as np #import this at module level
    #import math #import this at module level

    m = np.asarray(m)
    at = np.asarray(at)
    ab = np.asarray(ab)
    bt = np.asarray(bt)
    bb = np.asarray(bb)
    zt = np.asarray(zt)
    zb = np.asarray(zb)

    neig = len(m)

    if implementation == 'scalar':
        sin = math.sin
        cos = math.cos
        A = np.zeros([neig, neig], float)
        nlayers = len(zt)
        for layer in range(nlayers):
            for i in range(neig):
                A[i, i] += (3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                    - 3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                    - 3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                    + 3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                    - 3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                    + 3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                    + 3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                    - 3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                    + 3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer])**2
                    - 3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])**2
                    - 3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer])**2
                    + 3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])**2
                    - 3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer])**2
                    + 3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])**2
                    + 3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer])**2
                    - 3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])**2
                    - 3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer])**2
                    + 3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])**2
                    + 3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer])**2
                    - 3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])**2
                    + 3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer])**2
                    - 3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])**2
                    - 3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer])**2
                    + 3*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])**2
                    - 6*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                    + 6*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                    + 6*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                    - 6*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                    + 6*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                    - 6*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                    - 6*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                    + 6*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                    + 2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]**3*sin(m[i]*zb[layer])**2
                    + 2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]**3*cos(m[i]*zb[layer])**2
                    - 2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]**3*sin(m[i]*zt[layer])**2
                    - 2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]**3*cos(m[i]*zt[layer])**2
                    - 2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]**3*sin(m[i]*zb[layer])**2
                    - 2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]**3*cos(m[i]*zb[layer])**2
                    + 2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]**3*sin(m[i]*zt[layer])**2
                    + 2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]**3*cos(m[i]*zt[layer])**2
                    - 2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]**3*sin(m[i]*zb[layer])**2
                    - 2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]**3*cos(m[i]*zb[layer])**2
                    + 2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]**3*sin(m[i]*zt[layer])**2
                    + 2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]**3*cos(m[i]*zt[layer])**2
                    + 2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]**3*sin(m[i]*zb[layer])**2
                    + 2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]**3*cos(m[i]*zb[layer])**2
                    - 2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]**3*sin(m[i]*zt[layer])**2
                    - 2*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]**3*cos(m[i]*zt[layer])**2
                    - 3*zb[layer]*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])**2 +
                    3*zb[layer]*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])**2 -
                    3*zb[layer]*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])**2 +
                    3*zb[layer]*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])**2 +
                    6*zb[layer]*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*cos(m[i]*zb[layer])**2 -
                    6*zb[layer]*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*cos(m[i]*zt[layer])**2 -
                    6*zb[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                    + 6*zb[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                    - 6*zb[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                    + 6*zb[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                    + 12*zb[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                    - 12*zb[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                    + 3*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]**2*sin(m[i]*zb[layer])**2
                    + 3*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])**2
                    - 3*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]**2*sin(m[i]*zt[layer])**2
                    - 3*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])**2
                    + 3*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]**2*sin(m[i]*zb[layer])**2
                    + 3*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])**2
                    - 3*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]**2*sin(m[i]*zt[layer])**2
                    - 3*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])**2
                    - 6*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]**2*sin(m[i]*zb[layer])**2
                    - 6*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])**2
                    + 6*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]**2*sin(m[i]*zt[layer])**2
                    + 6*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])**2
                    - 6*zb[layer]**2*m[i]**2*(12*zb[layer]**2*m[i]**3 -
                    24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                    + 6*zb[layer]**2*m[i]**2*(12*zb[layer]**2*m[i]**3 -
                    24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                    + 6*zb[layer]**2*m[i]**3*(12*zb[layer]**2*m[i]**3 -
                    24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer])**2
                    + 6*zb[layer]**2*m[i]**3*(12*zb[layer]**2*m[i]**3 -
                    24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])**2
                    - 6*zb[layer]**2*m[i]**3*(12*zb[layer]**2*m[i]**3 -
                    24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer])**2
                    - 6*zb[layer]**2*m[i]**3*(12*zb[layer]**2*m[i]**3 -
                    24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])**2
                    + 6*zt[layer]*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zb[layer])**2 -
                    6*zt[layer]*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zt[layer])**2 -
                    3*zt[layer]*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])**2 +
                    3*zt[layer]*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])**2 -
                    3*zt[layer]*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])**2 +
                    3*zt[layer]*m[i]*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])**2 +
                    12*zt[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                    - 12*zt[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                    - 6*zt[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                    + 6*zt[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                    - 6*zt[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                    + 6*zt[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                    - 6*zt[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]**2*sin(m[i]*zb[layer])**2
                    - 6*zt[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])**2
                    + 6*zt[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]**2*sin(m[i]*zt[layer])**2
                    + 6*zt[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])**2
                    + 3*zt[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]**2*sin(m[i]*zb[layer])**2
                    + 3*zt[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])**2
                    - 3*zt[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]**2*sin(m[i]*zt[layer])**2
                    - 3*zt[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])**2
                    + 3*zt[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]**2*sin(m[i]*zb[layer])**2
                    + 3*zt[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])**2
                    - 3*zt[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]**2*sin(m[i]*zt[layer])**2
                    - 3*zt[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 - 24*zt[layer]*zb[layer]*m[i]**3
                    +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])**2
                    + 6*zt[layer]*zb[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 -
                    24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                    - 6*zt[layer]*zb[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 -
                    24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                    + 6*zt[layer]*zb[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 -
                    24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                    - 6*zt[layer]*zb[layer]*m[i]**2*(12*zb[layer]**2*m[i]**3 -
                    24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                    - 6*zt[layer]*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 -
                    24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer])**2
                    - 6*zt[layer]*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 -
                    24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])**2
                    + 6*zt[layer]*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 -
                    24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer])**2
                    + 6*zt[layer]*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 -
                    24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])**2
                    - 6*zt[layer]*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 -
                    24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer])**2
                    - 6*zt[layer]*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 -
                    24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])**2
                    + 6*zt[layer]*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 -
                    24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer])**2
                    + 6*zt[layer]*zb[layer]*m[i]**3*(12*zb[layer]**2*m[i]**3 -
                    24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])**2
                    - 6*zt[layer]**2*m[i]**2*(12*zb[layer]**2*m[i]**3 -
                    24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                    + 6*zt[layer]**2*m[i]**2*(12*zb[layer]**2*m[i]**3 -
                    24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                    + 6*zt[layer]**2*m[i]**3*(12*zb[layer]**2*m[i]**3 -
                    24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer])**2
                    + 6*zt[layer]**2*m[i]**3*(12*zb[layer]**2*m[i]**3 -
                    24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])**2
                    - 6*zt[layer]**2*m[i]**3*(12*zb[layer]**2*m[i]**3 -
                    24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer])**2
                    - 6*zt[layer]**2*m[i]**3*(12*zb[layer]**2*m[i]**3 -
                    24*zt[layer]*zb[layer]*m[i]**3 +
                    12*zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])**2)
            for i in range(neig-1):
                for j in range(i + 1, neig):
                    A[i, j] += (2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 2*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 2*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 2*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 2*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 6*m[j]*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 6*m[j]*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + 6*m[j]*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - 6*m[j]*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + 6*m[j]*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - 6*m[j]*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - 6*m[j]*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 6*m[j]*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + 4*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        - 4*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        - 4*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        + 4*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        - 4*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        + 4*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        + 4*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        - 4*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        + m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]**2*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]**2*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]**2*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - m[j]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]**2*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + 6*m[j]**2*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 6*m[j]**2*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 6*m[j]**2*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 6*m[j]**2*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 6*m[j]**2*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 6*m[j]**2*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 6*m[j]**2*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 6*m[j]**2*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 2*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 2*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 2*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 2*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 2*m[j]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 2*m[j]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + 2*m[j]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - 2*m[j]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + 2*m[j]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - 2*m[j]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - 2*m[j]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 2*m[j]**3*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - 4*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        + 4*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        + 4*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        - 4*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        + 4*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        - 4*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        - 4*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        + 4*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        - 2*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 2*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]**2*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + 2*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - 2*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]**2*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + 2*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - 2*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]**2*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - 2*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 2*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]**2*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - 2*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 2*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 2*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 2*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + m[j]**4*m[i]*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]**2*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]**2*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]**2*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]**2*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + zb[layer]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - zb[layer]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + zb[layer]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - zb[layer]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 2*zb[layer]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4
                        + 3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 2*zb[layer]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4
                        + 3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - zb[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + zb[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - zb[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + zb[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*zb[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4
                        + 3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*zb[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4
                        + 3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*zb[layer]*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        - 2*zb[layer]*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        + 2*zb[layer]*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        - 2*zb[layer]*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        - 4*zb[layer]*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        + 4*zb[layer]*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        + zb[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - zb[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + zb[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - zb[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - 2*zb[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 2*zb[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + 2*zb[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*zb[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*zb[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*zb[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 4*zb[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 4*zb[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 2*zb[layer]*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        + 2*zb[layer]*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        - 2*zb[layer]*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        + 2*zb[layer]*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        + 4*zb[layer]*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        - 4*zb[layer]*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        - 2*zb[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 2*zb[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - 2*zb[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 2*zb[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + 4*zb[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - 4*zb[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - zb[layer]*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + zb[layer]*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - zb[layer]*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + zb[layer]*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*zb[layer]*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4
                        + 3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*zb[layer]*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4
                        + 3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - zb[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + zb[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - zb[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + zb[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*zb[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*zb[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + zb[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - zb[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + zb[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - zb[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - 2*zb[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4
                        + 3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 2*zb[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4
                        + 3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - zb[layer]**2*m[i]**5*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + zb[layer]**2*m[i]**5*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + zb[layer]**2*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - zb[layer]**2*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + 2*zb[layer]**2*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*zb[layer]**2*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 2*zb[layer]**2*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 2*zb[layer]**2*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - zb[layer]**2*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + zb[layer]**2*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + zb[layer]**2*m[j]**5*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - zb[layer]**2*m[j]**5*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - 2*zt[layer]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4
                        + 3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 2*zt[layer]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4
                        + 3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + zt[layer]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - zt[layer]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + zt[layer]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - zt[layer]*m[i]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*zt[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4
                        + 3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*zt[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4
                        + 3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - zt[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + zt[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - zt[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + zt[layer]*m[i]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 4*zt[layer]*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        + 4*zt[layer]*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        + 2*zt[layer]*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        - 2*zt[layer]*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        + 2*zt[layer]*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        - 2*zt[layer]*m[j]*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        - 2*zt[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 2*zt[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + zt[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - zt[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + zt[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - zt[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - 4*zt[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 4*zt[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*zt[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*zt[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*zt[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*zt[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 4*zt[layer]*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        - 4*zt[layer]*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        - 2*zt[layer]*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        + 2*zt[layer]*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        - 2*zt[layer]*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])
                        + 2*zt[layer]*m[j]**3*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])
                        + 4*zt[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - 4*zt[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - 2*zt[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 2*zt[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - 2*zt[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 2*zt[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + 2*zt[layer]*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4
                        + 3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*zt[layer]*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4
                        + 3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - zt[layer]*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + zt[layer]*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - zt[layer]*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + zt[layer]*m[j]**4*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*zt[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*zt[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - zt[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + zt[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - zt[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + zt[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 2*zt[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4
                        + 3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 2*zt[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4
                        + 3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + zt[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - zt[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + zt[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - zt[layer]*m[j]**5*(zb[layer]**2*m[i]**6 - 3*zb[layer]**2*m[j]**2*m[i]**4 +
                        3*zb[layer]**2*m[j]**4*m[i]**2 - zb[layer]**2*m[j]**6 -
                        2*zt[layer]*zb[layer]*m[i]**6 + 6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + zt[layer]*zb[layer]*m[i]**5*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - zt[layer]*zb[layer]*m[i]**5*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + zt[layer]*zb[layer]*m[i]**5*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - zt[layer]*zb[layer]*m[i]**5*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - zt[layer]*zb[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + zt[layer]*zb[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - zt[layer]*zb[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + zt[layer]*zb[layer]*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - 2*zt[layer]*zb[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 2*zt[layer]*zb[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 2*zt[layer]*zb[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 2*zt[layer]*zb[layer]*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*zt[layer]*zb[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - 2*zt[layer]*zb[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + 2*zt[layer]*zb[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - 2*zt[layer]*zb[layer]*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + zt[layer]*zb[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - zt[layer]*zb[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + zt[layer]*zb[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - zt[layer]*zb[layer]*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - zt[layer]*zb[layer]*m[j]**5*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + zt[layer]*zb[layer]*m[j]**5*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - zt[layer]*zb[layer]*m[j]**5*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + zt[layer]*zb[layer]*m[j]**5*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bt[layer]*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - zt[layer]**2*m[i]**5*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + zt[layer]**2*m[i]**5*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + zt[layer]**2*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - zt[layer]**2*m[j]*m[i]**4*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + 2*zt[layer]**2*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - 2*zt[layer]**2*m[j]**2*m[i]**3*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - 2*zt[layer]**2*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + 2*zt[layer]**2*m[j]**3*m[i]**2*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - zt[layer]**2*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + zt[layer]**2*m[j]**4*m[i]*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + zt[layer]**2*m[j]**5*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - zt[layer]**2*m[j]**5*(zb[layer]**2*m[i]**6 -
                        3*zb[layer]**2*m[j]**2*m[i]**4 + 3*zb[layer]**2*m[j]**4*m[i]**2 -
                        zb[layer]**2*m[j]**6 - 2*zt[layer]*zb[layer]*m[i]**6 +
                        6*zt[layer]*zb[layer]*m[j]**2*m[i]**4 -
                        6*zt[layer]*zb[layer]*m[j]**4*m[i]**2 + 2*zt[layer]*zb[layer]*m[j]**6 +
                        zt[layer]**2*m[i]**6 - 3*zt[layer]**2*m[j]**2*m[i]**4 +
                        3*zt[layer]**2*m[j]**4*m[i]**2 -
                        zt[layer]**2*m[j]**6)**(-1)*bb[layer]*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]))

        #A is symmetric
        for i in range(neig - 1):
            for j in range(i + 1, neig):
                A[j, i] = A[i, j]

    elif implementation == 'fortran':
#        import geotecha.speccon.ext_integrals as ext_integ
#        A = ext_integ.dim1sin_abf_linear(m, at, ab, bt, bb, zt, zb)
        try:
            import geotecha.speccon.ext_integrals as ext_integ
            A = ext_integ.dim1sin_abf_linear(m, at, ab, bt, bb, zt, zb)
        except ImportError:
            A = dim1sin_abf_linear(m, at, ab, bt, bb, zt, zb, implementation='vectorized')

    else:#default is 'vectorized' using numpy
        sin = np.sin
        cos = np.cos
        A = np.zeros([neig, neig], float)

        diag =  np.diag_indices(neig)
        triu = np.triu_indices(neig, k = 1)
        tril = (triu[1], triu[0])

        mi = m[:, np.newaxis]
        A[diag] = np.sum(2*ab*bb*mi**3*zb**3*sin(mi*zb)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt + 12*mi**3*zt**2) +
            2*ab*bb*mi**3*zb**3*cos(mi*zb)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt + 12*mi**3*zt**2) -
            6*ab*bb*mi**3*zb**2*zt*sin(mi*zb)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt + 12*mi**3*zt**2)
            - 6*ab*bb*mi**3*zb**2*zt*cos(mi*zb)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt +
            12*mi**3*zt**2) + 6*ab*bb*mi**3*zb*zt**2*sin(mi*zb)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt
            + 12*mi**3*zt**2) + 6*ab*bb*mi**3*zb*zt**2*cos(mi*zb)**2/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2) - 2*ab*bb*mi**3*zt**3*sin(mi*zt)**2/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2) - 2*ab*bb*mi**3*zt**3*cos(mi*zt)**2/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2) -
            6*ab*bb*mi**2*zb**2*sin(mi*zb)*cos(mi*zb)/(12*mi**3*zb**2 - 24*mi**3*zb*zt +
            12*mi**3*zt**2) + 12*ab*bb*mi**2*zb*zt*sin(mi*zb)*cos(mi*zb)/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2) -
            6*ab*bb*mi**2*zt**2*sin(mi*zb)*cos(mi*zb)/(12*mi**3*zb**2 - 24*mi**3*zb*zt +
            12*mi**3*zt**2) + 3*ab*bb*mi*zb*sin(mi*zb)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt +
            12*mi**3*zt**2) - 3*ab*bb*mi*zb*cos(mi*zb)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt +
            12*mi**3*zt**2) - 3*ab*bb*mi*zt*sin(mi*zt)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt +
            12*mi**3*zt**2) + 6*ab*bb*mi*zt*cos(mi*zb)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt +
            12*mi**3*zt**2) - 3*ab*bb*mi*zt*cos(mi*zt)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt +
            12*mi**3*zt**2) + 3*ab*bb*sin(mi*zb)*cos(mi*zb)/(12*mi**3*zb**2 - 24*mi**3*zb*zt +
            12*mi**3*zt**2) - 3*ab*bb*sin(mi*zt)*cos(mi*zt)/(12*mi**3*zb**2 - 24*mi**3*zb*zt +
            12*mi**3*zt**2) + ab*bt*mi**3*zb**3*sin(mi*zb)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt +
            12*mi**3*zt**2) + ab*bt*mi**3*zb**3*cos(mi*zb)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt +
            12*mi**3*zt**2) - 3*ab*bt*mi**3*zb**2*zt*sin(mi*zb)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt
            + 12*mi**3*zt**2) - 3*ab*bt*mi**3*zb**2*zt*cos(mi*zb)**2/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2) + 3*ab*bt*mi**3*zb*zt**2*sin(mi*zt)**2/(12*mi**3*zb**2
            - 24*mi**3*zb*zt + 12*mi**3*zt**2) +
            3*ab*bt*mi**3*zb*zt**2*cos(mi*zt)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt + 12*mi**3*zt**2)
            - ab*bt*mi**3*zt**3*sin(mi*zt)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt + 12*mi**3*zt**2) -
            ab*bt*mi**3*zt**3*cos(mi*zt)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt + 12*mi**3*zt**2) -
            3*ab*bt*mi*zb*sin(mi*zb)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt + 12*mi**3*zt**2) +
            3*ab*bt*mi*zb*cos(mi*zt)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt + 12*mi**3*zt**2) +
            3*ab*bt*mi*zt*sin(mi*zt)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt + 12*mi**3*zt**2) -
            3*ab*bt*mi*zt*cos(mi*zb)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt + 12*mi**3*zt**2) -
            3*ab*bt*sin(mi*zb)*cos(mi*zb)/(12*mi**3*zb**2 - 24*mi**3*zb*zt + 12*mi**3*zt**2) +
            3*ab*bt*sin(mi*zt)*cos(mi*zt)/(12*mi**3*zb**2 - 24*mi**3*zb*zt + 12*mi**3*zt**2) +
            at*bb*mi**3*zb**3*sin(mi*zb)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt + 12*mi**3*zt**2) +
            at*bb*mi**3*zb**3*cos(mi*zb)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt + 12*mi**3*zt**2) -
            3*at*bb*mi**3*zb**2*zt*sin(mi*zb)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt + 12*mi**3*zt**2)
            - 3*at*bb*mi**3*zb**2*zt*cos(mi*zb)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt +
            12*mi**3*zt**2) + 3*at*bb*mi**3*zb*zt**2*sin(mi*zt)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt
            + 12*mi**3*zt**2) + 3*at*bb*mi**3*zb*zt**2*cos(mi*zt)**2/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2) - at*bb*mi**3*zt**3*sin(mi*zt)**2/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2) - at*bb*mi**3*zt**3*cos(mi*zt)**2/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2) - 3*at*bb*mi*zb*sin(mi*zb)**2/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2) + 3*at*bb*mi*zb*cos(mi*zt)**2/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2) + 3*at*bb*mi*zt*sin(mi*zt)**2/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2) - 3*at*bb*mi*zt*cos(mi*zb)**2/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2) - 3*at*bb*sin(mi*zb)*cos(mi*zb)/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2) + 3*at*bb*sin(mi*zt)*cos(mi*zt)/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2) + 2*at*bt*mi**3*zb**3*sin(mi*zb)**2/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2) + 2*at*bt*mi**3*zb**3*cos(mi*zb)**2/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2) - 6*at*bt*mi**3*zb**2*zt*sin(mi*zt)**2/(12*mi**3*zb**2
            - 24*mi**3*zb*zt + 12*mi**3*zt**2) -
            6*at*bt*mi**3*zb**2*zt*cos(mi*zt)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt + 12*mi**3*zt**2)
            + 6*at*bt*mi**3*zb*zt**2*sin(mi*zt)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt +
            12*mi**3*zt**2) + 6*at*bt*mi**3*zb*zt**2*cos(mi*zt)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt
            + 12*mi**3*zt**2) - 2*at*bt*mi**3*zt**3*sin(mi*zt)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt +
            12*mi**3*zt**2) - 2*at*bt*mi**3*zt**3*cos(mi*zt)**2/(12*mi**3*zb**2 - 24*mi**3*zb*zt +
            12*mi**3*zt**2) + 6*at*bt*mi**2*zb**2*sin(mi*zt)*cos(mi*zt)/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2) -
            12*at*bt*mi**2*zb*zt*sin(mi*zt)*cos(mi*zt)/(12*mi**3*zb**2 - 24*mi**3*zb*zt +
            12*mi**3*zt**2) + 6*at*bt*mi**2*zt**2*sin(mi*zt)*cos(mi*zt)/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2) + 3*at*bt*mi*zb*sin(mi*zb)**2/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2) + 3*at*bt*mi*zb*cos(mi*zb)**2/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2) - 6*at*bt*mi*zb*cos(mi*zt)**2/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2) - 3*at*bt*mi*zt*sin(mi*zt)**2/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2) + 3*at*bt*mi*zt*cos(mi*zt)**2/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2) + 3*at*bt*sin(mi*zb)*cos(mi*zb)/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2) - 3*at*bt*sin(mi*zt)*cos(mi*zt)/(12*mi**3*zb**2 -
            24*mi**3*zb*zt + 12*mi**3*zt**2), axis=1)

        mi = m[triu[0]][:, np.newaxis]
        mj = m[triu[1]][:, np.newaxis]
        A[triu] = np.sum(-ab*bb*mi**5*zb**2*sin(mj*zb)*cos(mi*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            2*ab*bb*mi**5*zb*zt*sin(mj*zb)*cos(mi*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            ab*bb*mi**5*zt**2*sin(mj*zb)*cos(mi*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            ab*bb*mi**4*mj*zb**2*sin(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*ab*bb*mi**4*mj*zb*zt*sin(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2
            - 3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2
            - 6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2)
            + ab*bb*mi**4*mj*zt**2*sin(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2
            - 3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2
            - 6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2)
            + 2*ab*bb*mi**4*zb*sin(mi*zb)*sin(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*ab*bb*mi**4*zt*sin(mi*zb)*sin(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            2*ab*bb*mi**3*mj**2*zb**2*sin(mj*zb)*cos(mi*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt +
            mi**6*zt**2 - 3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 +
            3*mi**2*mj**4*zb**2 - 6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 +
            2*mj**6*zb*zt - mj**6*zt**2) -
            4*ab*bb*mi**3*mj**2*zb*zt*sin(mj*zb)*cos(mi*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt +
            mi**6*zt**2 - 3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 +
            3*mi**2*mj**4*zb**2 - 6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 +
            2*mj**6*zb*zt - mj**6*zt**2) +
            2*ab*bb*mi**3*mj**2*zt**2*sin(mj*zb)*cos(mi*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt +
            mi**6*zt**2 - 3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 +
            3*mi**2*mj**4*zb**2 - 6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 +
            2*mj**6*zb*zt - mj**6*zt**2) + 4*ab*bb*mi**3*mj*zb*cos(mi*zb)*cos(mj*zb)/(mi**6*zb**2 -
            2*mi**6*zb*zt + mi**6*zt**2 - 3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt -
            3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 - 6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 -
            mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            4*ab*bb*mi**3*mj*zt*cos(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            2*ab*bb*mi**3*sin(mj*zb)*cos(mi*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*ab*bb*mi**3*sin(mj*zt)*cos(mi*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*ab*bb*mi**2*mj**3*zb**2*sin(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt +
            mi**6*zt**2 - 3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 +
            3*mi**2*mj**4*zb**2 - 6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 +
            2*mj**6*zb*zt - mj**6*zt**2) +
            4*ab*bb*mi**2*mj**3*zb*zt*sin(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt +
            mi**6*zt**2 - 3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 +
            3*mi**2*mj**4*zb**2 - 6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 +
            2*mj**6*zb*zt - mj**6*zt**2) -
            2*ab*bb*mi**2*mj**3*zt**2*sin(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt +
            mi**6*zt**2 - 3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 +
            3*mi**2*mj**4*zb**2 - 6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 +
            2*mj**6*zb*zt - mj**6*zt**2) - 6*ab*bb*mi**2*mj*sin(mi*zb)*cos(mj*zb)/(mi**6*zb**2 -
            2*mi**6*zb*zt + mi**6*zt**2 - 3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt -
            3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 - 6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 -
            mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            6*ab*bb*mi**2*mj*sin(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            ab*bb*mi*mj**4*zb**2*sin(mj*zb)*cos(mi*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            2*ab*bb*mi*mj**4*zb*zt*sin(mj*zb)*cos(mi*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2
            - 3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2
            - 6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2)
            - ab*bb*mi*mj**4*zt**2*sin(mj*zb)*cos(mi*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2
            - 3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2
            - 6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2)
            - 4*ab*bb*mi*mj**3*zb*cos(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            4*ab*bb*mi*mj**3*zt*cos(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            6*ab*bb*mi*mj**2*sin(mj*zb)*cos(mi*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            6*ab*bb*mi*mj**2*sin(mj*zt)*cos(mi*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            ab*bb*mj**5*zb**2*sin(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*ab*bb*mj**5*zb*zt*sin(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            ab*bb*mj**5*zt**2*sin(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*ab*bb*mj**4*zb*sin(mi*zb)*sin(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            2*ab*bb*mj**4*zt*sin(mi*zb)*sin(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*ab*bb*mj**3*sin(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            2*ab*bb*mj**3*sin(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            ab*bt*mi**4*zb*sin(mi*zb)*sin(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            ab*bt*mi**4*zb*sin(mi*zt)*sin(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            ab*bt*mi**4*zt*sin(mi*zb)*sin(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            ab*bt*mi**4*zt*sin(mi*zt)*sin(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*ab*bt*mi**3*mj*zb*cos(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*ab*bt*mi**3*mj*zb*cos(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            2*ab*bt*mi**3*mj*zt*cos(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            2*ab*bt*mi**3*mj*zt*cos(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*ab*bt*mi**3*sin(mj*zb)*cos(mi*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            2*ab*bt*mi**3*sin(mj*zt)*cos(mi*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            6*ab*bt*mi**2*mj*sin(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            6*ab*bt*mi**2*mj*sin(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            2*ab*bt*mi*mj**3*zb*cos(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            2*ab*bt*mi*mj**3*zb*cos(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*ab*bt*mi*mj**3*zt*cos(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*ab*bt*mi*mj**3*zt*cos(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            6*ab*bt*mi*mj**2*sin(mj*zb)*cos(mi*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            6*ab*bt*mi*mj**2*sin(mj*zt)*cos(mi*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            ab*bt*mj**4*zb*sin(mi*zb)*sin(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            ab*bt*mj**4*zb*sin(mi*zt)*sin(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            ab*bt*mj**4*zt*sin(mi*zb)*sin(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            ab*bt*mj**4*zt*sin(mi*zt)*sin(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            2*ab*bt*mj**3*sin(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*ab*bt*mj**3*sin(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            at*bb*mi**4*zb*sin(mi*zb)*sin(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            at*bb*mi**4*zb*sin(mi*zt)*sin(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            at*bb*mi**4*zt*sin(mi*zb)*sin(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            at*bb*mi**4*zt*sin(mi*zt)*sin(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*at*bb*mi**3*mj*zb*cos(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*at*bb*mi**3*mj*zb*cos(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            2*at*bb*mi**3*mj*zt*cos(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            2*at*bb*mi**3*mj*zt*cos(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*at*bb*mi**3*sin(mj*zb)*cos(mi*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            2*at*bb*mi**3*sin(mj*zt)*cos(mi*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            6*at*bb*mi**2*mj*sin(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            6*at*bb*mi**2*mj*sin(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            2*at*bb*mi*mj**3*zb*cos(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            2*at*bb*mi*mj**3*zb*cos(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*at*bb*mi*mj**3*zt*cos(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*at*bb*mi*mj**3*zt*cos(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            6*at*bb*mi*mj**2*sin(mj*zb)*cos(mi*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            6*at*bb*mi*mj**2*sin(mj*zt)*cos(mi*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            at*bb*mj**4*zb*sin(mi*zb)*sin(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            at*bb*mj**4*zb*sin(mi*zt)*sin(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            at*bb*mj**4*zt*sin(mi*zb)*sin(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            at*bb*mj**4*zt*sin(mi*zt)*sin(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            2*at*bb*mj**3*sin(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*at*bb*mj**3*sin(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            at*bt*mi**5*zb**2*sin(mj*zt)*cos(mi*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*at*bt*mi**5*zb*zt*sin(mj*zt)*cos(mi*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            at*bt*mi**5*zt**2*sin(mj*zt)*cos(mi*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            at*bt*mi**4*mj*zb**2*sin(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            2*at*bt*mi**4*mj*zb*zt*sin(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2
            - 3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2
            - 6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2)
            - at*bt*mi**4*mj*zt**2*sin(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2
            - 3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2
            - 6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2)
            + 2*at*bt*mi**4*zb*sin(mi*zt)*sin(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*at*bt*mi**4*zt*sin(mi*zt)*sin(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*at*bt*mi**3*mj**2*zb**2*sin(mj*zt)*cos(mi*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt +
            mi**6*zt**2 - 3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 +
            3*mi**2*mj**4*zb**2 - 6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 +
            2*mj**6*zb*zt - mj**6*zt**2) +
            4*at*bt*mi**3*mj**2*zb*zt*sin(mj*zt)*cos(mi*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt +
            mi**6*zt**2 - 3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 +
            3*mi**2*mj**4*zb**2 - 6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 +
            2*mj**6*zb*zt - mj**6*zt**2) -
            2*at*bt*mi**3*mj**2*zt**2*sin(mj*zt)*cos(mi*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt +
            mi**6*zt**2 - 3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 +
            3*mi**2*mj**4*zb**2 - 6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 +
            2*mj**6*zb*zt - mj**6*zt**2) + 4*at*bt*mi**3*mj*zb*cos(mi*zt)*cos(mj*zt)/(mi**6*zb**2 -
            2*mi**6*zb*zt + mi**6*zt**2 - 3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt -
            3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 - 6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 -
            mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            4*at*bt*mi**3*mj*zt*cos(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            2*at*bt*mi**3*sin(mj*zb)*cos(mi*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*at*bt*mi**3*sin(mj*zt)*cos(mi*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            2*at*bt*mi**2*mj**3*zb**2*sin(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt +
            mi**6*zt**2 - 3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 +
            3*mi**2*mj**4*zb**2 - 6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 +
            2*mj**6*zb*zt - mj**6*zt**2) -
            4*at*bt*mi**2*mj**3*zb*zt*sin(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt +
            mi**6*zt**2 - 3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 +
            3*mi**2*mj**4*zb**2 - 6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 +
            2*mj**6*zb*zt - mj**6*zt**2) +
            2*at*bt*mi**2*mj**3*zt**2*sin(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt +
            mi**6*zt**2 - 3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 +
            3*mi**2*mj**4*zb**2 - 6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 +
            2*mj**6*zb*zt - mj**6*zt**2) - 6*at*bt*mi**2*mj*sin(mi*zb)*cos(mj*zb)/(mi**6*zb**2 -
            2*mi**6*zb*zt + mi**6*zt**2 - 3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt -
            3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 - 6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 -
            mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            6*at*bt*mi**2*mj*sin(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            at*bt*mi*mj**4*zb**2*sin(mj*zt)*cos(mi*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*at*bt*mi*mj**4*zb*zt*sin(mj*zt)*cos(mi*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2
            - 3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2
            - 6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2)
            + at*bt*mi*mj**4*zt**2*sin(mj*zt)*cos(mi*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2
            - 3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2
            - 6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2)
            - 4*at*bt*mi*mj**3*zb*cos(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            4*at*bt*mi*mj**3*zt*cos(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            6*at*bt*mi*mj**2*sin(mj*zb)*cos(mi*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            6*at*bt*mi*mj**2*sin(mj*zt)*cos(mi*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            at*bt*mj**5*zb**2*sin(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            2*at*bt*mj**5*zb*zt*sin(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            at*bt*mj**5*zt**2*sin(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*at*bt*mj**4*zb*sin(mi*zt)*sin(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            2*at*bt*mj**4*zt*sin(mi*zt)*sin(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) -
            2*at*bt*mj**3*sin(mi*zb)*cos(mj*zb)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2) +
            2*at*bt*mj**3*sin(mi*zt)*cos(mj*zt)/(mi**6*zb**2 - 2*mi**6*zb*zt + mi**6*zt**2 -
            3*mi**4*mj**2*zb**2 + 6*mi**4*mj**2*zb*zt - 3*mi**4*mj**2*zt**2 + 3*mi**2*mj**4*zb**2 -
            6*mi**2*mj**4*zb*zt + 3*mi**2*mj**4*zt**2 - mj**6*zb**2 + 2*mj**6*zb*zt - mj**6*zt**2), axis=1)
        #A is symmetric
        A[tril] = A[triu]

    return A

def dim1sin_D_aDf_linear(m, at, ab, zt, zb, implementation='vectorized'):
    """Generate code to calculate spectral method integrations

    Performs integrations of `sin(mi * z) * D[a(z) * D[sin(mj * z),z],z]`
    between [0, 1] where a(z) i piecewise linear functions of z.
    Calulation of integrals is performed at each element of a square symmetric
    matrix (size depends on size of `m`)

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    at : ``list`` of ``float``
        property at top of each layer
    ab : ``list`` of ``float``
        property at bottom of each layer
    zt : ``list`` of ``float``
        normalised depth or z-coordinate at top of each layer. `zt[0]` = 0
    zb : ``list`` of ``float``
        normalised depth or z-coordinate at bottom of each layer. `zt[-1]` = 1
    implementation: str, optional
        functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        default = 'vectorized'.  If fortran extention module cannot be imported
        then 'vectorized' version will be used.  If anything other than
        'fortran' or 'scalar' is used then default vectorized version will be
        used

    Returns
    -------
    A : numpy.ndarray
        returns a square symmetric matrix, size determined by size of `m`

    See Also
    --------
    m_from_sin_mx : used to generate 'm' input parameter
    geotecha.speccon.integrals_generate_code.dim1sin_abf_linear : sympy
    integrations to produce the code to this function.

    Notes
    -----
    The `dim1sin_D_aDf_linear` matrix, :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{\\frac{d}{dz}\\left({a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\right)\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    ..math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` and :math:`b\\left(z\\right)` are piecewise
    linear functions w.r.t. :math:`z`, that within a layer are defined by:

    ..math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively."""
    #import numpy as np #import this at module level
    #import math #import this at module level

    m = np.asarray(m)
    at = np.asarray(at)
    ab = np.asarray(ab)
    zt = np.asarray(zt)
    zb = np.asarray(zb)

    neig = len(m)

    if implementation == 'scalar':
        sin = math.sin
        cos = math.cos
        A = np.zeros([neig, neig], float)
        nlayers = len(zt)
        for layer in range(nlayers):
            for i in range(neig):
                A[i, i] += ((zb[layer] - zt[layer])**(-1)*(ab[layer] - at[layer])*sin(m[i]*zb[layer])**2/2 - (zb[layer] -
                    zt[layer])**(-1)*(ab[layer] - at[layer])*sin(m[i]*zt[layer])**2/2 -
                    m[i]*((zb[layer] - zt[layer])**(-1)*(ab[layer] - at[layer])*(zb[layer] -
                    zt[layer]) + at[layer])*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) +
                    m[i]*at[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) -
                    m[i]**2*(-(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*cos(m[i]*zb[layer])**2 +
                    (4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*cos(m[i]*zb[layer])**2 -
                    2*m[i]*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                    + 2*m[i]*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])
                    + m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]**2*sin(m[i]*zb[layer])**2 +
                    m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])**2 -
                    m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]**2*sin(m[i]*zb[layer])**2 -
                    m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])**2 -
                    2*zb[layer]*m[i]*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) +
                    2*zb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]*sin(m[i]*zb[layer])**2 +
                    2*zb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]*cos(m[i]*zb[layer])**2 +
                    2*zt[layer]*m[i]*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) -
                    2*zt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]*sin(m[i]*zb[layer])**2 -
                    2*zt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]*cos(m[i]*zb[layer])**2) +
                    m[i]**2*(-(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*cos(m[i]*zt[layer])**2 +
                    (4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*cos(m[i]*zt[layer])**2 -
                    2*m[i]*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                    + 2*m[i]*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])
                    + m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]**2*sin(m[i]*zt[layer])**2 +
                    m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])**2 -
                    m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]**2*sin(m[i]*zt[layer])**2 -
                    m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])**2 -
                    2*zb[layer]*m[i]*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) +
                    2*zb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]*sin(m[i]*zt[layer])**2 +
                    2*zb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]*cos(m[i]*zt[layer])**2 +
                    2*zt[layer]*m[i]*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) -
                    2*zt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]*sin(m[i]*zt[layer])**2 -
                    2*zt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 -
                    4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]*cos(m[i]*zt[layer])**2))
            for i in range(neig-1):
                for j in range(i + 1, neig):
                    A[i, j] += ((zb[layer] - zt[layer])**(-1)*m[j]*(ab[layer] - at[layer])*(-m[i]*(m[i]**2 -
                        m[j]**2)**(-1)*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) - m[j]*(m[i]**2 -
                        m[j]**2)**(-1)*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])) - (zb[layer] -
                        zt[layer])**(-1)*m[j]*(ab[layer] - at[layer])*(-m[i]*(m[i]**2 -
                        m[j]**2)**(-1)*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) - m[j]*(m[i]**2 -
                        m[j]**2)**(-1)*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])) - m[j]*((zb[layer] -
                        zt[layer])**(-1)*(ab[layer] - at[layer])*(zb[layer] - zt[layer]) +
                        at[layer])*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                        m[j]*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) -
                        m[j]**2*(m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4
                        - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4
                        - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        + 2*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) -
                        2*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) +
                        m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4
                        - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])
                        - m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        + m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])
                        - zb[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        zb[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                        zb[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        zb[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                        zt[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        zt[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                        zt[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        zt[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]))
                        + m[j]**2*(m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4
                        - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4
                        - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        + 2*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) -
                        2*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) +
                        m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4
                        - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])
                        - m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        + m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])
                        - zb[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        zb[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                        zb[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        zb[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                        zt[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        zt[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) -
                        zt[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        zt[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 +
                        zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 -
                        zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])))

        #A is symmetric
        for i in range(neig - 1):
            for j in range(i + 1, neig):
                A[j, i] = A[i, j]

    elif implementation == 'fortran':
#        import geotecha.speccon.ext_integrals as ext_integ
#        A = ext_integ.dim1sin_D_aDf_linear(m, at, ab, zt, zb)
        try:
            import geotecha.speccon.ext_integrals as ext_integ
            A = ext_integ.dim1sin_d_adf_linear(m, at, ab, zt, zb)
        except ImportError:
            A = dim1sin_D_aDf_linear(m, at, ab, zt, zb, implementation='vectorized')

    else:#default is 'vectorized' using numpy
        sin = np.sin
        cos = np.cos
        A = np.zeros([neig, neig], float)

        diag =  np.diag_indices(neig)
        triu = np.triu_indices(neig, k = 1)
        tril = (triu[1], triu[0])

        mi = m[:, np.newaxis]
        A[diag] = np.sum(-ab*mi*sin(mi*zb)*cos(mi*zb) + at*mi*sin(mi*zt)*cos(mi*zt) -
            mi**2*(ab*mi**2*zb**2*sin(mi*zb)**2/(4*mi**2*zb - 4*mi**2*zt) +
            ab*mi**2*zb**2*cos(mi*zb)**2/(4*mi**2*zb - 4*mi**2*zt) -
            2*ab*mi**2*zb*zt*sin(mi*zb)**2/(4*mi**2*zb - 4*mi**2*zt) -
            2*ab*mi**2*zb*zt*cos(mi*zb)**2/(4*mi**2*zb - 4*mi**2*zt) -
            2*ab*mi*zb*sin(mi*zb)*cos(mi*zb)/(4*mi**2*zb - 4*mi**2*zt) +
            2*ab*mi*zt*sin(mi*zb)*cos(mi*zb)/(4*mi**2*zb - 4*mi**2*zt) -
            ab*cos(mi*zb)**2/(4*mi**2*zb - 4*mi**2*zt) + at*mi**2*zb**2*sin(mi*zb)**2/(4*mi**2*zb -
            4*mi**2*zt) + at*mi**2*zb**2*cos(mi*zb)**2/(4*mi**2*zb - 4*mi**2*zt) +
            at*cos(mi*zb)**2/(4*mi**2*zb - 4*mi**2*zt)) +
            mi**2*(-ab*mi**2*zt**2*sin(mi*zt)**2/(4*mi**2*zb - 4*mi**2*zt) -
            ab*mi**2*zt**2*cos(mi*zt)**2/(4*mi**2*zb - 4*mi**2*zt) - ab*cos(mi*zt)**2/(4*mi**2*zb -
            4*mi**2*zt) + 2*at*mi**2*zb*zt*sin(mi*zt)**2/(4*mi**2*zb - 4*mi**2*zt) +
            2*at*mi**2*zb*zt*cos(mi*zt)**2/(4*mi**2*zb - 4*mi**2*zt) -
            at*mi**2*zt**2*sin(mi*zt)**2/(4*mi**2*zb - 4*mi**2*zt) -
            at*mi**2*zt**2*cos(mi*zt)**2/(4*mi**2*zb - 4*mi**2*zt) -
            2*at*mi*zb*sin(mi*zt)*cos(mi*zt)/(4*mi**2*zb - 4*mi**2*zt) +
            2*at*mi*zt*sin(mi*zt)*cos(mi*zt)/(4*mi**2*zb - 4*mi**2*zt) +
            at*cos(mi*zt)**2/(4*mi**2*zb - 4*mi**2*zt)) + (ab - at)*sin(mi*zb)**2/(2*(zb - zt)) -
            (ab - at)*sin(mi*zt)**2/(2*(zb - zt)), axis=1)

        mi = m[triu[0]][:, np.newaxis]
        mj = m[triu[1]][:, np.newaxis]
        A[triu] = np.sum(-ab*mj*sin(mi*zb)*cos(mj*zb) + at*mj*sin(mi*zt)*cos(mj*zt) +
            mj**2*(ab*mi**2*sin(mi*zt)*sin(mj*zt)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) + 2*ab*mi*mj*cos(mi*zt)*cos(mj*zt)/(mi**4*zb -
            mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) +
            ab*mj**2*sin(mi*zt)*sin(mj*zt)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) - at*mi**3*zb*sin(mj*zt)*cos(mi*zt)/(mi**4*zb -
            mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) +
            at*mi**3*zt*sin(mj*zt)*cos(mi*zt)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) + at*mi**2*mj*zb*sin(mi*zt)*cos(mj*zt)/(mi**4*zb
            - mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) -
            at*mi**2*mj*zt*sin(mi*zt)*cos(mj*zt)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) - at*mi**2*sin(mi*zt)*sin(mj*zt)/(mi**4*zb -
            mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) +
            at*mi*mj**2*zb*sin(mj*zt)*cos(mi*zt)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) - at*mi*mj**2*zt*sin(mj*zt)*cos(mi*zt)/(mi**4*zb
            - mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) -
            2*at*mi*mj*cos(mi*zt)*cos(mj*zt)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) - at*mj**3*zb*sin(mi*zt)*cos(mj*zt)/(mi**4*zb -
            mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) +
            at*mj**3*zt*sin(mi*zt)*cos(mj*zt)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) - at*mj**2*sin(mi*zt)*sin(mj*zt)/(mi**4*zb -
            mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt)) -
            mj**2*(-ab*mi**3*zb*sin(mj*zb)*cos(mi*zb)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) + ab*mi**3*zt*sin(mj*zb)*cos(mi*zb)/(mi**4*zb -
            mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) +
            ab*mi**2*mj*zb*sin(mi*zb)*cos(mj*zb)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) - ab*mi**2*mj*zt*sin(mi*zb)*cos(mj*zb)/(mi**4*zb
            - mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) +
            ab*mi**2*sin(mi*zb)*sin(mj*zb)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) + ab*mi*mj**2*zb*sin(mj*zb)*cos(mi*zb)/(mi**4*zb
            - mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) -
            ab*mi*mj**2*zt*sin(mj*zb)*cos(mi*zb)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) + 2*ab*mi*mj*cos(mi*zb)*cos(mj*zb)/(mi**4*zb -
            mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) -
            ab*mj**3*zb*sin(mi*zb)*cos(mj*zb)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) + ab*mj**3*zt*sin(mi*zb)*cos(mj*zb)/(mi**4*zb -
            mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) +
            ab*mj**2*sin(mi*zb)*sin(mj*zb)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) - at*mi**2*sin(mi*zb)*sin(mj*zb)/(mi**4*zb -
            mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) -
            2*at*mi*mj*cos(mi*zb)*cos(mj*zb)/(mi**4*zb - mi**4*zt - 2*mi**2*mj**2*zb +
            2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt) - at*mj**2*sin(mi*zb)*sin(mj*zb)/(mi**4*zb -
            mi**4*zt - 2*mi**2*mj**2*zb + 2*mi**2*mj**2*zt + mj**4*zb - mj**4*zt)) + mj*(ab -
            at)*(-mi*cos(mi*zb)*cos(mj*zb)/(mi**2 - mj**2) - mj*sin(mi*zb)*sin(mj*zb)/(mi**2 -
            mj**2))/(zb - zt) - mj*(ab - at)*(-mi*cos(mi*zt)*cos(mj*zt)/(mi**2 - mj**2) -
            mj*sin(mi*zt)*sin(mj*zt)/(mi**2 - mj**2))/(zb - zt), axis=1)
        #A is symmetric
        A[tril] = A[triu]

    return A

def dim1sin_ab_linear(m, at, ab, bt, bb, zt, zb, implementation='vectorized'):
    """Create matrix of spectral integrations

    Performs integrations of `sin(mi * z) * a(z) *b(z)`
    between [0, 1] where a(z) and b(z) are piecewise linear functions of z.
    Calulation of integrals is performed at each element of a 1d array
    (size depends on size of `m`)

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    at : ``list`` of ``float``
        property at top of each layer
    ab : ``list`` of ``float``
        property at bottom of each layer
    bt : ``list`` of ``float``
        2nd property at top of each layer
    bb : ``list`` of ``float``
        2nd property at bottom of each layer
    zt : ``list`` of ``float``
        normalised depth or z-coordinate at top of each layer. `zt[0]` = 0
    zb : ``list`` of ``float``
        normalised depth or z-coordinate at bottom of each layer. `zt[-1]` = 1
    implementation: str, optional
        functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        default = 'vectorized'.  If fortran extention module cannot be imported
        then 'vectorized' version will be used.  If anything other than
        'fortran' or 'scalar' is used then default vectorized version will be
        used

    Returns
    -------
    A : numpy.ndarray
        returns a 1d array size determined by size of `m`. treat as column
        vector

    See Also
    --------
    m_from_sin_mx : used to generate 'm' input parameter
    geotecha.integrals_generate_code.dim1sin_ab_linear : use sympy
    to perform the integrals symbolically and generate expressions for
    this function

    Notes
    -----
    The `dim1sin_ab_linear` which should be treated as a column vector,
    :math:`A` is given by:

    .. math:: \\mathbf{A}_{i}=\\int_{0}^1{{a\\left(z\\right)}{b\\left(z\\right)}\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    ..math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` and :math:`b\\left(z\\right)` are piecewise
    linear functions w.r.t. :math:`z`, that within a layer are defined by:

    ..math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """

    #import numpy as np #import this at module level
    #import math #import this at module level

    m = np.asarray(m)
    at = np.asarray(at)
    ab = np.asarray(ab)
    bt = np.asarray(bt)
    bb = np.asarray(bb)
    zt = np.asarray(zt)
    zb = np.asarray(zb)

    neig = len(m)

    if implementation == 'scalar':
        sin = math.sin
        cos = math.cos
        A = np.zeros(neig, float)
        nlayers = len(zt)
        for layer in range(nlayers):
            for i in range(neig):
                A[i] += (2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zb[layer]) -
                    2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zt[layer]) -
                    2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer]) +
                    2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer]) -
                    2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer]) +
                    2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer]) +
                    2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*cos(m[i]*zb[layer]) -
                    2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*cos(m[i]*zt[layer]) +
                    2*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer]) -
                    2*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer]) -
                    2*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer]) +
                    2*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer]) -
                    2*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer]) +
                    2*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer]) +
                    2*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer]) -
                    2*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer]) -
                    m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])
                    + m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])
                    + m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])
                    - m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])
                    + m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])
                    - m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])
                    - m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])
                    + m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])
                    + zb[layer]*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*sin(m[i]*zb[layer]) -
                    zb[layer]*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*sin(m[i]*zt[layer]) +
                    zb[layer]*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*sin(m[i]*zb[layer]) -
                    zb[layer]*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*sin(m[i]*zt[layer]) -
                    2*zb[layer]*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*sin(m[i]*zb[layer]) +
                    2*zb[layer]*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*sin(m[i]*zt[layer]) -
                    zb[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer]) +
                    zb[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer]) -
                    zb[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer]) +
                    zb[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer]) +
                    2*zb[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer]) -
                    2*zb[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer]) -
                    zb[layer]**2*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*cos(m[i]*zb[layer]) +
                    zb[layer]**2*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*at[layer]*cos(m[i]*zt[layer]) -
                    2*zt[layer]*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*sin(m[i]*zb[layer]) +
                    2*zt[layer]*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*sin(m[i]*zt[layer]) +
                    zt[layer]*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*sin(m[i]*zb[layer]) -
                    zt[layer]*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*sin(m[i]*zt[layer]) +
                    zt[layer]*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*sin(m[i]*zb[layer]) -
                    zt[layer]*m[i]*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*sin(m[i]*zt[layer]) +
                    2*zt[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer]) -
                    2*zt[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer]) -
                    zt[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer]) +
                    zt[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer]) -
                    zt[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer]) +
                    zt[layer]*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer]) +
                    zt[layer]*zb[layer]*m[i]**2*(zb[layer]**2*m[i]**3 -
                    2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*cos(m[i]*zb[layer]) -
                    zt[layer]*zb[layer]*m[i]**2*(zb[layer]**2*m[i]**3 -
                    2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*at[layer]*cos(m[i]*zt[layer]) +
                    zt[layer]*zb[layer]*m[i]**2*(zb[layer]**2*m[i]**3 -
                    2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zb[layer]) -
                    zt[layer]*zb[layer]*m[i]**2*(zb[layer]**2*m[i]**3 -
                    2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bt[layer]*ab[layer]*cos(m[i]*zt[layer]) -
                    zt[layer]**2*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zb[layer]) +
                    zt[layer]**2*m[i]**2*(zb[layer]**2*m[i]**3 - 2*zt[layer]*zb[layer]*m[i]**3 +
                    zt[layer]**2*m[i]**3)**(-1)*bb[layer]*ab[layer]*cos(m[i]*zt[layer]))

    elif implementation == 'fortran':
#        import geotecha.speccon.ext_integrals as ext_integ
#        A = ext_integ.dim1sin_ab_linear(m, at, ab, bt, bb, zt, zb)
        try:
            import geotecha.speccon.ext_integrals as ext_integ
            A = ext_integ.dim1sin_ab_linear(m, at, ab, bt, bb, zt, zb)
        except ImportError:
            A = dim1sin_ab_linear(m, at, ab, bt, bb, zt, zb, implementation='vectorized')

    else:#default is 'vectorized' using numpy
        sin = np.sin
        cos = np.cos
        A = np.zeros(neig, float)

        mi = m[:, np.newaxis]
        A[:] = np.sum(-ab*bb*mi**2*zb**2*cos(mi*zb)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) +
            2*ab*bb*mi**2*zb*zt*cos(mi*zb)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) -
            ab*bb*mi**2*zt**2*cos(mi*zb)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) +
            2*ab*bb*mi*zb*sin(mi*zb)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) -
            2*ab*bb*mi*zt*sin(mi*zb)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) +
            2*ab*bb*cos(mi*zb)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) -
            2*ab*bb*cos(mi*zt)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) -
            ab*bt*mi*zb*sin(mi*zb)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) -
            ab*bt*mi*zb*sin(mi*zt)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) +
            ab*bt*mi*zt*sin(mi*zb)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) +
            ab*bt*mi*zt*sin(mi*zt)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) -
            2*ab*bt*cos(mi*zb)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) +
            2*ab*bt*cos(mi*zt)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) -
            at*bb*mi*zb*sin(mi*zb)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) -
            at*bb*mi*zb*sin(mi*zt)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) +
            at*bb*mi*zt*sin(mi*zb)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) +
            at*bb*mi*zt*sin(mi*zt)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) -
            2*at*bb*cos(mi*zb)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) +
            2*at*bb*cos(mi*zt)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) +
            at*bt*mi**2*zb**2*cos(mi*zt)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) -
            2*at*bt*mi**2*zb*zt*cos(mi*zt)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) +
            at*bt*mi**2*zt**2*cos(mi*zt)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) +
            2*at*bt*mi*zb*sin(mi*zt)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) -
            2*at*bt*mi*zt*sin(mi*zt)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) +
            2*at*bt*cos(mi*zb)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2) -
            2*at*bt*cos(mi*zt)/(mi**3*zb**2 - 2*mi**3*zb*zt + mi**3*zt**2), axis=1)


    return A

def dim1sin_abc_linear(m, at, ab, bt, bb, ct, cb, zt, zb, implementation='vectorized'):
    """Create matrix of spectral integrations

    Performs integrations of `sin(mi * z) * a(z) * b(z) * c(z)`
    between [0, 1] where a(z), b(z), c(z) are piecewise linear functions of z.
    Calulation of integrals is performed at each element of a 1d array
    (size depends on size of `m`)

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    at : ``list`` of ``float``
        property at top of each layer
    ab : ``list`` of ``float``
        property at bottom of each layer
    bt : ``list`` of ``float``
        2nd property at top of each layer
    bb : ``list`` of ``float``
        2nd property at bottom of each layer
    ct : ``list`` of ``float``
        3rd property at top of each layer
    cb : ``list`` of ``float``
        3rd property at bottom of each layer
    zt : ``list`` of ``float``
        normalised depth or z-coordinate at top of each layer. `zt[0]` = 0
    zb : ``list`` of ``float``
        normalised depth or z-coordinate at bottom of each layer. `zt[-1]` = 1
    implementation: str, optional
        functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        default = 'vectorized'.  If fortran extention module cannot be imported
        then 'vectorized' version will be used.  If anything other than
        'fortran' or 'scalar' is used then default vectorized version will be
        used

    Returns
    -------
    A : numpy.ndarray
        returns a 1d array size determined by size of `m`. treat as column
        vector

    See Also
    --------
    m_from_sin_mx : used to generate 'm' input parameter
    geotecha.integrals_generate_code.dim1sin_abc_linear : use sympy
    to perform the integrals symbolically and generate expressions for
    this function

    Notes
    -----
    The `dim1sin_abc_linear` which should be treated as a column vector,
    :math:`A` is given by:

    .. math:: \\mathbf{A}_{i}=\\int_{0}^1{{a\\left(z\\right)}{b\\left(z\\right)}{c\\left(z\\right)}\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    ..math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)`, :math:`b\\left(z\\right)`, and
    :math:`c\\left(z\\right)` are piecewise linear functions
    w.r.t. :math:`z`, that within a layer are defined by:

    ..math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """

    #import numpy as np #import this at module level
    #import math #import this at module level

    m = np.asarray(m)
    at = np.asarray(at)
    ab = np.asarray(ab)
    bt = np.asarray(bt)
    bb = np.asarray(bb)
    zt = np.asarray(zt)
    zb = np.asarray(zb)

    neig = len(m)

    if implementation == 'scalar':
        sin = math.sin
        cos = math.cos
        A = np.zeros(neig, float)
        nlayers = len(zt)
        for layer in range(nlayers):
            for i in range(neig):
                A[i] += (-6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*sin(m[i]*zb[layer]) +
                    6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*sin(m[i]*zt[layer]) +
                    6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*sin(m[i]*zb[layer]) -
                    6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*sin(m[i]*zt[layer]) +
                    6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*sin(m[i]*zb[layer]) -
                    6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*sin(m[i]*zt[layer]) -
                    6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*sin(m[i]*zb[layer]) +
                    6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*sin(m[i]*zt[layer]) +
                    6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*sin(m[i]*zb[layer]) -
                    6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*sin(m[i]*zt[layer]) -
                    6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*sin(m[i]*zb[layer]) +
                    6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*sin(m[i]*zt[layer]) -
                    6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*sin(m[i]*zb[layer]) +
                    6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*sin(m[i]*zt[layer]) +
                    6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*sin(m[i]*zb[layer]) -
                    6*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*sin(m[i]*zt[layer]) +
                    6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])
                    - 6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])
                    - 6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])
                    + 6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])
                    - 6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])
                    + 6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])
                    + 6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])
                    - 6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])
                    - 6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])
                    + 6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])
                    + 6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])
                    - 6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])
                    + 6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])
                    - 6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])
                    - 6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])
                    + 6*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])
                    + 3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*zb[layer]**2*sin(m[i]*zb[layer])
                    - 3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*zt[layer]**2*sin(m[i]*zt[layer])
                    - 3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zb[layer]**2*sin(m[i]*zb[layer])
                    + 3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zt[layer]**2*sin(m[i]*zt[layer])
                    - 3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zb[layer]**2*sin(m[i]*zb[layer])
                    + 3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zt[layer]**2*sin(m[i]*zt[layer])
                    + 3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zb[layer]**2*sin(m[i]*zb[layer])
                    - 3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zt[layer]**2*sin(m[i]*zt[layer])
                    - 3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zb[layer]**2*sin(m[i]*zb[layer])
                    + 3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zt[layer]**2*sin(m[i]*zt[layer])
                    + 3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zb[layer]**2*sin(m[i]*zb[layer])
                    - 3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zt[layer]**2*sin(m[i]*zt[layer])
                    + 3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zb[layer]**2*sin(m[i]*zb[layer])
                    - 3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zt[layer]**2*sin(m[i]*zt[layer])
                    - 3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*zb[layer]**2*sin(m[i]*zb[layer])
                    + 3*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*zt[layer]**2*sin(m[i]*zt[layer])
                    - m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*zb[layer]**3*cos(m[i]*zb[layer])
                    + m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*zt[layer]**3*cos(m[i]*zt[layer])
                    + m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zb[layer]**3*cos(m[i]*zb[layer])
                    - m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zt[layer]**3*cos(m[i]*zt[layer])
                    + m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zb[layer]**3*cos(m[i]*zb[layer])
                    - m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zt[layer]**3*cos(m[i]*zt[layer])
                    - m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zb[layer]**3*cos(m[i]*zb[layer])
                    + m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zt[layer]**3*cos(m[i]*zt[layer])
                    + m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zb[layer]**3*cos(m[i]*zb[layer])
                    - m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zt[layer]**3*cos(m[i]*zt[layer])
                    - m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zb[layer]**3*cos(m[i]*zb[layer])
                    + m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zt[layer]**3*cos(m[i]*zt[layer])
                    - m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zb[layer]**3*cos(m[i]*zb[layer])
                    + m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zt[layer]**3*cos(m[i]*zt[layer])
                    + m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*zb[layer]**3*cos(m[i]*zb[layer])
                    - m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*zt[layer]**3*cos(m[i]*zt[layer])
                    + 2*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*cos(m[i]*zb[layer]) -
                    2*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*cos(m[i]*zt[layer]) +
                    2*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*cos(m[i]*zb[layer]) -
                    2*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*cos(m[i]*zt[layer]) -
                    4*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*cos(m[i]*zb[layer]) +
                    4*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*cos(m[i]*zt[layer]) +
                    2*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*cos(m[i]*zb[layer]) -
                    2*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*cos(m[i]*zt[layer]) -
                    4*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*cos(m[i]*zb[layer]) +
                    4*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*cos(m[i]*zt[layer]) -
                    4*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*cos(m[i]*zb[layer]) +
                    4*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*cos(m[i]*zt[layer]) +
                    6*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*cos(m[i]*zb[layer]) -
                    6*zb[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*cos(m[i]*zt[layer]) +
                    2*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer])
                    - 2*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer])
                    + 2*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer])
                    - 2*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer])
                    - 4*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer])
                    + 4*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer])
                    + 2*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer])
                    - 2*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer])
                    - 4*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer])
                    + 4*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer])
                    - 4*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer])
                    + 4*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer])
                    + 6*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer])
                    - 6*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer])
                    - zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])
                    + zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])
                    - zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])
                    + zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])
                    + 2*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])
                    - 2*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])
                    - zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])
                    + zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])
                    + 2*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])
                    - 2*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])
                    + 2*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])
                    - 2*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])
                    - 3*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])
                    + 3*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])
                    + zb[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4
                    + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*sin(m[i]*zb[layer]) -
                    zb[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*sin(m[i]*zt[layer]) +
                    zb[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*sin(m[i]*zb[layer]) -
                    zb[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*sin(m[i]*zt[layer]) +
                    zb[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*sin(m[i]*zb[layer]) -
                    zb[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*sin(m[i]*zt[layer]) -
                    3*zb[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4
                    + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*sin(m[i]*zb[layer]) +
                    3*zb[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4
                    + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*sin(m[i]*zt[layer]) -
                    zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])
                    + zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4
                    + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])
                    - zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4
                    + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])
                    + zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4
                    + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])
                    - zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4
                    + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])
                    + zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4
                    + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])
                    + 3*zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])
                    - 3*zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])
                    - zb[layer]**3*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4
                    + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*cos(m[i]*zb[layer]) +
                    zb[layer]**3*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*at[layer]*cos(m[i]*zt[layer]) -
                    6*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*cos(m[i]*zb[layer]) +
                    6*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*cos(m[i]*zt[layer]) +
                    4*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*cos(m[i]*zb[layer]) -
                    4*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*cos(m[i]*zt[layer]) +
                    4*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*cos(m[i]*zb[layer]) -
                    4*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*cos(m[i]*zt[layer]) -
                    2*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*cos(m[i]*zb[layer]) +
                    2*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*cos(m[i]*zt[layer]) +
                    4*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*cos(m[i]*zb[layer]) -
                    4*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*cos(m[i]*zt[layer]) -
                    2*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*cos(m[i]*zb[layer]) +
                    2*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*cos(m[i]*zt[layer]) -
                    2*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*cos(m[i]*zb[layer]) +
                    2*zt[layer]*m[i]*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*cos(m[i]*zt[layer]) -
                    6*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer])
                    + 6*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer])
                    + 4*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer])
                    - 4*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer])
                    + 4*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer])
                    - 4*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer])
                    - 2*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer])
                    + 2*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer])
                    + 4*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer])
                    - 4*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer])
                    - 2*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zb[layer]*sin(m[i]*zb[layer])
                    + 2*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zt[layer]*sin(m[i]*zt[layer])
                    - 2*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zb[layer]*sin(m[i]*zb[layer])
                    + 2*zt[layer]*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zt[layer]*sin(m[i]*zt[layer])
                    + 3*zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])
                    - 3*zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])
                    - 2*zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])
                    + 2*zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])
                    - 2*zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])
                    + 2*zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])
                    + zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])
                    - zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])
                    - 2*zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])
                    + 2*zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])
                    + zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])
                    - zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])
                    + zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])
                    - zt[layer]*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])
                    - 2*zt[layer]*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*sin(m[i]*zb[layer]) +
                    2*zt[layer]*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*sin(m[i]*zt[layer]) -
                    2*zt[layer]*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*sin(m[i]*zb[layer]) +
                    2*zt[layer]*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*sin(m[i]*zt[layer]) +
                    2*zt[layer]*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*sin(m[i]*zb[layer]) -
                    2*zt[layer]*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*sin(m[i]*zt[layer]) -
                    2*zt[layer]*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*sin(m[i]*zb[layer]) +
                    2*zt[layer]*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*sin(m[i]*zt[layer]) +
                    2*zt[layer]*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*sin(m[i]*zb[layer]) -
                    2*zt[layer]*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*sin(m[i]*zt[layer]) +
                    2*zt[layer]*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*sin(m[i]*zb[layer]) -
                    2*zt[layer]*zb[layer]*m[i]**2*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*sin(m[i]*zt[layer]) +
                    2*zt[layer]*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])
                    - 2*zt[layer]*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])
                    + 2*zt[layer]*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])
                    - 2*zt[layer]*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])
                    - 2*zt[layer]*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])
                    + 2*zt[layer]*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])
                    + 2*zt[layer]*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])
                    - 2*zt[layer]*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])
                    - 2*zt[layer]*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])
                    + 2*zt[layer]*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])
                    - 2*zt[layer]*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])
                    + 2*zt[layer]*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])
                    + zt[layer]*zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*cos(m[i]*zb[layer]) -
                    zt[layer]*zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*at[layer]*cos(m[i]*zt[layer]) +
                    zt[layer]*zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*cos(m[i]*zb[layer]) -
                    zt[layer]*zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*at[layer]*cos(m[i]*zt[layer]) +
                    zt[layer]*zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*cos(m[i]*zb[layer]) -
                    zt[layer]*zb[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bt[layer]*ab[layer]*cos(m[i]*zt[layer]) +
                    3*zt[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4
                    + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*sin(m[i]*zb[layer]) -
                    3*zt[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4
                    + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*sin(m[i]*zt[layer]) -
                    zt[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*sin(m[i]*zb[layer]) +
                    zt[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*sin(m[i]*zt[layer]) -
                    zt[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*sin(m[i]*zb[layer]) +
                    zt[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*sin(m[i]*zt[layer]) -
                    zt[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*sin(m[i]*zb[layer]) +
                    zt[layer]**2*m[i]**2*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*sin(m[i]*zt[layer]) -
                    3*zt[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4
                    + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])
                    + 3*zt[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])
                    + zt[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4
                    + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zb[layer]*cos(m[i]*zb[layer])
                    - zt[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4
                    + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*zt[layer]*cos(m[i]*zt[layer])
                    + zt[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4
                    + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])
                    - zt[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4
                    + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])
                    + zt[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4
                    + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zb[layer]*cos(m[i]*zb[layer])
                    - zt[layer]**2*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4
                    + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*zt[layer]*cos(m[i]*zt[layer])
                    - zt[layer]**2*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*cos(m[i]*zb[layer]) +
                    zt[layer]**2*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*at[layer]*cos(m[i]*zt[layer]) -
                    zt[layer]**2*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*cos(m[i]*zb[layer]) +
                    zt[layer]**2*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bt[layer]*ab[layer]*cos(m[i]*zt[layer]) -
                    zt[layer]**2*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*cos(m[i]*zb[layer]) +
                    zt[layer]**2*zb[layer]*m[i]**3*(zb[layer]**3*m[i]**4 -
                    3*zt[layer]*zb[layer]**2*m[i]**4 + 3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*ct[layer]*bb[layer]*ab[layer]*cos(m[i]*zt[layer]) +
                    zt[layer]**3*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*cos(m[i]*zb[layer]) -
                    zt[layer]**3*m[i]**3*(zb[layer]**3*m[i]**4 - 3*zt[layer]*zb[layer]**2*m[i]**4 +
                    3*zt[layer]**2*zb[layer]*m[i]**4 -
                    zt[layer]**3*m[i]**4)**(-1)*cb[layer]*bb[layer]*ab[layer]*cos(m[i]*zt[layer]))

    elif implementation == 'fortran':
#        import geotecha.speccon.ext_integrals as ext_integ
#        A = ext_integ.dim1sin_abc_linear(m, at, ab, bt, bb, ct, cb, zt, zb)
        try:
            import geotecha.speccon.ext_integrals as ext_integ
            A = ext_integ.dim1sin_abc_linear(m, at, ab, bt, bb,  ct, cb, zt, zb)
        except ImportError:
            A = dim1sin_abc_linear(m, at, ab, bt, bb,  ct, cb, zt, zb, implementation='vectorized')

    else:#default is 'vectorized' using numpy
        sin = np.sin
        cos = np.cos
        A = np.zeros(neig, float)

        mi = m[:, np.newaxis]
        A[:] = np.sum(-ab*bb*cb*mi**3*zb**3*cos(mi*zb)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) +
            3*ab*bb*cb*mi**3*zb**2*zt*cos(mi*zb)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2
            - mi**4*zt**3) - 3*ab*bb*cb*mi**3*zb*zt**2*cos(mi*zb)/(mi**4*zb**3 - 3*mi**4*zb**2*zt +
            3*mi**4*zb*zt**2 - mi**4*zt**3) + ab*bb*cb*mi**3*zt**3*cos(mi*zb)/(mi**4*zb**3 -
            3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) +
            3*ab*bb*cb*mi**2*zb**2*sin(mi*zb)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 -
            mi**4*zt**3) - 6*ab*bb*cb*mi**2*zb*zt*sin(mi*zb)/(mi**4*zb**3 - 3*mi**4*zb**2*zt +
            3*mi**4*zb*zt**2 - mi**4*zt**3) + 3*ab*bb*cb*mi**2*zt**2*sin(mi*zb)/(mi**4*zb**3 -
            3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) +
            6*ab*bb*cb*mi*zb*cos(mi*zb)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 -
            mi**4*zt**3) - 6*ab*bb*cb*mi*zt*cos(mi*zb)/(mi**4*zb**3 - 3*mi**4*zb**2*zt +
            3*mi**4*zb*zt**2 - mi**4*zt**3) - 6*ab*bb*cb*sin(mi*zb)/(mi**4*zb**3 - 3*mi**4*zb**2*zt
            + 3*mi**4*zb*zt**2 - mi**4*zt**3) + 6*ab*bb*cb*sin(mi*zt)/(mi**4*zb**3 -
            3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) -
            ab*bb*ct*mi**2*zb**2*sin(mi*zb)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 -
            mi**4*zt**3) + 2*ab*bb*ct*mi**2*zb*zt*sin(mi*zb)/(mi**4*zb**3 - 3*mi**4*zb**2*zt +
            3*mi**4*zb*zt**2 - mi**4*zt**3) - ab*bb*ct*mi**2*zt**2*sin(mi*zb)/(mi**4*zb**3 -
            3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) -
            4*ab*bb*ct*mi*zb*cos(mi*zb)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 -
            mi**4*zt**3) - 2*ab*bb*ct*mi*zb*cos(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt +
            3*mi**4*zb*zt**2 - mi**4*zt**3) + 4*ab*bb*ct*mi*zt*cos(mi*zb)/(mi**4*zb**3 -
            3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) +
            2*ab*bb*ct*mi*zt*cos(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 -
            mi**4*zt**3) + 6*ab*bb*ct*sin(mi*zb)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2
            - mi**4*zt**3) - 6*ab*bb*ct*sin(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt +
            3*mi**4*zb*zt**2 - mi**4*zt**3) - ab*bt*cb*mi**2*zb**2*sin(mi*zb)/(mi**4*zb**3 -
            3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) +
            2*ab*bt*cb*mi**2*zb*zt*sin(mi*zb)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 -
            mi**4*zt**3) - ab*bt*cb*mi**2*zt**2*sin(mi*zb)/(mi**4*zb**3 - 3*mi**4*zb**2*zt +
            3*mi**4*zb*zt**2 - mi**4*zt**3) - 4*ab*bt*cb*mi*zb*cos(mi*zb)/(mi**4*zb**3 -
            3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) -
            2*ab*bt*cb*mi*zb*cos(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 -
            mi**4*zt**3) + 4*ab*bt*cb*mi*zt*cos(mi*zb)/(mi**4*zb**3 - 3*mi**4*zb**2*zt +
            3*mi**4*zb*zt**2 - mi**4*zt**3) + 2*ab*bt*cb*mi*zt*cos(mi*zt)/(mi**4*zb**3 -
            3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) + 6*ab*bt*cb*sin(mi*zb)/(mi**4*zb**3
            - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) -
            6*ab*bt*cb*sin(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3)
            - ab*bt*ct*mi**2*zb**2*sin(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 -
            mi**4*zt**3) + 2*ab*bt*ct*mi**2*zb*zt*sin(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt +
            3*mi**4*zb*zt**2 - mi**4*zt**3) - ab*bt*ct*mi**2*zt**2*sin(mi*zt)/(mi**4*zb**3 -
            3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) +
            2*ab*bt*ct*mi*zb*cos(mi*zb)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 -
            mi**4*zt**3) + 4*ab*bt*ct*mi*zb*cos(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt +
            3*mi**4*zb*zt**2 - mi**4*zt**3) - 2*ab*bt*ct*mi*zt*cos(mi*zb)/(mi**4*zb**3 -
            3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) -
            4*ab*bt*ct*mi*zt*cos(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 -
            mi**4*zt**3) - 6*ab*bt*ct*sin(mi*zb)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2
            - mi**4*zt**3) + 6*ab*bt*ct*sin(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt +
            3*mi**4*zb*zt**2 - mi**4*zt**3) - at*bb*cb*mi**2*zb**2*sin(mi*zb)/(mi**4*zb**3 -
            3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) +
            2*at*bb*cb*mi**2*zb*zt*sin(mi*zb)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 -
            mi**4*zt**3) - at*bb*cb*mi**2*zt**2*sin(mi*zb)/(mi**4*zb**3 - 3*mi**4*zb**2*zt +
            3*mi**4*zb*zt**2 - mi**4*zt**3) - 4*at*bb*cb*mi*zb*cos(mi*zb)/(mi**4*zb**3 -
            3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) -
            2*at*bb*cb*mi*zb*cos(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 -
            mi**4*zt**3) + 4*at*bb*cb*mi*zt*cos(mi*zb)/(mi**4*zb**3 - 3*mi**4*zb**2*zt +
            3*mi**4*zb*zt**2 - mi**4*zt**3) + 2*at*bb*cb*mi*zt*cos(mi*zt)/(mi**4*zb**3 -
            3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) + 6*at*bb*cb*sin(mi*zb)/(mi**4*zb**3
            - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) -
            6*at*bb*cb*sin(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3)
            - at*bb*ct*mi**2*zb**2*sin(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 -
            mi**4*zt**3) + 2*at*bb*ct*mi**2*zb*zt*sin(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt +
            3*mi**4*zb*zt**2 - mi**4*zt**3) - at*bb*ct*mi**2*zt**2*sin(mi*zt)/(mi**4*zb**3 -
            3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) +
            2*at*bb*ct*mi*zb*cos(mi*zb)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 -
            mi**4*zt**3) + 4*at*bb*ct*mi*zb*cos(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt +
            3*mi**4*zb*zt**2 - mi**4*zt**3) - 2*at*bb*ct*mi*zt*cos(mi*zb)/(mi**4*zb**3 -
            3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) -
            4*at*bb*ct*mi*zt*cos(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 -
            mi**4*zt**3) - 6*at*bb*ct*sin(mi*zb)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2
            - mi**4*zt**3) + 6*at*bb*ct*sin(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt +
            3*mi**4*zb*zt**2 - mi**4*zt**3) - at*bt*cb*mi**2*zb**2*sin(mi*zt)/(mi**4*zb**3 -
            3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) +
            2*at*bt*cb*mi**2*zb*zt*sin(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 -
            mi**4*zt**3) - at*bt*cb*mi**2*zt**2*sin(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt +
            3*mi**4*zb*zt**2 - mi**4*zt**3) + 2*at*bt*cb*mi*zb*cos(mi*zb)/(mi**4*zb**3 -
            3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) +
            4*at*bt*cb*mi*zb*cos(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 -
            mi**4*zt**3) - 2*at*bt*cb*mi*zt*cos(mi*zb)/(mi**4*zb**3 - 3*mi**4*zb**2*zt +
            3*mi**4*zb*zt**2 - mi**4*zt**3) - 4*at*bt*cb*mi*zt*cos(mi*zt)/(mi**4*zb**3 -
            3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) - 6*at*bt*cb*sin(mi*zb)/(mi**4*zb**3
            - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) +
            6*at*bt*cb*sin(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3)
            + at*bt*ct*mi**3*zb**3*cos(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 -
            mi**4*zt**3) - 3*at*bt*ct*mi**3*zb**2*zt*cos(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt +
            3*mi**4*zb*zt**2 - mi**4*zt**3) + 3*at*bt*ct*mi**3*zb*zt**2*cos(mi*zt)/(mi**4*zb**3 -
            3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) -
            at*bt*ct*mi**3*zt**3*cos(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 -
            mi**4*zt**3) + 3*at*bt*ct*mi**2*zb**2*sin(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt +
            3*mi**4*zb*zt**2 - mi**4*zt**3) - 6*at*bt*ct*mi**2*zb*zt*sin(mi*zt)/(mi**4*zb**3 -
            3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) +
            3*at*bt*ct*mi**2*zt**2*sin(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 -
            mi**4*zt**3) - 6*at*bt*ct*mi*zb*cos(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt +
            3*mi**4*zb*zt**2 - mi**4*zt**3) + 6*at*bt*ct*mi*zt*cos(mi*zt)/(mi**4*zb**3 -
            3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) + 6*at*bt*ct*sin(mi*zb)/(mi**4*zb**3
            - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3) -
            6*at*bt*ct*sin(mi*zt)/(mi**4*zb**3 - 3*mi**4*zb**2*zt + 3*mi**4*zb*zt**2 - mi**4*zt**3), axis=1)


    return A

def dim1sin_D_aDb_linear(m, at, ab, bt, bb, zt, zb, implementation='vectorized'):
    """Create matrix of spectral integrations

    Performs integrations of `sin(mi * z) * D[a(z) * D[b(z), z], z]`
    between [0, 1] where a(z) and b(z) are piecewise linear functions of z.
    Calulation of integrals is performed at each element of a 1d array
    (size depends on size of `m`)

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    at : ``list`` of ``float``
        property at top of each layer
    ab : ``list`` of ``float``
        property at bottom of each layer
    bt : ``list`` of ``float``
        2nd property at top of each layer
    bb : ``list`` of ``float``
        2nd property at bottom of each layer
    zt : ``list`` of ``float``
        normalised depth or z-coordinate at top of each layer. `zt[0]` = 0
    zb : ``list`` of ``float``
        normalised depth or z-coordinate at bottom of each layer. `zt[-1]` = 1

    Returns
    -------
    A : numpy.ndarray
        returns a 1d array size determined by size of `m`. treat as column
        vector

    See Also
    --------
    m_from_sin_mx : used to generate 'm' input parameter
    geotecha.integrals_generate_code.dim1sin_D_aDb_linear : use sympy
    to perform the integrals symbolically and generate expressions for
    this function

    Notes
    -----
    The `dim1sin_D_aDb_linear` which should be treated as a column vector,
    :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{\\frac{d}{dz}\\left({a\\left(z\\right)}\\frac{d}{dz}{b\\left(z\\right)}\\right)\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    ..math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` and :math:`b\\left(z\\right)` are piecewise
    linear functions w.r.t. :math:`z`, that within a layer are defined by:

    ..math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """


        #import numpy as np #import this at module level
    #import math #import this at module level

    m = np.asarray(m)
    at = np.asarray(at)
    ab = np.asarray(ab)
    bt = np.asarray(bt)
    bb = np.asarray(bb)
    zt = np.asarray(zt)
    zb = np.asarray(zb)

    neig = len(m)

    if implementation == 'scalar':
        sin = math.sin
        cos = math.cos
        A = np.zeros(neig, float)
        nlayers = len(zt)
        for layer in range(nlayers):
            for i in range(neig):
                A[i] += (-(zb[layer] - zt[layer])**(-2)*m[i]**(-1)*(bb[layer] - bt[layer])*(ab[layer] -
                    at[layer])*cos(m[i]*zb[layer]) + (zb[layer] -
                    zt[layer])**(-2)*m[i]**(-1)*(bb[layer] - bt[layer])*(ab[layer] -
                    at[layer])*cos(m[i]*zt[layer]) - (zb[layer] - zt[layer])**(-1)*(bb[layer] -
                    bt[layer])*((zb[layer] - zt[layer])**(-1)*(ab[layer] - at[layer])*(zb[layer] -
                    zt[layer]) + at[layer])*sin(m[i]*zb[layer]) + (zb[layer] -
                    zt[layer])**(-1)*(bb[layer] - bt[layer])*at[layer]*sin(m[i]*zt[layer]))

        for i in range(neig):
            A[i] += ((zb[-1] - zt[-1])**(-1)*(bb[-1] - bt[-1])*((zb[-1] - zt[-1])**(-1)*(ab[-1] - at[-1])*(zb[-1] -
                zt[-1]) + at[-1])*sin(m[i]*zb[-1]) - (zb[0] - zt[0])**(-1)*(bb[0] -
                bt[0])*at[0]*sin(m[i]*zt[0]))
    elif implementation == 'fortran':
        import geotecha.speccon.ext_integrals as ext_integ
        A = ext_integ.dim1sin_d_adb_linear(m, at, ab, bt, bb, zt, zb)
#        try:
#            import geotecha.speccon.ext_integrals as ext_integ
#            A = ext_integ.dim1sin_d_adb_linear(m, at, ab, bt, bb, zt, zb)
#        except ImportError:
#            A = dim1sin_D_aDb_linear(m, at, ab, bt, bb, zt, zb, implementation='vectorized')

    else:#default is 'vectorized' using numpy
        sin = np.sin
        cos = np.cos
        A = np.zeros(neig, float)

        mi = m[:, np.newaxis]
        A[:] = np.sum(-ab*(bb - bt)*sin(mi*zb)/(zb - zt) + at*(bb - bt)*sin(mi*zt)/(zb - zt) - (ab - at)*(bb -
            bt)*cos(mi*zb)/(mi*(zb - zt)**2) + (ab - at)*(bb - bt)*cos(mi*zt)/(mi*(zb - zt)**2), axis=1)
        mi = m
        A[:]+= ((zb[-1] - zt[-1])**(-1)*(bb[-1] - bt[-1])*((zb[-1] - zt[-1])**(-1)*(ab[-1] - at[-1])*(zb[-1] -
            zt[-1]) + at[-1])*sin(mi*zb[-1]) - (zb[0] - zt[0])**(-1)*(bb[0] -
            bt[0])*at[0]*sin(mi*zt[0]))
    return A

if __name__=='__main__':
#    eigs = np.array([2.46740110027, 22.2066099025])
#    dic = {'loadtim': np.array([0,0]), 'loadmag': np.array([0, 100]), 'eigs': eigs, 'tvals': np.array([-1,0,1])}
#    #print(EDload_linear(**dic))
#    #
#    dic = {'loadtim': np.array([0,0,10]), 'loadmag': np.array([0, -100,-100]), 'eigs': eigs, 'tvals': np.array([-1,0,1])}
#    #print (Eload_linear(**dic))


    import math
#    dic = {'m': [ math.pi/2,  3 * math.pi/2], 'at':[1],
#        'ab':[1],'zt':[0], 'zb':[1], 'z': [[0, 1], [0.1, 0.2], [0.5, 1]]}
#     print(dim1sin_a_linear_between(**dic))
    n=100
#    dic = {'m': m_from_sin_mx(np.arange(n),boundary=1),
#           'at':[1], 'ab':[1],'zt':[0], 'zb':[1],
#        'implementation': 'scalar'}

    dic = {'m': m_from_sin_mx(np.arange(n),boundary=1),
           'at':[1,2,1], 'ab':[2,1,2],'zt':[0,0.4,0.5], 'zb':[0.4,0.5,1],
        'implementation': 'fortran'}


    print(dim1sin_af_linear(**dic))