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
    
"""

from __future__ import division
import numpy as np

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
    
def dim1sin_af_linear(m, at, ab, zt, zb):
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
    geotecha.integrals_generate_code.generate_dim1sin_af_linear : use sympy 
    to perform the integrasl symbolically and generate expressions for 
    this function
    
    Notes
    -----
    The `dim1sin_af_linear' matrix, :math:`\\mathbf{A}` is given by:
        
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
            A[i, i] += -(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*ab[layer]*cos(m[i]*zb[layer])**2 + (4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*ab[layer]*cos(m[i]*zt[layer])**2 + (4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*cos(m[i]*zb[layer])**2 - (4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*cos(m[i]*zt[layer])**2 - 2*m[i]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) + 2*m[i]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) + 2*m[i]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) - 2*m[i]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) + m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]**2*sin(m[i]*zb[layer])**2 + m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]**2*cos(m[i]*zb[layer])**2 - m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]**2*sin(m[i]*zt[layer])**2 - m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]**2*cos(m[i]*zt[layer])**2 - m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]**2*sin(m[i]*zb[layer])**2 - m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]**2*cos(m[i]*zb[layer])**2 + m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]**2*sin(m[i]*zt[layer])**2 + m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]**2*cos(m[i]*zt[layer])**2 - 2*zb[layer]*m[i]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) + 2*zb[layer]*m[i]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) + 2*zb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]*sin(m[i]*zb[layer])**2 + 2*zb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*zb[layer]*cos(m[i]*zb[layer])**2 - 2*zb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]*sin(m[i]*zt[layer])**2 - 2*zb[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*at[layer]*zt[layer]*cos(m[i]*zt[layer])**2 + 2*zt[layer]*m[i]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*ab[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer]) - 2*zt[layer]*m[i]*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*ab[layer]*cos(m[i]*zt[layer])*sin(m[i]*zt[layer]) - 2*zt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]*sin(m[i]*zb[layer])**2 - 2*zt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zb[layer]*cos(m[i]*zb[layer])**2 + 2*zt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]*sin(m[i]*zt[layer])**2 + 2*zt[layer]*m[i]**2*(4*zb[layer]*m[i]**2 - 4*zt[layer]*m[i]**2)**(-1)*ab[layer]*zt[layer]*cos(m[i]*zt[layer])**2
        for i in range(neig-1):
            for j in range(i + 1, neig):
                A[i, j] += m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*ab[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) - m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*ab[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) - m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*at[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) + m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*at[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) - m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) + m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) + m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) - m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) + 2*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) - 2*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) - 2*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) + 2*m[j]*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) + m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) - m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) - m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) + m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) + m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*ab[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) - m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*ab[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) - m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*at[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) + m[j]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*at[layer]*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) + m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*ab[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) - m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*ab[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) - m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*at[layer]*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) + m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*at[layer]*zt[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) - m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*ab[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) + m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*ab[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) + m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*at[layer]*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) - m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*at[layer]*zt[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) - zb[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) + zb[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) + zb[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) - zb[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) + zb[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) - zb[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) - zb[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) + zb[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*at[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) + zt[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) - zt[layer]*m[i]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) - zt[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) + zt[layer]*m[j]*m[i]**2*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) - zt[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) + zt[layer]*m[j]**2*m[i]*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) + zt[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) - zt[layer]*m[j]**3*(zb[layer]*m[i]**4 - 2*zb[layer]*m[j]**2*m[i]**2 + zb[layer]*m[j]**4 - zt[layer]*m[i]**4 + 2*zt[layer]*m[j]**2*m[i]**2 - zt[layer]*m[j]**4)**(-1)*ab[layer]*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])                
                
    #A is symmetric
    for i in range(neig - 1):        
        for j in range(i + 1, neig):
            A[j, i] = A[i, j]                
    
    return A