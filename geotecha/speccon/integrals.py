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
"""Integrals and eigenvalues for consolidation using spectral methods.

Most of the functions have been generated using
geotecha.speccon.integrals_generate_code functions using sympy.

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
        Eigenvalue number in series to return.
    boundary : {0, 1}, optional
        Boundary condition.
        For 'Pervious Top Pervious Bottom (PTPB)', boundary = 0.
        For 'Pervious Top Impervious Bottom (PTIB)', boundary = 1.
        Default boundary=0.

    Returns
    -------
    out : ``float``
        The `i` th eigenvalue

    """

    from math import pi

    if boundary not in {0, 1}:
        raise ValueError('boundary = {}; must be 0 or 1.'.format(boundary))

    return pi * (i + 1 - boundary / 2.0)

def pdim1sin_af_linear(m, a, **kwargs):
    """Integration of sin(mi * z) * a(z) * sin(mj * z)
    between ztop and zbot where a(z) is piecewise linear.

    Wrapper for dim1sin_af_linear with PolyLine input.

    Calculation of integrals is performed at each element of a square symmetric
    matrix (size depends on size of `m`)


    Parameters
    ----------
    m : ``list`` of ``float``
       Eeigenvlaues of BVP. Generate with geoteca.speccon.m_from_sin_mx.
    a : PolyLine object
        PolyLine defining piecewise linear relationship.
    implementation : ['vectorized', 'scalar', 'fortran'], optional
        Functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        Default implementation='vectorized'.  If fortran extention module
        cannot be imported then 'vectorized' version will be used.
        If anything other than 'fortran' or 'scalar' is used then
        default vectorized version will be used.

    Returns
    -------
    A : numpy.ndarray
        A square symmetric matrix, size determined by size of `m`

    See Also
    --------
    m_from_sin_mx : Used to generate 'm' input parameter.
    dim1sin_af_linear : Wrapped function.
        inputs.

    Notes
    -----
    The `dim1sin_af_linear` matrix, :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{{a\\left(z\\right)}\\phi_i\\phi_j\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    .. math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` is a piecewise linear function
    with respect to :math:`z`, that within a layer are defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.


    See also
    --------
    dim1sin_af_linear

    """

    return dim1sin_af_linear(m, a.y1, a.y2, a.x1, a.x2, **kwargs)


def pdim1sin_abf_linear(m, a, b, **kwargs):
    """Integration of sin(mi * z) * a(z) * a(z) * sin(mj * z)
    between ztop and zbot where a(z) is piecewise linear.

    Wrapper for dim1sin_abf_linear to allow PolyLine inputs.


    Calulation of integrals is performed at each element of a square symmetric
    matrix (size depends on size of `m`).

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    a, b : PolyLine object
        PolyLine defining piecewise linear relationship.
    implementation : ['vectorized', 'scalar', 'fortran'], optional
        Functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        Default implementation='vectorized'.  If fortran extention module
        cannot be imported then 'vectorized' version will be used.
        If anything other than 'fortran' or 'scalar' is used then
        default vectorized version will be used.

    Returns
    -------
    A : numpy.ndarray
        A square symmetric matrix, size determined by size of `m`.

    See Also
    -------
    dim1sin_abf_linear : Wrapped function.


    Notes
    -----
    The `dim1sin_abf_linear` matrix, :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{{a\\left(z\\right)}{b\\left(z\\right)}\\phi_i\\phi_j\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    .. math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` and :math:`b\\left(z\\right)` are piecewise
    linear functions with respect to :math:`z`, that within a layer are defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """
    a, b = pwise.polyline_make_x_common(a,b)
    return dim1sin_abf_linear(m, a.y1, a.y2, b.y1, b.y2, a.x1, a.x2, **kwargs)


def pdim1sin_D_aDf_linear(m, a, **kwargs):
    """Integration of sin(mi * z) * D[a(z) * D[sin(mj * z),z],z]
    between ztop and zbot where a(z) is piecewise linear functions of z.


    Wrapper for dim1sin_D_aDf_linear to allow PolyLine inputs.


    Calulation of integrals is performed at each element of a square symmetric
    matrix (size depends on size of `m`)

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    a : PolyLine object
        PolyLine defining piecewise linear relationship.
    implementation : ['vectorized', 'scalar', 'fortran'], optional
        Functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        Default implementation='vectorized'.  If fortran extention module
        cannot be imported then 'vectorized' version will be used.
        If anything other than 'fortran' or 'scalar' is used then
        default vectorized version will be used.

    Returns
    -------
    A : numpy.ndarray
        A square symmetric matrix, size determined by size of `m`.

    See Also
    --------
    dim1sin_D_aDf_linear : Wrapped function.


    Notes
    -----
    The `dim1sin_D_aDf_linear` matrix, :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{\\frac{d}{dz}\\left({a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\right)\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    .. math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` is a piecewise
    linear functions with respect to :math:`z`, that within a layer is defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    To make the above integration simpler we integate by parts to get:

    .. math:: \\mathbf{A}_{i,j}= \\left.\\phi_i{a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\right|_{z=0}^{z=1} -\\int_{0}^1{{a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\frac{d\\phi_i}{dz}\\,dz}

    In this case the sine basis functions means the left term in the above
    equation is zero, leaving us with

    .. math:: \\mathbf{A}_{i,j}= -\\int_{0}^1{{a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\frac{d\\phi_i}{dz}\\,dz}


    """

    return dim1sin_D_aDf_linear(m, a.y1, a.y2, a.x1, a.x2, **kwargs)


def pdim1sin_ab_linear(m, a, b, **kwargs):
    """Integration of sin(mi * z) * a(z) * b(z)
    between ztop and zbot where a(z) and b(z) are piecewise linear functions of z.

    Wrapper for dim1sin_ab_linear to allow PolyLine inputs.


    Calulation of integrals is performed at each element of a 1d array
    (size depends on size of `m`).

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    a, b : PolyLine object
        PolyLine defining piecewise linear relationship.
    implementation : ['vectorized', 'scalar', 'fortran'], optional
        Functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        Default implementation='vectorized'.  If fortran extention module
        cannot be imported then 'vectorized' version will be used.
        If anything other than 'fortran' or 'scalar' is used then
        default vectorized version will be used.

    Returns
    -------
    A : numpy.ndarray
        A 1d array size determined by size of `m`. Treat as column
        vector.

    See Also
    --------
    dim1sin_ab_linear : Wrapped function.


    Notes
    -----
    The `dim1sin_ab_linear` which should be treated as a column vector,
    :math:`A` is given by:

    .. math:: \\mathbf{A}_{i}=\\int_{0}^1{{a\\left(z\\right)}{b\\left(z\\right)}\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    .. math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` and :math:`b\\left(z\\right)` are piecewise
    linear functions with respect to :math:`z`, that within a layer are defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """
    a, b = pwise.polyline_make_x_common(a,b)
    return dim1sin_ab_linear(m, a.y1, a.y2, b.y1, b.y2, a.x1, a.x2, **kwargs)


def pdim1sin_abc_linear(m, a, b, c, **kwargs):
    """Integrations of sin(mi * z) * a(z) * b(z) * c(z)
    between ztop and zbot where a(z), b(z), c(z) are piecewise linear functions of z.

    Wrapper for dim1sin_abc_linear to allow PolyLine inputs.


    Calulation of integrals is performed at each element of a 1d array
    (size depends on size of `m`).

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    a, b, c : PolyLine object
        PolyLine defining piecewise linear relationship.
    implementation : ['vectorized', 'scalar', 'fortran'], optional
        Functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        Default implementation='vectorized'.  If fortran extention module
        cannot be imported then 'vectorized' version will be used.
        If anything other than 'fortran' or 'scalar' is used then
        default vectorized version will be used.

    Returns
    -------
    A : numpy.ndarray
        A 1d array size determined by size of `m`. Treat as column
        vector.

    See Also
    --------
    dim1sin_abc_linear : Wrapped function.



    Notes
    -----
    The `dim1sin_abc_linear` which should be treated as a column vector,
    :math:`A` is given by:

    .. math:: \\mathbf{A}_{i}=\\int_{0}^1{{a\\left(z\\right)}{b\\left(z\\right)}{c\\left(z\\right)}\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    .. math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)`, :math:`b\\left(z\\right)`, and
    :math:`c\\left(z\\right)` are piecewise linear functions
    with respect to :math:`z`, that within a layer are defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """
    a, b, c = pwise.polyline_make_x_common(a,b,c)
    return dim1sin_abc_linear(m, a.y1, a.y2, b.y1, b.y2,c.y1, c.y2, a.x1, a.x2, **kwargs)


def pdim1sin_D_aDb_linear(m, a, b, **kwargs):
    """Integrations of `sin(mi * z) * D[a(z) * D[b(z), z], z]`
    between ztop and zbot where a(z) is a piecewise linear function of z,
    and b(z) is a linear function of z.

    Wrapper for dim1sin_D_aDb_linear to allow PolyLine inputs.

    Calulation of integrals is performed at each element of a 1d array
    (size depends on size of `m`).

    .. warning::
        The functions produced are set up to accept the b(z) input as
        piecewise linear, i.e. b.x1, b.x2, b.y1, b.y2 etc. It is up to the user to
        ensure that the b.x1 and b.x2 are such that they define a continuous
        linear function. eg. to define b(z)=z+1 then use
        b.x1=[0,0.4], b.x2=[0.4, 1], b.y1=[1,1.4], b.y2=[1.4,2].


    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    a, b : PolyLine object
        PolyLine defining piecewise linear relationship.
    implementation : ['vectorized', 'scalar', 'fortran'], optional
        Functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        Default implementation='vectorized'.  If fortran extention module
        cannot be imported then 'vectorized' version will be used.
        If anything other than 'fortran' or 'scalar' is used then
        default vectorized version will be used.

    Returns
    -------
    A : numpy.ndarray
        A 1d array size determined by size of `m`. Treat as column
        vector.

    See Also
    --------
    dim1sin_D_aDb_linear : Wrapped function.


    Notes
    -----
    The `dim1sin_D_aDb_linear` which should be treated as a column vector,
    :math:`A` is given by:

    .. math:: \\mathbf{A}_{i}=\\int_{0}^1{\\frac{d}{dz}\\left({a\\left(z\\right)}\\frac{d}{dz}{b\\left(z\\right)}\\right)\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    .. math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` is a piecewise
    linear functions with respect to :math:`z`, that within a layer is defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    :math:`b\\left(z\\right)` is a linear function of :math:`z` defined by

    .. math:: b\\left(z\\right) = b_t+\\left({b_b-b_t}\\right)z

    with :math:`t` and :math:`b` subscripts now representing 'top' and
    'bottom' of the profile respectively.

    Using the product rule for differentiation the above integral can be split
    into:

    .. math:: \\mathbf{A}_{i}=\\int_{0}^1{\\frac{da\\left(z\\right)}{dz}\\frac{db\\left(z\\right)}{dz}\\phi_i\\,dz} +
                              \\int_{0}^1{a\\left(z\\right)\\frac{d^2b\\left(z\\right)}{dz^2}\\phi_i\\,dz}

    The right hand term is zero because :math:`b\\left(z\\right)` is a
    continuous linear function so it's second derivative is zero.  The
    first derivative of :math:`b\\left(z\\right)` is a constant so the
    left term can be integrated by parts to give:

    .. math:: \\mathbf{A}_{i}=\\frac{db\\left(z\\right)}{dz}\\left(
                \\left.\\phi_i{a\\left(z\\right)}\\right|_{z=0}^{z=1} -
                -\\int_{0}^1{{a\\left(z\\right)}\\frac{d\\phi_i}{dz}\\,dz}
                \\right)



    """
    a, b = pwise.polyline_make_x_common(a,b)
    return dim1sin_D_aDb_linear(m, a.y1, a.y2, b.y1, b.y2, a.x1, a.x2, **kwargs)


def pEDload_linear(a, eigs, tvals, dT=1.0, **kwargs):
    """Integration of D[load(tau), tau] * exp(dT * eig * (t-tau)) between
    [0, t], where load(tau) is piecewise linear.

    Wrapper for EDload_linear to allow PolyLine inputs.

    Performs integrations involving a piecewise linear load.  A 2d array of
    dimensions A[len(tvals), len(eigs)]
    is produced where the 'i'th row of A contains the diagonal elements of the
    spectral 'E' matrix calculated for the time value tvals[i]. i.e. rows of
    this matrix will be assembled into the diagonal matrix 'E' elsewhere.

    Parameters
    ----------
    a : PolyLine
        PolyLine object representing piecewise linear load vs time.
    eigs : 1d numpy.ndarray
        List of eigenvalues.
    tvals : 1d numpy.ndarray`
        List of time values to calculate integral at.
    dT : ``float``, optional
        Time factor multiple (Default dT=1.0).
    implementation : ['vectorized', 'scalar', 'fortran'], optional
        Functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        Default implementation='vectorized'.  If fortran extention module
        cannot be imported then 'vectorized' version will be used.
        If anything other than 'fortran' or 'scalar' is used then
        default vectorized version will be used.


    Returns
    -------
    A : numpy.ndarray
        A 2d array of dimesnions A[len(tvals), len(eigs)].
        The 'i'th row of A is the diagonal elements of the spectral 'E' matrix
        calculated for the time tvals[i].


    See Also
    --------
    EDload_linear : Wrapped function.


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
    `EDload_linear` is from loading terms in the governing equation that are NOT
    differentiated wrt :math:`t`.
    The diagonal elements of :math:`E` are given by:

    .. math:: \\mathbf{E}_{i,i}=\\int_{0}^t{\\frac{d{\\sigma\\left(\\tau\\right)}}{d\\tau}{\\exp\\left({(dT\\left(t-\\tau\\right)\\lambda_i}\\right)}\\,d\\tau}

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

    return EDload_linear(a.x, a.y, eigs, tvals, dT, **kwargs)


def pEload_linear(a, eigs, tvals, dT=1.0, **kwargs):
    """Integration of load(tau) * exp(dT * eig * (t-tau)) between [0, t], where
    load(tau) is piecewise linear.

    Wrapper for Eload_linear to allow PolyLine inputs.

    Performs integrations involving a piecewise linear load.  A 2d array of
    dimensions A[len(tvals), len(eigs)]
    is produced where the 'i'th row of A contains the diagonal elements of the
    spectral 'E' matrix calculated for the time value tvals[i]. i.e. rows of
    this matrix will be assembled into the diagonal matrix 'E' elsewhere.

    Parameters
    ----------
    a : PolyLine
        PolyLine object representing piecewise linear load vs time.
    eigs : 1d numpy.ndarray
        List of eigenvalues.
    tvals : 1d numpy.ndarray`
        List of time values to calculate integral at.
    dT : ``float``, optional
        Time factor multiple (Default dT=1.0).
    implementation : ['vectorized', 'scalar', 'fortran'], optional
        Functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        Default implementation='vectorized'.  If fortran extention module
        cannot be imported then 'vectorized' version will be used.
        If anything other than 'fortran' or 'scalar' is used then
        default vectorized version will be used.


    Returns
    -------
    A : numpy.ndarray
        A 2d array of dimesnions A[len(tvals), len(eigs)].
        The 'i'th row of A is the diagonal elements of the spectral 'E' matrix
        calculated for the time tvals[i].


    See Also
    --------
    Eload_linear : Wrapped function, see for equations and other related
        functions.

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

    (note that this function,`Eload_linear`, rather than use :math:`t_s` and
    :math:`t_f`,
    explicitly finds increments that the current time falls in, falls after,
    and falls before and treates each case on it's own.)

    Each :math:`t` value of interest requires a separate diagonal matrix
    :math:`E`.  To use space more efficiently and to facilitate numpy
    broadcasting when using the results of the function, the diagonal elements
    of :math:`E` for each time value `t` value are stored in the rows of
    array :math:`A` returned by `Eload_linear`.  Thus:

    .. math:: \\mathbf{A}=\\left(\\begin{matrix}E_{0,0}(t_0)&E_{1,1}(t_0)& \cdots & E_{neig-1,neig-1}(t_0)\\\ E_{0,0}(t_1)&E_{1,1}(t_1)& \\cdots & E_{neig-1,neig-1}(t_1)\\\ \\vdots&\\vdots&\\ddots&\\vdots \\\ E_{0,0}(t_m)&E_{1,1}(t_m)& \cdots & E_{neig-1,neig-1}(t_m)\\end{matrix}\\right)

    """

    return Eload_linear(a.x, a.y, eigs, tvals, dT, **kwargs)


def pEDload_coslinear(a, omega, phase, eigs, tvals, dT=1.0, **kwargs):
    """Integration of D[cos(omega*tau+phase)*load(tau), tau] * exp(dT * eig * (t-tau)) between [0, t], where
    load(tau) is piecewise linear.

    Wrapper for EDload_coslinear to allow PolyLine inputs.

    Performs integrations involving a piecewise linear load.  A 2d array of
    dimensions A[len(tvals), len(eigs)]
    is produced where the 'i'th row of A contains the diagonal elements of the
    spectral 'E' matrix calculated for the time value tvals[i]. i.e. rows of
    this matrix will be assembled into the diagonal matrix 'E' elsewhere.

    Parameters
    ----------
    a : PolyLine
        PolyLine object representing piecewise linear load vs time.
        List of load magnitudes.
    omega, phase : float
        Parameters that describe a cyclic load cos(omega * t + phase).
    eigs : 1d numpy.ndarray
        List of eigenvalues.
    tvals : 1d numpy.ndarray`
        List of time values to calculate integral at.
    dT : ``float``, optional
        Time factor multiple (Default dT=1.0).
    implementation : ['vectorized', 'scalar', 'fortran'], optional
        Functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        Default implementation='vectorized'.  If fortran extention module
        cannot be imported then 'vectorized' version will be used.
        If anything other than 'fortran' or 'scalar' is used then
        default vectorized version will be used.


    Returns
    -------
    A : numpy.ndarray
        A 2d array of dimesnions A[len(tvals), len(eigs)].
        The 'i'th row of A is the diagonal elements of the spectral 'E' matrix
        calculated for the time tvals[i].


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
    `EDload_coslinear` is from loading terms in the governing equation that are NOT
    differentiated wrt :math:`t`.
    The diagonal elements of :math:`E` are given by:

    .. math:: \\mathbf{E}_{i,i}=\\int_{0}^t{\\frac{d{{\\cos\\left(\\omega\\tau+\\textrm{phase}\\right)}\\sigma\\left(\\tau\\right)}}{d\\tau}{\\exp\\left({(dT\\left(t-\\tau\\right)\\lambda_i}\\right)}\\,d\\tau}

    where

     :math:`\\lambda_i` is the `ith` eigenvalue of the problem,
     :math:`dT` is a time factor for numerical convienience,
     :math:`\\sigma\left(\\tau\\right)` is the time dependant portion of the loading function.

    When the time dependant loading term :math:`\\sigma\\left(\\tau\\right)` is
    piecewise in time. The contribution of each load segment is found by:

    .. math:: \\mathbf{E}_{i,i}=\\int_{t_s}^{t_f}{\\frac{d{{\\cos\\left(\\omega\\tau+\\textrm{phase}\\right)}\\sigma\\left(\\tau\\right)}}{d\\tau}\\exp\\left({dT\\left(t-\\tau\\right)*\\lambda_i}\\right)\\,d\\tau}

    where

    .. math:: t_s = \\min\\left(t,t_{increment\\:start}\\right)

    .. math:: t_f = \\min\\left(t,t_{increment\\:end}\\right)

    (note that this function,`EDload_coslinear`, rather than use :math:`t_s` and
    :math:`t_f`,
    explicitly finds increments that the current time falls in, falls after,
    and falls before and treates each case on it's own.)

    Each :math:`t` value of interest requires a separate diagonal matrix
    :math:`E`.  To use space more efficiently and to facilitate numpy
    broadcasting when using the results of the function, the diagonal elements
    of :math:`E` for each time value `t` value are stored in the rows of
    array :math:`A` returned by `EDload_coslinear`.  Thus:

    .. math:: \\mathbf{A}=\\left(\\begin{matrix}E_{0,0}(t_0)&E_{1,1}(t_0)& \cdots & E_{neig-1,neig-1}(t_0)\\\ E_{0,0}(t_1)&E_{1,1}(t_1)& \\cdots & E_{neig-1,neig-1}(t_1)\\\ \\vdots&\\vdots&\\ddots&\\vdots \\\ E_{0,0}(t_m)&E_{1,1}(t_m)& \cdots & E_{neig-1,neig-1}(t_m)\\end{matrix}\\right)

    See Also
    --------
    pEDload_coslinear : Same function with PolyLine inputs.

    """

    return EDload_coslinear(a.x, a.y, omega, phase, eigs, tvals, dT, **kwargs)


def pEload_coslinear(a, omega, phase, eigs, tvals, dT=1.0, **kwargs):
    """Integration of cos(omega*tau+phase)*load(tau) * exp(dT * eig * (t-tau)) between [0, t], where
    load(tau) is piecewise linear.


    Wrapper for Eload_coslinear to allow PolyLine inputs.


    Performs integrations involving a piecewise linear load.  A 2d array of
    dimensions A[len(tvals), len(eigs)]
    is produced where the 'i'th row of A contains the diagonal elements of the
    spectral 'E' matrix calculated for the time value tvals[i]. i.e. rows of
    this matrix will be assembled into the diagonal matrix 'E' elsewhere.

    Parameters
    ----------
    a : PolyLine
        PolyLine object representing piecewise linear load vs time.
    omega, phase : float
        Parameters that describe a cyclic load cos(omega * t + phase).
    eigs : 1d numpy.ndarray
        List of eigenvalues.
    tvals : 1d numpy.ndarray`
        List of time values to calculate integral at.
    dT : ``float``, optional
        Time factor multiple (Default dT=1.0).
    implementation : ['vectorized', 'scalar', 'fortran'], optional
        Functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        Default implementation='vectorized'.  If fortran extention module
        cannot be imported then 'vectorized' version will be used.
        If anything other than 'fortran' or 'scalar' is used then
        default vectorized version will be used.


    Returns
    -------
    A : numpy.ndarray
        A 2d array of dimesnions A[len(tvals), len(eigs)].
        The 'i'th row of A is the diagonal elements of the spectral 'E' matrix
        calculated for the time tvals[i].


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
    `Eload_coslinear` is from loading terms in the governing equation that are NOT
    differentiated wrt :math:`t`.
    The diagonal elements of :math:`E` are given by:

    .. math:: \\mathbf{E}_{i,i}=\\int_{0}^t{{\\cos\\left(\\omega\\tau+\\textrm{phase}\\right)}{\\sigma\\left(\\tau\\right)}{\exp\\left({(dT\\left(t-\\tau\\right)\\lambda_i}\\right)}\\,d\\tau}

    where

     :math:`\\lambda_i` is the `ith` eigenvalue of the problem,
     :math:`dT` is a time factor for numerical convienience,
     :math:`\\sigma\left(\\tau\\right)` is the time dependant portion of the loading function.

    When the time dependant loading term :math:`\\sigma\\left(\\tau\\right)` is
    piecewise in time. The contribution of each load segment is found by:

    .. math:: \\mathbf{E}_{i,i}=\\int_{t_s}^{t_f}{{\\cos\\left(\\omega\\tau+\\textrm{phase}\\right)}{\\sigma\\left(\\tau\\right)}\\exp\\left({dT\\left(t-\\tau\\right)*\\lambda_i}\\right)\\,d\\tau}

    where

    .. math:: t_s = \\min\\left(t,t_{increment\\:start}\\right)

    .. math:: t_f = \\min\\left(t,t_{increment\\:end}\\right)

    (note that this function,`Eload_coslinear`, rather than use :math:`t_s` and
    :math:`t_f`,
    explicitly finds increments that the current time falls in, falls after,
    and falls before and treates each case on it's own.)

    Each :math:`t` value of interest requires a separate diagonal matrix
    :math:`E`.  To use space more efficiently and to facilitate numpy
    broadcasting when using the results of the function, the diagonal elements
    of :math:`E` for each time value `t` value are stored in the rows of
    array :math:`A` returned by `Eload_coslinear`.  Thus:

    .. math:: \\mathbf{A}=\\left(\\begin{matrix}E_{0,0}(t_0)&E_{1,1}(t_0)& \cdots & E_{neig-1,neig-1}(t_0)\\\ E_{0,0}(t_1)&E_{1,1}(t_1)& \\cdots & E_{neig-1,neig-1}(t_1)\\\ \\vdots&\\vdots&\\ddots&\\vdots \\\ E_{0,0}(t_m)&E_{1,1}(t_m)& \cdots & E_{neig-1,neig-1}(t_m)\\end{matrix}\\right)

    See Also
    --------
    Eload_coslinear : Wrapped function.

    """

    return Eload_coslinear(a.x, a.y, omega, phase, eigs, tvals, dT, **kwargs)


def dim1sin(m, z):
    """sin(m*z) for each combination of m and z

    Calculates array A[len(z), len(m)]

    Parameters
    ----------
    m : ``list`` of ``float``
        Eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    z : ``list`` of ``float``
        Normalised depth or z-coordinate should be between 0 and 1.

    Returns
    -------
    A : numpy.ndarray
        A 1d array size A[len(z), len(m)].

    See Also
    --------
    m_from_sin_mx : used to generate 'm' input parameter

    Notes
    -----
    .. math:: \\mathbf{A}_{i,j}=\\sin\\left({m_j, z_i}\\right)

    """

    m = np.asarray(m)
    z = np.asarray(z)

    return np.sin(z[:, np.newaxis]*m[np.newaxis, :])


def dim1sin_avg_between(m, z):
    """Average of sin(m * z) between z1 and z2

    Integrate(sin(m*z), (z, z1, z2))/(z2-z1) for each combination of m
    and [z1,z2].

    Calculates array A[len(z), len(m)].

    Parameters
    ----------
    m : ``list`` of ``float``
        Eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    z : ``list`` of ``list`` of float``
        Normalised depth or z-coordinate pair should be two values
        between 0 and 1..

    Returns
    -------
    A : numpy.ndarray
        A 1d array size A[len(z), len(m)]

    See Also
    --------
    m_from_sin_mx : used to generate 'm' input parameter

    Notes
    -----
    .. math:: \\mathbf{A}_{i,j}=\\frac{1}{z_{2i}-z_{1i}}\\int_{z_{1i}}^{z_{2i}}{\\sin\\left({m_j, z}\\right)\\,dz}

    """

    from numpy import sin, cos
    m = np.asarray(m)
    z = np.asarray(z)

    a = z[:,0][:,np.newaxis]
    b = z[:,1][:,np.newaxis]
    mm = m[np.newaxis,:]

    return (cos(a*mm)-cos(b*mm))/mm/(b-a)


def pdim1sin_a_linear_between(m, a, z, **kwargs):
    """Integrations of `sin(mi * z) * a(z)`
    between [z1, z2] where a(z) is a piecewise linear functions of z.

    Wrapper for dim1sin_a_linear_between to allow PolyLine inputs.

    Calculates array A[len(z), len(m)].


    Parameters
    ----------
    m : ``list`` of ``float``
        Eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    a : PolyLine object
        PolyLine defining piecewise linear relationship.
    z : ``list`` of ``list`` of float``
        Normalised depth or z-coordinate pair should be two values
        between 0 and 1.

    Returns
    -------
    A : numpy.ndarray
        A 2d array size of size [len(z), len(m)].

    See Also
    --------
    dim1sin_a_linear_between : Wrapped function.


    Notes
    -----
    The `dim1sin_a_linear_between`, :math:`A`, is given by:

    .. math:: \\mathbf{A}_{i,j}=
                \\int_{z_1}^{z_2}{{a\\left(z\\right)}\\phi_j\\,dz}

    where the basis function :math:`\\phi_j` is given by:

    .. math:: \\phi_j\\left(z\\right)=\\sin\\left({m_j}z\\right)

    and :math:`a\\left(z\\right)` is a piecewise
    linear functions with respect to :math:`z`, that within a layer are defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """

    return dim1sin_a_linear_between(m, a.y1, a.y2, a.x1, a.x2, z, **kwargs)


def dim1sin_a_linear_between(m, at, ab, zt, zb, z):
    """Integrations of `sin(mi * z) * a(z)`
    between [z1, z2] where a(z) is a piecewise linear functions of z.

    Calculates array A[len(z), len(m)].


    Parameters
    ----------
    m : ``list`` of ``float``
        Eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    at : ``list`` of ``float``
        Property at top of each layer.
    ab : ``list`` of ``float``
        Property at bottom of each layer.
    zt : ``list`` of ``float``
        Normalised depth or z-coordinate at top of each layer. `zt[0]` = 0
    zb : ``list`` of ``float``
        Normalised depth or z-coordinate at bottom of each layer. `zt[-1]` = 1
    z : ``list`` of ``list`` of float``
        Normalised depth or z-coordinate pair should be two values
        between 0 and 1.

    Returns
    -------
    A : numpy.ndarray
        A 2d array size of size [len(z), len(m)].

    See Also
    --------
    m_from_sin_mx : Used to generate 'm' input parameter.
    geotecha.speccon.integrals_generate_code.dim1sin_a_linear_between : Generation
        of code for this function.
    pdim1sin_a_linear_between : Same function with PolyLine
        inputs.
    geotecha.piecewise.piecewise_linear_1d.segments_between_xi_and_xj : Used in the
        function.

    Notes
    -----
    The `dim1sin_a_linear_between`, :math:`A`, is given by:

    .. math:: \\mathbf{A}_{i,j}=
                \\int_{z_1}^{z_2}{{a\\left(z\\right)}\\phi_j\\,dz}

    where the basis function :math:`\\phi_j` is given by:

    .. math:: \\phi_j\\left(z\\right)=\\sin\\left({m_j}z\\right)

    and :math:`a\\left(z\\right)` is a piecewise
    linear functions with respect to :math:`z`, that within a layer are defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.


    """


    #import numpy as np #import this globally
    #import math #import this globally

    sin=math.sin
    cos=math.cos
    m = np.asarray(m)
    at = np.asarray(at)
    ab = np.asarray(ab)
    zt = np.asarray(zt)
    zb = np.asarray(zb)

    z = np.atleast_2d(z)

    z1 = z[:,0]
    z2 = z[:,1]

    z_for_interp = np.zeros(len(zt)+1)
    z_for_interp[:-1] = zt[:]
    z_for_interp[-1]=zb[-1]


    (segment_both,
     segment_z1_only,
     segment_z2_only,
     segments_between) = segments_between_xi_and_xj(z_for_interp, z1, z2)

    nz = len(z)
    neig = len(m)

    A = np.zeros((nz,neig), dtype=float)
    for i in range(nz):
        for layer in segment_both[i]:
            a_slope = (ab[layer] - at[layer]) / (zb[layer] - zt[layer])
            for j in range(neig):
                A[i,j] += (-a_slope*m[j]**(-2)*sin(m[j]*z1[i]) + a_slope*m[j]**(-2)*sin(m[j]*z2[i]) +
                    a_slope*m[j]**(-1)*z1[i]*cos(m[j]*z1[i]) -
                    a_slope*m[j]**(-1)*z2[i]*cos(m[j]*z2[i]) -
                    a_slope*zt[layer]*m[j]**(-1)*cos(m[j]*z1[i]) +
                    a_slope*zt[layer]*m[j]**(-1)*cos(m[j]*z2[i]) +
                    at[layer]*m[j]**(-1)*cos(m[j]*z1[i]) - at[layer]*m[j]**(-1)*cos(m[j]*z2[i]))
        for layer in segment_z1_only[i]:
            a_slope = (ab[layer] - at[layer]) / (zb[layer] - zt[layer])
            for j in range(neig):
                A[i,j] += (-a_slope*m[j]**(-2)*sin(m[j]*z1[i]) + a_slope*m[j]**(-2)*sin(zb[layer]*m[j]) +
                    a_slope*m[j]**(-1)*z1[i]*cos(m[j]*z1[i]) -
                    a_slope*zb[layer]*m[j]**(-1)*cos(zb[layer]*m[j]) -
                    a_slope*zt[layer]*m[j]**(-1)*cos(m[j]*z1[i]) +
                    a_slope*zt[layer]*m[j]**(-1)*cos(zb[layer]*m[j]) +
                    at[layer]*m[j]**(-1)*cos(m[j]*z1[i]) - at[layer]*m[j]**(-1)*cos(zb[layer]*m[j]))
        for layer in segments_between[i]:
            a_slope = (ab[layer] - at[layer]) / (zb[layer] - zt[layer])
            for j in range(neig):
                A[i,j] += (a_slope*m[j]**(-2)*sin(zb[layer]*m[j]) - a_slope*m[j]**(-2)*sin(zt[layer]*m[j]) -
                    a_slope*zb[layer]*m[j]**(-1)*cos(zb[layer]*m[j]) +
                    a_slope*zt[layer]*m[j]**(-1)*cos(zb[layer]*m[j]) -
                    at[layer]*m[j]**(-1)*cos(zb[layer]*m[j]) +
                    at[layer]*m[j]**(-1)*cos(zt[layer]*m[j]))
        for layer in segment_z2_only[i]:
            a_slope = (ab[layer] - at[layer]) / (zb[layer] - zt[layer])
            for j in range(neig):
                A[i,j] += (a_slope*m[j]**(-2)*sin(m[j]*z2[i]) - a_slope*m[j]**(-2)*sin(zt[layer]*m[j]) -
                    a_slope*m[j]**(-1)*z2[i]*cos(m[j]*z2[i]) +
                    a_slope*zt[layer]*m[j]**(-1)*cos(m[j]*z2[i]) -
                    at[layer]*m[j]**(-1)*cos(m[j]*z2[i]) + at[layer]*m[j]**(-1)*cos(zt[layer]*m[j]))
    return A


def dim1sin_af_linear(m, at, ab, zt, zb, implementation='vectorized'):
    """Integration of sin(mi * z) * a(z) * sin(mj * z)
    between ztop and zbot where a(z) is piecewise linear.


    Calculation of integrals is performed at each element of a square symmetric
    matrix (size depends on size of `m`)

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    at : ``list`` of ``float``
        Property at top of each layer.
    ab : ``list`` of ``float``
        Property at bottom of each layer.
    zt : ``list`` of ``float``
        Normalised depth or z-coordinate at top of each layer. `zt[0]` = 0
    zb : ``list`` of ``float``
        Normalised depth or z-coordinate at bottom of each layer. `zt[-1]` = 1
    implementation : ['vectorized', 'scalar', 'fortran'], optional
        Functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        Default implementation='vectorized'.  If fortran extention module
        cannot be imported then 'vectorized' version will be used.
        If anything other than 'fortran' or 'scalar' is used then
        default vectorized version will be used.

    Returns
    -------
    A : numpy.ndarray
        A square symmetric matrix, size determined by size of `m`.

    See Also
    --------
    m_from_sin_mx : Used to generate 'm' input parameter.
    geotecha.integrals_generate_code.dim1sin_af_linear : Generation
        of code for this function.
    pdim1sin_af_linear : Same function with PolyLine
        inputs.
    geotecha.speccon.ext_integrals.dim1sin_af_linear : Equivalent fortran
        function.

    Notes
    -----
    The `dim1sin_af_linear` matrix, :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{{a\\left(z\\right)}\\phi_i\\phi_j\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    .. math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` is a piecewise linear function
    with respect to :math:`z`, that within a layer are defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.




    """
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
            a_slope = (ab[layer] - at[layer]) / (zb[layer] - zt[layer])
            for i in range(neig):
                A[i, i] += (a_slope*m[i]**(-2)*sin(m[i]*zb[layer])**2/4 - a_slope*m[i]**(-2)*sin(m[i]*zt[layer])**2/4 +
                    a_slope*m[i]**(-1)*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])*zt[layer]/2 -
                    a_slope*m[i]**(-1)*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])/2 -
                    a_slope*zb[layer]*sin(m[i]*zb[layer])**2*zt[layer]/2 -
                    a_slope*zb[layer]*cos(m[i]*zb[layer])**2*zt[layer]/2 +
                    a_slope*zb[layer]**2*sin(m[i]*zb[layer])**2/4 +
                    a_slope*zb[layer]**2*cos(m[i]*zb[layer])**2/4 +
                    a_slope*zt[layer]**2*sin(m[i]*zt[layer])**2/4 +
                    a_slope*zt[layer]**2*cos(m[i]*zt[layer])**2/4 -
                    m[i]**(-1)*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])*at[layer]/2 +
                    m[i]**(-1)*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])*at[layer]/2 +
                    zb[layer]*sin(m[i]*zb[layer])**2*at[layer]/2 +
                    zb[layer]*cos(m[i]*zb[layer])**2*at[layer]/2 -
                    zt[layer]*sin(m[i]*zt[layer])**2*at[layer]/2 -
                    zt[layer]*cos(m[i]*zt[layer])**2*at[layer]/2)
            for i in range(neig-1):
                for j in range(i + 1, neig):
                    A[i, j] += (a_slope*m[i]**2*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        a_slope*m[i]**2*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        a_slope*m[i]**3*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*zt[layer] -
                        a_slope*m[i]**3*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        2*a_slope*m[j]*m[i]*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) -
                        2*a_slope*m[j]*m[i]*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) -
                        a_slope*m[j]*m[i]**2*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*zt[layer] +
                        a_slope*m[j]*m[i]**2*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                        a_slope*m[j]**2*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        a_slope*m[j]**2*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        a_slope*m[j]**2*m[i]*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*zt[layer] +
                        a_slope*m[j]**2*m[i]*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        a_slope*m[j]**3*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*zt[layer] -
                        a_slope*m[j]**3*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                        m[i]**3*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*at[layer] +
                        m[i]**3*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])*at[layer] +
                        m[j]*m[i]**2*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*at[layer] -
                        m[j]*m[i]**2*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])*at[layer] +
                        m[j]**2*m[i]*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*at[layer] -
                        m[j]**2*m[i]*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])*at[layer] -
                        m[j]**3*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*at[layer] +
                        m[j]**3*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])*at[layer])

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

        a_slope = (ab - at) / (zb - zt)

        mi = m[:, np.newaxis]
        A[diag] = np.sum(-a_slope*zb*sin(mi*zb)**2*zt/2 - a_slope*zb*cos(mi*zb)**2*zt/2 + a_slope*zb**2*sin(mi*zb)**2/4 +
            a_slope*zb**2*cos(mi*zb)**2/4 + a_slope*zt**2*sin(mi*zt)**2/4 +
            a_slope*zt**2*cos(mi*zt)**2/4 + a_slope*cos(mi*zb)*sin(mi*zb)*zt/(2*mi) -
            a_slope*zb*cos(mi*zb)*sin(mi*zb)/(2*mi) + a_slope*sin(mi*zb)**2/(4*mi**2) -
            a_slope*sin(mi*zt)**2/(4*mi**2) + zb*sin(mi*zb)**2*at/2 + zb*cos(mi*zb)**2*at/2 -
            zt*sin(mi*zt)**2*at/2 - zt*cos(mi*zt)**2*at/2 - cos(mi*zb)*sin(mi*zb)*at/(2*mi) +
            cos(mi*zt)*sin(mi*zt)*at/(2*mi), axis=1)

        mi = m[triu[0]][:, np.newaxis]
        mj = m[triu[1]][:, np.newaxis]
        A[triu] = np.sum(a_slope*mi**3*cos(mi*zb)*sin(mj*zb)*zt/(mi**4 - 2*mi**2*mj**2 + mj**4) -
            a_slope*mi**3*zb*cos(mi*zb)*sin(mj*zb)/(mi**4 - 2*mi**2*mj**2 + mj**4) -
            a_slope*mi**2*mj*cos(mj*zb)*sin(mi*zb)*zt/(mi**4 - 2*mi**2*mj**2 + mj**4) +
            a_slope*mi**2*mj*zb*cos(mj*zb)*sin(mi*zb)/(mi**4 - 2*mi**2*mj**2 + mj**4) +
            a_slope*mi**2*sin(mi*zb)*sin(mj*zb)/(mi**4 - 2*mi**2*mj**2 + mj**4) -
            a_slope*mi**2*sin(mi*zt)*sin(mj*zt)/(mi**4 - 2*mi**2*mj**2 + mj**4) -
            a_slope*mi*mj**2*cos(mi*zb)*sin(mj*zb)*zt/(mi**4 - 2*mi**2*mj**2 + mj**4) +
            a_slope*mi*mj**2*zb*cos(mi*zb)*sin(mj*zb)/(mi**4 - 2*mi**2*mj**2 + mj**4) +
            2*a_slope*mi*mj*cos(mi*zb)*cos(mj*zb)/(mi**4 - 2*mi**2*mj**2 + mj**4) -
            2*a_slope*mi*mj*cos(mi*zt)*cos(mj*zt)/(mi**4 - 2*mi**2*mj**2 + mj**4) +
            a_slope*mj**3*cos(mj*zb)*sin(mi*zb)*zt/(mi**4 - 2*mi**2*mj**2 + mj**4) -
            a_slope*mj**3*zb*cos(mj*zb)*sin(mi*zb)/(mi**4 - 2*mi**2*mj**2 + mj**4) +
            a_slope*mj**2*sin(mi*zb)*sin(mj*zb)/(mi**4 - 2*mi**2*mj**2 + mj**4) -
            a_slope*mj**2*sin(mi*zt)*sin(mj*zt)/(mi**4 - 2*mi**2*mj**2 + mj**4) -
            mi**3*cos(mi*zb)*sin(mj*zb)*at/(mi**4 - 2*mi**2*mj**2 + mj**4) +
            mi**3*cos(mi*zt)*sin(mj*zt)*at/(mi**4 - 2*mi**2*mj**2 + mj**4) +
            mi**2*mj*cos(mj*zb)*sin(mi*zb)*at/(mi**4 - 2*mi**2*mj**2 + mj**4) -
            mi**2*mj*cos(mj*zt)*sin(mi*zt)*at/(mi**4 - 2*mi**2*mj**2 + mj**4) +
            mi*mj**2*cos(mi*zb)*sin(mj*zb)*at/(mi**4 - 2*mi**2*mj**2 + mj**4) -
            mi*mj**2*cos(mi*zt)*sin(mj*zt)*at/(mi**4 - 2*mi**2*mj**2 + mj**4) -
            mj**3*cos(mj*zb)*sin(mi*zb)*at/(mi**4 - 2*mi**2*mj**2 + mj**4) +
            mj**3*cos(mj*zt)*sin(mi*zt)*at/(mi**4 - 2*mi**2*mj**2 + mj**4), axis=1)
        #A is symmetric
        A[tril] = A[triu]

    return A


def dim1sin_abf_linear(m, at, ab, bt, bb,  zt, zb, implementation='vectorized'):
    """Integration of sin(mi * z) * a(z) * a(z) * sin(mj * z)
    between ztop and zbot where a(z) is piecewise linear.

    Calulation of integrals is performed at each element of a square symmetric
    matrix (size depends on size of `m`).

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    at : ``list`` of ``float``
        Property at top of each layer.
    ab : ``list`` of ``float``
        Property at bottom of each layer.
    bt : ``list`` of ``float``
        2nd property at top of each layer.
    bb : ``list`` of ``float``
        2nd property at bottom of each layer.
    zt : ``list`` of ``float``
        Normalised depth or z-coordinate at top of each layer. `zt[0]` = 0
    zb : ``list`` of ``float``
        Normalised depth or z-coordinate at bottom of each layer. `zt[-1]` = 1
    implementation : ['vectorized', 'scalar', 'fortran'], optional
        Functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        Default implementation='vectorized'.  If fortran extention module
        cannot be imported then 'vectorized' version will be used.
        If anything other than 'fortran' or 'scalar' is used then
        default vectorized version will be used.

    Returns
    -------
    A : numpy.ndarray
        A square symmetric matrix, size determined by size of `m`.

    See Also
    --------
    m_from_sin_mx : Used to generate 'm' input parameter.
    geotecha.integrals_generate_code.dim1sin_abf_linear : Generation
        of code for this function.
    pdim1sin_abf_linear : Same function with PolyLine
        inputs.
    geotecha.speccon.ext_integrals.dim1sin_abf_linear : Equivalent fortran
        function.

    Notes
    -----
    The `dim1sin_abf_linear` matrix, :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{{a\\left(z\\right)}{b\\left(z\\right)}\\phi_i\\phi_j\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    .. math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` and :math:`b\\left(z\\right)` are piecewise
    linear functions with respect to :math:`z`, that within a layer are defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

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
            a_slope = (ab[layer] - at[layer]) / (zb[layer] - zt[layer])
            b_slope = (bb[layer] - bt[layer]) / (zb[layer] - zt[layer])
            for i in range(neig):
                A[i, i] += (a_slope*b_slope*m[i]**(-3)*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])/4 -
                    a_slope*b_slope*m[i]**(-3)*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])/4 -
                    a_slope*b_slope*m[i]**(-2)*sin(m[i]*zb[layer])**2*zt[layer]/2 +
                    a_slope*b_slope*m[i]**(-2)*zb[layer]*sin(m[i]*zb[layer])**2/4 -
                    a_slope*b_slope*m[i]**(-2)*zb[layer]*cos(m[i]*zb[layer])**2/4 +
                    a_slope*b_slope*m[i]**(-2)*zt[layer]*sin(m[i]*zt[layer])**2/4 +
                    a_slope*b_slope*m[i]**(-2)*zt[layer]*cos(m[i]*zt[layer])**2/4 -
                    a_slope*b_slope*m[i]**(-1)*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])*zt[layer]**2/2
                    +
                    a_slope*b_slope*m[i]**(-1)*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])*zt[layer]
                    -
                    a_slope*b_slope*m[i]**(-1)*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])/2
                    + a_slope*b_slope*zb[layer]*sin(m[i]*zb[layer])**2*zt[layer]**2/2 +
                    a_slope*b_slope*zb[layer]*cos(m[i]*zb[layer])**2*zt[layer]**2/2 -
                    a_slope*b_slope*zb[layer]**2*sin(m[i]*zb[layer])**2*zt[layer]/2 -
                    a_slope*b_slope*zb[layer]**2*cos(m[i]*zb[layer])**2*zt[layer]/2 +
                    a_slope*b_slope*zb[layer]**3*sin(m[i]*zb[layer])**2/6 +
                    a_slope*b_slope*zb[layer]**3*cos(m[i]*zb[layer])**2/6 -
                    a_slope*b_slope*zt[layer]**3*sin(m[i]*zt[layer])**2/6 -
                    a_slope*b_slope*zt[layer]**3*cos(m[i]*zt[layer])**2/6 +
                    a_slope*m[i]**(-2)*sin(m[i]*zb[layer])**2*bt[layer]/4 -
                    a_slope*m[i]**(-2)*sin(m[i]*zt[layer])**2*bt[layer]/4 +
                    a_slope*m[i]**(-1)*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])*zt[layer]*bt[layer]/2
                    -
                    a_slope*m[i]**(-1)*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])*bt[layer]/2
                    - a_slope*zb[layer]*sin(m[i]*zb[layer])**2*zt[layer]*bt[layer]/2 -
                    a_slope*zb[layer]*cos(m[i]*zb[layer])**2*zt[layer]*bt[layer]/2 +
                    a_slope*zb[layer]**2*sin(m[i]*zb[layer])**2*bt[layer]/4 +
                    a_slope*zb[layer]**2*cos(m[i]*zb[layer])**2*bt[layer]/4 +
                    a_slope*zt[layer]**2*sin(m[i]*zt[layer])**2*bt[layer]/4 +
                    a_slope*zt[layer]**2*cos(m[i]*zt[layer])**2*bt[layer]/4 +
                    b_slope*m[i]**(-2)*sin(m[i]*zb[layer])**2*at[layer]/4 -
                    b_slope*m[i]**(-2)*sin(m[i]*zt[layer])**2*at[layer]/4 +
                    b_slope*m[i]**(-1)*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])*zt[layer]*at[layer]/2
                    -
                    b_slope*m[i]**(-1)*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])*at[layer]/2
                    - b_slope*zb[layer]*sin(m[i]*zb[layer])**2*zt[layer]*at[layer]/2 -
                    b_slope*zb[layer]*cos(m[i]*zb[layer])**2*zt[layer]*at[layer]/2 +
                    b_slope*zb[layer]**2*sin(m[i]*zb[layer])**2*at[layer]/4 +
                    b_slope*zb[layer]**2*cos(m[i]*zb[layer])**2*at[layer]/4 +
                    b_slope*zt[layer]**2*sin(m[i]*zt[layer])**2*at[layer]/4 +
                    b_slope*zt[layer]**2*cos(m[i]*zt[layer])**2*at[layer]/4 -
                    m[i]**(-1)*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])*bt[layer]*at[layer]/2 +
                    m[i]**(-1)*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])*bt[layer]*at[layer]/2 +
                    zb[layer]*sin(m[i]*zb[layer])**2*bt[layer]*at[layer]/2 +
                    zb[layer]*cos(m[i]*zb[layer])**2*bt[layer]*at[layer]/2 -
                    zt[layer]*sin(m[i]*zt[layer])**2*bt[layer]*at[layer]/2 -
                    zt[layer]*cos(m[i]*zt[layer])**2*bt[layer]*at[layer]/2)
            for i in range(neig-1):
                for j in range(i + 1, neig):
                    A[i, j] += (2*a_slope*b_slope*m[i]**3*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        2*a_slope*b_slope*m[i]**3*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) -
                        2*a_slope*b_slope*m[i]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])*zt[layer] +
                        2*a_slope*b_slope*m[i]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        a_slope*b_slope*m[i]**5*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*zt[layer]**2 +
                        2*a_slope*b_slope*m[i]**5*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*zt[layer] -
                        a_slope*b_slope*m[i]**5*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        6*a_slope*b_slope*m[j]*m[i]**2*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                        6*a_slope*b_slope*m[j]*m[i]**2*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) -
                        4*a_slope*b_slope*m[j]*m[i]**3*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])*zt[layer] +
                        4*a_slope*b_slope*m[j]*m[i]**3*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) +
                        a_slope*b_slope*m[j]*m[i]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*zt[layer]**2 -
                        2*a_slope*b_slope*m[j]*m[i]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*zt[layer] +
                        a_slope*b_slope*m[j]*m[i]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                        6*a_slope*b_slope*m[j]**2*m[i]*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        6*a_slope*b_slope*m[j]**2*m[i]*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        2*a_slope*b_slope*m[j]**2*m[i]**3*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*zt[layer]**2 -
                        4*a_slope*b_slope*m[j]**2*m[i]**3*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*zt[layer] +
                        2*a_slope*b_slope*m[j]**2*m[i]**3*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        2*a_slope*b_slope*m[j]**3*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                        2*a_slope*b_slope*m[j]**3*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[j]*zt[layer])*sin(m[i]*zt[layer]) +
                        4*a_slope*b_slope*m[j]**3*m[i]*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])*zt[layer] -
                        4*a_slope*b_slope*m[j]**3*m[i]*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) -
                        2*a_slope*b_slope*m[j]**3*m[i]**2*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*zt[layer]**2 +
                        4*a_slope*b_slope*m[j]**3*m[i]**2*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*zt[layer] -
                        2*a_slope*b_slope*m[j]**3*m[i]**2*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                        2*a_slope*b_slope*m[j]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])*zt[layer] -
                        2*a_slope*b_slope*m[j]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) -
                        a_slope*b_slope*m[j]**4*m[i]*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*zt[layer]**2 +
                        2*a_slope*b_slope*m[j]**4*m[i]*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*zt[layer] -
                        a_slope*b_slope*m[j]**4*m[i]*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]**2*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        a_slope*b_slope*m[j]**5*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*zt[layer]**2 -
                        2*a_slope*b_slope*m[j]**5*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*zt[layer] +
                        a_slope*b_slope*m[j]**5*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]**2*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                        a_slope*m[i]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])*bt[layer] -
                        a_slope*m[i]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])*bt[layer] +
                        a_slope*m[i]**5*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*zt[layer]*bt[layer] -
                        a_slope*m[i]**5*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*bt[layer] +
                        2*a_slope*m[j]*m[i]**3*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])*bt[layer] -
                        2*a_slope*m[j]*m[i]**3*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])*bt[layer] -
                        a_slope*m[j]*m[i]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*zt[layer]*bt[layer] +
                        a_slope*m[j]*m[i]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*bt[layer] -
                        2*a_slope*m[j]**2*m[i]**3*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*zt[layer]*bt[layer] +
                        2*a_slope*m[j]**2*m[i]**3*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*bt[layer] -
                        2*a_slope*m[j]**3*m[i]*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])*bt[layer] +
                        2*a_slope*m[j]**3*m[i]*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])*bt[layer] +
                        2*a_slope*m[j]**3*m[i]**2*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*zt[layer]*bt[layer] -
                        2*a_slope*m[j]**3*m[i]**2*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*bt[layer] -
                        a_slope*m[j]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])*bt[layer] +
                        a_slope*m[j]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])*bt[layer] +
                        a_slope*m[j]**4*m[i]*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*zt[layer]*bt[layer] -
                        a_slope*m[j]**4*m[i]*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*bt[layer] -
                        a_slope*m[j]**5*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*zt[layer]*bt[layer] +
                        a_slope*m[j]**5*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*bt[layer] +
                        b_slope*m[i]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])*at[layer] -
                        b_slope*m[i]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])*at[layer] +
                        b_slope*m[i]**5*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*zt[layer]*at[layer] -
                        b_slope*m[i]**5*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*at[layer] +
                        2*b_slope*m[j]*m[i]**3*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])*at[layer] -
                        2*b_slope*m[j]*m[i]**3*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])*at[layer] -
                        b_slope*m[j]*m[i]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*zt[layer]*at[layer] +
                        b_slope*m[j]*m[i]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*at[layer] -
                        2*b_slope*m[j]**2*m[i]**3*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*zt[layer]*at[layer] +
                        2*b_slope*m[j]**2*m[i]**3*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*at[layer] -
                        2*b_slope*m[j]**3*m[i]*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zb[layer])*cos(m[j]*zb[layer])*at[layer] +
                        2*b_slope*m[j]**3*m[i]*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zt[layer])*cos(m[j]*zt[layer])*at[layer] +
                        2*b_slope*m[j]**3*m[i]**2*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*zt[layer]*at[layer] -
                        2*b_slope*m[j]**3*m[i]**2*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*at[layer] -
                        b_slope*m[j]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*sin(m[i]*zb[layer])*sin(m[j]*zb[layer])*at[layer] +
                        b_slope*m[j]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*sin(m[i]*zt[layer])*sin(m[j]*zt[layer])*at[layer] +
                        b_slope*m[j]**4*m[i]*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*zt[layer]*at[layer] -
                        b_slope*m[j]**4*m[i]*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*at[layer] -
                        b_slope*m[j]**5*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*zt[layer]*at[layer] +
                        b_slope*m[j]**5*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*at[layer] -
                        m[i]**5*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*bt[layer]*at[layer] +
                        m[i]**5*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])*bt[layer]*at[layer] +
                        m[j]*m[i]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*bt[layer]*at[layer] -
                        m[j]*m[i]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])*bt[layer]*at[layer] +
                        2*m[j]**2*m[i]**3*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*bt[layer]*at[layer] -
                        2*m[j]**2*m[i]**3*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])*bt[layer]*at[layer] -
                        2*m[j]**3*m[i]**2*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*bt[layer]*at[layer] +
                        2*m[j]**3*m[i]**2*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])*bt[layer]*at[layer] -
                        m[j]**4*m[i]*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*bt[layer]*at[layer] +
                        m[j]**4*m[i]*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])*bt[layer]*at[layer] +
                        m[j]**5*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*bt[layer]*at[layer] -
                        m[j]**5*(m[i]**6 - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 -
                        m[j]**6)**(-1)*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])*bt[layer]*at[layer])

        #A is symmetric
        for i in range(neig - 1):
            for j in range(i + 1, neig):
                A[j, i] = A[i, j]

    elif implementation == 'fortran':
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

        a_slope = (ab - at) / (zb - zt)
        b_slope = (bb - bt) / (zb - zt)

        mi = m[:, np.newaxis]
        A[diag] = np.sum(a_slope*b_slope*zb*sin(mi*zb)**2*zt**2/2 + a_slope*b_slope*zb*cos(mi*zb)**2*zt**2/2 -
            a_slope*b_slope*zb**2*sin(mi*zb)**2*zt/2 - a_slope*b_slope*zb**2*cos(mi*zb)**2*zt/2 +
            a_slope*b_slope*zb**3*sin(mi*zb)**2/6 + a_slope*b_slope*zb**3*cos(mi*zb)**2/6 -
            a_slope*b_slope*zt**3*sin(mi*zt)**2/6 - a_slope*b_slope*zt**3*cos(mi*zt)**2/6 -
            a_slope*b_slope*cos(mi*zb)*sin(mi*zb)*zt**2/(2*mi) +
            a_slope*b_slope*zb*cos(mi*zb)*sin(mi*zb)*zt/mi -
            a_slope*b_slope*zb**2*cos(mi*zb)*sin(mi*zb)/(2*mi) -
            a_slope*b_slope*sin(mi*zb)**2*zt/(2*mi**2) + a_slope*b_slope*zb*sin(mi*zb)**2/(4*mi**2)
            - a_slope*b_slope*zb*cos(mi*zb)**2/(4*mi**2) +
            a_slope*b_slope*zt*sin(mi*zt)**2/(4*mi**2) + a_slope*b_slope*zt*cos(mi*zt)**2/(4*mi**2)
            + a_slope*b_slope*cos(mi*zb)*sin(mi*zb)/(4*mi**3) -
            a_slope*b_slope*cos(mi*zt)*sin(mi*zt)/(4*mi**3) - a_slope*zb*sin(mi*zb)**2*zt*bt/2 -
            a_slope*zb*cos(mi*zb)**2*zt*bt/2 + a_slope*zb**2*sin(mi*zb)**2*bt/4 +
            a_slope*zb**2*cos(mi*zb)**2*bt/4 + a_slope*zt**2*sin(mi*zt)**2*bt/4 +
            a_slope*zt**2*cos(mi*zt)**2*bt/4 + a_slope*cos(mi*zb)*sin(mi*zb)*zt*bt/(2*mi) -
            a_slope*zb*cos(mi*zb)*sin(mi*zb)*bt/(2*mi) + a_slope*sin(mi*zb)**2*bt/(4*mi**2) -
            a_slope*sin(mi*zt)**2*bt/(4*mi**2) - b_slope*zb*sin(mi*zb)**2*zt*at/2 -
            b_slope*zb*cos(mi*zb)**2*zt*at/2 + b_slope*zb**2*sin(mi*zb)**2*at/4 +
            b_slope*zb**2*cos(mi*zb)**2*at/4 + b_slope*zt**2*sin(mi*zt)**2*at/4 +
            b_slope*zt**2*cos(mi*zt)**2*at/4 + b_slope*cos(mi*zb)*sin(mi*zb)*zt*at/(2*mi) -
            b_slope*zb*cos(mi*zb)*sin(mi*zb)*at/(2*mi) + b_slope*sin(mi*zb)**2*at/(4*mi**2) -
            b_slope*sin(mi*zt)**2*at/(4*mi**2) + zb*sin(mi*zb)**2*bt*at/2 + zb*cos(mi*zb)**2*bt*at/2
            - zt*sin(mi*zt)**2*bt*at/2 - zt*cos(mi*zt)**2*bt*at/2 -
            cos(mi*zb)*sin(mi*zb)*bt*at/(2*mi) + cos(mi*zt)*sin(mi*zt)*bt*at/(2*mi), axis=1)

        mi = m[triu[0]][:, np.newaxis]
        mj = m[triu[1]][:, np.newaxis]
        A[triu] = np.sum(-a_slope*b_slope*mi**5*cos(mi*zb)*sin(mj*zb)*zt**2/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            2*a_slope*b_slope*mi**5*zb*cos(mi*zb)*sin(mj*zb)*zt/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - a_slope*b_slope*mi**5*zb**2*cos(mi*zb)*sin(mj*zb)/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            a_slope*b_slope*mi**4*mj*cos(mj*zb)*sin(mi*zb)*zt**2/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - 2*a_slope*b_slope*mi**4*mj*zb*cos(mj*zb)*sin(mi*zb)*zt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            a_slope*b_slope*mi**4*mj*zb**2*cos(mj*zb)*sin(mi*zb)/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - 2*a_slope*b_slope*mi**4*sin(mi*zb)*sin(mj*zb)*zt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            2*a_slope*b_slope*mi**4*zb*sin(mi*zb)*sin(mj*zb)/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4
            - mj**6) + 2*a_slope*b_slope*mi**3*mj**2*cos(mi*zb)*sin(mj*zb)*zt**2/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) -
            4*a_slope*b_slope*mi**3*mj**2*zb*cos(mi*zb)*sin(mj*zb)*zt/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) +
            2*a_slope*b_slope*mi**3*mj**2*zb**2*cos(mi*zb)*sin(mj*zb)/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - 4*a_slope*b_slope*mi**3*mj*cos(mi*zb)*cos(mj*zb)*zt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            4*a_slope*b_slope*mi**3*mj*zb*cos(mi*zb)*cos(mj*zb)/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + 2*a_slope*b_slope*mi**3*cos(mi*zb)*sin(mj*zb)/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) -
            2*a_slope*b_slope*mi**3*cos(mi*zt)*sin(mj*zt)/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) - 2*a_slope*b_slope*mi**2*mj**3*cos(mj*zb)*sin(mi*zb)*zt**2/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            4*a_slope*b_slope*mi**2*mj**3*zb*cos(mj*zb)*sin(mi*zb)*zt/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) -
            2*a_slope*b_slope*mi**2*mj**3*zb**2*cos(mj*zb)*sin(mi*zb)/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - 6*a_slope*b_slope*mi**2*mj*cos(mj*zb)*sin(mi*zb)/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            6*a_slope*b_slope*mi**2*mj*cos(mj*zt)*sin(mi*zt)/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4
            - mj**6) - a_slope*b_slope*mi*mj**4*cos(mi*zb)*sin(mj*zb)*zt**2/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + 2*a_slope*b_slope*mi*mj**4*zb*cos(mi*zb)*sin(mj*zb)*zt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) -
            a_slope*b_slope*mi*mj**4*zb**2*cos(mi*zb)*sin(mj*zb)/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + 4*a_slope*b_slope*mi*mj**3*cos(mi*zb)*cos(mj*zb)*zt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) -
            4*a_slope*b_slope*mi*mj**3*zb*cos(mi*zb)*cos(mj*zb)/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + 6*a_slope*b_slope*mi*mj**2*cos(mi*zb)*sin(mj*zb)/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) -
            6*a_slope*b_slope*mi*mj**2*cos(mi*zt)*sin(mj*zt)/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4
            - mj**6) + a_slope*b_slope*mj**5*cos(mj*zb)*sin(mi*zb)*zt**2/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - 2*a_slope*b_slope*mj**5*zb*cos(mj*zb)*sin(mi*zb)*zt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            a_slope*b_slope*mj**5*zb**2*cos(mj*zb)*sin(mi*zb)/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4
            - mj**6) + 2*a_slope*b_slope*mj**4*sin(mi*zb)*sin(mj*zb)*zt/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - 2*a_slope*b_slope*mj**4*zb*sin(mi*zb)*sin(mj*zb)/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) -
            2*a_slope*b_slope*mj**3*cos(mj*zb)*sin(mi*zb)/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) + 2*a_slope*b_slope*mj**3*cos(mj*zt)*sin(mi*zt)/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + a_slope*mi**5*cos(mi*zb)*sin(mj*zb)*zt*bt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) -
            a_slope*mi**5*zb*cos(mi*zb)*sin(mj*zb)*bt/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) - a_slope*mi**4*mj*cos(mj*zb)*sin(mi*zb)*zt*bt/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + a_slope*mi**4*mj*zb*cos(mj*zb)*sin(mi*zb)*bt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) + a_slope*mi**4*sin(mi*zb)*sin(mj*zb)*bt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) - a_slope*mi**4*sin(mi*zt)*sin(mj*zt)*bt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) -
            2*a_slope*mi**3*mj**2*cos(mi*zb)*sin(mj*zb)*zt*bt/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4
            - mj**6) + 2*a_slope*mi**3*mj**2*zb*cos(mi*zb)*sin(mj*zb)*bt/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + 2*a_slope*mi**3*mj*cos(mi*zb)*cos(mj*zb)*bt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) -
            2*a_slope*mi**3*mj*cos(mi*zt)*cos(mj*zt)*bt/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) + 2*a_slope*mi**2*mj**3*cos(mj*zb)*sin(mi*zb)*zt*bt/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - 2*a_slope*mi**2*mj**3*zb*cos(mj*zb)*sin(mi*zb)*bt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            a_slope*mi*mj**4*cos(mi*zb)*sin(mj*zb)*zt*bt/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) - a_slope*mi*mj**4*zb*cos(mi*zb)*sin(mj*zb)*bt/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - 2*a_slope*mi*mj**3*cos(mi*zb)*cos(mj*zb)*bt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            2*a_slope*mi*mj**3*cos(mi*zt)*cos(mj*zt)*bt/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) - a_slope*mj**5*cos(mj*zb)*sin(mi*zb)*zt*bt/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + a_slope*mj**5*zb*cos(mj*zb)*sin(mi*zb)*bt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) - a_slope*mj**4*sin(mi*zb)*sin(mj*zb)*bt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) + a_slope*mj**4*sin(mi*zt)*sin(mj*zt)*bt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            b_slope*mi**5*cos(mi*zb)*sin(mj*zb)*zt*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) - b_slope*mi**5*zb*cos(mi*zb)*sin(mj*zb)*at/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - b_slope*mi**4*mj*cos(mj*zb)*sin(mi*zb)*zt*at/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            b_slope*mi**4*mj*zb*cos(mj*zb)*sin(mi*zb)*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) + b_slope*mi**4*sin(mi*zb)*sin(mj*zb)*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) - b_slope*mi**4*sin(mi*zt)*sin(mj*zt)*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) - 2*b_slope*mi**3*mj**2*cos(mi*zb)*sin(mj*zb)*zt*at/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + 2*b_slope*mi**3*mj**2*zb*cos(mi*zb)*sin(mj*zb)*at/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            2*b_slope*mi**3*mj*cos(mi*zb)*cos(mj*zb)*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) - 2*b_slope*mi**3*mj*cos(mi*zt)*cos(mj*zt)*at/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + 2*b_slope*mi**2*mj**3*cos(mj*zb)*sin(mi*zb)*zt*at/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) -
            2*b_slope*mi**2*mj**3*zb*cos(mj*zb)*sin(mi*zb)*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4
            - mj**6) + b_slope*mi*mj**4*cos(mi*zb)*sin(mj*zb)*zt*at/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - b_slope*mi*mj**4*zb*cos(mi*zb)*sin(mj*zb)*at/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) -
            2*b_slope*mi*mj**3*cos(mi*zb)*cos(mj*zb)*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) + 2*b_slope*mi*mj**3*cos(mi*zt)*cos(mj*zt)*at/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - b_slope*mj**5*cos(mj*zb)*sin(mi*zb)*zt*at/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            b_slope*mj**5*zb*cos(mj*zb)*sin(mi*zb)*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) - b_slope*mj**4*sin(mi*zb)*sin(mj*zb)*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) + b_slope*mj**4*sin(mi*zt)*sin(mj*zt)*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) - mi**5*cos(mi*zb)*sin(mj*zb)*bt*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) + mi**5*cos(mi*zt)*sin(mj*zt)*bt*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) + mi**4*mj*cos(mj*zb)*sin(mi*zb)*bt*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) - mi**4*mj*cos(mj*zt)*sin(mi*zt)*bt*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) + 2*mi**3*mj**2*cos(mi*zb)*sin(mj*zb)*bt*at/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - 2*mi**3*mj**2*cos(mi*zt)*sin(mj*zt)*bt*at/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) -
            2*mi**2*mj**3*cos(mj*zb)*sin(mi*zb)*bt*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) + 2*mi**2*mj**3*cos(mj*zt)*sin(mi*zt)*bt*at/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - mi*mj**4*cos(mi*zb)*sin(mj*zb)*bt*at/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + mi*mj**4*cos(mi*zt)*sin(mj*zt)*bt*at/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + mj**5*cos(mj*zb)*sin(mi*zb)*bt*at/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - mj**5*cos(mj*zt)*sin(mi*zt)*bt*at/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6), axis=1)
        #A is symmetric
        A[tril] = A[triu]

    return A


def dim1sin_D_aDf_linear(m, at, ab, zt, zb, implementation='vectorized'):
    """Integration of sin(mi * z) * D[a(z) * D[sin(mj * z),z],z]
    between ztop and zbot where a(z) is piecewise linear functions of z.


    Calulation of integrals is performed at each element of a square symmetric
    matrix (size depends on size of `m`)

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    at : ``list`` of ``float``
        Property at top of each layer.
    ab : ``list`` of ``float``
        Property at bottom of each layer.
    zt : ``list`` of ``float``
        Normalised depth or z-coordinate at top of each layer. `zt[0]` = 0
    zb : ``list`` of ``float``
        Normalised depth or z-coordinate at bottom of each layer. `zt[-1]` = 1
    implementation : ['vectorized', 'scalar', 'fortran'], optional
        Functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        Default implementation='vectorized'.  If fortran extention module
        cannot be imported then 'vectorized' version will be used.
        If anything other than 'fortran' or 'scalar' is used then
        default vectorized version will be used.

    Returns
    -------
    A : numpy.ndarray
        A square symmetric matrix, size determined by size of `m`.

    See Also
    --------
    m_from_sin_mx : Used to generate 'm' input parameter.
    geotecha.speccon.integrals_generate_code.dim1sin_D_aDf_linear_implementations : Generation
        of code for this function.
    pdim1sin_D_aDf_linear : Same function with PolyLine
        inputs.
    geotecha.speccon.ext_integrals.dim1sin_D_aDf_linear : Equivalent fortran
        function.


    Notes
    -----
    The `dim1sin_D_aDf_linear` matrix, :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{\\frac{d}{dz}\\left({a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\right)\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    .. math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` is a piecewise
    linear functions with respect to :math:`z`, that within a layer is defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    To make the above integration simpler we integate by parts to get:

    .. math:: \\mathbf{A}_{i,j}= \\left.\\phi_i{a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\right|_{z=0}^{z=1} -\\int_{0}^1{{a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\frac{d\\phi_i}{dz}\\,dz}

    In this case the sine basis functions means the left term in the above
    equation is zero, leaving us with

    .. math:: \\mathbf{A}_{i,j}= -\\int_{0}^1{{a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\frac{d\\phi_i}{dz}\\,dz}


    """

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
            a_slope = (ab[layer] - at[layer]) / (zb[layer] - zt[layer])
            for i in range(neig):
                A[i, i] += (m[i]**2*(-a_slope*m[i]**(-2)*sin(m[i]*zt[layer])**2/4 -
                    a_slope*zt[layer]**2*sin(m[i]*zt[layer])**2/4 -
                    a_slope*zt[layer]**2*cos(m[i]*zt[layer])**2/4 +
                    m[i]**(-1)*cos(m[i]*zt[layer])*sin(m[i]*zt[layer])*at[layer]/2 +
                    zt[layer]*sin(m[i]*zt[layer])**2*at[layer]/2 +
                    zt[layer]*cos(m[i]*zt[layer])**2*at[layer]/2) -
                    m[i]**2*(-a_slope*m[i]**(-2)*sin(m[i]*zb[layer])**2/4 -
                    a_slope*m[i]**(-1)*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])*zt[layer]/2 +
                    a_slope*m[i]**(-1)*zb[layer]*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])/2 -
                    a_slope*zb[layer]*sin(m[i]*zb[layer])**2*zt[layer]/2 -
                    a_slope*zb[layer]*cos(m[i]*zb[layer])**2*zt[layer]/2 +
                    a_slope*zb[layer]**2*sin(m[i]*zb[layer])**2/4 +
                    a_slope*zb[layer]**2*cos(m[i]*zb[layer])**2/4 +
                    m[i]**(-1)*cos(m[i]*zb[layer])*sin(m[i]*zb[layer])*at[layer]/2 +
                    zb[layer]*sin(m[i]*zb[layer])**2*at[layer]/2 +
                    zb[layer]*cos(m[i]*zb[layer])**2*at[layer]/2))
            for i in range(neig-1):
                for j in range(i + 1, neig):
                    A[i, j] += (m[j]*m[i]*(a_slope*m[i]**2*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) +
                        2*a_slope*m[j]*m[i]*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*sin(m[i]*zt[layer])*sin(m[j]*zt[layer]) +
                        a_slope*m[j]**2*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[i]*zt[layer])*cos(m[j]*zt[layer]) + m[i]**3*(m[i]**4 -
                        2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])*at[layer] -
                        m[j]*m[i]**2*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])*at[layer] -
                        m[j]**2*m[i]*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[j]*zt[layer])*sin(m[i]*zt[layer])*at[layer] +
                        m[j]**3*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[i]*zt[layer])*sin(m[j]*zt[layer])*at[layer]) -
                        m[j]*m[i]*(a_slope*m[i]**2*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) -
                        a_slope*m[i]**3*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*zt[layer] +
                        a_slope*m[i]**3*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) +
                        2*a_slope*m[j]*m[i]*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*sin(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        a_slope*m[j]*m[i]**2*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*zt[layer] -
                        a_slope*m[j]*m[i]**2*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        a_slope*m[j]**2*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[i]*zb[layer])*cos(m[j]*zb[layer]) +
                        a_slope*m[j]**2*m[i]*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*zt[layer] -
                        a_slope*m[j]**2*m[i]*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*zb[layer]*cos(m[j]*zb[layer])*sin(m[i]*zb[layer]) -
                        a_slope*m[j]**3*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*zt[layer] +
                        a_slope*m[j]**3*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*zb[layer]*cos(m[i]*zb[layer])*sin(m[j]*zb[layer]) +
                        m[i]**3*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*at[layer] -
                        m[j]*m[i]**2*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*at[layer] -
                        m[j]**2*m[i]*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[j]*zb[layer])*sin(m[i]*zb[layer])*at[layer] +
                        m[j]**3*(m[i]**4 - 2*m[j]**2*m[i]**2 +
                        m[j]**4)**(-1)*cos(m[i]*zb[layer])*sin(m[j]*zb[layer])*at[layer]))

        #A is symmetric
        for i in range(neig - 1):
            for j in range(i + 1, neig):
                A[j, i] = A[i, j]

    elif implementation == 'fortran':
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

        a_slope = (ab - at) / (zb - zt)

        mi = m[:, np.newaxis]
        A[diag] = np.sum(mi**2*(-a_slope*zt**2*sin(mi*zt)**2/4 - a_slope*zt**2*cos(mi*zt)**2/4 -
            a_slope*sin(mi*zt)**2/(4*mi**2) + zt*sin(mi*zt)**2*at/2 + zt*cos(mi*zt)**2*at/2 +
            cos(mi*zt)*sin(mi*zt)*at/(2*mi)) - mi**2*(-a_slope*zb*sin(mi*zb)**2*zt/2 -
            a_slope*zb*cos(mi*zb)**2*zt/2 + a_slope*zb**2*sin(mi*zb)**2/4 +
            a_slope*zb**2*cos(mi*zb)**2/4 - a_slope*cos(mi*zb)*sin(mi*zb)*zt/(2*mi) +
            a_slope*zb*cos(mi*zb)*sin(mi*zb)/(2*mi) - a_slope*sin(mi*zb)**2/(4*mi**2) +
            zb*sin(mi*zb)**2*at/2 + zb*cos(mi*zb)**2*at/2 + cos(mi*zb)*sin(mi*zb)*at/(2*mi)), axis=1)

        mi = m[triu[0]][:, np.newaxis]
        mj = m[triu[1]][:, np.newaxis]
        A[triu] = np.sum(mi*mj*(a_slope*mi**2*cos(mi*zt)*cos(mj*zt)/(mi**4 - 2*mi**2*mj**2 + mj**4) +
            2*a_slope*mi*mj*sin(mi*zt)*sin(mj*zt)/(mi**4 - 2*mi**2*mj**2 + mj**4) +
            a_slope*mj**2*cos(mi*zt)*cos(mj*zt)/(mi**4 - 2*mi**2*mj**2 + mj**4) +
            mi**3*cos(mj*zt)*sin(mi*zt)*at/(mi**4 - 2*mi**2*mj**2 + mj**4) -
            mi**2*mj*cos(mi*zt)*sin(mj*zt)*at/(mi**4 - 2*mi**2*mj**2 + mj**4) -
            mi*mj**2*cos(mj*zt)*sin(mi*zt)*at/(mi**4 - 2*mi**2*mj**2 + mj**4) +
            mj**3*cos(mi*zt)*sin(mj*zt)*at/(mi**4 - 2*mi**2*mj**2 + mj**4)) -
            mi*mj*(-a_slope*mi**3*cos(mj*zb)*sin(mi*zb)*zt/(mi**4 - 2*mi**2*mj**2 + mj**4) +
            a_slope*mi**3*zb*cos(mj*zb)*sin(mi*zb)/(mi**4 - 2*mi**2*mj**2 + mj**4) +
            a_slope*mi**2*mj*cos(mi*zb)*sin(mj*zb)*zt/(mi**4 - 2*mi**2*mj**2 + mj**4) -
            a_slope*mi**2*mj*zb*cos(mi*zb)*sin(mj*zb)/(mi**4 - 2*mi**2*mj**2 + mj**4) +
            a_slope*mi**2*cos(mi*zb)*cos(mj*zb)/(mi**4 - 2*mi**2*mj**2 + mj**4) +
            a_slope*mi*mj**2*cos(mj*zb)*sin(mi*zb)*zt/(mi**4 - 2*mi**2*mj**2 + mj**4) -
            a_slope*mi*mj**2*zb*cos(mj*zb)*sin(mi*zb)/(mi**4 - 2*mi**2*mj**2 + mj**4) +
            2*a_slope*mi*mj*sin(mi*zb)*sin(mj*zb)/(mi**4 - 2*mi**2*mj**2 + mj**4) -
            a_slope*mj**3*cos(mi*zb)*sin(mj*zb)*zt/(mi**4 - 2*mi**2*mj**2 + mj**4) +
            a_slope*mj**3*zb*cos(mi*zb)*sin(mj*zb)/(mi**4 - 2*mi**2*mj**2 + mj**4) +
            a_slope*mj**2*cos(mi*zb)*cos(mj*zb)/(mi**4 - 2*mi**2*mj**2 + mj**4) +
            mi**3*cos(mj*zb)*sin(mi*zb)*at/(mi**4 - 2*mi**2*mj**2 + mj**4) -
            mi**2*mj*cos(mi*zb)*sin(mj*zb)*at/(mi**4 - 2*mi**2*mj**2 + mj**4) -
            mi*mj**2*cos(mj*zb)*sin(mi*zb)*at/(mi**4 - 2*mi**2*mj**2 + mj**4) +
            mj**3*cos(mi*zb)*sin(mj*zb)*at/(mi**4 - 2*mi**2*mj**2 + mj**4)), axis=1)
        #A is symmetric
        A[tril] = A[triu]

    return A


def dim1sin_ab_linear(m, at, ab, bt, bb, zt, zb, implementation='vectorized'):
    """Integration of sin(mi * z) * a(z) * b(z)
    between ztop and zbot where a(z) and b(z) are piecewise linear functions of z.


    Calulation of integrals is performed at each element of a 1d array
    (size depends on size of `m`).

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    at : ``list`` of ``float``
        Property at top of each layer.
    ab : ``list`` of ``float``
        Property at bottom of each layer.
    bt : ``list`` of ``float``
        2nd property at top of each layer.
    bb : ``list`` of ``float``
        2nd property at bottom of each layer.
    zt : ``list`` of ``float``
        Normalised depth or z-coordinate at top of each layer. `zt[0]` = 0
    zb : ``list`` of ``float``
        Normalised depth or z-coordinate at bottom of each layer. `zt[-1]` = 1
    implementation : ['vectorized', 'scalar', 'fortran'], optional
        Functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        Default implementation='vectorized'.  If fortran extention module
        cannot be imported then 'vectorized' version will be used.
        If anything other than 'fortran' or 'scalar' is used then
        default vectorized version will be used.

    Returns
    -------
    A : numpy.ndarray
        A 1d array size determined by size of `m`. Treat as column
        vector

    See Also
    --------
    m_from_sin_mx : Used to generate 'm' input parameter.
    geotecha.speccon.integrals_generate_code.dim1sin_ab_linear_implementations : Generation
        of code for this function.
    pdim1sin_ab_linear : Same function with PolyLine
        inputs.
    geotecha.speccon.ext_integrals.dim1sin_ab_linear : Equivalent fortran
        function.


    Notes
    -----
    The `dim1sin_ab_linear` which should be treated as a column vector,
    :math:`A` is given by:

    .. math:: \\mathbf{A}_{i}=\\int_{0}^1{{a\\left(z\\right)}{b\\left(z\\right)}\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    .. math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` and :math:`b\\left(z\\right)` are piecewise
    linear functions with respect to :math:`z`, that within a layer are defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

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
            a_slope = (ab[layer] - at[layer]) / (zb[layer] - zt[layer])
            b_slope = (bb[layer] - bt[layer]) / (zb[layer] - zt[layer])
            for i in range(neig):
                A[i] += (2*a_slope*b_slope*m[i]**(-3)*cos(m[i]*zb[layer]) - 2*a_slope*b_slope*m[i]**(-3)*cos(m[i]*zt[layer])
                    - 2*a_slope*b_slope*m[i]**(-2)*sin(m[i]*zb[layer])*zt[layer] +
                    2*a_slope*b_slope*m[i]**(-2)*zb[layer]*sin(m[i]*zb[layer]) -
                    a_slope*b_slope*m[i]**(-1)*cos(m[i]*zb[layer])*zt[layer]**2 +
                    2*a_slope*b_slope*m[i]**(-1)*zb[layer]*cos(m[i]*zb[layer])*zt[layer] -
                    a_slope*b_slope*m[i]**(-1)*zb[layer]**2*cos(m[i]*zb[layer]) +
                    a_slope*m[i]**(-2)*sin(m[i]*zb[layer])*bt[layer] -
                    a_slope*m[i]**(-2)*sin(m[i]*zt[layer])*bt[layer] +
                    a_slope*m[i]**(-1)*cos(m[i]*zb[layer])*zt[layer]*bt[layer] -
                    a_slope*m[i]**(-1)*zb[layer]*cos(m[i]*zb[layer])*bt[layer] +
                    b_slope*m[i]**(-2)*sin(m[i]*zb[layer])*at[layer] -
                    b_slope*m[i]**(-2)*sin(m[i]*zt[layer])*at[layer] +
                    b_slope*m[i]**(-1)*cos(m[i]*zb[layer])*zt[layer]*at[layer] -
                    b_slope*m[i]**(-1)*zb[layer]*cos(m[i]*zb[layer])*at[layer] -
                    m[i]**(-1)*cos(m[i]*zb[layer])*bt[layer]*at[layer] +
                    m[i]**(-1)*cos(m[i]*zt[layer])*bt[layer]*at[layer])

    elif implementation == 'fortran':
        import geotecha.speccon.ext_integrals as ext_integ
        A = ext_integ.dim1sin_ab_linear(m, at, ab, bt, bb, zt, zb)
#        try:
#            import geotecha.speccon.ext_integrals as ext_integ
#            A = ext_integ.dim1sin_ab_linear(m, at, ab, bt, bb, zt, zb)
#        except ImportError:
#            A = dim1sin_ab_linear(m, at, ab, bt, bb, zt, zb, implementation='vectorized')

    else:#default is 'vectorized' using numpy
        sin = np.sin
        cos = np.cos
        A = np.zeros(neig, float)

        a_slope = (ab - at) / (zb - zt)
        b_slope = (bb - bt) / (zb - zt)
        mi = m[:, np.newaxis]
        A[:] = np.sum(-a_slope*b_slope*cos(mi*zb)*zt**2/mi + 2*a_slope*b_slope*zb*cos(mi*zb)*zt/mi -
            a_slope*b_slope*zb**2*cos(mi*zb)/mi - 2*a_slope*b_slope*sin(mi*zb)*zt/mi**2 +
            2*a_slope*b_slope*zb*sin(mi*zb)/mi**2 + 2*a_slope*b_slope*cos(mi*zb)/mi**3 -
            2*a_slope*b_slope*cos(mi*zt)/mi**3 + a_slope*cos(mi*zb)*zt*bt/mi -
            a_slope*zb*cos(mi*zb)*bt/mi + a_slope*sin(mi*zb)*bt/mi**2 - a_slope*sin(mi*zt)*bt/mi**2
            + b_slope*cos(mi*zb)*zt*at/mi - b_slope*zb*cos(mi*zb)*at/mi +
            b_slope*sin(mi*zb)*at/mi**2 - b_slope*sin(mi*zt)*at/mi**2 - cos(mi*zb)*bt*at/mi +
            cos(mi*zt)*bt*at/mi, axis=1)


    return A


def dim1sin_abc_linear(m, at, ab, bt, bb, ct, cb, zt, zb, implementation='vectorized'):
    """Integrations of sin(mi * z) * a(z) * b(z) * c(z)
    between ztop and zbot where a(z), b(z), c(z) are piecewise linear functions of z.

    Calulation of integrals is performed at each element of a 1d array
    (size depends on size of `m`).

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    at : ``list`` of ``float``
        Property at top of each layer.
    ab : ``list`` of ``float``
        Property at bottom of each layer.
    bt : ``list`` of ``float``
        2nd property at top of each layer.
    bb : ``list`` of ``float``
        2nd property at bottom of each layer.
    ct : ``list`` of ``float``
        3rd property at top of each layer.
    cb : ``list`` of ``float``
        3rd property at bottom of each layer.
    zt : ``list`` of ``float``
        Normalised depth or z-coordinate at top of each layer. `zt[0]` = 0
    zb : ``list`` of ``float``
        Normalised depth or z-coordinate at bottom of each layer. `zt[-1]` = 1
    implementation : ['vectorized', 'scalar', 'fortran'], optional
        Functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        Default implementation='vectorized'.  If fortran extention module
        cannot be imported then 'vectorized' version will be used.
        If anything other than 'fortran' or 'scalar' is used then
        default vectorized version will be used.

    Returns
    -------
    A : numpy.ndarray
        A 1d array size determined by size of `m`. Treat as column
        vector.

    See Also
    --------
    m_from_sin_mx : Used to generate 'm' input parameter.
    geotecha.speccon.integrals_generate_code.dim1sin_abc_linear_implementations : Generation
        of code for this function.
    pdim1sin_abc_linear : Same function with PolyLine
        inputs.
    geotecha.speccon.ext_integrals.dim1sin_abc_linear : Equivalent fortran
        function.


    Notes
    -----
    The `dim1sin_abc_linear` which should be treated as a column vector,
    :math:`A` is given by:

    .. math:: \\mathbf{A}_{i}=\\int_{0}^1{{a\\left(z\\right)}{b\\left(z\\right)}{c\\left(z\\right)}\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    .. math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)`, :math:`b\\left(z\\right)`, and
    :math:`c\\left(z\\right)` are piecewise linear functions
    with respect to :math:`z`, that within a layer are defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

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
    ct = np.asarray(ct)
    cb = np.asarray(cb)
    zt = np.asarray(zt)
    zb = np.asarray(zb)

    neig = len(m)

    if implementation == 'scalar':
        sin = math.sin
        cos = math.cos
        A = np.zeros(neig, float)
        nlayers = len(zt)
        for layer in range(nlayers):
            a_slope = (ab[layer] - at[layer]) / (zb[layer] - zt[layer])
            b_slope = (bb[layer] - bt[layer]) / (zb[layer] - zt[layer])
            c_slope = (cb[layer] - ct[layer]) / (zb[layer] - zt[layer])
            for i in range(neig):
                A[i] += (-6*a_slope*b_slope*c_slope*m[i]**(-4)*sin(m[i]*zb[layer]) +
                    6*a_slope*b_slope*c_slope*m[i]**(-4)*sin(m[i]*zt[layer]) -
                    6*a_slope*b_slope*c_slope*m[i]**(-3)*cos(m[i]*zb[layer])*zt[layer] +
                    6*a_slope*b_slope*c_slope*m[i]**(-3)*zb[layer]*cos(m[i]*zb[layer]) +
                    3*a_slope*b_slope*c_slope*m[i]**(-2)*sin(m[i]*zb[layer])*zt[layer]**2 -
                    6*a_slope*b_slope*c_slope*m[i]**(-2)*zb[layer]*sin(m[i]*zb[layer])*zt[layer] +
                    3*a_slope*b_slope*c_slope*m[i]**(-2)*zb[layer]**2*sin(m[i]*zb[layer]) +
                    a_slope*b_slope*c_slope*m[i]**(-1)*cos(m[i]*zb[layer])*zt[layer]**3 -
                    3*a_slope*b_slope*c_slope*m[i]**(-1)*zb[layer]*cos(m[i]*zb[layer])*zt[layer]**2
                    +
                    3*a_slope*b_slope*c_slope*m[i]**(-1)*zb[layer]**2*cos(m[i]*zb[layer])*zt[layer]
                    - a_slope*b_slope*c_slope*m[i]**(-1)*zb[layer]**3*cos(m[i]*zb[layer]) +
                    2*a_slope*b_slope*m[i]**(-3)*cos(m[i]*zb[layer])*ct[layer] -
                    2*a_slope*b_slope*m[i]**(-3)*cos(m[i]*zt[layer])*ct[layer] -
                    2*a_slope*b_slope*m[i]**(-2)*sin(m[i]*zb[layer])*zt[layer]*ct[layer] +
                    2*a_slope*b_slope*m[i]**(-2)*zb[layer]*sin(m[i]*zb[layer])*ct[layer] -
                    a_slope*b_slope*m[i]**(-1)*cos(m[i]*zb[layer])*zt[layer]**2*ct[layer] +
                    2*a_slope*b_slope*m[i]**(-1)*zb[layer]*cos(m[i]*zb[layer])*zt[layer]*ct[layer] -
                    a_slope*b_slope*m[i]**(-1)*zb[layer]**2*cos(m[i]*zb[layer])*ct[layer] +
                    2*a_slope*c_slope*m[i]**(-3)*cos(m[i]*zb[layer])*bt[layer] -
                    2*a_slope*c_slope*m[i]**(-3)*cos(m[i]*zt[layer])*bt[layer] -
                    2*a_slope*c_slope*m[i]**(-2)*sin(m[i]*zb[layer])*zt[layer]*bt[layer] +
                    2*a_slope*c_slope*m[i]**(-2)*zb[layer]*sin(m[i]*zb[layer])*bt[layer] -
                    a_slope*c_slope*m[i]**(-1)*cos(m[i]*zb[layer])*zt[layer]**2*bt[layer] +
                    2*a_slope*c_slope*m[i]**(-1)*zb[layer]*cos(m[i]*zb[layer])*zt[layer]*bt[layer] -
                    a_slope*c_slope*m[i]**(-1)*zb[layer]**2*cos(m[i]*zb[layer])*bt[layer] +
                    a_slope*m[i]**(-2)*sin(m[i]*zb[layer])*ct[layer]*bt[layer] -
                    a_slope*m[i]**(-2)*sin(m[i]*zt[layer])*ct[layer]*bt[layer] +
                    a_slope*m[i]**(-1)*cos(m[i]*zb[layer])*zt[layer]*ct[layer]*bt[layer] -
                    a_slope*m[i]**(-1)*zb[layer]*cos(m[i]*zb[layer])*ct[layer]*bt[layer] +
                    2*b_slope*c_slope*m[i]**(-3)*cos(m[i]*zb[layer])*at[layer] -
                    2*b_slope*c_slope*m[i]**(-3)*cos(m[i]*zt[layer])*at[layer] -
                    2*b_slope*c_slope*m[i]**(-2)*sin(m[i]*zb[layer])*zt[layer]*at[layer] +
                    2*b_slope*c_slope*m[i]**(-2)*zb[layer]*sin(m[i]*zb[layer])*at[layer] -
                    b_slope*c_slope*m[i]**(-1)*cos(m[i]*zb[layer])*zt[layer]**2*at[layer] +
                    2*b_slope*c_slope*m[i]**(-1)*zb[layer]*cos(m[i]*zb[layer])*zt[layer]*at[layer] -
                    b_slope*c_slope*m[i]**(-1)*zb[layer]**2*cos(m[i]*zb[layer])*at[layer] +
                    b_slope*m[i]**(-2)*sin(m[i]*zb[layer])*ct[layer]*at[layer] -
                    b_slope*m[i]**(-2)*sin(m[i]*zt[layer])*ct[layer]*at[layer] +
                    b_slope*m[i]**(-1)*cos(m[i]*zb[layer])*zt[layer]*ct[layer]*at[layer] -
                    b_slope*m[i]**(-1)*zb[layer]*cos(m[i]*zb[layer])*ct[layer]*at[layer] +
                    c_slope*m[i]**(-2)*sin(m[i]*zb[layer])*bt[layer]*at[layer] -
                    c_slope*m[i]**(-2)*sin(m[i]*zt[layer])*bt[layer]*at[layer] +
                    c_slope*m[i]**(-1)*cos(m[i]*zb[layer])*zt[layer]*bt[layer]*at[layer] -
                    c_slope*m[i]**(-1)*zb[layer]*cos(m[i]*zb[layer])*bt[layer]*at[layer] -
                    m[i]**(-1)*cos(m[i]*zb[layer])*ct[layer]*bt[layer]*at[layer] +
                    m[i]**(-1)*cos(m[i]*zt[layer])*ct[layer]*bt[layer]*at[layer])

    elif implementation == 'fortran':
        import geotecha.speccon.ext_integrals as ext_integ
        A = ext_integ.dim1sin_abc_linear(m, at, ab, bt, bb, ct, cb, zt, zb)
#        try:
#            import geotecha.speccon.ext_integrals as ext_integ
#            A = ext_integ.dim1sin_abc_linear(m, at, ab, bt, bb,  ct, cb, zt, zb)
#        except ImportError:
#            A = dim1sin_abc_linear(m, at, ab, bt, bb,  ct, cb, zt, zb, implementation='vectorized')

    else:#default is 'vectorized' using numpy
        sin = np.sin
        cos = np.cos
        A = np.zeros(neig, float)


        a_slope = (ab - at) / (zb - zt)
        b_slope = (bb - bt) / (zb - zt)
        c_slope = (cb - ct) / (zb - zt)

        mi = m[:, np.newaxis]
        A[:] = np.sum(a_slope*b_slope*c_slope*cos(mi*zb)*zt**3/mi - 3*a_slope*b_slope*c_slope*zb*cos(mi*zb)*zt**2/mi +
            3*a_slope*b_slope*c_slope*zb**2*cos(mi*zb)*zt/mi -
            a_slope*b_slope*c_slope*zb**3*cos(mi*zb)/mi +
            3*a_slope*b_slope*c_slope*sin(mi*zb)*zt**2/mi**2 -
            6*a_slope*b_slope*c_slope*zb*sin(mi*zb)*zt/mi**2 +
            3*a_slope*b_slope*c_slope*zb**2*sin(mi*zb)/mi**2 -
            6*a_slope*b_slope*c_slope*cos(mi*zb)*zt/mi**3 +
            6*a_slope*b_slope*c_slope*zb*cos(mi*zb)/mi**3 -
            6*a_slope*b_slope*c_slope*sin(mi*zb)/mi**4 + 6*a_slope*b_slope*c_slope*sin(mi*zt)/mi**4
            - a_slope*b_slope*cos(mi*zb)*zt**2*ct/mi + 2*a_slope*b_slope*zb*cos(mi*zb)*zt*ct/mi -
            a_slope*b_slope*zb**2*cos(mi*zb)*ct/mi - 2*a_slope*b_slope*sin(mi*zb)*zt*ct/mi**2 +
            2*a_slope*b_slope*zb*sin(mi*zb)*ct/mi**2 + 2*a_slope*b_slope*cos(mi*zb)*ct/mi**3 -
            2*a_slope*b_slope*cos(mi*zt)*ct/mi**3 - a_slope*c_slope*cos(mi*zb)*zt**2*bt/mi +
            2*a_slope*c_slope*zb*cos(mi*zb)*zt*bt/mi - a_slope*c_slope*zb**2*cos(mi*zb)*bt/mi -
            2*a_slope*c_slope*sin(mi*zb)*zt*bt/mi**2 + 2*a_slope*c_slope*zb*sin(mi*zb)*bt/mi**2 +
            2*a_slope*c_slope*cos(mi*zb)*bt/mi**3 - 2*a_slope*c_slope*cos(mi*zt)*bt/mi**3 +
            a_slope*cos(mi*zb)*zt*ct*bt/mi - a_slope*zb*cos(mi*zb)*ct*bt/mi +
            a_slope*sin(mi*zb)*ct*bt/mi**2 - a_slope*sin(mi*zt)*ct*bt/mi**2 -
            b_slope*c_slope*cos(mi*zb)*zt**2*at/mi + 2*b_slope*c_slope*zb*cos(mi*zb)*zt*at/mi -
            b_slope*c_slope*zb**2*cos(mi*zb)*at/mi - 2*b_slope*c_slope*sin(mi*zb)*zt*at/mi**2 +
            2*b_slope*c_slope*zb*sin(mi*zb)*at/mi**2 + 2*b_slope*c_slope*cos(mi*zb)*at/mi**3 -
            2*b_slope*c_slope*cos(mi*zt)*at/mi**3 + b_slope*cos(mi*zb)*zt*ct*at/mi -
            b_slope*zb*cos(mi*zb)*ct*at/mi + b_slope*sin(mi*zb)*ct*at/mi**2 -
            b_slope*sin(mi*zt)*ct*at/mi**2 + c_slope*cos(mi*zb)*zt*bt*at/mi -
            c_slope*zb*cos(mi*zb)*bt*at/mi + c_slope*sin(mi*zb)*bt*at/mi**2 -
            c_slope*sin(mi*zt)*bt*at/mi**2 - cos(mi*zb)*ct*bt*at/mi + cos(mi*zt)*ct*bt*at/mi, axis=1)


    return A


def dim1sin_D_aDb_linear(m, at, ab, bt, bb, zt, zb, implementation='vectorized'):
    """Integrations of `sin(mi * z) * D[a(z) * D[b(z), z], z]`
    between ztop and zbot where a(z) is a piecewise linear function of z,
    and b(z) is a linear function of z.

    Calulation of integrals is performed at each element of a 1d array
    (size depends on size of `m`).

    .. warning::
        The functions produced are set up to accept the b(z) input as
        piecewise linear, i.e. zt, zb, bt, bb etc. It is up to the user to
        ensure that the bt and bb are such that they define a continuous
        linear function. eg. to define b(z)=z+1 then use
        zt=[0,0.4], zb=[0.4, 1], bt=[1,1.4], bb=[1.4,2].


    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    at : ``list`` of ``float``
        Property at top of each layer.
    ab : ``list`` of ``float``
        Property at bottom of each layer.
    bt : ``list`` of ``float``
        2nd property at top of each layer.
    bb : ``list`` of ``float``
        2nd property at bottom of each layer.
    zt : ``list`` of ``float``
        Normalised depth or z-coordinate at top of each layer. `zt[0]` = 0
    zb : ``list`` of ``float``
        Normalised depth or z-coordinate at bottom of each layer. `zt[-1]` = 1
    implementation : ['vectorized', 'scalar', 'fortran'], optional
        Functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        Default implementation='vectorized'.  If fortran extention module
        cannot be imported then 'vectorized' version will be used.
        If anything other than 'fortran' or 'scalar' is used then
        default vectorized version will be used.

    Returns
    -------
    A : numpy.ndarray
        A 1d array size determined by size of `m`. Treat as column
        vector

    See Also
    --------
    m_from_sin_mx : Used to generate 'm' input parameter.
    geotecha.speccon.integrals_generate_code.dim1sin_D_aDb_linear_implementations : Generation
        of code for this function.
    pdim1sin_D_aDb_linear : Same function with PolyLine
        inputs.
    geotecha.speccon.ext_integrals.dim1sin_D_aDb_linear : Equivalent fortran
        function.


    Notes
    -----
    The `dim1sin_D_aDb_linear` which should be treated as a column vector,
    :math:`A` is given by:

    .. math:: \\mathbf{A}_{i}=\\int_{0}^1{\\frac{d}{dz}\\left({a\\left(z\\right)}\\frac{d}{dz}{b\\left(z\\right)}\\right)\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    .. math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` is a piecewise
    linear functions with respect to :math:`z`, that within a layer is defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    :math:`b\\left(z\\right)` is a linear function of :math:`z` defined by

    .. math:: b\\left(z\\right) = b_t+\\left({b_b-b_t}\\right)z

    with :math:`t` and :math:`b` subscripts now representing 'top' and
    'bottom' of the profile respectively.

    Using the product rule for differentiation the above integral can be split
    into:

    .. math:: \\mathbf{A}_{i}=\\int_{0}^1{\\frac{da\\left(z\\right)}{dz}\\frac{db\\left(z\\right)}{dz}\\phi_i\\,dz} +
                              \\int_{0}^1{a\\left(z\\right)\\frac{d^2b\\left(z\\right)}{dz^2}\\phi_i\\,dz}

    The right hand term is zero because :math:`b\\left(z\\right)` is a
    continuous linear function so it's second derivative is zero.  The
    first derivative of :math:`b\\left(z\\right)` is a constant so the
    left term can be integrated by parts to give:

    .. math:: \\mathbf{A}_{i}=\\frac{db\\left(z\\right)}{dz}\\left(
                \\left.\\phi_i{a\\left(z\\right)}\\right|_{z=0}^{z=1} -
                -\\int_{0}^1{{a\\left(z\\right)}\\frac{d\\phi_i}{dz}\\,dz}
                \\right)


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
            a_slope = (ab[layer] - at[layer]) / (zb[layer] - zt[layer])
            b_slope = (bb[layer] - bt[layer]) / (zb[layer] - zt[layer])
            for i in range(neig):
                A[i] += (b_slope*m[i]*(a_slope*m[i]**(-2)*cos(zt[layer]*m[i]) + at[layer]*m[i]**(-1)*sin(zt[layer]*m[i])) -
                    b_slope*m[i]*(a_slope*m[i]**(-2)*cos(zb[layer]*m[i]) +
                    a_slope*zb[layer]*m[i]**(-1)*sin(zb[layer]*m[i]) -
                    a_slope*zt[layer]*m[i]**(-1)*sin(zb[layer]*m[i]) +
                    at[layer]*m[i]**(-1)*sin(zb[layer]*m[i])))

        for i in range(neig):
            A[i] += (-(zb[0] - zt[0])**(-1)*(bb[0] - bt[0])*at[0]*sin(zt[0]*m[i]) + (zb[nlayers - 1] - zt[nlayers -
                1])**(-1)*(bb[nlayers - 1] - bt[nlayers - 1])*ab[nlayers - 1]*sin(zb[nlayers -
                1]*m[i]))
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

        a_slope = (ab - at) / (zb - zt)
        b_slope = (bb - bt) / (zb - zt)
        mi = m[:, np.newaxis]
        A[:] = np.sum(b_slope*mi*(a_slope*cos(mi*zt)/mi**2 + sin(mi*zt)*at/mi) - b_slope*mi*(-a_slope*sin(mi*zb)*zt/mi +
            a_slope*zb*sin(mi*zb)/mi + a_slope*cos(mi*zb)/mi**2 + sin(mi*zb)*at/mi), axis=1)
        mi = m
        A[:]+= (-(zb[0] - zt[0])**(-1)*sin(mi*zt[0])*(bb[0] - bt[0])*at[0] + sin(mi*zb[-1])*(zb[-1] -
            zt[-1])**(-1)*(bb[-1] - bt[-1])*ab[-1])
    return A


def Eload_linear(loadtim, loadmag, eigs, tvals, dT=1.0, implementation='vectorized'):
    """Integration of load(tau) * exp(dT * eig * (t-tau)) between [0, t], where
    load(tau) is piecewise linear.

    Performs integrations involving a piecewise linear load.  A 2d array of
    dimensions A[len(tvals), len(eigs)]
    is produced where the 'i'th row of A contains the diagonal elements of the
    spectral 'E' matrix calculated for the time value tvals[i]. i.e. rows of
    this matrix will be assembled into the diagonal matrix 'E' elsewhere.

    Parameters
    ----------
    loadtim : 1d numpy.ndarray
        List of times describing load application.
    loadmag : 1d numpy.ndarray
        List of load magnitudes.
    eigs : 1d numpy.ndarray
        List of eigenvalues.
    tvals : 1d numpy.ndarray`
        List of time values to calculate integral at.
    dT : ``float``, optional
        Time factor multiple (Default dT=1.0).
    implementation : ['vectorized', 'scalar', 'fortran'], optional
        Functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        Default implementation='vectorized'.  If fortran extention module
        cannot be imported then 'vectorized' version will be used.
        If anything other than 'fortran' or 'scalar' is used then
        default vectorized version will be used.


    Returns
    -------
    A : numpy.ndarray
        A 2d array of dimesnions A[len(tvals), len(eigs)].
        The 'i'th row of A is the diagonal elements of the spectral 'E' matrix
        calculated for the time tvals[i].


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

    (note that this function,`Eload_linear`, rather than use :math:`t_s` and
    :math:`t_f`,
    explicitly finds increments that the current time falls in, falls after,
    and falls before and treates each case on it's own.)

    Each :math:`t` value of interest requires a separate diagonal matrix
    :math:`E`.  To use space more efficiently and to facilitate numpy
    broadcasting when using the results of the function, the diagonal elements
    of :math:`E` for each time value `t` value are stored in the rows of
    array :math:`A` returned by `Eload_linear`.  Thus:

    .. math:: \\mathbf{A}=\\left(\\begin{matrix}E_{0,0}(t_0)&E_{1,1}(t_0)& \cdots & E_{neig-1,neig-1}(t_0)\\\ E_{0,0}(t_1)&E_{1,1}(t_1)& \\cdots & E_{neig-1,neig-1}(t_1)\\\ \\vdots&\\vdots&\\ddots&\\vdots \\\ E_{0,0}(t_m)&E_{1,1}(t_m)& \cdots & E_{neig-1,neig-1}(t_m)\\end{matrix}\\right)


    See Also
    --------
    pEload_linear : Same function with PolyLine inputs.
    geotecha.speccon.ext_integrals.eload_linear : Equivalent fortran function.
    geotecha.speccon.integrals_generate_code.Eload_linear_implementations : Generation
        of code for this function.
    Eload_coslinear : Similar function with additional cosine term.
    EDload_linear : Similar function but the time dependent
        loading function is differentiated with respect to time.


    """

    loadtim = np.asarray(loadtim)
    loadmag = np.asarray(loadmag)
    eigs = np.asarray(eigs)
    tvals = np.asarray(tvals)

    if implementation == 'scalar':
        sin = math.sin
        cos = math.cos
        exp = math.exp

        A = np.zeros([len(tvals), len(eigs)])

        (ramps_less_than_t, constants_less_than_t, steps_less_than_t,
            ramps_containing_t, constants_containing_t) = segment_containing_also_segments_less_than_xi(loadtim, loadmag, tvals, steps_or_equal_to = True)

        for i, t in enumerate(tvals):
            for k in constants_containing_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += (-exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT*eig) + loadmag[k]/(dT*eig))
            for k in constants_less_than_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += (exp(-dT*eig*(t - loadtim[k + 1]))*loadmag[k]/(dT*eig) - exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT*eig))
            for k in ramps_containing_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += ((-dT*eig*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*loadmag[k + 1] +
                        dT*eig*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*loadmag[k] -
                        dT*eig*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*loadmag[k] + dT*eig*loadtim[k]*(-dT*eig*loadtim[k +
                        1] + dT*eig*loadtim[k])**(-1)*loadmag[k + 1] + (-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*loadmag[k + 1] - (-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*loadmag[k])/(dT*eig) -
                        (-dT*eig*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*loadmag[k + 1] + dT*eig*t*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] +
                        dT*eig*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*(t -
                        loadtim[k])*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1] -
                        dT*eig*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*(t -
                        loadtim[k])*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] - dT*eig*loadtim[k +
                        1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k] + dT*eig*loadtim[k]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1] +
                        (-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k + 1] - (-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k])/(dT*eig))
            for k in ramps_less_than_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += (-(-dT*eig*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k + 1] + dT*eig*t*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] +
                        dT*eig*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*(t -
                        loadtim[k])*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1] -
                        dT*eig*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*(t -
                        loadtim[k])*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] - dT*eig*loadtim[k +
                        1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k] + dT*eig*loadtim[k]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1] +
                        (-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k + 1] - (-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k])/(dT*eig)
                        + (-dT*eig*t*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*loadmag[k + 1] + dT*eig*t*exp(-dT*eig*(t -
                        loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*loadmag[k] + dT*eig*(t - loadtim[k +
                        1])*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*loadmag[k + 1] - dT*eig*(t - loadtim[k +
                        1])*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*loadmag[k] + dT*eig*exp(-dT*eig*(t - loadtim[k +
                        1]))*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*loadmag[k
                        + 1] - dT*eig*loadtim[k + 1]*exp(-dT*eig*(t - loadtim[k +
                        1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*loadmag[k] +
                        exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*loadmag[k + 1] - exp(-dT*eig*(t - loadtim[k +
                        1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*loadmag[k])/(dT*eig))
    elif implementation == 'fortran':
        #note than all fortran subroutines are lowercase.

        import geotecha.speccon.ext_integrals as ext_integ
        A = ext_integ.eload_linear(loadtim, loadmag, eigs, tvals, dT)
        #previous two lines are just for development to make sure that
        #the fortran code is actually working.  They force
#        try:
#            from geotecha.speccon.ext_integrals import eload_linear as fn
#        except ImportError:
#            fn = Eload_linear
#         A = fn(loadtim, loadmag, eigs, tvals, dT)
    else:#default is 'vectorized' using numpy
        sin = np.sin
        cos = np.cos
        exp = np.exp

        A = np.zeros([len(tvals), len(eigs)])

        (ramps_less_than_t, constants_less_than_t, steps_less_than_t,
            ramps_containing_t, constants_containing_t) = segment_containing_also_segments_less_than_xi(loadtim, loadmag, tvals, steps_or_equal_to = True)

        eig = eigs[:, None]
        for i, t in enumerate(tvals):
            k = constants_containing_t[i]
            if len(k):
                A[i, :] += np.sum(-exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT*eig) + loadmag[k]/(dT*eig), axis=1)

            k = constants_less_than_t[i]
            if len(k):
                A[i, :] += np.sum(exp(-dT*eig*(t - loadtim[k + 1]))*loadmag[k]/(dT*eig) - exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT*eig), axis=1)

            k = ramps_containing_t[i]
            if len(k):
                A[i, :] += np.sum((-dT*eig*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*loadmag[k + 1] +
                        dT*eig*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*loadmag[k] -
                        dT*eig*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*loadmag[k] + dT*eig*loadtim[k]*(-dT*eig*loadtim[k +
                        1] + dT*eig*loadtim[k])**(-1)*loadmag[k + 1] + (-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*loadmag[k + 1] - (-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*loadmag[k])/(dT*eig) -
                        (-dT*eig*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*loadmag[k + 1] + dT*eig*t*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] +
                        dT*eig*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*(t -
                        loadtim[k])*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1] -
                        dT*eig*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*(t -
                        loadtim[k])*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] - dT*eig*loadtim[k +
                        1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k] + dT*eig*loadtim[k]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1] +
                        (-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k + 1] - (-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k])/(dT*eig), axis=1)

            k = ramps_less_than_t[i]
            if len(k):
                A[i, :] += np.sum(-(-dT*eig*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k + 1] + dT*eig*t*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] +
                        dT*eig*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*(t -
                        loadtim[k])*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1] -
                        dT*eig*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*(t -
                        loadtim[k])*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] - dT*eig*loadtim[k +
                        1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k] + dT*eig*loadtim[k]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1] +
                        (-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k + 1] - (-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k])/(dT*eig)
                        + (-dT*eig*t*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*loadmag[k + 1] + dT*eig*t*exp(-dT*eig*(t -
                        loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*loadmag[k] + dT*eig*(t - loadtim[k +
                        1])*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*loadmag[k + 1] - dT*eig*(t - loadtim[k +
                        1])*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*loadmag[k] + dT*eig*exp(-dT*eig*(t - loadtim[k +
                        1]))*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*loadmag[k
                        + 1] - dT*eig*loadtim[k + 1]*exp(-dT*eig*(t - loadtim[k +
                        1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*loadmag[k] +
                        exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k])**(-1)*loadmag[k + 1] - exp(-dT*eig*(t - loadtim[k +
                        1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k])**(-1)*loadmag[k])/(dT*eig), axis=1)
    return A


def EDload_linear(loadtim, loadmag, eigs, tvals, dT=1.0, implementation='vectorized'):
    """Integration of D[load(tau), tau] * exp(dT * eig * (t-tau)) between
    [0, t], where load(tau) is piecewise linear.

    Performs integrations involving a piecewise linear load.  A 2d array of
    dimensions A[len(tvals), len(eigs)]
    is produced where the 'i'th row of A contains the diagonal elements of the
    spectral 'E' matrix calculated for the time value tvals[i]. i.e. rows of
    this matrix will be assembled into the diagonal matrix 'E' elsewhere.

    Parameters
    ----------
    loadtim : 1d numpy.ndarray
        List of times describing load application.
    loadmag : 1d numpy.ndarray
        List of load magnitudes.
    eigs : 1d numpy.ndarray
        List of eigenvalues.
    tvals : 1d numpy.ndarray`
        List of time values to calculate integral at.
    dT : ``float``, optional
        Time factor multiple (Default dT=1.0).
    implementation : ['vectorized', 'scalar', 'fortran'], optional
        Functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        Default implementation='vectorized'.  If fortran extention module
        cannot be imported then 'vectorized' version will be used.
        If anything other than 'fortran' or 'scalar' is used then
        default vectorized version will be used.


    Returns
    -------
    A : numpy.ndarray
        A 2d array of dimesnions A[len(tvals), len(eigs)].
        The 'i'th row of A is the diagonal elements of the spectral 'E' matrix
        calculated for the time tvals[i].


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
    `EDload_linear` is from loading terms in the governing equation that are NOT
    differentiated wrt :math:`t`.
    The diagonal elements of :math:`E` are given by:

    .. math:: \\mathbf{E}_{i,i}=\\int_{0}^t{\\frac{d{\\sigma\\left(\\tau\\right)}}{d\\tau}{\\exp\\left({(dT\\left(t-\\tau\\right)\\lambda_i}\\right)}\\,d\\tau}

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


    See Also
    --------
    pEDload_linear : Same function with PolyLine inputs.
    geotecha.speccon.ext_integrals.edload_linear : Equivalent fortran function.
    geotecha.speccon.integrals_generate_code.EDload_linear_implementations : Generation
        of code for this function.
    EDload_coslinear : Similar function with additional cosine term.
    Eload_linear : Similar function but the time dependent loading function
        is not differentiated with repect to time.

    """
    loadtim = np.asarray(loadtim)
    loadmag = np.asarray(loadmag)
    eigs = np.asarray(eigs)
    tvals = np.asarray(tvals)

    if implementation == 'scalar':
        sin = math.sin
        cos = math.cos
        exp = math.exp

        A = np.zeros([len(tvals), len(eigs)])

        (ramps_less_than_t, constants_less_than_t, steps_less_than_t,
            ramps_containing_t, constants_containing_t) = segment_containing_also_segments_less_than_xi(loadtim, loadmag, tvals, steps_or_equal_to = True)

        for i, t in enumerate(tvals):
            for k in steps_less_than_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += (exp(-dT*eig*(t - loadtim[k]))*(loadmag[k + 1] - loadmag[k]))
            for k in ramps_containing_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += ((loadtim[k + 1] - loadtim[k])**(-1)*(loadmag[k + 1] - loadmag[k])/(dT*eig) - (loadtim[k + 1] -
                        loadtim[k])**(-1)*exp(-dT*eig*(t - loadtim[k]))*(loadmag[k + 1] -
                        loadmag[k])/(dT*eig))
            for k in ramps_less_than_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += (-(loadtim[k + 1] - loadtim[k])**(-1)*exp(-dT*eig*(t - loadtim[k]))*(loadmag[k + 1] -
                        loadmag[k])/(dT*eig) + exp(-dT*eig*(t - loadtim[k + 1]))*(loadtim[k + 1] -
                        loadtim[k])**(-1)*(loadmag[k + 1] - loadmag[k])/(dT*eig))

    elif implementation == 'fortran':
        #note than all fortran subroutines are lowercase.

        import geotecha.speccon.ext_integrals as ext_integ
        A = ext_integ.edload_linear(loadtim, loadmag, eigs, tvals, dT)
        #previous two lines are just for development to make sure that
        #the fortran code is actually working.  They force
#        try:
#            from geotecha.speccon.ext_integrals import edload_linear as fn
#        except ImportError:
#            fn = EDload_linear
#         A = fn(loadtim, loadmag, eigs, tvals, dT)
    else:#default is 'vectorized' using numpy
        sin = np.sin
        cos = np.cos
        exp = np.exp

        A = np.zeros([len(tvals), len(eigs)])

        (ramps_less_than_t, constants_less_than_t, steps_less_than_t,
            ramps_containing_t, constants_containing_t) = segment_containing_also_segments_less_than_xi(loadtim, loadmag, tvals, steps_or_equal_to = True)

        eig = eigs[:, None]


        for i, t in enumerate(tvals):
            k = steps_less_than_t[i]
            if len(k):
                A[i, :] += np.sum(exp(-dT*eig*(t - loadtim[k]))*(loadmag[k + 1] - loadmag[k]), axis=1)
            k = ramps_containing_t[i]
            if len(k):
                A[i, :] += np.sum((loadtim[k + 1] - loadtim[k])**(-1)*(loadmag[k + 1] - loadmag[k])/(dT*eig) - (loadtim[k + 1] -
                        loadtim[k])**(-1)*exp(-dT*eig*(t - loadtim[k]))*(loadmag[k + 1] -
                        loadmag[k])/(dT*eig), axis=1)
            k = ramps_less_than_t[i]
            if len(k):
                A[i, :] += np.sum(-(loadtim[k + 1] - loadtim[k])**(-1)*exp(-dT*eig*(t - loadtim[k]))*(loadmag[k + 1] -
                        loadmag[k])/(dT*eig) + exp(-dT*eig*(t - loadtim[k + 1]))*(loadtim[k + 1] -
                        loadtim[k])**(-1)*(loadmag[k + 1] - loadmag[k])/(dT*eig), axis=1)
    return A



def Eload_coslinear(loadtim, loadmag, omega, phase, eigs, tvals, dT=1.0, implementation='vectorized'):
    """Integration of cos(omega*tau+phase)*load(tau) * exp(dT * eig * (t-tau)) between [0, t], where
    load(tau) is piecewise linear.

    Performs integrations involving a piecewise linear load.  A 2d array of
    dimensions A[len(tvals), len(eigs)]
    is produced where the 'i'th row of A contains the diagonal elements of the
    spectral 'E' matrix calculated for the time value tvals[i]. i.e. rows of
    this matrix will be assembled into the diagonal matrix 'E' elsewhere.

    Parameters
    ----------
    loadtim : 1d numpy.ndarray
        List of times describing load application.
    loadmag : 1d numpy.ndarray
        List of load magnitudes.
    omega, phase : float
        Parameters that describe a cyclic load cos(omega * t + phase).
    eigs : 1d numpy.ndarray
        List of eigenvalues.
    tvals : 1d numpy.ndarray`
        List of time values to calculate integral at.
    dT : ``float``, optional
        Time factor multiple (Default dT=1.0).
    implementation : ['vectorized', 'scalar', 'fortran'], optional
        Functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        Default implementation='vectorized'.  If fortran extention module
        cannot be imported then 'vectorized' version will be used.
        If anything other than 'fortran' or 'scalar' is used then
        default vectorized version will be used.


    Returns
    -------
    A : numpy.ndarray
        A 2d array of dimesnions A[len(tvals), len(eigs)].
        The 'i'th row of A is the diagonal elements of the spectral 'E' matrix
        calculated for the time tvals[i].


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
    `Eload_coslinear` is from loading terms in the governing equation that are NOT
    differentiated wrt :math:`t`.
    The diagonal elements of :math:`E` are given by:

    .. math:: \\mathbf{E}_{i,i}=\\int_{0}^t{{\\cos\\left(\\omega\\tau+\\textrm{phase}\\right)}{\\sigma\\left(\\tau\\right)}{\exp\\left({(dT\\left(t-\\tau\\right)\\lambda_i}\\right)}\\,d\\tau}

    where

     :math:`\\lambda_i` is the `ith` eigenvalue of the problem,
     :math:`dT` is a time factor for numerical convienience,
     :math:`\\sigma\left(\\tau\\right)` is the time dependant portion of the loading function.

    When the time dependant loading term :math:`\\sigma\\left(\\tau\\right)` is
    piecewise in time. The contribution of each load segment is found by:

    .. math:: \\mathbf{E}_{i,i}=\\int_{t_s}^{t_f}{{\\cos\\left(\\omega\\tau+\\textrm{phase}\\right)}{\\sigma\\left(\\tau\\right)}\\exp\\left({dT\\left(t-\\tau\\right)*\\lambda_i}\\right)\\,d\\tau}

    where

    .. math:: t_s = \\min\\left(t,t_{increment\\:start}\\right)

    .. math:: t_f = \\min\\left(t,t_{increment\\:end}\\right)

    (note that this function,`Eload_coslinear`, rather than use :math:`t_s` and
    :math:`t_f`,
    explicitly finds increments that the current time falls in, falls after,
    and falls before and treates each case on it's own.)

    Each :math:`t` value of interest requires a separate diagonal matrix
    :math:`E`.  To use space more efficiently and to facilitate numpy
    broadcasting when using the results of the function, the diagonal elements
    of :math:`E` for each time value `t` value are stored in the rows of
    array :math:`A` returned by `Eload_coslinear`.  Thus:

    .. math:: \\mathbf{A}=\\left(\\begin{matrix}E_{0,0}(t_0)&E_{1,1}(t_0)& \cdots & E_{neig-1,neig-1}(t_0)\\\ E_{0,0}(t_1)&E_{1,1}(t_1)& \\cdots & E_{neig-1,neig-1}(t_1)\\\ \\vdots&\\vdots&\\ddots&\\vdots \\\ E_{0,0}(t_m)&E_{1,1}(t_m)& \cdots & E_{neig-1,neig-1}(t_m)\\end{matrix}\\right)

    See Also
    --------
    pEload_coslinear : Same function with PolyLine inputs.
    geotecha.speccon.ext_integrals.eload_coslinear : Equivalent fortran function.
    geotecha.speccon.integrals_generate_code.Eload_coslinear_implementations : Generation
        of code for this function.
    Eload_linear : Similar function with no cosine term.
    EDload_coslinear : Similar function but the combined
        loading function is differentiated with respect to time.


    """
    loadtim = np.asarray(loadtim)
    loadmag = np.asarray(loadmag)
    eigs = np.asarray(eigs)
    tvals = np.asarray(tvals)

    if implementation == 'scalar':
        sin = math.sin
        cos = math.cos
        exp = math.exp

        A = np.zeros([len(tvals), len(eigs)])

        (ramps_less_than_t, constants_less_than_t, steps_less_than_t,
            ramps_containing_t, constants_containing_t) = segment_containing_also_segments_less_than_xi(loadtim, loadmag, tvals, steps_or_equal_to = True)

        for i, t in enumerate(tvals):
            for k in constants_containing_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += ((cos(omega*t + phase)/(1 + omega**2/(dT**2*eig**2)) + omega*sin(omega*t + phase)/(dT*eig*(1 +
                        omega**2/(dT**2*eig**2))))*loadmag[k]/(dT*eig) - (cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))/(1 +
                        omega**2/(dT**2*eig**2)) + omega*exp(-dT*eig*(t - loadtim[k]))*sin(omega*t -
                        omega*(t - loadtim[k]) + phase)/(dT*eig*(1 +
                        omega**2/(dT**2*eig**2))))*loadmag[k]/(dT*eig))
            for k in constants_less_than_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += ((cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t - loadtim[k + 1]))/(1 +
                        omega**2/(dT**2*eig**2)) + omega*exp(-dT*eig*(t - loadtim[k +
                        1]))*sin(omega*t - omega*(t - loadtim[k + 1]) + phase)/(dT*eig*(1 +
                        omega**2/(dT**2*eig**2))))*loadmag[k]/(dT*eig) - (cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))/(1 +
                        omega**2/(dT**2*eig**2)) + omega*exp(-dT*eig*(t - loadtim[k]))*sin(omega*t -
                        omega*(t - loadtim[k]) + phase)/(dT*eig*(1 +
                        omega**2/(dT**2*eig**2))))*loadmag[k]/(dT*eig))
            for k in ramps_containing_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += ((-dT*eig*t*cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]
                        + dT*eig*t*cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k]
                        - 2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] - dT*eig*cos(omega*t +
                        phase)*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] + dT*eig*cos(omega*t +
                        phase)*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] -
                        omega*t*sin(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] +
                        omega*t*sin(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] - omega*sin(omega*t +
                        phase)*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] + omega*sin(omega*t +
                        phase)*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] + cos(omega*t +
                        phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]
                        - cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] -
                        omega**2*t*cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k]
                        - 2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) +
                        omega**2*t*cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k]
                        - 2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) -
                        omega**2*cos(omega*t + phase)*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) +
                        omega**2*cos(omega*t + phase)*loadtim[k]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) +
                        2*omega*sin(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) -
                        2*omega*sin(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) -
                        omega**3*t*sin(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k]
                        - 2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) +
                        omega**3*t*sin(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k]
                        - 2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2) -
                        omega**3*sin(omega*t + phase)*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2) +
                        omega**3*sin(omega*t + phase)*loadtim[k]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) -
                        omega**2*cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) +
                        omega**2*cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2))/(dT*eig)
                        - (-dT*eig*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1] +
                        dT*eig*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] +
                        dT*eig*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*cos(omega*t - omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k + 1] - dT*eig*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] -
                        dT*eig*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] +
                        dT*eig*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1] -
                        omega*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k + 1]
                        + omega*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k] +
                        omega*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*exp(-dT*eig*(t - loadtim[k]))*sin(omega*t - omega*(t -
                        loadtim[k]) + phase)*loadmag[k + 1] - omega*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k] -
                        omega*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k] +
                        omega*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k + 1] +
                        (-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1]
                        - (-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] -
                        omega**2*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1]/(dT*eig) +
                        omega**2*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT*eig) +
                        omega**2*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*cos(omega*t - omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k + 1]/(dT*eig) - omega**2*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT*eig) - omega**2*loadtim[k +
                        1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT*eig) + omega**2*loadtim[k]*(-dT*eig*loadtim[k +
                        1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1]/(dT*eig) +
                        2*omega*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT*eig) - 2*omega*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT*eig) - omega**3*t*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT**2*eig**2) + omega**3*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT**2*eig**2) + omega**3*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT**2*eig**2) - omega**3*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT**2*eig**2) - omega**3*loadtim[k +
                        1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT**2*eig**2) + omega**3*loadtim[k]*(-dT*eig*loadtim[k +
                        1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT**2*eig**2) - omega**2*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k +
                        1]/(dT**2*eig**2) + omega**2*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT**2*eig**2))/(dT*eig))
            for k in ramps_less_than_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += (-(-dT*eig*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1] +
                        dT*eig*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] +
                        dT*eig*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*cos(omega*t - omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k + 1] - dT*eig*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] -
                        dT*eig*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] +
                        dT*eig*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1] -
                        omega*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k + 1]
                        + omega*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k] +
                        omega*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*exp(-dT*eig*(t - loadtim[k]))*sin(omega*t - omega*(t -
                        loadtim[k]) + phase)*loadmag[k + 1] - omega*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k] -
                        omega*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k] +
                        omega*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k + 1] +
                        (-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1]
                        - (-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] -
                        omega**2*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1]/(dT*eig) +
                        omega**2*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT*eig) +
                        omega**2*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*cos(omega*t - omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k + 1]/(dT*eig) - omega**2*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT*eig) - omega**2*loadtim[k +
                        1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT*eig) + omega**2*loadtim[k]*(-dT*eig*loadtim[k +
                        1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1]/(dT*eig) +
                        2*omega*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT*eig) - 2*omega*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT*eig) - omega**3*t*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT**2*eig**2) + omega**3*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT**2*eig**2) + omega**3*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT**2*eig**2) - omega**3*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT**2*eig**2) - omega**3*loadtim[k +
                        1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT**2*eig**2) + omega**3*loadtim[k]*(-dT*eig*loadtim[k +
                        1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT**2*eig**2) - omega**2*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k +
                        1]/(dT**2*eig**2) + omega**2*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT**2*eig**2))/(dT*eig) + (-dT*eig*t*cos(omega*t -
                        omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t - loadtim[k +
                        1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]
                        + dT*eig*t*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t
                        - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] + dT*eig*(t - loadtim[k
                        + 1])*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t -
                        loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] - dT*eig*(t -
                        loadtim[k + 1])*cos(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] + dT*eig*cos(omega*t -
                        omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t - loadtim[k +
                        1]))*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] - dT*eig*loadtim[k
                        + 1]*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t -
                        loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] -
                        omega*t*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k
                        + 1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] +
                        omega*t*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k
                        + 1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] + omega*(t - loadtim[k
                        + 1])*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k +
                        1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] - omega*(t -
                        loadtim[k + 1])*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t -
                        loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] + omega*exp(-dT*eig*(t
                        - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] - omega*loadtim[k +
                        1]*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k +
                        1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] + cos(omega*t -
                        omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t - loadtim[k +
                        1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]
                        - cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t -
                        loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] -
                        omega**2*t*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t
                        - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) +
                        omega**2*t*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t
                        - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) + omega**2*(t
                        - loadtim[k + 1])*cos(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) -
                        omega**2*(t - loadtim[k + 1])*cos(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) +
                        omega**2*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t -
                        loadtim[k + 1]))*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) -
                        omega**2*loadtim[k + 1]*cos(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) +
                        2*omega*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k
                        + 1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) -
                        2*omega*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k
                        + 1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) -
                        omega**3*t*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t -
                        loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) +
                        omega**3*t*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t -
                        loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2) +
                        omega**3*(t - loadtim[k + 1])*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t
                        - omega*(t - loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) -
                        omega**3*(t - loadtim[k + 1])*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t
                        - omega*(t - loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2) +
                        omega**3*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t -
                        loadtim[k + 1]) + phase)*loadtim[k]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) -
                        omega**3*loadtim[k + 1]*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t -
                        omega*(t - loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2) -
                        omega**2*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t -
                        loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) +
                        omega**2*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t -
                        loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2))/(dT*eig))
    elif implementation == 'fortran':
        #note than all fortran subroutines are lowercase.

        import geotecha.speccon.ext_integrals as ext_integ
        A = ext_integ.eload_coslinear(loadtim, loadmag, omega, phase, eigs, tvals, dT)
        #previous two lines are just for development to make sure that
        #the fortran code is actually working.  They force
#        try:
#            from geotecha.speccon.ext_integrals import eload_linear as fn
#        except ImportError:
#            fn = Eload_coslinear
#         A = fn(loadtim, loadmag, omega, phase, eigs, tvals, dT)
    else:#default is 'vectorized' using numpy
        sin = np.sin
        cos = np.cos
        exp = np.exp

        A = np.zeros([len(tvals), len(eigs)])

        (ramps_less_than_t, constants_less_than_t, steps_less_than_t,
            ramps_containing_t, constants_containing_t) = segment_containing_also_segments_less_than_xi(loadtim, loadmag, tvals, steps_or_equal_to = True)

        eig = eigs[:, None]
        for i, t in enumerate(tvals):
            k = constants_containing_t[i]
            if len(k):
                A[i, :] += np.sum((cos(omega*t + phase)/(1 + omega**2/(dT**2*eig**2)) + omega*sin(omega*t + phase)/(dT*eig*(1 +
                        omega**2/(dT**2*eig**2))))*loadmag[k]/(dT*eig) - (cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))/(1 +
                        omega**2/(dT**2*eig**2)) + omega*exp(-dT*eig*(t - loadtim[k]))*sin(omega*t -
                        omega*(t - loadtim[k]) + phase)/(dT*eig*(1 +
                        omega**2/(dT**2*eig**2))))*loadmag[k]/(dT*eig), axis=1)

            k = constants_less_than_t[i]
            if len(k):
                A[i, :] += np.sum((cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t - loadtim[k + 1]))/(1 +
                        omega**2/(dT**2*eig**2)) + omega*exp(-dT*eig*(t - loadtim[k +
                        1]))*sin(omega*t - omega*(t - loadtim[k + 1]) + phase)/(dT*eig*(1 +
                        omega**2/(dT**2*eig**2))))*loadmag[k]/(dT*eig) - (cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))/(1 +
                        omega**2/(dT**2*eig**2)) + omega*exp(-dT*eig*(t - loadtim[k]))*sin(omega*t -
                        omega*(t - loadtim[k]) + phase)/(dT*eig*(1 +
                        omega**2/(dT**2*eig**2))))*loadmag[k]/(dT*eig), axis=1)

            k = ramps_containing_t[i]
            if len(k):
                A[i, :] += np.sum((-dT*eig*t*cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]
                        + dT*eig*t*cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k]
                        - 2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] - dT*eig*cos(omega*t +
                        phase)*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] + dT*eig*cos(omega*t +
                        phase)*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] -
                        omega*t*sin(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] +
                        omega*t*sin(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] - omega*sin(omega*t +
                        phase)*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] + omega*sin(omega*t +
                        phase)*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] + cos(omega*t +
                        phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]
                        - cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] -
                        omega**2*t*cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k]
                        - 2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) +
                        omega**2*t*cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k]
                        - 2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) -
                        omega**2*cos(omega*t + phase)*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) +
                        omega**2*cos(omega*t + phase)*loadtim[k]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) +
                        2*omega*sin(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) -
                        2*omega*sin(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) -
                        omega**3*t*sin(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k]
                        - 2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) +
                        omega**3*t*sin(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k]
                        - 2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2) -
                        omega**3*sin(omega*t + phase)*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2) +
                        omega**3*sin(omega*t + phase)*loadtim[k]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) -
                        omega**2*cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) +
                        omega**2*cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2))/(dT*eig)
                        - (-dT*eig*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1] +
                        dT*eig*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] +
                        dT*eig*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*cos(omega*t - omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k + 1] - dT*eig*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] -
                        dT*eig*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] +
                        dT*eig*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1] -
                        omega*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k + 1]
                        + omega*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k] +
                        omega*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*exp(-dT*eig*(t - loadtim[k]))*sin(omega*t - omega*(t -
                        loadtim[k]) + phase)*loadmag[k + 1] - omega*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k] -
                        omega*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k] +
                        omega*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k + 1] +
                        (-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1]
                        - (-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] -
                        omega**2*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1]/(dT*eig) +
                        omega**2*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT*eig) +
                        omega**2*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*cos(omega*t - omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k + 1]/(dT*eig) - omega**2*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT*eig) - omega**2*loadtim[k +
                        1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT*eig) + omega**2*loadtim[k]*(-dT*eig*loadtim[k +
                        1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1]/(dT*eig) +
                        2*omega*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT*eig) - 2*omega*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT*eig) - omega**3*t*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT**2*eig**2) + omega**3*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT**2*eig**2) + omega**3*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT**2*eig**2) - omega**3*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT**2*eig**2) - omega**3*loadtim[k +
                        1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT**2*eig**2) + omega**3*loadtim[k]*(-dT*eig*loadtim[k +
                        1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT**2*eig**2) - omega**2*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k +
                        1]/(dT**2*eig**2) + omega**2*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT**2*eig**2))/(dT*eig), axis=1)

            k = ramps_less_than_t[i]
            if len(k):
                A[i, :] += np.sum(-(-dT*eig*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1] +
                        dT*eig*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] +
                        dT*eig*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*cos(omega*t - omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k + 1] - dT*eig*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] -
                        dT*eig*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] +
                        dT*eig*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1] -
                        omega*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k + 1]
                        + omega*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k] +
                        omega*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*exp(-dT*eig*(t - loadtim[k]))*sin(omega*t - omega*(t -
                        loadtim[k]) + phase)*loadmag[k + 1] - omega*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k] -
                        omega*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k] +
                        omega*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k + 1] +
                        (-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1]
                        - (-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] -
                        omega**2*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1]/(dT*eig) +
                        omega**2*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT*eig) +
                        omega**2*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*cos(omega*t - omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k + 1]/(dT*eig) - omega**2*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT*eig) - omega**2*loadtim[k +
                        1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT*eig) + omega**2*loadtim[k]*(-dT*eig*loadtim[k +
                        1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1]/(dT*eig) +
                        2*omega*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT*eig) - 2*omega*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT*eig) - omega**3*t*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT**2*eig**2) + omega**3*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT**2*eig**2) + omega**3*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT**2*eig**2) - omega**3*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT**2*eig**2) - omega**3*loadtim[k +
                        1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT**2*eig**2) + omega**3*loadtim[k]*(-dT*eig*loadtim[k +
                        1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT**2*eig**2) - omega**2*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k +
                        1]/(dT**2*eig**2) + omega**2*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT**2*eig**2))/(dT*eig) + (-dT*eig*t*cos(omega*t -
                        omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t - loadtim[k +
                        1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]
                        + dT*eig*t*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t
                        - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] + dT*eig*(t - loadtim[k
                        + 1])*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t -
                        loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] - dT*eig*(t -
                        loadtim[k + 1])*cos(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] + dT*eig*cos(omega*t -
                        omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t - loadtim[k +
                        1]))*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] - dT*eig*loadtim[k
                        + 1]*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t -
                        loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] -
                        omega*t*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k
                        + 1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] +
                        omega*t*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k
                        + 1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] + omega*(t - loadtim[k
                        + 1])*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k +
                        1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] - omega*(t -
                        loadtim[k + 1])*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t -
                        loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] + omega*exp(-dT*eig*(t
                        - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] - omega*loadtim[k +
                        1]*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k +
                        1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] + cos(omega*t -
                        omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t - loadtim[k +
                        1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]
                        - cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t -
                        loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] -
                        omega**2*t*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t
                        - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) +
                        omega**2*t*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t
                        - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) + omega**2*(t
                        - loadtim[k + 1])*cos(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) -
                        omega**2*(t - loadtim[k + 1])*cos(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) +
                        omega**2*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t -
                        loadtim[k + 1]))*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) -
                        omega**2*loadtim[k + 1]*cos(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) +
                        2*omega*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k
                        + 1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) -
                        2*omega*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k
                        + 1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) -
                        omega**3*t*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t -
                        loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) +
                        omega**3*t*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t -
                        loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2) +
                        omega**3*(t - loadtim[k + 1])*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t
                        - omega*(t - loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) -
                        omega**3*(t - loadtim[k + 1])*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t
                        - omega*(t - loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2) +
                        omega**3*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t -
                        loadtim[k + 1]) + phase)*loadtim[k]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) -
                        omega**3*loadtim[k + 1]*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t -
                        omega*(t - loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2) -
                        omega**2*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t -
                        loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) +
                        omega**2*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t -
                        loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2))/(dT*eig), axis=1)
    return A


def EDload_coslinear(loadtim, loadmag, omega, phase,eigs, tvals, dT=1.0, implementation='vectorized'):
    """Integration of D[cos(omega*tau+phase)*load(tau), tau] * exp(dT * eig * (t-tau)) between [0, t], where
    load(tau) is piecewise linear.

    Performs integrations involving a piecewise linear load.  A 2d array of
    dimensions A[len(tvals), len(eigs)]
    is produced where the 'i'th row of A contains the diagonal elements of the
    spectral 'E' matrix calculated for the time value tvals[i]. i.e. rows of
    this matrix will be assembled into the diagonal matrix 'E' elsewhere.

    Parameters
    ----------
    loadtim : 1d numpy.ndarray
        List of times describing load application.
    loadmag : 1d numpy.ndarray
        List of load magnitudes.
    omega, phase : float
        Parameters that describe a cyclic load cos(omega * t + phase).
    eigs : 1d numpy.ndarray
        List of eigenvalues.
    tvals : 1d numpy.ndarray`
        List of time values to calculate integral at.
    dT : ``float``, optional
        Time factor multiple (Default dT=1.0).
    implementation : ['vectorized', 'scalar', 'fortran'], optional
        Functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        Default implementation='vectorized'.  If fortran extention module
        cannot be imported then 'vectorized' version will be used.
        If anything other than 'fortran' or 'scalar' is used then
        default vectorized version will be used.


    Returns
    -------
    A : numpy.ndarray
        A 2d array of dimesnions A[len(tvals), len(eigs)].
        The 'i'th row of A is the diagonal elements of the spectral 'E' matrix
        calculated for the time tvals[i].


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
    `EDload_coslinear` is from loading terms in the governing equation that are NOT
    differentiated wrt :math:`t`.
    The diagonal elements of :math:`E` are given by:

    .. math:: \\mathbf{E}_{i,i}=\\int_{0}^t{\\frac{d{{\\cos\\left(\\omega\\tau+\\textrm{phase}\\right)}\\sigma\\left(\\tau\\right)}}{d\\tau}{\\exp\\left({(dT\\left(t-\\tau\\right)\\lambda_i}\\right)}\\,d\\tau}

    where

     :math:`\\lambda_i` is the `ith` eigenvalue of the problem,
     :math:`dT` is a time factor for numerical convienience,
     :math:`\\sigma\left(\\tau\\right)` is the time dependant portion of the loading function.

    When the time dependant loading term :math:`\\sigma\\left(\\tau\\right)` is
    piecewise in time. The contribution of each load segment is found by:

    .. math:: \\mathbf{E}_{i,i}=\\int_{t_s}^{t_f}{\\frac{d{{\\cos\\left(\\omega\\tau+\\textrm{phase}\\right)}\\sigma\\left(\\tau\\right)}}{d\\tau}\\exp\\left({dT\\left(t-\\tau\\right)*\\lambda_i}\\right)\\,d\\tau}

    where

    .. math:: t_s = \\min\\left(t,t_{increment\\:start}\\right)

    .. math:: t_f = \\min\\left(t,t_{increment\\:end}\\right)

    (note that this function,`EDload_coslinear`, rather than use :math:`t_s` and
    :math:`t_f`,
    explicitly finds increments that the current time falls in, falls after,
    and falls before and treates each case on it's own.)

    Each :math:`t` value of interest requires a separate diagonal matrix
    :math:`E`.  To use space more efficiently and to facilitate numpy
    broadcasting when using the results of the function, the diagonal elements
    of :math:`E` for each time value `t` value are stored in the rows of
    array :math:`A` returned by `EDload_coslinear`.  Thus:

    .. math:: \\mathbf{A}=\\left(\\begin{matrix}E_{0,0}(t_0)&E_{1,1}(t_0)& \cdots & E_{neig-1,neig-1}(t_0)\\\ E_{0,0}(t_1)&E_{1,1}(t_1)& \\cdots & E_{neig-1,neig-1}(t_1)\\\ \\vdots&\\vdots&\\ddots&\\vdots \\\ E_{0,0}(t_m)&E_{1,1}(t_m)& \cdots & E_{neig-1,neig-1}(t_m)\\end{matrix}\\right)

    See Also
    --------
    pEDload_coslinear : Same function with PolyLine inputs.
    geotecha.speccon.ext_integrals.edload_coslinear : Equivalent fortran function.
    geotecha.speccon.integrals_generate_code.EDload_coslinear_implementations : Generation
        of code for this function.
    EDload_linear : Similar function with no cosine term.
    Eload_coslinear : Similar function but the combined
        loading function is not differentiated with respect to time.


    """
    loadtim = np.asarray(loadtim)
    loadmag = np.asarray(loadmag)
    eigs = np.asarray(eigs)
    tvals = np.asarray(tvals)

    if implementation == 'scalar':
        sin = math.sin
        cos = math.cos
        exp = math.exp

        A = np.zeros([len(tvals), len(eigs)])

        (ramps_less_than_t, constants_less_than_t, steps_less_than_t,
            ramps_containing_t, constants_containing_t) = segment_containing_also_segments_less_than_xi(loadtim, loadmag, tvals, steps_or_equal_to = True)

        for i, t in enumerate(tvals):
            for k in steps_less_than_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += (cos(omega*loadtim[k] + phase)*exp(-dT*eig*(t - loadtim[k]))*(loadmag[k + 1] - loadmag[k]))
            for k in ramps_containing_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += (omega*t*sin(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]
                        - omega*t*sin(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] -
                        omega*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k + 1]
                        + omega*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k] +
                        omega*sin(omega*t + phase)*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] - omega*sin(omega*t +
                        phase)*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] +
                        omega*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*exp(-dT*eig*(t - loadtim[k]))*sin(omega*t - omega*(t -
                        loadtim[k]) + phase)*loadmag[k + 1] - omega*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k] -
                        omega*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k] +
                        omega*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k + 1] -
                        cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] + cos(omega*t +
                        phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] +
                        (-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1]
                        - (-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] -
                        omega**2*t*cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k]
                        - 2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) +
                        omega**2*t*cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k]
                        - 2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) +
                        omega**2*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1]/(dT*eig) -
                        omega**2*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT*eig) -
                        omega**2*cos(omega*t + phase)*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) +
                        omega**2*cos(omega*t + phase)*loadtim[k]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) -
                        omega**2*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*cos(omega*t - omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k + 1]/(dT*eig) + omega**2*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT*eig) + omega**2*loadtim[k +
                        1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT*eig) - omega**2*loadtim[k]*(-dT*eig*loadtim[k +
                        1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1]/(dT*eig) -
                        2*omega*sin(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) +
                        2*omega*sin(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) +
                        2*omega*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT*eig) - 2*omega*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT*eig) + omega**3*t*sin(omega*t +
                        phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k +
                        1]/(dT**2*eig**2) - omega**3*t*sin(omega*t + phase)*(-dT*eig*loadtim[k + 1]
                        + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2) -
                        omega**3*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT**2*eig**2) + omega**3*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT**2*eig**2) + omega**3*sin(omega*t + phase)*loadtim[k +
                        1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2) -
                        omega**3*sin(omega*t + phase)*loadtim[k]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) +
                        omega**3*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*exp(-dT*eig*(t - loadtim[k]))*sin(omega*t - omega*(t -
                        loadtim[k]) + phase)*loadmag[k + 1]/(dT**2*eig**2) -
                        omega**3*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*exp(-dT*eig*(t - loadtim[k]))*sin(omega*t - omega*(t -
                        loadtim[k]) + phase)*loadmag[k]/(dT**2*eig**2) - omega**3*loadtim[k +
                        1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT**2*eig**2) + omega**3*loadtim[k]*(-dT*eig*loadtim[k +
                        1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT**2*eig**2) + omega**2*cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) -
                        omega**2*cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2) -
                        omega**2*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k +
                        1]/(dT**2*eig**2) + omega**2*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT**2*eig**2)
                        - omega**4*t*cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**3*eig**3) +
                        omega**4*t*cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k]
                        - 2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**3*eig**3) +
                        omega**4*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k +
                        1]/(dT**3*eig**3) - omega**4*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT**3*eig**3)
                        - omega**4*cos(omega*t + phase)*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**3*eig**3) +
                        omega**4*cos(omega*t + phase)*loadtim[k]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**3*eig**3) -
                        omega**4*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*cos(omega*t - omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k + 1]/(dT**3*eig**3) + omega**4*(-dT*eig*loadtim[k +
                        1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT**3*eig**3) + omega**4*loadtim[k +
                        1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT**3*eig**3) -
                        omega**4*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k +
                        1]/(dT**3*eig**3))
            for k in ramps_less_than_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += (-omega*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k + 1] +
                        omega*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k] +
                        omega*t*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k
                        + 1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] -
                        omega*t*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k
                        + 1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] - omega*(t - loadtim[k
                        + 1])*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k +
                        1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] + omega*(t -
                        loadtim[k + 1])*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t -
                        loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] +
                        omega*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*exp(-dT*eig*(t - loadtim[k]))*sin(omega*t - omega*(t -
                        loadtim[k]) + phase)*loadmag[k + 1] - omega*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k] -
                        omega*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k +
                        1]) + phase)*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] - omega*loadtim[k +
                        1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k] +
                        omega*loadtim[k + 1]*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t -
                        omega*(t - loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] +
                        omega*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k + 1] +
                        (-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1]
                        - (-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] -
                        cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t - loadtim[k
                        + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]
                        + cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t -
                        loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] +
                        omega**2*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1]/(dT*eig) -
                        omega**2*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT*eig) -
                        omega**2*t*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t
                        - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) +
                        omega**2*t*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t
                        - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) + omega**2*(t
                        - loadtim[k + 1])*cos(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) -
                        omega**2*(t - loadtim[k + 1])*cos(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) -
                        omega**2*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*cos(omega*t - omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k + 1]/(dT*eig) + omega**2*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT*eig) + omega**2*cos(omega*t - omega*(t -
                        loadtim[k + 1]) + phase)*exp(-dT*eig*(t - loadtim[k +
                        1]))*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) +
                        omega**2*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT*eig) -
                        omega**2*loadtim[k + 1]*cos(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) -
                        omega**2*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1]/(dT*eig) +
                        2*omega*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT*eig) - 2*omega*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT*eig) - 2*omega*exp(-dT*eig*(t - loadtim[k +
                        1]))*sin(omega*t - omega*(t - loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k +
                        1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) +
                        2*omega*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k
                        + 1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) -
                        omega**3*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT**2*eig**2) + omega**3*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT**2*eig**2) + omega**3*t*exp(-dT*eig*(t - loadtim[k +
                        1]))*sin(omega*t - omega*(t - loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k +
                        1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) -
                        omega**3*t*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t -
                        loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2) -
                        omega**3*(t - loadtim[k + 1])*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t
                        - omega*(t - loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) +
                        omega**3*(t - loadtim[k + 1])*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t
                        - omega*(t - loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2) +
                        omega**3*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*exp(-dT*eig*(t - loadtim[k]))*sin(omega*t - omega*(t -
                        loadtim[k]) + phase)*loadmag[k + 1]/(dT**2*eig**2) -
                        omega**3*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*exp(-dT*eig*(t - loadtim[k]))*sin(omega*t - omega*(t -
                        loadtim[k]) + phase)*loadmag[k]/(dT**2*eig**2) - omega**3*exp(-dT*eig*(t -
                        loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) -
                        omega**3*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT**2*eig**2) + omega**3*loadtim[k + 1]*exp(-dT*eig*(t -
                        loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2) +
                        omega**3*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT**2*eig**2) - omega**2*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k +
                        1]/(dT**2*eig**2) + omega**2*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT**2*eig**2)
                        + omega**2*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t
                        - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) -
                        omega**2*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t -
                        loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2) +
                        omega**4*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k +
                        1]/(dT**3*eig**3) - omega**4*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT**3*eig**3)
                        - omega**4*t*cos(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**3*eig**3) +
                        omega**4*t*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t
                        - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**3*eig**3) +
                        omega**4*(t - loadtim[k + 1])*cos(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**3*eig**3) -
                        omega**4*(t - loadtim[k + 1])*cos(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**3*eig**3) -
                        omega**4*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*cos(omega*t - omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k + 1]/(dT**3*eig**3) + omega**4*(-dT*eig*loadtim[k +
                        1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT**3*eig**3) + omega**4*cos(omega*t - omega*(t -
                        loadtim[k + 1]) + phase)*exp(-dT*eig*(t - loadtim[k +
                        1]))*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT**3*eig**3)
                        - omega**4*loadtim[k + 1]*cos(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**3*eig**3) -
                        omega**4*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k +
                        1]/(dT**3*eig**3))
            for k in constants_containing_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += (-omega*(sin(omega*t + phase)/(1 + omega**2/(dT**2*eig**2)) - omega*cos(omega*t + phase)/(dT*eig*(1 +
                        omega**2/(dT**2*eig**2))))*loadmag[k]/(dT*eig) + omega*(exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)/(1 +
                        omega**2/(dT**2*eig**2)) - omega*cos(omega*t - omega*(t - loadtim[k]) +
                        phase)*exp(-dT*eig*(t - loadtim[k]))/(dT*eig*(1 +
                        omega**2/(dT**2*eig**2))))*loadmag[k]/(dT*eig))
            for k in constants_less_than_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += (-omega*(exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k + 1]) + phase)/(1 +
                        omega**2/(dT**2*eig**2)) - omega*cos(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*exp(-dT*eig*(t - loadtim[k + 1]))/(dT*eig*(1 +
                        omega**2/(dT**2*eig**2))))*loadmag[k]/(dT*eig) + omega*(exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)/(1 +
                        omega**2/(dT**2*eig**2)) - omega*cos(omega*t - omega*(t - loadtim[k]) +
                        phase)*exp(-dT*eig*(t - loadtim[k]))/(dT*eig*(1 +
                        omega**2/(dT**2*eig**2))))*loadmag[k]/(dT*eig))

    elif implementation == 'fortran':
        #note than all fortran subroutines are lowercase.

        import geotecha.speccon.ext_integrals as ext_integ
        A = ext_integ.edload_coslinear(loadtim, loadmag, omega, phase, eigs, tvals, dT)
        #previous two lines are just for development to make sure that
        #the fortran code is actually working.  They force
#        try:
#            from geotecha.speccon.ext_integrals import edload_linear as fn
#        except ImportError:
#            fn = EDload_coslinear
#         A = fn(loadtim, loadmag, omega, phase, eigs, tvals, dT)
    else:#default is 'vectorized' using numpy
        sin = np.sin
        cos = np.cos
        exp = np.exp

        A = np.zeros([len(tvals), len(eigs)])

        (ramps_less_than_t, constants_less_than_t, steps_less_than_t,
            ramps_containing_t, constants_containing_t) = segment_containing_also_segments_less_than_xi(loadtim, loadmag, tvals, steps_or_equal_to = True)

        eig = eigs[:, None]


        for i, t in enumerate(tvals):
            k = steps_less_than_t[i]
            if len(k):
                A[i, :] += np.sum(cos(omega*loadtim[k] + phase)*exp(-dT*eig*(t - loadtim[k]))*(loadmag[k + 1] - loadmag[k]), axis=1)
            k = ramps_containing_t[i]
            if len(k):
                A[i, :] += np.sum(omega*t*sin(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]
                        - omega*t*sin(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] -
                        omega*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k + 1]
                        + omega*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k] +
                        omega*sin(omega*t + phase)*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] - omega*sin(omega*t +
                        phase)*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] +
                        omega*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*exp(-dT*eig*(t - loadtim[k]))*sin(omega*t - omega*(t -
                        loadtim[k]) + phase)*loadmag[k + 1] - omega*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k] -
                        omega*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k] +
                        omega*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k + 1] -
                        cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] + cos(omega*t +
                        phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] +
                        (-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1]
                        - (-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] -
                        omega**2*t*cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k]
                        - 2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) +
                        omega**2*t*cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k]
                        - 2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) +
                        omega**2*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1]/(dT*eig) -
                        omega**2*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT*eig) -
                        omega**2*cos(omega*t + phase)*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) +
                        omega**2*cos(omega*t + phase)*loadtim[k]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) -
                        omega**2*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*cos(omega*t - omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k + 1]/(dT*eig) + omega**2*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT*eig) + omega**2*loadtim[k +
                        1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT*eig) - omega**2*loadtim[k]*(-dT*eig*loadtim[k +
                        1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1]/(dT*eig) -
                        2*omega*sin(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) +
                        2*omega*sin(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) +
                        2*omega*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT*eig) - 2*omega*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT*eig) + omega**3*t*sin(omega*t +
                        phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k +
                        1]/(dT**2*eig**2) - omega**3*t*sin(omega*t + phase)*(-dT*eig*loadtim[k + 1]
                        + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2) -
                        omega**3*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT**2*eig**2) + omega**3*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT**2*eig**2) + omega**3*sin(omega*t + phase)*loadtim[k +
                        1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2) -
                        omega**3*sin(omega*t + phase)*loadtim[k]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) +
                        omega**3*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*exp(-dT*eig*(t - loadtim[k]))*sin(omega*t - omega*(t -
                        loadtim[k]) + phase)*loadmag[k + 1]/(dT**2*eig**2) -
                        omega**3*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*exp(-dT*eig*(t - loadtim[k]))*sin(omega*t - omega*(t -
                        loadtim[k]) + phase)*loadmag[k]/(dT**2*eig**2) - omega**3*loadtim[k +
                        1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT**2*eig**2) + omega**3*loadtim[k]*(-dT*eig*loadtim[k +
                        1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT**2*eig**2) + omega**2*cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) -
                        omega**2*cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2) -
                        omega**2*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k +
                        1]/(dT**2*eig**2) + omega**2*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT**2*eig**2)
                        - omega**4*t*cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**3*eig**3) +
                        omega**4*t*cos(omega*t + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k]
                        - 2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**3*eig**3) +
                        omega**4*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k +
                        1]/(dT**3*eig**3) - omega**4*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT**3*eig**3)
                        - omega**4*cos(omega*t + phase)*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**3*eig**3) +
                        omega**4*cos(omega*t + phase)*loadtim[k]*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**3*eig**3) -
                        omega**4*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*cos(omega*t - omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k + 1]/(dT**3*eig**3) + omega**4*(-dT*eig*loadtim[k +
                        1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT**3*eig**3) + omega**4*loadtim[k +
                        1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT**3*eig**3) -
                        omega**4*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k +
                        1]/(dT**3*eig**3), axis=1)
            k = ramps_less_than_t[i]
            if len(k):
                A[i, :] += np.sum(-omega*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k + 1] +
                        omega*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k] +
                        omega*t*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k
                        + 1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] -
                        omega*t*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k
                        + 1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] - omega*(t - loadtim[k
                        + 1])*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k +
                        1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] + omega*(t -
                        loadtim[k + 1])*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t -
                        loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] +
                        omega*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*exp(-dT*eig*(t - loadtim[k]))*sin(omega*t - omega*(t -
                        loadtim[k]) + phase)*loadmag[k + 1] - omega*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k] -
                        omega*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k +
                        1]) + phase)*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1] - omega*loadtim[k +
                        1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k] +
                        omega*loadtim[k + 1]*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t -
                        omega*(t - loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] +
                        omega*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k + 1] +
                        (-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1]
                        - (-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k] -
                        cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t - loadtim[k
                        + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]
                        + cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t -
                        loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k] +
                        omega**2*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1]/(dT*eig) -
                        omega**2*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT*eig) -
                        omega**2*t*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t
                        - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) +
                        omega**2*t*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t
                        - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) + omega**2*(t
                        - loadtim[k + 1])*cos(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) -
                        omega**2*(t - loadtim[k + 1])*cos(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) -
                        omega**2*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*cos(omega*t - omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k + 1]/(dT*eig) + omega**2*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT*eig) + omega**2*cos(omega*t - omega*(t -
                        loadtim[k + 1]) + phase)*exp(-dT*eig*(t - loadtim[k +
                        1]))*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) +
                        omega**2*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT*eig) -
                        omega**2*loadtim[k + 1]*cos(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) -
                        omega**2*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k + 1]/(dT*eig) +
                        2*omega*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t
                        - loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT*eig) - 2*omega*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT*eig) - 2*omega*exp(-dT*eig*(t - loadtim[k +
                        1]))*sin(omega*t - omega*(t - loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k +
                        1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT*eig) +
                        2*omega*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k
                        + 1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT*eig) -
                        omega**3*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT**2*eig**2) + omega**3*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT**2*eig**2) + omega**3*t*exp(-dT*eig*(t - loadtim[k +
                        1]))*sin(omega*t - omega*(t - loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k +
                        1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) -
                        omega**3*t*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t -
                        loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2) -
                        omega**3*(t - loadtim[k + 1])*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t
                        - omega*(t - loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) +
                        omega**3*(t - loadtim[k + 1])*exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t
                        - omega*(t - loadtim[k + 1]) + phase)*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2) +
                        omega**3*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*exp(-dT*eig*(t - loadtim[k]))*sin(omega*t - omega*(t -
                        loadtim[k]) + phase)*loadmag[k + 1]/(dT**2*eig**2) -
                        omega**3*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*exp(-dT*eig*(t - loadtim[k]))*sin(omega*t - omega*(t -
                        loadtim[k]) + phase)*loadmag[k]/(dT**2*eig**2) - omega**3*exp(-dT*eig*(t -
                        loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) -
                        omega**3*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) +
                        phase)*loadmag[k]/(dT**2*eig**2) + omega**3*loadtim[k + 1]*exp(-dT*eig*(t -
                        loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k +
                        1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2) +
                        omega**3*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)*loadmag[k +
                        1]/(dT**2*eig**2) - omega**2*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k +
                        1]/(dT**2*eig**2) + omega**2*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT**2*eig**2)
                        + omega**2*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t
                        - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**2*eig**2) -
                        omega**2*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t -
                        loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**2*eig**2) +
                        omega**4*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k +
                        1]/(dT**3*eig**3) - omega**4*t*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT**3*eig**3)
                        - omega**4*t*cos(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**3*eig**3) +
                        omega**4*t*cos(omega*t - omega*(t - loadtim[k + 1]) + phase)*exp(-dT*eig*(t
                        - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**3*eig**3) +
                        omega**4*(t - loadtim[k + 1])*cos(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**3*eig**3) -
                        omega**4*(t - loadtim[k + 1])*cos(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**3*eig**3) -
                        omega**4*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k
                        + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k +
                        1]/(dT**3*eig**3) + omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t -
                        loadtim[k])*cos(omega*t - omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k + 1]/(dT**3*eig**3) + omega**4*(-dT*eig*loadtim[k +
                        1] + dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*(t - loadtim[k])*cos(omega*t -
                        omega*(t - loadtim[k]) + phase)*exp(-dT*eig*(t -
                        loadtim[k]))*loadmag[k]/(dT**3*eig**3) + omega**4*cos(omega*t - omega*(t -
                        loadtim[k + 1]) + phase)*exp(-dT*eig*(t - loadtim[k +
                        1]))*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k + 1]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k]/(dT**3*eig**3)
                        - omega**4*loadtim[k + 1]*cos(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*exp(-dT*eig*(t - loadtim[k + 1]))*(-dT*eig*loadtim[k + 1] +
                        dT*eig*loadtim[k] - 2*omega**2*loadtim[k + 1]/(dT*eig) +
                        2*omega**2*loadtim[k]/(dT*eig) - omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*loadmag[k]/(dT**3*eig**3) -
                        omega**4*loadtim[k]*(-dT*eig*loadtim[k + 1] + dT*eig*loadtim[k] -
                        2*omega**2*loadtim[k + 1]/(dT*eig) + 2*omega**2*loadtim[k]/(dT*eig) -
                        omega**4*loadtim[k + 1]/(dT**3*eig**3) +
                        omega**4*loadtim[k]/(dT**3*eig**3))**(-1)*cos(omega*t - omega*(t -
                        loadtim[k]) + phase)*exp(-dT*eig*(t - loadtim[k]))*loadmag[k +
                        1]/(dT**3*eig**3), axis=1)
            k = constants_containing_t[i]
            if len(k):
                A[i,:] += np.sum(-omega*(sin(omega*t + phase)/(1 + omega**2/(dT**2*eig**2)) - omega*cos(omega*t + phase)/(dT*eig*(1 +
                        omega**2/(dT**2*eig**2))))*loadmag[k]/(dT*eig) + omega*(exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)/(1 +
                        omega**2/(dT**2*eig**2)) - omega*cos(omega*t - omega*(t - loadtim[k]) +
                        phase)*exp(-dT*eig*(t - loadtim[k]))/(dT*eig*(1 +
                        omega**2/(dT**2*eig**2))))*loadmag[k]/(dT*eig), axis=1)
            k = constants_less_than_t[i]
            if len(k):
                A[i,:] += np.sum(-omega*(exp(-dT*eig*(t - loadtim[k + 1]))*sin(omega*t - omega*(t - loadtim[k + 1]) + phase)/(1 +
                        omega**2/(dT**2*eig**2)) - omega*cos(omega*t - omega*(t - loadtim[k + 1]) +
                        phase)*exp(-dT*eig*(t - loadtim[k + 1]))/(dT*eig*(1 +
                        omega**2/(dT**2*eig**2))))*loadmag[k]/(dT*eig) + omega*(exp(-dT*eig*(t -
                        loadtim[k]))*sin(omega*t - omega*(t - loadtim[k]) + phase)/(1 +
                        omega**2/(dT**2*eig**2)) - omega*cos(omega*t - omega*(t - loadtim[k]) +
                        phase)*exp(-dT*eig*(t - loadtim[k]))/(dT*eig*(1 +
                        omega**2/(dT**2*eig**2))))*loadmag[k]/(dT*eig), axis=1)
    return A

    
def pdim1sin_DD_abDDf_linear(m, a, b, **kwargs):
    """Integration of sin(mi * z) * D[a(z) * b(z) D[sin(mj * z),z,2],z,2]
    between ztop and zbot where a(z) & b(z) is piecewise linear functions of z.

    Wrapper for dim1sin_DD_abDDf_linear to allow PolyLine inputs.


    Calulation of integrals is performed at each element of a square symmetric
    matrix (size depends on size of `m`).

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    a, b : PolyLine object
        PolyLine defining piecewise linear relationship.
    implementation : ['vectorized', 'scalar', 'fortran'], optional
        Functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        Default implementation='vectorized'.  If fortran extention module
        cannot be imported then 'vectorized' version will be used.
        If anything other than 'fortran' or 'scalar' is used then
        default vectorized version will be used.

    Returns
    -------
    A : numpy.ndarray
        A square symmetric matrix, size determined by size of `m`.

    See Also
    -------
    dim1sin_DD_abDDf_linear : Wrapped function.


    Notes
    -----
    The `dim1sin_DD_abDDf_linear` matrix, :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{\\frac{d^2}{dz^2}\\left({a\\left(z\\right)}{b\\left(z\\right)}\\frac{d^2\\phi_j}{dz^2}\\right)\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    .. math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` and :math:`b\\left(z\\right)` are piecewise
    linear functions with respect to :math:`z`, that within a layer is defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    To make the above integration simpler we integate by parts to get:

    .. math:: \\mathbf{A}_{i,j}= \\left.{\\frac{d}{dz}\\left({a\\left(z\\right)}{b\\left(z\\right)}\\frac{d^2\\phi_j}{dz^2}\\right)\\phi_i}\\right|_{z=0}^{z=1} 
                    - \\left.{{a\\left(z\\right)}{b\\left(z\\right)}\\frac{d^2\\phi_j}{dz^2}\\frac{d\\phi_i}{dz}}\\right|_{z=0}^{z=1} 
               +\\int_{0}^1{{a\\left(z\\right)}{b\\left(z\\right)}\\frac{d^2\\phi_j}{dz^2}\\frac{d^2\\phi_i}{dz^2}\\,dz}

    In this case the sine basis functions means the end point terms in the above
    equation are zero, leaving us with

    .. math:: \\mathbf{A}_{i,j}= \\int_{0}^1{{a\\left(z\\right)}{b\\left(z\\right)}\\frac{d^2\\phi_j}{dz^2}\\frac{d^2\\phi_i}{dz^2}\\,dz}


    """

    a, b = pwise.polyline_make_x_common(a,b)
    return dim1sin_DD_abDDf_linear(m, a.y1, a.y2, b.y1, b.y2, a.x1, a.x2, **kwargs)
    

def dim1sin_DD_abDDf_linear(m, at, ab, bt, bb,  zt, zb, implementation='vectorized'):
    """Integration of sin(mi * z) * D[a(z) * b(z) D[sin(mj * z),z,2],z,2]
    between ztop and zbot where a(z) & b(z) is piecewise linear functions of z.

    Calulation of integrals is performed at each element of a square symmetric
    matrix (size depends on size of `m`).

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    at : ``list`` of ``float``
        Property at top of each layer.
    ab : ``list`` of ``float``
        Property at bottom of each layer.
    bt : ``list`` of ``float``
        2nd property at top of each layer.
    bb : ``list`` of ``float``
        2nd property at bottom of each layer.
    zt : ``list`` of ``float``
        Normalised depth or z-coordinate at top of each layer. `zt[0]` = 0
    zb : ``list`` of ``float``
        Normalised depth or z-coordinate at bottom of each layer. `zt[-1]` = 1
    implementation : ['vectorized', 'scalar', 'fortran'], optional
        Functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest), 'vectorized' = numpy(fast).
        Default implementation='vectorized'.  If fortran extention module
        cannot be imported then 'vectorized' version will be used.
        If anything other than 'fortran' or 'scalar' is used then
        default vectorized version will be used.

    Returns
    -------
    A : numpy.ndarray
        A square symmetric matrix, size determined by size of `m`.

    See Also
    --------
    m_from_sin_mx : Used to generate 'm' input parameter.
    geotecha.integrals_generate_code.dim1sin_DD_abDDf_linear_implementations : Generation
        of code for this function.
    pdim1sin_DD_abDDf_linear : Same function with PolyLine
        inputs.
    geotecha.speccon.ext_integrals.dim1sin_dd_abddf_linear : Equivalent fortran
        function.

    Notes
    -----
    The `dim1sin_DD_abDDf_linear` matrix, :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{\\frac{d^2}{dz^2}\\left({a\\left(z\\right)}{b\\left(z\\right)}\\frac{d^2\\phi_j}{dz^2}\\right)\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    .. math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` and :math:`b\\left(z\\right)` are piecewise
    linear functions with respect to :math:`z`, that within a layer is defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    To make the above integration simpler we integate by parts to get:

    .. math:: \\mathbf{A}_{i,j}= \\left.{\\frac{d}{dz}\\left({a\\left(z\\right)}{b\\left(z\\right)}\\frac{d^2\\phi_j}{dz^2}\\right)\\phi_i}\\right|_{z=0}^{z=1} 
                    - \\left.{{a\\left(z\\right)}{b\\left(z\\right)}\\frac{d^2\\phi_j}{dz^2}\\frac{d\\phi_i}{dz}}\\right|_{z=0}^{z=1} 
               +\\int_{0}^1{{a\\left(z\\right)}{b\\left(z\\right)}\\frac{d^2\\phi_j}{dz^2}\\frac{d^2\\phi_i}{dz^2}\\,dz}

    In this case the sine basis functions means the end point terms in the above
    equation are zero, leaving us with

    .. math:: \\mathbf{A}_{i,j}= \\int_{0}^1{{a\\left(z\\right)}{b\\left(z\\right)}\\frac{d^2\\phi_j}{dz^2}\\frac{d^2\\phi_i}{dz^2}\\,dz}


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
            a_slope = (ab[layer] - at[layer]) / (zb[layer] - zt[layer])
            b_slope = (bb[layer] - bt[layer]) / (zb[layer] - zt[layer])
            for i in range(neig):
                A[i, i] += (-m[i]**4*(a_slope*b_slope*m[i]**(-3)*cos(zt[layer]*m[i])*sin(zt[layer]*m[i])/4 +
                    a_slope*b_slope*zt[layer]*m[i]**(-2)*sin(zt[layer]*m[i])**2/4 +
                    a_slope*b_slope*zt[layer]*m[i]**(-2)*cos(zt[layer]*m[i])**2/4 +
                    a_slope*b_slope*zt[layer]**3*sin(zt[layer]*m[i])**2/6 +
                    a_slope*b_slope*zt[layer]**3*cos(zt[layer]*m[i])**2/6 -
                    a_slope*bt[layer]*m[i]**(-2)*cos(zt[layer]*m[i])**2/4 -
                    a_slope*zt[layer]**2*bt[layer]*sin(zt[layer]*m[i])**2/4 -
                    a_slope*zt[layer]**2*bt[layer]*cos(zt[layer]*m[i])**2/4 -
                    b_slope*at[layer]*m[i]**(-2)*cos(zt[layer]*m[i])**2/4 -
                    b_slope*zt[layer]**2*at[layer]*sin(zt[layer]*m[i])**2/4 -
                    b_slope*zt[layer]**2*at[layer]*cos(zt[layer]*m[i])**2/4 -
                    bt[layer]*at[layer]*m[i]**(-1)*cos(zt[layer]*m[i])*sin(zt[layer]*m[i])/2 +
                    zt[layer]*bt[layer]*at[layer]*sin(zt[layer]*m[i])**2/2 +
                    zt[layer]*bt[layer]*at[layer]*cos(zt[layer]*m[i])**2/2) +
                    m[i]**4*(a_slope*b_slope*m[i]**(-3)*cos(zb[layer]*m[i])*sin(zb[layer]*m[i])/4 +
                    a_slope*b_slope*zb[layer]*m[i]**(-2)*sin(zb[layer]*m[i])**2/4 -
                    a_slope*b_slope*zb[layer]*m[i]**(-2)*cos(zb[layer]*m[i])**2/4 +
                    a_slope*b_slope*zb[layer]*zt[layer]*m[i]**(-1)*cos(zb[layer]*m[i])*sin(zb[layer]*m[i])
                    + a_slope*b_slope*zb[layer]*zt[layer]**2*sin(zb[layer]*m[i])**2/2 +
                    a_slope*b_slope*zb[layer]*zt[layer]**2*cos(zb[layer]*m[i])**2/2 -
                    a_slope*b_slope*zb[layer]**2*m[i]**(-1)*cos(zb[layer]*m[i])*sin(zb[layer]*m[i])/2
                    - a_slope*b_slope*zb[layer]**2*zt[layer]*sin(zb[layer]*m[i])**2/2 -
                    a_slope*b_slope*zb[layer]**2*zt[layer]*cos(zb[layer]*m[i])**2/2 +
                    a_slope*b_slope*zb[layer]**3*sin(zb[layer]*m[i])**2/6 +
                    a_slope*b_slope*zb[layer]**3*cos(zb[layer]*m[i])**2/6 +
                    a_slope*b_slope*zt[layer]*m[i]**(-2)*cos(zb[layer]*m[i])**2/2 -
                    a_slope*b_slope*zt[layer]**2*m[i]**(-1)*cos(zb[layer]*m[i])*sin(zb[layer]*m[i])/2
                    - a_slope*bt[layer]*m[i]**(-2)*cos(zb[layer]*m[i])**2/4 -
                    a_slope*zb[layer]*bt[layer]*m[i]**(-1)*cos(zb[layer]*m[i])*sin(zb[layer]*m[i])/2
                    - a_slope*zb[layer]*zt[layer]*bt[layer]*sin(zb[layer]*m[i])**2/2 -
                    a_slope*zb[layer]*zt[layer]*bt[layer]*cos(zb[layer]*m[i])**2/2 +
                    a_slope*zb[layer]**2*bt[layer]*sin(zb[layer]*m[i])**2/4 +
                    a_slope*zb[layer]**2*bt[layer]*cos(zb[layer]*m[i])**2/4 +
                    a_slope*zt[layer]*bt[layer]*m[i]**(-1)*cos(zb[layer]*m[i])*sin(zb[layer]*m[i])/2
                    - b_slope*at[layer]*m[i]**(-2)*cos(zb[layer]*m[i])**2/4 -
                    b_slope*zb[layer]*at[layer]*m[i]**(-1)*cos(zb[layer]*m[i])*sin(zb[layer]*m[i])/2
                    - b_slope*zb[layer]*zt[layer]*at[layer]*sin(zb[layer]*m[i])**2/2 -
                    b_slope*zb[layer]*zt[layer]*at[layer]*cos(zb[layer]*m[i])**2/2 +
                    b_slope*zb[layer]**2*at[layer]*sin(zb[layer]*m[i])**2/4 +
                    b_slope*zb[layer]**2*at[layer]*cos(zb[layer]*m[i])**2/4 +
                    b_slope*zt[layer]*at[layer]*m[i]**(-1)*cos(zb[layer]*m[i])*sin(zb[layer]*m[i])/2
                    - bt[layer]*at[layer]*m[i]**(-1)*cos(zb[layer]*m[i])*sin(zb[layer]*m[i])/2 +
                    zb[layer]*bt[layer]*at[layer]*sin(zb[layer]*m[i])**2/2 +
                    zb[layer]*bt[layer]*at[layer]*cos(zb[layer]*m[i])**2/2))
            for i in range(neig-1):
                for j in range(i + 1, neig):
                    A[i, j] += (-m[j]**2*m[i]**2*(2*a_slope*b_slope*sin(zt[layer]*m[j])*m[i]**3*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zt[layer]*m[i]) -
                        6*a_slope*b_slope*m[j]*cos(zt[layer]*m[j])*m[i]**2*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zt[layer]*m[i]) +
                        6*a_slope*b_slope*m[j]**2*sin(zt[layer]*m[j])*m[i]*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zt[layer]*m[i]) -
                        2*a_slope*b_slope*m[j]**3*cos(zt[layer]*m[j])*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zt[layer]*m[i]) +
                        a_slope*bt[layer]*sin(zt[layer]*m[j])*m[i]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zt[layer]*m[i]) +
                        2*a_slope*bt[layer]*m[j]*cos(zt[layer]*m[j])*m[i]**3*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zt[layer]*m[i]) -
                        2*a_slope*bt[layer]*m[j]**3*cos(zt[layer]*m[j])*m[i]*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zt[layer]*m[i]) -
                        a_slope*bt[layer]*m[j]**4*sin(zt[layer]*m[j])*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zt[layer]*m[i]) +
                        b_slope*at[layer]*sin(zt[layer]*m[j])*m[i]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zt[layer]*m[i]) +
                        2*b_slope*at[layer]*m[j]*cos(zt[layer]*m[j])*m[i]**3*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zt[layer]*m[i]) -
                        2*b_slope*at[layer]*m[j]**3*cos(zt[layer]*m[j])*m[i]*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zt[layer]*m[i]) -
                        b_slope*at[layer]*m[j]**4*sin(zt[layer]*m[j])*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zt[layer]*m[i]) -
                        bt[layer]*at[layer]*sin(zt[layer]*m[j])*m[i]**5*(m[i]**6 - 3*m[j]**2*m[i]**4
                        + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zt[layer]*m[i]) +
                        bt[layer]*at[layer]*m[j]*cos(zt[layer]*m[j])*m[i]**4*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zt[layer]*m[i]) +
                        2*bt[layer]*at[layer]*m[j]**2*sin(zt[layer]*m[j])*m[i]**3*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zt[layer]*m[i]) -
                        2*bt[layer]*at[layer]*m[j]**3*cos(zt[layer]*m[j])*m[i]**2*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zt[layer]*m[i]) -
                        bt[layer]*at[layer]*m[j]**4*sin(zt[layer]*m[j])*m[i]*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zt[layer]*m[i]) +
                        bt[layer]*at[layer]*m[j]**5*cos(zt[layer]*m[j])*(m[i]**6 - 3*m[j]**2*m[i]**4
                        + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zt[layer]*m[i])) +
                        m[j]**2*m[i]**2*(2*a_slope*b_slope*sin(zb[layer]*m[j])*m[i]**3*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) -
                        6*a_slope*b_slope*m[j]*cos(zb[layer]*m[j])*m[i]**2*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) +
                        6*a_slope*b_slope*m[j]**2*sin(zb[layer]*m[j])*m[i]*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) -
                        2*a_slope*b_slope*m[j]**3*cos(zb[layer]*m[j])*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) +
                        2*a_slope*b_slope*zb[layer]*sin(zb[layer]*m[j])*m[i]**4*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) +
                        4*a_slope*b_slope*zb[layer]*m[j]*cos(zb[layer]*m[j])*m[i]**3*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) -
                        4*a_slope*b_slope*zb[layer]*m[j]**3*cos(zb[layer]*m[j])*m[i]*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) -
                        2*a_slope*b_slope*zb[layer]*m[j]**4*sin(zb[layer]*m[j])*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) +
                        2*a_slope*b_slope*zb[layer]*zt[layer]*sin(zb[layer]*m[j])*m[i]**5*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) -
                        2*a_slope*b_slope*zb[layer]*zt[layer]*m[j]*cos(zb[layer]*m[j])*m[i]**4*(m[i]**6
                        - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i])
                        -
                        4*a_slope*b_slope*zb[layer]*zt[layer]*m[j]**2*sin(zb[layer]*m[j])*m[i]**3*(m[i]**6
                        - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i])
                        +
                        4*a_slope*b_slope*zb[layer]*zt[layer]*m[j]**3*cos(zb[layer]*m[j])*m[i]**2*(m[i]**6
                        - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i])
                        +
                        2*a_slope*b_slope*zb[layer]*zt[layer]*m[j]**4*sin(zb[layer]*m[j])*m[i]*(m[i]**6
                        - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i])
                        - 2*a_slope*b_slope*zb[layer]*zt[layer]*m[j]**5*cos(zb[layer]*m[j])*(m[i]**6
                        - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i])
                        - a_slope*b_slope*zb[layer]**2*sin(zb[layer]*m[j])*m[i]**5*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) +
                        a_slope*b_slope*zb[layer]**2*m[j]*cos(zb[layer]*m[j])*m[i]**4*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) +
                        2*a_slope*b_slope*zb[layer]**2*m[j]**2*sin(zb[layer]*m[j])*m[i]**3*(m[i]**6
                        - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i])
                        -
                        2*a_slope*b_slope*zb[layer]**2*m[j]**3*cos(zb[layer]*m[j])*m[i]**2*(m[i]**6
                        - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i])
                        - a_slope*b_slope*zb[layer]**2*m[j]**4*sin(zb[layer]*m[j])*m[i]*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) +
                        a_slope*b_slope*zb[layer]**2*m[j]**5*cos(zb[layer]*m[j])*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) -
                        2*a_slope*b_slope*zt[layer]*sin(zb[layer]*m[j])*m[i]**4*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) -
                        4*a_slope*b_slope*zt[layer]*m[j]*cos(zb[layer]*m[j])*m[i]**3*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) +
                        4*a_slope*b_slope*zt[layer]*m[j]**3*cos(zb[layer]*m[j])*m[i]*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) +
                        2*a_slope*b_slope*zt[layer]*m[j]**4*sin(zb[layer]*m[j])*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) -
                        a_slope*b_slope*zt[layer]**2*sin(zb[layer]*m[j])*m[i]**5*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) +
                        a_slope*b_slope*zt[layer]**2*m[j]*cos(zb[layer]*m[j])*m[i]**4*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) +
                        2*a_slope*b_slope*zt[layer]**2*m[j]**2*sin(zb[layer]*m[j])*m[i]**3*(m[i]**6
                        - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i])
                        -
                        2*a_slope*b_slope*zt[layer]**2*m[j]**3*cos(zb[layer]*m[j])*m[i]**2*(m[i]**6
                        - 3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i])
                        - a_slope*b_slope*zt[layer]**2*m[j]**4*sin(zb[layer]*m[j])*m[i]*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) +
                        a_slope*b_slope*zt[layer]**2*m[j]**5*cos(zb[layer]*m[j])*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) +
                        a_slope*bt[layer]*sin(zb[layer]*m[j])*m[i]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) +
                        2*a_slope*bt[layer]*m[j]*cos(zb[layer]*m[j])*m[i]**3*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) -
                        2*a_slope*bt[layer]*m[j]**3*cos(zb[layer]*m[j])*m[i]*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) -
                        a_slope*bt[layer]*m[j]**4*sin(zb[layer]*m[j])*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) -
                        a_slope*zb[layer]*bt[layer]*sin(zb[layer]*m[j])*m[i]**5*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) +
                        a_slope*zb[layer]*bt[layer]*m[j]*cos(zb[layer]*m[j])*m[i]**4*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) +
                        2*a_slope*zb[layer]*bt[layer]*m[j]**2*sin(zb[layer]*m[j])*m[i]**3*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) -
                        2*a_slope*zb[layer]*bt[layer]*m[j]**3*cos(zb[layer]*m[j])*m[i]**2*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) -
                        a_slope*zb[layer]*bt[layer]*m[j]**4*sin(zb[layer]*m[j])*m[i]*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) +
                        a_slope*zb[layer]*bt[layer]*m[j]**5*cos(zb[layer]*m[j])*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) +
                        a_slope*zt[layer]*bt[layer]*sin(zb[layer]*m[j])*m[i]**5*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) -
                        a_slope*zt[layer]*bt[layer]*m[j]*cos(zb[layer]*m[j])*m[i]**4*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) -
                        2*a_slope*zt[layer]*bt[layer]*m[j]**2*sin(zb[layer]*m[j])*m[i]**3*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) +
                        2*a_slope*zt[layer]*bt[layer]*m[j]**3*cos(zb[layer]*m[j])*m[i]**2*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) +
                        a_slope*zt[layer]*bt[layer]*m[j]**4*sin(zb[layer]*m[j])*m[i]*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) -
                        a_slope*zt[layer]*bt[layer]*m[j]**5*cos(zb[layer]*m[j])*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) +
                        b_slope*at[layer]*sin(zb[layer]*m[j])*m[i]**4*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) +
                        2*b_slope*at[layer]*m[j]*cos(zb[layer]*m[j])*m[i]**3*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) -
                        2*b_slope*at[layer]*m[j]**3*cos(zb[layer]*m[j])*m[i]*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) -
                        b_slope*at[layer]*m[j]**4*sin(zb[layer]*m[j])*(m[i]**6 - 3*m[j]**2*m[i]**4 +
                        3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) -
                        b_slope*zb[layer]*at[layer]*sin(zb[layer]*m[j])*m[i]**5*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) +
                        b_slope*zb[layer]*at[layer]*m[j]*cos(zb[layer]*m[j])*m[i]**4*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) +
                        2*b_slope*zb[layer]*at[layer]*m[j]**2*sin(zb[layer]*m[j])*m[i]**3*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) -
                        2*b_slope*zb[layer]*at[layer]*m[j]**3*cos(zb[layer]*m[j])*m[i]**2*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) -
                        b_slope*zb[layer]*at[layer]*m[j]**4*sin(zb[layer]*m[j])*m[i]*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) +
                        b_slope*zb[layer]*at[layer]*m[j]**5*cos(zb[layer]*m[j])*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) +
                        b_slope*zt[layer]*at[layer]*sin(zb[layer]*m[j])*m[i]**5*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) -
                        b_slope*zt[layer]*at[layer]*m[j]*cos(zb[layer]*m[j])*m[i]**4*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) -
                        2*b_slope*zt[layer]*at[layer]*m[j]**2*sin(zb[layer]*m[j])*m[i]**3*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) +
                        2*b_slope*zt[layer]*at[layer]*m[j]**3*cos(zb[layer]*m[j])*m[i]**2*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) +
                        b_slope*zt[layer]*at[layer]*m[j]**4*sin(zb[layer]*m[j])*m[i]*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) -
                        b_slope*zt[layer]*at[layer]*m[j]**5*cos(zb[layer]*m[j])*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) -
                        bt[layer]*at[layer]*sin(zb[layer]*m[j])*m[i]**5*(m[i]**6 - 3*m[j]**2*m[i]**4
                        + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) +
                        bt[layer]*at[layer]*m[j]*cos(zb[layer]*m[j])*m[i]**4*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) +
                        2*bt[layer]*at[layer]*m[j]**2*sin(zb[layer]*m[j])*m[i]**3*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) -
                        2*bt[layer]*at[layer]*m[j]**3*cos(zb[layer]*m[j])*m[i]**2*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i]) -
                        bt[layer]*at[layer]*m[j]**4*sin(zb[layer]*m[j])*m[i]*(m[i]**6 -
                        3*m[j]**2*m[i]**4 + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*cos(zb[layer]*m[i]) +
                        bt[layer]*at[layer]*m[j]**5*cos(zb[layer]*m[j])*(m[i]**6 - 3*m[j]**2*m[i]**4
                        + 3*m[j]**4*m[i]**2 - m[j]**6)**(-1)*sin(zb[layer]*m[i])))

        #A is symmetric
        for i in range(neig - 1):
            for j in range(i + 1, neig):
                A[j, i] = A[i, j]

    elif implementation == 'fortran':
        try:
            import geotecha.speccon.ext_integrals as ext_integ
            A = ext_integ.dim1sin_dd_abddf_linear(m, at, ab, bt, bb, zt, zb)
        except ImportError:
            A = dim1sin_DD_abDDf_linear(m, at, ab, bt, bb, zt, zb, implementation='vectorized')

    else:#default is 'vectorized' using numpy
        sin = np.sin
        cos = np.cos
        A = np.zeros([neig, neig], float)

        diag =  np.diag_indices(neig)
        triu = np.triu_indices(neig, k = 1)
        tril = (triu[1], triu[0])

        a_slope = (ab - at) / (zb - zt)
        b_slope = (bb - bt) / (zb - zt)

        mi = m[:, np.newaxis]
        A[diag] = np.sum(-mi**4*(a_slope*b_slope*zt**3*sin(mi*zt)**2/6 + a_slope*b_slope*zt**3*cos(mi*zt)**2/6 +
            a_slope*b_slope*zt*sin(mi*zt)**2/(4*mi**2) + a_slope*b_slope*zt*cos(mi*zt)**2/(4*mi**2)
            + a_slope*b_slope*cos(mi*zt)*sin(mi*zt)/(4*mi**3) - a_slope*zt**2*sin(mi*zt)**2*bt/4 -
            a_slope*zt**2*cos(mi*zt)**2*bt/4 - a_slope*cos(mi*zt)**2*bt/(4*mi**2) -
            b_slope*zt**2*sin(mi*zt)**2*at/4 - b_slope*zt**2*cos(mi*zt)**2*at/4 -
            b_slope*cos(mi*zt)**2*at/(4*mi**2) + zt*sin(mi*zt)**2*bt*at/2 + zt*cos(mi*zt)**2*bt*at/2
            - cos(mi*zt)*sin(mi*zt)*bt*at/(2*mi)) + mi**4*(a_slope*b_slope*zb*sin(mi*zb)**2*zt**2/2
            + a_slope*b_slope*zb*cos(mi*zb)**2*zt**2/2 - a_slope*b_slope*zb**2*sin(mi*zb)**2*zt/2 -
            a_slope*b_slope*zb**2*cos(mi*zb)**2*zt/2 + a_slope*b_slope*zb**3*sin(mi*zb)**2/6 +
            a_slope*b_slope*zb**3*cos(mi*zb)**2/6 -
            a_slope*b_slope*cos(mi*zb)*sin(mi*zb)*zt**2/(2*mi) +
            a_slope*b_slope*zb*cos(mi*zb)*sin(mi*zb)*zt/mi -
            a_slope*b_slope*zb**2*cos(mi*zb)*sin(mi*zb)/(2*mi) +
            a_slope*b_slope*cos(mi*zb)**2*zt/(2*mi**2) + a_slope*b_slope*zb*sin(mi*zb)**2/(4*mi**2)
            - a_slope*b_slope*zb*cos(mi*zb)**2/(4*mi**2) +
            a_slope*b_slope*cos(mi*zb)*sin(mi*zb)/(4*mi**3) - a_slope*zb*sin(mi*zb)**2*zt*bt/2 -
            a_slope*zb*cos(mi*zb)**2*zt*bt/2 + a_slope*zb**2*sin(mi*zb)**2*bt/4 +
            a_slope*zb**2*cos(mi*zb)**2*bt/4 + a_slope*cos(mi*zb)*sin(mi*zb)*zt*bt/(2*mi) -
            a_slope*zb*cos(mi*zb)*sin(mi*zb)*bt/(2*mi) - a_slope*cos(mi*zb)**2*bt/(4*mi**2) -
            b_slope*zb*sin(mi*zb)**2*zt*at/2 - b_slope*zb*cos(mi*zb)**2*zt*at/2 +
            b_slope*zb**2*sin(mi*zb)**2*at/4 + b_slope*zb**2*cos(mi*zb)**2*at/4 +
            b_slope*cos(mi*zb)*sin(mi*zb)*zt*at/(2*mi) - b_slope*zb*cos(mi*zb)*sin(mi*zb)*at/(2*mi)
            - b_slope*cos(mi*zb)**2*at/(4*mi**2) + zb*sin(mi*zb)**2*bt*at/2 +
            zb*cos(mi*zb)**2*bt*at/2 - cos(mi*zb)*sin(mi*zb)*bt*at/(2*mi)), axis=1)

        mi = m[triu[0]][:, np.newaxis]
        mj = m[triu[1]][:, np.newaxis]
        A[triu] = np.sum(-mi**2*mj**2*(2*a_slope*b_slope*mi**3*cos(mi*zt)*sin(mj*zt)/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) - 6*a_slope*b_slope*mi**2*mj*cos(mj*zt)*sin(mi*zt)/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + 6*a_slope*b_slope*mi*mj**2*cos(mi*zt)*sin(mj*zt)/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) -
            2*a_slope*b_slope*mj**3*cos(mj*zt)*sin(mi*zt)/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) + a_slope*mi**4*sin(mi*zt)*sin(mj*zt)*bt/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) + 2*a_slope*mi**3*mj*cos(mi*zt)*cos(mj*zt)*bt/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - 2*a_slope*mi*mj**3*cos(mi*zt)*cos(mj*zt)*bt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) - a_slope*mj**4*sin(mi*zt)*sin(mj*zt)*bt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) + b_slope*mi**4*sin(mi*zt)*sin(mj*zt)*at/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            2*b_slope*mi**3*mj*cos(mi*zt)*cos(mj*zt)*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) - 2*b_slope*mi*mj**3*cos(mi*zt)*cos(mj*zt)*at/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - b_slope*mj**4*sin(mi*zt)*sin(mj*zt)*at/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - mi**5*cos(mi*zt)*sin(mj*zt)*bt*at/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + mi**4*mj*cos(mj*zt)*sin(mi*zt)*bt*at/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + 2*mi**3*mj**2*cos(mi*zt)*sin(mj*zt)*bt*at/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) -
            2*mi**2*mj**3*cos(mj*zt)*sin(mi*zt)*bt*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) - mi*mj**4*cos(mi*zt)*sin(mj*zt)*bt*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) + mj**5*cos(mj*zt)*sin(mi*zt)*bt*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6)) + mi**2*mj**2*(-a_slope*b_slope*mi**5*cos(mi*zb)*sin(mj*zb)*zt**2/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            2*a_slope*b_slope*mi**5*zb*cos(mi*zb)*sin(mj*zb)*zt/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - a_slope*b_slope*mi**5*zb**2*cos(mi*zb)*sin(mj*zb)/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            a_slope*b_slope*mi**4*mj*cos(mj*zb)*sin(mi*zb)*zt**2/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - 2*a_slope*b_slope*mi**4*mj*zb*cos(mj*zb)*sin(mi*zb)*zt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            a_slope*b_slope*mi**4*mj*zb**2*cos(mj*zb)*sin(mi*zb)/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - 2*a_slope*b_slope*mi**4*sin(mi*zb)*sin(mj*zb)*zt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            2*a_slope*b_slope*mi**4*zb*sin(mi*zb)*sin(mj*zb)/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4
            - mj**6) + 2*a_slope*b_slope*mi**3*mj**2*cos(mi*zb)*sin(mj*zb)*zt**2/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) -
            4*a_slope*b_slope*mi**3*mj**2*zb*cos(mi*zb)*sin(mj*zb)*zt/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) +
            2*a_slope*b_slope*mi**3*mj**2*zb**2*cos(mi*zb)*sin(mj*zb)/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - 4*a_slope*b_slope*mi**3*mj*cos(mi*zb)*cos(mj*zb)*zt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            4*a_slope*b_slope*mi**3*mj*zb*cos(mi*zb)*cos(mj*zb)/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + 2*a_slope*b_slope*mi**3*cos(mi*zb)*sin(mj*zb)/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) -
            2*a_slope*b_slope*mi**2*mj**3*cos(mj*zb)*sin(mi*zb)*zt**2/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) +
            4*a_slope*b_slope*mi**2*mj**3*zb*cos(mj*zb)*sin(mi*zb)*zt/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) -
            2*a_slope*b_slope*mi**2*mj**3*zb**2*cos(mj*zb)*sin(mi*zb)/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - 6*a_slope*b_slope*mi**2*mj*cos(mj*zb)*sin(mi*zb)/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) -
            a_slope*b_slope*mi*mj**4*cos(mi*zb)*sin(mj*zb)*zt**2/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + 2*a_slope*b_slope*mi*mj**4*zb*cos(mi*zb)*sin(mj*zb)*zt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) -
            a_slope*b_slope*mi*mj**4*zb**2*cos(mi*zb)*sin(mj*zb)/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + 4*a_slope*b_slope*mi*mj**3*cos(mi*zb)*cos(mj*zb)*zt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) -
            4*a_slope*b_slope*mi*mj**3*zb*cos(mi*zb)*cos(mj*zb)/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + 6*a_slope*b_slope*mi*mj**2*cos(mi*zb)*sin(mj*zb)/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            a_slope*b_slope*mj**5*cos(mj*zb)*sin(mi*zb)*zt**2/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4
            - mj**6) - 2*a_slope*b_slope*mj**5*zb*cos(mj*zb)*sin(mi*zb)*zt/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + a_slope*b_slope*mj**5*zb**2*cos(mj*zb)*sin(mi*zb)/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            2*a_slope*b_slope*mj**4*sin(mi*zb)*sin(mj*zb)*zt/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4
            - mj**6) - 2*a_slope*b_slope*mj**4*zb*sin(mi*zb)*sin(mj*zb)/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - 2*a_slope*b_slope*mj**3*cos(mj*zb)*sin(mi*zb)/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            a_slope*mi**5*cos(mi*zb)*sin(mj*zb)*zt*bt/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) - a_slope*mi**5*zb*cos(mi*zb)*sin(mj*zb)*bt/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - a_slope*mi**4*mj*cos(mj*zb)*sin(mi*zb)*zt*bt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            a_slope*mi**4*mj*zb*cos(mj*zb)*sin(mi*zb)*bt/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) + a_slope*mi**4*sin(mi*zb)*sin(mj*zb)*bt/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) - 2*a_slope*mi**3*mj**2*cos(mi*zb)*sin(mj*zb)*zt*bt/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + 2*a_slope*mi**3*mj**2*zb*cos(mi*zb)*sin(mj*zb)*bt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            2*a_slope*mi**3*mj*cos(mi*zb)*cos(mj*zb)*bt/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) + 2*a_slope*mi**2*mj**3*cos(mj*zb)*sin(mi*zb)*zt*bt/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - 2*a_slope*mi**2*mj**3*zb*cos(mj*zb)*sin(mi*zb)*bt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            a_slope*mi*mj**4*cos(mi*zb)*sin(mj*zb)*zt*bt/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) - a_slope*mi*mj**4*zb*cos(mi*zb)*sin(mj*zb)*bt/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - 2*a_slope*mi*mj**3*cos(mi*zb)*cos(mj*zb)*bt/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) -
            a_slope*mj**5*cos(mj*zb)*sin(mi*zb)*zt*bt/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) + a_slope*mj**5*zb*cos(mj*zb)*sin(mi*zb)*bt/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - a_slope*mj**4*sin(mi*zb)*sin(mj*zb)*bt/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + b_slope*mi**5*cos(mi*zb)*sin(mj*zb)*zt*at/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) -
            b_slope*mi**5*zb*cos(mi*zb)*sin(mj*zb)*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) - b_slope*mi**4*mj*cos(mj*zb)*sin(mi*zb)*zt*at/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + b_slope*mi**4*mj*zb*cos(mj*zb)*sin(mi*zb)*at/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) + b_slope*mi**4*sin(mi*zb)*sin(mj*zb)*at/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) -
            2*b_slope*mi**3*mj**2*cos(mi*zb)*sin(mj*zb)*zt*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4
            - mj**6) + 2*b_slope*mi**3*mj**2*zb*cos(mi*zb)*sin(mj*zb)*at/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + 2*b_slope*mi**3*mj*cos(mi*zb)*cos(mj*zb)*at/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            2*b_slope*mi**2*mj**3*cos(mj*zb)*sin(mi*zb)*zt*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4
            - mj**6) - 2*b_slope*mi**2*mj**3*zb*cos(mj*zb)*sin(mi*zb)*at/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) + b_slope*mi*mj**4*cos(mi*zb)*sin(mj*zb)*zt*at/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) -
            b_slope*mi*mj**4*zb*cos(mi*zb)*sin(mj*zb)*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) - 2*b_slope*mi*mj**3*cos(mi*zb)*cos(mj*zb)*at/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - b_slope*mj**5*cos(mj*zb)*sin(mi*zb)*zt*at/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) +
            b_slope*mj**5*zb*cos(mj*zb)*sin(mi*zb)*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) - b_slope*mj**4*sin(mi*zb)*sin(mj*zb)*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) - mi**5*cos(mi*zb)*sin(mj*zb)*bt*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) + mi**4*mj*cos(mj*zb)*sin(mi*zb)*bt*at/(mi**6 - 3*mi**4*mj**2 + 3*mi**2*mj**4 -
            mj**6) + 2*mi**3*mj**2*cos(mi*zb)*sin(mj*zb)*bt*at/(mi**6 - 3*mi**4*mj**2 +
            3*mi**2*mj**4 - mj**6) - 2*mi**2*mj**3*cos(mj*zb)*sin(mi*zb)*bt*at/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) - mi*mj**4*cos(mi*zb)*sin(mj*zb)*bt*at/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6) + mj**5*cos(mj*zb)*sin(mi*zb)*bt*at/(mi**6 -
            3*mi**4*mj**2 + 3*mi**2*mj**4 - mj**6)), axis=1)
        #A is symmetric
        A[tril] = A[triu]

    return A
    
if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])