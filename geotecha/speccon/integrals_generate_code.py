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
"""use sympy to generate code for generating spectral method matrix subroutines"""

from __future__ import division, print_function

import sympy
import textwrap
import os

from geotecha.inputoutput.inputoutput import fcode_one_large_expr

def linear(x, x1, y1, x2, y2):
    """Interpolation between two points.

    """
    return (y2 - y1)/(x2 - x1) * (x-x1) + y1

def string_to_IndexedBase(s):
    """turn string into sympy.tensor.IndexedBase

    """

    return sympy.tensor.IndexedBase(s)

def create_layer_sympy_var_and_maps(layer_prop=['z','kv','kh','et', 'mv',
                                                'surz', 'vacz']):
    """Create sympy variables and maps for use with integrating \
    to generate 1d spectral equations.

    Each x in layer prop will get a 'top', and 'bot', suffix
    Each 'xtop' will get mapped to 'xt[layer]', each 'xbot' to 'xb[layer]
    and added to `prop_map` dict.
    Each s in layer prop will get a linear representation added to the
    `linear_expression` dict:
        x = (xbot-xtop)/(zbot-ztop)*(z-zbot) +xtop
    `prop_map` will also get a 'mi' to m[i], and mj to m[j] map.
    'z, mi, mj' will become global variables

    Parameters
    ----------
    layer_prop : ``list`` of ``str``, optional
        label for properties that vary in a layer
        (default is ['z','kv','kh','et', 'mv', 'surz',' vacz'])


    Returns
    -------
    prop_map : ``dict``
        maps the string version of a variable to the sympy.tensor.IndexedBase
        version e.g. prop_map['kvtop'] = kvt[layer]
    linear_expressions : ``dict``
        maps the string version of a variable to an expression describing how
        that varibale varies linearly within a layer

    Examples
    --------
    >>> prop_map, linear_expressions = create_layer_sympy_var_and_maps(layer_prop=['z','kv'])
    >>> prop_map
    {'kvtop': kvt[layer], 'mi': m[i], 'mj': m[j], 'zbot': zb[layer], 'ztop': zt[layer], 'kvbot': kvb[layer]}
    >>> linear_expressions
    {'z': z, 'kv': kvtop + (kvbot - kvtop)*(z - ztop)/(zbot - ztop)}

    """
    #http://www.daniweb.com/software-development/python/threads/111526/setting-a-string-as-a-variable-name
    m = sympy.tensor.IndexedBase('m')
    neig, nlayers = sympy.var('neig, nlayers', integer=True)
    i = sympy.tensor.Idx('i', neig)
    j = sympy.tensor.Idx('j', neig)
    layer = sympy.tensor.Idx('layer', nlayers)

    sympy.var('z, mi, mj')
    suffix={'t':'top','b': 'bot'}
    prop_map = {}
    linear_expressions ={}

    prop_map['mi'] = m[i]
    prop_map['mj'] = m[j]

    for prop in layer_prop:
        for s1, s3 in suffix.iteritems():
            vars()[prop + s1] = string_to_IndexedBase(prop + s1)
            sympy.var(prop + s3)
            prop_map[prop + s3] = vars()[prop + s1][layer]
        linear_expressions[prop]=linear(z, ztop, eval(prop+suffix['t']), zbot, eval(prop+suffix['b']))
    return (prop_map, linear_expressions)

def dim1sin_af_linear():
    """Generate code to calculate spectral method integrations

    Performs integrations of `sin(mi * z) * a(z) * sin(mj * z)` between [0, 1]
    where a(z) is a piecewise linear function of z.  Code is generated that
    will produce a square array with the appropriate integrals at each location

    Paste the resulting code (at least the loops) into `dim1sin_af_linear`.

    Notes
    -----
    The `dim1sin_af_linear` matrix, :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{{a\\left(z\\right)}\\phi_i\\phi_j\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    ..math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` is a piecewise linear function
    w.r.t. :math:`z`, that within a layer are defined by:

    ..math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """

    mp, p = create_layer_sympy_var_and_maps(layer_prop=['z','a'])

    phi_i = sympy.sin(mi * z)
    phi_j = sympy.sin(mj * z)

    fdiag = sympy.integrate(p['a'] * phi_i * phi_i, z)
    fdiag = fdiag.subs(z, mp['zbot']) - fdiag.subs(z, mp['ztop'])
    fdiag = fdiag.subs(mp)

    foff = sympy.integrate(p['a'] * phi_j * phi_i, z)
    foff = foff.subs(z, mp['zbot']) - foff.subs(z, mp['ztop'])
    foff = foff.subs(mp)

    text = """def dim1sin_af_linear(m, at, ab, zt, zb):
    import numpy as np
    from math import sin, cos

    neig = len(m)
    nlayers = len(zt)

    A = np.zeros([neig, neig], float)
    for layer in range(nlayers):
        for i in range(neig):
            A[i, i] += {}
        for i in range(neig-1):
            for j in range(i + 1, neig):
                A[i, j] += {}

    #A is symmetric
    for i in range(neig - 1):
        for j in range(i + 1, neig):
            A[j, i] = A[i, j]

    return A"""


    fn = text.format(fdiag, foff)

    return fn


def dim1sin_abf_linear():
    """Generate code to calculate spectral method integrations

    Performs integrations of `sin(mi * z) * a(z) *b(z) * sin(mj * z)`
    between [0, 1] where a(z) and b(z) are piecewise linear functions of z.
    Code is generated that will produce a square array with the appropriate
    integrals at each location

    Paste the resulting code (at least the loops) into `dim1sin_abf_linear`.

    Notes
    -----
    The `dim1sin_abf_linear` matrix, :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{{a\\left(z\\right)}{b\\left(z\\right)}\\phi_i\\phi_j\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    ..math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` and :math:`b\\left(z\\right)` are piecewise
    linear functions w.r.t. :math:`z`, that within a layer are defined by:

    ..math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """

    mp, p = create_layer_sympy_var_and_maps(layer_prop=['z','a', 'b'])

    phi_i = sympy.sin(mi * z)
    phi_j = sympy.sin(mj * z)

    fdiag = sympy.integrate(p['a'] * p['b'] * phi_i * phi_i, z)
    fdiag = fdiag.subs(z, mp['zbot']) - fdiag.subs(z, mp['ztop'])
    fdiag = fdiag.subs(mp)

    foff = sympy.integrate(p['a'] * p['b'] * phi_j * phi_i, z)
    foff = foff.subs(z, mp['zbot']) - foff.subs(z, mp['ztop'])
    foff = foff.subs(mp)

    text = """def dim1sin_abf_linear(m, at, ab, bt, bb,  zt, zb):
    import numpy as np
    from math import sin, cos

    neig = len(m)
    nlayers = len(zt)

    A = np.zeros([neig, neig], float)
    for layer in range(nlayers):
        for i in range(neig):
            A[i, i] += {}
        for i in range(neig-1):
            for j in range(i + 1, neig):
                A[i, j] += {}

    #A is symmetric
    for i in range(neig - 1):
        for j in range(i + 1, neig):
            A[j, i] = A[i, j]

    return A"""


    fn = text.format(fdiag, foff)

    return fn

def dim1sin_D_aDf_linear():
    """Generate code to calculate spectral method integrations

    Performs integrations of `sin(mi * z) * D[a(z) * D[sin(mj * z),z],z]`
    between [0, 1] where a(z) i piecewise linear functions of z.
    Code is generated that will produce a square array with the appropriate
    integrals at each location

    Paste the resulting code (at least the loops) into `dim1sin_D_aDf_linear`.

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
    each layer respectively.

    The integration requires some explanation.  The difficultly arises because
    piecewise linear :math:`a\\left(z\\right)` is defined using step functions at
    the top and bottom of each layer; those step functions when differentiated
    yield dirac-delta or impulse functions which must be handled specially
    when integrating against the spectral basis functions.  The graph below
    shows what happens when the :math:`a\\left(z\\right) distribution is
    differentiated for one layer.

    ::

        y(x)
        ^
        |
        |                            yb
        |                          /|
        |                         / |
        |                        /  |
        |                       /   |
        |                      /    |
        |                   yt|     |
        |                     |     |
        0--------------------xt----xb------------------------1------>x

    Differentiate:

    ::

        y'(x)
        ^
        |
        |                     y(xt) * Dirac(x - xt)
        |                     ^
        |                     |
        |                     |     |
        |                     |     |
        |               y'(xt)|-----|
        |                     |     |
        |                     |     v - y(xb) * Dirac(x - xb)
        0--------------------xt----xb------------------------1------>x

    With sympy/sage I've found it easier to perform the indefinite integrals
    first  and then sub in the bounds of integration. That is, the integral
    of f between a and b is F(b)-F(a) where F is the anti-derivative of f.


    The general expression for :math:`{A}` is:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{\\frac{d}{dz}\\left({a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\right)\\phi_i\\,dz}
        :label: full

    Expanding out the integrals in :eq:`full` yields:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{{a\\left(z\\right)}\\frac{d^2\\phi_j}{dZ^2}\\phi_i\\,dZ}+\\int_{0}^1{\\frac{d{a\\left(z\\right)}}{dZ}\\frac{d\\phi_j}{dZ}\\phi_i\\,dZ}

    Considering a single layer and separating the layer boundaries from the behaviour within a layer gives:

    .. math:: \\mathbf{A}_{i,j,\\text{layer}}=\\int_{z_t}^{z_b}{{a\\left(z\\right)}\\frac{d^2\\phi_j}{dz^2}\\phi_i\\,dZ}+\\int_{z_t}^{z_b}{\\frac{d{a\\left(z\\right)}}{dz}\\frac{d\\phi_j}{dz}\\phi_i\\,dz}+\\int_{0}^{1}{{a\\left(z\\right)}\\delta\\left(z-z_t\\right)\\frac{d\\phi_j}{dz}\\phi_i\\,dz}-\\int_{0}^{1}{{a\\left(z\\right)}\\delta\\left(z-z_b\\right)\\frac{d\\phi_j}{dz}\\phi_i\\,dz}

    Performing the dirac delta integrations:

    .. math:: \\mathbf{A}_{i,j,\\text{layer}}=\\int_{z_t}^{z_b}{{a\\left(z\\right)}\\frac{d^2\\phi_j}{dz^2}\\phi_i\\,dZ}+\\int_{z_t}^{z_b}{\\frac{d{a\\left(z\\right)}}{dz}\\frac{d\\phi_j}{dz}\\phi_i\\,dz}+\\left.{a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\phi_i\\right|_{z=z_t}-\\left.{a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\phi_i\\right|_{z=z_b}

    Now, to get it in the form of F(zb)-F(zt) we only take the zb part of the dirac integration:

    .. math:: F\\left(z\\right)=\\int{{a\\left(z\\right)}\\frac{d^2\\phi_j}{dz^2}\\phi_i\\,dZ}+\\int{\\frac{d{a\\left(z\\right)}}{dz}\\frac{d\\phi_j}{dz}\\phi_i\\,dz}-\\left.{a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\phi_i\\right|_{z=z}

    Finally we get:

    .. math:: \\mathbf{A}_{i,j,\\text{layer}}=F\\left(z_b\\right)-F\\left(z_t\\right)

    TODO: explain why dirac integrations disapear at end points because in this case it will always be sin(mx)*cos(mx) and so always be zero.

    """

    mp, p = create_layer_sympy_var_and_maps(layer_prop=['z','a'])

    phi_i = sympy.sin(mi * z)
    phi_j = sympy.sin(mj * z)

    fdiag = (sympy.integrate(p['a'] * sympy.diff(phi_i, z, 2) * phi_i, z))
    fdiag += (sympy.integrate(sympy.diff(p['a'], z) * sympy.diff(phi_i, z) * phi_i,z))
    fdiag -= (p['a'] * sympy.diff(phi_i, z) * phi_i)
        # note the 'negative' for the diff (a) part is because the step fn
        #at the top and bottom of the layer yields a dirac function that is
        #positive at ztop and negative at zbot. It works because definite
        #integral of f between ztop and zbot is F(ztop)- F(zbot).
        #i.e. I've created the indefininte integral such that when i sub
        #in ztop and zbot in the next step i get the correct contribution
        #for the step functions at ztop and zbot
    fdiag = fdiag.subs(z, mp['zbot']) - fdiag.subs(z, mp['ztop'])
    fdiag = fdiag.subs(mp)

    foff = (sympy.integrate(p['a'] * sympy.diff(phi_j, z, 2) * phi_i, z))
    foff += (sympy.integrate(sympy.diff(p['a'], z) * sympy.diff(phi_j, z) * phi_i,z))
    foff -= (p['a'] * sympy.diff(phi_j, z) * phi_i)
    foff = foff.subs(z, mp['zbot']) - foff.subs(z, mp['ztop'])
    foff = foff.subs(mp)

    text = """def dim1sin_D_aDf_linear(m, at, ab, zt, zb):
    import numpy as np
    from math import sin, cos

    neig = len(m)
    nlayers = len(zt)

    A = np.zeros([neig, neig], float)
    for layer in range(nlayers):
        for i in range(neig):
            A[i, i] += {}
        for i in range(neig-1):
            for j in range(i + 1, neig):
                A[i, j] += {}

    #A is symmetric
    for i in range(neig - 1):
        for j in range(i + 1, neig):
            A[j, i] = A[i, j]

    return A"""


    fn = text.format(fdiag, foff)

    return fn


def dim1sin_ab_linear():
    """Generate code to calculate spectral method integrations

    Performs integrations of `sin(mi * z) * a(z) * b(z)`
    between [0, 1] where a(z) and b(z) are piecewise linear functions of z.
    Code is generated that will produce a 1d array with the appropriate
    integrals at each location.

    Paste the resulting code (at least the loops) into `dim1sin_ab_linear`.

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

    mp, p = create_layer_sympy_var_and_maps(layer_prop=['z', 'a', 'b'])

    phi_i = sympy.sin(mi * z)

    fcol = sympy.integrate(p['a'] * p['b'] * phi_i, z)
    fcol = fcol.subs(z, mp['zbot']) - fcol.subs(z, mp['ztop'])
    fcol = fcol.subs(mp)

    text = """def dim1sin_ab_linear(m, at, ab, bt, bb, zt, zb):
    import numpy as np
    from math import sin, cos

    neig = len(m)
    nlayers = len(zt)

    A = np.zeros(neig, float)
    for layer in range(nlayers):
        for i in range(neig):
            A[i] += {}

    return A"""


    fn = text.format(fcol)
    return fn

def dim1sin_abc_linear():
    """Generate code to calculate spectral method integrations

    Performs integrations of `sin(mi * z) * a(z) * b(z) * c(z)`
    between [0, 1] where a(z), b(z), c(z) are piecewise linear functions of z.
    Code is generated that will produce a 1d array with the appropriate
    integrals at each location.

    Paste the resulting code (at least the loops) into `dim1sin_abc_linear`.

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

    mp, p = create_layer_sympy_var_and_maps(layer_prop=['z', 'a', 'b', 'c'])

    phi_i = sympy.sin(mi * z)

    fcol = sympy.integrate(p['a'] * p['b'] * p['c'] * phi_i, z)
    fcol = fcol.subs(z, mp['zbot']) - fcol.subs(z, mp['ztop'])
    fcol = fcol.subs(mp)

    text = """def dim1sin_abc_linear(m, at, ab, bt, bb, ct, cb, zt, zb):
    import numpy as np
    from math import sin, cos

    neig = len(m)
    nlayers = len(zt)

    A = np.zeros(neig, float)
    for layer in range(nlayers):
        for i in range(neig):
            A[i] += {}

    return A"""


    fn = text.format(fcol)
    return fn

def dim1sin_D_aDb_linear():
    """Generate code to calculate spectral method integrations

    Performs integrations of `sin(mi * z) * D[a(z) * D[b(z), z], z]`
    between [0, 1] where a(z) and b(z) are piecewise linear functions of z.
    Code is generated that will produce a 1d array with the appropriate
    integrals at each location.

    Paste the resulting code (at least the loops) into `dim1sin_D_aDb_linear`.

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

    TODO: explain why the dirac integrations at 0 and 1 must be omitted i.e. not for all cases do they not contribute only.

    """

    mp, p = create_layer_sympy_var_and_maps(layer_prop=['z', 'a', 'b'])

    phi_i = sympy.sin(mi * z)

    fcol = sympy.integrate(sympy.diff(p['a'],z) * sympy.diff(p['b'],z) * phi_i, z)
    fcol -= p['a']*sympy.diff(p['b'], z) * phi_i
    fcol = fcol.subs(z, mp['zbot']) - fcol.subs(z, mp['ztop'])
    fcol = fcol.subs(mp)


    ffirst = -p['a']*sympy.diff(p['b'], z) * phi_i
    ffirst = ffirst.subs(z, mp['ztop'])
    ffirst = ffirst.subs(mp)
    ffirst = ffirst.subs(list(sympy.tensor.get_indices(mp['ztop'])[0])[0], 0)

    flast = p['a']*sympy.diff(p['b'], z) * phi_i
    flast = flast.subs(z, mp['zbot'])
    flast = flast.subs(mp)
    flast=flast.subs(list(sympy.tensor.get_indices(mp['ztop'])[0])[0], -1)

    fends = ffirst + flast

    text = """def dim1sin_D_aDb_linear(m, at, ab, bt, bb, zt, zb):
    import numpy as np
    from math import sin, cos

    neig = len(m)
    nlayers = len(zt)

    A = np.zeros(neig, float)
    for layer in range(nlayers):
        for i in range(neig):
            A[i] += {}

    for i in range(neig):
        A[i] += {}

    return A"""


    fn = text.format(fcol, fends)
    return fn



def EDload_linear():
    """Generate code to perform time integration for spectral methods

    Perform time integrations of a piecewise linear load vs time


    Notes
    -----
    the default output for the integrals will have expression s like
    exp(-dT*eig*t)*exp(dT*eig*loadtim[k]).  when dT*eig*loadtim[k] is large
    the exponential may be so large as to cause an error.  YOu may need to
    manually alter the expression to exp(is large exp(-dT*eig*(t-loadtim[k]))
    in which case the term in the exponential will always be negative and not
    lead to any numerical blow up.

    """

    from sympy import exp, symbols
    t, tau, dT, eig= symbols('t, tau, dT, eig')
#    sympy.assume(dT>0)
#    sympy.assume(eig>0)
#    sympy.var('t, tau, dT, eig')
    loadmag = sympy.tensor.IndexedBase('loadmag')
    loadtim = sympy.tensor.IndexedBase('loadtim')
    tvals = sympy.tensor.IndexedBase('tvals')
    eigs = sympy.tensor.IndexedBase('eigs')
    i = sympy.tensor.Idx('i')
    j = sympy.tensor.Idx('j')
    k = sympy.tensor.Idx('k')

    mp = [(exp(-dT*eig*t)*exp(dT*eig*loadtim[k]),exp(-dT*eig*(t-loadtim[k]))),(exp(-dT*eig*t)*exp(dT*eig*loadtim[k+1]),exp(-dT*eig*(t-loadtim[k+1])))]
#    the default output for the integrals will have expression s like
#    exp(-dT*eig*t)*exp(dT*eig*loadtim[k]).  when dT*eig*loadtim[k] is large
#    the exponential may be so large as to cause an error.  YOu may need to
#    manually alter the expression to exp(is large exp(-dT*eig*(t-loadtim[k]))
#    in which case the term in the exponential will always be negative and not
#    lead to any numerical blow up.
#    load = linear(tau, loadtim[k], loadmag[k], loadtim[k+1], loadmag[k+1])
#    after_instant = (loadmag[k+1] - loadmag[k]) * exp(-dT * eig * (t - loadtim[k]))
#    mp does this automatically with subs

    Dload = sympy.diff(linear(tau, loadtim[k], loadmag[k], loadtim[k+1], loadmag[k+1]), tau)
    after_instant = (loadmag[k+1] - loadmag[k]) * exp(-dT * eig * (t - loadtim[k]))

    within_ramp = Dload * exp(-dT * eig * (t - tau))
    within_ramp = sympy.integrate(within_ramp, (tau, loadtim[k], t), risch=False, conds='none')
    within_ramp = within_ramp.subs(mp)

    after_ramp = Dload * exp(-dT * eig * (t - tau))
    after_ramp = sympy.integrate(after_ramp, (tau, loadtim[k], loadtim[k+1]), risch=False, conds='none')
    after_ramp = after_ramp.subs(mp)


    text = """def EDload_linear(loadtim, loadmag, eigs, tvals):

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
                A[i,j] += {}
        for k in ramps_containing_t[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += {}
        for k in ramps_less_than_t[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += {}
    return A"""

    fn = text.format(after_instant, within_ramp, after_ramp)
    return fn

def Eload_linear():
    """Generate code to perform time integration for spectral methods

    Perform time integrations of a piecewise linear load vs time




    """

    from sympy import exp

    sympy.var('t, tau, dT, eig')
    loadmag = sympy.tensor.IndexedBase('loadmag')
    loadtim = sympy.tensor.IndexedBase('loadtim')
    tvals = sympy.tensor.IndexedBase('tvals')
    eigs = sympy.tensor.IndexedBase('eigs')
    i = sympy.tensor.Idx('i')
    j = sympy.tensor.Idx('j')
    k = sympy.tensor.Idx('k')

    mp = [(exp(-dT*eig*t)*exp(dT*eig*loadtim[k]),exp(-dT*eig*(t-loadtim[k]))),(exp(-dT*eig*t)*exp(dT*eig*loadtim[k+1]),exp(-dT*eig*(t-loadtim[k+1])))]
#    the default output for the integrals will have expression s like
#    exp(-dT*eig*t)*exp(dT*eig*loadtim[k]).  when dT*eig*loadtim[k] is large
#    the exponential may be so large as to cause an error.  YOu may need to
#    manually alter the expression to exp(is large exp(-dT*eig*(t-loadtim[k]))
#    in which case the term in the exponential will always be negative and not
#    lead to any numerical blow up.
#    load = linear(tau, loadtim[k], loadmag[k], loadtim[k+1], loadmag[k+1])
#    after_instant = (loadmag[k+1] - loadmag[k]) * exp(-dT * eig * (t - loadtim[k]))
#    mp does this automatically with subs
    load = linear(tau, loadtim[k], loadmag[k], loadtim[k+1], loadmag[k+1])
    within_constant = loadmag[k] * exp(-dT * eig * (t - tau))
    within_constant = sympy.integrate(within_constant, (tau, loadtim[k], t))
    within_constant = within_constant.subs(mp)

    after_constant = loadmag[k] * exp(-dT * eig * (t - tau))
    after_constant = sympy.integrate(after_constant, (tau, loadtim[k], loadtim[k+1]))
    after_constant = after_constant.subs(mp)

    within_ramp = load * exp(-dT * eig * (t - tau))
    within_ramp = sympy.integrate(within_ramp, (tau, loadtim[k], t))
    within_ramp = within_ramp.subs(mp)

    after_ramp = load * exp(-dT * eig * (t - tau))
    after_ramp = sympy.integrate(after_ramp, (tau, loadtim[k], loadtim[k+1]))
    after_ramp = after_ramp.subs(mp)


    text = """def Eload_linear(loadtim, loadmag, eigs, tvals):

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
                A[i,j] += {}
        for k in constants_less_than_t[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += {}
        for k in ramps_containing_t[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += {}
        for k in ramps_less_than_t[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += {}
    return A"""

    fn = text.format(within_constant, after_constant, within_ramp, after_ramp)
    return fn



def Eload_coslinear():
    """Generate code to perform time integration for spectral methods

    Perform time integrations of a piecewise linear load vs time multipled
    by cos(omega * t + phase).


    Notes
    -----
    the default output for the integrals will have expression s like
    exp(-dT*eig*t)*exp(dT*eig*loadtim[k]).  when dT*eig*loadtim[k] is large
    the exponential may be so large as to cause an error.  YOu may need to
    manually alter the expression to exp(is large exp(-dT*eig*(t-loadtim[k]))
    in which case the term in the exponential will always be negative and not
    lead to any numerical blow up.

    """

    from sympy import exp

    sympy.var('t, tau, dT, eig, omega, phase')
    loadmag = sympy.tensor.IndexedBase('loadmag')
    loadtim = sympy.tensor.IndexedBase('loadtim')
    tvals = sympy.tensor.IndexedBase('tvals')
    eigs = sympy.tensor.IndexedBase('eigs')
    i = sympy.tensor.Idx('i')
    j = sympy.tensor.Idx('j')
    k = sympy.tensor.Idx('k')

    mp = [(exp(-dT*eig*t)*exp(dT*eig*loadtim[k]),
               exp(-dT*eig*(t-loadtim[k]))),
          (exp(-dT*eig*t)*exp(dT*eig*loadtim[k+1]),
               exp(-dT*eig*(t-loadtim[k+1]))),
          ((dT**2*eig**2*exp(dT*eig*t)*exp(-dT*eig*loadtim[k]) + omega**2*exp(dT*eig*t)*exp(-dT*eig*loadtim[k]))**(-1),
               (dT**2*eig**2 + omega**2)**(-1)*exp(-dT*eig*(t-loadtim[k]))),
          ((dT**2*eig**2*exp(dT*eig*t)*exp(-dT*eig*loadtim[k + 1]) + omega**2*exp(dT*eig*t)*exp(-dT*eig*loadtim[k + 1]))**(-1),
               (dT**2*eig**2 + omega**2)**(-1)*exp(-dT*eig*(t-loadtim[k + 1]))),
          ((dT**4*eig**4*exp(dT*eig*t)*exp(-dT*eig*loadtim[k]) + 2*dT**2*eig**2*omega**2*exp(dT*eig*t)*exp(-dT*eig*loadtim[k]) + omega**4*exp(dT*eig*t)*exp(-dT*eig*loadtim[k]))**(-1),
               (dT**2*eig**2 + omega**2)**(-2)*exp(-dT*eig*(t-loadtim[k]))),
          ((dT**4*eig**4*exp(dT*eig*t)*exp(-dT*eig*loadtim[k + 1]) + 2*dT**2*eig**2*omega**2*exp(dT*eig*t)*exp(-dT*eig*loadtim[k + 1]) + omega**4*exp(dT*eig*t)*exp(-dT*eig*loadtim[k + 1]))**(-1),
               (dT**2*eig**2 + omega**2)**(-2)*exp(-dT*eig*(t-loadtim[k + 1]))),]
#    the default output for the integrals will have expression s like
#    exp(-dT*eig*t)*exp(dT*eig*loadtim[k]).  when dT*eig*loadtim[k] is large
#    the exponential may be so large as to cause an error.  YOu may need to
#    manually alter the expression to exp(is large exp(-dT*eig*(t-loadtim[k]))
#    in which case the term in the exponential will always be negative and not
#    lead to any numerical blow up.
#    load = linear(tau, loadtim[k], loadmag[k], loadtim[k+1], loadmag[k+1])
#    after_instant = (loadmag[k+1] - loadmag[k]) * exp(-dT * eig * (t - loadtim[k]))
#    mp does this automatically with subs
    load1 = sympy.cos(omega * tau + phase)
    load2 = linear(tau, loadtim[k], loadmag[k], loadtim[k+1], loadmag[k+1])

#    load = sympy.cos(omega * tau + phase)*linear(tau, loadtim[k], loadmag[k], loadtim[k+1], loadmag[k+1])

    within_constant = loadmag[k] * load1 *  exp(-dT * eig * (t - tau))
    within_constant = sympy.integrate(within_constant, (tau, loadtim[k], t))
    within_constant = within_constant.subs(mp)

    after_constant = loadmag[k] * load1 * exp(-dT * eig * (t - tau))
    after_constant = sympy.integrate(after_constant, (tau, loadtim[k], loadtim[k+1]))
    after_constant = after_constant.subs(mp)

    within_ramp = load1 * load2 * exp(-dT * eig * (t - tau))
    within_ramp = sympy.integrate(within_ramp, (tau, loadtim[k], t))
    within_ramp = within_ramp.subs(mp)

    after_ramp = load1 *load2 * exp(-dT * eig * (t - tau))
    after_ramp = sympy.integrate(after_ramp, (tau, loadtim[k], loadtim[k+1]))
    after_ramp = after_ramp.subs(mp)


    text = """def Eload_coslinear(loadtim, loadmag, omega, phase, eigs, tvals, dT=1.0):

    from math import exp
    cos=math.cos
    sin=math.sin

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
                A[i,j] += ({})
        for k in constants_less_than_t[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += ({})
        for k in ramps_containing_t[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += ({})
        for k in ramps_less_than_t[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += ({})
    return A"""

    fn = text.format(tw(within_constant, 5), tw(after_constant, 5), tw(within_ramp, 5), tw(after_ramp, 5))
    return fn

def EDload_coslinear():
    """Generate code to perform time integration for spectral methods

    Perform time integrations of time derivative of a piecewise linear load
    vs time multipled by cos(omega * t + phase).


    Notes
    -----
    the default output for the integrals will have expression s like
    exp(-dT*eig*t)*exp(dT*eig*loadtim[k]).  when dT*eig*loadtim[k] is large
    the exponential may be so large as to cause an error.  YOu may need to
    manually alter the expression to exp(is large exp(-dT*eig*(t-loadtim[k]))
    in which case the term in the exponential will always be negative and not
    lead to any numerical blow up.

    """

    from sympy import exp

    sympy.var('t, tau, dT, eig, omega, phase')
    loadmag = sympy.tensor.IndexedBase('loadmag')
    loadtim = sympy.tensor.IndexedBase('loadtim')
    tvals = sympy.tensor.IndexedBase('tvals')
    eigs = sympy.tensor.IndexedBase('eigs')
    i = sympy.tensor.Idx('i')
    j = sympy.tensor.Idx('j')
    k = sympy.tensor.Idx('k')

    mp = [(exp(-dT*eig*t)*exp(dT*eig*loadtim[k]),
               exp(-dT*eig*(t-loadtim[k]))),
          (exp(-dT*eig*t)*exp(dT*eig*loadtim[k+1]),
               exp(-dT*eig*(t-loadtim[k+1]))),
          ((dT**2*eig**2*exp(dT*eig*t)*exp(-dT*eig*loadtim[k]) + omega**2*exp(dT*eig*t)*exp(-dT*eig*loadtim[k]))**(-1),
               (dT**2*eig**2 + omega**2)**(-1)*exp(-dT*eig*(t-loadtim[k]))),
          ((dT**2*eig**2*exp(dT*eig*t)*exp(-dT*eig*loadtim[k + 1]) + omega**2*exp(dT*eig*t)*exp(-dT*eig*loadtim[k + 1]))**(-1),
               (dT**2*eig**2 + omega**2)**(-1)*exp(-dT*eig*(t-loadtim[k + 1]))),
          ((dT**4*eig**4*exp(dT*eig*t)*exp(-dT*eig*loadtim[k]) + 2*dT**2*eig**2*omega**2*exp(dT*eig*t)*exp(-dT*eig*loadtim[k]) + omega**4*exp(dT*eig*t)*exp(-dT*eig*loadtim[k]))**(-1),
               (dT**2*eig**2 + omega**2)**(-2)*exp(-dT*eig*(t-loadtim[k]))),
          ((dT**4*eig**4*exp(dT*eig*t)*exp(-dT*eig*loadtim[k + 1]) + 2*dT**2*eig**2*omega**2*exp(dT*eig*t)*exp(-dT*eig*loadtim[k + 1]) + omega**4*exp(dT*eig*t)*exp(-dT*eig*loadtim[k + 1]))**(-1),
               (dT**2*eig**2 + omega**2)**(-2)*exp(-dT*eig*(t-loadtim[k + 1]))),]
#    the default output for the integrals will have expression s like
#    exp(-dT*eig*t)*exp(dT*eig*loadtim[k]).  when dT*eig*loadtim[k] is large
#    the exponential may be so large as to cause an error.  YOu may need to
#    manually alter the expression to exp(is large exp(-dT*eig*(t-loadtim[k]))
#    in which case the term in the exponential will always be negative and not
#    lead to any numerical blow up.
#    load = linear(tau, loadtim[k], loadmag[k], loadtim[k+1], loadmag[k+1])
#    after_instant = (loadmag[k+1] - loadmag[k]) * exp(-dT * eig * (t - loadtim[k]))
#    mp does this automatically with subs

    load1 = sympy.cos(omega * tau + phase)
    load2 = linear(tau, loadtim[k], loadmag[k], loadtim[k+1], loadmag[k+1])

    after_instant = (loadmag[k+1] - loadmag[k]) * sympy.cos(omega * loadtim[k] + phase) * exp(-dT * eig * (t - loadtim[k]))

    within_constant = loadmag[k] * sympy.diff(load1, tau) * exp(-dT * eig * (t - tau))
    within_constant = sympy.integrate(within_constant, (tau, loadtim[k], t))
    within_constant = within_constant.subs(mp)

    after_constant = loadmag[k] * sympy.diff(load1, tau) * exp(-dT * eig * (t - tau))
    after_constant = sympy.integrate(after_constant, (tau, loadtim[k], loadtim[k+1]))
    after_constant = after_constant.subs(mp)

    within_ramp = sympy.diff(load1 * load2, tau) * exp(-dT * eig * (t - tau))
    within_ramp = sympy.integrate(within_ramp, (tau, loadtim[k], t))
    within_ramp = within_ramp.subs(mp)

    after_ramp = sympy.diff(load1 * load2, tau) * exp(-dT * eig * (t - tau))
    after_ramp = sympy.integrate(after_ramp, (tau, loadtim[k], loadtim[k+1]))
    after_ramp = after_ramp.subs(mp)


    text = """def EDload_coslinear(loadtim, loadmag, omega, phase, eigs, tvals, dT=1.0):

    from math import exp
    cos=math.cos
    sin=math.sin

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
                A[i,j] += ({})
        for k in constants_containing_t[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += ({})
        for k in constants_less_than_t[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += ({})
        for k in ramps_containing_t[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += ({})
        for k in ramps_less_than_t[i]:
            for j, eig in enumerate(eigs):
                A[i,j] += ({})
    return A"""

    fn = text.format(tw(after_instant, 5), tw(within_constant,5), tw(after_constant,5), tw(within_ramp,5), tw(after_ramp,5))
    return fn


def dim1sin_a_linear_between_old():
    """Generate code to calculate spectral method integrations

    Performs integrations of `sin(mi * z) * a(z)`
    between [z1, z2] where a(z) is a piecewise linear functions of z.

    calculates array A[len(z), len(m)]

    Paste the resulting code (at least the loops) into `dim1sin_a_linear_between`.

    Notes
    -----
    The `dim1sin_a_linear_between`, :math:`A`, is given by:

    .. math:: \\mathbf{A}_{i}=\\int_{z_1}^z_2{{a\\left(z\\right)}\\phi_j\\,dz}

    where the basis function :math:`\\phi_j` is given by:

    ..math:: \\phi_j\\left(z\\right)=\\sin\\left({m_j}z\\right)

    and :math:`a\\left(z\\right)` is a piecewise
    linear functions w.r.t. :math:`z`, that within a layer are defined by:

    ..math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """

    mp, p = create_layer_sympy_var_and_maps(layer_prop=['z', 'a', 'b'])
    sympy.var('z1, z2')

    z1 = sympy.tensor.IndexedBase('z1')
    z2 = sympy.tensor.IndexedBase('z2')
    i = sympy.tensor.Idx('i')
    j = sympy.tensor.Idx('j')



    phi_j = sympy.sin(mj * z)

    f = sympy.integrate(p['a'] * phi_j, z)

    both = f.subs(z, z2[i]) - f.subs(z, z1[i])
    both = both.subs(mp)

    between = f.subs(z, mp['zbot']) - f.subs(z, mp['ztop'])
    between = between.subs(mp)

    z1_only = f.subs(z, mp['zbot']) - f.subs(z, z1[i])
    z1_only = z1_only.subs(mp)

    z2_only = f.subs(z, z2[i]) - f.subs(z, mp['ztop'])
    z2_only = z2_only.subs(mp)

    text = """dim1sin_a_linear_between(m, at, ab, zt, zb, z):
    import numpy as np
    from math import sin, cos
    m = np.asarray(m)
    z = np.asarray(z)

    a = z[:,0]
    b = z[:,1]

    z_for_interp=zt[:].append(zb[-1])


    (segment_both, segment_z1_only, segment_z2_only, segments_between) = segments_between_xi_and_xj(z_for_interp,a,b)

    nz = len(z)
    neig = len(m)
    for i in range(nz):
        for layer in segment_both:
            for j in range(neig):
                A[i,j] += {}
        for layer in segment_z1_only:
            for j in range(neig):
                A[i,j] += {}
        for layer in segments_between:
            for j in range(neig):
                A[i,j] += {}
        for layer in segment_z2_only:
            for j in range(neig):
                A[i,j] += {}
    return A"""


    fn = text.format(both, z1_only, between, z2_only)

    return fn

def dim1_ab_linear_between():
    """Generate code to calculate spectral method integrations

    Performs integrations of `a(z) * b(z)`
    between [z1, z2] where a(z) is a piecewise linear functions of z.
    calculates array A[len(z)]

    Paste the resulting code (at least the loops) into `piecewise_linear_1d.integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between`.

    Notes
    -----
    The `dim1sin_a_linear_between`, :math:`A`, is given by:

    .. math:: \\mathbf{A}_{i}=\\int_{z_1}^z_2{{a\\left(z\\right)}{b\\left(z\\right)}\\,dz}

    where :math:`a\\left(z\\right)` is a piecewise
    linear functions w.r.t. :math:`z`, that within a layer are defined by:

    ..math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """


#    Because this integration goes into piecewise_linear_1d rather than speccon.integrals I have had to use a separate map funciton to massage the variable names into a naming convention consistent with piecewise_linear_1d (this is m2 below).
#    As such don't base anything off this funciton unless you know what you are doing
    mp, p = create_layer_sympy_var_and_maps(layer_prop=['z', 'a', 'b'])
    sympy.var('z1, z2')

    z1 = sympy.tensor.IndexedBase('z1')
    z2 = sympy.tensor.IndexedBase('z2')
    i = sympy.tensor.Idx('i')
    j = sympy.tensor.Idx('j')


    sympy.var('x1a, x2a, y1a, y2a, x1b, x2b, y1b, y2b')
    seg = sympy.tensor.Idx('seg')
    x1a = sympy.tensor.IndexedBase('x1a')
    x2a = sympy.tensor.IndexedBase('x2a')
    y1a = sympy.tensor.IndexedBase('y1a')
    y2a = sympy.tensor.IndexedBase('y2a')
    y1b = sympy.tensor.IndexedBase('y1b')
    y2b = sympy.tensor.IndexedBase('y2b')
    xi = sympy.tensor.IndexedBase('xi')
    xj = sympy.tensor.IndexedBase('xj')
    #mp2 = {'zb[layer]': x2a[seg]}
    mp2 = [(mp['zbot'], x2a[seg]),
           (mp['ztop'], x1a[seg]),
           (mp['abot'], y2a[seg]),
           (mp['atop'], y1a[seg]),
           (mp['bbot'], y2b[seg]),
           (mp['btop'], y1b[seg]),
           (z1[i], xi[i]),
           (z2[i], xj[i])]
    #phi_j = sympy.sin(mj * z)

    f = sympy.integrate(p['a'] * p['b'], z)

    both = f.subs(z, z2[i]) - f.subs(z, z1[i])
    both = both.subs(mp).subs(mp2)

    between = f.subs(z, mp['zbot']) - f.subs(z, mp['ztop'])
    between = between.subs(mp).subs(mp2)

    z1_only = f.subs(z, mp['zbot']) - f.subs(z, z1[i])
    z1_only = z1_only.subs(mp).subs(mp2)

    z2_only = f.subs(z, z2[i]) - f.subs(z, mp['ztop'])
    z2_only = z2_only.subs(mp).subs(mp2)

    text = """A = np.zeros(len(xi))
    for i in range(len(xi)):
        for seg in segment_both[i]:
            A[i] += {}
        for seg in segment_xi_only[i]:
            A[i] += {}
        for seg in segments_between[i]:
            A[i] += {}
        for seg in segment_xj_only[i]:
            A[i] += {}

    return A"""


    fn = text.format(both, z1_only, between, z2_only)

    return fn


    fn = text.format(both, xi_only, between, xj_only)

    return fn


def dim1sin_D_aDf_linear_v2():
    """Generate code to calculate spectral method integrations

    Performs integrations of `sin(mi * z) * D[a(z) * D[sin(mj * z),z],z]`
    between [0, 1] where a(z) i piecewise linear functions of z.
    Code is generated that will produce a square array with the appropriate
    integrals at each location

    Paste the resulting code (at least the loops) into `dim1sin_D_aDf_linear`.

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
    each layer respectively.

    The integration requires some explanation.  The difficultly arises because
    piecewise linear :math:`a\\left(z\\right)` is defined using step functions at
    the top and bottom of each layer; those step functions when differentiated
    yield dirac-delta or impulse functions which must be handled specially
    when integrating against the spectral basis functions.  The graph below
    shows what happens when the :math:`a\\left(z\\right) distribution is
    differentiated for one layer.

    ::

        y(x)
        ^
        |
        |                            yb
        |                          /|
        |                         / |
        |                        /  |
        |                       /   |
        |                      /    |
        |                   yt|     |
        |                     |     |
        0--------------------xt----xb------------------------1------>x

    Differentiate:

    ::

        y'(x)
        ^
        |
        |                     y(xt) * Dirac(x - xt)
        |                     ^
        |                     |
        |                     |     |
        |                     |     |
        |               y'(xt)|-----|
        |                     |     |
        |                     |     v - y(xb) * Dirac(x - xb)
        0--------------------xt----xb------------------------1------>x

    With sympy/sage I've found it easier to perform the indefinite integrals
    first  and then sub in the bounds of integration. That is, the integral
    of f between a and b is F(b)-F(a) where F is the anti-derivative of f.


    The general expression for :math:`{A}` is:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{\\frac{d}{dz}\\left({a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\right)\\phi_i\\,dz}
        :label: full

    Expanding out the integrals in :eq:`full` yields:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{{a\\left(z\\right)}\\frac{d^2\\phi_j}{dZ^2}\\phi_i\\,dZ}+\\int_{0}^1{\\frac{d{a\\left(z\\right)}}{dZ}\\frac{d\\phi_j}{dZ}\\phi_i\\,dZ}

    Considering a single layer and separating the layer boundaries from the behaviour within a layer gives:

    .. math:: \\mathbf{A}_{i,j,\\text{layer}}=\\int_{z_t}^{z_b}{{a\\left(z\\right)}\\frac{d^2\\phi_j}{dz^2}\\phi_i\\,dZ}+\\int_{z_t}^{z_b}{\\frac{d{a\\left(z\\right)}}{dz}\\frac{d\\phi_j}{dz}\\phi_i\\,dz}+\\int_{0}^{1}{{a\\left(z\\right)}\\delta\\left(z-z_t\\right)\\frac{d\\phi_j}{dz}\\phi_i\\,dz}-\\int_{0}^{1}{{a\\left(z\\right)}\\delta\\left(z-z_b\\right)\\frac{d\\phi_j}{dz}\\phi_i\\,dz}

    Performing the dirac delta integrations:

    .. math:: \\mathbf{A}_{i,j,\\text{layer}}=\\int_{z_t}^{z_b}{{a\\left(z\\right)}\\frac{d^2\\phi_j}{dz^2}\\phi_i\\,dZ}+\\int_{z_t}^{z_b}{\\frac{d{a\\left(z\\right)}}{dz}\\frac{d\\phi_j}{dz}\\phi_i\\,dz}+\\left.{a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\phi_i\\right|_{z=z_t}-\\left.{a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\phi_i\\right|_{z=z_b}

    Now, to get it in the form of F(zb)-F(zt) we only take the zb part of the dirac integration:

    .. math:: F\\left(z\\right)=\\int{{a\\left(z\\right)}\\frac{d^2\\phi_j}{dz^2}\\phi_i\\,dZ}+\\int{\\frac{d{a\\left(z\\right)}}{dz}\\frac{d\\phi_j}{dz}\\phi_i\\,dz}-\\left.{a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\phi_i\\right|_{z=z}

    Finally we get:

    .. math:: \\mathbf{A}_{i,j,\\text{layer}}=F\\left(z_b\\right)-F\\left(z_t\\right)

    TODO: explain why dirac integrations disapear at end points because in this case it will always be sin(mx)*cos(mx) and so always be zero.

    """

    mp, p = create_layer_sympy_var_and_maps(layer_prop=['z','a'])

    phi_i = sympy.sin(mi * z)
    phi_j = sympy.sin(mj * z)

    #fdiag = (sympy.integrate(p['a'] * sympy.diff(phi_i, z, 2) * phi_i, z))
    #fdiag += (sympy.integrate(sympy.diff(p['a'], z) * sympy.diff(phi_i, z) * phi_i,z))
    fdiag = -sympy.integrate((p['a'] * sympy.diff(phi_i, z) * sympy.diff(phi_i,z)),z)
        # note the 'negative' for the diff (a) part is because the step fn
        #at the top and bottom of the layer yields a dirac function that is
        #positive at ztop and negative at zbot. It works because definite
        #integral of f between ztop and zbot is F(ztop)- F(zbot).
        #i.e. I've created the indefininte integral such that when i sub
        #in ztop and zbot in the next step i get the correct contribution
        #for the step functions at ztop and zbot
    fdiag = fdiag.subs(z, mp['zbot']) - fdiag.subs(z, mp['ztop'])
    fdiag = fdiag.subs(mp)

    #foff = (sympy.integrate(p['a'] * sympy.diff(phi_j, z, 2) * phi_i, z))
    #foff += (sympy.integrate(sympy.diff(p['a'], z) * sympy.diff(phi_j, z) * phi_i,z))
    foff = -sympy.integrate((p['a'] * sympy.diff(phi_j, z) * sympy.diff(phi_i,z)),z)
    foff = foff.subs(z, mp['zbot']) - foff.subs(z, mp['ztop'])
    foff = foff.subs(mp)

    text = """def dim1sin_D_aDf_linear(m, at, ab, zt, zb):
    import numpy as np
    from math import sin, cos

    neig = len(m)
    nlayers = len(zt)

    A = np.zeros([neig, neig], float)
    for layer in range(nlayers):
        for i in range(neig):
            A[i, i] += {}
        for i in range(neig-1):
            for j in range(i + 1, neig):
                A[i, j] += {}

    #A is symmetric
    for i in range(neig - 1):
        for j in range(i + 1, neig):
            A[j, i] = A[i, j]

    return A"""


    fn = text.format(fdiag, foff)

    return fn





def dim1sin_af_linear_vectorize():
    """Generate code to calculate spectral method integrations

    Performs integrations of `sin(mi * z) * a(z) * sin(mj * z)` between [0, 1]
    where a(z) is a piecewise linear function of z.  Code is generated that
    will produce a square array with the appropriate integrals at each location

    Paste the resulting code (at least the loops) into `dim1sin_af_linear`.

    Notes
    -----
    The `dim1sin_af_linear` matrix, :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{{a\\left(z\\right)}\\phi_i\\phi_j\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    ..math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` is a piecewise linear function
    w.r.t. :math:`z`, that within a layer are defined by:

    ..math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """

    mp, p = create_layer_sympy_var_and_maps_vectorized(layer_prop=['z','a'])

    phi_i = sympy.sin(mi * z)
    phi_j = sympy.sin(mj * z)

    fdiag = sympy.integrate(p['a'] * phi_i * phi_i, z)
    fdiag = fdiag.subs(z, zbot) - fdiag.subs(z, ztop)
    fdiag = fdiag.subs(mp)

    foff = sympy.integrate(p['a'] * phi_j * phi_i, z)
    foff = foff.subs(z, zbot) - foff.subs(z, ztop)
    foff = foff.subs(mp)

    text = """def dim1sin_af_linear(m, at, ab, zt, zb):
    import numpy as np
    from numpy import sin, cos

    neig = len(m)
    A = np.zeros((neig, neig), dtype = float)

    diag =  np.diag_indices_from(A)
    triu = np.triu_indices(neig, k = 1)
    tril = (triu[1], triu[0])

    mi = m[:, np.newaxis]
    A[diag] = np.sum({0}, axis=1)

    mi = m[triu[0]][:, np.newaxis]
    mj = m[triu[1]][:, np.newaxis]
    A[triu] = np.sum({1}, axis=1)

    A[tril] = A[triu]

    return A"""

    tw=textwrap.TextWrapper(width=100, subsequent_indent = "    "*2, break_long_words = False)

    fn = text.format('\n'.join(tw.wrap(str(fdiag))), '\n'.join(tw.wrap(str(foff))))

    return fn

def create_layer_sympy_var_and_maps_vectorized(layer_prop=['z','a']):
    """Similatr to create_layer_sympy_var_and_maps_vectorized but no [Layer] indexing

    See also
    --------
    create_layer_sympy_var_and_maps_vectorized

    """


    mp, p = create_layer_sympy_var_and_maps(layer_prop)

    for v in mp:
        mp[v] = sympy.var(v[:2])

    return mp, p


def dim1sin_af_linear_fortran():
    """Generate fortran code to calculate spectral method integrations


    """

    mp, p = create_layer_sympy_var_and_maps(layer_prop=['z','a'])

    phi_i = sympy.sin(mi * z)
    phi_j = sympy.sin(mj * z)

    fdiag = sympy.integrate(p['a'] * phi_i * phi_i, z)
    fdiag = fdiag.subs(z, mp['zbot']) - fdiag.subs(z, mp['ztop'])
    fdiag = fdiag.subs(mp)


    foff = sympy.integrate(p['a'] * phi_j * phi_i, z)
    foff = foff.subs(z, mp['zbot']) - foff.subs(z, mp['ztop'])
    foff = foff.subs(mp)

    text = """      SUBROUTINE dim1sin_af_linear(m, at, ab, zt, zb, a, neig, nlayers)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nlayers
        REAL(DP), intent(in), dimension(0:neig-1) ::m
        REAL(DP), intent(in), dimension(0:nlayers-1) :: at
        REAL(DP), intent(in), dimension(0:nlayers-1) :: ab
        REAL(DP), intent(in), dimension(0:nlayers-1) :: zt
        REAL(DP), intent(in), dimension(0:nlayers-1) :: zb
        REAL(DP), intent(out), dimension(0:neig-1, 0:neig-1) :: a
        INTEGER :: i , j, layer
        a=0.0D0
        DO layer = 0, nlayers-1
          DO i = 0, neig-1
{0}
            DO j = i+1, neig-1
{1}
            END DO
          END DO
        END DO

        DO i = 0, neig -2
          DO j = i + 1, neig-1
            a(j,i) = a(i, j)
          END DO
        END DO

      END SUBROUTINE"""


    fn = text.format(fcode_one_large_expr(fdiag, prepend='a(i, i) = a(i, i) + '),
                 fcode_one_large_expr(foff, prepend='a(i, j) = a(i, j) + '))

    return fn


def dim1sin_af_linear_implementations_old():
    """Generate code to calculate spectral method integrations

    Generate scalar and vectorized python code and fortran loops

    Performs integrations of `sin(mi * z) * a(z) * sin(mj * z)` between [0, 1]
    where a(z) is a piecewise linear function of z.  Code is generated that
    will produce a square array with the appropriate integrals at each location

    Paste the resulting code (at least the loops) into `dim1sin_af_linear`.

    Creates 3 implementations:
     - 'scalar', python loops (slowest)
     - 'vectorized', numpy (much faster than scalar)
     - 'fortran', fortran loops (fastest).  Needs to be compiled and interfaced
       with f2py.

    Returns
    -------
    fn: string
        Python code. with scalar (loops) and vectorized (numpy) implementations
        also calls the fortran version.
    fn2: string
        Fortran code.  needs to be compiled with f2py

    Notes
    -----
    The `dim1sin_af_linear` matrix, :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{{a\\left(z\\right)}\\phi_i\\phi_j\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    ..math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` is a piecewise linear function
    w.r.t. :math:`z`, that within a layer are defined by:

    ..math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """

    mpv, p = create_layer_sympy_var_and_maps_vectorized(layer_prop=['z','a'])
    mp, p = create_layer_sympy_var_and_maps(layer_prop=['z','a'])

    phi_i = sympy.sin(mi * z)
    phi_j = sympy.sin(mj * z)

    fdiag = sympy.integrate(p['a'] * phi_i * phi_i, z)
    fdiag_loops = fdiag.subs(z, mp['zbot']) - fdiag.subs(z, mp['ztop'])
    fdiag_loops = fdiag_loops.subs(mp)
    fdiag_vector = fdiag.subs(z, mpv['zbot']) - fdiag.subs(z, mpv['ztop'])
    fdiag_vector = fdiag_vector.subs(mpv)

    foff = sympy.integrate(p['a'] * phi_j * phi_i, z)
    foff_loops = foff.subs(z, mp['zbot']) - foff.subs(z, mp['ztop'])
    foff_loops = foff_loops.subs(mp)
    foff_vector = foff.subs(z, mpv['zbot']) - foff.subs(z, mpv['ztop'])
    foff_vector = foff_vector.subs(mpv)

    text_python = """def dim1sin_af_linear(m, at, ab, zt, zb, implementation='vectorized'):

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
                A[i, i] += ({0})
            for i in range(neig-1):
                for j in range(i + 1, neig):
                    A[i, j] += ({1})

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
        A[diag] = np.sum({2}, axis=1)

        mi = m[triu[0]][:, np.newaxis]
        mj = m[triu[1]][:, np.newaxis]
        A[triu] = np.sum({3}, axis=1)
        #A is symmetric
        A[tril] = A[triu]

    return A"""


#    note the the i=j part in the fortran loop  below is because
#      I changed the loop order from layer, i,j to layer, j,i which is
#      i think faster as first index of a fortran array loops faster
#      my sympy code is mased on m[i], hence the need for i=j.
    text_fortran = """      SUBROUTINE dim1sin_af_linear(m, at, ab, zt, zb, a, neig, nlayers)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nlayers
        REAL(DP), intent(in), dimension(0:neig-1) ::m
        REAL(DP), intent(in), dimension(0:nlayers-1) :: at
        REAL(DP), intent(in), dimension(0:nlayers-1) :: ab
        REAL(DP), intent(in), dimension(0:nlayers-1) :: zt
        REAL(DP), intent(in), dimension(0:nlayers-1) :: zb
        REAL(DP), intent(out), dimension(0:neig-1, 0:neig-1) :: a
        INTEGER :: i , j, layer
        a=0.0D0
        DO layer = 0, nlayers-1
          DO j = 0, neig-1
              i=j
{0}
            DO i = j+1, neig-1
{1}
            END DO
          END DO
        END DO

        DO j = 0, neig -2
          DO i = j + 1, neig-1
            a(j,i) = a(i, j)
          END DO
        END DO

      END SUBROUTINE"""





    fn = text_python.format(tw(fdiag_loops,5), tw(foff_loops,6), tw(fdiag_vector,3), tw(foff_vector,3))
    fn2 = text_fortran.format(fcode_one_large_expr(fdiag_loops, prepend='a(i, i) = a(i, i) + '),
                 fcode_one_large_expr(foff_loops, prepend='a(i, j) = a(i, j) + '))

    return fn, fn2

def dim1sin_abf_linear_implementations_old():
    """Generate code to calculate spectral method integrations

    Performs integrations of `sin(mi * z) * a(z) *b(z) * sin(mj * z)`
    between [0, 1] where a(z) and b(z) are piecewise linear functions of z.
    Code is generated that will produce a square array with the appropriate
    integrals at each location

    Paste the resulting code (at least the loops) into `dim1sin_abf_linear`.

    Creates 3 implementations:
     - 'scalar', python loops (slowest)
     - 'vectorized', numpy (much faster than scalar)
     - 'fortran', fortran loops (fastest).  Needs to be compiled and interfaced
       with f2py.

    Returns
    -------
    fn: string
        Python code. with scalar (loops) and vectorized (numpy) implementations
        also calls the fortran version.
    fn2: string
        Fortran code.  needs to be compiled with f2py

    Notes
    -----
    The `dim1sin_abf_linear` matrix, :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{{a\\left(z\\right)}{b\\left(z\\right)}\\phi_i\\phi_j\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    ..math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` and :math:`b\\left(z\\right)` are piecewise
    linear functions w.r.t. :math:`z`, that within a layer are defined by:

    ..math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """

    mpv, p = create_layer_sympy_var_and_maps_vectorized(layer_prop=['z','a', 'b'])
    mp, p = create_layer_sympy_var_and_maps(layer_prop=['z','a','b'])

    phi_i = sympy.sin(mi * z)
    phi_j = sympy.sin(mj * z)

    fdiag = sympy.integrate(p['a'] * p['b'] * phi_i * phi_i, z)
    fdiag_loops = fdiag.subs(z, mp['zbot']) - fdiag.subs(z, mp['ztop'])
    fdiag_loops = fdiag_loops.subs(mp)
    fdiag_vector = fdiag.subs(z, mpv['zbot']) - fdiag.subs(z, mpv['ztop'])
    fdiag_vector = fdiag_vector.subs(mpv)

    foff = sympy.integrate(p['a'] * p['b'] * phi_j * phi_i, z)
    foff_loops = foff.subs(z, mp['zbot']) - foff.subs(z, mp['ztop'])
    foff_loops = foff_loops.subs(mp)
    foff_vector = foff.subs(z, mpv['zbot']) - foff.subs(z, mpv['ztop'])
    foff_vector = foff_vector.subs(mpv)

    text_python = """def dim1sin_abf_linear(m, at, ab, bt, bb,  zt, zb, implementation='vectorized'):

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
                A[i, i] += ({0})
            for i in range(neig-1):
                for j in range(i + 1, neig):
                    A[i, j] += ({1})

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

        mi = m[:, np.newaxis]
        A[diag] = np.sum({2}, axis=1)

        mi = m[triu[0]][:, np.newaxis]
        mj = m[triu[1]][:, np.newaxis]
        A[triu] = np.sum({3}, axis=1)
        #A is symmetric
        A[tril] = A[triu]

    return A"""


#    note the the i=j part in the fortran loop  below is because
#      I changed the loop order from layer, i,j to layer, j,i which is
#      i think faster as first index of a fortran array loops faster
#      my sympy code is mased on m[i], hence the need for i=j.
    text_fortran = """      SUBROUTINE dim1sin_abf_linear(m, at, ab, bt, bb, zt, zb, a, neig, nlayers)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nlayers
        REAL(DP), intent(in), dimension(0:neig-1) ::m
        REAL(DP), intent(in), dimension(0:nlayers-1) :: at,ab,bt,bb,zt,zb
!        REAL(DP), intent(in), dimension(0:nlayers-1) :: at
!        REAL(DP), intent(in), dimension(0:nlayers-1) :: ab
!        REAL(DP), intent(in), dimension(0:nlayers-1) :: bt
!        REAL(DP), intent(in), dimension(0:nlayers-1) :: bb
!        REAL(DP), intent(in), dimension(0:nlayers-1) :: zt
!        REAL(DP), intent(in), dimension(0:nlayers-1) :: zb
        REAL(DP), intent(out), dimension(0:neig-1, 0:neig-1) :: a
        INTEGER :: i , j, layer
        a=0.0D0
        DO layer = 0, nlayers-1
          DO j = 0, neig-1
              i=j
{0}
            DO i = j+1, neig-1
{1}
            END DO
          END DO
        END DO

        DO j = 0, neig -2
          DO i = j + 1, neig-1
            a(j,i) = a(i, j)
          END DO
        END DO

      END SUBROUTINE"""





    fn = text_python.format(tw(fdiag_loops,5), tw(foff_loops,6), tw(fdiag_vector,3), tw(foff_vector,3))
    fn2 = text_fortran.format(fcode_one_large_expr(fdiag_loops, prepend='a(i, i) = a(i, i) + '),
                 fcode_one_large_expr(foff_loops, prepend='a(i, j) = a(i, j) + '))

    return fn, fn2

def dim1sin_D_aDf_linear_implementations_old():
    """Generate code to calculate spectral method integrations

    Performs integrations of `sin(mi * z) * D[a(z) * D[sin(mj * z),z],z]`
    between [0, 1] where a(z) i piecewise linear functions of z.
    Code is generated that will produce a square array with the appropriate
    integrals at each location

    Paste the resulting code (at least the loops) into `dim1sin_D_aDf_linear`.

    Creates 3 implementations:
     - 'scalar', python loops (slowest)
     - 'vectorized', numpy (much faster than scalar)
     - 'fortran', fortran loops (fastest).  Needs to be compiled and interfaced
       with f2py.

    Returns
    -------
    fn : string
        Python code. with scalar (loops) and vectorized (numpy) implementations
        also calls the fortran version.
    fn2 : string
        Fortran code.  needs to be compiled with f2py

    Notes
    -----
    The `dim1sin_D_aDf_linear` matrix, :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{\\frac{d}{dz}\\left({a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\right)\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    .. math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` and :math:`b\\left(z\\right)` are piecewise
    linear functions w.r.t. :math:`z`, that within a layer are defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    The integration requires some explanation.  The difficultly arises because
    piecewise linear :math:`a\\left(z\\right)` is defined using step functions at
    the top and bottom of each layer; those step functions when differentiated
    yield dirac-delta or impulse functions which must be handled specially
    when integrating against the spectral basis functions.  The graph below
    shows what happens when the :math:`a\\left(z\\right) distribution is
    differentiated for one layer.

    ::

        y(x)
        ^
        |
        |                            yb
        |                          /|
        |                         / |
        |                        /  |
        |                       /   |
        |                      /    |
        |                   yt|     |
        |                     |     |
        0--------------------xt----xb------------------------1------>x

    Differentiate:

    ::

        y'(x)
        ^
        |
        |                     y(xt) * Dirac(x - xt)
        |                     ^
        |                     |
        |                     |     |
        |                     |     |
        |               y'(xt)|-----|
        |                     |     |
        |                     |     v - y(xb) * Dirac(x - xb)
        0--------------------xt----xb------------------------1------>x

    With sympy/sage I've found it easier to perform the indefinite integrals
    first  and then sub in the bounds of integration. That is, the integral
    of f between a and b is F(b)-F(a) where F is the anti-derivative of f.


    The general expression for :math:`{A}` is:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{\\frac{d}{dz}\\left({a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\right)\\phi_i\\,dz}
        :label: full

    Expanding out the integrals in :eq:`full` yields:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{{a\\left(z\\right)}\\frac{d^2\\phi_j}{dZ^2}\\phi_i\\,dZ}+\\int_{0}^1{\\frac{d{a\\left(z\\right)}}{dZ}\\frac{d\\phi_j}{dZ}\\phi_i\\,dZ}

    Considering a single layer and separating the layer boundaries from the behaviour within a layer gives:

    .. math:: \\mathbf{A}_{i,j,\\text{layer}}=\\int_{z_t}^{z_b}{{a\\left(z\\right)}\\frac{d^2\\phi_j}{dz^2}\\phi_i\\,dZ}+\\int_{z_t}^{z_b}{\\frac{d{a\\left(z\\right)}}{dz}\\frac{d\\phi_j}{dz}\\phi_i\\,dz}+\\int_{0}^{1}{{a\\left(z\\right)}\\delta\\left(z-z_t\\right)\\frac{d\\phi_j}{dz}\\phi_i\\,dz}-\\int_{0}^{1}{{a\\left(z\\right)}\\delta\\left(z-z_b\\right)\\frac{d\\phi_j}{dz}\\phi_i\\,dz}

    Performing the dirac delta integrations:

    .. math:: \\mathbf{A}_{i,j,\\text{layer}}=\\int_{z_t}^{z_b}{{a\\left(z\\right)}\\frac{d^2\\phi_j}{dz^2}\\phi_i\\,dZ}+\\int_{z_t}^{z_b}{\\frac{d{a\\left(z\\right)}}{dz}\\frac{d\\phi_j}{dz}\\phi_i\\,dz}+\\left.{a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\phi_i\\right|_{z=z_t}-\\left.{a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\phi_i\\right|_{z=z_b}

    Now, to get it in the form of F(zb)-F(zt) we only take the zb part of the dirac integration:

    .. math:: F\\left(z\\right)=\\int{{a\\left(z\\right)}\\frac{d^2\\phi_j}{dz^2}\\phi_i\\,dZ}+\\int{\\frac{d{a\\left(z\\right)}}{dz}\\frac{d\\phi_j}{dz}\\phi_i\\,dz}-\\left.{a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\phi_i\\right|_{z=z}

    Finally we get:

    .. math:: \\mathbf{A}_{i,j,\\text{layer}}=F\\left(z_b\\right)-F\\left(z_t\\right)

    TODO: explain why dirac integrations disapear at end points because in this case it will always be sin(mx)*cos(mx) and so always be zero.

    """

#    NOTE: remember that fortran does not distinguish between upper and lower
#        case.  When f2py wraps a fortran function with upper case letters then
#        upper case letters will be converted to lower case. e.g. Therefore when
#        calling a fortran function called if fortran fn is
#        'dim1sin_D_aDf_linear' f2py will wrap it as 'dim1sin_d_adf_linear'

    mpv, p = create_layer_sympy_var_and_maps_vectorized(layer_prop=['z','a'])
    mp, p = create_layer_sympy_var_and_maps(layer_prop=['z','a'])

    phi_i = sympy.sin(mi * z)
    phi_j = sympy.sin(mj * z)

    fdiag = (sympy.integrate(p['a'] * sympy.diff(phi_i, z, 2) * phi_i, z))
    fdiag += (sympy.integrate(sympy.diff(p['a'], z) * sympy.diff(phi_i, z) * phi_i,z))
    fdiag -= (p['a'] * sympy.diff(phi_i, z) * phi_i)
    fdiag_loops = fdiag.subs(z, mp['zbot']) - fdiag.subs(z, mp['ztop'])
    fdiag_loops = fdiag_loops.subs(mp)
    fdiag_vector = fdiag.subs(z, mpv['zbot']) - fdiag.subs(z, mpv['ztop'])
    fdiag_vector = fdiag_vector.subs(mpv)

    foff = (sympy.integrate(p['a'] * sympy.diff(phi_j, z, 2) * phi_i, z))
    foff += (sympy.integrate(sympy.diff(p['a'], z) * sympy.diff(phi_j, z) * phi_i,z))
    foff -= (p['a'] * sympy.diff(phi_j, z) * phi_i)
    foff_loops = foff.subs(z, mp['zbot']) - foff.subs(z, mp['ztop'])
    foff_loops = foff_loops.subs(mp)
    foff_vector = foff.subs(z, mpv['zbot']) - foff.subs(z, mpv['ztop'])
    foff_vector = foff_vector.subs(mpv)

    text_python = """def dim1sin_D_aDf_linear(m, at, ab, zt, zb, implementation='vectorized'):

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
                A[i, i] += ({0})
            for i in range(neig-1):
                for j in range(i + 1, neig):
                    A[i, j] += ({1})

        #A is symmetric
        for i in range(neig - 1):
            for j in range(i + 1, neig):
                A[j, i] = A[i, j]

    elif implementation == 'fortran':
        try:
            import geotecha.speccon.ext_integrals as ext_integ
            A = ext_integ.dim1sin_D_aDf_linear(m, at, ab, zt, zb)
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
        A[diag] = np.sum({2}, axis=1)

        mi = m[triu[0]][:, np.newaxis]
        mj = m[triu[1]][:, np.newaxis]
        A[triu] = np.sum({3}, axis=1)
        #A is symmetric
        A[tril] = A[triu]

    return A"""


#    note the the i=j part in the fortran loop  below is because
#      I changed the loop order from layer, i,j to layer, j,i which is
#      i think faster as first index of a fortran array loops faster
#      my sympy code is mased on m[i], hence the need for i=j.
    text_fortran = """      SUBROUTINE dim1sin_D_aDf_linear(m, at, ab, zt, zb, a, neig, nlayers)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nlayers
        REAL(DP), intent(in), dimension(0:neig-1) ::m
        REAL(DP), intent(in), dimension(0:nlayers-1) :: at
        REAL(DP), intent(in), dimension(0:nlayers-1) :: ab
        REAL(DP), intent(in), dimension(0:nlayers-1) :: zt
        REAL(DP), intent(in), dimension(0:nlayers-1) :: zb
        REAL(DP), intent(out), dimension(0:neig-1, 0:neig-1) :: a
        INTEGER :: i , j, layer
        a=0.0D0
        DO layer = 0, nlayers-1
          DO j = 0, neig-1
              i=j
{0}
            DO i = j+1, neig-1
{1}
            END DO
          END DO
        END DO

        DO j = 0, neig -2
          DO i = j + 1, neig-1
            a(j,i) = a(i, j)
          END DO
        END DO

      END SUBROUTINE"""





    fn = text_python.format(tw(fdiag_loops,5), tw(foff_loops,6), tw(fdiag_vector,3), tw(foff_vector,3))
    fn2 = text_fortran.format(fcode_one_large_expr(fdiag_loops, prepend='a(i, i) = a(i, i) + '),
                 fcode_one_large_expr(foff_loops, prepend='a(i, j) = a(i, j) + '))

    return fn, fn2

def dim1sin_ab_linear_implementations_old():
    """Generate code to calculate spectral method integrations

    Performs integrations of `sin(mi * z) * a(z) * b(z)`
    between [0, 1] where a(z) and b(z) are piecewise linear functions of z.
    Code is generated that will produce a 1d array with the appropriate
    integrals at each location.

    Paste the resulting code (at least the loops) into `dim1sin_ab_linear`.

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


    mpv, p = create_layer_sympy_var_and_maps_vectorized(layer_prop=['z','a', 'b'])
    mp, p = create_layer_sympy_var_and_maps(layer_prop=['z','a','b'])

    phi_i = sympy.sin(mi * z)


    fcol = sympy.integrate(p['a'] * p['b'] * phi_i, z)
    fcol_loops = fcol.subs(z, mp['zbot']) - fcol.subs(z, mp['ztop'])
    fcol_loops = fcol_loops.subs(mp)
    fcol_vector = fcol.subs(z, mpv['zbot']) - fcol.subs(z, mpv['ztop'])
    fcol_vector = fcol_vector.subs(mpv)

    text_python = """def dim1sin_ab_linear(m, at, ab, bt, bb,  zt, zb, implementation='vectorized'):

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
                A[i] += ({0})

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

        mi = m[:, np.newaxis]
        A[:] = np.sum({1}, axis=1)


    return A"""


#    note the the i=j part in the fortran loop  below is because
#      I changed the loop order from layer, i,j to layer, j,i which is
#      i think faster as first index of a fortran array loops faster
#      my sympy code is mased on m[i], hence the need for i=j.
    text_fortran = """      SUBROUTINE dim1sin_ab_linear(m, at, ab, bt, bb, zt, zb, a, neig, nlayers)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nlayers
        REAL(DP), intent(in), dimension(0:neig-1) ::m
        REAL(DP), intent(in), dimension(0:nlayers-1) :: at,ab,bt,bb,zt,zb
        REAL(DP), intent(out), dimension(0:neig-1) :: a
        INTEGER :: i, layer
        a=0.0D0
        DO layer = 0, nlayers-1
          DO i = 0, neig-1
{0}
          END DO
        END DO

      END SUBROUTINE"""





    fn = text_python.format(tw(fcol_loops,5), tw(fcol_vector,3))
    fn2 = text_fortran.format(fcode_one_large_expr(fcol_loops, prepend='a(i) = a(i) + '))

    return fn, fn2

def dim1sin_abc_linear_implementations_old():
    """Generate code to calculate spectral method integrations

    Performs integrations of `sin(mi * z) * a(z) * b(z) * c(z)`
    between [0, 1] where a(z), b(z), c(z) are piecewise linear functions of z.
    Code is generated that will produce a 1d array with the appropriate
    integrals at each location.

    Paste the resulting code (at least the loops) into `dim1sin_abc_linear`.

    Creates 3 implementations:
     - 'scalar', python loops (slowest)
     - 'vectorized', numpy (much faster than scalar)
     - 'fortran', fortran loops (fastest).  Needs to be compiled and interfaced
       with f2py.

    Returns
    -------
    fn: string
        Python code. with scalar (loops) and vectorized (numpy) implementations
        also calls the fortran version.
    fn2: string
        Fortran code.  needs to be compiled with f2py

    Notes
    -----
    The `dim1sin_abc_linear` which should be treated as a column vector,
    :math:`A` is given by:

    .. math:: \\mathbf{A}_{i}=\\int_{0}^1{{a\\left(z\\right)}{b\\left(z\\right)}{c\\left(z\\right)}\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    .. math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)`, :math:`b\\left(z\\right)`, and
    :math:`c\\left(z\\right)` are piecewise linear functions
    w.r.t. :math:`z`, that within a layer are defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """

    mpv, p = create_layer_sympy_var_and_maps_vectorized(layer_prop=['z','a', 'b', 'c'])
    mp, p = create_layer_sympy_var_and_maps(layer_prop=['z','a','b', 'c'])

    phi_i = sympy.sin(mi * z)


    fcol = sympy.integrate(p['a'] * p['b'] * p['c'] * phi_i, z)
    fcol_loops = fcol.subs(z, mp['zbot']) - fcol.subs(z, mp['ztop'])
    fcol_loops = fcol_loops.subs(mp)
    fcol_vector = fcol.subs(z, mpv['zbot']) - fcol.subs(z, mpv['ztop'])
    fcol_vector = fcol_vector.subs(mpv)


    text_python = """def dim1sin_abc_linear(m, at, ab, bt, bb, ct, cb, zt, zb, implementation='vectorized'):

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
                A[i] += ({0})

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

        mi = m[:, np.newaxis]
        A[:] = np.sum({1}, axis=1)


    return A"""


#    note the the i=j part in the fortran loop  below is because
#      I changed the loop order from layer, i,j to layer, j,i which is
#      i think faster as first index of a fortran array loops faster
#      my sympy code is mased on m[i], hence the need for i=j.
    text_fortran = """      SUBROUTINE dim1sin_abc_linear(m, at, ab, bt, bb, ct, cb, zt, zb, a, neig, nlayers)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nlayers
        REAL(DP), intent(in), dimension(0:neig-1) ::m
        REAL(DP), intent(in), dimension(0:nlayers-1) :: at,ab,bt,bb,ct,cb, zt,zb
        REAL(DP), intent(out), dimension(0:neig-1) :: a
        INTEGER :: i, layer
        a=0.0D0
        DO layer = 0, nlayers-1
          DO i = 0, neig-1
{0}
          END DO
        END DO

      END SUBROUTINE"""





    fn = text_python.format(tw(fcol_loops,5), tw(fcol_vector,3))
    fn2 = text_fortran.format(fcode_one_large_expr(fcol_loops, prepend='a(i) = a(i) + '))

    return fn, fn2

def dim1sin_D_aDb_linear_implementations_old():
    """Generate code to calculate spectral method integrations

    Performs integrations of `sin(mi * z) * D[a(z) * D[b(z), z], z]`
    between [0, 1] where a(z) and b(z) are piecewise linear functions of z.
    Code is generated that will produce a 1d array with the appropriate
    integrals at each location.

    Paste the resulting code (at least the loops) into `dim1sin_D_aDb_linear`.

    Creates 3 implementations:
     - 'scalar', python loops (slowest)
     - 'vectorized', numpy (much faster than scalar)
     - 'fortran', fortran loops (fastest).  Needs to be compiled and interfaced
       with f2py.

    Returns
    -------
    fn : string
        Python code. with scalar (loops) and vectorized (numpy) implementations
        also calls the fortran version.
    fn2 : string
        Fortran code.  needs to be compiled with f2py

    Notes
    -----
    The `dim1sin_D_aDb_linear` which should be treated as a column vector,
    :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{\\frac{d}{dz}\\left({a\\left(z\\right)}\\frac{d}{dz}{b\\left(z\\right)}\\right)\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    .. math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` and :math:`b\\left(z\\right)` are piecewise
    linear functions w.r.t. :math:`z`, that within a layer are defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    TODO: explain why the dirac integrations at 0 and 1 must be omitted i.e. not for all cases do they not contribute only.

    """

    mp, p = create_layer_sympy_var_and_maps(layer_prop=['z', 'a', 'b'])
    mpv, p = create_layer_sympy_var_and_maps_vectorized(layer_prop=['z','a', 'b'])

    phi_i = sympy.sin(mi * z)

    fcol = sympy.integrate(sympy.diff(p['a'],z) * sympy.diff(p['b'],z) * phi_i, z)
    fcol -= p['a']*sympy.diff(p['b'], z) * phi_i
    fcol_loops = fcol.subs(z, mp['zbot']) - fcol.subs(z, mp['ztop'])
    fcol_loops = fcol_loops.subs(mp)
    fcol_vector = fcol.subs(z, mpv['zbot']) - fcol.subs(z, mpv['ztop'])
    fcol_vector = fcol_vector.subs(mpv)


    ffirst = -p['a']*sympy.diff(p['b'], z) * phi_i
    ffirst = ffirst.subs(z, mp['ztop'])
    ffirst_loops = ffirst.subs(mp)
    ffirst_loops = ffirst_loops.subs(list(sympy.tensor.get_indices(mp['ztop'])[0])[0], 0)

    flast = p['a']*sympy.diff(p['b'], z) * phi_i
    flast = flast.subs(z, mp['zbot'])
    flast_loops = flast.subs(mp)
    nlayers = sympy.tensor.Idx('nlayers')
    flast_loops=flast_loops.subs(list(sympy.tensor.get_indices(mp['ztop'])[0])[0], nlayers - 1)


    fends_loops = ffirst_loops + flast_loops

    mp.pop('mi')
    mp.pop('mj')


    ffirst_vector = ffirst.subs(mp)
    ffirst_vector = ffirst_vector.subs(list(sympy.tensor.get_indices(mp['ztop'])[0])[0], 0)

    flast_vector= flast.subs(mp)
    flast_vector=flast_vector.subs(list(sympy.tensor.get_indices(mp['ztop'])[0])[0], -1)

    fends_vector = ffirst_vector + flast_vector

#    text = """def dim1sin_D_aDb_linear(m, at, ab, bt, bb, zt, zb):
#    import numpy as np
#    from math import sin, cos
#
#    neig = len(m)
#    nlayers = len(zt)
#
#    A = np.zeros(neig, float)
#    for layer in range(nlayers):
#        for i in range(neig):
#            A[i] += {}
#
#    for i in range(neig):
#        A[i] += {}
#
#    return A"""



    text_python = """def dim1sin_D_aDb_linear(m, at, ab, bt, bb,  zt, zb, implementation='vectorized'):

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
                A[i] += ({0})

        for i in range(neig):
            A[i] += ({1})
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
        A[:] = np.sum({2}, axis=1)
        mi = m
        A[:]+= ({3})
    return A"""


#    note the the i=j part in the fortran loop  below is because
#      I changed the loop order from layer, i,j to layer, j,i which is
#      i think faster as first index of a fortran array loops faster
#      my sympy code is mased on m[i], hence the need for i=j.
    text_fortran = """      SUBROUTINE dim1sin_D_aDb_linear(m, at, ab, bt, bb, zt, zb, a, neig, nlayers)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nlayers
        REAL(DP), intent(in), dimension(0:neig-1) ::m
        REAL(DP), intent(in), dimension(0:nlayers-1) :: at,ab,bt,bb,zt,zb
        REAL(DP), intent(out), dimension(0:neig-1) :: a
        INTEGER :: i, layer
        a=0.0D0
        DO layer = 0, nlayers-1
          DO i = 0, neig-1
{0}
          END DO
        END DO

        DO i = 0, neig-1
{1}
        END DO

      END SUBROUTINE"""





    fn = text_python.format(tw(fcol_loops,5), tw(fends_loops,4), tw(fcol_vector,3), tw(fends_vector,3))
    fn2 = text_fortran.format(fcode_one_large_expr(fcol_loops, prepend='a(i) = a(i) + '),
                              fcode_one_large_expr(fends_loops, prepend='a(i) = a(i) + '))

    return fn, fn2



def tw(text, indents=3, width=100, break_long_words=False):
    """Rough text wrapper for long sympy expressions

    1st line will not be indented.


    Parameters
    ----------
    text : str
        Text to wrap
    width : int optional
        Rough width of warpping. Default width=100.
    indents : int, optional
        Multiple of 4 spaces that will be used to indent each line.
        Default indents=3.
    break_long_words : True/False, optional
        Default break_long_words=False.

    """
    subsequent_indent = " "*4*indents

    wrapper = textwrap.TextWrapper(width=width,
                                   subsequent_indent = subsequent_indent,
                                   break_long_words=break_long_words)

    lines = wrapper.wrap(str(text))
    lines[0] = lines[0].strip()
    return os.linesep.join(lines)




def Eload_linear_implementations():
    """Generate code to perform time integration for spectral methods

    Perform time integrations of a piecewise linear load vs time

    Paste the resulting code (at least the loops) into `Eload_linear`.

    Creates 3 implementations:
     - 'scalar', python loops (slowest)
     - 'vectorized', numpy (much faster than scalar)
     - 'fortran', fortran loops (fastest).  Needs to be compiled and interfaced
       with f2py.

    Returns
    -------
    fn: string
        Python code. with scalar (loops) and vectorized (numpy) implementations
        also calls the fortran version.
    fn2: string
        Fortran code.  needs to be compiled with f2py


    """

    from sympy import exp

    sympy.var('t, tau, dT, eig')
    loadmag = sympy.tensor.IndexedBase('loadmag')
    loadtim = sympy.tensor.IndexedBase('loadtim')
    tvals = sympy.tensor.IndexedBase('tvals')
    eigs = sympy.tensor.IndexedBase('eigs')
    i = sympy.tensor.Idx('i')
    j = sympy.tensor.Idx('j')
    k = sympy.tensor.Idx('k')


    t0, t1, sig0, sig1 = sympy.symbols('t0, t1, sig0, sig1')
    load = sig0 + (sig1 - sig0)/(t1 - t0) * (tau - t0)
    x, x0, x1 = sympy.symbols('x, x0, x1')
    load = load.subs(tau, x/(dT*eig)+t)
    dx_dtau = dT * eig
    mp = [(x0, -dT * eig*(t-t0)),
          (x1, -dT * eig*(t-t1))]

    mp2 = [(sig0, loadmag[k]),
           (sig1, loadmag[k+1]),
            (t0, loadtim[k]),
            (t1, loadtim[k+1])]

#    mp = [(exp(-dT*eig*t)*exp(dT*eig*loadtim[k]),
#           exp(-dT*eig*(t-loadtim[k]))),
#          (exp(-dT*eig*t)*exp(dT*eig*loadtim[k+1]),
#           exp(-dT*eig*(t-loadtim[k+1])))]
#    the default output for the integrals will have expression s like
#    exp(-dT*eig*t)*exp(dT*eig*loadtim[k]).  when dT*eig*loadtim[k] is large
#    the exponential may be so large as to cause an error.  YOu may need to
#    manually alter the expression to exp(is large exp(-dT*eig*(t-loadtim[k]))
#    in which case the term in the exponential will always be negative and not
#    lead to any numerical blow up.
#    load = linear(tau, loadtim[k], loadmag[k], loadtim[k+1], loadmag[k+1])
#    after_instant = (loadmag[k+1] - loadmag[k]) * exp(-dT * eig * (t - loadtim[k]))
#    mp does this automatically with subs
#    load = linear(tau, loadtim[k], loadmag[k], loadtim[k+1], loadmag[k+1])
    within_constant = sig0 * exp(x) / dx_dtau
    within_constant = sympy.integrate(within_constant, (x, x0, 0),risch=False, conds='none')
    within_constant = within_constant.subs(mp)
    within_constant = within_constant.subs(mp2)

    after_constant = sig0 * exp(x) / dx_dtau
    after_constant = sympy.integrate(after_constant, (x, x0, x1), risch=False, conds='none')
    after_constant = after_constant.subs(mp)
    after_constant = after_constant.subs(mp2)

    within_ramp = load * exp(x) / dx_dtau
    within_ramp = sympy.integrate(within_ramp, (x, x0, 0), risch=False, conds='none')
    within_ramp = within_ramp.subs(mp)
    within_ramp = within_ramp.subs(mp2)

    after_ramp = load * exp(x) / dx_dtau
    after_ramp = sympy.integrate(after_ramp, (x, x0, x1), risch=False, conds='none')
    after_ramp = after_ramp.subs(mp)
    after_ramp = after_ramp.subs(mp2)


    text_python = """\
def Eload_linear(loadtim, loadmag, eigs, tvals, dT=1.0, implementation='vectorized'):

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
                    A[i,j] += ({0})
            for k in constants_less_than_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += ({1})
            for k in ramps_containing_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += ({2})
            for k in ramps_less_than_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += ({3})
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
                A[i, :] += np.sum({0}, axis=1)

            k = constants_less_than_t[i]
            if len(k):
                A[i, :] += np.sum({1}, axis=1)

            k = ramps_containing_t[i]
            if len(k):
                A[i, :] += np.sum({2}, axis=1)

            k = ramps_less_than_t[i]
            if len(k):
                A[i, :] += np.sum({3}, axis=1)
    return A"""



    text_fortran = """\
      SUBROUTINE eload_linear(loadtim, loadmag, eigs, tvals,&
                              dT, a, neig, nload, nt)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nload
        INTEGER, intent(in) :: nt
        REAL(DP), intent(in), dimension(0:nload-1) :: loadtim
        REAL(DP), intent(in), dimension(0:nload-1) :: loadmag
        REAL(DP), intent(in), dimension(0:neig-1) :: eigs
        REAL(DP), intent(in), dimension(0:nt-1) :: tvals
        REAL(DP), intent(in) :: dT
        REAL(DP), intent(out), dimension(0:nt-1, 0:neig-1) :: a
        INTEGER :: i , j, k
        REAL(DP):: EPSILON
        a=0.0D0
        EPSILON = 0.0000005D0
        DO i = 0, nt-1
          DO k = 0, nload-2

            IF (tvals(i) < loadtim(k)) EXIT !t is before load step

            IF (tvals(i) >= loadtim(k + 1)) THEN
              !t is after the load step
              IF(ABS(loadtim(k) - loadtim(k + 1)) <= &
                (ABS(loadtim(k) + loadtim(k + 1))*EPSILON)) THEN
                !step load
                CONTINUE
              ELSEIF(ABS(loadmag(k) - loadmag(k + 1)) <= &
                    (ABS(loadmag(k) + loadmag(k + 1))*EPSILON)) THEN
                !constant load
                DO j=0, neig-1
{0}
                END DO
              ELSE
                !ramp load
                DO j=0, neig-1
{1}
                END DO
              END IF
            ELSE
              !t is in the load step
              IF(ABS(loadmag(k) - loadmag(k + 1)) <= &
                    (ABS(loadmag(k) + loadmag(k + 1))*EPSILON)) THEN
                !constant load
                DO j=0, neig-1
{2}
                END DO
              ELSE
                !ramp load
                DO j=0, neig-1
{3}
                END DO
              END IF
            END IF
          END DO
        END DO

      END SUBROUTINE



    """



    fn = text_python.format(
            tw(within_constant, 6),
            tw(after_constant, 6),
            tw(within_ramp, 6),
            tw(after_ramp, 6))


    mp3 = [(eig, eigs[j]),
           (t,tvals[i])]
    after_constant = after_constant.subs(mp3)
    after_ramp = after_ramp.subs(mp3)
    within_constant = within_constant.subs(mp3)
    within_ramp = within_ramp.subs(mp3)
    fn2 = text_fortran.format(
        fcode_one_large_expr(after_constant, prepend='a(i, j) = a(i, j) + '),
        fcode_one_large_expr(after_ramp, prepend='a(i, j) = a(i, j) + '),
        fcode_one_large_expr(within_constant, prepend='a(i, j) = a(i, j) + '),
        fcode_one_large_expr(within_ramp, prepend='a(i, j) = a(i, j) + '))

    return fn, fn2


def EDload_linear_implementations():
    """Generate code to perform time integration for spectral methods

    Perform time integrations of a piecewise linear load vs time

    Paste the resulting code (at least the loops) into `EDload_linear`.

    Creates 3 implementations:
     - 'scalar', python loops (slowest)
     - 'vectorized', numpy (much faster than scalar)
     - 'fortran', fortran loops (fastest).  Needs to be compiled and interfaced
       with f2py.

    Returns
    -------
    fn: string
        Python code. with scalar (loops) and vectorized (numpy) implementations
        also calls the fortran version.
    fn2: string
        Fortran code.  needs to be compiled with f2py


    """

    from sympy import exp

    sympy.var('t, tau, dT, eig')
    loadmag = sympy.tensor.IndexedBase('loadmag')
    loadtim = sympy.tensor.IndexedBase('loadtim')
    tvals = sympy.tensor.IndexedBase('tvals')
    eigs = sympy.tensor.IndexedBase('eigs')
    i = sympy.tensor.Idx('i')
    j = sympy.tensor.Idx('j')
    k = sympy.tensor.Idx('k')


    t0, t1, sig0, sig1 = sympy.symbols('t0, t1, sig0, sig1')
    load = sig0 + (sig1 - sig0)/(t1 - t0) * (tau - t0)

    x, x0, x1 = sympy.symbols('x, x0, x1')
    load = load.subs(tau, x/(dT*eig)+t)
    dx_dtau = dT * eig
    mp = [(x0, -dT * eig*(t-t0)),
          (x1, -dT * eig*(t-t1))]
#    the default output for the integrals will have expression s like
#    exp(-dT*eig*t)*exp(dT*eig*loadtim[k]).  when dT*eig*loadtim[k] is large
#    the exponential may be so large as to cause an error.  YOu may need to
#    manually alter the expression to exp(is large exp(-dT*eig*(t-loadtim[k]))
#    in which case the term in the exponential will always be negative and not
#    lead to any numerical blow up.

    mp2 = [(sig0, loadmag[k]),
           (sig1, loadmag[k + 1]),
           (t0, loadtim[k]),
           (t1, loadtim[k + 1])]


    Dload = sympy.diff(load, x) * dx_dtau
    after_instant = (sig1 - sig0) * exp(x0)
    after_instant = after_instant.subs(mp)
    after_instant = after_instant.subs(mp2)

    within_ramp = Dload * exp(x) / dx_dtau
    within_ramp = sympy.integrate(within_ramp, (x, x0, 0), risch=False, conds='none')
    within_ramp = within_ramp.subs(mp)
    within_ramp = within_ramp.subs(mp2)

    after_ramp = Dload * exp(x) / dx_dtau
    after_ramp = sympy.integrate(after_ramp, (x, x0, x1), risch=False, conds='none')
    after_ramp = after_ramp.subs(mp)
    after_ramp = after_ramp.subs(mp2)



    text_python = """\
def EDload_linear(loadtim, loadmag, eigs, tvals, dT=1.0, implementation='vectorized'):

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
                    A[i,j] += ({0})
            for k in ramps_containing_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += ({1})
            for k in ramps_less_than_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += ({2})

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
                A[i, :] += np.sum({0}, axis=1)
            k = ramps_containing_t[i]
            if len(k):
                A[i, :] += np.sum({1}, axis=1)
            k = ramps_less_than_t[i]
            if len(k):
                A[i, :] += np.sum({2}, axis=1)
    return A"""



    text_fortran = """\
      SUBROUTINE edload_linear(loadtim, loadmag, eigs, tvals,&
                               dT, a, neig, nload, nt)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nload
        INTEGER, intent(in) :: nt
        REAL(DP), intent(in), dimension(0:nload-1) :: loadtim
        REAL(DP), intent(in), dimension(0:nload-1) :: loadmag
        REAL(DP), intent(in), dimension(0:neig-1) :: eigs
        REAL(DP), intent(in), dimension(0:nt-1) :: tvals
        REAL(DP), intent(in) :: dT
        REAL(DP), intent(out), dimension(0:nt-1, 0:neig-1) :: a
        INTEGER :: i , j, k
        REAL(DP):: EPSILON
        a=0.0D0
        EPSILON = 0.0000005D0
        DO i = 0, nt-1
          DO k = 0, nload-2

            IF (tvals(i) < loadtim(k)) EXIT !t is before load step

            IF (tvals(i) >= loadtim(k + 1)) THEN
              !t is after the load step
              IF(ABS(loadtim(k) - loadtim(k + 1)) <= &
                (ABS(loadtim(k) + loadtim(k + 1))*EPSILON)) THEN
                !step load
                DO j=0, neig-1
{0}
                END DO
              ELSEIF(ABS(loadmag(k) - loadmag(k + 1)) <= &
                    (ABS(loadmag(k) + loadmag(k + 1))*EPSILON)) THEN
                !constant load
                CONTINUE
              ELSE
                !ramp load
                DO j=0, neig-1
{1}
                END DO
              END IF
            ELSE
              !t is in the load step
              IF(ABS(loadmag(k) - loadmag(k + 1)) <= &
                    (ABS(loadmag(k) + loadmag(k + 1))*EPSILON)) THEN
                !constant load
                CONTINUE
              ELSE
                !ramp load
                DO j=0, neig-1
{2}
                END DO
              END IF
            END IF
          END DO
        END DO

      END SUBROUTINE



    """



    fn = text_python.format(
            tw(after_instant, 6),
            tw(within_ramp, 6),
            tw(after_ramp, 6))


    mp3 = [(eig, eigs[j]),
           (t, tvals[i])]
    after_instant = after_instant.subs(mp3)
    after_ramp = after_ramp.subs(mp3)
    within_ramp = within_ramp.subs(mp3)
    fn2 = text_fortran.format(
        fcode_one_large_expr(after_instant, prepend='a(i, j) = a(i, j) + '),
        fcode_one_large_expr(after_ramp, prepend='a(i, j) = a(i, j) + '),
        fcode_one_large_expr(within_ramp, prepend='a(i, j) = a(i, j) + '))

    return fn, fn2


def Eload_coslinear_implementations():
    """Generate code to perform time integration for spectral methods

    Perform time integrations of a piecewise linear load vs time

    Paste the resulting code (at least the loops) into `Eload_coslinear`.

    Creates 3 implementations:
     - 'scalar', python loops (slowest)
     - 'vectorized', numpy (much faster than scalar)
     - 'fortran', fortran loops (fastest).  Needs to be compiled and interfaced
       with f2py.

    Returns
    -------
    fn: string
        Python code. with scalar (loops) and vectorized (numpy) implementations
        also calls the fortran version.
    fn2: string
        Fortran code.  needs to be compiled with f2py


    """

    from sympy import exp

    sympy.var('t, tau, dT, eig, omega, phase')
    loadmag = sympy.tensor.IndexedBase('loadmag')
    loadtim = sympy.tensor.IndexedBase('loadtim')
    tvals = sympy.tensor.IndexedBase('tvals')
    eigs = sympy.tensor.IndexedBase('eigs')
    i = sympy.tensor.Idx('i')
    j = sympy.tensor.Idx('j')
    k = sympy.tensor.Idx('k')


    t0, t1, sig0, sig1 = sympy.symbols('t0, t1, sig0, sig1')
    x, x0, x1 = sympy.symbols('x, x0, x1')
    A, B = sympy.symbols('A, B')
#    load1 = sympy.cos(omega * tau + phase)
    load1 = sympy.cos(A * x + B)
    load2 = sig0 + (sig1 - sig0)/(t1 - t0) * (tau - t0)

    load1 = load1.subs(tau, x/(dT*eig)+t)
    load2 = load2.subs(tau, x/(dT*eig)+t)

    dx_dtau = dT * eig
    mp = [(x0, -dT * eig*(t-t0)),
          (x1, -dT * eig*(t-t1)),
            (A, omega / (dT*eig)),
            (B, omega * t + phase)]

    mp2 = [(sig0, loadmag[k]),
           (sig1, loadmag[k+1]),
            (t0, loadtim[k]),
            (t1, loadtim[k+1])]
#    mp = [(exp(-dT*eig*t)*exp(dT*eig*loadtim[k]),
#           exp(-dT*eig*(t-loadtim[k]))),
#          (exp(-dT*eig*t)*exp(dT*eig*loadtim[k+1]),
#           exp(-dT*eig*(t-loadtim[k+1])))]
#    the default output for the integrals will have expression s like
#    exp(-dT*eig*t)*exp(dT*eig*loadtim[k]).  when dT*eig*loadtim[k] is large
#    the exponential may be so large as to cause an error.  YOu may need to
#    manually alter the expression to exp(is large exp(-dT*eig*(t-loadtim[k]))
#    in which case the term in the exponential will always be negative and not
#    lead to any numerical blow up.
#    load = linear(tau, loadtim[k], loadmag[k], loadtim[k+1], loadmag[k+1])
#    after_instant = (loadmag[k+1] - loadmag[k]) * exp(-dT * eig * (t - loadtim[k]))
#    mp does this automatically with subs
#    load = linear(tau, loadtim[k], loadmag[k], loadtim[k+1], loadmag[k+1])
    within_constant = sig0 * load1 * exp(x) / dx_dtau
    within_constant = sympy.integrate(within_constant, (x, x0, 0),risch=False, conds='none')
    within_constant = within_constant.subs(mp)
    within_constant = within_constant.subs(mp2)
    after_constant = sig0 * load1 * exp(x) / dx_dtau
    after_constant = sympy.integrate(after_constant, (x, x0, x1), risch=False, conds='none')
    after_constant = after_constant.subs(mp)
    after_constant = after_constant.subs(mp2)

    within_ramp = load1 * load2 * exp(x) / dx_dtau
    within_ramp = sympy.integrate(within_ramp, (x, x0, 0), risch=False, conds='none')
    within_ramp = within_ramp.subs(mp)
    within_ramp = within_ramp.subs(mp2)

    after_ramp = load1 * load2 * exp(x) / dx_dtau
    after_ramp = sympy.integrate(after_ramp, (x, x0, x1), risch=False, conds='none')
    after_ramp = after_ramp.subs(mp)
    after_ramp = after_ramp.subs(mp2)


    text_python = """\
def Eload_coslinear(loadtim, loadmag, omega, phase, eigs, tvals, dT=1.0, implementation='vectorized'):

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
                    A[i,j] += ({0})
            for k in constants_less_than_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += ({1})
            for k in ramps_containing_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += ({2})
            for k in ramps_less_than_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += ({3})
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
                A[i, :] += np.sum({0}, axis=1)

            k = constants_less_than_t[i]
            if len(k):
                A[i, :] += np.sum({1}, axis=1)

            k = ramps_containing_t[i]
            if len(k):
                A[i, :] += np.sum({2}, axis=1)

            k = ramps_less_than_t[i]
            if len(k):
                A[i, :] += np.sum({3}, axis=1)
    return A"""



    text_fortran = """\
      SUBROUTINE eload_coslinear(loadtim, loadmag, omega, phase, &
                                 eigs, tvals, dT, a, neig, nload, nt)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nload
        INTEGER, intent(in) :: nt
        REAL(DP), intent(in), dimension(0:nload-1) :: loadtim
        REAL(DP), intent(in), dimension(0:nload-1) :: loadmag
        REAL(DP), intent(in), dimension(0:neig-1) :: eigs
        REAL(DP), intent(in), dimension(0:nt-1) :: tvals
        REAL(DP), intent(in) :: dT
        REAL(DP), intent(in) :: omega
        REAL(DP), intent(in) :: phase
        REAL(DP), intent(out), dimension(0:nt-1, 0:neig-1) :: a
        INTEGER :: i , j, k
        REAL(DP):: EPSILON
        a=0.0D0
        EPSILON = 0.0000005D0
        DO i = 0, nt-1
          DO k = 0, nload-2

            IF (tvals(i) < loadtim(k)) EXIT !t is before load step

            IF (tvals(i) >= loadtim(k + 1)) THEN
              !t is after the load step
              IF(ABS(loadtim(k) - loadtim(k + 1)) <= &
                (ABS(loadtim(k) + loadtim(k + 1))*EPSILON)) THEN
                !step load
                CONTINUE
              ELSEIF(ABS(loadmag(k) - loadmag(k + 1)) <= &
                    (ABS(loadmag(k) + loadmag(k + 1))*EPSILON)) THEN
                !constant load
                DO j=0, neig-1
{0}
                END DO
              ELSE
                !ramp load
                DO j=0, neig-1
{1}
                END DO
              END IF
            ELSE
              !t is in the load step
              IF(ABS(loadmag(k) - loadmag(k + 1)) <= &
                    (ABS(loadmag(k) + loadmag(k + 1))*EPSILON)) THEN
                !constant load
                DO j=0, neig-1
{2}
                END DO
              ELSE
                !ramp load
                DO j=0, neig-1
{3}
                END DO
              END IF
            END IF
          END DO
        END DO

      END SUBROUTINE



    """



    fn = text_python.format(
            tw(within_constant, 6),
            tw(after_constant, 6),
            tw(within_ramp, 6),
            tw(after_ramp, 6))


    mp3 = [(eig, eigs[j]),
           (t,tvals[i])]
    after_constant = after_constant.subs(mp3)
    after_ramp = after_ramp.subs(mp3)
    within_constant = within_constant.subs(mp3)
    within_ramp = within_ramp.subs(mp3)
    fn2 = text_fortran.format(
        fcode_one_large_expr(after_constant, prepend='a(i, j) = a(i, j) + '),
        fcode_one_large_expr(after_ramp, prepend='a(i, j) = a(i, j) + '),
        fcode_one_large_expr(within_constant, prepend='a(i, j) = a(i, j) + '),
        fcode_one_large_expr(within_ramp, prepend='a(i, j) = a(i, j) + '))

    return fn, fn2

def EDload_coslinear_implementations():
    """Generate code to perform time integration for spectral methods

    Perform time integrations of a piecewise linear load vs time

    Paste the resulting code (at least the loops) into `EDload_coslinear`.

    Creates 3 implementations:
     - 'scalar', python loops (slowest)
     - 'vectorized', numpy (much faster than scalar)
     - 'fortran', fortran loops (fastest).  Needs to be compiled and interfaced
       with f2py.

    Returns
    -------
    fn: string
        Python code. with scalar (loops) and vectorized (numpy) implementations
        also calls the fortran version.
    fn2: string
        Fortran code.  needs to be compiled with f2py


    """

    from sympy import exp

    sympy.var('t, tau, dT, eig, omega, phase')
    loadmag = sympy.tensor.IndexedBase('loadmag')
    loadtim = sympy.tensor.IndexedBase('loadtim')
    tvals = sympy.tensor.IndexedBase('tvals')
    eigs = sympy.tensor.IndexedBase('eigs')
    i = sympy.tensor.Idx('i')
    j = sympy.tensor.Idx('j')
    k = sympy.tensor.Idx('k')


    t0, t1, sig0, sig1 = sympy.symbols('t0, t1, sig0, sig1')
    x, x0, x1 = sympy.symbols('x, x0, x1')
    A, B = sympy.symbols('A, B')
#    load1 = sympy.cos(omega * tau + phase)
    load1 = sympy.cos(A * x + B)
    load2 = sig0 + (sig1 - sig0)/(t1 - t0) * (tau - t0)
    load2 = load2.subs(tau, x/(dT*eig)+t)

    dx_dtau = dT * eig
    mp = [(x0, -dT * eig*(t-t0)),
          (x1, -dT * eig*(t-t1)),
            (A, omega / (dT*eig)),
            (B, omega * t + phase)]

    mp2 = [(sig0, loadmag[k]),
           (sig1, loadmag[k+1]),
            (t0, loadtim[k]),
            (t1, loadtim[k+1])]
#    the default output for the integrals will have expression s like
#    exp(-dT*eig*t)*exp(dT*eig*loadtim[k]).  when dT*eig*loadtim[k] is large
#    the exponential may be so large as to cause an error.  YOu may need to
#    manually alter the expression to exp(is large exp(-dT*eig*(t-loadtim[k]))
#    in which case the term in the exponential will always be negative and not
#    lead to any numerical blow up.

    mp2 = [(sig0, loadmag[k]),
           (sig1, loadmag[k + 1]),
           (t0, loadtim[k]),
           (t1, loadtim[k + 1])]


    Dload = sympy.diff(load1 * load2, x) * dx_dtau
    after_instant = (sig1 - sig0) * sympy.cos(omega * t0 + phase)*exp(-dT * eig * (t - t0))
    after_instant = after_instant.subs(mp)
    after_instant = after_instant.subs(mp2)

    within_constant = sig0 * sympy.diff(load1, x) * exp(x) #dx_dtau cancels from diff and substitution
    within_constant = sympy.integrate(within_constant, (x, x0, 0), risch=False, conds='none')
    within_constant = within_constant.subs(mp)
    within_constant = within_constant.subs(mp2)

    after_constant = sig0 * sympy.diff(load1, x) * exp(x) #dx_dtau cancels from diff and substitution
    after_constant = sympy.integrate(after_constant, (x, x0, x1), risch=False, conds='none')
    after_constant = after_constant.subs(mp)
    after_constant = after_constant.subs(mp2)


    within_ramp = Dload * exp(x) / dx_dtau
    within_ramp = sympy.integrate(within_ramp, (x, x0, 0), risch=False, conds='none')
    within_ramp = within_ramp.subs(mp)
    within_ramp = within_ramp.subs(mp2)

    after_ramp = Dload * exp(x) / dx_dtau
    after_ramp = sympy.integrate(after_ramp, (x, x0, x1), risch=False, conds='none')
    after_ramp = after_ramp.subs(mp)
    after_ramp = after_ramp.subs(mp2)



    text_python = """\
def EDload_coslinear(loadtim, loadmag, omega, phase,eigs, tvals, dT=1.0, implementation='vectorized'):

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
                    A[i,j] += ({0})
            for k in ramps_containing_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += ({1})
            for k in ramps_less_than_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += ({2})
            for k in constants_containing_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += ({3})
            for k in constants_less_than_t[i]:
                for j, eig in enumerate(eigs):
                    A[i,j] += ({4})

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
                A[i, :] += np.sum({0}, axis=1)
            k = ramps_containing_t[i]
            if len(k):
                A[i, :] += np.sum({1}, axis=1)
            k = ramps_less_than_t[i]
            if len(k):
                A[i, :] += np.sum({2}, axis=1)
            k = constants_containing_t[i]
            if len(k):
                A[i,:] += np.sum({3}, axis=1)
            k = constants_less_than_t[i]
            if len(k):
                A[i,:] += np.sum({4}, axis=1)
    return A"""



    text_fortran = """\
      SUBROUTINE edload_coslinear(loadtim, loadmag, omega, phase,&
                                  eigs, tvals, dT, a, neig, nload, nt)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nload
        INTEGER, intent(in) :: nt
        REAL(DP), intent(in), dimension(0:nload-1) :: loadtim
        REAL(DP), intent(in), dimension(0:nload-1) :: loadmag
        REAL(DP), intent(in), dimension(0:neig-1) :: eigs
        REAL(DP), intent(in), dimension(0:nt-1) :: tvals
        REAL(DP), intent(in) :: dT
        REAL(DP), intent(in) :: omega
        REAL(DP), intent(in) :: phase
        REAL(DP), intent(out), dimension(0:nt-1, 0:neig-1) :: a
        INTEGER :: i , j, k
        REAL(DP):: EPSILON
        a=0.0D0
        EPSILON = 0.0000005D0
        DO i = 0, nt-1
          DO k = 0, nload-2

            IF (tvals(i) < loadtim(k)) EXIT !t is before load step

            IF (tvals(i) >= loadtim(k + 1)) THEN
              !t is after the load step
              IF(ABS(loadtim(k) - loadtim(k + 1)) <= &
                (ABS(loadtim(k) + loadtim(k + 1))*EPSILON)) THEN
                !step load
                DO j=0, neig-1
{0}
                END DO
              ELSEIF(ABS(loadmag(k) - loadmag(k + 1)) <= &
                    (ABS(loadmag(k) + loadmag(k + 1))*EPSILON)) THEN
                !constant load
                DO j=0, neig-1
{1}
                END DO
              ELSE
                !ramp load
                DO j=0, neig-1
{2}
                END DO
              END IF
            ELSE
              !t is in the load step
              IF(ABS(loadmag(k) - loadmag(k + 1)) <= &
                    (ABS(loadmag(k) + loadmag(k + 1))*EPSILON)) THEN
                !constant load
                DO j=0, neig-1
{3}
                END DO
              ELSE
                !ramp load
                DO j=0, neig-1
{4}
                END DO
              END IF
            END IF
          END DO
        END DO

      END SUBROUTINE



    """



    fn = text_python.format(
            tw(after_instant, 6),
            tw(within_ramp, 6),
            tw(after_ramp, 6),
            tw(within_constant, 6),
            tw(after_constant, 6))


    mp3 = [(eig, eigs[j]),
           (t, tvals[i])]

    after_instant = after_instant.subs(mp3)
    after_ramp = after_ramp.subs(mp3)
    within_ramp = within_ramp.subs(mp3)
    after_constant = after_constant.subs(mp3)
    within_constant = within_constant.subs(mp3)

    fn2 = text_fortran.format(
        fcode_one_large_expr(after_instant, prepend='a(i, j) = a(i, j) + '),
        fcode_one_large_expr(after_constant, prepend='a(i, j) = a(i, j) + '),
        fcode_one_large_expr(after_ramp, prepend='a(i, j) = a(i, j) + '),
        fcode_one_large_expr(within_constant, prepend='a(i, j) = a(i, j) + '),
        fcode_one_large_expr(within_ramp, prepend='a(i, j) = a(i, j) + '))

    return fn, fn2







class SympyVarsFor1DSpectralDerivation(object):
    """Container for sympy vars, z, zt, zb, at, etc piecewise linear
    Spectral Galerkin integrations.


    Parameters
    ----------
    linear_var : ['z', 'x', 'y'], optional
        Independent variable for the linear function specification.
        f(linear_var) = at+a_slope * (linear_var-linear_vart).
        Default linear_var='z'.
    slope : True/False, optional
        If True (default), then the linear functions will be defined with a
        lumped slope term, e.g. a = atop + a_slope * (z - ztop) compared to
        if slope=False, where a = atop + (abot-atop)/(zbot-ztop)*(z-ztop).
        You will have to define a_slope in your code template, e.g.
        Python loops: a_slope = (ab[layer]-at[layer])/(zb[layer]-zt[layer]).
        Fortran loops: a_slope = (ab(layer)-at(layer)/(zb(layer)-zt(layer)).
        Vectorised: a_slope = (ab-at)/(zb-zt).


    Attributes
    ----------
    x, y, z : sympy.Symbol
        Independent variables.
    xtop, xbot, ytop, ybot, ztop, zbot : sympy.Symbol
        Symbols used in expressions to be integrated with sympy.integrate.
        After the integration these variables are usually replaced with
        the relevant xt, xb, yt etc. or xt[layer], yb[layer] etc.  These
        substitutions can be made using `map_to_add_index' or
        `map_top_to_t_bot_to_b`.
    i, j, k , layer : sympy.tensor.Idx
        Index variables.
    xt, xb, yt, yb, zt, zb : sympy.tensor.IndexedBase
        Variables for values at top and bottom of layer.  Used to define
        linear relationships.  These variable usually replace xtop, xbot
        after integrations.  The reason they are not used before integration
        is thatsympy doesn't seem to like the sympy.tensor.IndexedBase for
        integrations. Substitutions can be made using `map_to_add_index' or
        `map_top_to_t_bot_to_b`.
    at, ab, a, a_slope: sympy.Symbol and sympy.Expr
        Variables to define linear relationships.
        If `slope`=True, a = atop + a_slope * (z - ztop)
        If `slope`=False, a = atop + (abot - atop)/(zbot - ztop) * (z - ztop).
        Where z and ztop may change according to `linear_var`.
    bt, bb, b, b_slope: sympy.Symbol and sympy.Expr
        Variables to define linear relationships.
        If `slope`=True, b = atop + b_slope * (z - ztop)
        If `slope`=False, b = btop + (bbot - btop)/(zbot - ztop) * (z - ztop).
        Where z and ztop may change according to `linear_var`.
    ct, cb, c, c_slope: sympy.Symbol and sympy.Expr
        Variables to define linear relationships.
        If `slope`=True, c = ctop + c_slope * (z - ztop)
        If `slope`=False, c = ctop + (cbot - ctop)/(zbot - ztop) * (z - ztop).
        Where z and ztop may change according to `linear_var`.
    map_to_add_index : list of 2 element tuples
        A list to be used with the subs method of sympy expressions to
        add a index variables to the variables. A typical entries in
        `map_to_add_index` would be [(mi, m[i]), (mj, m[j]), (atop, at[layer]),
        (ztop, zt[layer]), ...].  Use this to after an integration invoving
        ztop, atop etc. to format the expression for use in function with
        loops.
    map_top_to_t_bot_to_b : list of 2 element tuples
        A list to be used with the subs method of sympy expressions to
        add change 'ztop' to 'zt', 'abot' to 'at' etc. Typical entries in
        `map_top_to_t_bot_to_b` would be [(atop, at), (ztop, zt), ...].
        Use this to after an integration invoving ztop, atop etc. to format
        the expression for use in a vectorised function.
    mi, mj : sympy.Symbol
        Variable for row and column eigs.
    m : sympy.tensor.IndexedBase
        IndexedBaseVariable of eigs.  used for m[i], and m[j].


    Examples
    --------
    >>> v = SympyVarsFor1DSpectralDerivation()
    >>> v.a
    a_slope*(z - ztop) + atop
    >>> v.b
    b_slope*(z - ztop) + btop
    >>> v.c
    c_slope*(z - ztop) + ctop
    >>> v.map_to_add_index
    [(atop, at[layer]), (abot, ab[layer]), (btop, bt[layer]), (bbot, bb[layer]), (ctop, ct[layer]), (cbot, cb[layer]), (ztop, zt[layer]), (zbot, zb[layer]), (mi, m[i]), (mj, m[j])]
    >>> v.a.subs(v.map_to_add_index)
    a_slope*(z - zt[layer]) + at[layer]
    >>> v.b.subs(v.map_top_to_t_bot_to_b)
    b_slope*(z - zt) + bt
    >>> v.mi
    mi
    >>> v.mi.subs(v.map_to_add_index)
    m[i]
    >>> v.mi.subs(v.map_top_to_t_bot_to_b)
    mi


    >>> v = SympyVarsFor1DSpectralDerivation(linear_var='x', slope=False)
    >>> v.a
    atop + (abot - atop)*(x - xtop)/(xbot - xtop)


    """

    def __init__(self, linear_var='z', slope=True):

        import sympy


        #integration variables
        self.x, self.y, self.z = sympy.symbols('x,y,z')
        self.xtop, self.xbot = sympy.symbols('xtop,xbot', nonzero=True)
        self.ytop, self.ybot = sympy.symbols('ytop,ybot', nonzero=True)
        self.ztop, self.zbot = sympy.symbols('ztop,zbot', nonzero=True)
        self.atop, self.abot = sympy.symbols('atop,abot', nonzero=True)
        self.btop, self.bbot = sympy.symbols('btop,bbot', nonzero=True)
        self.ctop, self.cbot = sympy.symbols('ctop,cbot', nonzero=True)

        # indexes
        self.i = sympy.tensor.Idx('i')
        self.j = sympy.tensor.Idx('j')
        self.k = sympy.tensor.Idx('k')
        self.layer = sympy.tensor.Idx('layer')

        self.xt = sympy.tensor.IndexedBase('xt')
        self.xb = sympy.tensor.IndexedBase('xb')
        self.yt = sympy.tensor.IndexedBase('yt')
        self.yb = sympy.tensor.IndexedBase('yb')
        self.zt = sympy.tensor.IndexedBase('zt')
        self.zb = sympy.tensor.IndexedBase('zb')

        self.m = sympy.tensor.IndexedBase('m')
        self.mi, self.mj = sympy.symbols('mi,mj', nonzero=True)
        mmap_to_add_index = [(self.mi, self.m[self.i]),
                             (self.mj, self.m[self.j])]


        #linear f(linear_var)
        self.linear_var = linear_var
        _x = getattr(self, linear_var)
        _xtop = getattr(self, linear_var + 'top')
        _xbot = getattr(self, linear_var + 'bot')
        _xt = getattr(self, linear_var + 't')
        _xb = getattr(self, linear_var + 'b')
        _xmap_to_add_index = [(_xtop, _xt[self.layer]),
                              (_xbot, _xb[self.layer])]
        _xmap_top_to_t_bot_to_b = [(_xtop, _xt),
                                   (_xbot, _xb)]


        self.at = sympy.tensor.IndexedBase('at')
        self.ab = sympy.tensor.IndexedBase('ab')
        self.a_slope = sympy.symbols('a_slope')
        if slope:
            self.a = (self.atop + self.a_slope * (_x - _xtop))
        else:
            self.a = (self.atop +
                      (self.abot - self.atop) / (_xbot - _xtop) * (_x - _xtop))

        amap_to_add_index = [(self.atop, self.at[self.layer]),
                             (self.abot, self.ab[self.layer])]
        amap_top_to_t_bot_to_b = [(self.atop, self.at),
                                   (self.abot, self.ab)]

        self.bt = sympy.tensor.IndexedBase('bt')
        self.bb = sympy.tensor.IndexedBase('bb')
        self.b_slope = sympy.symbols('b_slope')
        if slope:
            self.b = (self.btop + self.b_slope * (_x - _xtop))
        else:
            self.b = (self.btop +
                      (self.bbot - self.btop) / (_xbot - _xtop) * (_x - _xtop))

        bmap_to_add_index = [(self.btop, self.bt[self.layer]),
                             (self.bbot, self.bb[self.layer])]
        bmap_top_to_t_bot_to_b = [(self.btop, self.bt),
                                   (self.bbot, self.bb)]

        self.ct = sympy.tensor.IndexedBase('ct')
        self.cb = sympy.tensor.IndexedBase('cb')
        self.c_slope = sympy.symbols('c_slope')
        if slope:
            self.c = (self.ctop + self.c_slope * (_x - _xtop))
        else:
            self.c = (self.ctop +
                      (self.cbot - self.ctop) / (_xbot - _xtop) * (_x - _xtop))

        cmap_to_add_index = [(self.ctop, self.ct[self.layer]),
                             (self.cbot, self.cb[self.layer])]
        cmap_top_to_t_bot_to_b = [(self.ctop, self.ct),
                                   (self.cbot, self.cb)]

        # make sure mmap... is the first entry!!!!!!!
        self.map_to_add_index = (mmap_to_add_index +
                                 amap_to_add_index +
                                 bmap_to_add_index +
                                 cmap_to_add_index +
                                 _xmap_to_add_index)

        self.map_top_to_t_bot_to_b = (amap_top_to_t_bot_to_b +
                                      bmap_top_to_t_bot_to_b +
                                      cmap_top_to_t_bot_to_b +
                                     _xmap_top_to_t_bot_to_b)


def dim1sin_af_linear_implementations():
    """Generate code to calculate spectral method integrations

    Generate scalar and vectorized python code and fortran loops

    Performs integrations of `sin(mi * z) * a(z) * sin(mj * z)` between [0, 1]
    where a(z) is a piecewise linear function of z.  Code is generated that
    will produce a square array with the appropriate integrals at each location

    Paste the resulting code (at least the loops) into `dim1sin_af_linear`.

    Creates 3 implementations:
     - 'scalar', python loops (slowest)
     - 'vectorized', numpy (much faster than scalar)
     - 'fortran', fortran loops (fastest).  Needs to be compiled and interfaced
       with f2py.


    Returns
    -------
    fn : string
        Python code. with scalar (loops) and vectorized (numpy) implementations
        also calls the fortran version.
    fn2 : string
        Fortran code.  needs to be compiled with f2py

    Notes
    -----
    The `dim1sin_af_linear` matrix, :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{{a\\left(z\\right)}\\phi_i\\phi_j\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    .. math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` is a piecewise linear function
    w.r.t. :math:`z`, that within a layer are defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """



    v = SympyVarsFor1DSpectralDerivation('z')
    integ_kwargs = dict(risch=False, conds='none')

    phi_i = sympy.sin(v.mi * v.z)
    phi_j = sympy.sin(v.mj * v.z)

    fdiag = sympy.integrate(v.a * phi_i * phi_i, v.z, **integ_kwargs)
    fdiag_loops = fdiag.subs(v.z, v.zbot) - fdiag.subs(v.z, v.ztop)
    fdiag_loops = fdiag_loops.subs(v.map_to_add_index)
    fdiag_vector = fdiag.subs(v.z, v.zbot) - fdiag.subs(v.z, v.ztop)
    fdiag_vector = fdiag_vector.subs(v.map_top_to_t_bot_to_b)

    foff = sympy.integrate(v.a * phi_j * phi_i, v.z, **integ_kwargs)
    foff_loops = foff.subs(v.z, v.zbot) - foff.subs(v.z, v.ztop)
    foff_loops = foff_loops.subs(v.map_to_add_index)
    foff_vector = foff.subs(v.z, v.zbot) - foff.subs(v.z, v.ztop)
    foff_vector = foff_vector.subs(v.map_top_to_t_bot_to_b)

    text_python = """def dim1sin_af_linear(m, at, ab, zt, zb, implementation='vectorized'):

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
                A[i, i] += ({0})
            for i in range(neig-1):
                for j in range(i + 1, neig):
                    A[i, j] += ({1})

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
        A[diag] = np.sum({2}, axis=1)

        mi = m[triu[0]][:, np.newaxis]
        mj = m[triu[1]][:, np.newaxis]
        A[triu] = np.sum({3}, axis=1)
        #A is symmetric
        A[tril] = A[triu]

    return A"""


#    note the the i=j part in the fortran loop  below is because
#      I changed the loop order from layer, i,j to layer, j,i which is
#      i think faster as first index of a fortran array loops faster
#      my sympy code is mased on m[i], hence the need for i=j.
    text_fortran = """      SUBROUTINE dim1sin_af_linear(m, at, ab, zt, zb, a, neig, nlayers)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nlayers
        REAL(DP), intent(in), dimension(0:neig-1) ::m
        REAL(DP), intent(in), dimension(0:nlayers-1) :: at
        REAL(DP), intent(in), dimension(0:nlayers-1) :: ab
        REAL(DP), intent(in), dimension(0:nlayers-1) :: zt
        REAL(DP), intent(in), dimension(0:nlayers-1) :: zb
        REAL(DP), intent(out), dimension(0:neig-1, 0:neig-1) :: a
        INTEGER :: i , j, layer
        REAL(DP) :: a_slope

        a=0.0D0
        DO layer = 0, nlayers-1
          a_slope = (ab(layer) - at(layer)) / (zb(layer) - zt(layer))
          DO j = 0, neig-1
              i=j
{0}
            DO i = j+1, neig-1
{1}
            END DO
          END DO
        END DO

        DO j = 0, neig -2
          DO i = j + 1, neig-1
            a(j,i) = a(i, j)
          END DO
        END DO

      END SUBROUTINE"""





    fn = text_python.format(tw(fdiag_loops,5), tw(foff_loops,6), tw(fdiag_vector,3), tw(foff_vector,3))
    fn2 = text_fortran.format(fcode_one_large_expr(fdiag_loops, prepend='a(i, i) = a(i, i) + '),
                 fcode_one_large_expr(foff_loops, prepend='a(i, j) = a(i, j) + '))

    return fn, fn2


def dim1sin_abf_linear_implementations():
    """Generate code to calculate spectral method integrations

    Performs integrations of `sin(mi * z) * a(z) *b(z) * sin(mj * z)`
    between [0, 1] where a(z) and b(z) are piecewise linear functions of z.
    Code is generated that will produce a square array with the appropriate
    integrals at each location

    Paste the resulting code (at least the loops) into `dim1sin_abf_linear`.

    Creates 3 implementations:
     - 'scalar', python loops (slowest)
     - 'vectorized', numpy (much faster than scalar)
     - 'fortran', fortran loops (fastest).  Needs to be compiled and interfaced
       with f2py.

    Returns
    -------
    fn : string
        Python code. with scalar (loops) and vectorized (numpy) implementations
        also calls the fortran version.
    fn2 : string
        Fortran code.  needs to be compiled with f2py

    Notes
    -----
    The `dim1sin_abf_linear` matrix, :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{{a\\left(z\\right)}{b\\left(z\\right)}\\phi_i\\phi_j\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    .. math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` and :math:`b\\left(z\\right)` are piecewise
    linear functions w.r.t. :math:`z`, that within a layer are defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """

    v = SympyVarsFor1DSpectralDerivation('z')
    integ_kwargs = dict(risch=False, conds='none')

    phi_i = sympy.sin(v.mi * v.z)
    phi_j = sympy.sin(v.mj * v.z)

#    fdiag = sympy.integrate(p['a'] * p['b'] * phi_i * phi_i, z)
    fdiag = sympy.integrate(v.a * v.b * phi_i * phi_i, v.z, **integ_kwargs)
    fdiag_loops = fdiag.subs(v.z, v.zbot) - fdiag.subs(v.z, v.ztop)
    fdiag_loops = fdiag_loops.subs(v.map_to_add_index)
    fdiag_vector = fdiag.subs(v.z, v.zbot) - fdiag.subs(v.z, v.ztop)
    fdiag_vector = fdiag_vector.subs(v.map_top_to_t_bot_to_b)

#    foff = sympy.integrate(p['a'] * p['b'] * phi_j * phi_i, z)
    foff = sympy.integrate(v.a * v.b * phi_j * phi_i, v.z, **integ_kwargs)
    foff_loops = foff.subs(v.z, v.zbot) - foff.subs(v.z, v.ztop)
    foff_loops = foff_loops.subs(v.map_to_add_index)
    foff_vector = foff.subs(v.z, v.zbot) - foff.subs(v.z, v.ztop)
    foff_vector = foff_vector.subs(v.map_top_to_t_bot_to_b)

    text_python = """def dim1sin_abf_linear(m, at, ab, bt, bb,  zt, zb, implementation='vectorized'):

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
                A[i, i] += ({0})
            for i in range(neig-1):
                for j in range(i + 1, neig):
                    A[i, j] += ({1})

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
        A[diag] = np.sum({2}, axis=1)

        mi = m[triu[0]][:, np.newaxis]
        mj = m[triu[1]][:, np.newaxis]
        A[triu] = np.sum({3}, axis=1)
        #A is symmetric
        A[tril] = A[triu]

    return A"""


#    note the the i=j part in the fortran loop  below is because
#      I changed the loop order from layer, i,j to layer, j,i which is
#      i think faster as first index of a fortran array loops faster
#      my sympy code is mased on m[i], hence the need for i=j.
    text_fortran = """\
      SUBROUTINE dim1sin_abf_linear(m, at, ab, bt, bb, zt, zb, a, &
                                    neig, nlayers)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nlayers
        REAL(DP), intent(in), dimension(0:neig-1) ::m
        REAL(DP), intent(in), dimension(0:nlayers-1) :: at,ab,bt,bb,zt,zb
!        REAL(DP), intent(in), dimension(0:nlayers-1) :: at
!        REAL(DP), intent(in), dimension(0:nlayers-1) :: ab
!        REAL(DP), intent(in), dimension(0:nlayers-1) :: bt
!        REAL(DP), intent(in), dimension(0:nlayers-1) :: bb
!        REAL(DP), intent(in), dimension(0:nlayers-1) :: zt
!        REAL(DP), intent(in), dimension(0:nlayers-1) :: zb
        REAL(DP), intent(out), dimension(0:neig-1, 0:neig-1) :: a
        INTEGER :: i , j, layer
        REAL(DP) :: a_slope, b_slope


        a=0.0D0
        DO layer = 0, nlayers-1
          a_slope = (ab(layer) - at(layer)) / (zb(layer) - zt(layer))
          b_slope = (bb(layer) - bt(layer)) / (zb(layer) - zt(layer))
          DO j = 0, neig-1
              i=j
{0}
            DO i = j+1, neig-1
{1}
            END DO
          END DO
        END DO

        DO j = 0, neig -2
          DO i = j + 1, neig-1
            a(j,i) = a(i, j)
          END DO
        END DO

      END SUBROUTINE"""





    fn = text_python.format(tw(fdiag_loops,5), tw(foff_loops,6), tw(fdiag_vector,3), tw(foff_vector,3))
    fn2 = text_fortran.format(fcode_one_large_expr(fdiag_loops, prepend='a(i, i) = a(i, i) + '),
                 fcode_one_large_expr(foff_loops, prepend='a(i, j) = a(i, j) + '))

    return fn, fn2


def dim1sin_D_aDf_linear_implementations():
    """Generate code to calculate spectral method integrations

    Performs integrations of `sin(mi * z) * D[a(z) * D[sin(mj * z),z],z]`
    between [0, 1] where a(z) i piecewise linear functions of z.
    Code is generated that will produce a square array with the appropriate
    integrals at each location

    Paste the resulting code (at least the loops) into `dim1sin_D_aDf_linear`.

    Creates 3 implementations:
     - 'scalar', python loops (slowest)
     - 'vectorized', numpy (much faster than scalar)
     - 'fortran', fortran loops (fastest).  Needs to be compiled and interfaced
       with f2py.

    Returns
    -------
    fn : string
        Python code. with scalar (loops) and vectorized (numpy) implementations
        also calls the fortran version.
    fn2 : string
        Fortran code.  needs to be compiled with f2py

    Notes
    -----
    The `dim1sin_D_aDf_linear` matrix, :math:`A` is given by:

    .. math:: \\mathbf{A}_{i,j}=\\int_{0}^1{\\frac{d}{dz}\\left({a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\right)\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    .. math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` and :math:`b\\left(z\\right)` are piecewise
    linear functions w.r.t. :math:`z`, that within a layer are defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    To make the above integratin simpler we integate by parts to get:

    .. math:: \\mathbf{A}_{i,j}= \\left.\\phi_i{a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\right|_{z=0}^{z=1} -\\int_{0}^1{{a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\frac{d\\phi_i}{dz}\\,dz}

    In this case the sine basis functions means the left term in the above
    equation is zero, leaving us with

    .. math:: \\mathbf{A}_{i,j}= -\\int_{0}^1{{a\\left(z\\right)}\\frac{d\\phi_j}{dz}\\frac{d\\phi_i}{dz}\\,dz}


    """

#    NOTE: remember that fortran does not distinguish between upper and lower
#        case.  When f2py wraps a fortran function with upper case letters then
#        upper case letters will be converted to lower case. e.g. Therefore when
#        calling a fortran function called if fortran fn is
#        'dim1sin_D_aDf_linear' f2py will wrap it as 'dim1sin_d_adf_linear'



    v = SympyVarsFor1DSpectralDerivation('z')
    integ_kwargs = dict(risch=False, conds='none')

    phi_i = sympy.sin(v.mi * v.z)
    phi_j = sympy.sin(v.mj * v.z)

    fdiag = sympy.integrate(-sympy.diff(phi_i, v.z) *
                            v.a *
                            sympy.diff(phi_i, v.z),
                            v.z, **integ_kwargs)
    fdiag_loops = fdiag.subs(v.z, v.zbot) - fdiag.subs(v.z, v.ztop)
    fdiag_loops = fdiag_loops.subs(v.map_to_add_index)
    fdiag_vector = fdiag.subs(v.z, v.zbot) - fdiag.subs(v.z, v.ztop)
    fdiag_vector = fdiag_vector.subs(v.map_top_to_t_bot_to_b)

    foff = sympy.integrate(-sympy.diff(phi_i, v.z) *
                           v.a *
                           sympy.diff(phi_j, v.z),
                           v.z, **integ_kwargs)
    foff_loops = foff.subs(v.z, v.zbot) - foff.subs(v.z, v.ztop)
    foff_loops = foff_loops.subs(v.map_to_add_index)
    foff_vector = foff.subs(v.z, v.zbot) - foff.subs(v.z, v.ztop)
    foff_vector = foff_vector.subs(v.map_top_to_t_bot_to_b)




    text_python = """def dim1sin_D_aDf_linear(m, at, ab, zt, zb, implementation='vectorized'):

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
                A[i, i] += ({0})
            for i in range(neig-1):
                for j in range(i + 1, neig):
                    A[i, j] += ({1})

        #A is symmetric
        for i in range(neig - 1):
            for j in range(i + 1, neig):
                A[j, i] = A[i, j]

    elif implementation == 'fortran':
        try:
            import geotecha.speccon.ext_integrals as ext_integ
            #note the lower case when calling functions interfaced with f2py!
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
        A[diag] = np.sum({2}, axis=1)

        mi = m[triu[0]][:, np.newaxis]
        mj = m[triu[1]][:, np.newaxis]
        A[triu] = np.sum({3}, axis=1)
        #A is symmetric
        A[tril] = A[triu]

    return A"""


#    note the the i=j part in the fortran loop  below is because
#      I changed the loop order from layer, i,j to layer, j,i which is
#      i think faster as first index of a fortran array loops faster
#      my sympy code is mased on m[i], hence the need for i=j.
    text_fortran = """      SUBROUTINE dim1sin_D_aDf_linear(m, at, ab, zt, zb, a, neig, nlayers)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nlayers
        REAL(DP), intent(in), dimension(0:neig-1) ::m
        REAL(DP), intent(in), dimension(0:nlayers-1) :: at
        REAL(DP), intent(in), dimension(0:nlayers-1) :: ab
        REAL(DP), intent(in), dimension(0:nlayers-1) :: zt
        REAL(DP), intent(in), dimension(0:nlayers-1) :: zb
        REAL(DP), intent(out), dimension(0:neig-1, 0:neig-1) :: a
        INTEGER :: i , j, layer
        REAL(DP) :: a_slope

        a=0.0D0
        DO layer = 0, nlayers-1
          a_slope = (ab(layer) - at(layer)) / (zb(layer) - zt(layer))
          DO j = 0, neig-1
              i=j
{0}
            DO i = j+1, neig-1
{1}
            END DO
          END DO
        END DO

        DO j = 0, neig -2
          DO i = j + 1, neig-1
            a(j,i) = a(i, j)
          END DO
        END DO

      END SUBROUTINE"""





    fn = text_python.format(tw(fdiag_loops,5), tw(foff_loops,6), tw(fdiag_vector,3), tw(foff_vector,3))
    fn2 = text_fortran.format(fcode_one_large_expr(fdiag_loops, prepend='a(i, i) = a(i, i) + '),
                 fcode_one_large_expr(foff_loops, prepend='a(i, j) = a(i, j) + '))

    return fn, fn2


def dim1sin_ab_linear_implementations():
    """Generate code to calculate spectral method integrations

    Performs integrations of `sin(mi * z) * a(z) * b(z)`
    between [0, 1] where a(z) and b(z) are piecewise linear functions of z.
    Code is generated that will produce a 1d array with the appropriate
    integrals at each location.

    Paste the resulting code (at least the loops) into `dim1sin_ab_linear`.

    Notes
    -----
    The `dim1sin_ab_linear` which should be treated as a column vector,
    :math:`A` is given by:

    .. math:: \\mathbf{A}_{i}=\\int_{0}^1{{a\\left(z\\right)}{b\\left(z\\right)}\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    .. math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` and :math:`b\\left(z\\right)` are piecewise
    linear functions w.r.t. :math:`z`, that within a layer are defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """

    v = SympyVarsFor1DSpectralDerivation('z')
    integ_kwargs = dict(risch=False, conds='none')

    phi_i = sympy.sin(v.mi * v.z)
#    phi_j = sympy.sin(v.mj * v.z)

    fcol = sympy.integrate(v.a * v.b * phi_i, v.z, **integ_kwargs)
    fcol_loops = fcol.subs(v.z, v.zbot) - fcol.subs(v.z, v.ztop)
    fcol_loops = fcol_loops.subs(v.map_to_add_index)
    fcol_vector = fcol.subs(v.z, v.zbot) - fcol.subs(v.z, v.ztop)
    fcol_vector = fcol_vector.subs(v.map_top_to_t_bot_to_b)



#    mpv, p = create_layer_sympy_var_and_maps_vectorized(layer_prop=['z','a', 'b'])
#    mp, p = create_layer_sympy_var_and_maps(layer_prop=['z','a','b'])
#
#    phi_i = sympy.sin(mi * z)


#    fcol = sympy.integrate(p['a'] * p['b'] * phi_i, z)
#    fcol_loops = fcol.subs(z, mp['zbot']) - fcol.subs(z, mp['ztop'])
#    fcol_loops = fcol_loops.subs(mp)
#    fcol_vector = fcol.subs(z, mpv['zbot']) - fcol.subs(z, mpv['ztop'])
#    fcol_vector = fcol_vector.subs(mpv)

    text_python = """def dim1sin_ab_linear(m, at, ab, bt, bb,  zt, zb, implementation='vectorized'):

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
                A[i] += ({0})

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
        A[:] = np.sum({1}, axis=1)


    return A"""


#    note the the i=j part in the fortran loop  below is because
#      I changed the loop order from layer, i,j to layer, j,i which is
#      i think faster as first index of a fortran array loops faster
#      my sympy code is mased on m[i], hence the need for i=j.
    text_fortran = """\
      SUBROUTINE dim1sin_ab_linear(m, at, ab, bt, bb, zt, zb, &
                                   a, neig, nlayers)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nlayers
        REAL(DP), intent(in), dimension(0:neig-1) ::m
        REAL(DP), intent(in), dimension(0:nlayers-1) :: at,ab,bt,bb,zt,zb
        REAL(DP), intent(out), dimension(0:neig-1) :: a
        INTEGER :: i, layer
        REAL(DP) :: a_slope, b_slope
        a=0.0D0
        DO layer = 0, nlayers-1
          a_slope = (ab(layer) - at(layer)) / (zb(layer) - zt(layer))
          b_slope = (bb(layer) - bt(layer)) / (zb(layer) - zt(layer))
          DO i = 0, neig-1
{0}
          END DO
        END DO

      END SUBROUTINE"""





    fn = text_python.format(tw(fcol_loops,5), tw(fcol_vector,3))
    fn2 = text_fortran.format(fcode_one_large_expr(fcol_loops, prepend='a(i) = a(i) + '))

    return fn, fn2


def dim1sin_abc_linear_implementations():
    """Generate code to calculate spectral method integrations

    Performs integrations of `sin(mi * z) * a(z) * b(z) * c(z)`
    between [0, 1] where a(z), b(z), c(z) are piecewise linear functions of z.
    Code is generated that will produce a 1d array with the appropriate
    integrals at each location.

    Paste the resulting code (at least the loops) into `dim1sin_abc_linear`.

    Creates 3 implementations:
     - 'scalar', python loops (slowest)
     - 'vectorized', numpy (much faster than scalar)
     - 'fortran', fortran loops (fastest).  Needs to be compiled and interfaced
       with f2py.

    Returns
    -------
    fn: string
        Python code. with scalar (loops) and vectorized (numpy) implementations
        also calls the fortran version.
    fn2: string
        Fortran code.  needs to be compiled with f2py

    Notes
    -----
    The `dim1sin_abc_linear` which should be treated as a column vector,
    :math:`A` is given by:

    .. math:: \\mathbf{A}_{i}=\\int_{0}^1{{a\\left(z\\right)}{b\\left(z\\right)}{c\\left(z\\right)}\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    .. math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)`, :math:`b\\left(z\\right)`, and
    :math:`c\\left(z\\right)` are piecewise linear functions
    w.r.t. :math:`z`, that within a layer are defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """

    v = SympyVarsFor1DSpectralDerivation('z')
    integ_kwargs = dict(risch=False, conds='none')

    phi_i = sympy.sin(v.mi * v.z)

    fcol = sympy.integrate(v.a * v.b * v.c* phi_i, v.z, **integ_kwargs)
    fcol_loops = fcol.subs(v.z, v.zbot) - fcol.subs(v.z, v.ztop)
    fcol_loops = fcol_loops.subs(v.map_to_add_index)
    fcol_vector = fcol.subs(v.z, v.zbot) - fcol.subs(v.z, v.ztop)
    fcol_vector = fcol_vector.subs(v.map_top_to_t_bot_to_b)


    text_python = """def dim1sin_abc_linear(m, at, ab, bt, bb, ct, cb, zt, zb, implementation='vectorized'):

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
                A[i] += ({0})

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
        A[:] = np.sum({1}, axis=1)


    return A"""


#    note the the i=j part in the fortran loop  below is because
#      I changed the loop order from layer, i,j to layer, j,i which is
#      i think faster as first index of a fortran array loops faster
#      my sympy code is mased on m[i], hence the need for i=j.
    text_fortran = """\
      SUBROUTINE dim1sin_abc_linear(m, at, ab, bt, bb, ct, cb, &
                                    zt, zb, a, neig, nlayers)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nlayers
        REAL(DP), intent(in), dimension(0:neig-1) ::m
        REAL(DP), intent(in), dimension(0:nlayers-1) :: at,ab,bt,bb, &
                                                        ct,cb,zt,zb
        REAL(DP), intent(out), dimension(0:neig-1) :: a
        INTEGER :: i, layer
        REAL(DP) :: a_slope, b_slope, c_slope
        a=0.0D0
        DO layer = 0, nlayers-1
          a_slope = (ab(layer) - at(layer)) / (zb(layer) - zt(layer))
          b_slope = (bb(layer) - bt(layer)) / (zb(layer) - zt(layer))
          c_slope = (cb(layer) - ct(layer)) / (zb(layer) - zt(layer))
          DO i = 0, neig-1
{0}
          END DO
        END DO

      END SUBROUTINE"""





    fn = text_python.format(tw(fcol_loops,5), tw(fcol_vector,3))
    fn2 = text_fortran.format(fcode_one_large_expr(fcol_loops, prepend='a(i) = a(i) + '))

    return fn, fn2


def dim1sin_D_aDb_linear_implementations():
    """Generate code to calculate spectral method integrations

    Performs integrations of `sin(mi * z) * D[a(z) * D[b(z), z], z]`
    between [0, 1] where a(z) is a piecewisepiecewise linear function of z,
    and b(z) is a linear function of z.
    Code is generated that will produce a 1d array with the appropriate
    integrals at each location.

    Paste the resulting code (at least the loops) into `dim1sin_D_aDb_linear`.

    Creates 3 implementations:
     - 'scalar', python loops (slowest)
     - 'vectorized', numpy (much faster than scalar)
     - 'fortran', fortran loops (fastest).  Needs to be compiled and interfaced
       with f2py.

    .. warning::
        The functions produced are set up to accept the b(z) input as
        piecewise linear, i.e. zt, zb, bt, bb etc. It is up to the user to
        ensure that the bt and bb are such that they define a continuous
        linear function. eg. to define b(z)=z+1 then use
        zt=[0,0.4], zb=[0.4, 1], bt=[1,1.4], bb=[1.4,2].


    Returns
    -------
    fn : string
        Python code. with scalar (loops) and vectorized (numpy) implementations
        also calls the fortran version.
    fn2 : string
        Fortran code.  needs to be compiled with f2py

    Notes
    -----
    The `dim1sin_D_aDb_linear` which should be treated as a column vector,
    :math:`A` is given by:

    .. math:: \\mathbf{A}_{i}=\\int_{0}^1{\\frac{d}{dz}\\left({a\\left(z\\right)}\\frac{d}{dz}{b\\left(z\\right)}\\right)\\phi_i\\,dz}

    where the basis function :math:`\\phi_i` is given by:

    .. math:: \\phi_i\\left(z\\right)=\\sin\\left({m_i}z\\right)

    and :math:`a\\left(z\\right)` is a piecewise
    linear functions w.r.t. :math:`z`, that within a layer is defined by:

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

    v = SympyVarsFor1DSpectralDerivation('z')
    integ_kwargs = dict(risch=False, conds='none')

    phi_i = sympy.sin(v.mi * v.z)

    fcol = - sympy.diff(v.b, v.z) * sympy.integrate(sympy.diff(phi_i, v.z) *
                                                    v.a, v.z,
                                                    **integ_kwargs)
    fcol_loops = fcol.subs(v.z, v.zbot) - fcol.subs(v.z, v.ztop)
    fcol_loops = fcol_loops.subs(v.map_to_add_index)
    fcol_vector = fcol.subs(v.z, v.zbot) - fcol.subs(v.z, v.ztop)
    fcol_vector = fcol_vector.subs(v.map_top_to_t_bot_to_b)


    v = SympyVarsFor1DSpectralDerivation('z', slope=False)
    fend = sympy.diff(v.b, v.z) * phi_i * v.a

    fbot = fend.subs(v.z, v.zbot)
    fbot_loops = fbot.subs(v.map_to_add_index)
    nlayers = sympy.tensor.Idx('nlayers')
    fbot_loops = fbot_loops.subs([(v.layer, nlayers - 1)])
    fbot_vector = fbot.subs(v.map_to_add_index[2:])
    fbot_vector = fbot_vector.subs([(v.layer, -1)])

    ftop = fend.subs(v.z, v.ztop)
    ftop_loops = ftop.subs(v.map_to_add_index)
    ftop_loops = ftop_loops.subs([(v.layer, 0)])
    ftop_vector = ftop.subs(v.map_to_add_index[2:])
    ftop_vector = ftop_vector.subs([(v.layer, 0)])

    fends_loops = fbot_loops - ftop_loops
    fends_vector = fbot_vector - ftop_vector

    text_python = """def dim1sin_D_aDb_linear(m, at, ab, bt, bb,  zt, zb, implementation='vectorized'):

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
                A[i] += ({0})

        for i in range(neig):
            A[i] += ({1})
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
        A[:] = np.sum({2}, axis=1)
        mi = m
        A[:]+= ({3})
    return A"""


#    note the the i=j part in the fortran loop  below is because
#      I changed the loop order from layer, i,j to layer, j,i which is
#      i think faster as first index of a fortran array loops faster
#      my sympy code is mased on m[i], hence the need for i=j.
    text_fortran = """\
      SUBROUTINE dim1sin_D_aDb_linear(m, at, ab, bt, bb, zt, zb, &
                                      a, neig, nlayers)
        USE types
        IMPLICIT NONE

        INTEGER, intent(in) :: neig
        INTEGER, intent(in) :: nlayers
        REAL(DP), intent(in), dimension(0:neig-1) ::m
        REAL(DP), intent(in), dimension(0:nlayers-1) :: at,ab,bt,bb,zt,zb
        REAL(DP), intent(out), dimension(0:neig-1) :: a
        INTEGER :: i, layer
        REAL(DP) :: a_slope, b_slope
        a=0.0D0
        DO layer = 0, nlayers-1
          a_slope = (ab(layer) - at(layer)) / (zb(layer) - zt(layer))
          b_slope = (bb(layer) - bt(layer)) / (zb(layer) - zt(layer))
          DO i = 0, neig-1
{0}
          END DO
        END DO

        DO i = 0, neig-1
{1}
        END DO

      END SUBROUTINE"""





    fn = text_python.format(tw(fcol_loops,5), tw(fends_loops,4), tw(fcol_vector,3), tw(fends_vector,3))
    fn2 = text_fortran.format(fcode_one_large_expr(fcol_loops, prepend='a(i) = a(i) + '),
                              fcode_one_large_expr(fends_loops, prepend='a(i) = a(i) + '))

    return fn, fn2


def dim1sin_a_linear_between():
    """Generate code to calculate spectral method integrations

    Performs integrations of `sin(mi * z) * a(z)`
    between [z1, z2] where a(z) is a piecewise linear functions of z.

    Calculates array A[len(z), len(m)].

    Paste the resulting code into `dim1sin_a_linear_between`.

    Returns
    -------
    fn : string
        Python code with scalar (loops) implementation

    Notes
    -----
    The `dim1sin_a_linear_between`, :math:`A`, is given by:

    .. math:: \\mathbf{A}_{i,j}=
                \\int_{z_1}^{z_2}{{a\\left(z\\right)}\\phi_j\\,dz}

    where the basis function :math:`\\phi_j` is given by:

    .. math:: \\phi_j\\left(z\\right)=\\sin\\left({m_j}z\\right)

    and :math:`a\\left(z\\right)` is a piecewise
    linear functions w.r.t. :math:`z`, that within a layer are defined by:

    .. math:: a\\left(z\\right) = a_t+\\frac{a_b-a_t}{z_b-z_t}\\left(z-z_t\\right)

    with :math:`t` and :math:`b` subscripts representing 'top' and 'bottom' of
    each layer respectively.

    """

    v = SympyVarsFor1DSpectralDerivation('z', slope=True)
    integ_kwargs = dict(risch=False, conds='none')
    v.z1 = sympy.tensor.IndexedBase('z1')
    v.z2 = sympy.tensor.IndexedBase('z2')


    phi_j = sympy.sin(v.mj * v.z)

    f = sympy.integrate(v.a * phi_j, v.z, **integ_kwargs)

    both = f.subs(v.z, v.z2[v.i]) - f.subs(v.z, v.z1[v.i])
    both = both.subs(v.map_to_add_index)

    between = f.subs(v.z, v.zbot) - f.subs(v.z, v.ztop)
    between = between.subs(v.map_to_add_index)

    z1_only = f.subs(v.z, v.zbot) - f.subs(v.z, v.z1[v.i])
    z1_only = z1_only.subs(v.map_to_add_index)

    z2_only = f.subs(v.z, v.z2[v.i]) - f.subs(v.z, v.ztop)
    z2_only = z2_only.subs(v.map_to_add_index)


#    mp, p = create_layer_sympy_var_and_maps(layer_prop=['z', 'a', 'b'])
#    sympy.var('z1, z2')
#
#    z1 = sympy.tensor.IndexedBase('z1')
#    z2 = sympy.tensor.IndexedBase('z2')
#    i = sympy.tensor.Idx('i')
#    j = sympy.tensor.Idx('j')
#
#
#
#    phi_j = sympy.sin(mj * z)
#
#    f = sympy.integrate(p['a'] * phi_j, z)
#
#    both = f.subs(z, z2[i]) - f.subs(z, z1[i])
#    both = both.subs(mp)
#
#    between = f.subs(z, mp['zbot']) - f.subs(z, mp['ztop'])
#    between = between.subs(mp)
#
#    z1_only = f.subs(z, mp['zbot']) - f.subs(z, z1[i])
#    z1_only = z1_only.subs(mp)
#
#    z2_only = f.subs(z, z2[i]) - f.subs(z, mp['ztop'])
#    z2_only = z2_only.subs(mp)

    text = """dim1sin_a_linear_between(m, at, ab, zt, zb, z):
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
                A[i,j] += ({0})
        for layer in segment_z1_only[i]:
            a_slope = (ab[layer] - at[layer]) / (zb[layer] - zt[layer])
            for j in range(neig):
                A[i,j] += ({1})
        for layer in segments_between[i]:
            a_slope = (ab[layer] - at[layer]) / (zb[layer] - zt[layer])
            for j in range(neig):
                A[i,j] += ({2})
        for layer in segment_z2_only[i]:
            a_slope = (ab[layer] - at[layer]) / (zb[layer] - zt[layer])
            for j in range(neig):
                A[i,j] += ({3})
    return A"""


    fn = text.format(tw(both,5), tw(z1_only,5), tw(between,5), tw(z2_only,5))

    return fn

if __name__ == '__main__':
    pass
#    import nose
#    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])
#    print(EDload_coslinear_test())


#    #print(EDload_coslinear())
#    print(EDload_linear())
#    #print(Eload_coslinear())
#    print(Eload_linear())
#    #print(dim1_ab_linear_between())
#    #print(dim1sin_D_aDb_linear())
#    #print(dim1sin_D_aDb_linear_implementations())
#    print(dim1sin_D_aDf_linear())
#    #print(dim1sin_D_aDf_linear_implementations())
#    #print(dim1sin_D_aDf_linear_v2())
#    #print(dim1sin_a_linear_between())
#    #print(dim1sin_ab_linear())
#    #print(dim1sin_ab_linear_implementations())
#    #print(dim1sin_abc_linear())
#    #print(dim1sin_abc_linear_implementations())
#    print(dim1sin_abf_linear())
#    print(dim1sin_abf_linear_implementations())
#    print(dim1sin_af_linear())
#    print(dim1sin_af_linear_fortran())
#    print(dim1sin_af_linear_implementations())
#    #print(dim1sin_af_linear_vectorize())
#    fn, fn2=Eload_linear_implementations();print(fn);print('#'*40); print(fn2)
#    fn, fn2=EDload_linear_implementations();print(fn);print('#'*40); print(fn2)
#    fn, fn2=Eload_coslinear_implementations();print(fn);print('#'*40); print(fn2)
#    fn, fn2=EDload_coslinear_implementations();print(fn);print('#'*40); print(fn2)
#    fn, fn2=dim1sin_af_linear_implementations();print(fn);print('#'*40); print(fn2)
#    fn, fn2=dim1sin_abf_linear_implementations();print(fn);print('#'*40); print(fn2)
#    fn, fn2=dim1sin_D_aDf_linear_implementations();print(fn);print('#'*40); print(fn2)
#    fn, fn2=dim1sin_ab_linear_implementations();print(fn);print('#'*40); print(fn2)
#    fn, fn2=dim1sin_abc_linear_implementations();print(fn);print('#'*40); print(fn2)
#    fn, fn2=dim1sin_D_aDb_linear_implementations();print(fn);print('#'*40); print(fn2)
    print(dim1sin_a_linear_between())