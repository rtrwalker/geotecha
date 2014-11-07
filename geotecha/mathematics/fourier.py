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
"""some routines related to fourier transforms"""


from __future__ import division, print_function
import matplotlib.pyplot
import numpy as np
from scipy import integrate
from scipy.special import jn_zeros
from scipy.special import jn
from matplotlib import pyplot as plt
import functools

import unittest
from numpy.testing import assert_allclose


from geotecha.mathematics.quadrature import gl_quad
from geotecha.mathematics.quadrature import gk_quad
from geotecha.mathematics.quadrature import gauss_kronrod_abscissae_and_weights
from geotecha.mathematics.quadrature import gauss_legendre_abscissae_and_weights
from geotecha.mathematics.quadrature import shanks_table
from geotecha.mathematics.quadrature import shanks

#for fourier transform pairs see
# http://en.wikibooks.org/wiki/Signals_and_Systems/Table_of_Fourier_Transforms
def rect(x, *args):
    """rectangle function
    -0.5<=x<=0.5 = 1, otherwise=0"""
    if -0.5<=x<=0.5:
        return 1
    else:
        return 0


def unit_step(x, *args):
    if x >= 0:
        return 1
    else:
        return 0

#real and symmetric
def fourier1(x, a):
    """exp(- a * abs(x))"""
    return np.exp(-a * abs(x))
def fourier1_(x, a):
    """2 * a / (a**2 + x**2)"""
    return  2 * a / (a**2 + x**2)
#real and unsymmetric
def fourier2(t, tau):
    """rect(t/tau)"""
    return rect(t/tau)
def fourier2_(w, tau):
    """tau * sinc(tau*W/(2*pi))"""
    return tau * np.sinc(tau * w/(2*np.pi))

#unsymmetric which give complex
def fourier3(x, *args):
    """-0.5+unit_step(x)"""
    #note you cannot fourier transform this by quadrature because the
    #integral does not converge
    return -0.5 + unit_step(x)
def fourier3_(w, *args):
    """1/(1.j*w)"""
    return 1 / (1.j * w)
def fourier4(t, b):
    """exp(-b*t)*unit_step(t)"""
    return np.exp(-b*t)*unit_step(t)
def fourier4_(w, b):
    """1/(1.j*w + b)"""
    return 1/(1.j*w + b)
#sine transformation pairs
def sine1(x, b):
    """exp(-x*b)"""
    return np.exp(-x*b)
def sine1_(w, b):
    """w/(w**2+b**2)"""
    return w/(w**2+b**2)

def real_func(x, *myargs):
    """Real part of a function

    Basically return np.real(func(x, *myargs[1:])) where func is the first
    argument after x.

    Parameters
    ----------
    x : float
        value to evaluate function at
    func : function/callable
        Function from which to return the real part. Always the first
        argument after `x`
    myargs : optional
        any remaining arguments will be passed to func(x, *myargs[1:])

    Returns
    -------
    out : ndarray
        real part of func(x, *myargs)

    See also
    --------
    imag_func : imaginary part of function

    Examples
    --------
    >>> def f(x, a):
    ...  return a*x+a*1.j
    >>> real_func(2, f, 4)
    8.0
    >>> real_func(3.j,f, 2)
    0.0
    >>> real_func(np.array([3.j, 1+2.j]),f, 2)
    array([ 0.,  2.])

    """

    func = myargs[0]
    myargs = (x,) + myargs[1:]
    return +np.real(func(*myargs))


def imag_func(x, *myargs):
    """Imaginary part of a function

    Basically return np.imag(func(x, *myargs[1:])) where func is the first
    argument after x.

    Parameters
    ----------
    x : float
        value to evaluate function at
    func : function/callable
        Function from which to return the imaginary part. Always the first
        argument after `x`
    myargs : optional
        any remaining arguments will be passed to func(x, *myargs[1:])

    Returns
    -------
    out : ndarray
        imaginary part of func(x, *myargs)

    See also
    --------
    real_func : real part of function

    Examples
    --------
    >>> def f(x, a):
    ...  return a*x+a*1.j
    >>> imag_func(2, f, 4)
    4.0
    >>> imag_func(3.j,f, 2)
    8.0
    >>> imag_func(np.array([3.j, 2.j]),f, 2)
    array([ 8.,  6.])

    """

    func = myargs[0]
    myargs = (x,) + myargs[1:]
    return +np.imag(func(*myargs))


def func_mirror_for_even_weight(x, *myargs):
    """
    Given a composite function f(x) * w(x) where w(x) is an even weighting
    function, return g(x) such that g(x)*w(x) gives same value as f(-x)*w(-x).
    This can be useful in transforming a fourier cosine integral
    with negative integation limits to one with positive limits.

    Parameters
    ----------
    x : float
        value to evaluate function at
    func : function/callable
        function to mirror. Always the first argument after `x`
    myargs : optional
        any remaining arguments will be passed to func

    Returns
    -------
    out : ndarray
        value of func(-x, *myargs)

    See also
    --------
    func_mirror_for_odd_weight : mirror for an odd wieght function

    Examples
    --------
    >>> def f(x, a):
    ...  return a*x+1
    >>> func_mirror_for_even_weight(5, f, 2)
    -9
    >>> def ff(x, a):
    ...  return a*x + 1.j
    >>> func_mirror_for_even_weight(3, real_func, ff, 4)
    -12.0

    """

    y = -x
    func = myargs[0]
    myargs = (y,) + myargs[1:]
    return +func(*myargs)

def func_mirror_for_odd_weight(x, *myargs):
    """
    Given a composite function f(x) * w(x) where w(x) is an odd weighting
    function, return g(x) such that g(x)*w(x) gives same value as f(-x)*w(-x).
    This can be useful in transforming a fourier sine integral
    with negative integration limits to one with positive limits.

    Parameters
    ----------
    x : float
        value to evaluate function at
    func : function/callable
        function to mirror. Always the first argument after `x`
    myargs : optional
        any remaining arguments will be passed to func

    Returns
    -------
    out : ndarray
        value of -func(-x, *myargs)

    See also
    --------
    func_mirror_for_even_weight : mirror for an even wieght function

    Examples
    --------
    >>> def f(x, a):
    ...  return a*x+1
    >>> func_mirror_for_odd_weight(5, f, 2)
    9
    >>> def ff(x, a):
    ...  return a*x + 1.j
    >>> func_mirror_for_odd_weight(3, real_func, ff, 4)
    12.0

    """

    y = -x
    func = myargs[0]
    myargs = (y,) + myargs[1:]
    return -func(*myargs)


def cosine_transform(func, w, args=(), a=0.0, b=np.inf):
    """Fourier cosine transform

    note that any function that can divide by zero may cause problems because
    QUADPACK includes the end points in integration

    Parameters
    ----------
    func : function/callable
        function to transform.  `func` will be called func(x, *args). `func`
        must return a real.
    w : float
        transform
    args : tuple, optional
        arguments to pass to `func`
    a, b : float, optional
        integration limits. defualt a=0.0, b=np.inf

    Returns
    -------
    value : float
        value of transform
    err : float
        error estimate from quadpack

    Notes
    -----

    The fourier cosine trasnform is given by:

    .. math:: F_c=\mathcal{F}_c\\{f(x)\\}(w) =
                \\int_0^{\\infty}f(x)\\cos(wx)\\,\\mathrm{d}x
    """

    return integrate.quad(func, a, b, args=args, weight='cos', wvar=w)

def sine_transform(func, w, args=(), a=0.0, b=np.inf):
    """Fourier sine transform

    note that any function that can divide by zero may cause problems because
    QUADPACK includes the end points in integration

    Parameters
    ----------
    func : function/callable
        function to transform.  `func` will be called func(x, *args). `func`
        must return a real.
    w : float
        transform
    args : tuple, optional
        arguments to pass to `func`
    a, b : float, optional
        integration limits. defualt a=0.0, b=np.inf

    Returns
    -------
    value : float
        value of transform
    err : float
        error estimate from quadpack

    Notes
    -----

    The fourier sine transform is given by:

    .. math:: F_s=\mathcal{F}_s\\{f(x)\\}(w) =
                \\int_0^{\\infty}f(x)\\sin(wx)\\,\\mathrm{d}x
    """

    return integrate.quad(func, a, b, args=args, weight='sin', wvar=w)




class FourierTransform(object):
    """One dimensional Fourier transform using scipy.quad

    note that any function that can divide by zero may cause problems because
    QUADPACK includes the end points in integration

    Parameters
    ----------
    func : function
        function to transform. func is called by func(x, *args)
    args : tuple, optional
        tuple of arguments to pass to func.  default args=()
    inv : True/False, optional
        If True then the inverse Fourier transform will be performed.
        Default = False
    func_is_real : True/False, optional
        If True then func is purely real.  It returns a real number.
        default=False
    func_is_imag : True/False, optional
        If True then func is purely imaginary.  It returns a real number that
        which should be multiplied by i. Default=False
    real_part_even : True/False, optional
        If True then the real part of func is even. Default=False
    real_part_odd : True/False, optional
        If True then the real part of func is odd. Default=False
    imag_part_even : True/False, optional
        If True then the imaginary part of func is even. Default=False
    imag_part_odd : True/False, optional
        If True then the imaginary part of func is odd. Default=False
    a, b : float, optional
        integration limits. defualt a=0.0, b=np.inf

    Attributes
    ----------
    inv_sign : float
        the sign of sum expressions changes for the inverse fourier transform
        `inv_sign` accounts for that sign change.  If inv=True, inv_sign=-1;
        If inv=False, inv_sign=+1.
    inv_const : float
        for inverse fourier transform all expressions are multiplied by
        1/(2*pi).


    """
    def __init__(self, func, args=(), inv=False,
                 func_is_real=False, func_is_imag=False,
                 real_part_even=False, real_part_odd=False,
                 imag_part_even=False, imag_part_odd=False,
                 a=0.0, b=np.inf):

        self.func = func
        self.args = args
        self.inv = inv
        if self.inv:
            self.inv_const=(2.0*np.pi)**(-1) #fourier inverse is 1/(2*pi) * integral
            self.inv_sign = -1 # some fourier inverse terms are negative
        else:
            self.inv_const = 1
            self.inv_sign = 1

        self.func_is_real = func_is_real
        self.func_is_imag = func_is_imag
        self.real_part_even = real_part_even
        self.real_part_odd = real_part_odd
        self.imag_part_even = imag_part_even
        self.imag_part_odd = imag_part_odd
        self.a = a
        self.b = b

        self.fargs=(self.func,) + self.args

    def real_func(self, x):
        """real part of func"""
        return +np.real(self.func(x, *self.args))
    def imag_func(self, x):
        """imaginary part of func"""
        return +np.imag(self.func(x, *self.args))


    def mfe(self, x):
        """mirror func for even weight function"""
        return func_mirror_for_even_weight(x, *self.fargs)
    def mfo(self, x):
        """mirror func for odd weight function"""
        return func_mirror_for_odd_weight(x, *self.fargs)
    def mfre(self, x):
        """mirror real(func) for even weight function"""
        return +np.real(func_mirror_for_even_weight(x, *self.fargs))
    def mfro(self, x):
        """mirror real(func) for odd weight function"""
        return +np.real(func_mirror_for_odd_weight(x, *self.fargs))
    def mfie(self, x):
        """mirror imag(func) for even weight function"""
        return +np.imag(func_mirror_for_even_weight(x, *self.fargs))
    def mfio(self, x):
        """mirror imag(func) for odd weight function"""
        return +np.imag(func_mirror_for_odd_weight(x, *self.fargs))



    def __call__(self, s):
        """perform 1d fourier transform at s"""

        if self.func_is_real:
            if self.real_part_even:
                igral, err = [2 * self.inv_const * v for v in
                        cosine_transform(self.func, s, self.args,
                                         a=self.a, b=self.b)]
                return igral, err
            elif self.real_part_odd:
                igral, err = [2.j * self.inv_const * self.inv_sign * v for v in
                        sine_transform(self.func, s, self.args, a=self.a, b=self.b)]
                return -igral, err

            else:
                #func is real and exhibits no symmetry.
                igral = 0
                err = 0

                # real transform result [-inf, 0]
                ig, er = [self.inv_const * v for v in
                        cosine_transform(self.mfe, s, a=self.a, b=self.b)]
                igral += ig; err += er
                # real transform result [-, inf]
                ig, er = [self.inv_const * v for v in
                        cosine_transform(self.func, s, self.args, a=self.a, b=self.b)]
                igral += ig; err += er
                # imag transform result [0, inf]
                ig, er = [1.j * self.inv_const * self.inv_sign * v for v in
                        sine_transform(self.mfo, s, a=self.a, b=self.b)]
                igral -= ig; err += er
                # imag transform result [-inf, 0]
                ig, er = [1.j * self.inv_const * self.inv_sign * v for v in
                        sine_transform(self.func, s, self.args, a=self.a, b=self.b)]
                igral -= ig; err += er
                return igral, err

        if self.func_is_imag:
            if self.imag_part_even:
                igral, err = [2.j * self.inv_const * v for v in
                        cosine_transform(self.func, s, self.args, a=self.a, b=self.b)]
                return igral, err
            elif self.imag_part_odd:
                igral, err =  [2 * self.inv_const * self.inv_sign * v for v in
                        sine_transform(self.func, s, self.args, a=self.a, b=self.b)]
                return igral, err
            else:
                #func is imaginary and ehibits non symmetry
                igral = 0
                err = 0

                # imag transform result [-inf, 0]
                ig, er = [1.j * self.inv_const * v for v in
                        cosine_transform(self.mfe, s, a=self.a, b=self.b)]
                igral += ig; err += er
                # imag transform result [0, inf]
                ig,er = [1.j * self.inv_const * v for v in
                        cosine_transform(self.func, s, self.args, a=self.a, b=self.b)]
                igral += ig; err += er
                # real transform result [-inf, 0]
                ig, er = [self.inv_const * self.inv_sign * v for v in
                        sine_transform(self.mfo, s, a=self.a, b=self.b)]
                igral -= ig; err += er
                # real transform result [0, inf]
                ig, er = [self.inv_const * self.inv_sign * v for v in
                        sine_transform(self.func, s, self.args, a=self.a, b=self.b)]
                igral -= ig; err += er

                return igral, err

        #if we have reached here then func is complex
        #use real and imag parts of func
        igral = 0
        err = 0
        if self.real_part_even:
            ig, er = [2 * self.inv_const * v for v in
                        cosine_transform(self.real_func, s, a=self.a, b=self.b)]
            igral += ig; err += er
        elif self.real_part_odd:
            # 2 * I(real_func * sin(s*x), 0, +inf)
            ig, er = [2.j * self.inv_const * self.inv_sign * v for v in
                        sine_transform(self.real_func, s, a=self.a, b=self.b)]
            igral -= ig; err += er
        else:
            #real part of function exhibits no symmetry
            # real transform result [-inf, 0]
            ig, er = [self.inv_const * v for v in
                        cosine_transform(self.mfre, s, a=self.a, b=self.b)]
            igral += ig; err += er
            # real transform result [0, inf]
            ig, er = [self.inv_const * v for v in
                        cosine_transform(self.real_func, s, a=self.a, b=self.b)]
            igral += ig; err += er
            # imag transform result [-inf, 0]
            ig, er = [1.j * self.inv_const * self.inv_sign * v for v in
                        sine_transform(self.mfro, s, a=self.a, b=self.b)]
            igral -= ig; err += er
            # imag transform result [0, inf]
            ig, er = [1.j * self.inv_const * self.inv_sign * v for v in
                        sine_transform(self.real_func, s, a=self.a, b=self.b)]
            igral -= ig; err += er


        if self.imag_part_even:
            igral, err = [2.j * self.inv_const * v for v in
                        cosine_transform(self.func, s, self.args, a=self.a, b=self.b)]
            igral += ig; err += er
        elif self.real_part_odd:
            igral, err = [2 * self.inv_const * self.inv_sign * v for v in
                        sine_transform(self.func, s, self.args, a=self.a, b=self.b)]
            igral += ig; err += er
        else:
            #imag part of function exhibits no symmetry
            # imag transform result [-inf, 0]
            ig, er = [1.j * self.inv_const * v for v in
                        cosine_transform(self.mfie, s, a=self.a, b=self.b)]
            igral += ig; err += er
            # imag transform result [0, inf]
            ig, er = [1.j * self.inv_const * v for v in
                        cosine_transform(self.imag_func, s, a=self.a, b=self.b)]
            igral += ig; err += er
            # real transform result [-inf, 0]
            ig, er = [self.inv_const * self.inv_sign * v for v in
                        sine_transform(self.mfio, s, a=self.a, b=self.b)]
            igral += ig; err += er
            # real transform result [0, inf]
            ig, er = [self.inv_const * self.inv_sign * v for v in
                        sine_transform(self.imag_func, s, a=self.a, b=self.b)]
            igral += ig; err += er
        return igral, err




def vcosine_transform(f, s, args=(), m=20, ng=20, shanks_ind=None):
    """Cosine transform of f(x) at transform variable s

    Parameters
    ----------
    f : function or method
        function to apply cosine trasnform to.  f is called with
        f(x, *args)
    s : 1d array
        coordinate(s) to evaluate transform at
    args : tuple, optional
        arguments to pass to f
    m : int, optional
        number of segments to break the integration interval into.  Each
        segment will be between the zeros of the cos function, default=20
    ng : [2-20, 32, 64, 100], optional
        number of gauss points to use in integration.
    shanks_ind : int, optional
        Start position of intervals to start shanks extrapolatoin.
        default=None i.e. no extrapolation.
        Be careful when using shanks extrapolation; make sure you only begin
        to use it after the intgrand is well behaved.

    Returns
    -------
    f : 1d array of float
        value of transform at s

    Notes
    -----
    Careful with singularities.  Because there is no way to increase the
    integration points at a particular sport the infinite behaviur may not be
    captured well. For example x**-0.5 should transform to sqrt(pi/2*w) but
    due to the sinularity at x=0 it does not converge well even using ng=100.



    """


    si = np.atleast_1d(s)

    xk_, wk = gauss_legendre_abscissae_and_weights(ng)

    # integration intervals

    zeros = np.zeros(m + 1, dtype=float)
    zeros[1:] = (2 * np.arange(m) + 1) * np.pi/2

    aj = zeros[0:-1]
    bj = zeros[1:]

    #dims of array:
    # 0 or i dim is each transform coordinate
    # 1 or j dim is each integration interval
    # 2 or k dim is each integration point

    # 2 dim will be summed to get integral of each interval
    # 1 dim will be summed or shanks'ed to give transform at each coord
    #

    si = si[:, np.newaxis, np.newaxis]

    aj = aj[np.newaxis, :, np.newaxis]
    bj = bj[np.newaxis, :, np.newaxis]

    xk_ = xk_[np.newaxis, np.newaxis, :]

    wk = wk[np.newaxis, np.newaxis, :]


    aij = aj / si
    bij = bj / si

    bma = (bij - aij) / 2 # b minus a
    bpa = (aij + bij) /2 # b plus a


    xijk = bma * xk_ + bpa # xj_ are in [-1, 1] so need to transform to [a, b]

    fijk = f(xijk, *args)
    fijk *= np.cos(si * xijk)
#    fijk *= xijk

    igral = bma[:,:,0] * np.sum(fijk * wk, axis=2)



    if shanks_ind is None:
        return igral.sum(axis=1)
    else:
        #extrapolate
        igral.cumsum(axis=1 , out=igral)
        return shanks(igral, shanks_ind)

def v2dcosine_transform(f, s1, s2, args=(), m=20, ng=20, shanks_ind=None):
    """Cosine transform of f(x) at transform variable s

    Parameters
    ----------
    f : function or method
        function to apply cosine trasnform to.  f is called with
        f(x, y, *args)
    s1, s2 : 1d array
        transformation variables. a grid of points will be made
    args : tuple, optional
        arguments to pass to f
    m : int, optional
        number of segments to break the integration interval into.  Each
        segment will be between the zeros of the cos function, default=20
    ng : [2-20, 32, 64, 100], optional
        number of gauss points to use in integration.
    shanks_ind : int, optional
        Start position of intervals to start shanks extrapolatoin.
        default=None i.e. no extrapolation.
        Be careful when using shanks extrapolation; make sure you only begin
        to use it after the intgrand is well behaved.

    Returns
    -------
    f : 1d array of float
        value of transform at s

    Notes
    -----
    Careful with singularities.  Because there is no way to increase the
    integration points at a particular sport the infinite behaviur may not be
    captured well. For example x**-0.5 should transform to sqrt(pi/2*w) but
    due to the sinularity at x=0 it does not converge well even using ng=100.



    """

    #dims of array:
    # 0 or i dim is each of the 1st transform coordinate s1
    # 1 or j dim is each integration inteval corresponding to s1, a1, b1
    # 2 or k dim is each gauss point in interval a1, b1
    # 3 or l dim is each of the 2nd transform coordinate s2
    # 4 or m dim is each integration interval corresponding to s2, a2, b2
    # 5 or n dim is each gauss point in interval a2, b2

    # 5 dim will be summed to get integral of each interval a2, b2
    # 4 dim will be summed to get shanks'ed to give transform at each s2 coord
    # 2 dim will be summed to get integral of each interval a1, b1
    # 1 dim will be summed to get shanks'ed to give transform at each s1 coord


    si = np.atleast_1d(s1)
    sl = np.atleast_1d(s2)

    xk_, wk = gauss_legendre_abscissae_and_weights(ng)
    xn_, wn = xk_, wk

    # integration intervals
    zeros = np.zeros(m + 1, dtype=float)
    zeros[1:] = (2 * np.arange(m) + 1) * np.pi/2

    aj = am = zeros[0:-1]
    bj = bm = zeros[1:]

    si = si[:, None, None, None, None, None]

    aj = aj[None, :, None, None, None, None]
    bj = bj[None, :, None, None, None, None]

    xk_ = xk_[None, None, :, None, None, None]
    wk = wk[None, None, :, None, None, None]

    aij = aj / si
    bij = bj / si

    bmaij = (bij - aij) / 2 # b minus a
    bpaij = (aij + bij) /2 # b plus a


    sl = sl[ None, None, None, :, None, None]
    am = am[None, None, None, None, :, None]
    bm = bm[None, None, None, None, :, None]

    xn_ = xn_[None, None, None, None, None, :]
    wn = wn[None, None, None, None, None, :]

    alm = am / sl
    blm = bm / sl

    bmalm = (blm - alm) / 2 # b minus a
    bpalm = (alm + blm) /2 # b plus a


    xijk = bmaij * xk_ + bpaij # xj_ are in [-1, 1] so need to transform to [a, b]

    xlmn = bmalm * xn_ + bpalm

    fijklmn=f(xijk, xlmn, *args)
    fijklmn*=np.cos(si*xijk)
    fijklmn*=np.cos(sl*xlmn)

    fijklmn*=wk
    fijklmn*=wn

    fijklmn*=bmaij#[:,:,0,:,:,:]
    fijklmn*=bmalm#[:,:,:,:,:,0]

    igral = np.sum(fijklmn, axis=5)

    if shanks_ind is None:
        igral = igral.sum(axis=4)
    else:
        #extrapolate
        igral.cumsum(axis=4 , out=igral)
        igral= np.apply_along_axis(shanks, 4, igral, shanks_ind)

    igral = np.sum(igral, axis=2)

    if shanks_ind is None:
        igral = igral.sum(axis=1)
    else:
        #extrapolate
        igral.cumsum(axis=1 , out=igral)
        igral= np.apply_along_axis(shanks, 1, igral, shanks_ind)

    return igral

def cot(phi):
    return 1/np.tan(phi)

def csc(phi):
    return 1.0/np.sin(phi)

def vinv_laplace_2dcosine_transform(f, s1, s2, t, args=(),
                                    m=20, ng=20, shanks_ind=None,
                                    nlap=24, shift=0.0):
    """Cosine transform of f(x) at transform variable s

    Parameters
    ----------
    f : function or method
        function to apply cosine trasnform to.  f is called with
        f(x, y, *args)
    s1, s2, t : 1d array
        transformation variables. a grid of points will be made, s1 and s2
        are the x and y fourier directions, s3 is the t direction for the
        inverse laplace
    args : tuple, optional
        arguments to pass to f
    m : int, optional
        number of segments to break the integration interval into.  Each
        segment will be between the zeros of the cos function, default=20
    ng : [2-20, 32, 64, 100], optional
        number of gauss points to use in integration.
    shanks_ind : int, optional
        Start position of intervals to start shanks extrapolatoin.
        default=None i.e. no extrapolation.
        Be careful when using shanks extrapolation; make sure you only begin
        to use it after the intgrand is well behaved.
    nlap : even int, optional
        number of integration points for talbot method inverse laplace.
        if n is odd it will be rounded up to
        nearest even number default n = 24
    shift : float
        For inverse laplace transorm, shift contour to the right in case
        there is a pole on the positive real axis. default shift=0.0

    Returns
    -------
    f : ndarray of shape (len(s1), len(s2), len(s3)
       value of transform at s1, s2, s3

    Notes
    -----
    Careful with singularities in the fourier transform.  Because there is
    no way to increase the
    integration points at a particular sport the infinite behaviur may not be
    captured well. For example x**-0.5 should transform to sqrt(pi/2*w) but
    due to the sinularity at x=0 it does not converge well even using ng=100.



    """

    #dims of array:
    # 0 or i dim is each of the 1st fourier transform coordinate s1
    # 1 or j dim is each integration inteval corresponding to s1, a1, b1
    # 2 or k dim is each gauss point in interval a1, b1
    # 3 or l dim is each of the 2nd fourier transform coordinate s2
    # 4 or m dim is each integration interval corresponding to s2, a2, b2
    # 5 or n dim is each gauss point in interval a2, b2
    # 6 or o dim is each of the inv laplace transform coordinate t
    # 7 or p dim is the theta corresponding to t

    # 7 dim will be summed to get inv laplace for each t
    # 5 dim will be summed to get integral of each interval a2, b2
    # 4 dim will be summed to get shanks'ed to give transform at each s2 coord
    # 2 dim will be summed to get integral of each interval a1, b1
    # 1 dim will be summed to get shanks'ed to give transform at each s1 coord


    to = np.atleast_1d(t)
    nlap = nlap + nlap % 2
    if np.any(to==0):
            raise ValueError('Inverse transform can not be calculated for t=0')
    #   Initiate the inv laplace stepsize
    h = 2*np.pi/nlap;
    theta = (-np.pi + (np.arange(nlap) + 1./2)*h)

    to = to[None, None, None, None, None, :, None]
    thetap = theta[None, None, None, None, None, None, :]
    zop = shift + nlap/to*(0.5017*thetap*cot(0.6407*thetap) - 0.6122 + 0.2645j*thetap)
    dzop = nlap/to*(-0.5017*0.6407*thetap*(csc(0.6407*thetap)**2)+0.5017*cot(0.6407*thetap)+0.2645j)




#    theta = (-np.pi + (np.arange(self.n)+1./2)*h)[:, np.newaxis]
#    z = self.shift + self.n/t*(0.5017*theta*cot(0.6407*theta) - 0.6122 + 0.2645j*theta)
#    dz = self.n/t*(-0.5017*0.6407*theta*(csc(0.6407*theta)**2)+0.5017*cot(0.6407*theta)+0.2645j)
#    inv_laplace = (np.exp(z * t) * self.f(z, *args) * dz).sum(axis=0)
#    inv_laplace *= h / (2j * np.pi)







    si = np.atleast_1d(s1)
    sl = np.atleast_1d(s2)




    xk_, wk = gauss_legendre_abscissae_and_weights(ng)
    xn_, wn = xk_, wk

    # integration intervals
    zeros = np.zeros(m + 1, dtype=float)
    zeros[1:] = (2 * np.arange(m) + 1) * np.pi/2

    aj = am = zeros[0:-1]
    bj = bm = zeros[1:]

    si = si[:, None, None, None, None, None, None, None]

    aj = aj[None, :, None, None, None, None, None, None]
    bj = bj[None, :, None, None, None, None, None, None]

    xk_ = xk_[None, None, :, None, None, None, None, None]
    wk = wk[None, None, :, None, None, None, None, None]

    aij = aj / si
    bij = bj / si

    bmaij = (bij - aij) / 2 # b minus a
    bpaij = (aij + bij) /2 # b plus a


    sl = sl[None, None, None, :, None, None, None, None ]
    am = am[None, None, None, None, :, None, None, None]
    bm = bm[None, None, None, None, :, None, None, None]

    xn_ = xn_[None, None, None, None, None, :, None, None]
    wn = wn[None, None, None, None, None, :, None, None]

    alm = am / sl
    blm = bm / sl

    bmalm = (blm - alm) / 2 # b minus a
    bpalm = (alm + blm) /2 # b plus a


    xijk = bmaij * xk_ + bpaij # xj_ are in [-1, 1] so need to transform to [a, b]

    xlmn = bmalm * xn_ + bpalm

    fijklmnop=f(xijk, xlmn, zop, *args)
    fijklmnop*=np.cos(si*xijk)
    fijklmnop*=np.cos(sl*xlmn)

    fijklmnop*=wk
    fijklmnop*=wn

    fijklmnop*=bmaij#[:,:,0,:,:,:]
    fijklmnop*=bmalm#[:,:,:,:,:,0]

    fijklmnop*=np.exp(zop * to)
    fijklmnop*=dzop
    fijklmnop*=h / (2j * np.pi)

    igral = np.sum(fijklmnop,axis=7)
    igral = np.real(igral)
    igral = np.sum(igral, axis=5)

#    igral = np.sum(fijklmn, axis=5)

    if shanks_ind is None:
        igral = igral.sum(axis=4)
    else:
        #extrapolate
        igral.cumsum(axis=4 , out=igral)
        igral= np.apply_along_axis(shanks, 4, igral, shanks_ind)

    igral = np.sum(igral, axis=2)

    if shanks_ind is None:
        igral = igral.sum(axis=1)
    else:
        #extrapolate
        igral.cumsum(axis=1 , out=igral)
        igral= np.apply_along_axis(shanks, 1, igral, shanks_ind)

    return igral








# for cosine transform pairs see
# http://www.efunda.com/math/Fourier_transform/table.cfm?TransName=Fc

def cosine1(x, a):
    """exp(- a * x"""
    return np.exp(-a * x)
def cosine1_(x, a):
    """a / (a**2 + x**2)"""
    return  a / (a**2 + x**2)

def cosine2(x, a):
    """x**(-0.5)"""
    return x**(-0.5)
def cosine2_(x, a):
    """np.sqrt(np.pi/(2*x))"""
    return  np.sqrt(np.pi / (2*x))

def cosine3(x, y, a, b):
    return np.exp(-a * x) * np.exp(-b * y)
def cosine3_(x, y, a, b):
    return a / (a**2 + x**2) * b / (b**2 + y**2)


class test_vcosine_transform(unittest.TestCase):
    """tests for vcosine_transform"""

    def test_cosine1(self):
        s = np.array([0.5, 1, 1.6])
        args=(1.2,)

        shanks_ind=-5
        assert_allclose(vcosine_transform(cosine1, s, args,
                                          shanks_ind=shanks_ind),
                        cosine1_(s, *args), atol=1e-8)
#    def test_cosine2(self):
#        s = np.array([0.5, 1, 1.6])
#        args=(1.2,)
#
#        shanks_ind=-5
#        assert_allclose(vcosine_transform(cosine2, s, args, m=44, ng=100,
#                                          shanks_ind=shanks_ind),
#                        cosine2_(s, *args), atol=1e-8)



class test_v2dcosine_transform(unittest.TestCase):
    """tests for vcosine_transform"""

    def test_cosine3(self):
        s1 = np.array([0.5, 1, 1.6])
        s2 = np.array([0.6,1,2])
        args=(1.2, 1.4)

        shanks_ind=-5#None
        assert_allclose(v2dcosine_transform(cosine3, s1, s2, args,
                                          shanks_ind=shanks_ind),
                        cosine3_(s1[:, np.newaxis], s2[np.newaxis,:], *args), atol=1e-8)
#    def test_cosine2(self):
#        s = np.array([0.5, 1, 1.6])
#        args=(1.2,)
#        order=0
#        shanks_ind=-5
#        assert_allclose(vcosine_transform(cosine2, s, args, order=order,
#                                          shanks_ind=shanks_ind),
#                        cosine2_(s, *args), atol=1e-8)
#
#    def test_cosine4(self):
#        s = np.array([0.5, 1, 1.6])
#        order=3
#        args=(1.2, order)
#        shanks_ind=-5
#        assert_allclose(vcosine_transform(cosine4, s, args, order=order,
#                                          shanks_ind=shanks_ind),
#                        cosine4_(s, *args), atol=1e-8)
#    def test_cosine5(self):
#        s = np.array([0.5, 1, 1.6])
#        order=3
#        args=(1.2, order)
#        shanks_ind=-5
#        assert_allclose(vcosine_transform(cosine5, s, args, order=order,
#                                          shanks_ind=shanks_ind),
#                        cosine5_(s, *args), atol=1e-8)


def ilapcosine3(x, y, t, a, b, c):
    """np.exp(-a * x) * np.exp(-b * y) / (1+t+c)"""
    f= np.exp(-a * x)
    f*=np.exp(-b * y)
    f/=(1+t+c)
    return f

def ilapcosine3_(x, y, t, a, b, c):
    return a / (a**2 + x**2) * b / (b**2 + y**2) *np.exp(-(c + 1) * t)

class test_vinv_laplace_2dcosine_transform(unittest.TestCase):
    """tests for vinv_laplace_2dcosine_transform"""

    def test_cosine3(self):
        s1 = np.array([0.5, 1, 1.6])
        s2 = np.array([0.6,1,2])
        t = np.array([0.2, 2, 4,])
        args=(1.2, 1.4, 0.8)

        shanks_ind=-5#None
        nlap=24
        shift = 0.0
        assert_allclose(vinv_laplace_2dcosine_transform(
                            ilapcosine3, s1, s2, t,
                            args, shanks_ind=shanks_ind,
                            nlap=nlap, shift=shift),
                        ilapcosine3_(s1[:, None, None],
                                 s2[None,:, None],
                                 t[None, None, :], *args), atol=1e-8)


def scratch():
    """
    """

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])