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


from geotecha.math.quadrature import gl_quad
from geotecha.math.quadrature import gk_quad
from geotecha.math.quadrature import gauss_kronrod_abscissae_and_weights
from geotecha.math.quadrature import gauss_legendre_abscissae_and_weights
from geotecha.math.quadrature import shanks_table
from geotecha.math.quadrature import shanks

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


def cosine_transform(func, w, args=()):
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

    return integrate.quad(func, 0, np.inf, args=args, weight='cos', wvar=w)

def sine_transform(func, w, args=()):
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

    return integrate.quad(func, 0, np.inf, args=args, weight='sin', wvar=w)




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
                 imag_part_even=False, imag_part_odd=False):

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
                        cosine_transform(self.func, s, self.args)]
                return igral, err
            elif self.real_part_odd:
                igral, err = [2.j * self.inv_const * self.inv_sign * v for v in
                        sine_transform(self.func, s, self.args)]
                return -igral, err

            else:
                #func is real and exhibits no symmetry.
                igral = 0
                err = 0

                # real transform result [-inf, 0]
                ig, er = [self.inv_const * v for v in
                        cosine_transform(self.mfe, s)]
                igral += ig; err += er
                # real transform result [-, inf]
                ig, er = [self.inv_const * v for v in
                        cosine_transform(self.func, s, self.args)]
                igral += ig; err += er
                # imag transform result [0, inf]
                ig, er = [1.j * self.inv_const * self.inv_sign * v for v in
                        sine_transform(self.mfo, s)]
                igral -= ig; err += er
                # imag transform result [-inf, 0]
                ig, er = [1.j * self.inv_const * self.inv_sign * v for v in
                        sine_transform(self.func, s, self.args)]
                igral -= ig; err += er
                return igral, err

        if self.func_is_imag:
            if self.imag_part_even:
                igral, err = [2.j * self.inv_const * v for v in
                        cosine_transform(self.func, s, self.args)]
                return igral, err
            elif self.imag_part_odd:
                igral, err =  [2 * self.inv_const * self.inv_sign * v for v in
                        sine_transform(self.func, s, self.args)]
                return igral, err
            else:
                #func is imaginary and ehibits non symmetry
                igral = 0
                err = 0

                # imag transform result [-inf, 0]
                ig, er = [1.j * self.inv_const * v for v in
                        cosine_transform(self.mfe, s)]
                igral += ig; err += er
                # imag transform result [0, inf]
                ig,er = [1.j * self.inv_const * v for v in
                        cosine_transform(self.func, s, self.args)]
                igral += ig; err += er
                # real transform result [-inf, 0]
                ig, er = [self.inv_const * self.inv_sign * v for v in
                        sine_transform(self.mfo, s)]
                igral -= ig; err += er
                # real transform result [0, inf]
                ig, er = [self.inv_const * self.inv_sign * v for v in
                        sine_transform(self.func, s, self.args)]
                igral -= ig; err += er

                return igral, err

        #if we have reached here then func is complex
        #use real and imag parts of func
        igral = 0
        err = 0
        if self.real_part_even:
            ig, er = [2 * self.inv_const * v for v in
                        cosine_transform(self.real_func, s)]
            igral += ig; err += er
        elif self.real_part_odd:
            # 2 * I(real_func * sin(s*x), 0, +inf)
            ig, er = [2.j * self.inv_const * self.inv_sign * v for v in
                        sine_transform(self.real_func, s)]
            igral -= ig; err += er
        else:
            #real part of function exhibits no symmetry
            # real transform result [-inf, 0]
            ig, er = [self.inv_const * v for v in
                        cosine_transform(self.mfre, s)]
            igral += ig; err += er
            # real transform result [0, inf]
            ig, er = [self.inv_const * v for v in
                        cosine_transform(self.real_func, s)]
            igral += ig; err += er
            # imag transform result [-inf, 0]
            ig, er = [1.j * self.inv_const * self.inv_sign * v for v in
                        sine_transform(self.mfro, s)]
            igral -= ig; err += er
            # imag transform result [0, inf]
            ig, er = [1.j * self.inv_const * self.inv_sign * v for v in
                        sine_transform(self.real_func, s)]
            igral -= ig; err += er


        if self.imag_part_even:
            igral, err = [2.j * self.inv_const * v for v in
                        cosine_transform(self.func, s, self.args)]
            igral += ig; err += er
        elif self.real_part_odd:
            igral, err = [2 * self.inv_const * self.inv_sign * v for v in
                        sine_transform(self.func, s, self.args)]
            igral += ig; err += er
        else:
            #imag part of function exhibits no symmetry
            # imag transform result [-inf, 0]
            ig, er = [1.j * self.inv_const * v for v in
                        cosine_transform(self.mfie, s)]
            igral += ig; err += er
            # imag transform result [0, inf]
            ig, er = [1.j * self.inv_const * v for v in
                        cosine_transform(self.imag_func, s)]
            igral += ig; err += er
            # real transform result [-inf, 0]
            ig, er = [self.inv_const * self.inv_sign * v for v in
                        sine_transform(self.mfio, s)]
            igral += ig; err += er
            # real transform result [0, inf]
            ig, er = [self.inv_const * self.inv_sign * v for v in
                        sine_transform(self.imag_func, s)]
            igral += ig; err += er
        return igral, err


def scratch():
    """
    """

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])