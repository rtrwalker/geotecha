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
"""routines related to hankel_transforms"""


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
from numpy.polynomial.polynomial import Polynomial

from geotecha.math.quadrature import gl_quad
from geotecha.math.quadrature import gk_quad
from geotecha.math.quadrature import gauss_kronrod_abscissae_and_weights
from geotecha.math.quadrature import gauss_legendre_abscissae_and_weights
from geotecha.math.quadrature import shanks_table
from geotecha.math.quadrature import shanks

class HankelTransform(object):
    """Hankel transform of integer order

    Applies Gauss-Kronrod quadrature between zeros of the integrand. Then uses
    shanks extrapolation.

    Parameters
    ----------
    func : function or method
        function to apply hankel trasnform to.  `func` is called with
        func(r, *args). func should be vectorised in r.
    args : tuple, optional
        arguments to pass to f
    order : integer, optional
        order of hankel transform, default=0
    m : int, optional
        number of segments to break the integration interval into.  Each
        segment will be between the zeros of the bessel function or order=
        `order' divide by r, default=20
    points : list/array of float, optional
        points in addition to those defined by m at which to split the
        integral.  Use this if `f` itself oscillates or there are
        discontinuities.  Points will only be included that are less than
        the `m` zeros mentined above. default = None i.e. no extra points.
        Basically the function is never evaluated at any the points, rather
        they form the boundary four qauss quadrature.
    ng : [7,10,15,20,25,30], optional
        number of gauss points to use in integration after first zero.
        default=10. Number of Kronrod points will automatically by 2 * ng + 1
    ng0 : [7,10,15,20,25,30], optional
        number of gauss points to use in integrating between 0 and first zero
        default=20
    shanks_ind : int, optional
        Start position of intervals (not including the first interval) from
        which to begin shanks transformationdefault=None i.e. no extrapolation
        The first interval will never be included.
        Be careful when using shanks extrapolation; make sure you only begin
        to use it after the intgrand is well behaved.  use the plot_integrand
        method to check your integrand.

    Returns
    -------
    f : float
        value of transform at r

    Notes
    -----
    The Hankel Transform of order :math:`\\nu` is given by:

    .. math:: F_\\nu(s)=\mathcal{H}_\\nu\\{f(r)\\} =
                \\int_0^{\\infty}rf(r)J_\\nu(sr)\\,\\mathrm{d}r

    provided :math:`\\nu\\gt1/2` the inverse hankel transform is the same as
    the normal transform:

    .. math:: f(r)=\mathcal{H}_{\\nu}^{-1}\\{F_{\\nu}(s)\\} =
                \\int_0^{\\infty}sF_\\nu(s)J_\\nu(sr)\\,\\mathrm{d}s

    References
    ----------
    .. [1] Piessens, Robert. 2000. 'Chapter 9 The Hankel Transform'. In The
           Transforms and Applications Handbook, edited by Alexander
           D. Poularikas, 2nd edition. Boca Raton, USA: CRC Press.
    """


    def __init__(self, func, args=(), order=0, m=20, points=None, ng=10, ng0=20,
                 shanks_ind=None):


        self.func = func
        self.args = args
        self.order = order
        self.m = m
        if points is None:
            self.points = None
        else:
            self.points = np.atleast_1d(points)

        self.ng = ng
        self.ng0 = ng0
        self.shanks_ind = shanks_ind

        self.zeros_of_jn()

    def zeros_of_jn(self):
        """Roots of Jn for determining integration intervals (0 prepended"""

        self.jn_0s = np.zeros(self.m + 1, dtype=float)
        self.jn_0s[1:] = jn_zeros(self.order, self.m)
        return
    def _integrand(self, s, r, *args):
        """
        Parameters
        ----------
        s : float
            transform variable.
        r : float
            independent variable. i.e. integrate w.r.t. to r
        """

        return self.func(r, *args) * r * jn(self.order, s * r)

    def __call__(self, s, a=0, b=np.inf):
        """transform f(r, *args) at s

        Parameters
        ----------
        s : scalar
            transform variable, i.e. coordinate to evaluate transform at
        a, b : float, optional
            limits of integration.  default a=0, b=np.inf.  A hankel transform
            is usually from 0 to infinity.  However, if you have a function
            that is defined on [a,b] and zero elsewhere then you can restrict
            integration to that interval (shanks_ind will be ignored if
            b!=np.inf).

        Returns
        -------
        F : float
            transformed functin evaluated at s
        err_est : float
            error estimate.  For each interval (i.e. between bessel zeros
            and any specified points) sum up 200*abs(G-K)**1.5.  The error is
            calculated before any shanks extrapolation so the error is just a
            measure of the difference between the coarse gauss quadrature and
            the finer Kronrod quadrature.


        """


        integrand = functools.partial(self._integrand, s)

        zeros = self.jn_0s / s

        if not self.points is None:
            #zeros becomes both zeros and interesting points/singularities
            zeros = np.unique(list(zeros) +
                    list(self.points[(self.points < zeros[-1])
                                     & (self.points>0)]))

        if (a!=0) or (b!=np.inf):
            zeros = np.unique(zeros.clip(a, b))

        #1st segment
        igral0, err_est0 = gk_quad(integrand, zeros[0], zeros[1],
                                   self.args, self.ng0)
        #remaining segments
        if len(zeros)>2:
            igralm, err_estm = gk_quad(integrand, zeros[1:-1], zeros[2:],
                                       self.args, self.ng)
        else:
            return igral0[0], err_est0[0]


        if (self.shanks_ind is None) or (b!=np.inf):
            igral = igral0 + np.sum(igralm)
        else:
            igralm.cumsum(out=igralm)
            igral = igral0 + shanks(igralm, self.shanks_ind)

        err_est = (200*np.abs(err_est0))**1.5 + np.sum((200*np.abs(err_estm))**1.5)
        return igral[0], err_est[0]



    def plot_integrand(self, s, npts = 1000):
        """plot the integrand

        Parameters
        ----------
        s : float
            transform variable, i.e. point to evaluate transform at
        npts : int, optional
            number of points to plot. default= 1000

        Returns
        -------
        fig : matplotlib.Figure
            use plt.show to plot

        Notes
        -----
        use this to check if your parameters are appropriate

        """

        integrand = functools.partial(self._integrand, s)

        zeros=self.jn_0s / s

        if not self.points is None:
            zeros = np.unique(list(zeros) +
                    list(self.points[(self.points < zeros[-1])
                                     & (self.points>0)]))

        x = np.linspace(0.001, zeros[-1], npts)
#        y = self.f(x, *self.args) * x * jn(self.order, x * r)
        y = integrand(x, *self.args)

        fig = plt.figure()
        ax = fig.add_subplot('111')
        ax.plot(x,y, '-')
        ax.plot(zeros, np.zeros_like(zeros), 'ro', ms=5)
#        ax.plot(x, x*jn(self.order, r*x), label='rJn(r*s)')
        ax.set_title('$rf(r)J_\\nu(sr)\\,dr)$ at s={:.3g}, f(r) is {}(r, *{})'.format(s, self.func.__name__, self.args))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid()
        return fig


def vhankel_transform(f, r, args=(), order=0, m=20, ng=20, shanks_ind=None):
    """Hankel transform of f(r)

    Parameters
    ----------
    f : function or method
        function to apply hankel trasnform to.  f is called with
        f(s, *args)
    r : 1d array
        coordinate(s) to evaluate transform at
    args : tuple, optional
        arguments to pass to f
    order : integer, optional
        order of hankel transform, default=0
    m : int, optional
        number of segments to break the integration interval into.  Each
        segment will be between the zeros of the bessel function, default=20
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
        value of transform at r

    Notes
    -----
    The Hankel Transform of order :math:`\\nu` is given by:

    .. math:: F_\\nu(s)=\mathcal{H}_\\nu\\{f(r)\\} =
                \\int_0^{\\infty}rf(r)J_\\nu(sr)\\,\\mathrm{d}r

    provided :math:`\\nu\\gt1/2` the inverse hankel transform is the same as
    the normal transform:

    .. math:: f(r)=\mathcal{H}_{\\nu}^{-1}\\{F_{\\nu}(s)\\} =
                \\int_0^{\\infty}sF_\\nu(s)J_\\nu(sr)\\,\\mathrm{d}s


    Note that because this implementation does not allow for input of
    extra point to break up the integration inteval, there is no way to
    account for singularities and other oscillations.  If you need this control
    then see the HankelTransorm class which is not vectorized but provides a
    few more options.


    References
    ----------
    .. [1] Piessens, Robert. 2000. 'Chapter 9 The Hankel Transform'. In The
           Transforms and Applications Handbook, edited by Alexander
           D. Poularikas, 2nd edition. Boca Raton, USA: CRC Press.


    """


    ri = np.atleast_1d(r)

    xk_, wk = gauss_legendre_abscissae_and_weights(ng)

    # integration intervals
    zeros = np.zeros(m + 1, dtype=float)
    zeros[1:] = jn_zeros(order, m)

    aj = zeros[0:-1]
    bj = zeros[1:]


    ri = ri[:, np.newaxis, np.newaxis]

    aj = aj[np.newaxis, :, np.newaxis]
    bj = bj[np.newaxis, :, np.newaxis]

    xk_ = xk_[np.newaxis, np.newaxis, :]

    wk = wk[np.newaxis, np.newaxis, :]


    aij = aj / ri
    bij = bj / ri

    bma = (bij - aij) / 2 # b minus a
    bpa = (aij + bij) /2 # b plus a


    xijk = bma * xk_ + bpa # xj_ are in [-1, 1] so need to transform to [a, b]

    fijk = f(xijk, *args)
    fijk *= jn(order, ri * xijk)
    fijk *= xijk

    igral = bma[:,:,0] * np.sum(fijk * wk, axis=2)



    if shanks_ind is None:
        return igral.sum(axis=1)
    else:
        #extrapolate
        igral.cumsum(axis=1 , out=igral)
        return shanks(igral, shanks_ind)


#Hankel transform pairs
#zero order
def hankel1(s, a):
    """a/(s**2 + a**2)**1.5"""
    #H(hankel1)=exp(-a* r)
    return a/(s**2 + a**2)**1.5
def hankel1_(r, a):
    """exp(-a*r)"""
    return np.exp(-a*r)

def hankel2(s, *args):
    "1/s"
    #H(hankel2)=1/r
    return 1/s
def hankel2_(r, *args):
    "1/r"
    return 1/r

def hankel3(s,a):
    "1/s*jn(0,a/s)"
    #H(hankel3)=1/rJ0(2*(a*r)**0.5)
    return 1/s*jn(0,a/s)
def hankel3_(r, a):
    "1/rJ0(2*(a*r)**0.5)"
    return 1/r*jn(0,(2*(a*r)**0.5))

#integer order
def hankel4(s, a, v=0):
    """(sqrt(s**2+a**2)-a)**v/(s**v*sqrt(s**2+a**2))"""
    #H(hankel4)=exp(-a*r)/r
    return (np.sqrt(s**2 + a**2) - a)**v/(s**v*np.sqrt(s**2+a**2))

def hankel4_(r, a, *args):
    """exp(-a*r)/r"""
    return np.exp(-a*r)/r

def hankel5(s, a, v=0):
    """s**v/(2*a**2)**(v+1)*exp(-s**2/(4*a**2))"""
    #H(hankel5)=exp(-a**2*r**2)*r**v
    return s**v/(2*a**2)**(v+1)*np.exp(-s**2/(4*a**2))
def hankel5_(r, a, v=0):
    """exp(-a**2*r**2)*r**v"""
    return np.exp(-a**2*r**2)*r**v



def scratch():
    """
    """

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])