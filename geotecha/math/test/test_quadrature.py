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
"""Some test routines for the laplace module

"""
from __future__ import division, print_function

from nose import with_setup
from nose.tools.trivial import assert_almost_equal
from nose.tools.trivial import assert_raises
from nose.tools.trivial import ok_
from numpy.testing import assert_allclose

import numpy as np
import unittest

from scipy.special import jn_zeros
from scipy.special import jn

from geotecha.math.quadrature import gl_quad
from geotecha.math.quadrature import gk_quad
from geotecha.math.quadrature import vhankel_transform
from geotecha.math.quadrature import HankelTransform
from geotecha.math.quadrature import shanks
from geotecha.math.quadrature import shanks_table





#zero order
def f1(s, a):
    """a/(s**2 + a**2)**1.5"""
    #H(f1)=exp(-a* r)
    return a/(s**2 + a**2)**1.5
def f1_(r, a):
    """exp(-a*r)"""
    return np.exp(-a*r)

def f2(s, *args):
    "1/s"
    #H(f2)=1/r
    return 1/s
def f2_(r, *args):
    "1/r"
    return 1/r

def f3(s,a):
    "1/s*jn(0,a/s)"
    #H(f3)=1/rJ0(2*(a*r)**0.5)
    return 1/s*jn(0,a/s)
def f3_(r, a):
    "1/rJ0(2*(a*r)**0.5)"
    return 1/r*jn(0,(2*(a*r)**0.5))

#integer order
def f4(s, a, v=0):
    """(sqrt(s**2+a**2)-a)**v/(s**v*sqrt(s**2+a**2))"""
    #H(f4)=exp(-a*r)/r
    return (np.sqrt(s**2 + a**2) - a)**v/(s**v*np.sqrt(s**2+a**2))

def f4_(r, a, *args):
    """exp(-a*r)/r"""
    return np.exp(-a*r)/r

def f5(s, a, v=0):
    """s**v/(2*a**2)**(v+1)*exp(-s**2/(4*a**2))"""
    #H(f5)=exp(-a**2*r**2)*r**v
    return s**v/(2*a**2)**(v+1)*np.exp(-s**2/(4*a**2))
def f5_(r, a, v=0):
    """exp(-a**2*r**2)*r**v"""
    return np.exp(-a**2*r**2)*r**v


def test_gl_quad_polynomials():
    """tests for gl_quad exact polynomial"""
    for n in [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20, 32, 64, 100]:
        yield check_gl_quad, n

def check_gl_quad(n):
    """check that gl_quad integrates polynomial of degree 2*n-1 'exactly'

    This test is not rigorous because the ploynmials formed can be fairly well
    approximated using lower order schemes"""

    a, b = (-0.9, 1.3)
    coeff = np.arange(1, 2*n+1)
    coeff[::2]  =coeff[::2]*-1.0
    f = np.poly1d(coeff)
    F = f.integ()
    exact = F(b) - F(a)

    assert_allclose(gl_quad(f,a,b, n=n), exact, rtol=1e-14,atol=0)

def test_gk_quad_polynomials():
    """tests for gl_quad exact polynomial"""
    for n in [7,10,15,20,25,30]:
        yield check_gk_quad, n

def check_gk_quad(n):
    """check that gk_quad integrates polynomial of degree 3*n+1 (even) and
    3*n+2 (odd)exactly

    This test is not rigorous because the ploynmials formed can be fairly well
    approximated using lower order schemes
    """
    a, b = (-0.9, 1.3)
    coeff = np.arange(1, 3*n + 1 + (n%2))
    coeff[::2]  =coeff[::2]*-1.0
    f = np.poly1d(coeff)
    F = f.integ()
    exact = F(b) - F(a)

    assert_allclose(gk_quad(f,a,b, n=n)[0], exact, rtol=1e-14,atol=0)




#points=a/jn_zeros(0,40)
#ht3 = HankelTransform(f3, args=(a,), order=0, m=m, points=points, ng=10, ng0=20,
#                 shanks_ind=45)
#fig3=ht3.plot_integrand(r, 1000)
#print(f3.__name__, f3.__doc__, f3_.__name__, f3_.__doc__)
#y1, e1 = ht3(r)
#y2 = f3_(r, a)
#print(', '.join(['{:.6g}' for a in range(4)]).format(y1, e1, y2, y2-y1))



def test_hankel_transform_order_v():
    """tests for HankelTransform class of order v (v is int)

    for a variety of functions and known analytical transforms"""

#    order=0
    m=20
    ng=10
    ng0=20
    shanks_ind=-5

    s_ = [0.1, 0.5, 1.5]
    a_ = [0.9, 1.1]
    order_ = [0, 1, 4, 20]
    for func, func_ in [(f4, f4_), (f5, f5_)]:
        for order in order_:
            for s in s_:
                for a in a_:
                    args=(a, order)
                    yield (check_HankelTransfrom,
                           s, (func, func.__doc__), (func_, func_.__doc__),
                            args, order, m, ng, ng0, shanks_ind)

def test_hankel_transform_order_0():
    """tests for HankelTransform class of order 0

    for a variety of functions and known analytical transforms"""

    a=1.5
    order=0
    m=20
    ng=10
    ng0=20
    shanks_ind=-5

    s_ = [0.1, 0.5, 1.5]
    a_ = [0.9, 1.1]
    for func, func_ in [(f1, f1_), (f2, f2_), (f3, f3_), (f4, f4_)]:
        for s in s_:
            for a in a_:
                args=(a,)
                yield (check_HankelTransfrom,
                       s, (func, func.__doc__), (func_, func_.__doc__),
                        args, order, m, ng, ng0, shanks_ind)



def check_HankelTransfrom(s, (func, funcdoc), (func_, func_doc), args, order, m,
                          ng, ng0, shanks_ind):
    """check if a HankelTransfrom gives it's analytical solution

    Parameters
    ----------
    s : float
        transfrom variable
    func, funcdoc: function and functions doc
        function to transform
    func_, func_doc : function and functin doc
        analytical transform of `func`
    other: see HankelTransform
    """

    if func==f3:
        points= args[0] / jn_zeros(0, 70)
        atol=1e-4
    else:
        points=None
        atol=1e-5

    h = HankelTransform(func, args, order, m, points, ng, ng0, shanks_ind)
    assert_allclose(h(s)[0], func_(s, *args), atol=atol)


def x(*args):
    return args[0]

print(HankelTransform(x, order=0)(0.5, b=2.0)[0]
            , 2.28414)

class test_HankelTransform(unittest.TestCase):
    """one off tests for HankelTransform class"""

    def one(self, *args):
        return 1
    def x(self, *args):
        return args[0]

    def test_b_integration_limit_f_is_x(self):
        """Integrate(r*J(0, 0.5*r)*r, {r, 0,2})"""
        #wolfram alpha: Integrate[BesselJ[0,x*0.5]*x^2, {x, 0 ,2}]
        assert_allclose(
            HankelTransform(self.x, order=0)(0.5, b=2.0)[0]
            , 2.28414, atol=1e-5)

    def test_a_b_integration_limits_f_is_x(self):
        """Integrate(r*J(0, 0.5*r)*r, {r, 0.2, 2})"""
        #wolfram alpha : Integrate[BesselJ[0,x*0.5]*x^2, {x, 1.3 ,2}]
        assert_allclose(
            HankelTransform(self.x, order=0)(0.5, a=1.3, b=2.0)[0]
            , 1.59735, atol=1e-5)


class test_shanks(unittest.TestCase):
    """tests for shanks"""

    def test_pi(self):
        """4 * sum((-1)**k * (2 * k+1)**(-1))"""
        nn = 10
        seq = np.array([4*sum((-1)**n/(2*n+1) for n in range(m)) for
                        m in range(1, nn)])

        assert_allclose(shanks(seq, -8), np.pi, atol=1e-6)

    def test_pi_max_shanks_ind(self):
        """4 * sum((-1)**k * (2 * k+1)**(-1))"""
        nn = 10
        seq = np.array([4*sum((-1)**n/(2*n+1) for n in range(m)) for
                        m in range(1, nn)])

        assert_allclose(shanks(seq, -50), np.pi, atol=1e-8)





class test_vhankel_transform(unittest.TestCase):
    """tests for vhankel_transform"""

    def test_f1(self):
        s = np.array([0.5, 1, 1.6])
        args=(1.2,)
        order=0
        shanks_ind=-5
        assert_allclose(vhankel_transform(f1, s, args,order=order,
                                          shanks_ind=shanks_ind),
                        f1_(s, *args), atol=1e-8)

    def test_f2(self):
        s = np.array([0.5, 1, 1.6])
        args=(1.2,)
        order=0
        shanks_ind=-5
        assert_allclose(vhankel_transform(f2, s, args, order=order,
                                          shanks_ind=shanks_ind),
                        f2_(s, *args), atol=1e-8)

    def test_f4(self):
        s = np.array([0.5, 1, 1.6])
        order=3
        args=(1.2, order)
        shanks_ind=-5
        assert_allclose(vhankel_transform(f4, s, args, order=order,
                                          shanks_ind=shanks_ind),
                        f4_(s, *args), atol=1e-8)
    def test_f5(self):
        s = np.array([0.5, 1, 1.6])
        order=3
        args=(1.2, order)
        shanks_ind=-5
        assert_allclose(vhankel_transform(f5, s, args, order=order,
                                          shanks_ind=shanks_ind),
                        f5_(s, *args), atol=1e-8)




if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])