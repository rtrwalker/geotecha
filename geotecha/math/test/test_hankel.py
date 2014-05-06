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


from geotecha.math.hankel import vhankel_transform
from geotecha.math.hankel import HankelTransform



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
    for func, func_ in [(hankel4, hankel4_), (hankel5, hankel5_)]:
        for order in order_:
            for s in s_:
                for a in a_:
                    args=(a, order)
                    yield (check_HankelTransform,
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
    for func, func_ in [(hankel1, hankel1_), (hankel2, hankel2_), (hankel3, hankel3_), (hankel4, hankel4_)]:
        for s in s_:
            for a in a_:
                args=(a,)
                yield (check_HankelTransform,
                       s, (func, func.__doc__), (func_, func_.__doc__),
                        args, order, m, ng, ng0, shanks_ind)



def check_HankelTransform(s, (func, funcdoc), (func_, func_doc), args, order, m,
                          ng, ng0, shanks_ind):
    """check if a HankelTransform gives it's analytical solution

    Parameters
    ----------
    s : float
        transform variable
    func, funcdoc: function and functions doc
        function to transform
    func_, func_doc : function and functin doc
        analytical transform of `func`
    other: see HankelTransform
    """

    if func==hankel3:
        points= args[0] / jn_zeros(0, 70)
        atol=1e-4
    else:
        points=None
        atol=1e-5

    h = HankelTransform(func, args, order, m, points, ng, ng0, shanks_ind)
    assert_allclose(h(s)[0], func_(s, *args), atol=atol)




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





class test_vhankel_transform(unittest.TestCase):
    """tests for vhankel_transform"""

    def test_hankel1(self):
        s = np.array([0.5, 1, 1.6])
        args=(1.2,)
        order=0
        shanks_ind=-5
        assert_allclose(vhankel_transform(hankel1, s, args,order=order,
                                          shanks_ind=shanks_ind),
                        hankel1_(s, *args), atol=1e-8)

    def test_hankel2(self):
        s = np.array([0.5, 1, 1.6])
        args=(1.2,)
        order=0
        shanks_ind=-5
        assert_allclose(vhankel_transform(hankel2, s, args, order=order,
                                          shanks_ind=shanks_ind),
                        hankel2_(s, *args), atol=1e-8)

    def test_hankel4(self):
        s = np.array([0.5, 1, 1.6])
        order=3
        args=(1.2, order)
        shanks_ind=-5
        assert_allclose(vhankel_transform(hankel4, s, args, order=order,
                                          shanks_ind=shanks_ind),
                        hankel4_(s, *args), atol=1e-8)
    def test_hankel5(self):
        s = np.array([0.5, 1, 1.6])
        order=3
        args=(1.2, order)
        shanks_ind=-5
        assert_allclose(vhankel_transform(hankel5, s, args, order=order,
                                          shanks_ind=shanks_ind),
                        hankel5_(s, *args), atol=1e-8)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])