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
"""Test routines for the fourier module."""
from __future__ import division, print_function

from nose import with_setup
from nose.tools.trivial import assert_almost_equal
from nose.tools.trivial import assert_raises
from nose.tools.trivial import ok_
from numpy.testing import assert_allclose

import numpy as np
import unittest

from geotecha.mathematics.fourier import FourierTransform
from geotecha.mathematics.fourier import real_func
from geotecha.mathematics.fourier import imag_func
from geotecha.mathematics.fourier import func_mirror_for_even_weight
from geotecha.mathematics.fourier import func_mirror_for_odd_weight
from geotecha.mathematics.fourier import cosine_transform
from geotecha.mathematics.fourier import sine_transform
from geotecha.mathematics.fourier import vcosine_transform
from geotecha.mathematics.fourier import v2dcosine_transform


#fro fourier transform pairs see
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

class test_fourier_transform_object(unittest.TestCase):
    """tests for FourierTransfrom object"""

    def test_fourier1_normal_at_zero(self):
        """normal ft of exp(- a * abs(x)) == 2 * a / (a**2 + x**2)"""
        func = fourier1
        func_ = fourier1_
        a = 2.3
        args=(a,)
        s=0
        ft = FourierTransform(func, args=args,
                           inv=False, func_is_real=True,
                           real_part_even=True)
        assert_allclose(ft(s)[0], func_(s, *args), atol=0)

    def test_fourier1_normal(self):
        """normal ft of exp(- a * abs(x)) == 2 * a / (a**2 + x**2)"""
        func = fourier1
        func_ = fourier1_
        a = 2.3
        args=(a,)
        s=1.5
        ft = FourierTransform(func, args=args,
                           inv=False, func_is_real=True,
                           real_part_even=True)
        assert_allclose(ft(s)[0], func_(s, *args), atol=0)

    def test_fourier1_inverse(self):
        """inverse ft of 2 * a / (a**2 + x**2)==exp(- a * abs(x))"""
        func_ = fourier1
        func = fourier1_
        a = 2.3
        args=(a,)
        s = 1.5
        ft = FourierTransform(func, args=args,
                           inv=True, func_is_real=True,
                           real_part_even=True)
        assert_allclose(ft(s)[0], func_(s, *args), atol=0)

    def test_fourier2_normal(self):
        """normal ft of rect(t/tau)==tau * sinc(tau*W/(2*pi))"""
        #note that inverse of fourier2 will fail due to oscillations
        func = fourier2
        func_ = fourier2_
        a = 2.3
        args=(a,)
        s=1.5
        ft = FourierTransform(func, args=args,
                           inv=False, func_is_real=True)
        assert_allclose(ft(s)[0], func_(s, *args), atol=0)


    def test_fourier4_normal(self):
        """normal ft of exp(-b*t)*unit_step(t)==1/(1.j*w + b)"""
        func = fourier4
        func_ = fourier4_
        a = 2.3
        args=(a,)
        s=1.5
        ft = FourierTransform(func, args=args,
                           inv=False, func_is_real=True)
        assert_allclose(ft(s)[0], func_(s, *args), atol=0)

    def test_fourier4_inverse(self):
        """inverse ft of 1/(1.j*w + b)==exp(-b*t)*unit_step(t)"""
        func_ = fourier4
        func = fourier4_
        a = 2.3
        args=(a,)
        s = 1.5
        ft = FourierTransform(func, args=args,
                           inv=True, )
        assert_allclose(ft(s)[0], func_(s, *args), atol=0)


class test_cosine_transform(unittest.TestCase):
    """tests for cosine_transform"""

    def test_fourier1_normal_at_zero(self):
        """normal ft of exp(- a * abs(x)) == 2 * a / (a**2 + x**2)"""
        func = fourier1
        func_ = fourier1_
        a = 2.3
        args=(a,)
        s=0
        assert_allclose(cosine_transform(func, s, args)[0],
                        0.5*func_(s, *args), atol=0)
    def test_fourier1_normal(self):
        """normal ft of exp(- a * abs(x)) == 2 * a / (a**2 + x**2)"""
        func = fourier1
        func_ = fourier1_
        a = 2.3
        args=(a,)
        s=1.5
        assert_allclose(cosine_transform(func, s, args)[0],
                        0.5*func_(s, *args), atol=0)

class test_sine_transform(unittest.TestCase):
    """tests for sine_transform"""

    # for sine transformation pairs see:
    # https://www.efunda.com/math/Fourier_transform/table.cfm?TransName=Fs





    def test_sine1(self):
        """normal st of exp(-x*b) == w/(w**2+b**2)"""
        func = sine1
        func_ = sine1_
        a = 2.3
        args=(a,)
        s=1.4
        assert_allclose(sine_transform(func, s, args)[0],
                        func_(s, *args), atol=0)


class test_real_func(unittest.TestCase):
    """tests for real_func"""
    def freal(self, x):
        return 2*x
    def fimag(self, x):
        return 2.j*x
    def test_real_fn(self):
        assert_allclose(real_func(2, self.freal), 4)
    def test_imag_fn(self):
        assert_allclose(real_func(2, self.fimag), 0)
    def test_complex_fn(self):
        assert_allclose(real_func(2+2.j, self.freal), 4)

    def test_real_fn_array(self):
        assert_allclose(real_func(np.array([2,3]), self.freal), [4,6])
    def test_imag_fn_array(self):
        assert_allclose(real_func(np.array([2, 1.j]), self.fimag), [0, -2])
    def test_complex_fn_array(self):
        assert_allclose(real_func(np.array([2+2.j, 3+1.j]), self.freal), [4,6])



class test_imag_func(unittest.TestCase):
    """tests for imag_func"""
    def freal(self, x):
        return 2*x
    def fimag(self, x):
        return 2.j*x
    def test_real_fn(self):
        assert_allclose(imag_func(2, self.freal), 0)
    def test_imag_fn(self):
        assert_allclose(imag_func(2, self.fimag), 4)
    def test_complex_fn(self):
        assert_allclose(imag_func(2+2.j, self.freal), 4)

    def test_real_fn_array(self):
        assert_allclose(imag_func(np.array([2,3]), self.freal), [0,0])
    def test_imag_fn_array(self):
        assert_allclose(imag_func(np.array([2, 1.j]), self.fimag), [4, 0])
    def test_complex_fn_array(self):
        assert_allclose(imag_func(np.array([2+2.j, 3+1.j]), self.freal), [4,2])

class test_func_mirror_for_even_weight(unittest.TestCase):
    """tests for func_mirror_for_even_weight"""

    def xxx(self, x):
        return x
    def x_plus_a(self, x, a):
        return x + a
    def test_x(self):
        assert_allclose(func_mirror_for_even_weight(2,self.xxx), -2)

    def test_x_plus_a(self):
        assert_allclose(func_mirror_for_even_weight(np.array([2,4]),
                                                    self.x_plus_a, 8), [6, 4])

class test_func_mirror_for_odd_weight(unittest.TestCase):
    """tests for func_mirror_for_even_weight"""

    def xxx(self, x):
        return x
    def x_plus_a(self, x, a):
        return x + a
    def test_x(self):
        assert_allclose(func_mirror_for_odd_weight(2, self.xxx), 2)

    def test_x_plus_a(self):
        assert_allclose(func_mirror_for_odd_weight(np.array([2,4]),
                                                    self.x_plus_a, 8),
                                                    [-6, -4])


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
#
#        # This shows the difficulty of doing the cosine transform when there
#        # are singulatities.  The test fails for atol=1e-8.
#        s = np.array([0.5, 1, 1.6])
#        args=(1.2,)
#
#        shanks_ind=-5
#        assert_allclose(vcosine_transform(cosine2, s, args, m=44, ng=100,
#                                          shanks_ind=shanks_ind),
#                        cosine2_(s, *args), atol=1e-8)
#

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







if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])