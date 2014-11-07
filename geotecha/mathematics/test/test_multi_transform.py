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
"""Some test routines for the multi_transform module

"""
from __future__ import division, print_function

from nose import with_setup
from nose.tools.trivial import assert_almost_equal
from nose.tools.trivial import assert_raises
from nose.tools.trivial import ok_
from numpy.testing import assert_allclose

import numpy as np
import unittest

from geotecha.mathematics.multi_transform import ntransform

class test_ntransform(unittest.TestCase):
    """tests for ntransform"""
    #ntransform(func, transforms, transvars, args=None, opts=None):


#    def h_f_il1(self, x, y, z, a, b, c):
#        """
#        hankel: a/(x**2 + a**2)**1.5
#        fourier: exp(- b * abs(y))
#        inv laplace: 1/(1+z+c)
#        """
#        f1 = a/(x**2 + a**2)**1.5
#        f2 = np.exp(-b * abs(y))
#        f3 = 1/(1+z+c)
#        return f1*f2*f3
#
#    def h_f_il1_(self, x, y, z, a, b, c):
#        """
#        hankel: exp(-a*x)
#        fourier: 2 * b / (b**2 + y**2)
#        inv laplace: exp(-(c+1)*z)
#        """
#        f1 = np.exp(-a*x)
#        f2 = 2 * b / (b**2 + y**2)
#        f3 = np.exp(-(c + 1)*z)
#        return f1*f2*f3

    def h_il_f1(self, x, y, z, a, b, c):
        """
        hankel: a/(x**2 + a**2)**1.5
        inv laplace: 1/(1+y+b)
        fourier: exp(- c * abs(z))
        """
        f1 = a/(x**2 + a**2)**1.5
        f2 = 1/(1+y+b)
        f3 = np.exp(-c * abs(z))

        return f1*f2*f3

    def h_il_f1_(self, x, y, z, a, b, c):
        """
        hankel: exp(-a*x)
        fourier: 2 * b / (b**2 + y**2)
        inv laplace: exp(-(c+1)*z)
        """
        f1 = np.exp(-a*x)
        f2 = np.exp(-(b + 1)*y)
        f3 = 2 * c / (c**2 + z**2)
        return f1*f2*f3





    def f_f1(self, x, y, a, b):
        """
        fourier: exp(- a * abs(x))
        fourier: exp(- b * abs(y))
        """
        f1 = np.exp(-a * abs(x))
        f2 = np.exp(-b * abs(y))
        return f1*f2
    def f_f1_(self, x, y, a, b):
        """
        fourier: 2 * a / (a**2 + x**2)
        fourier: 2 * b / (b**2 + y**2)
        """
        f1 = 2 * a / (a**2 + x**2)
        f2 = 2 * b / (b**2 + y**2)
        return f1*f2

    def f_f_il1(self, x, y, z, a, b, c):
        """
        fourier: exp(- a * abs(x))
        fourier: exp(- b * abs(y))
        inv_laplace: 1/(1+z+c)
        """
        f1 = np.exp(-a * abs(x))
        f2 = np.exp(-b * abs(y))
        "L-1{1/(1+s+a)} = e^(-(a+1)*t)"
        f3 = 1/(1+z+c)
        return f1 * f2 * f3

    def f_f_il1_(self, x, y, z, a, b, c):
        """
        fourier: 2 * a / (a**2 + x**2)
        fourier: 2 * b / (b**2 + y**2)
        inv_laplace: exp(-(c+1)*z)
        """
        f1 = 2 * a / (a**2 + x**2)
        f2 = 2 * b / (b**2 + y**2)
        f3 = np.exp(-(c+1)*z)
        return f1 * f2 * f3

    def il_f_f1(self, x, y, z, a, b, c):
        """
        inv_laplace: 1/(1+x+a)
        fourier: exp(- b * abs(y))
        fourier: exp(- c * abs(z))
        """
        f1 = np.exp(-b * abs(y))
        f2 = np.exp(-c * abs(z))
        f3 = 1/(1+x+a)
        return f1 * f2 * f3

    def il_f_f1_(self, x, y, z, a, b, c):
        """
        inv_laplace: exp(-(a+1)*x)
        fourier: 2 * b / (b**2 + y**2)
        fourier: 2 * c / (c**2 + z**2)
        """
        f1 = 2 * b / (b**2 + y**2)
        f2 = 2 * c / (c**2 + z**2)
        f3 = np.exp(-(a+1)*x)
        return f1 * f2 * f3

    def h_f1(self, x, y, a, b):
        """
        hankel: a/(x**2 + a**2)**1.5
        fourier: exp(- b * abs(y))
        """
        f1 = a/(x**2 + a**2)**1.5
        f2 = np.exp(- b * np.abs(y))
        return f1*f2
    def h_f1_(self, x, y, a, b):
        """
        hankel: exp(-a*x)
        fourier: 2 * b / (b**2 + y**2)
        """
        f1 = np.exp(-a*x)
        f2 = 2 * b / (b**2 + y**2)
        return f1*f2


#    def test_h_f_il1(self):
#        a, b, c = 1.8, 2.2, 0.5
#        x, y, z = 1.1, 1.2, 1.3
#        tvar = (x, y, z)
#        args= (a, b, c)
#        transforms=['Hankel', 'Fourier', 'Laplace_inverse']
##        opts=[{'func_is_real': True, 'real_part_even':True, 'b':10}]*2
#        assert_allclose(ntransform(self.h_f_il1, transforms, tvar, args)[0],
#                        self.h_f_il1_(*(tvar+args)), atol=0)


    def test_f_f1(self):
        a, b, c = 1.8, 2.2, 0.5
        x, y, z = 1.1, 1.2, 1.3
        tvar = (x, y)
        args= (a, b)
        transforms=['Fourier', 'Fourier']
        opts=[{'func_is_real': True, 'real_part_even':True, 'b':10}]*2
        assert_allclose(ntransform(self.f_f1, transforms, tvar, args, opts)[0],
                        self.f_f1_(*(tvar+args)), atol=1e-5)
#    def test_f_f_il1(self):
#        a, b, c = 1.8, 2.2, 0.5
#        x, y, z = 1.1, 1.2, 1.3
#        tvar = (x, y, z)
#        args= (a, b, z)
#        transforms=['Fourier', 'Fourier', 'Laplace_inverse']
#        opts=[{'func_is_real': True, 'real_part_even':True, 'b':10},
#              {'func_is_real': True, 'real_part_even':True, 'b':10},
#                {'n': 42}]
#        assert_allclose(ntransform(self.f_f_il1, transforms, tvar, args, opts)[0],
#                        self.f_f_il1_(*(tvar+args)), atol=1e-5)


    def test_il_f_f1(self):
        a, b, c = 1.8, 2.2, 0.5
        x, y, z = 1.1, 1.2, 1.3
        tvar = (x, y, z)
        args= (a, b, z)
        transforms=['Laplace_inverse', 'Fourier', 'Fourier']
        opts=[{'n': 24},
            {'func_is_real': True, 'real_part_even':True, 'b':10},
              {'func_is_real': True, 'real_part_even':True, 'b':10},
                ]
        assert_allclose(ntransform(self.il_f_f1, transforms, tvar, args, opts)[0],
                        self.il_f_f1_(*(tvar+args)), atol=1e-5)

    def test_h_f1(self):
        a, b, c = 1.8, 2.2, 0.5
        x, y, z = 1.1, 1.2, 1.3
        tvar = (x, y)
        args= (a, b)
        transforms=['Hankel', 'Fourier']
        opts=[{'shanks_ind': -5},
              {'func_is_real': True, 'real_part_even':True, 'b':np.inf}]
        assert_allclose(ntransform(self.h_f1, transforms, tvar, args, opts)[0],
                        self.h_f1_(*(tvar+args)), atol=0)

    def test_h_il_f1(self):
        a, b, c = 1.8, 2.2, 0.5
        x, y, z = 1.1, 1.2, 1.3
        tvar = (x, y, z)
        args= (a, b, z)
        transforms=['Hankel', 'Laplace_inverse', 'Fourier']
        opts=[{'shanks_ind': -5},
              {'n': 24, 'vectorized':False},
              {'func_is_real': True, 'real_part_even':True, 'b':10}]
        assert_allclose(ntransform(self.h_il_f1, transforms, tvar, args, opts)[0],
                        self.h_il_f1_(*(tvar+args)), atol=1e-5)

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])