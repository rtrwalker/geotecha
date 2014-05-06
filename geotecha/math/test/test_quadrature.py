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

from geotecha.math.quadrature import gl_quad
from geotecha.math.quadrature import gk_quad
from geotecha.math.quadrature import shanks
from geotecha.math.quadrature import shanks_table




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



if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])