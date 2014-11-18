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
"""Testing routines for the mp_laplace module."""
from __future__ import division, print_function

from nose import with_setup
from nose.tools.trivial import assert_almost_equal
from nose.tools.trivial import assert_raises
from nose.tools.trivial import ok_
from numpy.testing import assert_allclose

import numpy as np
import textwrap
import matplotlib.pyplot as plt

try:
    import mpmath
except ImportError:
    try:
        import sympy.mpmath as mpmath
    except ImportError:
        raise ImportError("No mpmath module can be found."
                          "Checked mpmath and sympy.mpmath")


from geotecha.mathematics.mp_laplace import Talbot

dps = mpmath.mp.dps

def f1(s):
    "L-1{1/(1+s)} = e^(-t)"
    return 1/(1+s)

def f2(s,a):
    "L-1{1/(1+s+a)} = e^(-(a+1)*t)"
    return 1/(1+s+a)

def f3(s):
    "L-1{1/(s-1)} = e^t"
    return 1/(s-1)

def f4(s, a):
    "L-1{2*a*s/(s**2+a**2)**2} = t*sin(a*t)"
    return 2*a*s/(s**2+a**2)**2

def setup_dps():
    "set dps to existing mpmath.mp.dps"
    global dps
    dps = mpmath.mp.dps


def teardown_dps():
    "return dps to pre-test value of mpmath.mp.dps"
    global dps
    mpmath.mp.dps = dps

def mpallclose(x, y, rtol=mpmath.mpf('0.00001'),
               atol=mpmath.mpf('0.00000001')):
    """numpy.allclose for mpmath"""

    return np.all(np.less_equal(abs(x-y), atol + rtol * abs(y)))

@with_setup(setup_dps, teardown_dps)
def  test_talbot():
    """test for Talbot numerical inverse Laplace with mpmath"""

    a = Talbot(f=f1, n=24, shift=0.0, dps=None)
    #t=0 raise error:
    assert_raises(ValueError, a, 0)
    #single value of t:
    ok_(mpallclose(a(1), mpmath.exp(mpmath.mpf('-1.0'))))

    #2 values of t:
    ans = np.array([mpmath.exp(mpmath.mpf('-1.0')),
                  mpmath.exp(mpmath.mpf('-2.0'))])
    ok_(mpallclose(a([1,2]),ans))


@with_setup(setup_dps, teardown_dps)
def  test_talbot_dps_fail():
    "test for Talbot numerical inverse Laplace with mpmath insufficient dps"

    a = Talbot(f=f1, n=200, shift=0.0, dps=None)
    #t=0 raise error:
    assert_raises(ValueError, a, 0)
    #single value of t:
    ok_(not mpallclose(a(1), mpmath.exp(mpmath.mpf('-1.0'))))

@with_setup(setup_dps, teardown_dps)
def  test_talbot_dps_pass():
    """test for Talbot numerical inverse Laplace with mpmath sufficient dps"""

    a = Talbot(f=f1, n=200, shift=0.0, dps=30)
    #t=0 raise error:
    assert_raises(ValueError, a, 0)
    #single value of t:
    ok_(mpallclose(a(1), mpmath.exp(mpmath.mpf('-1.0'))))


@with_setup(setup_dps, teardown_dps)
def  test_talbot_dps_precision():
    """test for Talbot numerical inverse Laplace with mpmath high precision"""

    a = Talbot(f=f1, n=200, shift=0.0, dps=95)
    #t=0 raise error:
    assert_raises(ValueError, a, 0)
    #single value of t:
    ok_(mpallclose(a(1),
                   mpmath.exp(mpmath.mpf('-1.0')),
                   atol=mpmath.mpf('1e-80'),
                   rtol=mpmath.mpf('1e-40') ))

@with_setup(setup_dps, teardown_dps)
def  test_talbot_with_args():
    """test for Talbot numerical inverse Laplace with mpmath"""

    a = Talbot(f=f2, n=24, shift=0.0, dps=None)
    #t=0 raise error:
    assert_raises(ValueError, a, 0)
    #single value of t:
    ok_(mpallclose(a(1,args=(1,)), mpmath.exp(mpmath.mpf('-2.0'))))

    #2 values of t:
    ans = np.array([mpmath.exp(mpmath.mpf('-2.0')),
                  mpmath.exp(mpmath.mpf('-4.0'))])
    ok_(mpallclose(a([1,2],args=(1,)),ans))

@with_setup(setup_dps, teardown_dps)
def  test_talbot_with_shift():
    """test for Talbot numerical inverse Laplace with shift"""

    a = Talbot(f=f3, n=24, shift=1.0, dps=None)

    #single value of t:
    ok_(mpallclose(a(1), mpmath.exp(mpmath.mpf('1.0'))))

@with_setup(setup_dps, teardown_dps)
def  test_talbot_with_more_complicated():
    """test for Talbot numerical inverse Laplace with sin"""

    a = Talbot(f=f4, n=24, shift=0, dps=None)

    #single value of t:
    ok_(mpallclose(a(2, args=(1,)),
                   mpmath.mpf('2.0')*mpmath.sin(mpmath.mpf('2.0'))))


if __name__ == '__main__':

    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])