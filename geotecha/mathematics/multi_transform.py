# geotecha - A software suite for geotechncial engineering
# Copyright (C) 2018  Rohan T. Walker (rtrwalker@gmail.com)
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
"""Some routines related multi dimensional integral transforms"""


from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as plt
import functools

import unittest
from numpy.testing import assert_allclose



from geotecha.mathematics.laplace import Talbot
from geotecha.mathematics.hankel import HankelTransform
from geotecha.mathematics.fourier import FourierTransform
import collections



def ntransform(func, transforms, transvars, args=None, opts=None):
    """
    Multi-dimensional integral transforms over multiple variables.

    General idea and organisation of code is from scipy.integrate.nquad


    Parameters
    ----------
    func : callable
        The function to be integrated. Has arguments of ``x0, ... xn``,
        ``t0, tm``, where integration is carried out over ``x0, ... xn``, which
        must be floats. Function signature should be
        ``func(x0, x1, ..., xn, t0, t1, ..., tm)``. Integral transforms are
        carried out in order. That is, integration over ``x0`` is the
        innermost integral/transform, and ``xn`` is the outermost.
    transform : iterable object of string
        Each elmement of transforms is a string corresponding to one of the
        available transforms.  Current options are:

        - Hankel
        - Hankel_inverse
        - Fourier
        - Fourier_inverse
        - Laplace_inverse

    transvars : iterable
        Transformation variable for each transformation.
    args : iterable object, optional
        Additional arguments ``t0, ..., tn``, required by `func`.
    opts : iterable object or dict, optional
        Options to be passed to each transform. May be empty, a dict, or
        a sequence of dicts, or functions that return a dict. If empty, the
        default options of each transform are used. If a dict, the same
        options are used for all levels of integration. If a sequence, then each
        element of the sequence corresponds to a particular integration/
        transform. e.g.
        opts[0] corresponds to integration/transform over x0, and so on.
        See the individual transforms for options.

    Returns
    -------
    result : float
        The result of the integration.
    abserr : float
        The maximum of the estimates of the absolute error in the various
        integration results.

    See Also
    --------
    geotecha.mathematics.laplace.Talbot : numerical inverse Laplace
    geotecha.mathematics.hankel.HankelTransform : Hankel transform
    geotecha.mathematics.fourier.FourierTransform : Fourier transform


    Notes
    -----
    ntransform is quite temperamental.  Be careful with the following:

     - For some reason performing repeated multi dimensional fourier
       transforms with the default integration limit of np.inf does not
       work (I get a very small number... the wrong number).  This problem
       also happens when doing the same transforms using scipy's nquad for
       multidemensional integration.  There is some issue with the recursion
       and use of the underlying QUADPACK integration routines when performing
       integrations with a cos or sin weight function with infinte integration
       limits.  To get around this you could try truncating the integral by
       setting the `b` keyword agument to the FourierTransform object.  You
       will probably have to trial a few values; larger values give more
       accuracy but too large and the solution is gibberish.
     - Because the inverse laplace transform uses imaginary and any QUADPACK
       integations can only use real numbers.  A laplace transfrom usually must
       be the inner most intergal, i.e. first in the `transform` list.
       If performing multi-dimensional inverse lapace transforms then
       they must be at the front of the `transform` list; all inverse
       laplace transforms other than the first in the `transform` list must
       have {'vectorised': False}, in the corresponding `opts` dict.
       If `f` is not vectorised then the first inverse Laplace transform must
       also have {'vectorised': False}
     - Because the Hankel transform uses numpy broadcasting, it must always
       be the inner most integral, i.e. first in the `transform` list. If also
       doing an inverse laplace transform, then 'Laplace_inverse` must be
       second in the `transform` list with {'vectorised': False} in the
       corresponding `opts` dict.
     - Rules of thumb are 1. 'Hankel_transform', if it occurs, must be first
       in the `transform` list, 2. 'Laplace_inverse' must be before any
       'Fourier' or 'Fourier_inverse' instance in the `transform` list, 3. For
       any but the first 'Laplace_inverse' instance in the `transform` list,
       the corresponding `opts` dict must contain 'vectorized':False, 4.
       'Fourier' or 'Fourier_inverse' may fail using the default b=np.inf, try
       truncating the integral with 'b':35 in the corresponding `opts` dict.
     - Make sure that the order of the `transform` list matches the order
       or the args in function `f`.  First in list transforms first arg of
       `f`, 2nd in list transforms secnod arg of `f`.


    """

    depth = len(transforms)

    try:
        assert len(transvars)==depth
    except TypeError:
        raise TypeError('transvar must be iterable')
    except AssertionError:
        raise AssertionError('transvar and transforms must be the same length')

    if args is None:
        args = ()
    if opts is None:
        opts = [dict([])] * depth

    if isinstance(opts, dict):
        opts = [opts] * depth
    else:
        opts = [opt if isinstance(opt, collections.Callable) else
                _OptFunc(opt) for opt in opts]

    return _NTransform(func, transforms,
                       transvars, opts).integral_transform(*args)


class _OptFunc(object):
    """When called the object will return the variable used to initialize it

    Helper function for the ntransform function.  Useful when you are using a
    function that only accepts callable objects.  Say your function expects
    a callable object that returns a two element list.  If you always want
    to provide a known two element list then you could hard code a
    function to return your list, or you could pass _OptFunc(my_list) and
    avoid all the code snippets.

    Parameters
    ----------
    opt : anything
        The variable that will be returned when the object is called with
        any positional arguments.

    """
    def __init__(self, opt):
        self.opt = opt

    def __call__(self, *args):
        """Return stored dict."""
        return self.opt

class _NTransform(object):
    """Recursively perform integral transforms

    Engine room of the ntransform function.



    """

    def __init__(self, func, transforms, transvars, opts):
        self.abserr = 0
        self.func = func
        self.transforms = transforms
        self.transvars = transvars
        self.opts = opts
        self.maxdepth = len(transforms)


        self.tdict= {'Hankel': self.hankel,
                     'Hankel_inverse': self.hankel,
                     'Fourier': self.fourier,
                     'Fourier_inverse': self.fourier_inverse,
                     'Laplace_inverse': self.laplace_inverse,}


    def hankel(self, f, tvar, args, **opt):
        ht = HankelTransform(f, args, **opt)
        val, err = ht(tvar)
        return val, err
    def fourier(self, f, tvar, args, **opt):
        ft = FourierTransform(f, args, **opt)
        val, err = ft(tvar)
        return val, err
    def fourier_inverse(self,f, tvar, args, **opt):
        ft = FourierTransform(f, args, inv=True, **opt)
        val, err = ft(tvar)
        return val, err
    def laplace_inverse(self, f, tvar, args, **opt):
        ilt = Talbot(f, **opt)
        val = ilt(tvar, args)
        return val, 0.0

    def integral_transform(self, *args, **kwargs):
        """Perform the transforms"""

        depth = kwargs.pop('depth', 0)
        if kwargs:
            raise ValueError('unexpected kwargs')

        # Get the integration range and options for this depth.
        ind = -(depth + 1)
        transform = self.tdict[self.transforms[ind]]
        tvar = self.transvars[ind]

        fn_opt = self.opts[ind]
        opt = dict(fn_opt(*args))

        if depth + 1 == self.maxdepth:
            f = self.func
        else:
            f = functools.partial(self.integral_transform, depth=depth+1)

        value, abserr = transform(f, tvar, args=args, **opt)
        self.abserr = max(self.abserr, np.abs(abserr))
        if depth > 0:
            return value
        else:
            # Final result of n-D integration with error
            return value, self.abserr







#def f_f1(x, y, a, b):
#    f1 = np.exp(-a * abs(x))
#    f2 = np.exp(-b * abs(y))
#    return f1*f2
#def f_f1_(x, y, a, b):
#    f1 = 2 * a / (a**2 + x**2)
#    f2 = 2 * b / (b**2 + y**2)
#    return f1*f2
#
#def test_f_f1():
#    a, b, c = 1.8, 2.2, 0.5
#    x, y, z = 1.1, 1.2, 1.3
##    a, b, c = 0.15, 0.18,0.22
##    x, y, z = 0.1, 0.2, 0.3
#    tvar = (x, y)
#    args= (a, b)
#    transforms=['Fourier', 'Fourier']
#    opts=[{'func_is_real': True, 'real_part_even': True, 'b': 10}]*2
#    print(f_f1_(*(tvar+args)))
#    print(f_f1_(x, y, a, b))
#    assert_allclose(ntransform(f_f1, transforms, tvar, args, opts)[0],
#                    f_f1_(*(tvar+args)), atol=0)
#
#
#
#
#def aaa():
#    a, b= 0.15, 0.18
#    x, y = 0.1, 0.2
#    a, b= 1, 1
#    x, y = 20, 1
#    from scipy.integrate import nquad
#    from scipy.integrate import quad
#
#    tvar = (x, y)
#    args= (a, b)
#    transforms=['Fourier', 'Fourier']
#    opts=[{'func_is_real': True, 'real_part_even':True, 'b':10}]*2
#    val, err=ntransform(f_f1, transforms, tvar, args, opts)
#    expected = f_f1_(*(tvar+args))
#    print('expected', expected)
#    print('ntransform',val, err)
#
#    epsrel=1.49e-08
#    epsabs=1.49e-17
#    val, err = nquad(f_f1, [(0, np.inf), (0, 54)], args=(a, b),
#                    opts=[{'weight': 'cos', 'wvar': x, 'epsabs':epsabs, 'epsrel': epsrel}
#                            , {'weight': 'cos', 'wvar': y, 'epsabs':epsabs, 'epsrel': epsrel}])
#    print('nquad', 4*val, err)
#    val,err = quad(lambda x,a: np.exp(-a * abs(x)) , 0, np.inf, args=(a,),
#                   weight='cos', wvar=x)
#    val1, err1 = quad(lambda x,a: np.exp(-a * abs(x)) , 0, np.inf, args=(b,),
#                   weight='cos', wvar=y)
#    print('quad*quad', 4*val*val1, err+err1)

if __name__ == '__main__':
#    aaa()
#    test_f_f1()

    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])