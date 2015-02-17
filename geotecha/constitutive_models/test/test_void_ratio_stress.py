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

"""
Testing rotines for piecewise_linear_1d module.

"""
from __future__ import division, print_function

from nose import with_setup
from nose.tools.trivial import assert_almost_equal
from nose.tools.trivial import assert_raises
from nose.tools.trivial import ok_
from nose.tools.trivial import assert_false
from nose.tools.trivial import assert_equal
#from nose.tools.trivial import assertSequenceEqual
import unittest
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose

import numpy as np


from geotecha.constitutive_models.void_ratio_stress import YinAndGrahamSoilModel


DEBUG = False


def test_test_YinAndGrahamSoilModel_CRSN_methods():
    """test that 'step' and 'odeint' methods produce the same CRSN simulations"""

    a = YinAndGrahamSoilModel(lam=0.2, kap=0.04, psi=0.01, siga=20, ea=1, ta=1,
                              e0=0.95, estress0=18)

    tt = np.array([0.0, 1000, 1001, 1e4])
    edot = -np.array([1e-4, 1e-4, 1e-5, 1e-5])
    tvals, estress, e, edot_ = a.CRSN(tt, edot, method='odeint')
    tvals2, estress2, e2, edot_2 = a.CRSN(tt, edot, method='step')

    if DEBUG:
        title ='Compare 2 methods of CRSN simulation'
        print(title)
        print(' '.join(['{:>9s}']*4).format('t', 'odeint', 'step', 'diff'))
        for i, j, k in zip(tvals, estress, estress2):
            print(' '.join(['{:9.4f}']*4).format(i, j, k, j-k))
        print()


        fig, ax = plt.subplots()
#        plt.plot(t, settle, color='red', label = 'calced')
#        plt.plot(t, expected_settle, color='green', marker='o', ls='.', label='expected')

        #instant time line
        x = np.linspace(a.estress0, a.pstress0, 10)
        y = a.e0 - a.kap*np.log(x/a.estress0)
        ax.plot(x, y, label='instant time line')
        #reference time line
        x = np.linspace(a.pstress0, a.pstress0 + 40, 10)
        y = a.ea - a.lam*np.log(x/a.siga)
        ax.plot(x, y, label='reference time line')


        ax.plot(estress, e, label='odeint', marker='+', ms=5, markevery=10)

        ax.plot(estress2, e2, label='step', marker='o', ms=3, markevery=10)


        ax.set_xscale('log')
#        ax.invert_yaxis()
        ax.set_xlabel('Effective stress')
        ax.set_ylabel('Void ratio')
        ax.set_title(title)
        ax.grid()
        leg=plt.legend(loc=1)
        leg.draggable()



        plt.show()

    assert_allclose(estress, estress2, atol=0.25)




if __name__ == "__main__":
    DEBUG = True
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest', '--doctest-options=+ELLIPSIS'])
#    test_test_YinAndGrahamSoilModel_CRSN_methods()