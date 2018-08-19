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
        title ='Compare 2 methods of constant rate of void ratio (CRSN) simulation'
        print(title)
        print(' '.join(['{:>9s}']*4).format('t', 'odeint es', 'step es', 'diff'))
        for i, j, k in zip(tvals, estress, estress2):
            print(' '.join(['{:9.4f}']*4).format(i, j, k, j-k))
        print()


        fig, ax = plt.subplots()
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


def test_test_YinAndGrahamSoilModel_CRSS_methods():
    """test that 'step' and 'odeint' methods produce the same CRSS simulations"""

    a = YinAndGrahamSoilModel(lam=0.2, kap=0.04, psi=0.01, siga=20, ea=1, ta=1,
                              e0=0.95, estress0=18)

    tt = np.array([0.0, 20, 20.001, 80])
    estressdot = np.array([1, 1, 1e-1, 1e-1])
    tvals, estress, e, edot_ = a.CRSS(tt, estressdot, method='odeint')
    tvals2, estress2, e2, edot_2 = a.CRSS(tt, estressdot, method='step')

    if DEBUG:
        title ='Compare 2 methods of constat rate of stress (CRSS) simulation'
        print(title)
        print(' '.join(['{:>9s}']*4).format('t', 'odeint e', 'step e', 'diff'))
        for i, j, k in zip(tvals, e, e2):
            print(' '.join(['{:9.4f}']*4).format(i, j, k, j-k))
        print()


        fig, ax = plt.subplots()
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

    assert_allclose(e, e2, atol=0.0012)



def test_YinAndGrahamSoilModel_av_from_stress_CRSS_methods():
    """step through a CRSS test and check that av_from_stress gives corret"""

    a = YinAndGrahamSoilModel(lam=0.2, kap=0.04, psi=0.01, siga=20, ea=1, ta=1,
                              e0=0.90, estress0=18)



    tmax=30
    estressdot = 1.5    
    tt = np.linspace(0,tmax, 100)
#    tt = np.logspace(np.log10(0.01),np.log10(tmax),1000)
    
    dt = np.diff(tt)
    
    av = np.zeros_like(tt)
    av_check = np.zeros_like(tt)
    e = np.zeros_like(tt)
    estress = np.zeros_like(tt)
    igral = np.zeros_like(tt)
    
    av[0] = a.kap / a.estress0 + a.psi / a.t0 / estressdot    
    estress[0] = a.estress0
    e[0] = a.e0
    igral[0] = a._igral
    av_check[0]= a.av_from_stress(estress=estress[0], estressdot=estressdot)
    
    for i, dt_ in enumerate(dt):
        estress[i + 1] += estress[i] + dt_ * estressdot
        e[i + 1] = a.e_from_stress(estress=estress[i+1], dt=dt_)        
        av[i + 1] = (e[i]-e[i+1])/(estress[i+1]-estress[i])
        av_check[i+1]= a.av_from_stress(estress=estress[i+1], estressdot=estressdot)
        igral[i+1] = a._igral
        
    if DEBUG:
        
        title ='Compare av_from_stress with finite difference approximation'
        print(title)
        print(' '.join(['{:>12s}']*4).format('t', 'av findiff', 'av_f_stress', 'diff'))
        for i, j, k in zip(tt, av[0::10], av_check[0::10]):
            print(' '.join(['{:12.6f}']*4).format(i, j, k, j-k))
        print()
        
        fig = plt.figure()
        ax = fig.add_subplot('221')
        ax.plot(estress, av, label="step thrugh")
        ax.plot(estress, av_check, label="av_from_stress")
        ax.set_xlabel("estress")        
        ax.set_ylabel("av")        
        leg = ax.legend()
        leg.draggable()
        
        ax = fig.add_subplot('223')
        ax.plot(estress, e, label="stress path")
        
        ax.set_xlabel("estress")        
        ax.set_ylabel("e")        
        leg = ax.legend()
        leg.draggable()
        
        ax = fig.add_subplot('222')
        ax.plot(tt, av, label="step thrugh")
        ax.plot(tt, av_check, label="av_from_stress")
        ax.set_xlabel("time")        
        ax.set_ylabel("av")        
        leg = ax.legend()
        leg.draggable()
        
        ax = fig.add_subplot('224')
        ax.plot(tt, e, label="stress path")
        
        ax.set_xlabel("time")        
        ax.set_ylabel("e")        
        leg = ax.legend()
        leg.draggable()        
        
        fig.tight_layout()
        plt.show()

    assert_allclose(av, av_check, atol=0.0001)


if __name__ == "__main__":
#    DEBUG = True
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest', '--doctest-options=+ELLIPSIS'])

#    test_test_YinAndGrahamSoilModel_CRSN_methods()
#    test_test_YinAndGrahamSoilModel_CRSS_methods()
#    test_YinAndGrahamSoilModel_av_from_stress_CRSS_methods()