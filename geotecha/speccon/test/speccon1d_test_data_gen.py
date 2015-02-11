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
"""Quick and dirty code to generate some results for testing against
certain geotecha-speccon-speccon1d functions.


This module contains some `eval` commands that can be a security threat. Rather
than fix the problem I've hard coded in some 'if SAFE' (SAFE is a global
variable) instances that you will have to hardcode back to 'SAFE=TRUE'
for them to work.

"""

from __future__ import division, print_function

from collections import OrderedDict
import numpy as np
import sympy
from geotecha.piecewise.piecewise_linear_1d import PolyLine
from numpy import cos, sin
from geotecha.inputoutput.inputoutput import PrefixNumpyArrayString

from geotecha.speccon.speccon1d import dim1sin_E_Igamv_the_BC_abf_linear
from geotecha.speccon.speccon1d import dim1sin_E_Igamv_the_BC_D_aDf_linear
from geotecha.speccon.speccon1d import dim1sin_E_Igamv_the_BC_aDfDt_linear
from geotecha.speccon.speccon1d import dim1sin_E_Igamv_the_BC_deltaf_linear
from geotecha.speccon.speccon1d import dim1sin_E_Igamv_the_abmag_bilinear
from geotecha.speccon.speccon1d import dim1sin_E_Igamv_the_aDmagDt_bilinear
from geotecha.speccon.speccon1d import dim1sin_E_Igamv_the_deltamag_linear

SAFE=False

def gen_dim1sin_integrate_af():
    """Test case data generation for dim1sin_integrate_af."""




    outz = np.array([[0.2, 0.4], [0.4, 0.6]])
    z1 = outz[:, 0]
    z2 = outz[:, 1]
    m = np.array([1.0,2.0, 3.0])
    v_E_Igamv_the = np.ones((3,2), dtype=float)
    tvals = np.array([1.0, 3])
    top_vs_time = PolyLine([0,2,4],[0,2,2])
    bot_vs_time = PolyLine([0,2,4],[0,2,2])
    omega_phase = (1,2)
    a = PolyLine([0, 1], [1, 2])# y = 1 + z
    g = np.array([1.0,2.0])# this is interpolated from top_vs_time at t = 1, 3

    z1_, z2_, z_, m_, g_, t_, omega_, phase_ =sympy.symbols('z1_, z2_, z_, m_, g_, t_, omega_, phase_')

    top1= g_*(1-z_)
    top2= g_
    bot1= g_*z_
    print(g[np.newaxis,:]*z1[:,np.newaxis])
    mapping = [('z1_', 'z1[:,np.newaxis]'),
               ('z2_', 'z2[:,np.newaxis]'),
                ('m_', 'm[np.newaxis,:]'),
                ('g_', 'g[np.newaxis,:]'),
                ('t_', 'tvals[np.newaxis,:]'),
                ('omega_', 'omega_phase[0]'),
                ('phase_', 'omega_phase[1]')]
    fn=OrderedDict()
    fn['no_bc'] = sympy.integrate(sympy.sin(m_ * z_)*(1+z_), (z_, z1_, z2_))
    #anything other than fn['no_bc'] only contains the bc part
    fn['top_vs_time_drn_0'] = sympy.integrate(top1*(1+z_), (z_, z1_, z2_))
    fn['top_vs_time_drn_1'] = sympy.integrate(top2*(1+z_), (z_, z1_, z2_))
    fn['bot_vs_time_drn_0'] = sympy.integrate(bot1*(1+z_), (z_, z1_, z2_))
    fn['bot_vs_time_top_vs_time_drn_1'] = sympy.integrate((top2+bot1)*(1+z_), (z_, z1_, z2_))
    fn['top_vs_time_drn_0_omega_phase'] = sympy.integrate(sympy.cos(omega_*t_+phase_)*top1*(1+z_), (z_, z1_, z2_))
    fn['bot_vs_time_drn_0_omega_phase'] = sympy.integrate(sympy.cos(omega_*t_+phase_)*bot1*(1+z_), (z_, z1_, z2_))
    fn['bot_vs_time_top_vs_time_drn_1_double_loads']= 2*sympy.integrate((top2+bot1)*(1+z_), (z_, z1_, z2_))
    #no_bc part
    sout=str(fn['no_bc'])
    for k, v in mapping:
            sout = sout.replace(k, v)
    print('no_bc')
    print(sout)
    print('#'*10)
    if SAFE:
        no_bc=(np.dot(eval(sout), v_E_Igamv_the))
    print(no_bc)
    print('#'*10+'\n')

    for s, f in fn.items():
        if s == "no_bc":
            continue
        sout = str(f)
        for k, v in mapping:
            sout = sout.replace(k, v)

        print(s)
        print(sout)
        print('#'*10)
        if SAFE:
            print(no_bc + eval(sout))
        print('#'*10+'\n')


def gen_dim1sin_E_Igamv_the_BC_abf_linear():
    """Test case generation for dim1sin_E_Igamv_the_BC_abf_linear.

    2014-03-22"""

    #dim1sin_E_Igamv_the_BC_abf_linear(drn, m, eigs, tvals, Igamv, a, b, top_vs_time, bot_vs_time, top_omega_phase=None, bot_omega_phase=None, dT=1.0):
    outz = np.array([[0.2, 0.4], [0.4, 0.6]])
    z1 = outz[:, 0]
    z2 = outz[:, 1]
    m = np.array([1.0,2.0, 3.0])
    v_E_Igamv_the = np.ones((3,2), dtype=float)
    tvals = np.array([1.0, 3])
    top_vs_time = PolyLine([0,2,4],[0,2,2])
    bot_vs_time = PolyLine([0,2,4],[0,2,2])
    omega_phase = (1,2)
    a = PolyLine([0, 1], [1, 2])# y = 1 + z
    b = PolyLine([0, 1], [1, 2])# y = 1 + z
    g = np.array([1.0,2.0])# this is interpolated from top_vs_time at t = 1, 3
    Igamv = np.identity(3)
    eigs = np.ones(3)


    fn=OrderedDict()

    fn['no_bc'] = {'drn': 0, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
        'a': a, 'b':b, 'top_vs_time': None, 'bot_vs_time':None}
    fn['top_vs_time_drn_0']={'drn': 0, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
            'a': a, 'b':b, 'top_vs_time': [top_vs_time], 'bot_vs_time':None}
    fn['top_vs_time_drn_1']={'drn': 1, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
            'a': a, 'b':b, 'top_vs_time': [top_vs_time], 'bot_vs_time':None}
    fn['bot_vs_time_drn_0']={'drn': 0, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
            'a': a, 'b':b, 'top_vs_time': None,'bot_vs_time':[bot_vs_time]}
    fn['bot_vs_time_top_vs_time_drn_1']={'drn': 1, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
            'a': a, 'b':b, 'top_vs_time': [top_vs_time],'bot_vs_time':[bot_vs_time]}
    fn['test_top_vs_time_drn_0_omega_phase']={'drn': 0, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
            'a': a, 'b':b,
            'top_vs_time': [top_vs_time],
            'bot_vs_time':None,
            'top_omega_phase': [omega_phase]}
    fn['test_bot_vs_time_drn_0_omega_phase']={'drn': 0, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
            'a': a, 'b':b,
            'top_vs_time': None,
            'bot_vs_time':[bot_vs_time],
            'bot_omega_phase': [omega_phase]}
    fn['bot_vs_time_top_vs_time_drn_1_double_loads']={'drn': 1, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
            'a': a, 'b':b, 'top_vs_time': [top_vs_time, top_vs_time],'bot_vs_time':[bot_vs_time, bot_vs_time]}
#    print(dim1sin_E_Igamv_the_BC_abf_linear(0,m,eigs,tvals,Igamv, a,b, top_vs_time=None, bot_vs_time=None))

    for k, v in fn.items():
        print(k)
        print (dim1sin_E_Igamv_the_BC_abf_linear(**v))
        print('#'*10+'\n')


def gen_dim1sin_E_Igamv_the_BC_D_aDf_linear():
    """Test case generation for dim1sin_E_Igamv_the_BC_D_aDf_linear.

    2014-03-22"""

    #dim1sin_E_Igamv_the_BC_D_aDf_linear(drn, m, eigs, tvals, Igamv, a, top_vs_time, bot_vs_time, top_omega_phase=None, bot_omega_phase=None, dT=1.0):
    outz = np.array([[0.2, 0.4], [0.4, 0.6]])
    z1 = outz[:, 0]
    z2 = outz[:, 1]
    m = np.array([1.0,2.0, 3.0])
    v_E_Igamv_the = np.ones((3,2), dtype=float)
    tvals = np.array([1.0, 3])
    top_vs_time = PolyLine([0,2,4],[0,2,2])
    bot_vs_time = PolyLine([0,2,4],[0,2,2])
    omega_phase = (1,2)
    a = PolyLine([0, 1], [1, 2])# y = 1 + z
    b = PolyLine([0, 1], [1, 2])# y = 1 + z
    g = np.array([1.0,2.0])# this is interpolated from top_vs_time at t = 1, 3
    Igamv = np.identity(3)
    eigs = np.ones(3)


    fn=OrderedDict()

    fn['no_bc'] = {'drn': 0, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
        'a': a,  'top_vs_time': None, 'bot_vs_time':None}
    fn['top_vs_time_drn_0']={'drn': 0, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
            'a': a,  'top_vs_time': [top_vs_time], 'bot_vs_time':None}
    fn['top_vs_time_drn_1']={'drn': 1, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
            'a': a,  'top_vs_time': [top_vs_time], 'bot_vs_time':None}
    fn['bot_vs_time_drn_0']={'drn': 0, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
            'a': a,  'top_vs_time': None,'bot_vs_time':[bot_vs_time]}
    fn['bot_vs_time_top_vs_time_drn_1']={'drn': 1, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
            'a': a,  'top_vs_time': [top_vs_time],'bot_vs_time':[bot_vs_time]}
    fn['test_top_vs_time_drn_0_omega_phase']={'drn': 0, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
            'a': a,
            'top_vs_time': [top_vs_time],
            'bot_vs_time':None,
            'top_omega_phase': [omega_phase]}
    fn['test_bot_vs_time_drn_0_omega_phase']={'drn': 0, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
            'a': a,
            'top_vs_time': None,
            'bot_vs_time':[bot_vs_time],
            'bot_omega_phase': [omega_phase]}
    fn['bot_vs_time_top_vs_time_drn_1_double_loads']={'drn': 1, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
            'a': a,  'top_vs_time': [top_vs_time, top_vs_time],'bot_vs_time':[bot_vs_time, bot_vs_time]}
#    print(dim1sin_E_Igamv_the_BC_abf_linear(0,m,eigs,tvals,Igamv, a,b, top_vs_time=None, bot_vs_time=None))

    for k, v in fn.items():
        print(k)
        print (dim1sin_E_Igamv_the_BC_D_aDf_linear(**v))
        print('#'*10+'\n')



def gen_dim1sin_E_Igamv_the_BC_aDfDt_linear():
    """Test case generation for dim1sin_E_Igamv_the_BC_aDfDt_linear.

    2014-03-22"""

    #dim1sin_E_Igamv_the_BC_aDfDt_linear(drn, m, eigs, tvals, Igamv, a, top_vs_time, bot_vs_time, top_omega_phase=None, bot_omega_phase=None, dT=1.0):
    outz = np.array([[0.2, 0.4], [0.4, 0.6]])
    z1 = outz[:, 0]
    z2 = outz[:, 1]
    m = np.array([1.0,2.0, 3.0])
    v_E_Igamv_the = np.ones((3,2), dtype=float)
    tvals = np.array([1.0, 3])
    top_vs_time = PolyLine([0,2,4],[0,2,2])
    bot_vs_time = PolyLine([0,2,4],[0,2,2])
    omega_phase = (1,2)
    a = PolyLine([0, 1], [1, 2])# y = 1 + z
    b = PolyLine([0, 1], [1, 2])# y = 1 + z
    g = np.array([1.0,2.0])# this is interpolated from top_vs_time at t = 1, 3
    Igamv = np.identity(3)
    eigs = np.ones(3)


    fn=OrderedDict()

    fn['no_bc'] = {'drn': 0, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
        'a': a,  'top_vs_time': None, 'bot_vs_time':None}
    fn['top_vs_time_drn_0']={'drn': 0, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
            'a': a,  'top_vs_time': [top_vs_time], 'bot_vs_time':None}
    fn['top_vs_time_drn_1']={'drn': 1, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
            'a': a,  'top_vs_time': [top_vs_time], 'bot_vs_time':None}
    fn['bot_vs_time_drn_0']={'drn': 0, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
            'a': a,  'top_vs_time': None,'bot_vs_time':[bot_vs_time]}
    fn['bot_vs_time_top_vs_time_drn_1']={'drn': 1, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
            'a': a,  'top_vs_time': [top_vs_time],'bot_vs_time':[bot_vs_time]}
    fn['test_top_vs_time_drn_0_omega_phase']={'drn': 0, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
            'a': a,
            'top_vs_time': [top_vs_time],
            'bot_vs_time':None,
            'top_omega_phase': [omega_phase]}
    fn['test_bot_vs_time_drn_0_omega_phase']={'drn': 0, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
            'a': a,
            'top_vs_time': None,
            'bot_vs_time':[bot_vs_time],
            'bot_omega_phase': [omega_phase]}
    fn['bot_vs_time_top_vs_time_drn_1_double_loads']={'drn': 1, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
            'a': a,  'top_vs_time': [top_vs_time, top_vs_time],'bot_vs_time':[bot_vs_time, bot_vs_time]}
#    print(dim1sin_E_Igamv_the_BC_abf_linear(0,m,eigs,tvals,Igamv, a,b, top_vs_time=None, bot_vs_time=None))

    for k, v in fn.items():
        print(k)
        print (dim1sin_E_Igamv_the_BC_aDfDt_linear(**v))
        print('#'*10+'\n')


def gen_dim1sin_E_Igamv_the_BC_deltaf_linear():
    """Test case generation for dim1sin_E_Igamv_the_BC_deltaf_linear.

    2014-03-22"""

    #dim1sin_E_Igamv_the_BC_deltaf_linear(drn, m, eigs, tvals, Igamv, zvals, pseudo_k, top_vs_time, bot_vs_time, top_omega_phase=None, bot_omega_phase=None, dT=1.0):
    outz = np.array([[0.2, 0.4], [0.4, 0.6]])
    z1 = outz[:, 0]
    z2 = outz[:, 1]
    m = np.array([1.0,2.0, 3.0])
    v_E_Igamv_the = np.ones((3,2), dtype=float)
    tvals = np.array([1.0, 3])
    top_vs_time = PolyLine([0,2,4],[0,2,2])
    bot_vs_time = PolyLine([0,2,4],[0,2,2])
    omega_phase = (1,2)
    a = PolyLine([0, 1], [1, 2])# y = 1 + z
    b = PolyLine([0, 1], [1, 2])# y = 1 + z
    g = np.array([1.0,2.0])# this is interpolated from top_vs_time at t = 1, 3
    Igamv = np.identity(3)
    eigs = np.ones(3)
    zvals=[0.2]
    pseudo_k=[1000]

    fn=OrderedDict()

    fn['no_bc'] = {'drn': 0, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,

        'zvals':zvals,
        'pseudo_k': pseudo_k,
        'top_vs_time': None, 'bot_vs_time':None}
    fn['top_vs_time_drn_0']={'drn': 0, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,

        'zvals':zvals,
        'pseudo_k': pseudo_k,
          'top_vs_time': [top_vs_time], 'bot_vs_time':None}
    fn['top_vs_time_drn_1']={'drn': 1, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,

        'zvals':zvals,
        'pseudo_k': pseudo_k,
          'top_vs_time': [top_vs_time], 'bot_vs_time':None}
    fn['bot_vs_time_drn_0']={'drn': 0, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,

        'zvals':zvals,
        'pseudo_k': pseudo_k,
          'top_vs_time': None,'bot_vs_time':[bot_vs_time]}
    fn['bot_vs_time_top_vs_time_drn_1']={'drn': 1, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,

        'zvals':zvals,
        'pseudo_k': pseudo_k,
          'top_vs_time': [top_vs_time],'bot_vs_time':[bot_vs_time]}
    fn['test_top_vs_time_drn_0_omega_phase']={'drn': 0, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,

        'zvals':zvals,
        'pseudo_k': pseudo_k,

            'top_vs_time': [top_vs_time],
            'bot_vs_time':None,
            'top_omega_phase': [omega_phase]}
    fn['test_bot_vs_time_drn_0_omega_phase']={'drn': 0, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
            'zvals':zvals,
        'pseudo_k': pseudo_k,
            'top_vs_time': None,
            'bot_vs_time':[bot_vs_time],
            'bot_omega_phase': [omega_phase]}
    fn['bot_vs_time_top_vs_time_drn_1_double_loads']={'drn': 1, 'm': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,

        'zvals':zvals,
        'pseudo_k': pseudo_k,
          'top_vs_time': [top_vs_time, top_vs_time],'bot_vs_time':[bot_vs_time, bot_vs_time]}
#    print(dim1sin_E_Igamv_the_BC_abf_linear(0,m,eigs,tvals,Igamv, a,b, top_vs_time=None, bot_vs_time=None))

    for k, v in fn.items():
        print(k)
        print (dim1sin_E_Igamv_the_BC_deltaf_linear(**v))
        print('#'*10+'\n')


def gen_dim1sin_E_Igamv_the_abmag_bilinear():
    """Test case generation for dim1sin_E_Igamv_the_abmag_bilinear.

    2014-03-22"""

    #dim1sin_E_Igamv_the_abmag_bilinear(m, eigs, tvals, Igamv, a, b, mag_vs_depth, mag_vs_time, omega_phase=None, dT=1.0):
    m = np.array([1.0,2.0, 3.0])
    v_E_Igamv_the = np.ones((3,2), dtype=float)
    tvals = np.array([1.0, 3])
    mag_vs_time = PolyLine([0,2,4],[0,2,2])
    mag_vs_depth = PolyLine([0, 1], [1, 2])#y = (1+z)

    omega_phase = (1,2)
    a = PolyLine([0, 1], [1, 2])# y = 1 + z
    b = PolyLine([0, 1], [1, 2])# y = 1 + z
    g = np.array([1.0,2.0])# this is interpolated from top_vs_time at t = 1, 3
    Igamv = np.identity(3)
    eigs = np.ones(3)

    fn=OrderedDict()

    fn['single_load'] = {'m': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
               'a':a, 'b':b,
                'mag_vs_depth': [mag_vs_depth], 'mag_vs_time': [mag_vs_time]}
    fn['double_load'] = {'m': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
               'a':a, 'b':b,
                'mag_vs_depth': [mag_vs_depth]*2, 'mag_vs_time': [mag_vs_time]*2}
    fn['omega_phase'] = {'m': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
               'a':a, 'b':b,
                'mag_vs_depth': [mag_vs_depth], 'mag_vs_time': [mag_vs_time],'omega_phase': [omega_phase]}

    for k, v in fn.items():
        print(k)
        print (dim1sin_E_Igamv_the_abmag_bilinear(**v))
        print('#'*10+'\n')


def gen_dim1sin_E_Igamv_the_aDmagDt_bilinear():
    """Test case generation for dim1sin_E_Igamv_the_aDmagDt_bilinear

    2014-03-22"""

    #dim1sin_E_Igamv_the_aDmagDt_bilinear(m, eigs, tvals, Igamv, a, mag_vs_depth, mag_vs_time, omega_phase = None, dT=1.0):
    m = np.array([1.0,2.0, 3.0])
    v_E_Igamv_the = np.ones((3,2), dtype=float)
    tvals = np.array([1.0, 3])
    mag_vs_time = PolyLine([0,2,4],[0,2,2])
    mag_vs_depth = PolyLine([0, 1], [1, 2])#y = (1+z)

    omega_phase = (1,2)
    a = PolyLine([0, 1], [1, 2])# y = 1 + z
    b = PolyLine([0, 1], [1, 2])# y = 1 + z
    g = np.array([1.0,2.0])# this is interpolated from top_vs_time at t = 1, 3
    Igamv = np.identity(3)
    eigs = np.ones(3)

    fn=OrderedDict()

    fn['single_load'] = {'m': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
               'a':a,
                'mag_vs_depth': [mag_vs_depth], 'mag_vs_time': [mag_vs_time]}
    fn['double_load'] = {'m': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
               'a':a,
                'mag_vs_depth': [mag_vs_depth]*2, 'mag_vs_time': [mag_vs_time]*2}
    fn['omega_phase'] = {'m': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
               'a':a,
                'mag_vs_depth': [mag_vs_depth], 'mag_vs_time': [mag_vs_time],'omega_phase': [omega_phase]}

    for k, v in fn.items():
        print(k)
        print (dim1sin_E_Igamv_the_aDmagDt_bilinear(**v))
        print('#'*10+'\n')

def gen_dim1sin_E_Igamv_the_deltamag_linear():
    """Test case generation for dim1sin_E_Igamv_the_deltamag_linear

    2014-03-22"""

    #dim1sin_E_Igamv_the_deltamag_linear(m, eigs, tvals, Igamv, zvals, a, mag_vs_time, omega_phase=None, dT=1.0):
    m = np.array([1.0,2.0, 3.0])
    v_E_Igamv_the = np.ones((3,2), dtype=float)
    tvals = np.array([1.0, 3])
    mag_vs_time = PolyLine([0,2,4],[0,2,2])
    mag_vs_depth = PolyLine([0, 1], [1, 2])#y = (1+z)

    omega_phase = (1,2)
    a = PolyLine([0, 1], [1, 2])# y = 1 + z
    b = PolyLine([0, 1], [1, 2])# y = 1 + z
    g = np.array([1.0,2.0])# this is interpolated from top_vs_time at t = 1, 3
    Igamv = np.identity(3)
    eigs = np.ones(3)
    zvals=[0.2]
    pseudo_k=[1000]

    fn=OrderedDict()

    fn['single_load'] = {'m': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
               'zvals': zvals, 'pseudo_k':pseudo_k,
                'mag_vs_time': [mag_vs_time], 'omega_phase': [None]}
    fn['double_load'] = {'m': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
               'zvals': zvals*2, 'pseudo_k':pseudo_k*2,
                'mag_vs_time': [mag_vs_time]*2, 'omega_phase': [None]*2}
    fn['omega_phase'] = {'m': m, 'eigs':eigs, 'tvals':tvals,'Igamv':Igamv,
               'zvals': zvals, 'pseudo_k':pseudo_k,
                'mag_vs_time': [mag_vs_time],'omega_phase': [omega_phase]}

    for k, v in fn.items():
        print(k)
        print (dim1sin_E_Igamv_the_deltamag_linear(**v))
        print('#'*10+'\n')


if __name__=='__main__':
    PrefixNumpyArrayString().turn_on()
#    gen_dim1sin_integrate_af()
#    gen_dim1sin_E_Igamv_the_BC_abf_linear()
#    gen_dim1sin_E_Igamv_the_BC_D_aDf_linear()
#    gen_dim1sin_E_Igamv_the_BC_aDfDt_linear()
#    gen_dim1sin_E_Igamv_the_BC_deltaf_linear()
#    gen_dim1sin_E_Igamv_the_abmag_bilinear()
#    gen_dim1sin_E_Igamv_the_aDmagDt_bilinear()
#    gen_dim1sin_E_Igamv_the_deltamag_linear()