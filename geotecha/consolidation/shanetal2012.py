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
Shan et al (2012) "Exact solutions for one-dimensional consolidation of
single-layer unsaturated soil".


"""
from __future__ import print_function, division

import numpy as np
from matplotlib import pyplot as plt
import sympy

def _integral_for_homogenous_case_linear_initial_condition():
    """equ 52 from shan et al 2012 for depth-linear initial conditon"""
    z, H, M, utop, ubot = sympy.symbols('z, H, M, utop, ubot')


    f1 = utop + z * (ubot - utop) / H
    sn = sympy.sin(M * z / H)

    a = 2/H * sympy.integrate(f1 * sn, (z, 0,H))
    print(a)
    return


def _integral_for_exp_loading_dt():
    """equ 75 from shan et al 2012 for exponential loading diff wrt time"""

    #q(t) = q0 * exp[-b * t]
    M, eig, tau, t, q0, b= sympy.symbols('M, eig, tau, t, q0, b')

    f = q0 * (1-sympy.exp(-b * tau))
    f2 = sympy.exp(eig * M**2 * (t - tau))

    igral = sympy.integrate(sympy.diff(f, tau) * f2, (tau, 0, t), risch=False, conds='none')


    #wolfram:Integrate[Exp[g M**2 * t] * q0 * b * Exp[-b x] * Exp[-g * M**2 * x)], {x,0,t}]

    print('calculated', igral)

    print('manual', "-(b*q0*(exp(-b*t)-exp(eigs*M**2*t)))/(b+eigs*M**2)")

    return


def _integral_for_sin_loading_dt():
    """equ 75 from shan et al 2012 for sin loading diff wrt time"""

    #q(t) = q0 * sin(omega * t)
    r, tau, t, q0, omega= sympy.symbols('r, tau, t, q0, omega')

    f = q0 * sympy.sin(omega * tau)
    f2 = sympy.exp(-r * tau)

    igral = sympy.integrate(sympy.diff(f, tau) * f2, (tau, 0, t), risch=False, conds='none')

    #wolfram: Integrate[Exp[g M**2 * t] * q0 * D[Sin[k*x], x] * Exp[-g * M**2 * x)], {x,0,t}]
    print('calculated', igral)
    print('manual', "(omega*q0*(-eigs*M**2*cos(omega*t)+eigs*M**2*exp(eigs*M**2*t)+omega*sin(omega*t)))/(eigs**2*M**4+omega**2)")
    return


def _integral_for_exp_loading():
    """equ 75 from shan et al 2012 for exponential loading"""

    #q(t) = q0 * exp[-b * t]
    M, eig, tau, t, q0, b= sympy.symbols('M, eig, tau, t, q0, b')

    f = q0 * (1-sympy.exp(-b * tau))
    f2 = sympy.exp(eig * M**2 * (t - tau))

    igral = sympy.integrate(f * f2, (tau, 0, t), risch=False, conds='none')


    #wolfram:Integrate[Exp[g M**2 * t] * q0 *(1 - Exp[-b x]) * Exp[-g * M**2 * x)], {x,0,t}]

    print('calculated', igral)

    print('manual', "(q0*(exp(-b*t)-exp(eigs*M**2*t)))/(b+eigs*M**2)+(q0*(exp(eigs*M**2*t)-1))/(eigs*M**2)")

    return


def _integral_for_sin_loading():
    """equ 75 from shan et al 2012 for sin loading"""

    #q(t) = q0 * sin(omega * t)
    r, tau, t, q0, omega= sympy.symbols('r, tau, t, q0, omega')

    f = q0 * sympy.sin(omega * tau)
    f2 = sympy.exp(-r * tau)

    igral = sympy.integrate(f, tau * f2, (tau, 0, t), risch=False, conds='none')

    #wolfram: Integrate[Exp[g M**2 * t] * q0 * Sin[k*x] * Exp[-g * M**2 * x)], {x,0,t}]
    print('calculated', igral)
    print('manual', "-(q0*(-omega*exp(eigs*M**2*t)+eigs*M**2*sin(omega*t)+omega*cos(omega*t)))/(eigs**2*M**4+omega**2)")
    return



def shanetal2012(z, t, H, Cw, Cvw, Ca, Cva, drn=1, Csw=0, Csa=0,
                 uwi=(0, 0), uai=(0, 0), nterms=100,
                 f=None, f1=None, f2=None, f3=None, f4=None):
    """1D unsaturated consolidation

    Features:

     - Unsaturated soil.
     - Vertical flow.
     - Soil properties constant with time.
     - Initial pore pressure distribution is linear with depth.
       Load is uniform with depth but can be sinusoidal with time,
       or exponential with time.
     - Drainage boundaries in air and water phase can be pervious, impervious
       or piecewise linear with time or sinusoidal with time or exponential
       with time.
     - Pore pressure vs depth in air and water at various times.


    Parameters
    ----------
    z : float or 1d array/list of float
        Depth values for output.
    t : float or 1d array/list of float
        Time values for output
    H : float
        Drainage path length.
    Cw : float
        Cw = (1 - (m2w/m1kw)) / (m2w - m1kw)
    Cvw : float
        Cvw = kw / (gamw * m2w)
    Ca : float
        Ca = (m2a/m1ka) / (1 - m2a/m1ka - n(1-S)/(ua_*m1ka))
    Cva : float
        Cva = ka / {(wa/(R*T)) * m1ka*ua_*[1 - m2a/m1ka - n(1-S)/(ua_*m1ka)]}
        where wa=molecular mass of air= 28.966e-3 kg/mol for air,
        R=universal gas constant=8.31432 J/(mol.K), T = absolute
        temperature in Kelvin=273.16+t0 (K), t0=temperature in celsius, ua_=
        absolute air pressure=ua+uatm (kPa), ua=guage air pressure, uatm=
        atmospheric air pressure=101 kPa.  When ua is small or rapidly
        dissipates during consolidation ua_ can be considered a constant;
        so let ua_=uatm
    drn : int, optional
        Drainage condition. drn=0 is PTPB, drn=1 is PTIB.  Default drn=1.
    Csw : float, optional
        Csw = m1kw/m2w    default=0.
    Csa : float, optional
        Csa = (m2a/m1ka) / (1 - m2a/m1ka - n(1-S)/(ua_*m1ka)) Default=0,
    uwi : 2-element tuple, optioanl
        Initial pore water pressure at top and bottom of soil.  Initial pore
        water pressure within soil is assumed to vary linearly between the
        top and bottom values. Default uwi=(0,0).
    uai : 2-element tuple, optioanl
        Initial pore air pressure at top and bottom of soil.  Initial pore
        air pressure within soil is assumed to vary linearly between the
        top and bottom values. Default uai=(0,0).
    nterms : int, optional
        Number of terms to use in solution. Default nterms=100.
    f : dict, optional
        Ditionary describing a loading function. Default f=None i.e. no load
        e.g.
        f = {'type': 'exp', 'q0': 100, 'b': 0.00005} is a load described by
        q(t) = q0 * exp[-b * t].
        f = {'type': 'sin', 'q0': 100, 'omega': 2*np.pi/1e8} is a load
        described by q(t) = q0 * sin(omega*t).
    f1, f2 : dict, optional
        dict describing loading function for uwtop and uatop.
        Default f1=f2==None i.e. no load.
    f3, f4 : dict, optional
        dict describing loading function for uwbot and uabot.
        Default f3=f4==None i.e. no load.



    Returns
    -------
    porw, pora : 2d array of float
        Pore pressure at depth and time in water and air phase
        por is an array of size (len(z), len(t)).


    References
    ----------
    .. [1] Shan, Zhendong, Daosheng Ling, and Haojiang Ding. 2012. 'Exact
           Solutions for One-dimensional Consolidation of Single-layer
           Unsaturated Soil'. International Journal for Numerical and
           Analytical Methods in Geomechanics 36 (6): 708-22.
           doi:10.1002/nag.1026.

    """



    def E_exp_load_dt(q0,b, t, M, eigs):
        exp=np.exp
        return -(b*q0*(exp(-b*t)-exp(eigs*M**2*t)))/(b+eigs*M**2)
    def E_sin_load_dt(q0,omega, t, M, eigs):
        exp=np.exp
        sin = np.sin
        cos = np.cos
        return ((omega*q0*(-eigs*M**2*cos(omega*t)+eigs*M**2*exp(eigs*M**2*t)+
                omega*sin(omega*t)))/(eigs**2*M**4+omega**2))

    def E_exp_load(q0,b, t, M, eigs):
        exp=np.exp
        return ((q0*(exp(-b*t)-exp(eigs*M**2*t)))/(b+eigs*M**2)+
                (q0*(exp(eigs*M**2*t)-1))/(eigs*M**2))
    def E_sin_load(q0,omega, t, M, eigs):
        exp=np.exp
        sin = np.sin
        cos = np.cos
        return (-(q0*(-omega*exp(eigs*M**2*t)+eigs*M**2*sin(omega*t)+
                omega*cos(omega*t)))/(eigs**2*M**4+omega**2))

    permitted_loads = ['exp', 'sin']
    permitted_parameters={'exp': ['q0','b'],
                          'sin': ['q0', 'omega']}
    f_list =  ['f','f1','f2','f3','f4']
    for fname, ff in zip(f_list,[f,f1,f2,f3,f4]):
        if not ff is None:
            if not ff['type'] in permitted_loads:
                raise ValueError("{} load function must be one "
                    "of type {}; one of yours is "
                    "'{}'".format(fname, permitted_loads, ff['type']))
            for p in permitted_parameters[ff['type']]:
                if not p in ff:
                    raise ValueError("'{}' is missing from the "
                        "{} '{}' loading "
                        "function dict".format(p, fname, ff['type']))


    Cmat = np.array([[1, Cw], [Ca, 1]], dtype=float)
    Kmat = np.array([[Cvw, 0], [0, Cva]], dtype=float)

    a1 = np.sqrt(Cva**2 - 2 * Cva * Cvw + Cvw**2 + 4 * Ca * Cw * Cva * Cvw)
    r1 = 0.5 * (Cva + Cvw - a1)/ (1-Ca*Cw)
    r2 = 0.5 * (Cva + Cvw + a1)/ (1-Ca*Cw)

    sin = np.sin
    cos = np.cos
    if drn==1:
        M = ((2 * np.arange(nterms) + 1) / 2 * np.pi)
    elif drn==0:
        M = (np.arange(1,nterms+1) * np.pi)

    ICmat = np.linalg.inv(Cmat)
    Nmat = np.dot(ICmat,Kmat)

    eigs, v = np.linalg.eig(Nmat)
    v = np.asarray(v)



    if 0:
        #this is just to demonstrate that it is possible to do the
        #eigenvalues and eigenvectors analytically.
        a1 = np.sqrt(Cva**2 - 2 * Cva * Cvw + Cvw**2 + 4 * Ca * Cw * Cva * Cvw)
        r1 = 0.5 * (Cva + Cvw - a1)/ (1-Ca*Cw)
        r2 = 0.5 * (Cva + Cvw + a1)/ (1-Ca*Cw)
        eigs = np.array([r1,r2])
        v = np.array([[r1-Nmat[1,1], r2 - Nmat[1,1]], [Nmat[1,0], Nmat[1,0]]])



    #initial condition
    gw = 2 / H * (-H*uwi[1]*cos(M)/M + H*uwi[0]/M +
            H*uwi[1]*sin(M)/M**2 - H*uwi[0]*sin(M)/M**2)
    ga = 2 / H * (-H*uai[1]*cos(M)/M + H*uai[0]/M +
            H*uai[1]*sin(M)/M**2 - H*uai[0]*sin(M)/M**2)

    X0 = np.empty((1, 1, nterms, 2))
    X0[0, 0, :, 0] = gw
    X0[0, 0, :, 1] = ga

    #z, t, M, uw & ua
    Iv = np.linalg.inv(v)
    Iv_X0 = np.einsum('lp,ijkp', Iv, X0)

    E = np.empty((1, len(t), nterms, 2), dtype=float)
    E[0, :, :, :] = np.exp(     M[None, None, :, None] **2 / H**2 *
                                eigs[None, None, None, :] *
                                t[None, :, None, None])

    v_E_Iv_X0 = np.einsum('lp,ijkp', v, E * Iv_X0)
    S = np.sin(M[None, None, :,None] * z[:,None, None, None]/H)


    u = np.sum(S * v_E_Iv_X0, axis=2)


    #dsig
    E = np.zeros((1, len(t), nterms, 2), dtype=float)
    if not f is None:
        Csig = np.array([Csw, Csa], dtype=float)
        ICmat_Csig = np.dot(ICmat, Csig)

        igral_S = 2 * (1 - np.cos(M)) / M


        Hk = igral_S[None, None, :, None] * ICmat_Csig[None, None, None, :]
        Iv_X = np.einsum('lp,ijkp', Iv, Hk)
        if f['type']=='exp':
            q0 = f['q0']
            b = f['b']
            E[0, :, :, :] = E_exp_load_dt(q0, b,
                                        t[None, :, None, None],
                                        M[None, None, :, None]/H,
                                        eigs[None, None, None, :])
        elif f['type']=='sin':
            q0 = f['q0']
            omega = f['omega']
            E[0, :, :, :] = E_sin_load_dt(q0, omega,
                                        t[None, :, None, None],
                                        M[None, None, :, None] / H,
                                        eigs[None, None, None, :])

        v_E_Iv_X = np.einsum('lp,ijkp', v, E * Iv_X)
        u += np.sum(S * v_E_Iv_X, axis=2)


    #f1 diff wrt t. i.e f1*(h-z)/h, water top
    if not f1 is None:
        ff=f1
        Csig = np.array([-1, -Ca], dtype=float)
        ICmat_Csig = np.dot(ICmat, Csig)
        if drn==0:
            igral_S = (2*(M-sin(M)))/M**2 #2/H * Integrate[Sin[M*x / H] * (H-x)/H, {x, 0, H}]
        elif drn==1:
            igral_S = 2 * (1 - np.cos(M)) / M
        Hk = igral_S[None, None, :, None] * ICmat_Csig[None, None, None, :]
        Iv_X = np.einsum('lp,ijkp', Iv, Hk)
        if ff['type']=='exp':
            q0 = ff['q0']
            b = ff['b']
            E[0, :, :, :] = E_exp_load_dt(q0, b,
                                        t[None, :, None, None],
                                        M[None, None, :, None]/H,
                                        eigs[None, None, None, :])
            #add in homogenizing term
            if drn==0:
                u[:,:,0] += (q0 * (1 - np.exp(-b * t[None, :])) *
                                (1 - z[:, None]/H))
            elif drn==1:
                u[:,:,0] += q0 * (1 - np.exp(-b * t[None, :])) * (1)
        elif ff['type']=='sin':
            q0 = ff['q0']
            omega = ff['omega']
            E[0, :, :, :] = E_sin_load_dt(q0, omega,
                                        t[None, :, None, None],
                                        M[None, None, :, None] / H,
                                        eigs[None, None, None, :])
            #add in homogenizing term
            if drn==0:
                u[:,:,0] += q0 * np.sin(omega * t[None, :]) * (1-z[:, None]/H)
            elif drn==1:
                u[:,:,0] += q0 * np.sin(omega * t[None, :]) * (1)
        v_E_Iv_X = np.einsum('lp,ijkp', v, E * Iv_X)
        u += np.sum(S * v_E_Iv_X, axis=2)


    #f2 diff wrt t. i.e f2*(h-z)/h, air top
    if not f2 is None:
        ff=f2
        Csig = np.array([-Cw, -1], dtype=float)
        ICmat_Csig = np.dot(ICmat, Csig)
        if drn==0:
            igral_S = (2*(M-sin(M)))/M**2 #2/H * Integrate[Sin[M*x / H] * (H-x)/H, {x, 0, H}]
        elif drn==1:
            igral_S = 2 * (1 - np.cos(M)) / M
        Hk = igral_S[None, None, :, None] * ICmat_Csig[None, None, None, :]
        Iv_X = np.einsum('lp,ijkp', Iv, Hk)
        if ff['type']=='exp':
            q0 = ff['q0']
            b = ff['b']
            E[0, :, :, :] = E_exp_load_dt(q0, b,
                                        t[None, :, None, None],
                                        M[None, None, :, None]/H,
                                        eigs[None, None, None, :])
            #add in homogenizing term
            if drn==0:
                u[:,:,1] += (q0 * (1 - np.exp(-b * t[None, :])) *
                                (1 - z[:, None]/H))
            elif drn==1:
                u[:,:,1] += q0 * (1 - np.exp(-b * t[None, :])) * (1)
        elif ff['type']=='sin':
            q0 = ff['q0']
            omega = ff['omega']
            E[0, :, :, :] = E_sin_load_dt(q0, omega,
                                        t[None, :, None, None],
                                        M[None, None, :, None] / H,
                                        eigs[None, None, None, :])
            #add in homogenizing term
            if drn==0:
                u[:,:,1] += q0 * np.sin(omega * t[None, :]) * (1-z[:, None]/H)
            elif drn==1:
                u[:,:,1] += q0 * np.sin(omega * t[None, :]) * (1)
        v_E_Iv_X = np.einsum('lp,ijkp', v, E * Iv_X)
        u += np.sum(S * v_E_Iv_X, axis=2)



    #f3 diff wrt t. i.e f3*z/H, water bot
    if not f3 is None:
        ff=f3
        Csig = np.array([-1, -Ca], dtype=float)
        ICmat_Csig = np.dot(ICmat, Csig)
        if drn==0:
            igral_S = (2*(sin(M)-M*cos(M)))/M**2 #2/H * Integrate[Sin[M*x / H] * (x)/H, {x, 0, H}]
        elif drn==1:
            igral_S = (2*H*(sin(M)-M*cos(M)))/M**2 #2/H * Integrate[Sin[M*x / H] * x, {x, 0, H}]
        Hk = igral_S[None, None, :, None] * ICmat_Csig[None, None, None, :]
        Iv_X = np.einsum('lp,ijkp', Iv, Hk)
        if ff['type']=='exp':
            q0 = ff['q0']
            b = ff['b']
            E[0, :, :, :] = E_exp_load_dt(q0, b,
                                        t[None, :, None, None],
                                        M[None, None, :, None]/H,
                                        eigs[None, None, None, :])
            #add in homogenizing term
            if drn==0:
                u[:,:,0] += (q0 * (1 - np.exp(-b * t[None, :])) *
                                (z[:, None]/H))
            elif drn==1:
                u[:,:,0] += q0 * (1 - np.exp(-b * t[None, :])) * (z[:, None])
        elif ff['type']=='sin':
            q0 = ff['q0']
            omega = ff['omega']
            E[0, :, :, :] = E_sin_load_dt(q0, omega,
                                        t[None, :, None, None],
                                        M[None, None, :, None] / H,
                                        eigs[None, None, None, :])
            #add in homogenizing term
            if drn==0:
                u[:,:,0] += q0 * np.sin(omega * t[None, :]) * (z[:, None]/H)
            elif drn==1:
                u[:,:,0] += q0 * np.sin(omega * t[None, :]) * (z[:, None])
        v_E_Iv_X = np.einsum('lp,ijkp', v, E * Iv_X)
        u += np.sum(S * v_E_Iv_X, axis=2)


    #f4 diff wrt t. i.e f4*z, air bot
    if not f4 is None:
        ff=f4
        Csig = np.array([-Cw, -1], dtype=float)
        ICmat_Csig = np.dot(ICmat, Csig)
        if drn==0:
            igral_S = (2*(sin(M)-M*cos(M)))/M**2 #2/H * Integrate[Sin[M*x / H] * (x)/H, {x, 0, H}]
        elif drn==1:
            igral_S = (2*H*(sin(M)-M*cos(M)))/M**2 #2/H * Integrate[Sin[M*x / H] * x, {x, 0, H}]

        Hk = igral_S[None, None, :, None] * ICmat_Csig[None, None, None, :]
        Iv_X = np.einsum('lp,ijkp', Iv, Hk)
        if ff['type']=='exp':
            q0 = ff['q0']
            b = ff['b']
            E[0, :, :, :] = E_exp_load_dt(q0, b,
                                        t[None, :, None, None],
                                        M[None, None, :, None]/H,
                                        eigs[None, None, None, :])
            #add in homogenizing term
            if drn==0:
                u[:,:,1] += (q0 * (1 - np.exp(-b * t[None, :])) *
                                (z[:, None]/H))
            elif drn==1:
                u[:,:,1] += q0 * (1 - np.exp(-b * t[None, :])) * (z[:, None])
        elif ff['type']=='sin':
            q0 = ff['q0']
            omega = ff['omega']
            E[0, :, :, :] = E_sin_load_dt(q0, omega,
                                        t[None, :, None, None],
                                        M[None, None, :, None] / H,
                                        eigs[None, None, None, :])
            #add in homogenizing term
            if drn==0:
                u[:,:,1] += q0 * np.sin(omega * t[None, :]) * (z[:, None]/H)
            elif drn==1:
                u[:,:,1] += q0 * np.sin(omega * t[None, :]) * (z[:, None])
        v_E_Iv_X = np.einsum('lp,ijkp', v, E * Iv_X)
        u += np.sum(S * v_E_Iv_X, axis=2)


    uw = u[:,:,0]
    ua = u[:,:,1]
    return uw, ua







#########################





if __name__ == '__main__':


    kw = 1e-10
    ka = 10 * kw
    H=10
    Cw=-0.75
    Cvw=-5e-8
    Ca = -0.0775134
    Cva=-64504.4 * ka
    drn=1
    Csw=0.25
    Csa=0.155027
    uwi=(40, 40)
    uai=(20, 20)
    nterms=200
    f=f1=f2=f3=f4=None
#    f = {'type':'exp', 'q0':100.0, 'b':0.00005}
#    f = {'type':'sin', 'q0':100.0, 'omega':2*np.pi / 1e8}
    f3 = {'type':'sin', 'q0':100.0, 'omega':2*np.pi / 1e9}

    z = np.array([0, 3.0, 5.0, 8.0, 10.0])
    t = np.array([1e6,3e6, 1e8,3e8, 1e9, 2e9 ])

    z = np.linspace(0, H, 51)
#    z = np.array([4])
#    t = np.logspace(2,10,400)

#    t = np.array([0, 1e6])

    porw, pora = shanetal2012(z, t, H, Cw, Cvw, Ca, Cva, drn, Csw, Csa,
                 uwi, uai, nterms, f=f, f1=f1, f2=f2, f3=f3, f4=f4)
    if 1:

        print('\nporw', repr(porw))
        print('\npora', repr(pora))

    labels = ['water pore pressure', 'Air pore pressure']
    title = 'drn={}, ka/kw={:.3g}'.format(drn, ka/kw)
    for p, lab in zip([porw, pora], labels):
        fig=plt.figure()
        ax = fig.add_subplot(1,1,1)
        for j, t_ in enumerate(t):
            ax.plot(p[:,j], z, marker='o', label="t={0:.3g}".format(t[j]))

        ax.set_xlabel(lab)
        ax.set_ylabel('Depth')
        ax.invert_yaxis()
#        ax.set_xlim(0,1)
        ax.grid()
        ax.set_title(title)
        leg = plt.legend(loc=3 )
        leg.draggable()



    fig = plt.figure()
    ax=fig.add_subplot('111')
    ax.plot(t, porw[len(z)//2], 'bs', ls='-', ms=3,label='water')
    ax.plot(t, pora[len(z)//2], 'g*', ls='-', ms=3,label='air')
    ax.set_xlabel('Time')
    ax.set_ylabel('Pore pressure at z={}'.format(z[len(z)//2]))
    ax.set_xscale('log')
    ax.set_xlim(0)
    ax.grid()
    ax.set_title(title)
    leg = plt.legend(loc=3 )
    leg.draggable()



#    fig=plt.figure()
#    ax = fig.add_subplot(1,1,1)
#    ax.plot(t, avp, marker='o',)
#    ax.set_xlabel('time')
#    ax.set_ylabel('Overall average pore pressure')
#    ax.invert_yaxis()
##    ax.set_xlim(0,1)
#    ax.grid()
#
#    fig=plt.figure()
#    ax = fig.add_subplot(1,1,1)
#    ax.plot(t, settle, marker='o',)
#    ax.set_xlabel('time')
#    ax.set_ylabel('settlement')
#    ax.invert_yaxis()
##    ax.set_xlim(0,1)
#    ax.grid()

    plt.show()

