#!/usr/bin/env python
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
Finite elastic Euler-Bernoulli beam resting on viscoelastic foundation
with piecewise-linear properties, subjected to multiple moving point loads.


"""

from __future__ import division, print_function


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from geotecha.mathematics.root_finding import find_n_roots
from scipy import integrate
from scipy.integrate import odeint
import matplotlib.style
import matplotlib as mpl

import time
from datetime  import timedelta
import datetime
from collections import OrderedDict
import os

import geotecha.speccon.speccon1d as speccon1d
import geotecha.piecewise.piecewise_linear_1d as pwise
from geotecha.piecewise.piecewise_linear_1d import PolyLine
import geotecha.speccon.integrals as integ
import geotecha.inputoutput.inputoutput as inputoutput
from geotecha.inputoutput.inputoutput import SimpleTimer

from geotecha.plotting.one_d import save_figure

from numpy.testing import assert_allclose
from geotecha.plotting.one_d import MarkersDashesColors

DEBUG=True


class SpecBeam(object):
    """Finite elastic Euler-Bernoulli beam resting on
    viscoelastic foundation subjected to a moving load, piecewise-linear
    properties.

    An extension of Ding et al. (2012) [1]_ with piecewise linear
    material properties and non-linear foundation stiffness k3=0 in k3*w**3.


    You don't need all the parameters.  Basically if normalised values are
    'None' (i.e. default) then those properties will be calculated from
    the non normalised quantities.  All calculations are done in normalised
    space.  You only need the non-normalised variables if you want
    non-normalised output.


    Note: some of the code is a relic of
    geotecha.beam_on_foundation.dingetal2012 which is a straight up
    implementation of [1]_ .  Sorry for any confusion.

    Parameters
    ----------
    BC : ["SS","CC","FF"], optional
        Boundary condition. Can only have Simply Supported (SS).
        Clamped Clamped (CC) and Free Free (FF) are a relic from a differnet
        program.  Default BC="SS".
    nterms : integer, optional
        Number of terms for Galerkin truncation. Default nterms=50
    E : float, optional
        Young's modulus of beam.(E in [1]_).
    rho : float, optional
        Mass density of beam.
    I : float, optional
        Second moment of area of beam (I in [1]_).
    A : float, optional
        Cross sectional area of beam
    L : float, optional
        Length of beam. (L in [1]_).
    k1 : float, optional
        Mean stiffness of foundation.
    k3 : float, optional
        Non linear stiffness of foundation.
    mu : float, optional
        Viscous damping of foundation.
#    Fz : float, optional
#        Load.  This is a relic; loads now specified with moving_loads_...

    v : list of float, optional
        Velocity of each group of moving point loads.
    moving_loads_x : list of lists of floats, optional
        Each sublist contains x-coords of each point load in the moving load
        group.  The coords are relative to a reference point which will
        enter the beam at the corresponding time value in `moving_loads_t0`.
        Coords are usually negative for vehicle facing left to right.
    moving_loads_Fz : list of lists of floats, optional
        Each sublist contains magnitude of of each point load in moving load
        group.
    moving_loads_v : list of float, optional
        Velocity of each moving point load group.  When velocity is negative
        the x-coords in the corresponding member of `moving_loads_x` will
        be multiplied by -1 before the beam enters the beam from the right.
    moving_loads_offset : list of float, optional
        An initial distance by which the corresponding `moving_loads_x` is
        offset. Will default to zero if not inputted.
    moving_loads_t0 : list of lists, optional
        Time that moving load group enters the beam (strictyly speaking it
        is when the 0 coord in `moving_loads_x` enters the beam). If v>0
        moving load group starts at x=0, t=0 and moves to the right.
        If v<0 then moving load group starts at x=L,t=0 and moves to the
        left.  Defaults to 0.
    moving_loads_x0 : list of float, optional
        Position where moving load group appears on the beam.  Defaults to
        zero if ommited.
    moving_loads_L : list of float, optional
        Distance the load group travels on beam.  Defaults to L if omitted.
        Used in conjunction with moving loads_x0.
    tvals : 1d array of float, optional
        Output times
    xvals : 1d array of float, optional
        Output x-coord

    v_norm : list of float, optional
        normalised velocity = V * sqrt(rho / E) for each group of moving
        point loads. An example of a consistent
        set of units to get the correct v_norm is rho in kg/m^3, L in m, and
        E in Pa.
    moving_loads_x_norm : list of lists of floats, optional
        As per `moving_loads_x` but nomalised.
        x_norm = x / L
    moving_loads_Fz_norm : list of lists of floats, optional
        As per `moving_loads_Fz` but nomalised.
        Fz_norm = Fz / (E * A).
    moving_loads_v_norm : list of float, optional
        As per `moving_loads_v` but nomalised.
        v_norm= v * sqrt(rho / E) for each group of moving
        point loads. An example of a consistent
        set of units to get the correct v_norm is rho in kg/m^3, L in m, and
        E in Pa.
    moving_loads_offset_norm : list of float, optional
        As per `moving_loads_offset` but nomalised.
        offset_norm = offet / L
    moving_loads_t0_norm : list of lists, optional
        As per `moving_loads_t0` but nomalised.
        t0_norm = t0 / L * np.sqrt(E / rho)
    moving_loads_x0_norm : list of float, optional
        As per `moving_loads_x0`. x0_norm=x0/L.
    moving_loads_L_norm : list of float, optional
        As per `moving_loads_L_norm` but normalised.L_norm = L' / L

    tvals_norm : 1d array of float, optional
        Normalised output times.
        t_norm = t / L * np.sqrt(E / rho)
    xvals_norm : 1d array of float, optional
        Normalised output coords.
        x_norm = x / L


    stationary_loads_x : list of float, optional
        Location of stationary point loads.
    stationary_loads_vs_t : list of Polyline, optional
        Stationary load magnitude variation with time. PolyLine(time, magnitude).
    stationary_loads_omega_phase : list of 2 element tuples, optional
        (omega, phase) to define cyclic variation of stationary point loads. i.e.
        mag_vs_time * cos(omega*t + phase). If stationary_loads_omega_phase is None
        then cyclic component will be ignored.  If stationary_loads_omega_phase is a
        list then if any member is None then cyclic component will not be
        applied for that load combo.

    stationary_loads_x_norm : list of float, optional
        As per stationary_loads_x but normalised
    stationary_loads_vs_t_norm : list of Polyline, optional
        As per stationary_loads_vs_t but normalised
    stationary_loads_omega_phase_norm : list of 2 element tuples, optional
        As per stationary_loads_omega_phase but normalised.


    kf : float, optional
        Normalised modulus of elasticity = 1 / L * sqrt(I / A).
#    Fz_norm : float, optional
#        Normalised load = Fz / (E * A). This is a relic; loads now specified
#        with moving_loads_...
    mu_norm : float, optional
        Normalised damping = mu / A * sqrt(L**2 / (rho * E)).
    k1_norm : float, optional
        Normalised mean stiffness = k1 * L**2 / (E * A).
    k3_norm : float, optional
        Normalised non-linear stiffness = k3 * L**4 / (E * A)
    nquad : integer, optional
        Number of quadrature points for numerical integration of non-linear
        k3*w**3*w_ term.  Default nquad=30.  Note I've had errors when n>35.
        For the special case of nquad=None then integration will be performed
        using scipy.integrate.quad; this is slower.
    Ebar : PolyLine, optional
        PolyLine object representing piecewise linear E/Eref vs x.
        Default Ebar = PolyLine([0],[1],[1],[1]) i.e. uniform.
    rhobar : PolyLine, optional
        PolyLine object representing piecewise linear rho/rhoref vs x.
        Default rhobar = PolyLine([0],[1],[1],[1]) i.e. uniform.
    Ibar : PolyLine, optional
        PolyLine object representing piecewise linear I/Iref vs x.
        Default Ibar = PolyLine([0],[1],[1],[1]) i.e. uniform.
    Abar : PolyLine, optional
        PolyLine object representing piecewise linear A/Aref vs x.
        Default Abar = PolyLine([0],[1],[1],[1]) i.e. uniform.
    k1bar : PolyLine, optional
        PolyLine object representing piecewise linear k1/k1ref vs x.
        Default k1bar = PolyLine([0],[1],[1],[1]) i.e. uniform.
    k3bar : PolyLine, optional
        PolyLine object representing piecewise linear k3/k3ref vs x.
        Default k3bar = PolyLine([0],[1],[1],[1]) i.e. uniform.
    mubar : PolyLine, optional
        PolyLine object representing piecewise linear mu/muref vs x.
        Default mubar = PolyLine([0],[1],[1],[1]) i.e. uniform.
    use_analytical : [False,true]
        If true then system of second order ode resulting from spectral
        galerkin method will be solved analytically.  If False then the
        ode system will be solved numerically (potentially very slow).
        analytical approach is only avaialble when BC='SS' and
        k3=0 (or k3_norm=0).
    file_stem : string, optional
        File stem for saving deflection data.  Default file_stem="specbeam_".
    force_calc : [False, True]
        If True then calcualtion will happen regardless of presence of output
        file (file_stem)

    Attributes
    ----------
    phi : function
        Relevant Galerkin trial function.  Depends on `BC`.  See [1]_ for
        details.
    beta : 1d ndarray of `nterms` float
        beta terms in Galerkin trial function.
    xj : 1d ndarray of `nquad` float
        Quadrature integration points.
    Ij : 1d ndarray of `nquad` float
        Weighting coefficients for numerical integration.
    BC_coeff : int
        I don't think this is relevant anymore.
        Coefficent  to multiply the Fz and k3 terms in the ode.
        For `BC`="SS" BC_coeff=2, for `BC`="CC" or "FF" BC_coeff=2.
        See [1]_ Equation (31) and (32).
    t : 1d ndarray of float
        Raw time values. Only valid after running calulate_qk.
    t_norm : 1d ndarray of float
        Normlised time values = t / L * sqrt(E / rho).
        Only valid after running calulate_qk.
    qsol : 2d ndarray of shape(len(`t`), 2* `nterms`) float
        Values of Galerkin coefficients at each time i.e. qk(t) in [1]_.
        w(x) = sum(qk * phi_k(x)).
        Only valid after running calulate_qk.
    lam_qdotdot : 2d array of shape (nterms, nterms)
        spectral coefficeient matrix for qdotdot
    psi_q : 2d array of shape (nterms, nterms)
        spectral coefficeient matrix for q
    psi_qdot : 2d array of shape (nterms, nterms)
        spectral coefficeient matrix for qdot.
    k3barj : 1d array of float with len(nquad)
        value of k3 interpolated from k3bar at the quadrature points. Used
        in the  k3*w**3 term.
    L_norm : float
        Normalised length.  Always =1.
    defl_norm : array of float, size (len(xvals_norm), len(tvals_norm))
        normalised defelction at xval,tval
    defl : array of float, size (len(xvals_norm), len(tvals_norm))
        deflection at xval,tval.  ONly if self.L is not None.
    beta_block : 1d array float with size (2*nterms)
        Block column vector of repeated beta values.
    has_moving_loads : boolean
        If True then there are moving point loads.
    has_stationary_loads : boolean
        If True then there are stationary loads.
    moving_load_w_norm : list of ndarray
        Normalised deflections under each moving load at each time value.
        Each element of the list corresponds to each moving load.  The array
        is of size (len(mvpl.x), len(tvals))  i.e. for each axle in the moving
        load group.
    moving_load_w : list of ndarray
        Non normalised deflections under each moving load.

    Notes
    -----
    defl and def_norm will be 1d arrays if len(xvals)=1.

    References
    ----------
    .. [1] Ding, H., Chen, L.-Q., and Yang, S.-P. (2012).
           "Convergence of Galerkin truncation for dynamic response of
           finite beams on nonlinear foundations under a moving load."
           Journal of Sound and Vibration, 331(10), 2426-2442.
    .. [2] Walker, R.T.R. and Indraratna, B, (in press) "Moving loads on a
           viscoelastic foundation with special reference to railway
           transition zones". International Journal of Geomechanics.

    """

    def __init__(self,
                    BC="SS",
                    nterms=50,
                    E=None,
                    I=None,
                    rho=None,
                    A=None,
                    L=None,
                    k1=None,
                    k3=None,
                    mu=None,
#                    Fz=None, #This is a relic; loads now specified with moving_loads_...
#                    v=None, #This is a relic; loads now specified with moving_loads_...
#                    v_norm=None,

                    moving_loads_x=None,
                    moving_loads_Fz=None,
                    moving_loads_v=None,
                    moving_loads_offset=None,
                    moving_loads_t0=None,
                    moving_loads_x0=None,
                    moving_loads_L=None,

                    tvals=None,
                    xvals=None,
                    v_norm=None,
                    moving_loads_x_norm=None,
                    moving_loads_Fz_norm=None,
                    moving_loads_v_norm=None,
                    moving_loads_offset_norm=None,
                    moving_loads_t0_norm=None,
                    moving_loads_x0_norm=None,
                    moving_loads_L_norm=None,
                    tvals_norm=None,
                    xvals_norm=None,

                    stationary_loads_x=None,
                    stationary_loads_vs_t=None,
                    stationary_loads_omega_phase=None,
                    stationary_loads_x_norm=None,
                    stationary_loads_vs_t_norm=None,
                    stationary_loads_omega_phase_norm=None,



                    kf=None,
#                    Fz_norm=None, #This is a relic; loads now specified with moving_loads_...
                    mu_norm=None,
                    k1_norm=None,
                    k3_norm=None,
                    nquad=30,
                    Ebar=PolyLine([0],[1],[1],[1]),
                    rhobar=PolyLine([0],[1],[1],[1]),
                    Ibar=PolyLine([0],[1],[1],[1]),
                    Abar=PolyLine([0],[1],[1],[1]),
                    k1bar=PolyLine([0],[1],[1],[1]),
                    k3bar=PolyLine([0],[1],[1],[1]),
                    mubar=PolyLine([0],[1],[1],[1]),
                    implementation="vectorized",
                    use_analytical=False,
                    file_stem="specbeam_",
                    force_calc=False):


        self.BC = BC
        self.nterms = nterms
        self.E = E
        self.I = I
        self.rho = rho
        self.I = I
        self.A = A
        self.L = L
        self.k1 = k1
        self.k3 = k3
        self.mu = mu
#        self.Fz = #Fz This is a relic; loads now specified with moving_loads_...
#        self.v = v #This is a relic; loads now specified with moving_loads_...
#        self.v_norm = v_norm #This is a relic; loads now specified with moving_loads_...

        self.moving_loads_x=moving_loads_x
        self.moving_loads_Fz=moving_loads_Fz
        self.moving_loads_v=moving_loads_v
        self.moving_loads_offset=moving_loads_offset
        self.moving_loads_t0=moving_loads_t0
        self.moving_loads_x0=moving_loads_x0
        self.moving_loads_L=moving_loads_L
        self.tvals=tvals
        self.xvals=xvals
        #self.v_norm=v_norm #This is a relic; loads now specified with moving_loads_...
        self.moving_loads_x_norm=moving_loads_x_norm
        self.moving_loads_Fz_norm=moving_loads_Fz_norm
        self.moving_loads_v_norm=moving_loads_v_norm
        self.moving_loads_offset_norm=moving_loads_offset_norm
        self.moving_loads_t0_norm=moving_loads_t0_norm
        self.moving_loads_x0_norm=moving_loads_x0_norm
        self.moving_loads_L_norm=moving_loads_L_norm
        self.tvals_norm=tvals_norm
        self.xvals_norm=xvals_norm

        self.stationary_loads_x=stationary_loads_x
        self.stationary_loads_vs_t=stationary_loads_vs_t
        self.stationary_loads_omega_phase=stationary_loads_omega_phase
        self.stationary_loads_x_norm=stationary_loads_x_norm
        self.stationary_loads_vs_t_norm=stationary_loads_vs_t_norm
        self.stationary_loads_omega_phase_norm=stationary_loads_omega_phase_norm


        self.kf = kf
#        self.Fz_norm = Fz_norm
        self.mu_norm = mu_norm
        self.k1_norm = k1_norm
        self.k3_norm = k3_norm
        self.nquad = nquad


        self.Ebar = Ebar
        self.Ibar = Ibar
        self.Abar = Abar
        self.k1bar = k1bar
        self.k3bar = k3bar
        self.mubar = mubar
        self.rhobar = rhobar
        self.implementation = implementation
        self.use_analytical=use_analytical
        self.file_stem=file_stem
        self.force_calc=force_calc

        if (self.k3_norm is None) and self.k3 is None:
            self.k3=0
            self.k3_norm=0

        if (((self.BC.upper() in ["CC","FF"]) or
            (sum([v==0 for v in [self.k3, self.k3_norm]]) != 2))
            and self.use_analytical):
            raise ValueError ("If use_analytical=True, cannot have BC='CC, BC='FF', or either k3!=None or k3_norm!=None")

        # normalised parameters
        self.L_norm = 1
        if kf is None:
            self.kf = 1 / self.L * np.sqrt(self.I / self.A)
#        if v_norm is None: #This is a relic; loads now specified with moving_loads_...
#            self.v_norm = self.v * np.sqrt(self.rho / self.E) #This is a relic; loads now specified with moving_loads_...


        if all([v is None for v in
                [self.moving_loads_x_norm,
                 self.moving_loads_Fz_norm,
                 self.moving_loads_v_norm,
                 self.moving_loads_x,
                 self.moving_loads_Fz,
                 self.moving_loads_v]
                ]):
            #no moving loads
            self.has_moving_loads=False
        else:
            # at least one moving loads
            self.has_moving_loads=True

            if self.moving_loads_x_norm is None:
                self.moving_loads_x_norm = [[vv/self.L for vv in v] for v in self.moving_loads_x]
            if self.moving_loads_Fz_norm is None:
                self.moving_loads_Fz_norm = [[vv / (self.E * self.A) for vv in v] for v in self.moving_loads_Fz]
            if self.moving_loads_v_norm is None:
                self.moving_loads_v_norm = [v * np.sqrt(self.rho / self.E) for v in self.moving_loads_v]

            #optional moving load variables
            ###################
            if (self.moving_loads_offset is None):
                if not self.moving_loads_x is None:
                    #raw input values used:
                    self.moving_loads_offset = np.zeros(len(self.moving_loads_x))
                elif self.moving_loads_offset_norm is None:
                    #nomalised values unsed but default behaviour
                    self.moving_loads_offset_norm = np.zeros(len(self.moving_loads_x_norm))
            if self.moving_loads_offset_norm is None:
                self.moving_loads_offset_norm = [v /self.L for v in self.moving_loads_offset]
            ###################
            if (self.moving_loads_t0 is None):
                if not self.moving_loads_x is None:
                    #raw input values used:
                    self.moving_loads_t0 = np.zeros(len(self.moving_loads_x))
                elif self.moving_loads_t0_norm is None:
                    #nomalised values used but default behaviour
                    self.moving_loads_t0_norm = np.zeros(len(self.moving_loads_x_norm))
            if self.moving_loads_t0_norm is None:
                self.moving_loads_t0_norm = [v / self.L * np.sqrt(self.E / self.rho) for v in self.moving_loads_t0]
            ###################
            if self.moving_loads_x0 is None:
                if not self.moving_loads_x is None:
                    #raw input values used for moving loads but default for x
                    self.moving_loads_x0 = np.zeros(len(self.moving_loads_x))
                elif self.moving_loads_x0_norm is None:
                    #nomalised values used but default behaviour
                    self.moving_loads_x0_norm = np.zeros(len(self.moving_loads_x_norm))
            if self.moving_loads_x0_norm is None:
                self.moving_loads_x0_norm = [v /self.L for v in self.moving_loads_x0]
            ###################
            if self.moving_loads_L is None:
                if not self.moving_loads_x is None:
                    #raw input values used for moving loads but default for x
                    self.moving_loads_L = self.L * np.ones(len(self.moving_loads_x), dtype=float)
                elif self.moving_loads_L_norm is None:
                    #nomalised values used but default behaviour
                    self.moving_loads_L_norm = np.ones(len(self.moving_loads_x_norm),dtype=float)
            if self.moving_loads_L_norm is None:
                self.moving_loads_L_norm = [v /self.L for v in self.moving_loads_L]
            ###################


            self.moving_loads_norm = [MovingPointLoads(x=xv,
                                                       p=pv,
                                                       offset=offv,
                                                       v=vv,
                                                       t0=t0v,
                                                       L=Lv,
                                                       x0=x0v)
                                        for xv,pv,offv,vv,t0v,Lv,x0v in
                                        zip(self.moving_loads_x_norm,
                                            self.moving_loads_Fz_norm,
                                            self.moving_loads_offset_norm,
                                            self.moving_loads_v_norm,
                                            self.moving_loads_t0_norm,
                                            self.moving_loads_L_norm,
                                            self.moving_loads_x0_norm)]




            if not self.moving_loads_x is None:
                self.moving_loads = [MovingPointLoads(x=xv,
                                                       p=pv,
                                                       offset=offv,
                                                       v=vv,
                                                       t0=t0v,
                                                       L=self.L)
                                        for xv,pv,offv,vv,t0v in
                                        zip(self.moving_loads_x,
                                            self.moving_loads_Fz,
                                            self.moving_loads_offset,
                                            self.moving_loads_v,
                                            self.moving_loads_t0)]





        if self.tvals_norm is None:
            self.tvals_norm = self.tvals / self.L * np.sqrt(self.E / self.rho)
        if self.xvals_norm is None:
            self.xvals_norm = self.xvals / self.L


        if all([v is None for v in
                [self.stationary_loads_x_norm,
                self.stationary_loads_omega_phase_norm,
                self.stationary_loads_vs_t_norm,
                self.stationary_loads_x,
                self.stationary_loads_omega_phase,
                self.stationary_loads_vs_t_norm,]
               ]):
            self.has_stationary_loads=False
        else:
            self.has_stationary_loads=True
            #check if stationary loads omega phase is missing
            if self.stationary_loads_omega_phase is None:
                if not self.stationary_loads_x is None:
                    #use default omega_phase i.e. NONE
                    self.stationary_loads_omega_phase = len(self.stationary_loads_x) * [None]



            if self.stationary_loads_x_norm is None:
                self.stationary_loads_x_norm = [v / self.L for v in self.stationary_loads_x]

            if self.stationary_loads_vs_t_norm is None:
                #PolyLine(time, mag)
                #what if
                self.stationary_loads_vs_t_norm = (
                    [PolyLine(v.x / self.L * np.sqrt(self.E / self.rho), v.y / (self.E * self.A)) for v in self.stationary_loads_vs_t])

            if self.stationary_loads_omega_phase_norm is None:
                if self.stationary_loads_omega_phase is None:
                    #use default omega_phase_norm, i.e. None
                    self.stationary_loads_omega_phase_norm = len(self.stationary_loads_x_norm) * [None]
                else:
                    #only omega is effected, phase remains unchanged
                    self.stationary_loads_omega_phase_norm = (
                        [(v[0] * self.L / np.sqrt(self.E / self.rho),v[1]/ (self.E * self.A))
                            if not v is None else None for v in self.stationary_loads_omega_phase])







        if self.kf is None:
            self.kf = 1 / self.L * np.sqrt(self.I / self.A)
#        if Fz_norm is None: #This is a relic; loads now specified with moving_loads_...
#            self.Fz_norm = self.Fz / (self.E * self.A) #This is a relic; loads now specified with moving_loads_...

        if self.mu_norm is None:
            self.mu_norm = self.mu / self.A * np.sqrt(self.L**2 / (self.rho * self.E))
        if self.k1_norm is None:
            self.k1_norm = self.k1 * self.L**2 / (self.E * self.A)
        if self.k3_norm is None:
            self.k3_norm = self.k3 * self.L**4 / (self.E* self.A)



        # phi, beta, and quadrature points
        if self.BC == "SS":
            self.phi = self.phiSS
            self.beta = np.pi * (np.arange(self.nterms) + 1)
            #note self.beta is the same as rtrwalker's more usual self.m

            if not self.nquad is None:
                self.xj = 0.5 * (1 - np.cos(np.pi * np.arange(self.nquad) / (self.nquad - 1)))

            self.BC_coeff = 1# for DingEtal2012 it is two but it is not relevant anymore because i don't multiply the whole equation by 2.

        elif self.BC == "CC" or self.BC == "FF":
            #This is a relic; can only have "SS"
            raise ValueError("'BC' can only be 'SS'")
            def _f(beta):
                return 1 - np.cos(beta) * np.cosh(beta)

            self.beta = find_n_roots(_f, n=self.nterms, x0=0.001, dx=3.14159 / 10, p=1.1)

            if not self.nquad is None:
                self.xj = 0.5*(1 - np.cos(np.pi * np.arange(self.nquad)/(self.nquad - 3)))
                self.xj[0] = 0
                self.xj[1] = 0.0001
                self.xj[-2] = 0.0001
                self.xj[-1] = 0

            self.BC_coeff = 1 # Coefficeint to multiply Fz(vt) and k3*Integral terms by
            if self.BC == "CC":
                self.phi = self.phiCC

            if self.BC == "FF":
                self.phi = self.phiFF
                self.beta[1:] = self.beta[0:-1]
                self.beta[0] = 0.0

        else:
            raise ValueError("only BC='SS', 'CC' and 'FF' have been implemented, you have {}".format(self.BC))


        # quadrature weighting
        if not self.nquad is None:
            rhs = np.reciprocal(np.arange(self.nquad, dtype=float) + 1)
            lhs = self.xj[np.newaxis, :] ** np.arange(self.nquad)[:, np.newaxis]

            self.Ij = np.linalg.solve(lhs, rhs)

        self.vsvdot = np.zeros(2 * self.nterms) #vector of state values for odeint


    def phiSS(self, x, beta):
        return np.sin(beta * x)


    def phiCC(self, x, beta):

        def _xi(beta):
            return (np.cosh(beta) - np.cos(beta))/(np.sinh(beta) - np.sin(beta))

        return (np.cosh(beta * x)
                - np.cos(beta * x)
                + _xi(beta) * (np.sin(beta * x) - np.sinh(beta * x))
                )


    def phiFF(self, x, beta):

        def _xi(beta):

            return (-(np.cos(beta) - np.cosh(beta))/(np.sin(beta) - np.sinh(beta)))

        return (np.cos(beta * x)
                + np.cosh(beta * x)
                + _xi(beta) * (np.sin(beta * x) + np.sinh(beta * x))
                )


    def w(self, qk, x):
        """Nomalised vertcal deformation at x.

        Parameters
        ----------
        qk : 1d ndarray of nterm floats
            Galerkin coefficients.
        x : float
            nomalised distances to calculate deflection at.

        Returns
        -------
        w : float
            vertical deformation at x value.
        """

        return np.sum(qk * self.phi(x, self.beta))


    def vectorfield(self, vsv, tnow, p=()):
        """
        Parameters
        ----------
        vsv :  float
            Vector of the state variables.
            vsv = [q1, q2, ...qk, q1dot, q2dot, ..., qkdot]
            where qk is the kth galerkin coefficient and qkdot is the time
            derivative of the kth Galerkin coefficient.
        tnow : float
            Current time.
        p : various
            Vector of parameters

        Returns
        -------
        vsvdot : vector of state variables first derivatives
            vsvdot = [q1dot, q2dot, ...qkdot, q1dotdot, q2dotdot, ..., qkdotdot]

        """

        q = vsv[:self.nterms]
        qdot = vsv[self.nterms:]

        self.vsvdot[self.nterms:] = np.dot(self.psi_q, q)
        self.vsvdot[self.nterms:] += np.dot(self.psi_qdot, qdot)

        for i in range(self.nterms):
            self.vsvdot[i] = qdot[i]
#            self.vsvdot[self.nterms + i] = - self.mu_norm * qdot[i]
#
#            self.vsvdot[self.nterms + i] -= (self.k1_norm + self.kf**2 * self.beta[i]**4) * q[i]
#

#TODO: this is where the moving loads need to go
#            self.vsvdot[self.nterms + i] += self.BC_coeff * self.Fz_norm * self.phi(self.v_norm * tnow, self.beta[i])
            if tnow>0:
                np.random.rand()

            if self.has_moving_loads:
                for mvl in self.moving_loads_norm:
                    mvl_x, mvl_p = mvl.point_loads_on_beam(t=tnow)
                    self.vsvdot[self.nterms + i] += np.sum(self.BC_coeff * mvl_p * self.phi(mvl_x, self.beta[i]))

            if self.has_stationary_loads:
                for j in range(len(self.stationary_loads_x_norm)):
                    x=self.stationary_loads_x_norm[j]
                    mag_vs_time = self.stationary_loads_vs_t_norm[j]
                    mag = pwise.pinterp_x_y(a=mag_vs_time,xi=tnow)
                    omega_phase = self.stationary_loads_omega_phase_norm[j]
                    if omega_phase is None:
                        self.vsvdot[self.nterms + i] += self.BC_coeff * mag*np.sin(x*self.beta[i])
                    else:
                        omega,phase = omega_phase
                        self.vsvdot[self.nterms + i] += self.BC_coeff * mag*np.sin(x*self.beta[i])*np.cos(omega*tnow+phase)
            if 1:
                # DIY quadrature


                Fj = np.sum(q[:,None] * self.phi(self.xj[None, :], self.beta[:,None]), axis=0)**3
                y = np.sum(self.Ij * Fj * self.k3barj * self.phi(self.xj, self.beta[i]))
            else:
                raise ValueError("Haven't checked this yet")
                # scipy
#                print("yquad = {:12.2g}".format(y))
                y, err = integrate.quad(self.w_cubed_wi, 0, 1, args=(q,i))
#                print("yscip = {:12.2g}".format(y))
            #maybe the quad integrate is not great for the oscillating function
            self.vsvdot[self.nterms + i] -= self.BC_coeff * self.k3_norm * y

        self.vsvdot[self.nterms:] = np.linalg.solve(self.lam_qdotdot, self.vsvdot[self.nterms:])
        return self.vsvdot


    def w_cubed_wi(self, x, q, i):
        """Non-linear cube term for numerical integration"""

        return self.w(q, x)**3 * self.phi(x, self.beta[i])


    def calulate_qk(self, t=None, t_norm=None, **odeint_kwargs):
        """Calculate the nterm Galerkin coefficients qk at each time value

        Parameters
        ----------
        t : float or array of float, optional
            Raw time values.
        t_norm : float or array of float, optional
            Normalised time values. If t_norm==None then it will be
            calculated from raw t values and other params
            t_norm = t / L * sqrt(E / rho).

        Notes
        -----
        This method determines initializes self.t and self.t_norm and
        calculates `self.qsol`.

        """

        if t_norm is None:
            self.t_norm = t / self.L * np.sqrt(self.E / self.rho)
        else:
            self.t_norm = t_norm

        self.t = t

        vsv0 = np.zeros(2*self.nterms) # initial conditions


        self.time_independant_matrices()

        self.qsol = odeint(self.vectorfield,
                      vsv0,
                      self.t_norm,
                      args=(),
                      **odeint_kwargs)

    def time_independant_matrices(self):
        """Make all time independent matrices"""
        self.lam_qdotdot = integ.pdim1sin_abf_linear(m=self.beta,
                                                   a=self.rhobar,
                                                   b=self.Abar,
                                                   implementation=self.implementation)

        self.psi_q = - self.kf**2 * integ.pdim1sin_DD_abDDf_linear(
                                            m=self.beta,
                                            a = self.Ebar,
                                            b = self.Ibar,
                                            implementation=self.implementation)

        self.psi_q -= self.k1_norm * integ.pdim1sin_af_linear(
                                            m=self.beta,
                                            a = self.k1bar,
                                            implementation=self.implementation)

        self.psi_qdot = -self.mu_norm * integ.pdim1sin_af_linear(
                                            m=self.beta,
                                            a=self.mubar,
                                            implementation=self.implementation)

        self.k3barj = pwise.pinterp_x_y(self.k1bar, self.xj)


    def wofx(self, x=None, x_norm=None, tslice=slice(None, None, None), normalise_w=True):
        """Deflection at distance x, and times t[tslice]

        Parameters
        ----------
        x : float or ndarray of float, optional
            Raw values of x to calculate deflection at. Default x=None.
        x_norm : float or array of float, optional
            Normalised x values to calc deflection at.  If x_norm==None then
            it will be calculated frm `x` and other properties : x_norm = x / L.
            Default x_norm=None.
        tslice : slice object, optional
            slice to select subset of time values.  Default
            tslice=slice(None, None, None) i.e. all time values.
            Note the array of time values is already in the object (it was
            used to calc the qk galerkin coefficients).
        normalise_w : True/False, optional
            If True then output is normalised deflection.  Default nomalise_w=True.

        Returns
        -------
        w : array of float, shape(len(x),len(t))
            Deflections at x and self.t_norm[tslice]

        """

        if x_norm is None:
            x_norm = x / self.L

        x_norm = np.atleast_1d(x_norm)

        v = np.zeros((len(x_norm), len(self.tvals_norm[tslice])))

        for i, xx in enumerate(x_norm):
            for j, qq in enumerate(self.qsol[tslice, :self.nterms]):
                v[i, j] = self.w(qq, xx)

        if not normalise_w:
            v *= self.L

        if len(x_norm)==1:
            return v[0]
        else:
            return v


    def runme(self):

        if (self.force_calc) or (not self.load_defl()):
            if self.use_analytical:

                self.time_independant_matrices()



                #adjust matrices so that thaey are all onthe left hand side of the
                # equation.  i.e. analytical equation is arranged differently to
                # numerical equations (and I did the numerical formulation before
                # the analyitcal).
                self.lam_qdotdot*=1  # rho*A
                self.psi_q*=-1 #kf**2 * E*I +++++ k1
                self.psi_qdot*=-1 #mu


                # make the block beta.  ONly the second nterms elements will be relevant in calcs.
                self.beta_block = np.empty(2 * self.nterms, dtype=float)
                self.beta_block[:self.nterms] = self.beta
                self.beta_block[self.nterms:] = self.beta

                self._make_gam()
                self._make_psi()

                self._make_eigs_and_v()
                self.make_time_dependent_arrays()
                self.make_output()


            else:
                #using numerical answer
                self.calulate_qk(t_norm=self.tvals_norm)
                self.defl_norm = self.wofx(x_norm=self.xvals_norm, normalise_w=True)
                #Should be array of shape (len(self.xvals_norm), len(self.tvals_norm))

        if not self.L is None:
            self.defl = self.defl_norm * self.L


    def _make_gam(self):
        """Make the 2n =*2n block gamma matrix


        Used with `use_analytical`=True
        """

        #self.m_block = np.empty(2 * self.neig, dtype=float)
        #self.m_block[:self.neig] = self.m
        #self.m_block[self.neig:] = self.m


        self.gam = np.zeros((2*self.nterms, 2*self.nterms), dtype=float)


        self.gam[:self.nterms, :self.nterms] = np.eye(self.nterms) #top left
        self.gam[:self.nterms, self.nterms:] =  0 #top right
        self.gam[self.nterms:, :self.nterms] =  0 #bottom left
        self.gam[self.nterms:, self.nterms:] = self.lam_qdotdot[:,:]#botom right


        #self.gam[np.abs(self.gam)<1e-8] = 0.0

#        self.lam_qdotdot #+ve
#
#        self.psi_q #-ve
#
#        self.psi_qdot*=1 #-ve

        return

    def _make_psi(self):
        """Make the 2n =*2n block psi matrix

        Used with `use_analytical`=True"""

        self.psi = np.zeros((2*self.nterms, 2*self.nterms), dtype=float)

        self.psi[:self.nterms, :self.nterms] = 0 #top left
        self.psi[:self.nterms, self.nterms:] =  -np.eye(self.nterms) #top right
        self.psi[self.nterms:, :self.nterms] =  self.psi_q #bottom left
        self.psi[self.nterms:, self.nterms:] = self.psi_qdot#botom right


        #self.psi[np.abs(self.psi)<1e-8] = 0.0
        return

    def _make_eigs_and_v(self):
        """make Igam_psi, v and eigs, and Igamv

#TODO: make sure this is all still relevatn after copyint from speccon1d_unsat


        Finds the eigenvalues, `self.eigs`, and eigenvectors, `self.v` of
        inverse(gam)*psi.  Once found the matrix inverse(gamma*v), `self.Igamv`
        is determined.

        Notes
        -----
        From the original equation

        .. math:: \\mathbf{\\Gamma}\\mathbf{A}'=\\mathbf{\\Psi A}+loading\\:terms

        `self.eigs` and `self.v` are the eigenvalues and eigenvegtors of the matrix `self.Igam_psi`

        .. math:: \\left(\\mathbf{\\Gamma}^{-1}\\mathbf{\\Psi}\\right)

        """

#        self.psi[np.abs(self.psi) < 1e-8] = 0.0
        Igam_psi = np.dot(np.linalg.inv(self.gam), self.psi)
        self.eigs, self.v = np.linalg.eig(Igam_psi)
        self.v = np.asarray(self.v)
        self.Igamv = np.linalg.inv(np.dot(self.gam, self.v))

        if False:
            for i, eig in enumerate(self.eigs):
                print("{:s}, {:12.6f}".format(str(i+1).zfill(3),eig))

    def make_time_dependent_arrays(self):
        """make all time dependent arrays

        See also
        --------
        self.make_E_Igamv_the()

        """

        self.make_E_Igamv_the()
        self.v_E_Igamv_the = np.dot(self.v, self.E_Igamv_the)



    def make_E_Igamv_the(self):
        """sum contributions from all loads

        Calculates all contributions to E*inverse(gam*v)*theta part of solution
        defl=phi*vE*inverse(gam*v)*theta. i.e. surcharge, vacuum, top and bottom
        pore pressure boundary conditions. `make_E_Igamv_the` will create
        `self.E_Igamv_the`.  `self.E_Igamv_the`  is an array
        of size (nterms, len(tvals)). So the columns are the column array
        E*inverse(gam*v)*theta calculated at each output time.  This will allow
        us later to do defl = phi*v*self.E_Igamv_the

        See also
        --------
        _make_E_Igamv_the_mvl :  moiving point load contribution

        """


        self.E_Igamv_the = np.zeros((2*self.nterms, len(self.tvals_norm)), dtype=complex)

        if self.has_moving_loads:
            self._make_E_Igamv_the_mvpl() #moving point loads
            self.E_Igamv_the += self.E_Igamv_the_mvpl
        if self.has_stationary_loads:
            self._make_E_Igamv_the_stationary() #moving point loads
            self.E_Igamv_the += self.E_Igamv_the_stationary

        return

    def _make_E_Igamv_the_mvpl(self):
        """Moving point load contribution to self.E_Igam_the

        Makes `self.E_Igamv_mvpl`  which is an array
        of size (nterms, len(tvals))

        """

        self.E_Igamv_the_mvpl = speccon1d.dim1sin_E_Igamv_the_mvpl(
                            self.beta_block,
                            self.eigs,
                            self.tvals_norm,
                            self.Igamv,
                            self.moving_loads_norm,
                            dT=1.0,
                            theta_zero_indexes=slice(None, self.nterms),
                            implementation=self.implementation)

        return

    def _make_E_Igamv_the_stationary(self):
        """Stationary point load contribution to self.E_Igam_the

        Makes `self.E_Igamv_stationary`  which is an array
        of size (nterms, len(tvals))

        """
        pseudo_k = [1 for v in self.stationary_loads_x_norm]

        self.E_Igamv_the_stationary = speccon1d.dim1sin_E_Igamv_the_deltamag_linear(
                            self.beta_block,
                            self.eigs,
                            self.tvals_norm,
                            self.Igamv,
                            self.stationary_loads_x_norm,
                            pseudo_k,
                            self.stationary_loads_vs_t_norm,
                            self.stationary_loads_omega_phase_norm,
                            dT=1,
                            theta_zero_indexes=slice(None, self.nterms),
                            implementation=self.implementation)
        return

    def make_output(self):
        """make all output"""

        self._make_defl()

    def _make_defl(self):
        "Make deflection and deflection rate"""

        self.defl_norm = speccon1d.dim1sin_f(self.beta, self.xvals_norm,
            self.tvals_norm,
            self.v_E_Igamv_the[:self.nterms, :],
            drn=0)

        self.defl_norm = np.atleast_2d(self.defl_norm)

        #check if complex part is small
        if np.allclose(np.imag(self.defl_norm),0,atol=1e-12):
            self.defl_norm = np.real(self.defl_norm)
        else:
            raise ValueError('Imaginary parts of self.defl are > 1e-12')

        if not self.L is None:
            self.defl = self.defl_norm * self.L




        return

    def saveme(self):
        """Save deflection vs time to file

        Deflection will be saved to :
        'self.file_stem + "_defl.csv'
        Normalised deflection will be saved to:
        'self.file_stem + "_defl_norm.csv"

        Notes:
        -----
        Might not handle singel x value and single t vlue.

        """
#TODO: Need to account for single xvalue and single y value

        two_d_defl_norm = np.atleast_2d(self.defl_norm)


        inputoutput.save_grid_data_to_file(data_dicts=dict(
                                               name="_defl_norm",
                                               data=two_d_defl_norm,#self.defl_norm,
                                               row_labels=self.xvals_norm,
                                               row_labels_label="x_norm",
                                               column_labels=self.tvals_norm),
                                           file_stem=self.file_stem,
                                           create_directory=False)
        if not self.L is None:
            two_d_defl = np.atleast_2d(self.defl)
            inputoutput.save_grid_data_to_file(data_dicts=dict(
                                               name="_defl",
                                               data=two_d_defl,#self.defl,
                                               row_labels=self.xvals,
                                               row_labels_label="xvals",
                                               column_labels=self.tvals),
                                           file_stem=self.file_stem,
                                           create_directory=False)

    def load_defl(self):
        """Loads defl_norm from txt file if it exists

        Returns
        -------
        Loaded : boolean
            If True then self.defl either already exists or has been
            successfully loaded.


        """

#        fname = self.file_stem + "_defl_norm.csv"

        if hasattr(self, "defl_norm"):
            #defl_norm already exists in memory
            return True


        #check if out file exists
        fname = self.file_stem + "_defl_norm.csv"



        if not os.path.isfile(fname):
            return False

        try:
            with open(fname,'r') as f:
                lines = f.readlines()


            t_norm = np.array([float(v) for v in lines[0].split(",")[2:]])
            a = np.atleast_2d(np.genfromtxt(fname=fname,skip_header=1, dtype=float,delimiter=","))
            x_norm = a[:, 1]
            defl_norm = a[:, 2:]

            self.defl_norm = defl_norm

            if not self.L is None:
                self.defl = self.defl_norm * self.L
                self.xvals = self.xvals_norm * self.L

            print('loaded it')
            return True
        except:
            return False
    def onClick(self, event):
        if self.pause:
            self.ani.event_source.stop()
        else:
            self.ani.event_source.start()
        self.pause ^= True

    def plot_w_vs_x_overtime(self, ax=None, norm=True):
        """Plot the normalised deflection vs distance for all times

        """

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        self.load_defl()

        if norm:
            for t, defl in zip(self.tvals_norm, self.defl_norm.T):
                ax.plot(self.xvals_norm, defl,
                        label = "t={:5.2g}".format(t))
            ax.set_xlabel("x_norm")
            ax.set_ylabel("w_norm")
        else:
            for t, defl in zip(self.tvals, self.defl.T):
                ax.plot(self.xvals, defl,
                        label = "t={:5.2g}".format(t))
            ax.set_xlabel("x")
            ax.set_ylabel("w")

        leg = ax.legend()
        leg.draggable()


        return ax


    def animateme(self, xlim=None, ylim=None, norm=True, saveme=False,interval=50):
        """Animate the deflection vs distance over time plot

        Will display beam osciallations and evolving max and min deflection
        envelopes.  Position and magnitude of load will also be shown.

        Parameters
        ----------
        xlim : tuple of 2 floats, optional
            Limits of x-axis , default xlim=None, i.e. use Beam ends
        ylim : tuple of 2 floats, optional
            Limits of y-axis , default ylim=None, i.e. limits will be
            calculated ylim=(1.5*defl_min,1.1*defl_max)
        norm : True/False, optional
            If norm=True normalised values will be plotted. Default norm=True.
        saveme : False/True
            Whether to save the animation to disk. Default saveme=False.
        interval : float
            Number of miliseconds for each frame.  Default interval=50.

        """

        self.pause = False

        self.load_defl()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax2 =ax.twinx()

        if norm:
            xx = self.xvals_norm
            tt = self.tvals_norm
            defl = self.defl_norm
            if self.has_moving_loads:
                moving_loads = self.moving_loads_norm
            if self.has_stationary_loads:
                stationary_loads_x = self.stationary_loads_x_norm
                stationary_loads_vs_t = self.stationary_loads_vs_t_norm
                stationary_loads_omega_phase = self.stationary_loads_omega_phase_norm
            ax.set_xlabel("x_norm")
            ax.set_ylabel("w_norm")
            ax2.set_ylabel("Fz_norm")
        else:
            xx = self.xvals
            tt = self.tvals
            defl = self.defl
            if self.has_moving_loads:
                moving_loads = self.moving_loads
            if self.has_stationary_loads:
                stationary_loads_x = self.stationary_loads_x_norm
                stationary_loads_vs_t = self.stationary_loads_vs_t_norm
                stationary_loads_omega_phase = self.stationary_loads_omega_phase_norm
            ax.set_xlabel("x")
            ax.set_ylabel("w")
            ax2.set_ylabel("Fz")

        defl_min = np.min(defl)
        defl_max = np.max(defl)

        x_min = np.min(xx)
        x_max = np.max(xx)

        if not xlim is None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(x_min,x_max)
        if not ylim is None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(2*defl_min,1.5*defl_max)
            ax.set_ylim(1.5*defl_min,1.1*defl_max)



        max_defl_evolve = np.zeros_like(defl)
        min_defl_evolve = np.zeros_like(defl)


        max_defl_evolve[:,0] = defl[:,0]
        min_defl_evolve[:,0] = defl[:,0]
        for i in range(len(tt)):
            if i==0:
                continue
            max_defl_evolve[:,i] = np.maximum(defl[:,i,], max_defl_evolve[:,i-1])
            min_defl_evolve[:,i] = np.minimum(defl[:,i,], min_defl_evolve[:,i-1])


        ax.grid()
        ax.invert_yaxis()
        #moving point load



        data_line, = ax.plot((np.nan,np.nan),(np.nan,np.nan),
                            linestyle="-",
                            color='blue')

        min_line, = ax.plot((np.nan,np.nan),(np.nan,np.nan),
                            linestyle="-",
                            color='orange')
        max_line, = ax.plot((np.nan,np.nan),(np.nan,np.nan),
                            linestyle="-",
                            color='cyan')

        pmax=1e-50
        pmin=1e50




        p_scale_mvl=1e-50
        if self.has_moving_loads:
            # mvpl1_line1,mvpl1_line2,..., mvpl2_line1, mvpl2_line2, mvpl_line3, ...
            mvpl_lines = [ax2.annotate("", xy=(np.nan,np.nan), xytext=(np.nan, np.nan),
                             arrowprops=dict(facecolor='red', shrink=0.0)
                                             #arrowstyle="->",
                                             #edgecolor='red',
                                             #shrinkline=0
                                             #)
                             ) for mvpl in moving_loads for j in mvpl.xt]

            p_scale_mvl = abs(max([p for mvpl in moving_loads for p in mvpl.p]))
            pmax = max(pmax,max([p for mvpl in moving_loads for p in mvpl.p]))
            pmin = min(pmin,min([p for mvpl in moving_loads for p in mvpl.p]))
        else:
            mvpl_lines=[]


        p_scale_stationary=1e-50
        if self.has_stationary_loads:


            x_map = np.zeros(len(stationary_loads_x), dtype=int)
            x_map[0] = 0


            for i in range(1,len(stationary_loads_x)):
                for j in range(i+1):
                    if np.isclose(stationary_loads_x[i], stationary_loads_x[j]):
                        x_map[i]=j
                        break


            stationary_lines=[ax2.annotate("", xy=(np.nan,np.nan), xytext=(np.nan, np.nan),
                             arrowprops=dict(facecolor='green', shrink=0.0)
                                             #arrowstyle="->",
                                             #edgecolor='red',
                                             #shrinkline=0
                                             #)
                             ) for v in range(max(x_map)+1)]#stationary_loads_x]
            p_scale_stationary = abs(max([p for mag_vs_time in stationary_loads_vs_t for p in mag_vs_time.y]))

            pmin_temp=[]
            pmax_temp=[]
            for i in range(max(x_map)+1):
                sum_mag_max=0
                sum_mag_min=0
                for j in range(len(stationary_loads_x)):
                    if x_map[j]!=i:
                        continue
                    mag_max = max(stationary_loads_vs_t[j].y)
                    mag_min = min(stationary_loads_vs_t[j].y)
                    sum_mag_max +=mag_max
                    sum_mag_min +=mag_min
                pmin_temp.append(sum_mag_min)
                pmax_temp.append(sum_mag_max)

            pmax = max(pmax, max(pmax_temp))
            pmin = min(pmin, min(pmin_temp))

        else:
             stationary_lines=[]



        ax2.set_ylim(-2*max(abs(pmin) , abs(pmax)),1.5*max(abs(pmin) , abs(pmax)))

#        align_yaxis(ax, 0, ax2, 0)
        align_yaxis(ax,ax2)



        p_scale = max(p_scale_mvl, p_scale_stationary)
        p_scale *=2/abs(defl_min)/3 # max load will plot to half the scale

        time_template = 'time = {:.4f}'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        fig.tight_layout()

        def init():
            i=0

            if self.has_moving_loads:
                for mvpl in moving_loads:
                    for j in range(len(mvpl.xt)):
                        mvpl_lines[i].xy = (np.nan,np.nan)
                        mvpl_lines[i].xyann = (np.nan,np.nan)
                        i=i+1
            if self.has_stationary_loads:
                for j in range(max(x_map)+1):#len(stationary_loads_x)):
                    stationary_lines[j].xy = (np.nan,np.nan)
                    stationary_lines[j].xyann = (np.nan,np.nan)

            data_line.set_data([], [])
            min_line.set_data([], [])
            max_line.set_data([], [])
            time_text.set_text('')

            return tuple(stationary_lines) + tuple(mvpl_lines) + (data_line,) + (time_text,) + (min_line,)+(max_line,)


        def animate(frame):
            i=0
            if self.has_moving_loads:
                for mvpl in moving_loads:
                    if mvpl.v<0:
                        x0 = mvpl.x0 + mvpl.L
                    else:
                        x0 = mvpl.x0
                    thisx = mvpl.axle_positions(x0=x0, t=tt[frame],v0=mvpl.v,t0=mvpl.t0)
                    #thisy = -mvpl.p / p_scale#defl_max/5
                    thisy=mvpl.p
                    for j in range(len(mvpl.xt)):
                        mvpl_lines[i].xy = (thisx[j],0)
                        mvpl_lines[i].xyann = (thisx[j], thisy[j])
                        i=i+1

            if self.has_stationary_loads:
                for j in range(max(x_map)+1):#len(stationary_loads_x)):
                    sum_mag = 0
                    for k in range(len(stationary_loads_x)):
                        if x_map[k] != j:
                            continue

                        mag = pwise.pinterp_x_y(a=stationary_loads_vs_t[k],xi=tt[frame])
                        if stationary_loads_omega_phase[k] is None:
                            pass
                        else:
                            omega, phase = stationary_loads_omega_phase[k]
                            mag*=np.cos(omega*tt[frame]+phase)
                        sum_mag +=mag
                    stationary_lines[j].xy = (stationary_loads_x[j],0)
                    #stationary_lines[j].xyann = (stationary_loads_x[j], -mag/p_scale)
                    stationary_lines[j].xyann = (stationary_loads_x[j], sum_mag)



            min_line.set_data(xx, min_defl_evolve[:,frame])
            max_line.set_data(xx, max_defl_evolve[:,frame])

            data_line.set_data(xx, defl.T[frame])
            time_text.set_text(time_template.format(tt[frame]))
            return tuple(stationary_lines) + tuple(mvpl_lines) + (data_line,) + (time_text,) + (min_line,)+(max_line,)

        self.ani = animation.FuncAnimation(fig, animate, len(tt),
                                      interval=interval, blit=True, init_func=init,repeat=True)
        fig.canvas.mpl_connect('button_press_event', self.onClick)
#        anim_running = True
#
#        def onClick(event):
#            nonlocal anim_running
#            if anim_running:
#                ani.event_source.stop()
#                anim_running = False
#            else:
#                ani.event_source.start()
#                anim_running = True



        if saveme:
            pass
            self.ani.save(self.file_stem+"_anim.mp4", fps=15,codec="libx264")#changing the codec seems to work for me.
#            ani.save(self.file_stem+"_anim", fps=15)#changing the codec seems to work for me.

        return self.ani

    def defl_vs_time_under_loads(self, xlim_norm=(0,1)):
        """Deflection vs time under each moving load

        i.e. What is deflection under each axle

        Creates and populates self.moving_load_w_norm .
        Normalised deflections under each moving load at each time value.
        Each element of the list corresponds to each moving load.  The array
        is of size (len(mvpl.x), len(tvals))  i.e. for each axle in the moving
        load group.

        Parameters
        ----------
        xlim_norm : tuple, optional
            won't show defelction untill within these normalised bounds
            Default xlim_norm=(0,1)

        """

        self.moving_load_w_norm = []

        for i, mvpl in enumerate(self.moving_loads_norm):
            if mvpl.v<0:
                x0 = mvpl.x0 + mvpl.L
            else:
                x0 = mvpl.x0

            wvals = np.zeros((len(mvpl.x), len(self.tvals_norm)))

            for j, t in enumerate(self.tvals_norm):
                xnow = mvpl.axle_positions(x0=x0, t=t,v0=mvpl.v, t0=mvpl.t0)
                wnow = np.interp(xnow, self.xvals_norm, self.defl_norm[:,j])

                #remove points where load is not on beam.
                wnow[xnow<xlim_norm[0]] = np.nan
                wnow[xnow>xlim_norm[1]] = np.nan

                wvals[:,j] = wnow

            self.moving_load_w_norm.append(wvals)


        if not self.L is None:
            self.moving_load_w = [v*self.L for v in self.moving_load_w_norm]

        return

class MovingPointLoads(object):
    """Multi-axle vehicle represented by point loads in a line

    Parameters
    ----------
    x : 1d array or list
        x-coord of point loads.  Vehicle assumed to start at 0.
    p : 1d array or list
        Magnitude of point loads at corresponding x value.
    offset : float, optional
        x-coords will be offset by this distance.  Positive values will move
        x to the right (i.e. increase in x).
    v : float, optional
        Velocity.  Default v = None, i.e. you'll need to enter in various
        functions.
    L : float, optional
        Length of Beam.  Default L=None i.e. you'll need to explicitly
        enter in various functions. You can use a psuedo L with x0 to have the
        loads begin and end midspan.
    t0 : float, optional
        Time the point loads begin at x0 on the beam. default t0=0 ie. you'll
        have to explicitly enter in various functions.
    x0 : float, optional
        Start position of loads on beam. Defualt x0=0.0 .

    Attributes
    ----------
    xt : 1d array
        transformed x coords after mirror and offset are applied.
    start_time, end_time : 1d array
        start and end time for each point load traversing length L.

    """

    def __init__(self, x, p, offset=0,v=None,L=None,t0=0,x0=0.0):

        self.x = np.array(x, dtype=float)#array(x.copy())
        self.p = np.array(p, dtype=float)#np.asarray(p.copy())

        self.offset = offset

        self.xt = self.x.copy()
        self.xt += offset

        self.L = L
        self.t0 = t0
        self.v = v
        self.x0 = x0

        self.xnow = np.zeros_like(self.xt)
        self.pnow = np.zeros_like(self.xt)

        return


    def setv(self, v):
        """Set the vehicle's velocity, then calc start_time and end_time when
        vehicle's axles enter and leave the beam"""
        self._v = v
        if not v is None:
            if v > 0:
                self.start_time = self.t0 + (0 - self.xt) / v
                self.end_time = self.t0 + (self.L - self.xt) / v
            else:
                #train is travelling in negative direction
                # the (-1) is to mirror around reference point
                self.start_time = self.t0 - (-1)*self.xt / v
                self.end_time = self.t0 - (self.L + (-1)*self.xt) / v
        else:
            self.start_time = None
            self.end_time = None

        return

    def getv(self):
        return self._v

    def delv(self):
        del self._v
        del self.start_time
        del self.end_time
        return

    v = property(getv, setv, delv, "Velocity")

    def point_loads_on_beam(self, t):
        """Position and magnitude of point loads between [0, L] at time t

        Parameters
        ----------
        t : float
            Current time.

        Returns
        -------
        x : 1d array of float
            position of each load at time t.
        p : 1d array of float
            Load at time t.  Will be set to zero if load is not beteen
            0 and L.

        Notes
        -----
        Position can be offset by self.x0

        """

        if self.v >= 0:
            self.xnow[:] = self.axle_positions(t=t, v0=self.v, x0=self.x0)
        else:
            self.xnow[:] = self.axle_positions(t=t, v0=self.v, x0=self.L)

        self.pnow[:] = self.p

        self.pnow[((t< self.start_time) | (t>self.end_time))] = 0
        return self.xnow, self.pnow


    def convert_to_specbeam(self, v=None, L=None, t0=None, x0=None):
        """Convert vehicle at velocity v into
        surcharge_vs_time and surcharge_omega_phase PolyLines for use with
        spectral method.

        Parameters
        ----------
        v : float, optional
            Velocity of vehicle.  Default v=None in which case the v used
            during object initialization will be uses
        L : float, optional
            Length of beam.  Default L=None in which case the L used
            during object initialization will be uses
        t0 : float, optional
            time when vehicle reference point begins entering the beam. Default t0=0.
            If v>0 vehicle will enter from the right.  If v<0 vehicle
            will enter from the right. v=0 will result in an error.
        x0 : float, optional
            Position where loads start.  Default x0=none, i.e. value used
            during initialization will be used.

        Notes
        -----

        Vehicle is made up of axles at a relative distance dxi from a
        reference point.  dxi should be negative.  If velocity v is positive
        then the reference point enters the beam (at x=x0) at time t0.  The
        position of the ith axle is:
            xi = x0 + dxi + v * (t - t0)
        The ith axle enters the beam at position x0 at time
            tstarti = t0 - dxi / v
        and leaves the beam (x = x0 + L) at time
            tendi = t0+(L-dxi)/v


        If v is negative then the reference point enters the beam from
        the right, then we first flip the vehicle by making dx_i=-dxi.  The
        positiion and times entering and leaving the beam are
            xi = x0 + L + dx_i + v * (t - t0)

        and
            tstart = t0 - dx_i / v
            tend = t0 - (L + dx_i)/v

        For use in spec beam we will need sin(M*xi) in a form
        sin(omega*t+phase).

        for v>0
        xi = v * t + (x0 + dxi - v * t0)
        so omega/M = v and phase/M = x0 + dxi-v*t0

        for v<0
        xi = v * t + (x0 + L + dx_i - v * t0)
        so omega/M = v and phase/M = x0 + L + dx_i - v*t0

        Returns
        -------
        plines : list of PolyLine
        omega_phase : list of two element tuples
            frequency and phase angle for use in spec beam.  as per the notes
            above will need to multiply both omega and phase by M for use
            in specbeam.


        """

        if v is None:
            v = self.v
        if L is None:
            L = self.L
        if t0 is None:
            t0 = self.t0
        if x0 is None:
            x0 = self.x0

        plines=[]
        omega_phase=[]
        if v==0:
            raise ValueError("Can not have v=0 i.e. no static loads.")

        if v > 0:
            for xi, p in zip(self.xt, self.p):
                start_time = t0 + (0 - xi) / v
                end_time = t0 + (L - xi) / v
                plines.append(PolyLine([start_time,start_time,end_time,end_time],
                                       [0, p, p,0]))
                omega_phase.append((v, xi + x0 - v * t0))



        else:
            #train is travelling in negative direction
            for xi, p in zip(self.xt, self.p):
                xi *= -1 #mirror around reference point.
                start_time = t0 - xi / v
                end_time = t0 - (L + xi) / v
                plines.append(PolyLine([start_time,start_time,end_time,end_time],
                                       [0, p, p, 0]))
                omega_phase.append((v, x0 + L + xi - v*t0))
        return plines, omega_phase

    def plot_specbeam_PolyLines(self, ax=None, v=1, L=1, t0=0, x0=0):
        """Plot load_vs_time for each vehicle axis.

        Parameters
        ----------
        ax : Matplotlib.Axes, optional
            Axes to plot on.  Default ax=None i.e. a new fig and ax will be
            created.
        v : float, optional
            velocity, default v=1.
        L : float, optional
            Lenght of beam default L=1
        x0, t0 : float, optional
            Vehicle starts at positon x0 at time t0. Default x0=0, t0=0

        """

        plines, omega_phases = self.convert_to_specbeam(v=v, L=L, t0=0, x0=x0)

        if ax is None:
            fig, ax = plt.subplots()

        for i in range(len(plines)):
            omega,phase = omega_phases[i]
            xx = plines[i].x
            yy=plines[i].y
            label = "$i={},\\omega={},\\phi={}$".format(i,omega,phase)
            label = "${},{},{}$".format(i,omega,phase)
            ax.plot(xx,yy,label=label)

        ax.set_title("L={},v={},t0={},x0={}".format(L,v,t0,x0))
        ax.set_xlabel("time")
        ax.set_ylabel("magnitude")
        leg = ax.legend(title ="$i,\\omega,\\phi$",
                        labelspacing=0)
        leg.draggable(0)

        return ax

    def axle_positions(self, t=0, v0=0, t0=0, x0=0):
        """Positions of axles at given time

        Parameters
        ----------
        t : float
            Time of interest
        v0 : float, optional
            Initial speed of vehicle.  Default v0=0.
            If v0<0 then axle coords will be multiplied by -1 before moving.
        t0 : float, optional
            Time when vehicle starts moving at velocity v0. Default t0=0
        x0 : float, optional
            Start postion of vehicle

        Returns
        -------
        xnew : 1d array
            xcoords of axles at time t.

        """

        self.xnow[:]=self.xt[:]#xnew = self.xt.copy()
        if v0<0:
            self.xnow*=-1

        self.xnow+= x0 + v0*(t-t0)

        return self.xnow

    def plotme(self, ax=None,t=0, v0=0, t0=0, x0=0, xlim=None, ylim=None):
        """Plot the load at time t

        Each axle load is represented by an arrow, with head at 0 and tail
        at the load magnitude.

        Parameters
        ----------
        ax : pyplot.Axes, optional
            Axes object to plot on.  Default ax=None i.e. new figure and
            axes will be created.
        t : float, optional
            Time at which to plot the vehicle positoin. Default t=0.
        v0 : float, optional
            Initial velocity of vehicle.  default v0=0.  If v0<0 then
            positions will be multiplied by -1 before adjusting position.
        t0 : float, optional
            Time the vehicle starts moving. Default t0=0.
        x0 : float, optional
            Start position of vehicle
        xlim, ylim : tuple of size 2, default xlim=ylim=None.
            axes limits.

        Returns
        -------
        ax : pyplot.Axes opbect
           Axes object with plotted vehicle position.


        """

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        xnew = self.axle_positions(t=t, v0=v0, t0=t0, x0=x0)

        lines = [ax.annotate("", xy=(xnew[i], 0), xytext=(xnew[i], self.p[i]),
                             arrowprops=dict(facecolor='red', shrink=0.0)
                                             #arrowstyle="->",
                                             #edgecolor='red',
                                             #shrinkline=0
                                             #)
                             ) for i in range(len(self.xt))]

        if not xlim is None:
            ax.set_xlim(xlim)
        if not ylim is None:
            ax.set_ylim(ylim)

        return ax

    def animateme(self, x0=0, t0=0, tf=1,v0=1, xlim=None, ylim=None):
        """Animate moving vehicle

        Parameters
        ----------
        x0 : float, optional
            Start position of vehicle, default=0.
        t0 : float, optional
            Start time of animation. Default t0=0.
        tf : float, optional
            End time of animation. Default tf=1.
        v0 : float, optional
            Initial velocity of vehicle.  Default v0=1.  If v0<0 then
            positions will be multiplied by -1 before adjusting position.
        xlim, ylim : tuple of size 2, default xlim=ylim=None.
            axes limits.

        Returns
        -------
        ani : matplotlib animation
           Animation of moving vehicle


        """

        #http://stackoverflow.com/questions/15887820/animation-by-using-matplotlib-errorbar

        fig = plt.figure()
        ax = fig.add_subplot(111)
        if not xlim is None:
            ax.set_xlim(xlim)
        if not ylim is None:
            ax.set_ylim(ylim)


        ax.grid()


        lines = [ax.annotate("", xy=(np.nan,np.nan), xytext=(np.nan, np.nan),
                             arrowprops=dict(facecolor='red', shrink=0.0)
                                             #arrowstyle="->",
                                             #edgecolor='red',
                                             #shrinkline=0
                                             #)
                             ) for i in range(len(self.xt))]


        time_template = 'time = {:.1f}'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


        def init():

            time_text.set_text("")
            for i in range(len(self.xt)):
                lines[i].xy = (np.nan,np.nan)
                lines[i].xyann = (np.nan,np.nan)

            time_text.set_text('')
            return tuple(lines)+(time_text,)


        def animate(t):

            thisx = self.axle_positions(t=t,v0=v0,t0=t0,x0=x0)
            thisy = self.p

            for i in range(len(self.xt)):
                lines[i].xy = (thisx[i], 0)
                lines[i].xyann = (thisx[i],thisy[i])

            time_text.set_text(time_template.format(t))
            return tuple(lines) + (time_text,)

        ani = animation.FuncAnimation(fig, animate, np.linspace(t0,tf,100),
                                      interval=50, blit=True, init_func=init,repeat=True)

        return ani

    def time_axles_pass_point(self, x, v0=0, t0=0, x0=0):
        """time values when each moving load passes a certain point

        Parameters
        ----------
        x : float
            x-coord of interest
        v0 : float, optional
            Initial speed of vehicle.  Default v0=0.
            If v0<0 then axle coords will be multiplied by -1 before moving.
        t0 : float, optional
            Time when vehicle starts moving at velocity v0. Default t0=0
        x0 : float, optional
            Start postion of vehicle
        L : float, optional
            Length of beam.  Default L=None in which case the L used
            during object initialization will be uses.

        Returns
        -------
        t_pass : 1d array of float
            list of times corresponding to when each axle passes the given
            x value.

        Notes
        -----

        """

        self.xnow[:]=self.xt[:]
        if v0<0:
            self.xnow*=-1
        return (x-self.xnow-x0)/v0 + t0

    def time_axles_pass_beam_point(self,x):
        """Time values when each axle passes a particular point on a beam

        Parameters
        ----------
        x : float
            x coord of interest

        Returns
        -------
        t_pass : 1d aray of float
            time values when each axle passes the particualr point

        Notes
        -----
        If self.v < 0 then loads enter beam from the left, i.e. at time
        t0 axles will be at L+x0+xt

        Not really sure if I've accountered for self.x0!=0 for the
        negative velocity.  Just make self.x0 =0 for simplicity
        """

        if self.v>0:
            t_pass = self.time_axles_pass_point(x, v0=self.v, t0=self.t0, x0=self.x0)
        elif self.v<0:
            t_pass = self.time_axles_pass_point(x, v0=self.v, t0=self.t0, x0=(self.L+self.x0))
        else:
            raise ValueError("v can't be zero")


        return t_pass





##***Don't think I need this combo of align_yaxis & adjust_yaxis, I use a
##***simpler version of align_yaxis that is hardcoded to align at y=0 on both
##***axes
#def align_yaxis(ax1, v1, ax2, v2):
#    """Adjust y-axis bounds so that value v1 in ax1 is aligned to value v2
#    in ax2.
#
#    Should keep keep all data visible.
#
#    Parameters
#    ----------
#    ax1, ax2 : matplotlib.Axes
#        Axes objects in which to align particular y-axis values
#    v1, v2 : float
#        y-axis values on which to align.
#
#    See Also
#    --------
#    adjust_yaxis : helper routine to move shift yaxis
#
#    Notes
#    -----
#    Copied from http://stackoverflow.com/questions/10481990/matplotlib-axis-with-two-scales-shared-origin
#
#    """
#
#    _, y1 = ax1.transData.transform((0, v1))
#    _, y2 = ax2.transData.transform((0, v2))
#    adjust_yaxis(ax2,(y1-y2)/2,v2)
#    adjust_yaxis(ax1,(y2-y1)/2,v1)
#
#    return
#
#
#def adjust_yaxis(ax, ydif, v):
#    """Adjust y-axis bounds by shifting ydif but keeping value v in the same
#    visual location.
#
#    Parameters
#    ----------
#    ax : Matplotlib.Axes
#        Axes object to change
#    ydif : float
#
#    See Also
#    --------
#    align_yaxis : parent routine
#
#    Notes
#    -----
#    Copied from: http://stackoverflow.com/questions/10481990/matplotlib-axis-with-two-scales-shared-origin
#
#    """
#
#    inv = ax.transData.inverted()
#    _, dy = inv.transform((0, 0)) - inv.transform((0, ydif))
#    miny, maxy = ax.get_ylim()
#    miny, maxy = miny - v, maxy - v
#    if -miny>maxy or (-miny==maxy and dy > 0):
#        nminy = miny
#        nmaxy = miny*(maxy+dy)/(miny+dy)
#    else:
#        nmaxy = maxy
#        nminy = maxy*(miny+dy)/(maxy+dy)
#    ax.set_ylim(nminy+v, nmaxy+v)
#
#    return


def align_yaxis(ax1, ax2):
    """Adjust y-axis limits so zeros of the two axes align, zooming them out
    by same ratio.

    Parameters
    ----------
    ax1, ax2 : matpolotlib.Axes
        Axes to align.

    """


    axes = (ax1, ax2)
    extrema = [ax.get_ylim() for ax in axes]
    tops = [extr[1] / (extr[1] - extr[0]) for extr in extrema]
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [list(reversed(l)) for l in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    b_new_t = extrema[0][0] + tot_span * (extrema[0][1] - extrema[0][0])
    t_new_b = extrema[1][1] - tot_span * (extrema[1][1] - extrema[1][0])
    axes[0].set_ylim(extrema[0][0], b_new_t)
    axes[1].set_ylim(t_new_b, extrema[1][1])


def DAFrelative(alp, bet, k2_k1):
    """Relative Dynamic Amplification Factor, based on ratio of beam
    stiffness to some reference beam stiffness.

    Parameters
    ----------
    alp : float
        velocity ratio. alp=v/vcr; vcr= SQRT(2*SQRT(k*EI)/(rho*A))
    bet : float
        damping ratio.  bet=c/ccr; ccr=2*SQRT(rho*A*k)
    k2_k1 : float
        stiffness ratio of beam to current beam

    Returns
    -------
    DAF : float
        Relative Dynamic Amplification Factor

    Notes
    -----
    DAFrelative is a terse implementation for an appendix of code in
    Walker, R.T.R. and Indraratna, B, (in press) "Moving loads on a
    viscoelastic foundation with special reference to railway
    transition zones". International Journal of Geomechanics.

    """
    import numpy as np
    alp /= k2_k1**(0.25)
    bet *= (1/k2_k1)**0.5

    gam = np.roots([1.0, 0.0, 4.0 * alp**2, -8 * alp * bet, 4])
    cmat = np.zeros((4, 4), dtype=complex)
    cmat[0, :] = [1, 1, -1, -1]
    cmat[1, :] = [gam[0], gam[1], -gam[2], -gam[3]]
    cmat[2, :] = [gam[0]**2, gam[1]**2, -gam[2]**2, -gam[3]**2]
    cmat[3, :] = [gam[0]**3, gam[1]**3, -gam[2]**3, -gam[3]**3]

    rhs = np.array([0, 0, 0, 8.0], dtype=complex)
    A = np.linalg.solve(cmat, rhs)
    s = np.linspace(-8,8,1001, dtype=complex)
    disp = np.empty(len(s), dtype=complex)

    s_neg = s[s<0]
    s_pos = s[s>=0]

    disp[s>=0] = A[0]*np.exp(s_pos*gam[0])+A[1]*np.exp(s_pos*gam[1])
    disp[s<0] = A[2]*np.exp(s_neg*gam[2])+A[3]*np.exp(s_neg*gam[3])

    DAF = np.real(np.max(disp) * (1 / k2_k1)**0.75)

    return  DAF


def point_load_static_defl_shape_function(x, L, xc=0.0):
    """Deflection shape function for static point load on beam on foundation.

    Parameters
    ----------
    x : float
        x value
    L : float
        Characteristc length (1/lam)
    x0 : float, optional
        position of load.

    Returns
    -------
    nu : float
        shape function at x

    """


    return np.exp(-np.abs(x-xc)/L)*(np.cos(np.abs(x-xc)/L)+np.sin(np.abs(x-xc)/L))


def multi_static(xc, frel, wsingle=1, xlim=(-10,10), n=500, L=1):
    """Deflection shape for multiple static point loads

    Parameters
    ----------
    xc : float
        x-coord of each point load
    frel : float
        relative magnitude of each point load
    wsingle : float, optional
        maximum deflection under unit load
    xlim : tuple with two values
        xmin, xmax of evaluation
    L : float, optional
        Characterisic length

    Returns
    -------
    x,w : 1d ndarray
        x coords, and static deflection

    See Also
    --------
    point_load_static_defl_shape_function : single load function

    """

    x = np.linspace(xlim[0], xlim[1], n)
    w = np.zeros_like(x)

    for i, (xc_,frel_) in  enumerate(zip(xc, frel)):
        w+= frel_*point_load_static_defl_shape_function(x=x, L=L, xc=xc_)

    return x, w*wsingle


def dingetal_figure_8(v_nterms=(50,75,150,200)):
    """Reproduce Ding Et Al 2012 Figure 8 (might take a while).

    Note that a plot will be be saved to disk in current working directory
    as well as a timing file."""

    start_time0 = time.time()
    ftime = datetime.datetime.fromtimestamp(start_time0).strftime('%Y-%m-%d %H%M%S')
    with open(ftime + ".txt", "w") as f:
        f.write(ftime + os.linesep)

        fig=plt.figure()
        ax = fig.add_subplot("111")
        ax.set_xlabel("time, s")
        ax.set_ylabel("w, m")
        ax.set_xlim(3.5, 4.5)

        for v in v_nterms:
            vstr="nterms"
            vfmt="{}"
            start_time1 = time.time()

            f.write("*" * 70 + os.linesep)
            vequals = "{}={}".format(vstr, vfmt.format(v))
            f.write(vequals + os.linesep); print(vequals)

            t = np.linspace(0, 4.5, 400)
            pdict = OrderedDict(
                E = 6.998*1e9, #Pa
                rho = 2373, #kg/m3
                L = 160, #m
                #v_norm=0.01165,
                kf=5.41e-4,
                #Fz_norm=1.013e-4,
                mu_norm=39.263,
                k1_norm=97.552,
                k3_norm=2.497e6,
                nterms=v,
                BC="CC",
                nquad=20,
                moving_loads_x_norm=[[0]],
                moving_loads_Fz_norm=[[1.013e-4]],
                moving_loads_v_norm=[0.01165,],
                tvals=t,
                xvals_norm=np.array([.5]))



            f.write(repr(pdict) + os.linesep)

            for BC in ["SS"]:# "CC", "FF"]:
                pdict["BC"] = BC

                a = SpecBeam(**pdict)
                a.calulate_qk(t=t)

                x = t
                y = a.wofx(x_norm=0.5, normalise_w=False)

                ax.plot(x, y, label="x=0.5, {}".format(vequals))

            end_time1 = time.time()
            elapsed_time = (end_time1 - start_time1); print("Run time={}".format(str(timedelta(seconds=elapsed_time))))

            f.write("Run time={}".format(str(timedelta(seconds=elapsed_time))) + os.linesep)

        leg = ax.legend()
        leg.draggable()

        end_time0 = time.time()
        elapsed_time = (end_time0 - start_time0); print("Total run time={}".format(str(timedelta(seconds=elapsed_time))))
        f.write("Total run time={}".format(str(timedelta(seconds=elapsed_time))) + os.linesep)
    plt.savefig(ftime+".pdf")
    plt.show()




def SpecBeam_const_mat_midpoint_defl_runme_analytical_play():
    """Test SpecBeam for constant mat: close to Ding et al Figure 8, displacement vs time at
    beam midpoint (using the runme method) but with k3=0"""

    start_time0 = time.time()
    ftime = datetime.datetime.fromtimestamp(start_time0).strftime('%Y-%m-%d %H%M%S')

    # expected values are digitised from Ding et al 2012.
    expected_50terms_t = np.array(
      [ 3.511,  3.553,  3.576,  3.605,  3.621,  3.637,  3.651,  3.661,
        3.676,  3.693,  3.701,  3.717,  3.733,  3.743,  3.761,  3.777,
        3.79 ,  3.804,  3.822,  3.833,  3.848,  3.862,  3.877,  3.899,
        3.919,  3.94 ,  3.956,  3.97 ,  3.978,  3.99 ,  4.001,  4.014,
        4.025,  4.041,  4.052,  4.064,  4.076,  4.086,  4.094,  4.104,
        4.112,  4.123,  4.134,  4.146,  4.159,  4.172,  4.18 ,  4.192,
        4.212,  4.221,  4.234,  4.255,  4.273,  4.292,  4.324,  4.357,
        4.373,  4.399,  4.418,  4.445])

    #old values for k3~=0
#    expected_50terms_displacement = np.array(
#      [ -1.30900000e-04,   8.85900000e-05,   2.63900000e-04,
#         3.30200000e-04,   3.30500000e-04,   2.87100000e-04,
#         2.00000000e-04,   9.09500000e-05,  -8.36000000e-05,
#        -3.01800000e-04,  -4.32800000e-04,  -6.07300000e-04,
#        -7.59900000e-04,  -8.90900000e-04,  -9.99800000e-04,
#        -1.02100000e-03,  -9.99100000e-04,  -8.67700000e-04,
#        -5.61300000e-04,  -2.11300000e-04,   2.04300000e-04,
#         7.29200000e-04,   1.42900000e-03,   2.41300000e-03,
#         3.46300000e-03,   4.29400000e-03,   5.08100000e-03,
#         5.62800000e-03,   5.86800000e-03,   6.19600000e-03,
#         6.39300000e-03,   6.52500000e-03,   6.59100000e-03,
#         6.48200000e-03,   6.32900000e-03,   6.02300000e-03,
#         5.69600000e-03,   5.45500000e-03,   5.06200000e-03,
#         4.75600000e-03,   4.40700000e-03,   3.90400000e-03,
#         3.35800000e-03,   2.87800000e-03,   2.35300000e-03,
#         1.82900000e-03,   1.52300000e-03,   9.98700000e-04,
#         5.18300000e-04,   2.12500000e-04,  -4.95200000e-05,
#        -2.67600000e-04,  -3.76500000e-04,  -3.76100000e-04,
#        -2.00600000e-04,   1.87300000e-05,   1.06500000e-04,
#         1.94500000e-04,   1.94900000e-04,   1.30000000e-04])


    expected_50terms_displacement = np.array(
      [ -1.74400116e-05,   2.56595529e-04,   3.64963938e-04,
         3.96163257e-04,   3.52239419e-04,   2.55908617e-04,
         1.36296573e-04,   3.12193756e-05,  -1.48035583e-04,
        -3.73050108e-04,  -4.80427236e-04,  -6.84847774e-04,
        -8.61705766e-04,  -9.44268811e-04,  -1.02333177e-03,
        -1.00753661e-03,  -9.18192599e-04,  -7.32771117e-04,
        -3.72382730e-04,  -7.79760498e-05,   4.06998900e-04,
         9.37679667e-04,   1.57072235e-03,   2.58323632e-03,
         3.52928135e-03,   4.47726082e-03,   5.12406706e-03,
         5.61329890e-03,   5.84231170e-03,   6.13309619e-03,
         6.32505191e-03,   6.46203546e-03,   6.49032925e-03,
         6.39329339e-03,   6.23960924e-03,   5.99242067e-03,
         5.67177967e-03,   5.35598822e-03,   5.07919623e-03,
         4.69552191e-03,   4.37030991e-03,   3.90678952e-03,
         3.43034780e-03,   2.90774639e-03,   2.35231514e-03,
         1.82145021e-03,   1.51724930e-03,   1.09014092e-03,
         4.86434682e-04,   2.61200867e-04,  -5.02752907e-06,
        -2.99906282e-04,  -4.26188536e-04,  -4.46910518e-04,
        -3.09328219e-04,  -6.53887601e-05,   4.77786782e-05,
         1.84483472e-04,   2.30390682e-04,   2.18090273e-04])

    expected_200terms_t = np.array(
      [ 3.503,  3.519,  3.539,  3.556,  3.576,  3.595,  3.613,  3.632,
        3.651,  3.667,  3.69 ,  3.708,  3.727,  3.746,  3.766,  3.782,
        3.801,  3.822,  3.84 ,  3.856,  3.878,  3.896,  3.915,  3.933,
        3.951,  3.97 ,  3.99 ,  4.009,  4.027,  4.046,  4.067,  4.085,
        4.102,  4.122,  4.141,  4.159,  4.178,  4.197,  4.217,  4.234,
        4.252,  4.273,  4.291,  4.31 ,  4.329,  4.347,  4.368,  4.386,
        4.405,  4.423,  4.442,  4.46 ,  4.479,  4.495])

    #wrong : displ are with k3~=0
    expected_200terms_displacement = np.array(
      [  7.04000000e-08,   2.22800000e-05,   2.27000000e-05,
         1.23200000e-06,   2.35100000e-05,   2.39300000e-05,
         2.46400000e-06,  -1.89700000e-05,  -6.22700000e-05,
        -1.27500000e-04,  -1.70700000e-04,  -2.57800000e-04,
        -3.44800000e-04,  -4.53600000e-04,  -5.40600000e-04,
        -5.84000000e-04,  -5.61700000e-04,  -4.73800000e-04,
        -2.54900000e-04,   1.17100000e-04,   6.85900000e-04,
         1.51700000e-03,   2.50100000e-03,   3.70300000e-03,
         5.01500000e-03,   6.26200000e-03,   7.20200000e-03,
         7.61800000e-03,   7.39900000e-03,   6.74400000e-03,
         5.91400000e-03,   5.01800000e-03,   4.10100000e-03,
         3.22700000e-03,   2.50600000e-03,   1.85000000e-03,
         1.32600000e-03,   9.11400000e-04,   5.84000000e-04,
         3.00200000e-04,   1.47600000e-04,   3.87500000e-05,
        -4.82900000e-05,  -1.13400000e-04,  -1.13000000e-04,
        -1.34500000e-04,  -1.34000000e-04,  -1.33600000e-04,
        -1.11400000e-04,  -6.72600000e-05,  -6.68400000e-05,
        -6.64500000e-05,  -4.41700000e-05,  -4.38200000e-05])

    vfact=2
    t = np.linspace(0, 4.5*2.5/vfact, 500)
    x_norm = np.linspace(0,1,2001)
    #see Dingetal2012 Table 2 for material properties
    pdict = OrderedDict(
            E = 6.998*1e9, #Pa
            rho = 2373, #kg/m3
            L = 160, #m
            #v_norm=0.01165,
            kf=5.41e-4,
            #Fz_norm=1.013e-4,
            mu_norm=39.263,
            k1_norm=97.552,
#            k3_norm=2.497e6,
            nterms=200,
            BC="SS",
            nquad=20,
#            k1bar=PolyLine([0,0.5],[0.5,1],[1,1],[1,.25]),
#            k1bar=PolyLine([0,0.25,.75,1],[1.0,1,4,4]),
#            k1bar=PolyLine([0,0.4,.6,1],[1.0,1,.25,.25]),
            k1bar=PolyLine([0,0.4,.6,1],[1.0,1,4,4]),
#            k1bar=PolyLine([0,0.5],[0.5,1],[2,1],[2,1]),
            moving_loads_x_norm=[[0,-0.05,-0.1]],
            moving_loads_Fz_norm=[3*[1.013e-4]],
            moving_loads_v_norm=[0.01165*vfact],
            tvals=t,
#            tvals=np.array([4.0]),
#            xvals_norm=np.array([0.5]),
            xvals_norm=x_norm,

            use_analytical=True,
#            implementation="vectorized",
            implementation="fortran",
            file_stem="specbeam200moving_soft_to_stiff",
            force_calc=True,
            )



    a = SpecBeam(**pdict)
    a.runme()
    a.saveme()
    a.animateme(norm=True,saveme=True,interval=18)
#    a.plot_w_vs_x_overtime(norm=True)
    plt.show()
#    a.calulate_qk(t=t)
#
#    yall = a.wofx(x_norm=0.5, normalise_w=False)
    ycompare = np.interp(expected_50terms_t, t, a.defl)


    print("n=", pdict['nterms'])
    end_time0 = time.time()
    elapsed_time = (end_time0 - start_time0); print("Total run time={}".format(str(timedelta(seconds=elapsed_time))))

    if DEBUG:


        title ='SpecBeam, const mat prop vs Ding et al (2012) \nFigure 8 Displacement at Midpoint of beam but with k3=0'
        print(title)
        print(' '.join(['{:>9s}']*4).format('t', 'expect', 'calc', 'diff'))
        for i, j, k in zip(expected_50terms_t, expected_50terms_displacement, ycompare):
            print(' '.join(['{:9.4f}']*4).format(i, j, k, j-k))
        print()




        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot("111")
        ax.set_xlabel("time, s")
        ax.set_ylabel("w, m")
        ax.set_xlim(3.5, 4.5)
        ax.set_title(title)

        ax.plot(expected_50terms_t, expected_50terms_displacement,
                label="expect n=50", color='green', marker='o', ls='-')
        ax.plot(expected_200terms_t, expected_200terms_displacement,
                label="expect n=200", color='black', marker=None, ls='-')


        ax.plot(t, a.defl, label="calc n=50")
        ax.plot(expected_50terms_t, ycompare, label="calc, n=50 (compare)",
                color='red', marker='s', ms=4, ls=':')

        leg = ax.legend(loc='upper left')
        leg.draggable()

        plt.show()

    assert_allclose(expected_50terms_displacement, ycompare, atol=2.4e-4)



def SpecBeam_stationary_load_const_mat_midpoint_defl_runme_analytical_play():
    """Test SpecBeam for constant mat: close to Ding et al Figure 8, displacement vs time at
    beam midpoint (using the runme method) but with k3=0"""

    start_time0 = time.time()
    ftime = datetime.datetime.fromtimestamp(start_time0).strftime('%Y-%m-%d %H%M%S')

    # expected values are digitised from Ding et al 2012.
    expected_50terms_t = np.array(
      [ 3.511,  3.553,  3.576,  3.605,  3.621,  3.637,  3.651,  3.661,
        3.676,  3.693,  3.701,  3.717,  3.733,  3.743,  3.761,  3.777,
        3.79 ,  3.804,  3.822,  3.833,  3.848,  3.862,  3.877,  3.899,
        3.919,  3.94 ,  3.956,  3.97 ,  3.978,  3.99 ,  4.001,  4.014,
        4.025,  4.041,  4.052,  4.064,  4.076,  4.086,  4.094,  4.104,
        4.112,  4.123,  4.134,  4.146,  4.159,  4.172,  4.18 ,  4.192,
        4.212,  4.221,  4.234,  4.255,  4.273,  4.292,  4.324,  4.357,
        4.373,  4.399,  4.418,  4.445])

    #old values for k3~=0
#    expected_50terms_displacement = np.array(
#      [ -1.30900000e-04,   8.85900000e-05,   2.63900000e-04,
#         3.30200000e-04,   3.30500000e-04,   2.87100000e-04,
#         2.00000000e-04,   9.09500000e-05,  -8.36000000e-05,
#        -3.01800000e-04,  -4.32800000e-04,  -6.07300000e-04,
#        -7.59900000e-04,  -8.90900000e-04,  -9.99800000e-04,
#        -1.02100000e-03,  -9.99100000e-04,  -8.67700000e-04,
#        -5.61300000e-04,  -2.11300000e-04,   2.04300000e-04,
#         7.29200000e-04,   1.42900000e-03,   2.41300000e-03,
#         3.46300000e-03,   4.29400000e-03,   5.08100000e-03,
#         5.62800000e-03,   5.86800000e-03,   6.19600000e-03,
#         6.39300000e-03,   6.52500000e-03,   6.59100000e-03,
#         6.48200000e-03,   6.32900000e-03,   6.02300000e-03,
#         5.69600000e-03,   5.45500000e-03,   5.06200000e-03,
#         4.75600000e-03,   4.40700000e-03,   3.90400000e-03,
#         3.35800000e-03,   2.87800000e-03,   2.35300000e-03,
#         1.82900000e-03,   1.52300000e-03,   9.98700000e-04,
#         5.18300000e-04,   2.12500000e-04,  -4.95200000e-05,
#        -2.67600000e-04,  -3.76500000e-04,  -3.76100000e-04,
#        -2.00600000e-04,   1.87300000e-05,   1.06500000e-04,
#         1.94500000e-04,   1.94900000e-04,   1.30000000e-04])


    expected_50terms_displacement = np.array(
      [ -1.74400116e-05,   2.56595529e-04,   3.64963938e-04,
         3.96163257e-04,   3.52239419e-04,   2.55908617e-04,
         1.36296573e-04,   3.12193756e-05,  -1.48035583e-04,
        -3.73050108e-04,  -4.80427236e-04,  -6.84847774e-04,
        -8.61705766e-04,  -9.44268811e-04,  -1.02333177e-03,
        -1.00753661e-03,  -9.18192599e-04,  -7.32771117e-04,
        -3.72382730e-04,  -7.79760498e-05,   4.06998900e-04,
         9.37679667e-04,   1.57072235e-03,   2.58323632e-03,
         3.52928135e-03,   4.47726082e-03,   5.12406706e-03,
         5.61329890e-03,   5.84231170e-03,   6.13309619e-03,
         6.32505191e-03,   6.46203546e-03,   6.49032925e-03,
         6.39329339e-03,   6.23960924e-03,   5.99242067e-03,
         5.67177967e-03,   5.35598822e-03,   5.07919623e-03,
         4.69552191e-03,   4.37030991e-03,   3.90678952e-03,
         3.43034780e-03,   2.90774639e-03,   2.35231514e-03,
         1.82145021e-03,   1.51724930e-03,   1.09014092e-03,
         4.86434682e-04,   2.61200867e-04,  -5.02752907e-06,
        -2.99906282e-04,  -4.26188536e-04,  -4.46910518e-04,
        -3.09328219e-04,  -6.53887601e-05,   4.77786782e-05,
         1.84483472e-04,   2.30390682e-04,   2.18090273e-04])

    expected_200terms_t = np.array(
      [ 3.503,  3.519,  3.539,  3.556,  3.576,  3.595,  3.613,  3.632,
        3.651,  3.667,  3.69 ,  3.708,  3.727,  3.746,  3.766,  3.782,
        3.801,  3.822,  3.84 ,  3.856,  3.878,  3.896,  3.915,  3.933,
        3.951,  3.97 ,  3.99 ,  4.009,  4.027,  4.046,  4.067,  4.085,
        4.102,  4.122,  4.141,  4.159,  4.178,  4.197,  4.217,  4.234,
        4.252,  4.273,  4.291,  4.31 ,  4.329,  4.347,  4.368,  4.386,
        4.405,  4.423,  4.442,  4.46 ,  4.479,  4.495])

    #wrong : displ are with k3~=0
    expected_200terms_displacement = np.array(
      [  7.04000000e-08,   2.22800000e-05,   2.27000000e-05,
         1.23200000e-06,   2.35100000e-05,   2.39300000e-05,
         2.46400000e-06,  -1.89700000e-05,  -6.22700000e-05,
        -1.27500000e-04,  -1.70700000e-04,  -2.57800000e-04,
        -3.44800000e-04,  -4.53600000e-04,  -5.40600000e-04,
        -5.84000000e-04,  -5.61700000e-04,  -4.73800000e-04,
        -2.54900000e-04,   1.17100000e-04,   6.85900000e-04,
         1.51700000e-03,   2.50100000e-03,   3.70300000e-03,
         5.01500000e-03,   6.26200000e-03,   7.20200000e-03,
         7.61800000e-03,   7.39900000e-03,   6.74400000e-03,
         5.91400000e-03,   5.01800000e-03,   4.10100000e-03,
         3.22700000e-03,   2.50600000e-03,   1.85000000e-03,
         1.32600000e-03,   9.11400000e-04,   5.84000000e-04,
         3.00200000e-04,   1.47600000e-04,   3.87500000e-05,
        -4.82900000e-05,  -1.13400000e-04,  -1.13000000e-04,
        -1.34500000e-04,  -1.34000000e-04,  -1.33600000e-04,
        -1.11400000e-04,  -6.72600000e-05,  -6.68400000e-05,
        -6.64500000e-05,  -4.41700000e-05,  -4.38200000e-05])

    vfact=2
    t = np.linspace(0, 4.5*3/vfact, 200)
    x_norm = np.linspace(0,1,2001)
    #see Dingetal2012 Table 2 for material properties
    pdict = OrderedDict(
            E = 6.998*1e9, #Pa
            rho = 2373, #kg/m3
            L = 160, #m
            #v_norm=0.01165,
            kf=5.41e-4,
            #Fz_norm=1.013e-4,
            mu_norm=39.263,
            k1_norm=97.552,
#            k3_norm=2.497e6,
            nterms=200,
            BC="SS",
            nquad=20,
#            k1bar=PolyLine([0,0.5],[0.5,1],[1,1],[1,.25]),
#            k1bar=PolyLine([0,0.25,.75,1],[1.0,1,4,4]),
            k1bar=PolyLine([0,0.4,.6,1],[1.0,1,.25,.25]),
#            k1bar=PolyLine([0,0.5],[0.5,1],[2,1],[2,1]),

#            moving_loads_x_norm=[[0,-0.05,-0.1]],
#            moving_loads_Fz_norm=[3*[1.013e-4]],
#            moving_loads_v_norm=[0.01165*vfact],

#            stationary_loads_x=None,
#            stationary_loads_vs_t=None,
#            stationary_loads_omega_phase=None,
            stationary_loads_x_norm=[0.5, 0.5],
            stationary_loads_vs_t_norm=2*[PolyLine([0,t[-1]/160*np.sqrt(6.998*1e9/2373.)],[0.5*1.013e-4,0.5*1.013e-4])],
            stationary_loads_omega_phase_norm=[(.5,0),(0,0)],#None,
            tvals=t,
#            tvals=np.array([4.0]),
#            xvals_norm=np.array([0.5]),
            xvals_norm=x_norm,

            use_analytical=True,
#            implementation="vectorized",
            implementation="fortran",
            file_stem="specbeam200stationary",
            force_calc=True,
            )



    a = SpecBeam(**pdict)
    a.runme()
    a.saveme()
    a.animateme(norm=True)
#    a.plot_w_vs_x_overtime(norm=True)
    plt.show()







def FIGURE_verify_against_DingEtAl2012_fig8(v_nterms=(50,75,150,200), saveas=None):
    """Ding et al Figure 8, displacement vs time at
    beam midpoint (using the runme method) (50 and 200 terms)"""

    expected=dict()
    # expected values are digitised from Ding et al 2012.
    expected_50terms_t = np.array(
      [ 3.511,  3.553,  3.576,  3.605,  3.621,  3.637,  3.651,  3.661,
        3.676,  3.693,  3.701,  3.717,  3.733,  3.743,  3.761,  3.777,
        3.79 ,  3.804,  3.822,  3.833,  3.848,  3.862,  3.877,  3.899,
        3.919,  3.94 ,  3.956,  3.97 ,  3.978,  3.99 ,  4.001,  4.014,
        4.025,  4.041,  4.052,  4.064,  4.076,  4.086,  4.094,  4.104,
        4.112,  4.123,  4.134,  4.146,  4.159,  4.172,  4.18 ,  4.192,
        4.212,  4.221,  4.234,  4.255,  4.273,  4.292,  4.324,  4.357,
        4.373,  4.399,  4.418,  4.445])

    #old values for k3~=0
#    expected_50terms_displacement = np.array(
#      [ -1.30900000e-04,   8.85900000e-05,   2.63900000e-04,
#         3.30200000e-04,   3.30500000e-04,   2.87100000e-04,
#         2.00000000e-04,   9.09500000e-05,  -8.36000000e-05,
#        -3.01800000e-04,  -4.32800000e-04,  -6.07300000e-04,
#        -7.59900000e-04,  -8.90900000e-04,  -9.99800000e-04,
#        -1.02100000e-03,  -9.99100000e-04,  -8.67700000e-04,
#        -5.61300000e-04,  -2.11300000e-04,   2.04300000e-04,
#         7.29200000e-04,   1.42900000e-03,   2.41300000e-03,
#         3.46300000e-03,   4.29400000e-03,   5.08100000e-03,
#         5.62800000e-03,   5.86800000e-03,   6.19600000e-03,
#         6.39300000e-03,   6.52500000e-03,   6.59100000e-03,
#         6.48200000e-03,   6.32900000e-03,   6.02300000e-03,
#         5.69600000e-03,   5.45500000e-03,   5.06200000e-03,
#         4.75600000e-03,   4.40700000e-03,   3.90400000e-03,
#         3.35800000e-03,   2.87800000e-03,   2.35300000e-03,
#         1.82900000e-03,   1.52300000e-03,   9.98700000e-04,
#         5.18300000e-04,   2.12500000e-04,  -4.95200000e-05,
#        -2.67600000e-04,  -3.76500000e-04,  -3.76100000e-04,
#        -2.00600000e-04,   1.87300000e-05,   1.06500000e-04,
#         1.94500000e-04,   1.94900000e-04,   1.30000000e-04])


    expected_50terms_displacement = np.array(
      [ -1.74400116e-05,   2.56595529e-04,   3.64963938e-04,
         3.96163257e-04,   3.52239419e-04,   2.55908617e-04,
         1.36296573e-04,   3.12193756e-05,  -1.48035583e-04,
        -3.73050108e-04,  -4.80427236e-04,  -6.84847774e-04,
        -8.61705766e-04,  -9.44268811e-04,  -1.02333177e-03,
        -1.00753661e-03,  -9.18192599e-04,  -7.32771117e-04,
        -3.72382730e-04,  -7.79760498e-05,   4.06998900e-04,
         9.37679667e-04,   1.57072235e-03,   2.58323632e-03,
         3.52928135e-03,   4.47726082e-03,   5.12406706e-03,
         5.61329890e-03,   5.84231170e-03,   6.13309619e-03,
         6.32505191e-03,   6.46203546e-03,   6.49032925e-03,
         6.39329339e-03,   6.23960924e-03,   5.99242067e-03,
         5.67177967e-03,   5.35598822e-03,   5.07919623e-03,
         4.69552191e-03,   4.37030991e-03,   3.90678952e-03,
         3.43034780e-03,   2.90774639e-03,   2.35231514e-03,
         1.82145021e-03,   1.51724930e-03,   1.09014092e-03,
         4.86434682e-04,   2.61200867e-04,  -5.02752907e-06,
        -2.99906282e-04,  -4.26188536e-04,  -4.46910518e-04,
        -3.09328219e-04,  -6.53887601e-05,   4.77786782e-05,
         1.84483472e-04,   2.30390682e-04,   2.18090273e-04])

    expected_200terms_t = np.array(
      [ 3.503,  3.519,  3.539,  3.556,  3.576,  3.595,  3.613,  3.632,
        3.651,  3.667,  3.69 ,  3.708,  3.727,  3.746,  3.766,  3.782,
        3.801,  3.822,  3.84 ,  3.856,  3.878,  3.896,  3.915,  3.933,
        3.951,  3.97 ,  3.99 ,  4.009,  4.027,  4.046,  4.067,  4.085,
        4.102,  4.122,  4.141,  4.159,  4.178,  4.197,  4.217,  4.234,
        4.252,  4.273,  4.291,  4.31 ,  4.329,  4.347,  4.368,  4.386,
        4.405,  4.423,  4.442,  4.46 ,  4.479,  4.495])

    expected_200terms_displacement = np.array(
      [  7.04000000e-08,   2.22800000e-05,   2.27000000e-05,
         1.23200000e-06,   2.35100000e-05,   2.39300000e-05,
         2.46400000e-06,  -1.89700000e-05,  -6.22700000e-05,
        -1.27500000e-04,  -1.70700000e-04,  -2.57800000e-04,
        -3.44800000e-04,  -4.53600000e-04,  -5.40600000e-04,
        -5.84000000e-04,  -5.61700000e-04,  -4.73800000e-04,
        -2.54900000e-04,   1.17100000e-04,   6.85900000e-04,
         1.51700000e-03,   2.50100000e-03,   3.70300000e-03,
         5.01500000e-03,   6.26200000e-03,   7.20200000e-03,
         7.61800000e-03,   7.39900000e-03,   6.74400000e-03,
         5.91400000e-03,   5.01800000e-03,   4.10100000e-03,
         3.22700000e-03,   2.50600000e-03,   1.85000000e-03,
         1.32600000e-03,   9.11400000e-04,   5.84000000e-04,
         3.00200000e-04,   1.47600000e-04,   3.87500000e-05,
        -4.82900000e-05,  -1.13400000e-04,  -1.13000000e-04,
        -1.34500000e-04,  -1.34000000e-04,  -1.33600000e-04,
        -1.11400000e-04,  -6.72600000e-05,  -6.68400000e-05,
        -6.64500000e-05,  -4.41700000e-05,  -4.38200000e-05])


    expected=dict()
    expected[50]=(expected_50terms_t, expected_50terms_displacement,"o")
#    expected[75]=(expected_200terms_t, expected_200terms_displacement,"^")
    expected[200]=(expected_200terms_t, expected_200terms_displacement,"s")
    start_time0 = time.time()
    ftime = datetime.datetime.fromtimestamp(start_time0).strftime('%Y-%m-%d %H%M%S')
    with open(ftime + ".txt", "w") as f:
        f.write(ftime + os.linesep)


        fig = plt.figure(figsize=(3.54,3.54))
        matplotlib.rcParams.update({'font.size': 11})

        ax = fig.add_subplot("111")


        for v in v_nterms:
            vstr="nterms"
            vfmt="{}"
            start_time1 = time.time()

            f.write("*" * 70 + os.linesep)
#            vequals = "{}={}".format(vstr, vfmt.format(v))
            vequals = "{}".format(vfmt.format(v))
            f.write(vequals + os.linesep); print(vequals)

            t = np.linspace(0, 4.5, 400)
            pdict = OrderedDict(
                E = 6.998*1e9, #Pa
                rho = 2373, #kg/m3
                L = 160, #m
                #v_norm=0.01165,
                kf=5.41e-4,
                #Fz_norm=1.013e-4,
                mu_norm=39.263,
                k1_norm=97.552,
#                k3_norm=2.497e6,
                nterms=v,
                BC="CC",
                nquad=20,
                moving_loads_x_norm=[[0]],
                moving_loads_Fz_norm=[[1.013e-4]],
                moving_loads_v_norm=[0.01165,],
                tvals=t,
                xvals_norm=np.array([.5]),
                use_analytical=True,
                implementation="fortran",
                force_calc=False,
                file_stem="specbeam_fig_ding_{}".format(v))



            f.write(repr(pdict) + os.linesep)

            for BC in ["SS"]:# "CC", "FF"]:
                pdict["BC"] = BC

                a = SpecBeam(**pdict)
                a.runme()
                a.saveme()
#                a.calulate_qk(t=t)

                x = t
                y = a.defl[0,:]*1000#a.wofx(x_norm=0.5, normalise_w=False)

#                ax.plot(x, y, label="x=0.5, {}".format(vequals))
                line, = ax.plot(x, y, label=vequals)

                expect=expected.get(v,None)
                if not expect is None:
                    expect_x,expect_y, marker = expect
                    ax.plot(expect_x, expect_y*1000,
                            label="{} (Ding)".format(vequals),
                            linestyle="None",
                            marker=marker,
                            markeredgecolor = line.get_color(),
                            markerfacecolor='None',
                            markersize=5)

            end_time1 = time.time()
            elapsed_time = (end_time1 - start_time1); print("Run time={}".format(str(timedelta(seconds=elapsed_time))))

            f.write("Run time={}".format(str(timedelta(seconds=elapsed_time))) + os.linesep)

        #ax.set_title("$B=r_e,b_w=r_w,b_s=NA$")
        #ax.set_ylim(.6,1.05)
        #ax.xaxis.label.set_fontsize(30)
        #ax.yaxis.label.set_fontsize(30)
        #ax.set_ylabel("$\\frac{u_\\mathrm{a,max}}{u_\\mathrm{p,max}}$")
#        ax.set_ylabel("$u_\\mathrm{a,max}/u_\\mathrm{p,max}$")
#        ax.set_xlabel("$n_a=r_e/r_w$")
    #    ax.set_xscale("log")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Midpoint deflection (mm)")
        ax.set_xlim(3.5, 4.5)
        ax.grid()
        leg = ax.legend(title="$No.\.terms$",
                        loc="upper left",
#                        ncol=2,
                        labelspacing=.2,
                        handlelength=2,
                        fontsize=8
                        )


        ax.xaxis.labelpad = -1
        ax.yaxis.labelpad = 0

        fig.subplots_adjust(top=0.95, bottom=0.13, left=0.14, right=0.95)


#        leg = ax.legend()
#        leg.draggable()

        end_time0 = time.time()
        elapsed_time = (end_time0 - start_time0); print("Total run time={}".format(str(timedelta(seconds=elapsed_time))))
        f.write("Total run time={}".format(str(timedelta(seconds=elapsed_time))) + os.linesep)
#    plt.savefig(ftime+".pdf")
    if not saveas is None:
        save_figure(fig=fig,fname=saveas)
    plt.show()

def FIGURE_convergence_step_change(v_nterms=(50,75,150,200),saveas=None):
    """Check convergence with step change in stiffness at midpoint"""


    start_time0 = time.time()
    ftime = datetime.datetime.fromtimestamp(start_time0).strftime('%Y-%m-%d %H%M%S')
    with open(ftime + ".txt", "w") as f:
        f.write(ftime + os.linesep)


        fig = plt.figure(figsize=(3.54,3.54))
        matplotlib.rcParams.update({'font.size': 11})

        ax = fig.add_subplot("111")


        for v in v_nterms:
            vstr=""
            vfmt="{}"
            start_time1 = time.time()

            f.write("*" * 70 + os.linesep)
#            vequals = "{}={}".format(vstr, vfmt.format(v))
            vequals = "{}".format(vfmt.format(v))
            f.write(vequals + os.linesep); print(vequals)

            t = np.linspace(0, 4.5, 400)
            pdict = OrderedDict(
                E = 6.998*1e9, #Pa
                rho = 2373, #kg/m3
                L = 160, #m
                #v_norm=0.01165,
                kf=5.41e-4,
                #Fz_norm=1.013e-4,
                mu_norm=39.263,
                k1_norm=97.552,
                k1bar=PolyLine([0,0.5,.5,1],[1.0,1,.01,.01]),
#                k3_norm=2.497e6,
                nterms=v,
                BC="CC",
                nquad=20,
                moving_loads_x_norm=[[0]],
                moving_loads_Fz_norm=[[1.013e-4]],
                moving_loads_v_norm=[0.01165,],
                tvals=t,
                xvals_norm=np.array([.5]),
                use_analytical=True,
                implementation="fortran",
                force_calc=False,
                file_stem="specbeam_fig_convergence_step{}".format(v))



            f.write(repr(pdict) + os.linesep)

            for BC in ["SS"]:# "CC", "FF"]:
                pdict["BC"] = BC

                a = SpecBeam(**pdict)
                a.runme()
                a.saveme()
#                a.calulate_qk(t=t)

                x = t
                y = a.defl[0,:]*1000#a.wofx(x_norm=0.5, normalise_w=False)

#                ax.plot(x, y, label="x=0.5, {}".format(vequals))
                line, = ax.plot(x, y, label=vequals)


            end_time1 = time.time()
            elapsed_time = (end_time1 - start_time1); print("Run time={}".format(str(timedelta(seconds=elapsed_time))))

            f.write("Run time={}".format(str(timedelta(seconds=elapsed_time))) + os.linesep)

        #ax.set_title("$B=r_e,b_w=r_w,b_s=NA$")
        #ax.set_ylim(.6,1.05)
        #ax.xaxis.label.set_fontsize(30)
        #ax.yaxis.label.set_fontsize(30)
        #ax.set_ylabel("$\\frac{u_\\mathrm{a,max}}{u_\\mathrm{p,max}}$")
#        ax.set_ylabel("$u_\\mathrm{a,max}/u_\\mathrm{p,max}$")
#        ax.set_xlabel("$n_a=r_e/r_w$")
    #    ax.set_xscale("log")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Midpoint deflection (mm)")
        ax.set_xlim(3.5, 4.5)
        ax.grid()
        leg = ax.legend(title="$No.\.terms$",
                        loc="upper left",
#                        ncol=2,
                        labelspacing=.2,
                        handlelength=2,
                        fontsize=8
                        )


        ax.xaxis.labelpad = -1
        ax.yaxis.labelpad = 0

        fig.subplots_adjust(top=0.95, bottom=0.13, left=0.14, right=0.95)


#        leg = ax.legend()
#        leg.draggable()

        end_time0 = time.time()
        elapsed_time = (end_time0 - start_time0); print("Total run time={}".format(str(timedelta(seconds=elapsed_time))))
        f.write("Total run time={}".format(str(timedelta(seconds=elapsed_time))) + os.linesep)
#    plt.savefig(ftime+".pdf")
    if not saveas is None:
        save_figure(fig=fig,fname=saveas)
    plt.show()

def FIGURE_defl_envelope_vary_velocity(velocities_norm=(0.01165,),saveas=None,nterms=100,force_calc=True,nx=400,nt=400):
    """Deflection envelopes with different velocities"""


    start_time0 = time.time()
    ftime = datetime.datetime.fromtimestamp(start_time0).strftime('%Y-%m-%d %H%M%S')
    with open(ftime + ".txt", "w") as f:
        f.write(ftime + os.linesep)


        fig = plt.figure(figsize=(3.54,3.54))
        matplotlib.rcParams.update({'font.size': 11})

        ax = fig.add_subplot("111")


        for v in velocities_norm:
            vstr=""
            vfmt="{}"
            start_time1 = time.time()

            f.write("*" * 70 + os.linesep)
#            vequals = "{}={}".format(vstr, vfmt.format(v))
            vequals = "{}".format(vfmt.format(v))
            f.write(vequals + os.linesep); print(vequals)

            t = np.linspace(0, v/0.01165*4*2, nt)# 4.5

            pdict = OrderedDict(
                E = 6.998*1e9, #Pa
                rho = 2373, #kg/m3
                L = 160, #m
                A = 0.3*0.1, # m^2
                #v_norm=0.01165,
                kf=5.41e-4,
                #Fz_norm=1.013e-4,
                mu_norm=39.263/4/2/1000,
                mubar=PolyLine([0,0.05,0.05,.95,.95,1],[1,1,1,1,1,1]),
                k1_norm=97.552,
#                k1bar=PolyLine([0,0.2,0.3,1],[1.0,1,0.1,0.1]),
                k1bar=PolyLine([0,0.2,0.2,1],[1.0,1,1,1]),

#                k3_norm=2.497e6,
                nterms=nterms,
                BC="SS",
                nquad=20,
                moving_loads_x_norm=[[0]],
                moving_loads_Fz_norm=[[1.013e-4]],
                moving_loads_v_norm=[v,],#[0.01165,],
                tvals=t,
                xvals_norm=np.linspace(0,1,nx),#np.array([.5]),
                use_analytical=True,
                implementation="fortran",
                force_calc=force_calc,
                file_stem="specbeam_defl_envelope_velocity{}".format(v))



            I_raw = (pdict["kf"]*pdict["L"])**2*pdict["A"]
            k_raw = pdict["k1_norm"]*pdict["E"]*pdict["A"]/pdict["L"]**2

            c_raw = pdict["mu_norm"] * pdict["A"]*np.sqrt(pdict["rho"]*pdict["E"])/pdict["L"]
            c_crit = np.sqrt(4*pdict["rho"]*pdict["A"]*k_raw)
            beta = c_raw/c_crit

            v_raw = v / np.sqrt(pdict["rho"]/pdict["E"])
            v_crit = np.sqrt(2/pdict["rho"]/pdict["A"]*np.sqrt(k_raw*pdict["E"]*I_raw))
            alpha = v_raw/v_crit

            #tmax = pdict["L"]/ (v/np.sqrt(pdict["rho"]/pdict["E"]))
            tmax = pdict["L"]/ v_raw
            t = np.linspace(0, tmax, nt)
            pdict["tvals"]=t

            print(I_raw,k_raw,v_raw)
            print("velocity={:4.3g} km/hr".format(v_raw/1000*(60*60)))
            print("v/v_crit={:2.3g}, {:2.3g}".format(alpha, alpha/(0.1**0.25)))
            print("c/c_crit={:2.3g}, {:2.3g}".format(beta, beta/(0.1**0.5)))
            f.write(repr(pdict) + os.linesep)

            for BC in ["SS"]:# "CC", "FF"]:
                pdict["BC"] = BC

                a = SpecBeam(**pdict)
                a.runme()
                a.saveme()
#                a.calulate_qk(t=t)

                x = a.xvals_norm * pdict["L"]
                ymax = np.max(a.defl,axis=1)*1000
                ymin = np.min(a.defl,axis=1)*1000
#                y = a.defl[0,:]*1000#a.wofx(x_norm=0.5, normalise_w=False)

#                ax.plot(x, y, label="x=0.5, {}".format(vequals))
                line, = ax.plot(x, ymax, label=vequals)
                line, = ax.plot(x, ymin, color=line.get_color())


            end_time1 = time.time()
            elapsed_time = (end_time1 - start_time1); print("Run time={}".format(str(timedelta(seconds=elapsed_time))))

            f.write("Run time={}".format(str(timedelta(seconds=elapsed_time))) + os.linesep)

        #ax.set_title("$B=r_e,b_w=r_w,b_s=NA$")
        #ax.set_ylim(.6,1.05)
        #ax.xaxis.label.set_fontsize(30)
        #ax.yaxis.label.set_fontsize(30)
        #ax.set_ylabel("$\\frac{u_\\mathrm{a,max}}{u_\\mathrm{p,max}}$")
#        ax.set_ylabel("$u_\\mathrm{a,max}/u_\\mathrm{p,max}$")
#        ax.set_xlabel("$n_a=r_e/r_w$")
    #    ax.set_xscale("log")
        ax.set_xlabel("Distance, x (m)")
        ax.set_ylabel("Max/min deflection experienced (mm)")
#        ax.set_xlim(3.5, 4.5)
        ax.grid()
        leg = ax.legend(title="$v$",
                        loc="upper left",
#                        ncol=2,
                        labelspacing=.2,
                        handlelength=2,
                        fontsize=8
                        )
        leg.draggable()

        ax.xaxis.labelpad = -1
        ax.yaxis.labelpad = 0

        fig.subplots_adjust(top=0.95, bottom=0.13, left=0.14, right=0.95)


#        leg = ax.legend()
#        leg.draggable()

        end_time0 = time.time()
        elapsed_time = (end_time0 - start_time0); print("Total run time={}".format(str(timedelta(seconds=elapsed_time))))
        f.write("Total run time={}".format(str(timedelta(seconds=elapsed_time))) + os.linesep)
        a.animateme()


#    plt.savefig(ftime+".pdf")
    if not saveas is None:
        save_figure(fig=fig,fname=saveas)

    plt.show()

def FIGURE_DAF_constant_prop(saveas=None,nterms=100,force_calc=True,prefix="",
                             alphas=None,betas=None,nt=20,nx=55, end_damp=0,
                             xwindow=(0.3,0.7),
                             xeval=(0.3,0.7), numerical_DAF=True,
                             article_formatting=False):
    """Dynamic amplification plot, esveld fig 6.18. p120"""
    pass

    start_time0 = time.time()
    ftime = datetime.datetime.fromtimestamp(start_time0).strftime('%Y-%m-%d %H%M%S')
    with open(ftime + ".txt", "w") as f:
        f.write(ftime + os.linesep)


        fig = plt.figure(figsize=(3.54,3.54))
        matplotlib.rcParams.update({'font.size': 11})

        ax = fig.add_subplot("111")

        if alphas is None:
            alphas = np.linspace(1e-5, 3, 100)
        if betas is None:
            betas = [0, 0.05, 0.1, 0.3, 1.1, 2.0]


        pdict = OrderedDict(
                    E = 6.998*1e9, #Pa
                    rho = 2373, #kg/m3
                    L = 160, #m
                    A = 0.3*0.1, # m^2
                    I = 0.00022477900799999998,
                    k1 = 800002.6125,

#                    kf=5.41e-4,
                    #Fz_norm=1.013e-4,
#                    mu_norm=39.263/4/2/1000,
#                    mubar=PolyLine([0,0.05,0.05,.95,.95,1],[1,1,1,1,1,1]),
#                    k1_norm=97.552,
    #                k1bar=PolyLine([0,0.2,0.3,1],[1.0,1,0.1,0.1]),
#                    k1bar=PolyLine([0,0.2,0.2,1],[1.0,1,1,1]),

    #                k3_norm=2.497e6,
                    nterms=nterms,
                    BC="SS",
#                    nquad=20,
                    moving_loads_x_norm=[[0]],
                    moving_loads_Fz_norm=[[1.013e-4]],
#                    moving_loads_v_norm=[v,],#[0.01165,],
#                    tvals=t,
                    xvals_norm=np.linspace(xeval[0],xeval[1],nx),
                    use_analytical=True,
                    implementation="fortran",
                    force_calc=force_calc,
                    )


        iwindow = [np.searchsorted(pdict['xvals_norm'],v) for v in xwindow]

        if article_formatting:
            mdc = MarkersDashesColors(markersize=3)
            mdc.construct_styles(markers=[2,11,25,5,8,15],
                               dashes=[0],
                               marker_colors=[0,1,2,3,4,5],
                               line_colors=[0,1,2,3,4,5])

            styles=mdc(markers=[2,11,25,5,8,15],
                               dashes=[0],
                               marker_colors=[0,1,2,3,4,5],
                               line_colors=[0,1,2,3,4,5])

        for i, beta in enumerate(betas):
            x = np.zeros(len(alphas))
            y = np.zeros(len(alphas))
            for j, alpha in enumerate(alphas):

                start_time1 = time.time()

                f.write("*" * 70 + os.linesep)
#                vequals = "{}={}".format(vstr, vfmt.format(v))
                vequals = "alpha={}, beta={}".format(alpha, beta)
                f.write(vequals + os.linesep); print(vequals)


                v_crit = np.sqrt(2/pdict["rho"]/pdict["A"]*np.sqrt(pdict["k1"]*pdict["E"]*pdict["I"]))
                v_raw = v_crit*alpha

                c_crit = np.sqrt(4*pdict["rho"]*pdict["A"]*pdict["k1"])
                c_raw = c_crit * beta

                tmax = pdict["L"] / v_raw

#                end_damp = 0.05

                if end_damp==0:
                    pdict["mu"] = c_raw
                    pdict["mubar"]=PolyLine([0,1],[1,1])
                else:
                    if beta==0:
                        pdict["mu"] = c_crit
                        pdict["mubar"]=PolyLine([0,end_damp,end_damp,1-end_damp,1-end_damp,1],
                                            [1,1,0,0,1,1])
                    else:
                        pdict["mu"]= c_raw
                        pdict["mubar"]=PolyLine([0,end_damp,end_damp,1-end_damp,1-end_damp,1],
                                            [c_crit/c_raw,c_crit/c_raw,1,1,c_crit/c_raw,c_crit/c_raw])

#                pdict["mu"] = c_raw
                pdict["tvals"] = np.linspace(tmax*xeval[0],tmax*xeval[1],nt)
                pdict["moving_loads_v_norm"]=[v_raw*np.sqrt(pdict["rho"]/pdict["E"])]
                pdict["file_stem"] = prefix + "DAF_alp{:5.3f}_bet{:5.3f}_n{:d}".format(alpha,beta,nterms)

                a = SpecBeam(**pdict)
                a.runme()
                a.saveme()

                jjj=np.searchsorted(alphas, 0.7)
                if i==0 and j==jjj:
                    a.animateme()
                w_now = np.max(a.defl[iwindow[0]:iwindow[1],:])

                if j==0:

                    if numerical_DAF:
                        w_0 = w_now
                    else:
                        lam = (pdict['k1'] /(4*pdict['E']*pdict['I']))**0.25
                        Q = pdict["moving_loads_Fz_norm"][0][0]* pdict['E'] * pdict['A']
                        w_0 = Q * lam / (2*pdict['k1'])
                        print("analytical")

                DAF = w_now/w_0

                x[j] = alpha
                y[j] = DAF


                end_time1 = time.time()
                elapsed_time = (end_time1 - start_time1); print("Run time={}".format(str(timedelta(seconds=elapsed_time))))

            if article_formatting:
                line,=ax.plot(x, y,
                          label="${}$".format(beta),markevery=(i/5*0.09, 0.1),**styles[i])
            else:
                line,=ax.plot(x, y,
                          label="${}$".format(beta))

        ######Analytical
#        figdum, axdum = plt.subplots(figsize=(6,8))
        alpha=np.linspace(alphas[0], alphas[-1] ,100)
        for j, bet in enumerate(betas):
            DAF = np.empty_like(alpha)
            for i, alp in enumerate(alpha):



#                s, disp=moving_load_on_beam_on_elastic_foundation(alp=alp, bet=bet, ax=axdum)
#                axdum.cla()
#                mx = np.max(disp)
                DAF[i] = DAFinfinite(alp=alp, bet=bet)
            if j==0:
                ax.plot(alpha, DAF, label="analytical\ninf. beam",
                        marker="+",ms=3,color="black",ls='None',markevery=5)
            else:
                ax.plot(alpha, DAF,
                        marker="+",ms=3,color="black",ls='None',markevery=5)
#        figdum.clf()
        #####end analytical


        #ax.set_title("$B=r_e,b_w=r_w,b_s=NA$")
        ax.set_ylim(0,4)
        ax.set_xlim(0,3)
        #ax.xaxis.label.set_fontsize(30)
        #ax.yaxis.label.set_fontsize(30)
        #ax.set_ylabel("$\\frac{u_\\mathrm{a,max}}{u_\\mathrm{p,max}}$")
#        ax.set_ylabel("$u_\\mathrm{a,max}/u_\\mathrm{p,max}$")
#        ax.set_xlabel("$n_a=r_e/r_w$")
    #    ax.set_xscale("log")
        ax.set_xlabel("Velocity ratio, $\\alpha$")
        ax.set_ylabel("Deflection amplification factor")

#        ax.set_xlim(3.5, 4.5)
        ax.grid()
        leg = ax.legend(title="$\\mathrm{Damping\ ratio,\ }\\beta$",
                        loc="upper right",
#                        ncol=2,
                        labelspacing=.2,
                        handlelength=2,
                        fontsize=8
                        )
        leg.draggable()

        ax.xaxis.labelpad = -1
        ax.yaxis.labelpad = 0

        fig.subplots_adjust(top=0.95, bottom=0.13, left=0.14, right=0.95)




        end_time0 = time.time()
        elapsed_time = (end_time0 - start_time0); print("Total run time={}".format(str(timedelta(seconds=elapsed_time))))
        f.write("Total run time={}".format(str(timedelta(seconds=elapsed_time))) + os.linesep)
#        a.animateme()

    if not saveas is None:
        save_figure(fig=fig, fname=saveas)

    plt.show()

def FIGURE_DAF_transition_length(saveas=None,k_ratio=4,beta=0.1,nterms=100,force_calc=True):
    """Dynamic amplification factor length of transition"""


    start_time0 = time.time()
    ftime = datetime.datetime.fromtimestamp(start_time0).strftime('%Y-%m-%d %H%M%S')
    with open(ftime + ".txt", "w") as f:
        f.write(ftime + os.linesep)


        fig = plt.figure(figsize=(3.54,3.54))
        matplotlib.rcParams.update({'font.size': 11})

        ax = fig.add_subplot("111")


        alphas = np.linspace(1e-5, 3, 10)
        t_lengths = np.linspace(0,25.0,10)
        #betas = [0, 0.05, 0.1, 0.3, 1.1, 2.0]


        pdict = OrderedDict(
                    E = 6.998*1e9, #Pa
                    rho = 2373, #kg/m3
                    L = 160, #m
                    A = 0.3*0.1, # m^2
                    I = 0.00022477900799999998,
                    k1 = 800002.6125,

#                    kf=5.41e-4,
                    #Fz_norm=1.013e-4,
#                    mu_norm=39.263/4/2/1000,
#                    mubar=PolyLine([0,0.05,0.05,.95,.95,1],[1,1,1,1,1,1]),
#                    k1_norm=97.552,
    #                k1bar=PolyLine([0,0.2,0.3,1],[1.0,1,0.1,0.1]),
#                    k1bar=PolyLine([0,0.2,0.2,1],[1.0,1,1,1]),

    #                k3_norm=2.497e6,
                    nterms=nterms,
                    BC="SS",
#                    nquad=20,
                    moving_loads_x_norm=[[0]],
                    moving_loads_Fz_norm=[[1.013e-4]],
#                    moving_loads_v_norm=[v,],#[0.01165,],
#                    tvals=t,
                    xvals_norm=np.linspace(0.15,0.85,100),
                    use_analytical=True,
                    implementation="fortran",
                    force_calc=force_calc,
                    )


        z = np.zeros((len(alphas), len(t_lengths)), dtype=float)



        for i, t_length in enumerate(t_lengths):
            for j, alpha in enumerate(alphas):

                start_time1 = time.time()

                f.write("*" * 70 + os.linesep)
#                vequals = "{}={}".format(vstr, vfmt.format(v))
                vequals = "alpha={}, t_len={}, k_ratio={}, beta={}".format(alpha, t_length, k_ratio, beta)
                f.write(vequals + os.linesep); print(vequals)


                v_crit = np.sqrt(2/pdict["rho"]/pdict["A"]*np.sqrt(pdict["k1"]*pdict["E"]*pdict["I"]))
                v_raw = v_crit*alpha

                c_crit = np.sqrt(4*pdict["rho"]*pdict["A"]*pdict["k1"])
                c_raw = c_crit*beta

                tmax = pdict["L"] / v_raw


                pdict["mu"] = c_raw
                pdict["tvals"] = np.linspace(tmax*0.3,tmax*0.80,100)
                pdict["moving_loads_v_norm"]=[v_raw*np.sqrt(pdict["rho"]/pdict["E"])]
                pdict["file_stem"] = "DAF_alp{:5.3f}_bet{:5.3f}_tlen{:5.3f}_krat{:5.3f}_n{:d}".format(alpha,beta,t_length,k_ratio,nterms)
                pdict["k1bar"]=PolyLine([0, 0.4, 0.4+t_length/pdict["L"],1],[1.0,1.0,k_ratio,k_ratio])

                a = SpecBeam(**pdict)
                a.runme()
                a.saveme()

                w_now = np.max(a.defl)

                if j==0:
                    w_0 = w_now

                DAF = w_now/w_0

                z[j, i]=DAF



                end_time1 = time.time()
                elapsed_time = (end_time1 - start_time1); print("Run time={}".format(str(timedelta(seconds=elapsed_time))))

                if i==0 and j==2:
                    a.animateme()

        #contourv = np.linspace(.1,10,10)
        CS = ax.contour(alphas, t_lengths, z )
        ax.clabel(CS,inline=1,fontsize=8,fmt='%.1f')





        #ax.set_title("$B=r_e,b_w=r_w,b_s=NA$")
        ax.set_ylim(0,25)
        ax.set_xlim(0,3)
        #ax.xaxis.label.set_fontsize(30)
        #ax.yaxis.label.set_fontsize(30)
        #ax.set_ylabel("$\\frac{u_\\mathrm{a,max}}{u_\\mathrm{p,max}}$")
#        ax.set_ylabel("$u_\\mathrm{a,max}/u_\\mathrm{p,max}$")
#        ax.set_xlabel("$n_a=r_e/r_w$")
    #    ax.set_xscale("log")
        ax.set_xlabel("Velocity ratio, $\\alpha$")
        ax.set_ylabel("Transition length")

#        ax.set_xlim(3.5, 4.5)
        ax.grid()
#        leg = ax.legend(title="$\\mathrm{Damping\ ratio,\ }\\beta$",
#                        loc="upper right",
##                        ncol=2,
#                        labelspacing=.2,
#                        handlelength=2,
#                        fontsize=8
#                        )
#        leg.draggable()

        ax.xaxis.labelpad = -1
        ax.yaxis.labelpad = 0

        fig.subplots_adjust(top=0.95, bottom=0.13, left=0.14, right=0.95)




        end_time0 = time.time()
        elapsed_time = (end_time0 - start_time0); print("Total run time={}".format(str(timedelta(seconds=elapsed_time))))
        f.write("Total run time={}".format(str(timedelta(seconds=elapsed_time))) + os.linesep)
#        a.animateme()

    if not saveas is None:
        save_figure(fig=fig, fname=saveas)

    plt.show()


def FIGURE_DAF_transition_length_esveld(saveas=None,k_ratio=4,beta=0.1,nterms=100,force_calc=True):
    """Dynamic amplification factor length of transition, reproduce esveld figure"""


    start_time0 = time.time()
    ftime = datetime.datetime.fromtimestamp(start_time0).strftime('%Y-%m-%d %H%M%S')
    with open(ftime + ".txt", "w") as f:
        f.write(ftime + os.linesep)


        fig = plt.figure(figsize=(3.54,3.54))
        matplotlib.rcParams.update({'font.size': 11})

        ax = fig.add_subplot("111")

        fig2 = plt.figure(figsize=(3.54,3.54))
        ax2 = fig2.add_subplot("111")

        alphas = [0.0001, 0.4, 0.9]
        t_lengths = np.linspace(0,25.0,6)
        #betas = [0, 0.05, 0.1, 0.3, 1.1, 2.0]


        pdict = OrderedDict(
                    E = 6.998*1e9, #Pa
                    rho = 2373, #kg/m3
                    L = 200,#160, #m
                    A = 0.3*0.1, # m^2
                    I = 0.00022477900799999998,
                    k1 = 800002.6125,

#                    kf=5.41e-4,
                    #Fz_norm=1.013e-4,
#                    mu_norm=39.263/4/2/1000,
#                    mubar=PolyLine([0,0.05,0.05,.95,.95,1],[1,1,1,1,1,1]),
#                    k1_norm=97.552,
    #                k1bar=PolyLine([0,0.2,0.3,1],[1.0,1,0.1,0.1]),
#                    k1bar=PolyLine([0,0.2,0.2,1],[1.0,1,1,1]),

    #                k3_norm=2.497e6,
                    nterms=nterms,
                    BC="SS",
#                    nquad=20,
                    moving_loads_x_norm=[[0]],
                    moving_loads_Fz_norm=[[1.013e-4]],
#                    moving_loads_v_norm=[v,],#[0.01165,],
#                    tvals=t,
                    #xvals_norm=np.linspace(0.15,0.85,100)
                    use_analytical=True,
                    implementation="fortran",
                    force_calc=force_calc,
                    )


        z = np.zeros((len(alphas), len(t_lengths)), dtype=float)



        for i, t_length in enumerate(t_lengths):
            for j, alpha in enumerate(alphas):

                start_time1 = time.time()

                f.write("*" * 70 + os.linesep)
#                vequals = "{}={}".format(vstr, vfmt.format(v))
                vequals = "alpha={}, t_len={}, k_ratio={}, beta={}".format(alpha, t_length, k_ratio, beta)
                f.write(vequals + os.linesep); print(vequals)


                v_crit = np.sqrt(2/pdict["rho"]/pdict["A"]*np.sqrt(pdict["k1"]*pdict["E"]*pdict["I"]))
                v_raw = v_crit*alpha

                c_crit = np.sqrt(4*pdict["rho"]*pdict["A"]*pdict["k1"])
                c_raw = c_crit*beta

                tmax = pdict["L"] / v_raw

                #pdict["mubar"] = PolyLine([0,0.1,0.9,1],[1/beta,1,1,1/beta])
                #pdict["mubar"] = PolyLine([0,0.1,0.1,0.9,0.9,1],[1/beta,1/beta,1,1,1/beta,1/beta])
                tran_start = 0.3
                pdict["mu"] = c_raw
                pdict["tvals"] = np.linspace(tmax*(tran_start-0.1),tmax*(tran_start+t_length/pdict["L"]+0.1),200)
                pdict["xvals_norm"]=np.linspace(tran_start-0.1,tran_start+t_length/pdict["L"]+0.1,300)
                pdict["moving_loads_v_norm"]=[v_raw * np.sqrt(pdict["rho"]/pdict["E"])]
                pdict["file_stem"] = "DAF_alp{:5.3f}_bet{:5.3f}_tlen{:5.3f}_krat{:5.3f}_n{:d}".format(alpha,beta,t_length,k_ratio,nterms)
                pdict["k1bar"]=PolyLine([0, tran_start, tran_start+t_length/pdict["L"],1],[1.0,1.0,k_ratio,k_ratio])
                pdict["k1bar"]=PolyLine([0, tran_start, tran_start+t_length/pdict["L"],1],[k_ratio,k_ratio,1.0,1.0])

                xstart = 0.0 #start position on beam
                xend = 1.0 #end position on beam
                pdict["tvals"] = np.linspace(0,tmax*(xend-xstart), 200)
                pdict["moving_loads_x0_norm"] = [xstart]
                pdict["moving_loads_L_norm"] = [xend - xstart]
                pdict["xvals_norm"]=np.linspace(xstart,xend,200)

                pdict["tvals"] = np.linspace(tmax*0.25, tmax*0.8, 300)
                pdict["xvals_norm"]=np.linspace(0.25, 0.8, 300)

                a = SpecBeam(**pdict)
                a.runme()
                a.saveme()

                w_now = np.max(a.defl)

                defl_max_envelope = np.max(a.defl, axis=1)
                defl_min_envelope = np.min(a.defl, axis=1)
                line,=ax2.plot(pdict["xvals_norm"],defl_max_envelope,
                         label="$\\alpha={},\ tlen={}$".format(alpha,t_length))
                ax2.plot(pdict["xvals_norm"],defl_min_envelope,
                         color = line.get_color(),
                         )
                if j==0:
                    w_0 = w_now

                DAF = w_now / w_0

                z[j, i] = DAF



                end_time1 = time.time()
                elapsed_time = (end_time1 - start_time1); print("Run time={}".format(str(timedelta(seconds=elapsed_time))))

                if 1 and i==0:#i==0:#i==len(t_lengths)-1: #i is t_length counter,j is alpha counter
                    a.animateme()
#                    print("******************",t_length,alpha, DAF)
#                    print(repr(z))
        for j, alpha in enumerate(alphas[1:]):
            x = t_lengths
            y = z[j+1,:]
            ax.plot(x,y,
                    marker='o',
                    label="${}$".format(alpha))

        #contourv = np.linspace(.1,10,10)
#        CS = ax.contour(alphas, t_lengths, z )
 #       ax.clabel(CS,inline=1,fontsize=8,fmt='%.1f')





        ax.set_title("$\\beta={},kratio={}$".format(beta, k_ratio))
        ax.set_xlim(0,25)
        ax.set_ylim(0,12)
        #ax.xaxis.label.set_fontsize(30)
        #ax.yaxis.label.set_fontsize(30)
        #ax.set_ylabel("$\\frac{u_\\mathrm{a,max}}{u_\\mathrm{p,max}}$")
#        ax.set_ylabel("$u_\\mathrm{a,max}/u_\\mathrm{p,max}$")
#        ax.set_xlabel("$n_a=r_e/r_w$")
    #    ax.set_xscale("log")
        ax.set_xlabel("Transition length (m)")
        ax.set_ylabel("DAF")

#        ax.set_xlim(3.5, 4.5)
        ax.grid()
        leg = ax.legend(title="$\\alpha$",
                        loc="upper right",
#                        ncol=2,
                        labelspacing=.2,
                        handlelength=2,
                        fontsize=8
                        )
        leg.draggable()

        ax.xaxis.labelpad = -1
        ax.yaxis.labelpad = 0

        fig.subplots_adjust(top=0.95, bottom=0.13, left=0.14, right=0.95)




        end_time0 = time.time()
        elapsed_time = (end_time0 - start_time0); print("Total run time={}".format(str(timedelta(seconds=elapsed_time))))
        f.write("Total run time={}".format(str(timedelta(seconds=elapsed_time))) + os.linesep)
#        a.animateme()

    if not saveas is None:
        save_figure(fig=fig, fname=saveas)

    plt.show()

def FIGURE_DAFwrtinf_transition_length(saveas=None,k_ratio=4,beta=0.1,nterms=100,force_calc=True):
    """Dynamic amplification factor with rest to length of transition"""

    ####

    start_time0 = time.time()
    ftime = datetime.datetime.fromtimestamp(start_time0).strftime('%Y-%m-%d %H%M%S')
    with open(ftime + ".txt", "w") as f:
        f.write(ftime + os.linesep)

        matplotlib.rcParams.update({'font.size': 11})
        fig = plt.figure(figsize=(3.54,3.54))
        ax = fig.add_subplot("111")

        fig2 = plt.figure(figsize=(3.54,3.54))
        ax2 = fig2.add_subplot("111")

        fig3 = plt.figure(figsize=(3.54,3.54))
        ax3 = fig3.add_subplot("111")

        alphas = [0.0001, 0.1, 0.3, 0.4, 0.5, 0.9, 1.1, 1.3]
        t_lengths = np.linspace(25,0,6)
        #betas = [0, 0.05, 0.1, 0.3, 1.1, 2.0]


        pdict = OrderedDict(
                    E = 6.998*1e9, #Pa
                    rho = 2373, #kg/m3
                    L = 200,#160, #m
                    A = 0.3*0.1, # m^2
                    I = 0.00022477900799999998,
                    k1 = 800002.6125,

#                    kf=5.41e-4,
                    #Fz_norm=1.013e-4,
#                    mu_norm=39.263/4/2/1000,
#                    mubar=PolyLine([0,0.05,0.05,.95,.95,1],[1,1,1,1,1,1]),
#                    k1_norm=97.552,
    #                k1bar=PolyLine([0,0.2,0.3,1],[1.0,1,0.1,0.1]),
#                    k1bar=PolyLine([0,0.2,0.2,1],[1.0,1,1,1]),

    #                k3_norm=2.497e6,
                    nterms=nterms,
                    BC="SS",
#                    nquad=20,
                    moving_loads_x_norm=[[0]],
                    moving_loads_Fz_norm=[[1.013e-4]],
#                    moving_loads_v_norm=[v,],#[0.01165,],
#                    tvals=t,
                    #xvals_norm=np.linspace(0.15,0.85,100)
                    use_analytical=True,
                    implementation="fortran",
                    force_calc=force_calc,
                    )


        zmax = np.zeros((len(alphas), len(t_lengths)), dtype=float)
        zmin = np.zeros((len(alphas), len(t_lengths)), dtype=float)




        for i, t_length in enumerate(t_lengths):
            for j, alpha in enumerate(alphas):

                start_time1 = time.time()

                f.write("*" * 70 + os.linesep)
#                vequals = "{}={}".format(vstr, vfmt.format(v))
                vequals = "alpha={}, t_len={}, k_ratio={}, beta={}".format(alpha, t_length, k_ratio, beta)
                f.write(vequals + os.linesep); print(vequals)


                v_crit = np.sqrt(2/pdict["rho"]/pdict["A"]*np.sqrt(pdict["k1"]*pdict["E"]*pdict["I"]))
                v_raw = v_crit*alpha

                c_crit = np.sqrt(4*pdict["rho"]*pdict["A"]*pdict["k1"])
                c_raw = c_crit*beta

                tmax = pdict["L"] / v_raw

                #pdict["mubar"] = PolyLine([0,0.1,0.9,1],[1/beta,1,1,1/beta])
                #pdict["mubar"] = PolyLine([0,0.1,0.1,0.9,0.9,1],[1/beta,1/beta,1,1,1/beta,1/beta])
                tran_start = 0.3
                pdict["mu"] = c_raw
                pdict["tvals"] = np.linspace(tmax*(tran_start-0.1),tmax*(tran_start+t_length/pdict["L"]+0.1),200)
                pdict["xvals_norm"]=np.linspace(tran_start-0.1,tran_start+t_length/pdict["L"]+0.1,300)
                pdict["moving_loads_v_norm"]=[v_raw * np.sqrt(pdict["rho"]/pdict["E"])]
                pdict["file_stem"] = "DAF_alp{:5.3f}_bet{:5.3f}_tlen{:5.3f}_krat{:5.3f}_n{:d}".format(alpha,beta,t_length,k_ratio,nterms)
                pdict["k1bar"]=PolyLine([0, tran_start, tran_start+t_length/pdict["L"],1],[1.0,1.0,k_ratio,k_ratio]) #soft to stiff
                pdict["k1bar"]=PolyLine([0, tran_start, tran_start+t_length/pdict["L"],1],[k_ratio,k_ratio,1.0,1.0]) #stiff to soft

                xstart = 0.0 #start position on beam
                xend = 1.0 #end position on beam
                pdict["tvals"] = np.linspace(0,tmax*(xend-xstart), 200)
                pdict["moving_loads_x0_norm"] = [xstart]
                pdict["moving_loads_L_norm"] = [xend - xstart]
                pdict["xvals_norm"]=np.linspace(xstart,xend,200)

                pdict["tvals"] = np.linspace(tmax*0.25, tmax*0.8, 300)
                pdict["xvals_norm"]=np.linspace(0.25, 0.8, 300)

                a = SpecBeam(**pdict)
                a.runme()
                a.saveme()

                w_now_max = np.max(a.defl)
                w_now_min = np.min(a.defl)

                defl_max_envelope = np.max(a.defl, axis=1)
                defl_min_envelope = np.min(a.defl, axis=1)
                line,=ax3.plot(pdict["xvals_norm"],defl_max_envelope,
                         label="$\\alpha={},\ tlen={}$".format(alpha,t_length))
                ax3.plot(pdict["xvals_norm"],defl_min_envelope,
                         color = line.get_color(),
                         )
#                if i==0:
#                    w_0 = w_now_max
#
#                DAF = w_now / w_0

                zmax[j, i] = w_now_max
                zmin[j, i] = w_now_min



                end_time1 = time.time()
                elapsed_time = (end_time1 - start_time1); print("Run time={}".format(str(timedelta(seconds=elapsed_time))))

                if 0 and i==0:#i==0:#i==len(t_lengths)-1: #i is t_length counter,j is alpha counter
                    a.animateme()
#                    print("******************",t_length,alpha, DAF)
#                    print(repr(z))
        for j, alpha in enumerate(alphas):
            x = t_lengths
            y = zmax[j, :] / zmax[j,0]
            ax.plot(x,y,
                    marker='o',
                    label="${}$".format(alpha))
            y = zmin[j, :] / zmin[j, 0]
            ax2.plot(x,y,
                    marker='s',
                    label="${}$".format(alpha))
        #contourv = np.linspace(.1,10,10)
#        CS = ax.contour(alphas, t_lengths, z )
 #       ax.clabel(CS,inline=1,fontsize=8,fmt='%.1f')

        adjust = dict(top=0.9, bottom=0.13, left=0.19, right=0.95)
        ###################
        ax2.set_title("$\\beta={},k_1/k_2={}$".format(beta, k_ratio))
        ax2.set_xlabel("Transition length (m)")
        ax2.set_ylabel("DAFup w.r.t. infinite")
        ax2.grid()
        leg = ax2.legend(title="$\\alpha$",
                        loc="upper right",
#                        ncol=2,
                        labelspacing=.2,
                        handlelength=2,
                        fontsize=8
                        )
        leg.draggable()
        ax2.xaxis.labelpad = -1
        ax2.yaxis.labelpad = 0
        fig2.subplots_adjust(**adjust)
        ####################

        ax.set_title("$\\beta={},kratio={}$".format(beta, k_ratio))
#        ax.set_xlim(0,25)
#        ax.set_ylim(0,12)
        #ax.xaxis.label.set_fontsize(30)
        #ax.yaxis.label.set_fontsize(30)
        #ax.set_ylabel("$\\frac{u_\\mathrm{a,max}}{u_\\mathrm{p,max}}$")
#        ax.set_ylabel("$u_\\mathrm{a,max}/u_\\mathrm{p,max}$")
#        ax.set_xlabel("$n_a=r_e/r_w$")
    #    ax.set_xscale("log")
        ax.set_xlabel("Transition length (m)")
        ax.set_ylabel("DAF w.r.t. infinite")

#        ax.set_xlim(3.5, 4.5)
        ax.grid()
        leg = ax.legend(title="$\\alpha$",
                        loc="upper right",
#                        ncol=2,
                        labelspacing=.2,
                        handlelength=2,
                        fontsize=8
                        )
        leg.draggable()

        ax.xaxis.labelpad = -1
        ax.yaxis.labelpad = 0

        fig.subplots_adjust(**adjust)




        end_time0 = time.time()
        elapsed_time = (end_time0 - start_time0); print("Total run time={}".format(str(timedelta(seconds=elapsed_time))))
        f.write("Total run time={}".format(str(timedelta(seconds=elapsed_time))) + os.linesep)
#        a.animateme()

    if not saveas is None:
        save_figure(fig=fig, fname=saveas)
        save_figure(fig=fig2, fname=saveas + "_up")

#    plt.show()
    fig.clf()
    fig2.clf()

def FIGURE_DAFwrtinf_transition_lengthrev(saveas=None,k_ratio=4,beta=0.1,nterms=100,force_calc=True):
    """Dynamic amplification factor with rest to length of transition"""


    start_time0 = time.time()
    ftime = datetime.datetime.fromtimestamp(start_time0).strftime('%Y-%m-%d %H%M%S')
    with open(ftime + ".txt", "w") as f:
        f.write(ftime + os.linesep)

        matplotlib.rcParams.update({'font.size': 11})
        fig = plt.figure(figsize=(3.54,3.54))
        ax = fig.add_subplot("111")

        fig2 = plt.figure(figsize=(3.54,3.54))
        ax2 = fig2.add_subplot("111")

        fig3 = plt.figure(figsize=(3.54,3.54))
        ax3 = fig3.add_subplot("111")

        alphas = [0.0001, 0.1, 0.3, 0.4, 0.5, 0.9, 1.1, 1.3]
        t_lengths = np.linspace(25,0,6)
        #betas = [0, 0.05, 0.1, 0.3, 1.1, 2.0]


        pdict = OrderedDict(
                    E = 6.998*1e9, #Pa
                    rho = 2373, #kg/m3
                    L = 200,#160, #m
                    A = 0.3*0.1, # m^2
                    I = 0.00022477900799999998,
                    k1 = 800002.6125,

#                    kf=5.41e-4,
                    #Fz_norm=1.013e-4,
#                    mu_norm=39.263/4/2/1000,
#                    mubar=PolyLine([0,0.05,0.05,.95,.95,1],[1,1,1,1,1,1]),
#                    k1_norm=97.552,
    #                k1bar=PolyLine([0,0.2,0.3,1],[1.0,1,0.1,0.1]),
#                    k1bar=PolyLine([0,0.2,0.2,1],[1.0,1,1,1]),

    #                k3_norm=2.497e6,
                    nterms=nterms,
                    BC="SS",
#                    nquad=20,
                    moving_loads_x_norm=[[0]],
                    moving_loads_Fz_norm=[[1.013e-4]],
#                    moving_loads_v_norm=[v,],#[0.01165,],
#                    tvals=t,
                    #xvals_norm=np.linspace(0.15,0.85,100)
                    use_analytical=True,
                    implementation="fortran",
                    force_calc=force_calc,
                    )


        zmax = np.zeros((len(alphas), len(t_lengths)), dtype=float)
        zmin = np.zeros((len(alphas), len(t_lengths)), dtype=float)




        for i, t_length in enumerate(t_lengths):
            for j, alpha in enumerate(alphas):

                start_time1 = time.time()

                f.write("*" * 70 + os.linesep)
#                vequals = "{}={}".format(vstr, vfmt.format(v))
                vequals = "alpha={}, t_len={}, k_ratio={}, beta={}".format(alpha, t_length, k_ratio, beta)
                f.write(vequals + os.linesep); print(vequals)


                v_crit = np.sqrt(2/pdict["rho"]/pdict["A"]*np.sqrt(pdict["k1"]*pdict["E"]*pdict["I"]))
                v_raw = v_crit*alpha

                c_crit = np.sqrt(4*pdict["rho"]*pdict["A"]*pdict["k1"])
                c_raw = c_crit*beta

                tmax = pdict["L"] / v_raw

                #pdict["mubar"] = PolyLine([0,0.1,0.9,1],[1/beta,1,1,1/beta])
                #pdict["mubar"] = PolyLine([0,0.1,0.1,0.9,0.9,1],[1/beta,1/beta,1,1,1/beta,1/beta])
                tran_start = 0.6
                pdict["mu"] = c_raw
                pdict["tvals"] = np.linspace(tmax*(tran_start-0.1),tmax*(tran_start+t_length/pdict["L"]+0.1),200)
                pdict["xvals_norm"]=np.linspace(tran_start-0.1,tran_start+t_length/pdict["L"]+0.1,300)
                pdict["moving_loads_v_norm"]=[v_raw * np.sqrt(pdict["rho"]/pdict["E"])]
                pdict["file_stem"] = "revDAF_alp{:5.3f}_bet{:5.3f}_tlen{:5.3f}_krat{:5.3f}_n{:d}".format(alpha,beta,t_length,k_ratio,nterms)
                pdict["k1bar"]=PolyLine([0, tran_start, tran_start+t_length/pdict["L"],1],[1.0,1.0,k_ratio,k_ratio]) #soft to stiff
#                pdict["k1bar"]=PolyLine([0, tran_start, tran_start+t_length/pdict["L"],1],[k_ratio,k_ratio,1.0,1.0])#stiff to soft

                xstart = 0.0 #start position on beam
                xend = 1.0 #end position on beam
                pdict["tvals"] = np.linspace(0,tmax*(xend-xstart), 200)
                pdict["moving_loads_x0_norm"] = [xstart]
                pdict["moving_loads_L_norm"] = [xend - xstart]
                pdict["xvals_norm"]=np.linspace(xstart,xend,200)

                pdict["tvals"] = np.linspace(tmax*0.25, tmax*0.8, 300)
                pdict["xvals_norm"]=np.linspace(0.25, 0.8, 300)

                a = SpecBeam(**pdict)
                a.runme()
                a.saveme()

                w_now_max = np.max(a.defl)
                w_now_min = np.min(a.defl)

                defl_max_envelope = np.max(a.defl, axis=1)
                defl_min_envelope = np.min(a.defl, axis=1)
                line,=ax3.plot(pdict["xvals_norm"],defl_max_envelope,
                         label="$\\alpha={},\ tlen={}$".format(alpha,t_length))
                ax3.plot(pdict["xvals_norm"],defl_min_envelope,
                         color = line.get_color(),
                         )
#                if i==0:
#                    w_0 = w_now_max
#
#                DAF = w_now / w_0

                zmax[j, i] = w_now_max
                zmin[j, i] = w_now_min



                end_time1 = time.time()
                elapsed_time = (end_time1 - start_time1); print("Run time={}".format(str(timedelta(seconds=elapsed_time))))

                if 0 and i==0:#i==0:#i==len(t_lengths)-1: #i is t_length counter,j is alpha counter
                    a.animateme()
#                    print("******************",t_length,alpha, DAF)
#                    print(repr(z))
        for j, alpha in enumerate(alphas):
            x = t_lengths
            y = zmax[j, :] / zmax[j,0]
            ax.plot(x,y,
                    marker='o',
                    label="${}$".format(alpha))
            y = zmin[j, :] / zmin[j, 0]
            ax2.plot(x,y,
                    marker='s',
                    label="${}$".format(alpha))
        #contourv = np.linspace(.1,10,10)
#        CS = ax.contour(alphas, t_lengths, z )
 #       ax.clabel(CS,inline=1,fontsize=8,fmt='%.1f')

        adjust = dict(top=0.9, bottom=0.13, left=0.19, right=0.95)
        ###################
        ax2.set_title("$\\beta={},k_1/k_2={}$".format(beta, k_ratio))
        ax2.set_xlabel("Transition length (m)")
        ax2.set_ylabel("DAFup w.r.t. infinite")
        ax2.grid()
        leg = ax2.legend(title="$\\alpha$",
                        loc="upper right",
#                        ncol=2,
                        labelspacing=.2,
                        handlelength=2,
                        fontsize=8
                        )
        leg.draggable()
        ax2.xaxis.labelpad = -1
        ax2.yaxis.labelpad = 0
        fig2.subplots_adjust(**adjust)
        ####################

        ax.set_title("$\\beta={},kratio={}$".format(beta, k_ratio))
#        ax.set_xlim(0,25)
#        ax.set_ylim(0,12)
        #ax.xaxis.label.set_fontsize(30)
        #ax.yaxis.label.set_fontsize(30)
        #ax.set_ylabel("$\\frac{u_\\mathrm{a,max}}{u_\\mathrm{p,max}}$")
#        ax.set_ylabel("$u_\\mathrm{a,max}/u_\\mathrm{p,max}$")
#        ax.set_xlabel("$n_a=r_e/r_w$")
    #    ax.set_xscale("log")
        ax.set_xlabel("Transition length (m)")
        ax.set_ylabel("DAF w.r.t. infinite")

#        ax.set_xlim(3.5, 4.5)
        ax.grid()
        leg = ax.legend(title="$\\alpha$",
                        loc="upper right",
#                        ncol=2,
                        labelspacing=.2,
                        handlelength=2,
                        fontsize=8
                        )
        leg.draggable()

        ax.xaxis.labelpad = -1
        ax.yaxis.labelpad = 0

        fig.subplots_adjust(**adjust)




        end_time0 = time.time()
        elapsed_time = (end_time0 - start_time0); print("Total run time={}".format(str(timedelta(seconds=elapsed_time))))
        f.write("Total run time={}".format(str(timedelta(seconds=elapsed_time))) + os.linesep)
#        a.animateme()

    if not saveas is None:
        save_figure(fig=fig, fname=saveas)
        save_figure(fig=fig2, fname=saveas + "_up")

#    plt.show()
    fig.clf()
    fig2.clf()

def FIGURE_DAFwrtinf_transition_length_betaconst(saveas=None,k_ratio=4,beta=0.1,nterms=100,force_calc=True):
    """Dynamic amplification factor with rest to length of transition with
    beta term constant"""

    ####

    start_time0 = time.time()
    ftime = datetime.datetime.fromtimestamp(start_time0).strftime('%Y-%m-%d %H%M%S')
    with open(ftime + ".txt", "w") as f:
        f.write(ftime + os.linesep)

        matplotlib.rcParams.update({'font.size': 11})
        fig = plt.figure(figsize=(3.54,3.54))
        ax = fig.add_subplot("111")

        fig2 = plt.figure(figsize=(3.54,3.54))
        ax2 = fig2.add_subplot("111")

        fig3 = plt.figure(figsize=(3.54,3.54))
        ax3 = fig3.add_subplot("111")

        alphas = [0.0001, 0.1, 0.3, 0.4, 0.5, 0.9, 1.1, 1.3]
        t_lengths = np.linspace(25,0,6)
        #betas = [0, 0.05, 0.1, 0.3, 1.1, 2.0]


        pdict = OrderedDict(
                    E = 6.998*1e9, #Pa
                    rho = 2373, #kg/m3
                    L = 200,#160, #m
                    A = 0.3*0.1, # m^2
                    I = 0.00022477900799999998,
                    k1 = 800002.6125,

#                    kf=5.41e-4,
                    #Fz_norm=1.013e-4,
#                    mu_norm=39.263/4/2/1000,
#                    mubar=PolyLine([0,0.05,0.05,.95,.95,1],[1,1,1,1,1,1]),
#                    k1_norm=97.552,
    #                k1bar=PolyLine([0,0.2,0.3,1],[1.0,1,0.1,0.1]),
#                    k1bar=PolyLine([0,0.2,0.2,1],[1.0,1,1,1]),

    #                k3_norm=2.497e6,
                    nterms=nterms,
                    BC="SS",
#                    nquad=20,
                    moving_loads_x_norm=[[0]],
                    moving_loads_Fz_norm=[[1.013e-4]],
#                    moving_loads_v_norm=[v,],#[0.01165,],
#                    tvals=t,
                    #xvals_norm=np.linspace(0.15,0.85,100)
                    use_analytical=True,
                    implementation="fortran",
                    force_calc=force_calc,
                    )


        zmax = np.zeros((len(alphas), len(t_lengths)), dtype=float)
        zmin = np.zeros((len(alphas), len(t_lengths)), dtype=float)




        for i, t_length in enumerate(t_lengths):
            for j, alpha in enumerate(alphas):

                start_time1 = time.time()

                f.write("*" * 70 + os.linesep)
#                vequals = "{}={}".format(vstr, vfmt.format(v))
                vequals = "alpha={}, t_len={}, k_ratio={}, beta={}".format(alpha, t_length, k_ratio, beta)
                f.write(vequals + os.linesep); print(vequals)


                v_crit = np.sqrt(2/pdict["rho"]/pdict["A"]*np.sqrt(pdict["k1"]*pdict["E"]*pdict["I"]))
                v_raw = v_crit*alpha

                c_crit = np.sqrt(4*pdict["rho"]*pdict["A"]*pdict["k1"])
                c_raw = c_crit*beta

                tmax = pdict["L"] / v_raw

                #pdict["mubar"] = PolyLine([0,0.1,0.9,1],[1/beta,1,1,1/beta])
                #pdict["mubar"] = PolyLine([0,0.1,0.1,0.9,0.9,1],[1/beta,1/beta,1,1,1/beta,1/beta])
                tran_start = 0.3
                pdict["mu"] = c_raw
                pdict["tvals"] = np.linspace(tmax*(tran_start-0.1),tmax*(tran_start+t_length/pdict["L"]+0.1),200)
                pdict["xvals_norm"]=np.linspace(tran_start-0.1,tran_start+t_length/pdict["L"]+0.1,300)
                pdict["moving_loads_v_norm"]=[v_raw * np.sqrt(pdict["rho"]/pdict["E"])]
                pdict["file_stem"] = "betaconst_DAF_alp{:5.3f}_bet{:5.3f}_tlen{:5.3f}_krat{:5.3f}_n{:d}".format(alpha,beta,t_length,k_ratio,nterms)
                pdict["k1bar"]=PolyLine([0, tran_start, tran_start+t_length/pdict["L"],1],[1.0,1.0,k_ratio,k_ratio]) #soft to stiff
                pdict["k1bar"]=PolyLine([0, tran_start, tran_start+t_length/pdict["L"],1],[k_ratio, k_ratio,1.0,1.0]) #stiff to soft


                if t_length>0:
                    mubar_x = [0]
                    mubar_mag=[k_ratio]
                    for xx in np.linspace(tran_start, tran_start+t_length/pdict["L"], 6):
                        mubar_x.append(xx)

                        know = 1.0 * np.sqrt(k_ratio)*np.sqrt(1+(1/k_ratio-1)*(xx-tran_start)/(t_length/pdict["L"])) #stiff to soft
                        mubar_mag.append(know)
                    mubar_x.append(1.0)
                    mubar_mag.append(1.0)
                    pdict["mubar"] = PolyLine(mubar_x, mubar_mag)
                    for xx, mag in zip(mubar_x, mubar_mag):
                        print("{:8.3g}, {:8.3g}".format(xx, mag))
                else:
                    pdict["mubar"] = PolyLine([0],[1],[1],[1])

                xstart = 0.0 #start position on beam
                xend = 1.0 #end position on beam
                pdict["tvals"] = np.linspace(0,tmax*(xend-xstart), 200)
                pdict["moving_loads_x0_norm"] = [xstart]
                pdict["moving_loads_L_norm"] = [xend - xstart]
                pdict["xvals_norm"]=np.linspace(xstart,xend,200)

                pdict["tvals"] = np.linspace(tmax*0.25, tmax*0.8, 300)
                pdict["xvals_norm"]=np.linspace(0.25, 0.8, 300)

                a = SpecBeam(**pdict)
                a.runme()
                a.saveme()

                w_now_max = np.max(a.defl)
                w_now_min = np.min(a.defl)

                defl_max_envelope = np.max(a.defl, axis=1)
                defl_min_envelope = np.min(a.defl, axis=1)
                line,=ax3.plot(pdict["xvals_norm"],defl_max_envelope,
                         label="$\\alpha={},\ tlen={}$".format(alpha,t_length))
                ax3.plot(pdict["xvals_norm"],defl_min_envelope,
                         color = line.get_color(),
                         )
#                if i==0:
#                    w_0 = w_now_max
#
#                DAF = w_now / w_0

                zmax[j, i] = w_now_max
                zmin[j, i] = w_now_min



                end_time1 = time.time()
                elapsed_time = (end_time1 - start_time1); print("Run time={}".format(str(timedelta(seconds=elapsed_time))))

                if 1 and i==0:#i==0:#i==len(t_lengths)-1: #i is t_length counter,j is alpha counter
                    a.animateme()
#                    print("******************",t_length,alpha, DAF)
#                    print(repr(z))
        for j, alpha in enumerate(alphas):
            x = t_lengths
            y = zmax[j, :] / zmax[j,0]
            ax.plot(x,y,
                    marker='o',
                    label="${}$".format(alpha))
            y = zmin[j, :] / zmin[j, 0]
            ax2.plot(x,y,
                    marker='s',
                    label="${}$".format(alpha))
        #contourv = np.linspace(.1,10,10)
#        CS = ax.contour(alphas, t_lengths, z )
 #       ax.clabel(CS,inline=1,fontsize=8,fmt='%.1f')

        adjust = dict(top=0.9, bottom=0.13, left=0.19, right=0.95)
        ###################
        ax2.set_title("$\\beta={},k_1/k_2={}$".format(beta, k_ratio))
        ax2.set_xlabel("Transition length (m)")
        ax2.set_ylabel("DAFup w.r.t. infinite")
        ax2.grid()
        leg = ax2.legend(title="$\\alpha$",
                        loc="upper right",
#                        ncol=2,
                        labelspacing=.2,
                        handlelength=2,
                        fontsize=8
                        )
        leg.draggable()
        ax2.xaxis.labelpad = -1
        ax2.yaxis.labelpad = 0
        fig2.subplots_adjust(**adjust)
        ####################

        ax.set_title("$\\beta={},kratio={}$".format(beta, k_ratio))
#        ax.set_xlim(0,25)
#        ax.set_ylim(0,12)
        #ax.xaxis.label.set_fontsize(30)
        #ax.yaxis.label.set_fontsize(30)
        #ax.set_ylabel("$\\frac{u_\\mathrm{a,max}}{u_\\mathrm{p,max}}$")
#        ax.set_ylabel("$u_\\mathrm{a,max}/u_\\mathrm{p,max}$")
#        ax.set_xlabel("$n_a=r_e/r_w$")
    #    ax.set_xscale("log")
        ax.set_xlabel("Transition length (m)")
        ax.set_ylabel("DAF w.r.t. infinite")

#        ax.set_xlim(3.5, 4.5)
        ax.grid()
        leg = ax.legend(title="$\\alpha$",
                        loc="upper right",
#                        ncol=2,
                        labelspacing=.2,
                        handlelength=2,
                        fontsize=8
                        )
        leg.draggable()

        ax.xaxis.labelpad = -1
        ax.yaxis.labelpad = 0

        fig.subplots_adjust(**adjust)




        end_time0 = time.time()
        elapsed_time = (end_time0 - start_time0); print("Total run time={}".format(str(timedelta(seconds=elapsed_time))))
        f.write("Total run time={}".format(str(timedelta(seconds=elapsed_time))) + os.linesep)
#        a.animateme()

    if not saveas is None:
        save_figure(fig=fig, fname=saveas)
        save_figure(fig=fig2, fname=saveas + "_up")

#    plt.show()
    fig.clf()
    fig2.clf()



def moving_load_on_beam_on_elastic_foundation(alp, bet,slim=(-6,6),npts=1001):
    """Displacement vs distance from moving point load on infinite beam on
    elastic foundation.

    This is to generate data for a a single subplot of  esveld Figure 6.17

    Esveld, C. (2001). Modern railway track. MRT-productions Zaltbommel, The Netherlands, Netherlands.0

    s is in multiples of lambda.

    Parameters
    ----------
    alp : float
        velocity ratio.
    bet : float
        damping ratio
    slim : two element tuple of float, optional
        Distance, in multiples of characteristic length,  on either side
        of moving load to assess deflection at Default slim = (-6,6).
    npts : int, optional
        number of points witin slim to evaluate deflection.  Use an odd
        value to capture the mid point directly under the mmoving load.
        Default npts=1001.

    Returns
    -------
    s : 1D numpy array of float
        Distances, in multiples of characteristic length,
        from moving point load.
        w.r.t. normalised by characteristic lengths
    disp : 1D numpy array of float
        Displacements at s.

    """

    gam = np.roots([1.0, 0.0, 4.0*alp**2,-8*alp*bet,4])
    #print(gam)

    cmat = np.zeros((4,4), dtype=complex)
    cmat[0,:]=[1,1,-1,-1]
    cmat[1,:]=[gam[0], gam[1], -gam[2], -gam[3]]
    cmat[2,:]=[gam[0]**2, gam[1]**2, -gam[2]**2, -gam[3]**2]
    cmat[3,:]=[gam[0]**3, gam[1]**3, -gam[2]**3, -gam[3]**3]

    rhs = np.array([0, 0, 0, 8.0], dtype=complex)

    A=np.linalg.solve(cmat, rhs)
    #print(A)

    smin,smax=slim
    s=np.linspace(smin,smax,npts, dtype=complex)

    disp = np.empty(len(s), dtype=complex)

    s_neg = s[s<0]
    s_pos = s[s>=0]



    #assume that gam0 and gam1 are have negative real components
    disp[s>=0]=A[0]*np.exp(s_pos*gam[0])+A[1]*np.exp(s_pos*gam[1])
    disp[s<0]=A[2]*np.exp(s_neg*gam[2])+A[3]*np.exp(s_neg*gam[3])

    return  s, disp

def esveld_fig6p17():
    """Displacement shapes before and after a moving
    point load on infinite beam on elastic foundation for
    a sub crtical , critical, super critcal velocity ratio (alpha) and
    damping ratios (beta) , Esveld Figure 6p17.


    Esveld, C. (2001). Modern railway track. MRT-productions Zaltbommel, The Netherlands, Netherlands.

    Returns
    -------
    fig : matplotlib figure
        the deflection shape plot.

    """
    fig, axs = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(12,11))

    for i,alp in enumerate([0,.5,1,2]):
        for j,bet in enumerate([0,.1,1.1]):


            s, disp =moving_load_on_beam_on_elastic_foundation(alp=alp, bet=bet)
            axs[i,j].plot(s,disp)
            axs[i,j].set_ylim([-2,3])
            axs[i,j].invert_yaxis()
            axs[i,j].set_ylabel("rel vert displ")
            axs[i,j].set_xlabel("Rel dist, s $\\alpha="+str(alp)+",\\beta=" +str(bet) +"$")
            axs[i,j].grid()

    return fig


def esveld_fig6p18():
    """Dynamic Amplification Factor vs
    velocity ratio (alpha) for variuos damping ratios (beta) for moving
    point load on infinite beam on elastic foundation, Esveld Figure 6p18.


    Esveld, C. (2001). Modern railway track. MRT-productions Zaltbommel, The Netherlands, Netherlands.

    Returns
    -------
    fig : matplotlib figure
        the Dynamic amplification plotplot of
    """
    fig, ax = plt.subplots(figsize=(6,8))

    alpha=np.linspace(0,3,100)
    #DAF = np.empty_like(alp)
    for j,bet in enumerate([0,0.05,0.1,0.3,1.1,2.0]):
        DAF = np.empty_like(alpha)
        for i, alp in enumerate(alpha):



            s, disp=moving_load_on_beam_on_elastic_foundation(alp=alp, bet=bet)

            mx = np.max(disp)
            DAF[i] = mx
        ax.plot(alpha, DAF, label="$\\beta=" + str(bet) + "$")
    leg = ax.legend()
    leg.draggable()
    ax.set_ylim([0,4])
    ax.set_xlim([0,3])
    ax.grid()


    ax.set_ylabel("Deflection amplification factor")
    ax.set_xlabel("Rel dist, s $\\alpha="+str(alp)+",\\beta=" +str(bet) +"$")

    return fig




def case_study_PaixaoEtAl2014(
       saveas=None,
       nterms=100,
       force_calc=True,
       nx=100, nt=100):
    """Try and replicate Paixao et al 2014 for 6 rail cars going over
    transition using specbeam.

    Some eplots will be produced.

    You'll have to read the source code for this one.

    Parameters
    ----------
    saveas : string, optional
        Filename stem (not including extension) to save figure to.
        Default saveas=None, i.e. figure won't be saved to disk.
    nx : int, optional
        Number of points between `xlim[0]` and `xlim[1]` at which to
        calculate deflection.  Default nx = 100.
    nt : int, optional
        Number of evaluation times from when load enters `xlim[0]` and
        leaves `xlim[1]`.  Default nt=100
    nterms : int, optional
        Number of series terms in approximation.  More terms  leads to higher
        accuracy but also longer computation times.  Start of with a
        small enough value to observe behaviour and then increase
        to confirm. Default nterms=100 .
    force_calc : [True, False], optional
        When True all calcuations will be done from scratch. When False
        Current working directory will be searched for a results file
        named according to "saveas" and
        if found it will be loaded.  Be careful as name of results file may
        not be unique to your analysis paramters.  Safest to use
        force_calc = True, but if you are simply regenerating graphs then
        consider force_calc=False.  Default force_calc=True.

    """

    ####

    start_time0 = time.time()
    ftime = datetime.datetime.fromtimestamp(start_time0).strftime('%Y-%m-%d %H%M%S')


    #raw digitized data,
    #time in seconds but not sure what time zero represents, probably just
    # offset the specbeam data to line up with the peaks, which is what
    # paxior seems to do
    #Displacemtn in mm positive up

    fig12top_t_raw = np.array(
      [ 0.12883271,  0.13973975,  0.1478486 ,  0.15608597,  0.17517319,
        0.18598029,  0.19146229,  0.20236935,  0.21608873,  0.22693857,
        0.22956522,  0.24040088,  0.24591183,  0.25674748,  0.26498448,
        0.27853267,  0.28120274,  0.29483585,  0.30842746,  0.31379463,
        0.3274287 ,  0.34114769,  0.35198335,  0.35465343,  0.35733701,
        0.37639615,  0.39545432,  0.4090884 ,  0.43087358,  0.45003211,
        0.46359477,  0.48263944,  0.49891464,  0.51260468,  0.5317497 ,
        0.5398294 ,  0.55340654,  0.56148624,  0.5697242 ,  0.5860129 ,
        0.59135209,  0.59961803,  0.60769869,  0.61592217,  0.62133277,
        0.62955528,  0.63769288,  0.64029155,  0.64558731,  0.64814352,
        0.65066981,  0.65316812,  0.65569538,  0.65826509,  0.66657349,
        0.67219831,  0.67506813,  0.67793795,  0.68077883,  0.6836197 ,
        0.68901582,  0.69165694,  0.6942836 ,  0.69688226,  0.69922328,
        0.69942303,  0.70452001,  0.71281393,  0.71319992,  0.71572621,
        0.71609773,  0.71899553,  0.73024516,  0.73308604,  0.73861146,
        0.74690539,  0.75243081,  0.75796974,  0.76623568,  0.77981282,
        0.785152  ,  0.79336005,  0.80695166,  0.81506031,  0.81775837,
        0.82867991,  0.83953004,  0.85042263,  0.85580427,  0.86131522,
        0.86952423,  0.87773227,  0.88585539,  0.89944701,  0.90757013,
        0.91572124,  0.91839132,  0.93740703,  0.94564499,  0.9509697 ,
        0.95630888,  0.95873578,  0.95890755,  0.96363204,  0.96388969,
        0.96605991,  0.96614483,  0.9715699 ,  0.97710883,  0.97999313,
        0.98287646,  0.98563241,  0.98576076,  0.99131416,  0.99141355,
        0.99695345,  1.00222123,  1.0050621 ,  1.00738865,  1.00756041,
        1.00991591,  1.01001626,  1.0181529 ,  1.02092236,  1.02377771,
        1.02656165,  1.02944595,  1.02955982,  1.03237271,  1.03525604,
        1.03809691,  1.03819727,  1.04367927,  1.0437362 ,  1.04923268,
        1.05201662,  1.05755555,  1.06298062,  1.06562174,  1.06823392,
        1.07081811,  1.07335888,  1.07588614,  1.0783555 ,  1.08092521,
        1.08352388,  1.09192202,  1.09198957,  1.09481693,  1.09765395,
        1.10324113,  1.10332798,  1.10622289,  1.11435759,  1.11699196,
        1.11945264,  1.11961669,  1.12198086,  1.12455734,  1.12716276,
        1.13274994,  1.13314558,  1.1356159 ,  1.135751  ,  1.13599224,
        1.1388582 ,  1.14713765,  1.15000362,  1.15285993,  1.15573554,
        1.16401499,  1.17223654,  1.17773688,  1.18053529,  1.18878579,
        1.19957417,  1.19964172,  1.20222784,  1.21588217,  1.23223843,
        1.24307505,  1.25395028,  1.25933481,  1.2648062 ,  1.27838334,
        1.28658559,  1.29747046,  1.30295149,  1.31105725,  1.32193247,
        1.33287524,  1.33550961,  1.35997162,  1.37088544,  1.37355841,
        1.37621208,  1.37865346,  1.37880785,  1.3839222 ,  1.38636357,
        1.38651797,  1.39147792,  1.39164196,  1.39413159,  1.39693   ,
        1.39988281,  1.40250753,  1.4053928 ,  1.40826841,  1.40836491,
        1.41119227,  1.41950067,  1.42218329,  1.42756783,  1.4302022 ,
        1.43263392,  1.43280762,  1.43530689,  1.44065283,  1.4434802 ,
        1.44635581,  1.44915422,  1.45475106,  1.4548765 ,  1.45767492,
        1.45781966,  1.46336825,  1.46351299,  1.46639825,  1.47737962,
        1.48291855,  1.48825484,  1.48834169,  1.49352358,  1.49608076,
        1.49862828,  1.50096351,  1.5011758 ,  1.50361718,  1.506184  ,
        1.51171328,  1.51190628,  1.51747416,  1.52304204,  1.52586941,
        1.52874502,  1.53422605,  1.53962024,  1.54212916,  1.54224496,
        1.54717596,  1.5473979 ,  1.552464  ,  1.55556156,  1.56085925,
        1.56121629,  1.56417875,  1.56708331,  1.56711226,  1.57543996,
        1.57555575,  1.58654677,  1.58937414,  1.59763429,  1.60313462,
        1.60323112,  1.6087604 ,  1.61959702,  1.62764488,  1.62772208,
        1.64390464,  1.65211654,  1.65759757,  1.66575158,  1.68204993,
        1.69026184,  1.69568497,  1.70390652,  1.70666634,  1.71480104,
        1.72028208,  1.73385922,  1.74471514,  1.75010933,  1.75823438,
        1.76086875,  1.77181152,  1.77727326,  1.78542726,  1.79084075,
        1.7989465 ,  1.80154227,  1.80411875,  1.80644432,  1.80668557,
        1.80895325,  1.81151042,  1.81681776,  1.82501036,  1.82515511,
        1.83075194,  1.8336372 ,  1.83659001,  1.8421579 ,  1.85278222,
        1.85296557,  1.85806062,  1.86060814,  1.86316531,  1.86878144,
        1.87171495,  1.8719369 ,  1.88026459,  1.88313056,  1.88330425,
        1.88616057,  1.88904583,  1.89452686,  1.90278701,  1.90284491,
        1.9082584 ,  1.91638345,  1.91896957,  1.9215943 ,  1.92398742,
        1.92420937,  1.9264867 ,  1.92903422,  1.93440911,  1.94273681,
        1.9428526 ,  1.94854593,  1.94861348,  1.95417171,  1.95972029,
        1.9677199 ,  1.97011303,  1.97024812,  1.97540107,  1.97798719,
        1.9810365 ,  1.98362262,  1.98389281,  1.98681667,  1.98961509,
        1.99518297,  2.0007798 ,  2.00094385,  2.00651173,  2.01473328,
        2.01480083,  2.02035906,  2.03406164,  2.04493687,  2.0504179 ,
        2.05577349,  2.07209115,  2.09383194,  2.09923577,  2.11281292,
        2.12378463,  2.13738107,  2.14829489,  2.16183343,  2.17272795,
        2.17809319,  2.18892981,  2.2025745 ,  2.20807484,  2.22157478,
        2.22669877,  2.22684352,  2.23196751,  2.23433169,  2.23449574,
        2.23961973,  2.24493672,  2.25042741,  2.25327407,  2.25338987,
        2.25626548,  2.26182371,  2.26193951,  2.26751704,  2.27278578,
        2.27287263,  2.27808347,  2.28065029,  2.28319781,  2.2885148 ,
        2.29419848,  2.29714164,  2.30002691,  2.30292182,  2.3085669 ,
        2.31140391,  2.31431812,  2.31449182,  2.32820406,  2.33360789,
        2.33894418,  2.3415303 ,  2.34682799,  2.34943341,  2.35177829,
        2.35196164,  2.35710493,  2.36538438,  2.36822139,  2.36837579,
        2.37394367,  2.37407877,  2.38239682,  2.39029028,  2.39045432,
        2.3928957 ,  2.39531778,  2.39546252,  2.40054792,  2.40895281,
        2.40904931,  2.41211792,  2.41771475,  2.42050352,  2.42056141,
        2.42070616,  2.42633194,  2.43185157,  2.44012137,  2.44292944,
        2.45385291,  2.45665132,  2.47026706,  2.47836317,  2.49467117,
        2.51105638,  2.51909458,  2.53537364,  2.5435759 ,  2.54636466,
        2.55726883,  2.56274022,  2.57635595,  2.58449066,  2.59809675,
        2.60619285,  2.61155809,  2.62244296,  2.62796259,  2.63335678,
        2.6441934 ,  2.64952004,  2.65208686,  2.65717226,  2.65738456,
        2.6623831 ,  2.66488238,  2.66747815,  2.67304603,  2.6758734 ,
        2.67602779,  2.68161497,  2.68444234,  2.68729865,  2.69284723,
        2.69293408,  2.69831862,  2.70093369,  2.70350051,  2.70581644,
        2.70600943,  2.7138643 ,  2.71949008,  2.72247184,  2.72264553,
        2.72830991,  2.73125307,  2.73417693,  2.7370622 ,  2.73983166,
        2.75361145,  2.75900563,  2.76430332,  2.76687015,  2.76928257,
        2.76938872,  2.77458026,  2.779849  ,  2.78241583,  2.78542654,
        2.78803196,  2.79375423,  2.79664915,  2.80223633,  2.80769806,
        2.81575557,  2.81835134,  2.82073482,  2.82086026,  2.82099536,
        2.82335954,  2.82884057,  2.83448565,  2.83740951,  2.84023688,
        2.84035268,  2.84318969,  2.8460653 ,  2.84896021,  2.85185513,
        2.85742301,  2.86298124,  2.87659698,  2.87941469,  2.89031886,
        2.89854042,  2.90934809,  2.92025226,  2.9339452 ,  2.94472392,
        2.96654191,  2.98289817,  2.99927372,  3.01290876,  3.03735147,
        3.04829424,  3.06466014,  3.0726501 ,  3.07786095,  3.08304284,
        3.08557106,  3.08813789,  3.09345487,  3.09911925,  3.10474503,
        3.10487048,  3.10781364,  3.11340082,  3.11348767,  3.12694901,
        3.12704551,  3.12952548,  3.13210196,  3.13473633,  3.1373707 ,
        3.13749614,  3.14315087,  3.14607473,  3.14621948,  3.14908544,
        3.14914334,  3.15474017,  3.15765438,  3.16053965,  3.16337666,
        3.17162716,  3.17998381,  3.19369604,  3.20467741,  3.21547544,
        3.23462046,  3.24272621,  3.26179404,  3.27815994,  3.28909306,
        3.29174673,  3.30531422,  3.30805474,  3.32164153,  3.33527657,
        3.35438299,  3.36251769,  3.37613343,  3.38425849,  3.3952688 ,
        3.40890384,  3.42254853,  3.43348165,  3.44437617,  3.45804015,
        3.47168484,  3.48261796,  3.49624335,  3.50985909,  3.51528222,
        3.52077291,  3.53442725,  3.5425716 ,  3.55618734,  3.56709151,
        3.57801498,  3.58348636,  3.59440018])

    fig12top_w_raw = np.array(
      [  4.18764000e-02,   4.18600000e-02,   1.56698000e-02,
         3.65997000e-02,   3.65710000e-02,  -9.45000000e-05,
         1.03684000e-02,   1.03520000e-02,   4.17449000e-02,
         2.07862000e-02,  -1.58670000e-02,  -4.20614000e-02,
        -2.11273000e-02,  -4.73216000e-02,  -2.63916000e-02,
        -5.78256000e-02,  -7.87721000e-02,  -7.87926000e-02,
        -9.45199000e-02,  -1.25942000e-01,  -1.25962000e-01,
        -9.45692000e-02,  -1.20764000e-01,  -1.41710000e-01,
        -1.57421000e-01,  -1.67921000e-01,  -1.78421000e-01,
        -1.78441000e-01,  -1.88945000e-01,  -1.62796000e-01,
        -1.88995000e-01,  -2.04730000e-01,  -2.36168000e-01,
        -2.15246000e-01,  -1.94333000e-01,  -2.30994000e-01,
        -2.51957000e-01,  -2.88619000e-01,  -2.67689000e-01,
        -2.93891000e-01,  -3.35784000e-01,  -3.04383000e-01,
        -3.41044000e-01,  -3.25350000e-01,  -3.41065000e-01,
        -3.25370000e-01,  -3.41089000e-01,  -3.88214000e-01,
        -4.45813000e-01,  -5.08644000e-01,  -5.81947000e-01,
        -6.65720000e-01,  -7.39022000e-01,  -7.96618000e-01,
        -7.49510000e-01,  -6.86691000e-01,  -6.34339000e-01,
        -5.81988000e-01,  -5.40107000e-01,  -4.98227000e-01,
        -5.19177000e-01,  -5.50595000e-01,  -5.87248000e-01,
        -6.34372000e-01,  -7.75737000e-01,  -7.02439000e-01,
        -8.33337000e-01,  -7.91464000e-01,  -6.50104000e-01,
        -7.23406000e-01,  -5.87281000e-01,  -5.24458000e-01,
        -3.98821000e-01,  -3.56940000e-01,  -3.30770000e-01,
        -2.88898000e-01,  -2.62728000e-01,  -2.31323000e-01,
        -1.99922000e-01,  -2.20885000e-01,  -2.62778000e-01,
        -2.52319000e-01,  -2.68046000e-01,  -2.94236000e-01,
        -3.04712000e-01,  -2.99492000e-01,  -3.20451000e-01,
        -3.25703000e-01,  -3.51889000e-01,  -3.30955000e-01,
        -3.20496000e-01,  -3.10038000e-01,  -3.30992000e-01,
        -3.46720000e-01,  -3.67674000e-01,  -3.78158000e-01,
        -3.99104000e-01,  -4.25311000e-01,  -4.04381000e-01,
        -4.51509000e-01,  -4.93402000e-01,  -6.03353000e-01,
        -5.40526000e-01,  -8.07549000e-01,  -7.13309000e-01,
        -9.17501000e-01,  -8.86087000e-01,  -8.96566000e-01,
        -8.65161000e-01,  -8.07574000e-01,  -7.49987000e-01,
        -7.39520000e-01,  -6.92399000e-01,  -6.55759000e-01,
        -6.19109000e-01,  -5.87704000e-01,  -6.55775000e-01,
        -6.13894000e-01,  -7.60495000e-01,  -6.97668000e-01,
        -8.33797000e-01,  -7.97148000e-01,  -8.12867000e-01,
        -7.97164000e-01,  -7.50048000e-01,  -7.29110000e-01,
        -6.71523000e-01,  -6.29638000e-01,  -5.98229000e-01,
        -5.40642000e-01,  -4.98761000e-01,  -4.62112000e-01,
        -4.51649000e-01,  -4.30707000e-01,  -4.15008000e-01,
        -3.94070000e-01,  -3.62665000e-01,  -3.73144000e-01,
        -4.04562000e-01,  -4.46450000e-01,  -4.98810000e-01,
        -5.66877000e-01,  -6.40179000e-01,  -7.34424000e-01,
        -7.92019000e-01,  -8.39144000e-01,  -7.60622000e-01,
        -7.34444000e-01,  -6.97799000e-01,  -6.55919000e-01,
        -6.08807000e-01,  -5.77393000e-01,  -5.14570000e-01,
        -5.30290000e-01,  -5.66943000e-01,  -6.61187000e-01,
        -6.03596000e-01,  -7.34489000e-01,  -7.92085000e-01,
        -8.33974000e-01,  -7.86862000e-01,  -6.40265000e-01,
        -7.34510000e-01,  -6.87390000e-01,  -5.98385000e-01,
        -5.46033000e-01,  -5.09397000e-01,  -4.57045000e-01,
        -4.09929000e-01,  -3.57577000e-01,  -3.20940000e-01,
        -3.05246000e-01,  -2.89547000e-01,  -2.63374000e-01,
        -2.37208000e-01,  -2.79109000e-01,  -2.52931000e-01,
        -3.05291000e-01,  -3.00076000e-01,  -3.00101000e-01,
        -3.26295000e-01,  -3.36783000e-01,  -3.62969000e-01,
        -3.57741000e-01,  -3.78704000e-01,  -3.68245000e-01,
        -3.78733000e-01,  -3.68270000e-01,  -3.94460000e-01,
        -4.04948000e-01,  -3.94493000e-01,  -4.25911000e-01,
        -4.57361000e-01,  -4.52142000e-01,  -4.73089000e-01,
        -4.99271000e-01,  -6.03986000e-01,  -5.46395000e-01,
        -6.72057000e-01,  -7.76773000e-01,  -7.19181000e-01,
        -9.02435000e-01,  -8.39608000e-01,  -9.28617000e-01,
        -9.02443000e-01,  -8.18678000e-01,  -8.55331000e-01,
        -7.97744000e-01,  -7.45392000e-01,  -7.08743000e-01,
        -6.72098000e-01,  -6.24990000e-01,  -6.40701000e-01,
        -6.66887000e-01,  -6.98305000e-01,  -8.08256000e-01,
        -7.45429000e-01,  -8.29203000e-01,  -8.65860000e-01,
        -8.29215000e-01,  -7.76863000e-01,  -7.50689000e-01,
        -6.98342000e-01,  -6.51222000e-01,  -6.25048000e-01,
        -5.72692000e-01,  -5.36051000e-01,  -4.83695000e-01,
        -4.26108000e-01,  -3.99947000e-01,  -3.68541000e-01,
        -4.10434000e-01,  -3.79021000e-01,  -4.78505000e-01,
        -5.41336000e-01,  -6.04167000e-01,  -7.50767000e-01,
        -6.72234000e-01,  -7.76949000e-01,  -8.34545000e-01,
        -8.08375000e-01,  -7.35077000e-01,  -6.93201000e-01,
        -6.51324000e-01,  -6.14679000e-01,  -5.62328000e-01,
        -5.51865000e-01,  -5.72815000e-01,  -6.51353000e-01,
        -6.09468000e-01,  -8.03193000e-01,  -7.19424000e-01,
        -8.60793000e-01,  -7.24672000e-01,  -7.82271000e-01,
        -6.51382000e-01,  -5.67617000e-01,  -4.99558000e-01,
        -4.89087000e-01,  -4.36744000e-01,  -3.94859000e-01,
        -3.63462000e-01,  -3.26817000e-01,  -2.95416000e-01,
        -2.79717000e-01,  -2.43068000e-01,  -2.16899000e-01,
        -2.43093000e-01,  -2.90225000e-01,  -2.64048000e-01,
        -3.26899000e-01,  -3.16440000e-01,  -3.05977000e-01,
        -3.16461000e-01,  -3.37428000e-01,  -3.26969000e-01,
        -3.37448000e-01,  -3.21754000e-01,  -3.11287000e-01,
        -3.27006000e-01,  -3.16543000e-01,  -3.37506000e-01,
        -3.58465000e-01,  -3.79415000e-01,  -4.00370000e-01,
        -4.31787000e-01,  -4.21333000e-01,  -4.16105000e-01,
        -4.26589000e-01,  -4.42304000e-01,  -4.68494000e-01,
        -5.15618000e-01,  -5.73214000e-01,  -7.19814000e-01,
        -6.30809000e-01,  -7.98352000e-01,  -8.61183000e-01,
        -9.13547000e-01,  -9.08324000e-01,  -8.55968000e-01,
        -8.03621000e-01,  -7.46033000e-01,  -6.62268000e-01,
        -6.20392000e-01,  -7.25120000e-01,  -6.57057000e-01,
        -7.87955000e-01,  -8.56022000e-01,  -9.18853000e-01,
        -8.56034000e-01,  -7.82740000e-01,  -6.98971000e-01,
        -6.46627000e-01,  -5.94275000e-01,  -5.31448000e-01,
        -4.84332000e-01,  -4.26745000e-01,  -4.16282000e-01,
        -3.84881000e-01,  -3.63939000e-01,  -3.79654000e-01,
        -4.00608000e-01,  -4.52968000e-01,  -4.89621000e-01,
        -6.10044000e-01,  -5.31510000e-01,  -6.93817000e-01,
        -7.61884000e-01,  -7.88070000e-01,  -7.35726000e-01,
        -6.93842000e-01,  -6.04845000e-01,  -5.78667000e-01,
        -5.42026000e-01,  -5.05386000e-01,  -5.73460000e-01,
        -6.93883000e-01,  -6.46763000e-01,  -7.56718000e-01,
        -8.09078000e-01,  -6.88664000e-01,  -7.41024000e-01,
        -6.41548000e-01,  -5.68254000e-01,  -5.42080000e-01,
        -5.00203000e-01,  -4.47856000e-01,  -3.90264000e-01,
        -3.48388000e-01,  -3.32694000e-01,  -3.06516000e-01,
        -2.69875000e-01,  -2.43718000e-01,  -2.54205000e-01,
        -2.43742000e-01,  -2.80399000e-01,  -2.96131000e-01,
        -3.22342000e-01,  -3.43292000e-01,  -3.64255000e-01,
        -3.38094000e-01,  -3.53821000e-01,  -3.48602000e-01,
        -3.85271000e-01,  -3.90523000e-01,  -4.21945000e-01,
        -4.48139000e-01,  -4.42924000e-01,  -4.27226000e-01,
        -4.74367000e-01,  -5.94793000e-01,  -5.42437000e-01,
        -6.62864000e-01,  -7.98993000e-01,  -7.36166000e-01,
        -8.56593000e-01,  -9.08957000e-01,  -8.93258000e-01,
        -8.51378000e-01,  -8.09493000e-01,  -7.51906000e-01,
        -7.15265000e-01,  -6.73380000e-01,  -6.26268000e-01,
        -6.94339000e-01,  -6.62926000e-01,  -7.51939000e-01,
        -8.09534000e-01,  -8.77601000e-01,  -9.24729000e-01,
        -8.40968000e-01,  -7.62439000e-01,  -7.04851000e-01,
        -6.42028000e-01,  -5.73974000e-01,  -5.32094000e-01,
        -4.64035000e-01,  -4.01208000e-01,  -3.69815000e-01,
        -3.90766000e-01,  -4.32659000e-01,  -4.85019000e-01,
        -5.42618000e-01,  -5.84507000e-01,  -7.25872000e-01,
        -6.57809000e-01,  -7.73000000e-01,  -7.36363000e-01,
        -6.94483000e-01,  -6.36892000e-01,  -5.95015000e-01,
        -5.47895000e-01,  -4.95551000e-01,  -6.00275000e-01,
        -5.42684000e-01,  -6.47400000e-01,  -7.57351000e-01,
        -7.04995000e-01,  -8.41128000e-01,  -7.57372000e-01,
        -7.20722000e-01,  -5.95073000e-01,  -5.42725000e-01,
        -5.21787000e-01,  -5.00845000e-01,  -4.48489000e-01,
        -3.85670000e-01,  -3.59500000e-01,  -3.28099000e-01,
        -2.96690000e-01,  -2.91471000e-01,  -2.65297000e-01,
        -2.70553000e-01,  -3.01979000e-01,  -3.22946000e-01,
        -3.12499000e-01,  -3.64867000e-01,  -3.96306000e-01,
        -3.85847000e-01,  -3.64908000e-01,  -3.64925000e-01,
        -3.59698000e-01,  -3.64954000e-01,  -3.80673000e-01,
        -3.91164000e-01,  -4.22590000e-01,  -4.54012000e-01,
        -4.64500000e-01,  -4.38330000e-01,  -4.59280000e-01,
        -4.85475000e-01,  -5.32603000e-01,  -5.90199000e-01,
        -7.26332000e-01,  -6.47798000e-01,  -8.15345000e-01,
        -8.99118000e-01,  -9.46243000e-01,  -9.04366000e-01,
        -8.67721000e-01,  -8.10130000e-01,  -7.63018000e-01,
        -7.26373000e-01,  -6.79257000e-01,  -6.42616000e-01,
        -6.11203000e-01,  -6.37389000e-01,  -6.79278000e-01,
        -7.36873000e-01,  -8.88709000e-01,  -8.15411000e-01,
        -9.35842000e-01,  -8.73023000e-01,  -7.78786000e-01,
        -7.15959000e-01,  -6.37434000e-01,  -5.58904000e-01,
        -4.85610000e-01,  -4.28023000e-01,  -4.12320000e-01,
        -3.59985000e-01,  -3.80936000e-01,  -4.38535000e-01,
        -4.96131000e-01,  -6.11318000e-01,  -5.74669000e-01,
        -6.68917000e-01,  -7.36988000e-01,  -7.94584000e-01,
        -6.89876000e-01,  -7.37000000e-01,  -6.37533000e-01,
        -5.74710000e-01,  -5.27598000e-01,  -5.22370000e-01,
        -5.69503000e-01,  -6.16627000e-01,  -7.42285000e-01,
        -6.95165000e-01,  -6.48045000e-01,  -7.78939000e-01,
        -7.68476000e-01,  -7.00421000e-01,  -6.27127000e-01,
        -5.90482000e-01,  -5.48598000e-01,  -5.06717000e-01,
        -4.54365000e-01,  -3.91543000e-01,  -3.28720000e-01,
        -2.86843000e-01,  -2.50202000e-01,  -2.55459000e-01,
        -2.24049000e-01,  -2.24066000e-01,  -2.08371000e-01,
        -2.45037000e-01,  -2.45053000e-01,  -2.24131000e-01,
        -2.71268000e-01,  -2.71301000e-01,  -2.71326000e-01,
        -2.66115000e-01,  -2.66135000e-01,  -3.02821000e-01,
        -2.87131000e-01,  -2.87156000e-01,  -3.55230000e-01,
        -4.44244000e-01,  -5.43728000e-01,  -6.17030000e-01,
        -6.74626000e-01,  -7.26990000e-01,  -6.48464000e-01,
        -5.85645000e-01,  -5.38525000e-01,  -4.59995000e-01,
        -4.12883000e-01,  -3.81470000e-01,  -4.44318000e-01,
        -4.07668000e-01,  -4.96677000e-01,  -5.54273000e-01,
        -5.85691000e-01,  -6.22344000e-01,  -5.75223000e-01,
        -5.01934000e-01,  -4.28640000e-01,  -3.76284000e-01,
        -3.23932000e-01,  -3.02990000e-01,  -2.50642000e-01,
        -1.82584000e-01,  -1.24996000e-01,  -8.31159000e-02,
        -5.69503000e-02,   5.86434000e-03,   3.72573000e-02,
         6.34187000e-02,   2.15177000e-02,   4.24312000e-02,
         1.62410000e-02,   1.09766000e-02,   1.09520000e-02,
         2.14067000e-02,  -4.77531000e-03,  -3.09738000e-02,
        -2.57423000e-02,  -4.14696000e-02,  -4.14901000e-02,
        -3.62833000e-02,  -5.20024000e-02,  -5.72585000e-02,
        -7.82132000e-02,  -4.15805000e-02,  -4.16011000e-02,
        -3.63861000e-02,  -2.59313000e-02,  -3.11833000e-02,
        -2.07327000e-02,  -1.55177000e-02,  -5.06298000e-03,
        -1.03191000e-02,  -1.55752000e-02,  -2.60546000e-02,
        -1.55917000e-02,  -5.14106000e-03,  -2.08601000e-02,
        -2.61163000e-02,  -2.61327000e-02,  -2.09136000e-02,
        -1.56862000e-02,  -1.04671000e-02])



    fig13top_t_raw = np.array(
      [ 0.179243,  0.183404,  0.195889,  0.208374,  0.214617,  0.22294 ,
        0.229183,  0.239587,  0.24791 ,  0.258314,  0.264557,  0.27288 ,
        0.279123,  0.289527,  0.29785 ,  0.306174,  0.312416,  0.316578,
        0.333225,  0.335305,  0.351952,  0.37068 ,  0.374841,  0.376922,
        0.385246,  0.387326,  0.393569,  0.399811,  0.399811,  0.403973,
        0.408135,  0.416458,  0.418539,  0.422701,  0.426862,  0.433105,
        0.439347,  0.443509,  0.443509,  0.451832,  0.453913,  0.455994,
        0.462237,  0.464317,  0.466398,  0.468479,  0.474722,  0.476802,
        0.478883,  0.483045,  0.487207,  0.491368,  0.493449,  0.49553 ,
        0.505934,  0.510096,  0.514257,  0.5205  ,  0.522581,  0.526743,
        0.532985,  0.541308,  0.549632,  0.557955,  0.57044 ,  0.574602,
        0.587087,  0.591249,  0.593329,  0.603734,  0.612057,  0.614138,
        0.62038 ,  0.628704,  0.637027,  0.64535 ,  0.647431,  0.653674,
        0.659916,  0.664078,  0.674482,  0.678644,  0.686967,  0.689048,
        0.69321 ,  0.697371,  0.699452,  0.701533,  0.705695,  0.709856,
        0.711937,  0.716099,  0.71818 ,  0.726503,  0.732746,  0.734826,
        0.738988,  0.74315 ,  0.749392,  0.751473,  0.753554,  0.755635,
        0.759796,  0.761877,  0.766039,  0.770201,  0.772281,  0.774362,
        0.776443,  0.780605,  0.784766,  0.786847,  0.791009,  0.795171,
        0.801413,  0.809737,  0.813898,  0.815979,  0.820141,  0.822222,
        0.826383,  0.828464,  0.830545,  0.834707,  0.840949,  0.84303 ,
        0.84303 ,  0.845111,  0.859677,  0.863838,  0.865919,  0.870081,
        0.872162,  0.878404,  0.884647,  0.886728,  0.888808,  0.890889,
        0.895051,  0.897132,  0.901293,  0.903374,  0.905455,  0.91794 ,
        0.920021,  0.928344,  0.94291 ,  0.947072,  0.955395,  0.957476,
        0.963719,  0.965799,  0.974123,  0.980365,  0.984527,  0.990769,
        0.994931,  0.997012,  1.00534 ,  1.0095  ,  1.02198 ,  1.02614 ,
        1.03447 ,  1.04071 ,  1.04695 ,  1.04903 ,  1.05528 ,  1.0636  ,
        1.06776 ,  1.07192 ,  1.07816 ,  1.08441 ,  1.09065 ,  1.09897 ,
        1.11146 ,  1.11354 ,  1.11562 ,  1.11978 ,  1.12394 ,  1.12602 ,
        1.1281  ,  1.13019 ,  1.13227 ,  1.13851 ,  1.14059 ,  1.14475 ,
        1.15099 ,  1.15307 ,  1.15724 ,  1.15724 ,  1.16764 ,  1.17388 ,
        1.17596 ,  1.18013 ,  1.18221 ,  1.18637 ,  1.18845 ,  1.19469 ,
        1.19677 ,  1.19677 ,  1.20093 ,  1.20301 ,  1.21134 ,  1.21342 ,
        1.21966 ,  1.2259  ,  1.23215 ,  1.23423 ,  1.23631 ,  1.23839 ,
        1.24255 ,  1.24463 ,  1.24671 ,  1.24879 ,  1.25296 ,  1.2592  ,
        1.26336 ,  1.26544 ,  1.26752 ,  1.2696  ,  1.27376 ,  1.27793 ,
        1.28209 ,  1.28625 ,  1.28833 ,  1.29041 ,  1.29249 ,  1.29665 ,
        1.29873 ,  1.30081 ,  1.30914 ,  1.31122 ,  1.31122 ,  1.3133  ,
        1.31746 ,  1.32162 ,  1.3237  ,  1.32787 ,  1.34035 ,  1.34243 ,
        1.34867 ,  1.35284 ,  1.357   ,  1.3674  ,  1.37781 ,  1.39237 ,
        1.40278 ,  1.4111  ,  1.4215  ,  1.42358 ,  1.43399 ,  1.44439 ,
        1.4548  ,  1.46104 ,  1.47144 ,  1.47352 ,  1.48393 ,  1.49225 ,
        1.50057 ,  1.50474 ,  1.50682 ,  1.51722 ,  1.52554 ,  1.53595 ,
        1.54011 ,  1.54219 ,  1.54635 ,  1.54843 ,  1.5526  ,  1.55676 ,
        1.563   ,  1.56716 ,  1.5734  ,  1.57549 ,  1.57549 ,  1.57965 ,
        1.58173 ,  1.58589 ,  1.59005 ,  1.59629 ,  1.59837 ,  1.60046 ,
        1.60462 ,  1.6067  ,  1.61294 ,  1.61918 ,  1.62126 ,  1.62334 ,
        1.62959 ,  1.63167 ,  1.63375 ,  1.63791 ,  1.63999 ,  1.65248 ,
        1.65456 ,  1.65664 ,  1.6608  ,  1.66288 ,  1.66496 ,  1.66912 ,
        1.6712  ,  1.67328 ,  1.67537 ,  1.67953 ,  1.68577 ,  1.68785 ,
        1.69201 ,  1.69409 ,  1.69617 ,  1.70034 ,  1.7045  ,  1.70866 ,
        1.71282 ,  1.7149  ,  1.71698 ,  1.71906 ,  1.72322 ,  1.73363 ,
        1.73363 ,  1.73987 ,  1.74195 ,  1.74403 ,  1.74611 ,  1.75236 ,
        1.75444 ,  1.76068 ,  1.76692 ,  1.77316 ,  1.78565 ,  1.79397 ,
        1.80022 ,  1.80854 ,  1.81686 ,  1.82727 ,  1.83767 ,  1.84183 ,
        1.84599 ,  1.85848 ,  1.86472 ,  1.87721 ,  1.88137 ,  1.88761 ,
        1.89801 ,  1.90634 ,  1.91674 ,  1.92299 ,  1.92923 ,  1.93755 ,
        1.94796 ,  1.96044 ,  1.96252 ,  1.9646  ,  1.96876 ,  1.97084 ,
        1.97501 ,  1.97917 ,  1.98125 ,  1.98541 ,  1.98749 ,  1.99165 ,
        1.9979  ,  1.99998 ,  2.00206 ,  2.00622 ,  2.01454 ,  2.01662 ,
        2.0187  ,  2.02078 ,  2.02495 ,  2.02703 ,  2.02911 ,  2.03535 ,
        2.03743 ,  2.04159 ,  2.04367 ,  2.04575 ,  2.04784 ,  2.05408 ,
        2.05824 ,  2.06032 ,  2.0624  ,  2.06656 ,  2.07697 ,  2.07905 ,
        2.08321 ,  2.08529 ,  2.08737 ,  2.09153 ,  2.09361 ,  2.09569 ,
        2.09778 ,  2.10194 ,  2.1061  ,  2.11026 ,  2.11026 ,  2.11234 ,
        2.11442 ,  2.12066 ,  2.12483 ,  2.12899 ,  2.13315 ,  2.13939 ,
        2.14147 ,  2.14355 ,  2.1498  ,  2.15604 ,  2.15812 ,  2.16436 ,
        2.16852 ,  2.17269 ,  2.17477 ,  2.17685 ,  2.18517 ,  2.19141 ,
        2.19557 ,  2.20182 ,  2.2039  ,  2.21014 ,  2.2143  ,  2.21846 ,
        2.22263 ,  2.22679 ,  2.23511 ,  2.23719 ,  2.24343 ,  2.25384 ,
        2.26008 ,  2.26632 ,  2.27673 ,  2.28713 ,  2.29337 ,  2.3017  ,
        2.3121  ,  2.32251 ,  2.33083 ,  2.33915 ,  2.36204 ,  2.3662  ,
        2.37661 ,  2.38493 ,  2.39117 ,  2.39534 ,  2.39742 ,  2.3995  ,
        2.40574 ,  2.40782 ,  2.4099  ,  2.41198 ,  2.41614 ,  2.42239 ,
        2.42447 ,  2.42863 ,  2.43071 ,  2.43903 ,  2.44319 ,  2.44528 ,
        2.44944 ,  2.4536  ,  2.45568 ,  2.45984 ,  2.464   ,  2.46608 ,
        2.46816 ,  2.47233 ,  2.47649 ,  2.47857 ,  2.48273 ,  2.48481 ,
        2.48897 ,  2.50562 ,  2.50978 ,  2.51394 ,  2.5181  ,  2.52019 ,
        2.52227 ,  2.52435 ,  2.52851 ,  2.53267 ,  2.53683 ,  2.53891 ,
        2.54099 ,  2.54516 ,  2.54932 ,  2.55556 ,  2.55764 ,  2.55972 ,
        2.56388 ,  2.56804 ,  2.57013 ,  2.57429 ,  2.57637 ,  2.58053 ,
        2.58261 ,  2.58677 ,  2.59093 ,  2.5951  ,  2.59718 ,  2.59926 ,
        2.60134 ,  2.60342 ,  2.61382 ,  2.6159  ,  2.62007 ,  2.63671 ,
        2.64712 ,  2.65752 ,  2.66584 ,  2.67209 ,  2.68249 ,  2.68873 ,
        2.69706 ,  2.71578 ,  2.72619 ,  2.73243 ,  2.73867 ,  2.75532 ,
        2.76781 ,  2.77197 ,  2.78237 ,  2.79069 ,  2.79902 ,  2.80526 ,
        2.80942 ,  2.81566 ,  2.81775 ,  2.81983 ,  2.82399 ,  2.82607 ,
        2.82815 ,  2.83023 ,  2.83439 ,  2.83647 ,  2.83855 ,  2.8448  ,
        2.84688 ,  2.85104 ,  2.8552  ,  2.86144 ,  2.86769 ,  2.86977 ,
        2.87393 ,  2.87601 ,  2.87809 ,  2.88225 ,  2.88849 ,  2.89266 ,
        2.89474 ,  2.90098 ,  2.90306 ,  2.90514 ,  2.90722 ,  2.91138 ,
        2.91346 ,  2.92387 ,  2.93219 ,  2.93843 ,  2.95092 ,  2.95716 ,
        2.97173 ,  2.98005 ,  2.98837 ,  2.99254 ,  2.99878 ,  3.00086 ,
        3.00294 ,  3.0071  ,  3.01126 ,  3.01334 ,  3.01751 ,  3.02791 ,
        3.03623 ,  3.04456 ,  3.05288 ,  3.06953 ,  3.07161 ,  3.08825 ,
        3.10074 ,  3.10282 ,  3.11947 ,  3.12987 ,  3.14236 ,  3.15484 ,
        3.159   ,  3.16941 ,  3.17981 ,  3.19022 ,  3.19646 ,  3.20686 ,
        3.22143 ,  3.23391 ,  3.24432 ,  3.25888 ,  3.26513 ,  3.27553 ,
        3.28177 ,  3.29634 ,  3.30674 ,  3.31715 ,  3.32547 ,  3.33171 ,
        3.3442  ,  3.3546  ,  3.36293 ,  3.36917 ,  3.37749 ,  3.38581 ,
        3.38998 ,  3.3983  ,  3.40662 ,  3.41703 ,  3.42327 ,  3.43784 ,
        3.44408 ,  3.45448 ,  3.46489 ,  3.47737 ,  3.48569 ,  3.49194 ,
        3.51066 ,  3.52107 ,  3.52731 ,  3.54396 ,  3.54604 ,  3.55436 ,
        3.56477 ,  3.57309 ,  3.58141 ,  3.58557 ,  3.59598 ,  3.6043  ,
        3.61263 ,  3.62095 ,  3.63343 ,  3.64176 ])
    fig13top_w_raw = np.array(
      [ 0.        , -0.0198413 ,  0.        , -0.015873  ,  0.        ,
       -0.015873  ,  0.015873  , -0.0198413 , -0.00396825, -0.0198413 ,
        0.015873  , -0.0238095 ,  0.        , -0.0277778 , -0.00396825,
       -0.031746  , -0.0119048 , -0.031746  , -0.0277778 , -0.00793651,
       -0.00793651,  0.0277778 , -0.0119048 , -0.0634921 , -0.0992063 ,
       -0.138889  , -0.230159  , -0.484127  , -0.484127  , -0.603175  ,
       -0.702381  , -0.531746  , -0.611111  , -0.488095  , -0.452381  ,
       -0.361111  , -0.404762  , -0.599206  , -0.599206  , -0.714286  ,
       -0.753968  , -0.789683  , -0.72619   , -0.603175  , -0.539683  ,
       -0.47619   , -0.365079  , -0.31746   , -0.234127  , -0.18254   ,
       -0.150794  , -0.115079  , -0.0793651 , -0.047619  , -0.0753968 ,
       -0.00396825,  0.015873  , -0.015873  , -0.0674603 , -0.015873  ,
        0.0198413 , -0.0714286 ,  0.0119048 , -0.0436508 , -0.0595238 ,
        0.        , -0.0396825 , -0.031746  , -0.00793651, -0.0119048 ,
       -0.0357143 , -0.015873  , -0.0436508 , -0.015873  , -0.0396825 ,
       -0.0277778 ,  0.00396825, -0.0277778 , -0.0119048 ,  0.00396825,
        0.0238095 ,  0.0119048 , -0.0119048 , -0.0357143 , -0.0793651 ,
       -0.138889  , -0.214286  , -0.281746  , -0.357143  , -0.440476  ,
       -0.535714  , -0.722222  , -0.809524  , -0.710317  , -0.543651  ,
       -0.496032  , -0.460317  , -0.416667  , -0.464286  , -0.527778  ,
       -0.571429  , -0.619048  , -0.690476  , -0.777778  , -0.837302  ,
       -0.765873  , -0.714286  , -0.555556  , -0.412698  , -0.349206  ,
       -0.337302  , -0.269841  , -0.150794  , -0.0634921 , -0.123016  ,
       -0.138889  , -0.257937  , -0.345238  , -0.420635  , -0.484127  ,
       -0.619048  , -0.694444  , -0.77381   , -0.869048  , -0.634921  ,
       -0.730159  , -0.730159  , -0.496032  , -0.384921  , -0.5       ,
       -0.603175  , -0.706349  , -0.765873  , -0.84127   , -0.785714  ,
       -0.710317  , -0.615079  , -0.452381  , -0.420635  , -0.357143  ,
       -0.22619   , -0.142857  , -0.0873016 , -0.126984  , -0.0396825 ,
       -0.0238095 , -0.0515873 , -0.00396825, -0.0238095 , -0.0515873 ,
       -0.0357143 ,  0.00396825, -0.0357143 , -0.0436508 ,  0.00396825,
       -0.00793651, -0.0396825 ,  0.        , -0.0119048 , -0.0357143 ,
       -0.031746  ,  0.00396825, -0.0357143 , -0.00793651, -0.0119048 ,
       -0.0357143 , -0.0357143 , -0.0396825 , -0.0277778 , -0.00793651,
       -0.031746  ,  0.        ,  0.0198413 , -0.00396825, -0.015873  ,
       -0.0714286 , -0.119048  , -0.15873   , -0.190476  , -0.285714  ,
       -0.404762  , -0.519841  , -0.587302  , -0.638889  , -0.686508  ,
       -0.785714  , -0.638889  , -0.714286  , -0.448413  , -0.448413  ,
       -0.412698  , -0.496032  , -0.563492  , -0.626984  , -0.722222  ,
       -0.805556  , -0.888889  , -0.84127   , -0.5       , -0.5       ,
       -0.404762  , -0.373016  , -0.253968  , -0.138889  , -0.0952381 ,
       -0.123016  , -0.150794  , -0.257937  , -0.202381  , -0.285714  ,
       -0.337302  , -0.448413  , -0.539683  , -0.599206  , -0.742063  ,
       -0.865079  , -0.75      , -0.68254   , -0.603175  , -0.571429  ,
       -0.555556  , -0.436508  , -0.400794  , -0.464286  , -0.539683  ,
       -0.619048  , -0.686508  , -0.765873  , -0.825397  , -0.876984  ,
       -0.718254  , -0.571429  , -0.571429  , -0.492063  , -0.444444  ,
       -0.392857  , -0.309524  , -0.162698  , -0.142857  , -0.047619  ,
       -0.0277778 , -0.0634921 , -0.031746  , -0.0515873 , -0.00396825,
       -0.0357143 , -0.0119048 , -0.0396825 , -0.00793651, -0.0277778 ,
       -0.0515873 , -0.015873  , -0.0436508 , -0.015873  , -0.0238095 ,
        0.00793651, -0.031746  , -0.015873  , -0.0396825 , -0.015873  ,
        0.00396825,  0.015873  , -0.00793651, -0.015873  , -0.0753968 ,
       -0.115079  , -0.222222  , -0.301587  , -0.416667  , -0.507937  ,
       -0.694444  , -0.849206  , -0.75      , -0.599206  , -0.599206  ,
       -0.52381   , -0.5       , -0.464286  , -0.424603  , -0.464286  ,
       -0.511905  , -0.646825  , -0.72619   , -0.81746   , -0.912698  ,
       -0.718254  , -0.583333  , -0.642857  , -0.444444  , -0.376984  ,
       -0.293651  , -0.257937  , -0.162698  , -0.138889  , -0.166667  ,
       -0.202381  , -0.257937  , -0.309524  , -0.369048  , -0.464286  ,
       -0.559524  , -0.674603  , -0.75      , -0.813492  , -0.75      ,
       -0.698413  , -0.575397  , -0.535714  , -0.492063  , -0.448413  ,
       -0.396825  , -0.452381  , -0.503968  , -0.587302  , -0.65873   ,
       -0.746032  , -0.833333  , -0.547619  , -0.547619  , -0.468254  ,
       -0.404762  , -0.361111  , -0.305556  , -0.202381  , -0.146825  ,
       -0.115079  , -0.0833333 , -0.047619  , -0.0396825 , -0.047619  ,
       -0.0198413 , -0.0396825 , -0.015873  , -0.0515873 , -0.0277778 ,
       -0.0396825 , -0.0238095 , -0.0436508 , -0.0198413 , -0.0595238 ,
       -0.015873  , -0.0396825 , -0.00396825, -0.047619  , -0.0238095 ,
       -0.0436508 , -0.015873  ,  0.0119048 , -0.00793651, -0.015873  ,
       -0.0555556 , -0.103175  , -0.142857  , -0.242063  , -0.392857  ,
       -0.496032  , -0.623016  , -0.710317  , -0.809524  , -0.865079  ,
       -0.813492  , -0.690476  , -0.599206  , -0.484127  , -0.464286  ,
       -0.444444  , -0.492063  , -0.539683  , -0.626984  , -0.714286  ,
       -0.797619  , -0.865079  , -0.896825  , -0.809524  , -0.714286  ,
       -0.626984  , -0.543651  , -0.380952  , -0.297619  , -0.222222  ,
       -0.142857  , -0.0952381 , -0.142857  , -0.170635  , -0.202381  ,
       -0.289683  , -0.345238  , -0.412698  , -0.496032  , -0.563492  ,
       -0.650794  , -0.702381  , -0.813492  , -0.753968  , -0.753968  ,
       -0.555556  , -0.535714  , -0.488095  , -0.392857  , -0.424603  ,
       -0.543651  , -0.626984  , -0.698413  , -0.769841  , -0.829365  ,
       -0.77381   , -0.56746   , -0.456349  , -0.31746   , -0.25      ,
       -0.186508  , -0.103175  , -0.138889  , -0.0357143 , -0.0119048 ,
       -0.0714286 , -0.015873  ,  0.0198413 , -0.0238095 , -0.0674603 ,
       -0.031746  ,  0.00396825, -0.031746  , -0.0674603 ,  0.0238095 ,
       -0.0634921 , -0.00793651, -0.0634921 , -0.00793651, -0.0753968 ,
       -0.00793651, -0.0555556 , -0.00793651, -0.0555556 , -0.0277778 ,
       -0.0396825 , -0.00793651, -0.0238095 ,  0.015873  , -0.0436508 ,
       -0.150794  , -0.31746   , -0.238095  , -0.400794  , -0.527778  ,
       -0.670635  , -0.72619   , -0.801587  , -0.84127   , -0.730159  ,
       -0.59127   , -0.543651  , -0.480159  , -0.420635  , -0.484127  ,
       -0.543651  , -0.611111  , -0.678571  , -0.765873  , -0.849206  ,
       -0.714286  , -0.785714  , -0.587302  , -0.484127  , -0.40873   ,
       -0.34127   , -0.281746  , -0.218254  , -0.111111  , -0.198413  ,
       -0.289683  , -0.376984  , -0.452381  , -0.543651  , -0.642857  ,
       -0.718254  , -0.793651  , -0.587302  , -0.68254   , -0.531746  ,
       -0.484127  , -0.448413  , -0.40873   , -0.436508  , -0.519841  ,
       -0.563492  , -0.65873   , -0.722222  , -0.781746  , -0.853175  ,
       -0.65873   , -0.615079  , -0.507937  , -0.43254   , -0.388889  ,
       -0.325397  , -0.253968  , -0.198413  , -0.150794  , -0.0833333 ,
       -0.126984  , -0.0634921 , -0.0198413 , -0.0396825 ,  0.        ,
       -0.0357143 , -0.0119048 , -0.0357143 ,  0.        , -0.047619  ,
       -0.0119048 , -0.0436508 , -0.0198413 , -0.0515873 , -0.00396825,
       -0.0357143 , -0.0198413 , -0.0357143 ,  0.0198413 ,  0.        ,
        0.00396825, -0.015873  , -0.0634921 , -0.146825  , -0.210317  ,
       -0.27381   , -0.464286  , -0.34127   , -0.551587  , -0.646825  ,
       -0.72619   , -0.781746  , -0.825397  , -0.718254  , -0.626984  ,
       -0.535714  , -0.480159  , -0.444444  , -0.488095  , -0.559524  ,
       -0.623016  , -0.710317  , -0.793651  , -0.845238  , -0.793651  ,
       -0.611111  , -0.702381  , -0.436508  , -0.376984  , -0.313492  ,
       -0.246032  , -0.166667  , -0.0873016 , -0.103175  , -0.047619  ,
        0.        , -0.0674603 , -0.00793651, -0.047619  ,  0.015873  ,
       -0.103175  ,  0.0396825 ,  0.186508  ,  0.0833333 ,  0.031746  ,
       -0.142857  , -0.0277778 ,  0.0198413 , -0.0357143 , -0.0634921 ,
       -0.0436508 , -0.0238095 , -0.00396825, -0.0238095 , -0.00396825,
       -0.0198413 , -0.00793651, -0.0277778 , -0.0238095 , -0.031746  ,
        0.00396825, -0.0198413 ,  0.        , -0.0198413 , -0.0119048 ,
       -0.00396825, -0.0238095 ,  0.00396825, -0.00396825, -0.0277778 ,
       -0.0119048 , -0.0277778 ,  0.00396825, -0.0277778 , -0.015873  ,
       -0.0357143 , -0.0198413 , -0.00793651, -0.0277778 , -0.00396825,
       -0.0277778 , -0.00396825, -0.015873  , -0.00396825, -0.0119048 ,
       -0.00396825, -0.0277778 , -0.00793651, -0.0238095 , -0.00396825,
       -0.031746  , -0.0238095 , -0.031746  , -0.0198413 , -0.0277778 ,
       -0.0198413 , -0.0277778 , -0.015873  , -0.00396825, -0.0238095 ,
       -0.015873  , -0.015873  , -0.00793651, -0.015873  , -0.015873  ,
       -0.0119048 , -0.0198413 , -0.0119048 , -0.015873  , -0.0119048 ,
       -0.0277778 , -0.0198413 , -0.0238095 , -0.0277778 ])

    fig13bot_t_raw = np.array(
      [ 0.14382 ,  0.14955 ,  0.15719 ,  0.16674 ,  0.17247 ,  0.1973  ,
        0.21067 ,  0.22022 ,  0.229769,  0.241229,  0.254599,  0.266059,
        0.273699,  0.283249,  0.290889,  0.298529,  0.306169,  0.315719,
        0.321449,  0.336729,  0.348189,  0.355829,  0.357739,  0.359649,
        0.363469,  0.365379,  0.367289,  0.371109,  0.378749,  0.384479,
        0.388299,  0.390209,  0.394029,  0.399759,  0.407398,  0.411218,
        0.416948,  0.424588,  0.426498,  0.428408,  0.434138,  0.436048,
        0.439868,  0.443688,  0.445598,  0.449418,  0.455148,  0.457058,
        0.460878,  0.462788,  0.472338,  0.479978,  0.487618,  0.497168,
        0.504808,  0.514358,  0.521998,  0.533458,  0.539188,  0.556378,
        0.564018,  0.571658,  0.579297,  0.590757,  0.598397,  0.606037,
        0.617497,  0.619407,  0.634687,  0.644237,  0.649967,  0.657607,
        0.667157,  0.669067,  0.674797,  0.676707,  0.680527,  0.682437,
        0.688167,  0.697717,  0.703447,  0.705357,  0.711087,  0.716817,
        0.718727,  0.720637,  0.724457,  0.728277,  0.737827,  0.741647,
        0.745467,  0.747377,  0.749286,  0.756926,  0.760746,  0.762656,
        0.768386,  0.772206,  0.776026,  0.779846,  0.783666,  0.789396,
        0.793216,  0.795126,  0.798946,  0.802766,  0.808496,  0.812316,
        0.816136,  0.821866,  0.829506,  0.833326,  0.837146,  0.840966,
        0.844786,  0.850516,  0.858156,  0.861976,  0.861976,  0.863886,
        0.867706,  0.875346,  0.879166,  0.882986,  0.886806,  0.894446,
        0.903996,  0.911636,  0.921186,  0.928825,  0.938375,  0.946015,
        0.961295,  0.976575,  0.980395,  0.989945,  0.999495,  1.0205  ,
        1.03196 ,  1.03578 ,  1.04342 ,  1.04915 ,  1.06061 ,  1.06825 ,
        1.07971 ,  1.08353 ,  1.08926 ,  1.09499 ,  1.0969  ,  1.10072 ,
        1.10263 ,  1.10645 ,  1.11409 ,  1.11982 ,  1.12173 ,  1.12364 ,
        1.12746 ,  1.12937 ,  1.13701 ,  1.14274 ,  1.14656 ,  1.15038 ,
        1.15229 ,  1.1542  ,  1.15993 ,  1.16375 ,  1.16757 ,  1.17139 ,
        1.17521 ,  1.18094 ,  1.18285 ,  1.18667 ,  1.18858 ,  1.19049 ,
        1.20386 ,  1.20768 ,  1.20959 ,  1.21341 ,  1.21723 ,  1.21914 ,
        1.21914 ,  1.22678 ,  1.2306  ,  1.23251 ,  1.23442 ,  1.23824 ,
        1.24015 ,  1.24397 ,  1.24779 ,  1.25543 ,  1.25734 ,  1.26116 ,
        1.26307 ,  1.2688  ,  1.27262 ,  1.27453 ,  1.28026 ,  1.28217 ,
        1.28599 ,  1.28981 ,  1.29554 ,  1.29745 ,  1.30509 ,  1.31082 ,
        1.31655 ,  1.32419 ,  1.32992 ,  1.33947 ,  1.34902 ,  1.35475 ,
        1.36048 ,  1.37194 ,  1.37767 ,  1.38149 ,  1.38722 ,  1.39868 ,
        1.40632 ,  1.41396 ,  1.41969 ,  1.42924 ,  1.44261 ,  1.45216 ,
        1.46171 ,  1.46935 ,  1.47699 ,  1.48463 ,  1.49991 ,  1.50373 ,
        1.50946 ,  1.51519 ,  1.5171  ,  1.52092 ,  1.52474 ,  1.52856 ,
        1.53047 ,  1.53238 ,  1.53429 ,  1.54002 ,  1.54575 ,  1.54766 ,
        1.55148 ,  1.56103 ,  1.56103 ,  1.56867 ,  1.57058 ,  1.57631 ,
        1.58204 ,  1.58586 ,  1.59159 ,  1.59541 ,  1.59732 ,  1.59923 ,
        1.60878 ,  1.61069 ,  1.61451 ,  1.62024 ,  1.62215 ,  1.62597 ,
        1.6317  ,  1.63361 ,  1.63743 ,  1.64316 ,  1.64507 ,  1.64889 ,
        1.65462 ,  1.66035 ,  1.66417 ,  1.66608 ,  1.6699  ,  1.67563 ,
        1.67754 ,  1.68327 ,  1.689   ,  1.69091 ,  1.69282 ,  1.69855 ,
        1.70237 ,  1.70428 ,  1.70619 ,  1.71574 ,  1.71956 ,  1.71956 ,
        1.72147 ,  1.73102 ,  1.73484 ,  1.74057 ,  1.75012 ,  1.75203 ,
        1.75394 ,  1.75776 ,  1.76158 ,  1.7654  ,  1.76922 ,  1.77113 ,
        1.77304 ,  1.77877 ,  1.7845  ,  1.78641 ,  1.79023 ,  1.79787 ,
        1.8036  ,  1.80742 ,  1.81124 ,  1.81697 ,  1.82461 ,  1.83607 ,
        1.84371 ,  1.85708 ,  1.87618 ,  1.88191 ,  1.89528 ,  1.90292 ,
        1.9182  ,  1.92393 ,  1.93157 ,  1.9373  ,  1.93921 ,  1.94112 ,
        1.94303 ,  1.94876 ,  1.95067 ,  1.95258 ,  1.9564  ,  1.96022 ,
        1.96595 ,  1.96977 ,  1.97168 ,  1.97359 ,  1.98887 ,  1.99269 ,
        1.9946  ,  1.99651 ,  2.00224 ,  2.00606 ,  2.00988 ,  2.01561 ,
        2.01752 ,  2.02134 ,  2.02516 ,  2.03089 ,  2.0328  ,  2.03662 ,
        2.04426 ,  2.04999 ,  2.05572 ,  2.05763 ,  2.05954 ,  2.06527 ,
        2.071   ,  2.07673 ,  2.08246 ,  2.08437 ,  2.08437 ,  2.09201 ,
        2.09583 ,  2.09774 ,  2.10538 ,  2.1092  ,  2.11111 ,  2.11684 ,
        2.12257 ,  2.12639 ,  2.13021 ,  2.13212 ,  2.13594 ,  2.13785 ,
        2.14167 ,  2.14358 ,  2.1474  ,  2.15122 ,  2.15504 ,  2.16077 ,
        2.16841 ,  2.17414 ,  2.17605 ,  2.17987 ,  2.18942 ,  2.19324 ,
        2.20279 ,  2.20852 ,  2.21807 ,  2.2238  ,  2.22571 ,  2.22953 ,
        2.23526 ,  2.24099 ,  2.24481 ,  2.24863 ,  2.25436 ,  2.26009 ,
        2.26391 ,  2.27346 ,  2.28301 ,  2.29447 ,  2.29829 ,  2.30402 ,
        2.30784 ,  2.31357 ,  2.3193  ,  2.32503 ,  2.33267 ,  2.34031 ,
        2.35177 ,  2.3575  ,  2.36132 ,  2.36705 ,  2.37087 ,  2.37469 ,
        2.37851 ,  2.38042 ,  2.38233 ,  2.38997 ,  2.39379 ,  2.39952 ,
        2.40143 ,  2.40716 ,  2.4148  ,  2.41671 ,  2.42244 ,  2.42626 ,
        2.43199 ,  2.43581 ,  2.43772 ,  2.43963 ,  2.44345 ,  2.44727 ,
        2.45109 ,  2.453   ,  2.45491 ,  2.45873 ,  2.46637 ,  2.46828 ,
        2.47592 ,  2.47974 ,  2.48547 ,  2.4912  ,  2.49884 ,  2.50266 ,
        2.50457 ,  2.50839 ,  2.51603 ,  2.52367 ,  2.52749 ,  2.53131 ,
        2.53322 ,  2.53704 ,  2.53895 ,  2.54659 ,  2.55232 ,  2.55614 ,
        2.55996 ,  2.5676  ,  2.56951 ,  2.57333 ,  2.58288 ,  2.58479 ,
        2.59816 ,  2.60007 ,  2.60389 ,  2.60962 ,  2.61344 ,  2.61726 ,
        2.62108 ,  2.63063 ,  2.63636 ,  2.644   ,  2.64973 ,  2.65164 ,
        2.65928 ,  2.66501 ,  2.67265 ,  2.68029 ,  2.68602 ,  2.68984 ,
        2.69748 ,  2.7013  ,  2.71085 ,  2.7204  ,  2.73377 ,  2.73759 ,
        2.74714 ,  2.75286 ,  2.75668 ,  2.76814 ,  2.77196 ,  2.77578 ,
        2.78151 ,  2.78533 ,  2.78724 ,  2.79106 ,  2.79297 ,  2.79488 ,
        2.80061 ,  2.80252 ,  2.80634 ,  2.81016 ,  2.81398 ,  2.81589 ,
        2.81971 ,  2.82162 ,  2.82926 ,  2.83499 ,  2.8369  ,  2.84072 ,
        2.84263 ,  2.84263 ,  2.84645 ,  2.84836 ,  2.85218 ,  2.85409 ,
        2.856   ,  2.85791 ,  2.86173 ,  2.86555 ,  2.86746 ,  2.87319 ,
        2.88083 ,  2.88274 ,  2.88656 ,  2.88847 ,  2.89611 ,  2.89993 ,
        2.90375 ,  2.91139 ,  2.91903 ,  2.92667 ,  2.93431 ,  2.94004 ,
        2.94386 ,  2.94959 ,  2.95341 ,  2.96296 ,  2.96678 ,  2.96869 ,
        2.97633 ,  2.98206 ,  2.98397 ,  2.99161 ,  3.00116 ,  3.00307 ,
        3.01453 ,  3.01835 ,  3.02599 ,  3.03554 ,  3.05082 ,  3.0661  ,
        3.07183 ,  3.07756 ,  3.0852  ,  3.09857 ,  3.11003 ,  3.11958 ,
        3.1234  ,  3.13677 ,  3.15014 ,  3.15778 ,  3.16924 ,  3.17306 ,
        3.18261 ,  3.19025 ,  3.20935 ,  3.23036 ,  3.24182 ,  3.24373 ,
        3.25901 ,  3.27047 ,  3.28002 ,  3.28766 ,  3.29912 ,  3.31058 ,
        3.32013 ,  3.32777 ,  3.33732 ,  3.34687 ,  3.35833 ,  3.37552 ,
        3.38507 ,  3.39462 ,  3.40608 ,  3.41372 ,  3.41945 ,  3.429   ,
        3.43855 ,  3.4481  ,  3.46529 ,  3.46911 ,  3.47866 ,  3.49012 ,
        3.49967 ,  3.50922 ,  3.5245  ,  3.53405 ,  3.53596 ,  3.54169 ,
        3.54742 ,  3.54933 ,  3.55506 ,  3.56461 ,  3.57225 ,  3.58562 ,
        3.59135 ,  3.60663 ])
    fig13bot_w_raw = np.array(
      [  3.77358000e-03,  -1.13208000e-02,   7.54717000e-03,
        -1.13208000e-02,   1.50943000e-02,  -3.77358000e-02,
         2.22045000e-16,  -2.64151000e-02,   7.54717000e-03,
         3.77358000e-03,   1.50943000e-02,  -1.13208000e-02,
         7.54717000e-03,  -1.50943000e-02,   3.77358000e-03,
        -1.13208000e-02,   3.77358000e-03,  -1.50943000e-02,
         1.13208000e-02,   1.50943000e-02,  -1.50943000e-02,
        -9.81132000e-02,  -5.28302000e-02,  -1.47170000e-01,
        -2.79245000e-01,  -3.47170000e-01,  -4.07547000e-01,
        -4.86792000e-01,  -6.22642000e-01,  -5.66038000e-01,
        -5.24528000e-01,  -4.26415000e-01,  -3.47170000e-01,
        -3.01887000e-01,  -4.07547000e-01,  -3.81132000e-01,
        -5.13208000e-01,  -6.94340000e-01,  -5.43396000e-01,
        -6.30189000e-01,  -4.83019000e-01,  -4.52830000e-01,
        -3.66038000e-01,  -2.75472000e-01,  -1.84906000e-01,
        -1.62264000e-01,  -1.35849000e-01,  -5.66038000e-02,
        -1.09434000e-01,   3.01887000e-02,  -6.03774000e-02,
         4.15094000e-02,  -4.90566000e-02,   6.03774000e-02,
        -3.77358000e-02,   3.39623000e-02,  -4.15094000e-02,
         4.15094000e-02,   2.22045000e-16,   3.01887000e-02,
        -3.01887000e-02,   4.52830000e-02,  -7.54717000e-03,
         3.39623000e-02,  -7.54717000e-03,   3.77358000e-02,
        -2.64151000e-02,   3.77358000e-02,  -3.39623000e-02,
         4.90566000e-02,  -1.13208000e-02,   1.50943000e-02,
        -5.66038000e-02,  -2.07547000e-01,  -2.79245000e-01,
        -3.50943000e-01,  -4.45283000e-01,  -5.32075000e-01,
        -5.77358000e-01,  -5.09434000e-01,  -4.67925000e-01,
        -3.81132000e-01,  -3.05660000e-01,  -3.35849000e-01,
        -3.84906000e-01,  -4.26415000e-01,  -4.67925000e-01,
        -5.77358000e-01,  -6.60377000e-01,  -6.11321000e-01,
        -4.03774000e-01,  -3.24528000e-01,  -2.86792000e-01,
        -2.64151000e-01,  -6.79245000e-02,  -1.62264000e-01,
        -3.77358000e-03,  -6.03774000e-02,  -1.05660000e-01,
        -1.50943000e-01,  -2.18868000e-01,  -3.05660000e-01,
        -4.03774000e-01,  -4.86792000e-01,  -5.50943000e-01,
        -6.22642000e-01,  -5.66038000e-01,  -5.24528000e-01,
        -4.07547000e-01,  -3.28302000e-01,  -3.66038000e-01,
        -4.11321000e-01,  -4.79245000e-01,  -5.39623000e-01,
        -6.52830000e-01,  -6.26415000e-01,  -5.66038000e-01,
        -4.71698000e-01,  -4.71698000e-01,  -2.37736000e-01,
        -1.69811000e-01,  -1.24528000e-01,  -7.92453000e-02,
        -3.01887000e-02,   3.77358000e-03,   3.39623000e-02,
         3.77358000e-03,   5.28302000e-02,  -1.13208000e-02,
         5.66038000e-02,   2.22045000e-16,   2.26415000e-02,
         3.77358000e-02,   3.01887000e-02,   1.13208000e-02,
         3.39623000e-02,   1.88679000e-02,   3.77358000e-02,
         1.88679000e-02,   3.39623000e-02,   2.26415000e-02,
         5.28302000e-02,   1.13208000e-02,   3.01887000e-02,
         3.01887000e-02,   7.54717000e-03,  -3.77358000e-02,
        -1.24528000e-01,  -1.96226000e-01,  -2.79245000e-01,
        -3.66038000e-01,  -4.90566000e-01,  -6.11321000e-01,
        -5.58491000e-01,  -5.01887000e-01,  -4.56604000e-01,
        -4.03774000e-01,  -3.54717000e-01,  -3.24528000e-01,
        -3.58491000e-01,  -4.15094000e-01,  -4.90566000e-01,
        -5.58491000e-01,  -6.26415000e-01,  -6.41509000e-01,
        -6.03774000e-01,  -5.09434000e-01,  -3.05660000e-01,
        -2.67925000e-01,  -2.07547000e-01,  -2.52830000e-01,
        -1.24528000e-01,  -4.90566000e-02,  -1.88679000e-02,
        -3.01887000e-02,  -1.43396000e-01,  -2.15094000e-01,
        -2.64151000e-01,  -4.00000000e-01,  -5.77358000e-01,
        -5.77358000e-01,  -6.37736000e-01,  -5.84906000e-01,
        -5.20755000e-01,  -4.60377000e-01,  -4.26415000e-01,
        -3.96226000e-01,  -3.62264000e-01,  -3.20755000e-01,
        -4.37736000e-01,  -4.90566000e-01,  -5.20755000e-01,
        -5.73585000e-01,  -6.03774000e-01,  -6.33962000e-01,
        -6.75472000e-01,  -6.37736000e-01,  -4.18868000e-01,
        -3.20755000e-01,  -2.67925000e-01,  -2.41509000e-01,
        -1.09434000e-01,  -7.54717000e-02,   1.50943000e-02,
         7.92453000e-02,  -1.50943000e-02,   5.66038000e-02,
         1.13208000e-02,   1.13208000e-02,  -1.13208000e-02,
         6.03774000e-02,  -5.66038000e-02,   4.90566000e-02,
         7.54717000e-02,   1.13208000e-02,   2.26415000e-02,
         5.66038000e-02,   7.54717000e-03,   4.52830000e-02,
         2.64151000e-02,   3.39623000e-02,   3.77358000e-03,
         2.26415000e-02,   1.88679000e-02,   3.77358000e-03,
         4.15094000e-02,   2.64151000e-02,  -3.77358000e-03,
        -3.77358000e-02,  -7.92453000e-02,  -1.47170000e-01,
        -2.11321000e-01,  -2.94340000e-01,  -4.11321000e-01,
        -5.01887000e-01,  -5.69811000e-01,  -5.96226000e-01,
        -5.73585000e-01,  -4.79245000e-01,  -5.35849000e-01,
        -4.07547000e-01,  -3.01887000e-01,  -3.01887000e-01,
        -3.50943000e-01,  -4.37736000e-01,  -5.35849000e-01,
        -5.58491000e-01,  -6.07547000e-01,  -4.98113000e-01,
        -3.47170000e-01,  -2.37736000e-01,  -2.11321000e-01,
        -1.39623000e-01,  -2.15094000e-01,   3.01887000e-02,
        -1.50943000e-02,  -5.66038000e-02,  -1.09434000e-01,
        -1.39623000e-01,  -2.15094000e-01,  -3.62264000e-01,
        -4.30189000e-01,  -4.75472000e-01,  -5.50943000e-01,
        -6.07547000e-01,  -5.50943000e-01,  -4.33962000e-01,
        -3.77358000e-01,  -2.98113000e-01,  -3.47170000e-01,
        -4.18868000e-01,  -4.60377000e-01,  -4.79245000e-01,
        -5.09434000e-01,  -5.84906000e-01,  -6.52830000e-01,
        -5.58491000e-01,  -4.22642000e-01,  -3.43396000e-01,
        -2.67925000e-01,  -1.28302000e-01,  -1.28302000e-01,
        -3.77358000e-02,  -6.79245000e-02,  -1.50943000e-02,
         9.81132000e-02,  -5.66038000e-02,   1.50943000e-02,
         4.90566000e-02,   9.43396000e-02,  -1.13208000e-02,
        -6.79245000e-02,  -1.88679000e-02,   3.77358000e-02,
         8.30189000e-02,   4.15094000e-02,  -3.77358000e-02,
         2.22045000e-16,   5.66038000e-02,  -4.90566000e-02,
         1.13208000e-02,   6.79245000e-02,   1.50943000e-02,
        -4.15094000e-02,   5.28302000e-02,  -2.64151000e-02,
         3.77358000e-02,  -7.54717000e-03,   7.54717000e-03,
        -1.13208000e-02,   4.15094000e-02,   1.50943000e-02,
         4.52830000e-02,   1.13208000e-02,  -2.64151000e-02,
        -5.66038000e-02,  -1.05660000e-01,  -1.54717000e-01,
        -2.15094000e-01,  -3.32075000e-01,  -4.07547000e-01,
        -5.01887000e-01,  -5.84906000e-01,  -6.37736000e-01,
        -5.92453000e-01,  -5.28302000e-01,  -4.56604000e-01,
        -3.69811000e-01,  -3.28302000e-01,  -3.69811000e-01,
        -4.86792000e-01,  -6.37736000e-01,  -6.94340000e-01,
        -6.41509000e-01,  -5.69811000e-01,  -5.01887000e-01,
        -3.84906000e-01,  -2.90566000e-01,  -2.26415000e-01,
        -1.47170000e-01,  -6.03774000e-02,   2.22045000e-16,
        -6.03774000e-02,  -2.64151000e-02,  -5.66038000e-02,
        -1.35849000e-01,  -2.64151000e-01,  -4.30189000e-01,
        -5.50943000e-01,  -6.41509000e-01,  -5.66038000e-01,
        -3.39623000e-01,  -3.39623000e-01,  -3.73585000e-01,
        -3.47170000e-01,  -3.01887000e-01,  -4.33962000e-01,
        -5.09434000e-01,  -5.81132000e-01,  -6.18868000e-01,
        -6.52830000e-01,  -5.58491000e-01,  -4.67925000e-01,
        -4.22642000e-01,  -2.71698000e-01,  -2.30189000e-01,
        -2.11321000e-01,  -1.69811000e-01,  -1.09434000e-01,
        -6.03774000e-02,  -3.77358000e-03,   2.64151000e-02,
        -3.77358000e-03,   3.77358000e-02,   2.22045000e-16,
         5.66038000e-02,   2.22045000e-16,   3.01887000e-02,
         3.77358000e-03,   5.66038000e-02,   7.54717000e-03,
         3.77358000e-02,   7.54717000e-03,   6.79245000e-02,
         7.54717000e-03,  -2.64151000e-02,   2.26415000e-02,
         5.28302000e-02,  -1.13208000e-02,  -1.50943000e-02,
         1.88679000e-02,  -1.88679000e-02,   7.54717000e-03,
         2.64151000e-02,   3.77358000e-02,   2.64151000e-02,
        -7.54717000e-03,   2.26415000e-02,   1.50943000e-02,
        -1.13208000e-02,   3.77358000e-02,   3.77358000e-03,
         3.77358000e-03,  -5.28302000e-02,  -1.28302000e-01,
        -2.07547000e-01,  -2.94340000e-01,  -4.26415000e-01,
        -4.98113000e-01,  -5.39623000e-01,  -5.84906000e-01,
        -5.69811000e-01,  -5.35849000e-01,  -4.15094000e-01,
        -3.81132000e-01,  -3.47170000e-01,  -3.73585000e-01,
        -4.11321000e-01,  -5.24528000e-01,  -5.66038000e-01,
        -6.11321000e-01,  -5.77358000e-01,  -5.13208000e-01,
        -4.33962000e-01,  -3.47170000e-01,  -2.83019000e-01,
        -2.41509000e-01,  -1.84906000e-01,  -1.09434000e-01,
         1.50943000e-02,  -7.92453000e-02,  -1.01887000e-01,
        -7.92453000e-02,  -1.54717000e-01,  -3.77358000e-01,
        -4.64151000e-01,  -5.84906000e-01,  -5.43396000e-01,
        -4.41509000e-01,  -3.96226000e-01,  -3.62264000e-01,
        -2.98113000e-01,  -3.62264000e-01,  -4.37736000e-01,
        -5.09434000e-01,  -5.50943000e-01,  -5.96226000e-01,
        -6.30189000e-01,  -3.92453000e-01,  -2.60377000e-01,
        -2.33962000e-01,  -1.35849000e-01,  -6.03774000e-02,
        -1.88679000e-02,   2.22045000e-16,   4.52830000e-02,
         2.22045000e-16,   2.64151000e-02,   5.28302000e-02,
         1.50943000e-02,  -7.54717000e-03,   7.54717000e-03,
         3.39623000e-02,   7.54717000e-03,   3.39623000e-02,
         3.77358000e-03,   1.50943000e-02,   4.52830000e-02,
         1.88679000e-02,   2.64151000e-02,   1.50943000e-02,
         7.54717000e-03,  -1.50943000e-02,   2.26415000e-02,
         4.15094000e-02,   1.50943000e-02,   3.01887000e-02,
         7.54717000e-03,   1.88679000e-02,   3.01887000e-02,
         7.54717000e-03,   2.26415000e-02,   7.54717000e-03,
         1.88679000e-02,   2.22045000e-16,  -1.88679000e-02,
        -4.52830000e-02,  -7.54717000e-02,  -1.58491000e-01,
        -2.33962000e-01,  -3.66038000e-01,  -4.11321000e-01,
        -4.71698000e-01,  -5.96226000e-01,  -6.41509000e-01,
        -6.41509000e-01,  -5.32075000e-01,  -6.07547000e-01,
        -5.13208000e-01,  -3.96226000e-01,  -4.15094000e-01,
        -3.81132000e-01,  -3.28302000e-01,  -3.69811000e-01,
        -3.96226000e-01,  -3.96226000e-01,  -5.50943000e-01,
        -6.33962000e-01,  -6.60377000e-01,  -6.22642000e-01,
        -5.73585000e-01,  -5.47170000e-01,  -5.01887000e-01,
        -4.18868000e-01,  -3.20755000e-01,  -2.18868000e-01,
        -1.73585000e-01,  -7.92453000e-02,   1.50943000e-02,
         6.79245000e-02,   2.22045000e-16,   3.01887000e-02,
         9.05660000e-02,  -2.26415000e-02,   5.28302000e-02,
         3.77358000e-02,  -4.90566000e-02,   1.88679000e-02,
         9.81132000e-02,  -1.16981000e-01,   3.09434000e-01,
         2.98113000e-01,   6.03774000e-02,  -1.43396000e-01,
         8.67925000e-02,   2.22045000e-16,   3.01887000e-02,
        -5.66038000e-02,   5.28302000e-02,  -1.13208000e-02,
         4.52830000e-02,  -1.50943000e-02,   3.77358000e-02,
         3.77358000e-03,   1.50943000e-02,   3.77358000e-02,
        -3.39623000e-02,   1.13208000e-02,   2.64151000e-02,
         2.26415000e-02,  -7.54717000e-03,   2.26415000e-02,
        -1.13208000e-02,   2.64151000e-02,   4.15094000e-02,
         1.88679000e-02,   3.77358000e-03,  -1.88679000e-02,
         2.26415000e-02,   2.22045000e-16,   1.88679000e-02,
         1.13208000e-02,   4.15094000e-02,  -3.77358000e-03,
         2.26415000e-02,   3.01887000e-02,   1.13208000e-02,
         2.64151000e-02,   1.50943000e-02,  -7.54717000e-03,
         2.26415000e-02,  -3.77358000e-03,   1.13208000e-02,
         1.13208000e-02,   1.50943000e-02,  -1.13208000e-02,
         1.50943000e-02,   7.54717000e-03,   1.88679000e-02,
        -3.77358000e-03,   7.54717000e-03,  -3.77358000e-03,
         1.88679000e-02,   3.77358000e-03,   7.54717000e-03,
         1.88679000e-02,  -3.77358000e-03,   7.54717000e-03,
         2.26415000e-02,   3.77358000e-03,   3.77358000e-03,
        -3.77358000e-03,   1.88679000e-02,   3.01887000e-02,
         1.13208000e-02,  -1.13208000e-02,   2.22045000e-16,
         7.54717000e-03,   7.54717000e-03,   7.54717000e-03,
         1.50943000e-02,   7.54717000e-03])

    fig14top_t_raw = np.array(
      [ 0.747111,  0.75739 ,  0.761501,  0.769724,  0.777947,  0.78617 ,
        0.794393,  0.804672,  0.808783,  0.817006,  0.825229,  0.82934 ,
        0.837563,  0.84373 ,  0.851953,  0.856065,  0.862232,  0.870455,
        0.893068,  0.903346,  0.911569,  0.917736,  0.921848,  0.938293,
        0.942405,  0.944461,  0.946516,  0.950628,  0.952683,  0.956795,
        0.960906,  0.962962,  0.965018,  0.969129,  0.973241,  0.975296,
        0.977352,  0.983519,  0.985575,  0.987631,  0.991742,  0.993798,
        1.00202 ,  1.00819 ,  1.0123  ,  1.01641 ,  1.02052 ,  1.02258 ,
        1.02875 ,  1.0308  ,  1.03491 ,  1.03697 ,  1.04108 ,  1.04519 ,
        1.04725 ,  1.0493  ,  1.05136 ,  1.05547 ,  1.06164 ,  1.08014 ,
        1.09042 ,  1.10275 ,  1.10892 ,  1.11509 ,  1.12742 ,  1.13564 ,
        1.14592 ,  1.1562  ,  1.16237 ,  1.17059 ,  1.17676 ,  1.18498 ,
        1.1932  ,  1.20348 ,  1.20759 ,  1.21376 ,  1.22198 ,  1.24665 ,
        1.25899 ,  1.26104 ,  1.26515 ,  1.26721 ,  1.26926 ,  1.27132 ,
        1.27749 ,  1.27954 ,  1.2816  ,  1.28365 ,  1.28777 ,  1.29393 ,
        1.29599 ,  1.29804 ,  1.3001  ,  1.30216 ,  1.30421 ,  1.30627 ,
        1.31243 ,  1.32066 ,  1.32271 ,  1.32477 ,  1.33094 ,  1.3371  ,
        1.34121 ,  1.34327 ,  1.34533 ,  1.34944 ,  1.35149 ,  1.35355 ,
        1.35972 ,  1.36383 ,  1.37616 ,  1.38027 ,  1.38233 ,  1.38438 ,
        1.38644 ,  1.3885  ,  1.39261 ,  1.39466 ,  1.40083 ,  1.407   ,
        1.41316 ,  1.41522 ,  1.41728 ,  1.41933 ,  1.42755 ,  1.43167 ,
        1.43372 ,  1.43578 ,  1.44194 ,  1.44606 ,  1.45222 ,  1.45633 ,
        1.45839 ,  1.46045 ,  1.4625  ,  1.46456 ,  1.46867 ,  1.47278 ,
        1.47484 ,  1.47689 ,  1.48511 ,  1.49128 ,  1.49334 ,  1.49951 ,
        1.50156 ,  1.50978 ,  1.52006 ,  1.52829 ,  1.5324  ,  1.53856 ,
        1.54062 ,  1.5509  ,  1.55501 ,  1.56529 ,  1.57351 ,  1.57968 ,
        1.58173 ,  1.58996 ,  1.59407 ,  1.60435 ,  1.60846 ,  1.61463 ,
        1.62079 ,  1.62696 ,  1.63107 ,  1.63724 ,  1.64135 ,  1.64752 ,
        1.65368 ,  1.66191 ,  1.67219 ,  1.67835 ,  1.68246 ,  1.68863 ,
        1.69069 ,  1.69274 ,  1.69685 ,  1.70097 ,  1.70508 ,  1.70713 ,
        1.71124 ,  1.71741 ,  1.71947 ,  1.72358 ,  1.72769 ,  1.73386 ,
        1.74002 ,  1.74208 ,  1.74414 ,  1.74825 ,  1.7503  ,  1.75441 ,
        1.76058 ,  1.76264 ,  1.76469 ,  1.76675 ,  1.77086 ,  1.77292 ,
        1.77497 ,  1.77703 ,  1.78114 ,  1.78525 ,  1.78936 ,  1.79142 ,
        1.79758 ,  1.8017  ,  1.80581 ,  1.80786 ,  1.80992 ,  1.81403 ,
        1.81609 ,  1.81814 ,  1.82431 ,  1.82842 ,  1.83048 ,  1.83459 ,
        1.84075 ,  1.84898 ,  1.8572  ,  1.85926 ,  1.86337 ,  1.86748 ,
        1.87159 ,  1.8757  ,  1.87776 ,  1.87981 ,  1.88187 ,  1.88392 ,
        1.88804 ,  1.89009 ,  1.8942  ,  1.89626 ,  1.90243 ,  1.90859 ,
        1.91065 ,  1.91476 ,  1.92709 ,  1.93532 ,  1.94148 ,  1.94765 ,
        1.95587 ,  1.96204 ,  1.9641  ,  1.97026 ,  1.97643 ,  1.98054 ,
        1.99699 ,  2.00316 ,  2.00521 ,  2.01549 ,  2.02988 ,  2.04016 ,
        2.05044 ,  2.07305 ,  2.07922 ,  2.08744 ,  2.09566 ,  2.10183 ,
        2.10594 ,  2.108   ,  2.11211 ,  2.11622 ,  2.11828 ,  2.12444 ,
        2.1265  ,  2.12856 ,  2.13472 ,  2.14089 ,  2.14295 ,  2.145   ,
        2.14911 ,  2.15117 ,  2.15734 ,  2.16145 ,  2.16556 ,  2.16761 ,
        2.17173 ,  2.17378 ,  2.17789 ,  2.182   ,  2.18406 ,  2.18817 ,
        2.19023 ,  2.19434 ,  2.19639 ,  2.20051 ,  2.20462 ,  2.20667 ,
        2.21284 ,  2.2149  ,  2.22106 ,  2.22723 ,  2.22929 ,  2.23134 ,
        2.23545 ,  2.23751 ,  2.23956 ,  2.24368 ,  2.24573 ,  2.2519  ,
        2.25601 ,  2.25601 ,  2.26012 ,  2.26218 ,  2.26629 ,  2.27246 ,
        2.27657 ,  2.28068 ,  2.28273 ,  2.28479 ,  2.29096 ,  2.29918 ,
        2.30124 ,  2.30535 ,  2.30946 ,  2.31151 ,  2.31563 ,  2.31974 ,
        2.32179 ,  2.32385 ,  2.33413 ,  2.33618 ,  2.34235 ,  2.35057 ,
        2.35263 ,  2.3588  ,  2.36291 ,  2.37524 ,  2.3773  ,  2.38346 ,
        2.38758 ,  2.39785 ,  2.40402 ,  2.4143  ,  2.42047 ,  2.42869 ,
        2.43486 ,  2.44514 ,  2.45336 ,  2.46158 ,  2.48008 ,  2.49036 ,
        2.49859 ,  2.50681 ,  2.51298 ,  2.52531 ,  2.52942 ,  2.53148 ,
        2.53559 ,  2.53764 ,  2.54176 ,  2.54792 ,  2.54998 ,  2.55203 ,
        2.55409 ,  2.56231 ,  2.56642 ,  2.56848 ,  2.57259 ,  2.5767  ,
        2.58287 ,  2.58904 ,  2.59109 ,  2.5952  ,  2.59726 ,  2.60137 ,
        2.60548 ,  2.60959 ,  2.61165 ,  2.61371 ,  2.61782 ,  2.61987 ,
        2.62193 ,  2.62398 ,  2.62604 ,  2.6281  ,  2.63221 ,  2.64043 ,
        2.64865 ,  2.65276 ,  2.65482 ,  2.65893 ,  2.66304 ,  2.6651  ,
        2.66921 ,  2.67743 ,  2.67949 ,  2.68154 ,  2.68771 ,  2.69182 ,
        2.69593 ,  2.70005 ,  2.7021  ,  2.70621 ,  2.71032 ,  2.71444 ,
        2.71855 ,  2.7206  ,  2.72471 ,  2.72883 ,  2.73088 ,  2.73499 ,
        2.73705 ,  2.7391  ,  2.74322 ,  2.74733 ,  2.75761 ,  2.75966 ,
        2.76583 ,  2.77816 ,  2.78227 ,  2.79255 ,  2.80078 ,  2.809   ,
        2.81928 ,  2.82956 ,  2.84806 ,  2.85422 ,  2.86656 ,  2.87889 ,
        2.88917 ,  2.89945 ,  2.92617 ,  2.94056 ,  2.9529  ,  2.95495 ,
        2.95907 ,  2.96112 ,  2.96523 ,  2.96729 ,  2.97346 ,  2.97551 ,
        2.97962 ,  2.98785 ,  2.99196 ,  2.99401 ,  2.99607 ,  3.00018 ,
        3.01252 ,  3.01663 ,  3.01868 ,  3.02279 ,  3.03307 ,  3.03718 ,
        3.03924 ,  3.04952 ,  3.05157 ,  3.05363 ,  3.0598  ,  3.06391 ,
        3.07008 ,  3.07213 ,  3.07624 ,  3.08035 ,  3.08241 ,  3.08652 ,
        3.08858 ,  3.09269 ,  3.10091 ,  3.10297 ,  3.10502 ,  3.10913 ,
        3.11325 ,  3.11941 ,  3.12558 ,  3.12558 ,  3.12969 ,  3.1338  ,
        3.13791 ,  3.14614 ,  3.14819 ,  3.15025 ,  3.15436 ,  3.15847 ,
        3.16053 ,  3.16669 ,  3.16875 ,  3.17081 ,  3.17492 ,  3.17697 ,
        3.18931 ,  3.19547 ,  3.2037  ,  3.21192 ,  3.2222  ,  3.22631 ,
        3.23042 ,  3.23453 ,  3.24276 ,  3.25098 ,  3.25509 ,  3.26126 ,
        3.26948 ,  3.2777  ,  3.28593 ,  3.29209 ,  3.30237 ,  3.30854 ,
        3.33321 ,  3.34554 ,  3.35582 ,  3.36404 ,  3.37432 ,  3.37843 ,
        3.37843 ,  3.38254 ,  3.38666 ,  3.38871 ,  3.39077 ,  3.39488 ,
        3.39899 ,  3.40105 ,  3.40516 ,  3.40927 ,  3.41132 ,  3.41544 ,
        3.41749 ,  3.4216  ,  3.42366 ,  3.43188 ,  3.43394 ,  3.43599 ,
        3.4401  ,  3.44216 ,  3.44422 ,  3.44833 ,  3.45449 ,  3.46066 ,
        3.46477 ,  3.46683 ,  3.46889 ,  3.473   ,  3.47505 ,  3.47916 ,
        3.48533 ,  3.48944 ,  3.49767 ,  3.50383 ,  3.51206 ,  3.51617 ,
        3.52233 ,  3.5285  ,  3.53467 ,  3.54289 ,  3.547   ,  3.55111 ,
        3.55728 ,  3.55934 ,  3.56345 ,  3.58401 ,  3.58606 ,  3.58812 ,
        3.59223 ,  3.5984  ,  3.60456 ,  3.61073 ,  3.61895 ,  3.62512 ,
        3.63129 ,  3.6354  ,  3.64362 ,  3.6539  ,  3.66418 ,  3.66829 ,
        3.67446 ,  3.6909  ,  3.69501 ,  3.70324 ,  3.70735 ,  3.71968 ,
        3.72379 ,  3.73202 ,  3.74435 ,  3.75257 ,  3.76285 ,  3.77108 ,
        3.78135 ,  3.80191 ,  3.8163  ,  3.82452 ,  3.83891 ,  3.85125 ,
        3.86153 ,  3.88003 ,  3.88414 ,  3.89647 ,  3.91498 ,  3.9232  ,
        3.93553 ,  3.94787 ,  3.95609 ,  3.97254 ,  3.98282 ,  3.98898 ,
        3.99721 ,  4.00132 ,  4.01571 ,  4.02599 ,  4.03832 ,  4.05065 ,
        4.06093 ,  4.06916 ,  4.0856  ,  4.09588 ,  4.1041  ,  4.11438 ,
        4.13083 ,  4.13699 ,  4.14522 ,  4.1555  ,  4.16372 ,  4.18016 ,
        4.20278 ])
    fig14top_w_raw = np.array(
      [  2.22045000e-16,   2.22045000e-16,  -1.18110000e-02,
         2.22045000e-16,   2.22045000e-16,  -2.36220000e-02,
        -3.93701000e-03,  -3.93701000e-03,  -1.57480000e-02,
         2.22045000e-16,  -1.18110000e-02,   2.22045000e-16,
        -3.93701000e-03,   7.87402000e-03,  -1.18110000e-02,
        -1.57480000e-02,   2.22045000e-16,  -7.87402000e-03,
        -7.87402000e-03,  -1.18110000e-02,   2.22045000e-16,
        -7.87402000e-03,  -3.93701000e-03,   3.54331000e-02,
         3.93701000e-03,  -5.90551000e-02,  -1.25984000e-01,
        -1.92913000e-01,  -2.32283000e-01,  -2.71654000e-01,
        -3.34646000e-01,  -4.25197000e-01,  -5.15748000e-01,
        -5.94488000e-01,  -6.61417000e-01,  -7.28346000e-01,
        -7.67717000e-01,  -7.40157000e-01,  -6.85039000e-01,
        -5.31496000e-01,  -4.72441000e-01,  -4.21260000e-01,
        -3.54331000e-01,  -4.21260000e-01,  -4.72441000e-01,
        -5.31496000e-01,  -7.40157000e-01,  -7.91339000e-01,
        -7.51969000e-01,  -6.53543000e-01,  -5.74803000e-01,
        -6.22047000e-01,  -5.03937000e-01,  -3.93701000e-01,
        -2.87402000e-01,  -1.96850000e-01,  -1.10236000e-01,
        -6.29921000e-02,   2.22045000e-16,   3.54331000e-02,
         2.36220000e-02,   2.75591000e-02,   3.54331000e-02,
         1.96850000e-02,   2.36220000e-02,   3.54331000e-02,
         2.36220000e-02,   3.93701000e-02,   3.14961000e-02,
         3.14961000e-02,   1.57480000e-02,   3.14961000e-02,
         1.57480000e-02,   2.36220000e-02,   7.87402000e-03,
         3.54331000e-02,   1.96850000e-02,   5.90551000e-02,
        -3.93701000e-03,   3.54331000e-02,  -7.08661000e-02,
        -1.53543000e-01,  -2.28346000e-01,  -3.07087000e-01,
        -4.05512000e-01,  -4.80315000e-01,  -5.27559000e-01,
        -7.32283000e-01,  -7.63780000e-01,  -7.32283000e-01,
        -6.73228000e-01,  -5.98425000e-01,  -5.27559000e-01,
        -4.52756000e-01,  -3.97638000e-01,  -3.70079000e-01,
        -3.50394000e-01,  -4.44882000e-01,  -3.66142000e-01,
        -6.33858000e-01,  -6.81102000e-01,  -6.57480000e-01,
        -6.02362000e-01,  -5.19685000e-01,  -3.74016000e-01,
        -2.99213000e-01,  -2.32283000e-01,  -1.65354000e-01,
        -7.48031000e-02,  -1.18110000e-02,  -5.90551000e-02,
        -9.05512000e-02,  -1.29921000e-01,  -2.00787000e-01,
        -2.59843000e-01,  -3.22835000e-01,  -5.78740000e-01,
        -6.92913000e-01,  -7.75591000e-01,  -7.12598000e-01,
        -5.55118000e-01,  -4.88189000e-01,  -4.44882000e-01,
        -4.09449000e-01,  -3.74016000e-01,  -4.01575000e-01,
        -4.52756000e-01,  -5.31496000e-01,  -7.20472000e-01,
        -8.14961000e-01,  -7.59843000e-01,  -5.70866000e-01,
        -4.76378000e-01,  -4.09449000e-01,  -3.42520000e-01,
        -2.71654000e-01,  -2.00787000e-01,  -1.45669000e-01,
        -1.02362000e-01,  -4.72441000e-02,   1.18110000e-02,
         1.18110000e-02,   3.93701000e-03,   1.18110000e-02,
         3.54331000e-02,   2.75591000e-02,   2.75591000e-02,
         1.57480000e-02,   2.22045000e-16,   1.18110000e-02,
         3.54331000e-02,   1.96850000e-02,   1.96850000e-02,
         3.93701000e-02,   1.57480000e-02,   2.75591000e-02,
         4.72441000e-02,   3.54331000e-02,   1.57480000e-02,
         3.54331000e-02,   3.14961000e-02,   1.96850000e-02,
         3.14961000e-02,   3.93701000e-02,   2.75591000e-02,
         2.75591000e-02,   3.93701000e-02,   2.36220000e-02,
         3.54331000e-02,   5.51181000e-02,   4.33071000e-02,
         5.51181000e-02,   3.54331000e-02,  -9.05512000e-02,
        -1.65354000e-01,  -2.28346000e-01,  -2.99213000e-01,
        -4.01575000e-01,  -6.57480000e-01,  -6.88976000e-01,
        -7.40157000e-01,  -6.85039000e-01,  -5.43307000e-01,
        -4.84252000e-01,  -4.37008000e-01,  -3.54331000e-01,
        -3.81890000e-01,  -4.33071000e-01,  -5.00000000e-01,
        -5.98425000e-01,  -6.85039000e-01,  -7.44094000e-01,
        -6.81102000e-01,  -6.06299000e-01,  -5.35433000e-01,
        -4.60630000e-01,  -3.74016000e-01,  -2.83465000e-01,
        -1.96850000e-01,  -1.49606000e-01,  -1.06299000e-01,
        -7.87402000e-02,  -7.87402000e-03,   1.18110000e-02,
        -3.93701000e-03,  -9.84252000e-02,  -1.77165000e-01,
        -2.71654000e-01,  -3.81890000e-01,  -5.00000000e-01,
        -6.06299000e-01,  -7.24409000e-01,  -7.95276000e-01,
        -7.16535000e-01,  -6.41732000e-01,  -5.66929000e-01,
        -4.44882000e-01,  -3.81890000e-01,  -4.44882000e-01,
        -6.53543000e-01,  -7.44094000e-01,  -7.71654000e-01,
        -7.51969000e-01,  -7.00787000e-01,  -6.37795000e-01,
        -5.62992000e-01,  -4.80315000e-01,  -3.93701000e-01,
        -3.22835000e-01,  -2.48031000e-01,  -1.77165000e-01,
        -9.84252000e-02,  -5.11811000e-02,  -2.75591000e-02,
         1.96850000e-02,   3.54331000e-02,   2.75591000e-02,
         2.75591000e-02,   3.54331000e-02,   3.54331000e-02,
         1.57480000e-02,   2.36220000e-02,   4.33071000e-02,
         1.96850000e-02,   1.96850000e-02,   3.14961000e-02,
         7.87402000e-03,   1.57480000e-02,   3.54331000e-02,
         1.57480000e-02,   3.54331000e-02,   1.18110000e-02,
         2.75591000e-02,   3.14961000e-02,   5.51181000e-02,
         4.33071000e-02,   6.29921000e-02,   3.14961000e-02,
         1.57480000e-02,  -3.93701000e-02,  -9.05512000e-02,
        -1.69291000e-01,  -3.70079000e-01,  -4.76378000e-01,
        -5.90551000e-01,  -6.88976000e-01,  -7.59843000e-01,
        -7.20472000e-01,  -6.45669000e-01,  -5.59055000e-01,
        -4.84252000e-01,  -4.33071000e-01,  -3.62205000e-01,
        -3.93701000e-01,  -4.56693000e-01,  -5.31496000e-01,
        -6.37795000e-01,  -7.28346000e-01,  -7.75591000e-01,
        -7.36220000e-01,  -6.88976000e-01,  -6.10236000e-01,
        -4.44882000e-01,  -3.66142000e-01,  -2.83465000e-01,
        -2.08661000e-01,  -1.33858000e-01,  -7.87402000e-02,
        -2.75591000e-02,   3.93701000e-03,  -1.18110000e-02,
        -9.84252000e-02,  -1.73228000e-01,  -2.71654000e-01,
        -3.62205000e-01,  -4.72441000e-01,  -5.78740000e-01,
        -6.92913000e-01,  -7.44094000e-01,  -7.04724000e-01,
        -5.55118000e-01,  -5.55118000e-01,  -4.33071000e-01,
        -3.77953000e-01,  -3.34646000e-01,  -3.11024000e-01,
        -3.50394000e-01,  -4.01575000e-01,  -4.80315000e-01,
        -5.59055000e-01,  -6.45669000e-01,  -5.82677000e-01,
        -5.47244000e-01,  -4.29134000e-01,  -3.50394000e-01,
        -2.63780000e-01,  -1.96850000e-01,  -9.05512000e-02,
        -6.69291000e-02,  -4.33071000e-02,  -1.18110000e-02,
         1.18110000e-02,   3.93701000e-02,   2.36220000e-02,
         2.22045000e-16,   7.87402000e-03,   2.75591000e-02,
         1.18110000e-02,  -1.57480000e-02,   1.18110000e-02,
         3.93701000e-02,   7.87402000e-03,   2.36220000e-02,
         2.75591000e-02,  -3.93701000e-03,   3.93701000e-03,
         2.36220000e-02,   2.22045000e-16,   1.57480000e-02,
         1.57480000e-02,   2.75591000e-02,   1.57480000e-02,
         3.93701000e-02,   3.93701000e-02,   3.54331000e-02,
         4.33071000e-02,   2.22045000e-16,  -5.11811000e-02,
        -1.18110000e-01,  -1.85039000e-01,  -2.67717000e-01,
        -4.76378000e-01,  -5.98425000e-01,  -7.04724000e-01,
        -7.59843000e-01,  -6.85039000e-01,  -6.14173000e-01,
        -5.51181000e-01,  -4.76378000e-01,  -3.77953000e-01,
        -3.46457000e-01,  -4.25197000e-01,  -5.31496000e-01,
        -6.25984000e-01,  -6.96850000e-01,  -7.24409000e-01,
        -6.92913000e-01,  -6.33858000e-01,  -5.47244000e-01,
        -4.64567000e-01,  -4.01575000e-01,  -3.42520000e-01,
        -2.75591000e-01,  -2.04724000e-01,  -1.77165000e-01,
        -1.02362000e-01,  -5.51181000e-02,  -1.18110000e-02,
        -4.33071000e-02,  -1.65354000e-01,  -2.40157000e-01,
        -3.46457000e-01,  -5.82677000e-01,  -6.81102000e-01,
        -7.36220000e-01,  -6.96850000e-01,  -5.82677000e-01,
        -5.27559000e-01,  -4.64567000e-01,  -3.50394000e-01,
        -3.85827000e-01,  -4.48819000e-01,  -5.19685000e-01,
        -6.22047000e-01,  -7.28346000e-01,  -8.62205000e-01,
        -8.26772000e-01,  -7.28346000e-01,  -6.33858000e-01,
        -5.51181000e-01,  -3.85827000e-01,  -3.14961000e-01,
        -2.36220000e-01,  -1.65354000e-01,  -9.05512000e-02,
        -3.93701000e-02,  -1.18110000e-02,   1.57480000e-02,
         3.93701000e-02,   1.57480000e-02,   2.36220000e-02,
         1.96850000e-02,   1.18110000e-02,   3.93701000e-02,
         7.87402000e-03,   2.75591000e-02,   7.87402000e-03,
         3.14961000e-02,   1.57480000e-02,   3.14961000e-02,
         1.57480000e-02,   2.75591000e-02,   3.14961000e-02,
         5.51181000e-02,   2.75591000e-02,   2.22045000e-16,
        -9.84252000e-02,  -1.96850000e-01,  -3.07087000e-01,
        -3.93701000e-01,  -4.84252000e-01,  -7.00787000e-01,
        -7.75591000e-01,  -7.00787000e-01,  -5.47244000e-01,
        -4.60630000e-01,  -3.77953000e-01,  -3.54331000e-01,
        -3.70079000e-01,  -4.21260000e-01,  -6.25984000e-01,
        -6.81102000e-01,  -6.65354000e-01,  -5.51181000e-01,
        -3.77953000e-01,  -5.74803000e-01,  -4.64567000e-01,
        -2.12598000e-01,   3.93701000e-02,  -3.54331000e-02,
        -1.57480000e-02,  -5.11811000e-02,  -1.69291000e-01,
        -2.63780000e-01,  -3.50394000e-01,  -4.64567000e-01,
        -6.06299000e-01,  -6.92913000e-01,  -6.41732000e-01,
        -5.70866000e-01,  -4.96063000e-01,  -4.21260000e-01,
        -3.74016000e-01,  -3.26772000e-01,  -3.77953000e-01,
        -3.77953000e-01,  -5.47244000e-01,  -6.22047000e-01,
        -7.16535000e-01,  -6.41732000e-01,  -5.55118000e-01,
        -4.72441000e-01,  -3.97638000e-01,  -2.44094000e-01,
        -1.77165000e-01,  -1.41732000e-01,  -1.06299000e-01,
        -6.69291000e-02,  -3.14961000e-02,   7.87402000e-03,
         1.57480000e-02,   4.33071000e-02,   3.54331000e-02,
         4.33071000e-02,   1.96850000e-02,   7.87402000e-03,
         3.14961000e-02,   5.51181000e-02,   1.57480000e-02,
         1.57480000e-02,   4.72441000e-02,   3.54331000e-02,
         3.93701000e-03,   4.72441000e-02,   2.75591000e-02,
         2.75591000e-02,   4.72441000e-02,   3.14961000e-02,
         2.75591000e-02,   3.93701000e-02,   5.90551000e-02,
         6.29921000e-02,   4.72441000e-02,   3.93701000e-03,
         3.93701000e-03,  -1.14173000e-01,  -1.88976000e-01,
        -2.79528000e-01,  -3.77953000e-01,  -4.92126000e-01,
        -6.33858000e-01,  -7.48031000e-01,  -8.11024000e-01,
        -7.55906000e-01,  -6.81102000e-01,  -5.94488000e-01,
        -5.31496000e-01,  -4.05512000e-01,  -3.26772000e-01,
        -3.46457000e-01,  -3.81890000e-01,  -4.25197000e-01,
        -4.80315000e-01,  -5.39370000e-01,  -5.98425000e-01,
        -6.88976000e-01,  -6.14173000e-01,  -5.51181000e-01,
        -3.89764000e-01,  -3.26772000e-01,  -2.83465000e-01,
        -2.08661000e-01,  -1.33858000e-01,  -7.08661000e-02,
        -1.57480000e-02,   3.14961000e-02,   7.87402000e-03,
         4.72441000e-02,   1.57480000e-02,   3.54331000e-02,
        -7.87402000e-03,   5.51181000e-02,  -2.75591000e-02,
         8.26772000e-02,  -3.54331000e-02,  -1.18110000e-01,
         1.14173000e-01,   2.04724000e-01,   2.83465000e-01,
         2.83465000e-01,   1.25984000e-01,   2.75591000e-02,
        -1.25984000e-01,  -3.14961000e-02,  -1.25984000e-01,
         8.26772000e-02,  -3.54331000e-02,   5.51181000e-02,
        -3.93701000e-03,   5.51181000e-02,   2.22045000e-16,
         1.96850000e-02,   3.54331000e-02,  -3.93701000e-03,
         2.36220000e-02,   4.33071000e-02,   3.93701000e-03,
         2.75591000e-02,   1.96850000e-02,   1.57480000e-02,
         2.36220000e-02,   1.18110000e-02,   7.87402000e-03,
         2.36220000e-02,   1.57480000e-02,   1.96850000e-02,
         1.96850000e-02,   3.54331000e-02,   3.14961000e-02,
         1.57480000e-02,   2.36220000e-02,   1.57480000e-02,
         2.36220000e-02,   4.33071000e-02,   2.75591000e-02,
         2.75591000e-02,   4.33071000e-02,   2.75591000e-02,
         2.36220000e-02,   3.54331000e-02,   1.57480000e-02,
         2.75591000e-02,   1.18110000e-02,   1.57480000e-02,
         3.54331000e-02,   1.96850000e-02,   3.54331000e-02,
         1.96850000e-02,   3.93701000e-02,   4.72441000e-02,
         2.36220000e-02,   2.75591000e-02,   3.14961000e-02,
         3.54331000e-02,   4.72441000e-02,   3.14961000e-02,
         2.75591000e-02,   2.75591000e-02,   3.93701000e-02,
         3.14961000e-02,   3.93701000e-02,   3.54331000e-02,
         2.36220000e-02])

    fig14bot_t_raw = np.array(
      [ 0.578169,  0.584338,  0.592563,  0.600788,  0.613126,  0.621351,
        0.629576,  0.639857,  0.648082,  0.654251,  0.662476,  0.670701,
        0.67687 ,  0.683039,  0.689208,  0.691264,  0.695377,  0.699489,
        0.705658,  0.709771,  0.713883,  0.720052,  0.728277,  0.738559,
        0.74884 ,  0.759121,  0.763234,  0.771459,  0.773515,  0.777628,
        0.785853,  0.787909,  0.794078,  0.796134,  0.800247,  0.80436 ,
        0.808472,  0.812585,  0.818754,  0.824922,  0.826979,  0.831091,
        0.839316,  0.845485,  0.847541,  0.851654,  0.85371 ,  0.857823,
        0.861935,  0.866048,  0.870161,  0.872217,  0.876329,  0.880442,
        0.882498,  0.888667,  0.890723,  0.894836,  0.905117,  0.911286,
        0.919511,  0.927736,  0.938018,  0.946243,  0.956524,  0.970918,
        0.977087,  0.985312,  0.993537,  1.00382 ,  1.00999 ,  1.01616 ,
        1.02438 ,  1.04289 ,  1.05111 ,  1.06962 ,  1.07579 ,  1.07784 ,
        1.08401 ,  1.09224 ,  1.09635 ,  1.09841 ,  1.10252 ,  1.10458 ,
        1.11075 ,  1.1128  ,  1.11691 ,  1.11897 ,  1.12514 ,  1.12925 ,
        1.13336 ,  1.13542 ,  1.13748 ,  1.14159 ,  1.14776 ,  1.15187 ,
        1.15393 ,  1.15804 ,  1.1601  ,  1.16215 ,  1.16421 ,  1.16626 ,
        1.17243 ,  1.17449 ,  1.17655 ,  1.18066 ,  1.18683 ,  1.18888 ,
        1.19094 ,  1.193   ,  1.193   ,  1.19711 ,  1.20122 ,  1.20945 ,
        1.21356 ,  1.21562 ,  1.21767 ,  1.22178 ,  1.22384 ,  1.2259  ,
        1.23207 ,  1.23412 ,  1.23823 ,  1.2444  ,  1.24646 ,  1.25057 ,
        1.25057 ,  1.25263 ,  1.25674 ,  1.2588  ,  1.26497 ,  1.26702 ,
        1.27319 ,  1.2773  ,  1.27936 ,  1.28142 ,  1.28964 ,  1.2917  ,
        1.29375 ,  1.29581 ,  1.30198 ,  1.30404 ,  1.3102  ,  1.31432 ,
        1.31843 ,  1.32871 ,  1.33488 ,  1.33899 ,  1.34927 ,  1.35339 ,
        1.36367 ,  1.36984 ,  1.37806 ,  1.38629 ,  1.39451 ,  1.40274 ,
        1.41096 ,  1.41919 ,  1.43152 ,  1.43769 ,  1.44592 ,  1.44798 ,
        1.45209 ,  1.4562  ,  1.46031 ,  1.46237 ,  1.46854 ,  1.47676 ,
        1.48499 ,  1.4891  ,  1.50349 ,  1.51583 ,  1.51994 ,  1.522   ,
        1.52611 ,  1.52817 ,  1.53228 ,  1.5364  ,  1.53845 ,  1.54256 ,
        1.54462 ,  1.54668 ,  1.5549  ,  1.55901 ,  1.56724 ,  1.57341 ,
        1.57546 ,  1.57752 ,  1.57958 ,  1.58163 ,  1.58369 ,  1.58575 ,
        1.59191 ,  1.59808 ,  1.60014 ,  1.60631 ,  1.60836 ,  1.61042 ,
        1.61453 ,  1.6207  ,  1.62893 ,  1.63715 ,  1.63921 ,  1.64127 ,
        1.64332 ,  1.64743 ,  1.65155 ,  1.6536  ,  1.65566 ,  1.65977 ,
        1.66388 ,  1.668   ,  1.67005 ,  1.67211 ,  1.67828 ,  1.6865  ,
        1.68856 ,  1.69062 ,  1.69473 ,  1.69884 ,  1.7009  ,  1.70501 ,
        1.71118 ,  1.71324 ,  1.71529 ,  1.71735 ,  1.72352 ,  1.72557 ,
        1.72969 ,  1.73585 ,  1.74202 ,  1.74614 ,  1.7523  ,  1.75642 ,
        1.76053 ,  1.76259 ,  1.7667  ,  1.77492 ,  1.78315 ,  1.79343 ,
        1.7996  ,  1.81194 ,  1.82016 ,  1.8325  ,  1.84484 ,  1.85306 ,
        1.8654  ,  1.87568 ,  1.88185 ,  1.88802 ,  1.89624 ,  1.90036 ,
        1.91064 ,  1.92092 ,  1.92503 ,  1.93326 ,  1.94148 ,  1.94354 ,
        1.94765 ,  1.95176 ,  1.95588 ,  1.95793 ,  1.96205 ,  1.9641  ,
        1.96616 ,  1.97027 ,  1.9785  ,  1.98055 ,  1.98261 ,  1.98878 ,
        1.99495 ,  1.99906 ,  2.00317 ,  2.00523 ,  2.00728 ,  2.00934 ,
        2.01345 ,  2.02168 ,  2.02373 ,  2.0299  ,  2.03196 ,  2.03401 ,
        2.03607 ,  2.04224 ,  2.04635 ,  2.05252 ,  2.05663 ,  2.0628  ,
        2.06692 ,  2.07103 ,  2.07308 ,  2.07514 ,  2.0772  ,  2.07925 ,
        2.08337 ,  2.08953 ,  2.09159 ,  2.09365 ,  2.09776 ,  2.10187 ,
        2.10598 ,  2.11215 ,  2.11832 ,  2.12038 ,  2.12243 ,  2.12243 ,
        2.1286  ,  2.13477 ,  2.13683 ,  2.13889 ,  2.143   ,  2.14711 ,
        2.14917 ,  2.15122 ,  2.15328 ,  2.15534 ,  2.1615  ,  2.16562 ,
        2.16973 ,  2.18207 ,  2.18412 ,  2.19235 ,  2.19852 ,  2.20263 ,
        2.21085 ,  2.21908 ,  2.22731 ,  2.23347 ,  2.23759 ,  2.24581 ,
        2.25198 ,  2.26226 ,  2.27049 ,  2.28077 ,  2.28694 ,  2.29722 ,
        2.30133 ,  2.30544 ,  2.31367 ,  2.31984 ,  2.32806 ,  2.33834 ,
        2.34451 ,  2.35479 ,  2.36302 ,  2.36713 ,  2.37124 ,  2.37536 ,
        2.37947 ,  2.38153 ,  2.38564 ,  2.38975 ,  2.39181 ,  2.40003 ,
        2.40415 ,  2.4062  ,  2.41237 ,  2.4206  ,  2.42265 ,  2.42676 ,
        2.43088 ,  2.43293 ,  2.4391  ,  2.44733 ,  2.45144 ,  2.45555 ,
        2.45966 ,  2.46172 ,  2.46378 ,  2.46583 ,  2.472   ,  2.47817 ,
        2.48023 ,  2.4864  ,  2.49051 ,  2.49668 ,  2.49873 ,  2.50079 ,
        2.50696 ,  2.51313 ,  2.51724 ,  2.51724 ,  2.52135 ,  2.53369 ,
        2.5378  ,  2.53986 ,  2.54192 ,  2.54397 ,  2.54808 ,  2.55631 ,
        2.55837 ,  2.56248 ,  2.56453 ,  2.56865 ,  2.57276 ,  2.57482 ,
        2.57893 ,  2.5851  ,  2.59127 ,  2.59538 ,  2.60772 ,  2.61594 ,
        2.62828 ,  2.63445 ,  2.64062 ,  2.6509  ,  2.65707 ,  2.66735 ,
        2.67969 ,  2.6838  ,  2.68997 ,  2.70025 ,  2.70847 ,  2.71876 ,
        2.72904 ,  2.74754 ,  2.75577 ,  2.76194 ,  2.77428 ,  2.78456 ,
        2.78867 ,  2.79278 ,  2.79484 ,  2.79895 ,  2.80306 ,  2.80512 ,
        2.80718 ,  2.81129 ,  2.81334 ,  2.82157 ,  2.82568 ,  2.82774 ,
        2.8298  ,  2.83596 ,  2.84213 ,  2.84625 ,  2.8483  ,  2.85036 ,
        2.85241 ,  2.85653 ,  2.85858 ,  2.86475 ,  2.87092 ,  2.87298 ,
        2.87709 ,  2.8812  ,  2.88531 ,  2.88737 ,  2.89354 ,  2.89971 ,
        2.90382 ,  2.90588 ,  2.90999 ,  2.9141  ,  2.91616 ,  2.91822 ,
        2.92027 ,  2.92233 ,  2.9285  ,  2.93261 ,  2.93878 ,  2.94083 ,
        2.94495 ,  2.94906 ,  2.95112 ,  2.95523 ,  2.95728 ,  2.9614  ,
        2.96345 ,  2.96551 ,  2.96757 ,  2.96962 ,  2.97373 ,  2.9799  ,
        2.98196 ,  2.98402 ,  2.98813 ,  2.99018 ,  2.9943  ,  2.99635 ,
        2.99841 ,  3.00458 ,  3.0128  ,  3.01486 ,  3.0272  ,  3.03954 ,
        3.05187 ,  3.05599 ,  3.07449 ,  3.08066 ,  3.08683 ,  3.10122 ,
        3.10739 ,  3.11562 ,  3.11973 ,  3.1259  ,  3.13001 ,  3.14441 ,
        3.15469 ,  3.16086 ,  3.16291 ,  3.17319 ,  3.17731 ,  3.18759 ,
        3.19581 ,  3.20404 ,  3.21226 ,  3.21432 ,  3.21843 ,  3.22049 ,
        3.22254 ,  3.22666 ,  3.23077 ,  3.23283 ,  3.23488 ,  3.23694 ,
        3.24311 ,  3.24928 ,  3.24928 ,  3.25133 ,  3.25339 ,  3.2575  ,
        3.26573 ,  3.27395 ,  3.27806 ,  3.28218 ,  3.28423 ,  3.28629 ,
        3.28835 ,  3.29451 ,  3.29657 ,  3.30274 ,  3.30685 ,  3.30891 ,
        3.31096 ,  3.31508 ,  3.3233  ,  3.33564 ,  3.34798 ,  3.3562  ,
        3.36854 ,  3.37882 ,  3.39322 ,  3.4035  ,  3.41378 ,  3.42406 ,
        3.43228 ,  3.4364  ,  3.44051 ,  3.44257 ,  3.44874 ,  3.46313 ,
        3.47135 ,  3.48369 ,  3.48575 ,  3.49397 ,  3.49809 ,  3.50837 ,
        3.51248 ,  3.5207  ,  3.53099 ,  3.53921 ,  3.55361 ,  3.568   ,
        3.57417 ,  3.58239 ,  3.58856 ,  3.59473 ,  3.60296 ,  3.61118 ,
        3.62146 ,  3.62969 ,  3.64408 ,  3.65025 ,  3.65642 ,  3.66053 ,
        3.68109 ,  3.69138 ,  3.69755 ,  3.70166 ,  3.70988 ,  3.72428 ,
        3.73456 ,  3.75101 ,  3.76335 ,  3.79213 ,  3.81064 ,  3.82298 ,
        3.83532 ,  3.84148 ,  3.85177 ,  3.85588 ,  3.85999 ,  3.87233 ,
        3.87644 ,  3.88467 ,  3.89084 ,  3.897   ,  3.90317 ,  3.9114  ,
        3.92168 ,  3.93402 ,  3.94224 ,  3.95458 ,  3.96075 ,  3.96692 ,
        3.97103 ,  3.97926 ,  3.98748 ,  3.99571 ,  4.00393 ,  4.02038 ,
        4.03066 ,  4.04094 ])
    fig14bot_w_raw = np.array(
      [  3.65926000e-02,   6.91201000e-02,   2.84794000e-02,
         1.22288000e-02,   2.85034000e-02,   1.22529000e-02,
         3.25878000e-02,  -3.98557000e-03,   4.07395000e-02,
        -1.20988000e-02,   2.04312000e-02,  -3.94945000e-03,
         1.63830000e-02,   8.26012000e-03,   3.67226000e-02,
         4.89202000e-02,   3.26648000e-02,  -3.91573000e-03,
        -3.90851000e-03,   4.48768000e-02,   2.45564000e-02,
         3.26937000e-02,   2.05082000e-02,   3.27154000e-02,
         1.24022000e-02,   3.68045000e-02,  -1.60362000e-02,
         6.93393000e-02,   9.77970000e-02,   3.27611000e-02,
         1.24456000e-02,  -4.44626000e-02,  -1.29821000e-01,
        -1.78599000e-01,  -2.07050000e-01,  -2.68021000e-01,
        -4.55008000e-01,  -4.91588000e-01,  -4.99711000e-01,
        -4.79379000e-01,  -4.10270000e-01,  -2.72054000e-01,
        -2.88305000e-01,  -3.65533000e-01,  -4.18376000e-01,
        -4.95607000e-01,  -5.52516000e-01,  -5.76901000e-01,
        -5.97221000e-01,  -5.60631000e-01,  -4.22415000e-01,
        -3.41112000e-01,  -2.55741000e-01,  -1.74436000e-01,
        -1.41913000e-01,  -8.09300000e-02,  -1.18219000e-02,
         4.50935000e-02,  -3.21302000e-02,   6.95030000e-02,
         8.53707000e-03,   3.70020000e-02,   1.26238000e-02,
         4.10887000e-02,   2.48406000e-02,   4.92477000e-02,
         1.26695000e-02,   4.51995000e-02,   2.08189000e-02,
         2.89610000e-02,   1.67731000e-02,  -1.98050000e-02,
         2.89851000e-02,   5.51478000e-04,   4.12115000e-02,
         1.27779000e-02,   7.78258000e-02,   9.81534000e-02,
         4.93801000e-02,   8.19101000e-02,   4.67913000e-03,
        -7.25542000e-02,  -1.41655000e-01,  -1.86368000e-01,
        -2.39206000e-01,  -3.24570000e-01,  -3.97736000e-01,
        -4.87164000e-01,  -5.44068000e-01,  -5.03412000e-01,
        -4.50562000e-01,  -3.77389000e-01,  -3.20476000e-01,
        -2.59496000e-01,  -2.14773000e-01,  -2.26963000e-01,
        -2.71676000e-01,  -3.40777000e-01,  -4.13945000e-01,
        -4.87114000e-01,  -5.27762000e-01,  -5.88735000e-01,
        -5.72468000e-01,  -5.19620000e-01,  -4.22056000e-01,
        -3.32621000e-01,  -2.71638000e-01,  -2.06595000e-01,
        -1.37487000e-01,  -8.87036000e-02,  -8.87036000e-02,
         8.86217000e-03,   3.32572000e-02,   2.10717000e-02,
        -1.95738000e-02,  -6.02218000e-02,  -1.49650000e-01,
        -2.02491000e-01,  -2.79724000e-01,  -3.56958000e-01,
        -4.66707000e-01,  -5.23615000e-01,  -5.72390000e-01,
        -5.35798000e-01,  -4.95145000e-01,  -3.97579000e-01,
        -3.97579000e-01,  -3.04081000e-01,  -2.43101000e-01,
        -2.10578000e-01,  -2.91871000e-01,  -3.28454000e-01,
        -4.01618000e-01,  -5.43889000e-01,  -6.73968000e-01,
        -7.18681000e-01,  -6.73956000e-01,  -5.07287000e-01,
        -4.21919000e-01,  -3.32486000e-01,  -2.34917000e-01,
        -7.63785000e-02,  -2.35257000e-02,   8.69360000e-04,
         2.11994000e-02,   2.93415000e-02,   6.59341000e-02,
         5.78088000e-02,   7.00160000e-02,   4.56306000e-02,
         5.37727000e-02,   2.53246000e-02,   3.75294000e-02,
         5.01869000e-03,   6.19389000e-02,   3.75583000e-02,
         6.60232000e-02,   3.35125000e-02,   4.16570000e-02,
         2.13391000e-02,   3.35438000e-02,   5.79365000e-02,
         2.54210000e-02,   1.03553000e-03,   3.76257000e-02,
         7.01484000e-02,   2.13752000e-02,   5.39051000e-02,
         4.17196000e-02,   6.20497000e-02,   8.64568000e-02,
         5.80159000e-02,  -2.32801000e-02,  -8.83183000e-02,
        -1.37094000e-01,  -1.89937000e-01,  -2.42778000e-01,
        -3.03749000e-01,  -3.48462000e-01,  -3.97237000e-01,
        -4.66341000e-01,  -4.98859000e-01,  -4.54133000e-01,
        -3.24047000e-01,  -2.71192000e-01,  -2.14274000e-01,
        -2.58987000e-01,  -3.32156000e-01,  -3.84999000e-01,
        -4.78492000e-01,  -5.11010000e-01,  -5.63853000e-01,
        -5.96367000e-01,  -5.23189000e-01,  -4.37820000e-01,
        -3.52447000e-01,  -2.79274000e-01,  -2.06101000e-01,
        -1.20730000e-01,  -6.90190000e-03,   1.74980000e-02,
        -8.41184000e-02,  -6.88023000e-03,  -1.57284000e-01,
        -2.22323000e-01,  -2.71098000e-01,  -3.28004000e-01,
        -3.76782000e-01,  -4.33690000e-01,  -5.02791000e-01,
        -4.78396000e-01,  -4.45871000e-01,  -4.13348000e-01,
        -3.72695000e-01,  -2.95452000e-01,  -2.30402000e-01,
        -2.91375000e-01,  -3.40153000e-01,  -4.29579000e-01,
        -4.90550000e-01,  -6.08434000e-01,  -6.49080000e-01,
        -6.16552000e-01,  -5.67769000e-01,  -4.37685000e-01,
        -3.84837000e-01,  -3.07594000e-01,  -1.93771000e-01,
        -1.24660000e-01,  -2.70200000e-03,  -1.89549000e-02,
         5.01556000e-02,   8.67482000e-02,   3.79725000e-02,
         1.76521000e-02,   5.42399000e-02,   8.67650000e-02,
         4.20592000e-02,   5.42640000e-02,   9.56055000e-03,
         2.58279000e-02,  -1.88731000e-02,   9.59186000e-03,
         9.60631000e-03,  -2.28996000e-02,   3.80857000e-02,
         5.02952000e-02,  -1.06683000e-02,   8.28348000e-02,
         3.81266000e-02,   4.22013000e-02,   2.18809000e-02,
         7.06734000e-02,   6.25554000e-02,   9.50805000e-02,
         6.66349000e-02,   5.03843000e-02,  -6.74994000e-02,
        -1.40665000e-01,  -1.85376000e-01,  -2.34152000e-01,
        -2.95125000e-01,  -4.21136000e-01,  -4.73979000e-01,
        -5.18692000e-01,  -5.63403000e-01,  -5.22743000e-01,
        -4.29245000e-01,  -3.56072000e-01,  -3.03219000e-01,
        -2.46301000e-01,  -3.15402000e-01,  -4.17023000e-01,
        -4.61736000e-01,  -5.06449000e-01,  -5.67422000e-01,
        -6.20263000e-01,  -5.83668000e-01,  -5.06430000e-01,
        -3.43821000e-01,  -2.70648000e-01,  -1.60889000e-01,
        -9.17813000e-02,  -5.92537000e-02,  -1.85985000e-02,
         6.67746000e-02,   1.73872000e-03,  -1.20205000e-01,
        -1.73046000e-01,  -2.34017000e-01,  -3.07185000e-01,
        -3.76288000e-01,  -4.41327000e-01,  -4.94170000e-01,
        -5.26685000e-01,  -5.06353000e-01,  -4.65700000e-01,
        -4.16917000e-01,  -3.35611000e-01,  -2.78696000e-01,
        -2.21781000e-01,  -2.42099000e-01,  -3.59978000e-01,
        -4.29081000e-01,  -5.14444000e-01,  -5.14444000e-01,
        -6.36388000e-01,  -6.11991000e-01,  -5.42883000e-01,
        -4.16864000e-01,  -3.55884000e-01,  -2.78643000e-01,
        -2.21730000e-01,  -1.44492000e-01,  -5.91237000e-02,
         1.40494000e-02,   2.62518000e-02,   5.47119000e-02,
         8.72370000e-02,   5.87962000e-02,   7.50588000e-02,
         4.66131000e-02,   3.84903000e-02,   5.47552000e-02,
         3.03746000e-02,   5.07095000e-02,   2.63288000e-02,
         4.66613000e-02,   7.91864000e-02,   5.07408000e-02,
         9.13984000e-02,   3.85649000e-02,   1.03615000e-01,
         3.04565000e-02,   6.70491000e-02,  -6.10960000e-03,
         5.89359000e-02,   8.73960000e-02,   2.64300000e-02,
         6.30226000e-02,   4.27070000e-02,   6.71093000e-02,
         5.49214000e-02,   8.33887000e-02,   3.05528000e-02,
        -5.48082000e-02,  -1.27974000e-01,  -2.21465000e-01,
        -2.90566000e-01,  -3.63734000e-01,  -4.40965000e-01,
        -5.22261000e-01,  -6.27950000e-01,  -5.34444000e-01,
        -4.04358000e-01,  -3.55575000e-01,  -2.90528000e-01,
        -3.06778000e-01,  -3.47426000e-01,  -4.00267000e-01,
        -4.69368000e-01,  -5.91317000e-01,  -6.52285000e-01,
        -5.34389000e-01,  -3.51457000e-01,  -2.74217000e-01,
        -1.92911000e-01,  -1.40063000e-01,  -6.28251000e-02,
        -5.91213000e-03,   2.25504000e-02,  -5.89768000e-03,
        -5.87408000e-02,  -1.19709000e-01,  -2.01005000e-01,
        -2.94494000e-01,  -4.48963000e-01,  -5.05871000e-01,
        -5.46514000e-01,  -5.30247000e-01,  -4.77397000e-01,
        -4.77397000e-01,  -2.98530000e-01,  -3.31036000e-01,
        -4.40787000e-01,  -5.05825000e-01,  -5.50538000e-01,
        -6.31837000e-01,  -6.92808000e-01,  -6.39953000e-01,
        -5.58649000e-01,  -4.65149000e-01,  -3.71650000e-01,
        -3.02540000e-01,  -2.04974000e-01,  -1.39931000e-01,
        -8.30155000e-02,  -3.01627000e-02,  -5.76523000e-03,
         2.67599000e-02,   4.30345000e-02,   1.86539000e-02,
         4.30586000e-02,   1.46105000e-02,   3.08779000e-02,
         1.46298000e-02,   4.71574000e-02,   3.90393000e-02,
         1.05985000e-02,  -5.65686000e-03,   5.53260000e-02,
        -5.63759000e-03,   7.16078000e-02,  -9.68096000e-03,
         1.06563000e-02,  -9.64725000e-03,   1.47526000e-02,
        -5.56535000e-03,   1.88393000e-02,   2.69815000e-02,
        -4.21194000e-02,  -1.39676000e-01,  -2.04714000e-01,
        -2.45359000e-01,  -3.10395000e-01,  -3.83564000e-01,
        -4.48602000e-01,  -4.93312000e-01,  -5.66481000e-01,
        -6.47772000e-01,  -5.66466000e-01,  -4.89228000e-01,
        -4.32315000e-01,  -3.14422000e-01,  -2.65634000e-01,
        -3.26605000e-01,  -3.87578000e-01,  -4.48551000e-01,
        -5.01394000e-01,  -5.50170000e-01,  -5.94883000e-01,
        -6.68047000e-01,  -5.94869000e-01,  -4.52590000e-01,
        -3.87544000e-01,  -3.18434000e-01,  -2.28998000e-01,
        -2.16786000e-02,  -4.60616000e-02,  -6.23146000e-02,
        -8.26350000e-02,  -1.35478000e-01,  -1.88319000e-01,
        -2.41159000e-01,  -2.94003000e-01,  -3.59041000e-01,
        -4.11884000e-01,  -4.48467000e-01,  -4.93175000e-01,
        -5.09431000e-01,  -4.48448000e-01,  -3.79340000e-01,
        -3.22424000e-01,  -2.85834000e-01,  -2.65506000e-01,
        -2.81762000e-01,  -3.38670000e-01,  -3.95576000e-01,
        -4.60614000e-01,  -5.09392000e-01,  -5.70365000e-01,
        -6.35403000e-01,  -6.84179000e-01,  -6.11001000e-01,
        -5.45958000e-01,  -4.64655000e-01,  -3.79284000e-01,
        -3.22371000e-01,  -2.57326000e-01,  -1.31307000e-01,
        -6.21990000e-02,  -2.56064000e-02,  -9.33659000e-03,
         4.35113000e-02,   2.72656000e-02,   1.10199000e-02,
        -5.22579000e-03,  -1.33511000e-02,   1.91909000e-02,
         5.17185000e-02,   2.73355000e-02,   2.32873000e-02,
        -2.14209000e-02,  -1.08610000e-03,  -3.36016000e-02,
        -2.54643000e-02,   1.11259000e-02,  -5.11742000e-03,
        -4.16907000e-02,  -1.72933000e-02,   7.09937000e-03,
         3.04637000e-03,   2.74414000e-02,   2.33884000e-02,
         3.15281000e-02,  -9.82545000e-04,  -1.72331000e-02,
        -9.04014000e-02,  -1.79827000e-01,  -2.44866000e-01,
        -3.01774000e-01,  -3.70875000e-01,  -4.39976000e-01,
        -5.65989000e-01,  -7.04198000e-01,  -7.57042000e-01,
        -7.89555000e-01,  -7.32637000e-01,  -7.32637000e-01,
        -5.70033000e-01,  -4.92795000e-01,  -4.39944000e-01,
        -3.62699000e-01,  -3.99275000e-01,  -4.43985000e-01,
        -5.13086000e-01,  -5.82189000e-01,  -6.63488000e-01,
        -6.91941000e-01,  -6.18763000e-01,  -4.84614000e-01,
        -4.03306000e-01,  -2.24439000e-01,  -1.67526000e-01,
        -1.22809000e-01,  -7.80883000e-02,  -4.90791000e-03,
         2.35618000e-02,  -2.11392000e-02,   3.26070000e-03,
        -1.29850000e-02,   1.95474000e-02,  -1.29561000e-02,
         3.31609000e-03,  -1.29320000e-02,  -5.76354000e-02,
        -1.14536000e-01,  -1.42987000e-01,  -1.10462000e-01,
        -6.57438000e-02,  -4.54114000e-02,  -7.38498000e-02,
        -6.57101000e-02,  -6.16306000e-02,  -4.13030000e-02,
        -3.72284000e-02,  -6.38173000e-04,  -4.12765000e-02,
        -1.68815000e-02,  -3.71970000e-02,  -4.93801000e-02,
        -6.15656000e-02,  -2.90284000e-02,  -5.56294000e-04,
        -2.08743000e-02,  -5.39437000e-04,  -2.89875000e-02,
        -1.27201000e-02,  -2.89706000e-02,  -1.27008000e-02,
        -2.08189000e-02,  -5.33296000e-02,  -4.11176000e-02,
        -4.51754000e-02,  -2.89080000e-02,  -4.51297000e-03,
         3.64120000e-03,   1.17833000e-02,  -4.04578000e-04,
         1.99254000e-02,  -1.25853000e-02,   2.80820000e-02,
        -1.25564000e-02,   1.59182000e-02,  -3.27515000e-04,
        -2.93800000e-04,  -1.65323000e-02,   1.60025000e-02,
        -4.30827000e-03,  -5.71466000e-02,  -4.28900000e-03,
        -1.24143000e-02,   2.01109000e-02,   3.63855000e-02,
         3.86998000e-03,   2.42048000e-02,   1.20169000e-02,
         4.04794000e-02,   2.82915000e-02,   5.26914000e-02,
        -1.42084000e-04,   1.20675000e-02,   4.86625000e-02,
        -1.03553000e-04,   4.86842000e-02,   3.24312000e-02,
         4.46312000e-02,   2.83806000e-02,   3.65203000e-02,
         1.62048000e-02,   3.24746000e-02,   8.10359000e-03,
         2.84408000e-02,   2.84529000e-02])


    fig15top_t_raw = np.array(
      [ 1.01862,  1.04636,  1.05823,  1.06221,  1.07013,  1.07807,
        1.08598,  1.08997,  1.09985,  1.10979,  1.11571,  1.12365,
        1.14149,  1.15932,  1.16919,  1.17913,  1.18899,  1.19684,
        1.2047 ,  1.20672,  1.21058,  1.22052,  1.22055,  1.22458,
        1.22662,  1.23065,  1.23078,  1.23095,  1.23717,  1.23938,
        1.24354,  1.24761,  1.25735,  1.25922,  1.26111,  1.26297,
        1.26684,  1.27479,  1.27686,  1.28095,  1.28511,  1.28735,
        1.2915 ,  1.29735,  1.30115,  1.30297,  1.3068 ,  1.30866,
        1.31053,  1.31438,  1.31824,  1.32199,  1.32209,  1.32586,
        1.32976,  1.33171,  1.33364,  1.34151,  1.34744,  1.35736,
        1.36329,  1.37124,  1.37915,  1.38711,  1.395  ,  1.4069 ,
        1.41483,  1.43461,  1.44255,  1.44847,  1.4584 ,  1.49011,
        1.49799,  1.50786,  1.51375,  1.51767,  1.52556,  1.52956,
        1.52961,  1.53366,  1.53774,  1.53983,  1.54399,  1.54423,
        1.55047,  1.55268,  1.55878,  1.56463,  1.56649,  1.56821,
        1.56834,  1.57402,  1.57592,  1.5838 ,  1.58781,  1.59194,
        1.59421,  1.60038,  1.60812,  1.61198,  1.61384,  1.6157 ,
        1.61755,  1.6214 ,  1.62527,  1.62903,  1.62914,  1.63287,
        1.63476,  1.63865,  1.64656,  1.65053,  1.65062,  1.65276,
        1.65698,  1.65924,  1.66344,  1.66561,  1.6736 ,  1.6775 ,
        1.67928,  1.68318,  1.68502,  1.68691,  1.69279,  1.70274,
        1.70283,  1.70498,  1.7072 ,  1.71137,  1.71543,  1.71931,
        1.72118,  1.72305,  1.72688,  1.7287 ,  1.73251,  1.73435,
        1.73826,  1.74214,  1.74591,  1.74602,  1.74979,  1.75173,
        1.75564,  1.75755,  1.76345,  1.76942,  1.77732,  1.78528,
        1.79915,  1.813  ,  1.81897,  1.82488,  1.83282,  1.8467 ,
        1.8546 ,  1.86252,  1.87642,  1.91208,  1.91995,  1.92588,
        1.92982,  1.93967,  1.94357,  1.94753,  1.95151,  1.95352,
        1.95555,  1.95957,  1.96166,  1.96377,  1.96792,  1.97017,
        1.97436,  1.97654,  1.98057,  1.98844,  1.99021,  1.99232,
        1.996  ,  1.99787,  2.00374,  2.0117 ,  2.01377,  2.01595,
        2.02019,  2.0223 ,  2.02432,  2.03021,  2.03208,  2.03392,
        2.03579,  2.03964,  2.04349,  2.04532,  2.04718,  2.05104,
        2.05491,  2.05679,  2.05865,  2.06256,  2.06444,  2.0704 ,
        2.07445,  2.07657,  2.0787 ,  2.0808 ,  2.08295,  2.08718,
        2.08735,  2.09151,  2.09359,  2.10519,  2.10897,  2.11279,
        2.11867,  2.1227 ,  2.12274,  2.12682,  2.129  ,  2.13122,
        2.13537,  2.13936,  2.14324,  2.14705,  2.14894,  2.15082,
        2.15464,  2.1565 ,  2.15829,  2.1584 ,  2.16217,  2.16607,
        2.16796,  2.17183,  2.17372,  2.17763,  2.17956,  2.18348,
        2.18937,  2.19931,  2.20523,  2.21123,  2.22308,  2.22905,
        2.23497,  2.24093,  2.24882,  2.25677,  2.2627 ,  2.28055,
        2.29242,  2.29636,  2.30232,  2.30432,  2.31622,  2.3202 ,
        2.32612,  2.33997,  2.34786,  2.35178,  2.35569,  2.36552,
        2.36946,  2.37748,  2.37953,  2.38357,  2.38765,  2.38977,
        2.39202,  2.39435,  2.3966 ,  2.40267,  2.41053,  2.41242,
        2.4143 ,  2.41618,  2.41801,  2.42185,  2.42574,  2.43168,
        2.43765,  2.43776,  2.43993,  2.44218,  2.44828,  2.45414,
        2.45601,  2.45987,  2.46171,  2.46354,  2.46538,  2.46923,
        2.47113,  2.47687,  2.47698,  2.47874,  2.48457,  2.49042,
        2.49646,  2.50055,  2.50264,  2.50472,  2.50692,  2.509  ,
        2.50922,  2.51143,  2.51353,  2.52143,  2.52528,  2.52914,
        2.53099,  2.53286,  2.53674,  2.54459,  2.54664,  2.54879,
        2.55498,  2.55716,  2.56122,  2.5651 ,  2.56704,  2.57089,
        2.57279,  2.57662,  2.57845,  2.5803 ,  2.58222,  2.58613,
        2.59002,  2.59385,  2.59573,  2.59763,  2.60155,  2.60347,
        2.60738,  2.61533,  2.62524,  2.63119,  2.63512,  2.64701,
        2.651  ,  2.6609 ,  2.66484,  2.67475,  2.68465,  2.69456,
        2.70448,  2.71238,  2.71833,  2.72624,  2.7322 ,  2.74011,
        2.75003,  2.76189,  2.77177,  2.77768,  2.77962,  2.78556,
        2.78945,  2.79338,  2.80139,  2.80343,  2.80548,  2.8095 ,
        2.81158,  2.81574,  2.81593,  2.81831,  2.82251,  2.82857,
        2.83437,  2.83443,  2.84006,  2.84021,  2.84387,  2.84576,
        2.85165,  2.86158,  2.8617 ,  2.86392,  2.86816,  2.8722 ,
        2.87999,  2.88181,  2.88559,  2.88727,  2.88741,  2.89313,
        2.89503,  2.89888,  2.90074,  2.90462,  2.90851,  2.91237,
        2.92039,  2.92248,  2.92663,  2.93081,  2.93502,  2.93724,
        2.94134,  2.94914,  2.95296,  2.95485,  2.95867,  2.96257,
        2.97053,  2.97061,  2.97276,  2.97893,  2.98113,  2.98525,
        2.99296,  2.9931 ,  2.99873,  3.00041,  3.00054,  3.00615,
        3.00625,  3.01002,  3.01387,  3.01772,  3.01963,  3.02353,
        3.02742,  3.03134,  3.03527,  3.04121,  3.05115,  3.06104,
        3.067  ,  3.0749 ,  3.08088,  3.09077,  3.09476,  3.12841,
        3.13634,  3.14426,  3.15215,  3.16009,  3.17198,  3.17791,
        3.1819 ,  3.1878 ,  3.19965,  3.20356,  3.21144,  3.21533,
        3.22528,  3.22533,  3.22739,  3.23145,  3.23357,  3.23774,
        3.24193,  3.2442 ,  3.24643,  3.25256,  3.2584 ,  3.26027,
        3.2621 ,  3.2678 ,  3.26793,  3.27555,  3.28357,  3.28569,
        3.28788,  3.29617,  3.302  ,  3.30388,  3.30773,  3.30956,
        3.31334,  3.31515,  3.32096,  3.32283,  3.32289,  3.32473,
        3.33056,  3.33244,  3.33435,  3.34224,  3.34634,  3.34848,
        3.35461,  3.35487,  3.35711,  3.36127,  3.36532,  3.3712 ,
        3.37508,  3.37695,  3.37881,  3.38066,  3.38073,  3.38651,
        3.39447,  3.39849,  3.39864,  3.40284,  3.40506,  3.40716,
        3.41688,  3.41873,  3.4206 ,  3.42246,  3.42428,  3.43   ,
        3.43206,  3.43589,  3.4378 ,  3.43968,  3.44552,  3.44743,
        3.45129,  3.45134,  3.45521,  3.46315,  3.47705,  3.48694,
        3.49886,  3.50678,  3.52063,  3.53058,  3.54046,  3.54638,
        3.55627,  3.56019,  3.57605,  3.58401,  3.59192,  3.59788,
        3.60581,  3.60975,  3.61565,  3.62355,  3.6255 ,  3.63338,
        3.63732,  3.63926,  3.64518,  3.64919,  3.65123,  3.6533 ,
        3.65537,  3.65746,  3.66357,  3.66379,  3.66607,  3.67034,
        3.67645,  3.68227,  3.68409,  3.68592,  3.68971,  3.68979,
        3.69357,  3.69942,  3.70543,  3.71144,  3.71157,  3.71372,
        3.71591,  3.71796,  3.72382,  3.72944,  3.72959,  3.72968,
        3.73525,  3.73909,  3.74297,  3.74489,  3.74874,  3.7526 ,
        3.75451,  3.75843,  3.76233,  3.76625,  3.77814,  3.78209,
        3.796  ,  3.8039 ,  3.80985,  3.81774,  3.83163,  3.8376 ,
        3.84749,  3.85343,  3.86333,  3.87723,  3.8911 ,  3.90695,
        3.91883,  3.92673,  3.93266,  3.94658,  3.9743 ,  3.98417,
        3.98815,  4.00202,  4.00398,  4.01388,  4.01984,  4.03172,
        4.04757,  4.05154,  4.06741,  4.07929,  4.08322,  4.09117,
        4.09711,  4.111  ,  4.11495,  4.12288,  4.13078,  4.13672,
        4.14462,  4.1664 ,  4.18226,  4.19611,  4.20009,  4.20999,
        4.21395,  4.22784,  4.23776,  4.24767,  4.25163,  4.27342,
        4.28731,  4.3091 ,  4.33091,  4.34478,  4.35664,  4.37645,
        4.3824 ,  4.39229,  4.40417,  4.4121 ,  4.41607,  4.42398,
        4.43589,  4.44378,  4.46163,  4.47746])
    fig15top_w_raw = np.array(
      [ 0.        ,  0.        ,  0.00383142, -0.00383142,  0.        ,
       -0.0114943 , -0.00383142, -0.0153257 , -0.00383142, -0.0191571 ,
       -0.00766284, -0.0191571 , -0.0229885 , -0.0268199 , -0.00766284,
       -0.0268199 , -0.00383142,  0.0344828 ,  0.0689655 ,  0.045977  ,
        0.0996169 ,  0.0842912 ,  0.0689655 ,  0.0344828 ,  0.00383142,
       -0.0306513 , -0.0957854 , -0.183908  , -0.321839  , -0.440613  ,
       -0.536398  , -0.59387   , -0.509579  , -0.455939  , -0.40613   ,
       -0.348659  , -0.298851  , -0.314176  , -0.356322  , -0.425287  ,
       -0.524904  , -0.655172  , -0.747126  , -0.701149  , -0.62069   ,
       -0.54023   , -0.471264  , -0.409962  , -0.356322  , -0.298851  ,
       -0.245211  , -0.141762  , -0.191571  , -0.0957854 , -0.0613027 ,
       -0.045977  , -0.0191571 ,  0.00766284,  0.0114943 ,  0.00383142,
        0.0114943 , -0.00383142,  0.00383142, -0.0114943 ,  0.00383142,
       -0.00383142, -0.00383142,  0.0114943 ,  0.        ,  0.0114943 ,
        0.        , -0.00766284,  0.0153257 ,  0.0344828 ,  0.0613027 ,
        0.0804598 ,  0.0957854 ,  0.0766284 ,  0.0536398 ,  0.00766284,
       -0.0498084 , -0.10728   , -0.206897  , -0.329502  , -0.475096  ,
       -0.59387   , -0.670498  , -0.624521  , -0.563218  , -0.43295   ,
       -0.498084  , -0.363985  , -0.32567   , -0.302682  , -0.32567   ,
       -0.409962  , -0.555556  , -0.670498  , -0.578544  , -0.528736  ,
       -0.467433  , -0.40613   , -0.340996  , -0.283525  , -0.237548  ,
       -0.1341    , -0.191571  , -0.0727969 , -0.0268199 ,  0.00766284,
        0.0191571 ,  0.0114943 , -0.0306513 , -0.114943  , -0.241379  ,
       -0.383142  , -0.505747  , -0.597701  , -0.632184  , -0.601533  ,
       -0.501916  , -0.467433  , -0.398467  , -0.35249   , -0.318008  ,
       -0.344828  , -0.390805  , -0.475096  , -0.59387   , -0.701149  ,
       -0.747126  , -0.708812  , -0.651341  , -0.59387   , -0.528736  ,
       -0.448276  , -0.371648  , -0.298851  , -0.275862  , -0.233716  ,
       -0.1341    , -0.191571  , -0.0957854 , -0.0727969 , -0.045977  ,
       -0.0114943 ,  0.0114943 , -0.00383142,  0.00766284, -0.00766284,
       -0.0114943 , -0.00383142, -0.0153257 ,  0.        , -0.00766284,
       -0.0114943 ,  0.        ,  0.        , -0.0153257 , -0.0153257 ,
        0.0114943 ,  0.0153257 ,  0.0268199 ,  0.0536398 ,  0.0881226 ,
        0.0881226 ,  0.0766284 ,  0.0613027 ,  0.0383142 ,  0.00766284,
       -0.045977  , -0.111111  , -0.206897  , -0.340996  , -0.455939  ,
       -0.555556  , -0.590038  , -0.563218  , -0.45977   , -0.521073  ,
       -0.37931   , -0.32567   , -0.287356  , -0.306513  , -0.35249   ,
       -0.452107  , -0.590038  , -0.655172  , -0.678161  , -0.651341  ,
       -0.59387   , -0.524904  , -0.467433  , -0.409962  , -0.35249   ,
       -0.279693  , -0.214559  , -0.164751  , -0.118774  , -0.0651341 ,
       -0.00766284,  0.0191571 ,  0.0727969 ,  0.0613027 ,  0.0191571 ,
       -0.0536398 , -0.126437  , -0.187739  , -0.272031  , -0.409962  ,
       -0.494253  , -0.59387   , -0.643678  , -0.498084  , -0.409962  ,
       -0.337165  , -0.306513  , -0.337165  , -0.360153  , -0.417625  ,
       -0.521073  , -0.639847  , -0.735632  , -0.747126  , -0.704981  ,
       -0.632184  , -0.582375  , -0.532567  , -0.45977   , -0.402299  ,
       -0.302682  , -0.360153  , -0.264368  , -0.233716  , -0.183908  ,
       -0.137931  , -0.0957854 , -0.0651341 , -0.0421456 , -0.0191571 ,
        0.00383142, -0.0114943 ,  0.        , -0.0268199 , -0.0114943 ,
       -0.0268199 , -0.0114943 , -0.0229885 , -0.00383142, -0.0191571 ,
       -0.0114943 , -0.0229885 , -0.0153257 , -0.00383142, -0.0114943 ,
       -0.0229885 , -0.0268199 , -0.0344828 , -0.0268199 , -0.0153257 ,
        0.        ,  0.0229885 ,  0.0498084 ,  0.0842912 ,  0.0957854 ,
        0.0498084 ,  0.0114943 , -0.0268199 , -0.0842912 , -0.157088  ,
       -0.291188  , -0.467433  , -0.601533  , -0.666667  , -0.636015  ,
       -0.590038  , -0.54023   , -0.48659   , -0.409962  , -0.348659  ,
       -0.314176  , -0.310345  , -0.32567   , -0.37931   , -0.478927  ,
       -0.613027  , -0.689655  , -0.651341  , -0.59387   , -0.54023   ,
       -0.471264  , -0.394636  , -0.321839  , -0.268199  , -0.226054  ,
       -0.126437  , -0.180077  , -0.0689655 , -0.0114943 ,  0.0344828 ,
       -0.0153257 , -0.0766284 , -0.1341    , -0.183908  , -0.291188  ,
       -0.340996  , -0.455939  , -0.570881  , -0.632184  , -0.616858  ,
       -0.563218  , -0.509579  , -0.444444  , -0.386973  , -0.344828  ,
       -0.306513  , -0.344828  , -0.429119  , -0.551724  , -0.655172  ,
       -0.704981  , -0.662835  , -0.639847  , -0.586207  , -0.544061  ,
       -0.478927  , -0.402299  , -0.337165  , -0.302682  , -0.279693  ,
       -0.241379  , -0.176245  , -0.122605  , -0.0842912 , -0.0613027 ,
       -0.0306513 , -0.00766284, -0.0191571 , -0.0191571 , -0.0229885 ,
       -0.00766284, -0.0114943 , -0.0229885 , -0.0229885 , -0.0114943 ,
       -0.0153257 , -0.0114943 , -0.0153257 , -0.0229885 , -0.00766284,
       -0.0153257 , -0.00766284, -0.0153257 , -0.00766284, -0.0153257 ,
        0.        ,  0.0114943 ,  0.0268199 ,  0.045977  ,  0.0498084 ,
        0.0842912 ,  0.0996169 ,  0.0574713 ,  0.0268199 , -0.00383142,
       -0.0344828 , -0.0881226 , -0.183908  , -0.283525  , -0.482759  ,
       -0.605364  , -0.662835  , -0.590038  , -0.62069   , -0.463602  ,
       -0.536398  , -0.386973  , -0.340996  , -0.314176  , -0.32567   ,
       -0.386973  , -0.505747  , -0.64751   , -0.685824  , -0.62069   ,
       -0.536398  , -0.448276  , -0.295019  , -0.363985  , -0.252874  ,
       -0.210728  , -0.157088  , -0.0957854 , -0.0536398 , -0.0153257 ,
        0.0344828 , -0.0114943 , -0.0689655 , -0.16092   , -0.272031  ,
       -0.398467  , -0.521073  , -0.586207  , -0.524904  , -0.452107  ,
       -0.40613   , -0.337165  , -0.306513  , -0.32567   , -0.363985  ,
       -0.448276  , -0.563218  , -0.67433   , -0.754789  , -0.64751   ,
       -0.716475  , -0.559387  , -0.40613   , -0.475096  , -0.306513  ,
       -0.356322  , -0.260536  , -0.203065  , -0.145594  , -0.10728   ,
       -0.0766284 , -0.0421456 , -0.0191571 , -0.00383142, -0.00383142,
       -0.0229885 , -0.0153257 , -0.0229885 , -0.0114943 , -0.0268199 ,
       -0.0229885 , -0.0344828 , -0.0229885 , -0.0268199 , -0.0229885 ,
       -0.00766284, -0.0153257 , -0.0153257 , -0.00766284, -0.0229885 ,
       -0.00383142,  0.0153257 ,  0.0421456 ,  0.0651341 ,  0.0996169 ,
        0.0766284 ,  0.0536398 ,  0.0153257 , -0.0344828 , -0.10728   ,
       -0.210728  , -0.32567   , -0.471264  , -0.597701  , -0.693487  ,
       -0.639847  , -0.582375  , -0.509579  , -0.386973  , -0.452107  ,
       -0.298851  , -0.344828  , -0.417625  , -0.521073  , -0.704981  ,
       -0.64751   , -0.597701  , -0.54023   , -0.467433  , -0.371648  ,
       -0.287356  , -0.222222  , -0.164751  , -0.195402  , -0.122605  ,
       -0.0651341 , -0.0153257 ,  0.0191571 ,  0.0383142 , -0.0306513 ,
       -0.111111  , -0.206897  , -0.337165  , -0.467433  , -0.570881  ,
       -0.613027  , -0.582375  , -0.54023   , -0.482759  , -0.425287  ,
       -0.356322  , -0.390805  , -0.310345  , -0.329502  , -0.360153  ,
       -0.43295   , -0.555556  , -0.67433   , -0.735632  , -0.639847  ,
       -0.574713  , -0.521073  , -0.45977   , -0.375479  , -0.264368  ,
       -0.306513  , -0.237548  , -0.199234  , -0.149425  , -0.0996169 ,
       -0.0613027 , -0.0114943 , -0.0383142 ,  0.0114943 ,  0.        ,
       -0.0153257 , -0.00766284, -0.0229885 , -0.0229885 , -0.0153257 ,
       -0.0383142 , -0.0229885 , -0.0114943 , -0.00766284,  0.0153257 ,
        0.00766284, -0.00766284,  0.        , -0.0114943 , -0.0114943 ,
        0.        ,  0.0191571 ,  0.0306513 ,  0.045977  ,  0.0689655 ,
        0.0804598 ,  0.0996169 ,  0.114943  ,  0.0881226 ,  0.0574713 ,
        0.0153257 , -0.0306513 , -0.0881226 , -0.172414  , -0.283525  ,
       -0.43295   , -0.586207  , -0.67433   , -0.609195  , -0.528736  ,
       -0.452107  , -0.367816  , -0.40613   , -0.314176  , -0.268199  ,
       -0.298851  , -0.337165  , -0.398467  , -0.48659   , -0.59387   ,
       -0.624521  , -0.586207  , -0.421456  , -0.498084  , -0.544061  ,
       -0.35249   , -0.291188  , -0.252874  , -0.218391  , -0.164751  ,
       -0.111111  , -0.0766284 , -0.0536398 , -0.0229885 , -0.00383142,
       -0.00766284,  0.        , -0.0191571 , -0.00766284, -0.0114943 ,
        0.00383142, -0.00766284, -0.0229885 , -0.0153257 , -0.0114943 ,
       -0.0114943 , -0.0268199 , -0.0268199 , -0.0268199 , -0.0268199 ,
       -0.0114943 , -0.00766284, -0.0344828 , -0.0268199 , -0.00766284,
       -0.0191571 , -0.0191571 , -0.00766284, -0.00766284, -0.0153257 ,
       -0.0114943 , -0.0114943 , -0.0191571 , -0.0268199 , -0.0268199 ,
       -0.0114943 , -0.0229885 , -0.0229885 , -0.0344828 , -0.0268199 ,
       -0.0306513 , -0.0191571 , -0.0153257 , -0.00383142,  0.        ,
       -0.00383142,  0.00383142, -0.00766284, -0.00383142, -0.00383142,
       -0.0153257 , -0.0229885 , -0.0229885 , -0.0229885 , -0.0229885 ,
       -0.0344828 , -0.0344828 , -0.0421456 , -0.045977  , -0.0306513 ,
       -0.0306513 , -0.0383142 , -0.0306513 , -0.0268199 , -0.0268199 ,
       -0.0344828 , -0.0229885 , -0.0383142 , -0.0229885 , -0.0306513 ,
       -0.0229885 ])

    fig15bot_t_raw = np.array(
      [ 0.817702,  0.825628,  0.835535,  0.839498,  0.849406,  0.853369,
        0.863276,  0.869221,  0.879128,  0.891017,  0.896962,  0.910832,
        0.918758,  0.928666,  0.936592,  0.940555,  0.944518,  0.952444,
        0.958389,  0.968296,  0.976222,  0.980185,  0.990093,  0.996037,
        1.00198 ,  1.00594 ,  1.01189 ,  1.01387 ,  1.01783 ,  1.0218  ,
        1.02576 ,  1.02972 ,  1.0317  ,  1.03567 ,  1.03765 ,  1.04161 ,
        1.04557 ,  1.04954 ,  1.0535  ,  1.05548 ,  1.06341 ,  1.06737 ,
        1.07133 ,  1.0753  ,  1.07728 ,  1.07926 ,  1.08124 ,  1.0852  ,
        1.09115 ,  1.09511 ,  1.09709 ,  1.10106 ,  1.10304 ,  1.10502 ,
        1.10898 ,  1.12087 ,  1.13276 ,  1.13871 ,  1.14663 ,  1.1605  ,
        1.16645 ,  1.17041 ,  1.17834 ,  1.18626 ,  1.19815 ,  1.20608 ,
        1.21797 ,  1.22985 ,  1.24174 ,  1.25165 ,  1.26552 ,  1.27345 ,
        1.28336 ,  1.29326 ,  1.30119 ,  1.30713 ,  1.31506 ,  1.31902 ,
        1.321   ,  1.32497 ,  1.32695 ,  1.33091 ,  1.33686 ,  1.33884 ,
        1.3428  ,  1.34478 ,  1.35073 ,  1.35667 ,  1.35865 ,  1.36063 ,
        1.36262 ,  1.36658 ,  1.37252 ,  1.37649 ,  1.37847 ,  1.38243 ,
        1.38639 ,  1.38838 ,  1.38838 ,  1.39432 ,  1.39828 ,  1.40225 ,
        1.40423 ,  1.40621 ,  1.41017 ,  1.41414 ,  1.41612 ,  1.42008 ,
        1.42206 ,  1.42602 ,  1.43197 ,  1.43395 ,  1.43989 ,  1.44584 ,
        1.44782 ,  1.4498  ,  1.45178 ,  1.45377 ,  1.45773 ,  1.45971 ,
        1.46367 ,  1.46764 ,  1.46962 ,  1.4716  ,  1.47358 ,  1.47754 ,
        1.47952 ,  1.48349 ,  1.48745 ,  1.48943 ,  1.49736 ,  1.50132 ,
        1.5033  ,  1.50727 ,  1.51321 ,  1.51519 ,  1.51717 ,  1.52114 ,
        1.52312 ,  1.5251  ,  1.52906 ,  1.53303 ,  1.53501 ,  1.54293 ,
        1.55284 ,  1.56275 ,  1.56869 ,  1.5786  ,  1.58454 ,  1.58851 ,
        1.59247 ,  1.59842 ,  1.6103  ,  1.61427 ,  1.62219 ,  1.62814 ,
        1.63805 ,  1.65192 ,  1.66182 ,  1.66975 ,  1.67569 ,  1.67768 ,
        1.68164 ,  1.6856  ,  1.69353 ,  1.69551 ,  1.70344 ,  1.71136 ,
        1.71731 ,  1.71929 ,  1.72127 ,  1.72523 ,  1.73118 ,  1.7391  ,
        1.74307 ,  1.74703 ,  1.74901 ,  1.75297 ,  1.75495 ,  1.75694 ,
        1.7609  ,  1.76486 ,  1.76882 ,  1.77279 ,  1.77477 ,  1.7827  ,
        1.78468 ,  1.78864 ,  1.7926  ,  1.79458 ,  1.80449 ,  1.80647 ,
        1.80845 ,  1.81242 ,  1.8144  ,  1.82034 ,  1.82431 ,  1.82629 ,
        1.82827 ,  1.83223 ,  1.8362  ,  1.83818 ,  1.84214 ,  1.84809 ,
        1.85007 ,  1.85205 ,  1.85799 ,  1.86196 ,  1.86394 ,  1.8679  ,
        1.86988 ,  1.87384 ,  1.87583 ,  1.87979 ,  1.88177 ,  1.88375 ,
        1.8897  ,  1.89366 ,  1.89762 ,  1.90159 ,  1.90753 ,  1.91149 ,
        1.91347 ,  1.91546 ,  1.91942 ,  1.92536 ,  1.92735 ,  1.93725 ,
        1.93923 ,  1.94122 ,  1.9432  ,  1.94518 ,  1.94914 ,  1.95112 ,
        1.95509 ,  1.95707 ,  1.96103 ,  1.96499 ,  1.96698 ,  1.97292 ,
        1.97886 ,  1.99075 ,  1.9967  ,  2.00462 ,  2.01057 ,  2.01453 ,
        2.02642 ,  2.03237 ,  2.03633 ,  2.04227 ,  2.05614 ,  2.07001 ,
        2.07596 ,  2.08587 ,  2.1037  ,  2.11757 ,  2.12946 ,  2.1354  ,
        2.14333 ,  2.14927 ,  2.15918 ,  2.16314 ,  2.16711 ,  2.16909 ,
        2.17305 ,  2.17702 ,  2.18098 ,  2.18296 ,  2.18494 ,  2.18692 ,
        2.19089 ,  2.19485 ,  2.19683 ,  2.20079 ,  2.20476 ,  2.20674 ,
        2.20872 ,  2.21268 ,  2.22061 ,  2.22457 ,  2.22655 ,  2.23052 ,
        2.23646 ,  2.24042 ,  2.24439 ,  2.24637 ,  2.24835 ,  2.25231 ,
        2.25628 ,  2.26024 ,  2.26222 ,  2.2642  ,  2.26816 ,  2.27411 ,
        2.27807 ,  2.28005 ,  2.28402 ,  2.286   ,  2.29591 ,  2.29987 ,
        2.30185 ,  2.30383 ,  2.30779 ,  2.31176 ,  2.3177  ,  2.32167 ,
        2.32365 ,  2.32563 ,  2.33554 ,  2.33752 ,  2.34148 ,  2.34346 ,
        2.34544 ,  2.34941 ,  2.35337 ,  2.3613  ,  2.36526 ,  2.36922 ,
        2.3712  ,  2.37318 ,  2.37517 ,  2.37913 ,  2.38507 ,  2.38705 ,
        2.39102 ,  2.39696 ,  2.40291 ,  2.41083 ,  2.4148  ,  2.42669 ,
        2.43263 ,  2.43857 ,  2.45046 ,  2.4683  ,  2.48019 ,  2.48613 ,
        2.5     ,  2.50396 ,  2.51387 ,  2.5218  ,  2.52972 ,  2.54161 ,
        2.55152 ,  2.56539 ,  2.5753  ,  2.58322 ,  2.58917 ,  2.59313 ,
        2.59709 ,  2.60304 ,  2.60502 ,  2.607   ,  2.60898 ,  2.61295 ,
        2.61493 ,  2.61691 ,  2.62087 ,  2.62682 ,  2.6288  ,  2.63078 ,
        2.63474 ,  2.63672 ,  2.64267 ,  2.64663 ,  2.6506  ,  2.65258 ,
        2.65852 ,  2.66248 ,  2.66645 ,  2.67239 ,  2.67635 ,  2.67834 ,
        2.6823  ,  2.68428 ,  2.68626 ,  2.69023 ,  2.69419 ,  2.70013 ,
        2.70211 ,  2.7041  ,  2.70608 ,  2.71202 ,  2.71797 ,  2.71995 ,
        2.72193 ,  2.72787 ,  2.72986 ,  2.73382 ,  2.74174 ,  2.74571 ,
        2.74571 ,  2.74967 ,  2.7576  ,  2.76354 ,  2.7675  ,  2.7675  ,
        2.77147 ,  2.77345 ,  2.77741 ,  2.78336 ,  2.78732 ,  2.7893  ,
        2.79128 ,  2.79525 ,  2.79723 ,  2.80119 ,  2.80317 ,  2.80515 ,
        2.81902 ,  2.82893 ,  2.83289 ,  2.8428  ,  2.85469 ,  2.8646  ,
        2.87054 ,  2.87252 ,  2.87847 ,  2.88243 ,  2.88838 ,  2.89432 ,
        2.89828 ,  2.90423 ,  2.90819 ,  2.91414 ,  2.92206 ,  2.92404 ,
        2.9399  ,  2.94584 ,  2.95178 ,  2.95971 ,  2.96962 ,  2.97358 ,
        2.98745 ,  2.99538 ,  3.00132 ,  3.00529 ,  3.01321 ,  3.01519 ,
        3.01916 ,  3.02114 ,  3.0251  ,  3.02708 ,  3.03104 ,  3.03501 ,
        3.03699 ,  3.03897 ,  3.04293 ,  3.05086 ,  3.05284 ,  3.0568  ,
        3.05879 ,  3.06077 ,  3.06671 ,  3.07067 ,  3.07266 ,  3.07464 ,
        3.08058 ,  3.08653 ,  3.08851 ,  3.09643 ,  3.09842 ,  3.1004  ,
        3.10238 ,  3.10436 ,  3.10832 ,  3.1103  ,  3.11625 ,  3.12418 ,
        3.12814 ,  3.13012 ,  3.13408 ,  3.13606 ,  3.14399 ,  3.14795 ,
        3.14994 ,  3.1539  ,  3.15588 ,  3.16579 ,  3.16777 ,  3.16975 ,
        3.17768 ,  3.18362 ,  3.18758 ,  3.19155 ,  3.19749 ,  3.19749 ,
        3.20542 ,  3.20938 ,  3.21136 ,  3.21334 ,  3.21731 ,  3.21929 ,
        3.22325 ,  3.22523 ,  3.22721 ,  3.2292  ,  3.2391  ,  3.24505 ,
        3.25099 ,  3.26486 ,  3.27081 ,  3.27873 ,  3.29062 ,  3.30053 ,
        3.30449 ,  3.30846 ,  3.31836 ,  3.32233 ,  3.32827 ,  3.3362  ,
        3.34214 ,  3.35799 ,  3.37979 ,  3.39564 ,  3.40555 ,  3.41149 ,
        3.41942 ,  3.42933 ,  3.43527 ,  3.43924 ,  3.4432  ,  3.44716 ,
        3.44914 ,  3.45112 ,  3.45509 ,  3.45905 ,  3.46103 ,  3.46301 ,
        3.47094 ,  3.4749  ,  3.47688 ,  3.47887 ,  3.48283 ,  3.48481 ,
        3.49075 ,  3.49472 ,  3.50066 ,  3.50661 ,  3.50859 ,  3.51453 ,
        3.52048 ,  3.52246 ,  3.52642 ,  3.53038 ,  3.53435 ,  3.53831 ,
        3.54029 ,  3.54227 ,  3.54624 ,  3.5502  ,  3.55614 ,  3.55813 ,
        3.56605 ,  3.572   ,  3.57992 ,  3.58587 ,  3.59181 ,  3.59776 ,
        3.6037  ,  3.60766 ,  3.61955 ,  3.63144 ,  3.64729 ,  3.6572  ,
        3.66513 ,  3.68296 ,  3.69485 ,  3.70278 ,  3.71268 ,  3.72655 ,
        3.74042 ,  3.77015 ,  3.77213 ,  3.77609 ,  3.786   ,  3.79392 ,
        3.80185 ,  3.80581 ,  3.81176 ,  3.82563 ,  3.83554 ,  3.84941 ,
        3.87913 ,  3.88706 ,  3.90885 ,  3.91678 ,  3.93857 ,  3.9465  ,
        3.95641 ,  3.97226 ,  3.98019 ,  3.99208 ,  4.00991 ,  4.03567 ,
        4.04558 ,  4.06143 ,  4.06539 ,  4.07332 ,  4.08124 ,  4.08719 ,
        4.10106 ,  4.11097 ,  4.11889 ,  4.12682 ,  4.13276 ,  4.13673 ,
        4.14663 ,  4.16447 ,  4.17636 ,  4.1823  ,  4.2041  ,  4.20806 ,
        4.23184 ,  4.24967 ,  4.2576  ,  4.27543 ,  4.28138 ])
    fig15bot_w_raw = np.array(
      [ 0.00387597, -0.0193798 , -0.00775194, -0.0193798 , -0.00387597,
       -0.0155039 ,  0.        , -0.0155039 , -0.0387597 , -0.0232558 ,
       -0.0503876 , -0.0310078 , -0.0310078 , -0.0155039 , -0.0542636 ,
       -0.0426357 , -0.00387597, -0.00775194, -0.0348837 , -0.0310078 ,
       -0.0465116 ,  0.0116279 , -0.0116279 ,  0.0310078 ,  0.0271318 ,
        0.0116279 , -0.0542636 , -0.104651  , -0.155039  , -0.205426  ,
       -0.244186  , -0.302326  , -0.352713  , -0.426357  , -0.391473  ,
       -0.453488  , -0.372093  , -0.306202  , -0.27907   , -0.244186  ,
       -0.182171  , -0.228682  , -0.317829  , -0.364341  , -0.399225  ,
       -0.430233  , -0.488372  , -0.523256  , -0.484496  , -0.29845   ,
       -0.232558  , -0.170543  , -0.0891473 , -0.0348837 ,  0.00387597,
        0.0426357 ,  0.0232558 ,  0.0310078 ,  0.0193798 ,  0.0387597 ,
        0.0232558 ,  0.        ,  0.0155039 ,  0.00775194,  0.0271318 ,
        0.0116279 ,  0.00775194,  0.0232558 ,  0.0155039 ,  0.        ,
       -0.00387597,  0.0155039 , -0.0116279 , -0.0271318 ,  0.        ,
        0.0193798 ,  0.0116279 , -0.0348837 , -0.100775  , -0.162791  ,
       -0.186047  , -0.217054  , -0.275194  , -0.348837  , -0.406977  ,
       -0.465116  , -0.546512  , -0.5       , -0.434109  , -0.352713  ,
       -0.27907   , -0.224806  , -0.213178  , -0.236434  , -0.267442  ,
       -0.294574  , -0.379845  , -0.496124  , -0.496124  , -0.569767  ,
       -0.492248  , -0.437984  , -0.317829  , -0.25969   , -0.170543  ,
       -0.100775  , -0.0658915 , -0.0232558 ,  0.0348837 ,  0.0697674 ,
        0.0348837 , -0.108527  , -0.186047  , -0.24031   , -0.27907   ,
       -0.317829  , -0.360465  , -0.399225  , -0.449612  , -0.503876  ,
       -0.593023  , -0.542636  , -0.434109  , -0.496124  , -0.379845  ,
       -0.313953  , -0.267442  , -0.232558  , -0.263566  , -0.310078  ,
       -0.321705  , -0.496124  , -0.600775  , -0.639535  , -0.534884  ,
       -0.604651  , -0.44186   , -0.352713  , -0.27907   , -0.209302  ,
       -0.139535  , -0.0891473 , -0.0155039 ,  0.00387597, -0.0116279 ,
        0.0116279 ,  0.0155039 , -0.0116279 ,  0.00387597,  0.0542636 ,
        0.0155039 ,  0.0348837 ,  0.0387597 ,  0.0581395 ,  0.        ,
        0.0271318 ,  0.00387597,  0.0426357 ,  0.0116279 , -0.0155039 ,
       -0.0116279 ,  0.00775194,  0.0193798 , -0.0155039 , -0.0193798 ,
        0.00775194, -0.00387597, -0.00775194, -0.0271318 ,  0.        ,
        0.0193798 ,  0.0465116 ,  0.0232558 ,  0.0155039 , -0.0426357 ,
       -0.0891473 , -0.116279  , -0.151163  , -0.20155   , -0.263566  ,
       -0.310078  , -0.372093  , -0.445736  , -0.511628  , -0.531008  ,
       -0.468992  , -0.325581  , -0.290698  , -0.244186  , -0.20155   ,
       -0.255814  , -0.302326  , -0.352713  , -0.410853  , -0.476744  ,
       -0.565891  , -0.48062   , -0.364341  , -0.418605  , -0.27907   ,
       -0.182171  , -0.116279  , -0.0503876 ,  0.0271318 ,  0.0465116 ,
        0.0620155 ,  0.0387597 , -0.0581395 , -0.124031  , -0.158915  ,
       -0.197674  , -0.248062  , -0.317829  , -0.383721  , -0.44186   ,
       -0.488372  , -0.46124   , -0.422481  , -0.313953  , -0.255814  ,
       -0.162791  , -0.213178  , -0.27907   , -0.329457  , -0.368217  ,
       -0.51938   , -0.585271  , -0.53876   , -0.484496  , -0.44186   ,
       -0.368217  , -0.271318  , -0.193798  , -0.124031  , -0.0736434 ,
       -0.0193798 ,  0.00775194,  0.0193798 ,  0.0310078 ,  0.0620155 ,
        0.0271318 ,  0.0542636 , -0.0232558 , -0.0155039 , -0.0155039 ,
        0.00775194,  0.        , -0.0232558 , -0.00775194,  0.0116279 ,
        0.        ,  0.        ,  0.0116279 , -0.00387597,  0.0310078 ,
        0.0155039 ,  0.0310078 ,  0.0116279 ,  0.0503876 ,  0.0736434 ,
        0.0426357 ,  0.0155039 , -0.0930233 , -0.155039  , -0.193798  ,
       -0.217054  , -0.255814  , -0.306202  , -0.360465  , -0.403101  ,
       -0.457364  , -0.534884  , -0.589147  , -0.542636  , -0.472868  ,
       -0.426357  , -0.337209  , -0.282946  , -0.209302  , -0.271318  ,
       -0.321705  , -0.360465  , -0.430233  , -0.507752  , -0.488372  ,
       -0.445736  , -0.360465  , -0.313953  , -0.217054  , -0.131783  ,
       -0.0503876 , -0.00775194,  0.0271318 ,  0.0620155 ,  0.0348837 ,
       -0.0193798 , -0.0968992 , -0.139535  , -0.178295  , -0.267442  ,
       -0.387597  , -0.434109  , -0.476744  , -0.531008  , -0.406977  ,
       -0.317829  , -0.275194  , -0.224806  , -0.25969   , -0.290698  ,
       -0.325581  , -0.360465  , -0.414729  , -0.496124  , -0.577519  ,
       -0.434109  , -0.372093  , -0.317829  , -0.275194  , -0.224806  ,
       -0.0968992 , -0.0465116 , -0.0155039 ,  0.00387597,  0.0503876 ,
        0.0271318 ,  0.0736434 ,  0.0465116 ,  0.0155039 ,  0.        ,
        0.0271318 ,  0.0542636 ,  0.00387597,  0.0310078 , -0.0271318 ,
        0.0193798 , -0.0116279 ,  0.0193798 , -0.0465116 ,  0.00775194,
       -0.0310078 ,  0.0193798 , -0.0271318 ,  0.        , -0.0155039 ,
       -0.00775194, -0.0310078 , -0.0930233 , -0.182171  , -0.193798  ,
       -0.228682  , -0.27907   , -0.337209  , -0.410853  , -0.468992  ,
       -0.550388  , -0.600775  , -0.554264  , -0.48062   , -0.337209  ,
       -0.282946  , -0.236434  , -0.209302  , -0.189922  , -0.282946  ,
       -0.341085  , -0.468992  , -0.593023  , -0.658915  , -0.488372  ,
       -0.410853  , -0.294574  , -0.24031   , -0.166667  , -0.112403  ,
       -0.0852713 , -0.0503876 ,  0.0542636 ,  0.0116279 , -0.0581395 ,
       -0.162791  , -0.205426  , -0.232558  , -0.302326  , -0.387597  ,
       -0.44186   , -0.492248  , -0.550388  , -0.488372  , -0.282946  ,
       -0.282946  , -0.244186  , -0.205426  , -0.282946  , -0.391473  ,
       -0.391473  , -0.46124   , -0.523256  , -0.577519  , -0.515504  ,
       -0.418605  , -0.317829  , -0.248062  , -0.197674  , -0.158915  ,
       -0.104651  , -0.0581395 , -0.0155039 ,  0.0271318 ,  0.00387597,
        0.0813953 ,  0.0116279 , -0.00775194, -0.0271318 ,  0.00775194,
       -0.0193798 ,  0.0465116 ,  0.0155039 ,  0.0310078 , -0.00387597,
       -0.0232558 , -0.00775194, -0.0155039 , -0.0658915 , -0.0348837 ,
       -0.00775194, -0.0348837 , -0.0310078 ,  0.0193798 , -0.0232558 ,
       -0.00387597, -0.0271318 , -0.00387597, -0.0193798 ,  0.0116279 ,
        0.0348837 , -0.0271318 , -0.0930233 , -0.158915  , -0.213178  ,
       -0.255814  , -0.275194  , -0.310078  , -0.352713  , -0.410853  ,
       -0.476744  , -0.523256  , -0.503876  , -0.437984  , -0.356589  ,
       -0.294574  , -0.25969   , -0.228682  , -0.286822  , -0.24031   ,
       -0.337209  , -0.360465  , -0.430233  , -0.51938   , -0.496124  ,
       -0.406977  , -0.286822  , -0.224806  , -0.166667  , -0.0968992 ,
       -0.0426357 , -0.00387597,  0.0775194 ,  0.0271318 , -0.0581395 ,
       -0.131783  , -0.162791  , -0.213178  , -0.321705  , -0.368217  ,
       -0.445736  , -0.531008  , -0.472868  , -0.383721  , -0.313953  ,
       -0.24031   , -0.275194  , -0.325581  , -0.356589  , -0.465116  ,
       -0.465116  , -0.647287  , -0.608527  , -0.527132  , -0.437984  ,
       -0.325581  , -0.24031   , -0.158915  , -0.100775  , -0.0542636 ,
       -0.0155039 ,  0.00775194,  0.0542636 ,  0.0232558 ,  0.        ,
        0.        , -0.0155039 ,  0.        ,  0.0581395 ,  0.0232558 ,
        0.00387597,  0.        ,  0.0310078 ,  0.0465116 ,  0.0155039 ,
        0.0465116 ,  0.0620155 , -0.00387597,  0.0155039 ,  0.0348837 ,
        0.0658915 ,  0.0542636 ,  0.0658915 ,  0.0503876 , -0.0658915 ,
       -0.112403  , -0.166667  , -0.232558  , -0.275194  , -0.321705  ,
       -0.368217  , -0.44186   , -0.527132  , -0.581395  , -0.515504  ,
       -0.430233  , -0.352713  , -0.27907   , -0.224806  , -0.197674  ,
       -0.248062  , -0.302326  , -0.368217  , -0.44186   , -0.488372  ,
       -0.44186   , -0.360465  , -0.209302  , -0.158915  , -0.112403  ,
       -0.0542636 , -0.0232558 ,  0.0155039 ,  0.0465116 ,  0.0813953 ,
        0.0658915 ,  0.0310078 ,  0.0542636 ,  0.0387597 ,  0.0387597 ,
        0.0271318 , -0.0193798 , -0.00387597,  0.0116279 , -0.0232558 ,
        0.        , -0.0232558 , -0.0232558 , -0.00387597, -0.0271318 ,
       -0.00387597, -0.0271318 , -0.0155039 , -0.0310078 , -0.0426357 ,
       -0.0193798 , -0.0232558 ,  0.0155039 ,  0.0271318 ,  0.0116279 ,
        0.0155039 ,  0.00775194, -0.0465116 , -0.0155039 , -0.0465116 ,
       -0.0155039 , -0.00387597, -0.00387597,  0.00775194, -0.0310078 ,
        0.        ,  0.00387597,  0.00387597,  0.00387597, -0.0232558 ,
       -0.00775194, -0.0116279 ,  0.0116279 , -0.0310078 , -0.0155039 ,
       -0.0387597 , -0.0155039 , -0.0310078 ,  0.00387597, -0.0232558 ,
        0.00387597,  0.0232558 , -0.0155039 , -0.00387597, -0.0310078 ,
       -0.00387597, -0.0348837 , -0.0193798 , -0.0271318 ,  0.00387597,
       -0.0387597 ,  0.00387597,  0.00775194, -0.00387597, -0.0193798 ,
        0.        , -0.00775194])



    fig19b_bridge_to_emb_x_raw = np.array(
      [ -4.92072000e+00,  -4.90682000e+00,  -4.71917000e+00,
        -4.70637000e+00,  -4.58438000e+00,  -4.50665000e+00,
        -4.38320000e+00,  -4.37332000e+00,  -4.23944000e+00,
        -4.18165000e+00,  -4.10538000e+00,  -3.98010000e+00,
        -3.97169000e+00,  -3.84549000e+00,  -3.77142000e+00,
        -3.64413000e+00,  -3.63828000e+00,  -3.50952000e+00,
        -3.43874000e+00,  -3.30816000e+00,  -3.17208000e+00,
        -3.04040000e+00,  -3.03821000e+00,  -2.83739000e+00,
        -2.70351000e+00,  -2.63968000e+00,  -2.50251000e+00,
        -2.43905000e+00,  -2.23787000e+00,  -2.23494000e+00,
        -2.10308000e+00,  -1.96755000e+00,  -1.90135000e+00,
        -1.76674000e+00,  -1.69998000e+00,  -1.50008000e+00,
        -1.36694000e+00,  -1.23397000e+00,  -1.16722000e+00,
        -9.68049000e-01,  -9.00379000e-01,  -8.35635000e-01,
        -6.36282000e-01,  -5.67149000e-01,  -4.34735000e-01,
        -3.70357000e-01,  -3.02138000e-01,  -2.36114000e-01,
        -3.67613000e-02,   2.96285000e-02,   9.56527000e-02,
         4.96918000e-01,   7.63941000e-01,   1.16466000e+00,
         1.36383000e+00,   1.63176000e+00,   2.03212000e+00,
         2.09869000e+00,   2.63219000e+00,   2.63237000e+00,
         3.16641000e+00,   3.23207000e+00,   3.49964000e+00,
         3.76539000e+00,   3.96620000e+00,   4.29888000e+00,
         4.56645000e+00,   5.09922000e+00,   5.23346000e+00,
         5.76623000e+00,   5.83371000e+00,   6.43415000e+00,
         6.76683000e+00,   7.10134000e+00,   7.63465000e+00,
         7.70178000e+00,   8.30221000e+00,   8.83552000e+00,
         8.90283000e+00,   9.50345000e+00,   9.90307000e+00,
         1.01706000e+01,   1.11049000e+01,   1.11712000e+01,
         1.19057000e+01,   1.21726000e+01,   1.26399000e+01,
         1.30404000e+01,   1.34409000e+01,   1.43088000e+01,
         1.44423000e+01,   1.51098000e+01,   1.59107000e+01,
         1.59775000e+01,   1.69788000e+01,   1.69790000e+01,
         1.78464000e+01,   1.79132000e+01,   1.85808000e+01,
         1.87810000e+01,   1.91814000e+01,   1.92483000e+01,
         1.94484000e+01,   1.98478000e+01,   1.99151000e+01,
         2.04486000e+01,   2.05141000e+01,   2.08483000e+01,
         2.13135000e+01,   2.15147000e+01,   2.20478000e+01,
         2.21799000e+01,   2.25144000e+01,   2.28478000e+01,
         2.29131000e+01,   2.36480000e+01,   2.37131000e+01,
         2.44481000e+01,   2.46469000e+01,   2.52485000e+01,
         2.55806000e+01,   2.58485000e+01,   2.65810000e+01,
         2.67156000e+01,   2.73161000e+01,   2.73815000e+01,
         2.78496000e+01,   2.81822000e+01,   2.84500000e+01,
         2.89167000e+01,   2.92496000e+01,   2.95843000e+01,
         2.98504000e+01,   3.01180000e+01,   3.05843000e+01,
         3.07182000e+01,   3.16521000e+01,   3.17192000e+01,
         3.24533000e+01,   3.27869000e+01,   3.31874000e+01,
         3.35874000e+01,   3.40551000e+01,   3.46557000e+01,
         3.56569000e+01,   3.58573000e+01,   3.65913000e+01,
         3.69920000e+01,   3.76595000e+01,   3.77263000e+01,
         3.86607000e+01,   3.93281000e+01,   3.99958000e+01,
         4.11306000e+01,   4.11308000e+01,   4.19319000e+01,
         4.28663000e+01,   4.29330000e+01,   4.40011000e+01,
         4.50692000e+01,   4.52697000e+01,   4.64043000e+01,
         4.66044000e+01,   4.70053000e+01,   4.72050000e+01,
         4.81393000e+01,   4.88737000e+01,   4.90744000e+01,
         4.94746000e+01,   4.98087000e+01,   4.98755000e+01])

    fig19b_bridge_to_emb_w_raw = np.array(
      [-0.712329, -0.711006, -0.693151, -0.691329, -0.673973, -0.66868 ,
       -0.660274, -0.659334, -0.646595, -0.641096, -0.633839, -0.621918,
       -0.62089 , -0.605479, -0.599433, -0.589041, -0.588326, -0.572603,
       -0.566825, -0.556164, -0.550595, -0.545205, -0.54519 , -0.543817,
       -0.542902, -0.542466, -0.53872 , -0.536986, -0.523288, -0.522871,
       -0.50411 , -0.489385, -0.482192, -0.471203, -0.465753, -0.472331,
       -0.476712, -0.482184, -0.484932, -0.486976, -0.487671, -0.489268,
       -0.494185, -0.49589 , -0.512329, -0.51898 , -0.526027, -0.536986,
       -0.550685, -0.558927, -0.567123, -0.560543, -0.556164, -0.565315,
       -0.569863, -0.565469, -0.558904, -0.559816, -0.567121, -0.567123,
       -0.567123, -0.568743, -0.575342, -0.581584, -0.586301, -0.590857,
       -0.594521, -0.601086, -0.60274 , -0.610035, -0.610959, -0.616438,
       -0.619171, -0.621918, -0.626785, -0.627397, -0.632877, -0.635309,
       -0.635616, -0.638356, -0.641638, -0.643836, -0.649315, -0.649542,
       -0.652055, -0.653051, -0.654795, -0.654795, -0.654795, -0.654795,
       -0.654795, -0.654795, -0.657323, -0.657534, -0.657534, -0.657535,
       -0.660274, -0.660274, -0.660274, -0.660274, -0.660274, -0.660274,
       -0.663014, -0.670048, -0.671233, -0.679452, -0.681696, -0.693151,
       -0.704627, -0.709589, -0.723288, -0.726389, -0.734247, -0.739726,
       -0.740844, -0.753425, -0.754539, -0.767123, -0.769845, -0.778082,
       -0.784148, -0.789041, -0.798299, -0.8     , -0.805479, -0.806488,
       -0.813699, -0.816735, -0.819178, -0.827397, -0.827397, -0.827397,
       -0.830129, -0.832877, -0.839263, -0.841096, -0.846208, -0.846575,
       -0.849315, -0.85056 , -0.852055, -0.853318, -0.854795, -0.855708,
       -0.85723 , -0.857534, -0.859306, -0.860274, -0.860274, -0.860274,
       -0.86347 , -0.865753, -0.864739, -0.863014, -0.863014, -0.863014,
       -0.863014, -0.863014, -0.863014, -0.863014, -0.86273 , -0.861124,
       -0.860841, -0.860274, -0.860803, -0.863277, -0.865222, -0.865753,
       -0.865753, -0.865753, -0.865753])

    fig19b_emb_to_bridge_x_raw = np.array(
      [ -4.92072000e+00,  -4.90682000e+00,  -4.71917000e+00,
        -4.70637000e+00,  -4.58438000e+00,  -4.50665000e+00,
        -4.38320000e+00,  -4.37332000e+00,  -4.23944000e+00,
        -4.18165000e+00,  -4.10538000e+00,  -3.98010000e+00,
        -3.97169000e+00,  -3.84549000e+00,  -3.77142000e+00,
        -3.64413000e+00,  -3.63828000e+00,  -3.50952000e+00,
        -3.43874000e+00,  -3.30816000e+00,  -3.17208000e+00,
        -3.04040000e+00,  -3.03821000e+00,  -2.83739000e+00,
        -2.70351000e+00,  -2.63968000e+00,  -2.50251000e+00,
        -2.43905000e+00,  -2.23787000e+00,  -2.23494000e+00,
        -2.10308000e+00,  -1.96755000e+00,  -1.90135000e+00,
        -1.76674000e+00,  -1.69998000e+00,  -1.50008000e+00,
        -1.36694000e+00,  -1.23397000e+00,  -1.16722000e+00,
        -9.68049000e-01,  -9.00379000e-01,  -8.35635000e-01,
        -6.36282000e-01,  -5.67149000e-01,  -4.34735000e-01,
        -3.70357000e-01,  -3.02138000e-01,  -2.36114000e-01,
        -3.67613000e-02,   2.96285000e-02,   9.56527000e-02,
         4.96918000e-01,   7.63941000e-01,   1.16466000e+00,
         1.36383000e+00,   1.63176000e+00,   2.03212000e+00,
         2.09869000e+00,   2.63219000e+00,   2.63237000e+00,
         3.16641000e+00,   3.23207000e+00,   3.49964000e+00,
         3.76539000e+00,   3.96620000e+00,   4.29888000e+00,
         4.56645000e+00,   5.09922000e+00,   5.23346000e+00,
         5.76623000e+00,   5.83371000e+00,   6.43415000e+00,
         6.76683000e+00,   7.10134000e+00,   7.63465000e+00,
         7.70178000e+00,   8.30221000e+00,   8.83552000e+00,
         8.90283000e+00,   9.50345000e+00,   9.90307000e+00,
         1.01706000e+01,   1.11049000e+01,   1.11712000e+01,
         1.19057000e+01,   1.21726000e+01,   1.26399000e+01,
         1.30404000e+01,   1.34409000e+01,   1.43088000e+01,
         1.44423000e+01,   1.51098000e+01,   1.59107000e+01,
         1.59775000e+01,   1.69788000e+01,   1.69790000e+01,
         1.78464000e+01,   1.79132000e+01,   1.85808000e+01,
         1.87810000e+01,   1.91814000e+01,   1.92483000e+01,
         1.94484000e+01,   1.98478000e+01,   1.99151000e+01,
         2.04486000e+01,   2.05141000e+01,   2.08483000e+01,
         2.13135000e+01,   2.15147000e+01,   2.20478000e+01,
         2.21799000e+01,   2.25144000e+01,   2.28478000e+01,
         2.29131000e+01,   2.36480000e+01,   2.37131000e+01,
         2.44481000e+01,   2.46469000e+01,   2.52485000e+01,
         2.55806000e+01,   2.58485000e+01,   2.65810000e+01,
         2.67156000e+01,   2.73161000e+01,   2.73815000e+01,
         2.78496000e+01,   2.81822000e+01,   2.84500000e+01,
         2.89167000e+01,   2.92496000e+01,   2.95843000e+01,
         2.98504000e+01,   3.01180000e+01,   3.05843000e+01,
         3.07182000e+01,   3.16521000e+01,   3.17192000e+01,
         3.24533000e+01,   3.27869000e+01,   3.31874000e+01,
         3.35874000e+01,   3.40551000e+01,   3.46557000e+01,
         3.56569000e+01,   3.58573000e+01,   3.65913000e+01,
         3.69920000e+01,   3.76595000e+01,   3.77263000e+01,
         3.86607000e+01,   3.93281000e+01,   3.99958000e+01,
         4.11306000e+01,   4.11308000e+01,   4.19319000e+01,
         4.28663000e+01,   4.29330000e+01,   4.40011000e+01,
         4.50692000e+01,   4.52697000e+01,   4.64043000e+01,
         4.66044000e+01,   4.70053000e+01,   4.72050000e+01,
         4.81393000e+01,   4.88737000e+01,   4.90744000e+01,
         4.94746000e+01,   4.98087000e+01,   4.98755000e+01])

    fig19b_emb_to_bridge_w_raw = np.array(
      [-0.5043  , -0.50411 , -0.501545, -0.50137 , -0.50639 , -0.509589,
       -0.512126, -0.512329, -0.506849, -0.503306, -0.49863 , -0.496063,
       -0.49589 , -0.49589 , -0.49589 , -0.501129, -0.50137 , -0.508441,
       -0.512329, -0.515012, -0.517808, -0.512419, -0.512329, -0.50411 ,
       -0.49863 , -0.49515 , -0.487671, -0.485722, -0.479542, -0.479452,
       -0.47675 , -0.473973, -0.471263, -0.465753, -0.467125, -0.471233,
       -0.478087, -0.484932, -0.489058, -0.50137 , -0.509771, -0.517808,
       -0.531507, -0.53578 , -0.543966, -0.547945, -0.549347, -0.550704,
       -0.5548  , -0.556164, -0.556164, -0.556164, -0.555069, -0.553425,
       -0.554593, -0.556164, -0.560863, -0.561644, -0.569863, -0.569867,
       -0.582062, -0.583562, -0.58906 , -0.594521, -0.597614, -0.60274 ,
       -0.606404, -0.613699, -0.615353, -0.621918, -0.622657, -0.629233,
       -0.632877, -0.632877, -0.632877, -0.633489, -0.638969, -0.643836,
       -0.644354, -0.648978, -0.652055, -0.652633, -0.654651, -0.654795,
       -0.654795, -0.654795, -0.654795, -0.654795, -0.654795, -0.654795,
       -0.654795, -0.65604 , -0.657534, -0.657363, -0.654795, -0.654795,
       -0.659882, -0.660274, -0.660274, -0.660274, -0.663014, -0.664665,
       -0.6696  , -0.679452, -0.681389, -0.696745, -0.69863 , -0.708937,
       -0.723288, -0.728377, -0.741865, -0.745205, -0.752705, -0.76018 ,
       -0.761644, -0.776744, -0.778082, -0.786708, -0.789041, -0.797867,
       -0.80274 , -0.806409, -0.816438, -0.81782 , -0.823985, -0.824658,
       -0.82786 , -0.830137, -0.832886, -0.837678, -0.841096, -0.841096,
       -0.841096, -0.843094, -0.846575, -0.847262, -0.852055, -0.852055,
       -0.852055, -0.852055, -0.856167, -0.860274, -0.859075, -0.857534,
       -0.860274, -0.860862, -0.863014, -0.861986, -0.860274, -0.860457,
       -0.863014, -0.863014, -0.863014, -0.860274, -0.860274, -0.860274,
       -0.863014, -0.863014, -0.863014, -0.860707, -0.860274, -0.863014,
       -0.865753, -0.867582, -0.868493, -0.873973, -0.871233, -0.871233,
       -0.871233, -0.865753, -0.864659])


    fig12top_w = fig12top_w_raw
    fig12top_t = fig12top_t_raw + 0

    fig13top_w = fig13top_w_raw
    fig13top_t = fig13top_t_raw + 0
    fig13bot_w = fig13bot_w_raw
    fig13bot_t = fig13bot_t_raw + 0

    fig14top_w = fig14top_w_raw
    fig14top_t = fig14top_t_raw + 0
    fig14bot_w = fig14bot_w_raw
    fig14bot_t = fig14bot_t_raw + 0

    fig15top_w = fig15top_w_raw
    fig15top_t = fig15top_t_raw + 0
    fig15bot_w = fig15bot_w_raw
    fig15bot_t = fig15bot_t_raw + 0


    fig19b_emb_to_bridge_x = fig19b_emb_to_bridge_x_raw
    fig19b_emb_to_bridge_w = fig19b_emb_to_bridge_w_raw

    fig19b_bridge_to_emb_x = fig19b_bridge_to_emb_x_raw
    fig19b_bridge_to_emb_w = fig19b_bridge_to_emb_w_raw

    top_ts = [fig13top_t, fig14top_t, fig15top_t, fig12top_t]
    top_ws = [fig13top_w, fig14top_w, fig15top_w, fig12top_w]
    top_t_offset=[0.03719, 0.173177, 0.23286-0.01137,-0.456988 ]

    with open(ftime + ".txt", "w") as f:
        f.write(ftime + os.linesep)

        matplotlib.rcParams.update({'font.size': 11})
        fig = plt.figure(figsize=(3.54,4.5))

        gs0 = matplotlib.gridspec.GridSpec(
                   2, 1,
#                  width_ratios=[1, 2],
                   height_ratios=[1,1],
#                   hspace=0.05,
                   )
        ax = fig.add_subplot(gs0[0,0])
        ax2 = fig.add_subplot(gs0[1,0], sharex=ax)


        fig2 = plt.figure(figsize=(8,8))
        ax3 = fig2.add_subplot("111")
        #fig2 = plt.figure(figsize=(3.54,3.54))
        #ax2 = fig2.add_subplot("111")

        #fig3 = plt.figure(figsize=(3.54,3.54))
        #ax3 = fig3.add_subplot("111")


        #t_lengths = np.linspace(25,0,6)
        #betas = [0, 0.05, 0.1, 0.3, 1.1, 2.0]

        fig4 = plt.figure(figsize=(3.54,8))
        gs = matplotlib.gridspec.GridSpec(
                   4, 1,
#                  width_ratios=[1, 2],
#                   height_ratios=[1,4],
#                   hspace=0.05,
#                   hspace = 0.05,
#                   top=0.05,
                   )
        axs =[]
        for i in range(4):
            axs.append(fig4.add_subplot(gs[i,0]))
#            if i==0:
#
#            else:
#                axs.append(fig4.add_subplot(gs[i,0], sharex=axs[0]))

            if i!=3:
                axs[i].get_xaxis().set_ticks([])

            axs[i].set_ylim(-1.2,0.2)
            axs[i].set_xlim(0,4)

            axs[i].set_ylabel("Rail displ. (mm)")

        axs[3].set_xlabel("Time (s)")







        k1 = 160e6*0.5 #N/m
        step1 = 2*0.684881633*.9
        step2 = 3*0.5*0.9
        #1030
        #10000

        pier_width = 0.5
        pierk=2.5#10000/80/4
        beamk=2*0.9#10000/80/4
        m1=2
        m2=2
                                    #x,     k1bar,  mubar,  rho_bar
        xp,kp,mup,rhop = np.array(
                                  [[0.0,                1.,     1,      1], #start
                                   [30,                 1.,     1.,     1],#start of UGM wedge,
                                   [30+(40-30)*0.25,       1+(step1-1)*0.25**m1,   1,   1],
                                   [30+(40-30)*0.50,       1+(step1-1)*0.50**m1,   1,   1],
                                   [30+(40-30)*0.75,       1+(step1-1)*0.75**m1,   1,   1],
                                   [40,                 step1,     1.,     1],     #end of UGM wedge,
                                   [50,                 step1,     1.,     1], #start of CBM,
                                   [50+(57-50)*0.25,       step1+(step2-step1)*0.25**m2,   1,   1],
                                   [50+(57-50)*0.50,       step1+(step2-step1)*0.50**m2,   1,   1],
                                   [50+(57-50)*0.75,       step1+(step2-step1)*0.75**m2,   1,   1],
                                   [57,                 step2,     1.,     1], #end of CBM
                                   [60,                 step2,     1.,     1],#Start abutment pier1
                                   [60,                 pierk,     1.,     1],
                                   [60+2*pier_width,    pierk,     1.,     1],
                                   [60+2*pier_width,    beamk,     1.,     1], #end abutment pier 1
                                   [97.5-pier_width,    beamk,     1.,     1],#Start abutment pier 2
                                   [97.5-pier_width,    pierk,     1.,     1],
                                   [97.5,               pierk,     1.,     1],
#                                   [97.5+pier_width,    pierk,     1.,     1],
#                                   [97.5+pier_width,    beamk,     1.,     1], #end abutment pier 2
#                                   [135-pier_width,     beamk,     1.,     1],#Start abutment pier 3
#                                   [135-pier_width,     pierk,     1.,     1],
#                                   [135+pier_width,     pierk,     1.,     1],
#                                   [135+pier_width,     beamk,     1.,     1], #end abutment pier 3
#                                   [172.5-pier_width,   beamk,     1.,     1],#Start abutment pier 4
#                                   [172.5-pier_width,   pierk,     1.,     1],
#                                   [172.5,              pierk,     1.,     1], # end of beam
                                   ]).T


#                                   [60-pier_width,      step2,     1.,     1],#Start abutment pier1
#                                   [60-pier_width,      pierk,     1.,     1],
#                                   [60+pier_width,      pierk,     1.,     1],
#                                   [60+pier_width,      beamk,     1.,     1], #end abutment pier 1
#                                   [97.5-pier_width,    beamk,     1.,     1],#Start abutment pier 2
#                                   [97.5-pier_width,    pierk,     1.,     1],
#                                   [97.5,               pierk,     1.,     1],


#xp,kp,mup,rhop = np.array(
#                                  [[0.0,                1.,     1,      1], #start
#                                   [30,                 1.,     1.,     1],#start of UGM wedge,
#                                   [40,                 step1,     1.,     1],     #end of UGM wedge,
#                                   [50,                 step1,     1.,     1], #start of CBM,
#                                   [57,                 step2,     1.,     1], #end of CBM
#                                   [60,                 step2,     1.,     1],#Start abutment pier1
#                                   [60,                 pierk,     1.,     1],
#                                   [60+2*pier_width,    pierk,     1.,     1],
#                                   [60+2*pier_width,    beamk,     1.,     1], #end abutment pier 1
#                                   [97.5-pier_width,    beamk,     1.,     1],#Start abutment pier 2
#                                   [97.5-pier_width,    pierk,     1.,     1],
#                                   [97.5,               pierk,     1.,     1],
##                                   [97.5+pier_width,    pierk,     1.,     1],
##                                   [97.5+pier_width,    beamk,     1.,     1], #end abutment pier 2
##                                   [135-pier_width,     beamk,     1.,     1],#Start abutment pier 3
##                                   [135-pier_width,     pierk,     1.,     1],
##                                   [135+pier_width,     pierk,     1.,     1],
##                                   [135+pier_width,     beamk,     1.,     1], #end abutment pier 3
##                                   [172.5-pier_width,   beamk,     1.,     1],#Start abutment pier 4
##                                   [172.5-pier_width,   pierk,     1.,     1],
##                                   [172.5,              pierk,     1.,     1], # end of beam
#                                   ]).T
#
#
##                                   [60-pier_width,      step2,     1.,     1],#Start abutment pier1
##                                   [60-pier_width,      pierk,     1.,     1],
##                                   [60+pier_width,      pierk,     1.,     1],
##                                   [60+pier_width,      beamk,     1.,     1], #end abutment pier 1
##                                   [97.5-pier_width,    beamk,     1.,     1],#Start abutment pier 2
##                                   [97.5-pier_width,    pierk,     1.,     1],
##                                   [97.5,               pierk,     1.,     1],



        #6 train cars distance in (m), load in (kN)
        train_x = np.array(
        [  -3.85,   -6.55,  -22.85,  -25.55,  -29.75,  -32.45,  -48.75,
           -51.45,  -55.65,  -58.35,  -74.65,  -77.35,  -81.55,  -84.25,
           -100.55, -103.25, -107.45, -110.15, -126.45, -129.15, -133.35,
           -136.05, -152.35, -155.05])
        train_f = np.array(
        [-129.8, -133.2, -132.6, -129.2, -131.5, -134.9, -134.9, -131.5,
           -128.8, -128.8, -128. , -128.8, -132.7, -132.7, -132.7, -132.7,
           -133.2, -136.6, -136.6, -133.2, -130.2, -133.6, -134.2, -130.8])*1000



        L = xp[-1]

        train_v = 220*1000/60/60 #m/s
        tstart = 0
        tend = (64.5 + (-1) * train_x[-1]) / train_v

        tvals = np.linspace(tstart, tend, nt)


        # x=64.5--> Paix -4.5
        # x=18.9--> Paix 41.1
        # x=45.3--> Paix 14.7
        # x=59.1--> Paix 0.9
        #i.e. xrohan = 64.5 - xpaix -4.5 ; need to check
        #xpaix = -4.5 + (64.5-xrohan)

        x_interest = [18.9, 45.3, 59.1, 64.5]

        xvals = np.append(np.linspace(0,64.5,nx), x_interest)
        xvals.sort()
        i_interest = np.searchsorted(xvals, x_interest)

        pdict = OrderedDict(
                    E = 210e9, #Pa
                    rho = 7850, #kg/m^3
                    L = L, #m
                    A = 0.3*0.12*7.7e-3, # m^2
                    I = 2*30.55e-6,#m^4
                    k1 = k1,#160e6,#N/m
                    mu = 17e3,
#                    kf=5.41e-4,
                    #Fz_norm=1.013e-4,
#                    mu_norm=39.263/4/2/1000,
#                    mubar=PolyLine([0,0.05,0.05,.95,.95,1],[1,1,1,1,1,1]),
#                    k1_norm=97.552,
    #                k1bar=PolyLine([0,0.2,0.3,1],[1.0,1,0.1,0.1]),
#                    k1bar=PolyLine([0,0.2,0.2,1],[1.0,1,1,1]),

    #                k3_norm=2.497e6,

                    k1bar = PolyLine(xp/L, kp),
                    moving_loads_x = [train_x], #m
                    moving_loads_Fz = [train_f], #N
                    moving_loads_v = [train_v], #m/s
                    nterms=nterms,
                    BC="SS",
#                    nquad=20,
#                    moving_loads_x_norm=[[0]],
#                    moving_loads_Fz_norm=[[1.013e-4]],
#                    moving_loads_v_norm=[v,],#[0.01165,],
                    tvals=tvals,
                    xvals_norm=xvals/L,
                    use_analytical=True,
                    implementation="fortran",
                    force_calc=force_calc,

                    )


        #zmax = np.zeros((len(alphas), len(t_lengths)), dtype=float)
        #zmin = np.zeros((len(alphas), len(t_lengths)), dtype=float)



        start_time1 = time.time()

        f.write("*" * 70 + os.linesep)
        vequals = "Paixao et al 2016"
        f.write(vequals + os.linesep); print(vequals)


        pdict["file_stem"] = "Paixaoet al_n{:d}".format(nterms)

        a = SpecBeam(**pdict)
        a.runme()
        a.saveme()
        a.animateme()
        #w_now_max = np.max(a.defl)
        #w_now_min = np.min(a.defl)

        defl_max_envelope = np.max(a.defl, axis=1)
        defl_min_envelope = np.min(a.defl, axis=1)


        #xpaix = -4.5 + (64.5-xrohan)
        dist_from_abut = -4.5 + (64.5-pdict["xvals_norm"]*L)
#        line, =ax.plot(pdict["xvals_norm"]*L, defl_max_envelope*1000,
#                 label="hello")
#        ax.plot(pdict["xvals_norm"]*L, defl_min_envelope*1000,
#                 color = line.get_color(),
#                 )

        line, =ax.plot(dist_from_abut, defl_min_envelope*1000,
                 label="specbeam")

        ax.plot(fig19b_emb_to_bridge_x,fig19b_emb_to_bridge_w, label="axle FEM (Paixao)",
                marker="s", color="green",markerfacecolor="none",
                markevery=0.2,ms=5,markeredgecolor="green")

        for i, (ii,xx) in enumerate(zip(i_interest,x_interest)):

            axs[i].plot(a.tvals, a.defl[ii,:]*1000, label='specbeam')

            offset_t = top_t_offset[i]
            axs[i].plot(top_ts[i]-offset_t,top_ws[i], label="measured x={:.1f}m".format(-4.5 + (64.5-xx)))

            ax3.plot(a.tvals,a.defl[ii,:]*1000,
                     label="x={}".format(xx))



        end_time1 = time.time()
        elapsed_time = (end_time1 - start_time1); print("Run time={}".format(str(timedelta(seconds=elapsed_time))))



        adjust = dict(top=0.9, bottom=0.13, left=0.19, right=0.95)
        #peak envelope plot
        ###################
        #ax.set_title("$\\beta={},k_1/k_2={}$".format(beta, k_ratio))
#        ax.set_xlabel("x")
        ax.set_ylabel("Peak displacement (mm)")
        ax.grid()
#        leg = ax3.legend(title="$\\alpha$",
#                        loc="upper right",
##                        ncol=2,
#                        labelspacing=.2,
#                        handlelength=2,
#                        fontsize=8
#                        )
#        leg.draggable()
        ax.xaxis.labelpad = -1
        ax.yaxis.labelpad = 0
        ax.set_xlim(50, -5)
        leg = ax.legend(fontsize=8,
                        loc='upper left')
        leg.draggable()
        fig.subplots_adjust(top=0.95, bottom=0.13, left=0.19, right=0.9)
        ####################
        #Kplot
        dist_from_abut = -4.5 + (64.5-xp)
        ax2.plot(dist_from_abut, kp)#xp, kp)
        ax2.set_xlabel("Distance from abutment (m)")
        ax2.set_ylabel("$k/k_{ref}$")
        ax2.set_ylim(0.9, 3)
        for i, x in enumerate(x_interest):
            y = np.interp(x,xp,kp)
            if i==0:
                ax2.plot(-4.5 + (64.5-x),y,ls=':', marker='o', label="defl. vs time evaluation point")
            else:
                ax2.plot(-4.5 + (64.5-x),y,ls=':', marker='o')

            leg = axs[i].legend(fontsize=8,
                    loc='lower left')
            leg.draggable()

        leg = ax2.legend(fontsize=8,
                        loc='upper left',
                        numpoints=1)
        leg.draggable()

        #displacemetn vs tiem at points plot
        ###################
        #ax3.set_title("$\\beta={},k_1/k_2={}$".format(beta, k_ratio))
        ax3.set_xlabel("time (s)")
        ax3.set_ylabel("Displacement (mm)")
#        ax3.grid()
        leg = ax3.legend(title="$\\alpha$",
                        loc="upper right",
#                        ncol=2,
                        labelspacing=.2,
                        handlelength=2,
                        fontsize=8
                        )
#        leg.draggable()
        ax3.xaxis.labelpad = -1
        ax3.yaxis.labelpad = 0


        #compare with field
        fig4.subplots_adjust(top=0.98, bottom=0.08, left=0.19, right=0.9, hspace=0.04)

#        fig.subplots_adjust(**adjust)

        end_time0 = time.time()
        elapsed_time = (end_time0 - start_time0); print("Total run time={}".format(str(timedelta(seconds=elapsed_time))))
        f.write("Total run time={}".format(str(timedelta(seconds=elapsed_time))) + os.linesep)
#        a.animateme()

    if not saveas is None:
        save_figure(fig=fig, fname=saveas+"_fig1")
        save_figure(fig=fig4, fname=saveas+"_fig2")
#        save_figure(fig=fig2, fname=saveas + "_up")

    plt.show()
    fig.clf()
#    fig2.clf()

def transition_DAF1(saveas=None,
                    ax=None,
                    kratios=None,
                    alpmax=3,
                    npts=500,
                    betas=None,
                    article_formatting=False,
                    DAFmax=4):
    """Plot of Dynamic Amplification Factor (DAF) vs velocity ratio
    for different stiffness and damping values.

    Parameters
    ----------
    saveas : string, optional
        Filename (not including extension) to save figure to.
        Default saveas=None, i.e. figure won't be saved to disk.
    ax : matplotlib.Axes object, optional
        Axes object to plot on. Default ax=None, i.e. figure and axis
        will be created.  Nb this might return an error if used as the function
        will try and return then figure.
    kratios : list of float, optional
        striffness ratio values. Default kratios=None which will
        mean kratios=[1,2].
    alpmax : float, optional
        maximum velocit ratio considered (i.e x-axis max).  Default alpmax=3
    npts : int, optional
        velocity ratios (i.e x-axis values) will be npts between
        0 and alpmax. Default npts=500.
    betas : list of float, optional
        list of damping ratios to consider.  Default betas=None which will
        mean betas=[0,0.05, 0.1, 0.3, 1.1, 2.0]
    article_formatting : [False, True], optional
        You should never need this.  It was a quick hack to do some
        custom formatting for a journal article.  Dont even try messing with
        it.  will ony work with 6 beta values or less.  Default=False,
        i.e. no special formatting.
    DAFmax : float, optional
        Y-axis max.  Default DAFmax=4


    Returns
    -------
    fig : matplotlib figure
        the figure


    """

    if article_formatting:
        mdc = MarkersDashesColors(markersize=6,linewidth=1)

        mdc.construct_styles(markers=[2,25,11,5,8,15],
                       dashes=[0],
                       marker_colors=[0,1,2,3,4,5],
                   line_colors=[0,1,2,3,4,5])

        styles=mdc(markers=[2,25,11,5,8,15],
                       dashes=[0],
                       marker_colors=[0,1,2,3,4,5],
                   line_colors=[0,1,2,3,4,5])
        #a.demo_styles()

    start_time0 = time.time()
    #ftime = datetime.datetime.fromtimestamp(start_time0).strftime('%Y-%m-%d %H%M%S')
    print("Started transition_DAF1")

    if ax is None:
        fig,ax = plt.subplots(figsize=(3.54,3.54))
    matplotlib.rcParams.update({'font.size': 11})

    if kratios is None:
        kratios=[1,2]

    if betas is None:
        betas=[0, 0.05, 0.1, 0.3, 1.1, 2.0]

    for k, kratio in enumerate(kratios):
        start_time1 = time.time()
        print("doing kratio={}".format(kratio))

        alpha=np.linspace(0,alpmax,npts)

        for j,bet in enumerate(betas):
            DAF = np.empty_like(alpha)

            for i, alp in enumerate(alpha):
                DAF[i] = DAFinfinite(alp, bet=bet,kratio=kratio)

            labela = "$k_2/k_1={}$".format(kratio)
            labelb = "$\\beta={}$".format(bet)
            if k==0:
                if j==0:
                    label=labelb+", " + labela
                else:
                    label=labelb

            else:
                if j==0:
                    label = labela
                else:
                    label=None

            if j==0:
                if article_formatting:
                    line, =ax.plot(alpha, DAF, label=label,markevery=(0.03+[0,0,0,0,0,.06][k],500.0),**styles[k])
                else:
                    line, =ax.plot(alpha, DAF, label=label,
                                   #alpha=1,
    #                               marker='s',ms=2,
                                   )

            else:
                ax.plot(alpha, DAF, label=label,
                        #alpha=0.8-0.5*j/len(betas),
                               #marker='s',ms=2,
                               color = line.get_color())

        end_time1 = time.time()
        elapsed_time = (end_time1 - start_time1); print("Run time={}".format(str(timedelta(seconds=elapsed_time))))

    leg = ax.legend(
#                    title="$\\beta$",
#                    loc="upper center",
#                    ncol=2,
                    labelspacing=0,
                    handlelength=2,
                    fontsize=8,
                    numpoints=1,
                    )


    ax.set_ylim([0, DAFmax])
    ax.set_xlim([0, alpmax])
    ax.grid()


    ax.set_ylabel("$DAF_{wrt1}$")
    ax.set_xlabel("$\\alpha_1$")

    ax.xaxis.labelpad = -1
    ax.yaxis.labelpad = 0

    fig.subplots_adjust(top=0.95, bottom=0.13, left=0.14, right=0.95)


    end_time0 = time.time()
    elapsed_time = (end_time0 - start_time0); print("Total run time={}".format(str(timedelta(seconds=elapsed_time))))

    if not saveas is None:
        save_figure(fig=fig, fname=saveas)
    return fig

def transition_DAF1_at_specific_alp(saveas=None,
                                    ax=None,
                                    alphas=None,
                                    kratios=None,
                                    betas=None,
                                    DAFmax=4):
    """Plot of Dynamic Amplification Factor (DAF) vs stiffness ratio
    for different velocity ratios and damping values.

    Parameters
    ----------
    saveas : string, optional
        Filename (not including extension) to save figure to.
        Default saveas=None, i.e. figure won't be saved to disk.
    ax : matplotlib.Axes object, optional
        Axes object to plot on. Default ax=None, i.e. figure and axis
        will be created.  Nb this might return an error if used as the function
        will try and return then figure.
    alphas : list of float,optional
        Specific velocity ratios to calc at.  Default alphas=None which will
        use [0.1, 0.5, 0.8].
    kratios : list of float, optional
        striffness ratio values. Default kratios=None which will
        mean kratios=np.logspace(np.log10(0.1), np.log10(100), 100), i.e
        100 points logspaced between 0.1 and 100.
    betas : list of float, optional
        list of damping ratios to consider.  Default betas=None which will
    DAFmax : float, optional
        Y-axis max.  Default DAFmax=4

    Returns
    -------
    fig : matplotlib figure
        the figure


    """

    start_time0 = time.time()
    print("Started transition_DAF1_at_specific_alp")

    if ax is None:
        fig,ax = plt.subplots(figsize=(3.54,3.54))
        matplotlib.rcParams.update({'font.size': 11})

    if kratios is None:
        kratios=np.logspace(np.log10(0.1), np.log10(100), 100)

    if alphas is None:
        alphas = [0.1,0.5,0.8]

    if betas is None:
        betas = [0, 0.05, 0.1, 0.3, 1.1, 2.0]


    for i, alp in enumerate(alphas):
        start_time1 = time.time()
        print("doing alpha={}".format(alp))

        for j,bet in enumerate(betas):
            DAF = np.empty_like(kratios)

            for k, kratio in enumerate(kratios):
                DAF[k] = DAFinfinite(alp/kratio**(0.25), bet*(1/kratio)**0.5)
                DAF[k] *= (1/kratio)**(0.75)

            labela = "$\\alpha={}$".format(alp)

            labelb = "$\\beta={}$".format(bet)

            if i==0:
                if j==0:
                    label=labelb+", " + labela
                else:
                    label=labelb

            else:
                if j==0:
                    label = labela
                else:
                    label=None

            if j==0:
                line, =ax.plot(kratios, DAF, label=label,
#                               marker='s',ms=2,
                               )
            else:
                ax.plot(kratios, DAF, label=label,
#                               marker='s',ms=2,
                               color = line.get_color())


        end_time1 = time.time()
        elapsed_time = (end_time1 - start_time1); print("Run time={}".format(str(timedelta(seconds=elapsed_time))))

    leg = ax.legend()
    leg.draggable()
    leg = ax.legend(
#                    title="$\\beta$",
#                    loc="upper center",
#                    ncol=2,
                    labelspacing=.2,
                    handlelength=2,
                    fontsize=8,
#                    numpoints=1,
                    )
    ax.set_ylim([0, DAFmax])
    ax.set_xscale('log')
    ax.grid()


    ax.set_ylabel("Deflection amplification factor, w.r.t. k1")
    ax.set_xlabel("$k_2/k_1$")

    end_time0 = time.time()
    elapsed_time = (end_time0 - start_time0); print("Total run time={}".format(str(timedelta(seconds=elapsed_time))))

    if not saveas is None:
        save_figure(fig=fig, fname=saveas)
    return ax




def DAFinfinite(alp, bet, kratio=1.0):
    """Dynamic amplification factor for infinite beam on winkler foundation

    Parameters
    ----------
    alp : float
        Velocity ratio.  alp = v/vcr where critical velocity
        vcr=sqrt(2*sqrt(k*E*I)/(rho*A))**0.5
    bet : float
        Damping ratio.  bet=c/ccr where critical damping ccr=2*sqrt(rho*A*k).
        The damping ratio is with respect to the reference foundation stiffness.
    kratio : float, optional
        Ratio of foundation stiffness to reference value i.e. k2/k1 where
        k1 is the reference value. Default kratio=1

    Returns
    -------
    DAF : float
        Dynamic/deflection amplification factor with respect to the reference
        foundation stiffness, k1. i.e.
        deflection = DAF*[static deflection of point load  on beam on k1 ]
    Notes
    -----
    Only works for scalar input

    """
    s, disp = moving_load_on_beam_on_elastic_foundation(
                            alp/kratio**(0.25),
                            bet*(1/kratio)**0.5
                            ,slim=(-6,6),npts=1001)
    DAF = np.real(np.max(disp)*(1/kratio)**(0.75))
    return DAF

def transition_zones1(
            Lt_Lcs=[0.1,4],
            alphas=[0.8],
            reverses=[False,True],
            xi_Lc=None, #ratio of xinterest over characteristic length
            kratio=50,
            beta=0.2,
            alpha_design=0.8,
            ntrans=50,
            DAF_distribution="linear",
            tanhend=0.01,
            xlim = (-8, 8),
            nx=100,     #number of x points
            nt=100, #number of time values
            t_extend = 1.2, # factor to extend maximum evalutated time.
            nterms=150,
            ninterp = 15, # number of points to discretize the transition zone
            nloads = 1,
            load_spacing = 0.03,
            animateme=False,
            saveas=None,
            force_calc=True,
            article_formatting=False,
            ):

    """Exploration of deflection envelopes caused by point loads travelling
    through a stiffness transition zone on beam on visco-elastic foundation
    using Specbeam.


    - Transition between reference beam/foundation with stiffness k to beam on
    foundation with stiffness k*`kratio`.
    - Transition zone length Lt is a multiple of the characteristic length
    Lc of the reference beam/foundation. Lt = `Lt_Lc`*Lc
    - Transition zone divided into ntrans points.
    - Change in stiffness over transition zone is a calculated value based on
    a goal displacement profile.  The shape of the goal displacement profile
    is controlled be `DAF_distribution` which is usually linear.
    - At each evaluation point in transition zone the stiffness needed to
    achieve the desired  displacement value is back calcualted based on
    infinite beam theory with a point load travelling at velocity ratio
    of `alpha_design` with damping ratio `beta` (`alpha_design` and `beta`
    are w.r.t. the reference beam/foundation).
    - for input into specbeam the design stiffness profile piecewise linear
    interpolated at `ninterp` points to produce piecewise linear analysis
    profile.  `ninterp` should be less thatn or equal to `ntrans`
    - Moving load will then traverse the design stiffness profile and
    depending on dynamic effects the moving load displacement may or may not
    match the design displacements.  You can only design for one speed so
    deflection response might be differnt when actual speed is different
    to design speed.
    - Note only the stiffness changes.  We assume that damping determined
    by `beta` of the reference beam/foundation does not change in the
    transition zone.  Perhaps not realistic but simply an assumption to
    enable exploration.

    3 plots will be produced:
    - Line will be plotted based on permututions of `Lt_Lcs`, `alphas`,
    `reverses`.  So number of lines can quickly get out of hand.  Usually
    interested in varyin one of the paramters at a time.
    - Plot 1a) shows the design transtion stiffness profile
    Plot 1b) shows max deflection experienced at each point
    - Plot 2) is deflection vs time at the point which experiences the
        maximum deflection.  also plots a marker at the time when
        load passes the point.
    - Plot 3) is the deflection vs time curve under the moving point load.


    Parameters
    ----------
    Lt_Lcs : list of float, optional
        Transtion zone lengths as a mulitple of characteristic length.
        Default Lt_Lcs = [0.1,4].
    alphas : list of float, optional
        Velocity ratios to analyze.   Velocity rations are
        with respect to the reference beam/foundation.  Default alphas = [0.8]
    reverses : list of Bool, optional
        Directions of travel.  False means left to right from reference
        beam/foundation to other beam/foundation.  True means the reverse
        direction from right to left from the other beam/foundation to the
        reference beam foundation. Default reverses=[False,True].
    xi_Lc : [False, True], optional
        If True then deflection vs time will be plotted at the point where
        the maximum deflection was experienced. Default xi_Lc=None i.e. False
        and no plot will be created. Also plots a marker at the time when
        load passes the point.
    kratio : float optional
        stiffness ratio k2/k1 where k1 is the stiffness of the reference
        beam/foundation. Default kratio = 50.
    beta : float, optional
        Reference value of damping ratio. Default beta = 0.2
    alpha_design : float, optional
        Design velocity ratio used to back calculate the design stiffness
        ratio. Default alpha_design = 0.8.
    ntrans : int, optional
        Number of points to approximate the desired displacement
        transition profile into, to then back calculate the stiffness
        profile used for analysis. Default ntrans = 50 .
    DAF_distribution : ["linear", "tanh", "reverse"], optional
        Shape of goal transition deflection distrribution.
        "linear" isself explanatory and simplest.
        "tanh" is a hyperbolic tangent, with steepness of transtion
        controlled by `tanend`. Not well tested.
        "reverse" is a compund reverse curve of two curclar arcs tangent to
        the end points and tangent at the arc join. Not well tested
        Default DAF_distribution = "linear".
    tanend : float, optional
        value controling the steepness of a "tanh" `DAF_distribution`.
        Say transitioning between points (-0.5, -1) and (0.5, 1).  Tanh
        curve will never actually hit valus of -1 and +1, it will approach
        those values as x goes to += infinity. Using tanend=0.01 will control
        transtion such that the tanh curve evalutes to y=1+0.01 and y=1-0.01
        at the start/end points.  This evaluated value is vever actually used
        because we have given start and end points but it does control the
        position of points just inside the start/end points and hence the
        steepness of transition.
    xlim : two element tuple of float, optional
        Distances on either side of the transtion zone midpoint to evaluate
        deflection at.  The mid point of the transition would be at 0.
        For example default xlim=(-8,8) which would be
        8 characteristic lenghts on either side of the transition zone
        midpoint.   Note: Characteristic lenth Lc is with respect to the
        reference beam/foundation. Default xlim=(-8, 8).
    nx : int, optional
        Number of points between `xlim[0]` and `xlim[1]` at which to
        calculate deflection.  Default nx = 100.
    nt : int, optional
        Number of evaluation times from when load enters `xlim[0]` and
        leaves `xlim[1]`.  Default nt=100
    t_extend : float, optional
        factor to extend maximum evalutated time.  As per `nt' evaluation times
        are normally between first point entering `xlim[0]' and leaving
        `xlim[1]`.  However, with mulitple loads, loads can still be on the
        beam at that point. max eval time can be extended fby a factor.
        t_extend = 1.2 .
    ninterp : int, optional
        For use in specbeam the back calculated stiffness and damping
        transitions profiles are interpoalted at `ninterp` points.  A coarser
        discretization will improve computation times but might no capture
        the changing properties approraitely.
        Note: `ninterp` should be less thatn or equal to `ntrans`.
        Default ninterp = 15 .
    nterms : int, optional
        Number of series terms in approximation.  More terms  leads to higher
        accuracy but also longer computation times.  Start of with a
        small enough value to observe behaviour and then increase
        to confirm. Default nterms=150 .
    nloads : int, optional
        Number of eqully spaced moving point loads.  Default nloads = 1 .
    load_spacing : float, optional
        Spacing between the multiple points loads.  Units are multiples of
        Lc characteristic length.  Default load_spacing = 0.03 .
    animateme : [False, True], optional
        If True the deflectons and moving loads will be animated.
        Default anmateme=False.
    saveas : string, optional
        Filename stem (not including extension) to save figure to.
        Default saveas=None, i.e. figure won't be saved to disk.
    force_calc : [True, False], optional
        When True all calcuations will be done from scratch. When False
        Current working directory will be searched for a results file
        named according to "saveas" and
        if found it will be loaded.  Be careful as name of results file may
        not be unique to your analysis paramters.  Safest to use
        force_calc = True, but if you are simply regenerating graphs then
        consider force_calc=False.  Default force_calc=True.
    article_formatting : [False, True], optional
        You should never need this.  It was a quick hack to do some
        custom formatting for a journal article.  Dont even try messing with
        it.  will ony work with 3 Lt_Lcs values values or less.  Default=False,
        i.e. no special formatting.


    Notes
    -----
    While all paramters are normalised the raw properties that all others
    are based on are:
    - E = 6.998*1e9, #Pa
    - rho = 2373, #kg/m3
    - L = 160, #m
    - A = 0.3*0.1, # m^2
    - I = 0.00022477900799999998,
    - k1 = 800002.6125,

    """

    if article_formatting:
        mdc = MarkersDashesColors(markersize=4,linewidth=1)
#        mdc.demo_options()
        mdc.construct_styles(markers=[5,15,2],
                           dashes=[0],
                           marker_colors=[0,1,2],
                       line_colors=[0,1,2])

        styles=mdc(markers=[5,15,2],
                           dashes=[0],
                           marker_colors=[0,1,2],
                       line_colors=[0,1,2])
#        mdc.demo_styles()

    #figure
    fig = plt.figure(figsize=(3.54,5))
    gs = matplotlib.gridspec.GridSpec(2, 1,
#                       width_ratios=[1, 2],
                   height_ratios=[1,4],
                   hspace=0.05)

    matplotlib.rcParams.update({'font.size': 11})

    #DAF vs kratio plot
#    ax1 = fig.add_subplot(gs[1,0])
#    transition_DAF1_at_specific_alp(ax=ax1)
#    ax1.set_ylim(0,2)

    #DAF vs xtrans plot
    ax2 = ax2 = fig.add_subplot(gs[1,0])
    ax2.set_ylabel("$w/w_{static1}$")
#    ax2.set_xlim(-0.5,1.5)

    ax2.set_xlabel('$x/L_c$')
    ax2.set_ylim(0,2)
#    ax1.set_ylim(-10, 10)



    #k/k1 vs xtrans plot
    ax3 = fig.add_subplot(gs[0,0], )#sharex=ax2)
#        ax3.set_xlabel('$x/L_t$')
    ax3.set_ylabel('$k/k_1$')

    ax3.grid()

    ax3.xaxis.set_ticklabels([])

#    xlim = (-.5,1.5)
#    xlim = (-8,8)
    ax2.set_xlim(*xlim)
    ax2.grid()


    if not xi_Lc is None:
        fig4, ax4 = plt.subplots() # deflection vs time at point

    fig5 = plt.figure(figsize=(3.54, 3.54))
    ax5 = fig5.add_subplot('111')


    pdict = OrderedDict(
                E = 6.998*1e9, #Pa
                rho = 2373, #kg/m3
                L = 160, #m
                A = 0.3*0.1, # m^2
                I = 0.00022477900799999998,
                k1 = 800002.6125,
                nterms=nterms,
                BC="SS",
#                moving_loads_x_norm=[[0-vv*load_spacing for vv in range(nloads)]],
#                moving_loads_Fz_norm=[[1.013e-4]*nloads],
#                xvals_norm=np.linspace(xeval[0], xeval[1], nx),
                use_analytical=True,
                implementation="fortran",
                force_calc=force_calc,
                )








    for k, Lt_Lc in enumerate(Lt_Lcs):
        print("doing Lt_Lc=", Lt_Lc)
        trans_characteristic_factor = Lt_Lc# multiple of k1 characteristic length
        trans_middle = 0.5 # midpoint of transition is midpoint of beam

        lam = (pdict['k1']/(4*pdict['E']*pdict['I']))**0.25
        trans_len_raw = trans_characteristic_factor / lam
        print("Characteristic len ={}".format(1/lam))
        print("normalised:{}".format(1/lam/pdict["L"]))
        trans_len = trans_len_raw / pdict["L"]

    #    trans_len=0.2  #normalised values w.r.t. L

        print("tlen/char_len={}".format(trans_len/(1/lam/pdict["L"])))

        trans_start = trans_middle - trans_len/2
        trans_end = trans_start + trans_len
        print("tstart={:8.3f},tend={:8.3f},tlen={:8.3f}".format(trans_start,trans_end, trans_len))

        xeval = (trans_middle - abs(xlim[0])/lam/pdict["L"],
                 trans_middle + abs(xlim[1])/lam/pdict["L"]) # Points either side of midpoint to evluate between

        print("xeval:",xeval)

        xtrans = np.linspace(0, 1, ntrans)


        if nloads>1:
            print("loadspace/char_len={}".format(load_spacing*pdict["L"]/(1/lam)))


        # work out goal displacement distribution
        DAF1 = DAFinfinite(alp=alpha_design, bet=beta)
        DAF2 = DAFinfinite(alp=alpha_design,bet=beta,kratio=kratio)

        if DAF_distribution.lower()=="linear":
            DAF_goal = DAF1 + (DAF2-DAF1)*xtrans
        elif DAF_distribution.lower()=="tanh":

            b = tanhtranib(x=0, y=tanhend, a=0.5)
            print("b:",b)
            DAF_goal = (tanhtran(x=xtrans, a=0.5, b=b) - 0.5)/(1-2*tanhend) + 0.5

            DAF_goal = DAF_goal * DAF2 + (1-DAF_goal) * DAF1

        elif DAF_distribution.lower()=="reverse":
            DAF_goal = reverse_curve_transition(xtrans,0,DAF1, 1, DAF2)

        DAF_goal[0] = DAF1
        DAF_goal[-1] = DAF2

#        figg,axx=plt.subplots()
#        axx.plot(xtrans,DAF_goal)
#        plt.show()

        # back calucatee the stiffness profile needed to get the displacemtn profile
        kkk = np.linspace(1, kratio,300,dtype=float)
        yyy = np.zeros_like(kkk)
        for iii, krat in enumerate(kkk):
            yyy[iii] = DAFinfinite(alp=alpha_design,bet=beta,kratio=krat)

#        figg,axx=plt.subplots()
#        axx.plot(kkk,yyy,marker="o")
#        plt.show()

        ktrans = np.empty_like(xtrans)
        for iii, DAF in enumerate(DAF_goal):
            if iii==0:
                ktrans[0]=1
            elif iii==ntrans-1:
                ktrans[-1]=kratio
            else:
                ktrans[iii]=np.interp(DAF, yyy[::-1], kkk[::-1])


        ctrans = np.ones_like(xtrans)


        #plot the stiffness profile
        xx = [xlim[0]]; xx.extend((xtrans-trans_middle)*trans_len*pdict["L"]*lam); xx.append(xlim[1])
        yy = [1]; yy.extend(ktrans); yy.append(kratio)
        ax3.plot(xx, yy)

        DAFtrans = np.empty_like(xtrans)
        for i, alp in enumerate(alphas):


            for j, krat in enumerate(ktrans):
                DAFtrans[j] = DAFinfinite(alp=alp,bet=beta,kratio=krat)


            yy = [DAFtrans[0]]; yy.extend(DAFtrans); yy.append(DAFtrans[-1])
            label="$L_t/L_c={}, ideal$".format(Lt_Lc)
            if article_formatting:
                line,= ax2.plot(xx,yy,label=label,**styles[k])
            else:
                line,= ax2.plot(xx,yy,ls="-",label=label)

            #sample the k value to form a poly line for specbeam
            xtrans_spec = np.linspace(0, 1, ninterp)

            ktrans_spec = np.interp(xtrans_spec, xtrans, ktrans)

            pline_x = [0]
            pline_x.extend(trans_start + xtrans_spec * trans_len)
            pline_x.append(1)

            pline_k1 = [ktrans_spec[0]]
            pline_k1.extend(ktrans_spec)
            pline_k1.append(ktrans_spec[-1])

            k1bar = PolyLine(pline_x, pline_k1)

            #sample the c values to form a poly line
            ctrans_spec = np.interp(xtrans_spec, xtrans, ctrans)
            pline_mu = [1]
            pline_mu.extend(ctrans_spec)
            pline_mu.append(ctrans_spec[-1])
            mubar = PolyLine(pline_x, pline_mu)

            v_crit = np.sqrt(2/pdict["rho"]/pdict["A"]*np.sqrt(pdict["k1"]*pdict["E"]*pdict["I"]))
            v_raw = v_crit*alp

            c_crit = np.sqrt(4*pdict["rho"]*pdict["A"]*pdict["k1"])
            c_raw = c_crit * beta

            tmax = pdict["L"] / v_raw # time for first load to reach end of beam

            pdict["moving_loads_x_norm"]=[[0-vv*load_spacing for vv in range(nloads)]]
            pdict["moving_loads_Fz_norm"]=[[1.013e-4]*nloads]
            pdict["xvals_norm"]=np.linspace(xeval[0], xeval[1], nx)


            pdict["mu"] = c_raw
            pdict["tvals"] = np.linspace(tmax*xeval[0],tmax*xeval[1]*t_extend,nt)

            pdict['k1bar']=k1bar
            pdict['mubar']=mubar
            for reverse in reverses:
                if reverse:
                    pdict["moving_loads_v_norm"] = [-1*v_raw*np.sqrt(pdict["rho"]/pdict["E"])]
                else:
                    pdict["moving_loads_v_norm"] = [1*v_raw*np.sqrt(pdict["rho"]/pdict["E"])]
                #pdict["file_stem"] = prefix + "DAF_alp{:5.3f}_bet{:5.3f}_n{:d}".format(alpha,beta,nterms)

                a = SpecBeam(**pdict)
                a.runme()
                if animateme:
                    a.animateme()
                #a.saveme()

                max_envelope = np.max(a.defl, axis=1)


    #            xplot = (pdict['xvals_norm']-trans_start)/trans_len
                xplot = (pdict['xvals_norm']-trans_middle)*pdict['L']*lam

        #        print(pdict['xvals_norm'])
        #        lam = (pdict['k1']/(4*pdict['E']*pdict['I']))**0.25



                Q = pdict["moving_loads_Fz_norm"][0][0] * pdict['E'] * pdict['A']

                wstatic = Q * lam / (2*pdict['k1'])

                # account for multiple loads
                # pdict["moving_loads_x_norm"]=[[0-vv*load_spacing for vv in range(nloads)]]
                # pdict["moving_loads_Fz_norm"]=[[1.013e-4]*nloads]

                xmultim,ymulti = multi_static(xc=[0-vv*(load_spacing*pdict["L"]) for vv in range(nloads)],
                                              frel=[1]*nloads,
                                              L=1/lam,
                                              xlim = (0-load_spacing*pdict["L"]*(nloads + 1), load_spacing*pdict["L"]),
                                              wsingle=wstatic)
                wstatic = np.max(ymulti)




        #        print(lam, Q, wstatic)
        #        print(max_envelope)

                yplot = max_envelope / wstatic
        #        print(xplot.shape, yplot.shape)
        #        print(xplot)
                if reverse:
                    label = "$L_t/L_c={},\ \\leftarrow$".format(Lt_Lc)
                    if k==0:
                        label = "$spec\ \\leftarrow$".format(Lt_Lc)
                    else:
                        label=None
                    ax2.plot(xplot, yplot, dashes=[4,2,1,2],
                         color=line.get_color(),
                        label = label)
                else:
                    label = "$L_t/L_c={},\ \\rightarrow$".format(Lt_Lc)
                    if k==0:
                        label = "$spec\ \\rightarrow$".format(Lt_Lc)
                    else:
                        label=None
                    ax2.plot(xplot, yplot, ls='--',
                         color=line.get_color(),
                         label = label)



                #point vs time plot
                if not xi_Lc is None:
                    max_position=np.unravel_index(a.defl.argmax(), a.defl.shape)[0]
                    xeval=a.xvals_norm[max_position]
#                    xeval = trans_middle + xi_Lc / lam/pdict["L"]
                    xi_Lc_ = (xeval-trans_middle)*lam*pdict["L"]
                    ieval = np.searchsorted(a.xvals_norm,xeval)
                    xplot=a.tvals#tvals_norm
                    yplot = a.defl[ieval-1,:] + (xeval-a.xvals_norm[ieval-1])/(a.xvals_norm[ieval]-a.xvals_norm[ieval-1]) * (a.defl[ieval,:]-a.defl[ieval-1,:])
#                    yplot = a.defl[ieval,:] + (xeval-a.xvals_norm[ieval])/(a.xvals_norm[ieval+1]-a.xvals_norm[ieval]) * (a.defl[ieval+1,:]-a.defl[ieval,:])
                    yplot /= wstatic
                    if article_formatting:
                        ax4.plot(xplot, yplot,
                                label="$L_t/L_c={}$".format(Lt_Lc),
                                markevery=(k/3*0.09,0.15),
                                **styles[k])
                    else:
                        ax4.plot(xplot, yplot, color=line.get_color(),
                                label="$L_t/L_c={}$".format(Lt_Lc) )
                    for imvpl, mvpl in enumerate(a.moving_loads_norm):
                        t_pass = mvpl.time_axles_pass_beam_point(xeval)
                        t_pass *= pdict["L"] / np.sqrt(pdict["E"] / pdict["rho"])
                        ax4.plot(t_pass, np.interp(t_pass,xplot,yplot),
                                     color=line.get_color(),
                                     marker="o", ms=7, ls=':',
                                     label="$x/L_c={:.1f}$".format(xi_Lc_))



                #under loads
                a.defl_vs_time_under_loads(xlim_norm=(a.xvals_norm[0], a.xvals_norm[-1]))
                dash_list=[(None,None),[2,2],[3,2,1,2],[4,4]]
                for imvpl, mvpl in enumerate(a.moving_loads_norm):
                    for j in range(len(mvpl.x)):
                        if len(Lt_Lcs)==1:
                            label = label="$load {}$".format(j+1)
                        else:
                            label = label="$L_t/L_c={},\ load {}$".format(Lt_Lc, j+1)


                        ax5.plot(a.tvals, a.moving_load_w[imvpl][j] / wstatic,
                             color=line.get_color(),
                             label=label,
                             dashes=dash_list[j])





    leg = ax2.legend(
#                    title="$\\beta$",
                    loc="upper right",
#                    ncol=2,
                    labelspacing=.2,
                    handlelength=2,
                    fontsize=8,
                    numpoints=1,
                        )
    leg.draggable()

    ax2.xaxis.labelpad = -1
    ax2.yaxis.labelpad = 0
    ax3.xaxis.labelpad = -1
    ax3.yaxis.labelpad = 0

    yloc = plt.MaxNLocator(5)
    ax3.yaxis.set_major_locator(yloc)

    fig.subplots_adjust(top=0.95, bottom=0.13, left=0.14, right=0.95)


    if not xi_Lc is None:
        ax4.set_xlabel('$Time (s)$')
        ax4.set_ylabel('$w/w_{static1}$')
        leg = ax4.legend(
#                    title="",
                    labelspacing=.2,
                    handlelength=2,
                    fontsize=8,
                    numpoints=1,
                    loc="center left",)
        leg.draggable()

    ax5.set_xlabel('$Time (s)$')
    ax5.set_ylabel('$w/w_{static1}$')
    leg = ax5.legend(
#                    title="",
                labelspacing=.2,
                handlelength=2,
                fontsize=8,
                numpoints=1,
                loc="lower right",)

    leg.draggable()
    fig5.subplots_adjust(top=0.95, bottom=0.13, left=0.15, right=0.95)
    if not saveas is None:

        save_figure(fig=fig, fname=saveas + "_fig1")
        save_figure(fig=fig4, fname=saveas + "_fig4")
        save_figure(fig=fig5, fname=saveas + "_fig5")


    return fig


def transition_zones2(animateme=False,
           saveas=None,
           xlim = (-8, 8),
           ntrans = 50, # number of points to sample analytical trasnstion zone
            kratio = 50,
            beta = 0.2,
            alphas = [0.8],
            nx = 100,     #number of x points
            nt=100, #number of time values
            nterms=150,
            force_calc=True,
            ninterp = 15, # number of points to discretize the transition zone
            nloads = 1,
            load_spacing = 0.03,
            alpha_design = 0.8, #this will be ignored
            Lt_Lcs = [0.1,4],
            t_extend = 1.1, # factor to extend maximum evalutated time.
            DAF_distribution = "linear",
            tanhend=0.01,
            reverses=[False,True],
            xi_Lc=None, #ratio of xinterest over characteristic length
            prefix = "",
            article_formatting=True):
    """Essentially a duplicate of transition_zones1 to get a DAF* vs Lt/Lc
    plot for forward and reverse plots

    Be warned that this can produce hundreds of plots and take hours to run
    when all you are interested is the single resulting plot.

    Basically a transtion_zones1 analysis will be run.  The maximum
    defelction (w.r.t. to the infinite beam deflection at same speed
    and damping) for that run will be recorded.  That will be one point for
    one alpha value for one Lt/Lc value.  Points will be joined to form
    a line for for each alphas and reverses combination.


    I used this once for a journal article figure.



    """

    if article_formatting:
        mdc1 = MarkersDashesColors(markersize=4,linewidth=1)

        mdc1.construct_styles(markers=[2,25,11,5,8,15],
                           dashes=[0],
                           marker_colors=[0,1,2,3,4,5],
                       line_colors=[0,1,2,3,4,5])

        styles1=mdc1(markers=[2,25,11,5,8,15],
                           dashes=[0],
                           marker_colors=[0,1,2,3,4,5],
                       line_colors=[0,1,2,3,4,5])

        mdc2 = MarkersDashesColors(markersize=4,linewidth=1)

        mdc2.construct_styles(markers=[13,18,21,9,0,3],
                           dashes=[1],
                           marker_colors=[0,1,2,3,4,5],
                       line_colors=[0,1,2,3,4,5])

        styles2=mdc1(markers=[13,18,21,9,0,3],
                           dashes=[1],
                           marker_colors=[0,1,2,3,4,5],
                       line_colors=[0,1,2,3,4,5])



    #figure
    fig = plt.figure(figsize=(3.54,5))
    gs = matplotlib.gridspec.GridSpec(2, 1,
#                       width_ratios=[1, 2],
                   height_ratios=[1,4],
                   hspace=0.05)

    matplotlib.rcParams.update({'font.size': 11})

    #DAF vs kratio plot
#    ax1 = fig.add_subplot(gs[1,0])
#    transition_DAF1_at_specific_alp(ax=ax1)
#    ax1.set_ylim(0,2)

    #DAF vs xtrans plot
    ax2 = ax2 = fig.add_subplot(gs[1,0])
    ax2.set_ylabel("$w/w_{static1}$")
#    ax2.set_xlim(-0.5,1.5)

    ax2.set_xlabel('$x/L_c$')
    ax2.set_ylim(0,2)
#    ax1.set_ylim(-10, 10)



    #k/k1 vs xtrans plot
    ax3 = fig.add_subplot(gs[0,0], )#sharex=ax2)
#        ax3.set_xlabel('$x/L_t$')
    ax3.set_ylabel('$k/k_1$')
    ax3.grid()

    ax3.xaxis.set_ticklabels([])

#    xlim = (-.5,1.5)
#    xlim = (-8,8)
    ax2.set_xlim(*xlim)
    ax2.grid()


    if not xi_Lc is None:
        fig4, ax4 = plt.subplots() # deflection vs time at point

    fig5, ax5 = plt.subplots()


    fig6 = plt.figure(figsize=(3.54,3.54))
    ax6 = fig6.add_subplot('111')

    pdict = OrderedDict(
                E = 6.998*1e9, #Pa
                rho = 2373, #kg/m3
                L = 160, #m
                A = 0.3*0.1, # m^2
                I = 0.00022477900799999998,
                k1 = 800002.6125,
                nterms=nterms,
                BC="SS",
#                moving_loads_x_norm=[[0-vv*load_spacing for vv in range(nloads)]],
#                moving_loads_Fz_norm=[[1.013e-4]*nloads],
#                xvals_norm=np.linspace(xeval[0], xeval[1], nx),
                use_analytical=True,
                implementation="fortran",
                force_calc=force_calc,
                )






    max_ratios=[]
    for reverse in reverses:
        max_ratios.append(np.zeros((len(alphas), len(Lt_Lcs))))

    for k, Lt_Lc in enumerate(Lt_Lcs):
        print("doing Lt_Lc=", Lt_Lc)
        trans_characteristic_factor = Lt_Lc# multiple of k1 characteristic length
        trans_middle = 0.5
#        t_extend = 1.1 # factor to extend maximum evalutated time.
        lam = (pdict['k1']/(4*pdict['E']*pdict['I']))**0.25
        trans_len_raw = trans_characteristic_factor / lam
        print("Characteristic len ={}".format(1/lam))
        print("normalised:{}".format(1/lam/pdict["L"]))
        trans_len = trans_len_raw / pdict["L"]

    #    trans_len=0.2  #normalised values w.r.t. L

        print("tlen/char_len={}".format(trans_len/(1/lam/pdict["L"])))
        trans_start = trans_middle - trans_len/2
        trans_end = trans_start + trans_len
        print("tstart={:8.3f},tend={:8.3f},tlen={:8.3f}".format(trans_start,trans_end, trans_len))

        xeval = (min(0.4,trans_start - (0-xlim[0]) * trans_len),
                     max(0.6,trans_end + (xlim[1] - 1)* trans_len))
        xeval = (trans_middle - abs(xlim[0])/lam/pdict["L"], trans_middle + abs(xlim[1])/lam/pdict["L"])
        print("xeval:",xeval)

        xtrans = np.linspace(0, 1, ntrans)













        DAFtrans = np.empty_like(xtrans)

        for i, alp in enumerate(alphas):

            alpha_design = alp
            DAF1 = DAFinfinite(alp=alpha_design, bet=beta)
            DAF2 = DAFinfinite(alp=alpha_design,bet=beta,kratio=kratio)


            if DAF_distribution.lower()=="linear":
                DAF_goal = DAF1 + (DAF2-DAF1)*xtrans
            elif DAF_distribution.lower()=="tanh":
                #ideal transition
                b = tanhtranib(x=0, y=tanhend, a=0.5)
                print("b:",b)
                DAF_goal = (tanhtran(x=xtrans, a=0.5, b=b) - 0.5)/(1-2*tanhend) + 0.5

                DAF_goal = DAF_goal * DAF2 + (1-DAF_goal) * DAF1

            elif DAF_distribution.lower()=="reverse":
                DAF_goal = reverse_curve_transition(xtrans,0,DAF1, 1, DAF2)

            DAF_goal[0] = DAF1
            DAF_goal[-1] = DAF2

            kkk = np.linspace(1,kratio,300,dtype=float)
            yyy = np.zeros_like(kkk)
            for iii,krat in enumerate(kkk):
                yyy[iii]=DAFinfinite(alp=alpha_design,bet=beta,kratio=krat)

            ktrans = np.empty_like(xtrans)
            for iii, DAF in enumerate(DAF_goal):
                if iii==0:
                    ktrans[0]=1
                elif iii==ntrans-1:
                    ktrans[-1]=kratio
                else:
                    ktrans[iii]=np.interp(DAF, yyy[::-1], kkk[::-1])




            ctrans = np.ones_like(xtrans)



            xx = [xlim[0]]; xx.extend((xtrans-trans_middle)*trans_len*pdict["L"]*lam); xx.append(xlim[1])
            yy = [1]; yy.extend(ktrans); yy.append(kratio)


            for j, krat in enumerate(ktrans):
                DAFtrans[j] = DAFinfinite(alp=alp,bet=beta,kratio=krat)


            yy = [DAFtrans[0]]; yy.extend(DAFtrans); yy.append(DAFtrans[-1])
            label="$L_t/L_c={}, idealized$".format(Lt_Lc)
            line,= ax2.plot(xx,yy,ls="-",label=label)

            #mymodel



            #sample the k value to form a poly line

            xtrans_spec = np.linspace(0, 1, ninterp)
    #        if reverse:
    #            ktrans_spec = np.interp(xtrans_spec, xtrans, ktrans_reverse)
    #        else:
            ktrans_spec = np.interp(xtrans_spec, xtrans, ktrans)
    #        print(ktrans_spec)
            pline_x = [0]
            pline_x.extend(trans_start + xtrans_spec * trans_len)
            pline_x.append(1)

            pline_k1 = [ktrans_spec[0]]
            pline_k1.extend(ktrans_spec)
            pline_k1.append(ktrans_spec[-1])

            k1bar = PolyLine(pline_x, pline_k1)
    #        print(k1bar)

            #sample the c values to form a poly line
            ctrans_spec = np.interp(xtrans_spec, xtrans, ctrans)
            pline_mu = [1]
            pline_mu.extend(ctrans_spec)
            pline_mu.append(ctrans_spec[-1])
            mubar = PolyLine(pline_x, pline_mu)
    #        print(mubar)


    #        xeval = (trans_start - (0-xlim[0]) * trans_len,
    #                 trans_end + (xlim[1] - 1)* trans_len)
    #        print(xeval)
    #
    ##        xwindow = (xeval[0],xeval[1])
    #
    #        pdict = OrderedDict(
    #                    E = 6.998*1e9, #Pa
    #                    rho = 2373, #kg/m3
    #                    L = 160, #m
    #                    A = 0.3*0.1, # m^2
    #                    I = 0.00022477900799999998,
    #                    k1 = 800002.6125,
    #                    nterms=nterms,
    #                    BC="SS",
    #                    moving_loads_x_norm=[[0-vv*load_spacing for vv in range(nloads)]],
    #                    moving_loads_Fz_norm=[[1.013e-4]*nloads],
    #                    xvals_norm=np.linspace(xeval[0], xeval[1], nx),
    #                    use_analytical=True,
    #                    implementation="fortran",
    #                    force_calc=force_calc,
    #                    )



            v_crit = np.sqrt(2/pdict["rho"]/pdict["A"]*np.sqrt(pdict["k1"]*pdict["E"]*pdict["I"]))
            v_raw = v_crit*alp

            c_crit = np.sqrt(4*pdict["rho"]*pdict["A"]*pdict["k1"])
            c_raw = c_crit * beta

            tmax = pdict["L"] / v_raw


            pdict["moving_loads_x_norm"]=[[0-vv*load_spacing for vv in range(nloads)]]
            pdict["moving_loads_Fz_norm"]=[[1.013e-4]*nloads]
            pdict["xvals_norm"]=np.linspace(xeval[0], xeval[1], nx)


            pdict["mu"] = c_raw
            pdict["tvals"] = np.linspace(tmax*xeval[0],tmax*xeval[1]*t_extend,nt)

            pdict['k1bar']=k1bar
            pdict['mubar']=mubar
            for m, reverse in enumerate(reverses):
                if reverse:
                    pdict["moving_loads_v_norm"] = [-1*v_raw*np.sqrt(pdict["rho"]/pdict["E"])]
                    pdict["file_stem"] = prefix + "tans2_alp{:5.3f}_bet{:5.3f}_n{:d}_LtLc{:5.3f}_rev".format(alp,beta,nterms,Lt_Lc)
                else:
                    pdict["moving_loads_v_norm"] = [1*v_raw*np.sqrt(pdict["rho"]/pdict["E"])]
                    pdict["file_stem"] = prefix + "tans2_alp{:5.3f}_bet{:5.3f}_n{:d}_LtLc{:5.3f}_for".format(alp,beta,nterms,Lt_Lc)


                print(pdict["file_stem"])
                a = SpecBeam(**pdict)
                a.runme()
                if animateme:
                    a.animateme()
                a.saveme()

                max_envelope = np.max(a.defl, axis=1)




    #            xplot = (pdict['xvals_norm']-trans_start)/trans_len
                xplot = (pdict['xvals_norm']-trans_middle)*pdict['L']*lam

        #        print(pdict['xvals_norm'])
        #        lam = (pdict['k1']/(4*pdict['E']*pdict['I']))**0.25



                Q = pdict["moving_loads_Fz_norm"][0][0] * pdict['E'] * pdict['A']

                wstatic = Q * lam / (2*pdict['k1'])
        #        print(lam, Q, wstatic)
        #        print(max_envelope)

                yplot = max_envelope / wstatic


                max_ratio = np.max(a.defl)/ wstatic / DAF1
                max_ratios[m][i, k] = max_ratio

        #        print(xplot.shape, yplot.shape)
        #        print(xplot)
                if reverse:
                    label = "$L_t/L_c={},\ \\leftarrow$".format(Lt_Lc)
                    if k==0:
                        label = "$spec\ \\leftarrow$".format(Lt_Lc)
                    else:
                        label=None
                    ax2.plot(xplot, yplot, dashes=[4,2,1,2],
                         color=line.get_color(),
                        label = label)
                else:
                    label = "$L_t/L_c={},\ \\rightarrow$".format(Lt_Lc)
                    if k==0:
                        label = "$spec\ \\rightarrow$".format(Lt_Lc)
                    else:
                        label=None
                    ax2.plot(xplot, yplot, ls='--',
                         color=line.get_color(),
                         label = label)



                #point vs time plot
                if not xi_Lc is None:
                    max_position=np.unravel_index(a.defl.argmax(), a.defl.shape)[0]
                    xevalu=a.xvals_norm[max_position]
#                    xevalu = trans_middle + xi_Lc / lam/pdict["L"]
                    xi_Lc_ = (xevalu-trans_middle)*lam*pdict["L"]
                    ieval = np.searchsorted(a.xvals_norm,xevalu)
                    xplot=a.tvals_norm
                    yplot = a.defl[ieval-1,:] + (xevalu-a.xvals_norm[ieval-1])/(a.xvals_norm[ieval]-a.xvals_norm[ieval-1]) * (a.defl[ieval,:]-a.defl[ieval-1,:])
#                    yplot = a.defl[ieval,:] + (xevalu-a.xvals_norm[ieval])/(a.xvals_norm[ieval+1]-a.xvals_norm[ieval]) * (a.defl[ieval+1,:]-a.defl[ieval,:])
                    yplot /= wstatic
                    ax4.plot(xplot, yplot, color=line.get_color(),
                            label="$L_t/L_c={}$".format(Lt_Lc) )
                    for imvpl, mvpl in enumerate(a.moving_loads_norm):
                        t_pass = mvpl.time_axles_pass_beam_point(xevalu)
                        ax4.plot(t_pass, np.interp(t_pass,xplot,yplot),
                                     color=line.get_color(),
                                     marker="o", ls=':',
                                     label="$x/L_c={:.1f}$".format(xi_Lc_))



                #under loads
                a.defl_vs_time_under_loads(xlim_norm=(a.xvals_norm[0], a.xvals_norm[-1]))
                dash_list=[(None,None),[2,2],[3,2,1,2],[4,4]]
                for imvpl, mvpl in enumerate(a.moving_loads_norm):
                    for j in range(len(mvpl.x)):
                        ax5.plot(a.tvals, a.moving_load_w[imvpl][j] / wstatic,
                             color=line.get_color(),
                             label="$L_t/L_c={},\ load {}$".format(Lt_Lc, j+1),
                             dashes=dash_list[j])





    leg = ax2.legend(
#                    title="$\\beta$",
                    loc="upper right",
#                    ncol=2,
                    labelspacing=.2,
                    handlelength=2,
                    fontsize=8,
                    numpoints=1,
                        )
    leg.draggable()

    ax2.xaxis.labelpad = -1
    ax2.yaxis.labelpad = 0
    ax3.xaxis.labelpad = -1
    ax3.yaxis.labelpad = 0


    fig.subplots_adjust(top=0.95, bottom=0.13, left=0.14, right=0.95)


    if not xi_Lc is None:
        ax4.set_xlabel('$norm _time$')
        ax4.set_ylabel('$w/w_{static1}$')
        leg = ax4.legend(
#                    title="",
                    labelspacing=.2,
                    handlelength=2,
                    fontsize=8,
                    numpoints=1)
        leg.draggable()

    ax5.set_xlabel('$time$')
    ax5.set_ylabel('$w/w_{static1}$')
    leg = ax5.legend(
#                    title="",
                labelspacing=.2,
                handlelength=2,
                fontsize=8,
                numpoints=1)

    leg.draggable()


    for i, alp in enumerate(alphas):
        for m, reverse in enumerate(reverses):
            xplot = Lt_Lcs
            yplot = max_ratios[m][i,:]
            if reverse:
                label = "$\\alpha_1={},\ \\leftarrow$".format(alp)
#                if k==0:
#                    label = "$spec\ \\leftarrow$".format(Lt_Lc)
#                else:
#                    label=None
#                ax2.plot(xplot, yplot, dashes=[4,2,1,2],
#                     color=line.get_color(),
#                    label = label)
                if article_formatting:
                    ax6.plot(xplot, yplot,
                             label=label,
                             markevery=(i/6*0.1,0.1),
                             **styles1[i])
                else:
                    ax6.plot(xplot, yplot,
                             label=label,
                             ls='--',color=line.get_color())
            else:
                label = "$\\alpha_1={},\ \\rightarrow$".format(alp)
#                if k==0:
#                    label = "$spec\ \\rightarrow$".format(Lt_Lc)
#                else:
#                    label=None
#
#                ax2.plot(xplot, yplot, ls='--',
#                     color=line.get_color(),
#                     label = label)
                if article_formatting:
                    line,=ax6.plot(xplot, yplot,
                             label=label,
                             markevery=(i/6*0.1,0.1),
                             **styles2[i])
                else:
                    line,=ax6.plot(xplot, yplot,
                             label=label,
                             ls='-')


#            ax6.plot(xplot,yplot,label="$\\alpha_1={}$".format(alp))
    ax6.set_xlabel('$L_t/L_c$')
    ax6.set_ylabel('DAF*')

    def flip(items, ncol):
        import itertools
        return itertools.chain(*[items[i::ncol] for i in range(ncol)])
    handles, labels = ax6.get_legend_handles_labels()

    leg = ax6.legend(flip(handles, 2), flip(labels, 2),
                     ncol=2,
                     labelspacing=.1,
                     handlelength=2,
                     fontsize=8,
                     handletextpad=0.1,
                     borderpad=0.2,
                     columnspacing=0.2)
    leg.draggable()
#        for k, Lt_Lcs in enumerate(Lt_Lcs):
    fig6.subplots_adjust(top=0.98, bottom=0.15, left=0.18, right=0.98)

    if not saveas is None:
        save_figure(fig=fig6, fname=saveas)


    return fig

def tanhtran(x, a, b):
    """Transition between two functions using tanh

    t(x) Smoothly transitions between 0 and 1;

    Used with:
        [transition f(x) to g(x)](x) = g(x)*t(x) + (1-t(x))*f(x)

    Parameters
    ----------
    x : float
        x values to evaluate at
    a : float
        Position of tranistion.  Often a=0.
    b : float,
        b close to zero gives a sharp change.


    Returns
    -------
    tfunc: float
        0.5 + 0.5 * tanh((x-a)/b)

    See Also
    --------
    tanhtrani : inverse of tanhtran
    tanhtranib : get smoothing factor from speccific x,y

    """
    return 0.5+0.5*np.tanh((x-0.5)/b)


def tanhtrani(y, a, b):
    """Inverse of Transition between two functions using tanh

    x value that matches y


    Parameters
    ----------
    y : float
        y value to evaluate at
    a : float
        Position of tranistion.  Often a=0.
    b : float,
        b close to zero gives a sharp change.


    Returns
    -------
    tfunc: float
        b * arctanh((y - 0.5) / 0.5) + a

    See Also
    --------
    tanhtran : inverse of tanhtrani
    tanhtranib : get smoothing factor from specific point.

    """
    return b * np.arctanh((y - 0.5) / 0.5) + a


def tanhtranib(x,y,a):
    """Smoothing factor of tansition curve to get known (x,y)

    b value such that t(x) goes through (x,y)


    Parameters
    ----------
    x,y : float
        x values to evaluate at
    a : float
        Position of tranistion.  Often a=0.



    Returns
    -------
    tfunc: float
        (x-a) / arctanh((y - 0.5) / 0.5)

    See Also
    --------
    tanhtran : transition function
    tanhtrani : inverse of tanhtran

    """
    return (x-a) / np.arctanh((y - 0.5) / 0.5)



def FIG_DAF_envelope_vary_beta(ax=None, betas=None, alphas=None,
                               saveas=None,
                               nx=200,nt=200,nterms=100, force_calc=True,
                               t_extend=1,
                               ylim=None,
                               numerical_DAF=False,
                               article_formatting=False):
    """DAF envelope for whole beam with different alpha & beta

    A plot will be produced of DAF vs normalised distance along beam, where
    the dynamic deflection in the DAF calculation is the maximum experienced
    each x/L point.


    Parameters
    ----------
    ax : matplotlib.Axes object, optional
        Axes object to plot on.  Default ax=None i.e. fig and axis will be
        created.
    betas : list of float, optional
        list of damping ratios, Default betas=None which will use
        [0, 0.05, 0.1].
    alphas : list of float, optional
        list of velocity ratios, Default alphas=None which will use
        [0.7].
    saveas : string, optional
        Filename stem (not including extension) to save figure to.
        Default saveas=None, i.e. figure won't be saved to disk.
    nx : int, optional
        Number of points between `xlim[0]` and `xlim[1]` at which to
        calculate deflection.  Default nx = 200.
    nt : int, optional
        Number of evaluation times from when load enters `xlim[0]` and
        leaves `xlim[1]`.  Default nt=200
    nterms : int, optional
        Number of series terms in approximation.  More terms  leads to higher
        accuracy but also longer computation times.  Start of with a
        small enough value to observe behaviour and then increase
        to confirm. Default nterms=100 .
    force_calc : [True, False], optional
        When True all calcuations will be done from scratch. When False
        Current working directory will be searched for a results file
        named according to "saveas" and
        if found it will be loaded.  Be careful as name of results file may
        not be unique to your analysis paramters.  Safest to use
        force_calc = True, but if you are simply regenerating graphs then
        consider force_calc=False.  Default force_calc=True.
    t_extend : float, optional
        factor to extend maximum evalutated time.  As per `nt' evaluation times
        are normally between first point entering `xlim[0]' and leaving
        `xlim[1]`.  However, with mulitple loads, loads can still be on the
        beam at that point. max eval time can be extended fby a factor.
        t_extend = 1 .
    ylim : float, optional
        Y axis limit. default ylim=None i.e. auto axis scale
    numerical_DAF : [False, True], optional
        If True then the static deflection for use in Dynamic amplification
        factor calcuations (DAF = w/wstatic) willbe determined from a
        very slow moving load travelling over the beam/foundation.
        If False then the analytical expression for wstatic will be used.
        Default numerical_DAF=False.
    article_formatting : [False, True], optional
        You should never need this.  It was a quick hack to do some
        custom formatting for a journal article.  Dont even try messing with
        it.  will ony work with 3 Lt_Lcs values values or less.  Default=False,
        i.e. no special formatting.


    """

    if article_formatting:
            mdc = MarkersDashesColors(markersize=3,linewidth=1)


            styles=mdc(markers=[5,15,2],
                   dashes=[0],
                   marker_colors=[0,1,2],
                   line_colors=[0,1,2])

            mevery = [0.4,0.15,0.1]

    #used for figure 3 in publication
    mytimer=SimpleTimer()
    mytimer.start(0,"FIG_DAF_envelope_vary_beta")


    if ax is None:
        fig,ax = plt.subplots(figsize=(3.54,3.54))

    matplotlib.rcParams.update({'font.size': 11})



    pdict = OrderedDict(
                E = 6.998*1e9, #Pa
                rho = 2373, #kg/m3
                L = 160, #m
                A = 0.3*0.1, # m^2
                I = 0.00022477900799999998,
                k1 = 800002.6125,
                nterms=nterms,
                BC="SS",
#                moving_loads_x_norm=[[0-vv*load_spacing for vv in range(nloads)]],
#                moving_loads_Fz_norm=[[1.013e-4]*nloads],
#                xvals_norm=np.linspace(xeval[0], xeval[1], nx),
                moving_loads_x_norm=[[0]],
                moving_loads_Fz_norm=[[1.013e-4]],
                xvals_norm=np.linspace(0, 1, nx),
                use_analytical=True,
                implementation="fortran",
                force_calc=force_calc,
                )




    if alphas is None:
        alphas = [0.7]
    if betas is None:
        betas = [0, 0.05, 0.1]



    if numerical_DAF:
        alphas.insert(0,0.00001)

    for i,alp in enumerate(alphas):
        for j,bet in enumerate(betas):

            mytimer.start(1, "alpha={}, beta={}".format(alp, bet))
#            start_time1 = time.time()
#            print("doing alpha={}, beta={}".format(alp, bet))
            v_crit = np.sqrt(2/pdict["rho"]/pdict["A"]*np.sqrt(pdict["k1"]*pdict["E"]*pdict["I"]))
            v_raw = v_crit*alp

            c_crit = np.sqrt(4*pdict["rho"]*pdict["A"]*pdict["k1"])
            c_raw = c_crit * bet

            tmax = pdict["L"] / v_raw



            pdict["file_stem"] = "envelope_alp{:5.3f}_bet{:5.3f}_nx{:d}_nt{:d}_n{:d}".format(alp,bet,nx,nt,nterms)

            pdict["mu"] = c_raw
            pdict["tvals"] = np.linspace(tmax*0,tmax*t_extend, nt)

            pdict["moving_loads_v_norm"] = [1*v_raw*np.sqrt(pdict["rho"]/pdict["E"])]


            a = SpecBeam(**pdict)
            a.runme()
#            a.animateme()
            a.saveme()


            max_envelope = np.max(a.defl, axis=1)


            lam = (pdict['k1']/(4*pdict['E']*pdict['I']))**0.25
            Q = pdict["moving_loads_Fz_norm"][0][0] * pdict['E'] * pdict['A']

            if numerical_DAF:
                if i==0:
                    wstatic = np.max(max_envelope[int(nx*0.4):int(nx*0.6)])
                    continue
            else:
                wstatic = Q * lam / (2*pdict['k1'])


            xplot = pdict['xvals_norm']
            yplot = max_envelope / wstatic

#            label = "$\\alpha={:3.2g},\\ \\beta={:3.2g}$".format(alp,bet)
            label = "${:3.2g}$".format(bet)

            if article_formatting:
                line, = ax.plot(xplot, yplot,
                                label=label,
                                markevery=(j/3*0.9,mevery[j]),
                                **styles[j])
            else:
                line, = ax.plot(xplot, yplot,
                                label=label,
                                ls='-',)
#                 color=line.get_color())
            DAF = DAFinfinite(alp, bet)
            if j==0:
                label="analytical"
            else:
                label=None

            ax.plot(0.6, DAF,
                    marker='o', ms=6,
                    ls='None',
                    color = line.get_color(),
                    label=label)
            mytimer.finish(1)

    ax.grid()
    ax.set_xlabel("$x/L$")
    ax.set_ylabel("$DAF$")
    if not ylim is None:
        ax.set_ylim(*ylim)
    leg=ax.legend()
    leg = ax.legend(
                    title="$\\beta$",
                    loc="upper center",
#                    ncol=2,
                    labelspacing=.2,
                    handlelength=2,
                    fontsize=8,
                    numpoints=1,
                        )
    leg.draggable()

    ax.xaxis.labelpad = -1
    ax.yaxis.labelpad = 0

    mytimer.finish(0)
#    end_time0 = time.time()
#    elapsed_time = (end_time0 - start_time0); print("Total run time={}".format(str(timedelta(seconds=elapsed_time))))
    fig.subplots_adjust(top=0.95, bottom=0.13, left=0.14, right=0.95)


    if not saveas is None:
        save_figure(fig=fig, fname=saveas)
#        save_figure(fig=fig2, fname=saveas + "_up")
    return ax


def static_point_load_shape(x, L):
    """Deflection shape parameter for point load at x=0 on infinite beam on
    elastic foundation  with characteristic length L.

    Deflection is relative to the maximum value

    Parameters
    ----------
    x : float
        x coordinate(s)
    L : float
        Beam characteristic Length.

    Returns
    -------
    out : float
        Relative deflection at x.

    Notes
    -----
    FYI the defelctions are relative to w_max which is :
        L = Characteristic length = k1 / ((4 * E * I)**0.25)
        Q = Point load
        w_max = static = Q * lam / (2 * k1)


    """

    return np.exp(-np.abs(x)/L)*(np.cos(np.abs(x)/L)+np.sin(np.abs(x)/L))


def specbeam_vs_analytical_single_static_load():
    """Compare specbeam vs analytical solution for single static load
    at mid point of 'long' beam.

    Will produce a plot and an animation.

    Note that you can't really have a static load in specbeam.  You can
    have a slowly applied load that then remains constant.  Thus depending
    on the beam properties e.g. damping, you can get a maximum displacement
    envelope that is higher than the static case in that the momentum of the
    beam carries the deflection past the 'analytical" maximum and then
    oscillations are created untill they are damped out.

    """




    mytimer=SimpleTimer()
    mytimer.start(0,"single_static_load")
    vfact=2
    t = np.linspace(0, 2, 40)
    t[-1]=999
#    x = np.linspace(0,160,2001)
    #see Dingetal2012 Table 2 for material properties
    pdict = OrderedDict(
            E = 6.998*1e9, #Pa
            rho = 2373, #kg/m3
            L = 20, #m
            A = 0.3*0.1, # m^2
            I = 0.00022477900799999998,
            k1 = 800002.6125,
            #v_norm=0.01165,
#            kf=5.41e-4,
            #Fz_norm=1.013e-4,
            mu_norm=39.263,#/100,
#            k1_norm=97.552,
#            k3_norm=2.497e6,
            nterms=201,
            BC="SS",
            nquad=20,
#            k1bar=PolyLine([0,0.5],[0.5,1],[1,1],[1,.25]),
#            k1bar=PolyLine([0,0.25,.75,1],[1.0,1,4,4]),
#            k1bar=PolyLine([0,0.4,.6,1],[1.0,1,.25,.25]),
#            k1bar=PolyLine([0,0.5],[0.5,1],[2,1],[2,1]),

#            moving_loads_x_norm=[[0,-0.05,-0.1]],
#            moving_loads_Fz_norm=[3*[1.013e-4]],
#            moving_loads_v_norm=[0.01165*vfact],

#            stationary_loads_x=None,
#            stationary_loads_vs_t=None,
#            stationary_loads_omega_phase=None,
            stationary_loads_x_norm=[0.5],
#            stationary_loads_vs_t_norm=[PolyLine([0,8],[1,1])],
            stationary_loads_vs_t_norm=[PolyLine([0,1,1000],[0,1e-4,1e-4])],
#            stationary_loads_omega_phase_norm=[(.5,0),(0,0)],#None,
            tvals_norm=t,
#            tvals=np.array([4.0]),
#            xvals_norm=np.array([0.5]),
#            xvals=x,

            use_analytical=True,
#            implementation="vectorized",
            implementation="fortran",
            file_stem="single_static_load",
            force_calc=True,
            )

    pdict['xvals'] = np.linspace(0,pdict['L'], 2001)

    a = SpecBeam(**pdict)
    a.runme()
#    a.saveme()
    a.animateme(norm=True)
#    a.plot_w_vs_x_overtime(norm=True)

    max_envelope = np.max(a.defl, axis=1)
    min_envelope = np.min(a.defl, axis=1)



    max_index = np.argmax(a.defl[int((2001-1)/2),:])
    #print(a.defl[int((2001-1)/2),:])
    #print(max_index)
    maxy = a.defl[:, max_index]
    yyy = a.defl[:,-1]

    lam = (pdict['k1'] /(4*pdict['E']*pdict['I']))**0.25
    Q = pdict["stationary_loads_vs_t_norm"][0].y[1] * pdict['E'] * pdict['A']

    wstatic = Q * lam / (2*pdict['k1'])

    fig,ax = plt.subplots()
    ax.plot(a.xvals, maxy, label='When max reached')
#    line,=ax.plot(a.xvals, max_envelope, label='envelope')
#    ax.plot(a.xvals, min_envelope, color=line.get_color())
    ax.plot(a.xvals, yyy, label='last',lw=4)

    xstatic = np.linspace(0,pdict["L"], 5001)
    ystatic = wstatic * static_point_load_shape((xstatic-pdict["L"]/2),1/lam)

    ax.plot(xstatic, ystatic, label="analytical",
            ls='None', marker='o', markevery=0.02)

    ax.set_xlabel('x')
    ax.set_ylabel('Deflection')
    ax.set_title('Single static load at beam midpoint')

    leg = ax.legend()
    leg.draggable()

    mytimer.finish(0)
    plt.show()

def reverse_curve_transition(x, x1, y1, x2, y2):
    """Transition curve between (x1, y1) and (x2, y2)

    Connects two parallel horizontal lines by two circular arcs.

    Parameters
    ----------
    x : array of float
        value(s) at which to caluclate transition curve
    x1, y1 : float
        start point of transition
    x2, y2 : float
        end point of transition

    Returns
    -------
    y : float
        value of transition curve at x.

    Notes
    -----
    Two circular curves transition in a rectangle h high by L wide.  Each
    curve has radius R and traces out a curve of delta radians.
    h=2*R*(1-cos(delta))
    &
    L=2*R*(sin(delta))

    Solve these in wolfram alpha for delta :Solve[(1-Cos[x])/Sin[x]-h/L,x]
    to get : delta = 2*arctan(h/L)
    Then use equation of circle in particualr quadrant to get points on curve.
    adjust for x1,y1 and x2,y2 starting points.

    """

    x = np.atleast_1d(x)
    y = np.empty_like(x)

    if y1 == y2:
        #no transition --> horizontal line
        y[:]=y1
        return y

    xmid = (x2 + x1) / 2
    xm = x < xmid #LHS of transition
    xp = x >= xmid #RHS of transition

    if x2 > x1:
        xa, xb = x1, x2
        ya, yb = y1, y2
    elif x1>x2:
        xa, xb= x2, x1
        ya, yb= y2, y1
    else:
        raise ValueError("x1 must be differnet to x2")

    xmm = x <= xa #to left of transition
    xpp = x >= xb #to right of transition

    xm*=~xmm
    xp*=~xpp

    delta = 2 * np.arctan(abs(yb - ya) / abs(xb - xa))
    R = abs(xb - xa) / (2*np.sin(delta))

    if yb<ya:
        y[xm] = np.sqrt(R**2 - (x[xm]-xa)**2) + ya - R
        y[xp] = -np.sqrt(R**2 - (x[xp]-xb)**2) + yb + R
    else:
        y[xm] = -np.sqrt(R**2 - (x[xm]-xa)**2) + ya + R
        y[xp] = +np.sqrt(R**2 - (x[xp]-xb)**2) + yb - R
    y[xmm] = ya
    y[xpp] = yb

    return y





def article_figure_01():
    """Figure 1 Mid point deflection comparison with Ding et al 2012, figure
    from Walker and Indraratna (in press).

    References
    ----------
    .. [1] Walker, R.T.R. and Indraratna, B, (in press) "Moving loads on a
           viscoelastic foundation with special reference to railway
           transition zones". International Journal of Geomechanics.
    """

    ################################
    # FIGURE GENERATING CODE
    # Figure 1 Mid point deflection comparison with Ding et al 2012
    #################################
    mpl.style.use('classic')
    #Journal paper figure, verify against ding et al.
    FIGURE_verify_against_DingEtAl2012_fig8(v_nterms=(50,75,150,200),saveas="verify_Dingetal2012")#75,150,200))

    plt.show()


def article_figure_02():
    """Figure 2 Deflection amplification factor compared with analytical
    infinite beam solution, figure from Walker and Indraratna (in press).

    References
    ----------
    .. [1] Walker, R.T.R. and Indraratna, B, (in press) "Moving loads on a
           viscoelastic foundation with special reference to railway
           transition zones". International Journal of Geomechanics.
    """
    #################################
    # FIGURE GENERATING CODE
    # Figure 2 Deflection amplification factor compared with analytical infinite beam solution
    #################################
    mpl.style.use('classic')
    FIGURE_DAF_constant_prop(saveas="DAF_constant_astatic",
                             nterms=150,
                             force_calc=False,
#                                 betas=[0.05],
#                                 alphas=np.linspace(1e-5,3,4),
#                                 alphas=np.linspace(0.01,0.95,10),
                             nx=2000, nt=100,
                             prefix="static_a",
#                                 end_damp=0.04,
                             xwindow=(0.3,0.7),
                             xeval=(0, 1),
                             numerical_DAF=False,
                             article_formatting=True)
    plt.show()


def article_figure_03():
    """Figure 3 Low damping DAF for velocity ratio  alp=0p5,
    figure from Walker and Indraratna (in press).

    References
    ----------
    .. [1] Walker, R.T.R. and Indraratna, B, (in press) "Moving loads on a
           viscoelastic foundation with special reference to railway
           transition zones". International Journal of Geomechanics.
    """
    #################################
    # FIGURE GENERATING CODE
    # Figure 3 Low damping DAF for velocity ratio  alp=0.5
    #################################
    mpl.style.use('classic')
    ax=FIG_DAF_envelope_vary_beta(ax=None,
                               betas=[0, 0.05,0.5],
                               alphas=[0.5],
                           saveas="low_beta_DAF_envelope",
                           nx=1001,nt=1001,nterms=150,
                           force_calc=False,
                           t_extend=1,
                           ylim=(0.8,1.6),
                           numerical_DAF=False,
                           article_formatting=True)
    plt.show()

def article_figure_04():
    """Figure 4 Relative DAF for various stiffness ratios,
    figure from Walker and Indraratna (in press).

    References
    ----------
    .. [1] Walker, R.T.R. and Indraratna, B, (in press) "Moving loads on a
           viscoelastic foundation with special reference to railway
           transition zones". International Journal of Geomechanics.
    """
    #################################
    # FIGURE GENERATING CODE
    # Figure 4 Relative DAF for various stiffness ratios
    #################################
    mpl.style.use('classic')
    fig = transition_DAF1(kratios=[0.5, 1, 2, 5, 20, 50],
                          alpmax=3,
                          npts=500,
                          saveas="transitionDAF1",
                          article_formatting=True)
    plt.show()


def article_figure_05a_05b_05c_07():
    """Figure 5 a),b),&c) Max deflection envelope for  bet=0p1  k2_k1=10,
    and Figure 7 Deflection vs time at point which experiences maximum
    deflection  alp=0p8,  bet=0p1,  k2_k1=10, soft to stiff=forward,
    figure from Walker and Indraratna (in press).

    References
    ----------
    .. [1] Walker, R.T.R. and Indraratna, B, (in press) "Moving loads on a
           viscoelastic foundation with special reference to railway
           transition zones". International Journal of Geomechanics.
    """
    #################################
    # FIGURE GENERATING CODE
    # Figure 5 a),b),&c) Max deflection envelope for  bet=0.1,  k2_k1=10
    # will need to comment out code as appropriate for 5a, 5b, 5c.
    # Figure 7 Deflection vs time at point which experiences maximum deflection  alp=0.8,  bet=0.1,  k2_k1=10, soft to stiff=forward
    #################################
    mpl.style.use('classic')
    for alpha_design, alphas, reverses, saveas in zip(
                [0.3, 0.8, 0.8],
                [[0.3], [0.8] , [0.8]],
                [[False], [False] , [True]],
                ["Lt_lc_alp0.3_bet0.1_forward",
                 "Lt_lc_alp0.8_bet0.1_forward" ,
                 "Lt_lc_alp0.8_bet0.1_reverse"]
                 ):

        transition_zones1(xlim = (-8,8),
           ntrans = 50, # number of points to sample analytical trasnstion zone
            kratio = 10,
            beta = 0.1,
            nx = 400,     #number of x points
            nt=400, #number of time values
            nterms=150,
            force_calc=True,
            ninterp = 50, # number of points to discretize the transition zone
            nloads = 1,
            load_spacing = 0.03,
            Lt_Lcs = [0.01,2,5], #2,4,6],
            t_extend = 1.0, # factor to extend maximum evalutated time.
            DAF_distribution = "linear",
            tanhend=0.01,
            alpha_design = alpha_design, #alpha for back calculating k profile
            alphas = alphas,
            reverses=reverses,
            saveas=saveas,
            animateme=False,

            xi_Lc=True, #ratio of xinterest over characteristic length)
            article_formatting=True)
    plt.show()

def article_figure_06():
    """Additional deflection for transition effects,  bet=0p1,  alp=10, (dashed=stiff to soft, solid=soft to stiff),
    figure from Walker and Indraratna (in press).

    This takes forever to run.

    References
    ----------
    .. [1] Walker, R.T.R. and Indraratna, B, (in press) "Moving loads on a
           viscoelastic foundation with special reference to railway
           transition zones". International Journal of Geomechanics.
    """
    #################################
    # FIGURE GENERATING CODE
    # Figure 6 Additional deflection for transition effects,  bet=0.1,  alp=10, (dashed=stiff to soft, solid=soft to stiff)
    #################################
    #this is for generating DAF* vs Lt_Lc
    trans2(xlim = (-10,10),
       ntrans = 50, # number of points to sample analytical trasnstion zone
        kratio = 10,
        beta = 0.1,
        alphas = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8],
        nx = 400,     #number of x points
        nt=400, #number of time values
        nterms=150,
        force_calc=False,
        ninterp = 50, # number of points to discretize the transition zone
        nloads = 1,
        load_spacing = 0.03,
        #alpha_design = 0.3, #alpha for back calculating k profile, #this will be ignored
        Lt_Lcs = np.linspace(0.01, 15, 50), #2,4,6],
        t_extend = 1.2, # factor to extend maximum evalutated time.
        DAF_distribution = "linear",
        tanhend=0.01,
        reverses=[False,True],
        animateme=False,
        saveas="DAf_star_vs_Lt_Lc",
        xi_Lc=True, #ratio of xinterest over characteristic length)
        )
    plt.show()

def article_figure_08():
    """Figure 8 Deflection vs time under four moving loads alp=0p8,  bet=0p1,  k2/k1=10, soft to stiff,
    figure from Walker and Indraratna (in press).

    References
    ----------
    .. [1] Walker, R.T.R. and Indraratna, B, (in press) "Moving loads on a
           viscoelastic foundation with special reference to railway
           transition zones". International Journal of Geomechanics.
    """
    #################################
    # FIGURE GENERATING CODE
    # Figure 8 Deflection vs time under four moving loads alp=0.8,  bet=0.1,  k2/k1=10, soft to stiff
    #################################
    mpl.style.use('classic')
    #This is for multi-load
    for Lt in [0.01, 4, 8]:
        transition_zones1(xlim = (-10,10),
           ntrans = 50, # number of points to sample analytical trasnstion zone
            kratio = 10,
            beta = 0.1,
            alphas = [0.8],
            nx = 400,     #number of x points
            nt=400, #number of time values
            nterms=150,
            force_calc=True,
            ninterp=50, # number of points to discretize the transition zone
            nloads = 4,
            load_spacing = 0.03,
            alpha_design = 0.8, #alpha for back calculating k profile
            Lt_Lcs = [Lt], #2,4,6],
            t_extend = 1.2, # factor to extend maximum evalutated time.
            DAF_distribution = "linear",
            tanhend=0.01,
            reverses=[False],
            animateme=True,
            saveas="four_loads_Lt_lc{:.2g}_alp0.8_bet0.1_forward".format(Lt),
            xi_Lc=True, #ratio of xinterest over characteristic length)
            )
    plt.show()

def article_figure_09():
    """Figure 9 is  drawing in a word file,  this routine is just a marker
    to indicated that figure 9 has not been forgotten,
    figure from Walker and Indraratna (in press).

    References
    ----------
    .. [1] Walker, R.T.R. and Indraratna, B, (in press) "Moving loads on a
           viscoelastic foundation with special reference to railway
           transition zones". International Journal of Geomechanics.
    """
    #################################
    # FIGURE GENERATING CODE
    # Figure 9 is a drawing in a word file
    #################################
    print("Figure 9 is  drawing in a word file.  This routine is just a marker to indicated that figure 9 has not been forgotten")

def article_figure_10_11():
    """Figure 10 Measured vs modelled deflections at various locations in the
    transition zone, Figure 11 Peak displacement and stiffness distribution,
    figure from Walker and Indraratna (in press).

    References
    ----------
    .. [1] Walker, R.T.R. and Indraratna, B, (in press) "Moving loads on a
           viscoelastic foundation with special reference to railway
           transition zones". International Journal of Geomechanics.
    """
    #################################
    # FIGURE GENERATING CODE
    # Figure 10 Measured vs modelled deflections at various locations in the transition zone
    # Figure 11 Peak displacement and stiffness distribution.
    #################################
    mpl.style.use('classic')
    #generate comparison plots for Paixao case study
    case_study_PaixaoEtAl2014(saveas="Paixao", nterms=150, force_calc=False,
                   nx=500,
                   nt=500)


def article_figure_12():
    """Figure 12 python code for relative dynamic amplification factor
    calculation, figure from Walker and Indraratna (in press).

    References
    ----------
    .. [1] Walker, R.T.R. and Indraratna, B, (in press) "Moving loads on a
           viscoelastic foundation with special reference to railway
           transition zones". International Journal of Geomechanics.
    """

    #################################
    # FIGURE GENERATING CODE
    # Figure 12 python code for relative dynamic amplification factor calculation
    #################################
    mpl.style.use('classic')
    from geotecha.plotting.one_d import figure_from_source_code
    fig = figure_from_source_code(DAFrelative, figsize=(6.5,5))
    save_figure(fig, "DAFrelative")
    print("done")
    plt.show()



if __name__ == "__main__":
    import nose
    mpl.style.use('classic')
#    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest', '--doctest-options=+ELLIPSIS'])
#    test_DingEtAl2012()
#    test_DingEtAl2012_deflection_shape()
#    test_SpecBeam_const_mat_midpoint_defl()
#    test_SpecBeam_const_mat_deflection_shape()
#    test_SpecBeam_const_mat_midpoint_defl_runme()
#    test_SpecBeam_const_mat_deflection_shape_runme()
#    test_SpecBeam_const_mat_midpoint_defl_runme_analytical_50()
#    test_SpecBeam_const_mat_midpoint_defl_runme_analytical_200()
#    SpecBeam_const_mat_midpoint_defl_runme_analytical_play()
#    test_SpecBeam_const_mat_deflection_shape_stationary_load()
#    SpecBeam_stationary_load_const_mat_midpoint_defl_runme_analytical_play()
#    dingetal_figure_8()


############################################
# Article figures
############################################

#    article_figure_01()
#    article_figure_02()
#    article_figure_03()
#    article_figure_04()
#    article_figure_05a_05b_05c_07()
#    article_figure_06() # this can take forever to run!!!!!!!! i.e. 5+hours
#    article_figure_07()
#    article_figure_08()
#    article_figure_09()
#    article_figure_10_11()
#    article_figure_12()




############################################
# Other code
############################################
    if 0:
        ##An example of using the reverse transition curves
        x = np.linspace(-4, 4, 1000)
        x1, y1 = (-2, 2)

        fig, ax = plt.subplots()
        for x2, y2 in [(1,1), (1,.5), (1,2.5),(-3,3),(-3,0.7), (3,2)]:

            ax.plot(x1, y1, marker='o')
            ax.plot(x2, y2, marker='s')

            y = reverse_curve_transition(x, x1, y1, x2, y2)
    #        reverse_curve_transition(xtrans,0,DAF1, 1, DAF2)
            ax.plot(x,y)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Transiton curve")

        plt.show()

    if 0:
        #Use SpecBeam to analyse/check a single static load at midpoint
        specbeam_vs_analytical_single_static_load()

    if 0:
        #example of deflection envelope
        for alpha_design, alphas, reverses, saveas in zip(
                    [0.8],
                    [[0.8]],
                    [[False]],
                    ["Lt_lc_alp0.3_bet0.1_forward"]
                     ):

            transition_zones1(xlim = (-8,8),
               ntrans = 50, # number of points to sample analytical trasnstion zone
                kratio = 10,
                beta = 0.1,
                nx = 400,     #number of x points
                nt=400, #number of time values
                nterms=150,
                force_calc=False,
                ninterp = 50, # number of points to discretize the transition zone
                nloads = 1,
                load_spacing = 0.03,
                Lt_Lcs = [2], #2,4,6],
                t_extend = 1.0, # factor to extend maximum evalutated time.
                DAF_distribution = "linear",
                tanhend=0.01,
                alpha_design = alpha_design, #alpha for back calculating k profile
                alphas = alphas,
                reverses=reverses,
                saveas=saveas,
                animateme=True,

                xi_Lc=True, #ratio of xinterest over characteristic length)
                article_formatting=False)
        plt.show()

    if 0:
        esveld_fig6p17()
        esveld_fig6p18()
        plt.show()

    if 0:
        #Messing around with deflection shape from multiple static loads
        fig, ax = plt.subplots()

        x = np.linspace(-4, 4, 100)
        #single load, choose the location with xc:
        y = point_load_static_defl_shape_function(x, 1, xc=1)
        ax.plot(x, y ,label='single load')

        ax.set_xlabel("x/L_c")
        ax.set_ylabel("relative w")
        ax.invert_yaxis()

        #load centers equally spaced, with equal magnitude of 1:
        for n in [3]:
            xx,yy = multi_static(xc=np.arange(n)*3, frel=[1]*n)
            ax.plot(xx,yy, label="{} loads".format(n))


        ax.grid()
        leg = ax.legend()
        plt.show()


    if 0:
        #messing around with moving point loads
        a = MovingPointLoads([0.,-3,-4,-5],[20.,15,10,10 ])
        a.plotme(ax=None,t=0, v0=0, t0=0, x0=0, xlim = (-10,10), ylim=(0,100))
        plt.show()#x0,t0,tf,v

    if 0:
        #messing around with moving point loads
        a = MovingPointLoads([0.,-3,-4,-5],[20.,15,10,10 ])
#        a.animateme(x0=-5,t0=0,tf=15,v0=6, xlim = (-10,20), ylim=(0,100))
        a.animateme(x0=-5,t0=0,tf=15,v0=6, xlim = (-10,20), ylim=(0,100))
        plt.show()#x0,t0,tf,v

    if 0:
        #messing around with moving point loads
        a = MovingPointLoads([0.,-3,-4,-5],[20.,15,10,10 ])
        a.animateme(x0=20,t0=0,tf=15,v0=-6, xlim = (-10,20), ylim=(0,100))
        plt.show()#x0,t0,tf,v

    if 0:
        #messing around with moving point loads
        a = MovingPointLoads([0.,-3,-4,-5],[20.,15,10,10 ])

        a.plot_specbeam_PolyLines(ax=None, v=-2, L=15, t0=0)

        plt.show()#x0,t0,tf,v
    if 1:
        #messing around with moving point loads
        a = MovingPointLoads(x=[0.,-3,-4,-5],p=[20.,15,10,10 ],v=3,L=10,t0=0)
        fig, ax = plt.subplots()
        for t in [0,2,3,6,12]:
            x,p = a.point_loads_on_beam(t)

            ax.plot(x,p,marker='o',linestyle="None",label= "{}".format(t))
        ax.set_title("L={},v={}".format(a.L,a.v))
        ax.set_ylim(0,100)
        plt.show()
    if 0:
        #messing around with moving point loads
        a = MovingPointLoads(x=[0.,-3,-4,-5],p=[20.,15,10,10 ],v=-3,L=10,t0=0)
        fig, ax = plt.subplots()
        for t in [0,2,3,6,12]:
            x,p = a.point_loads_on_beam(t)

            ax.plot(x,p,marker='o',linestyle="None",label= "{}".format(t))
        ax.set_title("L={},v={}".format(a.L,a.v))
        ax.set_ylim(0,100)
        plt.show()

############################################
# Misc code that Rohan has forgotten the use
# of but was no doubt useful at the time.
############################################

    if 0:
        #k3!=0 so use the numberical, not sure
        start_time0 = time.time()
        ftime = datetime.datetime.fromtimestamp(start_time0).strftime('%Y-%m-%d %H%M%S')
        with open(ftime + ".txt", "w") as f:
            f.write(ftime + os.linesep)

            fig=plt.figure()
            ax = fig.add_subplot("111")
            ax.set_xlabel("time, s")
            ax.set_ylabel("w, m")
            ax.set_xlim(3.5, 4.5)

            for v in [50,75, ]:
                vstr="nterms"
                vfmt="{}"
                start_time1 = time.time()

                f.write("*" * 70 + os.linesep)
                vequals = "{}={}".format(vstr, vfmt.format(v))
                f.write(vequals + os.linesep); print(vequals)

                pdict = OrderedDict(
                    E = 6.998*1e9, #Pa
                    rho = 2373, #kg/m3
                    L = 160, #m
                    v_norm=0.01165,
                    kf=5.41e-4,
                    Fz_norm=1.013e-4,
                    mu_norm=39.263,
                    k1_norm=97.552,
                    k3_norm=2.497e6,
                    nterms=v,
                    BC="CC",
                    nquad=20,
                    )

                t = np.linspace(0, 4.5, 400)

                f.write(repr(pdict) + os.linesep)

                for BC in ["SS"]:# "CC", "FF"]:
                    pdict["BC"] = BC

                    a = SpecBeam(**pdict)
                    a.calulate_qk(t=t)

                    x = t
                    y = a.wofx(x_norm=0.5, normalise_w=False)

                    ax.plot(x, y, label="x=0.5, {}".format(vequals))

                end_time1 = time.time()
                elapsed_time = (end_time1 - start_time1); print("Run time={}".format(str(timedelta(seconds=elapsed_time))))

                f.write("Run time={}".format(str(timedelta(seconds=elapsed_time))) + os.linesep)

            leg = ax.legend()
            leg.draggable()

            end_time0 = time.time()
            elapsed_time = (end_time0 - start_time0); print("Total run time={}".format(str(timedelta(seconds=elapsed_time))))
            f.write("Total run time={}".format(str(timedelta(seconds=elapsed_time))) + os.linesep)
        plt.savefig(ftime+".pdf")
        plt.show()


    if 0:
        #Journal paper figure, convergence with step change
        FIGURE_convergence_step_change(v_nterms=(50,75,150,200),saveas="convergence_step_change")#75,150,200))
    if 0:
        #Journal paper figure, vary velocitiesconvergence with step change
#        FIGURE_convergence_step_change(v_nterms=(50,75,150),saveas="convergence_step_change")#75,150,200))
        FIGURE_defl_envelope_vary_velocity(
            velocities_norm=(0.01165/2, 0.01165, 0.01165*2, 0.01165*4, 0.01165*10/1.13),
            saveas=None,
            nterms=150,
            nx=200, nt=200,
            force_calc=True)


    if 0:
        #Journal paper figure, length of transition zone
        FIGURE_DAF_transition_length(saveas=None,k_ratio=4,beta=0.1,nterms=100,force_calc=True)
    if 0:
        FIGURE_DAF_transition_length_esveld(saveas=None,k_ratio=40,beta=0.05,nterms=100,force_calc=True)
    if 0:
        betas=[0.05, 0.1, 0.5, 1, 1.3]
        kratios=[1,2,5,10,20,100,1000,10000]

        betas=[1, 1.3]
        kratios=[1,2,5,10,20,100,1000,10000]

#        betas=[0.05]
#        kratios=[40]

        for beta in betas:
            for kratio in kratios:
                saveas="DAFwrtinf_t_len_bet{}_krat{}".format(beta,kratio)
#                saveas=None
                FIGURE_DAFwrtinf_transition_length(
                    saveas=saveas,k_ratio=kratio,beta=beta,nterms=150,force_calc=False)

    if 0:
        betas=[0.05, 0.1, 0.5, 1, 1.3]
        kratios=[1,2,5,10,20,100,1000,10000]

        betas=[1, 1.3]
#        kratios=[1,2,5,10,20,100,1000,10000]

#        betas=[0.05]
#        kratios=[40]

        for beta in betas:
            for kratio in kratios:
                saveas="revDAFwrtinf_t_len_bet{}_krat{}".format(beta,kratio)
#                saveas=None
                FIGURE_DAFwrtinf_transition_lengthrev(
                    saveas=saveas,k_ratio=kratio,beta=beta,nterms=150,force_calc=False)

