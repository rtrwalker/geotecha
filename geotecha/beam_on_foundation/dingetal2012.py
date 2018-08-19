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
Ding et al (2012) "Convergence of Galerkin truncation for dynamic response 
of finite beams on nonlinear foundations under a moving load".

"""

from __future__ import division, print_function


import numpy as np
import matplotlib.pyplot as plt
from geotecha.mathematics.root_finding import find_n_roots
from scipy import integrate
from scipy.integrate import odeint

import time
from datetime  import timedelta
import datetime
from collections import OrderedDict
import os


class DingEtAl2012(object):
    """Finite elastic Euler-Bernoulli beam resting on non-linear 
    viscoelastic foundation subjected to a moving load.
    
    An implementation of Ding et al. (2012) [1]_.


    You don't need all the parameters.  Basically if normalised values are
    equal None (i.e. default) then those properties will be calculated from 
    the non normalised quantities.  All calculations are done in normalised
    space.  You only need the non-normalised variables if you want 
    non-normalised output.
    

    Parameters
    ----------
    BC : ["SS", "CC", "FF"], optional
        Boundary condition. Simply Supported (SS), Clamped Clamped ("CC"), 
        Free Free (FF). Default BC="SS".
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
    Fz : float, optional
        Load.
    v : float, optional
        Speed of the moving force.        
    v_norm : float, optional
        normalised velocity = V * sqrt(rho / E). An example of a consistent 
        set of units to get the correct v_norm is rho in kg/m^3, L in m, and 
        E in Pa.
    kf : float, optional
        Normalised modulus of elasticity = 1 / L * sqrt(I / A).
    Fz_norm : float, optional
        Normalised load = Fz / (E * A).
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
        

    References
    ----------
    .. [1] Ding, H., Chen, L.-Q., and Yang, S.-P. (2012). 
           "Convergence of Galerkin truncation for dynamic response of 
           finite beams on nonlinear foundations under a moving load." 
           Journal of Sound and Vibration, 331(10), 2426-2442.

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
                    Fz=None,
                    v=None, 
                    v_norm=None,
                    kf=None,
                    Fz_norm=None,
                    mu_norm=None,
                    k1_norm=None,
                    k3_norm=None,
                    nquad=30):
                    
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
        self.Fz = Fz
        self.v = v
        self.v_norm = v_norm
        self.kf = kf
        self.Fz_norm = Fz_norm
        self.mu_norm = mu_norm
        self.k1_norm = k1_norm
        self.k3_norm = k3_norm
        self.nquad = nquad

        # normalised parameters
        if kf is None:
            self.kf = 1 / self.L * np.sqrt(self.I / self.A)
        if v_norm is None:
            self.v_norm = self.v * np.sqrt(self.rho / self.E)            
        if self.kf is None:
            self.kf = 1 / self.L * np.sqrt(self.I / self.A)
        if Fz_norm is None:
            self.Fz_norm = self.Fz / (self.E * self.A)
        
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
            
            if not self.nquad is None:
                self.xj = 0.5 * (1 - np.cos(np.pi * np.arange(self.nquad) / (self.nquad - 1)))            
            
            self.BC_coeff = 2
        
        elif self.BC == "CC" or self.BC == "FF":
            
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
        """Nomalised vertcal deformation at x

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

        for i in range(self.nterms):
            self.vsvdot[i] = qdot[i]
            self.vsvdot[self.nterms + i] = - self.mu_norm * qdot[i]

            self.vsvdot[self.nterms + i] -= (self.k1_norm + self.kf**2 * self.beta[i]**4) * q[i]

            self.vsvdot[self.nterms + i] += self.BC_coeff * self.Fz_norm * self.phi(self.v_norm * tnow, self.beta[i])


            if 1:
                # DIY quadrature
                Fj = np.sum(q[:,None] * self.phi(self.xj[None, :], self.beta[:,None]), axis=0)**3
                y = np.sum(self.Ij * Fj * self.phi(self.xj, self.beta[i]))
            else:
                # scipy
#                print("yquad = {:12.2g}".format(y))
                y, err = integrate.quad(self.w_cubed_wi, 0, 1, args=(q,i))
#                print("yscip = {:12.2g}".format(y))            
            #maybe the quad integrate is not great for the oscillating function
            self.vsvdot[self.nterms + i] -= self.BC_coeff * self.k3_norm * y

        return self.vsvdot


    def w_cubed_wi(self, x, q, i):
        """non-linear cube term for numerical integration"""
        
        return self.w(q, x)**3 * self.phi(x, self.beta[i])


    def calulate_qk(self, t=None, t_norm=None, **odeint_kwargs):
        """Calculate the nterm Galerkin  coefficients qk at each time value

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

        self.qsol = odeint(self.vectorfield,
                      vsv0,
                      self.t_norm,
                      args=(),
                      **odeint_kwargs)


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
        w : array of float
            Deflections at x and self.t_norm[tslice]
            
        """
            
        if x_norm is None:
            x_norm = x / self.L

        x_norm = np.atleast_1d(x_norm)
        
        v = np.zeros((len(x_norm), len(self.t_norm[tslice])))

        for i, xx in enumerate(x_norm):
            for j, qq in enumerate(self.qsol[tslice, :self.nterms]):
                v[i, j] = self.w(qq, xx)
        
        if not normalise_w is None:
            v *= self.L
            
        if len(x_norm)==1:
            return v[0] 
        else:        
            return v

def dingetal_figure_8():
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
        
        for v in [50, 75, 150, 200]:#[50, 75, 150, 200]   
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
                nquad=20,)
            
            t = np.linspace(0, 4.5, 400)
    
            f.write(repr(pdict) + os.linesep)
            
            for BC in ["SS"]:# "CC", "FF"]:
                pdict["BC"] = BC
    
                a = DingEtAl2012(**pdict)
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
                  
if __name__ == "__main__":
    pass
#    dingetal_figure_8()
        
        

    
