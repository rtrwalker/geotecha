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
This module has functions classes and common functionality for one dimensinal
Spectral Galerkin methods.

"""
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

import geotecha.inputoutput.inputoutput as inputoutput
import geotecha.piecewise.piecewise_linear_1d as pwise
from geotecha.piecewise.piecewise_linear_1d import PolyLine
import geotecha.speccon.integrals as integ



class Speccon1d(inputoutput.InputFileLoaderCheckerSaver):
    """Solve 1D parabolic partial differential equation using spectral method.

    Basically a base class to provide a broad template for one dimensional
    spectral method consolidation problems.

    See Also
    --------
    geotecha.inputoutput.InputFileLoaderCheckerSaver : Details on how to
        initialize the object and attribute checks.


    """

    def make_all(self):
        """Run checks, make all arrays, make output

        Generally run this after attributes have been entered

        See also
        --------
        check_input_attributes
        make_time_independent_arrays
        make_time_dependent_arrays
        make_output

        """
        self.check_input_attributes()
        self.make_time_independent_arrays()
        self.make_time_dependent_arrays()
        self.make_output()

        if getattr(self, 'save_data_to_file', False):
            self._save_data()
        if (getattr(self, 'save_figures_to_file', False) or
                getattr(self, 'show_figures', False)):
            self.produce_plots()
            if getattr(self, 'save_figures_to_file', False):
                self._save_figures()
            if getattr(self, 'show_figures', False):
                plt.show()

        return

    def make_time_independent_arrays(self):
        """Make all time-independent arrays; To be overridden in subclasses."""
        raise NotImplementedError("make_time_independent_arrays")

    def make_time_dependent_arrays(self):
        """Make all time-independent arrays; To be overridden in subclasses."""
        raise NotImplementedError("make_time_dependent_arrays")

    def make_output(self):
        """Make all output i.e. data tables; To be overridden in subclasses."""
        raise NotImplementedError("make_output")


def dim1sin_f(m,
              outz,
              tvals,
              v_E_Igamv_the,
              drn,
              top_vs_time=None,
              bot_vs_time=None,
              top_omega_phase=None,
              bot_omega_phase=None):
    """Assemble output u(Z,t) = phi * v_E_Igam_v_the + utop(t) * (1-Z) + ubot(t)*Z.

    Basically calculates the phi part for each outz value, then dot product
    with v_E_Igamv_the (which has elsewhere been calculated at each tvals
    value).  Then account for non-zero boundary conditions by
    adding utop(t)*(1-Z) and ubot(t)*Z parts for each outz, tvals pair.

    Use sin(m*Z) for the phi part.


    Parameters
    ----------
    m : ``list`` of ``float``
        Eigenvalues of BVP, the m in sin(m*Z). Generate with
        geotecha.speccon.m_from_sin_mx.
    outz : 1d numpy.ndarray
        Depths to evaluate at.
    tvals : 1d numpy.ndarray
        Time values to evaluate at
    eigs : 1d numpy.ndarray
        List of eigenvalues of the spectral matrix i.e. Eigenvalues of the
        square Igam_psi matrix.
    tvals : 1d numpy.ndarray`
        List of time values to evaluate E matrix at.
    v_E_Igamv_the : ndarray of size (neig, len(tvals))
        Speccon matrix.
    drn : [0,1]
        Drainage condition,
        drn=0 for Pervious top pervious bottom (PTPB).
        drn=1 for Pervious top impoervious bottom (PTIB).
    top_vs_time, bot_vs_time : list of PolyLine
        Piecewise linear magnitude vs time for the top and bottom boundary.
        Use ``None`` if there is no variation.
    top_omega_phase, bot_omega_phase : list of 2 element tuples, optional
        (omega, phase) for use in cos(omega * t + phase) * mag_vs_time
        if omega_phase is None then mag_vs_time will not be multiplied by a
        cosine.  If any element of omega_phase is None then in that particular
        loading combo, mag_vs_time will not be multiplied by a cosine.


    Returns
    -------
    u : np.ndarray
        Pore pressure at depth and time. Array of size (len(outz), len(tvals)).


    Notes
    -----
    The :math:`\\phi` term is simply: :math:`\\sin\\left({m Z}\\right)`
    evaluated at each required depth.



    """

    phi = integ.dim1sin(m, outz)
    u = np.dot(phi, v_E_Igamv_the)
    #top part
    if not top_vs_time is None:
        if top_omega_phase is None:
            top_omega_phase = [None] * len(top_vs_time)
        for mag_vs_time, om_ph in zip(top_vs_time, top_omega_phase):
            if not om_ph is None:
                omega, phase = om_ph
                mult = np.cos(omega * tvals + phase)
            else:
                mult = 1

            if drn==1:
                u += pwise.pinterp_x_y(mag_vs_time, tvals, choose_max=True) * mult
            else:
                u += pwise.pinterp_xa_ya_multipy_x1b_x2b_y1b_y2b(mag_vs_time, PolyLine([0], [1], [1], [0]), tvals, outz, achoose_max=True) * mult
    #bot part
    if not bot_vs_time is None:
        if bot_omega_phase is None:
            bot_omega_phase = [None] * len(bot_vs_time)
        for mag_vs_time, om_ph in zip(bot_vs_time, bot_omega_phase):
            if not om_ph is None:
                omega, phase = om_ph
                mult = np.cos(omega * tvals + phase)
            else:
                mult=1
            u += pwise.pinterp_xa_ya_multipy_x1b_x2b_y1b_y2b(mag_vs_time, PolyLine([0], [1], [0], [1]), tvals, outz, achoose_max=True) * mult
    return u


def dim1sin_avgf(m,
                 z,
                 tvals,
                 v_E_Igamv_the,
                 drn,
                 top_vs_time=None,
                 bot_vs_time=None,
                 top_omega_phase=None,
                 bot_omega_phase=None):
    """Average u(Z,t) between Z1 and Z2 where
    u(Z,t) = phi * v_E_Igam_v_the + utop(t) * (1-Z) + ubot(t)*Z.

    Basically calculates the average phi part for each z pair value, then dot product
    with v_E_Igamv_the (which has elsewhere been calculated at each tvals
    value).  Then account for non-zero boundary conditions by
    adding average of utop(t)*(1-Z) and ubot(t)*Z parts for each z, tvals pair.

    Use sin(m*Z) for the phi part.


    Parameters
    ----------
    m : ``list`` of ``float``
        Eigenvalues of BVP, the m in sin(m*Z). Generate with
        geotecha.speccon.m_from_sin_mx.
    z : size (n, 2) 2d numpy.ndarray
        Depths to evaluate average between.
    tvals : 1d numpy.ndarray
        Time values to evaluate at.
    eigs : 1d numpy.ndarray
        List of eigenvalues of the spectral matrix i.e. Eigenvalues of the
        square Igam_psi matrix.
    tvals : 1d numpy.ndarray`
        List of time values to evaluate E matrix at.
    v_E_Igamv_the : ndarray of size (neig, len(tvals))
        Speccon matrix.
    drn : [0,1]
        Drainage condition,
        drn=0 for Pervious top pervious bottom (PTPB).
        drn=1 for Pervious top impoervious bottom (PTIB).
    top_vs_time, bot_vs_time : list of PolyLine
        Piecewise linear magnitude vs time for the top and bottom boundary.
        Use ``None`` if there is no variation.
    top_omega_phase, bot_omega_phase : list of 2 element tuples, optional
        (omega, phase) for use in cos(omega * t + phase) * mag_vs_time
        if omega_phase is None then mag_vs_time will not be multiplied by a
        cosine.  If any element of omega_phase is None then in that particular
        loading combo, mag_vs_time will not be multiplied by a cosine.


    Returns
    -------
    uavg : np.ndarray
        Average pore pressure between depths at each time.
        Array of size (len(z), len(tvals)).


    Notes
    -----
    The average of the :math:`\\phi` term is:

    .. math:: \\mathbf{\\phi}_{i\\textrm{average}}=
                \\frac{1}{Z_2-Z_1}
                \\int_{z_1}^{z_2}{\\sin\\left({m_i Z}\\right)\\,dZ}



    """

    phi = integ.dim1sin_avg_between(m, z)


    avg = np.dot(phi, v_E_Igamv_the)

    z1 = np.asarray(z)[:,0]
    z2 = np.asarray(z)[:,1]

    #top part
    if not top_vs_time is None:
        if top_omega_phase is None:
            top_omega_phase = [None] * len(top_vs_time)
        for mag_vs_time, om_ph in zip(top_vs_time, top_omega_phase):
            if not om_ph is None:
                omega, phase = om_ph
                mult = np.cos(omega * tvals + phase)
            else:
                mult = 1
            if drn==1:
                #bottom part
                avg += pwise.pinterp_x_y(mag_vs_time, tvals, choose_max=True)*mult
            else:
                avg += pwise.pxa_ya_multipy_avg_x1b_x2b_y1b_y2b_between(mag_vs_time,
                                                                        PolyLine([0], [1], [1], [0]),
                                                                        tvals, z1, z2, achoose_max=True)*mult
    #bottom part
    if not bot_vs_time is None:
        if bot_omega_phase is None:
            bot_omega_phase = [None] * len(bot_vs_time)
        for mag_vs_time, om_ph in zip(bot_vs_time, bot_omega_phase):
            if not om_ph is None:
                omega, phase = om_ph
                mult = np.cos(omega * tvals + phase)
            else:
                mult=1
            avg += pwise.pxa_ya_multipy_avg_x1b_x2b_y1b_y2b_between(mag_vs_time,
                                                                        PolyLine([0], [1], [0], [1]),
                                                                        tvals, z1,z2, achoose_max=True) * mult

    return avg


def dim1sin_integrate_af(m,
                         z,
                         tvals,
                         v_E_Igamv_the,
                         drn,
                         a,
                         top_vs_time=None,
                         bot_vs_time=None,
                         top_omega_phase=None,
                         bot_omega_phase=None):
    """Integrate u(Z,t) between Z1 and Z2 where
    u(Z,t) = phi * v_E_Igam_v_the + utop(t) * (1-Z) + ubot(t)*Z.

    Basically calculates the integral phi part for each z pair value, then dot product
    with v_E_Igamv_the (which has elsewhere been calculated at each tvals
    value).  Then account for non-zero boundary conditions by
    adding average of utop(t)*(1-Z) and ubot(t)*Z parts for each z, tvals pair.

    Use sin(m*Z) for the phi part.


    Parameters
    ----------
    m : ``list`` of ``float``
        Eigenvalues of BVP, the m in sin(m*Z). Generate with
        geotecha.speccon.m_from_sin_mx.
    z : size (n, 2) 2d numpy.ndarray
        Depths to evaluate integral between.
    tvals : 1d numpy.ndarray
        Time values to evaluate at.
    eigs : 1d numpy.ndarray
        List of eigenvalues of the spectral matrix i.e. Eigenvalues of the
        square Igam_psi matrix.
    tvals : 1d numpy.ndarray`
        List of time values to evaluate E matrix at.
    v_E_Igamv_the : ndarray of size (neig, len(tvals))
        Speccon matrix.
    drn : [0,1]
        Drainage condition,
        drn=0 for Pervious top pervious bottom (PTPB).
        drn=1 for Pervious top impoervious bottom (PTIB).
    top_vs_time, bot_vs_time : list of PolyLine
        Piecewise linear magnitude vs time for the top and bottom boundary.
        Use ``None`` if there is no variation.
    top_omega_phase, bot_omega_phase : list of 2 element tuples, optional
        (omega, phase) for use in cos(omega * t + phase) * mag_vs_time
        if omega_phase is None then mag_vs_time will not be multiplied by a
        cosine.  If any element of omega_phase is None then in that particular
        loading combo, mag_vs_time will not be multiplied by a cosine.


    Returns
    -------
    uintegral : np.ndarray
        Integral of pore pressure between depths at each time.
        Array of size (len(z), len(tvals)).


    Notes
    -----
    The integral of the :math:`\\phi` term is:

    .. math:: \\mathbf{\\phi}_{i\\textrm{integral}}=
                \\int_{z_1}^{z_2}{\\sin\\left({m_i Z}\\right)\\,dZ}




    """


    z1 = np.array(z)[:,0]
    z2 = np.array(z)[:,1]
    #a*u part
    phi = integ.pdim1sin_a_linear_between(m, a, z)

    out = np.dot(phi, v_E_Igamv_the)

    #top part
    if not top_vs_time is None:
        if top_omega_phase is None:
            top_omega_phase = [None] * len(top_vs_time)
        for mag_vs_time, om_ph in zip(top_vs_time, top_omega_phase):
            if not om_ph is None:
                omega, phase = om_ph
                mult = np.cos(omega * tvals + phase)
            else:
                mult = 1
            if drn==1:
                out += mult * pwise.pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(
                    mag_vs_time,
                    a,
                    PolyLine(a.x1, a.x2, np.ones_like(a.x1), np.ones_like(a.x2)),
                    tvals, z1, z2, achoose_max=True)
            else:
                out += mult * pwise.pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(
                    mag_vs_time,
                    a,
                    PolyLine(a.x1, a.x2, 1-a.x1, 1-a.x2),
                    tvals, z1, z2, achoose_max=True)


    #bot part
    if not bot_vs_time is None:
        if bot_omega_phase is None:
            bot_omega_phase = [None] * len(bot_vs_time)
        for mag_vs_time, om_ph in zip(bot_vs_time, bot_omega_phase):
            if not om_ph is None:
                omega, phase = om_ph
                mult = np.cos(omega * tvals + phase)
            else:
                mult=1
            out += mult * pwise.pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(
                    mag_vs_time,
                    a,
                    PolyLine(a.x1, a.x2, a.x1, a.x2),
                    tvals, z1, z2, achoose_max=True)
    #self.set *= self.H * self.mvref
    return out


def dim1sin_E_Igamv_the_BC_aDfDt_linear(drn,
                                        m,
                                        eigs,
                                        tvals,
                                        Igamv,
                                        a,
                                        top_vs_time,
                                        bot_vs_time,
                                        top_omega_phase=None,
                                        bot_omega_phase=None,
                                        dT=1.0,
                                        theta_zero_indexes=None,
                                        implementation='vectorized'):
    """Calculate E and theta parts and assemble E_Igamv_the matrix that arises
    from homogenising a(Z)*D[u(Z, t), t] for non_zero top and bottom boundary
    conditions.

    When accounting for non-zero boundary conditions we homogenise the
    governing equation by letting u(Z,t) = v(Z,t) + utop(t)*(1-Z) + ubot(t)*Z
    and solving for v(Z, t).  This function calculates the time dependent
    E part, the depth dependent theta part, and then assembles
    E*inverse(gam*v)*theta which forms part of solution
    v(Z,t)=phi*v*E*inverse(gam*v)*theta.  The E and theta parts arise
    by subbing the boundary conditions into into governing equation terms of
    the form a(z)*D[u(Z,t), t].

    The contribution of each `mag_vs_time`-`omega_phase` pairing
    are superposed.    The result is an array
    of size (neig, len(tvals)). So each column is the are the column vector
    E*inverse(gam*v)*theta calculated at each output time.  This will allow
    us later to do v(Z,t) = phi*v*E_Igamv_the.

    Uses sin(m*z) in the calculation of theta.

    Parameters
    ----------
    drn : [0,1]
        Drainage condition,
        drn=0 for Pervious top pervious bottom (PTPB).
        drn=1 for Pervious top impoervious bottom (PTIB).
    m : ``list`` of ``float``
        Eigenvalues of BVP, the m in sin(m*Z). Generate with
        geotecha.speccon.m_from_sin_mx.
    eigs : 1d numpy.ndarray
        List of eigenvalues of the spectral matrix i.e. Eigenvalues of the
        square Igam_psi matrix.
    tvals : 1d numpy.ndarray`
        List of time values to evaluate E matrix at.
    Igamv : ndarray
        Speccon matrix.  Igamv = inverse of [gam * v])
    a : PolyLine
        Piewcewise linear function.  e.g. for 1d consolidation surcharge
        loading term is mv*D[sigma(z, t), t] so `a` would be mv.
    top_vs_time, bot_vs_time : list of PolyLine
        Piecewise linear magnitude vs time for the top and bottom boundary.
        Use ``None`` if there is no variation.
    top_omega_phase, bot_omega_phase : list of 2 element tuples, optional
        (omega, phase) for use in cos(omega * t + phase) * mag_vs_time
        if omega_phase is None then mag_vs_time will not be multiplied by a
        cosine.  If any element of omega_phase is None then in that particular
        loading combo, mag_vs_time will not be multiplied by a cosine.
    dT : ``float``, optional
        Time factor multiple for numerical convieniece. Default dT=1.0.
    theta_zero_indexes : slice/list etc., optional
        A slice object, list, etc that can be used for numpy fancy indexing.
        Any specified index of the theta vector will be set to zero.  This is
        useful when using the spectral method with block matrices and the
        loading term only refers to a subset of the equations.  When using
        block matrices m should be the same size as the block matrix.
        Default theta_zero_indexes=None i.e. no elements of theta will be
        set to zero.

    Returns
    -------
    E_Igamv_the : ndarray
        Loading matrix of size (neig, len(tvals)).

    Notes
    -----
    Assuming the loads are formulated as the product of separate time and depth
    dependant functions as well as a cyclic component:

    .. math:: \\sigma\\left({Z,t}\\right)=
                \\sigma\\left({Z}\\right)
                \\sigma\\left({t}\\right)
                \\cos\\left(\\omega t + \\phi\\right)

    the solution to the consolidation equation using the spectral method has
    the form:

    .. math:: u\\left(Z,t\\right)=
                \\mathbf{\\phi v E}
                \\left(\\mathbf{\\Gamma v}\\right)^{-1}
                \\mathbf{\\theta}


    When we consider non-zero boundary conditions, additional loading terms are
    created when we sub in the following into the original governing equation.

    .. math:: u\\left({Z,t}\\right)=
                v\\left({Z,t}\\right) +
                u_{top}\\left({t}\\right)\\left({1-Z}\\right) +
                u_{bot}\\left({b}\\right)Z

    Two additional loading terms are created with each substitution, one
    for the top boundary condition and one for the bottom boundary condition.

    This function calculates :math:`\\mathbf{E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}`
    when substitutions are made in terms of the following form:


    .. math:: a\\left({Z}\\right)\\frac{\\partial u}{\\partial t}

    It is assumed that :math:`u_{top}\\left({t}\\right)` and
    :math:`u_{bot}\\left({t}\\right)` are piecewise linear
    in time with a cyclic component, and that multiple functions are
    superposed.  Also :math:`a\\left(Z\\right)`
    is a piecewise linear function w.r.t. :math:`Z`


    For this particular function the :math:`\\mathbf{\\theta}` vector for
    each load is given by:

    .. math:: \\mathbf{\\theta}_{i}=
                \\int_{0}^1{
                  {a\\left(Z\\right)}
                  {\\sigma\\left(Z\\right)}
                  f\\left({Z}\\right)
                  \\phi_i\\,dZ}

    Where :math:`f\\left({Z}\\right)` is the appropriate z-dependent term
    corresponding to either :math:`u_{top}` or :math:`u_{bot}` homogenisations.


    The :math:`\\mathbf{E}` matrix for each load is given by:

    .. math:: \\mathbf{E}_{i,j}=
                \\int_{0}^{t_j}{
                  \\frac{d{
                    {\\cos\\left(\\omega\\tau+\\textrm{phase}\\right)}
                    \\sigma\\left(\\tau\\right)}}
                    {d\\tau}
                  {\\exp\\left({(dT\\left(t-\\tau\\right)\\lambda_i}\\right)}
                  \\,d\\tau}


    where

     - :math:`\\lambda_i` is the `ith` eigenvalue of the problem,
     - :math:`dT` is a time factor for numerical convienience,
     - :math:`\\sigma\left(\\tau\\right)` is the piecewise linear time
       dependant load.


    Note that the listed equations above are in terms of normalised depth Z,
    with depth integrations between [0, 1].  However, IF YOU KNOW WHAT YOU
    ARE DOING the integrations can be done using non-normalised depths.
    The first z value in the piecewise definition a(z) must still be 0
    however the end point for integration will be the final z value in the
    definition of a(z).  If you are doing this then your `m` values will
    include the normalising Factor. e.g. m = [pi/2/H, 3*pi/2/H] and a(z) is
    defined in two layers [0, z1], [z1, zend] as opposed to
    m = [pi/2, 3*pi/2] and a(Z) is two layers [0, z1/H], [z1/H, zend/H].

    """

    E_Igamv_the = np.zeros((len(m), len(tvals)))




    if not a is None:
        if drn==1:
            zdist = PolyLine(a.x1,a.x2, np.ones_like(a.x1), np.ones_like(a.x2))
            #bot_vs_time=None
        else:
            zdist = PolyLine(a.x1,a.x2, a.x2[-1]-a.x1, a.x2[-1]-a.x2)


        if not top_vs_time is None:
            if top_omega_phase is None:
                top_omega_phase = [None] * len(top_vs_time)

            theta = integ.pdim1sin_ab_linear(m, a, zdist)
            if not theta_zero_indexes is None:
                theta[theta_zero_indexes] = 0.0
            for top_vs_t, om_ph in zip(top_vs_time, top_omega_phase):
                if not om_ph is None:
                    omega, phase = om_ph
                    E = integ.pEDload_coslinear(top_vs_t, omega, phase, eigs, tvals, dT, implementation=implementation)
                else:
                    E = integ.pEDload_linear(top_vs_t, eigs, tvals, dT, implementation=implementation)
                E_Igamv_the += (E*np.dot(Igamv, theta)).T


        if not bot_vs_time is None:
            if bot_omega_phase is None:
                bot_omega_phase = [None] * len(bot_vs_time)

            theta = integ.pdim1sin_ab_linear(m, a, PolyLine(a.x1,a.x2,a.x1,a.x2))
            if not theta_zero_indexes is None:
                theta[theta_zero_indexes] = 0.0
            for bot_vs_t, om_ph in zip(bot_vs_time, bot_omega_phase):
                if not om_ph is None:
                    omega, phase = om_ph
                    E = integ.pEDload_coslinear(bot_vs_t, omega, phase, eigs, tvals, dT, implementation=implementation)
                else:
                    E = integ.pEDload_linear(bot_vs_t, eigs, tvals, dT, implementation=implementation)
                E_Igamv_the += (E*np.dot(Igamv, theta)).T

    #theta is 1d array, Igamv is nieg by neig array, np.dot(Igamv, theta)
    #and np.dot(theta, Igamv) will give differetn 1d arrays.
    #Basically np.dot(Igamv, theta) gives us what we want i.e.
    #theta was treated as a column array.  The alternative
    #np.dot(theta, Igamv) would have treated theta as a row vector.
    return E_Igamv_the


def dim1sin_E_Igamv_the_BC_abf_linear(drn,
                                      m,
                                      eigs,
                                      tvals,
                                      Igamv,
                                      a,
                                      b,
                                      top_vs_time=None,
                                      bot_vs_time=None,
                                      top_omega_phase=None,
                                      bot_omega_phase=None,
                                      dT=1.0,
                                      theta_zero_indexes=None,
                                      implementation='vectorized'):
    """Calculate E and theta parts and assemble E_Igamv_the matrix that arises
    from homogenising a(Z)*b(Z)*u(Z,t) for non_zero top and bottom boundary
    conditions.

    When accounting for non-zero boundary conditions we homogenise the
    governing equation by letting u(Z,t) = v(Z,t) + utop(t)*(1-Z) + ubot(t)*Z
    and solving for v(Z, t).  This function calculates the time dependent
    E part, the depth dependent theta part, and then assembles
    E*inverse(gam*v)*theta which forms part of solution
    v(Z,t)=phi*v*E*inverse(gam*v)*theta.  The E and theta parts arise
    by subbing the boundary conditions into into governing equation terms of
    the form a(z)*b(z)*u(Z,t)

    The contribution of each `mag_vs_time`-`omega_phase` pairing
    are superposed.    The result is an array
    of size (neig, len(tvals)). So each column is the are the column vector
    E*inverse(gam*v)*theta calculated at each output time.  This will allow
    us later to do v(Z,t) = phi*v*E_Igamv_the.

    Uses sin(m*z) in the calculation of theta.

    Parameters
    ----------
    drn : [0,1]
        Drainage condition,
        drn=0 for Pervious top pervious bottom (PTPB).
        drn=1 for Pervious top impoervious bottom (PTIB).
    m : ``list`` of ``float``
        Eigenvalues of BVP, the m in sin(m*Z). Generate with
        geotecha.speccon.m_from_sin_mx.
    eigs : 1d numpy.ndarray
        List of eigenvalues of the spectral matrix i.e. Eigenvalues of the
        square Igam_psi matrix.
    tvals : 1d numpy.ndarray`
        List of time values to evaluate E matrix at.
    Igamv : ndarray
        Speccon matrix.  Igamv = inverse of [gam * v])
    a, b : PolyLine
        Piewcewise linear function.  e.g. for 1d consolidation surcharge
        radial draiange term is dTh*kh*et*u(Z,t) `a` would be kh, `b` would
        be et.
    top_vs_time, bot_vs_time : list of PolyLine
        Piecewise linear magnitude vs time for the top and bottom boundary.
        Use ``None`` if there is no variation.
    top_omega_phase, bot_omega_phase : list of 2 element tuples, optional
        (omega, phase) for use in cos(omega * t + phase) * mag_vs_time
        if omega_phase is None then mag_vs_time will not be multiplied by a
        cosine.  If any element of omega_phase is None then in that particular
        loading combo, mag_vs_time will not be multiplied by a cosine.
    dT : ``float``, optional
        Time factor multiple for numerical convieniece. Default dT=1.0.
    theta_zero_indexes : slice/list etc., optional
        A slice object, list, etc that can be used for numpy fancy indexing.
        Any specified index of the theta vector will be set to zero.  This is
        useful when using the spectral method with block matrices and the
        loading term only refers to a subset of the equations.  When using
        block matrices m should be the same size as the block matrix.
        Default theta_zero_indexes=None i.e. no elements of theta will be
        set to zero.

    Returns
    -------
    E_Igamv_the : ndarray
        Loading matrix of size (neig, len(tvals)).

    Notes
    -----
    Assuming the loads are formulated as the product of separate time and depth
    dependant functions as well as a cyclic component:

    .. math:: \\sigma\\left({Z,t}\\right)=
                \\sigma\\left({Z}\\right)
                \\sigma\\left({t}\\right)
                \\cos\\left(\\omega t + \\phi\\right)

    the solution to the consolidation equation using the spectral method has
    the form:

    .. math:: u\\left(Z,t\\right)=
                \\mathbf{\\phi v E}
                \\left(\\mathbf{\\Gamma v}\\right)^{-1}
                \\mathbf{\\theta}


    When we consider non-zero boundary conditions, additional loading terms are
    created when we sub in the following into the original governing equation.

    .. math:: u\\left({Z,t}\\right)=
                v\\left({Z,t}\\right) +
                u_{top}\\left({t}\\right)\\left({1-Z}\\right) +
                u_{bot}\\left({b}\\right)Z

    Two additional loading terms are created with each substitution, one
    for the top boundary condition and one for the bottom boundary condition.

    This function calculates :math:`\\mathbf{E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}`
    when substitutions are made in terms of the following form:


    .. math:: a\\left({Z}\\right)
              b\\left({Z}\\right)
              u\\left({Z,t}\\right)

    It is assumed that :math:`u_{top}\\left({t}\\right)` and
    :math:`u_{bot}\\left({t}\\right)` are piecewise linear
    in time with a cyclic component, and that multiple functions are
    superposed.  Also :math:`a\\left(Z\\right)` and :math:`b\\left(Z\\right)`
    are piecewise linear functions with respect to :math:`Z`


    For this particular function the :math:`\\mathbf{\\theta}` vector for
    each load is given by:

    .. math:: \\mathbf{\\theta}_{i}=
                \\int_{0}^1{
                  {a\\left(Z\\right)}
                  {b\\left(Z\\right)}
                  {\\sigma\\left(Z\\right)}
                  f\\left({Z}\\right)
                  \\phi_i\\,dZ}

    Where :math:`f\\left({Z}\\right)` is the appropriate z-dependent term
    corresponding to either :math:`u_{top}` or :math:`u_{bot}` homogenisations.


    The :math:`\\mathbf{E}` matrix for each load is given by:

    .. math:: \\mathbf{E}_{i,j}=
                \\int_{0}^{t_j}{
                  {\\cos\\left(\\omega\\tau+\\textrm{phase}\\right)}
                  {\\sigma\\left(\\tau\\right)}
                  {\\exp\\left({(dT\\left(t-\\tau\\right)\\lambda_i}\\right)}
                  \\,d\\tau}


    where

     - :math:`\\lambda_i` is the `ith` eigenvalue of the problem,
     - :math:`dT` is a time factor for numerical convienience,
     - :math:`\\sigma\left(\\tau\\right)` is the piecewise linear time
       dependant load.


    Note that the listed equations above are in terms of normalised depth Z,
    with depth integrations between [0, 1].  However, IF YOU KNOW WHAT YOU
    ARE DOING the integrations can be done using non-normalised depths.
    The first z value in the piecewise definition a(z) must still be 0
    however the end point for integration will be the final z value in the
    definition of a(z).  If you are doing this then your `m` values will
    include the normalising Factor. e.g. m = [pi/2/H, 3*pi/2/H] and a(z) is
    defined in two layers [0, z1], [z1, zend] as opposed to
    m = [pi/2, 3*pi/2] and a(Z) is two layers [0, z1/H], [z1/H, zend/H].


    """

    E_Igamv_the = np.zeros((len(m), len(tvals)))

    if sum([v is None for v in [a, b]]) == 0:
        a, b = pwise.polyline_make_x_common(a, b)
        if drn==1:
            zdist = PolyLine(a.x1,a.x2, np.ones_like(a.x1), np.ones_like(a.x2))
            #bot_vs_time=None
        else:
            zdist = PolyLine(a.x1,a.x2, a.x2[-1]-a.x1, a.x2[-1]-a.x2)

        if not top_vs_time is None:
            if top_omega_phase is None:
                top_omega_phase = [None] * len(top_vs_time)

            theta = integ.pdim1sin_abc_linear(m, a,b, zdist)
            if not theta_zero_indexes is None:
                theta[theta_zero_indexes] = 0.0
            for top_vs_t, om_ph in zip(top_vs_time, top_omega_phase):
                if not om_ph is None:
                    omega, phase = om_ph
                    E = integ.pEload_coslinear(top_vs_t, omega, phase, eigs, tvals, dT, implementation=implementation)
                else:
                    E = integ.pEload_linear(top_vs_t, eigs, tvals, dT, implementation=implementation)
                E_Igamv_the += (E*np.dot(Igamv, theta)).T

        if not bot_vs_time is None:
            if bot_omega_phase is None:
                bot_omega_phase = [None] * len(bot_vs_time)
            theta = integ.pdim1sin_abc_linear(m, a, b, PolyLine(a.x1,a.x2,a.x1,a.x2))
            if not theta_zero_indexes is None:
                theta[theta_zero_indexes] = 0.0
            for bot_vs_t, om_ph in zip(bot_vs_time, bot_omega_phase):
                if not om_ph is None:
                    omega, phase = om_ph
                    E = integ.pEload_coslinear(bot_vs_t, omega, phase, eigs, tvals, dT, implementation=implementation)
                else:
                    E = integ.pEload_linear(bot_vs_t, eigs, tvals, dT, implementation=implementation)
                E_Igamv_the += (E*np.dot(Igamv, theta)).T

    #theta is 1d array, Igamv is nieg by neig array, np.dot(Igamv, theta)
    #and np.dot(theta, Igamv) will give differetn 1d arrays.
    #Basically np.dot(Igamv, theta) gives us what we want i.e.
    #theta was treated as a column array.  The alternative
    #np.dot(theta, Igamv) would have treated theta as a row vector.
    return E_Igamv_the


def dim1sin_E_Igamv_the_BC_abDfDt_linear(drn,
                                         m,
                                         eigs,
                                         tvals,
                                         Igamv,
                                         a,
                                         b,
                                         top_vs_time=None,
                                         bot_vs_time=None,
                                         top_omega_phase=None,
                                         bot_omega_phase=None,
                                         dT=1.0,
                                         theta_zero_indexes=None,
                                         implementation='vectorized'):
    """Calculate E and theta parts and assemble E_Igamv_the matrix that arises
    from homogenising a(Z)*b(Z)*D[u(Z, t), t] for non_zero top and bottom
    boundary conditions.

    When accounting for non-zero boundary conditions we homogenise the
    governing equation by letting u(Z,t) = v(Z,t) + utop(t)*(1-Z) + ubot(t)*Z
    and solving for v(Z, t).  This function calculates the time dependent
    E part, the depth dependent theta part, and then assembles
    E*inverse(gam*v)*theta which forms part of solution
    v(Z,t)=phi*v*E*inverse(gam*v)*theta.  The E and theta parts arise
    by subbing the boundary conditions into into governing equation terms of
    the form a(Z)*b(Z)*D[u(Z,t), t].

    The contribution of each `mag_vs_time`-`omega_phase` pairing
    are superposed.    The result is an array
    of size (neig, len(tvals)). So each column is the are the column vector
    E*inverse(gam*v)*theta calculated at each output time.  This will allow
    us later to do v(Z,t) = phi*v*E_Igamv_the.

    Uses sin(m*Z) in the calculation of theta.

    Parameters
    ----------
    drn : [0,1]
        Drainage condition,
        drn=0 for Pervious top pervious bottom (PTPB).
        drn=1 for Pervious top impoervious bottom (PTIB).
    m : ``list`` of ``float``
        Eigenvalues of BVP, the m in sin(m*Z). Generate with
        geotecha.speccon.m_from_sin_mx.
    eigs : 1d numpy.ndarray
        List of eigenvalues of the spectral matrix i.e. Eigenvalues of the
        square Igam_psi matrix.
    tvals : 1d numpy.ndarray`
        List of time values to evaluate E matrix at.
    Igamv : ndarray
        Speccon matrix.  Igamv = inverse of [gam * v])
    a, b : PolyLine
        Piewcewise linear function.  e.g. for 1d consolidation surcharge
        radial draiange term is dTh*kh*et*u(Z,t) `a` would be kh, `b` would
        be et.
    top_vs_time, bot_vs_time : list of PolyLine
        Piecewise linear magnitude vs time for the top and bottom boundary.
        Use ``None`` if there is no variation.
    top_omega_phase, bot_omega_phase : list of 2 element tuples, optional
        (omega, phase) for use in cos(omega * t + phase) * mag_vs_time
        if omega_phase is None then mag_vs_time will not be multiplied by a
        cosine.  If any element of omega_phase is None then in that particular
        loading combo, mag_vs_time will not be multiplied by a cosine.
    dT : ``float``, optional
        Time factor multiple for numerical convieniece. Default dT=1.0.
    theta_zero_indexes : slice/list etc., optional
        A slice object, list, etc that can be used for numpy fancy indexing.
        Any specified index of the theta vector will be set to zero.  This is
        useful when using the spectral method with block matrices and the
        loading term only refers to a subset of the equations.  When using
        block matrices m should be the same size as the block matrix.
        Default theta_zero_indexes=None i.e. no elements of theta will be
        set to zero.

    Returns
    -------
    E_Igamv_the : ndarray
        Loading matrix of size (neig, len(tvals)).

    Notes
    -----
    Assuming the loads are formulated as the product of separate time and depth
    dependant functions as well as a cyclic component:

    .. math:: \\sigma\\left({Z,t}\\right)=
                \\sigma\\left({Z}\\right)
                \\sigma\\left({t}\\right)
                \\cos\\left(\\omega t + \\phi\\right)

    the solution to the consolidation equation using the spectral method has
    the form:

    .. math:: u\\left(Z,t\\right)=
                \\mathbf{\\phi v E}
                \\left(\\mathbf{\\Gamma v}\\right)^{-1}
                \\mathbf{\\theta}


    When we consider non-zero boundary conditions, additional loading terms are
    created when we sub in the following into the original governing equation.

    .. math:: u\\left({Z,t}\\right)=
                v\\left({Z,t}\\right) +
                u_{top}\\left({t}\\right)\\left({1-Z}\\right) +
                u_{bot}\\left({b}\\right)Z

    Two additional loading terms are created with each substitution, one
    for the top boundary condition and one for the bottom boundary condition.

    This function calculates :math:`\\mathbf{E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}`
    when substitutions are made in terms of the following form:


    .. math:: a\\left({Z}\\right)
              b\\left({Z}\\right)
              \\frac{\\partial u}{\\partial t}

    It is assumed that :math:`u_{top}\\left({t}\\right)` and
    :math:`u_{bot}\\left({t}\\right)` are piecewise linear
    in time with a cyclic component, and that multiple functions are
    superposed.  Also :math:`a\\left(Z\\right)` and :math:`b\\left(Z\\right)`
    are piecewise linear functions w.r.t. :math:`Z`


    For this particular function the :math:`\\mathbf{\\theta}` vector for
    each load is given by:

    .. math:: \\mathbf{\\theta}_{i}=
                \\int_{0}^1{
                  {a\\left(Z\\right)}
                  {b\\left(Z\\right)}
                  {\\sigma\\left(Z\\right)}
                  f\\left({Z}\\right)
                  \\phi_i\\,dZ}

    Where :math:`f\\left({Z}\\right)` is the appropriate z-dependent term
    corresponding to either :math:`u_{top}` or :math:`u_{bot}` homogenisations.


    The :math:`\\mathbf{E}` matrix for each load is given by:

    .. math:: \\mathbf{E}_{i,j}=
                \\int_{0}^{t_j}{
                  \\frac{d{
                    {\\cos\\left(\\omega\\tau+\\textrm{phase}\\right)}
                    \\sigma\\left(\\tau\\right)}}
                    {d\\tau}
                  {\\exp\\left({(dT\\left(t-\\tau\\right)\\lambda_i}\\right)}
                  \\,d\\tau}


    where

     - :math:`\\lambda_i` is the `ith` eigenvalue of the problem,
     - :math:`dT` is a time factor for numerical convienience,
     - :math:`\\sigma\left(\\tau\\right)` is the piecewise linear time
       dependant load.


    Note that the listed equations above are in terms of normalised depth Z,
    with depth integrations between [0, 1].  However, IF YOU KNOW WHAT YOU
    ARE DOING the integrations can be done using non-normalised depths.
    The first z value in the piecewise definition a(z) must still be 0
    however the end point for integration will be the final z value in the
    definition of a(z).  If you are doing this then your `m` values will
    include the normalising Factor. e.g. m = [pi/2/H, 3*pi/2/H] and a(z) is
    defined in two layers [0, z1], [z1, zend] as opposed to
    m = [pi/2, 3*pi/2] and a(Z) is two layers [0, z1/H], [z1/H, zend/H].


    """

    E_Igamv_the = np.zeros((len(m), len(tvals)))

    if sum([v is None for v in [a, b]]) == 0:
        a, b = pwise.polyline_make_x_common(a, b)
        if drn==1:
            zdist = PolyLine(a.x1,a.x2, np.ones_like(a.x1), np.ones_like(a.x2))
            #bot_vs_time=None
        else:
            zdist = PolyLine(a.x1,a.x2, a.x2[-1]-a.x1, a.x2[-1]-a.x2)

        if not top_vs_time is None:
            if top_omega_phase is None:
                top_omega_phase = [None] * len(top_vs_time)

            theta = integ.pdim1sin_abc_linear(m, a, b, zdist)
            if not theta_zero_indexes is None:
                theta[theta_zero_indexes] = 0.0
            for top_vs_t, om_ph in zip(top_vs_time, top_omega_phase):
                if not om_ph is None:
                    omega, phase = om_ph
                    E = integ.pEDload_coslinear(top_vs_t, omega, phase, eigs, tvals, dT, implementation=implementation)
                else:
                    E = integ.pEDload_linear(top_vs_t, eigs, tvals, dT, implementation=implementation)
                E_Igamv_the += (E*np.dot(Igamv, theta)).T

        if not bot_vs_time is None:
            if bot_omega_phase is None:
                bot_omega_phase = [None] * len(bot_vs_time)
            theta = integ.pdim1sin_abc_linear(m, a, b, PolyLine(a.x1,a.x2,a.x1,a.x2))
            if not theta_zero_indexes is None:
                theta[theta_zero_indexes] = 0.0
            for bot_vs_t, om_ph in zip(bot_vs_time, bot_omega_phase):
                if not om_ph is None:
                    omega, phase = om_ph
                    E = integ.pEDload_coslinear(bot_vs_t, omega, phase, eigs, tvals, dT, implementation=implementation)
                else:
                    E = integ.pEDload_linear(bot_vs_t, eigs, tvals, dT, implementation=implementation)
                E_Igamv_the += (E*np.dot(Igamv, theta)).T

    #theta is 1d array, Igamv is nieg by neig array, np.dot(Igamv, theta)
    #and np.dot(theta, Igamv) will give differetn 1d arrays.
    #Basically np.dot(Igamv, theta) gives us what we want i.e.
    #theta was treated as a column array.  The alternative
    #np.dot(theta, Igamv) would have treated theta as a row vector.
    return E_Igamv_the


def dim1sin_E_Igamv_the_BC_D_aDf_linear(drn,
                                        m,
                                        eigs,
                                        tvals,
                                        Igamv,
                                        a,
                                        top_vs_time,
                                        bot_vs_time,
                                        top_omega_phase=None,
                                        bot_omega_phase=None,
                                        dT=1.0,
                                        theta_zero_indexes=None,
                                        implementation='vectorized'):
    """Calculate E and theta parts and assemble E_Igamv_the matrix that arises
    from homogenising D[a(Z)*D[u(Z, t),Z],Z] for non_zero top and bottom
    boundary conditions.

    When accounting for non-zero boundary conditions we homogenise the
    governing equation by letting u(Z,t) = v(Z,t) + utop(t)*(1-Z) + ubot(t)*Z
    and solving for v(Z, t).  This function calculates the time dependent
    E part, the depth dependent theta part, and then assembles
    E*inverse(gam*v)*theta which forms part of solution
    v(Z,t)=phi*v*E*inverse(gam*v)*theta.  The E and theta parts arise
    by subbing the boundary conditions into into governing equation terms of
    the form D[a(Z)*D[u(Z, t),Z],Z].

    The contribution of each `mag_vs_time`-`omega_phase` pairing
    are superposed.    The result is an array
    of size (neig, len(tvals)). So each column is the are the column vector
    E*inverse(gam*v)*theta calculated at each output time.  This will allow
    us later to do v(Z,t) = phi*v*E_Igamv_the.

    Uses sin(m*z) in the calculation of theta.

    Parameters
    ----------
    drn : [0,1]
        Drainage condition,
        drn=0 for Pervious top pervious bottom (PTPB).
        drn=1 for Pervious top impoervious bottom (PTIB).
    m : ``list`` of ``float``
        Eigenvalues of BVP, the m in sin(m*Z). Generate with
        geotecha.speccon.m_from_sin_mx.
    eigs : 1d numpy.ndarray
        List of eigenvalues of the spectral matrix i.e. Eigenvalues of the
        square Igam_psi matrix.
    tvals : 1d numpy.ndarray`
        List of time values to evaluate E matrix at.
    Igamv : ndarray
        Speccon matrix.  Igamv = inverse of [gam * v])
    a : PolyLine
        Piewcewise linear function.  e.g. for 1d consolidation surcharge
        radial draiange term is D[kv(z)*D[u(Z,t), Z],Z] so `a` would be kv.
        be et.
    top_vs_time, bot_vs_time : list of PolyLine
        Piecewise linear magnitude vs time for the top and bottom boundary.
        Use ``None`` if there is no variation.
    top_omega_phase, bot_omega_phase : list of 2 element tuples, optional
        (omega, phase) for use in cos(omega * t + phase) * mag_vs_time
        if omega_phase is None then mag_vs_time will not be multiplied by a
        cosine.  If any element of omega_phase is None then in that particular
        loading combo, mag_vs_time will not be multiplied by a cosine.
    dT : ``float``, optional
        Time factor multiple for numerical convieniece. Default dT=1.0.
    theta_zero_indexes : slice/list etc., optional
        A slice object, list, etc that can be used for numpy fancy indexing.
        Any specified index of the theta vector will be set to zero.  This is
        useful when using the spectral method with block matrices and the
        loading term only refers to a subset of the equations.  When using
        block matrices m should be the same size as the block matrix.
        Default theta_zero_indexes=None i.e. no elements of theta will be
        set to zero.

    Returns
    -------
    E_Igamv_the : ndarray
        Loading matrix of size (neig, len(tvals)).

    Notes
    -----
    Assuming the loads are formulated as the product of separate time and depth
    dependant functions as well as a cyclic component:

    .. math:: \\sigma\\left({Z,t}\\right)=
                \\sigma\\left({Z}\\right)
                \\sigma\\left({t}\\right)
                \\cos\\left(\\omega t + \\phi\\right)

    the solution to the consolidation equation using the spectral method has
    the form:

    .. math:: u\\left(Z,t\\right)=
                \\mathbf{\\phi} \\mathbf{v} \\mathbf{E}
                \\left(\\mathbf{\\Gamma v}\\right)^{-1}
                \\mathbf{\\theta}


    When we consider non-zero boundary conditions, additional loading terms are
    created when we sub in the following into the original governing equation.

    .. math:: u\\left({Z,t}\\right)=
                v\\left({Z,t}\\right) +
                u_{top}\\left({t}\\right)\\left({1-Z}\\right) +
                u_{bot}\\left({b}\\right)Z

    Two additional loading terms are created with each substitution, one
    for the top boundary condition and one for the bottom boundary condition.

    This function calculates :math:`\\mathbf{E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}`
    when substitutions are made in terms of the following form:

    .. math:: \\frac{\\partial}{\\partial Z}
                \\left(
                  {a\\left({Z}\\right)
                  \\frac{\\partial u\\left({Z,t}\\right)}{\\partial Z}}
                \\right)


    It is assumed that :math:`u_{top}\\left({t}\\right)` and
    :math:`u_{bot}\\left({t}\\right)` are piecewise linear
    in time with a cyclic component, and that multiple functions are
    superposed.  Also :math:`a\\left(Z\\right)`
    is a piecewise linear function with respect to :math:`Z`


    For this particular function the :math:`\\mathbf{\\theta}` vector for
    each load is given by:

    .. math:: \\mathbf{\\theta}_{i}=
                \\int_{0}^1{
                  \\frac{\\partial}{\\partial Z}
                  \\left(
                    {a\\left({Z}\\right)
                    \\frac{\\partial \\sigma\\left({Z}\\right)}{\\partial Z}}
                  \\right)
                  f\\left({Z}\\right)
                  \\phi_i\\,dZ}

    Where :math:`f\\left({Z}\\right)` is the appropriate z-dependent term
    corresponding to either :math:`u_{top}` or :math:`u_{bot}` homogenisations.


    The :math:`\\mathbf{E}` matrix for each load is given by:

    .. math:: \\mathbf{E}_{i,j}=
                \\int_{0}^{t_j}{
                  {\\cos\\left(\\omega\\tau+\\textrm{phase}\\right)}
                  {\\sigma\\left(\\tau\\right)}
                  {\\exp\\left({(dT\\left(t-\\tau\\right)\\lambda_i}\\right)}
                  \\,d\\tau}


    where

     - :math:`\\lambda_i` is the `ith` eigenvalue of the problem,
     - :math:`dT` is a time factor for numerical convienience,
     - :math:`\\sigma\left(\\tau\\right)` is the piecewise linear time
       dependant load.


    Note that the listed equations above are in terms of normalised depth Z,
    with depth integrations between [0, 1].  However, IF YOU KNOW WHAT YOU
    ARE DOING the integrations can be done using non-normalised depths.
    The first z value in the piecewise definition a(z) must still be 0
    however the end point for integration will be the final z value in the
    definition of a(z).  If you are doing this then your `m` values will
    include the normalising Factor. e.g. m = [pi/2/H, 3*pi/2/H] and a(z) is
    defined in two layers [0, z1], [z1, zend] as opposed to
    m = [pi/2, 3*pi/2] and a(Z) is two layers [0, z1/H], [z1/H, zend/H].

    """

    E_Igamv_the = np.zeros((len(m), len(tvals)))

    if not a is None:
        if drn==1:
            zdist = PolyLine(a.x1,a.x2, np.ones_like(a.x1), np.ones_like(a.x2))
            #bot_vs_time=None
        else:
            zdist = PolyLine(a.x1,a.x2, a.x2[-1]-a.x1, a.x2[-1]-a.x2)

        if not top_vs_time is None:
            if top_omega_phase is None:
                top_omega_phase = [None] * len(top_vs_time)

            theta = integ.pdim1sin_D_aDb_linear(m, a, zdist)
            if not theta_zero_indexes is None:
                theta[theta_zero_indexes] = 0.0
            for top_vs_t, om_ph in zip(top_vs_time, top_omega_phase):
                if not om_ph is None:
                    omega, phase = om_ph
                    E = integ.pEload_coslinear(top_vs_t, omega, phase, eigs, tvals, dT, implementation=implementation)
                else:
                    E = integ.pEload_linear(top_vs_t, eigs, tvals, dT, implementation=implementation)
                E_Igamv_the += (E*np.dot(Igamv, theta)).T

        if not bot_vs_time is None:
            if bot_omega_phase is None:
                bot_omega_phase = [None] * len(bot_vs_time)

            theta = integ.pdim1sin_D_aDb_linear(m, a, PolyLine(a.x1,a.x2,a.x1,a.x2))
            if not theta_zero_indexes is None:
                theta[theta_zero_indexes] = 0.0
            for bot_vs_t, om_ph in zip(bot_vs_time, bot_omega_phase):
                if not om_ph is None:
                    omega, phase = om_ph
                    E = integ.pEload_coslinear(bot_vs_t, omega, phase, eigs, tvals, dT, implementation=implementation)
                else:
                    E = integ.pEload_linear(bot_vs_t, eigs, tvals, dT, implementation=implementation)
                E_Igamv_the += (E*np.dot(Igamv, theta)).T

    #theta is 1d array, Igamv is nieg by neig array, np.dot(Igamv, theta)
    #and np.dot(theta, Igamv) will give differetn 1d arrays.
    #Basically np.dot(Igamv, theta) gives us what we want i.e.
    #theta was treated as a column array.  The alternative
    #np.dot(theta, Igamv) would have treated theta as a row vector.
    return E_Igamv_the


def dim1sin_E_Igamv_the_BC_deltaf_linear(drn,
                                         m,
                                         eigs,
                                         tvals,
                                         Igamv,
                                         zvals,
                                         pseudo_k,
                                         top_vs_time,
                                         bot_vs_time,
                                         top_omega_phase=None,
                                         bot_omega_phase=None,
                                         dT=1.0,
                                         theta_zero_indexes=None,
                                         implementation='vectorized'):
    """Calculate E and theta parts and assemble E_Igamv_the matrix that arises
    from homogenising delta(Z-zd)*u(Z,t) for non_zero top and bottom
    boundary conditions.

    When accounting for non-zero boundary conditions we homogenise the
    governing equation by letting u(Z,t) = v(Z,t) + utop(t)*(1-Z) + ubot(t)*Z
    and solving for v(Z, t).  This function calculates the time dependent
    E part, the depth dependent theta part, and then assembles
    E*inverse(gam*v)*theta which forms part of solution
    v(Z,t)=phi*v*E*inverse(gam*v)*theta.  The E and theta parts arise
    by subbing the boundary conditions into into governing equation terms of
    the form delta(Z-zd)*u(Z,t).

    The contribution of each `mag_vs_time`-`omega_phase` pairing
    are superposed.    The result is an array
    of size (neig, len(tvals)). So each column is the are the column vector
    E*inverse(gam*v)*theta calculated at each output time.  This will allow
    us later to do v(Z,t) = phi*v*E_Igamv_the.

    Uses sin(m*z) in the calculation of theta.

    Parameters
    ----------
    drn : [0,1]
        Drainage condition,
        drn=0 for Pervious top pervious bottom (PTPB).
        drn=1 for Pervious top impoervious bottom (PTIB).
    m : ``list`` of ``float``
        Eigenvalues of BVP, the m in sin(m*Z). Generate with
        geotecha.speccon.m_from_sin_mx.
    eigs : 1d numpy.ndarray
        List of eigenvalues of the spectral matrix i.e. Eigenvalues of the
        square Igam_psi matrix.
    tvals : 1d numpy.ndarray`
        List of time values to evaluate E matrix at.
    Igamv : ndarray
        Speccon matrix.  Igamv = inverse of [gam * v])
    zvals : list of float
        z values defining each delta function, zd.
    pseudo_k : list of float
        Coefficients to multiply each delta function by.
    top_vs_time, bot_vs_time : list of PolyLine
        Piecewise linear magnitude vs time for the top and bottom boundary.
        Use ``None`` if there is no variation.
    top_omega_phase, bot_omega_phase : list of 2 element tuples, optional
        (omega, phase) for use in cos(omega * t + phase) * mag_vs_time
        if omega_phase is None then mag_vs_time will not be multiplied by a
        cosine.  If any element of omega_phase is None then in that particular
        loading combo, mag_vs_time will not be multiplied by a cosine.
    dT : ``float``, optional
        Time factor multiple for numerical convieniece. Default dT=1.0.
    theta_zero_indexes : slice/list etc., optional
        A slice object, list, etc that can be used for numpy fancy indexing.
        Any specified index of the theta vector will be set to zero.  This is
        useful when using the spectral method with block matrices and the
        loading term only refers to a subset of the equations.  When using
        block matrices m should be the same size as the block matrix.
        Default theta_zero_indexes=None i.e. no elements of theta will be
        set to zero.

    Returns
    -------
    E_Igamv_the : ndarray
        Loading matrix of size (neig, len(tvals)).

    Notes
    -----
    Assuming the loads are formulated as the product of separate time and depth
    dependant functions as well as a cyclic component:

    .. math:: \\sigma\\left({Z,t}\\right)=
                \\sigma\\left({Z}\\right)
                \\sigma\\left({t}\\right)
                \\cos\\left(\\omega t + \\phi\\right)

    the solution to the consolidation equation using the spectral method has
    the form:

    .. math:: u\\left(Z,t\\right)=
                \\mathbf{\\phi v E}
                \\left(\\mathbf{\\Gamma v}\\right)^{-1}
                \\mathbf{\\theta}


    When we consider non-zero boundary conditions, additional loading terms are
    created when we sub in the following into the original governing equation.

    .. math:: u\\left({Z,t}\\right)=
                v\\left({Z,t}\\right) +
                u_{top}\\left({t}\\right)\\left({1-Z}\\right) +
                u_{bot}\\left({b}\\right)Z

    Two additional loading terms are created with each substitution, one
    for the top boundary condition and one for the bottom boundary condition.

    This function calculates :math:`\\mathbf{E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}`
    when substitutions are made in terms of the following form:


    .. math:: k_{\\textrm{pseudo}}
              \\delta\\left({Z-Z_d}\\right)
              u\\left({Z,t}\\right)

    It is assumed that :math:`u_{top}\\left({t}\\right)` and
    :math:`u_{bot}\\left({t}\\right)` are piecewise linear
    in time with a cyclic component, and that multiple functions are
    superposed.


    For this particular function the :math:`\\mathbf{\\theta}` vector for
    each load is given by:

    .. math:: \\mathbf{\\theta}_{i}=
                \\int_{0}^1{
                  k_{\\textrm{pseudo}}
                  \\delta\\left({Z-Z_d}\\right)
                  {\\sigma\\left(Z\\right)}
                  f\\left({Z}\\right)
                  \\phi_i\\,dZ}

    Where :math:`f\\left({Z}\\right)` is the appropriate z-dependent term
    corresponding to either :math:`u_{top}` or :math:`u_{bot}` homogenisations.


    The :math:`\\mathbf{E}` matrix for each load is given by:

    .. math:: \\mathbf{E}_{i,j}=
                \\int_{0}^{t_j}{
                  {\\cos\\left(\\omega\\tau+\\textrm{phase}\\right)}
                  {\\sigma\\left(\\tau\\right)}
                  {\\exp\\left({(dT\\left(t-\\tau\\right)\\lambda_i}\\right)}
                  \\,d\\tau}


    where

     - :math:`\\lambda_i` is the `ith` eigenvalue of the problem,
     - :math:`dT` is a time factor for numerical convienience,
     - :math:`\\sigma\left(\\tau\\right)` is the piecewise linear time
       dependant load.


    Note that this function, unlike many similar functions in this module,
    has only been formulated for Normalised depths between [0,1]


    """

    E_Igamv_the = np.zeros((len(m), len(tvals)))
    zvals = np.asarray(zvals)

    if drn==1:
        zdist = np.ones_like(zvals)
        #bot_vs_time=None
    else:
        zdist = 1.0 - zvals

    zdist = 1 - zvals * (1 - drn)
    if not top_vs_time is None:
        if top_omega_phase is None:
            top_omega_phase = [None] * len(top_vs_time)

        for top_vs_t, om_ph in zip(top_vs_time, top_omega_phase):
            if not om_ph is None:
                omega, phase = om_ph
                E = integ.pEload_coslinear(top_vs_t, omega, phase, eigs, tvals, dT, implementation=implementation)
            else:
                E = integ.pEload_linear(top_vs_t, eigs, tvals, dT, implementation=implementation)
            for z, zd, k in zip(zvals, zdist, pseudo_k):
                theta = k * np.sin(z * m) * zd
                if not theta_zero_indexes is None:
                    theta[theta_zero_indexes] = 0.0
                E_Igamv_the += (E*np.dot(Igamv, theta)).T

    if not bot_vs_time is None:
        if bot_omega_phase is None:
            bot_omega_phase = [None] * len(bot_vs_time)

        for bot_vs_t, om_ph in zip(bot_vs_time, bot_omega_phase):
            if not om_ph is None:
                omega, phase = om_ph
                E = integ.pEload_coslinear(bot_vs_t, omega, phase, eigs, tvals, dT, implementation=implementation)
            else:
                E = integ.pEload_linear(bot_vs_t, eigs, tvals, dT, implementation=implementation)
            for z, k in zip(zvals, pseudo_k):
                theta = k * np.sin(z * m) * z
                if not theta_zero_indexes is None:
                    theta[theta_zero_indexes] = 0.0
                E_Igamv_the += (E*np.dot(Igamv, theta)).T

    #theta is 1d array, Igamv is nieg by neig array, np.dot(Igamv, theta)
    #and np.dot(theta, Igamv) will give differetn 1d arrays.
    #Basically np.dot(Igamv, theta) gives us what we want i.e.
    #theta was treated as a column array.  The alternative
    #np.dot(theta, Igamv) would have treated theta as a row vector.
    return E_Igamv_the


def dim1sin_E_Igamv_the_deltamag_linear(m,
                                        eigs,
                                        tvals,
                                        Igamv,
                                        zvals,
                                        pseudo_k,
                                        mag_vs_time,
                                        omega_phase=None,
                                        dT=1.0,
                                        theta_zero_indexes=None,
                                        implementation='vectorized'):
    """Calculate E and theta parts and assemble E_Igamv_the matrix
    for loading terms of the form a(Z) * delta(Z-Zd)*mag(t) where mag is
    piecewise linear in time multiplied by cos(omega * t + phase).


    Make the E*inverse(gam*v)*theta part of solution
    u(Z,t)=phi*v*E*inverse(gam*v)*theta for terms of the form
    a(Z) * delta(Z-Zd)*mag(t).
    The contribution of each `mag_vs_time`-`omega_phase` pairing and each zval
    are superposed. The result is an array
    of size (neig, len(tvals)). So each column is the are the column vector
    E*inverse(gam*v)*theta calculated at each output time.  This will allow
    us later to do u(Z,t) = phi*v*E_Igamv_the.

    Uses sin(m*Z) in the calculation of theta.


    Parameters
    ----------
    drn : [0,1]
        Drainage condition,
        drn=0 for Pervious top pervious bottom (PTPB).
        drn=1 for Pervious top impoervious bottom (PTIB).
    m : ``list`` of ``float``
        Eigenvalues of BVP, the m in sin(m*Z). Generate with
        geotecha.speccon.m_from_sin_mx.
    eigs : 1d numpy.ndarray
        List of eigenvalues of the spectral matrix i.e. Eigenvalues of the
        square Igam_psi matrix.
    tvals : 1d numpy.ndarray`
        List of time values to evaluate E matrix at.
    Igamv : ndarray
        Speccon matrix.  Igamv = inverse of [gam * v])
    zvals : list of float
        z values defining each delta function, zd.
    pseudo_k : list of float
        Coefficients to multiply each delta function by.
    dT : ``float``, optional
        Time factor multiple for numerical convieniece. Default dT=1.0.
    theta_zero_indexes : slice/list etc., optional
        A slice object, list, etc that can be used for numpy fancy indexing.
        Any specified index of the theta vector will be set to zero.  This is
        useful when using the spectral method with block matrices and the
        loading term only refers to a subset of the equations.  When using
        block matrices m should be the same size as the block matrix.
        Default theta_zero_indexes=None i.e. no elements of theta will be
        set to zero.

    Returns
    -------
    E_Igamv_the : ndarray
        Loading matrix of size (neig, len(tvals)).

    Notes
    -----
    Assuming the loads are formulated as the product of separate time and depth
    dependant functions as well as a cyclic component:

    .. math:: \\sigma\\left({Z,t}\\right)=
                \\sigma\\left({Z}\\right)
                \\sigma\\left({t}\\right)
                \\cos\\left(\\omega t + \\phi\\right)

    the solution to the consolidation equation using the spectral method has
    the form:

    .. math:: u\\left(Z,t\\right)=
                \\mathbf{\\phi v E}
                \\left(\\mathbf{\\Gamma v}\\right)^{-1}
                \\mathbf{\\theta}

    This function calculates :math:`\\mathbf{E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}`
    for terms of the following form:

    .. math:: k_{\\textrm{pseudo}}
              \\delta\\left({Z-Z_d}\\right)
              \\sigma\\left({t}\\right)

    It is assumed that :math:`\\sigma\\left({t}\\right)` is piecewise linear
    in time with a cyclic component, and that multiple functions are
    superposed.


    For this particular function the :math:`\\mathbf{\\theta}` vector for
    each load is given by:

    .. math:: \\mathbf{\\theta}_{i}=
                \\int_{0}^1{
                  k_{\\textrm{pseudo}}
                  \\delta\\left({Z-Z_d}\\right)
                  \\phi_i\\,dZ}


    The :math:`\\mathbf{E}` matrix for each load is given by:

    .. math:: \\mathbf{E}_{i,j}=
                \\int_{0}^{t_j}{
                  {\\cos\\left(\\omega\\tau+\\textrm{phase}\\right)}
                  {\\sigma\\left(\\tau\\right)}
                  {\\exp\\left({(dT\\left(t-\\tau\\right)\\lambda_i}\\right)}
                  \\,d\\tau}


    where

     - :math:`\\lambda_i` is the `ith` eigenvalue of the problem,
     - :math:`dT` is a time factor for numerical convienience,
     - :math:`\\sigma\left(\\tau\\right)` is the piecewise linear time
       dependant load.


    """

    E_Igamv_the = np.zeros((len(m), len(tvals)))

    if omega_phase is None:
            omega_phase = [None] * len(mag_vs_time)

    for z, k, mag_vs_t, om_ph in zip(zvals, pseudo_k, mag_vs_time, omega_phase):
        if mag_vs_t is None:
            continue
        theta = k * np.sin(z * m)
        if not theta_zero_indexes is None:
            theta[theta_zero_indexes] = 0.0
        if not om_ph is None:
            omega, phase = om_ph
            E = integ.pEload_coslinear(mag_vs_t, omega, phase, eigs, tvals, dT, implementation=implementation)
        else:
            E = integ.pEload_linear(mag_vs_t, eigs, tvals, dT, implementation=implementation)
        E_Igamv_the += (E*np.dot(Igamv, theta)).T


    return E_Igamv_the


def dim1sin_E_Igamv_the_aDmagDt_bilinear(m,
                                         eigs,
                                         tvals,
                                         Igamv,
                                         a,
                                         mag_vs_depth,
                                         mag_vs_time,
                                         omega_phase=None,
                                         dT=1.0,
                                         theta_zero_indexes=None,
                                         implementation='vectorized'):
    """Calculate E and theta parts and assemble E_Igamv_the matrix
    for loading terms of the form a(Z) * D[mag(t, Z),t] where mag is
    piecewise linear in depth and time and multiplied by cos(omega * t + phase).


    Make the E*inverse(gam*v)*theta part of solution
    u(Z,t)=phi*v*E*inverse(gam*v)*theta for terms of the form
    a(Z) * D[mag(t, Z),t].
    The contribution of each `mag_vs_time`-`omega_phase` pairing and
    are superposed. The result is an array
    of size (neig, len(tvals)). So each column is the column vector
    E*inverse(gam*v)*theta calculated at each output time.  This will allow
    us later to do u(Z,t) = phi*v*E_Igamv_the.


    Uses sin(m*z) in the calculation of theta.

    Parameters
    ----------
    drn : [0,1]
        Drainage condition,
        drn=0 for Pervious top pervious bottom (PTPB).
        drn=1 for Pervious top impoervious bottom (PTIB).
    m : ``list`` of ``float``
        Eigenvalues of BVP, the m in sin(m*Z). Generate with
        geotecha.speccon.m_from_sin_mx.
    eigs : 1d numpy.ndarray
        List of eigenvalues of the spectral matrix i.e. Eigenvalues of the
        square Igam_psi matrix.
    tvals : 1d numpy.ndarray`
        List of time values to evaluate E matrix at.
    Igamv : ndarray
        Speccon matrix.  Igamv = inverse of [gam * v])
    a : PolyLine
        Piewcewise linear function.  e.g. for 1d consolidation surcharge
        loading term is mv*D[sigma(z, t), t] so `a` would be mv.
    dT : ``float``, optional
        Time factor multiple for numerical convieniece. Default dT=1.0.
    theta_zero_indexes : slice/list etc., optional
        A slice object, list, etc that can be used for numpy fancy indexing.
        Any specified index of the theta vector will be set to zero.  This is
        useful when using the spectral method with block matrices and the
        loading term only refers to a subset of the equations.  When using
        block matrices m should be the same size as the block matrix.
        Default theta_zero_indexes=None i.e. no elements of theta will be
        set to zero.

    Returns
    -------
    E_Igamv_the : ndarray
        Loading matrix of size (neig, len(tvals)).

    Notes
    -----
    Assuming the loads are formulated as the product of separate time and depth
    dependant functions as well as a cyclic component:

    .. math:: \\sigma\\left({Z,t}\\right)=
                \\sigma\\left({Z}\\right)
                \\sigma\\left({t}\\right)
                \\cos\\left(\\omega t + \\phi\\right)

    the solution to the consolidation equation using the spectral method has
    the form:

    .. math:: u\\left(Z,t\\right)=
                \\mathbf{\\phi v E}
                \\left(\\mathbf{\\Gamma v}\\right)^{-1}
                \\mathbf{\\theta}


    This function calculates :math:`\\mathbf{E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}`
    for terms of the following form:


    .. math:: a\\left({Z}\\right)\\frac{\\partial \\sigma}{\\partial t}

    It is assumed that :math:`\\sigma\\left({t}\\right)` is piecewise linear
    in time with a cyclic component, and that multiple functions are
    superposed.  Also :math:`a\\left(Z\\right)`
    is a piecewise linear function with respect to :math:`Z`.


    For this particular function the :math:`\\mathbf{\\theta}` vector for
    each load is given by:

    .. math:: \\mathbf{\\theta}_{i}=
                \\int_{0}^1{
                  {a\\left(Z\\right)}
                  {\\sigma\\left(Z\\right)}
                  \\phi_i\\,dZ}


    The :math:`\\mathbf{E}` matrix for each load is given by:

    .. math:: \\mathbf{E}_{i,j}=
                \\int_{0}^{t_j}{
                  \\frac{d{
                    {\\cos\\left(\\omega\\tau+\\textrm{phase}\\right)}
                    \\sigma\\left(\\tau\\right)}}
                    {d\\tau}
                  {\\exp\\left({(dT\\left(t-\\tau\\right)\\lambda_i}\\right)}
                  \\,d\\tau}


    where

     - :math:`\\lambda_i` is the `ith` eigenvalue of the problem,
     - :math:`dT` is a time factor for numerical convienience,
     - :math:`\\sigma\left(\\tau\\right)` is the piecewise linear time
       dependant load.


    Note that the listed equations above are in terms of normalised depth Z,
    with depth integrations between [0, 1].  However, IF YOU KNOW WHAT YOU
    ARE DOING the integrations can be done using non-normalised depths.
    The first z value in the piecewise definition a(z) must still be 0
    however the end point for integration will be the final z value in the
    definition of a(z).  If you are doing this then your `m` values will
    include the normalising Factor. e.g. m = [pi/2/H, 3*pi/2/H] and a(z) is
    defined in two layers [0, z1], [z1, zend] as opposed to
    m = [pi/2, 3*pi/2] and a(Z) is two layers [0, z1/H], [z1/H, zend/H].


    """

    E_Igamv_the = np.zeros((len(m), len(tvals)))


    if sum([v is None for v in [mag_vs_depth, mag_vs_time]])==0:
        if omega_phase is None:
            omega_phase = [None] * len(mag_vs_time)

        for mag_vs_t, mag_vs_z, om_ph in zip(mag_vs_time, mag_vs_depth, omega_phase):
            a, mag_vs_z = pwise.polyline_make_x_common(a, mag_vs_z)
            theta = integ.pdim1sin_ab_linear(m, a, mag_vs_z)
            if not theta_zero_indexes is None:
                theta[theta_zero_indexes] = 0.0

            if not om_ph is None:
                omega, phase = om_ph
                E = integ.pEDload_coslinear(mag_vs_t, omega, phase, eigs, tvals, dT, implementation=implementation)
            else:
                E = integ.pEDload_linear(mag_vs_t, eigs, tvals, dT, implementation=implementation)


            #theta is 1d array, Igamv is nieg by neig array, np.dot(Igamv, theta)
            #and np.dot(theta, Igamv) will give differetn 1d arrays.
            #Basically np.dot(Igamv, theta) gives us what we want i.e.
            #theta was treated as a column array.  The alternative
            #np.dot(theta, Igamv) would have treated theta as a row vector.
            E_Igamv_the += (E*np.dot(Igamv, theta)).T

    return E_Igamv_the


def dim1sin_E_Igamv_the_abmag_bilinear(m,
                                       eigs,
                                       tvals,
                                       Igamv,
                                       a,
                                       b,
                                       mag_vs_depth,
                                       mag_vs_time,
                                       omega_phase=None,
                                       dT=1.0,
                                       theta_zero_indexes=None,
                                       implementation='vectorized'):
    """Calculate E and theta parts and assemble E_Igamv_the matrix
    for loading terms of the form a(Z)*b(Z)*mag(t, Z) where mag is
    piecewise linear in depth and time and multiplied by cos(omega * t + phase).


    Make the E*inverse(gam*v)*theta part of solution
    u(Z,t)=phi*v*E*inverse(gam*v)*theta for terms of the form
    a(Z)*b(Z)*mag(t, Z).
    The contribution of each `mag_vs_time`-`omega_phase` pairing and
    are superposed. The result is an array
    of size (neig, len(tvals)). So each column is the column vector
    E*inverse(gam*v)*theta calculated at each output time.  This will allow
    us later to do u(Z,t) = phi*v*E_Igamv_the.


    Uses sin(m*Z) in the calculation of theta.

    Parameters
    ----------
    drn : [0,1]
        Drainage condition,
        drn=0 for Pervious top pervious bottom (PTPB).
        drn=1 for Pervious top impoervious bottom (PTIB).
    m : ``list`` of ``float``
        Eigenvalues of BVP, the m in sin(m*Z). Generate with
        geotecha.speccon.m_from_sin_mx.
    eigs : 1d numpy.ndarray
        List of eigenvalues of the spectral matrix i.e. Eigenvalues of the
        square Igam_psi matrix.
    tvals : 1d numpy.ndarray`
        List of time values to evaluate E matrix at.
    Igamv : ndarray
        Speccon matrix.  Igamv = inverse of [gam * v])
    a, b : PolyLine
        Piewcewise linear function.  e.g. for 1d consolidation surcharge
        radial draiange term is dTh*kh*et*mag(Z,t) `a` would be kh, `b` would
        be et.
    dT : ``float``, optional
        Time factor multiple for numerical convieniece. Default dT=1.0.
    theta_zero_indexes : slice/list etc., optional
        A slice object, list, etc that can be used for numpy fancy indexing.
        Any specified index of the theta vector will be set to zero.  This is
        useful when using the spectral method with block matrices and the
        loading term only refers to a subset of the equations.  When using
        block matrices m should be the same size as the block matrix.
        Default theta_zero_indexes=None i.e. no elements of theta will be
        set to zero.

    Returns
    -------
    E_Igamv_the : ndarray
        Loading matrix of size (neig, len(tvals)).

    Notes
    -----
    Assuming the loads are formulated as the product of separate time and depth
    dependant functions as well as a cyclic component:

    .. math:: \\sigma\\left({Z,t}\\right)=
                \\sigma\\left({Z}\\right)
                \\sigma\\left({t}\\right)
                \\cos\\left(\\omega t + \\phi\\right)

    the solution to the consolidation equation using the spectral method has
    the form:

    .. math:: u\\left(Z,t\\right)=
                \\mathbf{\\phi v E}
                \\left(\\mathbf{\\Gamma v}\\right)^{-1}
                \\mathbf{\\theta}


    This function calculates :math:`\\mathbf{E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}`
    for terms of the following form:


    .. math:: a\\left({Z}\\right)b\\left({Z}\\right)\\sigma\\left({Z, t}\\right)

    It is assumed that :math:`\\sigma\\left({t}\\right)` is piecewise linear
    in time with a cyclic component, and that multiple functions are
    superposed.  Also :math:`a\\left(Z\\right)` and :math:`b\\left(Z\\right)`
    are piecewise linear functions with respect to :math:`Z`.


    For this particular function the :math:`\\mathbf{\\theta}` vector for
    each load is given by:

    .. math:: \\mathbf{\\theta}_{i}=
                \\int_{0}^1{
                  {a\\left(Z\\right)}
                  {b\\left(Z\\right)}
                  {\\sigma\\left(Z\\right)}
                  \\phi_i\\,dZ}


    The :math:`\\mathbf{E}` matrix for each load is given by:

    .. math:: \\mathbf{E}_{i,j}=
                \\int_{0}^{t_j}{
                  {\\cos\\left(\\omega\\tau+\\textrm{phase}\\right)}
                  {\\sigma\\left(\\tau\\right)}
                  {\\exp\\left({(dT\\left(t-\\tau\\right)\\lambda_i}\\right)}
                  \\,d\\tau}


    where

     - :math:`\\lambda_i` is the `ith` eigenvalue of the problem,
     - :math:`dT` is a time factor for numerical convienience,
     - :math:`\\sigma\left(\\tau\\right)` is the piecewise linear time
       dependant load.


    Note that the listed equations above are in terms of normalised depth Z,
    with depth integrations between [0, 1].  However, IF YOU KNOW WHAT YOU
    ARE DOING the integrations can be done using non-normalised depths.
    The first z value in the piecewise definition a(z) must still be 0
    however the end point for integration will be the final z value in the
    definition of a(z).  If you are doing this then your `m` values will
    include the normalising Factor. e.g. m = [pi/2/H, 3*pi/2/H] and a(z) is
    defined in two layers [0, z1], [z1, zend] as opposed to
    m = [pi/2, 3*pi/2] and a(Z) is two layers [0, z1/H], [z1/H, zend/H].




    #########






    Loading dependant E_Igamv_the matrix for a(z)*b(z)*D[mag(z, t), t] where mag is bilinear in depth and time multiplied by cos(omega*t + phase)

    Make the E*inverse(gam*v)*theta part of solution u=phi*v*E*inverse(gam*v)*theta.
    The contribution of each `mag_vs_time`-`mag_vs_depth`-`omega_phase` pair are superposed.
    The result is an array
    of size (neig, len(tvals)). So the columns are the column array
    E*inverse(gam*v)*theta calculated at each output time.  This will allow
    us later to do u = phi*v*E_Igamv_the

    Uses sin(m*z) in the calculation of theta.

    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    eigs : 1d numpy.ndarray
        list of eigenvalues
    tvals : 1d numpy.ndarray`
        list of time values to calculate integral at
    Igamv : ndarray
        speccon matrix
    a : PolyLine
        Piewcewise linear function.  e.g. for 1d consolidation surcharge
        loading term is mv*D[sigma(z, t), t] so a would be mv.
    b : PolyLine
        Piewcewise linear function.  e.g. for 1d consolidation vacuum term is
         kh*et*w(z,t) so a would be `kh`, `b` would be `et`
    mag_vs_depth : list of PolyLine
        Piecewise linear magnitude  vs depth.
    mag_vs_time : list of PolyLine
        Piecewise linear magnitude vs time
    omega_phase : list of 2 element tuples, optional
        (omega, phase) for use in cos(omega * t + phase) * mag_vs_time
        if omega_phase is None then mag_vs_time will not be multiplied by a
        cosine.  If any element of omega_phase is None then in that particular
        loading combo, mag_vs_time will not be multiplied by a cosine.
    dT : ``float``, optional
        time factor multiple (default = 1.0)
    theta_zero_indexes : slice/list etc., optional=None
        a slice object, list, etc that can be used for numpy fancy indexing.
        Any specified index of the theta vector will be set to zero.  This is
        useful when using the spectral method with block matrices and the
        loading term only refers to a subset of the equations.  When using
        block matrices m should be the same size as the block matrix.
        default=None i.e. no elements of theta will be set to zero.

    Returns
    -------
    E_Igamv_the: ndarray
        loading matrix

    Notes
    -----
    Assuming the loads are formulated as the product of separate time and depth
    dependant functions as well as a cyclic component:

    .. math:: \\sigma\\left({Z,t}\\right)=\\sigma\\left({Z}\\right)\\sigma\\left({t}\\right)\\cos\\left(\\omega t + \\phi\\right)

    the solution to the consolidation equation using the spectral method has
    the form:

    .. math:: u\\left(Z,t\\right)=\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}

    In this instance :math:`\\sigma\\left({Z}\\right)`
    and :math:`\\sigma\\left({t}\\right)` are piecewise linear in depth and
    time (hence the 'bilinear' in the function name) there is also a cyclic
    component.

    `dim1sin_E_Igamv_the_abmag_bilinear` will calculate
    :math:`\\mathbf{E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}`
    for terms with the form:

    .. math:: a\\left({z}\\right)b\\left({z}\\right)\\frac{\\partial\\sigma\\left({Z,t}\\right)}{\\partial t}

    where :math:`a\\left(z\\right)`, :math:`b\\left(z\\right)` are
    piecewise linear functions w.r.t. :math:`z`.


    """

    E_Igamv_the = np.zeros((len(m), len(tvals)))


    if sum([v is None for v in [mag_vs_depth, mag_vs_time]])==0:
        if omega_phase is None:
            omega_phase = [None] * len(mag_vs_time)

        for mag_vs_t, mag_vs_z, om_ph in zip(mag_vs_time, mag_vs_depth, omega_phase):
            a, b , mag_vs_z = pwise.polyline_make_x_common(a, b, mag_vs_z)
            theta = integ.pdim1sin_abc_linear(m, a, b, mag_vs_z)
            if not theta_zero_indexes is None:
                theta[theta_zero_indexes] = 0.0
            if not om_ph is None:
                omega, phase = om_ph
                E = integ.pEload_coslinear(mag_vs_t, omega, phase, eigs, tvals, dT, implementation=implementation)
            else:
                E = integ.pEload_linear(mag_vs_t, eigs, tvals, dT, implementation=implementation)

            #theta is 1d array, Igamv is nieg by neig array, np.dot(Igamv, theta)
            #and np.dot(theta, Igamv) will give differetn 1d arrays.
            #Basically np.dot(Igamv, theta) gives us what we want i.e.
            #theta was treated as a column array.  The alternative
            #np.dot(theta, Igamv) would have treated theta as a row vector.
            E_Igamv_the += (E*np.dot(Igamv, theta)).T

    return E_Igamv_the


def dim1sin_foft_Ipsiw_the_BC_D_aDf_linear(drn,
                                           m,
                                           eigs,
                                           tvals,
                                           Ipsiw,
                                           a,
                                           top_vs_time,
                                           bot_vs_time,
                                           top_omega_phase=None,
                                           bot_omega_phase=None,
                                           theta_zero_indexes=None):
    """Calculate the f(t) and theta parts and assemble the foft_Ipsiw_the
    matrix that arises from homgenising D[a(Z)*D[u(Z, t),Z],Z] terms with
    non_zero top and bottom boundary conditions when modelling
    drains/wells/columns with finite permeability.

    When accounting for non-zero boundary conditions we homogenise the
    governing equation by letting u(Z,t) = v(Z,t) + utop(t)*(1-Z) + ubot(t)*Z
    and solving for v(Z, t).  For some problems the homogenisation process
    produces an additional term that has t be added to the usual solution of
    v(Z,t)=phi*v*E*inverse(gam*v)*theta
    i.e. v(Z,t)=phi*v*E*inverse(gam*v)*theta + f(t)*Ipsiw*theta
    This function calculates the f(t) and theta parts of the second term and
    then assembles the foft_Ipsiw_the matrix. These parts arise
    by subbing the boundary conditions into into governing equation terms of
    the form D[a(Z)*D[u(Z, t),Z],Z].


    Uses sin(m*Z) in the calculation of theta.

    Parameters
    ----------
    drn : [0,1]
        Drainage condition,
        drn=0 for Pervious top pervious bottom (PTPB).
        drn=1 for Pervious top impoervious bottom (PTIB).
    m : ``list`` of ``float``
        Eigenvalues of BVP, the m in sin(m*Z). Generate with
        geotecha.speccon.m_from_sin_mx.
    eigs : 1d numpy.ndarray
        List of eigenvalues of the spectral matrix i.e. Eigenvalues of the
        square Igam_psi matrix.
    tvals : 1d numpy.ndarray`
        List of time values to evaluate E matrix at.
    Ipsiw : ndarray, square matrix
        Speccon matrix.  Ipsiw was originally
        formulated/denoted for vertical peremability term in the well
        resistance flow equation for vertical drain consolidatoin with
        well resistance.  It is different, but still a square matrix, for
        the stone column consolidation problem.  As long as you know what you
        are doing Ipsiw can be any appropriate square matrix.
    a : PolyLine
        Piewcewise linear function.  e.g. for 1d consolidation surcharge
        radial draiange term is D[kv(z)*D[u(Z,t), Z],Z] so `a` would be kv.
        be et.
    top_vs_time, bot_vs_time : list of PolyLine
        Piecewise linear magnitude vs time for the top and bottom boundary.
        Use ``None`` if there is no variation.
    top_omega_phase, bot_omega_phase : list of 2 element tuples, optional
        (omega, phase) for use in cos(omega * t + phase) * mag_vs_time
        if omega_phase is None then mag_vs_time will not be multiplied by a
        cosine.  If any element of omega_phase is None then in that particular
        loading combo, mag_vs_time will not be multiplied by a cosine.
    dT : ``float``, optional
        Time factor multiple for numerical convieniece. Default dT=1.0.
    theta_zero_indexes : slice/list etc., optional
        A slice object, list, etc that can be used for numpy fancy indexing.
        Any specified index of the theta vector will be set to zero.  This is
        useful when using the spectral method with block matrices and the
        loading term only refers to a subset of the equations.  When using
        block matrices m should be the same size as the block matrix.
        Default theta_zero_indexes=None i.e. no elements of theta will be
        set to zero.

    Returns
    -------
    foft_Ipsiw_the: ndarray
        Additional homgenising term of size (neig, len(t)).

    Notes
    -----
    Assuming the loads are formulated as the product of separate time and depth
    dependant functions as well as a cyclic component:

    .. math:: \\sigma\\left({Z,t}\\right)=
                \\sigma\\left({Z}\\right)
                \\sigma\\left({t}\\right)
                \\cos\\left(\\omega t + \\phi\\right)

    the solution to consolidation equation using the spectral method has a
    similar form to the following:

    .. math:: u\\left(Z,t\\right)=
                \\mathbf{\\phi} \\mathbf{v} \\mathbf{E}
                \\left(\\mathbf{\\Gamma v}\\right)^{-1}
                \\mathbf{\\theta}

    When we consider non-zero boundary conditions, additional loading terms are
    created when we sub in the following into the original governing equation.

    .. math:: u\\left({Z,t}\\right)=
                v\\left({Z,t}\\right) +
                u_{top}\\left({t}\\right)\\left({1-Z}\\right) +
                u_{bot}\\left({b}\\right)Z

    As well as loading terms the process may produce an extra term to be
    added to the final solution of the following form:

    .. math::   \\mathbf{\\phi}
                \\sigma\\left({t}\\right)
                \\mathbf{\\psi}_{w}^{1}
                \\mathbf{\\theta}_w

    There are actually two of these terms one
    for the top boundary condition and one for the bottom boundary condition.

    This function calculates :math:`\\sigma\\left({t}\\right)\\mathbf{\\psi}_{w}^{-1}\\mathbf{\\theta}_w`

    when substitutions are made in terms of the following form:

    .. math:: \\frac{\\partial}{\\partial Z}
                \\left(
                  {a\\left({Z}\\right)
                  \\frac{\\partial u\\left({Z,t}\\right)}{\\partial Z}}
                \\right)


    It is assumed that :math:`u_{top}\\left({t}\\right)` and
    :math:`u_{bot}\\left({t}\\right)` are piecewise linear
    in time with a cyclic component, and that multiple functions are
    superposed.  Also :math:`a\\left(Z\\right)`
    is a piecewise linear function with respect to :math:`Z`


    For this particular function the :math:`\\mathbf{\\theta}` vector for
    each load is given by:

    .. math:: \\mathbf{\\theta}_{i}=
                \\int_{0}^1{
                  \\frac{\\partial}{\\partial Z}
                  \\left(
                    {a\\left({Z}\\right)
                    \\frac{\\partial \\sigma\\left({Z}\\right)}{\\partial Z}}
                  \\right)
                  f\\left({Z}\\right)
                  \\phi_i\\,dZ}

    Where :math:`f\\left({Z}\\right)` is the appropriate z-dependent term
    corresponding to either :math:`u_{top}` or :math:`u_{bot}` homogenisations.


    The time dependent function evaluated at each time produces a matrix
    that we will call :math:`\\mathbf{E}` (not to be confused with other E
    matrices) which is given by:

    .. math:: \\mathbf{E}_{j}=
                 {\\sigma\\left(t_j\\right)}
                 {\\cos\\left(\\omega t_j+\\textrm{phase}\\right)}


    Note that the listed equations above are in terms of normalised depth Z,
    with depth integrations between [0, 1].  However, IF YOU KNOW WHAT YOU
    ARE DOING the integrations can be done using non-normalised depths.
    The first z value in the piecewise definition a(z) must still be 0
    however the end point for integration will be the final z value in the
    definition of a(z).  If you are doing this then your `m` values will
    include the normalising Factor. e.g. m = [pi/2/H, 3*pi/2/H] and a(z) is
    defined in two layers [0, z1], [z1, zend] as opposed to
    m = [pi/2, 3*pi/2] and a(Z) is two layers [0, z1/H], [z1/H, zend/H].

    """

    foft_Ipsiw_the = np.zeros((len(m), len(tvals)))


    if not a is None:
        if drn==1:
            zdist = PolyLine(a.x1,a.x2, np.ones_like(a.x1), np.ones_like(a.x2))
            #bot_vs_time=None
        else:
            zdist = PolyLine(a.x1,a.x2, a.x2[-1]-a.x1, a.x2[-1]-a.x2)

        if not top_vs_time is None:
            if top_omega_phase is None:
                top_omega_phase = [None] * len(top_vs_time)

            theta = integ.pdim1sin_D_aDb_linear(m, a, zdist)
            if not theta_zero_indexes is None:
                theta[theta_zero_indexes] = 0.0
            for top_vs_t, om_ph in zip(top_vs_time, top_omega_phase):
                E = pwise.pinterp_x_y(top_vs_t, tvals)
                if not om_ph is None:
                    omega, phase = om_ph
                    E*= np.cos(omega * tvals + phase)
                foft_Ipsiw_the += E[None, :]*np.dot(Ipsiw, theta)[:, None]

        if not bot_vs_time is None:
            if bot_omega_phase is None:
                bot_omega_phase = [None] * len(bot_vs_time)

            theta = integ.pdim1sin_D_aDb_linear(m, a, PolyLine(a.x1,a.x2,a.x1,a.x2))
            if not theta_zero_indexes is None:
                theta[theta_zero_indexes] = 0.0
            for bot_vs_t, om_ph in zip(bot_vs_time, bot_omega_phase):
                E = pwise.pinterp_x_y(bot_vs_t, tvals)
                if not om_ph is None:
                    omega, phase = om_ph
                    E*= np.cos(omega * tvals + phase)
                foft_Ipsiw_the += E[None, :]*np.dot(Ipsiw, theta)[:, None]

    #theta is 1d array, Igamv is nieg by neig array, np.dot(Igamv, theta)
    #and np.dot(theta, Igamv) will give differetn 1d arrays.
    #Basically np.dot(Igamv, theta) gives us what we want i.e.
    #theta was treated as a column array.  The alternative
    #np.dot(theta, Igamv) would have treated theta as a row vector.
    return foft_Ipsiw_the


def dim1sin_E_Igamv_the_mvpl(m,
                            eigs,
                            tvals,
                            Igamv,
                            moving_loads,
#                            pseudo_k,
#                            mag_vs_time,
#                            omega_phase=None,
                            dT=1.0,
                            theta_zero_indexes=None,
                            implementation='vectorized'):
    """Calculate E and theta parts and assemble E_Igamv_the matrix
    for loading terms of the form a(Z) * delta(Z-Zd)*mag(t) where mag is
    piecewise linear in time multiplied by cos(omega * t + phase).


    Make the E*inverse(gam*v)*theta part of solution
    u(Z,t)=phi*v*E*inverse(gam*v)*theta for terms of the form
    a(Z) * delta(Z-Zd)*mag(t).
    The contribution of each `mag_vs_time`-`omega_phase` pairing and each zval
    are superposed. The result is an array
    of size (neig, len(tvals)). So each column is the are the column vector
    E*inverse(gam*v)*theta calculated at each output time.  This will allow
    us later to do u(Z,t) = phi*v*E_Igamv_the.

    Uses sin(m*Z) in the calculation of theta.


    Parameters
    ----------
    m : ``list`` of ``float``
        Eigenvalues of BVP, the m in sin(m*Z). Generate with
        geotecha.speccon.m_from_sin_mx.
    eigs : 1d numpy.ndarray
        List of eigenvalues of the spectral matrix i.e. Eigenvalues of the
        square Igam_psi matrix.
    tvals : 1d numpy.ndarray`
        List of time values to evaluate E matrix at.
    Igamv : ndarray
        Speccon matrix.  Igamv = inverse of [gam * v])
    moving_loads : list of MovingPointLoads objects
        List of loads to apply .
    dT : ``float``, optional
        Time factor multiple for numerical convieniece. Default dT=1.0.
    theta_zero_indexes : slice/list etc., optional
        A slice object, list, etc that can be used for numpy fancy indexing.
        Any specified index of the theta vector will be set to zero.  This is
        useful when using the spectral method with block matrices and the
        loading term only refers to a subset of the equations.  When using
        block matrices m should be the same size as the block matrix.
        Default theta_zero_indexes=None i.e. no elements of theta will be
        set to zero.

    Returns
    -------
    E_Igamv_the : ndarray
        Loading matrix of size (neig, len(tvals)).

    Notes
    -----
    Assuming the loads are formulated as the product of separate time and depth
    dependant functions as well as a cyclic component:


    """

    E_the = np.zeros((len(tvals), len(eigs), len(m)))
    #axes are (t,eig,theta), they will be transposed at the end of the function.

    if not theta_zero_indexes is None:
        avoid = range(len(m))[theta_zero_indexes]
    else:
        avoid = []

    for mvpl in moving_loads:
        plines, omega_phases = mvpl.convert_to_specbeam()

        for mag_vs_t, omega_phase in zip(plines, omega_phases):
            omega, phase = omega_phase
            for i, mi in enumerate(m):
                if i in avoid:
                    continue
                E_the[:, :, i] += (
                        integ.pEload_sinlinear(mag_vs_t,
                                         omega*mi, phase*mi,
                                         eigs,
                                         tvals, dT,
                                         implementation=implementation))


#    if not theta_zero_indexes is None:
#         E_the[:, :, theta_zero_indexes] = 0.0

    E_the *= Igamv[np.newaxis, :, :]

    E_Igamv_the = E_the.sum(axis=-1).T





#    if omega_phase is None:
#            omega_phase = [None] * len(mag_vs_time)
#
#    for z, k, mag_vs_t, om_ph in zip(zvals, pseudo_k, mag_vs_time, omega_phase):
#        if mag_vs_t is None:
#            continue
#        theta = k * np.sin(z * m)
#        if not theta_zero_indexes is None:
#            theta[theta_zero_indexes] = 0.0
#        if not om_ph is None:
#            omega, phase = om_ph
#            E = integ.pEload_coslinear(mag_vs_t, omega, phase, eigs, tvals, dT, implementation=implementation)
#        else:
#            E = integ.pEload_linear(mag_vs_t, eigs, tvals, dT, implementation=implementation)
#        E_Igamv_the += (E*np.dot(Igamv, theta)).T


    return E_Igamv_the



if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])