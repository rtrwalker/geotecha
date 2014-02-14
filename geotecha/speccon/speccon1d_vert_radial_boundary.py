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
Created on Wed Oct 16 16:08:19 2013

@author: Rohan Walker
"""
from __future__ import division, print_function
import geotecha.piecewise.piecewise_linear_1d as pwise
from geotecha.piecewise.piecewise_linear_1d import PolyLine
import geotecha.speccon.integrals as integ
import random



import itertools

import geotecha.inputoutput.inputoutput as inputoutput

import geotecha.speccon.speccon1d as speccon1d

import geotecha.plotting.one_d #import MarkersDashesColors as MarkersDashesColors

import sys
import textwrap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

try:
    #for python 2 to 3 stuff see http://python3porting.com/stdlib.html
    #for using BytesIO instead of StringIO see http://stackoverflow.com/a/3423935/2530083
    from io import BytesIO as StringIO
except ImportError:
    from StringIO import StringIO



class speccon1d_vr(speccon1d.Speccon1d):
    """
    speccon1d(reader)

    1d consolidation with:

    - vertical and radial drainage (radial drainage uses the eta method)
    - material properties that are constant in time but piecewsie linear with
      depth

      - vertical permeability
      - horizontal permeability
      - lumped drain parameter eta
      - volume compressibilty

    - surcharge and vacuum loading

      - distribution with depth does not change over time
      - magnitude varies piecewise linear with time
      - multiple loads can be superposed

    - pore pressure boundary conditions at top and bottom vary piecewise
      linear with time
    - calculates

      - excess pore pressure at depth
      - average excess pore pressure between depths
      - settlement between depths





    Parameters
    ----------
    reader : object that can be run with exec to produce a module
        reader can be for examplestring, fileobject, StringIO.`reader`
        should contain a statements such as H = 1, drn=0 corresponding to the
        input attributes listed below.  The user should pick an appropriate
        combination of attributes for their analysis.  e.g. don't put dTh=,
        kh=, et=, if you are not modelling radial drainage.  You do not have to
        initialize with `reader` but you should know what you are doing.

    Attributes
    ----------
    H : float, optional
        total height of soil profile. default = 1.0
    mvref : float, optional
        reference value of volume compressibility mv (used with `H` in
        settlement calculations). default = 1.0
    kvref : float, optional
        reference value of vertical permeability kv (only used for pretty
        output). default = 1.0
    khref : float, optional
        reference value of horizontal permeability kh (only used for
        pretty output). default = 1.0
    etref : float, optional
        reference value of lumped drain parameter et (only used for pretty
        output). default = 1.0
    drn : {0, 1}, optional
        drainage boundary condition. default = 0
        0 = Pervious top pervious bottom (PTPB)
        1 = Pervious top impoervious bottom (PTIB)
    dT : float, optional
        convienient normaliser for time factor multiplier. default = 1.0
    neig: int, optional
        number of series terms to use in solution. default = 2
    dTv: float, optional
        vertical reference time factor multiplier.  dTv is calculated with
        the chosen reference values of kv and mv: dTv = kv /(mv*gamw) / H ^ 2
    dTh : float, optional
        horizontal reference time factor multiplier.  dTh is calculated with
        the reference values of kh, et, and mv: dTh = kh / (mv * gamw) * et
    mv : PolyLine, optional
        normalised volume compressibility PolyLine(depth, mv)
    kh : PolyLine, optional
        normalised horizontal permeability PolyLine(depth, kh)
    kv : PolyLine , optional
        normalised vertical permeability PolyLine(depth, kv)
    et : PolyLine, optional
        normalised vertical drain parameter PolyLine(depth, et).
        et = 2 / (mu * re^2) where mu is smear-zone/geometry parameter and re
        is radius of influence of vertical drain
    surcharge_vs_depth : list of Polyline, optional
        surcharge variation with depth. PolyLine(depth, multiplier)
    surcharge_vs_time : list of Polyline, optional
        surcharge magnitude variation with time. PolyLine(time, magnitude)
    surcharge_omega_phase : list of 2 element tuples, optional
        (omega, phase) to define cyclic variation of surcharve. i.e.
        mag_vs_time * cos(omega*t + phase). if surcharge_omega_phase is None
        then cyclic componenet will be ignored.  if surcharge_omega_phase is a
        list then if any member is None then cyclic component will not be
        applied for that load combo.

    vacuum_vs_depth : list of Polyline, optinal
        vacuum variation with depth. PolyLine(depth, multiplier)
    vacuum_vs_time : list of Polyline, optional
        vacuum magnitude variation with time. Polyline(time, magnitude)
    vacuum_omega_phase : list of 2 element tuples, optional
        (omega, phase) to define cyclic variation of vacuum. i.e.
        mag_vs_time * cos(omega*t + phase). if vacuum_omega_phase is None
        then cyclic componenet will be ignored.  if vacuum_omega_phase is a
        list then if any member is None then cyclic component will not be
        applied for that load combo.
    top_vs_time : list of Polyline, optional
        top p.press variation with time. Polyline(time, magnitude)
    top_omega_phase : list of 2 element tuples, optional
        (omega, phase) to define cyclic variation of top BC. i.e.
        mag_vs_time * cos(omega*t + phase). if top_omega_phase is None
        then cyclic componenet will be ignored.  if top_omega_phase is a
        list then if any member is None then cyclic component will not be
        applied for that load combo.
    bot_vs_time : list of Polyline, optional
        bottom p.press variation with time. Polyline(time, magnitude).
        When drn=1, i.e. PTIB, bot_vs_time is equivilent to saying
        D[u(1,t), Z] = bot_vs_time
    bot_omega_phase : list of 2 element tuples, optional
        (omega, phase) to define cyclic variation of bot BC. i.e.
        mag_vs_time * cos(omega*t + phase). if bot_omega_phase is None
        then cyclic componenet will be ignored.  if bot_omega_phase is a
        list then if any member is None then cyclic component will not be
        applied for that load combo.
    fixed_ppress: list of 3 element tuple, optional
        (zfixed, pseudo_k, PolyLine(time, magnitude)).  zfixed is the
        normalised z at which pore pressure is fixed. pseudo_k is a
        permeability-like coefficient that controls how quickly the pore
        pressure reduces to the fixed value (pseudo_k should be as high as
        possible without causing numerical difficulties). If the third
        element of the tuple is None then the pore pressure will be fixed at
        zero rather than a prescribed mag_vs_time PolyLine
    fixed_ppress_omega_phase : list of 2 element tuples, optional
        (omega, phase) to define cyclic variation of fixed ppress. i.e.
        mag_vs_time * cos(omega*t + phase). if fixed_ppress _omega_phase is
        None then cyclic componenet will be ignored.  if vacuum_omega_phase is
        a list then if any member is None then cyclic component will not be
        applied for that load combo.
    ppress_z : list_like of float, optional
        normalised z to calc pore pressure at
    avg_ppress_z_pairs : list of two element list of float, optional
        nomalised zs to calc average pore pressure between
        e.g. average of all profile is [[0,1]]
    settlement_z_pairs : list of two element list of float, optional
        normalised depths to calculate normalised settlement between.
        e.g. surface settlement would be [[0, 1]]
    tvals : list of float
        times to calculate output at
    ppress_z_tval_indexes: list/array of int, slice, optional
        indexes of `tvals` at which to calculate ppress_z. i.e. only calc
        ppress_z at a subset of the `tvals` values.  default =
        slice(None, None) i.e. use all the `tvals`.
    avg_ppress_z_pairs_tval_indexes: list/array of int, slice, optional
        indexes of `tvals` at which to calculate avg_ppress_z_pairs.
        i.e. only calc avg_ppress_z_pairs at a subset of the `tvals` values.
        default = slice(None, None) i.e. use all the `tvals`.
    settlement_z_pairs_tval_indexes: list/array of int, slice, optional
        indexes of `tvals` at which to calculate settlement_z_pairs.
        i.e. only calc settlement_z_pairs at a subset of the `tvals` values.
        default = slice(None, None) i.e. use all the `tvals`.
    por : ndarray, only present if ppress_z is input
        calculated pore pressure at depths correspoinding to `ppress_z` and
        times corresponding to `tvals`.  This is an output array of
        size (len(ppress_z), len(tvals[ppress_z_tval_indexes])).
    avp : ndarray, only present if avg_ppress_z_pairs is input
        calculated average pore pressure between depths correspoinding to
        `avg_ppress_z_pairs` and times corresponding to `tvals`.  This is an
        output array of size
        (len(avg_ppress_z_pairs), len(tvals[avg_ppress_z_pairs_tval_indexes])).
    set : ndarray, only present if settlement_z_pairs is input
        settlement between depths coreespoinding to `settlement_z_pairs` and
        times corresponding to `tvals`.  This is an output array of size
        (len(avg_ppress_z_pairs), len(tvals[settlement_z_pairs_tval_indexes]))
    implementation: ['scalar', 'vectorized','fortran'], optional
        where possible use the `implementation`, implementation.  'scalar'=
        python loops (slowest), 'vectorized' = numpy (fast), 'fortran' =
        fortran extension (fastest).  Note only some functions have multiple
        implementations.

    RLzero: float, optional
        reduced level of the top of the soil layer.  If RLzero is not None
        then all depths (in plots and results) will be transformed to an RL
        by RL = RLzero - z*H.  If RLzero is None (i.e. the default) then all
        depths will be reported  z*H (i.e. positive numbers).

    plot_properties : dict of dict, optional
        dictionary that passes some overridessome of the plot properties.
        Each member of `plot_properties` will correspond to one of the plots.
        e.g. plot_properties['por'] will pass properties to the pore pressure
        vs depth plot.

    Notes
    -----
    #TODO: explain lists of input must have same len.
    governing equation:



    References
    ----------
    All based on work by Dr Rohan Walker [1]_, [2]_, [3]_, [4]_

    .. [1] Walker, Rohan. 2006. 'Analytical Solutions for Modeling Soft Soil Consolidation by Vertical Drains'. PhD Thesis, Wollongong, NSW, Australia: University of Wollongong.
    .. [2] Walker, R., and B. Indraratna. 2009. 'Consolidation Analysis of a Stratified Soil with Vertical and Horizontal Drainage Using the Spectral Method'. Geotechnique 59 (5) (January): 439-449. doi:10.1680/geot.2007.00019.
    .. [3] Walker, Rohan, Buddhima Indraratna, and Nagaratnam Sivakugan. 2009. 'Vertical and Radial Consolidation Analysis of Multilayered Soil Using the Spectral Method'. Journal of Geotechnical and Geoenvironmental Engineering 135 (5) (May): 657-663. doi:10.1061/(ASCE)GT.1943-5606.0000075.
    .. [4] Walker, Rohan T. 2011. Vertical Drain Consolidation Analysis in One, Two and Three Dimensions'. Computers and Geotechnics 38 (8) (December): 1069-1077. doi:10.1016/j.compgeo.2011.07.006.

    """

    def _setup(self):

        self._attributes = (
            'H drn dT neig mvref kvref khref etref dTh dTv mv kh kv et '
            'surcharge_vs_depth surcharge_vs_time '
            'vacuum_vs_depth vacuum_vs_time '
            'top_vs_time bot_vs_time '
            'ppress_z avg_ppress_z_pairs settlement_z_pairs tvals '
            'implementation ppress_z_tval_indexes '
            'avg_ppress_z_pairs_tval_indexes settlement_z_pairs_tval_indexes '
            'fixed_ppress surcharge_omega_phase vacuum_omega_phase '
            'fixed_ppress_omega_phase top_omega_phase bot_omega_phase '
            'RLzero '
            'plot_properties').split()

        self._attribute_defaults = {
            'H': 1.0, 'drn': 0, 'dT': 1.0, 'neig': 2, 'mvref':1.0,
            'kvref': 1.0, 'khref': 1.0, 'etref': 1.0,
            'implementation': 'vectorized',
            'ppress_z_tval_indexes': slice(None, None),
            'avg_ppress_z_pairs_tval_indexes': slice(None, None),
            'settlement_z_pairs_tval_indexes': slice(None, None),
            'plot_properties': dict() }

        self._attributes_that_should_be_lists= (
            'surcharge_vs_depth surcharge_vs_time surcharge_omega_phase '
            'vacuum_vs_depth vacuum_vs_time vacuum_omega_phase '
            'top_vs_time top_omega_phase '
            'bot_vs_time bot_omega_phase '
            'fixed_ppress fixed_ppress_omega_phase').split()

        self._attributes_that_should_have_same_x_limits = (
            'mv kv kh et surcharge_vs_depth vacuum_vs_depth').split()

        self._attributes_that_should_have_same_len_pairs = (
            'surcharge_vs_depth surcharge_vs_time '
            'surcharge_vs_time surcharge_omega_phase '
            'vacuum_vs_depth vacuum_vs_time '
            'vacuum_vs_time vacuum_omega_phase '
            'top_vs_time top_omega_phase '
            'bot_vs_time bot_omega_phase '
            'fixed_ppress_omega_phase fixed_ppress').split() #pairs that should have the same length

        self._attributes_to_force_same_len = [
            "surcharge_vs_time surcharge_omega_phase".split(),
            "vacuum_vs_time vacuum_omega_phase".split(),
            "fixed_ppress fixed_ppress_omega_phase".split(),
            "top_vs_time top_omega_phase".split(),
            "bot_vs_time bot_omega_phase".split()]

        self._zero_or_all = [
            'dTh kh et'.split(),
            'dTv kv'.split(),
            'surcharge_vs_depth surcharge_vs_time'.split(),
            'vacuum_vs_depth vacuum_vs_time'.split()]
        self._at_least_one = [
            ['mv'],
            'dTh dTv'.split(),
            'kh kv'.split(),
            ('surcharge_vs_time vacuum_vs_time top_vs_time '
                'bot_vs_time fixed_ppress').split(),
            ['tvals'],
            'ppress_z avg_ppress_z_pairs settlement_z_pairs'.split()]

        self._one_implies_others = [
            'vacuum_vs_time dTh kh et'.split(),
            'vacuum_vs_depth dTh kh et'.split(),
            ('surcharge_omega_phase surcharge_vs_depth '
                'surcharge_vs_time').split(),
            'vacuum_omega_phase vacuum_vs_depth vacuum_vs_time'.split(),
            'fixed_ppress_omega_phase fixed_ppress'.split(),
            'top_omega_phase top_vs_time'.split(),
            'bot_omega_phase bot_vs_time'.split()]


        #these explicit initializations are just to make coding easier
        self.H = self._attribute_defaults.get('H', None)
        self.drn = self._attribute_defaults.get('drn', None)
        self.dT = self._attribute_defaults.get('dT', None)
        self.neig = self._attribute_defaults.get('neig', None)
        self.mvref = self._attribute_defaults.get('mvref', None)
        self.kvref = self._attribute_defaults.get('kvref', None)
        self.khref = self._attribute_defaults.get('khref', None)
        self.etref = self._attribute_defaults.get('etref', None)
        self.dTh = None
        self.dTv = None
        self.mv = None
        self.kh = None
        self.kv = None
        self.et = None
        self.surcharge_vs_depth = None
        self.surcharge_vs_time = None
        self.surcharge_omega_phase = None

        self.vacuum_vs_depth = None
        self.vacuum_vs_time = None
        self.vacuum_omega_phase = None
        self.top_vs_time = None
        self.top_omega_phase = None
        self.bot_vs_time = None
        self.bot_omega_phase = None
        self.fixed_ppress_omega_phase = None
        self.fixed_ppress = None

        self.ppress_z = None
        self.avg_ppress_z_pairs = None
        self.settlement_z_pairs = None
        self.tvals = None
        self.RLzero = None

        self.plot_properties = self._attribute_defaults.get('plot_properties', None)

        self.ppress_z_tval_indexes = self._attribute_defaults.get('ppress_z_tval_indexes', None)
        self.avg_ppress_z_pairs_tval_indexes = self._attribute_defaults.get('avg_ppress_z_pairs_tval_indexes', None)
        self.settlement_z_pairs_tval_indexes = self._attribute_defaults.get('settlement_z_pairs_tval_indexes', None)


        return


#    def make_all(self):
#        """run checks, make all arrays, make output
#
#        Generally run this after input is in place (either through
#        initializing the class with a reader/text/fileobject or
#        through some other means)
#
#        See also
#        --------
#        check_all
#        make_time_independent_arrays
#        make_time_dependent_arrays
#        make_output
#
#        """
#
#        self.check_all()
#        self.make_time_independent_arrays()
#        self.make_time_dependent_arrays()
#        self.make_output()
#        return

    def make_time_independent_arrays(self):
        """make all time independent arrays


        See also
        --------
        self._make_m : make the basis function eigenvalues
        self._make_gam : make the mv dependent gamma matrix
        self._make_psi : make the kv, kh, et dependent psi matrix
        self._make_eigs_and_v : make eigenvalues, eigenvectors and I_gamv

        """





        self._make_m()
        self._make_gam()
        self._make_psi()
        self._make_eigs_and_v()

        return

    def make_time_dependent_arrays(self):
        """make all time dependent arrays

        See also
        --------
        self.make_E_Igamv_the()

        """
        self.tvals = np.asarray(self.tvals)
        self.make_E_Igamv_the()
        self.v_E_Igamv_the=np.dot(self.v, self.E_Igamv_the)
        return

    def make_output(self):
        """make all output"""

        if not self.ppress_z is None:
            self._make_por()
        if not self.avg_ppress_z_pairs is None:
            self._make_avp()
        if not self.settlement_z_pairs is None:
            self._make_set()
        return

    def _make_m(self):
        """make the basis function eigenvalues

        Notes
        -----

        .. math:: m_i =\\pi*\\left(i+1-drn/2\\right)

        for :math:`i = 1\:to\:neig-1`

        """
        if sum(v is None for v in[self.neig, self.drn])!=0:
            raise ValueError('neig and/or drn is not defined')
        self.m = integ.m_from_sin_mx(np.arange(self.neig), self.drn)
        return

    def _make_m(self):
        """make the basis function eigenvalues

        m in u = sin(m * Z)

        Notes
        -----

        .. math:: m_i =\\pi*\\left(i+1-drn/2\\right)

        for :math:`i = 1\:to\:neig-1`

        """

        if sum(v is None for v in[self.neig, self.drn])!=0:
            raise ValueError('neig and/or drn is not defined')
        self.m = integ.m_from_sin_mx(np.arange(self.neig), self.drn)
        return

    def _make_gam(self):
        """make the mv dependant gam matrix

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----

        Creates the :math: `\Gam` matrix which occurs in the following equation:

        .. math:: \\mathbf{\\Gamma}\\mathbf{A}'=\\mathbf{\\Psi A}+loading\\:terms

        `self.gam`, :math:`\Gamma` is given by:

        .. math:: \\mathbf{\Gamma}_{i,j}=\\int_{0}^1{{m_v\\left(z\\right)}{sin\\left({m_j}z\\right)}{sin\\left({m_i}z\\right)}\,dz}


        """
#        self.gam = integ.dim1sin_af_linear(self.m,self.mv.y1, self.mv.y2, self.mv.x1, self.mv.x2)
        self.gam = integ.pdim1sin_af_linear(self.m,self.mv, implementation=self.implementation)
        self.gam[np.abs(self.gam)<1e-8]=0.0
        return

    def _make_psi(self):
        """make kv, kh, et dependant psi matrix

        Notes
        -----
        Creates the :math: `\Psi` matrix which occurs in the following equation:

        .. math:: \\mathbf{\\Gamma}\\mathbf{A}'=\\mathbf{\\Psi A}+loading\\:terms

        `self.psi`, :math:`\Psi` is given by:

        .. math:: \\mathbf{\Psi}_{i,j}=dT_h\\mathbf{A}_{i,j}=\\int_{0}^1{{k_h\\left(z\\right)}{\eta\\left(z\\right)}\\phi_i\\phi_j\\,dz}-dT_v\\int_{0}^1{\\frac{d}{dz}\\left({k_z\\left(z\\right)}\\frac{d\\phi_j}{dz}\\right)\\phi_i\\,dz}

        """

        self.psi = np.zeros((self.neig, self.neig))
        #kv part
        if sum([v is None for v in [self.kv, self.dTv]])==0:
#            self.psi -= self.dTv / self.dT * integ.dim1sin_D_aDf_linear(self.m, self.kv.y1, self.kv.y2, self.kv.x1, self.kv.x2)
            self.psi -= self.dTv / self.dT * integ.pdim1sin_D_aDf_linear(self.m, self.kv, implementation=self.implementation)
        #kh & et part
        if sum([v is None for v in [self.kh, self.et, self.dTh]])==0:
            kh, et = pwise.polyline_make_x_common(self.kh, self.et)
#            self.psi += self.dTh / self.dT * integ.dim1sin_abf_linear(self.m,kh.y1, kh.y2, et.y1, et.y2, kh.x1, kh.x2)
            self.psi += self.dTh / self.dT * integ.pdim1sin_abf_linear(self.m, self.kh, self.et, implementation=self.implementation)
        #fixed pore pressure part
        if not self.fixed_ppress is None:
            for (zfixed, pseudo_k, mag_vs_time) in self.fixed_ppress:
                self.psi += pseudo_k / self.dT * np.sin(self.m[:, np.newaxis] * zfixed) * np.sin(self.m[np.newaxis, :] * zfixed)

#                self.psi += k * np.sin(self.m[:, np.newaxis] * z) * np.sin(self.m[np.newaxis, :] * z)
        self.psi[np.abs(self.psi) < 1e-8]=0.0
        return

    def _make_eigs_and_v(self):
        """make Igam_psi, v and eigs, and Igamv

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

        Igam_psi = np.dot(np.linalg.inv(self.gam), self.psi)

        self.eigs, self.v = np.linalg.eig(Igam_psi)

        self.Igamv = np.linalg.inv(np.dot(self.gam, self.v))
        return

    def make_E_Igamv_the(self):
        """sum contributions from all loads

        Calculates all contributions to E*inverse(gam*v)*theta part of solution
        u=phi*vE*inverse(gam*v)*theta. i.e. surcharge, vacuum, top and bottom
        pore pressure boundary conditions. `make_load_matrices will create
        `self.E_Igamv_the`.  `self.E_Igamv_the`  is an array
        of size (neig, len(tvals)). So the columns are the column array
        E*inverse(gam*v)*theta calculated at each output time.  This will allow
        us later to do u = phi*v*self.E_Igamv_the

        See also
        --------
        _make_E_Igamv_the_surcharge :  surchage contribution
        _make_E_Igamv_the_vacuum : vacuum contribution
        _make_E_Igamv_the_BC : top boundary pore pressure contribution
        _make_E_Igamv_the_bot : bottom boundary pore pressure contribution

        """

        self.E_Igamv_the = np.zeros((self.neig,len(self.tvals)))
        if sum([v is None for v in [self.surcharge_vs_depth, self.surcharge_vs_time]])==0:
            self._make_E_Igamv_the_surcharge()
            self.E_Igamv_the += self.E_Igamv_the_surcharge
        if sum([v is None for v in [self.vacuum_vs_depth, self.vacuum_vs_time, self.et, self.kh,self.dTh]])==0:
            if self.dTh!=0:
                self._make_E_Igamv_the_vacuum()
                self.E_Igamv_the += self.E_Igamv_the_vacuum
        if not self.top_vs_time is None or not self.bot_vs_time is None:
            self._make_E_Igamv_the_BC()
            self.E_Igamv_the += self.E_Igamv_the_BC
        if not self.fixed_ppress is None:
            self._make_E_Igamv_the_fixed_ppress()
            self.E_Igamv_the +=self.E_Igamv_the_fixed_ppress

        return

    def _make_E_Igamv_the_surcharge(self):
        """make the surcharge loading matrices

        Make the E*inverse(gam*v)*theta part of solution u=phi*vE*inverse(gam*v)*theta.
        The contribution of each surcharge load is added and put in
        `self.E_Igamv_the_surcharge`. `self.E_Igamv_the_surcharge` is an array
        of size (neig, len(tvals)). So the columns are the column array
        E*inverse(gam*v)*theta calculated at each output time.  This will allow
        us later to do u = phi*v*self.E_Igamv_the_surcharge

        Notes
        -----
        Assuming the load are formulated as the product of separate time and depth
        dependant functions:

        .. math:: \\sigma\\left({Z,t}\\right)=\\sigma\\left({Z}\\right)\\sigma\\left({t}\\right)

        the solution to the consolidation equation using the spectral method has
        the form:

        .. math:: u\\left(Z,t\\right)=\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}

        `_make_E_Igamv_the_surcharge` will create `self.E_Igamv_the_surcharge` which is
        the :math:`\\mathbf{E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}`
        part of the solution for all surcharge loads

        """
        self.E_Igamv_the_surcharge = speccon1d.dim1sin_E_Igamv_the_aDmagDt_bilinear(self.m, self.eigs, self.tvals, self.Igamv, self.mv, self.surcharge_vs_depth, self.surcharge_vs_time, self.surcharge_omega_phase, self.dT)
#        self.E_Igamv_the_surcharge  = speccon1d.dim1sin_E_Igamv_the_aDmagDt_bilinear(self.m, self.eigs, self.mv, self.surcharge_vs_depth, self.surcharge_vs_time, self.tvals, self.Igamv, self.dT)
        return


    def _make_E_Igamv_the_vacuum(self):
        """make the vacuum loading matrices

        Make the E*inverse(gam*v)*theta part of solution u=phi*vE*inverse(gam*v)*theta.
        The contribution of each vacuum load is added and put in
        `self.E_Igamv_the_vacuum`. `self.E_Igamv_the_vacuum` is an array
        of size (neig, len(tvals)). So the columns are the column array
        E*inverse(gam*v)*theta calculated at each output time.  This will allow
        us later to do u = phi*v*self.E_Igamv_the_vacuum

        Notes
        -----
        Assuming the load are formulated as the product of separate time and depth
        dependant functions:

        .. math:: \\sigma\\left({Z,t}\\right)=\\sigma\\left({Z}\\right)\\sigma\\left({t}\\right)

        the solution to the consolidation equation using the spectral method has
        the form:

        .. math:: u\\left(Z,t\\right)=\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}

        `_make_E_Igamv_the_surcharge` will create `self.E_Igamv_the_surcharge` which is
        the :math:`\\mathbf{E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}`
        part of the solution for all surcharge loads

        """

#        self.E_Igamv_the_vacuum = self.dTh * speccon1d.dim1sin_E_Igamv_the_abmag_bilinear(self.m, self.eigs, self.kh, self.et,
#                                                                        self.vacuum_vs_depth, self.vacuum_vs_time, self.tvals, self.Igamv, self.dT)
        self.E_Igamv_the_vacuum = self.dTh * speccon1d.dim1sin_E_Igamv_the_abmag_bilinear(self.m, self.eigs, self.tvals, self.Igamv, self.kh, self.et,
                                                                        self.vacuum_vs_depth, self.vacuum_vs_time, self.vacuum_omega_phase, self.dT)
#        speccon1d.dim1sin_E_Igamv_the_aDmagDt_bilinear(self.m, self.eigs, self.tvals, self.Igamv, self.mv, self.surcharge_vs_depth, self.surcharge_vs_time, self.surcharge_omega_phase, self.dT)
        return

    def _make_E_Igamv_the_fixed_ppress(self):
        """make the fixed pore pressure loading matrices

        """

        self.E_Igamv_the_fixed_ppress = np.zeros((self.neig, len(self.tvals)))

        if not self.fixed_ppress is None:
            zvals = [v[0] for v in self.fixed_ppress]
            pseudo_k = [v[1] for v in self.fixed_ppress]
            mag_vs_time = [v[2] for v in self.fixed_ppress]
            self.E_Igamv_the_fixed_ppress += speccon1d.dim1sin_E_Igamv_the_deltamag_linear(self.m, self.eigs, self.tvals, self.Igamv, zvals, pseudo_k, mag_vs_time, self.fixed_ppress_omega_phase, self.dT)

    def _make_E_Igamv_the_BC(self):
        """make the boundary condition loading matrices

        """
        self.E_Igamv_the_BC = np.zeros((self.neig, len(self.tvals)))
        #mv * du/dt component
        #self.E_Igamv_the_BC -= speccon1d.dim1sin_E_Igamv_the_BC_aDfDt_linear(self.drn, self.m, self.eigs, self.mv, self.top_vs_time, self.bot_vs_time, self.tvals, self.Igamv, self.dT)
        self.E_Igamv_the_BC -= speccon1d.dim1sin_E_Igamv_the_BC_aDfDt_linear(self.drn, self.m, self.eigs, self.tvals, self.Igamv, self.mv, self.top_vs_time, self.bot_vs_time, self.top_omega_phase, self.bot_omega_phase, self.dT)

        #dTh * kh * et * u component
        if sum([v is None for v in [self.et, self.kh, self.dTh]])==0:
            if self.dTh!=0:
#                self.E_Igamv_the_BC -= self.dTh  * speccon1d.dim1sin_E_Igamv_the_BC_abf_linear(self.drn, self.m, self.eigs, self.kh, self.et, self.top_vs_time, self.bot_vs_time, self.tvals, self.Igamv, self.dT)
                self.E_Igamv_the_BC -= self.dTh  * speccon1d.dim1sin_E_Igamv_the_BC_abf_linear(self.drn, self.m, self.eigs, self.tvals, self.Igamv, self.kh, self.et, self.top_vs_time, self.bot_vs_time, self.top_omega_phase, self.bot_omega_phase, self.dT)

        #dTv * d/dZ(kv * du/dZ) component
        if sum([v is None for v in [self.kv, self.dTv]])==0:
            if self.dTv!=0:
#                self.E_Igamv_the_BC += self.dTv * speccon1d.dim1sin_E_Igamv_the_BC_D_aDf_linear(self.drn, self.m, self.eigs, self.kv, self.top_vs_time, self.bot_vs_time, self.tvals, self.Igamv, self.dT)
                self.E_Igamv_the_BC += self.dTv * speccon1d.dim1sin_E_Igamv_the_BC_D_aDf_linear(self.drn, self.m, self.eigs, self.tvals, self.Igamv, self.kv, self.top_vs_time, self.bot_vs_time, self.top_omega_phase, self.bot_omega_phase, self.dT)

        #the pseudo_k * delta(z-zfixed)*u component, i.e. the fixed_ppress part
        if not self.fixed_ppress is None:
            zvals = [v[0] for v in self.fixed_ppress]
            pseudo_k = [v[1] for v in self.fixed_ppress]
            self.E_Igamv_the_BC -= speccon1d.dim1sin_E_Igamv_the_BC_deltaf_linear(self.drn, self.m, self.eigs, self.tvals, self.Igamv, zvals, pseudo_k, self.top_vs_time, self.bot_vs_time, self.top_omega_phase, self.bot_omega_phase, self.dT)

    def _make_por(self):
        """make the pore pressure output

        makes `self.por`, the average pore pressure at depths corresponding to
        self.ppress_z and times corresponding to self.tvals.  `self.por`  has size
        (len(ppress_z), len(tvals)).

        Notes
        -----
        Solution to consolidation equation with spectral method for pore pressure at depth is :

        .. math:: u\\left(Z,t\\right)=\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}+u_{top}\\left({t}\\right)\\left({1-Z}\\right)+u_{bot}\\left({t}\\right)\\left({Z}\\right)

        For pore pressure :math:`\\Phi` is simply :math:`sin\\left({mZ}\\right)` for each value of m


        """

        self.por= speccon1d.dim1sin_f(self.m, self.ppress_z, self.tvals[self.ppress_z_tval_indexes], self.v_E_Igamv_the[:, self.ppress_z_tval_indexes], self.drn, self.top_vs_time, self.bot_vs_time, self.top_omega_phase, self.bot_omega_phase)
        return

    def _make_avp(self):
        """calculate average pore pressure

        makes `self.avp`, the average pore pressure at depths corresponding to
        self.avg_ppress_z_pairs and times corresponding to self.tvals.  `self.avp`  has size
        (len(ppress_z), len(tvals)).


        Notes
        -----
        The average pore pressure between Z1 and Z2 is given by:

        .. math:: \\overline{u}\\left(\\left({Z_1,Z_2}\\right),t\\right)=\\int_{Z_1}^{Z_2}{\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}+u_{top}\\left({t}\\right)\\left({1-Z}\\right)+u_{bot}\\left({t}\\right)\\left({Z}\\right)\,dZ}/\\left({Z_2-Z_1}\\right)

        """

        self.avp=speccon1d.dim1sin_avgf(self.m, self.avg_ppress_z_pairs, self.tvals[self.avg_ppress_z_pairs_tval_indexes], self.v_E_Igamv_the[:,self.avg_ppress_z_pairs_tval_indexes], self.drn, self.top_vs_time, self.bot_vs_time, self.top_omega_phase, self.bot_omega_phase)
        return

    def _make_set(self):
        """calculate settlement

        makes `self.set`, the average pore pressure at depths corresponding to
        self.settlement_z_pairs and times corresponding to self.tvals.  `self.set`  has size
        (len(ppress_z), len(tvals)).


        Notes
        -----
        The average settlement between Z1 and Z2 is given by:

        .. math:: \\overline{\\rho}\\left(\\left({Z_1,Z_2}\\right),t\\right)=\\int_{Z_1}^{Z_2}{m_v\\left({Z}\\right)\\left({\\sigma\\left({Z,t}\\right)-u\\left({Z,t}\\right)}\\right)\\,dZ}


        .. math:: \\overline{\\rho}\\left(\\left({Z_1,Z_2}\\right),t\\right)=\\int_{Z_1}^{Z_2}{m_v\\left({Z}\\right)\\sigma\\left({Z,t}\\right)\\,dZ}+\\int_{Z_1}^{Z_2}{m_v\\left({Z}\\right)\\left({\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}+u_{top}\\left({t}\\right)\\left({1-Z}\\right)+u_{bot}\\left({t}\\right)\\left({Z}\\right)}\\right)\\,dZ}

        """

        z1 = np.asarray(self.settlement_z_pairs)[:,0]
        z2 = np.asarray(self.settlement_z_pairs)[:,1]

        self.set=-speccon1d.dim1sin_integrate_af(self.m, self.settlement_z_pairs, self.tvals[self.settlement_z_pairs_tval_indexes],
                                       self.v_E_Igamv_the[:,self.settlement_z_pairs_tval_indexes],
                                        self.drn, self.mv, self.top_vs_time, self.bot_vs_time, self.top_omega_phase, self.bot_omega_phase)

        if not self.surcharge_vs_time is None:

#            self.set+=pwise.pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between_super(self.surcharge_vs_time, self.surcharge_vs_depth, self.mv, self.tvals[self.settlement_z_pairs_tval_indexes], z1, z2, achoose_max=True)
            self.set += pwise.pxa_ya_cos_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between_super(self.surcharge_vs_time, self.surcharge_vs_depth, self.mv, self.tvals[self.settlement_z_pairs_tval_indexes], z1, z2, omega_phase = self.surcharge_omega_phase, achoose_max=True)


#        if sum([v is None for v in [self.cyclic_surcharge_vs_depth, self.cyclic_surcharge_vs_time, self.cyclic_surcharge_omega_phase]])==0:
#            self.set+=pwise.pxa_ya_cos_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between_super(self.cyclic_surcharge_vs_time, self.cyclic_surcharge_omega_phase, self.cyclic_surcharge_vs_depth, self.mv, self.tvals[self.settlement_z_pairs_tval_indexes], z1, z2, achoose_max=True)

        self.set *= self.H * self.mvref
        return


#    def _plot_pore_pressure(self):
#        """make a pore_pressure vs depth plot
#
#        por_prop dictionary options
#        ---------------------------
#        ==================  ================================================
#        option              description
#        ==================  ================================================
#        fig_prop            dict of prop to pass to plt.figure
#        styles              List of dict.  Each dict is for one line.
#                            Each dict contains kwargs for plt.plot
#                            See geotecha.plotting.one_d.MarkersDashesColors
#        xlabel              x-axis label.
#        ylabel              y-axis label
#        has_legend          True or False. default is True
#        legend_prop         dict of prop to pass to ax.legend
#        ==================  ================================================
#
#        """
#
#
#        por_prop = self.plot_properties.pop('por', dict())
#        fig_prop = por_prop.pop('fig_prop', {})
#        legend_prop = por_prop.pop('legend_prop',
#                                   {'title': 'time:', 'fontsize': 8})
#
#        styles = por_prop.pop('style', None)
#        if styles is None:
#            mcd = geotecha.plotting.one_d.MarkersDashesColors(
##                color = 'black',
#                default_marker={'markersize': 7})
#            mcd.construct_styles(markers = range(32), dashes=[0],
#                                 marker_colors=None, line_colors=None)
#
#
#        styles = itertools.cycle(mcd.styles)
#
#
#
#        z = speccon1d.depth_to_reduced_level(self.ppress_z, self.H,
#                                                 self.RLzero)
#        t = self.tvals[self.ppress_z_tval_indexes]
#
#        fig = plt.figure(**fig_prop)
#
#
#
#
#
#        plt.plot(self.por, z)
#
#
#
#        xlabel = por_prop.pop('xlabel', 'Pore pressure, u')
#        plt.xlabel(xlabel)
#
#        if self.RLzero is None:
#            plt.gca().invert_yaxis()
#            ylabel = por_prop.pop('ylabel', 'Depth, z')
#        else:
#            ylabel = por_prop.pop('ylabel', 'RL')
#
#        plt.ylabel(ylabel)
#
#        #apply style to each line
#        [geotecha.plotting.one_d.apply_dict_to_object(line, d)
#            for line, d in zip(fig.gca().get_lines(), styles)]
#        #apply markevery to each line
##        [geotecha.plotting.one_d.apply_dict_to_object(line, d)
##            for line, d in zip(fig.gca().get_lines(), [{'markevery': 0.1}]*len(t))]
#        random.seed(1)
#        [geotecha.plotting.one_d.apply_dict_to_object(line, d)
#            for line, d in
#                zip(fig.gca().get_lines(),
#                    [{'markevery': (random.random()* 0.1, 0.1)} for v in t])]
#        #apply label to each line
#        line_labels = [{'label': '%.3g' % v} for v in t]
#        [geotecha.plotting.one_d.apply_dict_to_object(line, d)
#            for line, d in zip(fig.gca().get_lines(), line_labels)]
#
#        has_legend = por_prop.pop('has_legend', True)
#        if has_legend:
#            leg = fig.gca().legend(**legend_prop)
#            leg.draggable(True)
#
#
#    def _plot_average_pore_pressure(self):
#        """make an average_pore_pressure vs time plot
#
#        avp_prop dictionary options
#        ---------------------------
#        ==================  ================================================
#        option              description
#        ==================  ================================================
#        fig_prop            dict of prop to pass to plt.figure
#        styles              List of dict.  Each dict is for one line.
#                            Each dict contains kwargs for plt.plot
#                            See geotecha.plotting.one_d.MarkersDashesColors
#        xlabel              x-axis label.
#        ylabel              y-axis label
#        has_legend          True or False. default is True
#        legend_prop         dict of prop to pass to ax.legend
#        ==================  ================================================
#
#        """
#
#        avp_prop = self.plot_properties.pop('avp', dict())
#        fig_prop = avp_prop.pop('fig_prop', {})
#        legend_prop = avp_prop.pop('legend_prop',
#                                   {'title': 'Depth interval:', 'fontsize': 8})
#
#        styles = avp_prop.pop('style', None)
#        if styles is None:
#            mcd = geotecha.plotting.one_d.MarkersDashesColors(
##                color = 'black',
#                default_marker={'markersize': 7})
#            mcd.construct_styles(markers = range(32), dashes=[0],
#                                 marker_colors=None, line_colors=None)
#
#
#
#        styles = itertools.cycle(mcd.styles)
#
#
#        t = self.tvals[self.avg_ppress_z_pairs_tval_indexes]
#
#        z_pairs = speccon1d.depth_to_reduced_level(
#            np.asarray(self.avg_ppress_z_pairs), self.H, self.RLzero)
#
#
#        fig = plt.figure(**fig_prop)
#
#
#
#
#
#        plt.plot(t, self.avp.T)
#
#
#
#        xlabel = avp_prop.pop('xlabel', 'Time, t')
#        plt.xlabel(xlabel)
#        ylabel = avp_prop.pop('ylabel', 'Average pore pressure')
#        plt.ylabel(ylabel)
#
#        #apply style to each line
#        [geotecha.plotting.one_d.apply_dict_to_object(line, d)
#            for line, d in zip(fig.gca().get_lines(), styles)]
#        #apply markevery to each line
#        random.seed(1)
#        [geotecha.plotting.one_d.apply_dict_to_object(line, d)
#            for line, d in zip(fig.gca().get_lines(), [{'markevery': (random.random()* 0.1, 0.1)} for v in z_pairs])]
#
#        #apply label to each line
#        line_labels = [{'label': '%.3g to %.3g' % (z1, z2)} for z1, z2 in z_pairs]
#        [geotecha.plotting.one_d.apply_dict_to_object(line, d)
#            for line, d in zip(fig.gca().get_lines(), line_labels)]
#
#        has_legend = avp_prop.pop('has_legend', True)
#        if has_legend:
#            leg = fig.gca().legend(**legend_prop)
#            leg.draggable(True)


    def _plot_por(self):
        """plot depth vs pore pressure for various times

        """
        t = self.tvals[self.ppress_z_tval_indexes]
        line_labels = [{'label': '%.3g' % v} for v in t]
        por_prop = self.plot_properties.pop('por', dict())
        if not 'xlabel' in por_prop:
            por_prop['xlabel'] = 'Pore pressure'


        fig_por = self._plot_vs_depth(self.por, self.ppress_z,
                                      line_labels=line_labels, H = self.H,
                                      RLzero=self.RLzero,
                                      prop_dict=por_prop)

    def _plot_avp(self):
        """plot average pore pressure vs time for various depth intervals

        """

        t = self.tvals[self.avg_ppress_z_pairs_tval_indexes]
        z_pairs = speccon1d.depth_to_reduced_level(
            np.asarray(self.avg_ppress_z_pairs), self.H, self.RLzero)
        line_labels = [{'label': '%.3g to %.3g' % (z1, z2)} for
            z1, z2 in z_pairs]

        avp_prop = self.plot_properties.pop('avp', dict())
        if not 'ylabel' in avp_prop:
            avp_prop['ylabel'] = 'Average pore pressure'
        fig_avp = self._plot_vs_time(t, self.avp.T,
                           line_labels=line_labels,
                           prop_dict=avp_prop)
    def _plot_set(self):
        """plot settlement vs time fro various depth intervals


        """
        t = self.tvals[self.settlement_z_pairs_tval_indexes]
        z_pairs = speccon1d.depth_to_reduced_level(
            np.asarray(self.settlement_z_pairs), self.H, self.RLzero)
        line_labels = [{'label': '%.3g to %.3g' % (z1, z2)} for
            z1, z2 in z_pairs]

        set_prop = self.plot_properties.pop('set', dict())
        if not 'ylabel' in set_prop:
            set_prop['ylabel'] = 'Settlement'
        fig_set = self._plot_vs_time(t, self.set.T,
                           line_labels=line_labels,
                           prop_dict=set_prop)
        fig_set.gca().invert_yaxis()



    def produce_plots(self):
        """produce plots of analysis"""

        geotecha.plotting.one_d.pleasing_defaults()


        matplotlib.rcParams.update({'font.size': 11})
        matplotlib.rcParams.update({'font.family': 'serif'})


        #por
        if not self.ppress_z is None:
            self._plot_por()


        #avp
        if not self.avg_ppress_z_pairs is None:
            self._plot_avp()

        #settle
        if not self.settlement_z_pairs is None:
            self._plot_set()

        self._plot_loads()

    def _plot_loads(self):
        """plot loads

        """


        load_triples=[]
        load_names = []
        ylabels=[]
        #surcharge
        if not self.surcharge_vs_time is None:
            load_names.append('surch')
            ylabels.append('Surcharge')
            load_triples.append(
                [(vs_time, vs_depth, omega_phase) for
                    vs_time, vs_depth, omega_phase  in
                    zip(self.surcharge_vs_time, self.surcharge_vs_depth,
                    self.surcharge_omega_phase)])


        return self._plot_generic_loads(load_triples, load_names, ylabels=ylabels)


    def _plot_generic_loads(self, load_triples, load_names, ylabels=None, trange = None):
        """plot loads

        Parameters
        ----------
        load_triples : list of list of 3 element tuples
            (load_vs_time, load_vs_depth, load_omega_phase) PolyLines.
            load_triples[i] will be a list of load triples for the ith plot
            load_triples[i][j] will be the jth load triple for the ith plot
        load_names : list of string
            string to prepend to legend entries for each load
        ylabels : list of string, optional
            ylabels for each of the axes, Default = None i.e. y0, y1, y2 etc
        tmax : 2 element tuple
            (tmin, tmax) max and min times to plot loads for


        """


        n = len(load_triples)

        gs = matplotlib.gridspec.GridSpec(n,2, width_ratios=[5,1])
        fig = plt.figure()
        plt.subplot(gs[0])

        #determine tmax etc
        if trange is None:
            for i, (triples, name, ylabel)  in enumerate(zip(load_triples, load_names, ylabels)):
                for j, (vs_time, vs_depth, omega_phase) in enumerate(triples):
                    if not vs_time is None:
                        tmin = np.min(vs_time.x)
                        tmax = np.max(vs_time.x)
        else:
            tmin, tmax = trange

        if ylabels is None:
            ylabels = ['y%d' % v for v in range(n)]
        ax1 = []
        ax2 = []
        for i, (triples, name, ylabel)  in enumerate(zip(load_triples, load_names, ylabels)):
            ax1.append(plt.subplot(gs[i, 0]))
            ax2.append(plt.subplot(gs[i, 1]))

            for j, (vs_time, vs_depth, omega_phase) in enumerate(triples):
                if vs_time is None: #allow for fixed ppress
                    vs_time = PolyLine([tmin, tmax], [0.0, 0.0])

                dx = (tmax-tmin)/20.0
                if not omega_phase is None:
                    omega, phase = omega_phase
                    dx = min(dx, 1/(omega/(2*np.pi))/40)

                print(dx, omega)
#                print(vs_time.x)
                x = [np.linspace(x1, x2, max(int((x2-x1)//dx), 4)) for
                        (x1, x2, y1, y2) in zip(vs_time.x[:-1], vs_time.x[1:], vs_time.y[:-1], vs_time.y[1:])]
                        #if abs(y2-y1) > 1e-5 and abs(x2-x1) > 1e-5]

                y = [np.linspace(y1, y2, max(int((x2-x1)//dx), 4)) for
                        (x1, x2, y1, y2) in zip(vs_time.x[:-1], vs_time.x[1:], vs_time.y[:-1], vs_time.y[1:])]
                        #if abs(y2-y1) > 1e-5 and abs(x2-x1) > 1e-5]

                x = np.array([val for subl in x for val in subl])
                y = np.array([val for subl in y for val in subl])

                if not omega_phase is None:
                    y *= np.cos(omega * x + phase)

                mark_every = None
#                print(x, y)
                ax1[-1].plot(x, y, '-o',label=name)
                ax1[-1].set_xlabel('Time')
                ax1[-1].set_ylabel(ylabel)

        return fig

    def _plot_vs_time(self, t, y, line_labels, prop_dict={}):
        """plot y vs t for variuos values


        """

        fig_prop = prop_dict.pop('fig_prop', {'figsize':(18/2.54, 18/1.61/2.54)})
        legend_prop = prop_dict.pop('legend_prop',
                                   {'title': 'Depth interval:', 'fontsize': 9})

        styles = prop_dict.pop('style', None)
        if styles is None:
            mcd = geotecha.plotting.one_d.MarkersDashesColors(
#                color = 'black',
                default_marker={'markersize': 7})
            mcd.construct_styles(markers = range(32), dashes=[0],
                                 marker_colors=None, line_colors=None)


        styles = itertools.cycle(mcd.styles)



#        z = speccon1d.depth_to_reduced_level(z, H, RLzero)

#        t = self.tvals[self.ppress_z_tval_indexes]


        fig = plt.figure(**fig_prop)
        plt.plot(t, y)

        xlabel = prop_dict.pop('xlabel', 'Time, t')
        plt.xlabel(xlabel)
        ylabel = prop_dict.pop('ylabel', 'y')
        plt.ylabel(ylabel)

        #apply style to each line
        [geotecha.plotting.one_d.apply_dict_to_object(line, d)
            for line, d in zip(fig.gca().get_lines(), styles)]
        #apply markevery to each line
        random.seed(1)
        [geotecha.plotting.one_d.apply_dict_to_object(line, d)
            for line, d in
                zip(fig.gca().get_lines(),
                    [{'markevery': (random.random()* 0.1, 0.1)} for v in y])]

        [geotecha.plotting.one_d.apply_dict_to_object(line, d)
            for line, d in zip(fig.gca().get_lines(), line_labels)]

        has_legend = prop_dict.pop('has_legend', True)

        if has_legend:
            leg = fig.gca().legend(**legend_prop)
            leg.draggable(True)

        return fig



    def _plot_vs_depth(self, x, z, line_labels=None, H = 1.0, RLzero=None, prop_dict={}):
        """plot z vs x for various t values

        Parameters
        ----------



        """

        fig_prop = prop_dict.pop('fig_prop', {'figsize':(18/2.54, 18/1.61/2.54)})
        legend_prop = prop_dict.pop('legend_prop',
                                   {'title': 'time:', 'fontsize': 9})

        styles = prop_dict.pop('style', None)
        if styles is None:
            mcd = geotecha.plotting.one_d.MarkersDashesColors(
#                color = 'black',
                default_marker={'markersize': 7})
            mcd.construct_styles(markers = range(32), dashes=[0],
                                 marker_colors=None, line_colors=None)


        styles = itertools.cycle(mcd.styles)



        z = speccon1d.depth_to_reduced_level(z, H, RLzero)

#        t = self.tvals[self.ppress_z_tval_indexes]


        fig = plt.figure(**fig_prop)
        plt.plot(x, z)

        xlabel = prop_dict.pop('xlabel', 'x')
        plt.xlabel(xlabel)

        if RLzero is None:
            plt.gca().invert_yaxis()
            ylabel = prop_dict.pop('ylabel', 'Depth, z')
        else:
            ylabel = prop_dict.pop('ylabel', 'RL')

        plt.ylabel(ylabel)

        #apply style to each line
        [geotecha.plotting.one_d.apply_dict_to_object(line, d)
            for line, d in zip(fig.gca().get_lines(), styles)]
        #apply markevery to each line
        random.seed(1)
        [geotecha.plotting.one_d.apply_dict_to_object(line, d)
            for line, d in
                zip(fig.gca().get_lines(),
                    [{'markevery': (random.random()* 0.1, 0.1)} for v in x])]
        #apply label to each line
#        line_labels = [{'label': '%.3g' % v} for v in t]
        [geotecha.plotting.one_d.apply_dict_to_object(line, d)
            for line, d in zip(fig.gca().get_lines(), line_labels)]

        has_legend = prop_dict.pop('has_legend', True)

        if has_legend:
            leg = fig.gca().legend(**legend_prop)
            leg.draggable(True)

        return fig




def program(reader, writer):
    """run speccon1d_vert_radial_boundary program

    Parameters
    ----------
    reader : file-like object
        Object to read input from.  Can be string, file, StringIO
    writer : file-like object
        Object to write output to can be file or StringIO
    """


    inp = make_module_from_text(reader)


    if inp.__dict__.get('echo', True):
        #maybe process here
        writer.write(reader)

    writer.write(str(inp.a)+'\n')
    writer.write(str(inp.b)+'\n')
    writer.write(str(inp.c)+'\n')

    return



















if __name__ == '__main__':



    my_code = textwrap.dedent("""\
from geotecha.piecewise.piecewise_linear_1d import PolyLine
import numpy as np
H = 1
drn = 0
dT = 1
#dTh = 5
dTv = 0.1 * 0.25
neig = 10


mvref = 2.0
kvref = 1.0
khref = 1.0
etref = 1.0

mv = PolyLine([0,1], [0.5,0.5])
#kh = PolyLine([0,1], [1,1])
kv = PolyLine([0,1], [5,5])
#et = PolyLine([0,0.48,0.48, 0.52, 0.52,1], [0, 0,1,1,0,0])
#et = PolyLine([0,1], [1,1])
surcharge_vs_depth = PolyLine([0,1], [1,1])
surcharge_vs_time = PolyLine([0,1,10], [0,100,100])
surcharge_omega_phase = (2*np.pi*0.5, -np.pi/2)


#vacuum_vs_depth = PolyLine([0,1], [1,1])
#vacuum_vs_time = PolyLine([0,0,20], [0,-0.2,-0.2])
#vacuum_omega_phase = (2*np.pi*50, -np.pi/2)

#top_vs_time = PolyLine([0,0.0,10], [0,-100,-100])
#top_omega_phase = (2*np.pi*1, -np.pi/2)
#bot_vs_time = PolyLine([0,0.0,3], [0,-0.2,-0.2])
#bot_vs_time = PolyLine([0,0.0,10], [0, -0.2, -0.2])

#bot_vs_time = PolyLine([0,0.0,0.4,10], [0, -2500, -250, -210])
#bot_omega_phase = (2*np.pi*2, -np.pi/2)


#fixed_ppress = (0.2, 1000, PolyLine([0, 0.0, 8], [0,-0.3,-0.3]))
#fixed_ppress_omega_phase = (2*np.pi*2, -np.pi/2)


ppress_z = np.linspace(0,1,100)
avg_ppress_z_pairs = [[0,1],[0.4, 0.5]]
settlement_z_pairs = [[0,1],[0.4, 0.5]]
#tvals = np.linspace(0,3,10)
tvals = [0,0.05,0.1]+list(np.linspace(0.2,5,100))
tvals = np.linspace(0, 5, 100)
tvals = np.logspace(-2, 0.3,50)
ppress_z_tval_indexes = np.arange(len(tvals))[::len(tvals)//7]
#avg_ppress_z_pairs_tval_indexes = slice(None,None)#[0,4,6]
#settlement_z_pairs_tval_indexes = slice(None, None)#[0,4,6]

implementation='scalar'
implementation='vectorized'
#implementation='fortran'
RLzero = -12.0
plot_properties={}
    """)


#    writer = StringIO()
#    program(my_code, writer)
#    print('writer\n%s' % writer.getvalue())
    #a = calculate_normalised(my_code)
    a = speccon1d_vr(my_code)
    a.make_all()
    slope = (a.por[-1,:]-a.por[-2,:]) / (a.ppress_z[-1]-a.ppress_z[-2])
#    print(repr(a.tvals))
#    print(repr(slope))
#    a._make_gam()
#    #print(a.gam)
#    a._make_psi()
#    #print(a.psi)
#    a._make_eigs_and_v()
#    #print(a.eigs)
#    #print(a.v)
#    a._make_E_Igamv_the()
#    #print(a.E_Igamv_the)
#    a.make_por()
#    #print(a.por)
#    print(len(a.tvals))
    #print(a.por.shape)

    a.produce_plots()
    plt.show()

    if False:
        plt.figure()
        plt.plot(a.por, a.ppress_z)
        plt.xlabel('Pore pressure')
        plt.ylabel('Normalised depth, Z')
        plt.gca().invert_yaxis()
#        plt.show()
        plt.figure()
        plt.plot(a.tvals[a.avg_ppress_z_pairs_tval_indexes], a.avp.T, '-o')
        plt.xlabel('Time')
        plt.ylabel('Average pore pressure')
        #plt.gca().invert_yaxis()
#        plt.show()
##        print(a.set)
        plt.figure()
        plt.plot(a.tvals[a.settlement_z_pairs_tval_indexes], a.set.T)
        plt.xlabel('Time')
        plt.ylabel('settlement')
        plt.gca().invert_yaxis()
        plt.show()
#    for i, p in enumerate(a.por.T):
#
#        print(a.tvals[i])
#        plt.plot(p, a.ppress_z)
#
#    plt.show()




#def main(argv = None):
#
#    parser = argparse.ArgumentParser()
#    parser.add_argument('-f', '--files', metavar='files', type=str,
#                        nargs='+',
#                        help="files to process")
#
#
#
#    if isinstance(files, str):
#        files = [files] # convert single filename to list


