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
Nogami and Li (2003) 'Consolidation of Clay with a System of Vertical and
Horizontal Drains'.

"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import geotecha.inputoutput.inputoutput as inputoutput
import math
import textwrap
import scipy.optimize
import geotecha.piecewise.piecewise_linear_1d as pwise
import cmath
import time
from geotecha.mathematics.root_finding import find_n_roots
import geotecha.plotting.one_d
import scipy.special
#from scipy.special import j0, y0, j1, y1

besselj = scipy.special.jn
bessely = scipy.special.yn

from geotecha.inputoutput.inputoutput import GenericInputFileArgParser


class NogamiAndLi2003(inputoutput.InputFileLoaderCheckerSaver):
    """Multi-layer vertical and radial consolidation using matrix transfer

    Partially implements the article by Nogami and Li (2003) [1]_. While the
    article includes special treatment for sand layers and geotextile layers,
    this implementation only considers 'soil' layers.  (Sand layers are
    just normal layers  with high kv and low mv).

    The coding is not of high quality.  The main use is for verification of
    speccon models noting that Nogami and Li (2003) use rigourous methods
    where as speccon uses equal-strain assumptions for the radial flow part.

    Features:

     - Multiple layers.
     - Vertical flow and radial flow to a central drain (no smear zone).
     - Load is uniform with depth but varies piecewise-linear with time.
     - No assumptions on radial distribution of strain (i.e. NOT equal-strain).
     - pore pressure vs depth at various times.  Either at a particular radial
       coordinate or averaged in the radial direction.
     - Average pore pressure vs time.  Average is over the entire soil layer.


    .. warning::
        The 'Parameters' and 'Attributes' sections below require further
        explanation.  The parameters listed below are not used to explicitly
        initialize the object.  Rather they are defined in either a
        multi-line string or a file-like object using python syntax.
        It is the file object or string object that is used to initialize
        the object.  Each  'parameter' will be turned into an attribute that
        can be accessed using conventional python dot notation, after the
        object has been initialised.  The attributes listed below are
        calculated values (i.e. they could be interpreted as results) which
        are accessible using dot notation after all calculations are
        complete.


    Parameters
    ----------
    z : list/array of float
        Depth to calc pore pressure at.
    t : list/array of float
        Time values to calc average pore pressure at.
    tpor : list/array of float
        Time values to calc pore pressure profiles at.
    h : list/array of float
        Layer thicknesses.
    nv, nh : tuple of 2 int, optional
        Number of series terms to use in vertical and horizontal direction.
        Default nv=nh=5.
    kv, kh : list/array of float
        Layer vertical and horizontal permeability divided by unit weight of
        water.
    mv : list/array of float
        Layer volume compressibility.
    bctop, bcbot : [0, 1]
        Boundary condition. bctop=0 is free draining, bctop=1 is
        impervious.
    surcharge_vs_time : PolyLine
        Piecewise linear variation of surcharge with time
    r1, r0 : float optional
        drain influence zone and drain radius. If either is none then only
        vertical drainage will be considered.
    rcalc : float, optional
        Radial coordinate at which to calc pore pressure.  Default rcalc=None
        i.e. pore pressure is averaged in the radial direction.
    radial_roots_x0 : float, optional
        Starting point for finding radial eigenvalues.
        Default radial_roots_x0=1e-3.
    radial_roots_dx : float, optional
        Starting increment for finding radial eigenvalues.
        Default radial_roots_dx=1e-3.
    radial_roots_p : float, optional
        Succesive increment length increase factor for finding radial
        eigenvalues. default radial_roots_p=1.05.
    vertical_roots_x0 : float, optional
        Starting point for finding vertical eigenvalues.
        Default vertical_roots_x0=1e-7.
    vertical_roots_dx : float, optional
        Starting increment for finding vertical eigenvalues.
        Default vertical_roots_dx=1e-7.
    vertical_roots_p : float, optional
        Succesive increment lenght increase factor for finding vertical
        eigenvalues.  Default vertical_roots_p=1.05.
    max_iter : int, optional
        Max iterations when searching for eigenvalue intervals.
        Default max_iter=10000
    show_vert_eigs : True/False, optional
        If true a vertical eigen value plot will be made.
        Default show_vert_eigs=False
    plot_properties : dict of dict, optional
        dictionary that overrides some of the plot properties.
        Each member of `plot_properties` will correspond to one of the plots.

        ==================  ============================================
        plot_properties     description
        ==================  ============================================
        por                 dict of prop to pass to pore pressure plot.
        avp                 dict of prop to pass to average pore
                            pressure plot.
        set                 dict of prop to pass to settlement plot.
        ==================  ============================================
        see geotecha.plotting.one_d.plot_vs_depth and
        geotecha.plotting.one_d.plot_vs_time for options to specify in
        each plot dict.
    save_data_to_file : True/False, optional
        If True data will be saved to file.  Default save_data_to_file=False
    save_figures_to_file : True/False
        If True then figures will be saved to file.
        Default save_figures_to_file=False
    show_figures : True/False, optional
        If True the after calculation figures will be shown on screen.
        Default show_figures=False.
    directory : string, optional
        Path to directory where files should be stored.
        Default directory=None which
        will use the current working directory.  Note if you keep getting
        directory does not exist errors then try putting an r before the
        string definition. i.e. directory = r'C:\\Users\\...'
    overwrite : True/False, optional
        If True then existing files will be overwritten.
        Default overwrite=False.
    prefix : string, optional
         Filename prefix for all output files.  Default prefix= 'out'
    create_directory : True/Fase, optional
        If True a new sub-folder with name based on  `prefix` and an
        incremented number will contain the output
        files. Default create_directory=True.
    data_ext : string, optional
        File extension for data files. Default data_ext='.csv'
    input_ext : string, optional
        File extension for original and parsed input files. default = ".py"
    figure_ext : string, optional
        File extension for figures.  Can be any valid matplotlib option for
        savefig. Default figure_ext=".eps". Others include 'pdf', 'png'.
    title : str, optional
        A title for the input file.  This will appear at the top of data files.
        Default title=None, i.e. no title.
    author : str, optional
        Author of analysis. Default='unknown'.


    Attributes
    ----------
    por : array of shape (len(z), len(tpor))
        Pore pressure vs depth at various times.  Only present if tpor defined.
        If rcalc defined then porepressure will be at r=rcalc.  If rcalc is
        not defined then pore pressure is averaged radially
    avp : array of shape (1, len(t))
        Averge pore pressure of profile various times.  Only present if t
        defined. If rcalc defined then pore pressure will be at r=rcalc.
        If rcalc is not defined then pore pressure is averaged radially
    set : array of shape (1, len(t))
        Surface settlement at various times.  Only present if t
        defined. If rcalc defined then settlement will be at r=rcalc.
        If `rcalc` is not defined then settlement is averaged radially.


    Notes
    -----
    It is possbile to initialize the object without a file-like object or
    multi-line string, i.e. using the default reader=None.  This is not
    recommended because you have to explicitly set each attribute.  It will
    most likely be easier to use a string or file object and then do any
    custom modifications to the attributes afterwards.


    This program relies on numerical root finding, which can be extremely
    troublesome in for the vertical eigenvalue case here (Mainly because I
    never figured out how to normalise in the z direction... but that is
    another story). You will probably need to fine tune the vertical_roots
    parameters to ensure the correct eigenvalues have been found, so:

     1. Run the program with the defaults.  If it actually excecutes go to
        step 3.
     2. Increase a combo of `vertical_roots_dx`, `vertical_roots_p` and
        `max_iter` untill the program excecutes.
     3. Does your pore pressure vs depth plots look ok.  If yes, then possibly
        accept the results.  But better to check eigen values in step 4.
     4. Run the method `_plot_vert_roots` with enough points to smoothly
        show the characteristic curve.  zoom in on the roots and check if
        all the roots are found (usually the problems occur with leftmost line.
        If not alter `vertical_roots_dx`, `vertical_roots_p` and
        `max_iter` untill all roots are captured.  Basically if you choose
        `vertical_roots_dx` tiny, `vertical_roots_p`=1, and `max_iter` huge
        then you will find all the roots but it may take a long time.

    Root finding is very hard when there are clumps of closely spaced roots
    but the clumps themselves are far apart.

    Also note that there are errors in eq.24a and eq.24b in the published
    article of Nogami and Li.  Also I could never get the vertical
    normalisation to work.  Also I've done my own normalising for the radial
    part.

    See Also
    --------
    geotecha.piecewise.piecewise_linear_1d.PolyLine : How to specify loadings


    References
    ----------
    .. [1] Nogami, Toyoaki, and Maoxin Li. 'Consolidation of Clay with a
           System of Vertical and Horizontal Drains'. Journal of
           Geotechnical and Geoenvironmental Engineering 129, no. 9
           (2003): 838-48. doi:10.1061/(ASCE)1090-0241(2003)129:9(838).

    """

    def _setup(self):
        """This method overwrites the _setup method in
        inputoutput.InputFileLoaderCheckerSaver

        """

        self._attribute_defaults = {'bctop': 0, 'bcbot': 0,
                                    'radial_roots_x0': 1e-3,
                                    'radial_roots_dx': 1e-3,
                                    'radial_roots_p': 1.05,
                                    'vertical_roots_x0': 1e-7,
                                    'vertical_roots_dx': 1e-7,
                                    'vertical_roots_p': 1.05,
                                    'max_iter': 10000,
                                    'prefix': 'nl2003_',
                                    'show_vert_eigs': False}

        self._attributes = ('z t tpor nv nh h kv kh mv bctop bcbot '
            'surcharge_vs_time r0 r1 rcalc '
            'radial_roots_x0 radial_roots_dx radial_roots_p '
            'vertical_roots_x0 vertical_roots_dx vertical_roots_p '
            'max_iter show_vert_eigs' ).split()
        self._attributes_that_should_have_same_len_pairs = [
            'h kv'.split(),
            'kv mv'.split(),
            'h mv'.split(),
            'h kh'.split(),
            'h kv'.split(),
            'h mv'.split()] #pairs that should have the same length

        self._attributes_that_should_be_lists= []
        self._attributes_that_should_have_same_x_limits = []


        self.z = None
        self.t = None
        self.tpor = None
        self.nv = self._attribute_defaults.get('nv', None)
        self.nh = self._attribute_defaults.get('nh', None)
        self.h = None
        self.kv = None
        self.kh = None
        self.mv = None
        self.bctop = self._attribute_defaults.get('bctop', None)
        self.bcbot = self._attribute_defaults.get('bcbot', None)
        self.r0 = None
        self.r1 = None
        self.rcalc = None

        self.radial_roots_x0 = self._attribute_defaults.get('radial_roots_x0', None)
        self.radial_roots_dx = self._attribute_defaults.get('radial_roots_dx', None)
        self.radial_roots_p = self._attribute_defaults.get('radial_roots_p', None)
        self.vertical_roots_x0 = self._attribute_defaults.get('vertical_roots_x0', None)
        self.vertical_roots_dx = self._attribute_defaults.get('vertical_roots_dx', None)
        self.vertical_roots_p = self._attribute_defaults.get('vertical_roots_p', None)
        self.max_iter = self._attribute_defaults.get('max_iter', None)
        self.show_vert_eigs=self._attribute_defaults.get('show_vert_eigs', None)
        self.surcharge_vs_time = None

        self._zero_or_all = [
            'h kv mv'.split(),
            'r0 r1'.split()]
        self._at_least_one = [
            ['mv'],
            ['surcharge_vs_time'],
            'kv kh'.split(),
            'tpor t'.split(),]
        self._one_implies_others = ['r0 r1 kh nh'.split(),
                                    'r1 r0 kh nh'.split(),
                                    'kh r0 r1 nh'.split(),
                                    'nh kh r0 r1'.split()]

    def _calc_derived_properties(self):
        """Calculate properties/ratios derived from input"""

        self.check_input_attributes()

#        if self.rcalc is None:
#            self.rcalc=self.r1


        if not self.t is None:
            self.t = np.asarray(self.t)

        self.z = np.asarray(self.z)
        self.kv = np.asarray(self.kv)
        self.mv = np.asarray(self.mv)

        if not self.kh is None:
            self.kh = np.asarray(self.kh)
            self.ch = self.kh / self.mv
        else:
            self.nh = 1
            self.ch = np.zeros_like(self.mv)

        self.h = np.asarray(self.h)

        self.nlayers = len(self.kv)

        self.zlayer = np.cumsum(self.h)

        self.cv = self.kv / self.mv

#        self.use_normalised = True
        #use_normalised is only relevant fot the radial component.
        #I couldn't get non-normalised to work, hence why it is hard coded in
        #rather than a user variable.  I could never get the vertical
        #normalisation working so I gave up trying

        if self.bctop == 0:
            self.phia0 = np.array([0.0, 1.0])
        elif self.bctop == 1:
            self.phia0 = np.array([1.0, 0.0])

        #index of phib at bottom to check
        if self.bcbot == 0:
            self.phi_i_check = 0
        elif self.bcbot == 1:
            self.phi_i_check = 1

    def un_normalised_average(self,s):
        """u(r) part of u(r, z, t) = u(r) * phi(z) * T(t), averaged betw r0 r1
        """

        r0 = self.r0
        r1 = self.r1
        nn = r1/r0

        return -self.un_normalised(r0,s,1)*2 / (nn**2-1)/s

    def un_normalised(self, r, s, order):
        """u(r) part of u(r, z, t) = u(r) * phi(z) * T(t)

        This version is normalised w.r.t. r0

        """

        r0 = self.r0
        r1 = self.r1
        return besselj(order, r/r0*s)*bessely(0, s) - besselj(0, s)*bessely(order, r/r0*s)



    def _radial_characteristic_curve(self, s):
        """Zeros of this function provide the radial eigenvalues"""

        r0 = self.r0
        r1 = self.r1

        return self.un_normalised(r1, s, 1)


    def _find_sn(self):
        """Find the radial eigenvalues"""
        if self.kh is None:
            self._sn=np.array([0])
        else:
            self._sn = find_n_roots(self._radial_characteristic_curve,
                               n=self.nh,x0=self.radial_roots_x0,
                               dx=self.radial_roots_dx,
                               p = self.radial_roots_p, max_iter=self.max_iter)


    def _beta(self, alp, s):
        """beta from alp**2 = cv*beta**2 + cr * s**2 / r0**2"""


        if self.r0 is None:
            r0=1
        else:
            r0 = self.r0
        a = 1/self.cv * alp**2 -(self.ch/self.cv)*s**2/r0**2
        return np.sqrt(np.array(a, dtype=complex))



    def _calc_phia_and_phidota(self):
        """Determine the pore pressure and pore pressure gradient at top of
        each layer

        Can only be done after finding alp
        """


        sin = cmath.sin
        cos = cmath.cos

        self._phia = np.zeros((self.nh, self.nv, self.nlayers), dtype=float)
        self._phidota = np.zeros((self.nh, self.nv, self.nlayers), dtype=float)

        self._phia[:,:,0] = self.phia0[0]
        self._phidota[:,:,0] = self.phia0[1]
#        print('o'*40)
        for i in range(self.nh):
            s = self._sn[i]

            square = np.zeros((2,2), dtype=complex)
            for j in range(self.nv):
                alp = self._alp[i, j]
                phia = np.array([self.phia0[0], self.phia0[1]], dtype=complex)
#                print(i, 's=', s)
#                print(j, 'alp=', alp)
                for k in range(self.nlayers):
                    h = self.h[k]
                    beta = self._betamn[i, j, k]
                    if cmath.polar(beta)[0]==0:
                        phib = np.array([phia[0],0], dtype=complex)
#                        phib[0] = phia[0]
#                        phib[1] = 0+0j
                    else:
                        square[0,0] = cos(beta*h)
                        square[0,1] = sin(beta*h) / beta
                        square[1,0] = -beta*sin(beta*h)
                        square[1,1] = cos(beta*h)

                        phib = np.dot(square, phia)
#                    print(k, beta, phia, phib)
                    if k != self.nlayers-1: # we are not in the last layer
                        # transfer phib to next layers phia
                        phia[0] = phib[0]
                        phia[1] = phib[1] * self.kv[k] /  self.kv[k+1]
                        self._phia[i,j,k + 1] = phia[0].real
                        self._phidota[i, j, k+1] = phia[1].real#phib[1] * self.kv[i] /  self.kv[i+1]

                #check
#                print('_', alp, 's', phib.real)
                if abs(phib[self.phi_i_check].real)>0.1:
                    pass
                    print('bottom BC not satisfied. ih=',i,'jv=', j )
#                    raise ValueError('Bottom BC not satisfied')

    def _vertical_characteristic_curve(self, alp, s):
        """the roots of this function will give the vertical eigenvalues

        Parameters
        ----------
        alp : float
            alp common for all layers
        s : float
            radial eigen value

        """

        sin = cmath.sin
        cos = cmath.cos

        phia = np.array([self.phia0[0], self.phia0[1]], dtype=complex)
        square = np.zeros((2,2), dtype=complex)

        beta = self._beta(alp, s)
#        print("*", 's=', s)
#        print('alp=', alp)
        for i, h in enumerate(self.h):
            if cmath.polar(beta[i])[0]==0:
                phib = np.array([phia[0], 0], dtype=complex)
#                phib[0] = phia[0]
#                phib[1] = 0+0j
            else:
                square[0,0] = cos(beta[i]*h)
                square[0,1] = sin(beta[i]*h) / beta[i]
                square[1,0] = -beta[i]* sin(beta[i]*h)
                square[1,1] = cos(beta[i]*h)

                phib = np.dot(square, phia)
#            print(i,  beta[i], phia, phib)
            if i != self.nlayers - 1: # we are not in the last layer
                #transfer phib to the next layer phia
                phia[0]= phib[0]
                phia[1] = phib[1] * self.kv[i] /  self.kv[i+1]
        ret = phib[self.phi_i_check].real

        return ret


    def _find_alp(self):
        """find alp by matrix transfer method"""

        self._alp = np.zeros((self.nh, self.nv), dtype=float)

        for n, s in enumerate(self._sn):
            if s==0:
                alp_start_offset = min(1e-7, self.vertical_roots_dx)
            else:
                alp_start_offset = 0
            if n==0:
                alp=self.vertical_roots_x0
            else:
                alp = self._alp[n-1,0]
            self._alp[n,:] = find_n_roots(self._vertical_characteristic_curve,
                args=(s,),n= self.nv, x0 = alp+alp_start_offset,
                dx = self.vertical_roots_dx, p = self.vertical_roots_p,
                max_iter=self.max_iter, fsolve_kwargs={})

    def _calc_Cn(self):
        """Calc Cn part of the coefficient Cmn"""

        self._Cn = np.zeros(self.nh, dtype=float)

        if self.kh is None:
            self._Cn[:]=1
            return

        r0 = self.r0
        r1 = self.r1

        for n, s in enumerate(self._sn):
            numer = -r0**2/s * self.un_normalised(r0, s, 1)
            denom = r0**2/2 * (r1**2/r0**2 * self.un_normalised(r1, s, 0)**2 -
                        self.un_normalised(r0, s, 1)**2)
            self._Cn[n] = numer / denom

    def _calc_betamn(self):
        """calc beta for each layer and each eigenvalue combination"""

        self._betamn = np.zeros((self.nh, self.nv, self.nlayers), dtype=complex)

        for i in range(self.nh):
            s = self._sn[i]
            for j in range(self.nv):
                alp = self._alp[i, j]
                self._betamn[i,j,:] = self._beta(alp, s)



    def _calc_Amn_and_Bmn(self):
        """calc coefficeints Amn and Bmn for each layer and eigenvalue
        combination"""


        sin = cmath.sin
        cos = cmath.cos

        self._Amn = np.zeros((self.nh, self.nv, self.nlayers), dtype=complex)
        self._Bmn = np.zeros((self.nh, self.nv, self.nlayers), dtype=complex)

        for i in range(self.nh):
            s = self._sn[i]
            square = np.zeros((2,2), dtype=float)
            for j in range(self.nv):
                alp = self._alp[i, j]
                phia = self.phia0
                for k in range(self.nlayers):
                    h = self.h[k]
                    bet = self._betamn[i, j, k]
                    phi_a = self._phia[i, j, k]
                    phi_a_dot = self._phidota[i, j, k]

                    if cmath.polar(bet)[0]==0:
                        self._Amn[i,j,k] = h*phi_a
                        self._Bmn[i,j,k] = h*phi_a**2
                    else:
                        self._Amn[i,j,k] = (phi_a*sin(bet*h)/bet -
                            phi_a_dot*cos(bet*h)/bet**2 + phi_a_dot/bet**2)
                        self._Bmn[i,j,k] = (h*phi_a**2*sin(bet*h)**2/2 +
                            h*phi_a**2*cos(bet*h)**2/2 +
                            phi_a**2*sin(bet*h)*cos(bet*h)/(2*bet) +
                            h*phi_a_dot**2*sin(bet*h)**2/(2*bet**2) +
                            h*phi_a_dot**2*cos(bet*h)**2/(2*bet**2) -
                            phi_a*phi_a_dot*cos(bet*h)**2/bet**2 +
                            phi_a*phi_a_dot/bet**2 -
                            phi_a_dot**2*sin(bet*h)*cos(bet*h)/(2*bet**3))


    def _calc_Cm(self):
        """Calc Cm part of the coefficient Cmn"""
        self._calc_Amn_and_Bmn()
        self._Cm = np.zeros((self.nh, self.nv), dtype=complex)


        for i in range(self.nh):
            s = self._sn[i]
            for j in range(self.nv):
                alp = self._alp[i, j]
#                phia = self.phia0
                numer = 0.0
                denom = 0.0
                for k in range(self.nlayers):
                    h = self.h[k]
                    mv = self.mv[k]
                    Amn = self._Amn[i,j,k]
                    Bmn = self._Bmn[i,j,k]
                    numer += mv*Amn
                    denom += mv*Bmn
                self._Cm[i,j] = numer / denom





    def _calc_Cmn(self):
        """calc the coefficient Cmn = Cm * Cn"""

        self._calc_Cn()
        self._calc_Cm()

        self._Cmn = np.zeros((self.nh, self.nv), dtype=complex)
        for i in range(self.nh):
            Cn = self._Cn[i]
            for j in range(self.nv):
                Cm = self._Cm[i,j]
                self._Cmn[i,j] = Cm * Cn




    def calc(self):
        """Perform all calculations"""

        self._calc_derived_properties()
        self._find_sn()

        self._find_alp()

        self._calc_betamn()
        self._calc_phia_and_phidota()
        self._calc_Cmn()
        self._calc_por()

        return


    def make_all(self):
        """Check input, make_output produce files and plots"""


        self.check_input_attributes()
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

    def make_output(self):
        """make all output"""



        self._calc_derived_properties()
        self._find_sn()

        self._find_alp()

        self._calc_betamn()
        self._calc_phia_and_phidota()
        self._calc_Cmn()
#        self._calc_por()


        header1 = "program: nogamiandli2003; geotecha version: {}; author: {}; date: {}\n".format(self.version, self.author, time.strftime('%Y/%m/%d %H:%M:%S'))
        if not self.title is None:
            header1 += "{}\n".format(self.title)

        if not self.rcalc is None:
            extra = " at r={0:.3g}".format(self.rcalc)
        else:
            extra=""


        self._grid_data_dicts = []
        if not self.tpor is None:
            self._calc_por()

            labels = ['{:.3g}'.format(v) for v in self.z]
            d = {'name': '_data_por',
                 'data': self.por.T,
                 'row_labels': self.tpor,
                 'row_labels_label': 'Time',
                 'column_labels': labels,
                 'header': header1 + 'Pore pressure at depth'+extra}
            self._grid_data_dicts.append(d)

        if not self.t is None:
            self._calc_avp()

            labels = ['{:.3g} to {:.3g}'.format(0, sum(self.h))]
            d = {'name': '_data_avp',
                 'data': self.avp.T,
                 'row_labels': self.t,
                 'row_labels_label': 'Time',
                 'column_labels': labels,
                 'header': header1 + 'Average pore pressure between depths' + extra}
            self._grid_data_dicts.append(d)

            labels = ['{:.3g} to {:.3g}'.format(0, sum(self.h))]
            d = {'name': '_data_set',
                 'data': self.avp.T,
                 'row_labels': self.t,
                 'row_labels_label': 'Time',
                 'column_labels': labels,
                 'header': header1 + 'settlement between depths' + extra}
            self._grid_data_dicts.append(d)

        return

    def produce_plots(self):
        """produce plots of analysis"""

#        geotecha.plotting.one_d.pleasing_defaults()

#        matplotlib.rcParams['figure.dpi'] = 80
#        matplotlib.rcParams['savefig.dpi'] = 80

        matplotlib.rcParams.update({'font.size': 11})
        matplotlib.rcParams.update({'font.family': 'serif'})

        self._figures=[]
        #por
        if not self.tpor is None:
            f=self._plot_por()
            title = 'fig_por'
            f.set_label(title)
            f.canvas.manager.set_window_title(title)
            self._figures.append(f)

        if not self.t is None:
            f=self._plot_avp()
            title = 'fig_avp'
            f.set_label(title)
            f.canvas.manager.set_window_title(title)
            self._figures.append(f)

            f=self._plot_set()
            title = 'fig_set'
            f.set_label(title)
            f.canvas.manager.set_window_title(title)
            self._figures.append(f)


        if self.show_vert_eigs:
            f = self._plot_vert_roots(1000)
            title = 'vertical characteristic curve and eigs'
            f.set_label(title)
            f.canvas.manager.set_window_title(title)
            self._figures.append(f)

    def _plot_por(self):
        """plot depth vs pore pressure for various times

        """

        if not self.rcalc is None:
            extra = " at r={0:.3g}".format(self.rcalc)
        else:
            extra=" (radial average)"

        t = self.tpor
        line_labels = ['{:.3g}'.format(v) for v in t]
        por_prop = self.plot_properties.pop('por', dict())
        if not 'xlabel' in por_prop:
            por_prop['xlabel'] = 'Pore pressure'+extra

        #to do
        fig_por = geotecha.plotting.one_d.plot_vs_depth(self.por, self.z,
                                      line_labels=line_labels,
                                      prop_dict=por_prop)
        return fig_por

    def _plot_avp(self):
        """plot average pore pressure of profile"""

        if not self.rcalc is None:
            extra = " at r={0:.3g}".format(self.rcalc)
        else:
            extra=" (radial average)"

        t = self.t
        line_labels = ['{:.3g} to {:.3g}'.format(0, sum(self.h))]

        avp_prop = self.plot_properties.pop('avp', dict())
        if not 'ylabel' in avp_prop:
            avp_prop['ylabel'] = 'Average pore pressure'+extra
        fig_avp = geotecha.plotting.one_d.plot_vs_time(t, self.avp.T,
                           line_labels=line_labels,
                           prop_dict=avp_prop)
        return fig_avp

    def _plot_set(self):
        """plot surface settlement"""

        if not self.rcalc is None:
            extra = " at r={0:.3g}".format(self.rcalc)
        else:
            extra=" (radial average)"

        t = self.t
        line_labels = ['{:.3g} to {:.3g}'.format(0, sum(self.h))]

        set_prop = self.plot_properties.pop('set', dict())
        if not 'ylabel' in set_prop:
            set_prop['ylabel'] = 'surface settlement'+extra
        fig_set = geotecha.plotting.one_d.plot_vs_time(t, self.set.T,
                           line_labels=line_labels,
                           prop_dict=set_prop)
        fig_set.gca().invert_yaxis()
        return fig_set


    def _calc_Tm(self, alp, t):
        """calculate the Tm expression at a given time

        Parameters
        ----------
        alp : float
            eigenvalue for layer
        t : float
            time value

        Returns
        -------
        Tm: float
            time dependant function

        """


        loadmag = self.surcharge_vs_time.y
        loadtim = self.surcharge_vs_time.x
        (ramps_less_than_t, constants_less_than_t, steps_less_than_t,
            ramps_containing_t, constants_containing_t) = pwise.segment_containing_also_segments_less_than_xi(loadtim, loadmag, t, steps_or_equal_to = True)

        exp = math.exp
        Tm=0
        cv = 1 # I copied the Tm function from SchiffmanAndStein1970
        i=0 #only one time value
        for k in steps_less_than_t[i]:
            sig1 = loadmag[k]
            sig2 = loadmag[k+1]

            Tm += (sig2-sig1)*exp(-cv * alp**2 * (t-loadtim[k]))
        for k in ramps_containing_t[i]:
            sig1 = loadmag[k]
            sig2 = loadmag[k+1]
            t1 = loadtim[k]
            t2 = loadtim[k+1]

#            Tm += (-sig1 + sig2)/(alp**2*cv*(-t1 + t2)) - (-sig1 + sig2)*exp(-alp**2*cv*t)*exp(alp**2*cv*t1)/(alp**2*cv*(-t1 + t2))
            Tm += (-sig1 + sig2)/(alp**2*cv*(-t1 + t2)) - (-sig1 + sig2)*exp(-alp**2*cv*(t-t1))/(alp**2*cv*(-t1 + t2))

        for k in ramps_less_than_t[i]:
            sig1 = loadmag[k]
            sig2 = loadmag[k+1]
            t1 = loadtim[k]
            t2 = loadtim[k+1]
#            Tm += -(-sig1 + sig2)*exp(-alp**2*cv*t)*exp(alp**2*cv*t1)/(alp**2*cv*(-t1 + t2)) + (-sig1 + sig2)*exp(-alp**2*cv*t)*exp(alp**2*cv*t2)/(alp**2*cv*(-t1 + t2))
            Tm += -(-sig1 + sig2)*exp(-alp**2*cv*(t-t1))/(alp**2*cv*(-t1 + t2)) + (-sig1 + sig2)*exp(-alp**2*cv*(t-t2))/(alp**2*cv*(-t1 + t2))
        return Tm

    def _calc_un(self):
        """u(r) part of u(r, z, t) = u(r) * phi(z) * T(t)"""

        self._un = np.ones_like(self._sn)


        if not self.kh is None:
            for i, s in enumerate(self._sn):
                if not self.rcalc is None:
                    self._un[i] = self.un_normalised(self.rcalc, s, 0)
                else:
                    self._un[i] = self.un_normalised_average(s)


    def _calc_por(self):
        """calculate the pore pressure"""

        sin = cmath.sin
        cos = cmath.cos

#        if self.tpor is None:
#            self.tpor==self.t
        if self.tpor is None:
            return

        self.por = np.zeros((len(self.z), len(self.tpor)), dtype=float)




        z_in_layer = np.searchsorted(self.zlayer, self.z)

        self._calc_un()
        for p, t in enumerate(self.tpor):
            for i in range(self.nh):
                s = self._sn[i]
                un = self._un[i]
                for j in range(self.nv):
                    alp = self._alp[i, j]
                    Tm = self._calc_Tm(alp, t)
                    for k, z in enumerate(self.z):
                        layer = z_in_layer[k]
                        zlay = z - (self.zlayer[layer] - self.h[layer])

                        bet = self._betamn[i, j, layer]
                        Cmn = self._Cmn[i, j].real
                        phi_a = self._phia[i, j, layer]
                        phi_a_dot = self._phidota[i, j, layer]
                        phi = (cos(bet * zlay) * phi_a +
                            sin(bet * zlay)/bet * phi_a_dot)
                        self.por[k, p] += Cmn * un * phi.real * Tm


    def _calc_avp(self):
        """calculate the average pore pressure"""

        sin = cmath.sin
        cos = cmath.cos

        h_all = sum(self.h)

        if self.t is None:
            return

        self.avp = np.zeros((1, len(self.t)), dtype=float)
        self.set = np.zeros((1, len(self.t)), dtype=float)

        z_in_layer = np.searchsorted(self.zlayer, self.z)

        self._calc_un()
        for p, t in enumerate(self.t):
            for i in range(self.nh):
                s = self._sn[i]
                un = self._un[i]
                for j in range(self.nv):
                    alp = self._alp[i, j]
                    Tm = self._calc_Tm(alp, t)

                    load = pwise.pinterp_x_y(self.surcharge_vs_time, t)
                    for layer, h in enumerate(self.h):
#                        layer = z_in_layer[k]
#                        zlay = z - (self.zlayer[layer] - self.h[layer])

                        bet = self._betamn[i, j, layer]
                        Cmn = self._Cmn[i, j].real
                        phi_a = self._phia[i, j, layer]
                        phi_a_dot = self._phidota[i, j, layer]
                        phi = (sin(bet * h) / bet * phi_a +
                                (1-cos(bet * h))/bet**2*phi_a_dot)
                        self.avp[0, p] += Cmn * un * phi.real / h_all * Tm
                        self.set[0, p] += self.mv[layer] * (load * h /self.nh/self.nv - Cmn *
                                            un * phi.real * Tm)

    def _plot_vert_roots(self, npt=200):
        """Plot the vertical characteristic curve and it's roots

        After a 'successful' run, use this to check the validity
        of the calculated vertical eigenvalues and ensure none are missing

        Parameters
        ----------
        npt : int, optional
            number of points to plot. default=200

        """

        fig = plt.figure(figsize=(40, 8))
        ax = fig.add_subplot('111')
        for i in range(self.nh):
            s = self._sn[i]
#            amin=self._alp_min()[i]
#            amin=0.001
            amin = 0.3*self._alp[i,0]
            x = np.linspace(amin, self._alp[i,-1], npt)


            y = np.zeros_like(x)
            for j,_x in enumerate(x):

                y[j] = self._vertical_characteristic_curve(_x, s)
            #        print(x[i],y[i])


            #    print(y)
            ax.plot(x, y, ls='-', marker='.', markersize=3)
            c = ax.get_lines()[-1].get_color()
            ax.set_ylim((-1,1))
            ax.plot(self._alp[i,:], np.zeros_like(self._alp[i,:]), 'o', color=c)
        ax.set_title('vertical_roots_x0={}, vertical_roots_dx={}, vertical_roots_p={}'.format(self.vertical_roots_x0, self.vertical_roots_dx, self.vertical_roots_p))
        ax.set_xlabel('beta')
        ax.set_ylabel('value of characterisrtic curve')
        ax.grid()
        fig.tight_layout()
        return fig



def main():
    """Run nogamiandli2003 as script"""
    a = GenericInputFileArgParser(obj=NogamiAndLi2003,
                                  methods=[('make_all', [], {})],
                                 pass_open_file=True)

    a.main()

if __name__ == '__main__':
#    import nose
#    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
##    nose.runmodule(argv=['nose', '--verbosity=3'])
    main()




