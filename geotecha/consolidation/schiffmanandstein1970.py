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
Schiffman and Stein 1970 "One-Dimensional consolidation of layered systems".

"""
from __future__ import print_function, division

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import geotecha.inputoutput.inputoutput as inputoutput
import math
import textwrap
import scipy.optimize
import geotecha.piecewise.piecewise_linear_1d as pwise

from geotecha.mathematics.root_finding import find_n_roots
import time
import geotecha.plotting.one_d


from geotecha.inputoutput.inputoutput import GenericInputFileArgParser

class SchiffmanAndStein1970(inputoutput.InputFileLoaderCheckerSaver):
    """One-dimensional consolidation of layered systems

    Implements Schiffman and Stein (1970) [1]_.

    Features:

     - Vertical flow, multiple layers.
     - Soil properties constant with time.
     - Drained or Undrained boundary condtions. Also possbile to have
       stiff high peremability impeding layers at top and bottom.
     - Load is uniform with depth but varies piecewise-linear with time.
     - Pore pressure vs depth at various times.
     - Average pore pressure vs time.  Average is over the entire soil layer.
     - Settlement vs time.  Settlement is over whole profile.


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
        Time values to calc time variation of parameters at.
    tpor : list/array of float
        Time values to calc pore pressure profiles at.
    h : list/array of float
        Layer thicknesses.
    n : int, optional
        Number of series terms to use.  Default n=5.
    kv : list/array of float
        Layer vertical permeability divided by unit weight of water.
    mv : list/array of float
        Layer volume compressibility.
    bctop, bcbot : [0, 1, 3]
        Boundary condition. bctop=0 is free draining, bctop=1 is
        impervious, bctop=3 is impeded by stiff layer of thickness htop
        and permeability ktop.
    htop, hbot : float, optional
        Thickness of top and bottom impeding layer. Only used if bcbot==3 or
        bctop==3.
    ktop, kbot : float, optional
        Top and bottom impeding layer vertical permeability divided by
        unit weight of water. Only used if bcbot==3 or bctop==3.
    surcharge_vs_time : PolyLine
        Piecewise linear variation of surcharge with time
    show_vert_eigs : True/False, optional
        If true a vertical eigen value plot will be made.  This is useful to
        to chekc if the correct eigenvalues have been found.
        Default show_vert_eigs=False
    plot_properties : dict of dict, optional
        dictionary that overrides some of the plot properties.
        Each member of `plot_properties` will correspond to one of the plots.

        ==================  ============================================
        plot_properties    description
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
    title: str, optional
        A title for the input file.  This will appear at the top of data files.
        Default title=None, i.e. no title.
    author: str, optional
        Author of analysis. Default='unknown'.


    Attributes
    ----------
    por : array of shape (len(z), len(tpor))
        Pore pressure vs depth at various times.  Only present if tpor defined.
    avp : array of shape (1, len(t))
        Averge pore pressure of profile various times.  Only present if t
        defined.
    set : array of shape (1, len(t))
        Surface settlement at various times.  Only present if t
        defined.


    See Also
    --------
    geotecha.piecewise.piecewise_linear_1d.PolyLine : how to specify loadings


    References
    ----------
    .. [1] Schiffman, R. L, and J. R Stein. 'One-Dimensional Consolidation
           of Layered Systems'. Journal of the Soil Mechanics and
           Foundations Division 96, no. 4 (1970): 1499-1504.

    """

    def _setup(self):
        self._attribute_defaults = {'n': 5,
                                    'prefix': 'ss1970_',
                                    'show_vert_eigs': False}
        self._attributes = 'z t tpor n h kv mv bctop bcbot htop ktop hbot kbot surcharge_vs_time show_vert_eigs'.split()
        self._attributes_that_should_have_same_len_pairs = [
        'h kv'.split(),
        'kv mv'.split(),
        'h mv'.split()] #pairs that should have the same length

        self._attributes_that_should_be_lists= []
        self._attributes_that_should_have_same_x_limits = []


        self.z = None
        self.t = None
        self.tpor = None
        self.n = self._attribute_defaults.get('n', None)
        self.prefix = self._attribute_defaults.get('prefix', None)
        self.show_vert_eigs = self._attribute_defaults.get('show_vert_eigs', None)
        self.h = None
        self.kv = None
        self.mv = None
        self.bctop = None
        self.bcbot = None
        self.htop = None
        self.ktop = None
        self.hbot = None
        self.kbot = None
        self.surcharge_vs_time = None

        self._zero_or_all = [
            'h kv mv'.split(),
            'htop ktop'.split(),
            'hbot kbot'.split(),
            'z t'.split()]
        self._at_least_one = [['surcharge_vs_time']]
        self._one_implies_others = []


    def _calc_derived_properties(self):
        """Calculate properties/ratios derived from input"""

        self.check_input_attributes()

        self.t = np.asarray(self.t)
        self.z = np.asarray(self.z)
        self.kv = np.asarray(self.kv)
        self.mv = np.asarray(self.mv)
        self.h = np.asarray(self.h)

        self.nlayers = len(self.kv)

        self.zlayer = np.cumsum(self.h)
#        print (self.zlayer)
#        self.zlayer = np.zeros(nlayers +1, dtype=float)
#        self.zlayer[1:] = np.cumsum(self.h)


        self.cv = self.kv / self.mv



        if self.bctop == 0:
            self.atop = 0
            self.btop = -1
        elif self.bctop == 1:
            self.atop = 1
            self.btop = 0
        elif self.bctop == 3:
            self.atop = h[0]
            self.btop = ktop * h[0] / (kv[0] * htop)
        else:
            raise ValueError('bctop must be 0, 1, or 2. you have bctop = {}'.foramt(self.bctop))

        if self.bcbot == 0:
            self.abot = 0
            self.bbot = 1
        elif self.bcbot == 1:
            self.abot = 1
            self.bbot = 0
        elif self.bcbot == 3:
            self.abot = h[-1]
            self.bbot = kbot * h[-1] / (kv[-1] * hbot)
        else:
            raise ValueError('bctop must be 0, 1, or 2. you have bctop = {}'.format(self.bctop))

        self.BC = np.zeros((2 * self.nlayers, 2* self.nlayers), dtype=float)



    def produce_plots(self):
        """produce plots of analysis"""

#        geotecha.plotting.one_d.pleasing_defaults()

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

            f = self.plot_characteristic_curve_and_roots(1000)
            title = 'vertical characteristic curve and eigs'
            f.set_label(title)
            f.canvas.manager.set_window_title(title)
            self._figures.append(f)

    def _plot_por(self):
        """plot depth vs pore pressure for various times

        """

        t = self.tpor
        line_labels = ['{:.3g}'.format(v) for v in t]
        por_prop = self.plot_properties.pop('por', dict())
        if not 'xlabel' in por_prop:
            por_prop['xlabel'] = 'Pore pressure'

        #to do
        fig_por = geotecha.plotting.one_d.plot_vs_depth(self.por, self.z,
                                      line_labels=line_labels,
                                      prop_dict=por_prop)
        return fig_por

    def _plot_avp(self):
        """plot average pore pressure of profile"""


        t = self.t
        line_labels = ['{:.3g} to {:.3g}'.format(0, sum(self.h))]

        avp_prop = self.plot_properties.pop('avp', dict())
        if not 'ylabel' in avp_prop:
            avp_prop['ylabel'] = 'Average pore pressure'
        fig_avp = geotecha.plotting.one_d.plot_vs_time(t, self.avp.T,
                           line_labels=line_labels,
                           prop_dict=avp_prop)
        return fig_avp

    def _plot_set(self):
        """plot surface settlement"""



        t = self.t
        line_labels = ['{:.3g} to {:.3g}'.format(0, sum(self.h))]

        set_prop = self.plot_properties.pop('set', dict())
        if not 'ylabel' in set_prop:
            set_prop['ylabel'] = 'surface settlement'
        fig_set = geotecha.plotting.one_d.plot_vs_time(t, self.set.T,
                           line_labels=line_labels,
                           prop_dict=set_prop)
        fig_set.gca().invert_yaxis()
        return fig_set

    def make_all(self):


#        self.check_input_attributes()
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
        """Perform all calculations"""

        self._calc_derived_properties()

        self._find_beta()


        self._calc_Bm_and_Cm()

        self._calc_Am()


        header1 = "program: schiffmanandstein1970; geotecha version: {}; author: {}; date: {}\n".format(self.version, self.author, time.strftime('%Y/%m/%d %H:%M:%S'))
        if not self.title is None:
            header1 += "{}\n".format(self.title)




        self._grid_data_dicts = []
        if not self.tpor is None:

            self.calc_por()
            labels = ['{:.3g}'.format(v) for v in self.z]
            d = {'name': '_data_por',
                 'data': self.por.T,
                 'row_labels': self.tpor,
                 'row_labels_label': 'Time',
                 'column_labels': labels,
                 'header': header1 + 'Pore pressure at depth'}
            self._grid_data_dicts.append(d)

        if not self.t is None:
            self.calc_settle_and_avp()

            labels = ['{:.3g} to {:.3g}'.format(0, sum(self.h))]
            d = {'name': '_data_avp',
                 'data': self.avp.T,
                 'row_labels': self.t,
                 'row_labels_label': 'Time',
                 'column_labels': labels,
                 'header': header1 + 'Average pore pressure between depths'}
            self._grid_data_dicts.append(d)

            labels = ['{:.3g} to {:.3g}'.format(0, sum(self.h))]
            d = {'name': '_data_set',
                 'data': self.avp.T,
                 'row_labels': self.t,
                 'row_labels_label': 'Time',
                 'column_labels': labels,
                 'header': header1 + 'settlement between depths'}
            self._grid_data_dicts.append(d)

        if self._debug:
            print ('beta')
            print (self._beta)
            print('Bm')
            print(self._Bm)
            print('Cm')
            print(self._Cm)
            print('Am')
            print(self._Am)

        return

    def _find_beta(self):
        """find the eigenvalues of the solution

        """

        H = self.zlayer[-1]

        x0 = 0.1 / H**2
        self._beta0 = np.empty(self.n, dtype=float)

        self._beta0[:] = find_n_roots(self._characteristic_eqn, n=self.n,
            x0=x0, dx=x0, p=1.01)


        return







    def _characteristic_eqn(self, beta0):
        """function for characteristic equation

        Roots are the eigenvalues of problem beta 1

        """

        self._make_BC(beta0)

        return np.linalg.det(self.BC)

    def _make_BC(self, beta0):
        """make boundary condition matrix

        for use in characteristic equatin and in determining coefficients B,C

        """

        beta = np.zeros_like(self.h, dtype=float)
        beta[0] = beta0
        for i in xrange(1, self.nlayers):
            beta[i] = np.sqrt(self.cv[i-1] / self.cv[i] * beta[i-1]**2)

        alpha = self.kv[:-1] / self.kv[1:]

        self.BC[0, 0] = self.btop * (-1)
        self.BC[0, 1] = self.atop * beta[0]

        self.BC[-1, -2] = (self.bbot * math.cos(beta[-1] * self.zlayer[-1]) -
                     self.abot * beta[-1] * math.sin(beta[-1] * self.zlayer[-1]))
        self.BC[-1, -1] = (self.bbot * math.sin(beta[-1] * self.zlayer[-1]) +
                     self.abot * beta[-1] * math.cos(beta[-1] * self.zlayer[-1]))

        for i in xrange(self.nlayers - 1):
            #1st equation
            #TODO: row is wrong
            row = 2 * i + 1
            self.BC[row, 2 * i] = math.cos(beta[i] * self.zlayer[i])#Bi coeef
            self.BC[row, 2 * i + 1] = math.sin(beta[i] * self.zlayer[i])#Ci coeef#C coeff
            self.BC[row, 2 * i + 2] = -math.cos(beta[i+1] * self.zlayer[i]) #Bi+1 coeef
            self.BC[row, 2 * i + 3] = -math.sin(beta[i+1] * self.zlayer[i])#Ci+1 coeff

            #2nd equation
            row += 1
            self.BC[row, 2 * i] = - alpha[i] * beta[i] * math.sin(beta[i] * self.zlayer[i])#Bi coeef
            self.BC[row, 2 * i + 1] = alpha[i] * beta[i] * math.cos(beta[i] * self.zlayer[i])#Ci coeef#C coeff
            self.BC[row, 2 * i + 2] = beta[i+1] * math.sin(beta[i+1] * self.zlayer[i]) #Bi+1 coeef
            self.BC[row, 2 * i + 3] = - beta[i+1] * math.cos(beta[i+1] * self.zlayer[i])#Ci+1 coeff

        return beta

    def plot_characteristic_curve_and_roots(self, npts=400):

        x = np.linspace(0, self._beta0[-1] + (self._beta0[-1]-self._beta0[-2])/8, npts)
        y = np.zeros_like(x)
        for i in xrange(len(x)):
            y[i]=self._characteristic_eqn(x[i])
#        plt.gcf().clear()
        fig = plt.figure(figsize=(30,5))
        ax = fig.add_subplot('111')
        ax.plot(x ,y,'-', marker='.', markersize=3)
        ax.plot(self._beta0, np.zeros_like(self._beta0),'ro', markersize=6)
        ax.set_ylabel('det(A)')
        ax.set_xlabel('beta0')
#        plt.gca().set_ylim(-0.1,0.1)
        ax.grid()
        fig.tight_layout()
        return fig

    def _calc_Bm_and_Cm(self):
        """calculate the coefficinets Bm and Cm"""
        self._Bm = np.zeros((self.n, self.nlayers), dtype=float)
        self._Cm = np.zeros((self.n, self.nlayers), dtype=float)

        self._beta = np.zeros((self.n, self.nlayers), dtype=float)

        self._Cm[:, -1] = 1.0
        for i, beta in enumerate(self._beta0):
            self._beta[i, :] = self._make_BC(beta)
            self.BC[np.abs(self.BC)<1e-10]=0
            if self._debug and i==0:
                print('BC for beta0')
                print(self.BC)
            b = -self.BC[:-1, -1]
            a = self.BC[:-1, :-1]
            x = np.linalg.solve(a, b)
            self._Bm[i, :] = x[::2]
            self._Cm[i, :-1] = x[1::2]

    def _Tm_integrations(self):
        """symbolic integration of the Tm coefficient

        just used as a step to derive some code"""

        import sympy

        cv, beta, t, tau, t1, t2, sig1, sig2 =  sympy.var('cv, beta, t, tau, t1, t2, sig1, sig2')

        q = sig1 + (sig2 - sig1) / (t2 - t1) * tau

        f = sympy.diff(q, tau) * sympy.exp(-cv * beta**2 * (t - tau))

        #uniform laod
        #within ramp
        Tm = sympy.integrate(f, (tau, t1, t))
        print('Tm within a ramp load')
        print(Tm)
#        after ramp
        Tm = sympy.integrate(f, (tau, t1, t2))
        print('Tm after a ramp load')
        print(Tm)
        return

    def _avp_integrations(self):
        """symbolic integration of for avp average pore pressure

        just used as a step to derive some code"""

        import sympy

        z, mv, Bm, Cm, beta, f, Zm, z1, z2 = sympy.var('z, mv, Bm, Cm, beta, f, Zm, z1, z2')

        Zm = Bm * sympy.cos(beta * z) + Cm * sympy.sin(beta * z)

        f = sympy.integrate(Zm, (z, z1, z2))
        print('summation term for avp')
        print(f)

        return

    def _Am_integrations(self):
        """symbolic integration of for Am coefficient

        just used as a step to derive some code"""

        import sympy

        z, mv, Bm, Cm, beta, f, Zm, z1, z2 = sympy.var('z, mv, Bm, Cm, beta, f, Zm, z1, z2')

        Zm = Bm * sympy.cos(beta * z) + Cm * sympy.sin(beta * z)

        #uniform initial pore pressure
        numerator = mv * sympy.integrate(Zm, (z, z1, z2))
        denominator = mv * (sympy.integrate(Zm**2, (z, z1, z2)))
#        Am = numerator / denominator
        print('Am numerator - uniform initial pore pressure')
        print(numerator)
        print('Am denominator - uniform initial pore pressure')
        print(denominator)
#        print('**')
#        print(Am)
    def _calc_Am(self):
        """make the Am coefficients"""

        cos = math.cos
        sin = math.sin
        self._Am = np.zeros(self.n, dtype=float)

        _z2 = self.zlayer
        _z1 = self.zlayer - self.h

        for m in range(self.n):
            numer = 0
            denom = 0
            for i in range(self.nlayers):
                z1=_z1[i]
                z2=_z2[i]
                mv = self.mv[i]
                Bm = self._Bm[m, i]
                Cm = self._Cm[m, i]
                beta = self._beta[m, i]

                numer += mv*(-Bm*sin(beta*z1)/beta + Bm*sin(beta*z2)/beta +
                    Cm*cos(beta*z1)/beta - Cm*cos(beta*z2)/beta)
                denom += mv*(-Bm**2*z1*sin(beta*z1)**2/2 -
                    Bm**2*z1*cos(beta*z1)**2/2 + Bm**2*z2*sin(beta*z2)**2/2 +
                    Bm**2*z2*cos(beta*z2)**2/2 -
                    Bm**2*sin(beta*z1)*cos(beta*z1)/(2*beta) +
                    Bm**2*sin(beta*z2)*cos(beta*z2)/(2*beta) +
                    Bm*Cm*cos(beta*z1)**2/beta - Bm*Cm*cos(beta*z2)**2/beta -
                    Cm**2*z1*sin(beta*z1)**2/2 - Cm**2*z1*cos(beta*z1)**2/2 +
                    Cm**2*z2*sin(beta*z2)**2/2 + Cm**2*z2*cos(beta*z2)**2/2 +
                    Cm**2*sin(beta*z1)*cos(beta*z1)/(2*beta) -
                    Cm**2*sin(beta*z2)*cos(beta*z2)/(2*beta))

            Am = numer / denom
            self._Am[m] = Am

        return

    def _calc_Tm(self, cv, beta, t):
        """calculate the Tm expression at a given time

        Parameters
        ----------
        cv : float
            coefficient of vertical consolidation for layer
        beta : float
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
        i=0 #only one time value
        for k in steps_less_than_t[i]:
            sig1 = loadmag[k]
            sig2 = loadmag[k+1]

            Tm += (sig2-sig1)*exp(-cv * beta**2 * (t-loadtim[k]))
        for k in ramps_containing_t[i]:
            sig1 = loadmag[k]
            sig2 = loadmag[k+1]
            t1 = loadtim[k]
            t2 = loadtim[k+1]

#            Tm += (-sig1 + sig2)/(beta**2*cv*(-t1 + t2)) - (-sig1 + sig2)*exp(-beta**2*cv*t)*exp(beta**2*cv*t1)/(beta**2*cv*(-t1 + t2))
            Tm += (-sig1 + sig2)/(beta**2*cv*(-t1 + t2)) - (-sig1 + sig2)*exp(-beta**2*cv*(t-t1))/(beta**2*cv*(-t1 + t2))

        for k in ramps_less_than_t[i]:
            sig1 = loadmag[k]
            sig2 = loadmag[k+1]
            t1 = loadtim[k]
            t2 = loadtim[k+1]
#            Tm += -(-sig1 + sig2)*exp(-beta**2*cv*t)*exp(beta**2*cv*t1)/(beta**2*cv*(-t1 + t2)) + (-sig1 + sig2)*exp(-beta**2*cv*t)*exp(beta**2*cv*t2)/(beta**2*cv*(-t1 + t2))
            Tm += -(-sig1 + sig2)*exp(-beta**2*cv*(t-t1))/(beta**2*cv*(-t1 + t2)) + (-sig1 + sig2)*exp(-beta**2*cv*(t-t2))/(beta**2*cv*(-t1 + t2))
        return Tm

    def calc_settle_and_avp(self):

        self.set = np.zeros(len(self.t), dtype=float)
        self.avp = np.zeros(len(self.t), dtype=float)
        _z2 = self.zlayer
        _z1 = self.zlayer - self.h
#        print(_z1,_z2)
        sin = math.sin
        cos = math.cos
        for j, t in enumerate(self.t):
            settle=0
            avp = 0
            q = pwise.pinterp_x_y(self.surcharge_vs_time, t)[0]
            settle = np.sum(self.mv * self.h) * q


            for layer in range(self.nlayers):
                for m in range(self.n):
                    z1=_z1[layer]
                    z2=_z2[layer]
                    Am = self._Am[m]
                    mv = self.mv[layer]
                    Bm = self._Bm[m, layer]
                    Cm = self._Cm[m, layer]
                    beta = self._beta[m, layer]
                    cv = self.cv[layer]

                    Zm_integral = -Bm*sin(beta * z1)/beta + Bm * sin(beta * z2)/beta + Cm * cos(beta*z1)/beta - Cm*cos(beta*z2)/beta
                    Tm = self._calc_Tm(cv, beta, t)

                    avp += Zm_integral * Tm * Am

                    settle -= mv * Zm_integral * Tm * Am


            self.set[j] = settle
            self.avp[j] = avp / self.zlayer[-1]

        return




    def calc_por(self):

        if self.tpor is None:
            self.tpor==self.t

        self.por = np.zeros((len(self.z), len(self.tpor)), dtype=float)


        z_in_layer = np.searchsorted(self.zlayer, self.z)

        for j, t in enumerate(self.tpor):
            for m in range(self.n):
                for k, z in enumerate(self.z):
                    layer = z_in_layer[k]

                    Am = self._Am[m]

                    Bm = self._Bm[m, layer]
                    Cm = self._Cm[m, layer]
                    beta = self._beta[m, layer]
                    cv = self.cv[layer]
                    Zm = Bm * math.cos(beta * z) + Cm * math.sin(beta * z)
#                    Tm = math.exp(-cv * beta**2 * t)
                    Tm = self._calc_Tm(cv, beta, t)

                    self.por[k, j] += Am * Zm * Tm

def main():
    """Run schiffmanandstein1970 as script"""
    a = GenericInputFileArgParser(obj=SchiffmanAndStein1970,
                                  methods=[('make_all', [], {})],
                                 pass_open_file=True)

    a.main()

if __name__ == '__main__':
#    import nose
#    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
##    nose.runmodule(argv=['nose', '--verbosity=3'])
    main()

