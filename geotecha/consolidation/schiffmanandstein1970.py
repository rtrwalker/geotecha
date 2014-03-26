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
module for Schiffman and Stein 1970 multi layer consolidation

"""
from __future__ import print_function, division

import numpy as np
from matplotlib import pyplot as plt
import geotecha.inputoutput.inputoutput as inputoutput
import math
import textwrap
import scipy.optimize
import geotecha.piecewise.piecewise_linear_1d as pwise

from geotecha.math.root_finding import find_n_roots



def plot_one_dim_consol(z, t, por=None, doc=None, settle=None, uavg=None):

    if not por is None:
        plt.figure()
        plt.plot(por,z)
        plt.ylabel('Depth, z')
        plt.xlabel('Pore pressure')
        plt.gca().invert_yaxis()

    if not doc is None:
        plt.figure()
        plt.plot(t, doc)
        plt.xlabel('Time, t')
        plt.ylabel('Degree of consolidation')
        plt.gca().invert_yaxis()
        plt.semilogx()

    if not settle is None:
        plt.figure()
        plt.plot(t, settle)
        plt.xlabel('Time, t')
        plt.ylabel('settlement')
        plt.gca().invert_yaxis()
        plt.semilogx()

    if not uavg is None:
        plt.figure()
        plt.plot(t, uavg)
        plt.xlabel('Time, t')
        plt.ylabel('Average pore pressure')
#        plt.gca().invert_yaxis()
        plt.semilogx()

    return



class SchiffmanAndStein1970(inputoutput.InputFileLoaderCheckerSaver):
    """Multi-layer consolidation


    Attributes
    ----------
    z : list/array of float
        depth to calc pore pressure at
    t : list/array of float
        time values to calc at
    tpor : list/array of float
        time values to calc pore pressure profiles at
    h : list/array of float
        layer depths thickness
    n : int, optional
        number of series terms to use. default n=5
    kv : list/array of float
        layer vertical permeability divided by unit weight of water
    mv : list/array of float
        layer volume compressibility
    bctop, bcbot : [0, 1, 2]
        boundary condition. bctop=0 is free draining, bctop=1 is
        impervious, bctop = impeded.
    htop, hbot : float
        thickness of impeding layer.
    ktop, hbot : float
        impeding layer vertical permeability divided py unit weight of
        water
    surcharge_vs_time : PolyLine
        piecewise linear variation of surcharge with time

    """

    def _setup(self):
        self._attribute_defaults = {'n': 5}
        self._attributes = 'z t tpor n h kv mv bctop bcbot htop ktop hbot kbot surcharge_vs_time'.split()
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


#    def __init__(self, reader=None):
#        self._debug = False
#        self._setup()
#
#        inputoutput.initialize_objects_attributes(self,
#                                                  self._attributes,
#                                                  self._attribute_defaults,
#                                                  not_found_value = None)
#
#        self._input_text = None
#        if not reader is None:
#            if isinstance(reader, str):
#                self._input_text = reader
#            else:
#                self._input_text = reader.read()
#
#            inputoutput.copy_attributes_from_text_to_object(reader,self,
#                self._attributes, self._attribute_defaults,
#                not_found_value = None)
#
#    def check_all(self):
#        """perform checks on attributes
#
#        Notes
#        -----
#
#        See also
#        --------
#        geotecha.inputoutput.inputoutput.check_attribute_combinations
#        geotecha.inputoutput.inputoutput.check_attribute_is_list
#        geotecha.inputoutput.inputoutput.check_attribute_PolyLines_have_same_x_limits
#        geotecha.inputoutput.inputoutput.check_attribute_pairs_have_equal_length
#
#        """
#
#
#        inputoutput.check_attribute_combinations(self,
#                                                 self._zero_or_all,
#                                                 self._at_least_one,
#                                                 self._one_implies_others)
#        inputoutput.check_attribute_is_list(self, self._attributes_that_should_be_lists, force_list=True)
#        inputoutput.check_attribute_PolyLines_have_same_x_limits(self, attributes=self._attributes_that_should_have_same_x_limits)
#        inputoutput.check_attribute_pairs_have_equal_length(self, attributes=self._attributes_that_should_have_same_len_pairs)
#
#        return


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
            raise ValueError('bctop must be 0, 1, or 2. you have bctop = %s' % self.bctop)

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
            raise ValueError('bctop must be 0, 1, or 2. you have bctop = %s' % self.bctop)

        self.BC = np.zeros((2 * self.nlayers, 2* self.nlayers), dtype=float)

    def calc(self):
        """Perform all calculations"""

        self._calc_derived_properties()

        self._find_beta()


        self._calc_Bm_and_Cm()

        self._calc_Am()

        self.calc_por()

        self.calc_settle_and_uavg()

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

    def plot_characteristic_curve_and_roots(self):

        x = np.linspace(0, self._beta0[-1] + (self._beta0[-1]-self._beta0[-2])/8, 400)
        y = np.zeros_like(x)
        for i in xrange(len(x)):
            y[i]=self._characteristic_eqn(x[i])
        plt.gcf().clear()
        plt.plot(x ,y,'-')
        plt.plot(self._beta0, np.zeros_like(self._beta0),'ro')
        plt.ylabel('det(A)')
        plt.xlabel('beta0')
#        plt.gca().set_ylim(-0.1,0.1)
        plt.grid()
        plt.show()
        plt.gcf().clear()

        return

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

    def _uavg_integrations(self):
        """symbolic integration of for uavg average pore pressure

        just used as a step to derive some code"""

        import sympy

        z, mv, Bm, Cm, beta, f, Zm, z1, z2 = sympy.var('z, mv, Bm, Cm, beta, f, Zm, z1, z2')

        Zm = Bm * sympy.cos(beta * z) + Cm * sympy.sin(beta * z)

        f = sympy.integrate(Zm, (z, z1, z2))
        print('summation term for uavg')
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

    def calc_settle_and_uavg(self):

        self.settle = np.zeros(len(self.t), dtype=float)
        self.uavg = np.zeros(len(self.t), dtype=float)
        _z2 = self.zlayer
        _z1 = self.zlayer - self.h
#        print(_z1,_z2)
        sin = math.sin
        cos = math.cos
        for j, t in enumerate(self.t):
            settle=0
            uavg = 0
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

                    uavg += Zm_integral * Tm * Am

                    settle -= mv * Zm_integral * Tm * Am


            self.settle[j] = settle
            self.uavg[j] = uavg / self.zlayer[-1]

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


if __name__ == '__main__':

    my_code = textwrap.dedent("""\
    from geotecha.piecewise.piecewise_linear_1d import PolyLine
    import numpy as np

    h = np.array([10, 20, 30, 20])
    cv = np.array([0.0411, 0.1918, 0.0548, 0.0686])
    mv = np.array([3.07e-3, 1.95e-3, 9.74e-4, 1.95e-3])
    #kv = np.array([7.89e-6, 2.34e-5, 3.33e-6, 8.35e-6])
    kv = cv*mv

    bctop = 0
    #htop = None
    #ktop = None
    bcbot = 0
    #hbot = None
    #kbot = None

    n = 40
    surcharge_vs_time = PolyLine([0,0,10], [0,100,100])
    z = np.concatenate((np.linspace(0, np.sum(h[:1]), 25, endpoint=False),
                        np.linspace(np.sum(h[:1]), np.sum(h[:2]), 25, endpoint=False),
                        np.linspace(np.sum(h[:2]), np.sum(h[:3]), 25, endpoint=False),
                        np.linspace(np.sum(h[:3]), np.sum(h), 25, endpoint=True)))



    tpor = np.array([740, 2930, 7195], dtype=float)

    t = np.logspace(1, 4.5, 30)

    t = np.array(
        [1.21957046e+02,   1.61026203e+02,   2.12611233e+02,
         2.80721620e+02,   3.70651291e+02,   4.89390092e+02,
         740.0,   8.53167852e+02,   1.12648169e+03,
         1.48735211e+03,   1.96382800e+03,   2930.0,
         3.42359796e+03,   4.52035366e+03,   5.96845700e+03,
         7195.0,   1.04049831e+04,   1.37382380e+04,
         1.81393069e+04,   2.39502662e+04,   3.16227766e+04])

    #z = np.linspace(0.0, np.sum(h), 100)

    #
    #z=np.array([0,10])
    #t = np.linspace(0,10000, 100)
    ##t=np.array([1])
    #
    #h=np.array([1])
    #kv=np.array([1])
    #mv=np.array([1])
    #surcharge_vs_time = PolyLine([0,0,8], [0,100,100])
    #surcharge_vs_time = PolyLine([0,0.1,8], [0,100,100])
    #
    #z = np.linspace(0.0, np.sum(h), 7)
    ##t = np.linspace(0, 10, 50)
    #t = np.logspace(-1,1.8,100)
    """)

    a = SchiffmanAndStein1970(my_code)


#
#    a._calc_derived_properties()
#    a._find_beta()
##    a.plot_characteristic_curve_and_roots()
#    a._make_BC(a._beta0[0])
#    a._Am_integrations()
#    a._Tm_integrations()
#    a._uavg_integrations()
#    a._debug=True

    a.calc()
    plot_one_dim_consol(a.z, a.t, por=a.por, uavg=a.uavg, settle=a.settle)
    plt.show()
    print(repr(a.z))
    print('*')
    print(repr(a.por))
    print('*')
    print(repr(a.uavg))
    print('*')
    print(repr(a.settle))
    print('*')
    print(repr(a.t))
#    plt.plot(a.por,a.z)
#    plt.ylabel('Depth, z')
#    plt.xlabel('Pore pressure')
#    plt.gca().invert_yaxis()
#    plt.grid()
#    plt.show()
#    x = np.linspace(0, 20, 400)
##    x = np.array([0.1])
#    y = np.zeros_like(x)
#    for i in xrange(len(x)):
#        y[i]=a._characteristic_eqn(x[i])
##        print(x[i],y[i])
#
#    print(np.sum(y[0:-1] * y[1:] < 0))
#    plt.plot(x ,y,'-')
#    plt.plot(a._beta0, np.zeros_like(a._beta0),'ro')
##    plt.gca().set_ylim(-0.1,0.1)
#    plt.grid()
#    plt.show()
#    print(inputoutput.code_for_explicit_attribute_initialization('z t n h kv mv bctop bcbot htop ktop hbot kbot'.split(), {'n': 5} ))