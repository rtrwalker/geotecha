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

from geotecha.optimize.root_finding import find_n_roots

import scipy.special
#from scipy.special import j0, y0, j1, y1

besselj = scipy.special.jn
bessely = scipy.special.yn


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



class NogamiAndLi2003(object):
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
    nv, nh : tuple of 2 int, optional
        number of series terms to use in vertical and horizontal direction.
        default nv=nh=5

    kv, kh : list/array of float
        layer vertical and horizontal permeability divided by unit weight of
        water
    mv : list/array of float
        layer volume compressibility
    bctop, bcbot : [0, 1]
        boundary condition. bctop=0 is free draining, bctop=1 is
        impervious.
    surcharge_vs_time : PolyLine
        piecewise linear variation of surcharge with time
    r1, r0 : float optional
        drain influence zone and drain radius. if either is none then only
        vertical drainage will be considered.


    """

    def _setup(self):
        self._attribute_defaults = {'bctop':0, 'bcbot':0}
        self._attributes = ('z t tpor nv nh h kv kh mv bctop bcbot '
            'surcharge_vs_time r0 r1').split()
        self._attributes_that_should_have_same_len_pairs = (
            'h kv '
            'kv mv '
            'h mv '
#            'h kh '
            'h kv '
            'h mv').split() #pairs that should have the same length

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
        self.bctop = self._attribute_defaults.get('top', None)
        self.bcbot = self._attribute_defaults.get('bcbot', None)
        self.r0 = None
        self.r1 = None

        self.surcharge_vs_time = None

        self._zero_or_all = [
            'h kv mv'.split(),
            'z t'.split(),
            'r0 r1'.split()]
        self._at_least_one = [
            ['mv'],
            ['surcharge_vs_time'],
            'kv kh'.split(),]
        self._one_implies_others = ['r0 r1 kh nh'.split(),
                                    'r1 r0 kh nh'.split(),
                                    'kh r0 r1 nh'.split(),
                                    'nh kh r0 r1'.split()]


    def __init__(self, reader=None):
        self._debug = False
        self._setup()

        inputoutput.initialize_objects_attributes(self,
                                                  self._attributes,
                                                  self._attribute_defaults,
                                                  not_found_value = None)

        self._input_text = None
        if not reader is None:
            if isinstance(reader, str):
                self._input_text = reader
            else:
                self._input_text = reader.read()

            inputoutput.copy_attributes_from_text_to_object(reader,self,
                self._attributes, self._attribute_defaults,
                not_found_value = None)

    def check_all(self):
        """perform checks on attributes

        Notes
        -----

        See also
        --------
        geotecha.inputoutput.inputoutput.check_attribute_combinations
        geotecha.inputoutput.inputoutput.check_attribute_is_list
        geotecha.inputoutput.inputoutput.check_attribute_PolyLines_have_same_x_limits
        geotecha.inputoutput.inputoutput.check_attribute_pairs_have_equal_length

        """


        inputoutput.check_attribute_combinations(self,
                                                 self._zero_or_all,
                                                 self._at_least_one,
                                                 self._one_implies_others)
        inputoutput.check_attribute_is_list(self, self._attributes_that_should_be_lists, force_list=True)
        inputoutput.check_attribute_PolyLines_have_same_x_limits(self, attributes=self._attributes_that_should_have_same_x_limits)
        inputoutput.check_attribute_pairs_have_equal_length(self, attributes=self._attributes_that_should_have_same_len_pairs)

        return


    def _calc_derived_properties(self):
        """Calculate properties/ratios derived from input"""

        self.check_all()

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

        self.use_normalised = True

        if self.bctop == 0:
            self.phia0 = np.array([0.0, 1.0])
        elif self.bctop == 1:
            self.phia0 = np.array([1.0, 0.0])

        #index of phib at bottom to check
        if self.bcbot == 0:
            self.phi_i_check = 0
        elif self.bcbot == 1:
            self.phi_i_check = 1




##############
    def un_normalised(self, r, s, order):
        r0 = self.r0
        r1 = self.r1
        return besselj(order, r/r0*s)*bessely(0, s) - besselj(0, s)*bessely(order, r/r0*s)
#
#    def _radial_characteristic_curve(self, s):
#        r0 = self.r0
#        r1 = self.r1
#        return self.un_normalised(r1, s, 1)

###########

    def un(self, r, s):
        r0 = self.r0
        if self.use_normalised:
            return besselj(order, r/r0*s)*bessely(0, s) - besselj(0, s)*bessely(order, r/r0*s)
        else:
            return besselj(0, r*s)*bessely(0, r0*s) - besselj(0, r0*s)*bessely(0, r*s)

    def _radial_characteristic_curve(self, s):
        r0 = self.r0
        r1 = self.r1
        if self.use_normalised:
            return self.un_normalised(r1, s, 1)
        else:
            return besselj(1, r1*s)*bessely(0, r0*s) - besselj(0, r0*s)*bessely(1, r1*s)


    def _find_sn(self):
        """find the radial eigen values"""
        if self.kh is None:
            self._sn=np.array([0])
        else:
            self._sn = find_n_roots(self._radial_characteristic_curve,
                               n=self.nh, p = 1.01)

    def _alp_min(self):
        if self.use_normalised:
            if self.r0 is None:
                r0=1
            else:
                r0 = self.r0
            return np.max(np.sqrt(self.ch[:, np.newaxis]) * self._sn[np.newaxis, :] / r0, axis = 0)
#            return np.max(np.sqrt(self.ch[:, np.newaxis]/ self.cv[:, np.newaxis]) * self._sn[np.newaxis, :] / r0, axis = 0)
        else:
            return np.max(np.sqrt(self.ch[:, np.newaxis]) * self._sn[np.newaxis, :], axis = 0)

    def _beta(self, alp, s):
        if self.use_normalised:
            if self.r0 is None:
                r0=1
            else:
                r0 = self.r0

            a = np.abs(1/self.cv * alp**2 -(self.ch/self.cv)*s**2/r0**2)
            b = np.empty_like(a, dtype=float)
            b[(a>=0)] = np.sqrt(a[(a>=0)])
            b[(a<0)] = -np.sqrt(np.abs(a[a<0]))
            return b
#            return np.sqrt(np.abs(np.abs(1/self.cv * alp**2 -(self.ch/self.cv)*s**2/r0**2)))
#            return np.sqrt(alp**2 -(self.ch/self.cv)*s**2/r0**2)
        else:
            return np.sqrt(1/self.cv * alp**2 -(self.ch/self.cv)*s**2)


    def _calc_phia_and_phidota(self):

        self._phia = np.zeros((self.nh, self.nv, self.nlayers), dtype=float)
        self._phidota = np.zeros((self.nh, self.nv, self.nlayers), dtype=float)

        self._phia[:,:,0] = self.phia0[0]
        self._phidota[:,:,0] = self.phia0[1]

        for i in range(self.nh):
            s = self._sn[i]
            square = np.zeros((2,2), dtype=float)
            for j in range(self.nv):
                alp = self._alp[i, j]
                phia = np.array([self.phia0[0], self.phia0[1]])
                for k in range(self.nlayers):
                    h = self.h[k]
                    beta = self._betamn[i, j, k]

                    square[0,0] = math.cos(beta*h)
                    square[0,1] = math.sin(beta*h) / beta
                    square[1,0] = -beta*math.sin(beta*h)
                    square[1,1] = math.cos(beta*h)

                    phib = np.dot(square, phia)

                    if k != self.nlayers-1: # we are not in the last layer
                        # transfer phib to next layers phia
                        phia[0] = phib[0]
                        phia[1] = phib[1] * self.kv[k] /  self.kv[k+1]
                        self._phia[i,j,k + 1] = phia[0]
                        self._phidota[i, j, k+1] = phia[1]#phib[1] * self.kv[i] /  self.kv[i+1]

                #check
                if abs(phib[self.phi_i_check])>0.01:
                    raise ValueError('Bottom BC not satisfied')

    def _vertical_characteristic_curve(self, alp, s):

        phia = np.array([self.phia0[0], self.phia0[1]])
        square = np.zeros((2,2), dtype=float)

        beta = self._beta(alp, s)

        for i, h in enumerate(self.h):
            if beta[i]==0:
                phib[:] = phia[:]
            else:
                square[0,0] = math.cos(beta[i]*h)
                square[0,1] = math.sin(beta[i]*h) / beta[i]
                square[1,0] = -beta[i]* math.sin(beta[i]*h)
                square[1,1] = math.cos(beta[i]*h)

                phib = np.dot(square, phia)

            if i != self.nlayers - 1: # we are not in the last layer
                #transfer phib to the next layer phia
                phia[0]= phib[0]
                phia[1] = phib[1] * self.kv[i] /  self.kv[i+1]

        return phib[self.phi_i_check]


    def _find_alp(self):

        self._alp = np.zeros((self.nh, self.nv), dtype=float)

        alp_min = self._alp_min()

        for n, s in enumerate(self._sn):
            if s==0:
                alp_start_offset = 0.0001
            else:
                alp_start_offset = 0
            alp = alp_min[n]
            alp=0.0001
            self._alp[n,:] = find_n_roots(self._vertical_characteristic_curve,
                args=(s,),n= self.nv, x0 = alp+alp_start_offset,
                dx = 0.01, p = 1.01)

    def _calc_Cn(self):

        self._Cn = np.zeros(self.nh, dtype=float)

        if self.kh is None:
            self._Cn[:]=1
            return

        r0 = self.r0
        r1 = self.r1

        for n, s in enumerate(self._sn):
            if self.use_normalised:
                numer = -r0**2/s * self.un_normalised(r0, s, 1)
                denom = r0**2/2 * (r1**2/r0**2 * self.un_normalised(r1, s, 0)**2 -
                    self.un_normalised(r0, s, 1)**2)
            else:
                numer = -(r0/s * (besselj(1, r0*s)*bessely(0, r0*s) -
                    besselj(0, r0*s)*bessely(1, r0*s)))

                denom = (0.5 * (r1**2 * (besselj(0, r1*s)*bessely(0, r0*s) -
                    besselj(0, r0*s)*bessely(0, r1*s))**2 +
                    r0**2 * (besselj(1, r0*s)*bessely(0, r0*s) -
                    besselj(0, r0*s)*bessely(1, r0*s))**2))

            self._Cn[n] = numer / denom

    def _calc_betamn(self):
        self._betamn = np.zeros((self.nh, self.nv, self.nlayers), dtype=float)

        for i in range(self.nh):
            s = self._sn[i]
            for j in range(self.nv):
                alp = self._alp[i, j]
                self._betamn[i,j,:] = self._beta(alp, s)



    def _calc_Amn_and_Bmn(self):

        sin = math.sin
        cos = math.cos

        self._Amn = np.zeros((self.nh, self.nv, self.nlayers), dtype=float)
        self._Bmn = np.zeros((self.nh, self.nv, self.nlayers), dtype=float)

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

                    if bet==0:
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

        self._calc_Amn_and_Bmn()
        self._Cm = np.zeros((self.nh, self.nv), dtype=float)


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
        self._calc_Cn()
        self._calc_Cm()

        self._Cmn = np.zeros((self.nh, self.nv), dtype=float)
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

#        print(self._Cn)
        self._calc_por()

#        self._find_beta()
#
#
#        self._calc_Bm_and_Cm()
#
#        self._calc_Am()
#
#        self.calc_por()
#
#        self.calc_settle_and_uavg()
#
#        if self._debug:
#            print ('beta')
#            print (self._beta)
#            print('Bm')
#            print(self._Bm)
#            print('Cm')
#            print(self._Cm)
#            print('Am')
#            print(self._Am)

        return

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

        self._un= np.ones_like(self._sn)

        if not self.kh is None:
            rx = self.r1
            for i, s in enumerate(self._sn):
                if self.use_normalised:
                    self._un[i] = self.un_normalised(rx, s, 0)
                else:
                    self._un[i] = self.un(rx, s)

    def _calc_por(self):

        if self.tpor is None:
            self.tpor==self.t

        self.por = np.zeros((len(self.z), len(self.tpor)), dtype=float)


        z_in_layer = np.searchsorted(self.zlayer, self.z)

#        rx = self.r1
        self._calc_un()
        for p, t in enumerate(self.tpor):
            for i in range(self.nh):
                s = self._sn[i]
                un = self._un[i]
                for j in range(self.nv):
                    alp = self._alp[i, j]
                    Tm = self._calc_Tm(alp, t)
    #                self._alp = np.zeros((self.nh, self.nv), dtype=float)
                    for k, z in enumerate(self.z):
                        layer = z_in_layer[k]
                        zlay = z - (self.zlayer[layer] - self.h[layer])
#                        print(z, zlay)
                        bet = self._betamn[i, j, layer]
                        Cmn = self._Cmn[i, j]
                        phi_a = self._phia[i, j, layer]
                        phi_a_dot = self._phidota[i, j, layer]
                        phi = (math.cos(bet * zlay) * phi_a +
                            math.sin(bet * zlay)/bet * phi_a_dot)
                        self.por[k, p] += Cmn * un * phi * Tm
#                        if z==1 and t==0:
#                            print(self.un(rx,s))
#                            print(bet)
#                            print(Cmn)
#                            print(phi_a)
#                            print(phi_a_dot)
#                            print(phi)






#
#
if __name__ == '__main__':

    my_code = textwrap.dedent("""\
from geotecha.piecewise.piecewise_linear_1d import PolyLine
import numpy as np

#surcharge_vs_time = PolyLine([0,0,10], [0,100,100])
#h = np.array([10, 20, 30, 20])
#cv = np.array([0.0411, 0.1918, 0.0548, 0.0686])
#mv = np.array([3.07e-3, 1.95e-3, 9.74e-4, 1.95e-3])
##kv = np.array([7.89e-6, 2.34e-5, 3.33e-6, 8.35e-6])
#kv = cv*mv
#kh = kv * 0.0003
#r1 = 0.6
#r0= 0.05
#
#bctop = 0
#
#bcbot = 0
#
#
#nv = 15
#nh = 5
#
#
#z = np.concatenate((np.linspace(0, np.sum(h[:1]), 25, endpoint=False),
#                    np.linspace(np.sum(h[:1]), np.sum(h[:2]), 25, endpoint=False),
#                    np.linspace(np.sum(h[:2]), np.sum(h[:3]), 25, endpoint=False),
#                    np.linspace(np.sum(h[:3]), np.sum(h), 25, endpoint=True)))
#
#
#
#tpor = np.array([740, 2930, 7195], dtype=float)
#
#t = np.logspace(1, 4.5, 30)
#
#t = np.array(
#    [1.21957046e+02,   1.61026203e+02,   2.12611233e+02,
#     2.80721620e+02,   3.70651291e+02,   4.89390092e+02,
#     740.0,   8.53167852e+02,   1.12648169e+03,
#     1.48735211e+03,   1.96382800e+03,   2930.0,
#     3.42359796e+03,   4.52035366e+03,   5.96845700e+03,
#     7195.0,   1.04049831e+04,   1.37382380e+04,
#     1.81393069e+04,   2.39502662e+04,   3.16227766e+04])

surcharge_vs_time = PolyLine([0,0,10], [0,100,100])
h = np.array([0.5,0.5])
#f = 0.001
cv = np.array([1,20])
mv = np.array([1,1])

#h = np.array([1])
#cv = np.array([1])
#mv = np.array([1])

kv = cv*mv
kh = kv
#kh = np.array([1,5.7])
r0 = 0.1
r1 = 20 * r0


bctop = 0

bcbot = 1


nv = 12
nh = 6

z = np.linspace(0,np.sum(h),100)
tpor = np.array([0.0,0.1, 0.3, 1])
t = np.linspace(0,3, 50)
    """)

    a = NogamiAndLi2003(my_code)


#    a.calc()
#    a._calc_derived_properties()
#    a._find_sn()
#    print(a._sn)
#    print(a._alp_min())
#    print(a._alp)
#    steps = 200
#    x = np.linspace(a._alp_min()[2]+0.00001,a._alp_min()[2]+0.8 , 200 )
#    y = np.zeros_like(x)
#    for i,_x in enumerate(x):
#        y[i] = a._vertical_characteristic_curve(_x, a._sn[2])
#    plt.plot(x,y)
#    plt.show()

#    a._find_beta()
##    a.plot_characteristic_curve_and_roots()
#    a._make_BC(a._beta0[0])
#    a._Am_integrations()
#    a._Tm_integrations()
#    a._uavg_integrations()
#    a._debug=True

    a.calc()
    print('alp', a._alp[0,:2])
    print('bet', a._betamn[0,:2,:])
#    print(a._betamn)
#    print(a._Cmn)
#    print('hello')
#    print(a._Bmn)
#    plot_one_dim_consol(a.z, a.t, por=a.por, uavg=a.uavg, settle=a.settle)
    plot_one_dim_consol(a.z, a.tpor, por=a.por, uavg=None, settle=None)


    i = 0
    s = a._sn[0]
    amin=a._alp_min()[i]
    amin=0.001
    x = np.linspace(amin, a._alp[i,-1], 200)

    y = np.zeros_like(x)
    for j,_x in enumerate(x):

        y[j] = a._vertical_characteristic_curve(_x, s)
#        print(x[i],y[i])

    plt.figure()
    plt.plot(x, y, '-+')
    plt.plot(a._alp[i,:], np.zeros_like(a._alp[i,:]),'o')
    plt.ylim((np.min(y[-150:]),np.max(y[-150:])))
    plt.plot(a._alp_min()[i], 0, 'r^')



    plt.show()



#    print(repr(a.z))
#    print('*')
#    print(repr(a.por))
#    print('*')
#    print(repr(a.uavg))
#    print('*')
#    print(repr(a.settle))
#    print('*')
#    print(repr(a.t))
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
