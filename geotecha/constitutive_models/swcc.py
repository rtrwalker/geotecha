# geotecha - A software suite for geotechncial engineering
# Copyright (C) 2015  Rohan T. Walker (rtrwalker@gmail.com)
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

"""Some soil water characteristic curves (SWCC) for unsaturated soil"""

from __future__ import print_function, division

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy.optimize import fsolve
import sympy



class SWCC(object):
    """Base class for defining soil water characteristic curve"""

    def w_from_psi(self, psi, **kwargs):
        """Water content from suction"""
        raise NotImplementedError("w_from_psi must be implemented")

    def psi_from_w(self, w, **kwargs):
        """Suction from water content"""
        raise NotImplementedError("w_from_psi must be implemented")

    def dw_dpsi(self, psi, **kwargs):
        """Slope of SWCC dw/dpsi"""
        raise NotImplementedError("dw_dpsi must be implemented")

    def psi_and_w_for_plotting(self, **kwargs):
        """Suction and water content for plotting that plot the method"""
        # should return a tuple of x and y values
        # x-values are peremability, y-values are void ratio.
        raise NotImplementedError("psi_and_w_for_plotting must be implemented")

    def plot_model(self, **kwargs):
        """Plot the void ratio-permeability"""
        ax = kwargs.pop('ax', plt.gca())
        x, y = self.psi_and_w_for_plotting(**kwargs)

        ax.plot(x, y)
        return



class SWCC_FredlundAndXing1994(SWCC):
    """Soil water characteristic curve from Fredlund and Xing 1994

    Parameters
    ----------

    a : float
        Fitting parameter corresponding to soil suction at the inflection
        point.
    n : float
        Fitting parameter designating th rate of desaturation.
    m : float
        Third fitting parameter
    ws : float, optional
        Saturated water content. Default ws=1.0 i.e. basically a degree of
        saturation.
    psir : float, optional
        Residual suction. Default psir=1e6.
    correction : float, optional
        Manual correction factor.  Default correction=None i.e. correction
        factor is caclulated.

    References
    ----------
    .. [1] Fredlund, D.G., Anqing Xing, and Shangyan Huang.
           "Predicting the Permeability Function for Unsaturated Soils
           Using the Soil-Water Characteristic Curve."
           Canadian Geotechnical Journal 31, no. 4 (1994): 533-46.
           doi:10.1139/t94-062.


    Notes
    -----
    The equation for the soil water charcteristic curve is given by:

    .. math:: w(\\psi) = C(\\psi)
        \\frac{w_s}
            {\\left[{\\ln\\left[{\\exp(1)+
                                 \\left({\\psi/a}\\right)^n}\\right]}\\right]^m}

    where the correction factor is given by:

    .. math:: C(\\psi) = \\left[{1 - \\frac{\\ln\\left({1+\\psi/\\psi_r}\\right)}
                               {\\ln\\left({1+10^6/\\psi_r}\\right)}}\\right]


    SWCC_FredlundAndXing1994()
    """

    def __init__(self, a, n, m, ws=1.0, psir=1.0e6, correction=None):


        self.a = a
        self.n = n
        self.m = m
        self.ws = ws
        self.psir = psir
        self.correction = correction

    def w_from_psi(self, psi, **kwargs):
        """water content from suction

        Parameters
        ----------
        psi : float
            Suction

        Returns
        -------
        w : float
            Suction corresponding to psi

        Examples
        --------
        >>> a = SWCC_FredlundAndXing1994(a=3000, n=1.5, m=1, ws=60, correction=1)
        >>> a.w_from_psi(2000)
        50.73...
        >>> b = SWCC_FredlundAndXing1994(a=100, n=2, m=4, correction=1)
        >>> b.w_from_psi(90)
        0.395...
        >>> c = SWCC_FredlundAndXing1994(a=427, n=0.794, m=0.613, psir=3000)
        >>> c.w_from_psi(10000)
        0.40...


        """

        psir = self.psir
        a = self.a
        n = self.n
        m = self.m
        ws = self.ws

        if self.correction is None:
            corr = (1.0 - np.log(1 + psi / psir) /
                          np.log(1.0 + 1.0e6 / psir) )
        else:
            corr = self.correction


        return (corr * ws /
                np.log(np.exp(1.0) + (psi / a)**n)**m)


    def _derive_dw_dpsi():
        """Derive the slope dw_dpsi with sympy

        Returns
        -------
        dw_dpsi1 : sympy expression
            Derivative including correction factor.
        dw_dpsi2 : sympy expression
            Derivative with correction factor equal to one.
         """
        psi, a, n, m, psir, ws = sympy.symbols('psi, a, n, m, psir, ws')
        from sympy import log, exp
        C1 = (1 - log(1 + psi / psir) /
                          log(1.0 + 1.0e6 / psir))

        C2 = 1

        w = C1 * ws / log(exp(1) + (psi / a)**n)**m
        dw_dpsi1 = sympy.diff(w, psi)

        w = C2 * ws / log(exp(1) + (psi / a)**n)**m
        dw_dpsi2 = sympy.diff(w, psi)
        return  dw_dpsi1, dw_dpsi2

    def _numerical_check_of_dw_dpsi(**kwargs):
        """Produce plot of dw_dpsi, and check against numerical slope

        only needs to be close
        Parameters
        ----------
        ax : matplotlib.Axes object, optional
            Axes to plot on.

        """

        if 'ax' in kwargs:
            ax=kwargs['ax']
        else:
            ax=plt.gca()
        a = SWCC_FredlundAndXing1994(a=427, n=0.794, m=0.613, psir=3000)
        b = SWCC_FredlundAndXing1994(a=200, n=2, m=1, ws=40, correction=1)
        text = ('a={a:g}, n={n:g}, m={m:g}, psir={psir:g}, '
                'ws={ws:g}, correction={correction}')
        labels = [text.format(a=v.a, n=v.n, m=v.m, psir=v.psir, ws=v.ws,
                                   correction=v.correction) for v in [a, b]]


        for v, label in zip([a, b], labels):
            x, y = v.psi_and_w_for_plotting(npts=100)
            dx = np.diff(x)
            dy = np.diff(y)
            xm = x[:-1] + dx
            slope_n = dy / dx
            slope_a = v.dw_dpsi(xm)
            ax.plot(xm, slope_n, marker='o', ms=2, ls='.',
                     label="numerical: " + label)
            ax.plot(xm, slope_a,
                     label="analytical: " + label)
            plt.gca().set_xscale('log')
            ax.set_xlabel('$\\psi$')
            ax.set_ylabel('$dw/d\\psi$')
        leg = ax.legend(loc=4)
        leg.draggable()
#            print(np.isclose(slope_a, slope_n))
#            print(slope_a-slope_n)
        plt.show()

    def dw_dpsi(self, psi, **kwargs):
        """Slope of SWCC dw/dpsi

        Parameters
        ----------
        psi : float
            Suction

        Returns
        -------
        dw_dpsi : float
            Suction corresponding to psi

        Examples
        --------
        >>> a = SWCC_FredlundAndXing1994(a=3000, n=1.5, m=1, ws=60, correction=1)
        >>> a.dw_dpsi(2000)
        -0.00536...
        >>> b = SWCC_FredlundAndXing1994(a=100, n=2, m=4, correction=1)
        >>> b.dw_dpsi(90)
        -0.00640...
        >>> c = SWCC_FredlundAndXing1994(a=427, n=0.794, m=0.613, psir=3000)
        >>> c.dw_dpsi(10000)
        -1.317...e-05


        """

        psir = self.psir
        a = self.a
        n = self.n
        m = self.m
        ws = self.ws

        E = np.exp(1.0)
        log = np.log
        if self.correction is None:
            return (-m*n*ws*(psi/a)**n*(1 - log(psi/psir + 1)/
                    log(1.0 + 1000000.0/psir))*log((psi/a)**n + E)**(-m)/
                    (psi*((psi/a)**n + E)*log((psi/a)**n + E)) -
                    ws*log((psi/a)**n + E)**(-m)/(psir*(psi/psir + 1)*
                    log(1.0 + 1000000.0/psir)))
        else:
            return self.correction * (-m*n*ws*(psi/a)**n*
                log((psi/a)**n + E)**(-m)/(psi*((psi/a)**n + E)*
                log((psi/a)**n + E)))




    def psi_and_w_for_plotting(self, **kwargs):
        """Suction and water content for plotting

        Parameters
        ----------
        npts : int, optional
            Number of points to return.  Default npts=100.
        xmin, xmax : float, optional
            Range of x (i.e. effective stress) values from which
            to return points. Default xmin=1, xmax=1e6.

        Returns
        -------
        x, y : 1d ndarray
            `npts` permeability, and void ratio values between
            `xmin` and `xmax`.

        """

        npts = kwargs.get('npts', 100)
        xmin, xmax = kwargs.get('xmin', 1.0), kwargs.get('xmax', 1e6)

        x = np.logspace(np.log10(xmin), np.log10(xmax), npts)
        y = self.w_from_psi(x)
        return x, y


    def k_from_psi(self, psi, **kwargs):
        """Relative permeability from suction

        Parameters
        ---------_
        psi : float
            Soil suction. Can be a 1d array.
        aev : float, optional
            Air entry soil suction. Default aev=1.0
        npts : int, optional
            Numper of intervals to break integral into. Default npts=500.

        Returns
        -------
        k : float
            Relative permeability.


        Examples
        --------
        >>> b = SWCC_FredlundAndXing1994(a=2.77, n=11.2, m=0.45, psir=300)
        >>> b.k_from_psi(4)
        0.0520...



        Notes
        -----
        The permability relative to the saturated value is given by:

        .. math:: k_r(\\psi) =  \\left.
            \\int_{\\ln(\\psi)}^{b}
              {\\frac{\\theta(e^y) - \\theta(\\psi)}{e^y} \\theta^{\\prime}(e^y)\\,dy}
            \\middle/
              \\int_{\\ln(\\psi_{\\textrm{aev}})}^{b}
              {\\frac{\\theta(e^y) - \\theta_s}{e^y} \\theta^{\\prime}(e^y)\\,dy}
            \\right.

        where, :math:`b=\\ln(10^6)`, :math:`\\theta(\\psi)` is the volumetric
        water content at a given soil suction.  :math:`\\psi_{\\textrm{aev}}`
        is the air entry value of soil suction (must be positive for the
        log integral to work).  Each integral is performed by dividing the
        integration inteval into N sections, evaluating the integrand at the
        mid point of each inteval, then summing the areas of each section.


        If you enter a single elment array you will get a scalar returned.


        References
        ----------
        .. [1] Fredlund, D.G., Anqing Xing, and Shangyan Huang.
               "Predicting the Permeability Function for Unsaturated
               Soils Using the Soil-Water Characteristic Curve.
               Canadian Geotechnical Journal 31, no. 4 (1994): 533-46.
               doi:10.1139/t94-062.


        """

        aev = kwargs.get('aev', 1.0)
        npts = kwargs.get('npts', 500)
        psi = np.atleast_1d(psi)

        b = np.log(1.0e6) #conceivably this could be the suction at residual
        a1 = np.log(psi)
        dy1 = (b - a1) / npts

        y1 = a1[:, None] + dy1[:, None]*(np.arange(npts)[None, :] + 0.5)
        ey1 = np.exp(y1)

        f1 = (self.w_from_psi(ey1)-self.w_from_psi(psi[:,None])) * self.dw_dpsi(ey1) / ey1

        numer = np.sum(f1, axis=1)

        a2 = np.log(aev)

        dy2 = (b - a2) / npts

        y2 = a2 + dy2*(np.arange(npts) + 0.5)
        ey2 = np.exp(y2)

        f2 = (self.w_from_psi(ey2) - self.ws) * self.dw_dpsi(ey2) / ey2
#        ws_numerical = self.w_from_psi(aev)
#        f2 = (self.w_from_psi(ey2) - ws_numerical) * self.dw_dpsi(ey2) / ey2
        denom = np.sum(f2)

        k = (numer * dy1) / (denom * dy2)
        if k.size==1:
            return k[0]
        else:
            return k
        return k


class SWCC_PhamAndFredlund2008(SWCC):
    """Soil water characteristic curve from Fredlund and Xing 2008

    Be careful interpreting results when suction is less than 1.
    At low suction values we expect wc approx equal to wsat-wr, but the
    s1 * log10(psi) term gives large negative numbers for psi<<1,
    whereas at psi=1 the term dissapears as expected.

    Parameters
    ----------

    a : float
        Curve fitting parameter.
    b : float
        Curve fitting parameter.
    wr : float
        Residual gravimetric water content.
    s1 : float
        Initial slope of SWCC (i.e. Cc/Gs)
    ws : float
        Saturated water content. Default ws=1.0 i.e. basically a degree of
        saturation.
    psir : float, optional
        Residual suction. Default psir=None, i.e. psir = (2.7*a) **(1/b).
    correction : float, optional
        Manual correction factor.  Default correction=None i.e. correction
        factor is caclulated.

    References
    ----------
    .. [1] Pham, Hung Q., and Delwyn G. Fredlund. "Equations for the
           Entire Soil-Water Characteristic Curve of a Volume Change
           Soil." Canadian Geotechnical Journal 45, no. 4
           (April 1, 2008): 443-53.
           doi:10.1139/T07-117.



    Notes
    -----
    The equation for the soil water charcteristic curve is given by:

    .. math:: w(\\psi) = C(\\psi)
        \\left[{\\left({w_{sat} - S_l \\log(\\psi) -w_r}\\right)
            \\frac{a}{\\psi^{b}+a} + w_r}\\right]



    where the correction factor is given by:

    .. math:: C(\\psi) = \\left[{1 - \\frac{\\ln\\left({1+\\psi/\\psi_r}\\right)}
                               {\\ln\\left({1+10^6/\\psi_r}\\right)}}\\right]


    SWCC_PhamAndFredlund2008()
    """

    def __init__(self, ws, a, b, wr, s1, psir=None, correction=None):

        self.ws = ws
        self.a = a
        self.b = b
        self.wr = wr
        self.s1 = s1
        if psir is None:
            self.psir = (2.7*self.a)**(1/self.b)
        else:
            self.psir = psir
#        self.psir = psir
        self.correction = correction

    def w_from_psi(self, psi, **kwargs):
        """water content from suction

        Parameters
        ----------
        psi : float
            Suction

        Returns
        -------
        w : float
            Suction corresponding to psi

#        Examples
#        --------
#        >>> a = SWCC_FredlundAndXing1994(a=3000, n=1.5, m=1, ws=60, correction=1)
#        >>> a.w_from_psi(2000)
#        50.73...
#        >>> b = SWCC_FredlundAndXing1994(a=100, n=2, m=4, correction=1)
#        >>> b.w_from_psi(90)
#        0.395...
#        >>> c = SWCC_FredlundAndXing1994(a=427, n=0.794, m=0.613, psir=3000)
#        >>> c.w_from_psi(10000)
#        0.40...


        """

        psir = self.psir
        ws = self.ws
        a = self.a
        b = self.b
        wr = self.wr
        s1 = self.s1


        if self.correction is None:
            corr = (1.0 - np.log(1. + psi / psir) /
                          np.log(1.0 + 1.0e6 / psir) )
        else:
            corr = self.correction


        return (corr * ((ws - s1 * np.log10(psi) - wr) *
                    a / (psi**b + a) + wr))


    def _derive_dw_dpsi():
        """Derive the slope dw_dpsi with sympy

        Returns
        -------
        dw_dpsi1 : sympy expression
            Derivative including correction factor.
        dw_dpsi2 : sympy expression
            Derivative with correction factor equal to one.
         """
        psi, a, n, m, psir, ws = sympy.symbols('psi, a, n, m, psir, ws')
        psi, ws, a, b, wr, s1, psir = sympy.symbols('psi, ws, a, b, wr, s1, psir')
        from sympy import log, exp
        C1 = (1 - log(1 + psi / psir) /
                          log(1.0 + 1.0e6 / psir))

        C2 = 1

        l10 = sympy.symbols('l10')

        w = C1 * ((ws - s1 * log(psi)/l10 - wr) * a / (psi**b + a) + wr)
        dw_dpsi1 = sympy.diff(w, psi)

        w = C2 * ((ws - s1 * log(psi)/l10 - wr) * a / (psi**b + a) + wr)
        dw_dpsi2 = sympy.diff(w, psi)
        return  dw_dpsi1, dw_dpsi2

    def _numerical_check_of_dw_dpsi(**kwargs):
        """Produce plot of dw_dpsi, and check against numerical slope

        only needs to be close
        Parameters
        ----------
        ax : matplotlib.Axes object, optional
            Axes to plot on.

        """

        if 'ax' in kwargs:
            ax=kwargs['ax']
        else:
            ax=plt.gca()
        a = SWCC_PhamAndFredlund2008(a = 3.1e6,
                                     b = 3.377,
                                     wr = 0.128,
                                     s1 = 0.115,
                                     ws=0.262,
                                     psir=3000)
        b = SWCC_PhamAndFredlund2008(
                                     a = 1.1e6,
                                     b = 2.,
                                     wr = 0.128,
                                     s1 = 0.1,
                                     ws=0.262,
                                     correction=1)
        text = ('ws={ws:g}, a={a:g}, b={b:g}, wr={wr:g}, s1={s1:g}, '
                'psir={psir:g}, correction={correction}')
        labels = [text.format(ws=v.ws, a=v.a, b=v.b, wr=v.wr,
                              s1=v.s1, psir=v.psir,
                                   correction=v.correction) for v in [a, b]]


        for v, label in zip([a, b], labels):
            x, y = v.psi_and_w_for_plotting(npts=100)
            dx = np.diff(x)
            dy = np.diff(y)
            xm = x[:-1] + dx
            slope_n = dy / dx
            slope_a = v.dw_dpsi(xm)
            ax.plot(xm, slope_n, marker='o', ms=2, ls='.',
                     label="numerical: " + label)
            ax.plot(xm, slope_a,
                     label="analytical: " + label)
            plt.gca().set_xscale('log')
            ax.set_xlabel('$\\psi$')
            ax.set_ylabel('$dw/d\\psi$')
        leg = ax.legend(loc=4)
        leg.draggable()
#            print(np.isclose(slope_a, slope_n))
#            print(slope_a-slope_n)
        plt.show()

    def dw_dpsi(self, psi, **kwargs):
        """Slope of SWCC dw/dpsi

        Parameters
        ----------
        psi : float
            Suction

        Returns
        -------
        dw_dpsi : float
            Suction corresponding to psi

#        Examples
#        --------
#        >>> a = SWCC_FredlundAndXing1994(a=3000, n=1.5, m=1, ws=60, correction=1)
#        >>> a.dw_dpsi(2000)
#        -0.00536...
#        >>> b = SWCC_FredlundAndXing1994(a=100, n=2, m=4, correction=1)
#        >>> b.dw_dpsi(90)
#        -0.00640...
#        >>> c = SWCC_FredlundAndXing1994(a=427, n=0.794, m=0.613, psir=3000)
#        >>> c.dw_dpsi(10000)
#        -1.317...e-05


        """

        psir = self.psir
        ws = self.ws
        a = self.a
        b = self.b
        wr = self.wr
        s1 = self.s1

        E = np.exp(1.0)
        log = np.log
        l10 = np.log(10.)
        if self.correction is None:
            return ((1 - log(psi/psir + 1)/log(1.0 + 1000000.0/psir))*
             (-a*b*psi**b*(-wr + ws - s1*log(psi)/l10)/(psi*(a + psi**b)**2)
             - a*s1/(l10*psi*(a + psi**b))) - (a*(-wr + ws -
             s1*log(psi)/l10)/(a + psi**b) + wr)/(psir*(psi/psir + 1)*
             log(1.0 + 1000000.0/psir)))
        else:
            return self.correction * (-a*b*psi**b*(-wr + ws -
                s1*log(psi)/l10)/(psi*(a + psi**b)**2) -
                a*s1/(l10*psi*(a + psi**b)))




    def psi_and_w_for_plotting(self, **kwargs):
        """Suction and water content for plotting

        Parameters
        ----------
        npts : int, optional
            Number of points to return.  Default npts=100.
        xmin, xmax : float, optional
            Range of x (i.e. effective stress) values from which
            to return points. Default xmin=1, xmax=1e6.

        Returns
        -------
        x, y : 1d ndarray
            `npts` permeability, and void ratio values between
            `xmin` and `xmax`.

        """

        npts = kwargs.get('npts', 100)
        xmin, xmax = kwargs.get('xmin', 1.0), kwargs.get('xmax', 1e6)

        x = np.logspace(np.log10(xmin), np.log10(xmax), npts)
        y = self.w_from_psi(x)
        return x, y


    def k_from_psi(self, psi, **kwargs):
        """Relative permeability from suction

        If k_from_psi returns a number greater than zero then try reducing
        aev.

        Paramters
        ---------
        psi : float
            Soil suction. Can be a 1d array.
        aev : float, optional
            Air entry soil suction. Default aev=0.001
        npts : int, optional
            Numper of intervals to break integral into. Default npts=500.

        Returns
        -------
        k : float
            Relative permeability.


#        Examples
#        --------
#        >>> b = SWCC_FredlundAndXing1994(a=2.77, n=11.2, m=0.45, psir=300)
#        >>> b.k_from_psi(4)
#        0.0520...



        Notes
        -----
        The permability relative to the saturated value is given by:

        .. math:: k_r(\\psi) =  \\left.
            \\int_{\\ln(\\psi)}^{b}
              {\\frac{\\theta(e^y) - \\theta(\\psi)}{e^y} \\theta^{\\prime}(e^y)\\,dy}
            \\middle/
              \\int_{\\ln(\\psi_{\\textrm{aev}})}^{b}
              {\\frac{\\theta(e^y) - \\theta_s}{e^y} \\theta^{\\prime}(e^y)\\,dy}
            \\right.

        where, :math:`b=\\ln(10^6)`, :math:`\\theta(\\psi)` is the volumetric
        water content at a given soil suction.  :math:`\\psi_{\\textrm{aev}}`
        is the air entry value of soil suction (must be positive for the
        log integral to work).  Each integral is performed by dividing the
        integration inteval into N sections, evaluating the integrand at the
        mid point of each inteval, then summing the areas of each section.


        If you enter a single element array you will get a scalar returned.


        References
        ----------
        .. [1] Fredlund, D.G., Anqing Xing, and Shangyan Huang.
               "Predicting the Permeability Function for Unsaturated
               Soils Using the Soil-Water Characteristic Curve.
               Canadian Geotechnical Journal 31, no. 4 (1994): 533-46.
               doi:10.1139/t94-062.


        """

        aev = kwargs.get('aev', 0.001)
        npts = kwargs.get('npts', 500)
        psi = np.atleast_1d(psi)

        b = np.log(1.0e6) #conceivably this could be the suction at residual
        a1 = np.log(psi)
        dy1 = (b - a1) / npts

        y1 = a1[:, None] + dy1[:, None]*(np.arange(npts)[None, :] + 0.5)
        ey1 = np.exp(y1)

        f1 = (self.w_from_psi(ey1)-self.w_from_psi(psi[:,None])) * self.dw_dpsi(ey1) / ey1

        numer = np.sum(f1, axis=1)

        a2 = np.log(aev)

        dy2 = (b - a2) / npts

        y2 = a2 + dy2*(np.arange(npts) + 0.5)
        ey2 = np.exp(y2)

#        f2 = (self.w_from_psi(ey2) - self.ws) * self.dw_dpsi(ey2) / ey2
        ws_numerical = self.w_from_psi(aev)
        f2 = (self.w_from_psi(ey2) - ws_numerical) * self.dw_dpsi(ey2) / ey2
        denom = np.sum(f2)

        k = (numer*dy1) / (denom*dy2)
        if k.size==1:
            return k[0]
        else:
            return k
        return k



def kwrel_from_discrete_swcc(psi, psi_swcc, vw_swcc):
    """Relative permeability from integrating discrete soil water
    characteristic curve.

    Parameters
    ----------
    psi: float
        Suction values to calculate relative permeability at. Suction is
        positve.
    psi_swcc : 1d array of float
        Suction values defining soil water charteristic curve.
    vw_swcc : 1d array of float
        Volumetric water content corresponding to psi_swcc


    Returns
    -------
    krel : 1d array of float
        relative permeability values corresponding to psi.

    Notes
    -----
    This integrates the whole SWCC.  Initial slope of SWCC is non-zero then
    peremability reduction with suction will commence straight from first
    value of psi_swcc.

    Uses _[1] method but does it assuming taht you already have all
    the SWCC points (_[1] can calculate volumetric water content from suction).
    Main differnce is that in _[1] vw and dvw/dpsi at the midpoint are
    calculated analytically, whereas here midpoint value is half the
    segment endpoint values, and slope is approx value at segmetn endpoint.

    Will return scalar if scalar or single elemtn array is input.

    If you get some sort of index error then probably your pis input is
    beyond your data range.

    Examples
    --------
    >>> b = SWCC_FredlundAndXing1994(a=2.77, n=11.2, m=0.45, psir=300)
    >>> x = np.logspace(-3,6,500)
    >>> y = b.w_from_psi(x)
    >>> kwrel_from_discrete_swcc(4, x, y)
    0.0550...



    References
    ----------
    .. [1] Fredlund, D.G., Anqing Xing, and Shangyan Huang.
           "Predicting the Permeability Function for Unsaturated
           Soils Using the Soil-Water Characteristic Curve.
           Canadian Geotechnical Journal 31, no. 4 (1994): 533-46.
           doi:10.1139/t94-062.
    """




    dlogx = np.log10(psi_swcc[1:]) - np.log10(psi_swcc[:-1]) #interval in log10-space

    xbar = psi_swcc[:-1]*np.power(10, 0.5*dlogx) # x value at midpoint
    ybar = 0.5 * (vw_swcc[:-1] + vw_swcc[1:]) # y value at interval
    logslope = (vw_swcc[1:] - vw_swcc[:-1]) / dlogx #interval slope in log10-space
    slopebar = logslope/np.log(10)/xbar #slope at interval midpoint in normal space

    idx = np.searchsorted(psi_swcc, psi, side="right")-1
    # if index error then chances are

    #TODO: limit what values of idx, to be within range
    f1 = dlogx[:,None] * (ybar[:,None] - ybar[None,idx]) / xbar[:,None] * slopebar[:,None]
    #note in f1 each row is an interval, each column is a psi value

    row, col = np.indices(f1.shape)

    f1[row<idx] = 0.0 #ignore values before psi

    numer = np.sum(f1, axis=0)

    denom = np.sum(dlogx[:]*(ybar[:] - ybar[0])/xbar[:] * slopebar[:])

    krel = numer / denom


    if krel.size==1:
        return krel[0]
    else:
        return krel


def karel_air_from_saturation(Sr, qfit=0.5):
    """Relative air peremability (w.r.t. dry_ka)

    krel = (1-Sr)**0.5*(1-Sr**(1 / qfit))**(2*qfit)

    Parameters
    ----------
    Sr : 1d array of float
        Degree of saturation at which to calc relative permeability of air.
    qfit : float, optional
        Fitting parameter.  Generally between between 0 and 1.
        Default qfit=1

    References
    ----------
    .. [1] Ba-Te, B., Limin Zhang, and Delwyn G. Fredlund. "A General
           Air-Phase Permeability Function for Airflow through Unsaturated
           Soils." In Proceedings of Geofrontiers 2005 Cngress,
           2961-85. Austin, Tx: ASCE, 2005. doi:10.1061/40787(166)29.

    """


    return (1 - Sr)**0.5*(1 - Sr**(1 / qfit))**(2 * qfit)



if __name__ == "__main__":
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest', '--doctest-options=+ELLIPSIS'])

    if 0:
        a = SWCC_FredlundAndXing1994(a=3000, n=1.5, m=1, ws=60, correction=1)

        b = SWCC_FredlundAndXing1994(a=200, n=2, m=1, ws=40, correction=1)
        c = SWCC_FredlundAndXing1994(a=10, n=4, m=1, ws=30, correction=1)

        [v.plot_model() for v in [a, b, c]]
        plt.gca().set_xscale('log')
        plt.gca().set_ylim([0,70])
        plt.gca().grid(which='major', linestyle='-')
        plt.gca().grid(which='minor', linestyle='--')
        plt.show()

    if 0:
        b = SWCC_FredlundAndXing1994(a=100, n=2, m=4)
        b = SWCC_FredlundAndXing1994(a=427, n=0.794, m=0.613, psir=3000)
        b.plot_model()
        plt.gca().set_xscale('log')
#        plt.gca().set_ylim([0,70])
        plt.gca().grid(which='major', linestyle='-')
        plt.gca().grid(which='minor', linestyle='--')
        plt.show()
    if 0:
        a, b=SWCC_FredlundAndXing1994._derive_dw_dpsi()

        print(a)
        print(b)
    if 0:
        SWCC_FredlundAndXing1994._numerical_check_of_dw_dpsi()
#        b = SWCC_FredlundAndXing1994(a=427, n=0.794, m=0.613, psir=3000)
#        b = SWCC_FredlundAndXing1994(a=200, n=2, m=1, ws=40, correction=1)
#        x, y = b.psi_and_w_for_plotting(npts=10000)
#        dx = np.diff(x)
#        dy = np.diff(y)
#        xm = x[:-1] + dx
#        slope_n = dy / dx
#        slope_a = b.dw_dpsi(xm)
#        plt.plot(xm, slope_n, marker='o', ms=2, ls='.')
#        plt.plot(xm, slope_a, color='red')
#        plt.gca().set_xscale('log')
#        print(np.isclose(slope_a, slope_n))
#        print(slope_a-slope_n)
#        plt.show()


    if 0:

#        b = SWCC_FredlundAndXing1994(a=427, n=0.794, m=0.613, psir=3000)
#        b = SWCC_FredlundAndXing1994(a=8.34, n=9.9, m=0.44, psir=30)
#         b = SWCC_FredlundAndXing1994(a=6.01, n=11.86, m=0.36, psir=30)
        b = SWCC_FredlundAndXing1994(a=2.77, n=11.2, m=0.45, psir=300)
#        b = SWCC_FredlundAndXing1994(a=2.7, n=2.05, m=0.36, psir=100)



#        b = SWCC_FredlundAndXing1994(a=200, n=2, m=1, ws=40, correction=1)

        x, y = b.psi_and_w_for_plotting(npts=100)

        k = b.k_from_psi(x, aev=1)

        fig = plt.figure()
        ax = fig.add_subplot('211')
        ax.plot(x, y)
        ax.set_xlabel('$psi$')
        ax.set_ylabel('w')
        ax.set_xscale('log')
        ax.grid(which='major', linestyle='-')
        ax.grid(which='minor', linestyle='--')

        ax = fig.add_subplot('212', sharex=ax)
        ax.plot(x, k)
        ax.set_xlabel('$psi$')
        ax.set_ylabel('$k/k_{sat}$')
        ax.set_yscale('log')
        ax.set_ylim((0.001, 1))
        ax.set_ylim((0.001, 10))
        ax.grid(which='major', linestyle='-')
        ax.grid(which='minor', linestyle='--')


        plt.show()
    if 0:

        a, b=SWCC_PhamAndFredlund2008._derive_dw_dpsi()

        print(a)
        print()
        print(b)
    if 0:
        SWCC_PhamAndFredlund2008._numerical_check_of_dw_dpsi()


    if 0:

        b = SWCC_PhamAndFredlund2008(a = 50000.,
                                     b = 2.,
                                     wr = 0.12,
                                     s1 = 0.04,
                                     ws=0.5,
                                     correction=None)
#        b = SWCC_PhamAndFredlund2008(a = 25000.,
#                                     b = 3.,
#                                     wr = 0.12,
#                                     s1 = 0.04,
#                                     ws=0.5,
#                                     correction=None)
#        b = SWCC_PhamAndFredlund2008(a = 50000.,
#                                     b = 1.5,
#                                     wr = 0.12,
#                                     s1 = 0.04,
#                                     ws=0.5,
#                                     correction=None)



        x, y = b.psi_and_w_for_plotting(npts=100)

        k = b.k_from_psi(x, aev=1)

        fig = plt.figure()
        ax = fig.add_subplot('211')
        ax.plot(x, y)
        ax.set_xlabel('$psi$')
        ax.set_ylabel('w')
        ax.set_xscale('log')
        ax.grid(which='major', linestyle='-')
        ax.grid(which='minor', linestyle='--')

        ax = fig.add_subplot('212', sharex=ax)
        ax.plot(x, k)
        ax.set_xlabel('$psi$')
        ax.set_ylabel('$k/k_{sat}$')
        ax.set_yscale('log')
        ax.set_ylim((0.001, 1))
        ax.set_ylim((0.001, 10))
        ax.grid(which='major', linestyle='-')
        ax.grid(which='minor', linestyle='--')


        plt.show()

    if 1:
        b = SWCC_FredlundAndXing1994(a=2.77, n=11.2, m=0.45, psir=300)


        x = np.logspace(-3,6, 500)
        y = b.w_from_psi(x)
        x2 = x[0:-1]*10**(0.5*(np.log10(x[1]) - np.log10(x[0])))
#        print(x)
#        print(x2)
        k_obj = b.k_from_psi(x)


        np.logspace
        krel = kwrel_from_discrete_swcc(x2[:-2], x, y)
#        krel = kwrel_from_discrete_swcc(x[:-1], x, y)

        fig, ax = plt.subplots()

        ax.plot(x, k_obj, label="expected")
        ax.plot(x2[:-2], krel, label="numerical", ls="None", marker='o')
#        ax.plot(x[:-1], krel, label="numerical", ls="None", marker='o')
        ax.set_xscale('log')
        ax.set_xlabel('psi')
        ax.set_ylabel('krel')
        plt.show()

