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
module for one-dimensional constitutive models
"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib



def CcCr_e_from_stresses(estress, pstress, Cc, Cr, siga, ea):
    """void ratio from from stress for CcCr soil model

    Parameters
    ----------
    estress : float
        current effective stress
    pstress : float
        current preconsolidation stress
    Cc : float
        compressibility index
    Cr : float
        recompression index
    siga, ea : float
        point on compression line fixing it in effective stress-void ratio
        space

    Returns
    -------
    e : float
        void ratio corresponding to current stress state

    Examples
    --------
    On recompression line:
    >>> CcCr_e_from_stresses(40, 50, 3, 0.5, 10, 5)
    2.95154...

    On compression line:
    >>> CcCr_e_from_stresses(60, 50, 3, 0.5, 10, 5)
    2.66554...

    Array inputs:
    >>> CcCr_e_from_stresses(np.array([40, 60]), np.array([50, 55]),
    ... 3, 0.5, 10, 5)
    array([ 2.95154499,  2.66554625])

    """


    max_past = np.maximum(pstress, estress)

    # void ratio at preconsolidation pressure
    e = ea - Cc * np.log10(max_past / siga)
    # void ratio at current effetive stress
    e += Cr * np.log10(max_past / estress)

    return e


def CcCr_estress_from_e(e, pstress, Cc, Cr, siga, ea):
    """void ratio from from stress for CcCr soil model

    Parameters
    ----------
    e : float
        current void ratio
    pstress : float
        current preconsolidation stress
    Cc : float
        compressibility index
    Cr : float
        recompression index
    siga, ea : float
        point on compression line fixing it in effective stress-void ratio
        space

    Returns
    -------
    estress : float
        effective stress corresponding to current void ratio

    Examples
    --------
    On recompression line:
    >>> CcCr_estress_from_e(2.95154499, 50, 3, 0.5, 10, 5)
    40.0...

    On compression line:
    >>> CcCr_estress_from_e(2.66554625, 50, 3, 0.5, 10, 5)
    59.999...

    Array inputs:
    >>> CcCr_estress_from_e(np.array([ 2.95154499,  2.66554625]), 50,
    ... 3, 0.5, 10, 5)
    array([ 40.0...,  59.99...])

    """

    # void ratio at preconsolidation pressure
    ep = ea - Cc * np.log10(pstress / siga)

    dpCc = pstress * (10.0**((ep - e) / Cc) - 1.0)
    dpCr = pstress * (10.0**((ep - e) / Cr) - 1.0)
    estress = pstress + np.minimum(dpCc, dpCr)

    return estress


def CcCr_av_from_stresses(estress, pstress, Cc, Cr, siga, ea):
    """av from from stress for CcCr soil model

    Parameters
    ----------
    estress : float
        current effective stress
    pstress : float
        current preconsolidation stress
    Cc : float
        compressibility index
    Cr : float
        recompression index
    siga, ea : float
        point on compression line fixing it in effective stress-void ratio
        space

    Returns
    -------
    av : float
        slope of void-ratio vs effective stress plot at current stress state

    Examples
    --------
    On recompression line:
    >>> CcCr_av_from_stresses(40, 50, 3, 0.5, 10, 5)
    0.00542868...

    On compression line:
    >>> CcCr_av_from_stresses(60, 50, 3, 0.5, 10, 5)
    0.02171472...

    Array inputs:
    >>> CcCr_av_from_stresses(np.array([40, 60]), np.array([50, 55]),
    ... 3, 0.5, 10, 5)
    array([ 0.00542868,  0.02171472])

    """

    chooser = np.array((Cc, Cc, Cr), dtype=float)

    Cx = chooser[np.sign(estress-pstress)]

    av = 0.43429448190325182 * Cx / estress

    return av

def av_e_from_stresses(estress, av, siga, ea):
    """void ratio from from stress for av soil model

    Parameters
    ----------
    estress : float
        current effective stress
    av : float
        slope of compression line
    siga, ea : float
        effective stress and void ratio specifying point on compression
        line fixing

    Returns
    -------
    e : float
        void ratio corresponding to current stress state

    Examples
    --------
    >>> av_e_from_stresses(20, 1.5, 19, 4)
    2.5

    Array inputs:
    >>> av_e_from_stresses(np.array([20, 21]), 1.5, 19, 4)
    array([ 2.5,  1. ])
    """

    e = ea - av * (estress - siga)

    return e


def av_estress_from_e(e, av, siga, ea):
    """effective stress from void ratio for av soil model

    Parameters
    ----------
    e : float
        current void ratio
    av : float
        slope of compression line
    siga, ea : float
        effective stress and void ratio specifying point on compression
        line fixing

    Returns
    -------
    estress : float
        effective stress corresponding to current void ratio

    Examples
    --------
    >>> av_estress_from_e(1, 1.5, 19, 4)
    21.0

    Array inputs:
    >>> av_estress_from_e(np.array([1, 2.5]), 1.5, 19, 4)
    array([ 21.,  20.])

    """

    sig = siga + (ea - e) / av

    return  sig

def ck_k_from_e(e, ck, ka, ea):
    """permeability from void ratio for ck peremability model

    Parameters
    ----------
    e : float
        current void ratio
    ck : float
        slope of semi-log void-ratio vs permeability line
    ka, ea : float
        peremability and void ratio of point specifying point on permeability
        line.

    Returns
    -------
    k : float
        permeability corresponding to current void ratio

    Examples
    --------
    >>> ck_k_from_e(1, 3, 10,3)
    2.15443...

    Array inputs
    >>> ck_k_from_e(np.array([1, 1.5]), 3, 10,3)
    array([ 2.15443469,  3.16227766])

    """


    k = ka * 10.0**((e - ea) / ck)
    return k


class OneDimensionalVoidRatioEffectiveStress(object):
    """Base class for defining 1D void ratio-effective stress relationships"""

    def e_from_stress(self, estress, **kwargs):
        """void ratio from effective stress"""
        raise NotImplementedError("e_from_stress must be implemented")

    def stress_from_e(self, e, **kwargs):
        """effective stress from void ratio"""
        raise NotImplementedError("stress_from_e must be implemented")

    def e_and_stress_for_plotting(self, **kwargs):
        """void ratio and stress values that plot the method"""
        # should return a tuple of x and y values
        raise NotImplementedError("e_and_stress_for_plotting must be implemented")

    def av_from_stress(self, estress, **kwargs):
        """slope of void ratio from effective stress"""
        raise NotImplementedError("av_from_stress must be implemented")

    def plot_constitutive_model(self, **kwargs):
        """plot the void ratio-stress points"""
        ax = kwargs.pop('ax', plt.gca())
        x, y = self.e_and_stress_for_plotting(**kwargs)

        ax.plot(x, y)
        return

class AVSoilModel(OneDimensionalVoidRatioEffectiveStress):
    """linear void ratio-effective stress realationship

    Parameters
    ----------
    av : float
        slope of compression line
    siga, ea : float
        effective stress and void ratio specifying point on av line

    """

    def __init__(self, av, siga, ea):
        self.av = av
        self.siga = siga
        self.ea = ea

    def e_from_stress(self, estress, **kwargs):
        """void ratio from effective stress

        Parameters
        ----------
        estress : float
            current effective stress

        Returns
        -------
        e : float
            void ratio corresponding to current stress state

        Examples
        --------
        >>> a = AVSoilModel(av=1.5, siga=19, ea=4)
        >>> a.e_from_stress(20)
        2.5

        Array inputs:
        >>> a = AVSoilModel(av=1.5, siga=19, ea=4)
        >>> a.e_from_stress(np.array([20, 21]))
        array([ 2.5,  1. ])

        """

        return self.ea - self.av * (estress - self.siga)

    def stress_from_e(self, e, **kwargs):
        """effective stress from void ratio

        Parameters
        ----------
        e : float
            current void ratio

        Returns
        -------
        estress : float
            effective stress corresponding to current void ratio

        Examples
        --------
        >>> a = AVSoilModel(av=1.5, siga=19, ea=4)
        >>> a.stress_from_e(1)
        21.0

        Array inputs:
        >>> a = AVSoilModel(av=1.5, siga=19, ea=4)
        >>> a.stress_from_e(np.array([1, 2.5]))
        array([ 21.,  20.])

        """

        estress = self.siga + (self.ea - e) / self.av

        return  estress

    def e_and_stress_for_plotting(self, **kwargs):
        """void ratio and stress values that plot the model

        Parameters
        ----------
        npts : int, optional
            number of points to return.  Default npts=100
        xmin, ymin : float, optional
            range of x (i.e. effective stress) values from which
            to return points. Default xmin, ymin=1, 100

        Returns
        -------
        x, y : 1d ndarray
            `npts` stress, and void ratio values between `xmin` and `ymin`

        """

        npts = kwargs.get('n', 100)
        xmin, xmax = kwargs.get('xmin', 1.0), kwargs.get('xmax', 100)

        x = np.linspace(xmin, xmax, npts)
        y = self.e_from_stress(x)
        return x, y

    def av_from_stress(self, *args, **kwargs):
        """slope of void ratio from effective stress"""
        return self.av


class CcCrSoilModel(OneDimensionalVoidRatioEffectiveStress):
    """semi-log void ratio-effective stress realationship

    Parameters
    ----------
    Cc : float
        compressibility index, slope of e-log(sig) line
    Cr : float
        recompression index, slope of e-log(sig) line
    siga, ea : float
        point on compression line fixing it in effective stress-void ratio
        space

    """

    def __init__(self, Cc, Cr, siga, ea):
        self.Cc = Cc
        self.Cr = Cr
        self.siga = siga
        self.ea = ea

    def e_from_stress(self, estress, **kwargs):
        """void ratio from effective stress

        Parameters
        ----------
        estress : float
            current effective stress
        pstress : float, optional
            reconsolidation stress.  Default pstress=estress i.e. normally
            consolidated.

        Returns
        -------
        e : float
            void ratio corresponding to current stress state

        Examples
        --------
        On recompression line:
        >>> a = CcCrSoilModel(Cc=3, Cr=0.5, siga=10, ea=5)
        >>> a.e_from_stress(estress=40, pstress=50)
        2.95154...

        On compression line:
        >>> a = CcCrSoilModel(Cc=3, Cr=0.5, siga=10, ea=5)
        >>> a.e_from_stress(estress=60, pstress=50)
        2.66554...

        Array inputs:
        >>> a = CcCrSoilModel(Cc=3, Cr=0.5, siga=10, ea=5)
        >>> a.e_from_stress(estress=np.array([40, 60]),
        ... pstress=np.array([50, 55]))
        array([ 2.95154499,  2.66554625])

        Normally consolidated (pstress not specified):
        >>> a = CcCrSoilModel(Cc=3.0, Cr=0.5, siga=10, ea=5)
        >>> a.e_from_stress(estress=10)
        5.0

        """


        pstress = kwargs.get('pstress', estress)
        ea = self.ea
        Cc = self.Cc
        Cr = self.Cr
        siga = self.siga

        max_past = np.maximum(pstress, estress)

        # void ratio at preconsolidation pressure
        e = ea - Cc * np.log10(max_past / siga)
        # void ratio at current effetive stress
        e += Cr * np.log10(max_past / estress)


        return e

    def stress_from_e(self, e, **kwargs):
        """effective stress from void ratio

        Parameters
        ----------
        e : float
            current void ratio
        pstress : float, optional
            reconsolidation stress.  Default pstress=estress i.e. normally
            consolidated.

        Returns
        -------
        estress : float
            effective stress corresponding to current void ratio

        Examples
        --------
        On recompression line:
        >>> a = CcCrSoilModel(Cc=3.0, Cr=0.5, siga=10, ea=5)
        >>> a.stress_from_e(e=2.95154499, pstress=50)
        40...

        On compression line:
        >>> a = CcCrSoilModel(Cc=3, Cr=0.5, siga=10, ea=5)
        >>> a.stress_from_e(e=2.66554625, pstress=50)
        59.999...

        Array inputs:
        >>> a = CcCrSoilModel(Cc=3, Cr=0.5, siga=10, ea=5)
        >>> a.stress_from_e(e=np.array([ 2.95154499,  2.66554625]), pstress=50)
        array([ 40.0...,  59.99...])

        Normally consolidated:
        >>> a = CcCrSoilModel(Cc=3.0, Cr=0.5, siga=10, ea=5)
        >>> a.stress_from_e(e=5)
        10.0

        """

        pstress = kwargs.get('pstress', None)
        ea = self.ea
        Cc = self.Cc
        Cr = self.Cr
        siga = self.siga

        fact = 2.3025850929940459 # 10**(x)==exp(fact*x)


        if pstress is None:
            # Normally consolidated
            estress = siga * np.exp(fact * (ea - e) / Cc)
            return estress

        # void ratio at preconsolidation pressure
        ep = ea - Cc * np.log10(pstress / siga)

        # stress change from (pstress, ep) if on extended Cc line
        dpCc = pstress * (np.exp(fact * (ep - e) / Cc) - 1.0)
        # stress change from (pstress, ep) if on extended Cr line
        dpCr = pstress * (np.exp(fact * (ep - e) / Cr) - 1.0)

        estress = pstress + np.minimum(dpCc, dpCr)

        return  estress

    def e_and_stress_for_plotting(self, **kwargs):
        """void ratio and stress values that plot the model

        Parameters
        ----------
        pstress : float, optional
            reconsolidation stress.  Default behaviour is normally
            consolidated.
        npts : int, optional
            number of points to return.  Default npts=100
        xmin, ymin : float, optional
            range of x (i.e. effective stress) values from which
            to return points. Default xmin, ymin=1, 100


        Returns
        -------
        x, y : 1d ndarray
            `npts` stress, and void ratio values between `xmin` and `ymin`

        """

        npts = kwargs.get('n', 100)
        xmin, xmax = kwargs.get('xmin', 1.0), kwargs.get('xmax', 100)

        x = np.linspace(xmin, xmax, npts)
        pstress = kwargs.get('pstress', x)
        i = np.searchsorted(x, pstress)
        x[i] = pstress
        y = self.e_from_stress(x, pstress=pstress)

        return x, y


    def av_from_stress(self, estress, **kwargs):
        """slope of void ratio from effective stress

        Parameters
        ----------
        estress : float
            current effective stress
        pstress : float, optional
            reconsolidation stress.  Default pstress=estress i.e. normally
            consolidated.

        Returns
        -------
        av : float
            slope of void-ratio vs effective stress plot at current stress
            state

        Examples
        --------
        On recompression line:
        >>> a = CcCrSoilModel(Cc=3.0, Cr=0.5, siga=10, ea=5)
        >>> a.av_from_stress(estress=40, pstress=50)
        0.00542868...

        On compression line:
        >>> a = CcCrSoilModel(Cc=3.0, Cr=0.5, siga=10, ea=5)
        >>> a.av_from_stress(estress=60, pstress=50)
        0.02171472...

        Array inputs:
        >>> a = CcCrSoilModel(Cc=3.0, Cr=0.5, siga=10, ea=5)
        >>> a.av_from_stress(estress=np.array([40, 60.0]),
        ... pstress=np.array([50, 55.0]))
        array([ 0.00542868,  0.02171472])

        """

        pstress = kwargs.get('pstress', estress)

        ea = self.ea
        Cc = self.Cc
        Cr = self.Cr

        chooser = np.array((Cc, Cc, Cr), dtype=float)

        # appropriate value of Cc or Cr
        Cx = chooser[np.sign(estress - pstress, dtype=int)]

        av = 0.43429448190325182 * Cx / estress

        return av


class PwiseLinearSoilModel(OneDimensionalVoidRatioEffectiveStress):
    """Pwise linear void ratio-effective stress realationship

    x and y data can be interpolated natural-natural, natural-log10,
    log10-natural, or log10, log10

    Parameters
    ----------
    siga, ea : 1d array
        effective stress values and void ratio values defining a one-to-one
        relationship.  Slope of void ratio-effectinve stress plot should never
        fall below `Cr`
    Cr : float
        recompression index, slope of void rato-effective stress line.  Note
        that `Cr` is the slope in whatever scales of slog and ylog that have
        been chosen.
    xlog, ylog : True/False, Optional
        If True then interpolation on each axis is assumed to be logarithmic
        with base 10.
        Default=False

    """

    def __init__(self, siga, ea, Cr, xlog=False, ylog=False, xbase=10, ybase=10):
        self.siga = np.asarray(siga, dtype=float)
        self.ea = np.asarray(ea, dtype=float)
        self.xlog = xlog
        self.ylog = ylog
        self.Cr = Cr

        if np.any(np.diff(self.siga) <= 0):
            raise ValueError("'siga' must be in monotonically increasing order.")

        if np.any(np.diff(self.ea) >= 0):
            raise ValueError("'ea' must be in monotomically decreasng order.")

        if len(siga)!=len(ea):
            raise IndexError("'siga' and 'ea' must be the same length.")

        self.log_siga = np.log10(self.siga)
        self.log_ea = np.log10(self.ea)

    def e_from_stress(self, estress, **kwargs):
        """void ratio from effective stress

        Parameters
        ----------
        estress : float
            current effective stress.  etress must be within the range of the
            soil model points.
        pstress : float, optional
            reconsolidation stress.  Default pstress=estress i.e. normally
            consolidated. pstress must be in the range of the soil model
            points.

        Returns
        -------
        e : float
            void ratio corresponding to current stress state

        Examples
        --------
        On recompression line:
        >>> x, y = np.array([1,2,2.5]), np.array([4, 3.5, 2])
        >>> a = PwiseLinearSoilModel(siga=x, ea=y, Cr=0.2)
        >>> a.e_from_stress(estress=1.25, pstress=2.25)
        2.95...

        On compression line:
        >>> a.e_from_stress(estress=2.25, pstress=2)
        2.75...

        Normally consolidated (pstress not specified):
        >>> a.e_from_stress(estress=2.25)
        2.75...

        Array inputs:
        >>> a.e_from_stress(estress=np.array([1.25, 2.25]), pstress=2.25)
        array([ 2.95,  2.75])


        Logarithmic effective stress scale:
        On recompression line:
        >>> x, y = np.array([1,2,2.5]), np.array([4, 3.5, 2])
        >>> a = PwiseLinearSoilModel(siga=x, ea=y, Cr=0.2, xlog=True)
        >>> a.e_from_stress(estress=1.25, pstress=2.25)
        2.75930...

        On compression line:
        >>> a.e_from_stress(estress=2.25, pstress=2)
        2.70824...

        Normally consolidated (pstress not specified):
        >>> a.e_from_stress(estress=2.25)
        2.70824...

        Array inputs:
        >>> a.e_from_stress(estress=np.array([1.25, 2.25]), pstress=2.25)
        array([ 2.75930...,  2.70824...])


        Logarithmic void ratio scale:
        On recompression line:
        >>> x, y = np.array([1,2,2.5]), np.array([4, 3.5, 2])
        >>> a = PwiseLinearSoilModel(siga=x, ea=y, Cr=0.1, ylog=True)
        >>> a.e_from_stress(estress=1.25, pstress=2.25)
        3.330803...

        On compression line:
        >>> a.e_from_stress(estress=2.25, pstress=2)
        2.64575...

        Normally consolidated (pstress not specified):
        >>> a.e_from_stress(estress=2.25)
        2.64575...

        Array inputs:
        >>> a.e_from_stress(estress=np.array([1.25, 2.25]), pstress=2.25)
        array([ 3.330803...,  2.64575...])

        Logarithmic effective stress and void ratio scales:
        On recompression line:
        >>> x, y = np.array([1,2,2.5]), np.array([4, 3.5, 2])
        >>> a = PwiseLinearSoilModel(siga=x, ea=y, Cr=0.1, xlog=True, ylog=True)
        >>> a.e_from_stress(estress=1.25, pstress=2.25)
        2.76255...

        On compression line:
        >>> a.e_from_stress(estress=2.25, pstress=2)
        2.604857...

        Normally consolidated (pstress not specified):
        >>> a.e_from_stress(estress=2.25)
        2.604857...

        Array inputs:
        >>> a.e_from_stress(estress=np.array([1.25, 2.25]), pstress=2.25)
        array([ 2.76255...,  2.604857...])

        """

        pstress = kwargs.get('pstress', estress)
        Cr = self.Cr
        max_past = np.maximum(pstress, estress)

        #transform x data if needed
        if self.xlog:
            siga = self.log_siga
            # np.log10(max_past, out=max_past) # doesn't work for single values
            max_past = np.log10(max_past)
            estress = np.log10(estress)
        else:
            siga=self.siga
        #transform y data if needed
        if self.ylog:
            ea = self.log_ea
        else:
            ea = self.ea

        # void ratio at preconsolidation pressure
        e = np.interp(max_past, siga, ea)
        # void ratio at current effetive stress
        e += Cr * (max_past - estress)

        # transform y back if needed
        if self.ylog:
            # np.power(10, e, out=e) # doesn't work for single values
            e = np.power(10, e)

        return e

    def stress_from_e(self, e, **kwargs):
        """effective stress from void ratio

        Parameters
        ----------
        e : float
            current void ratio
        pstress : float, optional
            reconsolidation stress.  Default pstress=estress i.e. normally
            consolidated.

        Returns
        -------
        estress : float
            effective stress corresponding to current void ratio

        Examples
        --------
        On recompression line:
        >>> x, y = np.array([1,2,2.5]), np.array([4, 3.5, 2])
        >>> a = PwiseLinearSoilModel(siga=x, ea=y, Cr=0.1)
        >>> a.stress_from_e(e=2.8, pstress=2.25)
        1.75...

        On compression line:
        >>> a.stress_from_e(e=2.65, pstress=2)
        2.28333...

        Normally consolidated (pstress not specified):
        >>> a.stress_from_e(e=2.65)
        2.28333...

        Array inputs:
        >>> a.stress_from_e(e=np.array([2.8, 2.65]),
        ... pstress=np.array([2.25, 2.0]))
        array([ 1.75...,  2.28333...])


        Logarithmic effective stress scale:
        On recompression line:
        >>> x, y = np.array([1,2,2.5]), np.array([4, 3.5, 2])
        >>> a = PwiseLinearSoilModel(siga=x, ea=y, Cr=0.1, xlog=True)
        >>> a.stress_from_e(e=2.73, pstress=2.25)
        1.363494...

        On compression line:
        >>> a.stress_from_e(e=2.73, pstress=2)
        2.2427...

        Normally consolidated (pstress not specified):
        >>> a.stress_from_e(e=2.73)
        2.2427...

        Array inputs:
        >>> a.stress_from_e(e=2.73, pstress=np.array([2.25, 2.]))
        array([ 1.363494...,  2.2427...])


        Logarithmic void ratio scale:
        On recompression line:
        >>> x, y = np.array([1,2,2.5]), np.array([4, 3.5, 2])
        >>> a = PwiseLinearSoilModel(siga=x, ea=y, Cr=0.1, ylog=True)
        >>> a.stress_from_e(e=2.75, pstress=2.25)
        2.082163...

        On compression line:
        >>> a.stress_from_e(e=2.65, pstress=2)
        2.24856...

        Normally consolidated (pstress not specified):
        >>> a.stress_from_e(e=2.65)
        2.24856...

        Array inputs:
        >>> a.stress_from_e(e=np.array([2.75, 2.65]),
        ... pstress=np.array([2.25, 2]))
        array([ 2.082163...,  2.24856...])

        Logarithmic effective stress and void ratio scales:
        On recompression line:
        >>> x, y = np.array([1,2,2.5]), np.array([4, 3.5, 2])
        >>> a = PwiseLinearSoilModel(siga=x, ea=y, Cr=0.1, xlog=True, ylog=True)
        >>> a.stress_from_e(e=2.65, pstress=2.25)
        1.8948013...

        On compression line:
        >>> a.stress_from_e(e=2.65, pstress=2)
        2.23463...

        Normally consolidated (pstress not specified):
        >>> a.stress_from_e(e=2.65)
        2.23463...

        Array inputs:
        >>> a.stress_from_e(e=2.65, pstress=np.array([2.25, 2]))
        array([ 1.8948013...,  2.23463...])

        """


        #transform x data if needed
        if self.xlog:
            siga = self.log_siga
#            estress = np.log10(estress)
        else:
            siga=self.siga
        #transform y data if needed
        if self.ylog:
            ea = self.log_ea
            e = np.log10(e)
        else:
            ea = self.ea

        pstress = kwargs.get('pstress', None)


        #fact = 2.3025850929940459 # 10**(x)==exp(fact*x)


        if pstress is None:
            # Normally consolidated
            estress = np.interp(e, ea[::-1], siga[::-1])
            if self.xlog:
                estress = np.power(10.0, estress)
#                estress *= fact
#                np.exp(estress, out=estress)
            return estress

        if self.xlog:
            pstress = np.log10(pstress)


        Cr = self.Cr
        # void ratio at preconsolidation pressure
        ep = np.interp(pstress, siga, ea)

        # stress change from (pstress, ep) if on pwise line
        dp_interp = np.interp(e, ea[::-1], siga[::-1]) - pstress
        # stress change from (pstress, ep) if on extended Cr line
        dpCr = (ep - e) / Cr

        # effective stress at current void ratio
        estress = pstress + np.minimum(dp_interp, dpCr)

        # transform x back if needed
        if self.xlog:
            estress = np.power(10.0, estress)

        return  estress

    def e_and_stress_for_plotting(self, **kwargs):
        """void ratio and stress values that plot the model

        Parameters
        ----------
        pstress : float, optional
            reconsolidation stress.  Default behaviour is normally
            consolidated.
        npts : int, optional
            number of points to return.  Default npts=100
        xmin, ymin : float, optional
            range of x (i.e. effective stress) values from which
            to return points. Default minumum of model siga


        Returns
        -------
        x, y : 1d ndarray
            `npts` stress, and void ratio values between `xmin` and `ymin`

        """

        npts = kwargs.get('n', 100)
        xmin, xmax = np.min(self.siga), np.max(self.siga)

        x = np.linspace(xmin, xmax, npts)
        pstress = kwargs.get('pstress', x)
        y = self.e_from_stress(x, pstress=pstress)

        return x, y


    def av_from_stress(self, estress, **kwargs):
        """slope of void ratio from effective stress

        Parameters
        ----------
        estress : float
            current effective stress
        pstress : float, optional
            reconsolidation stress.  Default pstress=estress i.e. normally
            consolidated.

        Returns
        -------
        av : float
            slope of void-ratio vs effective stress plot at current stress
            state

        Examples
        --------
        On recompression line:
        >>> x, y = np.array([1,2,2.5]), np.array([4, 3.5, 2])
        >>> a = PwiseLinearSoilModel(siga=x, ea=y, Cr=0.1)
        >>> a.av_from_stress(estress=1.25, pstress=2.25)
        0.10...

        On compression line:
        >>> a.av_from_stress(estress=2.25, pstress=2)
        3.0...

        Normally consolidated (pstress not specified):
        >>> a.av_from_stress(estress=2.25)
        3.0...

        Array inputs:
        >>> a.av_from_stress(estress=np.array([1.25, 2.25]),
        ... pstress=np.array([2.25, 2.]))
        array([ 0.1,  3. ])


        Logarithmic effective stress scale:
        On recompression line:
        >>> x, y = np.array([1,2,2.5]), np.array([4, 3.5, 2])
        >>> a = PwiseLinearSoilModel(siga=x, ea=y, Cr=0.1, xlog=True)
        >>> a.av_from_stress(estress=1.25, pstress=2.25)
        0.034743...

        On compression line:
        >>> a.av_from_stress(estress=2.25, pstress=2)
        2.987...

        Normally consolidated (pstress not specified):
        >>> a.av_from_stress(estress=2.25)
        2.987...

        Array inputs:
        >>> a.av_from_stress(estress=np.array([1.25, 2.25]),
        ... pstress=np.array([2.25, 2.]))
        array([ 0.034743...,  2.987...])


        Logarithmic void ratio scale:
        On recompression line:
        >>> x, y = np.array([1,2,2.5]), np.array([4, 3.5, 2])
        >>> a = PwiseLinearSoilModel(siga=x, ea=y, Cr=0.1, ylog=True)
        >>> a.av_from_stress(estress=1.25, pstress=2.25)
        0.76694...

        On compression line:
        >>> a.av_from_stress(estress=2.25, pstress=2)
        2.9612...

        Normally consolidated (pstress not specified):
        >>> a.av_from_stress(estress=2.25)
        2.9612...

        Array inputs:
        >>> a.av_from_stress(estress=np.array([1.25, 2.25]), pstress=2.25)
        array([ 0.76694...,  2.9612...])

        Logarithmic effective stress and void ratio scales:
        On recompression line:
        >>> x, y = np.array([1,2,2.5]), np.array([4, 3.5, 2])
        >>> a = PwiseLinearSoilModel(siga=x, ea=y, Cr=0.1, xlog=True, ylog=True)
        >>> a.av_from_stress(estress=1.25, pstress=2.25)
        0.2210045...

        On compression line:
        >>> a.av_from_stress(estress=2.25, pstress=2)
        2.9034...

        Normally consolidated (pstress not specified):
        >>> a.av_from_stress(estress=2.25)
        2.9034...

        Array inputs:
        >>> a.av_from_stress(estress=np.array([1.25, 2.25]), pstress=2.25)
        array([ 0.2210045...,  2.9034...])

        """

        pstress = kwargs.get('pstress', estress)
        Cr = self.Cr
        max_past = np.maximum(pstress, estress)

        #transform x data if needed
        if self.xlog:
            siga = self.log_siga
            # np.log10(max_past, out=max_past) # doesn't work for single values
            max_past = np.log10(max_past)
            estress = np.log10(estress)
        else:
            siga=self.siga
        #transform y data if needed
        if self.ylog:
            ea = self.log_ea
        else:
            ea = self.ea

        # interval at preconsolidatio stress
        i = np.searchsorted(siga, max_past)
        Cc = (ea[i - 1] - ea[i]) / (siga[i] - siga[i - 1])
        # void ratio at preconsolidation pressure
        e = ea[i-1] - Cc * (max_past - siga[i - 1])
        # void ratio at current effetive stress
        e += Cr * (max_past - estress)

        Cx = np.where(max_past > estress, Cr, Cc) # may need float comparison

        # modify Cx slope for log axes
        fact = 2.3025850929940459 #fact = log(10)
        dx, dy = 1.0, 1.0
        # transform y back if needed
        if self.ylog:
            dy = fact * np.power(10.0, e)
        if self.xlog:
            dx = fact * np.power(10.0, estress)

        av = Cx * dy / dx
        return av


if __name__ == '__main__':
#    print(CcCr_estress_from_e(2.95154499, 50, 3, 0.5, 10, 5))
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest', '--doctest-options=+ELLIPSIS'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])


#    print(repr(CcCr_e_from_stresses(np.array([40, 60]), np.array([50, 60]), 3, 0.5, 10, 5)))

#    a = AVSoilModel(av=1.5, siga=19, ea=4)
#    a.plot_constitutive_model()
#    plt.show()
    a = CcCrSoilModel(Cc=3.0, Cr=0.5, siga=10, ea=5)
    siga = np.linspace(1,80, 10)

    ea = np.cos(siga/100)
#    print(np.all(np.diff(ea) < 0))
#    print(siga, ea)
    siga = np.array([1,5,10,15,50,80, 100], dtype=float)
    ea = np.array([5,4.9,4.7,4.0,3,2, 1.8], dtype=float)
#    xlog, ylog=(1, 1)
#    xlog, ylog=(0, 0)
    xlog, ylog=(0, 1)
    xlog, ylog=(1, 0)

    x, y = np.array([1,2,2.5]), np.array([4, 3.5, 2])
    siga, ea = x, y
    a = PwiseLinearSoilModel(siga=x, ea=y, Cr=0.1, xlog=xlog, ylog=ylog)
    a.e_from_stress(estress=1.25, pstress=2.25)
#    a = PwiseLinearSoilModel(siga=siga, ea=ea, Cr=0.02, xlog=xlog, ylog=ylog)
    plt.plot(siga, ea, marker='*', ls=".")



    for i in [2.25]:#np.linspace(10, 40, 3):

        x, y = a.e_and_stress_for_plotting(pstress=i)
        plt.plot(x, y)

        x = 1.25
        y = a.e_from_stress(x, pstress=i)
        plt.plot(x, y, marker='o')
        av = a.av_from_stress(x, pstress=i)
        print(x, y)
        print(repr(av))

        xmin = max(x-3, np.min(siga))
        xmax = min(np.max(siga), x+3)

        xx = np.linspace(x, xmax, 100)
        x0 = xx[0]
        y0 = a.e_from_stress(xx[0], pstress=i)
        dx = np.diff(xx)
        av = a.av_from_stress(xx[1:], pstress=i)
        yy=np.zeros_like(xx)
        yy[0]=y0
        yy[1:] = y0+np.cumsum(dx*(-av))
        plt.plot(xx,yy, ls='.', marker='o')


#
#        y = 2.8
#        x = a.stress_from_e(y, pstress=i)
#        plt.plot(x, y, marker='s')
#        print(x, y)

    a.plot_constitutive_model()
    if xlog:
        plt.gca().set_xscale('log')
    if ylog:
        plt.gca().set_yscale('log')
    plt.grid()
    plt.show()
#    print(b)