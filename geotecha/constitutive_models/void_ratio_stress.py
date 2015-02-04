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
Relationships between void-ratio and effective stress.

"""


from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import functools
from scipy.optimize import fixed_point

class OneDimensionalVoidRatioEffectiveStress(object):
    """Base class for defining 1D void ratio-effective stress relationships"""

    def e_from_stress(self, estress, **kwargs):
        """Void ratio from effective stress"""
        raise NotImplementedError("e_from_stress must be implemented")

    def stress_from_e(self, e, **kwargs):
        """Effective stress from void ratio"""
        raise NotImplementedError("stress_from_e must be implemented")

    def e_and_stress_for_plotting(self, **kwargs):
        """Void ratio and stress values that plot the method"""
        # should return a tuple of x and y values
        raise NotImplementedError("e_and_stress_for_plotting must be "
                                  "implemented")

    def av_from_stress(self, estress, **kwargs):
        """Slope of void ratio from effective stress"""
        raise NotImplementedError("av_from_stress must be implemented")

    def plot_model(self, **kwargs):
        """plot the void ratio-stress points"""
        ax = kwargs.pop('ax', plt.gca())
        x, y = self.e_and_stress_for_plotting(**kwargs)

        ax.plot(x, y)
        return

class AvSoilModel(OneDimensionalVoidRatioEffectiveStress):
    """Linear void ratio-effective stress relationship

    Parameters
    ----------
    av : float
        Slope of compression line.
    siga, ea : float
        Effective stress and void ratio specifying point on av line.

    """

    def __init__(self, av, siga, ea):
        self.av = av
        self.siga = siga
        self.ea = ea

    def e_from_stress(self, estress, **kwargs):
        """Void ratio from effective stress

        Parameters
        ----------
        estress : float
            Current effective stress.

        Returns
        -------
        e : float
            Void ratio corresponding to current stress state.

        Examples
        --------
        >>> a = AvSoilModel(av=1.5, siga=19, ea=4)
        >>> a.e_from_stress(20)
        2.5

        Array inputs:

        >>> a = AvSoilModel(av=1.5, siga=19, ea=4)
        >>> a.e_from_stress(np.array([20, 21]))
        array([ 2.5,  1. ])

        """

        return self.ea - self.av * (estress - self.siga)

    def stress_from_e(self, e, **kwargs):
        """Effective stress from void ratio

        Parameters
        ----------
        e : float
            Current void ratio.

        Returns
        -------
        estress : float
            Effective stress corresponding to current void ratio

        Examples
        --------
        >>> a = AvSoilModel(av=1.5, siga=19, ea=4)
        >>> a.stress_from_e(1)
        21.0

        Array inputs:

        >>> a = AvSoilModel(av=1.5, siga=19, ea=4)
        >>> a.stress_from_e(np.array([1, 2.5]))
        array([ 21.,  20.])

        """

        estress = self.siga + (self.ea - e) / self.av

        return  estress

    def e_and_stress_for_plotting(self, **kwargs):
        """Void ratio and stress values that plot the model

        Parameters
        ----------
        npts : int, optional
            Number of points to return.  Default npts=100.
        xmin, xmax : float, optional
            range of x (i.e. effective stress) values from which
            to return points. Default xmin=1, xmax=100

        Returns
        -------
        x, y : 1d ndarray
            `npts` stress, and void ratio values between `xmin` and `xmax`.

        """

        npts = kwargs.get('npts', 100)
        xmin, xmax = kwargs.get('xmin', 1.0), kwargs.get('xmax', 100)

        x = np.linspace(xmin, xmax, npts)
        y = self.e_from_stress(x)
        return x, y

    def av_from_stress(self, *args, **kwargs):
        """Slope of void ratio from effective stress"""
        return self.av


class CcCrSoilModel(OneDimensionalVoidRatioEffectiveStress):
    """Semi-log void ratio-effective stress realationship

    Parameters
    ----------
    Cc : float
        Compressibility index, slope of e-log(sig) line.
    Cr : float
        Recompression index, slope of e-log(sig) line.
    siga, ea : float
        Point on compression line fixing it in effective stress-void ratio
        space.

    """

    def __init__(self, Cc, Cr, siga, ea):
        self.Cc = Cc
        self.Cr = Cr
        self.siga = siga
        self.ea = ea

    def e_from_stress(self, estress, **kwargs):
        """Void ratio from effective stress

        Parameters
        ----------
        estress : float
            Current effective stress.
        pstress : float, optional
            Preconsolidation stress.  Default pstress=estress i.e. normally
            consolidated.

        Returns
        -------
        e : float
            Void ratio corresponding to current stress state.

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
        """Effective stress from void ratio

        Parameters
        ----------
        e : float
            Current void ratio.
        pstress : float, optional
            Preconsolidation stress.  Default pstress=estress i.e. normally
            consolidated.

        Returns
        -------
        estress : float
            Effective stress corresponding to current void ratio.

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
        """Void ratio and stress values that plot the model

        Parameters
        ----------
        pstress : float, optional
            Preconsolidation stress.  Default behaviour is normally
            consolidated.
        npts : int, optional
            Number of points to return.  Default npts=100.
        xmin, xmax : float, optional
            range of x (i.e. effective stress) values from which
            to return points. Default xmin=1, xmax=100


        Returns
        -------
        x, y : 1d ndarray
            `npts` stress, and void ratio values between `xmin` and `xmax`.

        """

        npts = kwargs.get('npts', 100)
        xmin, xmax = kwargs.get('xmin', 1.0), kwargs.get('xmax', 100)

        x = np.linspace(xmin, xmax, npts)
        pstress = kwargs.get('pstress', x)
        i = np.searchsorted(x, pstress)
        x[i] = pstress
        y = self.e_from_stress(x, pstress=pstress)

        return x, y


    def av_from_stress(self, estress, **kwargs):
        """Slope of void ratio from effective stress

        Parameters
        ----------
        estress : float
            Current effective stress.
        pstress : float, optional
            Preconsolidation stress.  Default pstress=estress i.e. normally
            consolidated.

        Returns
        -------
        av : float
            Slope of void-ratio vs effective stress plot at current stress
            state.

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
    log10-natural, or log10, log10.

    Parameters
    ----------
    siga, ea : 1d array
        Effective stress values and void ratio values defining a one-to-one
        relationship.  Slope of void ratio-effectinve stress plot should never
        fall below `Cr`.
    Cr : float
        Precompression index, slope of void rato-effective stress line.  Note
        that `Cr` is the slope in whatever scales of slog and ylog that have
        been chosen.
    xlog, ylog : True/False, Optional
        If True then interpolation on each axis is assumed to be logarithmic
        with base 10. Default xlog=ylog=False

    """

    def __init__(self, siga, ea, Cr, xlog=False, ylog=False):
        self.siga = np.asarray(siga, dtype=float)
        self.ea = np.asarray(ea, dtype=float)
        self.xlog = xlog
        self.ylog = ylog
        self.Cr = Cr
        #TODO: adjust for different logarithm bases.

        self.siga_slice = slice(None)
        self.ea_slice = slice(None)

        if np.any(np.diff(self.siga) <= 0):
            # siga is in decreasing order
            # reverse the slice for e_from_stress interpolation
            self.siga_slice = slice(None, None, -1)


        if np.any(np.diff(self.ea) <= 0):
            # ea is in decreasing order
            # reverse the slice for stress_from_e interpolation
            self.ea_slice = slice(None, None, -1)
#            raise ValueError("'ea' must be in monotomically decreasng order.")

        if len(siga) != len(ea):
            raise IndexError("'siga' and 'ea' must be the same length.")

        self.log_siga = np.log10(self.siga)
        self.log_ea = np.log10(self.ea)

    def e_from_stress(self, estress, **kwargs):
        """Void ratio from effective stress

        Parameters
        ----------
        estress : float
            Current effective stress.  etress must be within the range of the
            soil model points.
        pstress : float, optional
            Preconsolidation stress.  Default pstress=estress i.e. normally
            consolidated. pstress must be in the range of the soil model
            points.

        Returns
        -------
        e : float
            Void ratio corresponding to current stress state.

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


        Logarithmic void ratio scale
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


        Logarithmic effective stress and void ratio scales
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


        Increasing vs decreasing inputs

        >>> ea = np.arange(1,10)
        >>> siga = 3 * ea
        >>> np.isclose(PwiseLinearSoilModel(siga, ea,
        ... Cr=0.1).e_from_stress(7.2),
        ... PwiseLinearSoilModel(siga[::-1], ea[::-1],
        ... Cr=0.1).e_from_stress(7.2))
        True


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
            siga = self.siga
        #transform y data if needed
        if self.ylog:
            ea = self.log_ea
        else:
            ea = self.ea

        # void ratio at preconsolidation pressure
        e = np.interp(max_past, siga[self.siga_slice], ea[self.siga_slice])
        # void ratio at current effetive stress
        e += Cr * (max_past - estress)

        # transform y back if needed
        if self.ylog:
            # np.power(10, e, out=e) # doesn't work for single values
            e = np.power(10, e)

        return e

    def stress_from_e(self, e, **kwargs):
        """Effective stress from void ratio

        Parameters
        ----------
        e : float
            Current void ratio.
        pstress : float, optional
            Preconsolidation stress.  Default pstress=estress i.e. normally
            consolidated.

        Returns
        -------
        estress : float
            Effective stress corresponding to current void ratio.

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


        Logarithmic effective stress scale
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


        Logarithmic void ratio scale
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


        Logarithmic effective stress and void ratio scales
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


        Increasing vs decreasing inputs

        >>> ea = np.arange(1,10)
        >>> siga = 3 * ea
        >>> np.isclose(PwiseLinearSoilModel(siga, ea, Cr=0.1).stress_from_e(3.0, pstress=4.0),
        ... PwiseLinearSoilModel(siga[::-1], ea[::-1], Cr=0.1).stress_from_e(3.0, pstress=4.0))
        True

        """


        #transform x data if needed
        if self.xlog:
            siga = self.log_siga
#            estress = np.log10(estress)
        else:
            siga = self.siga
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
            estress = np.interp(e, ea[self.ea_slice], siga[self.ea_slice])
            if self.xlog:
                estress = np.power(10.0, estress)
#                estress *= fact
#                np.exp(estress, out=estress)
            return estress

        if self.xlog:
            pstress = np.log10(pstress)


        Cr = self.Cr
        # void ratio at preconsolidation pressure
        ep = np.interp(pstress, siga[self.siga_slice], ea[self.siga_slice])

        # stress change from (pstress, ep) if on pwise line
        dp_interp = np.interp(e, ea[self.ea_slice], siga[self.ea_slice]) - pstress
        # stress change from (pstress, ep) if on extended Cr line
        dpCr = (ep - e) / Cr

        # effective stress at current void ratio
        estress = pstress + np.minimum(dp_interp, dpCr)

        # transform x back if needed
        if self.xlog:
            estress = np.power(10.0, estress)

        return  estress

    def e_and_stress_for_plotting(self, **kwargs):
        """Void ratio and stress values that plot the model

        Parameters
        ----------
        pstress : float, optional
            Preconsolidation stress.  Default behaviour is normally
            consolidated.
        npts : int, optional
            Number of points to return.  Default npts=100
        xmin, xmax : float, optional
            Range of x (i.e. effective stress) values from which
            to return points. Default minumum of model `siga`.


        Returns
        -------
        x, y : 1d ndarray
            `npts` stress, and void ratio values between `xmin` and `xmax`.

        """

        npts = kwargs.get('npts', 100)
        xmin, xmax = np.min(self.siga), np.max(self.siga)

        x = np.linspace(xmin, xmax, npts)
        pstress = kwargs.get('pstress', x)
        y = self.e_from_stress(x, pstress=pstress)

        return x, y


    def av_from_stress(self, estress, **kwargs):
        """Slope of void ratio from effective stress

        Parameters
        ----------
        estress : float
            Current effective stress.
        pstress : float, optional
            Preconsolidation stress.  Default pstress=estress i.e. normally
            consolidated.

        Returns
        -------
        av : float
            Slope of void-ratio vs effective stress plot at current stress
            state.

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


        Logarithmic effective stress scale
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


        Logarithmic void ratio scale
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


        Logarithmic effective stress and void ratio scales
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


        Increasing vs decreasing inputs

        >>> ea = np.arange(1,10)
        >>> siga = 3 * ea
        >>> np.isclose(PwiseLinearSoilModel(siga, ea, Cr=0.1).av_from_stress(7.2),
        ... PwiseLinearSoilModel(siga[::-1], ea[::-1], Cr=0.1).av_from_stress(7.2))
        True


        """

        pstress = kwargs.get('pstress', estress)
        Cr = self.Cr
        max_past = np.maximum(pstress, estress)

        #transform x data if needed
        if self.xlog:
            siga = self.log_siga[self.siga_slice]
            # np.log10(max_past, out=max_past) # doesn't work for single values
            max_past = np.log10(max_past)
            estress = np.log10(estress)
        else:
            siga = self.siga[self.siga_slice]
        #transform y data if needed
        if self.ylog:
            ea = self.log_ea[self.siga_slice]
        else:
            ea = self.ea[self.siga_slice]

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


class FunctionSoilModel(OneDimensionalVoidRatioEffectiveStress):
    """Functional definition of void-ratio stress relationship

    User provides python functions.

    Parameters
    ----------
    fn_e_from_stress: callable object
        Function to obtain void ratio from stress.  fn_e_from_stress should
        be the inverse function of fn_stress_from_e.
    fn_stress_from_e : callable object
        Function to obtain stress from void ratio. fn_stress_from_e should
        be the inverse function of fn_e_from_stress.
    fn_av_from_stress: callable object
        Function to obtain slope of void ratio-stress relationship.
        fn_av_from_stress should be negative the derivative of
        fn_e_from_stress w.r.t. stress
        be the inverse function of fn_k_from_e.
    *args, **kwargs : anything
        Positional and keyword arguments to be passed to the
        fn_e_from_stress, fn_stress_from_e, fn_av_from_stress functions.  Note
        that any additional args and kwargs passed to the functions will be
        appended to the args, and kwargs.  You may get into a mess for
        example with e_from_stress because normally the first postional
        argument passed to such fucntions is estress.  If you add your own
        positional arguments, then k will be after you own arguments.
        Just be aware that the usual way to call methods of the base object
        `OneDimensionalVoidRatioEffectiveStress` a single positonal arguement,
        e.g. estress, e, followed by any required keyword arguments.

    Notes
    -----
    Any function should be able to accept additonal keywords.


    Examples
    --------
    >>> def efs(s, b=2):
    ...     return s * b
    >>> def sfe(e, b=2):
    ...     return e / b
    >>> def afs(s, b=2):
    ...     return -b
    >>> a = FunctionSoilModel(efs, sfe, afs, b=5)
    >>> a.e_from_stress(3)
    15
    >>> a.stress_from_e(15)
    3.0
    >>> a.av_from_stress(3)
    -5

    Prconsolidation stress:

    >>> def efs2(s, pstress=None, b=2):
    ...     if pstress is None:
    ...         pstress=8
    ...     return s * b + pstress
    >>> a = FunctionSoilModel(efs2, sfe, afs, b=5)
    >>> a.e_from_stress(3)
    23

    """

    def __init__(self,
                 fn_e_from_stress,
                 fn_stress_from_e,
                 fn_av_from_stress,
                 *args,
                 **kwargs):

        self.fn_e_from_stress = fn_e_from_stress
        self.fn_stress_from_e = fn_stress_from_e
        self.fn_av_from_stress = fn_av_from_stress
        self.args = args
        self.kwargs = kwargs

        self.e_from_stress = functools.partial(fn_e_from_stress,
                                               *args,
                                               **kwargs)
        self.stress_from_e = functools.partial(fn_stress_from_e,
                                               *args,
                                               **kwargs)
        self.av_from_stress = functools.partial(fn_av_from_stress,
                                               *args,
                                               **kwargs)

        return


class YinAndGrahamSoilModel(OneDimensionalVoidRatioEffectiveStress):
    """Yin and Graham creep void ratio-effective stress realationship

    Parameters
    ----------
    lam : float
        Compressibility index, slope of the referece time line
        e-ln(sig) line.
    kap : float
        Recompression index, slope of e-ln(sig) line.
    psi : float
        Creep parameter.  Slope of the void ratio-ln(time) line.
    siga, ea : float
        Point on reference time line fixing it in effective
        stress-void ratio space.
    ta : float
        Reference time corresponding to the reference time line.
    e0 : float, optional
        Initial void ratio. Default e0=None.
        If specifying initial conditions then only use two of the four
        available (e0, estress0, pstress0, t0).
    estress0 : float, optional
        Initial effective stress. Default estress=None.
        If specifying initial conditions then only use two of the four
        available (e0, estress0, pstress0, t0).
    pstress0 : float, optional
        Initial 'preconsolidation` stress on the reference time line.
        Default pstress0=None.
        If specifying initial conditions then only use two of the four
        available (e0, estress0, pstress0, t0).
    t0 : float, optional
        Initial equivalent time at the initial conditon. Default t0=None.
        If specifying initial conditions then only use two of the four
        available (e0, estress0, pstress0, t0).

    Attributes
    ----------
    _alpha : float
        Composite material parameter (lam - kap / psi)
    _psi_zero_model : CcCrSoilModel
        soil model when psi=0.  i.e. CcCrSoilModel following instant time line
        and reference time line.
    _igral : float
        Ccurrent value of the material model integration w.r.t. time.
        Igral = integrate[1/t0 * (estress/etress0)**alpha, t,0,t]


    Examples
    --------
    Combos of initial conditions give the same results

    >>> fixed_param = dict(lam=0.2, kap=0.04, psi=0.01, siga=20, ea=1, ta=1)
    >>> oparam = dict(e0=0.95, estress0=18, pstress0=28.06637954,
    ... t0=1220.73731)
    >>> a = YinAndGrahamSoilModel(e0=oparam['e0'],
    ... estress0=oparam['estress0'], **fixed_param)
    >>> [a.e0, a.estress0, a.pstress0, a.t0]
    [0.95..., 18..., 28.0..., 1220.737...]
    >>> a = YinAndGrahamSoilModel(e0=oparam['e0'],
    ... pstress0=oparam['pstress0'], **fixed_param)
    >>> [a.e0, a.estress0, a.pstress0, a.t0]
    [0.95..., 18..., 28.0..., 1220.737...]
    >>> a = YinAndGrahamSoilModel(e0=oparam['e0'],
    ... t0=oparam['t0'], **fixed_param)
    >>> [a.e0, a.estress0, a.pstress0, a.t0]
    [0.95..., 18..., 28.0..., 1220.737...]
    >>> a = YinAndGrahamSoilModel(estress0=oparam['estress0'],
    ... pstress0=oparam['pstress0'], **fixed_param)
    >>> [a.e0, a.estress0, a.pstress0, a.t0]
    [0.95..., 18..., 28.0..., 1220.737...]
    >>> a = YinAndGrahamSoilModel(estress0=oparam['estress0'],
    ... t0=oparam['t0'], **fixed_param)
    >>> [a.e0, a.estress0, a.pstress0, a.t0]
    [0.95..., 18..., 28.0..., 1220.737...]
    >>> a = YinAndGrahamSoilModel(pstress0=oparam['pstress0'],
    ... t0=oparam['t0'], **fixed_param)
    >>> [a.e0, a.estress0, a.pstress0, a.t0]
    [0.95..., 18..., 28.0..., 1220.737...]

    """

    def __init__(self, lam, kap, psi, siga, ea, ta,
                 e0=None, estress0=None, pstress0=None, t0=None):
        self.lam = lam
        self.kap = kap
        self.psi = psi
        self.siga = siga
        self.ea = ea
        self.ta = ta

        self._alpha = (self.lam - self.kap) / self.psi
        self._psi_zero_model = CcCrSoilModel(Cc=np.log(10.0)*self.lam,
                                             Cr=np.log(10.0)*self.kap,
                                             siga=self.siga,
                                             ea=self.ea)

        self.e0, self.estress0, self.pstress0, self.t0 = (
            self.initial_conditions(e0=e0,
                                    estress0=estress0,
                                    pstress0=pstress0,
                                    t0=t0))

        self._igral = 0#np.zeros(len(np.asarray(self.e0)), dtype=float)



    def initial_conditions(self, e0=None, estress0=None, pstress0=None, t0=None):
        """Calculate the initial conditions

        Only use two of the paramters, the other two will be calculated.

        Parameters
        ----------
        e0 : float, optional
            Initial void ratio. Default e0=None.
            If specifying initial conditions then only use two of the four
            available (e0, estress0, pstress0, t0).
        estress0 : float, optional
            Initial effective stress. Default estress=None.
            If specifying initial conditions then only use two of the four
            available (e0, estress0, pstress0, t0).
        pstress0 : float, optional
            Initial 'preconsolidation` stress on the reference time line.
            Default pstress0=None.
            If specifying initial conditions then only use two of the four
            available (e0, estress0, pstress0, t0).
        t0 : float, optional
            Initial equivalent time at the initial conditon. Default t0=None.
            If specifying initial conditions then only use two of the four
            available (e0, estress0, pstress0, t0).

        Returns
        -------
        e0 : float
            Initial void ratio
        estress0 : float
            Initial effective stress
        pstress0 : float
            Inital preconsolidation stress
        t0 : float
            Initial equivalent time.

        """

        d = dict(e0=e0, estress0=estress0, pstress0=pstress0, t0=t0)
        ic_list = ['e0', 'estress0', 'pstress0','t0']
        ic_given = [v for v in ic_list if not d.get(v, None) is None]
        if len(ic_given)==0:
            #no initial conditions set
            return None, None, None, None
        elif len(ic_given)==2:
            #2 of the 4 initial condition parameters are given, calc the others
            e0 = d.get('e0', None)
            estress0 = d.get('estress0', None)
            pstress0 = d.get('pstress0', None)
            t0 = d.get('t0', None)
            ea = self.ea
            siga = self.siga
            alpha = self._alpha
            kap = self.kap
            lam = self.lam
            psi = self.psi
            ta = self.ta
            if ic_given==['e0', 'estress0']:
                numer = (ea - e0 - kap * np.log(estress0) + lam * np.log(siga))
                denom = (lam - kap)
                pstress0 = np.exp(numer / denom)
                t0 = (pstress0 / estress0)**(alpha) * ta
            elif ic_given==['e0', 'pstress0']:
                numer = (ea - e0- (lam-kap) * np.log(pstress0) + lam * np.log(siga))
                denom = (kap)
                estress0 = np.exp(numer / denom)
                t0 = (pstress0 / estress0)**(alpha) * ta
            elif ic_given==['e0', 't0']:
                OCR = (t0/ta)**(1/alpha)
                numer = (ea - e0 + lam * np.log(siga) + kap * np.log(OCR))
                denom = (lam)
                pstress0 = np.exp(numer / denom)
                estress0 = pstress0 / OCR
            elif ic_given==['estress0', 'pstress0']:
                OCR = pstress0 / estress0
                t0 = (OCR)**alpha * ta
                e0 = (ea - lam * np.log(pstress0 / siga)
                      + kap * np.log(OCR))
            elif ic_given==['estress0', 't0']:
                OCR = (t0/ta)**(1/alpha)
                pstress0 = OCR * estress0
                e0 = ea - lam * np.log(pstress0 / siga) + kap * np.log(OCR)
            elif ic_given==['pstress0', 't0']:
                OCR = (t0/ta)**(1/alpha)
                estress0 = pstress0 / OCR
                e0 = (ea - lam * np.log(pstress0 / siga)
                      + kap * np.log(OCR))
            return e0, estress0, pstress0, t0
        else:

            #too many initial conditoin parameters given.
            raise ValueError("If specifying initial conditions ({}) then only "
                    "two and only two can be specified.  "
                    "You have {}.".format(ic_list, ",".join(ic_given)))




    def e_from_stress(self, estress, **kwargs):
        """Void ratio from effective stress

        Parameters
        ----------
        estress : float
            Current effective stress. estress is assumed to be the effective
            stress at the end of an increment of length dt. i.e. at time t+dt.
        dt : float, optional
            Time increment. Default dt=None.  When dt is None, the void
            ratio is calculted assuming psi=0.  i.e. along the instant
            time line and then down the reference time line.
        pstress : float, optional
            Preconsolidation stress.  This is only relevant if dt is not None.
            Default pstress=estress i.e. on the reference time line.

        Returns
        -------
        e : float
            Void ratio corresponding to current stress state.

        Notes
        -----
        _igral will be updated!

        The void ratio stress relationship is:

        .. math::

           e=e0-\\kappa\\ln\\left({\\frac{\\sigma^{\\prime}_z}{\\sigma^{\\prime}_{z0}}}\\right)
             -\\psi\\left[{
               \\int_{0}^{t}{
                 \\frac{1}{t_0}
                 \\left({
                   \\frac{\\sigma^{\\prime}_z}
                   {\\sigma^{\\prime}_{z0}}}\\right)^{\\alpha}\\,dt
                             }+1
                           }\\right]

        To calculate the integral we keep track of the integration at time t
        and then use a finte difference approximation to calculate the
        integral at t+dt:

        .. math::

           e=e0-\\kappa\\ln\\left({\\frac{\\sigma^{\\prime}_{z,t+\\bigtriangleup t}}{\\sigma^{\\prime}_{z0}}}\\right)
             -\\psi\\left[{
               \\int_{0}^{t}{
                 \\frac{1}{t_0}
                 \\left({
                   \\frac{\\sigma^{\\prime}_z}
                   {\\sigma^{\\prime}_{z0}}}\\right)^{\\alpha}\\,dt
                             }+\\frac{\\bigtriangleup t}{t_0}\\left({
                   \\frac{\\sigma^{\\prime}_{z,t+\\bigtriangleup  t}}
                   {\\sigma^{\\prime}_{z0}}}\\right)^{\\alpha}+1
                           }\\right]


        YinAndGrahamSoilModel().e_from_stress()

        Examples
        --------


        """

        dt = kwargs.get('dt', None)
        pstress = kwargs.get('pstress', estress)

        if dt is None:
            e = self._psi_zero_model.e_from_stress(estress, pstress)
            return e
        else:
            # estress is assumed to be the effective stress at the end
            # of an increment of lenght dt.
            e0 = self.e0
            kap = self.kap
            lam = self.lam
            psi = self.psi
            estress0 = self.estress0

            inc = dt/self.t0 * (estress / estress0) ** self._alpha
            self._igral += inc
            e = (e0 - kap * np.log(estress / estress0)
                 - psi * np.log(self.igral + 1))

        return e

    def stress_from_e(self, e, **kwargs):
        """Effective stress from void ratio

        Parameters
        ----------
        e : float
            Current void ratio.  e is assumed to be the effective
            stress at the end of an increment of length dt. i.e. at time t+dt.
        dt : float, optional
            Time increment. Default dt=0.

        Returns
        -------
        estress : float
            Effective stress corresponding to current void ratio.

        Notes
        -----
        _igral will be updated!

        The void ratio stress relationship is:

        .. math::

           e=e0-\\kappa\\ln\\left({\\frac{\\sigma^{\\prime}_z}{\\sigma^{\\prime}_{z0}}}\\right)
             -\\psi\\left[{
               \\int_{0}^{t}{
                 \\frac{1}{t_0}
                 \\left({
                   \\frac{\\sigma^{\\prime}_z}
                   {\\sigma^{\\prime}_{z0}}}\\right)^{\\alpha}\\,dt
                             }+1
                           }\\right]

        To calculate the integral we keep track of the integration at time t
        and then use a finte difference approximation to calculate the
        integral at t+dt:

        .. math::

           e=e0-\\kappa\\ln\\left({\\frac{\\sigma^{\\prime}_{z,t+\\bigtriangleup t}}{\\sigma^{\\prime}_{z0}}}\\right)
             -\\psi\\left[{
               \\int_{0}^{t}{
                 \\frac{1}{t_0}
                 \\left({
                   \\frac{\\sigma^{\\prime}_z}
                   {\\sigma^{\\prime}_{z0}}}\\right)^{\\alpha}\\,dt
                             }+\\frac{\\bigtriangleup t}{t_0}\\left({
                   \\frac{\\sigma^{\\prime}_{z,t+\\bigtriangleup  t}}
                   {\\sigma^{\\prime}_{z0}}}\\right)^{\\alpha}+1
                           }\\right]

        Knowing :math:`e_0` and the integral at time :math:`t`, after
        rearranging we perform fixed point iteration to find the stress
        :math:`\\sigma^{\\prime}_{z,t+\\bigtriangleup  t}`.

        YinAndGrahamSoilModel().stress_from_e()

        Examples
        --------


        """

        dt = kwargs.get('dt', 0)
        e0 = self.e0
        kap = self.kap
        lam = self.lam
        psi = self.psi
        estress0 = self.estress0


        def fn(estress):
            inc = dt / self.t0 * (estress / estress0) ** self._alpha
            numer = e0 - e - psi * np.log(self._igral + inc + 1)
            es=estress0 * np.exp(numer/kap)

            return es

        estress = fixed_point(fn, estress0)
        inc = dt / self.t0 * (estress / estress0) ** self._alpha
        self._igral += inc

        return  estress
#
#    def e_and_stress_for_plotting(self, **kwargs):
#        """Void ratio and stress values that plot the model
#
#        Parameters
#        ----------
#        pstress : float, optional
#            Preconsolidation stress.  Default behaviour is normally
#            consolidated.
#        npts : int, optional
#            Number of points to return.  Default npts=100.
#        xmin, xmax : float, optional
#            range of x (i.e. effective stress) values from which
#            to return points. Default xmin=1, xmax=100
#
#
#        Returns
#        -------
#        x, y : 1d ndarray
#            `npts` stress, and void ratio values between `xmin` and `xmax`.
#
#        """
#
#        npts = kwargs.get('npts', 100)
#        xmin, xmax = kwargs.get('xmin', 1.0), kwargs.get('xmax', 100)
#
#        x = np.linspace(xmin, xmax, npts)
#        pstress = kwargs.get('pstress', x)
#        i = np.searchsorted(x, pstress)
#        x[i] = pstress
#        y = self.e_from_stress(x, pstress=pstress)
#
#        return x, y
#
#
#    def av_from_stress(self, estress, **kwargs):
#        """Slope of void ratio from effective stress
#
#        Parameters
#        ----------
#        estress : float
#            Current effective stress.
#        pstress : float, optional
#            Preconsolidation stress.  Default pstress=estress i.e. normally
#            consolidated.
#
#        Returns
#        -------
#        av : float
#            Slope of void-ratio vs effective stress plot at current stress
#            state.
#
#        Examples
#        --------
#        On recompression line:
#
#        >>> a = CcCrSoilModel(Cc=3.0, Cr=0.5, siga=10, ea=5)
#        >>> a.av_from_stress(estress=40, pstress=50)
#        0.00542868...
#
#        On compression line:
#
#        >>> a = CcCrSoilModel(Cc=3.0, Cr=0.5, siga=10, ea=5)
#        >>> a.av_from_stress(estress=60, pstress=50)
#        0.02171472...
#
#        Array inputs:
#
#        >>> a = CcCrSoilModel(Cc=3.0, Cr=0.5, siga=10, ea=5)
#        >>> a.av_from_stress(estress=np.array([40, 60.0]),
#        ... pstress=np.array([50, 55.0]))
#        array([ 0.00542868,  0.02171472])
#
#        """
#
#        pstress = kwargs.get('pstress', estress)
#
#        Cc = self.Cc
#        Cr = self.Cr
#
#        chooser = np.array((Cc, Cc, Cr), dtype=float)
#
#        # appropriate value of Cc or Cr
#        Cx = chooser[np.sign(estress - pstress, dtype=int)]
#
#        av = 0.43429448190325182 * Cx / estress
#
#        return av

if __name__ == '__main__':
#    print(CcCr_estress_from_e(2.95154499, 50, 3, 0.5, 10, 5))
#    import nose
#    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest', '--doctest-options=+ELLIPSIS'])
#    a = YinAndGraham(lam=0.2, kap=0.04, psi=0.001, siga=20, ea=1, ta=1,
#                     e0=1.5)

    fixed_param = dict(lam=0.2, kap=0.04, psi=0.01, siga=20, ea=1, ta=1)
    oparam = dict(e0=0.95, estress0=18)
    oparam = dict(e0=np.array([0.95, 0.90]), estress0=np.array([18,20]))
    a = YinAndGrahamSoilModel(e0=oparam['e0'], estress0=oparam['estress0'], **fixed_param)
    b=a._psi_zero_model.plot_model()
    plt.gca().plot(a.siga, a.ea , marker="o", ms=10)
    plt.gca().plot(a.estress0, a.e0, marker = 's', ms=5)
    plt.gca().plot(a.pstress0, a.ea-a.lam*np.log(a.pstress0/a.siga), marker = '^', ms=5)

    print([a.e0, a.estress0, a.pstress0, a.t0])
    bb = a.stress_from_e(a.e0*0.9, dt=10000000)
    print(bb)

    plt.gca().plot(bb, a.e0*.9, marker = 'h', ms=5)
    plt.show()

    plt.show()
