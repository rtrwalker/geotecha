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
module for permeability relationships
"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class PermeabilityVoidRatioRelationship(object):
    """Base class for defining permeability void ratio relationships"""

    def e_from_k(self, estress, **kwargs):
        """void ratio from permeability"""
        raise NotImplementedError("e_from_k must be implemented")

    def k_from_e(self, e, **kwargs):
        """permeability from void ratio"""
        raise NotImplementedError("k_from_e must be implemented")

    def e_and_k_for_plotting(self, **kwargs):
        """void ratio and permeability that plot the method"""
        # should return a tuple of x and y values
        # x-values are peremability, y-values are void ratio.
        raise NotImplementedError("e_and_k_for_plotting must be implemented")

    def plot_model(self, **kwargs):
        """plot the void ratio-permeability"""
        ax = kwargs.pop('ax', plt.gca())
        x, y = self.e_and_k_for_plotting(**kwargs)

        ax.plot(x, y)
        return


class CkPermeabilityModel(PermeabilityVoidRatioRelationship):
    """Semi-log void ratio permeability relationship

    Parameters
    ----------
    Ck : float
        slope of void-ratio vs permeability
    ka, ea : float
        permeability and void ratio specifying a point on e-k line

    """

    def __init__(self, Ck, ka, ea):
        self.Ck = Ck
        self.ka = ka
        self.ea = ea

    def e_from_k(self, k, **kwargs):
        """void ratio from permeability

        Parameters
        ----------
        k : float
            current permeability

        Returns
        -------
        e : float
            void ratio corresponding to current permeability

        Examples
        --------
        >>> a = CkPermeabilityModel(Ck=1.5, ka=10, ea=4)
        >>> a.e_from_k(8.0)
        3.854...

        Array Inputs
        >>> a.e_from_k(np.array([8.0, 4.0]))
        array([ 3.854...,  3.403...])

        """

        return self.ea + self.Ck * np.log10(k/self.ka)

    def k_from_e(self, e, **kwargs):
        """permeability from void ratio

        Parameters
        ----------
        e : float
            void ratio

        Returns
        -------
        k : float
            permeability corresponding to current void ratio

        Examples
        --------
        >>> a = CkPermeabilityModel(Ck=1.5, ka=10, ea=4)
        >>> a.k_from_e(3.855)
        8.0...

        Array Inputs
        >>> a.k_from_e(np.array([3.855, 3.404]))
        array([ 8.0...,  4.0...])






        """

        return self.ka * np.power(10.0, (e - self.ea) / self.Ck)


    def e_and_k_for_plotting(self, **kwargs):
        """void ratio and permeability that plot the model

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
            `npts` permeability, and void ratio values between
            `xmin` and `ymin`

        """

        npts = kwargs.get('n', 100)
        xmin, xmax = kwargs.get('xmin', 1.0), kwargs.get('xmax', 100)

        x = np.linspace(xmin, xmax, npts)
        y = self.e_from_k(x)
        return x, y

class ConstantPermeabilityModel(PermeabilityVoidRatioRelationship):
    """Permeability constant with void ratio

    Note that the method e_from_k is meaningless because there are multiple
    e values for single k value

    Parameters
    ----------
    ka : float
        permeability


    """

    def __init__(self, ka):
        self.ka = ka

    def e_from_k(self, k, **kwarks):
        """void ratio from permeability"""
        raise ValueError("e_from_k is meaningless for a constant permeability model")


    def k_from_e(self, e, **kwargs):
        """permeability from void ratio

        Parameters
        ----------
        e : float
            void ratio

        Returns
        -------
        ka : float
            the constant permeability

        Examples
        --------
        >>> a = ConstantPermeabilityModel(ka=2.5)
        >>> a.k_from_e(3.855)
        2.5

        Array Inputs
        >>> a.k_from_e(np.array([3.855, 3.404]))
        array([ 2.5,  2.5])

        """

        return np.ones_like(e, dtype=float) * self.ka


    def e_and_k_for_plotting(self, **kwargs):
        """void ratio and permeability that plot the model

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
            `npts` permeability, and void ratio values between
            `xmin` and `ymin`

        """

        npts = kwargs.get('n', 100)
        xmin, xmax = kwargs.get('xmin', 1.0), kwargs.get('xmax', 100)

        x = np.linspace(xmin, xmax, npts)
        y = self.e_from_k(x)
        return x, y


class PwiseLinearPermeabilityModel(PermeabilityVoidRatioRelationship):
    """Piecewise linear ratio permeability relationship

    x and y data can be interpolated natural-natural, natural-log10,
    log10-natural, or log10, log10

    Parameters
    ----------
    ka, ea : 1d array
        permeability values and void ratio values defining a one-to-one
        relationship.
    xlog, ylog : True/False, Optional
        If True then interpolation on each axis is assumed to be logarithmic
        with base 10. x refers to the peremability axis, y to the void ratio
        axis.  Default=False

    """

    def __init__(self, ka, ea, xlog=False, ylog=False, xbase=10, ybase=10):
        self.ka = np.asarray(ka, dtype=float)
        self.ea = np.asarray(ea, dtype=float)
        self.xlog = xlog
        self.ylog = ylog

        #TODO: adjust for different logarithm bases.


        if np.any(np.diff(self.ka) <= 0):
            raise ValueError("'ka' must be in monotonically increasing order.")

        if np.any(np.diff(self.ea) <= 0):
            raise ValueError("'ea' must be in monotomically increasing order.")

        if len(ka)!=len(ea):
            raise IndexError("'ka' and 'ea' must be the same length.")

        self.log_ka = np.log10(self.ka)
        self.log_ea = np.log10(self.ea)

    def e_from_k(self, k, **kwargs):
        """void ratio from permeability

        Parameters
        ----------
        k : float
            current permeability

        Returns
        -------
        e : float
            void ratio corresponding to current permeability

        Examples
        --------
        >>> a = PwiseLinearPermeabilityModel(ka=np.array([1.0, 2.0]), ea=np.array([5, 7.0]))
        >>> a.e_from_k(1.25)
        5.5

        Array Inputs
        >>> a.e_from_k(np.array([1.25, 1.75]))
        array([ 5.5,  6.5])

        Logarithmic permeability scale:
        >>> a = PwiseLinearPermeabilityModel(ka=np.array([1.0, 2.0]),
        ... ea=np.array([5, 7.0]), xlog=True)
        >>> a.e_from_k(1.25)
        5.643...
        >>> a.e_from_k(np.array([1.25, 1.75]))
        array([ 5.643...,  6.614...])

        Logarithmic void ratio scale:
        >>> a = PwiseLinearPermeabilityModel(ka=np.array([1.0, 2.0]),
        ... ea=np.array([5, 7.0]), ylog=True)
        >>> a.e_from_k(1.25)
        5.4387...
        >>> a.e_from_k(np.array([1.25, 1.75]))
        array([ 5.4387...,  6.435...])

        Logarithmic permeability and void ratio scale:
        >>> a = PwiseLinearPermeabilityModel(ka=np.array([1.0, 2.0]),
        ... ea=np.array([5, 7.0]), xlog=True, ylog=True)
        >>> a.e_from_k(1.25)
        5.572...
        >>> a.e_from_k(np.array([1.25, 1.75]))
        array([ 5.572...,  6.560...])



        """


        if self.xlog:
            ka = self.log_ka
            k = np.log10(k)
        else:
            ka = self.ka

        if self.ylog:
            ea = self.log_ea
        else:
            ea = self.ea

        e = np.interp(k, ka, ea)

        if self.ylog:
            e = np.power(10, e)

        return e



        return self.ea + self.Ck * np.log10(k/self.ka)

    def k_from_e(self, e, **kwargs):
        """permeability from void ratio

        Parameters
        ----------
        e : float
            void ratio

        Returns
        -------
        k : float
            permeability corresponding to current void ratio

        Examples
        --------
        >>> a = PwiseLinearPermeabilityModel(ka=np.array([1.0, 2.0]), ea=np.array([5, 7.0]))
        >>> a.k_from_e(5.5)
        1.25

        Array Inputs
        >>> a.k_from_e(np.array([5.5, 6.5]))
        array([ 1.25,  1.75])

        Logarithmic permeability scale:
        >>> a = PwiseLinearPermeabilityModel(ka=np.array([1.0, 2.0]),
        ... ea=np.array([5, 7.0]), xlog=True)
        >>> a.k_from_e(5.644)
        1.25...

        >>> a.k_from_e(np.array([5.644, 6.615]))
        array([ 1.25...,  1.75...])

        Logarithmic void ratio scale:
        >>> a = PwiseLinearPermeabilityModel(ka=np.array([1.0, 2.0]),
        ... ea=np.array([5, 7.0]), ylog=True)
        >>> a.k_from_e(5.4388)
        1.25...

        >>> a.k_from_e(np.array([5.4388, 6.436]))
        array([ 1.25...,  1.75...])

        Logarithmic permeability and void ratio scale:
        >>> a = PwiseLinearPermeabilityModel(ka=np.array([1.0, 2.0]),
        ... ea=np.array([5, 7.0]), xlog=True, ylog=True)
        >>> a.k_from_e(5.573)
        1.25...
        >>> a.k_from_e(np.array([5.573, 6.561]))
        array([ 1.25...,  1.75...])


        """

        if self.xlog:
            ka = self.log_ka
        else:
            ka = self.ka

        if self.ylog:
            ea = self.log_ea
            e = np.log10(e)
        else:
            ea = self.ea

        k = np.interp(e, ea, ka)

        if self.xlog:
            k = np.power(10, k)

        return k





    def e_and_k_for_plotting(self, **kwargs):
        """void ratio and permeability that plot the model

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
            `npts` permeability, and void ratio values between
            `xmin` and `ymin`

        """

        npts = kwargs.get('n', 100)
        xmin, xmax = np.min(self.ka), np.max(self.ka)

        x = np.linspace(xmin, xmax, npts)

        y = self.e_from_k(x)
        return x, y


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest', '--doctest-options=+ELLIPSIS'])
    pass
    a = CkPermeabilityModel(Ck=1.5, ka=10, ea=4)
#    b = a.e_from_k(8.0)
#    print(b)
    a.plot_model()
    plt.show()