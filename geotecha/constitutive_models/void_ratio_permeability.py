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
Relationships between permeability and void-ratio
"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import functools

class PermeabilityVoidRatioRelationship(object):
    """Base class for defining permeability void ratio relationships"""

    def e_from_k(self, estress, **kwargs):
        """Void ratio from permeability"""
        raise NotImplementedError("e_from_k must be implemented")

    def k_from_e(self, e, **kwargs):
        """Permeability from void ratio"""
        raise NotImplementedError("k_from_e must be implemented")

    def e_and_k_for_plotting(self, **kwargs):
        """Void ratio and permeability that plot the method"""
        # should return a tuple of x and y values
        # x-values are peremability, y-values are void ratio.
        raise NotImplementedError("e_and_k_for_plotting must be implemented")

    def plot_model(self, **kwargs):
        """Plot the void ratio-permeability"""
        ax = kwargs.pop('ax', plt.gca())
        x, y = self.e_and_k_for_plotting(**kwargs)

        ax.plot(x, y)
        return


class CkPermeabilityModel(PermeabilityVoidRatioRelationship):
    """Semi-log void ratio permeability relationship

    Parameters
    ----------
    Ck : float
        Slope of void-ratio vs permeability.
    ka, ea : float
        Permeability and void ratio specifying a point on e-k line.


    """

    def __init__(self, Ck, ka, ea):
        self.Ck = Ck
        self.ka = ka
        self.ea = ea

    def e_from_k(self, k, **kwargs):
        """Void ratio from permeability

        Parameters
        ----------
        k : float
            Current permeability.

        Returns
        -------
        e : float
            Void ratio corresponding to current permeability.

        Examples
        --------
        >>> a = CkPermeabilityModel(Ck=1.5, ka=10, ea=4)
        >>> a.e_from_k(8.0)
        3.854...

        Array Inputs:

        >>> a.e_from_k(np.array([8.0, 4.0]))
        array([ 3.854...,  3.403...])


        """

        return self.ea + self.Ck * np.log10(k/self.ka)

    def k_from_e(self, e, **kwargs):
        """Permeability from void ratio

        Parameters
        ----------
        e : float
            Void ratio.

        Returns
        -------
        k : float
            Permeability corresponding to current void ratio.

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
        """Void ratio and permeability that plot the model

        Parameters
        ----------
        npts : int, optional
            Number of points to return.  Default npts=100.
        xmin, xmax : float, optional
            Range of x (i.e. effective stress) values from which
            to return points. Default xmin=1, xmax=100.

        Returns
        -------
        x, y : 1d ndarray
            `npts` permeability, and void ratio values between
            `xmin` and `xmax`.

        """

        npts = kwargs.get('n', 100)
        xmin, xmax = kwargs.get('xmin', 1.0), kwargs.get('xmax', 100)

        x = np.linspace(xmin, xmax, npts)
        y = self.e_from_k(x)
        return x, y

class ConstantPermeabilityModel(PermeabilityVoidRatioRelationship):
    """Permeability constant with void ratio

    Note that the method e_from_k is meaningless because there are multiple
    e values for single k value.

    Parameters
    ----------
    ka : float
        Permeability.


    """

    def __init__(self, ka):
        self.ka = ka

    def e_from_k(self, k, **kwarks):
        """Void ratio from permeability, parameters are irrelevant."""
        raise ValueError("e_from_k is meaningless for a constant permeability model")


    def k_from_e(self, e, **kwargs):
        """Permeability from void ratio

        Parameters
        ----------
        e : float
            Void ratio.

        Returns
        -------
        ka : float
            The constant permeability.

        Examples
        --------
        >>> a = ConstantPermeabilityModel(ka=2.5)
        >>> a.k_from_e(3.855)
        2.5

        Array Inputs:

        >>> a.k_from_e(np.array([3.855, 3.404]))
        array([ 2.5,  2.5])

        """

        return np.ones_like(e, dtype=float) * self.ka


    def e_and_k_for_plotting(self, **kwargs):
        """Void ratio and permeability that plot the model

        Parameters
        ----------
        npts : int, optional
            Number of points to return.  Default npts=100
        xmin, xmax : float, optional
            Range of x (i.e. effective stress) values from which
            to return points. Default xmin=1, xmax=100.

        Returns
        -------
        x, y : 1d ndarray
            `npts` permeability, and void ratio values between
            `xmin` and `xmax`.

        """

        npts = kwargs.get('n', 100)
        xmin, xmax = kwargs.get('xmin', 1.0), kwargs.get('xmax', 100)

        x = np.linspace(xmin, xmax, npts)
        y = self.e_from_k(x)
        return x, y


class PwiseLinearPermeabilityModel(PermeabilityVoidRatioRelationship):
    """Piecewise linear ratio permeability relationship

    x and y data can be interpolated natural-natural, natural-log10,
    log10-natural, or log10, log10.

    Parameters
    ----------
    ka, ea : 1d array
        Permeability values and void ratio values defining a one-to-one
        relationship.
    xlog, ylog : True/False, Optional
        If True then interpolation on each axis is assumed to be logarithmic
        with base 10. x refers to the peremability axis, y to the void ratio
        axis.  Default xlog=ylog=False.

    """

    def __init__(self, ka, ea, xlog=False, ylog=False, xbase=10, ybase=10):
        self.ka = np.asarray(ka, dtype=float)
        self.ea = np.asarray(ea, dtype=float)
        self.xlog = xlog
        self.ylog = ylog

        #TODO: adjust for different logarithm bases.

        self.ka_slice = slice(None)
        self.ea_slice = slice(None)

        if np.any(np.diff(self.ka) <= 0):
            # ka is in decreasing order
            # reverse the slice for e_from_k interpolation
            self.ka_slice = slice(None, None, -1)
#            raise ValueError("'ka' must be in monotonically increasing order.")

        if np.any(np.diff(self.ea) <= 0):
            # ea is in decreasing order
            # reverse the slice for k_from_e interpolation
            self.ea_slice = slice(None, None, -1)
#            raise ValueError("'ea' must be in monotomically increasing order.")

        if len(ka)!=len(ea):
            raise IndexError("'ka' and 'ea' must be the same length.")

        self.log_ka = np.log10(self.ka)
        self.log_ea = np.log10(self.ea)

    def e_from_k(self, k, **kwargs):
        """Void ratio from permeability

        Parameters
        ----------
        k : float
            Current permeability.

        Returns
        -------
        e : float
            Void ratio corresponding to current permeability.

        Examples
        --------
        >>> a = PwiseLinearPermeabilityModel(ka=np.array([1.0, 2.0]), ea=np.array([5, 7.0]))
        >>> a.e_from_k(1.25)
        5.5

        Array Inputs:

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

        Increasing vs decreasing inputs:

        >>> ea = np.arange(1,10)
        >>> ka = 3 * ea
        >>> np.isclose(PwiseLinearPermeabilityModel(ka, ea).e_from_k(7.2),
        ... PwiseLinearPermeabilityModel(ka[::-1], ea[::-1]).e_from_k(7.2))
        True

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

        e = np.interp(k, ka[self.ka_slice], ea[self.ka_slice])

        if self.ylog:
            e = np.power(10, e)

        return e





    def k_from_e(self, e, **kwargs):
        """Permeability from void ratio

        Parameters
        ----------
        e : float
            Void ratio.

        Returns
        -------
        k : float
            Permeability corresponding to current void ratio.

        Examples
        --------
        >>> a = PwiseLinearPermeabilityModel(ka=np.array([1.0, 2.0]), ea=np.array([5, 7.0]))
        >>> a.k_from_e(5.5)
        1.25

        Array Inputs:

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

        Increasing vs decreasing inputs:

        >>> ea = np.arange(1,10)
        >>> ka = 3 * ea
        >>> np.isclose(PwiseLinearPermeabilityModel(ka, ea).k_from_e(3.0),
        ... PwiseLinearPermeabilityModel(ka[::-1], ea[::-1]).k_from_e(3.0))
        True

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

        k = np.interp(e, ea[self.ea_slice], ka[self.ea_slice])

        if self.xlog:
            k = np.power(10, k)

        return k





    def e_and_k_for_plotting(self, **kwargs):
        """Void ratio and permeability that plot the model

        Parameters
        ----------
        npts : int, optional
            Number of points to return.  Default npts=100.
        xmin, xmax : float, optional
            Range of x (i.e. effective stress) values from which
            to return points. Default min and max of model `ka`.

        Returns
        -------
        x, y : 1d ndarray
            `npts` permeability, and void ratio values between
            `xmin` and `xmax`.

        """

        npts = kwargs.get('n', 100)
        xmin, xmax = np.min(self.ka), np.max(self.ka)

        x = np.linspace(xmin, xmax, npts)

        y = self.e_from_k(x)
        return x, y




class FunctionPermeabilityModel(PermeabilityVoidRatioRelationship):
    """Functional definition of permeability void-ratio realationship

    User provides python functions.

    Parameters
    ----------
    fn_e_from_k: callable object
        Function to obtain void ratio from permeability.  fn_e_from_k should
        be the inverse function of fn_k_from_e.
    fn_k_from_e : callable object
        Function to obtain peremability from void ratio. fn_k_from_e should
        be the inverse function of fn_e_from_k.
    *args, **kwargs : anything
        Positional and keyword arguments to be passed to the
        fn_e_from_k, fn_k_from_efunctions.  Note
        that any additional args and kwargs passed to the functions will be
        appended to the args, and kwargs.  You may get into a mess for
        example with e_from_k because normally the first postional
        argument passed to such fucntions is k.  If you add your own
        positional arguments, then k will be after you own arguments.
        Just be aware that the usual way to call methods of the base object
        `PermeabilityVoidRatioRelationship` a single positonal arguement,
        e.g. k, e, followed by any required keyword arguments.

    Notes
    -----
    Any function should be able to accept additonal keywords.

    Examples
    --------
    >>> def efk(k, b=2):
    ...     return k * b
    >>> def kfe(e, b=2):
    ...     return e / b
    >>> a = FunctionPermeabilityModel(efk, kfe, b=5)
    >>> a.e_from_k(3)
    15
    >>> a.k_from_e(15)
    3.0


    """

    def __init__(self,
                 fn_e_from_k,
                 fn_k_from_e,
                 *args,
                 **kwargs):

        self.fn_e_from_k = fn_e_from_k
        self.fn_k_from_e = fn_k_from_e
        self.args = args
        self.kwargs = kwargs

        self.e_from_k = functools.partial(fn_e_from_k,
                                               *args,
                                               **kwargs)
        self.k_from_e = functools.partial(fn_k_from_e,
                                               *args,
                                               **kwargs)

        return



if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest', '--doctest-options=+ELLIPSIS'])
    pass
#    a = CkPermeabilityModel(Ck=1.5, ka=10, ea=4)
##    b = a.e_from_k(8.0)
##    print(b)
#    a.plot_model()
#    plt.show()



