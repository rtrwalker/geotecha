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

"""Relationships between velocity and hydraulic gradient"""


from __future__ import print_function, division

import numpy as np
from numpy import (isscalar, asarray)
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize
from collections import OrderedDict

from geotecha.mathematics import root_finding

class OneDimensionalFlowRelationship(object):
    """Base class for defining velocity vs hydraulic gradient relationships

    Attributes
    ----------
    non_Darcy: True
        A class variable indicating if model has a non-Darcian flow
        relationship.  Can be overridden in subclasses.  Basically
        the Darcian flow equations are simple and it might be simpler to
        bypass the use of a OneDimensionalFlowRelationship.

    """

    non_Darcy = True

    def v_from_i(self, hyd_grad, **kwargs):
        """Velocity from hydraulic gradient"""
        #Note this should work for +ve and -ve hydraulic gradients
        raise NotImplementedError("v_from_i must be implemented.")

    def i_from_v(self, velocity, **kwargs):
        """Hydraulic gradient from velocity """
        #Note this should work for +ve and -ve velocities
        raise NotImplementedError("i_from_v must be implemented.")

    def dv_di(self, hyd_grad, **kwargs):
        """Slope of velocity vs hydraulic gradient relationship"""
        #Note this should work for +ve and -ve hydraulic gradients
        raise NotImplementedError("dv_di must be implemented.")

    def vdrain_strain_rate(self, eta, head, **kwargs):
        """Vertical drain strain rate as based on the eta method"""
        raise NotImplementedError("vdrain_strain_rate must be implemented.")

    def v_and_i_for_plotting(self, **kwargs):
        """Velocity and hydraulic gradient that that plot the relationship"""
        # should return a tuple of x and y values
        # x-values are hydraulic gradient, y-values are velocity.
        raise NotImplementedError("v_and_i_for_plotting must be implemented.")

    def plot_model(self, **kwargs):
        """Plot the velocity-hydraulic gradient relationship"""
        ax = kwargs.pop('ax', plt.gca())
        x, y = self.v_and_i_for_plotting(**kwargs)
#        label = "k={:g}".format(self.k)
        label=None
        ax.plot(x, y, label=label)
        ax.set_xlabel('Hydraulic gradient, i')
        ax.set_ylabel('Flow velocity, v')
        return


    def __str__(self):
        """Need to define self._attribute_list, a lsit of attribute names"""

        #self._attribute_list is a list str containing names of attributes
        attribute_list = getattr(self, '_attribute_list', [])



        attribute_string = "\n".join(["{} = {:g}".format(v,
                                      getattr(self, v,"not defined.")) for
                                      v in attribute_list])

        return attribute_string


class DarcyFlowModel(OneDimensionalFlowRelationship):
    """Darcian flow model

    Parameters
    ----------
    k : float, optional
        Darcian permeability. Default k=1.


    Notes
    -----
    Darcian flow is described by:

    .. math:: v = ki

    """

    non_Darcy = False

    def __init__(self, k=1.0):
        self._attribute_list = ['k']
        self.k = k

    def v_from_i(self, hyd_grad, **kwargs):
        """Velocity from hydraulic gradient

        Parameters
        ----------
        hyd_grad : float
            Hydraulic gradient.
        **kwargs : any
            Any keyword arguments are ignored.

        Returns
        -------
        v : float
            Flow velocity.


        Examples
        --------
        >>> a = DarcyFlowModel(k=3)
        >>> a.v_from_i(8.0)
        24.0
        >>> a.v_from_i(-8.0)
        -24.0
        >>> a.v_from_i(np.array([5.0, 8.0]))
        array([15., 24.])


        """
        return self.k * hyd_grad

    def i_from_v(self, velocity, **kwargs):
        """Hydraulic gradient from velocity

        Parameters
        ----------
        v : float
            Flow velocity.
        **kwargs : any
            Any keyword arguments are ignored.

        Returns
        -------
        hyd_grad : float
            Hydraulic gradient.

        Examples
        --------
        >>> a = DarcyFlowModel(k=3)
        >>> a.i_from_v(24.0)
        8.0
        >>> a.i_from_v(-24.0)
        -8.0
        >>> a.i_from_v(np.array([15, 24]))
        array([5., 8.])


        """
        return velocity / self.k

    def dv_di(self, hyd_grad, **kwargs):
        """Slope of velocity vs hydraulic gradient relationship

        Parameters
        ----------
        hyd_grad : float
            Hydraulic gradient.
        **kwargs : any
            Any keyword arguments are ignored.

        Returns
        -------
        slope : float
            slope of velocity-hydraulic gradient relationship at `hyd_grad`.

        Examples
        --------
        >>> a = DarcyFlowModel(k=3)
        >>> a.dv_di(8.0)
        3.0
        >>> a.dv_di(-8.0)
        -3.0
        >>> a.dv_di(np.array([5.0, 8.0]))
        array([3., 3.])


        """

        return np.ones_like(hyd_grad, dtype=float) * self.k * np.sign(hyd_grad)

    def vdrain_strain_rate(self, eta, head, **kwargs):
        """Vertical drain strain rate as based on the eta method""

        [strain rate] = head * self.k * eta


        Parameters
        ----------
        eta : float
            Value of vertical drain geometry, peremability parameter. This
            value should be calculated based on Darcy's law (see
            geotecha.consolidation.smearzones.drain_eta)
        head : float
            Hydraulic head driving the flow.  For vertical drains this is
            usually the difference between the average head in the soil and
            the head in the drain.
        **kwargs : any
            Any additional keyword arguments are ignored.

        Returns
        -------
        strain_rate : float
            Strain rate based on eta method.

        See Also
        --------
        geotecha.consolidation.smearzones : Functions to determine eta.

        Examples
        --------
        >>> a = DarcyFlowModel(k=3)
        >>> a.vdrain_strain_rate(eta=2.5, head=4)
        30.0

        """

        return head * self.k * eta

    def v_and_i_for_plotting(self, **kwargs):
        """Velocity and hydraulic gradient that plot the relationship

        Parameters
        ----------
        npts : int, optional
            Number of points to return.  Default npts=100.
        xmin, xmax : float, optional
            Range of x (i.e. hydraulic gradient) values from which
            to return points. Default xmin=0, xmax=50.

        Returns
        -------
        x, y : 1d ndarray
            `npts` permeability, and void ratio values between
            `xmin` and `xmax`.

        """

        npts = kwargs.get('npts', 100)
        xmin, xmax = kwargs.get('xmin', 0), kwargs.get('xmax', 50)

        x = np.linspace(xmin, xmax, npts)
        y = self.v_from_i(x)
        return x, y


class HansboNonDarcianFlowModel(OneDimensionalFlowRelationship):
    """Hansbo non-darcian flow model

    Various combinations of parameters can be used to specify the model.
    Basically three parameters are required, one of which must be kstar
    or klinear.  If n==1 or i0=0, or iL=0 then the model will reduce to Darcy
    flow model with peremability equal to klinear (or kstar if klinear is not
    defined).

    Parameters
    ----------
    klinear : float, optional
        This is the slope of the linear portion of the v-i relationship
        for i>iL. klinear = kstar * n * iL**(n-1).  Default klinear=None.
        `klinear` and `kstar` cannot both be None, you
        must use one or the other.
    kstar : float, optional
        Permability coefficient in the power law, v=kstar*i**n, flow section,
        i<iL. Default kstar=None. `klinear` and `kstar` cannot both be None.
    n : float, optional
        Exponent for non-Darcian portion of flow. n must be  greater than or
        equal to 1.  Default n=None.
    i0 : float, optional
        x-axis intercept of the linear portion of the flow law.
        Default i0=None. i0 must be greater or equal to zero.
    iL : float, optional
        Limiting hydraulic gradient, beyond which v-i relationship is linear.
        iL = i0 * n / (n -1).  Default iL=None.


    Notes
    -----
    Hansbo's Non-darcian flow relationship is defined by:

    .. math::

       v=\\left\\{\\begin{array}{lr}
        k^{\\ast}i^{n} & i<i_{L} \\\\
        k_{\\rm{linear}} \\left({i-i_0}\\right)
            & i\\geq i_{L} \\end{array}\\right.

    where,

    .. math:: k_{\\rm{linear}} = k^{\\ast}ni_L^\\left(n-1\\right)

    .. math:: i_L=\\frac{i_0 n}{\\left(n-1\\right)}


    Examples
    --------
    These examples show the various ways to define the model:
    >>> p = dict(kstar=2, n=1.3, iL=16.2402611294, i0=3.7477525683, klinear=6)
    >>> a = HansboNonDarcianFlowModel(kstar=p['kstar'], n=p['n'], iL=p['iL'])
    >>> print(a)
    kstar = 2
    n = 1.3
    iL = 16.240...
    i0 = 3.747...
    klinear = 6
    >>> a = HansboNonDarcianFlowModel(kstar=p['kstar'], n=p['n'], i0=p['i0'])
    >>> print(a)
    kstar = 2
    n = 1.3
    iL = 16.240...
    i0 = 3.747...
    klinear = 6
    >>> a = HansboNonDarcianFlowModel(kstar=p['kstar'], iL=p['iL'], i0=p['i0'])
    >>> print(a)
    kstar = 2
    n = 1.3
    iL = 16.240...
    i0 = 3.747...
    klinear = 6
    >>> a = HansboNonDarcianFlowModel(kstar=p['kstar'], iL=p['iL'], klinear=p['klinear'])
    >>> print(a)
    kstar = 2
    n = 1.3
    iL = 16.240...
    i0 = 3.747...
    >>> a = HansboNonDarcianFlowModel(kstar=p['kstar'], i0=p['i0'], klinear=p['klinear'])
    >>> print(a)
    kstar = 2
    n = 1.3
    iL = 16.240...
    i0 = 3.747...
    klinear = 6
    >>> a = HansboNonDarcianFlowModel(n=p['n'], iL=p['iL'], klinear=p['klinear'])
    >>> print(a)
    kstar = 2
    n = 1.3
    iL = 16.240...
    i0 = 3.747...
    klinear = 6
    >>> a = HansboNonDarcianFlowModel(n=p['n'], i0=p['i0'], klinear=p['klinear'])
    >>> print(a)
    kstar = 2
    n = 1.3
    iL = 16.240...
    i0 = 3.747...
    klinear = 6
    >>> a = HansboNonDarcianFlowModel(iL=p['iL'], i0=p['i0'], klinear=p['klinear'])
    >>> print(a)
    kstar = 2
    n = 1.3
    iL = 16.240...
    i0 = 3.747...
    klinear = 6


    If the behaviour is not as you expect when i0=0, iL=0, n==1 then use very
    small values of i0 or iL, n just greater than 1.


    """

    non_Darcy = True

    def __init__(self, klinear=None, kstar=None, n=None, i0=None, iL=None):




        d = OrderedDict()
        d['kstar']=kstar
        d['n']=n
        d['iL']=iL
        d['i0']=i0
        d['klinear'] = klinear
        self._attribute_list = list(d.keys())
        params = [v for v in d if not d[v] is None]
#        self._attribute_list = ('kstar', 'n', 'iL', 'i0', 'klinear')
#        params = [v for v in self._attribute_list if not eval(v) is None]
        if len(params) != 3 or (kstar is None and klinear is None):
            raise TypeError("Wrong number of parameters to define model.'"
                            "Need three only of ['kstar', 'n', 'iL', "
                            "i0', 'klinear'] at least one of which is "
                            "kstar or klinear. You have {}.".format(params))


        if len([v for v in (kstar, n, iL) if not v is None]) == 3:

            if n == 1 or iL == 0:
                self.kstar = kstar
                self.n = 1
                self.iL = 0
                self.i0 = 0
                self.k_linear = kstar
            else:
                self.kstar = kstar
                self.n = n
                self.iL = iL
                self.i0 = (n - 1) / n * iL
                self.klinear = kstar * n * iL**(n - 1)
        elif len([v for v in (kstar, n, i0) if not v is None]) == 3:
            if n == 1 or i0 == 0:
                self.kstar = kstar
                self.n = 1
                self.iL = 0
                self.i0 = 0
                self.klinear = kstar
            else:
                self.kstar = kstar
                self.n = n
                self.iL = i0 * n / (n - 1)
                self.i0 = i0
                self.klinear = kstar * n * self.iL**(n - 1)
        elif len([v for v in (kstar, n, klinear) if not v is None]) == 3:
            if n == 1:
                self.kstar = klinear
                self.n = 1
                self.iL = 0
                self.i0 = 0
                self.klinear = klinear
            else:
                self.kstar = kstar
                self.n = n
                self.iL = (klinear / (kstar * n)) ** (1/(n - 1))
                self.i0 = self.iL * (n - 1) / n
                self.klinear = klinear
        elif len([v for v in (kstar, iL, i0) if not v is None]) == 3:
            if iL == i0:
                self.kstar = kstar
                self.n = 1
                self.iL = 0
                self.i0 = 0
                self.klinear = kstar
            else:
                self.kstar = kstar
                self.n = iL / (iL - i0)
                self.iL = iL
                self.i0 = i0
                self.klinear = kstar * self.n * iL **(self.n - 1)
        elif len([v for v in (kstar, iL, klinear) if not v is None]) == 3:
            if iL == 0:
                self.kstar = klinear
                self.n = 1
                self.iL = 0
                self.i0 = 0
                self.klinear = klinear
            else:
                self.kstar = kstar

                def fn(_n, _klinear, _kstar, _iL):
                    return np.log(_klinear/_kstar/_n) / np.log(_iL) + 1



#                self.n = scipy.optimize.fixed_point(fn,
#                                                    1.0001,
#                                                    args=(klinear, kstar, iL))
                # note as at 20151106 scipy.optimize.fixed_point may not
                # converge for this function because convergence acceleration
                # is used.  Use my modified version instead:
                self.n = root_finding.fixed_point_no_accel(fn,
                                                           1.00001,
                                                           args=(klinear, kstar, iL))

#                self.n = klinear / kstar / iL**(n - 1)
                self.iL = iL
                self.i0 = iL * (self.n - 1) / self.n
                self.klinear = klinear
        elif len([v for v in (kstar, i0, klinear) if not v is None]) == 3:
            if i0 == 0:
                self.kstar = klinear

                self.n = 1
                self.iL = 0
                self.i0 = 0
                self.klinear = klinear
            else:
                self.kstar = kstar

                def fn(_n, _klinear, _kstar, _i0):

                    return np.log(_klinear/_kstar/_n) / np.log((_i0 *_n/(_n - 1))) + 1


#                self.n = scipy.optimize.fixed_point(fn,
#                                                    1.00001,
#                                                    args=(klinear, kstar, i0))
                # note as at 20151106 scipy.optimize.fixed_point may not
                # converge for this function because convergence acceleration
                # is used.  Use my modified version instead:
                self.n = root_finding.fixed_point_no_accel(fn,
                                                           1.00001,
                                                           args=(klinear, kstar, i0))

                self.iL = i0 * self.n / (self.n - 1)
                self.i0 = i0
                self.klinear = klinear
        #elif len([v for v in (n, iL, i0) if not v is None]) == 3:
        #    #need kstar or klinear

        elif len([v for v in (n, iL, klinear) if not v is None]) == 3:
            if n == 1 or iL == 0:
                self.n = 1
                self.iL = 0
                self.i0 = 0
                self.klinear = klinear
                self.kstar = klinear
            else:
                self.n = n
                self.iL = iL
                self.i0 = iL * (n - 1) / n
                self.klinear = klinear
                self.kstar = klinear / (n * iL**(n - 1))
        elif len([v for v in (n, i0, klinear) if not v is None]) == 3:
            if n == 1 or i0 == 0:
                self.n = 1
                self.iL = 0
                self.i0 = 0
                self.klinear = klinear
                self.kstar = klinear
            else:
                self.n = n
                self.iL = i0 * n / (n - 1)
                self.i0 = i0
                self.klinear = klinear
                self.kstar = klinear / (self.n * self.iL**(self.n - 1))
        elif len([v for v in (iL, i0, klinear) if not v is None]) == 3:
            self.n = iL / (iL - i0)
            self.iL = iL
            self.i0 = i0
            self.klinear = klinear
            self.kstar = klinear / (self.n * iL**(self.n - 1))


        if self.iL < self.i0:
            raise ValueError('iL must be greater than i0.')
        if self.iL <= self.i0 and self.i0 > 0:
            raise ValueError('iL must be greater than i0.')
        if self.n < 1:
            raise ValueError('n must be greater than 0.')

        self.vL = self.klinear * (self.iL - self.i0)


    def v_from_i(self, hyd_grad, **kwargs):
        """Velocity from hydraulic gradient

        Parameters
        ----------
        hyd_grad : float
            Hydraulic gradient.
        **kwargs : any
            Any keyword arguments are ignored.

        Returns
        -------
        v : float
            Flow velocity.


        Examples
        --------
        >>> a = HansboNonDarcianFlowModel(kstar=2, n=1.3, iL=16.2402611294)
        >>> a.v_from_i(8.0)
        29.857...
        >>> a.v_from_i(-8.0)
        -29.857...
        >>> a.v_from_i(np.array([8.0, 20.0]))
        array([29.857..., 97.513...])


        """

        abs_hyd_grad = abs(hyd_grad)
        return np.where(abs_hyd_grad >= self.iL,
                        self.klinear * (abs_hyd_grad - self.i0),
                        self.kstar * abs_hyd_grad**self.n)*np.sign(hyd_grad)*1

    def i_from_v(self, velocity, **kwargs):
        """Hydraulic gradient from velocity

        Parameters
        ----------
        v : float
            Flow velocity.
        **kwargs : any
            Any keyword arguments are ignored.

        Returns
        -------
        velocity : float
            Flow velocity.

        Examples
        --------
        >>> a = HansboNonDarcianFlowModel(kstar=2, n=1.3, iL=16.2402611294)
        >>> a.i_from_v(29.858)
        8.0...
        >>> a.i_from_v(-29.858)
        -8.0...
        >>> a.i_from_v(np.array([29.858, 97.514]))
        array([ 8.0..., 20.0...])


        """

        absvelocity = np.abs(velocity)
        return np.where(absvelocity >= self.vL,
                        absvelocity / self.klinear  + self.i0,
                        (absvelocity / self.kstar)**(1/self.n)) * np.sign(velocity)*1

    def dv_di(self, hyd_grad, **kwargs):
        """Slope of velocity vs hydraulic gradient relationship

        Parameters
        ----------
        hyd_grad : float
            Hydraulic gradient.
        **kwargs : any
            Any keyword arguments are ignored.

        Returns
        -------
        slope : float
            slope of velocity-hydraulic gradient relationship at `hyd_grad`.

        Examples
        --------
        >>> a = HansboNonDarcianFlowModel(kstar=2, n=1.3, iL=16.2402611294)
        >>> a.dv_di(8.0)
        4.851...
        >>> a.dv_di(-8.0)
        -4.851...
        >>> a.dv_di(np.array([8.0, 20.0]))
        array([4.851..., 6...])

        """

        abs_hyd_grad = abs(hyd_grad)
        return np.where(abs_hyd_grad >= self.iL,
                        self.klinear,
                        self.kstar * self.n * abs_hyd_grad**(self.n - 1)) * np.sign(hyd_grad)*1

    def vdrain_strain_rate(self, eta, head, **kwargs):
        """Vertical drain strain rate as based on the eta method""

        [strain rate] = head**self.nflow * self.klinear * gamw**(nflow - 1) * eta


        Note that `vdrain_strain_rate` only uses the exponential portion of
        the Non-Darcian flow relationship.  If hydraulic gradients are
        greater than the limiting value iL then the flow rates will be
        overestimated.

        Parameters
        ----------
        eta : float
            Value of vertical drain geometry, peremability parameter. This
            value should be calculated based on Hansbo's non-Darcian flow
            model (see geotecha.consolidation.smearzones.non_darcy_drain_eta)
        head : float
            Hydraulic head driving the flow.  For vertical drains this is
            usually the difference between the average head in the soil and
            the head in the drain.
        gamw : float, optional
            Unit weight of water. Note that this gamw must be consistent with
            the value used to determine eta.  Default gamw=10.
        **kwargs : any
            Any additional keyword arguments, other than 'gamw', are ignored.

        Returns
        -------
        strain_rate : float
            Strain rate based on eta method.

        See Also
        --------
        geotecha.consolidation.smearzones : Functions to determine eta.

        Examples
        --------
        >>> a = HansboNonDarcianFlowModel(klinear=2, n=1.3, iL=16.2402611294)
        >>> a.vdrain_strain_rate(eta=0.1, head=4, gamw=10)
        2.419...


        """

        gamw = kwargs.get('gamw', 10)
        return head**self.n * self.klinear * gamw**(self.n - 1) * eta

    def v_and_i_for_plotting(self, **kwargs):
        """Velocity and hydraulic gradient that plot the relationship

        Parameters
        ----------
        npts : int, optional
            Number of points to return.  Default npts=100.
        xmin, xmax : float, optional
            Range of x (i.e. hydraulic gradient) values from which
            to return points. Default xmin=0, xmax=50.

        Returns
        -------
        x, y : 1d ndarray
            `npts` permeability, and void ratio values between
            `xmin` and `xmax`.

        """

        npts = kwargs.get('npts', 100)
        xmin, xmax = kwargs.get('xmin', 0), kwargs.get('xmax', 50)

        x = np.linspace(xmin, xmax, npts)
        y = self.v_from_i(x)
        return x, y


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest', '--doctest-options=+ELLIPSIS'])
    pass

    if 1:
        p = dict(kstar=2, n=1.3, iL=16.2402611294, i0=3.7477525683, klinear=6)
        a = HansboNonDarcianFlowModel(kstar=p['kstar'], iL=p['iL'], klinear=p['klinear'])
        print(a)
#        a.plot_model()
#        plt.show()
        b = HansboNonDarcianFlowModel(kstar=p['kstar'], i0=p['i0'], klinear=p['klinear'])
        print(b)
