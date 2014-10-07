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
        ------------
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
        ------------
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


if __name__ == '__main__':
#    import nose
#    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest', '--doctest-options=+ELLIPSIS'])
    pass
    a = CkPermeabilityModel(Ck=1.5, ka=10, ea=4)
#    b = a.e_from_k(8.0)
#    print(b)
    a.plot_model()
    plt.show()