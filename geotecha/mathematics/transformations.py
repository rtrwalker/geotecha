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

"""this module implements some transformations"""

from __future__ import print_function, division
import numpy as np

def depth_to_reduced_level(z, H = 1.0, rlzero=None):
    """convert normalised depth to rl, or simply actual depth


    Parameters
    ----------
    z : float, or ndarray
        normalised depth
    H : float, optional
        height of soil layer.  Default `H` = 1.0
    rlzero : float, optional
        reduced level of soil surface. default = None.  If rlzero=None then
        function will return non normalised depth H*z, otherwise it will
        return rl=rlzero - H * z

    Returns
    -------
    out: float or ndarray
        actual depth or reduced level

    """

    if not rlzero is None:
        return rlzero - z * H
    else:
        return z * H


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])