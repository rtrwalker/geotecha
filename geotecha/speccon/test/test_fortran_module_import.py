# geotecha - A software suite for geotechncial engineering
# Copyright (C) 2018  Rohan T. Walker (rtrwalker@gmail.com)
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
"""Test fortran extension model import, usually code will default to using
some other routines if fortran import fails.

"""
from __future__ import division, print_function

from nose import with_setup
from nose.tools.trivial import assert_almost_equal
from nose.tools.trivial import assert_raises
from nose.tools.trivial import ok_

import unittest
from numpy.testing import assert_allclose


def test_ext_integrals_import():
    """Check geotecha.speccon.ext_integrals can be imported."""
#    assert_raises(ImportError, import_ext_integrals)
#    assert(import_ext_integrals)
#    import geotecha.speccon.ext_integrals as ext_integ
    try:
        import geotecha.speccon.ext_integrals as ext_integ
    except ImportError:
        ok_(False, "Failed to import geotecha.speccon.ext_integrals (fortran), numpy versions will be used instead.")

def test_ext_epus_import():
    """Check geotecha.constitutive_models.epus_ext can be imported."""

    try:
        import geotecha.constitutive_models.epus_ext as epus_ext
    except ImportError:
        ok_(False, "Failed to import geotecha.constitutive_models.epus_ext (fortran), scalar versions will be used instead.")





if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])
#    test_ext_integrals_import()