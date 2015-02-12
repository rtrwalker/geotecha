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

"""Smear zones associated with vertical drain installation.

Smear zone permeability distributions etc.

"""

from __future__ import print_function, division

import numpy as np
from matplotlib import pyplot as plt
#import cmath
from numpy import log, sqrt

def mu_ideal(n, *args):
    """Smear zone permeability/geometry parameter for ideal drain (no smear)

    mu parameter in equal strain radial consolidation equations e.g.
    u = u0 * exp(-8*Th/mu)

    Parameters
    ----------
    n : float or ndarray of float
        Ratio of drain influence radius to drain radius (re/rw).
    args : anything
        `args` does not contribute to any calculations it is merely so you
        can have other arguments such as s and kappa which are used in other
        smear zone formulations.

    Returns
    -------
    mu : float
        Smear zone permeability/geometry parameter.

    Notes
    -----
    The :math:`\\mu` parameter is given by:

    .. math:: \\mu=\\frac{n^2}{\\left({n^2-1}\\right)}
              \\left({\\ln\\left({n}\\right)-\\frac{3}{4}}\\right)+
              \\frac{1}{\\left({n^2-1}\\right)}\\left({1-\\frac{1}{4n^2}}
              \\right)

    where:

    .. math:: n = \\frac{r_e}{r_w}

    :math:`r_w` is the drain radius, :math:`r_e` is the drain influence radius


    References
    ----------
    .. [1] Hansbo, S. 1981. "Consolidation of Fine-Grained Soils by
           Prefabricated Drains". In 10th ICSMFE, 3:677-82.
           Rotterdam-Boston: A.A. Balkema.


    """

    n = np.asarray(n) 
    if np.any(n <= 1):
        raise ValueError('n must be greater than 1. You have n = {}'.format(
            ', '.join([str(v) for v in np.atleast_1d(n)])))

    term1 = n**2 / (n**2 - 1) * (log(n) - 0.75)
    term2 = 1 / (n**2 - 1) * (1 - 1/(4 * n**2))
    mu = term1 + term2
    return mu


def mu_constant(n, s, kap):
    """Smear zone parameter for smear zone with constant permeability


    mu parameter in equal strain radial consolidation equations e.g.
    u = u0 * exp(-8*Th/mu)

    Parameters
    ----------
    n : float or ndarray of float
        Ratio of drain influence radius  to drain radius (re/rw).
    s : float or ndarray of float
        Ratio of smear zone radius to  drain radius (rs/rw)
    kap : float or ndarray of float.
        Ratio of undisturbed horizontal permeability to smear zone
        horizontal permeanility (kh / ks).

    Returns
    -------
    mu : float
        smear zone permeability/geometry parameter

    Notes
    -----
    The :math:`\\mu` parameter is given by:

    .. math:: \\mu=\\frac{n^2}{\\left({n^2-1}\\right)}
          \\left({\\ln\\left({\\frac{n}{s}}\\right)
          +\\kappa\\ln\\left({s}\\right)
          -\\frac{3}{4}}\\right)+
          \\frac{s^2}{\\left({n^2-1}\\right)}\\left({1-\\frac{s^2}{4n^2}}
          \\right)
          +\\frac{\\kappa}{\\left({n^2-1}\\right)}\\left({\\frac{s^4-1}{4n^2}}
          -s^2+1
          \\right)

    where:

    .. math:: n = \\frac{r_e}{r_w}

    .. math:: s = \\frac{r_s}{r_w}

    .. math:: \\kappa = \\frac{k_h}{k_s}

    :math:`r_w` is the drain radius, :math:`r_e` is the drain influence radius,
    :math:`r_s` is the smear zone radius, :math:`k_h` is the undisturbed
    horizontal permeability, :math:`k_s` is the smear zone horizontal
    permeability.

    References
    ----------
    .. [1] Hansbo, S. 1981. 'Consolidation of Fine-Grained Soils by
           Prefabricated Drains'. In 10th ICSMFE, 3:677-82.
           Rotterdam-Boston: A.A. Balkema.
    """
    n = np.asarray(n)
    s = np.asarray(s)
    kap = np.asarray(kap)
    if np.any(n <= 1.0):
        raise ValueError('n must be greater than 1. You have n = {}'.format(
            ', '.join([str(v) for v in np.atleast_1d(n)])))

    if np.any(s < 1.0):
        raise ValueError('s must be greater than 1. You have s = {}'.format(
            ', '.join([str(v) for v in np.atleast_1d(s)])))

    if np.any(kap <= 0.0):
        raise ValueError('kap must be greater than 0. You have kap = '
                         '{}'.format(', '.join([str(v) for v in
                                                np.atleast_1d(kap)])))

    if np.any(s > n):
        raise ValueError('s must be less than n. You have s = '
                         '{} and n = {}'.format(
                             ', '.join([str(v) for v in np.atleast_1d(s)]),
                             ', '.join([str(v) for v in np.atleast_1d(n)])))



    term1 = n**2 / (n**2 - 1) * (log(n/s) + kap * log(s) - 0.75)
    term2 = s**2 / (n**2 - 1) * (1 - s**2 /(4 * n**2))
    term3 = kap / (n**2 - 1) * ((s**4 - 1) / (4 * n**2) - s**2 +1)
    mu = term1 + term2 + term3
    return mu

def _sx(n, s):
    """Value of s=r/rw marking the start of overlapping linear smear zones

    `s` is usually larger than `n` when considering overlapping smear zones

    Parameters
    ----------
    n : float or ndarray of float
        Ratio of drain influence radius  to drain radius (re/rw).
    s : float or ndarray of float
        Ratio of smear zone radius to  drain radius (rs/rw).

    Returns
    -------
    sx : float or ndarray of float
        Value of s=r/rw marking the start of the overlapping zone


    Notes
    -----

    .. math:: \\kappa_X= 1+\\frac{\\kappa-1}{s-1}\\left({s_X-1}\\right)

    .. math:: s_X = 2n-s

    See also
    --------
    mu_overlapping_linear : uses _sx
    _kapx : used in mu_overlapping_linear

    """

    sx = 2 * n - s
    return sx


def _kapx(n, s, kap):
    """Value of kap=kh/ks for overlap part of intersecting linear smear zones

    Assumes `s` is greater than `n`.

    Parameters
    ----------
    n : float or ndarray of float
        Ratio of drain influence radius  to drain radius (re/rw).
    s : float or ndarray of float
        Ratio of smear zone radius to  drain radius (rs/rw)
    kap : float or ndarray of float.
        Ratio of undisturbed horizontal permeability to smear zone
        horizontal permeanility (kh / ks).

    Returns
    -------
    kapx : float
        Value of kap=kh/ks for overlap part of intersecting linear smear zones

    Notes
    -----

    .. math:: \\kappa_X= 1+\\frac{\\kappa-1}{s-1}\\left({s_X-1}\\right)

    .. math:: s_X = 2n-s


    See also
    --------
    mu_overlapping_linear : uses _kapx
    _sx : used in mu_overlapping_linear

    """
    sx = _sx(n, s)
    kapx = 1 + (kap - 1) / (s - 1) * (sx - 1)
    return kapx


def mu_overlapping_linear(n, s, kap):
    """\
    Smear zone parameter for smear zone with overlapping linear permeability


    mu parameter in equal strain radial consolidation equations e.g.
    u = u0 * exp(-8*Th/mu)

    Parameters
    ----------
    n : float or ndarray of float
        Ratio of drain influence radius  to drain radius (re/rw).
    s : float or ndarray of float
        Ratio of smear zone radius to  drain radius (rs/rw).
    kap : float or ndarray of float
        Ratio of undisturbed horizontal permeability to permeability at
        the drain-soil interface (kh / ks).

    Returns
    -------
    mu : float
        Smear zone permeability/geometry parameter.

    Notes
    -----
    The smear zone parameter :math:`\\mu` is given by:

    .. math::

          \\mu_X =
            \\left\\{\\begin{array}{lr}
                \\mu_L\\left({n,s,\\kappa}\\right) & n\\geq s \\\\
                \\frac{\\kappa}{\\kappa_X}\\mu_L
                    \\left({n, s_X,\\kappa_x}\\right)
                    & \\frac{s+1}{2}<n<s \\\\
                \\frac{\\kappa}{\\kappa_X}\\mu_I
                    \\left({n}\\right) & n\\leq \\frac{s+1}{2}
             \\end{array}\\right.


    where :math:`\\mu_L` is the :math:`\\mu` parameter for non_overlapping
    smear zones with linear permeability, :math:`\\mu_I` is the :math:`\\mu`
    parameter for no smear zone, and:

    .. math:: \\kappa_X= 1+\\frac{\\kappa-1}{s-1}\\left({s_X-1}\\right)

    .. math:: s_X = 2n-s

    and:

    .. math:: n = \\frac{r_e}{r_w}

    .. math:: s = \\frac{r_s}{r_w}

    .. math:: \\kappa = \\frac{k_h}{k_s}

    :math:`r_w` is the drain radius, :math:`r_e` is the drain influence radius,
    :math:`r_s` is the smear zone radius, :math:`k_h` is the undisturbed
    horizontal permeability, :math:`k_s` is the smear zone horizontal
    permeability

    See also
    --------
    mu_linear : :math:`\\mu` for non-overlapping smear zones

    mu_ideal :  :math:`\\mu` for ideal drain with no smear zone

    References
    ----------
    .. [1] Walker, R., and B. Indraratna. 2007. 'Vertical Drain Consolidation
           with Overlapping Smear Zones'. Geotechnique 57 (5): 463-67.
           doi:10.1680/geot.2007.57.5.463.

    """

    def mu_intersecting(n, s, kap):
        """mu for intersecting smear zones that do not completely overlap"""
        sx = _sx(n, s)
        kapx = _kapx(n, s, kap)
        mu = mu_linear(n, sx, kapx) * kap / kapx
        return mu

    n = np.asarray(n)
    s = np.asarray(s)
    kap = np.asarray(kap)
    if np.any(n <= 1.0):
        raise ValueError('n must be greater than 1. You have n = {}'.format(
            ', '.join([str(v) for v in np.atleast_1d(n)])))

    if np.any(s < 1.0):
        raise ValueError('s must be greater than 1. You have s = {}'.format(
            ', '.join([str(v) for v in np.atleast_1d(s)])))

    if np.any(kap <= 0.0):
        raise ValueError('kap must be greater than 0. You have kap = '
                         '{}'.format(', '.join([str(v) for v in
                                                np.atleast_1d(kap)])))

    is_array = any([isinstance(v, np.ndarray) for v in [n, s, kap]])

    n = np.atleast_1d(n)
    s = np.atleast_1d(s)
    kap = np.atleast_1d(kap)


    if len([v for v in [n, s] if v.shape == kap.shape]) != 2:
        raise ValueError('n, s, and kap must have the same shape.  You have '
                         'lengths for n, s, kap of {}, {}, {}.'.format(
                             len(n), len(s), len(kap)))


    ideal = np.isclose(s, 1) | np.isclose(kap, 1)
    normal = (n >= s) & (~ideal)
    all_disturbed = (2 * n - s <= 1) & (~ideal)
    intersecting = ~(ideal | normal | all_disturbed)

    mu = np.empty_like(n, dtype=float)

    mu[ideal] = mu_ideal(n[ideal])
    mu[normal] = mu_linear(n[normal], s[normal], kap[normal])
    mu[all_disturbed] = kap[all_disturbed] * mu_ideal(n[all_disturbed])
    mu[intersecting] = mu_intersecting(n[intersecting], s[intersecting],
                                       kap[intersecting])

    if is_array:
        return mu
    else:
        return mu[0]


def mu_linear(n, s, kap):
    """Smear zone parameter for smear zone linear variation of permeability

    mu parameter in equal strain radial consolidation equations e.g.
    u = u0 * exp(-8*Th/mu)

    Parameters
    ----------
    n : float or ndarray of float
        Ratio of drain influence radius  to drain radius (re/rw).
    s : float or ndarray of float
        Ratio of smear zone radius to  drain radius (rs/rw).
    kap : float or ndarray of float
        Ratio of undisturbed horizontal permeability to permeability at
        the drain-soil interface (kh / ks).

    Returns
    -------
    mu : float
        Smear zone permeability/geometry parameter.

    Notes
    -----
    For :math:`s\\neq\\kappa`, :math:`\\mu` is given by:

    .. math:: \\mu=\\frac{n^2}{\\left({n^2-1}\\right)}
          \\left[{
              \\ln\\left({\\frac{n}{s}}\\right)
              -\\frac{3}{4}
              +\\frac{s^2}{n^2}\\left({1-\\frac{s^2}{4n^2}}\\right)
              -\\frac{\\kappa}{B}\\ln\\left({\\frac{\\kappa}{s}}\\right)
              +\\frac{\\kappa B}{A^2 n^2}\\left({2-\\frac{B^2}{A^2 n^2}}
                  \\right)\\ln\\left({\\kappa}\\right)
              -\\frac{\\kappa\\left({s-1}\\right)}{A n^2}
                  \\left\\{
                      2
                      +\\frac{1}{n^2}
                          \\left[
                          {\\frac{A-B}{A}\\left({\\frac{1}{A}}-\\frac{s+1}{2}
                          \\right)}
                          -\\frac{s+1}{2}
                          -\\frac{\\left({s-1}\\right)^2}{3}
                          \\right]
                  \\right\\}
          }\\right]

    and for the special case :math:`s=\\kappa`, :math:`\\mu` is given by:

    .. math:: \\mu=\\frac{n^2}{\\left({n^2-1}\\right)}
          \\left[{
              \\ln\\left({\\frac{n}{s}}\\right)
              -\\frac{3}{4}
              +s-1
              -\\frac{s^2}{n^2}\\left({1-\\frac{s^2}{12n^2}}\\right)
              -\\frac{s}{n^2}\\left({2-\\frac{1}{3n^2}}\\right)
          }\\right]

    where :math:`A` and :math:`B` are:

    .. math:: A=\\frac{\\kappa-1}{s-1}

    .. math:: B=\\frac{s-\\kappa}{s-1}

    and:

    .. math:: n = \\frac{r_e}{r_w}

    .. math:: s = \\frac{r_s}{r_w}

    .. math:: \\kappa = \\frac{k_h}{k_s}

    :math:`r_w` is the drain radius, :math:`r_e` is the drain influence radius,
    :math:`r_s` is the smear zone radius, :math:`k_h` is the undisturbed
    horizontal permeability, :math:`k_s` is the smear zone horizontal
    permeability

    References
    ----------
    .. [1] Walker, R., and B. Indraratna. 2007. 'Vertical Drain Consolidation
           with Overlapping Smear Zones'. Geotechnique 57 (5): 463-67.
           doi:10.1680/geot.2007.57.5.463.
    """


    def mu_s_neq_kap(n, s, kap):
        """mu for when s != kap"""

        A = (kap - 1) / (s - 1)
        B = (s - kap) / (s - 1)

        term1 = n**2 / (n**2 - 1)
        term2 = (log(n / s) + s ** 2 / (n ** 2) *
                 (1 - s ** 2 / (4 * n ** 2)) - 3 / 4)
        term3 = kap * (1 - s ** 2 / n ** 2)
        term4 = (1 / B * log(s / kap)
                 - 1 / (n ** 2 * A ** 2) * (kap - 1 - B * log(kap)))
        term5 = term2 + term3 * term4

        term6 = 1 / (n ** 2 * B)
        term7 = (s ** 2 * log(s) - (s ** 2 - 1) / 2
            + 1 / A ** 2 * ((kap ** 2 - 1) / 2 - kap ** 2 * log(kap) + 2 * B * (kap * log(kap) - (kap - 1))))
        term8 = -1 / (n ** 4 * A ** 2)

        term9 = (B / 3 * (s ** 2 - 1) + 2 / 3 * (s ** 2 * kap - 1) - (s ** 2 - 1)
            + B / A ** 2 * ((kap ** 2 - 1) / 2 - kap ** 2 * log(kap) + 2 * B * (kap * log(kap) - (kap - 1))))


        term10 = kap * (term6 * term7 + term8 * term9)

        mu = term1 * (term5 + term10)
        return mu

    def mu_s_eq_kap(n, s):
        """mu for s == kap"""

        term1 = n ** 2 / (n ** 2 - 1)
        term2 = (log(n / s)
                + s ** 2 / (n ** 2) * (1 - s ** 2 / (4 * n ** 2)) - 3 / 4)
        term3 = (-s / n ** 2 * (1 - s ** 2 / n ** 2) * (s - 1)
                + (1 - s ** 2 / n ** 2) * (s - 1))
        term4 = (s / n ** 4 * (s ** 2 - 1)
                - 2 * s / (3 * n ** 4) * (s ** 3 - 1)
                - (s / n ** 2 - s ** 2 / n ** 2) * (s - 1))
        mu = term1 * (term2 + term3 + term4)
        return mu

    n = np.asarray(n)
    s = np.asarray(s)
    kap = np.asarray(kap)
    if np.any(n<=1.0):
        raise ValueError('n must be greater than 1. You have n = {}'.format(
            ', '.join([str(v) for v in np.atleast_1d(n)])))

    if np.any(s<1.0):
        raise ValueError('s must be greater than 1. You have s = {}'.format(
            ', '.join([str(v) for v in np.atleast_1d(s)])))

    if np.any(kap<=0.0):
        raise ValueError('kap must be greater than 0. You have kap = '
                '{}'.format(', '.join([str(v) for v in np.atleast_1d(kap)])))

    if np.any(s>=n):
        raise ValueError('s must be less than n. You have s = '
                '{} and n = {}'.format(
                ', '.join([str(v) for v in np.atleast_1d(s)]),
                ', '.join([str(v) for v in np.atleast_1d(n)])))




    is_array = any([isinstance(v, np.ndarray) for v in [n, s, kap]])

    n = np.atleast_1d(n)
    s = np.atleast_1d(s)
    kap = np.atleast_1d(kap)


    if len([v for v in [n, s] if v.shape==kap.shape])!=2:
        raise ValueError('n, s, and kap must have the same shape.  You have '
            'lengths for n, s, kap of {}, {}, {}.'.format(
            len(n), len(s), len(kap)))

    mu = np.empty_like(n, dtype=float)


    ideal = np.isclose(s, 1) | np.isclose(kap, 1)
    s_eq_kap = np.isclose(s, kap) & ~ideal
    s_neq_kap = ~np.isclose(s, kap) & ~ideal


    mu[ideal] = mu_ideal(n[ideal])
    mu[s_eq_kap] = mu_s_eq_kap(n[s_eq_kap], s[s_eq_kap])
    mu[s_neq_kap] = mu_s_neq_kap(n[s_neq_kap], s[s_neq_kap], kap[s_neq_kap])

    if is_array:
        return mu
    else:
        return mu[0]

def mu_parabolic(n, s, kap):
    """Smear zone parameter for parabolic variation of permeability

    mu parameter in equal strain radial consolidation equations e.g.
    u = u0 * exp(-8*Th/mu)

    Parameters
    ----------
    n : float or ndarray of float
        Ratio of drain influence radius  to drain radius (re/rw).
    s : float or ndarray of float
        Ratio of smear zone radius to  drain radius (rs/rw).
    kap : float or ndarray of float
        Ratio of undisturbed horizontal permeability to permeability at
        the drain-soil interface (kh/ks).

    Returns
    -------
    mu : float
        Smear zone permeability/geometry parameter

    Notes
    -----
    The smear zone parameter :math:`\\mu` is given by:


    .. math:: \\mu = \\frac{n^2}{\\left({n^2-1}\\right)}
                \\left({
                    \\frac{A^2}{n^2}\\mu_1+\\mu_2
                }\\right)

    where,

    .. math:: \\mu_1=
                \\frac{1}{A^2-B^2}
                    \\left({
                        s^2\\ln\\left({s}\\right)
                        -\\frac{1}{2}\\left({s^2-1}\\right)
                    }\\right)
                -\\frac{1}{\\left({A^2-B^2}\\right)C^2}
                    \\left({
                        \\frac{A^2}{2}\\ln\\left({\\kappa}\\right)
                        +\\frac{ABE}{2}+\\frac{1}{2}-B
                        -\\left({A^2-B^2}\\right)\\ln\\left({\\kappa}\\right)
                    }\\right)
                +\\frac{1}{n^2C^4}
                    \\left({
                        -\\left({\\frac{A^2}{2}+B^2}\\right)
                            \\ln\\left({\\kappa}\\right)
                        +\\frac{3ABE}{2}+\\frac{1}{2}-3B
                    }\\right)

    .. math:: \\mu_2=
                \\ln\\left({\\frac{n}{s}}\\right)
                -\\frac{3}{4}
                +\\frac{s^2}{n^2}\\left({1-\\frac{s^2}{4n^2}}\\right)
                +A^2\\left({1-\\frac{s^2}{n^2}}\\right)
                \\left[{
                    \\frac{1}{A^2-B^2}
                    \\left({
                        \\ln\\left({\\frac{s}{\\sqrt{\\kappa}}}\\right)
                        -\\frac{BE}{2A}
                    }\\right)
                    +\\frac{1}{n^2C^2}
                    \\left({
                        \\ln\\left({\\sqrt{\\kappa}}\\right)
                        -\\frac{BE}{2A}
                    }\\right)
                }\\right]



    where :math:`A`, :math:`B`, :math:`C` and :math:`E` are:

    .. math:: A=\\sqrt{\\frac{\\kappa}{\\kappa-1}}

    .. math:: B=\\frac{s}{s-1}

    .. math:: C=\\frac{1}{s-1}

    .. math:: E=\\ln\\left({\\frac{A+1}{A-1}}\\right)

    and:

    .. math:: n = \\frac{r_e}{r_w}

    .. math:: s = \\frac{r_s}{r_w}

    .. math:: \\kappa = \\frac{k_h}{k_s}


    :math:`r_w` is the drain radius, :math:`r_e` is the drain influence radius,
    :math:`r_s` is the smear zone radius, :math:`k_h` is the undisturbed
    horizontal permeability, :math:`k_s` is the smear zone horizontal
    permeability


    References
    ----------
    .. [1] Walker, Rohan, and Buddhima Indraratna. 2006. 'Vertical Drain
           Consolidation with Parabolic Distribution of Permeability in
           Smear Zone'. Journal of Geotechnical and Geoenvironmental
           Engineering 132 (7): 937-41.
           doi:10.1061/(ASCE)1090-0241(2006)132:7(937).

    """


    def mu_p(n, s, kap):
        """mu for parabolic smear"""

        A = sqrt((kap / (kap - 1)))
        B = s / (s - 1)
        C = 1 / (s - 1)

        term1 = (log(n / s) - 3 / 4 +
                s ** 2 / n ** 2 * (1 - s ** 2 / (4 * n ** 2)))
        term2 = (1 - s ** 2 / n ** 2) * A ** 2
        term3 = 1 / (A ** 2 - B ** 2)
        term4 = (log(s / sqrt(kap))) - (B / (2 * A) * log((A + 1) / (A - 1)))
        term5 = 1 / (n ** 2 * C ** 2)
        term6 = (log(sqrt(kap))) - (B / (2 * A) * log((A + 1) / (A - 1)))

        mu2 = term1 + term2 * ((term3 * term4) + (term5 * term6))


        term7 = (A ** 2 / n ** 2 * (1 / (A ** 2 - B ** 2)) * (s ** 2 * log(s)
            - 1 / 2 * (s ** 2 - 1)))
        term8 = -1 / (n ** 2 * C ** 2) * A ** 2 * (1 / (A ** 2 - B ** 2))
        term9 = (A ** 2 / 2 * log(kap) + B * A / 2 * log((A + 1) / (A - 1))
            + 1 / 2 - B - (A ** 2 - B ** 2) * log(kap))

        term12 = A ** 2 / 2 * log(kap)
        term13 = (B * A / 2 * log((A + 1) / (A - 1)))
        term14 = 1 / 2 - B
        term15 = -(A ** 2 - B ** 2) * log(kap)


        term10 = A ** 2 / (n ** 4 * C ** 4)
        term11 = (-(A ** 2 / 2 + B ** 2) * (log(kap)) +
        3 / 2 * A * B * log((A + 1) / (A - 1)) + 1 / 2 - 3 * B)

        mu1 = term7 + (term8 * term9) + (term10 * term11)

        mu = n ** 2 / (n ** 2 - 1) * (mu1 + mu2)
        return mu

    n = np.asarray(n)
    s = np.asarray(s)
    kap = np.asarray(kap)
    if np.any(n<=1.0):
        raise ValueError('n must be greater than 1. You have n = {}'.format(
            ', '.join([str(v) for v in np.atleast_1d(n)])))

    if np.any(s<1.0):
        raise ValueError('s must be greater than 1. You have s = {}'.format(
            ', '.join([str(v) for v in np.atleast_1d(s)])))

    if np.any(kap<=0.0):
        raise ValueError('kap must be greater than 0. You have kap = '
                '{}'.format(', '.join([str(v) for v in np.atleast_1d(kap)])))

    if np.any(s>n):
        raise ValueError('s must be less than n. You have s = '
                '{} and n = {}'.format(
                ', '.join([str(v) for v in np.atleast_1d(s)]),
                ', '.join([str(v) for v in np.atleast_1d(n)])))




    is_array = any([isinstance(v, np.ndarray) for v in [n, s, kap]])

    n = np.atleast_1d(n)
    s = np.atleast_1d(s)
    kap = np.atleast_1d(kap)


    if len([v for v in [n, s] if v.shape==kap.shape])!=2:
        raise ValueError('n, s, and kap must have the same shape.  You have '
            'lengths for n, s, kap of {}, {}, {}.'.format(
            len(n), len(s), len(kap)))

    mu = np.empty_like(n, dtype=float)

    ideal = np.isclose(s, 1) | np.isclose(kap, 1)



    mu[ideal] = mu_ideal(n[ideal])
    mu[~ideal] = mu_p(n[~ideal], s[~ideal], kap[~ideal])


    if is_array:
        return mu
    else:
        return mu[0]


def mu_piecewise_constant(s, kap, n=None, kap_m=None):
    """Smear zone parameter for piecewise constant permeability distribution


    mu parameter in equal strain radial consolidation equations e.g.
    u = u0 * exp(-8*Th/mu)

    Parameters
    ----------
    s : list or 1d ndarray of float
        Ratio of segment outer radii to drain radius (r_i/r_0). The first value
        of s should be greater than 1, i.e. the first value should be s_1;
        s_0=1 at the drain soil interface is implied.
    kap : list or ndarray of float
        Ratio of undisturbed horizontal permeability to permeability in each
        segment kh/khi.
    n, kap_m : float, optional
        If `n` and `kap_m` are given then they will each be appended to `s` and
        `kap`. This allows the specification of a smear zone separate to the
        specification of the drain influence radius.
        Default n=kap_m=None, i.e. soilpermeability is completely described
        by `s` and `kap`. If n is given but kap_m is None then the last
        kappa value in kap will be used.


    Returns
    -------
    mu : float
        Smear zone permeability/geometry parameter

    Notes
    -----

    The smear zone parameter :math:`\\mu` is given by:

    .. math:: \\mu = \\frac{n^2}{\\left({n^2-1}\\right)}
                \\sum\\limits_{i=1}^{m} \\kappa_i
                    \\left[{
                        \\frac{s_i^2}{n^2}\\ln
                            \\left({
                                \\frac{s_i}{s_{i-1}}
                            }\\right)
                        -\\frac{s_i^2-s_{i-1}^2}{2n^2}
                        -\\frac{\\left({s_i^2-s_{i-1}^2}\\right)^2}{4n^4}
                    }\\right]
                    +\\psi_i\\frac{s_i^2-s_{i-1}^2}{n^2}

    where,

    .. math:: \\psi_{i} = \\sum\\limits_{j=1}^{i-1}\\kappa_j
                \\left[{
                        \\ln
                            \\left({
                                \\frac{s_j}{s_{j-1}}
                            }\\right)
                        -\\frac{s_j^2-s_{j-1}^2}{2n^2}
                    }\\right]

    and:

    .. math:: n = \\frac{r_m}{r_0}

    .. math:: s_i = \\frac{r_i}{r_0}

    .. math:: \\kappa_i = \\frac{k_h}{k_{hi}}


    :math:`r_0` is the drain radius, :math:`r_m` is the drain influence radius,
    :math:`r_i` is the outer radius of the ith segment,
    :math:`k_h` is the undisturbed
    horizontal permeability in the ith segment,
    :math:`k_{hi}` is the horizontal
    permeability in the ith segment

    References
    ----------
    .. [1] Walker, Rohan. 2006. 'Analytical Solutions for Modeling Soft Soil
           Consolidation by Vertical Drains'. PhD Thesis, Wollongong, NSW,
           Australia: University of Wollongong. http://ro.uow.edu.au/theses/501
    .. [2] Walker, Rohan T. 2011. 'Vertical Drain Consolidation Analysis in
           One, Two and Three Dimensions'. Computers and
           Geotechnics 38 (8): 1069-77. doi:10.1016/j.compgeo.2011.07.006.

    """


    s = np.atleast_1d(s)
    kap = np.atleast_1d(kap)

    if not n is None:
        s_temp = np.empty(len(s) + 1, dtype=float)
        s_temp[:-1] = s
        s_temp[-1] = n
        kap_temp = np.empty(len(kap) + 1, dtype=float)
        kap_temp[:-1] = kap
        if kap_m is None:
            kap_temp[-1] = kap[-1]
        else:
            kap_temp[-1] = kap_m
        s = s_temp
        kap = kap_temp

    n = np.asarray(n)
    s = np.asarray(s)
    kap = np.asarray(kap)
    if len(s)!=len(kap):
        raise ValueError('s and kap must have the same shape.  You have '
            'lengths for s, kap of {}, {}.'.format(
            len(s), len(kap)))

    if np.any(s<=1.0):
        raise ValueError('must have all s>=1. You have s = {}'.format(
            ', '.join([str(v) for v in np.atleast_1d(s)])))

    if np.any(kap<=0.0):
        raise ValueError('all kap must be greater than 0. You have kap = '
                '{}'.format(', '.join([str(v) for v in np.atleast_1d(kap)])))

    if np.any(np.diff(s) <= 0):
        raise ValueError('s must increase left to right you have s = '
        '{}'.format(', '.join([str(v) for v in np.atleast_1d(s)])))


    n = s[-1]
    s_ = np.ones_like(s , dtype=float)
    s_[1:] = s[:-1]

    sumi = 0
    for i in range(len(s)):
        psi = 0
        for j in range(i):
            psi += kap[j] * (log(s[j] / s_[j])
                - 0.5 * (s[j] ** 2 / n ** 2 - s_[j] ** 2 / n ** 2))
        psi /= kap[i]

        sumi += kap[i] * (
            s[i] ** 2 / n ** 2 * log(s[i] / s_[i])
            + (psi - 0.5) * (s[i] ** 2 / n ** 2 - s_[i] ** 2 / n ** 2)
            - 0.25 * (s[i] ** 2 - s_[i] ** 2) ** 2 / n ** 4
            )
    mu = sumi * n ** 2 / (n ** 2 - 1)
    return mu


def mu_piecewise_linear(s, kap, n=None, kap_m=None):
    """Smear zone parameter for piecewise linear permeability distribution


    mu parameter in equal strain radial consolidation equations e.g.
    u = u0 * exp(-8*Th/mu)

    Parameters
    ----------
    s : list or 1d ndarray of float
        Ratio of radii to drain radius (r_i/r_0). The first value
        of s should be 1, i.e. at the drain soil interface.
    kap : list or ndarray of float
        Ratio of undisturbed horizontal permeability to permeability at each
        value of s.
    n, kap_m : float, optional
        If `n` and `kap_m` are given then they will each be appended to `s` and
        `kap`. This allows the specification of a smear zone separate to the
        specification of the drain influence radius.
        Default n=kap_m=None, i.e. soilpermeability is completely described
        by `s` and `kap`. If n is given but kap_m is None then the last
        kappa value in kap will be used.


    Returns
    -------
    mu : float
        Smear zone permeability/geometry parameter.

    Notes
    -----
    With permeability in the ith segment defined by:

    .. math:: \\frac{k_i}{k_{ref}}= \\frac{1}{\\kappa_{i-1}}
                    \\left({A_ir/r_w+B_i}\\right)

    .. math:: A_i = \\frac{\\kappa_{i-1}/\\kappa_i-1}{s_i-s_{i-1}}

    .. math:: B_i = \\frac{s_i-s_{i-1}\\kappa_{i-1}/\\kappa_i}{s_i-s_{i-1}}

    the smear zone :math:`\\mu` parameter is given by:

    .. math:: \\mu = \\frac{n^2}{n^2-1}
                        \\left[{
                            \\sum\\limits_{i=1}^{m}\\kappa_{i-1}\\theta_i
                            + \\Psi_i
                            \\left({
                                \\frac{s_i^2-s_{i-1}^2}{n^2}
                            }\\right)
                            +\\mu_w
                        }\\right]

    where,

    .. math:: \\theta_i = \\left\\{
                            \\begin{array}{lr}
                            \\frac{s_i^2}{n^2}\\ln
                                \\left[{\\frac{s_i}{s_{i-1}}}\\right]
                                -\\frac{s_i^2-s_{i-1}^2}{2n^2}
                                -\\frac{\\left({s_i^2-s_{i-1}^2}\\right)^2}{4n^4}
                            & \\textrm{for } \\frac{\\kappa_{i-1}}{\\kappa_i}=1 \\\\
                            \\frac{\\left({s_i^2-s_{i-1}^2}\\right)}{3n^4}
                                \\left({3n^2-s_{i-1}^2-2s_{i-1}s_i}\\right)
                            & \\textrm{for }\\frac{\\kappa_{i-1}}{\\kappa_i}=
                             \\frac{s_i}{s_{i-1}} \\\\
                             \\begin{multline}
                            \\frac{s_i}{B_i n^2}\\ln\\left[{
                                \\frac{\\kappa_i s_i}{\\kappa_{i-1}s_{i-1}}}\\right]
                            -\\frac{s_i-s_{i-1}}{A_in^2}
                                \\left({1-\\frac{B_i^2}{A_i^2n^4}}\\right)
                            \\\\-\\frac{\\left({s_i-s_{i-1}}\\right)^2}{3A_in^2}
                                \\left({2s_i+s_{i-1}}\\right)
                            \\\\+\\frac{B_i}{A_i^2 n^4}\\ln\\left[{
                                \\frac{\\kappa_{i-1}}{\\kappa_i}}\\right]
                                \\left({1-\\frac{B_i^2}{A_i^2n^2}}\\right)
                            \\\\+\\frac{B_i}{2A_i^2 n^4}
                                \\left({
                                2s_i^2\\ln\\left[{
                                \\frac{\\kappa_{i-1}}{\\kappa_i}}\\right]
                                -s_i^2 + s_{i-1}^2
                                }\\right)
                            \\end{multline}
                            & \\textrm{otherwise}
                            \\end{array}\\right.

    .. math:: \\Psi_i = \\sum\\limits_{j=1}^{i-1}\\kappa_{j-1}\\psi_j

    .. math:: \\psi_i = \\left\\{
                            \\begin{array}{lr}
                            \\ln\\left[{\\frac{s_j}{s_{j-1}}}\\right]
                            - \\frac{s_j^2- s_{j-1}^2}{2n^2}
                            & \\textrm{for } \\frac{\\kappa_{j-1}}{\\kappa_j}=1 \\\\
                            \\frac{\\left({s_j - s_{j-1}}\\right)
                                \\left({n^2-s_js_{j-1}}\\right)}{s_jn^2}
                            & \\textrm{for }\\frac{\\kappa_{j-1}}{\\kappa_j}=
                             \\frac{s_j}{s_{j-1}} \\\\
                             \\begin{multline}
                            \\frac{1}{B_i}\\ln\\left[{\\frac{s_j}{s_{j-1}}}\\right]
                                +\\ln\\left[{\\frac{\\kappa_{j-1}}{\\kappa_j}}\\right]
                                \\left({\\frac{B_j}{A_j^2n^2}-\\frac{1}{B_j}}\\right)
                                \\\\-\\frac{s_j-s_{j-1}}{A_j^2n^2}
                            \\end{multline}
                            & \\textrm{otherwise}
                            \\end{array}\\right.



    and:

    .. math:: n = \\frac{r_m}{r_0}

    .. math:: s_i = \\frac{r_i}{r_0}

    .. math:: \\kappa_i = \\frac{k_h}{k_{ref}}


    :math:`r_0` is the drain radius, :math:`r_m` is the drain influence radius,
    :math:`r_i` is the radius of the ith radial point,
    :math:`k_{ref}` is a convienient refernce permeability, usually
    the undisturbed
    horizontal permeability,
    :math:`k_{hi}` is the horizontal
    permeability at the ith radial point

    References
    ----------
    Derived by Rohan Walker in 2011 and 2014.
    Derivation steps are the same as for mu_piecewise_constant in appendix of
    [1]_ but permeability is linear in a segemetn as in [2]_.

    .. [1] Walker, Rohan. 2006. 'Analytical Solutions for Modeling Soft Soil
           Consolidation by Vertical Drains'. PhD Thesis, Wollongong, NSW,
           Australia: University of Wollongong. http://ro.uow.edu.au/theses/501
    .. [2] Walker, R., and B. Indraratna. 2007. 'Vertical Drain Consolidation
           with Overlapping Smear Zones'. Geotechnique 57 (5): 463-67.
           doi:10.1680/geot.2007.57.5.463.

    """


    s = np.atleast_1d(s)
    kap = np.atleast_1d(kap)

    if not n is None:
        s_temp = np.empty(len(s) + 1, dtype=float)
        s_temp[:-1] = s
        s_temp[-1] = n
        kap_temp = np.empty(len(kap) + 1, dtype=float)
        kap_temp[:-1] = kap
        if kap_m is None:
            kap_temp[-1] = kap[-1]
        else:
            kap_temp[-1] = kap_m
        s = s_temp
        kap = kap_temp

    n = np.asarray(n)
    s = np.asarray(s)
    kap = np.asarray(kap)
    if len(s)!=len(kap):
        raise ValueError('s and kap must have the same shape.  You have '
            'lengths for s, kap of {}, {}.'.format(
            len(s), len(kap)))

    if np.any(s < 1.0):
        raise ValueError('must have all s>=1. You have s = {}'.format(
            ', '.join([str(v) for v in np.atleast_1d(s)])))

    if np.any(kap<=0.0):
        raise ValueError('all kap must be greater than 0. You have kap = '
                '{}'.format(', '.join([str(v) for v in np.atleast_1d(kap)])))

    if np.any(np.diff(s) < 0):
        raise ValueError('All s must satisfy s[i]>s[i-1]. You have s = '
        '{}'.format(', '.join([str(v) for v in np.atleast_1d(s)])))

    if not np.isclose(s[0], 1):
        raise ValueError('First value of s should be 1.  You '
                            'have s[0]={}'.format(s[0]))




    n = s[-1]

    sumi = 0
    for i in range(1, len(s)):
        sumj = 0
        for j in range(1, i):
#            term1 = 0
            if np.isclose(s[j - 1], s[j]):
                term1=0
            elif np.isclose(kap[j - 1], kap[j]):
                term1 = (log(s[j] / s[j - 1])
                    - (s[j] ** 2 - s[j - 1] ** 2) / 2 / n ** 2)
            elif np.isclose(kap[j-1] / kap[j], s[j] / s[j - 1]):
                term1 = (s[j] - s[j - 1]) * (n ** 2 -
                        s[j - 1] * s[j]) / s[j] / n ** 2
            else:
                A = (kap[j-1] / kap[j] - 1) / (s[j] - s[j - 1])
                B = (s[j] - kap[j-1] / kap[j] * s[j - 1]) / (s[j] - s[j - 1])
                term1 = (1 / B * log(s[j] / s[j - 1])
                    + (B / A ** 2 / n ** 2 - 1 / B) * log(kap[j-1] / kap[j])
                    - (s[j] - s[j - 1]) / A / n ** 2)

            sumj += kap[j-1] * term1

#        term1 = 0
        if np.isclose(s[i - 1], s[i]):
            term1=0
        elif np.isclose(kap[i - 1], kap[i]):
            term1 = (s[i] ** 2 / n ** 2 * log(s[i] / s[i - 1])
                - (s[i] ** 2 - s[i - 1] ** 2) / 2 / n ** 2
                - (s[i] ** 2 - s[i - 1] ** 2) ** 2 / 4 / n ** 4)
        elif np.isclose(kap[i-1] / kap[i], s[i] / s[i - 1]):
            term1 = ((s[i] - s[i - 1]) ** 2 / 3 / n ** 4 * (3 * n ** 2 -
                    s[i - 1] ** 2 - 2 * s[i - 1] * s[i]))
        else:
            A = (kap[i-1] / kap[i] - 1) / (s[i] - s[i - 1])
            B = (s[i] - kap[i-1] / kap[i] * s[i - 1]) / (s[i] - s[i - 1])
            term1 = (s[i] ** 2 / B / n ** 2 * log(kap[i] * s[i] /
                    kap[i-1] / s[i - 1])
                - (s[i] - s[i - 1]) / A / n ** 2 *
                (1 - B ** 2 / A ** 2 / n ** 2)
                - (s[i] - s[i - 1]) ** 2 / 3 / A / n ** 4 *
                (s[i - 1] + 2 * s[i])
                + B / A ** 2 / n ** 2 * log(kap[i-1] / kap[i]) *
                (1 - B ** 2 / A ** 2 / n ** 2)
                + B / 2 / A ** 2 / n ** 4 * (s[i] ** 2 * (2 * log(kap[i-1] /
                kap[i]) - 1) + s[i - 1] ** 2))

        sumi += kap[i-1] * term1 + sumj * (s[i] ** 2 - s[i - 1] ** 2) / n ** 2

    mu = sumi * n ** 2 / (n ** 2 - 1)

    return mu


def mu_well_resistance(kh, qw, n, H, z=None):
    """Additional smear zone parameter for well resistance


    Parameters
    ----------
    kh : float
        The normalising permeability used in calculating kappa for smear zone
        calcs.  Usually the undisturbed permeability i.e. the kh in
        kappa = kh/ks
    qw : float
        Drain discharge capacity.  qw = kw * pi * rw**2.  Make sure
        the kw used has the same units as kh.
    n : float
        Ratio of drain influence radius  to drain radius (re/rw).
    H : float
        Length of drainage path.
    z : float, optional
        Evaluation depth. Default = None, in which case the well resistance
        factor will be averaged.

    Returns
    -------
    mu : float
        mu parameter for well resistance

    Notes
    -----
    The smear zone parameter :math:`\\mu_w` is given by:


    .. math:: \\mu_w = \\frac{k_h}{q_w}\\pi z
                        \\left({2H-z}\\right)
                        \\left({1-\\frac{1}{n^2}}\\right)

    when :math:`z` is None then the average :math:`\\mu_w` is given by:

    .. math:: \\mu_{w\\textrm{average}} = \\frac{2k_h H^2}{3q_w}\\pi
                        \\left({1-\\frac{1}{n^2}}\\right)

    where,

    .. math:: n = \\frac{r_e}{r_w}

    .. math:: q_w = k_w \\pi r_w^2
    :math:`r_w` is the drain radius, :math:`r_e` is the drain influence radius,
    :math:`k_h` is the undisturbed horizontal permeability,
    :math:`k_w` is the drain permeability

    References
    ----------
    .. [1] Hansbo, S. 1981. 'Consolidation of Fine-Grained Soils by
           Prefabricated Drains'. In 10th ICSMFE, 3:677-82.
           Rotterdam-Boston: A.A. Balkema.

    """

    n = np.asarray(n)
    if n<=1.0:
        raise ValueError('n must be greater than 1. You have n = {}'.format(
            n))

    if z is None:
        mu = 2 * kh * H**2 / 3 / qw * np.pi * (1 - 1 / n**2)
    else:
        mu = kh / qw * np.pi * z * (2 * H - z) * (1 - 1 / n**2)

    return mu

def k_parabolic(n, s, kap, si):
    """Permeability distribution for smear zone with parabolic permeability

    Normalised with respect to undisturbed permeability.  i.e. if you want the
    actual permeability then multiply by whatever you used to determine kap.


    Permeability is parabolic with value 1/kap at the drain soil interface
    i.e. at s=1 k=k0=1/kap.  for si>s, permeability=1.

    Parameters
    ----------
    n : float
        Ratio of drain influence radius  to drain radius (re/rw).
    s : float
        Ratio of smear zone radius to  drain radius (rs/rw).
    kap : float
        Ratio of undisturbed horizontal permeability to permeability at
        the drain-soil interface (kh / ks).
    si : float of ndarray of float
        Normalised radial coordinate(s) at which to calc the permeability
        i.e. si=ri/rw

    Returns
    -------
    permeability : float or ndarray of float
        Normalised permeability (i.e. ki/kh) at the si values.

    Notes
    -----
    Parabolic distribution of permeability in smear zone is given by:

    .. math:: \\frac{k_h^\\prime\\left({r}\\right)}{k_h}=
                \\frac{\\kappa-1}{\\kappa}
                \\left({A-B+C\\frac{r}{r_w}}\\right)
                \\left({A+B-C\\frac{r}{r_w}}\\right)

    where :math:`A`, :math:`B`, :math:`C` are:

    .. math:: A=\\sqrt{\\frac{\\kappa}{\\kappa-1}}

    .. math:: B=\\frac{s}{s-1}

    .. math:: C=\\frac{1}{s-1}

    and:

    .. math:: n = \\frac{r_e}{r_w}

    .. math:: s = \\frac{r_s}{r_w}

    .. math:: \\kappa = \\frac{k_h}{k_s}

    :math:`r_w` is the drain radius, :math:`r_e` is the drain influence radius,
    :math:`r_s` is the smear zone radius, :math:`k_h` is the undisturbed
    horizontal permeability, :math:`k_s` is the smear zone horizontal
    permeability


    References
    ----------
    .. [1] Walker, Rohan, and Buddhima Indraratna. 2006. 'Vertical Drain
           Consolidation with Parabolic Distribution of Permeability in
           Smear Zone'. Journal of Geotechnical and Geoenvironmental
           Engineering 132 (7): 937-41.
           doi:10.1061/(ASCE)1090-0241(2006)132:7(937).

    """
    n = np.asarray(n)
    s = np.asarray(s)
    kap = np.asarray(kap)
    if n<=1.0:
        raise ValueError('n must be greater than 1. You have n = {}'.format(
            n))

    if s<1.0:
        raise ValueError('s must be greater than 1. You have s = {}'.format(
            s))

    if kap<=0.0:
        raise ValueError('kap must be greater than 0. You have kap = '
                '{}'.format(kap))

    if s>n:
        raise ValueError('s must be less than n. You have s = '
                '{} and n = {}'.format(s, n))


    si = np.atleast_1d(si)
    if np.any((si < 1) | (si > n)):
        raise ValueError('si must satisfy 1 >= si >= n)')

    def parabolic_part(n,s, kap, si):
        """Parbolic smear zone part i.e from si=1 to si=s"""

        A = sqrt((kap / (kap - 1)))
        B = s / (s - 1)
        C = 1 / (s - 1)

        k0 = 1 / kap
        return k0*(kap-1)*(A - B + C * si)*(A + B - C * si)


    if np.isclose(s,1) or np.isclose(kap, 1):
        return np.ones_like(si, dtype=float)

    smear = (si < s)

    permeability = np.ones_like(si, dtype=float)
    permeability[smear] = parabolic_part(n, s, kap, si[smear])

    return permeability


def k_linear(n, s, kap, si):
    """Permeability distribution for smear zone with linear permeability

    Normalised with respect to undisturbed permeability.  i.e. if you want the
    actual permeability then multiply by whatever you used to determine kap.

    Permeability is linear with value 1/kap at the drain soil interface
    i.e. at s=1 k=k0=1/kap.  for si>s, permeability=1.

    Parameters
    ----------
    n : float
        Ratio of drain influence radius  to drain radius (re/rw).
    s : float
        Ratio of smear zone radius to  drain radius (rs/rw).
    kap : float
        Ratio of undisturbed horizontal permeability to permeability at
        the drain-soil interface (kh / ks).
    si : float of ndarray of float
        Normalised radial coordinate(s) at which to calc the permeability
        i.e. si=ri/rw.

    Returns
    -------
    permeability : float or ndarray of float
        Normalised permeability (i.e. ki/kh) at the si values.

    Notes
    -----
    Linear distribution of permeability in smear zone is given by:

    .. math::

       \\frac{k_h^\\prime\\left({r}\\right)}{k_h}=
        \\left\\{\\begin{array}{lr}
        \\frac{1}{\\kappa}
                    \\left({A\\frac{r}{r_w}+B}\\right)
                    & s\\neq\\kappa \\\\
                    \\frac{r}{\\kappa r_w}
                    & s=\\kappa \\end{array}\\right.

    where :math:`A` and :math:`B` are:

    .. math:: A=\\frac{\\kappa-1}{s-1}

    .. math:: B=\\frac{s-\\kappa}{s-1}

    and:

    .. math:: n = \\frac{r_e}{r_w}

    .. math:: s = \\frac{r_s}{r_w}

    .. math:: \\kappa = \\frac{k_h}{k_s}

    :math:`r_w` is the drain radius, :math:`r_e` is the drain influence radius,
    :math:`r_s` is the smear zone radius, :math:`k_h` is the undisturbed
    horizontal permeability, :math:`k_s` is the smear zone horizontal
    permeability

    References
    ----------
    .. [1] Walker, R., and B. Indraratna. 2007. 'Vertical Drain Consolidation
           with Overlapping Smear Zones'. Geotechnique 57 (5): 463-67.
           doi:10.1680/geot.2007.57.5.463.

    """

    def s_neq_kap_part(n, s, kap, si):
        """Linear permeability in smear zome when s!=kap"""

        A = (kap - 1) / (s - 1)
        B = (s - kap) / (s - 1)


        k0 = 1 / kap

        return k0*(A*si+B)

    def s_eq_kap_part(n, s, si):
        """Linear permeability in smear zome when s!=kap"""

        k0 = 1 / kap
        return k0 * si

    if n<=1.0:
        raise ValueError('n must be greater than 1. You have n = {}'.format(
            n))

    if s<1.0:
        raise ValueError('s must be greater than 1. You have s = {}'.format(
            s))

    if kap<=0.0:
        raise ValueError('kap must be greater than 0. You have kap = '
                '{}'.format(kap))

    if s>n:
        raise ValueError('s must be less than n. You have s = '
                '{} and n = {}'.format(s, n))

    si = np.atleast_1d(si)
    if np.any((si < 1) | (si > n)):
        raise ValueError('si must satisfy 1 >= si >= n)')

    if np.isclose(s,1) or np.isclose(kap, 1):
        return np.ones_like(si, dtype=float)


    smear = (si < s)
    permeability = np.ones_like(si, dtype=float)
    if np.isclose(s, kap):
        permeability[smear] = s_eq_kap_part(n, s, si[smear])
    else:
        permeability[smear] = s_neq_kap_part(n, s, kap, si[smear])

    return permeability


def k_overlapping_linear(n, s, kap, si):
    """Permeability smear zone with overlapping linear permeability

    Normalised with respect to undisturbed permeability.  i.e. if you want the
    actual permeability then multiply by whatever you used to determine kap.

    mu parameter in equal strain radial consolidation equations e.g.
    u = u0 * exp(-8*Th/mu)

    Parameters
    ----------
    n : float
        Ratio of drain influence radius  to drain radius (re/rw).
    s : float
        Ratio of smear zone radius to  drain radius (rs/rw).
    kap : float
        Ratio of undisturbed horizontal permeability to permeability at
        the drain-soil interface (kh / ks).
    si : float of ndarray of float
        Normalised radial coordinate(s) at which to calc the permeability
        i.e. si=ri/rw

    Returns
    -------
    permeability : float or ndarray of float
        Normalised permeability (i.e. ki/kh) at the si values.

    Notes
    -----

    When :math:`n>s` the permeability is no different from the linear case.

    When :math:`n\\leq (s+1)/2` then all the soil is disturbed
    and the permeability everywhere is equal to :math:`1/\\kappa`.

    When :math:`(s+1)/2<n<s` then the smear zones overlap.
    the permeability for :math:`r/r_w<s_X` is given by:

    .. math:: \\frac{k_h^\\prime\\left({r}\\right)}{k_h}=
                \\left\\{\\begin{array}{lr}
                \\frac{1}{\\kappa}
                            \\left({A\\frac{r}{r_w}+B}\\right)
                            & s\\neq\\kappa \\\\
                            \\frac{r}{\\kappa r_w}
                            & s=\\kappa \\end{array}\\right.

    In the overlapping part, :math:`r/r_w>s_X`, the permeability is given by:

    .. math::  k_h(r)=\\kappa_X/\\kappa

    where :math:`A` and :math:`B` are:

    .. math:: A=\\frac{\\kappa-1}{s-1}

    .. math:: B=\\frac{s-\\kappa}{s-1}

    .. math:: \\kappa_X= 1+\\frac{\\kappa-1}{s-1}\\left({s_X-1}\\right)

    .. math:: s_X = 2n-s




    and:

    .. math:: n = \\frac{r_e}{r_w}

    .. math:: s = \\frac{r_s}{r_w}

    .. math:: \\kappa = \\frac{k_h}{k_s}

    :math:`r_w` is the drain radius, :math:`r_e` is the drain influence radius,
    :math:`r_s` is the smear zone radius, :math:`k_h` is the undisturbed
    horizontal permeability, :math:`k_s` is the smear zone horizontal
    permeability

    References
    ----------
    .. [1] Walker, R., and B. Indraratna. 2007. 'Vertical Drain Consolidation
           with Overlapping Smear Zones'. Geotechnique 57 (5): 463-67.
           doi:10.1680/geot.2007.57.5.463.

    """

    def mu_intersecting(n, s, kap):
        """mu for intersecting smear zones that do not completely overlap"""
        sx = _sx(n, s)
        kapx =  _kapx(n, s, kap)
        mu = mu_linear(n, sx, kapx) * kap / kapx
        return mu
    
    n = np.asarray(n)
    s = np.asarray(s)
    kap = np.asarray(kap)
    if n<=1.0:
        raise ValueError('n must be greater than 1. You have n = {}'.format(
            n))

    if s<1.0:
        raise ValueError('s must be greater than 1. You have s = {}'.format(
            s))

    if kap<=0.0:
        raise ValueError('kap must be greater than 0. You have kap = '
                '{}'.format(kap))



    si = np.atleast_1d(si)
    if np.any((si < 1) | (si > n)):
        raise ValueError('si must satisfy 1 >= si >= n)')

    if np.isclose(s,1) or np.isclose(kap, 1):
        permeability = np.ones_like(si, dtype=float)
    elif (2*n-s <=1):
        permeability = np.ones_like(si, dtype=float) / kap
    elif (n>=s):
        permeability = k_linear(n, s, kap, si)
    else:
        sx = _sx(n, s)
        kapx =  _kapx(n, s, kap)

        smear = (si < sx)
        permeability = np.ones_like(si, dtype=float)
        A = (kap - 1) / (s - 1)
        B = (s - kap) / (s - 1)
        permeability[smear] = 1/kap*(A*si[smear] + B)
        permeability[~smear] = 1/kap*kapx#1 / kapx

    return permeability




def u_ideal(n, si, uavg=1, uw=0, muw=0):
    """Pore pressure at radius for ideal drain with no smear zone

    Parameters
    ----------
    n : float
        Ratio of drain influence radius  to drain radius (re/rw).
    si : float of ndarray of float
        Normalised radial coordinate(s) at which to calc the pore pressure
        i.e. si=ri/rw.
    uavg : float, optional = 1
        Average pore pressure in soil. default = 1.  when `uw`=0 , then if
        uavg=1.
    uw : float, optional
        Pore pressure in drain, default = 0.
    muw : float, optional
        Well resistance mu parameter

    Returns
    -------
    u : float or ndarray of float
        Pore pressure at specified si


    Notes
    -----
    The uavg is calculated from the eta method.  It is not the uavg used when
    considering the vacuum as an equivalent surcharge.  You would have to do
    other manipulations for that.

    Noteing that :math:`s_i=r_i/r_w`, the radial pore pressure distribution is given by:

    .. math:: u(r) = \\frac{u_{avg}-u_w}{\\mu+\\mu_w}
                            \\left[{
                                \\ln\\left({\\frac{r}{r_w}}\\right)
                                -\\frac{(r/r_w)^2-1}{2n^2}
                                +\\mu_w
                            }\\right]+u_w


    where:

    .. math:: n = \\frac{r_e}{r_w}

    :math:`r_w` is the drain radius, :math:`r_e` is the drain influence radius.

    References
    ----------
    .. [1] Hansbo, S. 1981. 'Consolidation of Fine-Grained Soils by
           Prefabricated Drains'. In 10th ICSMFE, 3:677-82.
           Rotterdam-Boston: A.A. Balkema.

    """
    n = np.asarray(n)
    if n<=1.0:
        raise ValueError('n must be greater than 1. You have n = {}'.format(
            n))

    si = np.atleast_1d(si)
    if np.any((si < 1) | (si > n)):
        raise ValueError('si must satisfy 1 >= si >= n)')


    mu = mu_ideal(n)
    term1 = (uavg - uw) / (mu + muw)
    term2 = log(si) - 1 / (2 * n**2) * (si**2 - 1) + muw

    u = term1 * term2 + uw
    return u

def u_constant(n, s, kap, si, uavg=1, uw=0, muw=0):
    """Pore pressure at radius for constant permeability smear zone

    Parameters
    ----------
    n : float
        Ratio of drain influence radius  to drain radius (re/rw).
    s : float
        Ratio of smear zone radius to  drain radius (rs/rw).
    kap : float
        Ratio of undisturbed horizontal permeability to permeability at
        the drain-soil interface (kh / ks).
    si : float of ndarray of float
        Normalised radial coordinate(s) at which to calc the pore pressure
        i.e. si=ri/rw.
    uavg : float, optional = 1
        Average pore pressure in soil. default = 1.  when `uw`=0 , then if
        uavg=1.
    uw : float, optional
        Pore pressure in drain, default = 0.
    muw : float, optional
        Well resistance mu parameter.

    Returns
    -------
    u : float or ndarray of float
        Pore pressure at specified si

    Notes
    -----
    The uavg is calculated from the eta method.  It is not the uavg used when
    considering the vacuum as an equivalent surcharge.  You would have to do
    other manipulations for that.

    Noteing that :math:`s_i=r_i/r_w`, the radial pore pressure distribution
    in the smear zone is given by:

    .. math:: u^\\prime(r) = \\frac{u_{avg}-u_w}{\\mu+\\mu_w}
                            \\left[{
                                \\kappa\\left({
                                \\ln\\left({s_i}\\right)
                                -\\frac{1}{2n^2}\\left({s_i^2-1}\\right)
                                }\\right)
                                +\\mu_w
                            }\\right]+u_w

    The pore pressure in the undisturbed zone is:

    .. math:: u(r) = \\frac{u_{avg}-u_w}{\\mu+\\mu_w}
                            \\left[{
                                \\ln\\left({\\frac{s_i}{s}}\\right)
                                -\\frac{1}{2n^2}\\left({s_i^2-s^2}\\right)
                                +\\kappa\\left[{
                                \\ln\\left({s}\\right)
                                -\\frac{1}{2n^2}\\left({s^2-1}\\right)
                            }\\right]
                                +\\mu_w
                            }\\right]+u_w

        where:

    .. math:: n = \\frac{r_e}{r_w}

    .. math:: s = \\frac{r_s}{r_w}

    .. math:: \\kappa = \\frac{k_h}{k_s}

    :math:`r_w` is the drain radius, :math:`r_e` is the drain influence radius,
    :math:`r_s` is the smear zone radius, :math:`k_h` is the undisturbed
    horizontal permeability, :math:`k_s` is the smear zone horizontal
    permeability

    References
    ----------
    .. [1] Hansbo, S. 1981. 'Consolidation of Fine-Grained Soils by
           Prefabricated Drains'. In 10th ICSMFE, 3:677-82.
           Rotterdam-Boston: A.A. Balkema.

    """

    def constant_part(n, s, kap, si):
        """u in smear zone with constant permeability i.e from si=1 to si=s"""

        term2 = log(si) - 1 / (2 * n ** 2) * (si ** 2 - 1)
        u = kap * term2
        return u

    def undisturbed_part(n, s, kap, si):
        """u outside of smear zone with constant permeability i.e from si=1 to si=s"""

        term4 = (log(si / s) - 1 / (2 * n ** 2) * (si ** 2 - s ** 2)
                    + kap * (log(s) - 1 / (2 * n ** 2) * (s ** 2 - 1)))

        u = term4
        return u
    n = np.asarray(n)
    s = np.asarray(s)
    kap = np.asarray(kap)        
    if n<=1.0:
        raise ValueError('n must be greater than 1. You have n = {}'.format(
            n))

    if s<1.0:
        raise ValueError('s must be greater than 1. You have s = {}'.format(
            s))

    if kap<=0.0:
        raise ValueError('kap must be greater than 0. You have kap = '
                '{}'.format(kap))

    if s>n:
        raise ValueError('s must be less than n. You have s = '
                '{} and n = {}'.format(s, n))
    si = np.atleast_1d(si)
    if np.any((si < 1) | (si > n)):
        raise ValueError('si must satisfy 1 >= si >= n)')


    if np.isclose(s, 1) or np.isclose(kap, 1):
        return u_ideal(n, si, uavg, uw, muw)




    mu = mu_constant(n, s, kap)
    term1 = (uavg - uw) / (mu + muw)
    term2 = np.empty_like(si, dtype=float)
    smear = (si < s)
    term2[smear] = constant_part(n, s, kap, si[smear])
    term2[~smear] = undisturbed_part(n, s, kap, si[~smear])


    u = term1 * (term2 + muw) + uw
    return u



def u_linear(n, s, kap, si, uavg=1, uw=0, muw=0):
    """Pore pressure at radius for linear smear zone

    Parameters
    ----------
    n : float
        Ratio of drain influence radius  to drain radius (re/rw).
    s : float
        Ratio of smear zone radius to  drain radius (rs/rw).
    kap : float
        Ratio of undisturbed horizontal permeability to permeability at
        the drain-soil interface (kh / ks).
    si : float of ndarray of float
        Normalised radial coordinate(s) at which to calc the pore pressure
        i.e. si=ri/rw.
    uavg : float, optional = 1
        Average pore pressure in soil. default = 1.  when `uw`=0 , then if
        uavg=1.
    uw : float, optional
        Pore pressure in drain, default = 0.
    muw : float, optional
        Well resistance mu parameter.

    Returns
    -------
    u : float or ndarray of float
        Pore pressure at specified si.

    Notes
    -----
    The uavg is calculated from the eta method.  It is not the uavg used when
    considering the vacuum as an equivalent surcharge.  You would have to do
    other manipulations for that.

    Noteing that :math:`s_i=r_i/r_w`, the radial pore pressure distribution
    in the smear zone is given by:

    .. math:: u^\\prime(r) = \\frac{u_{avg}-u_w}{\\mu+\\mu_w}
                            \\left[{
                                \\kappa\\left({\\frac{1}{B}\\ln\\left({s_i}\\right)
                                +\\left({\\frac{B}{A^2n^2}-\\frac{1}{B}}\\right)
                                \\ln\\left({B+As_i}\\right)
                                +\\frac{1-s_i}{An^2}
                                }\\right)
                                +\\mu_w
                            }\\right]+u_w

    The pore pressure in the undisturbed zone is:

    .. math:: u(r) = \\frac{u_{avg}-u_w}{\\mu+\\mu_w}
                            \\left[{
                                \\ln\\left({\\frac{s_i}{s}}\\right)
                                -\\frac{s_i^2-s^2}{2n^2}
                                +\\kappa
                            \\left[{
                                \\frac{1}{B}\\ln\\left({s}\\right)
                                +\\left({\\frac{B}{A^2n^2}-\\frac{1}{B}}\\right)
                                \\ln\\left({\\kappa}\\right)
                                +\\frac{1-s}{An^2}
                            }\\right]
                                +\\mu_w
                            }\\right]+u_w

    for the special case where :math:`s=\\kappa` the pore pressure
    in the undisturbed zone is:

    .. math:: u^\\prime(r) = \\frac{u_{avg}-u_w}{\\mu+\\mu_w}
                            \\left[{
                                s\\frac{\\left({n^2-s_i}\\right)
                                    \\left({s_i-1}\\right)}{n^2s_i}
                                +\\mu_w
                            }\\right]+u_w

    The pore pressure in the undisturbed zone is:

        .. math:: u(r) = \\frac{u_{avg}-u_w}{\\mu+\\mu_w}
                            \\left[{
                                \\ln\\left({\\frac{s_i}{s}}\\right)
                                +s-1+\\frac{s}{n^2}
                                -\\frac{s_i^2-s^2}{2n^2}
                                +\\mu_w
                            }\\right]+u_w

    where:

    .. math:: n = \\frac{r_e}{r_w}

    .. math:: s = \\frac{r_s}{r_w}

    .. math:: \\kappa = \\frac{k_h}{k_s}

    :math:`r_w` is the drain radius, :math:`r_e` is the drain influence radius,
    :math:`r_s` is the smear zone radius, :math:`k_h` is the undisturbed
    horizontal permeability, :math:`k_s` is the smear zone horizontal
    permeability

    If :math:`s=1` or :math:`\\kappa=1` then u_ideal will be used.

    References
    ----------
    .. [1] Walker, R., and B. Indraratna. 2007. 'Vertical Drain Consolidation
           with Overlapping Smear Zones'. Geotechnique 57 (5): 463-67.
           doi:10.1680/geot.2007.57.5.463.

    """

    def linear_part(n, s, kap, si):
        """u in smear zone with linear permeability i.e from si=1 to si=s"""


        if np.isclose(s, kap):
            term2 = -1 / si - 1 / n ** 2 * (si - 1) + 1
            u = kap * term2
            return u
        else:
            A = (kap - 1) / (s - 1)
            B = (s - kap) / (s - 1)

            term2 = log(si) - log(A * si + B)
            term3 = A * si + B - 1 - B * log(A * si + B)

            u = (1 / B * term2 - 1 / (n ** 2 * A ** 2) * term3)
            return kap * u

        return u

    def undisturbed_part(n, s, kap, si):
        """u outside of smear zone with linear permeability i.e from si=1 to si=s"""

        if np.isclose(s, kap):

            term2 = log(si / s) - 1 / (2 * n ** 2) * (si ** 2 - s ** 2)
            term3 = -1 / s - 1 / n ** 2 * (s - 1) + 1

            u = (term2 + kap * term3)
            return u
        else:
            A = (kap - 1) / (s - 1)
            B = (s - kap) / (s - 1)

            term2 = log(si / s) - 1 / (2 * n ** 2) * (si ** 2 - s ** 2)

            term3 = (1 / B * log(s / kap) - 1 / (n ** 2 * A ** 2) *
                        (kap - 1 - B * log(kap)))

            u = (term2 + kap * term3)
            return u
    n = np.asarray(n)
    s = np.asarray(s)
    kap = np.asarray(kap)            
    if n<=1.0:
        raise ValueError('n must be greater than 1. You have n = {}'.format(
            n))

    if s<1.0:
        raise ValueError('s must be greater than 1. You have s = {}'.format(
            s))

    if kap<=0.0:
        raise ValueError('kap must be greater than 0. You have kap = '
                '{}'.format(kap))

    if s>n:
        raise ValueError('s must be less than n. You have s = '
                '{} and n = {}'.format(s, n))
    si = np.atleast_1d(si)
    if np.any((si < 1) | (si > n)):
        raise ValueError('si must satisfy 1 >= si >= n)')


    if np.isclose(s, 1) or np.isclose(kap, 1):
        return u_ideal(n, si, uavg, uw, muw)


    mu = mu_linear(n, s, kap)
    term1 = (uavg - uw) / (mu + muw)
    term2 = np.empty_like(si, dtype=float)
    smear = (si < s)
    term2[smear] = linear_part(n, s, kap, si[smear])
    term2[~smear] = undisturbed_part(n, s, kap, si[~smear])


    u = term1 * (term2 + muw) + uw
    return u


def u_parabolic(n, s, kap, si, uavg=1, uw=0, muw=0):
    """Pore pressure at radius for parabolic smear zone

    Parameters
    ----------
    n : float
        Ratio of drain influence radius  to drain radius (re/rw).
    s : float
        Ratio of smear zone radius to  drain radius (rs/rw).
    kap : float
        Ratio of undisturbed horizontal permeability to permeability at
        the drain-soil interface (kh / ks).
    si : float of ndarray of float
        Normalised radial coordinate(s) at which to calc the pore pressure
        i.e. si=ri/rw.
    uavg : float, optional = 1
        Average pore pressure in soil. default = 1.  when `uw`=0 , then if
        uavg=1.
    uw : float, optional
        Pore pressure in drain, default = 0.
    muw : float, optional
        Well resistance mu parameter.

    Returns
    -------
    u : float of ndarray of float
        Pore pressure at specified si.

    Notes
    -----
    The uavg is calculated from the eta method.  It is not the uavg used when
    considering the vacuum as an equivalent surcharge.  You would have to do
    other manipulations for that.

    Noteing that :math:`s_i=r_i/r_w`, the radial pore pressure distribution
    in the smear zone is given by:

    .. math:: u^\\prime(r) = \\frac{u_{avg}-u_w}{\\mu+\\mu_w}
                            \\left[{
                                \\frac{\\kappa}{\\kappa-1}\\left\\{{
                                \\frac{1}{A^2-B^2}
                                    \\left({
                                    \\ln\\left({s_i}\\right)
                                    -\\frac{1}{2A}
                                        \\left[{
                                        \\left({A-B}\\right)F
                                        +\\left({A+B}\\right)G
                                        }\\right]
                                    }\\right)
                                    +\\frac{1}{2n^2AC}
                                        \\left[{
                                        \\left({A+B}\\right)F
                                        +\\left({A-B}\\right)G
                                        }\\right]
                                }\\right\\}
                                +\\mu_w
                            }\\right]+u_w

    The pore pressure in the undisturbed zone is:

    .. math:: u(r) = \\frac{u_{avg}-u_w}{\\mu+\\mu_w}
                            \\left[{
                                \\ln\\left({\\frac{s_i}{s}}\\right)
                                -\\frac{s_i^2-s^2}{2n^2}
                                +A^2
                            \\left[{
                                \\frac{1}{A^2-B^2}
                                    \\left({
                                        \\ln\\left({s}\\right)
                                        -\\frac{1}{2}\\left[{
                                            \\ln\\left({\\kappa}\\right)
                                            +\\frac{BE}{A}}\\right]
                                    }\\right)
                                +\\frac{1}{2n^2C^2}
                                    \\left({\\ln\\left({\\kappa}\\right)
                                    -\\frac{BE}{A}}\\right)
                            }\\right]
                                +\\mu_w
                            }\\right]+u_w

    where :math:`A`, :math:`B`, :math:`C`, :math:`E`, :math:`F`, and
    :math:`G`  are:

    .. math:: A=\\sqrt{\\frac{\\kappa}{\\kappa-1}}

    .. math:: B=\\frac{s}{s-1}

    .. math:: C=\\frac{1}{s-1}

    .. math:: E=\\ln\\left({\\frac{A+1}{A-1}}\\right)

    .. math:: F(r/r_w) = \\ln\\left({\\frac{A+B-Cs_i}{A+1}}\\right)

    .. math:: G(r/r_w) = \\ln\\left({\\frac{A-B+Cs_i}{A-1}}\\right)


    and:

    .. math:: n = \\frac{r_e}{r_w}

    .. math:: s = \\frac{r_s}{r_w}

    .. math:: \\kappa = \\frac{k_h}{k_s}


    :math:`r_w` is the drain radius, :math:`r_e` is the drain influence radius,
    :math:`r_s` is the smear zone radius, :math:`k_h` is the undisturbed
    horizontal permeability, :math:`k_s` is the smear zone horizontal
    permeability

    References
    ----------
    .. [1] Walker, Rohan, and Buddhima Indraratna. 2006. 'Vertical Drain
           Consolidation with Parabolic Distribution of Permeability in
           Smear Zone'. Journal of Geotechnical and Geoenvironmental
           Engineering 132 (7): 937-41.
           doi:10.1061/(ASCE)1090-0241(2006)132:7(937).

    """

    def parabolic_part(n, s, kap, si):
        """u in smear zone with parabolic permeability i.e from si=1 to si=s"""


        A = sqrt((kap / (kap - 1)))
        B = s / (s - 1)
        C = 1 / (s - 1)
        E = log((A + 1)/(A - 1))
        F = log((A + B - C * si) / (A + 1))
        G = log((A - B + C * si) / (A - 1))

        term1 = kap / (kap - 1)

        term2 = 1 / (A ** 2 - B ** 2)
        term3 = log(si)
        term4 = -1 / (2 * A)
        term5 = (A - B) * F + (A + B) * G
        term6 = term2 * (term3 + term4 * term5)

        term7 = 1 / (2 * n ** 2 * A * C ** 2)
        term8 = (A + B) * F + (A - B) * G
        term9 = term7 * term8

        u = term1 * (term6 + term9)
        return u

    def undisturbed_part(n, s, kap, si):
        """u outside of smear zone with parabolic permeability i.e from si=1 to si=s"""

        A = sqrt((kap / (kap - 1)))
        B = s / (s - 1)
        C = 1 / (s - 1)
        E = log((A + 1)/(A - 1))

        term1 = 1
        term2 = log(si / s) - 1 / (2 * n ** 2) * (si ** 2 - s ** 2)

        term3 = 1 / (A ** 2 - B ** 2)
        term4 = log(s) - 1 / 2 * (log(kap) + B / A * E)
        term5 = 1 / (2 * n ** 2 * C ** 2)
        term6 = (log(kap) - B / A * E)
        term7 = kap / (kap - 1) * (term3 * term4 + term5 * term6)

        u = term1 * (term2 + term7)
        return u
        
    n = np.asarray(n)
    s = np.asarray(s)
    kap = np.asarray(kap)
    if n<=1.0:
        raise ValueError('n must be greater than 1. You have n = {}'.format(
            n))

    if s<1.0:
        raise ValueError('s must be greater than 1. You have s = {}'.format(
            s))

    if kap<=0.0:
        raise ValueError('kap must be greater than 0. You have kap = '
                '{}'.format(kap))

    if s>n:
        raise ValueError('s must be less than n. You have s = '
                '{} and n = {}'.format(s, n))
    si = np.atleast_1d(si)
    if np.any((si < 1) | (si > n)):
        raise ValueError('si must satisfy 1 >= si >= n)')


    if np.isclose(s, 1) or np.isclose(kap, 1):
        return u_ideal(n, si, uavg, uw, muw)




    mu = mu_parabolic(n, s, kap)
    term1 = (uavg - uw) / (mu + muw)
    term2 = np.empty_like(si, dtype=float)
    smear = (si < s)
    term2[smear] = parabolic_part(n, s, kap, si[smear])
    term2[~smear] = undisturbed_part(n, s, kap, si[~smear])


    u = term1 * (term2 + muw) + uw
    return u


def u_piecewise_constant(s, kap, si, uavg=1, uw=0, muw=0, n=None, kap_m=None):
    """Pore pressure at radius for piecewise constant permeability distribution


    Parameters
    ----------
    s : list or 1d ndarray of float
        Ratio of segment outer radii to drain radius (r_i/r_0). The first value
        of s should be greater than 1, i.e. the first value should be s_1;
        s_0=1 at the drain soil interface is implied.
    kap : list or ndarray of float
        Ratio of undisturbed horizontal permeability to permeability in each
        segment kh/khi.
    si : float of ndarray of float
        Normalised radial coordinate(s) at which to calc the pore pressure
        i.e. si=ri/rw.
    uavg : float, optional = 1
        Average pore pressure in soil. default = 1.  when `uw`=0 , then if
        uavg=1.
    uw : float, optional
        Pore pressure in drain, default = 0.
    muw : float, optional
        Well resistance mu parameter
    n, kap_m : float, optional
        If `n` and `kap_m` are given then they will each be appended to `s` and
        `kap`. This allows the specification of a smear zone separate to the
        specification of the drain influence radius.
        Default n=kap_m=None, i.e. soilpermeability is completely described
        by `s` and `kap`. If n is given but kap_m is None then the last
        kappa value in kap will be used.

    Returns
    -------
    u : float of ndarray of float
        Pore pressure at specified si.

    Notes
    -----
    The pore pressure in the ith segment is given by:

    .. math:: u_i(r) = \\frac{u_{avg}-u_w}{\\mu+\\mu_w}
                            \\left[{
                                \\kappa_i\\left({\\ln\\left({\\frac{r}{r_{i-1}}}\\right)
                                -\\frac{r^2/r_0^2-s_{i-1}^2}{2n^2}}\\right)
                                +\\psi_i+\\mu_w
                            }\\right]+u_w

    where,

    .. math:: \\psi_{i} = \\sum\\limits_{j=1}^{i-1}\\kappa_j
                \\left[{
                        \\ln
                            \\left({
                                \\frac{s_j}{s_{j-1}}
                            }\\right)
                        -\\frac{s_j^2-s_{j-1}^2}{2n^2}
                    }\\right]

    and:

    .. math:: n = \\frac{r_m}{r_0}

    .. math:: s_i = \\frac{r_i}{r_0}

    .. math:: \\kappa_i = \\frac{k_h}{k_{hi}}


    :math:`r_0` is the drain radius, :math:`r_m` is the drain influence radius,
    :math:`r_i` is the outer radius of the ith segment,
    :math:`k_h` is the undisturbed
    horizontal permeability in the ith segment,
    :math:`k_{hi}` is the horizontal
    permeability in the ith segment

    References
    ----------
    .. [1] Walker, Rohan. 2006. 'Analytical Solutions for Modeling Soft Soil
           Consolidation by Vertical Drains'. PhD Thesis, Wollongong, NSW,
           Australia: University of Wollongong. http://ro.uow.edu.au/theses/501
    .. [2] Walker, Rohan T. 2011. 'Vertical Drain Consolidation Analysis in
           One, Two and Three Dimensions'. Computers and
           Geotechnics 38 (8): 1069-77. doi:10.1016/j.compgeo.2011.07.006.

    """


    s = np.atleast_1d(s)
    kap = np.atleast_1d(kap)

    if not n is None:
        s_temp = np.empty(len(s) + 1, dtype=float)
        s_temp[:-1] = s
        s_temp[-1] = n
        kap_temp = np.empty(len(kap) + 1, dtype=float)
        kap_temp[:-1] = kap
        if kap_m is None:
            kap_temp[-1] = kap[-1]
        else:
            kap_temp[-1] = kap_m
        s = s_temp
        kap = kap_temp

    if len(s)!=len(kap):
        raise ValueError('s and kap must have the same shape.  You have '
            'lengths for s, kap of {}, {}.'.format(
            len(s), len(kap)))

    if np.any(s<=1.0):
        raise ValueError('must have all s>=1. You have s = {}'.format(
            ', '.join([str(v) for v in np.atleast_1d(s)])))

    if np.any(kap<=0.0):
        raise ValueError('all kap must be greater than 0. You have kap = '
                '{}'.format(', '.join([str(v) for v in np.atleast_1d(kap)])))

    if np.any(np.diff(s) <= 0):
        raise ValueError('s must increase left to right you have s = '
        '{}'.format(', '.join([str(v) for v in np.atleast_1d(s)])))





    n = s[-1]
    si = np.atleast_1d(si)
    if np.any((si < 1) | (si > n)):
        raise ValueError('si must satisfy 1 >= si >= s[-1])')

    s_ = np.ones_like(s)
    s_[1:] = s[:-1]

    u = np.empty_like(si, dtype=float )

    segment = np.searchsorted(s, si)

    mu = mu_piecewise_constant(s, kap)

    term1 = (uavg - uw) / (mu + muw)

    for ii, i in enumerate(segment):

        sumj = 0
        for j in range(i):
            sumj += (kap[j] * (log(s[j] / s_[j])
                    - 0.5 * (s[j] ** 2 / n ** 2 - s_[j] ** 2 / n ** 2)))
        sumj = sumj / kap[i]

        u[ii] = kap[i] * (
                log(si[ii] / s_[i])
                - 0.5 * (si[ii] ** 2 / n ** 2 - s_[i] ** 2 / n ** 2)
                + sumj
                ) + muw

    u *= term1
    u += uw
    return u


def u_piecewise_linear(s, kap, si, uavg=1, uw=0, muw=0, n=None, kap_m=None):
    """Pore pressure at radius for piecewise constant permeability distribution

    Parameters
    ----------
    s : list or 1d ndarray of float
        Ratio of radii to drain radius (r_i/r_0). The first value
        of s should be 1, i.e. at the drain soil interface.
    kap : list or ndarray of float
        Ratio of undisturbed horizontal permeability to permeability at each
        value of s.
    si : float of ndarray of float
        Normalised radial coordinate(s) at which to calc the pore pressure
        i.e. si=ri/rw.
    uavg : float, optional = 1
        Average pore pressure in soil. default = 1.  when `uw`=0 , then if
        uavg=1.
    uw : float, optional
        Pore pressure in drain, default = 0.
    muw : float, optional
        Well resistance mu parameter.
    n, kap_m : float, optional
        If `n` and `kap_m` are given then they will each be appended to `s` and
        `kap`. This allows the specification of a smear zone separate to the
        specification of the drain influence radius.
        Default n=kap_m=None, i.e. soilpermeability is completely described
        by `s` and `kap`. If n is given but kap_m is None then the last
        kappa value in kap will be used.


    Returns
    -------
    u : float or ndarray of float
        Pore pressure at specified si.

    Notes
    -----
    With permeability in the ith segment defined by:

    .. math:: \\frac{k_i}{k_{ref}}= \\frac{1}{\\kappa_{i-1}}
                    \\left({A_ir/r_w+B_i}\\right)

    .. math:: A_i = \\frac{\\kappa_{i-1}/\\kappa_i-1}{s_i-s_{i-1}}

    .. math:: B_i = \\frac{s_i-s_{i-1}\\kappa_{i-1}/\\kappa_i}{s_i-s_{i-1}}

    The pore pressure in the ith segment is given by:

    .. math:: u_i(s) = \\frac{u_{avg}-u_w}{\\mu+\\mu_w}
                            \\left[{
                            \\sum\\limits_{i=1}^{m}\\kappa_{i-1}\\phi_i
                            + \\Psi_i
                            +\\mu_w
                        }\\right]+u_w

    where,

    .. math:: \\phi_i = \\left\\{
                            \\begin{array}{lr}
                            \\ln\\left[{\\frac{s}{s_{i-1}}}\\right]
                            - \\frac{s^2- s_{i-1}^2}{2n^2}
                            & \\textrm{for } \\frac{\\kappa_{i-1}}{\\kappa_i}=1 \\\\
                            \\frac{\\left({s - s_{i-1}}\\right)
                                \\left({n^2-ss_{i-1}}\\right)}{sn^2}
                            & \\textrm{for }\\frac{\\kappa_{i-1}}{\\kappa_i}=
                             \\frac{s_i}{s_{i-1}} \\\\
                             \\begin{multline}
                            \\frac{1}{B_i}\\ln\\left[{\\frac{s}{s_{i-1}}}\\right]
                                +\\ln\\left[{A_is+B_i}\\right]
                                \\left({\\frac{B_i}{A_i^2n^2}-\\frac{1}{B_i}}\\right)
                                \\\\-\\frac{s-s_{i-1}}{A_i^2n^2}
                            \\end{multline}
                            & \\textrm{otherwise}
                            \\end{array}\\right.

    .. math:: \\Psi_i = \\sum\\limits_{j=1}^{i-1}\\kappa_{j-1}\\psi_j

    .. math:: \\psi_i = \\left\\{
                            \\begin{array}{lr}
                            \\ln\\left[{\\frac{s_j}{s_{j-1}}}\\right]
                            - \\frac{s_j^2- s_{j-1}^2}{2n^2}
                            & \\textrm{for } \\frac{\\kappa_{j-1}}{\\kappa_j}=1 \\\\
                            \\frac{\\left({s_j - s_{j-1}}\\right)
                                \\left({n^2-s_js_{j-1}}\\right)}{s_jn^2}
                            & \\textrm{for }\\frac{\\kappa_{j-1}}{\\kappa_j}=
                             \\frac{s_j}{s_{j-1}} \\\\
                             \\begin{multline}
                            \\frac{1}{B_i}\\ln\\left[{\\frac{s_j}{s_{j-1}}}\\right]
                                +\\ln\\left[{\\frac{\\kappa_{j-1}}{\\kappa_j}}\\right]
                                \\left({\\frac{B_j}{A_j^2n^2}-\\frac{1}{B_j}}\\right)
                                \\\\-\\frac{s_j-s_{j-1}}{A_j^2n^2}
                            \\end{multline}
                            & \\textrm{otherwise}
                            \\end{array}\\right.



    and:

    .. math:: n = \\frac{r_m}{r_0}

    .. math:: s_i = \\frac{r_i}{r_0}

    .. math:: \\kappa_i = \\frac{k_h}{k_{ref}}


    :math:`r_0` is the drain radius, :math:`r_m` is the drain influence radius,
    :math:`r_i` is the radius of the ith radial point,
    :math:`k_{ref}` is a convienient refernce permeability, usually
    the undisturbed
    horizontal permeability,
    :math:`k_{hi}` is the horizontal
    permeability at the ith radial point

    References
    ----------
    Derived by Rohan Walker in 2011 and 2014.
    Derivation steps are the same as for mu_piecewise_constant in appendix of
    [1]_ but permeability is linear in a segemetn as in [2]_.

    .. [1] Walker, Rohan. 2006. 'Analytical Solutions for Modeling Soft Soil
           Consolidation by Vertical Drains'. PhD Thesis, Wollongong, NSW,
           Australia: University of Wollongong. http://ro.uow.edu.au/theses/501
    .. [2] Walker, R., and B. Indraratna. 2007. 'Vertical Drain Consolidation
           with Overlapping Smear Zones'. Geotechnique 57 (5): 463-67.
           doi:10.1680/geot.2007.57.5.463.

    """


    s = np.atleast_1d(s)
    kap = np.atleast_1d(kap)

    if not n is None:
        s_temp = np.empty(len(s) + 1, dtype=float)
        s_temp[:-1] = s
        s_temp[-1] = n
        kap_temp = np.empty(len(kap) + 1, dtype=float)
        kap_temp[:-1] = kap
        if kap_m is None:
            kap_temp[-1] = kap[-1]
        else:
            kap_temp[-1] = kap_m
        s = s_temp
        kap = kap_temp

    if len(s)!=len(kap):
        raise ValueError('s and kap must have the same shape.  You have '
            'lengths for s, kap of {}, {}.'.format(
            len(s), len(kap)))

    if np.any(s<1.0):
        raise ValueError('must have all s>=1. You have s = {}'.format(
            ', '.join([str(v) for v in np.atleast_1d(s)])))

    if np.any(kap<=0.0):
        raise ValueError('all kap must be greater than 0. You have kap = '
                '{}'.format(', '.join([str(v) for v in np.atleast_1d(kap)])))

    if np.any(np.diff(s) < 0):
        raise ValueError('s must increase left to right. you have s = '
        '{}'.format(', '.join([str(v) for v in np.atleast_1d(s)])))


    n = s[-1]
    si = np.atleast_1d(si)
    if np.any((si < 1) | (si > n)):
        raise ValueError('si must satisfy 1 >= si >= s[-1])')

    s_ = np.ones_like(s)
    s_[1:] = s[:-1]

    u = np.empty_like(si, dtype=float)

    segment = np.searchsorted(s, si)
    segment[segment==0] = 1 # put si=1 in first segment

    mu = mu_piecewise_linear(s, kap)

    term1 = (uavg - uw) / (mu + muw)

    for ii, i in enumerate(segment):



        #phi
        if np.isclose(kap[i-1]/kap[i], 1.0):
            phi = log(si[ii]/s[i-1]) - (si[ii]**2 - s[i-1]**2)/(2 * n**2)
        elif np.isclose(kap[i-1]/kap[i], s[i]/s[i-1]):
            phi = (si[ii]-s[i-1]) * (n**2 - s[i-1]*si[ii]) / (si[ii] * n**2)
        else:
            A = (kap[i-1] / kap[i] - 1) / (s[i] - s[i-1])
            B = (s[i] - s[i-1] * kap[i-1] / kap[i])/ (s[i] - s[i-1])
            phi = (1/B * log(si[ii]/s[i-1])
                    + (B/A**2/n**2 - 1/B) * log(A*si[ii] + B)
                    - (si[ii]-s[i-1])/A/n**2)

        psi = 0
        for j in range(1, i):
            if np.isclose(s[j - 1], s[j]):
                pass
            elif np.isclose(kap[j-1]/kap[j], 1.0):
                psi += kap[j-1]*(log(s[j]/s[j-1]) - (s[j]**2 - s[j-1]**2)/(2 * n**2))
            elif np.isclose(kap[j-1]/kap[j], s[j]/s[j-1]):
                psi += kap[j-1]*((s[j]-s[j-1]) * (n**2 - s[j-1]*s[j]) / (s[j] * n**2))
            else:
                A = (kap[j-1] / kap[j]-1) / (s[j] - s[j-1])
                B = (s[j] - s[j-1] * kap[j-1] / kap[j])/ (s[j] - s[j-1])
                psi += kap[j-1]*((1/B * log(s[j]/s[j-1])
                        + (B/A**2/n**2 - 1/B) * log(A*s[j] + B)
                        - (s[j]-s[j-1])/A/n**2))

        u[ii]=kap[i-1] * phi + psi + muw

    u *= term1
    u += uw
    return u


def re_from_drain_spacing(sp, pattern = 'Triangle'):
    """Calculate drain influence radius from drain spacing

    Parameters
    ----------
    sp : float
        Distance between drain centers.
    pattern : ['Triangle', 'Square'], optional
        Drain installation pattern. default = 'Triangle'.

    Returns
    -------
    re : float
        drain influence radius

    Notes
    -----

    The influence radius, :math:`r_e`, is given by:

    .. math:: r_e =
                \\left\\{\\begin{array}{lr}
                S_p \\frac{1}{\\sqrt{\\pi}}=S_p\\times 0.564189583
                & \\textrm{square pattern}\\\\
                S_p \\sqrt{\\frac{\\sqrt{3}}{2\\pi}}=S_p\\times 0.525037567
                & \\textrm{triangular pattern}
                \\end{array}\\right.

    References
    ----------
    Eta method is described in [1]_.

    .. [1] Walker, Rohan T. 2011. 'Vertical Drain Consolidation Analysis in
           One, Two and Three Dimensions'. Computers and
           Geotechnics 38 (8): 1069-77. doi:10.1016/j.compgeo.2011.07.006.

    """


    if np.any(np.atleast_1d(sp) <= 0):
        raise ValueError('sp must be greater than zero.  '
                            'You have sp={}'.format(sp))

    if pattern[0].upper()=='T':
        re = 0.525037567904332 * sp # factor = (3**0.5/2/np.pi)**0.5
    elif pattern[0].upper()=='S':
        re = 0.5641895835477563 * sp #factor = 1 / np.pi**0.5
    else:
        raise ValueError("pattern must begin with 'T' for triangular "
                        " or 'S' for square.  You have pattern="
                        "{}".format(pattern))
    return re

def drain_eta(re, mu_function, *args, **kwargs):
    """Calculate the vertical drain eta parameter for a specific smear zone

    eta = 2 / re**2 / (mu+muw)

    eta is used in radial consolidation equations u= u0 * exp(-eta*kh/gamw*t)

    Parameters
    ----------
    re : float
        Drain influence radius.
    mu_function : obj
        The mu_funtion to use. e.g. mu_ideal, mu_constant, mu_linear,
        mu_overlapping_linear, mu_parabolic, mu_piecewise_constant,
        mu_piecewise_linear.
    muw : float, optional
        Well resistance mu term, default=0.
    *args, **kwargs : various
        The arguments to pass to the mu_function.

    Returns
    -------
    eta : float
        Value of eta parameter

    Examples
    --------

    >>> drain_eta(1.5, mu_ideal, 10)
    0.56317834043349857
    >>> drain_eta(1.5, mu_constant, 5, 1.5, 1.6, muw=1)
    0.41158377241444855

    """

    muw = kwargs.pop('muw', 0)
    eta = 2 / re**2 / (mu_function(*args, **kwargs)+muw)
    return eta


def back_calc_drain_spacing_from_eta(eta, pattern, mu_function, rw, s, kap, muw=0):
    """Back calculate the required drain spacing to achieve a given eta

    eta = 2 / re**2 / (mu + muw)

    eta is used in radial consolidation equations u= u0 * exp(-eta*kh/gamw*t)


    Parameters
    ----------
    eta : float
        eta value.
    pattern : ['Triangle', 'Square']
        Drain installation pattern.
    mu_function : obj
        The mu_funtion to use. e.g. mu_ideal, mu_constant, mu_linear,
        mu_overlapping_linear, mu_parabolic, mu_piecewise_constant,
        mu_piecewise_linear.
    rw : float
        Drain/well radius.
    s : float or 1d array_like of float
        Ratio of smear zone radius to drain radius (rs/rw).  s can only be
        a 1d array is using a mu_piecewise function
    kap : float or 1d array_like of float
        Ratio of undisturbed horizontal permeability to permeability at
        in smear zone (kh / ks) (often at the drain-soil interface).  Be
        careful when defining s and kap for mu_piecewise_constant, and
        mu_piecewise_linear because the last value of kap will be used at
        the influence drain periphery.  In general the last value of kap
        should be one, representing the start of the undisturbed zone.
    muw : float, optional
        Well resistance mu term, default=0.

    Returns
    -------
    sp : float
        Drain spacing to get the required eta value
    re : float
        Drain influence radius
    n : float
        Ratio of drain influence radius to drain radius, re/rw

    Notes
    -----
    When using mu_piecewise_linear or mu_piecewise_constant only define s and
    kap up to the start of the undisturbed zone.  re will be varied.

    For anyting other than mu_overlapping_linear do not trust any returned
    spacing that gives an n value less than the extent of the smear zone.


    """

    def calc_eta(sp, eta, rw, s, kap, mu_function, pattern, muw=0):
        """eta from a given spacing value

        used in root finding

        """
        re = re_from_drain_spacing(sp, pattern)
        n = re/rw

        if mu_function != mu_ideal:
            if  n < np.max(s):
                if mu_function != mu_overlapping_linear:
                    raise ValueError('In determining required drain '
                        'spacing, n has fallen '
                        'below s. s={}, n={}'.format(np.max(s), n))

        if mu_function in [mu_piecewise_constant, mu_piecewise_linear]:
            eta_ = drain_eta(re, mu_function, s, kap, n = n, muw = muw)
        else:
            eta_ = drain_eta(re, mu_function, n, s, kap, muw=muw)
        return eta_ - eta

    from scipy.optimize import fsolve


    if not mu_function in [mu_piecewise_constant, mu_piecewise_linear]:
        if len(np.atleast_1d(s))>1:
            raise ValueError('for mu_function={}, you cannot have multiple '
                'values for s. s={}'.format(mu_function.__name__, s))
        if len(np.atleast_1d(kap))>1:
            raise ValueError('for mu_function={}, you cannot have multiple '
                'values for kap. kap={}'.format(mu_function.__name__, kap))

    x0 = rw * np.max(s) / 0.5 * 2 # this ensures guess is beyond smear zone
    calc_eta(x0, eta, rw, s, kap, mu_function, pattern, muw )
    sp = fsolve(calc_eta, x0,
                args=(eta, rw, s, kap, mu_function, pattern, muw))

    re = re_from_drain_spacing(sp[0], pattern)
    n = re/rw

    if mu_function != mu_ideal:
        if  n < np.max(s):
            if mu_function != mu_overlapping_linear:
                raise ValueError('calculated spacing results in n<s. s={}, n={}'.format(np.max(s), n))
    return sp[0], re, n








########################################################################



#scratch()
def scratch():
    """scratch pad for testing latex markup for docstrings

   """
    #scratch()
    pass


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest', '--doctest-options=+ELLIPSIS'])



    eta = 5
    pattern = 't'
    mu_function = mu_overlapping_linear
    rw = 0.05
    s = 5#[5,6]
    kap = 2#[2,1]
    muw = 1

    print(back_calc_drain_spacing_from_eta(eta, pattern, mu_function, rw, s, kap, muw))
#u_constant()
#k_overlapping_linear(()

    scratch()
#    print('lin',u_linear(5,2,3,[1.5,4]))
#    print('pwise', u_piecewise_linear([1,2,5],[3,1,1],[1.5,4]))
#    x = np.array(
#        [1.,   1.06779661,  1.13559322,  1.20338983,  1.27118644,
#        1.33898305,  1.40677966,  1.47457627,  1.54237288,  1.61016949,
#        1.6779661 ,  1.74576271,  1.81355932,  1.88135593,  1.94915254,
#        2.01694915,  2.08474576,  2.15254237,  2.22033898,  2.28813559,
#        2.3559322 ,  2.42372881,  2.49152542,  2.55932203,  2.62711864,
#        2.69491525,  2.76271186,  2.83050847,  2.89830508,  2.96610169,
#        3.03389831,  3.10169492,  3.16949153,  3.23728814,  3.30508475,
#        3.37288136,  3.44067797,  3.50847458,  3.57627119,  3.6440678 ,
#        3.71186441,  3.77966102,  3.84745763,  3.91525424,  3.98305085,
#        4.05084746,  4.11864407,  4.18644068,  4.25423729,  4.3220339 ,
#        4.38983051,  4.45762712,  4.52542373,  4.59322034,  4.66101695,
#        4.72881356,  4.79661017,  4.86440678,  4.93220339,  5., 30       ])
#
#    y = 1.0/np.array(
#        [ 0.5       ,  0.50847458,  0.51694915,  0.52542373,  0.53389831,
#        0.54237288,  0.55084746,  0.55932203,  0.56779661,  0.57627119,
#        0.58474576,  0.59322034,  0.60169492,  0.61016949,  0.61864407,
#        0.62711864,  0.63559322,  0.6440678 ,  0.65254237,  0.66101695,
#        0.66949153,  0.6779661 ,  0.68644068,  0.69491525,  0.70338983,
#        0.71186441,  0.72033898,  0.72881356,  0.73728814,  0.74576271,
#        0.75423729,  0.76271186,  0.77118644,  0.77966102,  0.78813559,
#        0.79661017,  0.80508475,  0.81355932,  0.8220339 ,  0.83050847,
#        0.83898305,  0.84745763,  0.8559322 ,  0.86440678,  0.87288136,
#        0.88135593,  0.88983051,  0.89830508,  0.90677966,  0.91525424,
#        0.92372881,  0.93220339,  0.94067797,  0.94915254,  0.95762712,
#        0.96610169,  0.97457627,  0.98305085,  0.99152542,  1., 1.        ])
#
#    mu_piecewise_linear(x,y)
#    mu_overlapping_linear(np.array([5,10]),
#                                    np.array([7, 12]),
#                                    np.array([1.6, 1.5,]))
#    mu_piecewise_linear([1, 5],
#                        [1, 1])
#
#    s=80
#    n=18
#    kap=8
#    x = np.linspace(1,n,50)
#    y = k_overlapping_linear(n,s, kap, x)
#    plt.plot(x,y)
#    plt.gca().grid()
#    plt.show()
#
#    xp = np.array(
#        [1.,    1.06779661,  1.13559322,  1.20338983,  1.27118644,
#        1.33898305,  1.40677966,  1.47457627,  1.54237288,  1.61016949,
#        1.6779661 ,  1.74576271,  1.81355932,  1.88135593,  1.94915254,
#        2.01694915,  2.08474576,  2.15254237,  2.22033898,  2.28813559,
#        2.3559322 ,  2.42372881,  2.49152542,  2.55932203,  2.62711864,
#        2.69491525,  2.76271186,  2.83050847,  2.89830508,  2.96610169,
#        3.03389831,  3.10169492,  3.16949153,  3.23728814,  3.30508475,
#        3.37288136,  3.44067797,  3.50847458,  3.57627119,  3.6440678 ,
#        3.71186441,  3.77966102,  3.84745763,  3.91525424,  3.98305085,
#        4.05084746,  4.11864407,  4.18644068,  4.25423729,  4.3220339 ,
#        4.38983051,  4.45762712,  4.52542373,  4.59322034,  4.66101695,
#        4.72881356,  4.79661017,  4.86440678,  4.93220339,  5., 30       ])
#
#    yp = 1.0/np.array(
#        [ 0.5       ,  0.51680552,  0.53332376,  0.54955473,  0.56549842,
#        0.58115484,  0.59652399,  0.61160586,  0.62640046,  0.64090779,
#        0.65512784,  0.66906061,  0.68270612,  0.69606435,  0.70913531,
#        0.72191899,  0.7344154 ,  0.74662453,  0.75854639,  0.77018098,
#        0.7815283 ,  0.79258834,  0.8033611 ,  0.8138466 ,  0.82404481,
#        0.83395576,  0.84357943,  0.85291583,  0.86196495,  0.8707268 ,
#        0.87920138,  0.88738868,  0.89528871,  0.90290147,  0.91022695,
#        0.91726515,  0.92401609,  0.93047975,  0.93665613,  0.94254525,
#        0.94814708,  0.95346165,  0.95848894,  0.96322896,  0.9676817 ,
#        0.97184717,  0.97572537,  0.97931629,  0.98261994,  0.98563631,
#        0.98836541,  0.99080724,  0.99296179,  0.99482907,  0.99640908,
#        0.99770181,  0.99870727,  0.99942545,  0.99985636,  1., 1.        ])
#
#
#
#    n=30
#    s=5
#    kap=2
#    muw=0
#    uw=-0.2
#    x = np.linspace(1, s, 60)
#    y = k_linear(n, s, kap, x)
#
#    x = np.linspace(1, n, 400)
#    y = u_ideal(n,x, uw=uw,muw=muw)
#    y2 = u_parabolic(n,s,kap,x, uw=uw,muw=muw)
#    y3 = u_linear(n,s,kap,x, uw=uw,muw=muw)
#    y4 = u_constant(n,s,kap,x, uw=uw,muw=muw)
#    y5 = u_piecewise_constant([s,n], [kap,1],x, uw=uw,muw=muw)
##    y6 = u_piecewise_linear([1,s,s,n], [kap,kap,1,1], x, uw=uw,muw=muw)
##
##    y7 = u_piecewise_linear([1,s,n], [kap,1,1], x, uw=uw,muw=muw)
##    y8 = u_piecewise_linear(xp, yp, x, uw=uw,muw=muw)
##    print(repr(x))
##    print(repr(y))
#    plt.plot(x,y, '-',label='ideal')
#    plt.plot(x, y2, '--',label='para')
#    plt.plot(x, y3, dashes=[5,2,2,2],label='lin')
#    plt.plot(x, y4, dashes=[8,2],label='const')
#    plt.plot(x, y5,'+',ms=2, label='pwisec')
##    plt.plot(x, y6,'o',ms=3, label='pwisel')
##    plt.plot(x, y7,'^',ms=3, label='pwisel_lin')
##    plt.plot(x, y8,'^',ms=3, label='pwisel_para')
#    leg=plt.gca().legend(loc=4)
#    plt.gca().grid()
#    plt.show()

    mu_piecewise_constant([1.5,5],
                                              [1.6,1])

    scratch()
    print(mu_parabolic(30,5,2))
    print(k_parabolic(30, 5, 2, [1, 1.13559322]))
    k_parabolic(20,1,2,[4,6,7])
#    mu_linear()
#    nose.runmodule(argv=['nose', '--verbosity=3'])
#    print(mu_ideal(0.5))
#    print(mu_linear(np.array([50,100]),
#                      np.array([10,20]),
#                      np.array([5,3])))
