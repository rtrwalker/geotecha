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
module for expressions and functions relating to vertical drain smear zone
permeability distributions

"""
from __future__ import print_function, division

import numpy as np
from matplotlib import pyplot as plt
#import cmath
from numpy import log, sqrt

def mu_ideal(n, *args):
    """smear zone permeability/geometry parameter for ideal drain (no smear zone)

    mu parameter in equal strain radial consolidation equations e.g.
    u = u0 * exp(-8*Th/mu)

    Parameters
    ----------
    n : float or ndarray of float
        ratio of drain influence radius  to drain radius (re/rw).

    args : anything
        args does not contribute to any calculations it is merely so you
        can have other arguments such as s and kappa which are used in other
        smear zone formulations

    Returns
    -------
    mu : float
        smear zone permeability/geometry parameter

    Notes
    -----
    The :math:`\\mu` parameter is given by:

    .. math:: \\mu=\\frac{n^2}{\\left({n^2-1}\\right)}
              \\left({\\ln\\left({n}\\right)-\\frac{3}{4}}\\right)+
              \\frac{1}{\\left({n^2-1}\\right)}\\left({1-\\frac{1}{4n^2}}
              \\right)

    where:

    .. math:: n = \\frac{r_e}{r_w}

    :math:r_w is the drain radius, :math:r_e is the drain influence radius

    """

    if np.any(n<=1):
        raise ValueError('n must be greater than 1. You have n = {}'.format(
            ', '.join([str(v) for v in np.atleast_1d(n)])))

    term1 = n**2 / (n**2 - 1) * (log(n) - 0.75)
    term2 = 1 / (n**2 - 1) * (1 - 1/(4 * n**2))
    mu = term1 + term2
    return mu


def mu_constant(n, s, kap):
    """vertical drain mu, for smear zone with constant permeability


    mu parameter in equal strain radial consolidation equations e.g.
    u = u0 * exp(-8*Th/mu)

    Parameters
    ----------
    n : float or ndarray of float
        ratio of drain influence radius  to drain radius (re/rw).
    s : float or ndarray of float
        ratio of smear zone radius to  drain radius (rs/rw)
    kap : float or ndarray of float
        ratio of undisturbed horizontal permeability to smear zone
        horizontal permeanility (kh / ks)

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

    :math:r_w is the drain radius, :math:r_e is the drain influence radius,
    :math:r_s is the smear zone radius, :math:k_h is the undisturbed
    horizontal permeability, :math:k_s is the smear zone horizontal
    permeability

    """

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



    term1 = n**2 / (n**2 - 1) * (log(n/s) + kap * log(s) - 0.75)
    term2 = s**2 / (n**2 - 1) * (1 - s**2 /(4 * n**2))
    term3 = kap / (n**2 - 1) * ((s**4 - 1) / (4 * n**2) - s**2 +1)
    mu = term1 + term2 + term3
    return mu

def _sx(n, s):
    """s value at intersection of linear smearzones


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
    """kap value at intersection of linear smearzones


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
    kapx =  1 + (kap - 1) / (s - 1) * (sx - 1)
    return kapx
def mu_overlapping_linear(n, s, kap):
    """vertical drain mu, for smear zone with linear variation of permeability
    that can overlap


    mu parameter in equal strain radial consolidation equations e.g.
    u = u0 * exp(-8*Th/mu)

    Parameters
    ----------
    n : float or ndarray of float
        ratio of drain influence radius  to drain radius (re/rw).
    s : float or ndarray of float
        ratio of smear zone radius to  drain radius (rs/rw)
    kap : float or ndarray of float
        ratio of undisturbed horizontal permeability to permeability at
        the drain-soil interface (kh / ks)

    Returns
    -------
    mu : float
        smear zone permeability/geometry parameter

    Notes
    -----
    The smear zone parameter :math:`\\mu` is given by:

    .. math:: \\mu_X =
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

    """

    def mu_intersecting(n, s, kap):
        """mu for intersecting smear zones that do not completely overlap"""
        sx = _sx(n, s)
        kapx =  _kapx(n, s, kap)
        mu = mu_linear(n, sx, kapx) * kap / kapx
        return mu

    if np.any(n<=1.0):
        raise ValueError('n must be greater than 1. You have n = {}'.format(
            ', '.join([str(v) for v in np.atleast_1d(n)])))

    if np.any(s<1.0):
        raise ValueError('s must be greater than 1. You have s = {}'.format(
            ', '.join([str(v) for v in np.atleast_1d(s)])))

    if np.any(kap<=0.0):
        raise ValueError('kap must be greater than 0. You have kap = '
                '{}'.format(', '.join([str(v) for v in np.atleast_1d(kap)])))

    is_array = any([isinstance(v, np.ndarray) for v in [n, s, kap]])

    n = np.atleast_1d(n)
    s = np.atleast_1d(s)
    kap = np.atleast_1d(kap)


    if len([v for v in [n, s] if v.shape==kap.shape])!=2:
        raise ValueError('n, s, and kap must have the same shape.  You have '
            'lengths for n, s, kap of {}, {}, {}.'.format(
            len(n), len(s), len(kap)))


    ideal = np.isclose(s, 1) | np.isclose(kap, 1)
    normal = (n >= s) & (~ideal)
    all_disturbed = (2*n-s <=1) & (~ideal)
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
    """vertical drain mu, for smear zone with linear variation of permeability


    mu parameter in equal strain radial consolidation equations e.g.
    u = u0 * exp(-8*Th/mu)

    Parameters
    ----------
    n : float or ndarray of float
        ratio of drain influence radius  to drain radius (re/rw).
    s : float or ndarray of float
        ratio of smear zone radius to  drain radius (rs/rw)
    kap : float or ndarray of float
        ratio of undisturbed horizontal permeability to permeability at
        the drain-soil interface (kh / ks)

    Returns
    -------
    mu : float
        smear zone permeability/geometry parameter

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

    """


    def mu_s_neq_kap(n,s,kap):
        """mu for when s != kap"""

        A = (kap - 1) / (s - 1)
        B = (s - kap) / (s - 1)

        term1 = n**2 / (n**2 - 1)
        term2 = log(n / s) + s ** 2 / (n ** 2) * (1 - s ** 2 / (4 * n ** 2)) - 3 / 4
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

    def mu_s_eq_kap(n,s):
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
    """vertical drain mu, smear zone with parabolic variation of permeability


    mu parameter in equal strain radial consolidation equations e.g.
    u = u0 * exp(-8*Th/mu)

    Parameters
    ----------
    n : float or ndarray of float
        ratio of drain influence radius  to drain radius (re/rw).
    s : float or ndarray of float
        ratio of smear zone radius to  drain radius (rs/rw)
    kap : float or ndarray of float
        ratio of undisturbed horizontal permeability to permeability at
        the drain-soil interface (kh / ks)

    Returns
    -------
    mu : float
        smear zone permeability/geometry parameter

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





def mu_piecewise_constant(s, kap):
    """vertical drain mu, smear zone with piecewise constant variation of
    permeability


    mu parameter in equal strain radial consolidation equations e.g.
    u = u0 * exp(-8*Th/mu)

    Parameters
    ----------
    s : list or 1d ndarray of float
        ratio of segment outer radii to drain radius (r_i/r_0). The first value
        of s should be greater than 1, i.e. the first value should be s_1;
        s_0=1 at the drain soil interface is implied
    kap : list or ndarray of float
        ratio of undisturbed horizontal permeability to permeability in each
        segment kh/khi

    Returns
    -------
    mu : float
        smear zone permeability/geometry parameter

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

    .. math:: \\psi_{i} = \\sum\\limits_{j=1}^{i-1}
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


    """


    s = np.atleast_1d(s)
    kap = np.atleast_1d(kap)

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
    s_ = np.ones_like(s)
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

def mu_piecewise_linear(s, kap):
    """vertical drain mu, smear zone with piecewise linear variation of
    permeability


    mu parameter in equal strain radial consolidation equations e.g.
    u = u0 * exp(-8*Th/mu)

    Parameters
    ----------
    s : list or 1d ndarray of float
        ratio of radii to drain radius (r_i/r_0). The first value
        of s should be 1, i.e. at the drain soil interface
    kap : list or ndarray of float
        ratio of undisturbed horizontal permeability to permeability at each
        value of s.

    Returns
    -------
    mu : float
        smear zone permeability/geometry parameter

    Notes
    -----

    The smear zone parameter :math:`\\mu` is given by:



#    .. math:: \\mu = \\frac{n^2}{\\left({n^2-1}\\right)}
#                \\sum\\limits_{i=1}^{m} \\kappa_i
#                    \\left[{
#                        \\frac{s_i^2}{n^2}\\ln
#                            \\left({
#                                \\frac{s_i}{s_{i-1}}
#                            }\\right)
#                        -\\frac{s_i^2-s_{i-1}^2}{2n^2}
#                        -\\frac{\\left({s_i^2-s_{i-1}^2}\\right)^2}{4n^4}
#                    }\\right]
#                    +\\psi_i\\frac{s_i^2-s_{i-1}^2}{n^2}
#
#    where,
#
#    .. math:: \\psi_{i} = \\sum\\limits_{j=1}^{i-1}
#                \\left[{
#                        \\ln
#                            \\left({
#                                \\frac{s_j}{s_{j-1}}
#                            }\\right)
#                        -\\frac{s_j^2-s_{j-1}^2}{2n^2}
#                    }\\right]

    and:

    .. math:: n = \\frac{r_m}{r_0}

    .. math:: s_i = \\frac{r_i}{r_0}

    .. math:: \\kappa_i = \\frac{k_h}{k_{hi}}


    :math:`r_0` is the drain radius, :math:`r_m` is the drain influence radius,
    :math:`r_i` is the radius at the ith radial point for defining the
    permeability,
    :math:`k_h` is the undisturbed
    horizontal permeability,
    :math:`k_{hi}` is the horizontal
    permeability at the ith radial point for defining permeability


    """


    s = np.atleast_1d(s)
    kap = np.atleast_1d(kap)

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


def k_parabolic(n, s, kap, si):
    """Permeability distribution for smear zone with parabolic permeability

    Normalised with respect to undisturbed permeability.  i.e. if you want the
    actual permeability then multiply by whatever you used to determine kap.


    permeability is parabolic with value 1/kap at the drain soil interface
    i.e. at s=1 k=k0=1/kap.  for si>s, permeability=1.

    Parameters
    ----------
    n : float
        ratio of drain influence radius  to drain radius (re/rw).
    s : float
        ratio of smear zone radius to  drain radius (rs/rw)
    kap : float
        ratio of undisturbed horizontal permeability to permeability at
        the drain-soil interface (kh / ks)
    si : float of ndarray of float
        normalised radial coordinate(s) at which to calc the permeability
        i.e. si=ri/rw

    Returns
    -------
    permeability : float or ndarray of float
        normalised permeability (i.e. ki/kh) at the si values.

    Notes
    -----
    parabolic distribution of permeability in smear zone is given by:

    .. math:: \\frac{k_h^'\\left({r}\\right)}{k_h}=
                \\frac{\\kappa-1}{\\kappa}
                \\left({A-B+C\\frac{r}{r_w}}\\right)
                \\left({A+B-C\\frac{r}{r_w}}\\right)

    where :math:`A`, :math:`B`, :math:`C` and :math:`E` are:

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

    """

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
        """parbolic smear zone part i.e from si=1 to si=s"""

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


    permeability is linear with value 1/kap at the drain soil interface
    i.e. at s=1 k=k0=1/kap.  for si>s, permeability=1.

    Parameters
    ----------
    n : float
        ratio of drain influence radius  to drain radius (re/rw).
    s : float
        ratio of smear zone radius to  drain radius (rs/rw)
    kap : float
        ratio of undisturbed horizontal permeability to permeability at
        the drain-soil interface (kh / ks)
    si : float of ndarray of float
        normalised radial coordinate(s) at which to calc the permeability
        i.e. si=ri/rw

    Returns
    -------
    permeability : float or ndarray of float
        normalised permeability (i.e. ki/kh) at the si values.

    Notes
    -----
    linear distribution of permeability in smear zone is given by:

    .. math:: \\frac{k_h^'\\left({r}\\right)}{k_h}=
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

    """

    def s_neq_kap_part(n, s, kap, si):
        """linear permeability in smear zome when s!=kap"""

        A = (kap - 1) / (s - 1)
        B = (s - kap) / (s - 1)


        k0 = 1 / kap

        return k0*(A*si+B)

    def s_eq_kap_part(n, s, si):
        """linear permeability in smear zome when s!=kap"""

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


class VerticalDrainSmearZone(object):
    """Smear zone around a vertical drain

    Optional keyword arguments:

          ============   =========================================
          Keyword        Description
          ============   =========================================
          *style*        [ 'sci' (or 'scientific') | 'plain' ]
                         plain turns off scientific notation
          *scilimits*    (m, n), pair of integers; if *style*
                         is 'sci', scientific notation will
                         be used for numbers outside the range
                         10`m`:sup: to 10`n`:sup:.
                         Use (0,0) to include all numbers.
          *useOffset*    [True | False | offset]; if True,
                         the offset will be calculated as needed;
                         if False, no offset will be used; if a
                         numeric offset is specified, it will be
                         used.
          *axis*         [ 'x' | 'y' | 'both' ]
          *useLocale*    If True, format the number according to
                         the current locale.  This affects things
                         such as the character used for the
                         decimal separator.  If False, use
                         C-style (English) formatting.  The
                         default setting is controlled by the
                         axes.formatter.use_locale rcparam.
          ============   =========================================

    """


    def __init__(self, **kwargs):
        pass



########################################################################
import unittest
from numpy.testing import assert_allclose
import nose
from nose.tools.trivial import assert_raises
from nose.tools.trivial import ok_

class test_mu_ideal(unittest.TestCase):
    """tests for mu_ideal"""

    def test_ideal(self):
        assert_allclose(mu_ideal(np.array([5,10,20,50,100])),
                        [0.936497825, 1.578343528,
                         2.253865374, 3.163688441,
                         3.855655749])

    def test_n_less_than_one(self):
        assert_raises(ValueError, mu_ideal, 0.5)

class test_mu_constant(unittest.TestCase):
    """tests for mu_constant"""

    def test_ideal(self):
        assert_allclose(mu_constant(np.array([5,10,20,50,100]),
                                    np.array([1,1,1,1,1]),
                                    np.array([5,10,20,50,100])),
                        [0.936497825, 1.578343528,
                         2.253865374, 3.163688441,
                         3.855655749])

    def test_constant(self):
        assert_allclose(mu_constant(np.array([5,10,20,50,100]),
                                    np.array([1.5, 2, 1, 4, 8]),
                                    np.array([1.6, 1, 5, 0.4, 4])),
                        [1.159679143, 1.578343528,
                         2.253865374, 2.335174298,
                         10.07573309])

    def test_n_less_than_one(self):
        assert_raises(ValueError, mu_constant, 0.5, 2, 5)

    def test_s_less_than_one(self):
        assert_raises(ValueError, mu_constant, 50, 0.5, 5)

    def test_kap_less_than_zero(self):
        assert_raises(ValueError, mu_constant, 50, 0.5, -5)

    def test_s_greater_than_n(self):
        assert_raises(ValueError, mu_constant, 50, 100, 5)


class test_mu_linear(unittest.TestCase):
    """tests for mu_linear"""

    def test_ideal(self):
        assert_allclose(mu_linear(np.array([5,10,20,50,100]),
                                    np.array([1,1,1,1,1]),
                                    np.array([5,10,20,50,100])),
                        [0.936497825, 1.578343528,
                         2.253865374, 3.163688441,
                         3.855655749])

    def test_linear(self):
        assert_allclose(mu_linear(np.array([5,10,20,50,100]),
                                    np.array([1.5, 2, 1, 4, 8]),
                                    np.array([1.6, 1, 5, 4, 4])),
                        [1.040117086, 1.578343528,
                         2.253865374, 4.774441621,
                         6.625207688])

    def test_n_less_than_one(self):
        assert_raises(ValueError, mu_linear, 0.5, 2, 5)

    def test_s_less_than_one(self):
        assert_raises(ValueError, mu_linear, 50, 0.5, 5)

    def test_kap_less_than_zero(self):
        assert_raises(ValueError, mu_linear, 50, 0.5, -5)

    def test_s_greater_than_n(self):
        assert_raises(ValueError, mu_linear, 50, 100, 5)

    def test_s_n_unequal_len(self):
        assert_raises(ValueError, mu_linear, [50,40], 100, 5)

class test_mu_overlapping_linear(unittest.TestCase):
    """tests for mu_overlapping_linear"""

    def test_ideal(self):
        assert_allclose(mu_overlapping_linear(np.array([5,10,20,50,100]),
                                    np.array([1,1,1,2,1]),
                                    np.array([5,10,20,1,100])),
                        [0.936497825, 1.578343528,
                         2.253865374, 3.163688441,
                         3.855655749])

    def test_linear(self):
        assert_allclose(mu_overlapping_linear(np.array([5,10,20,50,100]),
                                    np.array([1.5, 2, 1, 4, 8]),
                                    np.array([1.6, 1, 5, 4, 4])),
                        [1.040117086, 1.578343528,
                         2.253865374, 4.774441621,
                         6.625207688])

    def test_all_disturbed(self):
        assert_allclose(mu_overlapping_linear(np.array([5,10]),
                                    np.array([20, 30]),
                                    np.array([1.6, 1.5,])),
                        [1.498396521, 2.367515292])

    def test_intersecting(self):
        assert_allclose(mu_overlapping_linear(np.array([5,10]),
                                    np.array([7, 12]),
                                    np.array([1.6, 1.5,])),
                        [1.387620117, 2.200268994])

    def test_all_at_once(self):
        assert_allclose(mu_overlapping_linear(
                                    np.array([5,  10, 10, 10, 100]),
                                    np.array([1.5, 1, 30, 12, 8]),
                                    np.array([1.6, 1,1.5,1.5, 1])),
                        [1.040117086, 1.578343528,
                         2.367515292, 2.200268994,
                         3.855655749])

    def test_n_less_than_one(self):
        assert_raises(ValueError, mu_overlapping_linear, 0.5, 2, 5)

    def test_s_less_than_one(self):
        assert_raises(ValueError, mu_overlapping_linear, 50, 0.5, 5)

    def test_kap_less_than_zero(self):
        assert_raises(ValueError, mu_overlapping_linear, 50, 0.5, -5)


    def test_s_n_unequal_len(self):
        assert_raises(ValueError, mu_overlapping_linear, [50,40], 100, 5)


class test_mu_parabolic(unittest.TestCase):
    """tests for mu_parabolic"""

    def test_ideal(self):
        assert_allclose(mu_parabolic(np.array([5,10,20,50,100]),
                                    np.array([1,1,1,1,1]),
                                    np.array([5,10,20,50,100])),
                        [0.936497825, 1.578343528,
                         2.253865374, 3.163688441,
                         3.855655749])

    def test_parabolic(self):
        assert_allclose(mu_parabolic(np.array([5,10,20,50,100]),
                                    np.array([1.5, 2, 1, 4, 8]),
                                    np.array([1.6, 1, 5, 4, 4])),
                        [1.006231891, 1.578343528,
                         2.253865374, 4.258523315,
                         5.834098317])

    def test_n_less_than_one(self):
        assert_raises(ValueError, mu_parabolic, 0.5, 2, 5)

    def test_s_less_than_one(self):
        assert_raises(ValueError, mu_parabolic, 50, 0.5, 5)

    def test_kap_less_than_zero(self):
        assert_raises(ValueError, mu_parabolic, 50, 0.5, -5)

    def test_s_greater_than_n(self):
        assert_raises(ValueError, mu_parabolic, 50, 100, 5)

    def test_s_n_unequal_len(self):
        assert_raises(ValueError, mu_parabolic, [50,40], 100, 5)

class test_mu_piecewise_constant(unittest.TestCase):
    """tests for mu_piecewise_constant"""

    def test_ideal(self):
        assert_allclose(mu_piecewise_constant(5,
                                              1),
                        0.936497825)
    def test_ideal_multi(self):
        assert_allclose(mu_piecewise_constant([3,5],
                                              [1,1]),
                        0.936497825)

    def test_const(self):
        assert_allclose(mu_piecewise_constant([1.5, 5],
                                              [1.6, 1]),
                        1.159679143)

    def test_const_multi(self):
        assert_allclose(mu_piecewise_constant([1.3, 1.4, 1.5, 5],
                                              [1.6, 1.6, 1.6, 1]),
                        1.159679143)

    def test_const_two_smear_zones(self):
        assert_allclose(mu_piecewise_constant([1.5, 3, 5,],
                                              [2, 3, 1.0]),
                        2.253304564)

    def test_parabolic(self):
        """piecewise constant approximation of parabolic with n = 30, s=5, kap=2"""


        x = np.array(
        [   1.06779661,  1.13559322,  1.20338983,  1.27118644,
        1.33898305,  1.40677966,  1.47457627,  1.54237288,  1.61016949,
        1.6779661 ,  1.74576271,  1.81355932,  1.88135593,  1.94915254,
        2.01694915,  2.08474576,  2.15254237,  2.22033898,  2.28813559,
        2.3559322 ,  2.42372881,  2.49152542,  2.55932203,  2.62711864,
        2.69491525,  2.76271186,  2.83050847,  2.89830508,  2.96610169,
        3.03389831,  3.10169492,  3.16949153,  3.23728814,  3.30508475,
        3.37288136,  3.44067797,  3.50847458,  3.57627119,  3.6440678 ,
        3.71186441,  3.77966102,  3.84745763,  3.91525424,  3.98305085,
        4.05084746,  4.11864407,  4.18644068,  4.25423729,  4.3220339 ,
        4.38983051,  4.45762712,  4.52542373,  4.59322034,  4.66101695,
        4.72881356,  4.79661017,  4.86440678,  4.93220339,  5., 30       ])

        y = 1.0/np.array(
        [ 0.5       ,  0.51680552,  0.53332376,  0.54955473,  0.56549842,
        0.58115484,  0.59652399,  0.61160586,  0.62640046,  0.64090779,
        0.65512784,  0.66906061,  0.68270612,  0.69606435,  0.70913531,
        0.72191899,  0.7344154 ,  0.74662453,  0.75854639,  0.77018098,
        0.7815283 ,  0.79258834,  0.8033611 ,  0.8138466 ,  0.82404481,
        0.83395576,  0.84357943,  0.85291583,  0.86196495,  0.8707268 ,
        0.87920138,  0.88738868,  0.89528871,  0.90290147,  0.91022695,
        0.91726515,  0.92401609,  0.93047975,  0.93665613,  0.94254525,
        0.94814708,  0.95346165,  0.95848894,  0.96322896,  0.9676817 ,
        0.97184717,  0.97572537,  0.97931629,  0.98261994,  0.98563631,
        0.98836541,  0.99080724,  0.99296179,  0.99482907,  0.99640908,
        0.99770181,  0.99870727,  0.99942545,  0.99985636,  1.        ])

        assert_allclose(mu_piecewise_constant(x,y),
                        3.2542191564, atol=0.03)

    def test_linear(self):
        """piecewise constant approximation of linear with n = 30, s=5, kap=2"""


        x = np.array(
        [   1.06779661,  1.13559322,  1.20338983,  1.27118644,
        1.33898305,  1.40677966,  1.47457627,  1.54237288,  1.61016949,
        1.6779661 ,  1.74576271,  1.81355932,  1.88135593,  1.94915254,
        2.01694915,  2.08474576,  2.15254237,  2.22033898,  2.28813559,
        2.3559322 ,  2.42372881,  2.49152542,  2.55932203,  2.62711864,
        2.69491525,  2.76271186,  2.83050847,  2.89830508,  2.96610169,
        3.03389831,  3.10169492,  3.16949153,  3.23728814,  3.30508475,
        3.37288136,  3.44067797,  3.50847458,  3.57627119,  3.6440678 ,
        3.71186441,  3.77966102,  3.84745763,  3.91525424,  3.98305085,
        4.05084746,  4.11864407,  4.18644068,  4.25423729,  4.3220339 ,
        4.38983051,  4.45762712,  4.52542373,  4.59322034,  4.66101695,
        4.72881356,  4.79661017,  4.86440678,  4.93220339,  5., 30       ])

        y = 1.0/np.array(
        [ 0.5       ,  0.50847458,  0.51694915,  0.52542373,  0.53389831,
        0.54237288,  0.55084746,  0.55932203,  0.56779661,  0.57627119,
        0.58474576,  0.59322034,  0.60169492,  0.61016949,  0.61864407,
        0.62711864,  0.63559322,  0.6440678 ,  0.65254237,  0.66101695,
        0.66949153,  0.6779661 ,  0.68644068,  0.69491525,  0.70338983,
        0.71186441,  0.72033898,  0.72881356,  0.73728814,  0.74576271,
        0.75423729,  0.76271186,  0.77118644,  0.77966102,  0.78813559,
        0.79661017,  0.80508475,  0.81355932,  0.8220339 ,  0.83050847,
        0.83898305,  0.84745763,  0.8559322 ,  0.86440678,  0.87288136,
        0.88135593,  0.88983051,  0.89830508,  0.90677966,  0.91525424,
        0.92372881,  0.93220339,  0.94067797,  0.94915254,  0.95762712,
        0.96610169,  0.97457627,  0.98305085,  0.99152542,  1.        ])

        assert_allclose(mu_piecewise_constant(x,y),
                        3.482736134, atol=0.03)


    def test_s_increasing(self):
        assert_raises(ValueError, mu_piecewise_constant,[1.5,1,2], [1,1,1] )

    def test_s_less_than_one(self):
        assert_raises(ValueError, mu_piecewise_constant,[0.5,1,2], [1,1,1] )

    def test_kap_less_than_zero(self):
        assert_raises(ValueError, mu_piecewise_constant,[1.5,1.6,2], [-2,1,1] )

    def test_s_n_unequal_len(self):
        assert_raises(ValueError, mu_piecewise_constant, [2,4], [1])

class test_mu_piecewise_linear(unittest.TestCase):
    """tests for mu_piecewise_linear"""

    def test_ideal(self):
        assert_allclose(mu_piecewise_linear([1, 5],
                                              [1,1]),
                        0.936497825)
    def test_ideal_multi(self):
        assert_allclose(mu_piecewise_linear([1,3,5],
                                              [1,1,1]),
                        0.936497825)

    def test_const(self):
        assert_allclose(mu_piecewise_linear([1, 1.5, 1.5, 5],
                                              [1.6, 1.6, 1, 1]),
                        1.159679143)

    def test_const_multi(self):
        assert_allclose(mu_piecewise_linear([1.0, 1.3, 1.4, 1.5, 1.5, 5.0],
                                            [1.6, 1.6, 1.6, 1.6, 1.0, 1.0]),
                        1.159679143)

    def test_const_two_smear_zones(self):
        assert_allclose(mu_piecewise_linear([1.0, 1.5, 1.5, 3.0, 3.0, 5.0],
                                            [2.0, 2.0, 3.0, 3.0, 1.0, 1.0]),
                        2.253304564)

    def test_parabolic(self):
        """piecewise constant approximation of parabolic with n = 30, s=5, kap=2"""


        x = np.array(
        [1.,    1.06779661,  1.13559322,  1.20338983,  1.27118644,
        1.33898305,  1.40677966,  1.47457627,  1.54237288,  1.61016949,
        1.6779661 ,  1.74576271,  1.81355932,  1.88135593,  1.94915254,
        2.01694915,  2.08474576,  2.15254237,  2.22033898,  2.28813559,
        2.3559322 ,  2.42372881,  2.49152542,  2.55932203,  2.62711864,
        2.69491525,  2.76271186,  2.83050847,  2.89830508,  2.96610169,
        3.03389831,  3.10169492,  3.16949153,  3.23728814,  3.30508475,
        3.37288136,  3.44067797,  3.50847458,  3.57627119,  3.6440678 ,
        3.71186441,  3.77966102,  3.84745763,  3.91525424,  3.98305085,
        4.05084746,  4.11864407,  4.18644068,  4.25423729,  4.3220339 ,
        4.38983051,  4.45762712,  4.52542373,  4.59322034,  4.66101695,
        4.72881356,  4.79661017,  4.86440678,  4.93220339,  5., 30       ])

        y = 1.0/np.array(
        [ 0.5       ,  0.51680552,  0.53332376,  0.54955473,  0.56549842,
        0.58115484,  0.59652399,  0.61160586,  0.62640046,  0.64090779,
        0.65512784,  0.66906061,  0.68270612,  0.69606435,  0.70913531,
        0.72191899,  0.7344154 ,  0.74662453,  0.75854639,  0.77018098,
        0.7815283 ,  0.79258834,  0.8033611 ,  0.8138466 ,  0.82404481,
        0.83395576,  0.84357943,  0.85291583,  0.86196495,  0.8707268 ,
        0.87920138,  0.88738868,  0.89528871,  0.90290147,  0.91022695,
        0.91726515,  0.92401609,  0.93047975,  0.93665613,  0.94254525,
        0.94814708,  0.95346165,  0.95848894,  0.96322896,  0.9676817 ,
        0.97184717,  0.97572537,  0.97931629,  0.98261994,  0.98563631,
        0.98836541,  0.99080724,  0.99296179,  0.99482907,  0.99640908,
        0.99770181,  0.99870727,  0.99942545,  0.99985636,  1., 1.        ])

        assert_allclose(mu_piecewise_linear(x,y),
                        3.2542191564, atol=1e-4)

    def test_linear(self):
        """piecewise linear approximation of linear with n = 30, s=5, kap=2"""


        x = np.array(
        [1.,   1.06779661,  1.13559322,  1.20338983,  1.27118644,
        1.33898305,  1.40677966,  1.47457627,  1.54237288,  1.61016949,
        1.6779661 ,  1.74576271,  1.81355932,  1.88135593,  1.94915254,
        2.01694915,  2.08474576,  2.15254237,  2.22033898,  2.28813559,
        2.3559322 ,  2.42372881,  2.49152542,  2.55932203,  2.62711864,
        2.69491525,  2.76271186,  2.83050847,  2.89830508,  2.96610169,
        3.03389831,  3.10169492,  3.16949153,  3.23728814,  3.30508475,
        3.37288136,  3.44067797,  3.50847458,  3.57627119,  3.6440678 ,
        3.71186441,  3.77966102,  3.84745763,  3.91525424,  3.98305085,
        4.05084746,  4.11864407,  4.18644068,  4.25423729,  4.3220339 ,
        4.38983051,  4.45762712,  4.52542373,  4.59322034,  4.66101695,
        4.72881356,  4.79661017,  4.86440678,  4.93220339,  5., 30       ])

        y = 1.0/np.array(
        [ 0.5       ,  0.50847458,  0.51694915,  0.52542373,  0.53389831,
        0.54237288,  0.55084746,  0.55932203,  0.56779661,  0.57627119,
        0.58474576,  0.59322034,  0.60169492,  0.61016949,  0.61864407,
        0.62711864,  0.63559322,  0.6440678 ,  0.65254237,  0.66101695,
        0.66949153,  0.6779661 ,  0.68644068,  0.69491525,  0.70338983,
        0.71186441,  0.72033898,  0.72881356,  0.73728814,  0.74576271,
        0.75423729,  0.76271186,  0.77118644,  0.77966102,  0.78813559,
        0.79661017,  0.80508475,  0.81355932,  0.8220339 ,  0.83050847,
        0.83898305,  0.84745763,  0.8559322 ,  0.86440678,  0.87288136,
        0.88135593,  0.88983051,  0.89830508,  0.90677966,  0.91525424,
        0.92372881,  0.93220339,  0.94067797,  0.94915254,  0.95762712,
        0.96610169,  0.97457627,  0.98305085,  0.99152542,  1., 1.        ])

        assert_allclose(mu_piecewise_linear(x,y),
                        3.482736134, atol=1e-8)


    def test_s_increasing(self):
        assert_raises(ValueError, mu_piecewise_linear,[1.5,1,2], [1,1,1] )

    def test_s_less_than_one(self):
        assert_raises(ValueError, mu_piecewise_linear,[0.5,1,2], [1,1,1] )

    def test_kap_less_than_zero(self):
        assert_raises(ValueError, mu_piecewise_linear,[1.5,1.6,2], [-2,1,1] )

    def test_s_n_unequal_len(self):
        assert_raises(ValueError, mu_piecewise_linear, [2,4], [1])

    def test_first_s_not_one(self):
        assert_raises(ValueError, mu_piecewise_linear,[1.1,1.6,2], [1,1,1])

class test_k_parabolic(unittest.TestCase):
    """tests for k_parabolic"""

    def test_ideal(self):
        assert_allclose(k_parabolic(20,1,2,[4,6,7]),
                        [1,1,1])
    def test_ideal2(self):
        assert_allclose(k_parabolic(20,2,1,[4,6,7]),
                        [1,1,1])
    def test_within_smear_zone(self):
        assert_allclose(k_parabolic(30, 5, 2, [1, 1.13559322]),
                        [0.5, 0.53332376])
    def test_outside_smear_zone(self):
        assert_allclose(k_parabolic(30, 5, 2, [5, 8]),
                        [1, 1])
    def test_n_less_than_one(self):
        assert_raises(ValueError, k_parabolic, 0.5, 2, 5, 1)

    def test_s_less_than_one(self):
        assert_raises(ValueError, k_parabolic, 50, 0.5, 5, 1)

    def test_kap_less_than_zero(self):
        assert_raises(ValueError, k_parabolic, 50, 0.5, -5,1)

    def test_s_greater_than_n(self):
        assert_raises(ValueError, k_parabolic, 50, 100, 5, 1)

    def test_si_greater_than_n(self):
        assert_raises(ValueError, k_parabolic, 50, 5, 5, 100)

    def test_si_less_than_1(self):
        assert_raises(ValueError, k_parabolic, 50, 5, 5, 0.5)

class test_k_linear(unittest.TestCase):
    """tests for k_linear"""

    def test_ideal(self):
        assert_allclose(k_linear(20,1,2,[4,6,7]),
                        [1,1,1])
    def test_ideal2(self):
        assert_allclose(k_linear(20,2,1,[4,6,7]),
                        [1,1,1])
    def test_within_smear_zone(self):
        assert_allclose(k_linear(30, 5, 2, [1, 1.13559322]),
                        [0.5, 0.51694915])
    def test_outside_smear_zone(self):
        assert_allclose(k_linear(30, 5, 2, [5, 8]),
                        [1, 1])
    def test_n_less_than_one(self):
        assert_raises(ValueError, k_linear, 0.5, 2, 5, 1)

    def test_s_less_than_one(self):
        assert_raises(ValueError, k_linear, 50, 0.5, 5, 1)

    def test_kap_less_than_zero(self):
        assert_raises(ValueError, k_linear, 50, 0.5, -5,1)

    def test_s_greater_than_n(self):
        assert_raises(ValueError, k_linear, 50, 100, 5, 1)

    def test_si_greater_than_n(self):
        assert_raises(ValueError, k_linear, 50, 5, 5, 100)

    def test_si_less_than_1(self):
        assert_raises(ValueError, k_linear, 50, 5, 5, 0.5)


#scratch()
def scratch():
    """scratch

    The smear zone parameter :math:`\\mu` is given by:

    .. math:: \\mu_X =
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


    .. math:: \\mu_X =
                \\left\\{\\begin{array}{lr}
                    \\mu_L\\left({n,s,\\kappa}\\right) & n\\geq s \\\\
                    \\frac{\\kappa}{\\kappa_X}\\mu_L\\
                        left({n, s_X,\\kappa_x}\\right) & n<s \\\\
                    \\frac{\\kappa}{\\kappa_X}\\mu_I\\
                        left({n}\\right) & 2n-s\\leq 1
                 \\end{array}\\right.





    .. math:: \\frac{k_h^'\\left({r}\\right)}{k_h}=
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




    where :math:`A`, :math:`B`, :math:`C` and :math:`E` are:

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


    .. math:: \\psi_{i} = \\sum\\limits_{j=1}^{i-1}
                \\left[{
                        \\ln
                            \\left({
                                \\frac{s_j}{s_{j-1}}
                            }\\right)
                        -\\frac{s_j^2-s_{j-1}^2}{2n^2}
                    }\\right]

    and:

    .. math:: n = \\frac{r_e}{r_w}

    .. math:: s_i = \\frac{r_i}{r_w}

    .. math:: \\kappa_i = \\frac{k_h}{k_{hi}}


    :math:`r_w` is the drain radius, :math:`r_e` is the drain influence radius,
    :math:`r_s` is the smear zone radius, :math:`k_h` is the undisturbed
    horizontal permeability, :math:`k_s` is the smear zone horizontal
    permeability

   """
    pass
if __name__ == '__main__':



    scratch()
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
    mu_overlapping_linear(np.array([5,10]),
                                    np.array([7, 12]),
                                    np.array([1.6, 1.5,]))
    mu_piecewise_linear([1, 5],
                        [1, 1])



    n=30
    s=5
    kap=2
    x = np.linspace(1, s, 60)
    y = k_linear(n, s, kap, x)

#    print(repr(x))
#    print(repr(y))
#    plt.plot(x,y, 'o')
#
#    plt.show()

    mu_piecewise_constant([1.5,5],
                                              [1.6,1])

    scratch()
    print(mu_parabolic(30,5,2))
    print(k_parabolic(30, 5, 2, [1, 1.13559322]))
    k_parabolic(20,1,2,[4,6,7])
#    mu_linear()
    #    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])
#    print(mu_ideal(0.5))
#    print(mu_linear(np.array([50,100]),
#                      np.array([10,20]),
#                      np.array([5,3])))