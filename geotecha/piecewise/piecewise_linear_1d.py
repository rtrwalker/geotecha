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
One dimensional piecwise linear relationships and manipulations

"""
from __future__ import print_function, division

import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import copy
import operator


def has_steps(x):
    """Check if data points have any step changes

    True if any two consecutive x values are equal.

    Parameters
    ----------
    x : array_like
        x-coordinates

    Returns
    -------
    out : boolean
        Returns True if any two consecutive x values are equal.

    """
    #TODO: maybe check for bad segments such as x=[2,2] y=[10,10]
    x = np.asarray(x)

    return np.any(np.diff(x)==0)


def is_initially_increasing(x):
    """Are first two values increasing?

    Finds 1st instance where x[i+1] != x[i] and checks if x[i+1] > x[i].

    Parameters
    ----------
    x : array_like
        1 dimensional data to check

    Returns
    -------
    out : ``int``
        Returns True if 2nd value is greater than the 1st value.
        Returns False if 2nd value is less than the 1st value.

    """

    # This might be slow for long lists, perhaps just loop through
    # until x[i+1]!=x[i]
    if x[1] != x[0]:
        return x[1] > x[0]
    x = np.asarray(x)
    return np.where(np.diff(x) != 0)[0][0] > 0


#used info from http://stackoverflow.com/questions/4983258/python-how-to-check-list-monotonicity
def strictly_increasing(x):
    """Checks all x[i+1] > x[i]

    Parameters
    ----------
    x : 1d array
        Series of x values

    Returns
    -------
    out : True/False
        True if each x value is greater than the preceeding value.

    See Also
    --------
    strictly_decreasing : Check for decreasing values, no equality allowed.
    non_increasing : Less stringent decreasing check, allows equal values.
    non_decreasing : Less stringent increasing check, allows equal values.

    """
    x = np.asarray(x)
    return np.all(np.diff(x)>0)


def strictly_decreasing(x):
    """Checks all x[i+1] < x[i]

    Parameters
    ----------
    x : 1d array
        Series of x values

    Returns
    -------
    out : True/False
        True if each x value is less than the preceeding value.

    See Also
    --------
    strictly_increasing : Check for increasing values, no equality allowed.
    non_increasing : Less stringent decreasing check, allows equal values.
    non_decreasing : Less stringent increasing check, allows equal values.

    """
    x = np.asarray(x)
    return np.all(np.diff(x)<0)


def non_increasing(x):
    """Checks all x[i+1] <= x[i]

    Parameters
    ----------
    x : 1d array
        Series of x values

    Returns
    -------
    out : True/False
        True if each x value is less than or equal to the preceeding value.

    See Also
    --------
    strictly_increasing : Check for increasing values, no equality allowed.
    strictly_decreasing : Check for decreasing values, no equality allowed.
    non_decreasing : Less stringent increasing check, allows equal values.

    """
    x = np.asarray(x)
    return np.all(np.diff(x)<=0)


def non_decreasing(x):
    """Checks all x[i+1] >= x[i]

    Parameters
    ----------
    x : 1d array
        Series of x values

    Returns
    -------
    out : True/False
        True if each x value is greater than or equal to the preceeding value.

    See Also
    --------
    strictly_increasing : Check for increasing values, no equality allowed
    strictly_decreasing : Check for decreasing values, no equality allowed
    non_increasing : Less stringent decreasing check, allows equal values

    """
    x = np.asarray(x)
    return np.all(np.diff(x)>=0)


def non_increasing_and_non_decreasing_parts(x, include_end_point=False):
    """Indexes of each non-increasing and non-decreasing section of a list

    Parameters
    ----------
    x : list or 1d array
        List of values.
    include_end_point : True/False, optional
        If True then the index of the last point in a non-increasing or
        non-decreasing run is included.  Default include_end_point=False, i.e.
        only the start index of each line segment is given.

    Returns
    -------

    A : list of list
        Each element of A is a list of the start indices of each line
        segment that is part of a particular non-increasing or non-decreasing
        run.

    Notes
    -----
    This funciton only returns start indices for each line segment.
    Lets say x is [0, 4 , 5.5] then A will be [[0,1]].  If you do x[A[0]] then
    you will get [0, 4] i.e. no end point. To get the whole increasing or
    decreasing portion including the end point you need to do something like
    x[A[0].append[A[0][-1]+1]].

    """

    #TODO: maybe return slice object rather than a list of all the indexes as they will be contiguous anyway
    x = np.asarray(x)
    sign_changes = np.sign(np.diff(x))
    A = [[0]]

    current_sign = sign_changes[np.where(sign_changes != 0)[0][0]]

    for i, sgn in enumerate(sign_changes.tolist()):
        if i == 0:
            continue
        if sgn != 0 and sgn != current_sign:
            if include_end_point:
                A[-1].append(i)
            A.append([])
            current_sign = sgn
        A[-1].append(i)
    if include_end_point: #append final end point
        A[-1].append(A[-1][-1] + 1)
    return A


def force_strictly_increasing(x, y = None, keep_end_points = True, eps = 1e-15):
    """Force a non-decreasing or non-increasing list to be strictly increasing

    Adds or subtracts tiny amounts (multiples of `eps`) from the x values in
    step changes to ensure no two consecutive x values are equal (i.e. make x
    strictly increasing).  The adjustments are small enough that for all
    intents and purposes the data behaves as before; it can now however be
    easily used in straightforward interpolation functions that require
    strictly increasing x data.

    Any data where x is non-increasing will be reversed.

    Parameters
    ----------
    x : array_like
        List of x coordinates.
    y : array_like, optional
        List of y coordinates, (Default y=None, if known that x is
        non-decreasing then y will not be affected).
    keep_end_points : ``boolean``, optional
        Determines which x value of the step change is adjusted.
        Consider x=[1,1] and y=[20,40]. If keep_end_points==True then new
        data will be x=[0.9999,1], y=[20,40].  If keep_end_points==False then
        data will be x=[1, 1.0001], y=[20,40]
    eps : float, optional
        Amount to add/subtract from x (default is 1e-15).  To ensure
        consecutive step changes are handled correctly multipes of `eps`
        will be added and subtracted. e.g. if there are a total of five
        steps in the data then the first step would get 5*`eps` adjustment,
        the second step 4*`eps` adjustment and so on.

    """

    x = np.asarray(x)
    y = np.asarray(y)

    if strictly_increasing(x):
        return x, y

    if strictly_decreasing(x):
        return x[::-1], y[::-1]

    if non_increasing(x):
        x = x[::-1]
        y = y[::-1]

    if not non_decreasing(x):
        raise ValueError("x data is neither non-increasing, nor non-decreasing, therefore cannot force to strictly increasing")


    steps = np.where(np.diff(x) == 0)[0]

    if keep_end_points:
        f = -1 * eps
        d = 0
        dx = np.arange(len(steps), 0, -1) * f
    else:
        f = 1 * eps
        d = 1
        dx = np.arange(1, len(steps) + 1) * f

    x[steps + d] = x[steps + d] + dx
    return x, y


def force_non_decreasing(x, y = None):
    """Force non-increasing x, y data to non_decreasing by reversing the data

    Leaves already non-decreasing data alone.

    Parameters
    ----------
    x, y: array_like
        x and y coordinates

    Returns
    -------
    x, y : 1d ndarray, 1d ndarray
        Non-decreasing version of x, y

    """
    x = np.asarray(x)
    y = np.asarray(y)

    if non_decreasing(x):
        return x, y

    if not non_increasing(x):
        raise ValueError("x data is neither non-increasing, nor non-decreasing, therefore cannot force to non-decreasing")

    return x[::-1], y[::-1]


def start_index_of_ramps(x, y):
    """Find the start indices of the ramp line segments in x, y data.

    An example of a 'ramp' x=[0,2], y=[10,15]. i.e. not a vertical line and
    not a horizontal line.

    Assumes data is non_decreasing.

    Parameters
    ----------
    x, y : array_like
        x and y coords (must be non-decreasing).

    Returns
    -------
    out : 1d ndarray
        Start indices of all ramps.

    """

    x = np.asarray(x)
    y = np.asarray(y)


    return  np.where((np.diff(x) != 0) & (np.diff(y) != 0))[0]


def start_index_of_constants(x, y):
    """Find the start indices of the constant line-segments in x, y data.

    An example of a 'constant' x=[0,2], y=[15,15]. i.e. a horizontal line

    Assumes data is non_decreasing.

    Segments such as x=[1,1], y=[2,2] are ignored.

    Parameters
    ----------
    x, y : array_like
        x and y coords (must be non-decreasing).

    Returns
    -------
    out : 1d ndarray
        Start indices of all constant segments.

    """

    x = np.asarray(x)
    y = np.asarray(y)


    return np.delete(np.where(np.diff(y)==0)[0], np.where((np.diff(x)==0) & (np.diff(y)==0))[0])


def start_index_of_steps(x, y):
    """Find the start indices of the step line-segments in x, y data.

    An example of a 'step' x=[0, 0], y=[10, 15]. i.e. a vertical line.

    Assumes data is non_decreasing.

    Segments such as x=[1, 1], y=[2, 2] are ignored.

    Parameters
    ----------
    x, y : array_like
        x and y coords (must be non-decreasing)

    Returns
    -------
    out : 1d ndarray
        Start indices of all step segments.

    """

    x = np.asarray(x)
    y = np.asarray(y)


    return np.delete(np.where(np.diff(x)==0)[0], np.where((np.diff(x)==0) & (np.diff(y)==0))[0])


def segment_containing_xi(x, xi, subset=None, choose_max=False):
    """Find start index of line segment in which xi falls

    Find index where x[i] <= xi <= x[i+1] ignoring steps. `choose_max`
    determines what happens when more than one segment satisfies the
    condition e.g. the boundary between two segments; taking either the
    maximum index (`choose_max` True), or the minimum index (`choose_max`=
    False).

    Parameters
    ----------
    x : array_like, float
        x coordinates.
    xi : array_like, float
        Values to place in segments.
    subset : array_like, optional
        Restrict search to segments starting with indices in `subset`.
        Default subset=None i.e.  search all segments.
    choose_max : boolean, optional
        When False (default), the minumum index that satisfies the condition
        is returned. When True the maximum index that satisfies the condition
        is returned. Default choose_max=False. See Notes below

    Returns
    -------
    A : list of single element lists
        Each sub-list is the start index of the segment that contains xi.
        Returning each value in a list allows "for i in A[0]" type constructs
        which for an empty list will do nothing.

    Notes
    -----
    If x data has switch backs (i.e. not non-decreasing and not non-increasing)
    then if `choose_max` = True then the index returned will be in the last
    group of segments.  This can be useful as when determining when something
    ends. e.g. say x are strictly incresing time values and y zig zags up and
    down for a time then decays to zero.  A regular interpolation will give
    you any y value at a given x value.  But you might want to answer a
    question such as 'at what time does y finally fall below 10'.  You want
    to search only the last decaying section.  By plugging the y values in as
    `x` and using `choose_max` = True, you will get the segment where this
    happens.

    This function is somewhat similar to the `numpy.digitize` function that
    places x values into bins. `segment_containing_xi` however, does not insist
    on monotonic data and won't break down if steps are included; it is also
    possible to put both bounds in the same segement. Say for x = [0,1,2,3] and
    xi1 = 1 and xi2 = 2.  If `choose_max` is False for both points then xi1
    will be in segment 0, and xi2 in segment 1.  But if you use `choose_max` =
    True for xi1 then it be in segment 1 (the same as xi2).

    """

    x = np.asarray(x)

    xi = np.atleast_1d(xi)
#    if isinstance(xi, numbers.Number):
#        xi = np.array([xi])
#    xi = np.asarray(xi)

    if subset is None:
        subset = np.arange(len(x)-1)
    if len(subset)==0: #subset isempty
        return [np.array([],dtype = int) for v in xi]

    subset = np.asarray(subset)
    #this commented one doesn't work for descending
    #A = [subset[(v>=x[subset]) & (v<=x[subset+1]) & (x[subset]!=x[subset+1])] for v in xi]

    A = [subset[(abs(v-x[subset])<=abs(x[subset]-x[subset+1])) & (abs(v-x[subset+1])<=abs(x[subset]-x[subset+1])) & (x[subset]!=x[subset+1])] for v in xi]

    if choose_max:
        f = -1
    else:
        f = 0

    for i, v in enumerate(A):
        if v.size>0:
            A[i] = [v[f]]
        else:
            A[i] = []

    return A


def segments_less_than_xi(x, xi, subset=None, or_equal_to=False):
    """Find start index of line segments that end before xi

    Finds all segments where end point of segment is less than xi.

    Assumes non-decreasing `x` data.

    Parameters
    ----------
    x : array_like, float
        x coordinates.
    xi : array_like, float
        Values to check if segments start after.
    subset : array_like, optional
        Restrict search to segments starting with indices in `subset`.
        Default subset=None, which searches all segments.
    or_equal_to : ``boolean``, optional
        If False (default) then conditon is formulated with '<'.  If True
        then condition is formulated with '<='. Generally only used to
        include/exclude step loads.

    Returns
    -------
    out : list of 1d numpy.ndarray
        List contains len(xi) 1d numpy.ndarray corresponding to xi.

    """
    x = np.asarray(x)

    xi = np.atleast_1d(xi)





    if subset is None:
        subset = np.arange(len(x)-1)


    subset = np.asarray(subset)

    if or_equal_to:
        return [subset[x[subset+1] <= v] for v in xi]
    else:
        return [subset[x[subset+1] < v] for v in xi]


def ramps_constants_steps(x, y):
    """Find the ramp, constant, and step line segments in x, y data

    Returns
    -------
    ramps : 1d ndarray
        `start_index_of_ramps`
    constants : 1d ndarray
        `start_index_of_constants`
    steps : 1d ndarray
        `start_index_of_constants`

    See Also
    --------
    start_index_of_ramps : Find ramps.
    start_index_of_steps : Find steps.
    start_index_of_constants : Find constants.

    """

    x = np.asarray(x)
    y = np.asarray(y)

    ramps = start_index_of_ramps(x,y)
    constants = start_index_of_constants(x,y)
    steps = start_index_of_steps(x,y)

    return (ramps, constants, steps)


def segment_containing_also_segments_less_than_xi(x, y, xi,
                                                  steps_or_equal_to=True,
                                                  ramp_const_or_equal_to=False,
                                                  choose_max=False):
    """Determine ramp, constant and step segments containing xi as well as
    segments less than xi.

    Function does minimal calculations itself, essentially calling other
    functions and returning a tuple.  Useful for piecewise linear loading
    functions when you segements after a certain time are irrelevant.

    Parameters
    ----------
    x, y : array_like
        x and y coords (must be non-decreasing).
    xi : array_like, float
        Values to check check segments against.
    steps_or_equal_to : ``boolean``, optional
        If True (default) then any step segment that xi falls on/in will be
        included in the steps 'less than' list.
    ramp_const_or_equal_to : ``boolean``, optional
        If False (default) then any ramp or constant segment that xi falls
        on the start will not be included in the ramps and constants
        'less than' list.
    choose_max : ``boolean``, optional
        If False (default), then the minimum segment of multiple ramp segments
        that contain xi falls will be included in the 'contains' lists.
        If True then the maximum segment will be included.

    Returns
    -------
    ramps_less_than_xi : ndarray
        `segments_less_than_xi` for ramps.
    constants_less_than_xi : ndarray
        `segments_less_than_xi` for constants.
    steps_less_than_xi : ndarray
        `segments_less_than_xi` for steps.
    ramps_containing_xi : ndarray
        Start index of ramp segment containing xi.
    constants_containing_xi : ndarray
        Start index of constant segment containing xi.


    See Also
    --------
    ramps_constants_steps : find ramp, constant, and step segments
    segments_less_than_xi : find segments after xi
    segment_containing_xi : find segments that contain xi

    """

    x = np.asarray(x)
    y = np.asarray(y)

    xi = np.atleast_1d(xi)


    ramps, constants, steps = ramps_constants_steps(x,y)

    ramps_less_than_xi = segments_less_than_xi(x, xi,
                                           subset=ramps,
                                           or_equal_to=ramp_const_or_equal_to)
    constants_less_than_xi = segments_less_than_xi(x, xi,
                                           subset=constants,
                                           or_equal_to=ramp_const_or_equal_to)
    steps_less_than_xi = segments_less_than_xi(x, xi,
                                           subset=steps,
                                           or_equal_to=steps_or_equal_to)

    # the temptation to call segment_containing_xi here with subset=ramps and then subset=constants might lead to xi being in both a ramp or a constant
    contains = segment_containing_xi(x, xi, choose_max=choose_max)
    ramps_containing_xi = [np.array([],dtype=int)] * len(xi)
    constants_containing_xi = [np.array([],dtype=int)] * len(xi)

    for i, arr in enumerate(contains): #decide when xi falls on edge of segment wether it is in a ramp or a constant segment
        for v in arr: #this will skip empty arrays
            if v in ramps:
                ramps_containing_xi[i] = np.array([v], dtype = int)
            else:
                constants_containing_xi[i] = np.array([v], dtype = int)

    return (ramps_less_than_xi, constants_less_than_xi, steps_less_than_xi,
            ramps_containing_xi, constants_containing_xi)


def segment_containing_xi_also_containing_xj(x, xi, xj, subset=None):
    """Find start index of segments that xi and xj fall in trying to have
    them in the same section.

    Find highest i where x[i] <= xi <= x[i+1] ignoring steps.
    Find lowest j where  x[j] <= xj <= x[j+1] ignoring steps.
    This will minimise the number of segments between xi and xj.
    Usually xj should be greater than xi.

    Function does no calculations itself, rather calling segment_containing_xi
    twice and returning a tuple.

    Parameters
    ----------
    x : array_like, float
        x coordinates.
    xi, xj : array_like, float
        x values to from which to determin containing segment.
    subset : array_like, optional
        Restrict search to segments starting with indices in `subset`.
        Default subset=None which searches all segments.


    Returns
    -------
    seg_xi, seg_xj : list of single element lists
        Each sub-list is the start index of the segement that contains
        xi or xj.

    See also
    --------
    segments_containing_xi : Function called for `xi` and `xj`.

    """
#commented code may provide minimal speed up
#    x = np.asarray(x)
#
#    if isinstance(xi, numbers.Number):
#        xi = np.array([xi])
#    xi = np.asarray(xi)
#
#    if isinstance(xj, numbers.Number):
#        xj = np.array([xj])
#    xj = np.asarray(xj)
#
#
#
#    if subset is None:
#        subset = np.arange(len(x)-1)
#    if len(subset)==0: #subset isempty
#        return ([np.array([],dtype = int) for v in xi], [np.array([],dtype = int) for v in xi])
#
#    subset = np.asarray(subset)


    return (segment_containing_xi(x, xi, subset, choose_max=True), segment_containing_xi(x, xj, subset, choose_max = False))


def segments_between_xi_and_xj(x, xi, xj):
    """Find line segments that exclusively contain both, only one of, and
    in between xi and xj.

    Determine if xi and xj are both in the same segment.
    When xi and xj are in different segments find the segment that contains xi
    and the segment that contains xj and any segments in between them.
    Results are only obvious when x is strictly increasing and xi<xj.

    This is useful when integrating between two points; the integration limits
    in each relevant segment are different.

    Parameters
    ----------
    x : array_like, float
        x coordinates
    xi, xj : array_like, float
        x values to from which to determine segments.

    Returns
    -------
    segment_both : list of single element lists
        Each sub-list is the start index of the segment that contains
        xi or xj.
    segment_xi_only : list of single element lists
        When xi and xj not in the same segment, segment_xi_only will be the
        segment that contains xi.
    segment_xj_only : list of single element lists
        When xi and xj not in the same segment, segment_xj_only will be the
        segment that contains xj.
    segments_between : list of single element lists
        When xi and xj not in the same segment, segments_between will be the
        segments in between xi and xj but not containing xi or xj.

    See also
    --------
    segment_containing_xi_also_containing_xj : Find segment of xi and xj.

    """

    ix1, ix2 = segment_containing_xi_also_containing_xj(x,xi,xj)

    segment_both = [np.array([],dtype=int) for v in ix1]
    segment_xi_only = segment_both[:]
    segment_xj_only = segment_both[:]
    segments_between = segment_both[:]

    for i, (i1,i2) in enumerate(zip(ix1,ix2)):
        if len(i1)==0 or len(i2)==0:
            continue
        if i1[0]==i2[0]:
            segment_both[i] = np.insert(segment_both[i],0,i1[0])
#            np.insert(segment_both[i],0,i1[0])
        else:
            segment_xi_only[i]=np.insert(segment_xi_only[i],0,i1[0])
            segment_xj_only[i]=np.insert(segment_xj_only[i],0,i2[0])
            segments_between[i]=np.r_[i1[0]+1:i2[0]]
#            np.insert(segment_xi_only[i],0,i1[0])
#            np.insert(segment_xj_only[i],0,i2[0])
#            segments_between[i]=np.r_[i1[0]+1:i2[0]]

    return (segment_both, segment_xi_only, segment_xj_only, segments_between)


def convert_x1_x2_y1_y2_to_x_y(x1, x2, y1, y2):
    """Convert data defined at start and end of each segment to a line of data

    x1_x2_y1_y2 data is defined by x1[1:]==x2[:-1]

    Parameters
    ----------
    x1, y1 : array_like, float
        x and y values at start of each segment.
    x2, y2 : array_like, float
        x and y values at end of each segment (note x1[1:]==x2[:-1]).

    Returns
    -------
    x, y : 1d ndarray, float
        x and y data of continuous line that matches the x1_x2_y1_y2 data.

    See also
    --------
    convert_x_y_to_x1_x2_y1_y2 : Reverse of this function.

    Notes
    -----
    Graphs showing x1_x2_y1_y2 data and x_y data are shown below.

    ::

        x1_x2_y1_y2 type data
        y                                              y2[2]
        ^                                             /|
        |                                            / |
        |                           y2[0]           /  |
        |                          /|              /   |
        |                         / |       y1[2] /    |
        |                        /  |            |     |
        |                       /   |y1[1]  y2[1]|     |
        |                 y1[0]/    |------------|     |
        |                     | (0) |     (1)    | (2) |
        |                     |     |            |     |
        |----------------------------------------------------------->x
                            x1[0] x1[1]        x1[2]
                                  x2[0]        x2[1] x2[2]

        e.g. x1 = [0.0, 0.3, 0.7], y1 = [1, 1, 2]
             x2 = [0.3, 0.7, 1.0], y2 = [3, 1, 4]

        x_y_data
        y                                              y[5]
        ^                                             /|
        |                                            / |
        |                           y[1]            /  |
        |                          /|              /   |
        |                         / |         y[4]/    |
        |                        /  |            |     |
        |                       /   |y[2]    y[3]|     |
        |                  y[0]/    |------------|     |
        |                     |     |            |     |
        |                     |     |            |     |
        |----------------------------------------------------------->x
                            x[0]   x[1]         x[3]  x[5]
                                   x[2]         x[4]

        e.g. x = [0.0, 0.3, 0.3, 0.7, 0.7, 1.0]
             y = [1.0, 3.0, 1.0, 1.0, 2.0, 4.0]

    """
    #TODO: include an option to collapse segments where step changes are tiny
    #and where consecutive segments lie on a straight line see np.allclose with
    #atol and rtol.  Maybe collapse close steps first and then check for #
    #straight lines. probably better to do it in a separate function e.g.
    #def tidy_up_x1_x2_y1_y2() and tidy_up_x_y().


    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    n = len(x1)
    if not all(len(v) == n for v in [x2,y1,y2]):
            raise ValueError("x1, x2, y1, y2 must be of same length")

    if not np.allclose(x1[1:],x2[:-1]):
        raise ValueError("data is not x1_x2_y1_y2, i.e. x1[1:] != x2[:-1]")





    x = x1[:]
    x = np.append(x, x2[-1])
    y = y1[:]
    y = np.append(y, y2[-1])


    insert_index = np.where(y2[:-1]!=y1[1:])[0]

    x = np.insert(x, insert_index + 1, x2[insert_index])
    y = np.insert(y, insert_index + 1, y2[insert_index])

    return x, y


def convert_x_y_to_x1_x2_y1_y2(x, y):
    """Convert a line of data to data defined at start and end of each segment


    Parameters
    ----------
    x1, y1 : array_like, float
        x and y values at start of each segment.
    x2, y2 : array_like, float
        x and y values at end of each segment (note x1[1:]==x2[:-1]).

    Returns
    -------
    x1, y1 : 1d array, float
        x and y values at start of each segment.
    x2, y2 : 1d array, float
        x and y values at end of each segment (note x1[1:]==x2[:-1]).

    See also
    --------
    convert_x1_x2_y1_y2_to_x_y : Reverse of this function.

    Notes
    -----
    Graphs showing x1_x2_y1_y2 data and x_y data are shown below.

    ::



        x_y_data
        y                                              y[5]
        ^                                             /|
        |                                            / |
        |                           y[1]            /  |
        |                          /|              /   |
        |                         / |         y[4]/    |
        |                        /  |            |     |
        |                       /   |y[2]    y[3]|     |
        |                  y[0]/    |------------|     |
        |                     |     |            |     |
        |                     |     |            |     |
        |----------------------------------------------------------->x
                            x[0]   x[1]         x[3]  x[5]
                                   x[2]         x[4]

        e.g. x = [0.0, 0.3, 0.3, 0.7, 0.7, 1.0]
             y = [1.0, 3.0, 1.0, 1.0, 2.0, 4.0]

        x1_x2_y1_y2 type data
        y                                              y2[2]
        ^                                             /|
        |                                            / |
        |                           y2[0]           /  |
        |                          /|              /   |
        |                         / |       y1[2] /    |
        |                        /  |            |     |
        |                       /   |y1[1]  y2[1]|     |
        |                 y1[0]/    |------------|     |
        |                     | (0) |     (1)    | (2) |
        |                     |     |            |     |
        |----------------------------------------------------------->x
                            x1[0] x1[1]        x1[2]
                                  x2[0]        x2[1] x2[2]

        e.g. x1 = [0.0, 0.3, 0.7], y1 = [1, 1, 2]
             x2 = [0.3, 0.7, 1.0], y2 = [3, 1, 4]

    """
    #TODO: include an option to collapse segments where step changes are tiny
    #and where consecutive segments lie on a straight line see np.allclose with
    #atol and rtol.  Maybe collapse close steps first and then check for #
    #straight lines. probably better to do it in a separate function e.g.
    #def tidy_up_x1_x2_y1_y2() and tidy_up_x_y().


    x = np.asarray(x)
    y = np.asarray(y)

    if len(x)!=len(y):
            raise ValueError("x and y must be of same length")

    segs = np.where(x[:-1]!=x[1:])[0]



    x1 = x[segs]
    x2 = x[segs+1]
    y1 = y[segs]
    y2 = y[segs+1]

    return x1, x2, y1, y2


def pinterp_x1_x2_y1_y2(a, xi, **kwargs):
    """Linear interpolation using PolyLine; wrapper for interp_x1_x2_y1_y2.

    Parameters
    ----------
    a : PolyLine object
        PolyLine to interpolate from.
    xi : float, 1d array of float
        Values to interpolate at.
    **kwargs : keyword arguments
        Keyword arguments will be passed through `interp_x1_x2_y1_y2`.

    Returns
    -------
    A : 1d ndarray, float
        Interpolated y value corresponding to xi


    See also
    --------
    interp_x1_x2_y1_y2 : Wrapped function.

    """

    return interp_x1_x2_y1_y2(a.x1, a.x2, a.y1, a.y2, xi, **kwargs)


def interp_x1_x2_y1_y2(x1,x2,y1,y2, xi, choose_max=False):
    """Linear interpolation of x1_x2_y1_y2 data

    x1_x2_y1_y2 data is defined by x1[1:]==x2[:-1]
    if xi is beyond bounds of x then the first or last value of y will be
    returned as appropriate

    Parameters
    ----------
    x1, y1 : array_like, float
        x and y values at start of each segment
    x2, y2 : array_like, float
        x and y values at end of each segment (note x1[1:]==x2[:-1])
    xi : array_like, float
        x values to interpolate at

    Returns
    -------
    A : 1d ndarray, float
        Interpolated y value corresponding to xi.

    """

    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    n = len(x1)
    if not all(len(v) == n for v in [x2,y1,y2]):
            raise ValueError("x1, x2, y1, y2 must be of same length")

    if not np.allclose(x1[1:],x2[:-1]):
        raise ValueError("data is not x1_x2_y1_y2, i.e. x1[1:] != x2[:-1]")

    xi = np.atleast_1d(xi)
#    if isinstance(xi, numbers.Number):
#        xi = np.array([xi])
#    xi = np.asarray(xi)

    segs = segment_containing_xi(np.append(x1,x2[-1]), xi, subset = None, choose_max = choose_max)


    A = np.empty(len(segs))
    #mask = np.logical_not(map(len,segs))


    #A = [[] for v in segs]

    for i, v in enumerate(segs):

        if len(v)>0:
            j = v[0]
            A[i] = y1[j] + (y2[j] - y1[j]) / (x2[j] - x1[j]) * (xi[i] - x1[j])
        else:
            if abs(xi[i]-x1[0]) <abs(xi[i] - x2[-1]): #xi[i] beyond 1st value
                A[i]=y1[0]
            else: #xi[i] is beyond last value
                A[i]=y2[-1]

    return A


def pinterp_x_y(a, xi, **kwargs):
    """Linear interpolation using PolyLine; wrapper for interp_x_y.

    Parameters
    ----------
    a : PolyLine object
        PolyLine to interpolate from.
    xi : float, 1d array of float
        Values to interpolate at.
    **kwargs : keyword arguments
        Keyword arguments will be passed through `interp_x_y`.

    Returns
    -------
    A : 1d ndarray, float
        Interpolated y value corresponding to xi


    See also
    --------
    interp_x_y : Wrapped function.


    """

    return interp_x_y(a.x, a.y, xi, **kwargs)


def interp_x_y(x,y,xi, choose_max = False):
    """Linear interpolation of x_y data


    If xi is beyond bounds of x then the first or last value of y will be
    returned as appropriate.

    Parameters
    ----------
    x, y : array_like, float
        x and y values.
    xi : array_like, float
        x values to interpolate at.

    Returns
    -------
    A : 1d ndarray, float
        Interpolated y value corresponding to xi.


    """

    x = np.asarray(x)

    y = np.asarray(y)



    if len(x)!=len(y):
            raise ValueError("x and y must be of same length")



    xi = np.atleast_1d(xi)
#    if isinstance(xi, numbers.Number):
#        xi = np.array([xi])
#    xi = np.asarray(xi)

    segs = segment_containing_xi(x, xi, subset=None, choose_max=choose_max)


    A = np.empty(len(segs))
    #mask = np.logical_not(map(len,segs))


    #A = [[] for v in segs]

    for i, v in enumerate(segs):

        if len(v)>0:
            j = v[0]
            A[i] = y[j] + (y[j+1] - y[j]) / (x[j+1] - x[j]) * (xi[i] - x[j])
        else:
            if abs(xi[i] - x[0]) < abs(xi[i] - x[-1]): #xi[i] beyond 1st value
                A[i]=y[0]
            else: #xi[i] is beyond last value
                A[i]=y[-1]

    return A


def remove_superfluous_from_x_y(x, y, atol=1e-08):
    """Remove points that are on a line between other points

    Intermediate points are judged to be 'on a line' and therefore superfluous
    when the distance of the point from the line is below `atol`.

    Parameters
    ----------
    x, y : 1d array_like, float
        x and y values.
    atol : float, optional
        atol is the threshold. Default atol=1e-8.

    Returns
    -------
    xnew, ynew : 1d ndarray, float
        Cleaned up x and y values.

    Notes
    -----
    #TODO: put in equation for distance of point to a line from wolfram http://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html

    """
    n = len(x)
    if n!=len(y):
        raise ValueError("x and y must be of sam length, "
                         "{} vs {}".format(n, len(y)))

    x = np.asarray(x)
    y = np.asarray(y)

    if n<=2:
        return x, y

    ikeep = list(range(len(x)))


    j = 0
    x0 = x[j]
    y0 = y[j]
    for i in range(2,n):
        x1=x[i]
        y1=y[i]
        dx = x1-x0
        dy = y1-y0
        d = math.sqrt(dx**2+dy**2)
        if np.allclose(d,0):
            ikeep.remove(i-1)
            continue

        di = abs((dx*(y0-y[j+1:i])-dy*(x0-x[j+1:i]))/d)
        if np.any(di>atol): #points not on line anymore
            j = i - 1
            x0 = x[j]
            y0 = y[j]
        else: #points on line
            ikeep.remove(i-1)

    return x[ikeep], y[ikeep]


def pinterp_xa_ya_multipy_x1b_x2b_y1b_y2b(a, b, xai, xbi, **kwargs):
    """Interpolate a composite function made of two PolyLines; wrapper for
    interp_xa_ya_multipy_x1b_x2b_y1b_y2b.

    Evaluate f(xai)*g(xbi) where f(xa) and f(xb) are defined by PolyLine
    objects.


    Parameters
    ----------
    a, b : PolyLine objects
        PolyLine objects that make up the separable interpolation function
        a(xa) * b(xb)
        x and y values of x_y part of interpolation function.
    xai : array_like, float
        x values to interpolate at for a part
    xbi : array_like, float
        x values to interpolate at for b part
    achoose_max : ``boolean``, optional
        If False (default), if xai falls on boundary of segments choose the
        minimum segment to interpolate within.
    bchoose_max : ``boolean``, optional
        If True (default), if xbi falls on boundary of segments choose the
        maximum segment to interpolate within.
    **kwargs : keyword arguments
        Keyword arguments such as achoose_max and bchoose_max that will be
        passed through to interp_xa_ya_multipy_x1b_x2b_y1b_y2b.


    See also
    --------
    interp_xa_ya_multipy_x1b_x2b_y1b_y2b : Wrapped Function.


    """

    return interp_xa_ya_multipy_x1b_x2b_y1b_y2b(a.x, a.y, b.x1, b.x2, b.y1, b.y2, xai, xbi,**kwargs)


def interp_xa_ya_multipy_x1b_x2b_y1b_y2b(xa, ya,
                                         x1b, x2b, y1b, y2b,
                                         xai, xbi,
                                         achoose_max=False,
                                         bchoose_max=True):
    """Interpolate a composite function made of x_y and x1b_x2b_y1b_y2b
    piecewise-linear representations.

    Evaluate f(xai)*g(xbi) where f(xa) is defined
    with x_y data and g(xb) is defined by x1_x2_y1_y2 data.

    Does little calculation, mostly calls other functions.
    Calculates array A[len(xbi),len(xai)]


    Parameters
    ----------
    xa, ya : 1d array_like, float
        x and y values of x_y part of interpolation function.
    x1b, y1b : array_like, float
        x and y values at start of each segment for x1_x2_y1_y2 part of
        interpolation function
    x2b, y2b : array_like, float
        x and y values at end of each segment for x1_x2_y1_y2 part of
        interpolation function  (note x1[1:]==x2[:-1])
    xai : array_like, float
        x values to interpolate at for x_y part
    xbi : array_like, float
        x values to interpolate at for x1_x2_y1_y2 part
    achoose_max : ``boolean``, optional
        If False (default), if xai falls on boundary of segments choose the
        minimum segment to interpolate within.
    bchoose_max : ``boolean``, optional
        If True (default), if xbi falls on boundary of segments choose the
        maximum segment to interpolate within.

    See also
    --------
    interp_x_y : Interpolate the x_y part.
    interp_x1_x2_y1_y2 : Interpolate the x1_x2_y1_y2 part.

    """

    yai = interp_x_y(xa, ya, xai, choose_max=achoose_max)
    ybi = interp_x1_x2_y1_y2(x1b, x2b, y1b, y2b, xbi, choose_max=bchoose_max)

    return ybi[:, np.newaxis] * yai[np.newaxis,:]

def pavg_x_y_between_xi_xj(a, xi, xj, **kwargs):
    """Average between xi and xj for PolyLine data; wrapper for
    avg_x_y_between_xi_xj.


    Parameters
    ----------
    a : PolyLine object
        PolyLine containing data for averaging.
    xi, xj : array_like, float
        x values to interpolate between.

    Returns
    -------
    A : 1d array of float
        Interpolated values. len(A)=len(xi)


    See also
    --------
    avg_x_y_between_xi_xj : Wrapped function.


    """

    return avg_x_y_between_xi_xj(a.x, a.y, xi, xj, **kwargs)


def avg_x_y_between_xi_xj(x, y, xi, xj):
    """Average between xi and xj piecewise-linear x_y data


    Parameters
    ----------
    x, y : 1d array_like, float
        x and y values of x_y part of interpolation function.

    xi, xj : array_like, float
        x values to interpolate between.

    Returns
    -------
    A : 1d array of float
        Interpolated values. len(A)=len(xi)


    See also
    --------
    integrate_x_y_between_xi_xj : Integration is intemediate step in
        average calculation.

    """

    xi = np.atleast_1d(xi)
    xj = np.atleast_1d(xj)

    return integrate_x_y_between_xi_xj(x, y, xi, xj) / (xj - xi)


def pintegrate_x_y_between_xi_xj(a, xi, xj, **kwargs):
    """Integrate PolyLine data between xi and xj; wrapper for
    integrate_x_y_between_xi_xj.

    Parameters
    ----------
    a : PolyLine object
        PolyLine containing data for integrating.
    xi, xj : array_like, float
        x values to interpolate between.

    Returns
    -------
    A : 1d array of float
        Interpolated values. len(A) == len(xi).


    See also
    --------
    integrate_x_y_between_xi_xj : Wrapped function.

    """

    return integrate_x_y_between_xi_xj(a.x, a.y, xi, xj, **kwargs)


def integrate_x_y_between_xi_xj(x, y, xi, xj):
    """Integrate piecewise-linear x_y data between xi and xj

    Parameters
    ----------
    x, y : 1d array_like, float
        x and y values for piecewise linear integration.
    xi, xj : array_like, float
        x values to interpolate between.

    Returns
    -------
    A : 1d array of float
        Interpolated values. len(A) == len(xi)


    See also
    --------
    interp_x_y : Interpolate the x_y part.
    segments_between_xi_and_xj : Line segments between xi and xj.


    """

    x = np.asarray(x)
    y = np.asarray(y)
    xi = np.atleast_1d(xi)
    xj = np.atleast_1d(xj)

    (segment_both, segment_xi_only, segment_xj_only, segments_between) = segments_between_xi_and_xj(x, xi, xj)
    yi = interp_x_y(x, y, xi, choose_max = True)
    yj = interp_x_y(x, y, xj, choose_max = False)


    A = np.zeros(len(xi))
    for i in range(len(xi)):
        for layer in segment_both[i]:
            A[i] += (yi[i] + yj[i]) * 0.5 * (xj[i] - xi[i])
        for layer in segment_xi_only[i]:
            A[i] += (yi[i] + y[layer + 1]) * 0.5 * (x[layer + 1] - xi[i])
        for layer in segments_between[i]:
            A[i] += (y[layer] + y[layer + 1]) * 0.5 * (x[layer + 1] - x[layer])
        for layer in segment_xj_only[i]:
            A[i] += (y[layer] + yj[i]) * 0.5 * (xj[i] - x[layer])
    return A


def pintegrate_x1_x2_y1_y2_between_xi_xj(a, xi, xj, **kwargs):
    """Integrate PolyLine data between xi and xj; wrapper for
    integrate_x1_x2_y1_y2_between_xi_xj.

     Parameters
    ----------
    a : PolyLine object
        PolyLine containing data for integrating.
    xi, xj : array_like, float
        x values to integrate between.

    Returns
    -------
    A : 1d array of float
        Results of integrations. len(A) == len(xi).

    See also
    --------
    integrate_x1_x2_y1_y2_between_xi_xj : Wrapped function.

    """

    return integrate_x1_x2_y1_y2_between_xi_xj(a.x1, a.x2, a.y1, a.y2, xi, xj, **kwargs)


def integrate_x1_x2_y1_y2_between_xi_xj(x1, x2, y1, y2, xi, xj):
    """Integrate layered x1_x2_y1_y2 data between xi and xj

    Parameters
    ----------
    x1, y1 : array_like, float
        x and y values at start of each segment.
    x2, y2 : array_like, float
        x and y values at end of each segment (note x1[1:]==x2[:-1]).
    xi, xj : array_like, float
        x values to integrate between.

    Returns
    -------
    A : 1d array of float
        Results of integrations. len(A) == len(xi).

    See also
    --------
    interp_x1_x2_y1_y2 : Interpolation of x1_x2_y1_y2 data.
    segments_between_xi_and_xj : Line segments between xi and xj.


    """

    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    xi = np.atleast_1d(xi)
    xj = np.atleast_1d(xj)

    x_for_interp = np.zeros(len(x1)+1)
    x_for_interp[:-1] = x1[:]
    x_for_interp[-1] = x2[-1]


    (segment_both, segment_xi_only, segment_xj_only, segments_between) = segments_between_xi_and_xj(x_for_interp, xi, xj)

    yi = interp_x1_x2_y1_y2(x1,x2,y1,y2,xi, choose_max = True)
    yj = interp_x1_x2_y1_y2(x1,x2,y1,y2,xj, choose_max = False)


    A = np.zeros(len(xi))
    for i in range(len(xi)):
        for layer in segment_both[i]:
            A[i] += (yi[i] + yj[i]) * 0.5 * (xj[i] - xi[i])
        for layer in segment_xi_only[i]:
            A[i] += (yi[i] + y2[layer]) * 0.5 * (x2[layer] - xi[i])
        for layer in segments_between[i]:
            A[i] += (y1[layer] + y2[layer]) * 0.5 * (x2[layer] - x1[layer])
        for layer in segment_xj_only[i]:
            A[i] += (y1[layer] + yj[i]) * 0.5 * (xj[i] - x1[layer])
    return A


def pavg_x1_x2_y1_y2_between_xi_xj(a, xi, xj, **kwargs):
    """Average of PolyLine data between xi and xj; wrapper for
    avg_x1_x2_y1_y2_between_xi_xj.

    Parameters
    ----------
    a : PolyLine object
        PolyLine containing data for integrating.
    xi, xj : array_like, float
        x values to integrate between.

    Returns
    -------
    A : 1d array of float
        Average for each xi, xj pair. len(A) == len(xi).

    See also
    --------
    avg_x1_x2_y1_y2_between_xi_xj : Wrapped function.

    """

    return avg_x1_x2_y1_y2_between_xi_xj(a.x1, a.x2, a.y1, a.y2, xi, xj, **kwargs)


def avg_x1_x2_y1_y2_between_xi_xj(x1, x2, y1, y2, xi, xj):
    """Average of x1_x2_y1_y2 data between xi and xj

    Parameters
    ----------
    x1, y1 : array_like, float
        x and y values at start of each segment
    x2, y2 : array_like, float
        x and y values at end of each segment (note x1[1:]==x2[:-1])
    xi, xj : array_like, float
        x values to integrate between

    Returns
    -------
    A : 1d array of float
        Average for each xi, xj pair. len(A) == len(xi).

    See also
    --------
    integrate_x1_x2_y1_y2_between_xi_xj : Integration is intermediate
        step for average calculation.

    """


    xi = np.atleast_1d(xi)
    xj = np.atleast_1d(xj)

    return integrate_x1_x2_y1_y2_between_xi_xj(x1, x2, y1, y2, xi, xj) / (xj - xi)


def pxa_ya_multipy_avg_x1b_x2b_y1b_y2b_between(a, b,
                                               xai, xbi, xbj,
                                               **kwargs):
    """Respectively interpolate at one point, and average between two points,
    the a composiute function of two PolyLine objects.

    Evaluate f(xai) * integrate[g(xb), (xbi, xbj)] / (xbi - xbj),
    where f(xa) and g(xb) are PolyLine objects; wrapper
    for xa_ya_multipy_avg_x1b_x2b_y1b_y2b_between.


    Parameters
    ----------
    a, b : PolyLine object
        PolyLine object s containing data.  The `b` Polyline will be averaged
        between xbi, and xbj; the `a` PolyLine will be interpolated at xai.
    xai : array_like, float
        x values to interpolate at for x_y part.
    xbi, xbj : array_like, float
        x values to average between for the x1_x2_y1_y2 part.
    achoose_max : ``boolean``, optional
        If False (default), when xai falls on boundary of segments choose the
        minimum segment to interpolate within.

    Returns
    -------
    A : 2d array of float
        Result of averageing and interpolating.
        shape(A) == (len(xbi), len(xai)).

    See also
    --------
    xa_ya_multipy_avg_x1b_x2b_y1b_y2b_between : Wrapped function.

    """

    return xa_ya_multipy_avg_x1b_x2b_y1b_y2b_between(a.x,a.y,b.x1, b.x2, b.y1, b.y2, xai, xbi, xbj, **kwargs)



def xa_ya_multipy_avg_x1b_x2b_y1b_y2b_between(xa, ya,
                                              x1b, x2b, y1b, y2b,
                                              xai, xbi, xbj,
                                              achoose_max=False):
    """Respectively interpolate at one point, and average between two points,
    the x_y and x1_x2_y1_y2 portions of a composite function of
    piecewise-linear representations.

    Evaluate f(xai) * integrate[g(xb), (xbi, xbj)] / (xbi - xbj),
    where f(xa) is defined with x_y data and g(xb) is defined with
    x1_x2_y1_y2 data.

    Parameters
    ----------
    xa, ya : 1d array_like, float
        x and y values of x_y part of function.
    x1b, y1b : array_like, float
        x and y values at start of each segment for x1_x2_y1_y2 part of
        function.
    x2b, y2b : array_like, float
        x and y values at end of each segment for x1_x2_y1_y2 part of
        function  (note x1[1:]==x2[:-1])
    xai : array_like, float
        x values to interpolate at for x_y part.
    xbi, xbj : array_like, float
        x values to average between for the x1_x2_y1_y2 part.
    achoose_max : ``boolean``, optional
        If False (default), when xai falls on boundary of segments choose the
        minimum segment to interpolate within.

    Returns
    -------
    A : 2d array of float
        Result. shape(A) == (len(xbi), len(xai)), i.e. rows represent

    See also
    --------
    interp_x_y : Interpolate the x_y part.
    avg_x1_x2_y1_y2_between_xi_xj : Average the x1_x2_y1_y2 part between
        xbi, xbj.


    """

    yai = interp_x_y(xa, ya, xai, choose_max=achoose_max)
    ybi = avg_x1_x2_y1_y2_between_xi_xj(x1b, x2b, y1b, y2b, xbi, xbj)
    return ybi[:, np.newaxis] * yai[np.newaxis,:]


def pintegrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between(a, b,
                                                                xi, xj,
                                                                **kwargs):
    """Integrate between two points a composite function made of two
    PolyLine objects; wrapper for
    integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between.


    Evaluate integrate[f(x)*g(x), (xi, xj)]
    where f(x) and g(x) are defined by PolyLine objects.

    The two PolyLines must have the same x values, i.e. the layers must
    match up, x1a==x1b, x2a==x2b.

    Parameters
    ----------
    a, b : PolyLine object
        PolyLine objects multiplied together to define the composite function.
        x and y values at start of each segment for first x1_x2_y1_y2 part of
        composite function.
    xi, xj : array_like, float
        x values to integrate between.

    Returns
    -------
    A : len(xi) 1d array of float
        Integrations for each xi, xj pair.

    See also
    --------
    integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between : Wrapped
        function.
    polyline_make_x_common : Ensure PolyLine object have same x values.


    """

    a, b = polyline_make_x_common(a, b)
    return integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between(a.x1,a.x2,a.y1,a.y2,b.x1,b.x2,b.y1,b.y2,xi,xj, **kwargs)



def integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between(x1a, x2a,
                                                               y1a, y2a,
                                                               x1b, x2b,
                                                               y1b, y2b,
                                                               xi, xj):
    """Integrate between two points a composite function made of two
    x1_x2_y1_y2 piecewise-linear data representations.

    Evaluate integrate[f(x)*g(x), (xi, xj)]
    where f(x) and g(x) are defined by x1_x2_y1_y2 data.


    The two data sets must have the same x values, i.e. the layers must
    match up, x1a==x1b, x2a==x2b.

    Parameters
    ----------
    x1a, y1a : array_like, float
        x and y values at start of each segment for first x1_x2_y1_y2 part of
        composite function.
    x2a, y2a : array_like, float
        x and y values at end of each segment for first x1_x2_y1_y2 part of
        composite function  (note x1[1:]==x2[:-1]).
    x1b, y1b : array_like, float
        x and y values at start of each segment for second x1_x2_y1_y2 part of
        composite function.
    x2b, y2b : array_like, float
        x and y values at end of each segment for second x1_x2_y1_y2 part of
        composite function  (note x1[1:]==x2[:-1]).
    xi, xj : array_like, float
        x values to integrate between.

    Returns
    -------
    A : len(xi) 1d array of float
        Results of integrations.

    """

    x1a = np.asarray(x1a)
    x2a = np.asarray(x2a)
    y1a = np.asarray(y1a)
    y2a = np.asarray(y2a)
    x1b = np.asarray(x1b)
    x2b = np.asarray(x2b)
    y1b = np.asarray(y1b)
    y2b = np.asarray(y2b)

    if (not np.allclose(x1a, x1b)) or (not np.allclose(x2a, x2b)): #they may be different sizes
        raise ValueError("x values are different; they must be the same: \nx1a = {0}\nx1b = {1}\nx2a = {2}\nx2b = {3}".format(x1a,x1b, x2a, x2b))
        #sys.exit(0)

    xi = np.atleast_1d(xi)
    xj = np.atleast_1d(xj)

    x_for_interp = np.zeros(len(x1a)+1)
    x_for_interp[:-1] = x1a[:]
    x_for_interp[-1] = x2a[-1]


    (segment_both,
     segment_xi_only,
     segment_xj_only,
     segments_between) = segments_between_xi_and_xj(x_for_interp, xi, xj)


    A = np.zeros(len(xi))
    for i in range(len(xi)):
        for seg in segment_both[i]:
            a_slope = (y2a[seg] - y1a[seg]) / (x2a[seg] - x1a[seg])
            b_slope = (y2b[seg] - y1b[seg]) / (x2b[seg] - x1b[seg])
            A[i] += (-a_slope*b_slope*xi[i]**3/3 + a_slope*b_slope*xj[i]**3/3 - (-a_slope*b_slope*x1a[seg] +
                a_slope*y1b[seg]/2 + b_slope*y1a[seg]/2)*xi[i]**2 + (-a_slope*b_slope*x1a[seg] +
                a_slope*y1b[seg]/2 + b_slope*y1a[seg]/2)*xj[i]**2 - (a_slope*b_slope*x1a[seg]**2 -
                a_slope*x1a[seg]*y1b[seg] - b_slope*x1a[seg]*y1a[seg] + y1b[seg]*y1a[seg])*xi[i] +
                (a_slope*b_slope*x1a[seg]**2 - a_slope*x1a[seg]*y1b[seg] - b_slope*x1a[seg]*y1a[seg]
                + y1b[seg]*y1a[seg])*xj[i])
        for seg in segment_xi_only[i]:
            a_slope = (y2a[seg] - y1a[seg]) / (x2a[seg] - x1a[seg])
            b_slope = (y2b[seg] - y1b[seg]) / (x2b[seg] - x1b[seg])
            A[i] += (a_slope*b_slope*x2a[seg]**3/3 - a_slope*b_slope*xi[i]**3/3 - (-a_slope*b_slope*x1a[seg] +
                a_slope*y1b[seg]/2 + b_slope*y1a[seg]/2)*xi[i]**2 - (a_slope*b_slope*x1a[seg]**2 -
                a_slope*x1a[seg]*y1b[seg] - b_slope*x1a[seg]*y1a[seg] + y1b[seg]*y1a[seg])*xi[i] +
                x2a[seg]*(a_slope*b_slope*x1a[seg]**2 - a_slope*x1a[seg]*y1b[seg] -
                b_slope*x1a[seg]*y1a[seg] + y1b[seg]*y1a[seg]) +
                x2a[seg]**2*(-a_slope*b_slope*x1a[seg] + a_slope*y1b[seg]/2 + b_slope*y1a[seg]/2))
        for seg in segments_between[i]:
            a_slope = (y2a[seg] - y1a[seg]) / (x2a[seg] - x1a[seg])
            b_slope = (y2b[seg] - y1b[seg]) / (x2b[seg] - x1b[seg])
            A[i] += (-a_slope*b_slope*x1a[seg]**3/3 + a_slope*b_slope*x2a[seg]**3/3 -
                x1a[seg]*(a_slope*b_slope*x1a[seg]**2 - a_slope*x1a[seg]*y1b[seg] -
                b_slope*x1a[seg]*y1a[seg] + y1b[seg]*y1a[seg]) -
                x1a[seg]**2*(-a_slope*b_slope*x1a[seg] + a_slope*y1b[seg]/2 + b_slope*y1a[seg]/2) +
                x2a[seg]*(a_slope*b_slope*x1a[seg]**2 - a_slope*x1a[seg]*y1b[seg] -
                b_slope*x1a[seg]*y1a[seg] + y1b[seg]*y1a[seg]) +
                x2a[seg]**2*(-a_slope*b_slope*x1a[seg] + a_slope*y1b[seg]/2 + b_slope*y1a[seg]/2))
        for seg in segment_xj_only[i]:
            a_slope = (y2a[seg] - y1a[seg]) / (x2a[seg] - x1a[seg])
            b_slope = (y2b[seg] - y1b[seg]) / (x2b[seg] - x1b[seg])
            A[i] += (-a_slope*b_slope*x1a[seg]**3/3 + a_slope*b_slope*xj[i]**3/3 + (-a_slope*b_slope*x1a[seg] +
                a_slope*y1b[seg]/2 + b_slope*y1a[seg]/2)*xj[i]**2 + (a_slope*b_slope*x1a[seg]**2 -
                a_slope*x1a[seg]*y1b[seg] - b_slope*x1a[seg]*y1a[seg] + y1b[seg]*y1a[seg])*xj[i] -
                x1a[seg]*(a_slope*b_slope*x1a[seg]**2 - a_slope*x1a[seg]*y1b[seg] -
                b_slope*x1a[seg]*y1a[seg] + y1b[seg]*y1a[seg]) -
                x1a[seg]**2*(-a_slope*b_slope*x1a[seg] + a_slope*y1b[seg]/2 + b_slope*y1a[seg]/2))

    return A


def pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between_super(a,b,c, xai,xbi,xbj, **kwargs):
    """Superpose results of respectively interpolate at one point, and
    integrate between two points, the first PolyLine and the multiplied second
    and third PolyLine portions of a composite function of piecewise-linear
    representations.

    For each f(xa), g(xb) pair sum f(xai) * integrate[g(xb) * h(xb), (xbi, xbj)],
    where f(xa), g(xb), and h(xb) are defined with PolyLine objects.

    The function is a bit specialised in that `a` and `b` can be lists of
    PolyLine objects.  The contribution of each composite function
    (a[0], b[0], c), (a[1], b[1], c), ... will be summed.

    The `b` and `c` PolyLine objects do NOT need to have the same x values,
    they will be forced to do so using `polyline_make_x_common`.

    Parameters
    ----------
    a, b, c : PolyLine object, or list of PolyLine objects
        PolyLine obects making up the composite function. `a` and `b` can be
        lists of equal length.
    xai : array_like, float
        x values to interpolate the `a` part of the composite function.
    xbi, xbj : array_like, float
        x values to integrate the b * c  part of the composite function.
    achoose_max : ``boolean``, optional
        If False (default), when xai falls on boundary of segments choose the
        minimum segment to interpolate within.


    Returns
    -------
    A : (len(xbi), len(xai)) 2d array
        Results for each xbi, xbj pair and xai value.


    See also
    --------
    xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between :
        Wrapped function.
    pxa_ya_cos_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between_super :
        Same function with additional cosine term.
    pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between :
        Similar function without superposition.

    """

#    b, c = polyline_make_x_common(b, c)
#    return xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(a.x,a.y,b.x1,b.x2,b.y1,b.y2, c.x1, c.x2, c.y1, c.y2, xai,xbi,xbj, **kwargs)

    if not isinstance(a,list):
        a = [a]
    if not isinstance(b, list):
        b = [b]

    if len(a)!=len(b):
        raise ValueError("a and b must be lengths of equal length. len(a) = {0}, len(b) = {1}".format(len(a), len(b)))

    out = np.zeros((len(xbi), len(xai)))
    for aa, bb in zip(a, b):
        bb, cc = polyline_make_x_common(bb, c)
        out += xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(aa.x,aa.y,bb.x1,bb.x2,bb.y1,bb.y2, cc.x1, cc.x2, cc.y1, cc.y2, xai,xbi,xbj, **kwargs)
    return out

def pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(a, b, c, xai,xbi,xbj, **kwargs):
    """Respectively interpolate at one point, and integrate between two
    points, the first PolyLine and the multiplied second and third PolyLine
    portions of a composite function of piecewise-linear representations.

    Evaluate f(xai) * integrate[g(xb) * h(xb), (xbi, xbj)],
    where f(xa), g(xb), and h(xb) are defined with PolyLine objects.

    The `b` and `c` PolyLine objects do NOT need to have the same x values,
    they will be forced to do so using `polyline_make_x_common`.

    Parameters
    ----------
    a, b, c : PolyLine object
        PolyLine obects making up the composite function.
    xai : array_like, float
        x values to interpolate the `a` part of the composite function.
    xbi, xbj : array_like, float
        x values to integrate the b * c  part of the composite function.
        between.
    achoose_max : ``boolean``, optional
        If False (default), when xai falls on boundary of segments choose the
        minimum segment to interpolate within.


    Returns
    -------
    A : (len(xbi), len(xai)) 2d array
        Results for each xbi, xbj pair and xai value.


    See also
    --------
    xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between : Wrapped
        Function.
    pxa_ya_cos_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between : Similar
        function with additional cosine term.
    pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between_super : Same
        function with superposition.

    """

    b, c = polyline_make_x_common(b, c)
    return xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(a.x,a.y,b.x1,b.x2,b.y1,b.y2, c.x1, c.x2, c.y1, c.y2, xai,xbi,xbj, **kwargs)



def xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(xa,ya,x1b,x2b,y1b,y2b, x1c, x2c, y1c, y2c, xai,xbi,xbj, achoose_max=False):
    """Respectively interpolate at one point, and integrate between two
    points, the x_y and two multipled x1_x2_y1_y2 portions of a composite
    function of piecewise-linear representations.

    Evaluate f(xai) * integrate[g(xb) * h(xb), (xbi, xbj)],
    where f(xa) is defined with x_y data and g(xb) and h(xb) are defined with
    x1_x2_y1_y2 data.

    Parameters
    ----------
    xa, ya : 1d array_like, float
        x and y values of x_y part of the composite function.
    x1b, y1b : array_like, float
        x and y values at start of each segment for the first x1_x2_y1_y2
        part of the composite function.
    x2b, y2b : array_like, float
        x and y values at end of each segment for the first x1_x2_y1_y2 part of
        the composite function (note x1[1:]==x2[:-1]).
    x1c, y1c : array_like, float
        x and y values at start of each segment for the secondx1_x2_y1_y2
        part of the composite function.
    x2c, y2c : array_like, float
        x and y values at end of each segment for the second x1_x2_y1_y2
        part of the composite function  (note x1[1:]==x2[:-1]).
    xai : array_like, float
        x values to interpolate the x_y part at.
    xbi, xbj : array_like, float
        x values to integrate the x1b_x2b_y1b_y2b * x1c_x2c_y1c_y2c part
        between.
    achoose_max : ``boolean``, optional
        If False (default), when xai falls on boundary of segments choose the
        minimum segment to interpolate within.


    Returns
    -------
    A : (len(xbi), len(xai)) 2d array
        Results for each xbi, xbj pair and xai value.

    See Also
    --------
    pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between :
        Same function with PolyLine inputs.
    pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between_super :
        PolyLine inputs and superposition.
    xa_ya_cos_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between :
        Similar function with additional cosine term.

    """

    yai = interp_x_y(xa, ya, xai, choose_max=achoose_max)
    ybi = integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between(x1b, x2b, y1b, y2b, x1c, x2c, y1c, y2c, xbi, xbj)
    return ybi[:, np.newaxis] * yai[np.newaxis,:]


#def pxa_ya_cos_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between_super(a,omega_phase, b,c, xai,xbi,xbj, **kwargs):
#    """wrapper for xa_ya_cos_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between to allow PolyLine input.
#
#    Parameters
#    ----------
#    omega_phase: 2 element tuple
#        (omega, phase) for use in cos(omega * t + phase)
#
#    Notes
#    -----
#    `a` and `b` `omega_phase` can be lists that will be superposed.  This is not available in
#    the original function
#
#    See also
#    --------
#    xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between
#    pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between : similar polyline wrapper but no superposition
#
#    """
#
##    b, c = polyline_make_x_common(b, c)
##    return xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(a.x,a.y,b.x1,b.x2,b.y1,b.y2, c.x1, c.x2, c.y1, c.y2, xai,xbi,xbj, **kwargs)
#
#    if not isinstance(a,list):
#        a = [a]
#    if not isinstance(b, list):
#        b = [b]
#    if not isinstance(b, list):
#        omega_phase = [omega_phase]
#    if len(a)!=len(b):
#        raise ValueError("a and b must be lengths of equal length. len(a) = {0}, len(b) = {1}".format(len(a), len(b)))
#    if len(a)!=len(omega_phase):
#        raise ValueError("a and omega_phase must be lengths of equal length. len(a) = {0}, len(omega_phase) = {1}".format(len(a), len(omega_phase)))
#    out = np.zeros((len(xbi), len(xai)))
#    for aa, bb, (omega, phase) in zip(a, b, omega_phase):
#        bb, cc = polyline_make_x_common(bb, c)
#        out += xa_ya_cos_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(aa.x,aa.y,omega, phase, bb.x1,bb.x2,bb.y1,bb.y2, cc.x1, cc.x2, cc.y1, cc.y2, xai,xbi,xbj, **kwargs)
#    return out

def pxa_ya_cos_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between_super(a,b,c, xai, xbi, xbj, omega_phase=None, **kwargs):
    """Superpose results of  respectively interpolate at one point, and
    integrate between two points, the cosine multiplied by PolyLine, and two
    multipled PolyLine portions of a composite function of piecewise-linear
    representations.

    For each f(xa), g(xb) pair sum
    f(xai) * cos(omega*xai + phase) * integrate[g(xb) * h(xb), (xbi, xbj)],
    where f(xa), g(xb), and h(xb) are PolyLine objects.

    The function is a bit specialised in that `a` and `b` and `omega_phase`
    can be lists of PolyLine objects.  The contribution of each composite function
    (a[0], omega_phase[0], b[0], c), (a[1], b[1], omega_phase[1], c), ...
    will be summed.

    The `b` and `c` PolyLine objects do NOT need to have the same x values,
    they will be forced to do so using `polyline_make_x_common`.


    Parameters
    ----------
    a, b, c : PolyLine object
        PolyLine obects making up the composite function.
    xai :  array_like, float
        x values to interpolate the `a` part of the composite function.
    xbi, xbj : array_like, float
        x values to integrate the b * c  part of the composite function.
        between.
    omega_phase : 2 element tuple, optional
        (omega, phase) for use in cos(omega * t + phase), Default
        omega_phase=None i.e. no cosine term.
    achoose_max : ``boolean``, optional
        If False (default), when xai falls on boundary of segments choose the
        minimum segment to interpolate within.


    Returns
    -------
    A : (len(xbi), len(xai)) 2d array
        Results for each xbi, xbj pair and xai value.


    See also
    --------
    xa_ya_cos_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between :
        Wrapped function
    pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between_super :
        Same function without cosine term.
    pxa_ya_cos_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between :
        Similar function without superposition.


    """

#    b, c = polyline_make_x_common(b, c)
#    return xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(a.x,a.y,b.x1,b.x2,b.y1,b.y2, c.x1, c.x2, c.y1, c.y2, xai,xbi,xbj, **kwargs)

    if not isinstance(a,list):
        a = [a]
    if not isinstance(b, list):
        b = [b]
    if omega_phase is None:
        omega_phase = [None] * len(a)
    if not isinstance(b, list):
        omega_phase = [omega_phase]

    if len(a)!=len(b):
        raise ValueError("a and b must be lengths of equal length. len(a) = {0}, len(b) = {1}".format(len(a), len(b)))
    if len(a)!=len(omega_phase):
        raise ValueError("a and omega_phase must be lengths of equal length. len(a) = {0}, len(omega_phase) = {1}".format(len(a), len(omega_phase)))
    out = np.zeros((len(xbi), len(xai)))
    for aa, bb, om_ph in zip(a, b, omega_phase):
        bb, cc = polyline_make_x_common(bb, c)
        if not om_ph is None:
            omega, phase = om_ph
            out += xa_ya_cos_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(aa.x,aa.y,omega, phase, bb.x1,bb.x2,bb.y1,bb.y2, cc.x1, cc.x2, cc.y1, cc.y2, xai,xbi,xbj, **kwargs)
        else:
            out += xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(aa.x, aa.y, bb.x1, bb.x2, bb.y1,bb.y2, cc.x1, cc.x2, cc.y1, cc.y2, xai,xbi,xbj, **kwargs)
    return out

def pxa_ya_cos_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(a, omega, phase, b, c, xai,xbi,xbj, **kwargs):
    """Respectively interpolate at one point, and integrate between two
    points, the cosine multiplied by PolyLine, and two multipled PolyLine
    portions of a composite function of piecewise-linear representations.

    Evaluate f(xai) * cos(omega*xai + phase) * integrate[g(xb) * h(xb), (xbi, xbj)],
    where f(xa), g(xb), and h(xb) are PolyLine objects.

    The `b` and `c` PolyLine objects do NOT need to have the same x values,
    they will be forced to do so using `polyline_make_x_common`.

    Parameters
    ----------
    a, b, c : PolyLine object
        PolyLine obects making up the composite function.
    omega, phase : float
        Values for defining cos(omega * t + phase).
    xai : array_like, float
        x values to interpolate the `a` part of the composite function.
    xbi, xbj : array_like, float
        x values to integrate the b * c  part of the composite function.
        between.
    achoose_max : ``boolean``, optional
        If False (default), when xai falls on boundary of segments choose the
        minimum segment to interpolate within.


    Returns
    -------
    A : (len(xbi), len(xai)) 2d array
        Results for each xbi, xbj pair and xai value.

    See also
    --------
    xa_ya_cos_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between : Wrapped
        Function.
    pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between : Same
        function without the cosine part.
    pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between_super : Same
        function with superposition.



    """

    b, c = polyline_make_x_common(b, c)
    return xa_ya_cos_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(a.x,a.y,omega, phase, b.x1,b.x2,b.y1,b.y2, c.x1, c.x2, c.y1, c.y2, xai,xbi,xbj, **kwargs)

def xa_ya_cos_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(xa, ya, omega, phase, x1b,x2b,y1b,y2b, x1c, x2c, y1c, y2c, xai,xbi,xbj, achoose_max=False):
    """Respectively interpolate at one point, and integrate between two
    points, the cosine multiplied by x_y, and two multipled x1_x2_y1_y2
    portions of a composite function of piecewise-linear representations.

    Evaluate f(xai) * cos(omega*xai + phase) * integrate[g(xb) * h(xb), (xbi, xbj)],
    where f(xa) is defined with x_y data and g(xb) and h(xb) are defined with
    x1_x2_y1_y2 data.

    Parameters
    ----------
    xa, ya : 1d array_like, float
        x and y values of x_y part of interpolation function.
    omega, phase : float
        Values for defining cos(omega * t + phase).
    x1b, y1b : array_like, float
        x and y values at start of each segment for the first 1st x1_x2_y1_y2
        part of the composite function.
    x2b, y2b : array_like, float
        x and y values at end of each segment for the first 1st x1_x2_y1_y2
        part of the composite function  (note x1[1:]==x2[:-1])
    x1c, y1c : array_like, float
        x and y values at start of each segment for the second x1_x2_y1_y2
        part of the composite function.
    x2c, y2c : array_like, float
        x and y values at end of each segment for the second x1_x2_y1_y2
        part of the composite function  (note x1[1:]==x2[:-1]).
    xai : array_like, float
        x values to interpolate the xc_yc part at and evaluate the
        cos(omega*xai + phase) part.
    xbi, xbj : array_like, float
        x values to integrate the x1b_x2b_y1b_y2b * x1c_x2c_y1c_y2c part
        between.
    achoose_max : ``boolean``, optional
        If False (default), when xai falls on boundary of segments choose the
        minimum segment to interpolate within.


    Returns
    -------
    A : (len(xbi), len(xai)) 2d array
        Results for each xbi, xbj pair and xai value.

    See Also
    --------
    pxa_ya_cos_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between :
        Same function with PolyLine inputs.
    pxa_ya_cos_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between_super :
        PolyLine inputs and superposition.
    xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between :
        Similar function without cosine term.

    """

    xai = np.asarray(xai)
    yai = interp_x_y(xa, ya, xai, choose_max=achoose_max) * np.cos(omega * xai + phase)

    ybi = integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between(x1b, x2b, y1b, y2b, x1c, x2c, y1c, y2c, xbi, xbj)
    return ybi[:, np.newaxis] * yai[np.newaxis,:]

class PolyLine(object):
    """A Polyline is a series of x y points joined by straight lines

    Provides some extra functionality beyond just using an x array and y array
    to represent a multi-point line.

    Functinality of PolyLines:


    - PolyLines can be intialized in different ways e.g. use a single array
      of x-y points; use separate x-arrays and y-arrays; use x values at the
      start and end of each interval and y values at the start and end of each
      interval.  This can be useful if you want to work with layer/interval
      data but to plot those intervals you want x-y points.
    - Multiply a PolyLine by a scalar and only the y values will be changed.
    - Add to/from a PolyLine with a scalar and only the y values will be
      changed.
    - PolyLines can be added together to create a new PolyLine; the x values
      of each PolyLine will be maintained.  Any x values that are not common
      to both PolyLines will have their y values interpolated and then added.

    Parameters
    ----------
    *args : array like
        A PolyLine object can be initialized with 1, 2, or 4 positional
        arguments (a different number of arguments will raise an error):

         - A single n-by-2 two-dimensional array of n (x, y) points.  i.e.
           first column is x values, second column is y values.
         - Two one-dimensional arrays of equal length.  The first array is the
           x values.  The second array is the corresponding y values.
         - Four one-dimensional arrays of equal length. First array x1 is x
           values at the start of a interval.  The second array x2 is x values
           at the end of a interval (note x1[1:] must equal x2[0:-1]).  The
           third array y1 is the y values at the start of each interval.  The
           fourth array y2 is the y values at the end of each interval.


    Attributes
    ----------
    xy : 2d numpy array
        n by 2 array containing x and y values for each of the n points i.e.
        [[x0, y0], [x1, y1], ..., [xn, yn]].
    x, y : 1d numpy array
        Arrays containing all the x values and all the y values in the
        PolyLine.
    x1, x2, y1, y2 : 1d numpy array
        When you want the start and end values of each of the intervals/layers
        in the PolyLine.  x1 is the x values at the start of each interval,
        x2 is the x values at the end of each interval. y1 is the y values at
        the start of each interval, y2 is the y values at the end of each
        interval.  Note that when dealing with intervals any vertical intervals
        will be lost e.g. say our x-y values are defined by joining the dots:
        PolyLine([0,1,1,2], [4,4,5,5])  is defined by 4 points but  it will
        have only 2 layers/intervals i.e. x1 will be [0,1], x2 will be [1,0],
        y1 will be [4,5], y2 will be [4,5].  Just be careful when initialising
        PolyLines using x1_x2_y1_y2 values, any initial or final vertical
        section  cannot be defined.
    x1_x2_y1_y2 : tuple of 4 1d arrays
        The x1, x2, y1, y2 arrays returned in a tuple.
    atol, rtol : float
        Absolute and relative tolerance when comparing equality of points in a
        PolyLine using numpy.allclose
    _prefix_for_numpy_array_repr : string
        When using the repr function on Numpy arrays the default output will
        print array([...]).  Because PolyLines use numpy arrays to store data
        when using repr on PolyLines you would get PolyLine(array([...])).
        Now if in your code you have done import numpy.array as array" or some
        such import then you can just copy the repr of a PolylIne into you code
        .  However I usually use "import numpy as np"  so ideally I want
        'np.' prepended to all my numpy array reprs.
        _prefix_for_numpy_array_repr does just this with the default prefix
        = "np."  (if you wish to change the prefix for all numpy arrays not
        just the PolyLine repr then see numpy.`set_string_function` )

    """


    def __init__(self, *args):

        if not len(args) in [1,2,4]:
            #For the following error consider subclassing exception as per http://stackoverflow.com/a/1964247/2530083
            raise TypeError('{:d} args given; Line can only be initialized with '
                '1, 2, or 4 args: 1 (x-y coords), 2 (x-coords, y-coords), or '
                '4 (x1-coords, x2-coords, y1-coords, y2-coords)'.format(len(args)))
        self._xy = None
        self._x = None
        self._y = None
        self._x1 = None
        self._x2 = None
        self._y1 = None
        self._y2 = None
        self.atol = 1e-5
        self.rtol = 1e-8
        self._prefix_for_numpy_array_repr = "np."


        if len(args)==1:
            #args[0] is an 2d array of n xy data points; shape=(n,2)
            self._xy = np.asarray(args[0], dtype=float)

            if ((self._xy.ndim != 2) or
                (self._xy.shape[0] < 2) or
                (self._xy.shape[1] != 2)):
                raise TypeError('x_y data must be 2d array_like with shape '
                    '(n, 2) with n >= 2.  Yours has shape '
                    '({:d}, {:d}).'.format(*self._xy.shape))
            return

        if len(args)==2:
            #args[0] and args[1] are 1d arrays of n x and y data points
            if ((len(args[0]) != len(args[1])) or
                (len(args[0]) < 2) or
                (len(args[1]) < 2)):
                raise TypeError('x and y must be of same length with at '
                    'least two values. len(x)={:d}, '
                    'len(y)={:d}'.format(len(args[0]), len(args[1])))

            self._xy = np.empty([len(args[0]), 2], dtype=float)
            self._xy[:, 0]=args[0][:]
            self._xy[:, 1]=args[1][:]
            self._x = self._xy[:, 0]
            self._y = self._xy[:, 1]
            return

        if len(args)==4:
            #data is in segments with
            #args[0] and args[1] are 1d arrays of n x values at start and end of the n segments
            #args[2] and args[3] are 1d arrays of n y values at start and end of the n segments
            if len(set(map(len, args))) != 1:
                raise TypeError('x1, x2, y1, y2 must have same length. '
                    'You have lengths '
                    '[{:d}, {:d}, {:d}, {:d}]'.format(*tuple(map(len, args))))

            self._x1 = np.asarray(args[0], dtype=float)
            self._x2 = np.asarray(args[1], dtype=float)
            self._y1 = np.asarray(args[2], dtype=float)
            self._y2 = np.asarray(args[3], dtype=float)
            return

    @property
    def xy(self):
        """Get the xy values."""
        if self._xy is None:
            x, y = convert_x1_x2_y1_y2_to_x_y(self.x1, self.x2, self.y1, self.y2)
            self._xy = np.empty([len(x), 2], dtype=float)
            self._xy[:,0] = x
            self._xy[:,1] = y
        return self._xy

    @property
    def x1_x2_y1_y2(self):
        """Get the x1_x2_y1_y2 values"""
        if self._x1 is None:
            self._x1, self._x2, self._y1, self._y2 = convert_x_y_to_x1_x2_y1_y2(self.x, self.y)
        return self._x1, self._x2, self._y1, self._y2

    @property
    def x(self):
        """Get the x values of xy data."""
        if self._x is None:
            if self._xy is None:
                self.xy
            self._x = self._xy[:, 0]
        return self._x

    @property
    def y(self):
        """Get the y values of xy data."""
        if self._y is None:
            if self._xy is None:
                self.xy
            self._y = self._xy[:, 1]
        return self._y

    @property
    def x1(self):
        """Get the x1 values of x1_x2_y1_y2 data."""
        return self.x1_x2_y1_y2[0]
    @property
    def x2(self):
        """Get the x2 values of x1_x2_y1_y2 data."""
        return self.x1_x2_y1_y2[1]
    @property
    def y1(self):
        """Get the y1 values of x1_x2_y1_y2 data."""
        return self.x1_x2_y1_y2[2]
    @property
    def y2(self):
        """Get the y2 values of x1_x2_y1_y2 data."""
        return self.x1_x2_y1_y2[3]

    def __str__(self):
        """Return a string representation of the xy data"""
        return "PolyLine({})".format(str(self.xy))

#    def __repr__(self):
#        """A string repr of the PolyLine that will recreate the PolyLine"""
#        return "PolyLine({}{})".format(self._prefix_for_numpy_array_repr,
#                                    repr(self.xy))
    def __repr__(self):
        """A string repr of the PolyLine that will recreate the PolyLine"""
        return "PolyLine({})".format(repr(self.xy))

    def __add__(self, other):
        return self._add_substract(other, op=operator.add)
    def __radd__(self, other):
        return self._add_substract(other, op=operator.add)

    def __sub__(self, other):
        return self._add_substract(other, op=operator.sub)
    def __rsub__(self, other):
        return self._add_substract(other, op=operator.sub).__mul__(-1)

    def __mul__(self, other):
        if isinstance(other, PolyLine):
            raise TypeError('Cannot multiply two PolyLines together.  '
                'You will get a quadratic that I cannot handle.')
            sys.exit(0)
        try:
            return PolyLine(self.x, self.y * other)
#            a = copy.deepcopy(self)
#            a.xy #ensure xy has been initialized
#            a._xy[:,1] *= other
        except TypeError:
            print("unsupported operand type(s) for *: 'PolyLine' and "
                "'{}'".format(other.__class__.__name__))
            sys.exit(0)
#        return a
    def __rmul__(self,other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, PolyLine):
            raise TypeError('Cannot divide two PolyLines together.  You will get a quadratic that I cannot handle')
            sys.exit(0)
        try:
            return PolyLine(self.x, self.y / other)
#            a = copy.deepcopy(self)
#            a.xy #ensure xy has been initialized
#            a._xy[:,1] /= other
        except TypeError:
            print("Unsupported operand type(s) for /: 'PolyLine' and "
                "'{}'".format(other.__class__.__name__))
            sys.exit(0)
#        return a
    def __rtruediv__(self, other):
        if isinstance(other, PolyLine):
            raise TypeError('Cannot divide two PolyLines together.  You will get a quadratic that I cannot handle')
            sys.exit(0)
        try:
            return PolyLine(self.x, other / self.y)
#            a = copy.deepcopy(self)
#            a.xy #ensure xy has been initialized
#            a._xy[:,1] = other/a._xy[:,1]
        except TypeError:
            print("unsupported operand type(s) for /: 'PolyLine' and "
                "'{}'".format(other.__class__.__name__))
            sys.exit(0)
#        return a
        return self.__mul__(other)
    def __eq__(self, other):
        if len(self.xy)!=len(other.xy):
            return False
        else:
            return np.allclose(self.xy, other.xy, rtol=self.rtol, atol=self.atol)
    def __ne__(self, other):
        return not self.__eq__(other)

    def _add_substract(self, other, op = operator.add):
        """Addition or subtraction of PolyLine objects"""

        mp = {operator.add: operator.iadd, operator.sub: operator.isub}
        iop = mp[op]

        if isinstance(other, PolyLine):

            if ((not (non_increasing(self.x) or non_decreasing(self.x))) or
                (not (non_increasing(other.x) or non_decreasing(other.x)))):
                raise TypeError('Your PolyLines have switchbacks in them; cannot add together.')
                sys.exit(0)


            #1. reverse values if decreasing
            if not is_initially_increasing(self.x):
                xa = self.x[::-1]
                ya = self.y[::-1]
            else:
                xa = self.x[:]
                ya = self.y[:]
            if not is_initially_increasing(other.x):
                xb = other.x[::-1]
                yb = other.y[::-1]
            else:
                xb = other.x[:]
                yb = other.y[:]

            xa, ya = remove_superfluous_from_x_y(xa, ya)
            xb, yb = remove_superfluous_from_x_y(xb, yb)



            data= np.zeros([len(xa)+len(xb),4])
            data[:len(xa), 0]= xa[:]
            data[:len(xa), 1]= ya[:]
            data[:len(xa), 2]= 0 #line
            data[:len(xa), 3]= np.arange(len(xa)) #orig position
            data[len(xa):, 0]= xb[:]
            data[len(xa):, 1]= yb[:]
            data[len(xa):, 2]= 1 #line
            data[len(xa):, 3]= np.arange(len(xb)) #orig position

            data = data[np.lexsort((data[:,3],data[:,2],data[:,0]))]


            pnts = []
            xnow =np.nan
            start = 0
            stop = 0

            i=0
            while i < len(data):
                x = data[i, 0]
                #ind = np.where(np.abs(data[:,0] - xnow) < (self.atol + self.rtol * np.abs(xnow)))[0]
                ind = abs(data[:,0] - x) < (self.atol + self.rtol * abs(x))

                j = np.where((ind) & (data[:, 2] == 0))[0]
                if len(j) > 0:
                    #use point a line point
                    y1 = data[j[0], 1]
                    y1_= data[j[-1],1]
                else: #interpolate
                    y1 = interp_x_y(xa, ya, x, choose_max=False)
                    y1_= y1

                j = np.where((ind) & (data[:, 2] == 1))[0]
                if len(j) > 0:
                    #use point a line point
                    y2 = data[j[0], 1]
                    y2_= data[j[-1],1]
                else: #interpolate
                    y2 = interp_x_y(xb, yb, x, choose_max=False)
                    y2_= y2

                y = op(y1, y2)
                y_ = op(y1_, y2_)
                pnts.append([x, y])
                if abs(y - y_)> (self.atol + self.rtol * abs(y_)):
                    pnts.append([x, y_])

                i= np.nonzero(ind)[0][-1]+1

            return PolyLine(pnts)


        try:
            return PolyLine(self.x, iop(self.y, other))
#            a = copy.deepcopy(self)
#            #a._xy[:,1] += other
#
#            a.xy #ensure xy has been initialized
#            iop(a._xy[:,1], other)
        except TypeError:
            print("Unsupported operand type(s) for +: 'PolyLine' and "
                "'{}'".format(other.__class__.__name__))
#            sys.exit(0)
        return a


def polyline_make_x_common(*p_lines):
    """Add appropriate points to multiple PolyLine objetcs so that each has
    matching x1_x2 intevals.

    Parameters
    ----------
    p_lines : PolyLine
        One or more instances of PolyLine.

    Returns
    -------
    out : tuple of PolyLine
        Same number of Polyline's as `p_lines`.

    """

    xa = []
    ya = []

    for i, line in enumerate(p_lines):
        if not isinstance(line, PolyLine):
            raise TypeError("p_lines[{:d}] is not a PolyLine".format(i))
            sys.exit(0)
        if not (non_increasing(line.x) or non_decreasing(line.x)):
                raise TypeError('PolyLine #{:d} has switchbacks.'.format(i))
                sys.exit(0)

        if not is_initially_increasing(line.x):
            xa.append(line.x[::-1])
            ya.append(line.y[::-1])
        else:
            xa.append(line.x[:])
            ya.append(line.y[:])

    if len(p_lines)==1:
        return p_lines[0]

    data= np.zeros([sum(map(len,xa)),4])
    start = 0
    for i, x in enumerate(xa):
        stop = start + len(x)
        data[start:stop, 0]= x[:]
        data[start:stop, 1]= ya[i][:]
        data[start:stop, 2]= i #line
        data[start:stop, 3]= np.arange(len(x)) #orig position
        start = stop

    data = data[np.lexsort((data[:,3],data[:,2],data[:,0]))]


    pnts = [[] for i in xa]
    xnow =np.nan
    start = 0
    stop = 0

    atol = p_lines[0].atol
    rtol = p_lines[0].rtol

    i=0
    while i < len(data):
        x = data[i, 0]
        #ind = np.where(np.abs(data[:,0] - xnow) < (self.atol + self.rtol * np.abs(xnow)))[0]
        ind = abs(data[:,0] - x) < (atol + rtol * abs(x))# find all x values close to current x
        for k in range(len(xa)):

            j = np.where((ind) & (data[:, 2] == k))[0]
            if len(j) > 0:
                #use point a line point
                y1 = data[j[0], 1]
                y1_= data[j[-1],1]
            else: #interpolate
                y1 = interp_x_y(xa[k], ya[k], x, choose_max=False)
                y1_= y1
            pnts[k].append([x,y1])
            if abs(y1 - y1_)> (atol + rtol * abs(y1_)):
                pnts[k].append([x, y1_])

        i = np.nonzero(ind)[0][-1]+1
    return tuple(map(PolyLine, pnts))




def subdivide_x_y_into_segments(x, y, dx=None, min_segments = 2,
        just_before = None, logx=False, logy=False, logxzero=0.1,
        logyzero=0.1, rtol=1e-5, atol=1e-8):
    """Subdivide piecewise-linear line segment x, y data by interpolation

    Subsegments are evenly spaced in linear or log space.

    Parameters
    ----------
    x : 1d array-like
        List of x values to subdivide.
    y : 1d array-like
        List of y values to subdivide (y subdivision is based on
        interpolation at the new x values).
    dx : float, optional
        Approximate length or log10(length) of subsegment.  Say logx is True,
        segment x values are (1, 10), and x=0.2 then
        int((log(10)-log(1))/0.2)=5 subsegments will be created
        (i.e. 5 extra points will be inserted in the segment).  The new x
        values will be will be 10**(log(1)+0.2), 10**(log(1)+0.4),
        10**(log(1)+0.6) etc. Default dx=None i.e. no dx check and
        `min_segments` will be used.
    min_segments : int, optional
        Minuimum number of subsegments per inteval. This will be used if
        int(segment_lenght/dx)<min_segments. Default min_segments=2.
    just_before : float, optional
        If `just_before` is not None then in terms of subdividing, each
        segment will be treated as if it begins at its starting value but
        ends a distance of `just_before` multiplied by the interval length
        before its end value. This
        means that an extra point will be added just before all of the
        original points. Default just_before=None i.e dont use just before
        point.  Use a small number e.g. 1e-6.
        `just_before` can be useful for example when getting
        times to evaluate pore pressure during a soil consoliation
        analysis.  Say you have a PolyLine representing load_vs_time. If
        there are step changes in your load then there will be step
        changes in your pore pressure.  To capture the step change in your
        output you need an output time just before and just after the step
        change in your load.  Using `just_before` can achieve this.
    logx, logy : True/False, optional
        Use log scale on x and y axes.  Default logx=logy=False.
    logxzero, logyzero : float, optional
        If log scale is used, force zero value to be given number.
        Useful when 1st point is zero but you really want to start from close
        to zero. Default logxzero=logyzero=0.01.
    rtol, atol : float, optional
        For determining equal to zero when using log scale with numpy.
        Default atol=1e-8, rtol = 1e-5.


    Returns
    -------
    xnew, ynew : 1d array
        New x and y coordinates.

    See Also
    --------
    subdivide_x_into_segments : Subdivide only x data.

    """


    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)


    if len(x)!=len(y):
        raise ValueError('x and y must have same length '
                     'len(x)={:d}, len(y)={:d}'.format(len(x), len(y)))
    if logx:
        x[np.abs(x) <= (atol + rtol * np.abs(x))] = logxzero
        if np.any(x<0):
            raise ValueError('When logx=True cannot have negative '
                                'x values, x=' + x)
        x = np.log10(x)
    if logy:
        y[np.abs(y) <= (atol + rtol * np.abs(y))] = logyzero
        if np.any(y<0):
            raise ValueError('When logy=True cannot have negative '
                                'y values, y=' + x)
        y = np.log10(y)


    if dx is None:
        dx = 2*(np.max(x)-np.min(x))

    if just_before is None:
        xnew = [np.linspace(x1, x2, max(int(abs((x2-x1)//dx)), min_segments),
                         endpoint = False) for (x1, x2) in zip(x[:-1], x[1:])]

        ynew = [np.linspace(y1, y2, max(int(abs((x2-x1)//dx)), min_segments),
                            endpoint = False) for
                (x1, x2, y1, y2) in zip(x[:-1], x[1:], y[:-1], y[1:])]
    else:
        xnew = [np.linspace(x1, x1 + (1-just_before) * (x2 - x1),
                         max(int(abs((x2-x1)//dx)), min_segments)+1,
                         endpoint = True) for
                (x1, x2) in zip(x[:-1], x[1:])]


        ynew = [np.linspace(y1, y1 + (1-just_before) * (y2 - y1),
                         max(int(abs((x2-x1)//dx)), min_segments)+1,
                         endpoint = True) for
                (x1, x2, y1, y2) in zip(x[:-1], x[1:],
                                        y[:-1], y[1:])]

    # add in final point
    xnew.append([x[-1]])
    ynew.append([y[-1]])

    xnew = np.array([val for subl in xnew for val in subl])
    ynew = np.array([val for subl in ynew for val in subl])


    if logx:

        xnew = 10**xnew
    if logy:
        ynew = 10**ynew

    return (xnew, ynew)


def subdivide_x_into_segments(x, dx=None, min_segments=2,
        just_before = None, logx=False, logxzero=0.1,
        rtol=1e-5, atol=1e-8):
    """Subdivide sequential x values into subsegments.

    Intervals are evenly spaced in linear or log space

    Parameters
    ----------
    x : 1d array-like
        List of x values to subdivide.
    dx : float, optional
        Approximate length or log10(length) of subsegment.  Say logx is True,
        segment x values are (1, 10), and x=0.2 then
        int((log(10)-log(1))/0.2)=5 subsegments will be created
        (i.e. 5 extra points will be inserted in the segment).  The new x
        values will be will be 10**(log(1)+0.2), 10**(log(1)+0.4),
        10**(log(1)+0.6) etc. Default dx=None i.e. no dx check and
        `min_segments` will be used.
    min_segments : int, optional
        Minuimum number of subsegments per inteval. This will be used if
        int(segment_lenght/dx)<min_segments. Default min_segments=2.
    just_before : float, optional
        If `just_before` is not None then in terms of subdividing, each
        segment will be treated as if it begins at its starting value but
        ends a distance of `just_before` multiplied by the interval length
        before its end value. This
        means that an extra point will be added just before all of the
        original points. Default just_before=None i.e dont use just before
        point.  Use a small number e.g. 1e-6.
        `just_before` can be useful for example when getting
        times to evaluate pore pressure during a soil consoliation
        analysis.  Say you have a PolyLine representing load_vs_time. If
        there are step changes in your load then there will be step
        changes in your pore pressure.  To capture the step change in your
        output you need an output time just before and just after the step
        change in your load.  Using `just_before` can achieve this.
    logx : True/False, optional
        Use log scale on x data.  Default logx=False.
    logxzero : float, optional
        If log scale is used, force zero value to be given number.
        Useful when 1st point is zero but you really want to start from close
        to zero. Default logxzero=0.01.
    rtol, atol : float, optional
        For determining equal to zero when using log scale with numpy.
        Default atol=1e-8, rtol = 1e-5.

    Returns
    -------
    xnew : 1d array
        new x values

    See Also
    --------
    subdivide_x_y_into_segments : Subdivide both x and y data, y data is
        interpolated from the new x values.


    """


    x = np.asarray(x, dtype=float)



    if logx:
        x[np.abs(x) <= (atol + rtol * np.abs(x))] = logxzero
        if np.any(x<0):
            raise ValueError('When logx=True cannot have negative '
                                'x values, x=' + x)
        x = np.log10(x)


    if dx is None:
        dx = 2*(np.max(x)-np.min(x))

    if just_before is None:
        xnew = [np.linspace(x1, x2, max(int(abs((x2-x1)//dx)), min_segments),
                         endpoint = False) for (x1, x2) in zip(x[:-1], x[1:])]

    else:
        xnew = [np.linspace(x1, x1 + (1-just_before) * (x2 - x1),
                         max(int(abs((x2-x1)//dx)), min_segments)+1,
                         endpoint = True) for
                (x1, x2) in zip(x[:-1], x[1:])]


    # add in final point
    xnew.append([x[-1]])


    xnew = np.array([val for subl in xnew for val in subl])



    if logx:

        xnew = 10**xnew


    return xnew


def layer_coords(h, segs, min_segs=1):
    """Divide layer heights into segments and calculate offset from start

    Subdivide each layer. Calculate offset.

    Useful when you are interested in the end points of each layer as well as
    some intermediate points.

    Parameters
    ----------
    h : list/array
        1d array of layer heights.
    segs : int
        Approximate number of segments to subdivide the whole profile into.
    min_segs : int, optional
        Minimum number of segments to subdivide a layer into. Default
        min_segs=2.

    Returns
    -------
    z : 1d array
        Depths.


    Examples
    --------
    >>> layer_coords([4], 2)
    array([ 0.,  2.,  4.])
    >>> layer_coords([2], 1, 4)
    array([ 0. ,  0.5,  1. ,  1.5,  2. ])
    >>> layer_coords([2,4,2], 5, 2)
    array([ 0.,  1.,  2.,  4.,  6.,  7.,  8.])
    >>> layer_coords([5,1,5], 3, 2)
    array([  0. ,   2.5,   5. ,   5.5,   6. ,   8.5,  11. ])


    """


    h = np.asarray(h, dtype=float)

    z2 = np.cumsum(h)
    z1 = z2 - h
    dx = np.sum(h) / segs
    num = [max(height//dx, min_segs) for height in h]

    zlist = [np.linspace(z1_, z2_, num_, endpoint=False) for (z1_, z2_, num_) in zip(z1, z2, num)]
    zlist.append([z2[-1]])

    z = np.array([val for subl in zlist for val in subl])

    return z


def subdivide_into_elements(n=2, h=1.0, p=1, symmetry=True):
    """Subdivide a length into elements where sizes are proportional to
    adjacent elements.


     - element_0 = x * p**0
     - element_1 = x * p**1
     - element_2 = x * p**2
     - etc.

    x is such that sum of `n` elements equals `h`.

    Parameters
    ----------
    n : int, optional
        Number of elements to subdivide into. Default n=2.
    h : float, optional
        Length to subdivide.  Default h=1.
    p : float, optional
        Ratio of subsequent elements.  If p<1 then succesive elements
        will reduce in length.  If p>1 then successive elements
        will increase in lenght. Default p=1
    symmetry : True/False, optional
        If True then elements will be symmetrical abount middle.  Default
        symmetry=True.

    Returns
    -------
    out : array of float
        Length of each element. Should sum to `h`.

    See also
    --------
    np.logspace
    np.linspace
    subdivide_x_into_segments
    subdivide_x_y_into_segments


    Examples
    --------
    >>> subdivide_into_elements(n=3, h=6.0, p=1, symmetry=True)
    array([ 2.,  2.,  2.])
    >>> subdivide_into_elements(n=4, h=6.0, p=1, symmetry=True)
    array([ 1.5,  1.5,  1.5,  1.5])
    >>> subdivide_into_elements(n=4, h=6.0, p=1, symmetry=False)
    array([ 1.5,  1.5,  1.5,  1.5])
    >>> subdivide_into_elements(n=3, h=6.0, p=2, symmetry=True)
    array([ 1.5,  3. ,  1.5])
    >>> subdivide_into_elements(n=3, h=6.0, p=0.5, symmetry=True)
    array([ 2.4,  1.2,  2.4])
    >>> subdivide_into_elements(n=4, h=6.0, p=2, symmetry=True)
    array([ 1.,  2.,  2.,  1.])
    >>> subdivide_into_elements(n=4, h=6.0, p=0.5, symmetry=True)
    array([ 2.,  1.,  1.,  2.])
    >>> subdivide_into_elements(n=4, h=6.0, p=0.5, symmetry=False)
    array([ 3.2,  1.6,  0.8,  0.4])
    >>> subdivide_into_elements(n=3, h=3.5, p=2, symmetry=False)
    array([ 0.5,  1. ,  2. ])
    >>> sum(subdivide_into_elements(n=20, h=3.5, p=1.05, symmetry=False))
    3.5

    """

    if p<=0:
        raise ValueError("p must be greater than 0")
    if n<=1:
        raise ValueError("p must be an integer greater than 0")



    if p==1:
        return np.diff(np.linspace(0,h, n+1))

    ppower = np.arange(n)
    if symmetry:
        if n%2==0: #even
            x = h
            x /= 2 * (p**(n / 2) - 1) / (p - 1)
            ppower[n / 2:] = np.arange(n / 2)[::-1]
        else: #odd
            x = h
            x /=  2 * (p**((n + 1) / 2) - 1) / (p - 1) - p**((n - 1) / 2)
            ppower[(n + 1) / 2:] = np.arange((n - 1) / 2)[::-1]
    else:
        x = h / (p**n - 1) * (p - 1)
        ppower = np.arange(n)

    return x*p**ppower



if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])
