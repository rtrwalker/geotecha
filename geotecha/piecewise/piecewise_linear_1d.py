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
module for piecewise 1d linear relationships

"""
from __future__ import print_function, division

import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import copy
import operator
#import operator
#import numbers

def has_steps(x):
    """check if data points have any step changes

    True if any two consecutive x values are equal

    Parameters
    ----------
    x : array_like
        x-coordinates

    Returns
    -------
    out : boolean
        returns true if any two consecutive x values are equal

    """
    #TODO: maybe check for bad segments such as x=[2,2] y=[10,10]
    x = np.asarray(x)

    return np.any(np.diff(x)==0)


def is_initially_increasing(x):
    """Are first two values increasing?

    finds 1st instance where x[i+1] != x[i] and checks if x[i+1] > x[i]

    Parameters
    ----------
    x : array_like
        1 dimensional data to check

    Returns
    -------
    out : ``int``
        returns True is if 2nd value is greater than the 1st value
        returns False if 2nd value is less than the 1st value

    """

    #this might be slow for long lists, perhaps just loop through until x[i+1]!=x[i]
    if x[1]!=x[0]:
        return x[1]>x[0]
    x = np.asarray(x)
    return np.where(np.diff(x)!=0)[0][0]>0




#used info from http://stackoverflow.com/questions/4983258/python-how-to-check-list-monotonicity
def strictly_increasing(x):
    """Checks all x[i+1] > x[i]"""
    x = np.asarray(x)
    return np.all(np.diff(x)>0)

def strictly_decreasing(x):
    """Checks all x[i+1] < x[i]"""
    x = np.asarray(x)
    return np.all(np.diff(x)<0)

def non_increasing(x):
    """Checks all x[i+1] <= x[i]"""
    x = np.asarray(x)
    return np.all(np.diff(x)<=0)

def non_decreasing(x):
    """Checks all x[i+1] >= x[i]"""
    x = np.asarray(x)
    return np.all(np.diff(x)>=0)



def non_increasing_and_non_decreasing_parts(x, include_end_point = False):
    """split up a list into sections that are non-increasing and non-decreasing

    Returns
    -------

    A : 2d array
        each element of A is a list of the start indices of each line segment that is part of a particular
        non-increasing or non-decreasing run

    Notes
    -----
    This funciton only returns start indices for each line segment.
    Lets say x is [0,4 , 5.5] then A will be [[0,1]].  If you do x[A[0]] then
    you will get [0, 4] i.e. no end point. To get the whole increasing or
    decreasing portion including the end point you need to do something like
    x[A[0].append[A[0][-1]+1]].

    """

    #TODO: maybe return slice object rather than a list of all the indecies as they will be contiguous anyway
    x = np.asarray(x)
    sign_changes = np.sign(np.diff(x))
    A = [[0]]

    current_sign = sign_changes[np.where(sign_changes!=0)[0][0]]

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
    """force a non-decreasing or non-increasing list into a strictly increasing

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
        list of x coordinates
    y : array_like, optional
        list of y coordinates, (default = None, if known that x is
        non-decreasing then y will not be affected)
    keep_end_points : ``boolean``, optional
        determines which x value of the step change is adjusted.
        Consider x=[1,1] and y=[20,40]. If keep_end_points==True then new
        data will be x=[0.9999,1], y=[20,40].  If keep_end_points==False then
        data will be x=[1, 0.9999], y=[20,40]
    eps : float, optional
        amount to add/subtract from x (default is 1e-15).  To ensure
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
        raise ValueError, "x data is neither non-increasing, nor non-decreasing, therefore cannot force to strictly increasing"


    steps = np.where(np.diff(x) == 0)[0]

    if keep_end_points:
        f = -1 * eps
        d = 0
        dx = np.arange(len(steps),0, -1) * f
    else:
        f = 1 * eps
        d = 1
        dx = np.arange(1,len(steps)+1) * f

    x[steps + d] = x[steps + d] + dx
    return x, y

def force_non_decreasing(x, y = None):
    """force non-increasing x, y data to non_decreasing by reversing the data

    Leaves already non-decreasing data alone.

    Parameters
    ----------
    x, y: array_like
        x and y coordinates

    Returns
    -------
    x,y : 1d ndarray, 1d ndarray
        non-decreasing version of x, y

    """
    x = np.asarray(x)
    y = np.asarray(y)

    if non_decreasing(x):
        return x, y

    if not non_increasing(x):
        raise ValueError, "x data is neither non-increasing, nor non-decreasing, therefore cannot force to non-decreasing"

    return x[::-1], y[::-1]










def start_index_of_ramps(x, y):
    """find the start indices of the ramp segments in x, y data.

    An example of a 'ramp' x=[0,2], y=[10,15]. i.e. not a vertical line and
    not a horizontal line.

    Assumes data is non_decreasing.

    Parameters
    ----------
    x, y : array_like
        x and y coords (must be non-decreasing)

    Returns
    -------
    out : 1d ndarray
        start indices of all ramps.

    """

    x = np.asarray(x)
    y = np.asarray(y)



    return  np.where((np.diff(x)!=0) & (np.diff(y)!=0))[0]

def start_index_of_constants(x, y):
    """find the start indices of the constant segments in x, y data.

    An example of a 'ramp' x=[0,2], y=[15,15]. i.e. a horizontal line

    Assumes data is non_decreasing.

    Segments such as x=[1,1], y=[2,2] are ignored.

    Parameters
    ----------
    x, y : array_like
        x and y coords (must be non-decreasing)

    Returns
    -------
    out : 1d ndarray
        start indices of all constant segments.

    """

    x = np.asarray(x)
    y = np.asarray(y)


    return np.delete(np.where(np.diff(y)==0)[0], np.where((np.diff(x)==0) & (np.diff(y)==0))[0])

def start_index_of_steps(x, y):
    """find the start indices of the step segments in x, y data.

    An example of a 'step' x=[0,0], y=[10,15]. i.e. a vertical line

    Assumes data is non_decreasing.

    Segments such as x=[1,1], y=[2,2] are ignored.

    Parameters
    ----------
    x, y : array_like
        x and y coords (must be non-decreasing)

    Returns
    -------
    out : 1d ndarray
        start indices of all step segments.

    """

    x = np.asarray(x)
    y = np.asarray(y)


    return np.delete(np.where(np.diff(x)==0)[0], np.where((np.diff(x)==0) & (np.diff(y)==0))[0])



def segment_containing_xi(x, xi, subset = None, choose_max = False):
    """Find start index of segment in which xi falls

    Find index where x[i] <= xi <= x[i+1] ignoring steps. `choose_max`
    determines what happens when more than one segment satisfies the
    condition e.g. the boundary between two segments; taking either the
    maximum index (`choose_max` True), or the minimum index (`choose_max`=
    False).

    Parameters
    ----------
    x : array_like, float
        x coordinates
    xi : array_like, float
    subset : array_like, optional
        restrict search to segments starting with indices in `subset`.
        Default = search all segments)
    choose_max : boolean, optional
        When False (default), the minumum index that satisfies the condition
        is returned. When True the maximum index that satisfies the condition
        is returned.

    Returns
    -------
    A: list of single element lists
        each sub-list is the start index of the segement that contains xi.
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



def segments_less_than_xi(x, xi, subset = None, or_equal_to = False):
    """Find start index of segments that end before xi

    Finds all segments where end point of segment is less than xi

    Assumes non-decreasing `x` data

    Parameters
    ----------
    x : array_like, float
        x coordinates
    xi : array_like, float
        values to check if segments start after
    subset : array_like, optional
        restrict search to segments starting with indices in `subset`.
        (Default is to search all segments)
    or_equal_to : ``boolean``, optional
        if False (default) then conditon is formulated with '<'.  If True
        then condition is formulated with '<='. Generally ony used to
        include/exclude step loads.

    Returns
    -------
    out: list of 1d numpy.ndarray
        list contains len(xi) 1d numpy.ndarray corresponding to xi

    """
    x = np.asarray(x)

    xi = np.atleast_1d(xi)
#    if isinstance(xi, numbers.Number):
#        xi = np.array([xi])
#    xi = np.asarray(xi)




    if subset is None:
        subset = np.arange(len(x)-1)

#    if len(subset)==0: #subset isempty
#        return [np.array([],dtype = int) for v in xi]

    subset = np.asarray(subset)

    if or_equal_to:
        return [subset[x[subset+1] <= v] for v in xi]
    else:
        return [subset[x[subset+1] < v] for v in xi]




def ramps_constants_steps(x, y):
    """find the ramp segments, constant segments and step segments in x, y data

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
    start_index_of_ramps : find ramps
    start_index_of_steps : find steps
    start_index_of_constants : find constants

    """

    x = np.asarray(x)
    y = np.asarray(y)

    ramps = start_index_of_ramps(x,y)
    constants = start_index_of_constants(x,y)
    steps = start_index_of_steps(x,y)

    return (ramps, constants, steps)

def segment_containing_also_segments_less_than_xi(x, y, xi, steps_or_equal_to = True, ramp_const_or_equal_to = False, choose_max = False):
    """For ramps, constants and steps find segments that contain xi. Also segments that are after xi

    Function does minimal calculations itself, essentially calling other
    functions and returning a tuple.

    Parameters
    ----------
    x, y : array_like
        x and y coords (must be non-decreasing)
    xi : array_like, float
        values to check if segments start after
    steps_or_equal_to : ``boolean``, optional
        if True (default) then any step segment that xi falls on/in will be
        included in the steps 'less than' list.
    steps_or_equal_to : ``boolean``, optional
        if True (default) then any ramp or constant segment that xi falls
        on/in will be included in the steps 'less than' list.
    choose_max : ``boolean``, optional
        if False (default), then the minimum segment of multiple ramp segments
        that contain xi falls will be included in the 'contains' lists.
        If True then the maximum segment will be included.

    Returns
    -------
    ramps_less_than_xi : ndarray
        `segments_less_than_xi` for ramps
    constants_less_than_xi : ndarray
        `segments_less_than_xi` for constants
    steps_less_than_xi : ndarray
        `segments_less_than_xi` for steps
    ramps_containing_xi : ndarray
        start index of ramp segment containing xi
    constants_containing_xi :
        start index of constant segment containing xi


    See Also
    --------
    ramps_constants_steps : find ramp, constant, and step segments
    segments_less_than_xi : find segments after xi
    segment_containing_xi : find segments that contain xi

    """

    x = np.asarray(x)
    y = np.asarray(y)

    xi = np.atleast_1d(xi)
#    if isinstance(xi, numbers.Number):
#        xi = np.array([xi])
#    xi = np.asarray(xi)


    ramps, constants, steps = ramps_constants_steps(x,y)

    ramps_less_than_xi = segments_less_than_xi(x, xi, subset = ramps, or_equal_to = ramp_const_or_equal_to)
    constants_less_than_xi = segments_less_than_xi(x, xi, subset = constants, or_equal_to = ramp_const_or_equal_to)
    steps_less_than_xi = segments_less_than_xi(x, xi, subset = steps, or_equal_to = steps_or_equal_to)

    # the temptation to call segment_containing_xi here with subset=ramps and then subset=constants might lead to xi being in both a ramp or a constant
    contains = segment_containing_xi(x, xi, choose_max = choose_max)
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
    """Find start index of segments that xi and xj fall in trying to have them in the same section.

    Find highest i where x[i] <= xi <= x[i+1] ignoring steps.
    Find lowest j where  x[j] <= xj <= x[j+1] ignoring steps.
    This will minimise the number of segments between xi and xj.
    Usually xj should be greater than xi.
    Function does no calculations itself, rather calling segment_containing_xi
    twice and returning a tuple.

    Parameters
    ----------
    x : array_like, float
        x coordinates
    xi, xj : array_like, float
        x values to from which to determin containing segment.
    subset : array_like, optional
        restrict search to segments starting with indices in `subset`.
        Default = search all segments)


    Returns
    -------
    seg_xi, seg_xj: list of single element lists
        each sub-list is the start index of the segement that contains xi or xj.

    See also
    --------
    segments__containing_xi : function called for `xi` and `xj`

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
    """Find segments that exclusively contain both, only one of, and in between xi and xj

    Determine if xi and xj are both in the same segment
    When xi and xj are in different segments find the segment that contains xi
    and the segment that contains xj and any segments in between them.
    Results are only obvious when x is strictly increasing and xi>xj
    Parameters
    ----------
    x : array_like, float
        x coordinates
    xi, xj : array_like, float
        x values to from which to determin segments.

    Returns
    -------
    segment_both : list of single element lists
        each sub-list is the start index of the segement that contains xi or xj.
    segment_xi_only : list of single element lists
        when xi and xj not in the same segement, segment_xi_only will be the
        segment that contains xi.
    segment_xj_only : list of single element lists
        when xi and xj not in the same segement, segment_xj_only will be the
        segment that contains xj.
    segments_between : list of single element lists
        when xi and xj not in the same segement, segments_between will be the
        segments in between xi and xj but not containing xi or xj.

    See also
    --------
    segment_containing_xi_also_containing_xj : find segment of xi and xj

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
    """convert data defined at start and end of each segment to a line of data

    x1_x2_y1_y2 data is defined by x1[1:]==x2[:-1]

    Parameters
    ----------
    x1, y1 : array_like, float
        x and y values at start of each segment
    x2, y2 : array_like, float
        x and y values at end of each segment (note x1[1:]==x2[:-1])

    Returns
    -------
    x, y : 1d ndarray, float
        x and y data of continuous line that matches the x1_x2_y1_y2 data

    See also
    --------
    `convert_x_y_to_x1_x2_y1_y2` : reverse of this function

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
    """convert a line of data to data defined at start and end of each segment


    Parameters
    ----------
    x1, y1 : array_like, float
        x and y values at start of each segment
    x2, y2 : array_like, float
        x and y values at end of each segment (note x1[1:]==x2[:-1])

    Returns
    -------
    x1, y1 : 1d array, float
        x and y values at start of each segment
    x2, y2 : 1d array, float
        x and y values at end of each segment (note x1[1:]==x2[:-1])

    See also
    --------
    `convert_x1_x2_y1_y2_to_x_y` : reverse of this function

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
    """wrapper for interp_x1_x2_y1_y2 to allow PolyLine inputs

    See also
    --------
    interp_x1_x2_y1_y2

    """

    return interp_x1_x2_y1_y2(a.x1,a.x2,a.y1,a.y2, xi, **kwargs)


def interp_x1_x2_y1_y2(x1,x2,y1,y2, xi, choose_max = False):
    """interpoolate x1_x2_y1_y2 data

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
        interpolated y value corresponding to xi



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
    """wrapper for interp_x_y to accept PolyLine inputs

    See also
    --------
    interp_x_y

    """

    return interp_x_y(a.x, a.y, xi, **kwargs)

def interp_x_y(x,y,xi, choose_max = False):
    """interpoolate x, y data


    if xi is beyond bounds of x then the first or last value of y will be
    returned as appropriate

    Parameters
    ----------
    x, y : array_like, float
        x and y values
    xi : array_like, float
        x values to interpolate at

    Returns
    -------
    A : 1d ndarray, float
        interpolated y value corresponding to xi


    """

    x = np.asarray(x)

    y = np.asarray(y)



    if len(x)!=len(y):
            raise ValueError("x and y must be of same length")



    xi = np.atleast_1d(xi)
#    if isinstance(xi, numbers.Number):
#        xi = np.array([xi])
#    xi = np.asarray(xi)

    segs = segment_containing_xi(x, xi, subset = None, choose_max = choose_max)


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



def remove_superfluous_from_x_y(x,y, atol=1e-08):
    """Remove points that are on a line between other points

    Intermediate points are judged to be 'on a line' and therefore superfluous
    whenis determined by distance of the point from the line is below `atol`.

    Parameters
    ----------
    x, y : 1d array_like, float
        x and y values
    atol : float, optional
        atol is the threshold

    Returns
    -------
    xnew, ynew : 1d ndarray, float
        cleaned up x and y values

    Notes
    -----
    #TODO: put in equation for distance of point to a line from wolfram http://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html

    """
    n = len(x)
    if n!=len(y):
        raise ValueError("x and y must be of sam length, %s vs %s" % (n, len(y)))

    x = np.asarray(x)
    y = np.asarray(y)

    if n<=2:
        return x, y

    ikeep = range(len(x))


    j = 0
    x0 = x[j]
    y0 = y[j]
    for i in xrange(2,n):
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
    """wrapper for interp_xa_ya_multipy_x1b_x2b_y1b_y2b to use PolyLine inputs

    See also
    --------
    interp_xa_ya_multipy_x1b_x2b_y1b_y2b

    """

    return interp_xa_ya_multipy_x1b_x2b_y1b_y2b(a.x, a.y, b.x1, b.x2, b.y1, b.y2, xai, xbi,**kwargs)

def interp_xa_ya_multipy_x1b_x2b_y1b_y2b(xa, ya, x1b, x2b, y1b, y2b, xai, xbi, achoose_max=False, bchoose_max=True):
    """interpolate where f(a, b) defined as g(a)*h(b) where g(a) is defined with x_y data and h(b) is defined by x1_x2_y1_y2 data

    Does little calculation, mostly calls other functions
    calculates array A[len(xbi),len(xai)]
    Parameters
    ----------
    xa, ya : 1d array_like, float
        x and y values of x_y part of interpolation function
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
        if False (default), if xai falls on boundary of segments choose the
        minimum segment to interpolate within.
    bchoose_max : ``boolean``, optional
        if true (default), if xbi falls on boundary of segments choose the
        maximum segment to interpolate within.

    See also
    --------
    interp_x_y : interpolate the x_y part
    interp_x1_x2_y1_y2 : interpolate the x1_x2_y1_y2 part

    """

    yai = interp_x_y(xa, ya, xai, choose_max=achoose_max)
    ybi = interp_x1_x2_y1_y2(x1b, x2b, y1b, y2b, xbi, choose_max=bchoose_max)

    return ybi[:, np.newaxis] * yai[np.newaxis,:]

def pavg_x_y_between_xi_xj(a, xi, xj, **kwargs):
    """wrapper for avg_x_y_between_xi_xj to allow polyline inputs

    See also
    --------
    avg_x_y_between_xi_xj


    """

    return avg_x_y_between_xi_xj(a.x, a.y, xi, xj, **kwargs)


def avg_x_y_between_xi_xj(x, y, xi, xj):
    """find average between xi and xj of x_y


    calculates array A[len(xi)]
    Parameters
    ----------
    x, y : 1d array_like, float
        x and y values of x_y part of interpolation function

    xi, xj : array_like, float
        x values to interpolate between

    See also
    --------
    integrate_x_y_between_xi_xj : integration is intemediate step in average calculation

    """

    xi = np.atleast_1d(xi)
    xj = np.atleast_1d(xj)

    return integrate_x_y_between_xi_xj(x, y, xi, xj) / (xj - xi)

def pintegrate_x_y_between_xi_xj(a, xi, xj, **kwargs):
    """wrapper for integrate_x_y_between_xi_xj to allow PolyLine inputs

    See also
    --------
    integrate_x_y_between_xi_xj

    """

    return integrate_x_y_between_xi_xj(a.x, a.y, xi, xj, **kwargs)


def integrate_x_y_between_xi_xj(x, y, xi, xj):
    """integrate x_y data between xi and xj"


    calculates array A[len(xi)]
    Parameters
    ----------
    x, y : 1d array_like, float
        x and y values of x_y part of interpolation function

    xi, xj : array_like, float
        x values to integrate between

    See also
    --------
    interp_x_y : interpolate the x_y part
    segments_between_xi_and_xj : segments between xi and xj
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
    """wrapper for integrate_x1_x2_y1_y2_between_xi_xj to allow PolyLine inputs

    See also
    --------
    integrate_x1_x2_y1_y2_between_xi_xj

    """

    return integrate_x1_x2_y1_y2_between_xi_xj(a.x1, a.x2, a.y1, a.y2, xi, xj, **kwargs)


def integrate_x1_x2_y1_y2_between_xi_xj(x1, x2, y1, y2, xi, xj):
    """integrate x1_x2_y1_y2 data between xi and xj"


    calculates array A[len(xi)]
    Parameters
    ----------
    x1, y1 : array_like, float
        x and y values at start of each segment
    x2, y2 : array_like, float
        x and y values at end of each segment (note x1[1:]==x2[:-1])
    xi, xj : array_like, float
        x values to integrate between

    See also
    --------


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
    """wrapper for avg_x1_x2_y1_y2_between_xi_xj to use PolyLine inputs

    See also
    --------
    avg_x1_x2_y1_y2_between_xi_xj

    """

    return avg_x1_x2_y1_y2_between_xi_xj(a.x1, a.x2, a.y1, a.y2, xi, xj, **kwargs)


def avg_x1_x2_y1_y2_between_xi_xj(x1, x2, y1, y2, xi, xj):
    """average of x1_x2_y1_y2 data between xi and xj"


    calculates array A[len(xi)]

    Parameters
    ----------
    x1, y1 : array_like, float
        x and y values at start of each segment
    x2, y2 : array_like, float
        x and y values at end of each segment (note x1[1:]==x2[:-1])
    xi, xj : array_like, float
        x values to integrate between

    See also
    --------
    integrate_x1_x2_y1_y2_between_xi_xj : integration is intermediate step for average calculation

    """


    xi = np.atleast_1d(xi)
    xj = np.atleast_1d(xj)

    return integrate_x1_x2_y1_y2_between_xi_xj(x1, x2, y1, y2, xi, xj) / (xj - xi)


def pxa_ya_multipy_avg_x1b_x2b_y1b_y2b_between(a,b, xai, xbi, xbj, **kwargs):
    """wrapper for xa_ya_multipy_avg_x1b_x2b_y1b_y2b_between to have PolyLine inputs

    See also
    --------
    xa_ya_multipy_avg_x1b_x2b_y1b_y2b_between

    """

    return xa_ya_multipy_avg_x1b_x2b_y1b_y2b_between(a.x,a.y,b.x1, b.x2, b.y1, b.y2, xai, xbi, xbj, **kwargs)



def xa_ya_multipy_avg_x1b_x2b_y1b_y2b_between(xa, ya, x1b, x2b, y1b, y2b, xai, xbi, xbj, achoose_max=False):
    """average the x1_x2_y1_y2 part between xbi, and xbj of f(a, b) which is defined as g(a)*h(b) where g(a) is defined with x_y data and h(b) is defined by x1_x2_y1_y2 data

    Does little calculation, mostly calls other functions
    calculates array A[len(xbi), len(xai)]

    Parameters
    ----------
    xa, ya : 1d array_like, float
        x and y values of x_y part of interpolation function
    x1b, y1b : array_like, float
        x and y values at start of each segment for x1_x2_y1_y2 part of
        interpolation function
    x2b, y2b : array_like, float
        x and y values at end of each segment for x1_x2_y1_y2 part of
        interpolation function  (note x1[1:]==x2[:-1])
    xai : array_like, float
        x values to interpolate at for x_y part
    xbi, xbj : array_like, float
        x values to average between for the x1_x2_y1_y2 part
    achoose_max : ``boolean``, optional
        if False (default), if xai falls on boundary of segments choose the
        minimum segment to interpolate within.

    See also
    --------
    interp_x_y : interpolate the x_y part
    avg_x1_x2_y1_y2_between_xi_xj : average the x1_x2_y1_y2 part between xbi, xbj


    """

    yai = interp_x_y(xa, ya, xai, choose_max=achoose_max)
    ybi = avg_x1_x2_y1_y2_between_xi_xj(x1b, x2b, y1b, y2b, xbi, xbj)
    return ybi[:, np.newaxis] * yai[np.newaxis,:]

def pintegrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between(a,b,xi,xj, **kwargs):
    """wrapper for integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between to allow PolyLine inputs

    See also
    --------
    integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between

    """

    a, b = polyline_make_x_common(a, b)
    return integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between(a.x1,a.x2,a.y1,a.y2,b.x1,b.x2,b.y1,b.y2,xi,xj, **kwargs)



def integrate_x1a_x2a_y1a_y2a_multiply_x1b_x2b_y1b_y2b_between(x1a, x2a, y1a,y2a,x1b,x2b,y1b,y2b,xi,xj):
    """integrate between xi, xj the multiplication of two x1_x2_y1_y2 funcitons

    calculates array A[len(xi)]
    currently works only for x1a==x1b, x2a==x2b
    Parameters
    ----------
    x1a, y1a : array_like, float
        x and y values at start of each segment 2nd x1_x2_y1_y2 part of
        function
    x2a, y2a : array_like, float
        x and y values at end of each segment for 2nd x1_x2_y1_y2 part of
        function  (note x1[1:]==x2[:-1])
    x1b, y1b : array_like, float
        x and y values at start of each segment 2nd x1_x2_y1_y2 part of
        function
    x2b, y2b : array_like, float
        x and y values at end of each segment for 2nd x1_x2_y1_y2 part of
        function  (note x1[1:]==x2[:-1])
    xi, xj : array_like, float
        x values to average between


    Notes
    -----
    TODO: I think this only works if the two distributions have the same z values?? not sure
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
        raise ValueError ("x values are different; they must be the same: \nx1a = {0}\nx1b = {1}\nx2a = {2}\nx2b = {3}".format(x1a,x1b, x2a, x2b))
        #sys.exit(0)

    xi = np.atleast_1d(xi)
    xj = np.atleast_1d(xj)

    x_for_interp = np.zeros(len(x1a)+1)
    x_for_interp[:-1] = x1a[:]
    x_for_interp[-1] = x2a[-1]


    (segment_both, segment_xi_only, segment_xj_only, segments_between) = segments_between_xi_and_xj(x_for_interp, xi, xj)

#    yi = interp_x1_x2_y1_y2(x1,x2,y1,y2,xi, choose_max = True)
#    yj = interp_x1_x2_y1_y2(x1,x2,y1,y2,xj, choose_max = False)


    A = np.zeros(len(xi))
    for i in range(len(xi)):
        for seg in segment_both[i]:
            A[i] += -(-6*x1a[seg]*x2a[seg] + 3*x1a[seg]**2 + 3*x2a[seg]**2)**(-1)*(y1b[seg]*y1a[seg] - y1b[seg]*y2a[seg] - y2b[seg]*y1a[seg] + y2b[seg]*y2a[seg])*xi[i]**3 + (-6*x1a[seg]*x2a[seg] + 3*x1a[seg]**2 + 3*x2a[seg]**2)**(-1)*(y1b[seg]*y1a[seg] - y1b[seg]*y2a[seg] - y2b[seg]*y1a[seg] + y2b[seg]*y2a[seg])*xj[i]**3 - (-4*x1a[seg]*x2a[seg] + 2*x1a[seg]**2 + 2*x2a[seg]**2)**(-1)*(x1a[seg]*y1b[seg]*y2a[seg] + x1a[seg]*y2b[seg]*y1a[seg] - 2*x1a[seg]*y2b[seg]*y2a[seg] - 2*x2a[seg]*y1b[seg]*y1a[seg] + x2a[seg]*y1b[seg]*y2a[seg] + x2a[seg]*y2b[seg]*y1a[seg])*xi[i]**2 + (-4*x1a[seg]*x2a[seg] + 2*x1a[seg]**2 + 2*x2a[seg]**2)**(-1)*(x1a[seg]*y1b[seg]*y2a[seg] + x1a[seg]*y2b[seg]*y1a[seg] - 2*x1a[seg]*y2b[seg]*y2a[seg] - 2*x2a[seg]*y1b[seg]*y1a[seg] + x2a[seg]*y1b[seg]*y2a[seg] + x2a[seg]*y2b[seg]*y1a[seg])*xj[i]**2 - (-2*x1a[seg]*x2a[seg] + x1a[seg]**2 + x2a[seg]**2)**(-1)*(-x1a[seg]*x2a[seg]*y1b[seg]*y2a[seg] - x1a[seg]*x2a[seg]*y2b[seg]*y1a[seg] + x1a[seg]**2*y2b[seg]*y2a[seg] + x2a[seg]**2*y1b[seg]*y1a[seg])*xi[i] + (-2*x1a[seg]*x2a[seg] + x1a[seg]**2 + x2a[seg]**2)**(-1)*(-x1a[seg]*x2a[seg]*y1b[seg]*y2a[seg] - x1a[seg]*x2a[seg]*y2b[seg]*y1a[seg] + x1a[seg]**2*y2b[seg]*y2a[seg] + x2a[seg]**2*y1b[seg]*y1a[seg])*xj[i]
        for seg in segment_xi_only[i]:
            A[i] += (-6*x1a[seg]*x2a[seg] + 3*x1a[seg]**2 + 3*x2a[seg]**2)**(-1)*(y1b[seg]*y1a[seg] - y1b[seg]*y2a[seg] - y2b[seg]*y1a[seg] + y2b[seg]*y2a[seg])*x2a[seg]**3 - (-6*x1a[seg]*x2a[seg] + 3*x1a[seg]**2 + 3*x2a[seg]**2)**(-1)*(y1b[seg]*y1a[seg] - y1b[seg]*y2a[seg] - y2b[seg]*y1a[seg] + y2b[seg]*y2a[seg])*xi[i]**3 + (-4*x1a[seg]*x2a[seg] + 2*x1a[seg]**2 + 2*x2a[seg]**2)**(-1)*(x1a[seg]*y1b[seg]*y2a[seg] + x1a[seg]*y2b[seg]*y1a[seg] - 2*x1a[seg]*y2b[seg]*y2a[seg] - 2*x2a[seg]*y1b[seg]*y1a[seg] + x2a[seg]*y1b[seg]*y2a[seg] + x2a[seg]*y2b[seg]*y1a[seg])*x2a[seg]**2 - (-4*x1a[seg]*x2a[seg] + 2*x1a[seg]**2 + 2*x2a[seg]**2)**(-1)*(x1a[seg]*y1b[seg]*y2a[seg] + x1a[seg]*y2b[seg]*y1a[seg] - 2*x1a[seg]*y2b[seg]*y2a[seg] - 2*x2a[seg]*y1b[seg]*y1a[seg] + x2a[seg]*y1b[seg]*y2a[seg] + x2a[seg]*y2b[seg]*y1a[seg])*xi[i]**2 + (-2*x1a[seg]*x2a[seg] + x1a[seg]**2 + x2a[seg]**2)**(-1)*(-x1a[seg]*x2a[seg]*y1b[seg]*y2a[seg] - x1a[seg]*x2a[seg]*y2b[seg]*y1a[seg] + x1a[seg]**2*y2b[seg]*y2a[seg] + x2a[seg]**2*y1b[seg]*y1a[seg])*x2a[seg] - (-2*x1a[seg]*x2a[seg] + x1a[seg]**2 + x2a[seg]**2)**(-1)*(-x1a[seg]*x2a[seg]*y1b[seg]*y2a[seg] - x1a[seg]*x2a[seg]*y2b[seg]*y1a[seg] + x1a[seg]**2*y2b[seg]*y2a[seg] + x2a[seg]**2*y1b[seg]*y1a[seg])*xi[i]
        for seg in segments_between[i]:
            A[i] += -(-6*x1a[seg]*x2a[seg] + 3*x1a[seg]**2 + 3*x2a[seg]**2)**(-1)*(y1b[seg]*y1a[seg] - y1b[seg]*y2a[seg] - y2b[seg]*y1a[seg] + y2b[seg]*y2a[seg])*x1a[seg]**3 + (-6*x1a[seg]*x2a[seg] + 3*x1a[seg]**2 + 3*x2a[seg]**2)**(-1)*(y1b[seg]*y1a[seg] - y1b[seg]*y2a[seg] - y2b[seg]*y1a[seg] + y2b[seg]*y2a[seg])*x2a[seg]**3 - (-4*x1a[seg]*x2a[seg] + 2*x1a[seg]**2 + 2*x2a[seg]**2)**(-1)*(x1a[seg]*y1b[seg]*y2a[seg] + x1a[seg]*y2b[seg]*y1a[seg] - 2*x1a[seg]*y2b[seg]*y2a[seg] - 2*x2a[seg]*y1b[seg]*y1a[seg] + x2a[seg]*y1b[seg]*y2a[seg] + x2a[seg]*y2b[seg]*y1a[seg])*x1a[seg]**2 + (-4*x1a[seg]*x2a[seg] + 2*x1a[seg]**2 + 2*x2a[seg]**2)**(-1)*(x1a[seg]*y1b[seg]*y2a[seg] + x1a[seg]*y2b[seg]*y1a[seg] - 2*x1a[seg]*y2b[seg]*y2a[seg] - 2*x2a[seg]*y1b[seg]*y1a[seg] + x2a[seg]*y1b[seg]*y2a[seg] + x2a[seg]*y2b[seg]*y1a[seg])*x2a[seg]**2 - (-2*x1a[seg]*x2a[seg] + x1a[seg]**2 + x2a[seg]**2)**(-1)*(-x1a[seg]*x2a[seg]*y1b[seg]*y2a[seg] - x1a[seg]*x2a[seg]*y2b[seg]*y1a[seg] + x1a[seg]**2*y2b[seg]*y2a[seg] + x2a[seg]**2*y1b[seg]*y1a[seg])*x1a[seg] + (-2*x1a[seg]*x2a[seg] + x1a[seg]**2 + x2a[seg]**2)**(-1)*(-x1a[seg]*x2a[seg]*y1b[seg]*y2a[seg] - x1a[seg]*x2a[seg]*y2b[seg]*y1a[seg] + x1a[seg]**2*y2b[seg]*y2a[seg] + x2a[seg]**2*y1b[seg]*y1a[seg])*x2a[seg]
        for seg in segment_xj_only[i]:
            A[i] += -(-6*x1a[seg]*x2a[seg] + 3*x1a[seg]**2 + 3*x2a[seg]**2)**(-1)*(y1b[seg]*y1a[seg] - y1b[seg]*y2a[seg] - y2b[seg]*y1a[seg] + y2b[seg]*y2a[seg])*x1a[seg]**3 + (-6*x1a[seg]*x2a[seg] + 3*x1a[seg]**2 + 3*x2a[seg]**2)**(-1)*(y1b[seg]*y1a[seg] - y1b[seg]*y2a[seg] - y2b[seg]*y1a[seg] + y2b[seg]*y2a[seg])*xj[i]**3 - (-4*x1a[seg]*x2a[seg] + 2*x1a[seg]**2 + 2*x2a[seg]**2)**(-1)*(x1a[seg]*y1b[seg]*y2a[seg] + x1a[seg]*y2b[seg]*y1a[seg] - 2*x1a[seg]*y2b[seg]*y2a[seg] - 2*x2a[seg]*y1b[seg]*y1a[seg] + x2a[seg]*y1b[seg]*y2a[seg] + x2a[seg]*y2b[seg]*y1a[seg])*x1a[seg]**2 + (-4*x1a[seg]*x2a[seg] + 2*x1a[seg]**2 + 2*x2a[seg]**2)**(-1)*(x1a[seg]*y1b[seg]*y2a[seg] + x1a[seg]*y2b[seg]*y1a[seg] - 2*x1a[seg]*y2b[seg]*y2a[seg] - 2*x2a[seg]*y1b[seg]*y1a[seg] + x2a[seg]*y1b[seg]*y2a[seg] + x2a[seg]*y2b[seg]*y1a[seg])*xj[i]**2 - (-2*x1a[seg]*x2a[seg] + x1a[seg]**2 + x2a[seg]**2)**(-1)*(-x1a[seg]*x2a[seg]*y1b[seg]*y2a[seg] - x1a[seg]*x2a[seg]*y2b[seg]*y1a[seg] + x1a[seg]**2*y2b[seg]*y2a[seg] + x2a[seg]**2*y1b[seg]*y1a[seg])*x1a[seg] + (-2*x1a[seg]*x2a[seg] + x1a[seg]**2 + x2a[seg]**2)**(-1)*(-x1a[seg]*x2a[seg]*y1b[seg]*y2a[seg] - x1a[seg]*x2a[seg]*y2b[seg]*y1a[seg] + x1a[seg]**2*y2b[seg]*y2a[seg] + x2a[seg]**2*y1b[seg]*y1a[seg])*xj[i]

    return A


def pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between_super(a,b,c, xai,xbi,xbj, **kwargs):
    """wrapper for xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between to allow PolyLine input.

    Notes
    -----
    `a` and `b` can be lists that will be superposed.  This is not available in
    the original function

    See also
    --------
    xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between
    pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between : similar polyline wrapper but no superposition

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

def pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(a,b,c, xai,xbi,xbj, **kwargs):
    """wrapper for xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between to allow PolyLine input

    See also
    --------
    xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between

    """

    b, c = polyline_make_x_common(b, c)
    return xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(a.x,a.y,b.x1,b.x2,b.y1,b.y2, c.x1, c.x2, c.y1, c.y2, xai,xbi,xbj, **kwargs)



def xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(xa,ya,x1b,x2b,y1b,y2b, x1c, x2c, y1c, y2c, xai,xbi,xbj, achoose_max=False):
    """interpolate the xa_ya part at xai, and integrate the x1b_x2b_y1b_y2b * x1c_x2c_y1c_y2c part between xbi, and xbj of f(a, b, c) which is defined as g(a)*h(b)*h(c) where g(a) is defined with x_y data and h(b) and h(c) is defined by x1_x2_y1_y2 data

    Does little calculation, mostly calls other functions
    calculates array A[len(xbi), len(xai)]
    Parameters
    ----------
    xa, ya : 1d array_like, float
        x and y values of x_y part of interpolation function
    x1b, y1b : array_like, float
        x and y values at start of each segment 1st x1_x2_y1_y2 part of
        function
    x2b, y2b : array_like, float
        x and y values at end of each segment for 1st x1_x2_y1_y2 part of
        function  (note x1[1:]==x2[:-1])
    x1c, y1c : array_like, float
        x and y values at start of each segment 2nd x1_x2_y1_y2 part of
        function
    x2c, y2c : array_like, float
        x and y values at end of each segment for 2nd x1_x2_y1_y2 part of
        function  (note x1[1:]==x2[:-1])
    xai, array_like, float
        x values to interpolate the xc_yc part at
    xbi, xbj : array_like, float
        x values to integrate the x1b_x2b_y1b_y2b * x1c_x2c_y1c_y2c part between
    achoose_max : ``boolean``, optional
        if False (default), if xai falls on boundary of segments choose the
        minimum segment to interpolate within.

    See also
    --------


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
    """wrapper for xa_ya_cos_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between to allow PolyLine input.

    Parameters
    ----------
    omega_phase: 2 element tuple
        (omega, phase) for use in cos(omega * t + phase)

    Notes
    -----
    `a` and `b` `omega_phase` can be lists that will be superposed.  This is not available in
    the original function

    See also
    --------
    xa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between
    pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between : similar polyline wrapper but no superposition

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
    """wrapper for xa_ya_cos_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between to allow PolyLine input

    See also
    --------
    xa_ya_cos_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between

    """

    b, c = polyline_make_x_common(b, c)
    return xa_ya_cos_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(a.x,a.y,omega, phase, b.x1,b.x2,b.y1,b.y2, c.x1, c.x2, c.y1, c.y2, xai,xbi,xbj, **kwargs)

def xa_ya_cos_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(xa, ya, omega, phase, x1b,x2b,y1b,y2b, x1c, x2c, y1c, y2c, xai,xbi,xbj, achoose_max=False):
    """interpolate the xa_ya * cos(omega*t + phase) part at xai, and integrate the x1b_x2b_y1b_y2b * x1c_x2c_y1c_y2c part between xbi, and xbj of f(a, b, c) which is defined as g(a)*h(b)*h(c) where g(a) is defined with x_y data and h(b) and h(c) is defined by x1_x2_y1_y2 data

    Does little calculation, mostly calls other functions
    calculates array A[len(xbi), len(xai)]
    Parameters
    ----------
    xa, ya : 1d array_like, float
        x and y values of x_y part of interpolation function
    omega, phase:
        values for defining cos(omega * t + phase)
    x1b, y1b : array_like, float
        x and y values at start of each segment 1st x1_x2_y1_y2 part of
        function
    x2b, y2b : array_like, float
        x and y values at end of each segment for 1st x1_x2_y1_y2 part of
        function  (note x1[1:]==x2[:-1])
    x1c, y1c : array_like, float
        x and y values at start of each segment 2nd x1_x2_y1_y2 part of
        function
    x2c, y2c : array_like, float
        x and y values at end of each segment for 2nd x1_x2_y1_y2 part of
        function  (note x1[1:]==x2[:-1])
    xai, array_like, float
        x values to interpolate the xc_yc part at
    xbi, xbj : array_like, float
        x values to integrate the x1b_x2b_y1b_y2b * x1c_x2c_y1c_y2c part between
    achoose_max : ``boolean``, optional
        if False (default), if xai falls on boundary of segments choose the
        minimum segment to interpolate within.

    See also
    --------


    """

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
      start and end of each interval and yvalues at the start and end of each
      interval.  This can be useful if you want to work with layer/interval
      data but to plot those intervals you want x-y points.
    - Multiply a PolyLine by a scalar and only the y values will be changed
    - Add to/from a PolyLine with a scalar and only the y values will be
      changed.
    - PolyLines can be added together to create a new PolyLine; the x values
      of each PolyLine will be maintained.  Any x values that are not common
      to both PolyLines will have their y values interpolated and then added.

    Attributes
    ---------
    xy : 2d numpy array
        n by 2 array containing x and y values for each of the n points i.e.
        [[x0, y0], [x1, y1], ..., [xn, yn]]
    x, y : 1d numpy array
        arrays containing all the x values  and all the y values in the
        PolyLine
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
        the x1, x2, y1, y2 arrays returned in a tuple.
    atol, rtol : float
        absolute and relative tolerance when comparing equality of points in a
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
            raise TypeError('%d args given; Line can only be initialized with '
                '1, 2, or 4 args: 1 (x-y coords), 2 (x-coords, y-coords), or '
                '4 (x1-coords, x2-coords, y1-coords, y2-coords)' % len(args))
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
                    '(%d, %d).' % self._xy.shape)
            return

        if len(args)==2:
            #args[0] and args[1] are 1d arrays of n x and y data points
            if ((len(args[0]) != len(args[1])) or
                (len(args[0]) < 2) or
                (len(args[1]) < 2)):
                raise TypeError('x and y must be of same length with at least two values. len(x)=%d, len(y)=%d' % (len(args[0]), len(args[1])))

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
                    'You have lengths [%d, %d, %d, %d]' % tuple(map(len, args)))

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
        return "PolyLine(%s)" % (str(self.xy))
#    def __repr__(self):
#        """A string repr of the PolyLine that will recreate the Ployline"""
#        return "PolyLine(%s%s)" % (self._prefix_for_numpy_array_repr,
#                                    repr(self.xy))
    def __repr__(self):
        """A string repr of the PolyLine that will recreate the Ployline"""
        return "PolyLine(%s)" % (repr(self.xy))

    def __add__(self, other):
        return self._add_substract(other, op = operator.add)
    def __radd__(self, other):
        return self._add_substract(other, op = operator.add)

    def __sub__(self, other):
        return self._add_substract(other, op = operator.sub)
    def __rsub__(self, other):
        return self._add_substract(other, op = operator.sub).__mul__(-1)

    def __mul__(self, other):
        if isinstance(other, PolyLine):
            raise TypeError('cannot multiply two PolyLines together.  You will get a quadratic that I cannot handle')
            sys.exit(0)
        try:
            return PolyLine(self.x, self.y * other)
#            a = copy.deepcopy(self)
#            a.xy #ensure xy has been initialized
#            a._xy[:,1] *= other
        except TypeError:
            print("unsupported operand type(s) for *: 'PolyLine' and '%s'" % other.__class__.__name__)
            sys.exit(0)
#        return a
    def __rmul__(self,other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, PolyLine):
            raise TypeError('cannot divide two PolyLines together.  You will get a quadratic that I cannot handle')
            sys.exit(0)
        try:
            return PolyLine(self.x, self.y / other)
#            a = copy.deepcopy(self)
#            a.xy #ensure xy has been initialized
#            a._xy[:,1] /= other
        except TypeError:
            print("unsupported operand type(s) for /: 'PolyLine' and '%s'" % other.__class__.__name__)
            sys.exit(0)
#        return a
    def __rtruediv__(self, other):
        if isinstance(other, PolyLine):
            raise TypeError('cannot divide two PolyLines together.  You will get a quadratic that I cannot handle')
            sys.exit(0)
        try:
            return PolyLine(self.x, other / self.y)
#            a = copy.deepcopy(self)
#            a.xy #ensure xy has been initialized
#            a._xy[:,1] = other/a._xy[:,1]
        except TypeError:
            print("unsupported operand type(s) for /: 'PolyLine' and '%s'" % other.__class__.__name__)
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
        """addition or subtraction"""

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
            print("unsupported operand type(s) for +: 'PolyLine' and '%s'" % other.__class__.__name__)
#            sys.exit(0)
        return a




def polyline_make_x_common(*p_lines):
    """add points to multiple PolyLine's so that each have matching x1_x2 intevals

    Parameters
    ----------
    p_lines: PloyLine
        one or more instances of PolyLine

    Returns
    -------
    out : tuple of PolyLine
        same number of Polyline's as p_lines

    """

    xa=[]
    ya=[]

    for i, line in enumerate(p_lines):
        if not isinstance(line, PolyLine):
            raise TypeError("p_lines[%d] is not a PolyLine" % i)
            sys.exit(0)
        if not (non_increasing(line.x) or non_decreasing(line.x)):
                raise TypeError('PolyLine #%d has switchbacks.' % i)
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
    """subdivide each line segment into subsegments

    subsegements are evenly spaced in linear or log space

    Parameters
    ----------
    x : 1d array-like
        list of xvalues to subdivide
    y : 1d array-like
        list of y values to subdivide (based on x values)
        no y values
    dx : float, optional
        approxmate log10(length) of subsegment.  Say segment is from 1-10 units
        and dx=0.2 then int((log(10)-log(1))/0.2)=5 subsegments
        (i.e. 5 extra points will be inserted in thh segement) will be
        created at log(1+0.2), log(1+0.4), log(1+0.6) etc.
        default = None i.e. no dx check use min_segments.
    min_segments : int, optional
        minuimum number of subsegments per inteval. this will be used if
        int(segment_lenght/dx)<min_segments. default = 2
    just_before : float, optional
        If just_before is not None then in terms of subdividing each
        segment will be treated as if it begins at its starting value but
        ends a distance of `just_before` multiplied by the interval length
        before its end value. This
        means that an extra point will be added just before all of the
        original points. default = None i.e dont use just before point.
        Use a small number e.g. 1e-6.
        `just_before` can be useful for example when getting
        times to evaluate pore pressure at in a soil consoliation
        analysis.  Say you have a PolyLine representing load_vs_time. If
        there are step changes in your load then there will be step
        changes in your pore pressure.  To capture the step change in your
        output you need an output time just before and just after the step
        change in your load.  Using `just_before` can achieve this.
    logx, logy: True/False, optional
        use log scale on x and y axes.  default = False
    logxzero, logyzero: float, optional
        if log scale is used force zero value to be given number.
        Useful when 1st point iszero but you really want to start from close
        to zero. default = 0.01
    rtol, atol: float, optional
        for determining equal to zero when using log scale with numpy.
        default atol = 1e-8 , rtol = 1e-5
    Returns
    -------
    xnew, ynew : 1d array
        new x and y coordinates

    """


    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)


    if len(x)!=len(y):
        raise (ValueError('x and y must have same length '
                     'len(x)=%d, len(y)=%d' % (len(x), len(y))))
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

def subdivide_x_into_segments(x, dx=None, min_segments = 2,
        just_before = None, logx=False, logxzero=0.1,
        rtol=1e-5, atol=1e-8):
    """subdivide each inteval into subsegments

    Intervals are evenly spaced in linear or log space

    Parameters
    ----------
    x : 1d array-like
        list of xvalues to subdivide
    dx : float, optional
        approxmate log10(length) of subsegment.  Say segment is from 1-10 units
        and dx=0.2 then int((log(10)-log(1))/0.2)=5 subsegments
        (i.e. 5 extra points will be inserted in thh segement) will be
        created at log(1+0.2), log(1+0.4), log(1+0.6) etc.
        default = None i.e. no dx check use min_segments.
    min_segments : int, optional
        minuimum number of subsegments per inteval. this will be used if
        int(segment_lenght/dx)<min_segments. default = 2
    just_before : float, optional
        If just_before is not None then in terms of subdividing each
        segment will be treated as if it begins at its starting value but
        ends a distance of `just_before` multiplied by the interval length
        before its end value. This
        means that an extra point will be added just before all of the
        original points. default = None i.e dont use just before point.
        Use a small number e.g. 1e-6.
        `just_before` can be useful for example when getting
        times to evaluate pore pressure at in a soil consoliation
        analysis.  Say you have a PolyLine representing load_vs_time. If
        there are step changes in your load then there will be step
        changes in your pore pressure.  To capture the step change in your
        output you need an output time just before and just after the step
        change in your load.  Using `just_before` can achieve this.
    logx: True/False, optional
        use log scale on x axis.  default = False
    logxzero: float, optional
        if log scale is used force zero value to be given number.
        Useful when 1st point iszero but you really want to start from close
        to zero. default = 0.01
    rtol, atol: float, optional
        for determining equal to zero when using log scale with numpy.
        default atol = 1e-8 , rtol = 1e-5
    Returns
    -------
    xnew: 1d array
        new x values

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


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])