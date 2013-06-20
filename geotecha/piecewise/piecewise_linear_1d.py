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
import operator
import numbers

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
    
    if isinstance(xi, numbers.Number):    
        xi = np.array([xi])        
    xi = np.asarray(xi)
    
    if subset is None:
        subset = np.arange(len(x)-1)       
    if len(subset)==0: #subset isempty
        return [np.array([],dtype = int) for v in xi]        
        
    subset = np.asarray(subset)

    A = [subset[(v>=x[subset]) & (v<=x[subset+1]) & (x[subset]!=x[subset+1])] for v in xi]
        
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
    
    if isinstance(xi, numbers.Number):    
        xi = np.array([xi])        
    xi = np.asarray(xi)
    
    
                        
    
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
    
    if isinstance(xi, numbers.Number):    
        xi = np.array([xi])        
    xi = np.asarray(xi)
    
    
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
    segment_both: list of single element lists
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
            np.insert(segment_both[i],0,i1[0])
        else:
            np.insert(segment_xi_only[i],0,i1[0])
            np.insert(segment_xj_only[i],0,i2[0])
            segments_between[i]=np.r_[i1[0]+1:i2[0]]
            
    return (segment_both, segment_xi_only, segment_xj_only, segments_between)

    
if __name__ == '__main__':
    #print(strictly_increasing([0,  0.5,  1,  1.5,  2]))
    #print(non_increasing_and_non_decreasing_parts([0,  0.5,  1,  1.5,  2]))
    #print (force_strictly_increasing([0, 0.5, 1, 0.75, 1.5, 2], keep_end_points=True))
   # print(force_strictly_increasing([0,  0.4,   0.4,  1,  2.5,  3,  3], keep_end_points = True, eps=0.01) )
    #print(force_strictly_increasing([0,  0.4,   0.4,  1,  2.5,  3,  3], keep_end_points = False, eps=0.01) )
    #print (force_strictly_increasing([0,  0.4,   0.4,  1,  2.5,  3,  3]))
    #a = {'x': [0,  0,  1,  1,  2],                          'y': [0, 10, 10, 30, 30]}
    a = {'x': [0,  0.4,  1,  1.5,  2],                          'y': [0, 10, 10, 30, 30]}
#    print(ramps_constants_steps(**a))
    #print (start_index_of_ramps(**a))   
#    print (segment_containing_xi(a['x'], [-1,0,0.2,1,1.7,8]))
#    print (segment_containing_xi(a['x'], [-1,0,0.2,1,1.7,8], choose_max = True))
#    print(segment_containing_xi(x=[0,  0.4,   0.4,  1,  2.5,  3,  3], xi=[0,1.2], subset=None, choose_max=True))
    #print (segment_containing_xi(a['x'], [-1,0,0.2,1,1.7]))
    two_ramps_two_steps = {'x': [0,  0.4,   0.4,  1,  2.5,  3,  3],
                                    'y': [0,  10.0, 20.0, 20, 30.0, 30, 40]}
    two_steps = {'x': [0,  0,  1,  1,  2],
                          'y': [0, 10, 10, 30, 30]}                                    
    #print(segment_containing_also_segments_less_than_xi(xi = 0.42, **two_ramps_two_steps))
    #(ramps_less_than_xi, constants_less_than_xi, steps_less_than_xi, 
    #        ramps_containing_xi, constants_containing_xi) = segment_containing_also_segments_less_than_xi(xi = 0.42, **two_ramps_two_steps)        
    #all(map(np.allclose, ramps_less_than_xi, [3]))
    #all(map(np.allclose, constants_less_than_xi, [4]))
    #all(map(np.allclose, steps_less_than_xi, [5]))
    #print(ramps_less_than_xi)    
    #print(ramps_containing_xi)    
    #print(all(map(np.allclose, ramps_containing_xi, [[]])))
    #all(map(np.allclose, constants_containing_xi, [2]))
    #print(segments_less_than_xi(x=two_steps['x'], xi= 0, subset=None, or_equal_to=False))
    #print (segment_containing_xi(a['x'], 4))
    dic = {'x': np.array([0,0,10]), 'y': np.array([0, -100,-100]), 'xi': np.array([-1,0,1])}
    print (segment_containing_also_segments_less_than_xi(**dic))
#    ramps, constatnts, steps = ramps_constants_steps([0,0],[0,10])
#    print(ramps)
#    
#    print(segment_containing_xi([0,0],2,[]))
#    print(segment_containing_xi(x=[0,  0,  1,  1,  2], xi=[-1,0,0.2, 1, 2, 3.4], subset=None, choose_max=True))
    #print(segment_containing_xi(x=[0,  0.4,   0.4,  1,  2.5,  3,  3], xi=[0], subset=[0,3]))
    #print(segment_containing_xi(x=[0,  0.4,   0.4,  1,  2.5,  3,  3], xi=[0.4], choose_max=True))
    #print(segments_less_than_xi(x=[0,  0,  1,  1,  2], xi=-1, subset=None, or_equal_to=False))
    #print(segments_less_than_xi(x=[0,  0.4,   0.4,  1,  2.5,  3,  3], xi=0.42, subset=[2,4], or_equal_to=False))
#import sys
#sys.float_info.epsilon


#def passes_vertical_line_test(x, y):
