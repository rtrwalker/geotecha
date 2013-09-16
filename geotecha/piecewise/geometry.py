# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 09:44:51 2013

@author: Rohan Walker
"""
from __future__ import division, print_function

import sympy
import matplotlib.pylab as plt
import numpy as np


def xyz_from_pts(pts, close_polygon = False):
    """ extract separate x coords, ycoords, z coords from an x,y,z coords
    
    Parameters
    ----------
    pts : array_like
        array of x, y or x, y, z points
    close_polygon : boolean
        if True then 1st point will be repeated i.e. to close the polygon
    
    Returns
    -------
    x,y,z : 1d ndarrays
        1d array of x coords, y coords, z coords
        
    """

    pts = np.asarray(pts)    
    
    if close_polygon:
        x = np.zeros(len(pts)+1, dtype=float)
        y = np.zeros(len(pts)+1, dtype=float)
        z = np.zeros(len(pts)+1, dtype=float)

        x[:-1] = pts[:,0]
        x[-1] = x[0]
        
        y[:-1] = pts[:,1]
        y[-1] = y[0]
        if len(pts[0]) == 3:            
            z[:-1] = pts[:,2]
            z[-1] = z[0]
        
    else:
        x = pts[:,0]
        y = pts[:,1]
        
        if len(pts[0]) == 3:
            z = pts[:,2]
        else:        
            z = np.zeros_like(x)
            
    return x, y, z
    
def eqn_of_plane(pts):
    """Equation of plane defined by polygon points
    
    a,b,c,d from a*x+b*y+c*z+d = 0
    looking at the plane, counter-clockwise (CCW) points the positive normal 
    is towards you.  For clockwise points (CW) the positive normal is away 
    from you.
    
    Parameters
    ----------
    pts : array_like
        array of x, y or x, y, z points
        
    
    Returns
    -------
    [a,b,c] : ndarray with 3 elemetns
        direction cosines of normal to plane
    d : float
        constant in plane equation
        
    """
    
    pts = np.asarray(pts)    
    x, y, z = xyz_from_pts(pts,True)    
    a = np.sum((y[:-1] - y[1:]) * (z[:-1] + z[1:]))
    b = np.sum((z[:-1] - z[1:]) * (x[:-1] + x[1:]))
    c = np.sum((x[:-1] - x[1:]) * (y[:-1] + y[1:]))
    
    n = np.array([a, b, c], dtype=float )    
    n /= np.linalg.norm(n)        
    p = np.array([np.sum(x[:-1]), np.sum(y[:-1]), np.sum(z[:-1])]) / len(pts)
    d = -np.dot(p,n)    
    return n, d
    
    
    
if __name__ == "__main__":    
    shp = dict()
    shp['unit square'] = [[0,0],[1,0],[1,1],[0,1]]
    shp['right tri'] = [[0,0],[1,0],[0,1]]    
    shp['octahedral tri'] = [[1,0,0],[0,1,0],[0,0,1]]
    shp['3D tri'] = [[1,-2,0],[3,1,4],[0,-1,2]]
    print(eqn_of_plane(shp['unit square']))
    print(eqn_of_plane(shp['octahedral tri']))
    print(eqn_of_plane(shp['3D tri']))
    
    
    

                                