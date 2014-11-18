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
"""Some routines loosely related to geometry."""


from __future__ import division, print_function
import numpy as np
import sympy
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


def xyz_from_pts(pts, close_polygon = False):
    """Extract x, y and z values from an (n, 3) or (n, 2) shaped array

    Parameters
    ----------
    pts : array_like
        Array of x, y or x, y, z points.
    close_polygon : boolean, optional
        If True then 1st point will be repeated i.e. to close the polygon.
        Default close_polygon=False.


    Returns
    -------
    x, y, z : 1d ndarrays
        1d array of x coords, y coords, z coords.

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

    a, b, c, d from a*x+b*y+c*z+d = 0
    Looking at the plane, counter-clockwise (CCW) points the positive normal
    is towards you.  For clockwise points (CW) the positive normal is away
    from you.

    Parameters
    ----------
    pts : array_like
        aAray of x, y or x, y, z points.


    Returns
    -------
    [a, b, c] : ndarray with 3 elements
        Direction cosines of normal to plane.
    d : float
        Constant in plane equation.

    """

    pts = np.asarray(pts)
    x, y, z = xyz_from_pts(pts, True)
    a = np.sum((y[:-1] - y[1:]) * (z[:-1] + z[1:]))
    b = np.sum((z[:-1] - z[1:]) * (x[:-1] + x[1:]))
    c = np.sum((x[:-1] - x[1:]) * (y[:-1] + y[1:]))

    n = np.array([a, b, c], dtype=float )
    n /= np.linalg.norm(n)
    p = np.array([np.sum(x[:-1]), np.sum(y[:-1]), np.sum(z[:-1])]) / len(pts)
    d = -np.dot(p,n)
    return n, d


def replace_x0_and_x1_with_vect(s,xyz = ['x','y','z']):
    """Replaces strings x0 with x[:-1],  x1 with x[1:], y0 with y[:-1] ...

    Parameters
    ----------
    s : string, or sympy_expression
        Expression to make replacements on
    xyz : list of strings
        Replacements will be made for each element of xyz.
        default xyz=['x','y','z'].

    Returns
    -------
    out : string
        string with all replacements made

    """

    endings = ['[:-1]', '[1:]']
    s = str(s)
    for x in xyz:
        for i, ending in enumerate(endings):
            s = s.replace('{}{:d}'.format(x,i), '{}{}'.format(x,ending))
    return s

def integrate_f_over_polygon_code(f):
    """Generate code that will integrate a function over a polygon

    Parameters
    ----------
    f : sympy expression
        Expression to be integrated over the polygon.

    Returns
    -------
    out : str
        Multiline string of function code

    """

    x, y, z=sympy.symbols('x,y,z')
    x0, x1, y0, y1, z0, z1=sympy.symbols('x0,x1,y0,y1,z0,z1')
    t = sympy.symbols('t')
    s2 = [(x, x0+t*(x1-x0)), (y, y0+t*(y1-y0)), (z, z0+t*(z1-z0))]

    ff = sympy.integrate(f,x)
    ff = ff.subs(s2)
    ff = sympy.integrate(ff, (t,0,1))
    ff *= (y1 - y0) #if integrating in y direction 1st then *-(x1-x0)

    ff = replace_x0_and_x1_with_vect(ff)

    template ="""def ifxy(pts):
    "Integrate f = {} over polygon"

    x, y, z = xyz_from_pts(pts, True)

    return np.sum({})"""

    return template.format(str(f), ff)

def integrate_f_over_polyhedra_code(f):
    """Generate code that will integrate a function over a polyhedra

    Parameters
    ----------
    f : sympy expression
        Expression to be integrated over the polyhedron.

    Returns
    -------
    out : str
        Multiline string of funciton code.

    """
    x,y,z=sympy.symbols('x,y,z')
    x0,x1,y0,y1,z0,z1=sympy.symbols('x0,x1,y0,y1,z0,z1')
    n1,n2,n3,d = sympy.symbols('n1,n2,n3,d')

    t = sympy.symbols('t')
    s1 = [(x,-n2/n1*y-n3/n1*z-d/n1)] #for integration order x,y,z *(z1-z0), for x,z,y *-(y1-y0), ignore n1==0
    #s1 = [(y,-n1/n2*x-n3/n2*z-d/n2)] #for integration order y,x,z *-(z1-z0), for y,z,x *(x1-x0), ignore n2==0
    #s1 = [(z,-n1/n3*x-n2/n3*y-d/n3)] #for integration order z,x,y *(y1-y0), for z,y,x *-(x1-x0), ignore n3==0
    s2 = [(x,x0+t*(x1-x0)),(y,y0+t*(y1-y0)),(z,z0+t*(z1-z0))]

    ff = sympy.integrate(f, x)
    ff = ff.subs(s1)
    ff = sympy.integrate(ff, y)
    ff = ff.subs(s2)
    ff = sympy.integrate(ff, (t,0,1))
    ff *= z1 - z0

    ff = replace_x0_and_x1_with_vect(ff)

    template ="""def ifxyz(faces):
    "Integrate f = {} over polyhedron"

    x, y, z = xyz_from_pts(pts,True)
    igral = 0
    for pts in faces:
        (n1,n2,n3),d = eqn_of_plane(pts)
        if n1==0:
            continue
        igral += np.sum({})

    return igral"""

    return template.format(str(f), ff)

def polygon_area(pts):
    """Area of polygon defined by points

    Parameters
    ----------
    pts : array_like
        Array of x, y or x, y, z points.

    Returns
    -------
    a : float
        Area of polygon.

    """

    pts = np.asarray(pts)
    x, y, z = xyz_from_pts(pts,True)
    n, d = eqn_of_plane(pts)

    i = np.argmax(n) #project polygon onto plane that is perpendicular to this direction, then integrate the area

    j = range(3)
    j.pop(i) #remaining directions
    e = (x, y, z)
    def if1dxdy(x, y):
        """integrate f=1 over polygon of x,y points"""
        #generate code using integrate_f_over_polygon_code(1)

        return np.sum((x[:-1]/2 + x[1:]/2)*(-y[:-1] + y[1:]))

    return if1dxdy(e[j[0]], e[j[1]]) / n[i]


def polygon_centroid(pts):
    """Centroid of polygon defined by points

    Parameters
    ----------
    pts : array_like
        Array of x, y or x, y, z points.

    Returns
    -------
    [xc, yc, zc] : ndarray of float
        Coordinates of centroid.

    """

    def ifxdxdy(x,y):
        """integrate f=x over polygon of x,y points"""
        #generate code using integrate_f_over_polygon_code(x)
        return np.sum((-y[:-1] + y[1:])*(x[:-1]**2/6 + x[:-1]*x[1:]/6 + x[1:]**2/6))
    def ifydxdy(x,y):
        """integrate f=y over polygon of x,y points"""
        #generate code using integrate_f_over_polygon_code(y)
        return np.sum((-y[:-1] + y[1:])*(x[:-1]*y[:-1]/3 + x[:-1]*y[1:]/6 +
            x[1:]*y[:-1]/6 + x[1:]*y[1:]/3))


    pts = np.asarray(pts)
    x, y, z = xyz_from_pts(pts, True)
    n, d = eqn_of_plane(pts)
    a = polygon_area(pts)


    xc=0.0
    yc=0.0
    zc=0.0

    calcd = [False,False,False]


    if n[0]!=0:#project on yz plane
        if not calcd[1]:
            yc = ifxdxdy(y, z)/ (n[0] * a)
            calcd[1]= True
        if not calcd[2]:
            zc = ifydxdy(y, z)/ (n[0] * a)
            calcd[2]= True
    if n[1]!=0: #project on zx plane
        if not calcd[2]:
            zc = ifxdxdy(z, x)/ (n[1] * a)
            calcd[2]= True
        if not calcd[0]:
            xc = ifydxdy(z, x)/ (n[1] * a)
            calcd[0]= True
    if n[2]!=0: #project on xy plane
        if not calcd[0]:
            xc = ifxdxdy(x, y)/ (n[2] * a)
            calcd[0]= True
        if not calcd[1]:
            yc = ifydxdy(x, y)/ (n[2] * a)
            calcd[1]= True

    return np.array([xc, yc, zc])


def polygon_2nd_moment_of_area(pts):
    """2nd moment of area of polygon defined by points

    Parameters
    ----------
    pts : array_like
        Array of x, y or x, y, z points.

    Returns
    -------
    [Ixx, Iyy, Izz] : ndarray of float
        2nd moment of area about centroidal x, y, and z axes.

    """

    def ifx2dxdy(x,y,xc):
        """Integrate f = (x - xc)**2 over polygon of x,y points"""
        #generate code using integrate_f_over_polygon_code((x-xc)**2)
        return np.sum((-y[:-1] + y[1:])*(x[:-1]**3/12 + x[:-1]**2*x[1:]/12 -
            x[:-1]**2*xc/3 + x[:-1]*x[1:]**2/12 - x[:-1]*x[1:]*xc/3 +
            x[:-1]*xc**2/2 + x[1:]**3/12 - x[1:]**2*xc/3 + x[1:]*xc**2/2))

    def ify2dxdy(x,y,yc):
        """integrate f = (y - yc)**2 over polygon of x,y points"""
        #generate code using integrate_f_over_polygon_code((y-yc)**2)
        return np.sum((-y[:-1] + y[1:])*(x[:-1]*y[:-1]**2/4 +
            x[:-1]*y[:-1]*y[1:]/6 - 2*x[:-1]*y[:-1]*yc/3 + x[:-1]*y[1:]**2/12 -
            x[:-1]*y[1:]*yc/3 + x[:-1]*yc**2/2 + x[1:]*y[:-1]**2/12 +
            x[1:]*y[:-1]*y[1:]/6 - x[1:]*y[:-1]*yc/3 + x[1:]*y[1:]**2/4 -
            2*x[1:]*y[1:]*yc/3 + x[1:]*yc**2/2))


    pts = np.asarray(pts)
    x, y, z = xyz_from_pts(pts, True)
    n, d = eqn_of_plane(pts)
    a = polygon_area(pts)

    xc,yc,zc = polygon_centroid(pts)

    ixx=0.0
    iyy=0.0
    izz=0.0

    calcd = [False, False, False]


    if n[0]!=0:#project on yz plane
        if not calcd[1]:
            iyy = ifx2dxdy(y, z, yc)/ (n[0] * a)
            calcd[1]= True
        if not calcd[2]:
            izz = ify2dxdy(y, z, zc)/ (n[0] * a)
            calcd[2]= True
    if n[1]!=0: #project on zx plane
        if not calcd[2]:
            izz = ifx2dxdy(z, x, zc)/ (n[1] * a)
            calcd[2]= True
        if not calcd[0]:
            ixx = ify2dxdy(z, x, xc)/ (n[1] * a)
            calcd[0]= True
    if n[2]!=0: #project on xy plane
        if not calcd[0]:
            ixx = ifx2dxdy(x, y, xc)/ (n[2] * a)
            calcd[0]= True
        if not calcd[1]:
            iyy = ify2dxdy(x, y, yc)/ (n[2] * a)
            calcd[1]= True

    return np.array([ixx, iyy, izz])

def polyhedron_volume(faces):
    """Volume of polyhedron defined by faces defined py pts

    Parameters
    ----------
    faces : list
        A list of pts arrays defining x,y,z coords of face vertices.


    Returns
    -------
    v : float
        Volume of polyhedron.

    Notes
    -----
    I think points on a face have to be defined in anti clockwise (CCW) order
    to give a positive volume.  No checks are done to check the order.


    """

    v = 0.0
    for pts in faces:
        pts = np.asarray(pts)
        x, y, z = xyz_from_pts(pts,True)
        (n1,n2,n3),d = eqn_of_plane(pts)
        if n1==0:
            continue
        #generate code with integrate_f_over_polyhedra_code(1)
        v += np.sum((-z[:-1] + z[1:])*((-2*d*y[:-1] - n2*y[:-1]**2 -
            2*n3*y[:-1]*z[:-1])/(2*n1) + (d*y[:-1] - d*y[1:] + n2*y[:-1]**2 -
            n2*y[:-1]*y[1:] + 2*n3*y[:-1]*z[:-1] - n3*y[:-1]*z[1:] -
            n3*y[1:]*z[:-1])/(2*n1) + (-n2*y[:-1]**2 + 2*n2*y[:-1]*y[1:] -
            n2*y[1:]**2 - 2*n3*y[:-1]*z[:-1] + 2*n3*y[:-1]*z[1:] +
            2*n3*y[1:]*z[:-1] - 2*n3*y[1:]*z[1:])/(6*n1)))



    return v

def make_hexahedron(coords):
    """Assemble the face vertices of a hexahedron


    Parameters
    ----------
    coords : 8 by 3 ndarray
        x, y, z coords of 8 corner nodes of hexahedron.  node numbering is as
        per Smith and Grifiths.

    Returns
    -------
    faces : list of pts arrays
        List of pts arrays.  Each list defines vertices of a face.

    Notes
    -----
    Node numbering:

    ::

                                x
            (6)-------(7)       ^
            /         /|        |  y
           /         / |        | /
          /         /  |        |/
        (2)-------(3) (8)       |-------->x
         |         |   /
         |         |  /
         |         | /
        (1)-------(4)



    """
    coords = np.asarray(coords)
    face_nodes=np.array([[1,4,3,2],[3,4,8,7],[8,5,6,7],[2,6,5,1],[6,2,3,7],[5,8,4,1]], dtype=int)
    face_nodes-=1
    return [coords[v,:] for v in face_nodes]

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])
#    a = [[-1,-1,-1],
#         [-1,-1, 1],
#        [1,-1, 1],
#        [1,-1,-1],
#        [-1,1,-1],
#        [-1,1,1],
#        [1,1,1],
#        [1,1,-1]]
#
#
#    b = [
#            [[0,0,0],
#            [0,0,1],
#            [1,0,0],
#            [0,0,0]],
#
#            [[1,0,0],
#            [0,0,1],
#            [0,1,0],
#            [1,0,0]],
#
#            [[0,1,0],
#            [0,0,1],
#            [0,0,0],
#            [0,1,0]],
#
#            [[0,0,0],
#            [1,0,0],
#            [0,1,0],
#            [0,0,0]]
#        ]
#    c = [[[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0]],
#         [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]],
#         [[0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0]],
#         [[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0]]]
##    for v in b:
##        print(v[::-1])
#    c = [v[::-1] for v in b]
#    print(c)
##    b= make_hexahedron(a)
##    print(b)
#    print(polyhedron_volume(c))



#    shp = dict()
#    shp['unit square'] = [[0,0],[1,0],[1,1],[0,1]]
#    shp['right tri'] = [[0,0],[1,0],[0,1]]
#    shp['3D tri'] = [[1,-2,0],[3,1,4],[0,-1,2]]
#    shp['2D tri'] = [[1,0],[3,4],[0,2]]
#    shp['octahedral tri'] = [[1,0,0],[0,1,0],[0,0,1]]
#    for shape,pts in shp.iteritems():
#        print(shape)
#        print(eqn_of_plane(pts))
#        print(polygon_area(pts))
#        print(polygon_centroid(pts))
#        print(polygon_2nd_moment_of_area(pts))
#        print()
#
#    #print(integrate_f_over_polygon_code(1))
#    x,y,z = sympy.symbols('x,y,z')
#    xc,yc,zc = sympy.symbols('xc,yc,zc')
##    print(integrate_f_over_polygon_code((x-xc)**2))
##    print(integrate_f_over_polygon_code((y-yc)**2))
##    print(integrate_f_over_polygon_code((z-zc)**2))
##
#
#    shp = dict()
#    vert = dict()
#    vert['2unit cube'] = [[-1,-1,-1], [-1,-1,1],[1,-1,1],[1,-1,-1],
#                           [-1,1,-1], [-1,1,1],[1,1,1], [1,1,-1]]
#    shp['2unit cube'] = make_hexahedron(vert['2unit cube'])
#    for shape,faces in shp.iteritems():
#        print(shape)
#        print(polyhedron_volume(faces))
#        print()
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        ax.scatter(np.array(vert[shape])[:,0], np.array(vert[shape])[:,1],np.array(vert[shape])[:,2])
#        ax.set_xlabel('X')
#        ax.set_ylabel('Y')
#        ax.set_zlabel('Z')
#        plt.show()

