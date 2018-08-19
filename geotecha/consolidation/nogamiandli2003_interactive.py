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

"""
This is really just way to interactively run a nomagimandli2003 analysis
without polluting the the if __name__ == '__main__' area of
the nogamiandli2003 module.


See Also
--------
geotecha.consolidation.nogamiandli2003.NogamiAndLi2003 : calculations

"""
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

import geotecha.piecewise.piecewise_linear_1d as pwise
from geotecha.piecewise.piecewise_linear_1d import PolyLine
import geotecha.mathematics.transformations as transformations
import textwrap

from geotecha.consolidation.nogamiandli2003 import NogamiAndLi2003



if __name__ == '__main__':


    my_code = textwrap.dedent("""\
#from geotecha.piecewise.piecewise_linear_1d import PolyLine
#import numpy as np
###################################################
#Schiffman and stein1970
#surcharge_vs_time = PolyLine([0,0,10], [0,100,100])
#h = np.array([10, 20, 30, 20])*1
#cv = np.array([0.0411, 0.1918, 0.0548, 0.0686])*1e8
#mv = np.array([3.07e-3, 1.95e-3, 9.74e-4, 1.95e-3])
##kv = np.array([7.89e-6, 2.34e-5, 3.33e-6, 8.35e-6])
#kv = cv*mv
##kh = np.array([  1.26177000e-04,   3.74010000e-04,   5.33752000e-05, 1.33770000e-04])*1e-4
###kh = kv * 0.0003
##r1 = 0.6
##r0= 0.05
#
#bctop = 0
#
#bcbot = 0
#
#
#nv = 17
##nh = 5
#
#
#z = np.concatenate((np.linspace(0, np.sum(h[:1]), 25, endpoint=False),
#                    np.linspace(np.sum(h[:1]), np.sum(h[:2]), 25, endpoint=False),
#                    np.linspace(np.sum(h[:2]), np.sum(h[:3]), 25, endpoint=False),
#                    np.linspace(np.sum(h[:3]), np.sum(h), 25, endpoint=True)))
#
#
#
#tpor = np.array([740, 2930, 7195], dtype=float)
#
#t = np.logspace(1, 4.5, 30)
#
#t = np.array(
#    [1.21957046e+02,   1.61026203e+02,   2.12611233e+02,
#     2.80721620e+02,   3.70651291e+02,   4.89390092e+02,
#     740.0,   8.53167852e+02,   1.12648169e+03,
#     1.48735211e+03,   1.96382800e+03,   2930.0,
#     3.42359796e+03,   4.52035366e+03,   5.96845700e+03,
#     7195.0,   1.04049831e+04,   1.37382380e+04,
#     1.81393069e+04,   2.39502662e+04,   3.16227766e+04])
################################################


##########################################################
#Figure 10 of Nogami and li 2003.
surcharge_vs_time = PolyLine([0,0,10], [0,100,100])

hs=0.05
h = np.array([1, hs, hs, 1, hs, hs, 0.5])
lam = 100 #5
kv = np.array([1,lam/hs, lam/hs, 1, lam/hs, lam/hs, 1])
mv = np.array([1, 1, 1, 1, 1, 1, 1])
kh = kv

r0 = 0.05
r1 = 20 * r0
#rcalc = r1

bctop = 0

bcbot = 1


nv = 15
nh = 5

z = np.linspace(0,np.sum(h),100)
tpor = np.array([0.01,0.1, 0.4])
t = np.linspace(0,0.5, 50)
t = np.logspace(-2,0.8, 10)


z = np.array(
       [ 0.    ,  0.0625,  0.125 ,  0.1875,  0.25  ,  0.3125,  0.375 ,
        0.4375,  0.5   ,  0.5625,  0.625 ,  0.6875,  0.75  ,  0.8125,
        0.875 ,  0.9375,  1.    ,  1.025 ,  1.05  ,  1.075 ,  1.1   ,
        1.1625,  1.225 ,  1.2875,  1.35  ,  1.4125,  1.475 ,  1.5375,
        1.6   ,  1.6625,  1.725 ,  1.7875,  1.85  ,  1.9125,  1.975 ,
        2.0375,  2.1   ,  2.125 ,  2.15  ,  2.175 ,  2.2   ,  2.2625,
        2.325 ,  2.3875,  2.45  ,  2.5125,  2.575 ,  2.6375,  2.6999   ])


t = np.array(
      [ 0.01      ,  0.01603286,  0.02570525,  0.04121285,  0.06607597,
        0.1, 0.10593866,  0.16984993,  0.27231794,  0.4, 0.43660343,  0.7 ])



max_iter=20000
#radial_roots_x0 = 1e-3
#radial_roots_dx = 1e-3
#radial_roots_p = 1.05
if lam in [100,5,0]:
    vertical_roots_x0 = {100:3, 5: 2.2}[lam]
    vertical_roots_dx = {100:1e-3, 5: 1e-3}[lam]
else:
    vertical_roots_x0 = 1e-7
    vertical_roots_dx = 1e-3
vertical_roots_p = 1.01

directory= r"C:\\Users\\Rohan Walker\\Documents\\temp" #may always need the r
save_data_to_file= True
save_figures_to_file= True
show_figures= True
overwrite=True

#prefix="silly"

#create_directory=True
#data_ext = '.csv'
#input_ext='.py'
figure_ext='.png'
show_vert_eigs=True
    """)

    a = NogamiAndLi2003(my_code)

    a.make_all()

#    print(repr(a.por))
#    print("")
#    print(repr(a.avp))
#    print("")
#    print(repr(a.set))
#    print("")







