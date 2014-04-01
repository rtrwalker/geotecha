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
This is really just way to interactively run a nomagimandli2003 analysis
without polluting the the if __name__ == '__main__' area of
geotecha.consolidation.nogamiandli2003

"""
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

import geotecha.piecewise.piecewise_linear_1d as pwise
from geotecha.piecewise.piecewise_linear_1d import PolyLine
import geotecha.math.transformations as transformations
import textwrap

from geotecha.consolidation.schiffmanandstein1970 import SchiffmanAndStein1970



if __name__ == '__main__':


    my_code = textwrap.dedent("""\
#from geotecha.piecewise.piecewise_linear_1d import PolyLine
#import numpy as np

h = np.array([10, 20, 30, 20])
cv = np.array([0.0411, 0.1918, 0.0548, 0.0686])
mv = np.array([3.07e-3, 1.95e-3, 9.74e-4, 1.95e-3])
#kv = np.array([7.89e-6, 2.34e-5, 3.33e-6, 8.35e-6])
kv = cv*mv

bctop = 0
#htop = None
#ktop = None
bcbot = 0
#hbot = None
#kbot = None

n = 40
surcharge_vs_time = PolyLine([0,0,10], [0,100,100])
z = np.concatenate((np.linspace(0, np.sum(h[:1]), 10, endpoint=False),
                    np.linspace(np.sum(h[:1]), np.sum(h[:2]), 10, endpoint=False),
                    np.linspace(np.sum(h[:2]), np.sum(h[:3]), 10, endpoint=False),
                    np.linspace(np.sum(h[:3]), np.sum(h), 10, endpoint=True)))



tpor = np.array([740, 2930, 7195], dtype=float)

t = np.logspace(1, 4.5, 30)

t = np.array(
    [1.21957046e+02,   1.61026203e+02,   2.12611233e+02,
     2.80721620e+02,   3.70651291e+02,   4.89390092e+02,
     740.0,   8.53167852e+02,   1.12648169e+03,
     1.48735211e+03,   1.96382800e+03,   2930.0,
     3.42359796e+03,   4.52035366e+03,   5.96845700e+03,
     7195.0,   1.04049831e+04,   1.37382380e+04,
     1.81393069e+04,   2.39502662e+04,   3.16227766e+04])

#z = np.linspace(0.0, np.sum(h), 100)

#
#z=np.array([0,10])
#t = np.linspace(0,10000, 100)
##t=np.array([1])
#
#h=np.array([1])
#kv=np.array([1])
#mv=np.array([1])
#surcharge_vs_time = PolyLine([0,0,8], [0,100,100])
#surcharge_vs_time = PolyLine([0,0.1,8], [0,100,100])
#
#z = np.linspace(0.0, np.sum(h), 7)
##t = np.linspace(0, 10, 50)
#t = np.logspace(-1,1.8,100)

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

    a = SchiffmanAndStein1970(my_code)

    a.make_all()

    print(repr(a.z))
    print(repr(a.por))









