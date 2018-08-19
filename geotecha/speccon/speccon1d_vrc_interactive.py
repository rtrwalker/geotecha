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
This is really just way to interactively run a speccon1d_vrc analysis
without polluting the the if __name__ == '__main__' area of
geotecha.speccon.speccon1d_vrc

"""
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

import geotecha.piecewise.piecewise_linear_1d as pwise
from geotecha.piecewise.piecewise_linear_1d import PolyLine
import geotecha.mathematics.transformations as transformations
import textwrap

from geotecha.speccon.speccon1d_vrc import Speccon1dVRC

if __name__ == '__main__':

    my_code = textwrap.dedent("""\
#from geotecha.piecewise.piecewise_linear_1d import PolyLine
#import numpy as np
H = 1
drn = 1
dT = 1
dTh = 0.5
dTv = 0.1 * 0.25
dThc = 1000
dTvc = 1000
neig = 20


mvref = 2.0
kvref = 1.0
khref = 1.0
kvcref = 1.0
khcref = 1.0
etref = 1.0


re=1.5
rc=0.1
n = re/rc

# work out lumped mv parameter
mv_x = np.array([0.0,1.0])
ms_y = np.array([0.5,0.5])
mc_y = np.array([0.5,0.5])/200
mv_y = ms_y/(1+(ms_y/mc_y-1)/n**2)
mv = PolyLine(mv_x, mv_y)

kh = PolyLine([0,1], [1,1])
kv = PolyLine([0,1], [5,5])
khc = PolyLine([0,1], [1,1])
kvc = PolyLine([0,1], [5,5])

#et = PolyLine([0,0.48,0.48, 0.52, 0.52,1], [0, 0,1,1,0,0])
et = PolyLine([0,1], [1,1])

#surcharge_vs_depth = PolyLine([0,1], [1,1]),
#surcharge_vs_time = PolyLine([0,0,10], [0,10,10])
#surcharge_omega_phase = (2*np.pi*0.5, -np.pi/2)
surcharge_vs_depth = PolyLine([0,1], [1,1])
surcharge_vs_time = PolyLine([0,0,10], [0,10,10])
#surcharge_omega_phase = [(2*np.pi*0.5, -np.pi/2), None]


#dT = 1
#dTh = 0.5
#dTv = 0.1 * 0.25
#mv = PolyLine([0,1], [0.5, 0.5])
#surcharge_vs_depth = [PolyLine([0,1], [1,1]), PolyLine([0,1], [1,1])]
#surcharge_vs_time = [PolyLine([0,0,10], [0,10,10]), PolyLine([0,0,10], [0,10,10])]



#top_vs_time = PolyLine([0,0.0,10], [0,-100,-100])
#top_omega_phase = (2*np.pi*1, -np.pi/2)
#bot_vs_time = PolyLine([0,0.0,3], [0,-10,-10])
#bot_vs_time = PolyLine([0,0.0,10], [0, -10, -10])

#bot_vs_time = PolyLine([0,0.0,0.4,10], [0, -2500, -250, -210])
#bot_omega_phase = (2*np.pi*2, -np.pi/2)


#fixed_ppress = (0.2, 1000, PolyLine([0, 0.0, 8], [0,-30,-30]))
#fixed_ppress_omega_phase = (2*np.pi*2, -np.pi/2)

#pumping = (0.5, PolyLine([0,0,10], [0,5,5]))


ppress_z = np.linspace(0,1,50)
avg_ppress_z_pairs = [[0,1],[0.4, 0.5]]
settlement_z_pairs = [[0,1],[0.4, 0.5]]
#tvals = np.linspace(0,3,10)
tvals = [0,0.05,0.1]+list(np.linspace(0.2,5,100))
tvals = np.linspace(0, 5, 100)
tvals = np.logspace(-2, 0.3,50)
ppress_z_tval_indexes = np.arange(len(tvals))[::len(tvals)//7]
#avg_ppress_z_pairs_tval_indexes = slice(None, None)#[0,4,6]
#settlement_z_pairs_tval_indexes = slice(None, None)#[0,4,6]

implementation='scalar'
implementation='vectorized'
#implementation='fortran'
#RLzero = -12.0
plot_properties={}

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

    """)


    a = Speccon1dVRC(my_code)

    a.make_all()

    print('\nt', repr(a.tvals))
    print('\nz', repr(a.ppress_z))
    print('\npor', repr(a.por))
    print('\npors', repr(a.pors))
    print('\nporc', repr(a.porc))
    print('\navp', repr(a.avp))
    print('\navps', repr(a.avps))
    print('\navpc', repr(a.avpc))
    print('\nset', repr(a.set))


