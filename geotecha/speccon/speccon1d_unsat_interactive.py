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
This is really just way to interactively run a speccon1d_unsat analysis
without polluting the the if __name__ == '__main__' area of
geotecha.speccon.speccon1d_unsat

"""
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

import geotecha.piecewise.piecewise_linear_1d as pwise
from geotecha.piecewise.piecewise_linear_1d import PolyLine
import geotecha.mathematics.transformations as transformations
import textwrap

from geotecha.speccon.speccon1d_unsat import Speccon1dUnsat


if __name__ == '__main__':

#
#    my_code = textwrap.dedent("""\
##from geotecha.piecewise.piecewise_linear_1d import PolyLine
##import numpy as np
## fredlund and rahandjo test 1
#H = 0.11815 #m
#drn = 0
#neig = 30
#
#mvref = 1e-4 #1/kPa
#kwref = 1.0e-10 #m/s
#kwref *= 60 #m/min
#
#karef = 8e-13 #m/s
#Daref = karef * 60 / 10 #m/min from equation ka=Da*g
#
#wa = 28.966e-3 #kg / kmol
#R = 8.31432 #J/(mol.K)
#ua_= 101 #kPa
#T = 273.16 + 23
#dTa = Daref /(mvref) / (wa/(R*T))/ua_ / H ** 2
#dTw = kwref / mvref / 10 / H**2
#dT = dTw
#
#kw = PolyLine([0,1], [1,1])
#Da = PolyLine([0,1], [1,1])
#S = PolyLine([0,1], [0.7887] * 2)
#n = PolyLine([0,1], [0.517] * 2)
#
#m1kw = PolyLine([0,1], [-0.8]*2)
#m2w = PolyLine([0,1], [-2.41]*2)
#m1ka = PolyLine([0,1], [-0.37]*2)
#m2a = PolyLine([0,1], [1.11]*2)
#
#
#surcharge_vs_depth = PolyLine([0,1], [1,1])
#surcharge_vs_time = PolyLine([0,0,10000], [0,202,202])
##surcharge_omega_phase = (2*np.pi*0.5, -np.pi/2)
#
##wtop_vs_time = PolyLine([0,0.0,10], [0,-100,-100])
##wtop_omega_phase = (2*np.pi*1, -np.pi/2)
##wbot_vs_time = PolyLine([0,0.0,3], [0,-10,-10])
##wbot_omega_phase = (2*np.pi*1, -np.pi/2)
#
##atop_vs_time = PolyLine([0,0.0,10], [0,-100,-100])
##atop_omega_phase = (2*np.pi*1, -np.pi/2)
##abot_vs_time = PolyLine([0,0.0,3], [0,-10,-10])
##abot_omega_phase = (2*np.pi*1, -np.pi/2)
#
#
#ppress_z = np.linspace(0,1,50)
#avg_ppress_z_pairs = [[0,1]]
#settlement_z_pairs = [[0,1]]
##tvals = np.linspace(0,1000,10)
#tvals = [0,0.05,0.1]+list(np.linspace(0.2,5,100))
#tvals = np.linspace(0, 5, 100)
#tvals = np.logspace(0, 5, 50)
##ppress_z_tval_indexes = np.arange(len(tvals))[::len(tvals)//7]
##avg_ppress_z_pairs_tval_indexes = slice(None, None)#[0,4,6]
##settlement_z_pairs_tval_indexes = slice(None, None)#[0,4,6]
#
#implementation='scalar'
#implementation='vectorized'
##implementation='fortran'
##RLzero = -12.0
#plot_properties={}
#
#directory= r"C:\\Users\\Rohan Walker\\Documents\\temp" #may always need the r
#save_data_to_file= False
#save_figures_to_file= False
#show_figures= False
#overwrite=True
#
##prefix="silly"
#
##create_directory=True
##data_ext = '.csv'
##input_ext='.py'
#figure_ext='.png'
#
#    """)
#

    my_code = textwrap.dedent("""\
#from geotecha.piecewise.piecewise_linear_1d import PolyLine
#import numpy as np
# shan et al,
H = 10 #m
drn = 1
neig = 400

mvref = 1e-4 #1/kPa
kwref = 1.0e-10 #m/s

karef = kwref * 100 #m/s
Daref = karef / 10 # from equation ka=Da*g

wa = 29.0e-3 #kg / mol
R = 8.31432 #J/(mol.K)
ua_= 111 #kPa
T = 273.16 + 20
dTa = Daref /(mvref) / (wa*ua_/(R*T))/ H ** 2
dTw = kwref / mvref/ 10 / H**2
dT = max(dTw,dTa)

kw = PolyLine([0,1], [1,1])
Da = PolyLine([0,1], [1,1])
S = PolyLine([0,1], [0.8] * 2)
n = PolyLine([0,1], [0.5] * 2)

m1kw = PolyLine([0,1], [-0.5]*2)
m2w =  PolyLine([0,1], [-2.0]*2)
m1ka = PolyLine([0,1], [-2.0]*2)
m2a =  PolyLine([0,1], [1.0]*2)


surcharge_vs_depth = PolyLine([0,1], [1,1])
surcharge_vs_time = PolyLine([0,0,1e12], [0,100,100])
#surcharge_omega_phase = (2*np.pi*0.5, -np.pi/2)

wtop_vs_time = PolyLine([0, 0.0, 1e12], [0,100,100])
wtop_omega_phase = (2*np.pi/1e5, -np.pi/2)
#wbot_vs_time = PolyLine([0,0.0,1e10, 1e10, 1e11], [0,40,40,0,0])
#wbot_omega_phase = (2*np.pi*1, -np.pi/2)

#atop_vs_time = PolyLine([0,0.0,1e10, 1e10, 1e11], [0,20,20,0,0])
#atop_omega_phase = (2*np.pi*1, -np.pi/2)
#abot_vs_time = PolyLine([0,0.0,1e10, 1e10, 1e11], [0,20,20,0,0])
#abot_omega_phase = (2*np.pi*1, -np.pi/2)


ppress_z = np.linspace(0,1,50)
ppress_z = [0.02]
avg_ppress_z_pairs = [[0,1]]
settlement_z_pairs = [[0,1]]
#tvals = np.linspace(0, 1000,10)
tvals = [0,0.05,0.1]+list(np.linspace(0.2,5,100))
tvals = np.linspace(0, 5, 100)
tvals = np.logspace(0, 6, 100)
#ppress_z_tval_indexes = np.arange(len(tvals))[::len(tvals)//7]
#avg_ppress_z_pairs_tval_indexes = slice(None, None)#[0,4,6]
#settlement_z_pairs_tval_indexes = slice(None, None)#[0,4,6]

implementation='scalar'
implementation='vectorized'
implementation='fortran'
#RLzero = -12.0
plot_properties={}

directory= r"C:\\Users\\Rohan Walker\\Documents\\temp" #may always need the r
save_data_to_file= False
save_figures_to_file= False
show_figures= False
overwrite=True

#prefix="silly"

#create_directory=True
#data_ext = '.csv'
#input_ext='.py'
figure_ext='.png'

    """)

    a = Speccon1dUnsat(my_code)

    a.make_all()

#    a.print_eigs()




    if 1:
        #only use this if there is one z interval
        fig = plt.figure()
        ax=fig.add_subplot('111')
        ax.plot(a.tvals, a.set[0], 'ro', ls='-', label='combined')
        ax.plot(a.tvals, a.setw[0], 'bs', ls='-', label='water')
        ax.plot(a.tvals, a.seta[0], 'g*', ls='-', label='air')
        ax.set_xlabel('Time')
        ax.set_ylabel('Settlement')
        ax.set_xscale('log')
#        ax.invert_yaxis()
        ax.set_xlim(0)
        ax.grid()
        leg = plt.legend(loc=3 )
        leg.draggable()

        fig = plt.figure()
        ax=fig.add_subplot('111')
        ax.plot(a.tvals, a.avpw[0], 'bs', ls='-', label='water')
        ax.plot(a.tvals, a.avpa[0], 'g*', ls='-', label='air')
        ax.set_xlabel('Time')
        ax.set_ylabel('average pore pressure')
        ax.set_xscale('log')
        ax.set_xlim(0)
        ax.grid()
        leg = plt.legend(loc=3 )
        leg.draggable()


        fig = plt.figure()
        ax=fig.add_subplot('111')
        ax.plot(a.tvals, a.porw[len(a.ppress_z)//2], 'bs', ls='-', label='water')
        ax.plot(a.tvals, a.pora[len(a.ppress_z)//2], 'g*', ls='-', label='air')
        ax.set_xlabel('Time')
        ax.set_ylabel('Pore pressure at z={}'.format(a.ppress_z[len(a.ppress_z)//2]*a.H))
        ax.set_xscale('log')
        ax.set_xlim(0)
        ax.grid()
        leg = plt.legend(loc=3 )
        leg.draggable()
        plt.show()



