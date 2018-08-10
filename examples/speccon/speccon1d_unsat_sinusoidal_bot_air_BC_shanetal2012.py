# speccon1d_unsat example (if viewing this in docs, plots are at bottom of page)

# Unsaturated soil 1 dimensional consolidation. sinusoidal varying bot air
# pressure boundary condition.
# Compare with Shan et al. (2012) Fig2a and Fig5a
# The orignal Shan et al. (2012)
# is implemented separately in geotecha.consolidation.shanetal2012.

#note there are more examples like this in the geotecha tesing routines for
# soeccon1d_unsat.  Look in the source code.

# Shan, Zhendong, Daosheng Ling, and Haojiang Ding. 2012. 'Exact
# Solutions for One-dimensional Consolidation of Single-layer
# Unsaturated Soil'. International Journal for Numerical and
# Analytical Methods in Geomechanics 36 (6): 708-22.
# doi:10.1002/nag.1026.

# This file should be run with python.  It will not work if run with the
# speccon1d_vr.exe script program.

from __future__ import division, print_function
import numpy as np
from geotecha.speccon.speccon1d_unsat import Speccon1dUnsat
import matplotlib.pyplot as plt

#PTPB drainage
#dsig = 100kPa instantly
#uabot = sin(2*np.pi/1e8)
#
#ka/kw=10
#
#other data:
#n = 0.50
#S=0.80
#kw=10^10m/s
#m1kw=-0.5x10**4 kPa-1
#h=10m
#mw2=-2.0x10**4 kPa-1
#ma1k=-2.0x10**4 kPa-1
#ma2=1.0x10**4 kPa-1
#
#gamw= 10000N
#ua_=uatm=101kPa,
#R=8.31432J/molK
#t0 = 20 degrees C,
#T =(t0+273.16)K,
#wa=29x10**3 kg/mol
#
#Note to get (dua, duw) = (0.2, 0.4) * dsig need ua_=111kPa


#Expected values
#t = timeua, tuw, tset = time values for pore air and pore water pressure settlement output
#z = depth values
#pora, porw = excess pore pressure at time t and depth z in air and soil.

z = np.array([0, 3.0, 5.0, 8.0, 10.0])/10
t = np.array([1e6,3e6, 1e8,3e8, 1e9, 2e9 ])

porw = np.array(
 [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
      0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
   [  2.51160269e+01,   2.53809684e+01,   2.86794389e+01,
      1.84225700e+01,   1.31402505e+01,   1.30905228e+01],
   [  2.52028754e+01,   2.56521851e+01,   3.96023735e+01,
      2.31985517e+01,   2.13815989e+01,   2.13201321e+01],
   [  2.53451599e+01,   2.60799853e+01,   3.20328797e+01,
      1.30290743e+01,   2.24473606e+01,   2.24112313e+01],
   [  2.67914232e-14,   1.68851653e-14,   9.04637595e-15,
      2.05636474e-15,   7.48375877e-15,   7.47623126e-15]])

pora = np.array(
 [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
      0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
   [  1.69193072e-01,   5.22191992e-01,   1.75980381e+01,
      2.85434458e+01,  -4.25318313e-02,  -4.25288413e-02],
   [  2.84908997e-01,   8.83551215e-01,   2.93404622e+01,
      4.75690971e+01,  -5.86231831e-02,  -5.86194872e-02],
   [  4.74493752e-01,   1.46227901e+00,   4.69852488e+01,
      7.60973130e+01,  -4.53849520e-02,  -4.53827796e-02],
   [  6.28314397e-01,   1.88484397e+00,   5.87785252e+01,
      9.51056516e+01,  -2.45053354e-14,  -4.89982710e-14]])


#############################
##shanetal2012 input
#kw = 1e-10
#ka = 10 * kw
#H=10
#Cw=-0.75
#Cvw=-5e-8
#Ca = -0.0775134
#Cva=-64504.4 * ka
#drn=0
#Csw=0.25
#Csa=0.155027
#uwi=(40, 40)
#uai=(20, 20)
#nterms=200
#f=f1=f2=f3=f4=None
#f4 = dict([('type', 'sin'), ('q0',100.0), ('omega',2*np.pi / 1e9)])
#z = np.array([0, 3.0, 5.0, 8.0, 10.0])
#t = np.array([1e6,3e6, 1e8,3e8, 1e9, 2e9 ])
#
#porw, pora = shanetal2012(z, t, H, Cw, Cvw, Ca, Cva, drn, Csw, Csa,
#             uwi, uai, nterms, f=f, f1=f1, f2=f2, f3=f3, f4=f4)
#########################

reader = ("""\
H = 10 #m
drn = 0
neig = 30

mvref = 1e-4 #1/kPa
kwref = 1.0e-10 #m/s

karef = kwref * 10 #m/s
Daref = karef / 10 # from equation ka=Da*g

wa = 29.0e-3 #kg / mol
R = 8.31432 #J/(mol.K)
ua_= 111 #kPa
T = 273.16 + 20
dTa = Daref /(mvref) / (wa*ua_/(R*T))/ H ** 2
dTw = kwref / mvref / 10 / H**2
dT = max(dTw, dTa)

kw = PolyLine([0,1], [1,1])
Da = PolyLine([0,1], [1,1])
S = PolyLine([0,1], [0.8] * 2)
n = PolyLine([0,1], [0.5] * 2)

m1kw = PolyLine([0,1], [-0.5]*2)
m2w =  PolyLine([0,1], [-2.0]*2)
m1ka = PolyLine([0,1], [-2.0]*2)
m2a =  PolyLine([0,1], [1.0]*2)



surcharge_vs_depth = PolyLine([0,1], [1,1])
surcharge_vs_time = PolyLine([0, 0, 1e12], [0, 100, 100])


abot_vs_time = PolyLine([0, 0.0, 1e12], [0,100,100])
abot_omega_phase = (2*np.pi/1e9, -np.pi/2)

ppress_z = np.{z}
#avg_ppress_z_pairs = [[0,1]]
#settlement_z_pairs = [[0,1]]

tvals = np.{t}

#ppress_z_tval_indexes = slice(None, len(tua)+len(tuw))
#avg_ppress_z_pairs_tval_indexes = slice(None, None)#[0,4,6]
#settlement_z_pairs_tval_indexes = slice(len(tua)+len(tuw),len(tua)+len(tuw)+len(tset))

save_data_to_file= False
save_figures_to_file= False
show_figures= False

    """.format(t=repr(t), z = repr(z)))



a = Speccon1dUnsat(reader)
a.make_all()


# custom plots
title = ("Shan et al. (2012) Unsaturated soil. instant surcharge"
         " + \nsinusoidal bot air pressure boundary consdition.")
fig = plt.figure(figsize=(9,6))
fig.suptitle(title)
#z vs ua
ax1 = fig.add_subplot("121")
ax1.set_xlabel('Excess pore pressure in air, kPa')
ax1.set_ylabel('Depth')
ax1.invert_yaxis()
ax1.plot(pora, z,
         ls="None", color='Blue', marker="+", ms=5,
         label='expected')
ax1.plot(a.pora, z,
         ls='-', color='red', marker='o', ms=5, markerfacecolor='None',
         markeredgecolor='red',
         label='calculated')

#z vs uw
ax2 = fig.add_subplot("122")
ax2.set_xlabel('Excess pore pressure in water, kPa')
ax2.set_ylabel('Depth')
ax2.invert_yaxis()
ax2.plot(porw, z,
         ls="None", color='Blue', marker="+", ms=5,
         label='expected')
ax2.plot(a.porw, z,
         ls='-', color='red', marker='o', ms=5, markerfacecolor='None',
         markeredgecolor='red',
         label='calculated')

fig.subplots_adjust(top=0.9)#, bottom=0.15, left=0.13, right=0.94)
#fig.tight_layout()
plt.show()



