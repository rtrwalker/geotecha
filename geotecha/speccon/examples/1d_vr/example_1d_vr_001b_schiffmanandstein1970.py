# Example of using a multi-line string to initialize a Speccon1dVR object
# This is the most flexible approach especially for paramteric studies
# and custom plots.
# Only issues are a) no syntax highlighting because text is with a string,
# b) difficult to debug the code within a string.
# Within the string limited calculations are available, outside the string
# in a normal python environment you can do whatever calculations you want
# and place the results in the string.

# Vertical consolidation of four soil layers
# Schiffman, R. L, and J. R Stein. (1970) 'One-Dimensional Consolidation of
# Layered Systems'. Journal of the Soil Mechanics and Foundations
# Division 96, no. 4 (1970): 1499-1504.

# Examine the effect of different layer thicknesses

from __future__ import division, print_function
import numpy as np
from geotecha.speccon.speccon1d_vr import Speccon1dVR
import matplotlib.pyplot as plt

# changing parameters
h_values = [np.array([10, 20, 30, 20]),
            np.array([5, 20, 40, 20]),
            np.array([10, 50, 5, 10])] # layer ticknesses
tprofile = 3500 # time to calc pore ressure profiles


# the reader string is a template with {} indicating where parameters will be
# inserted.  Use double curly braces {{}} if you need curly braces in your
# string.
reader = """\
# Parameters from Schiffman and Stein 1970
h = np.{h} # article has np.array([10, 20, 30, 20]) # feet
cv = np.array([0.0411, 0.1918, 0.0548, 0.0686]) # square feet per day
mv = np.array([3.07e-3, 1.95e-3, 9.74e-4, 1.95e-3]) # square feet per kip
#kv = np.array([7.89e-6, 2.34e-5, 3.33e-6, 8.35e-6]) # feet per day
kv = cv*mv # assume kv values are actually kv/gamw


# speccon1d_vr parameters
drn = 0
neig = 60

H = np.sum(h)
z2 = np.cumsum(h) / H # Normalized Z at bottom of each layer
z1 = (np.cumsum(h) - h) / H # Normalized Z at top of each layer

mvref = mv[0] # Choosing 1st layer as reference value
kvref = kv[0] # Choosing 1st layer as reference value

dTv = 1 / H**2 * kvref / mvref

mv = PolyLine(z1, z2, mv/mvref, mv/mvref)
kv = PolyLine(z1, z2, kv/kvref, kv/kvref)

surcharge_vs_time = PolyLine([0,0,30000], [0,100,100])
surcharge_vs_depth = PolyLine([0,1], [1,1]) # Load is uniform with depth



ppress_z = np.linspace(0,1,200)

tvals = np.linspace(0.01, 3.2e4, 100)
i = np.searchsorted(tvals,{tprofile})
tvals = np.insert(tvals, i, {tprofile})
ppress_z_tval_indexes=[i]

avg_ppress_z_pairs = [[0,1]]
settlement_z_pairs = [[0,1]]

author = "Dr. Rohan Walker"
"""






fig = plt.figure(figsize=(15,8))
ax1 = fig.add_subplot('131')
ax1.set_xlabel('ppress')
ax1.set_ylabel('depth')
ax1.invert_yaxis()
ax1.grid()

ax2 = fig.add_subplot('132')
ax2.set_xlabel('time')
ax2.set_ylabel('average pore pressure')


ax3 = fig.add_subplot('133')
ax3.set_xlabel('time')
ax3.set_ylabel('settlement')
ax3.invert_yaxis()



for h in h_values:
    a = Speccon1dVR(reader.format(h=repr(h),
                                  tprofile=tprofile))
    a.make_all()
    label='{}'.format(str(list(h)))
    ax1.plot(a.por[:,0], a.ppress_z*a.H, label=label)
    ax2.plot(a.tvals, a.avp[0], label=label)
    ax3.plot(a.tvals, a.set[0], label=label)

leg_title='layer thickness'
ax1.set_title('ppress at t={:g} days'.format(tprofile))
leg1 = ax1.legend(title=leg_title,loc=6)
leg1.draggable()

leg2 = ax2.legend(title=leg_title)
leg2.draggable()

leg3 = ax3.legend(title=leg_title)
leg3.draggable()

plt.show()