# Example of using an existing file as a template for speccon1d_vr.
# Run in a python interpreter.

# Examine the effect of number of series terms on pore pressure profiles
# on example_1d_vr_001_schiffmanandstein1970.py

import numpy as np
from geotecha.speccon.speccon1d_vr import Speccon1dVR
import matplotlib.pyplot as plt



fname = "example_1d_vr_001_schiffmanandstein1970.py"
tindex = 11 # time index of pore pressure profile to plot.

with open(fname) as f:
    template = f.read()

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



for neig in np.arange(2,8,1):
    # Values in an input file can be overridden by appending the new values.
    # More complicated find and replace may be needed if variables depend
    # on each other.
    reader = (template
              + '\n' + 'neig={}'.format(neig)
              + '\n' + 'ppress_z_tval_indexes=[{}]'.format(tindex)
              + '\n' + 'save_data_to_file = False'
              + '\n' + 'save_figures_to_file = False')


    a = Speccon1dVR(reader)
    a.make_all()

    ax1.plot(a.por[:,0], a.ppress_z*a.H, label='{}'.format(neig))
    ax2.plot(a.tvals, a.avp[0], label='{}'.format(neig))
    ax3.plot(a.tvals, a.set[0], label='{}'.format(neig))

ax1.set_title('ppress at t={:g} days'.format(a.tvals[tindex]))
leg1 = ax1.legend(title='neig',loc=6)
leg1.draggable()

leg2 = ax2.legend(title='neig')
leg2.draggable()

leg3 = ax3.legend(title='neig')
leg3.draggable()

plt.show()