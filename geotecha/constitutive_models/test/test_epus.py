from __future__ import division, print_function
import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt
import math
from numpy.testing import assert_allclose
import unittest
import nose
import inspect

from geotecha.constitutive_models import epus as epus


"""Testing routines for epus module"""


DEBUG=False

def test_EPUSvsVB_case01():
    """EPUS vs VB version, Test Case #01

    wsat=0.262, a = 3.1 * 10 ** 6, b = 3.377, wr = 0.128, sl = 0.115; LogDS = 0.6, LogRS = 2, Ccs = SimpleSWCC.sl, Css = 0.019, Ccd = 0, Assumption = 1, Gs = 2.7, K0 = 1, beta = 0.1, pm = 1, Pore_shape = 1
    1. Load to 20kPa, 2. Unload to 1kPa, 3. Dry to 10^6kPa, Wetting to 30kPa, 5. Load to 1500kPa, 100 pts per step.

    20 points per loading step,
    400 points in PoresizeDistribution
    """
    #get test data
    fname = "EPUS_test_data_case01.csv" #needs to be in same directory as file

#    fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                          fname)
    #inspect approach at http://stackoverflow.com/a/18489147/2530083
    mname = os.path.abspath(os.path.dirname(inspect.getsourcefile(lambda:0)))
    fpath = os.path.join(mname,
                          fname)

    data = np.loadtxt(fpath, skiprows=4, dtype=float, delimiter=',', unpack=True)
    expect = dict()
    names=['ss', 'st', 'e', 'w', 'Sr', 'vw']
    for i, v in enumerate(names):
        expect[v] = data[i]


    # input data
    pdict = dict()
    SWCC = epus.CurveFittingSWCC(wsat=0.262,
                            a = 3.1 * 10 ** 6,
                            b = 3.377,
                            wr = 0.128,
                            sl = 0.115)


    stp = epus.StressPath([dict(ist=True, npp=20, vl=20, name="1. Load to 20kPa"),
                      dict(ist=True, npp=20, vl=1, name="2. Unload to 1kPa"),
                      dict(ist=False, npp=20, vl=1e6, name="3. Dry to 10^6kPa"),
                      dict(ist=False, npp=20, vl=30, name="4. Wetting to 30kPa"),
                      dict(ist=False, npp=20, vl=1500, name="5. Load to 1500kPa")])
    pdict=dict(
         SimpleSWCC=SWCC,
         stp=stp,
         logDS=0.6,
         logRS=2,
         Css=0.019,
         beta=0.1,
         soilname='Artificial silt',
         username='Hung Pham',
         Npoint=400)

    #calclate
    a = epus.EPUS(**pdict)

    a.Calresults()
    a.stp.combine_datapoints()
    dp = a.stp.datapoints_combined



    if DEBUG:

        fig = plt.figure(figsize=(10,12))

        ax = fig.add_subplot(321)
        ax.set_xlabel('Suction')
        ax.set_ylabel('$w$')
        ax.set_xscale('log')
        for dp_step in a.stp.datapoints:
            ax.plot(dp_step.ss, dp_step.w, marker="+", ls='none', label='calc')
        ax.plot(expect['ss'], expect['w'], linewidth=2, label='expected')

        handles, labels = ax.get_legend_handles_labels()
        leg = ax.legend([handles[0], handles[-1]], [labels[0], labels[-1]])
        leg.draggable()

        ax = fig.add_subplot(322)
        ax.set_xlabel('Suction')
        ax.set_ylabel('$e$')
        ax.set_xscale('log')
        for dp_step in a.stp.datapoints:
            ax.plot(dp_step.ss, dp_step.e, marker="+", ls='none')
        ax.plot(expect['ss'], expect['e'], linewidth=2)

        ax = fig.add_subplot(323)
        ax.set_xlabel('Suction')
        ax.set_ylabel('$\\theta$')
        ax.set_xscale('log')
        for dp_step in a.stp.datapoints:
            ax.plot(dp_step.ss, dp_step.vw, marker="+", ls='none')
        ax.plot(expect['ss'], expect['vw'], linewidth=2)

        ax = fig.add_subplot(324)
        ax.set_xlabel('Stress')
        ax.set_ylabel('$e$')
        ax.set_xscale('log')
        for dp_step in a.stp.datapoints:
            ax.plot(dp_step.st, dp_step.e, marker="+", ls='none')
        ax.plot(expect['st'], expect['e'], linewidth=2)

        ax = fig.add_subplot(325)
        ax.set_xlabel('Suction')
        ax.set_ylabel('$S_r$')
        ax.set_xscale('log')
        for dp_step in a.stp.datapoints:
            ax.plot(dp_step.ss, dp_step.Sr, marker="+", ls='none')
        ax.plot(expect['ss'], expect['Sr'], linewidth=2)

        plt.show()




    #compare
    assert_allclose(dp.ss, expect['ss'], atol=1e-6, rtol=1e-6)
    assert_allclose(dp.st, expect['st'], atol=1e-6, rtol=1e-6)
    assert_allclose(dp.e, expect['e'], atol=1e-2, rtol=1e-6) # looks like the acuracy is lost here
    assert_allclose(dp.w, expect['w'], atol=1e-6, rtol=1e-6)
    assert_allclose(dp.Sr, expect['Sr'], atol=1e-2, rtol=1e-6) # Sr is calced from e
    assert_allclose(dp.vw, expect['vw'], atol=1e-2, rtol=1e-6) # vw is calced from e

    # the error in void ratio e, I think arises in the calculation of f.RV
    # in the EPUS.DryPoreSize method.  The VB version that I am testing against
    # using single precision floating point numbers, here I use double.  The
    # calculation can involves subtraction of two numbers that are very close.
    # So while each number may have be accurate to 5 sig figs, when subtracted
    # the first 4 sig figs may be irrelevant so the final result might only
    # be accurate to 1 significant figure.  I think this is called
    # "catasophic cancellation" or "Loss of significance"

if __name__ =="__main__":
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest', '--doctest-options=+ELLIPSIS'])



#    DEBUG=True
#    test_EPUSvsVB_case01()