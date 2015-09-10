# geotecha - A software suite for geotechncial engineering
# Copyright (C) 2015  Rohan T. Walker (rtrwalker@gmail.com)
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

from __future__ import division, print_function
import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt

try:
    import geotecha.constitutive_models.epus_ext as epus_ext
    # remeber that all fortran variables and routines will be lower case
    # regardless of what case they are in the source code.
    _SUCCESSFUL_FORTRAN_IMPORT = True
except ImportError:
    print("Failed to import epus_ext; EPUS will use slow scalar version instead.")
    _SUCCESSFUL_FORTRAN_IMPORT = False

from geotecha.constitutive_models import void_ratio_permeability
from geotecha.constitutive_models import swcc

import math
log10 = math.log10
log = math.log
#log10 = np.log10
#log = np.log


"""Elasto-Plastic for Unsaturated Soils, adapted from VB code in Phd thesis of
Pham (2005).

"""

class EPUS(object):
    """Elasto-Plastic for Unsaturated Soils


    This code is adapted from visual basic code in Appendix E of thesis of
    Pham (2005) [2]_.  Journal article is [1]_.

    Parameters
    ----------
    SimpleSWCC : CurveFittingSWCC object
        Soil water characteristic curve.
    stp : StressPath object
        object containing info abount the stress path.
    logDS : float, optional
        Distance ratio between boundary drying and boundary drying curve.
        In log cycles. default logDS=0.
    logRS : float, optional
        Slope ratio between boundary drying and boundary drying curve.
        In log cycles. Default logRS=0.
    Ccs : float, optional
        semi-log slope of saturated compression curve.  Default=None, i.e.
        Ccs will be equal to SimpleSWCC.sl.
    Css : float, optional
        semi-log slope of saturated recompression curve.  Default Css=None
        i.e Css=Ccs*0.1
    Ccd : float, optional
        semi-log slope of dry compression curve, default Ccd=0.
    Gs : float, optional
        Specific gravity of solids default Gs=2.7.
    Assumption : int, optional
        Assumption=0 dry pores are incompressible,
        Assumption=1 dry pores are compressible. Default Assumption=1.
    beta : float, optionalB
        air entrapped in a collapsible pore is proportional to the volume
        of the pore and can be denoted by an entrapped air parameter beta.
        Default beta=0.
    pm : float, optional
        Soil parameter (Sr)^pm for calculating the yield stress of dry pores.
        Default pm=1.
    Pore_shape : float, optional
        For describing the change in AEV/WEV of the soil. Default
        Pore_shape=1.
    K0 : float, optional
        Lateral earth pressure coefficient. Default K0=1 i.e. isotropic.
    soilname : str, optional
        Default soilname='unknown'.
    username : str, optional
        Default username='unknown'
    Npoint : int, optional
        Number of data points along the pore size distribution curve.
        Default Npoint=1000
    NumSr : int, optional
        Number points divided along the wetting and drying Sr.
        Default NumSr=400
    MaxSuction : int, optional
        log Scale or 1000000. Default MaxSuction=6
    MinSuction : int, optional
        log scale or 0.001. Default MinSuction=-3
    implementation : ['fortran', 'scalar', ], optional
        Functional implementation: 'scalar' = python loops (slow),
        'fortran' = fortran code (fastest).  Currently there is no
        'vectorized' = numpy(fast) version.
        Default implementation='fortran'.  If fortran extention module
        cannot be imported then 'scalar' version will be used.
        If anything other than 'scalar' is used then
        default fortran version will be used.

    Attributes
    ----------
    f : PoresizeDistribution object
        Pore size distribution function at initial condition.
    Suction : float
        Current value of suction.
    Stress : float
        Current value of net normal stress.
    RefSr : float
        Reference value of Saturation.
    Curveon : int
         = 1: on drying; = 2: on horizontal scanning; = 3: on wetting
    errocc : bool
        Some sort of error code.
    Srdry : 1d array of float (size is NumSr)
        Drying degree of saturation SWCC (zero net mean stress)
    Srwet : 1d array of float (size is NumSr)
        Wetting degree of saturation SWCC  (zero net mean stress)

    References
    ----------
    .. [1] Pham, H. Q., and Fredlund, D. G. (2011).
           'Volume-mass unsaturated soil constitutive model for drying-wetting
           under isotropic loading-unloading conditions. Canadian Geotechnical
           Journal, 48(2), 280-313.
    .. [2] Pham, H. Q. (2005). 'A volume-mass constitutive model for
           unsaturated soils'. PhD Thesis, University of Saskatchewan,
           Saskatoon, Saskatchewan, Canada.



    """

    def __init__(self,
                 SimpleSWCC,
                 stp,
                 logDS=0.0,
                 logRS=0.0,
                 Ccs=None,
                 Css=None,
                 Ccd=0.0,
                 Gs=2.7,
                 Assumption=1,
                 beta=0.0,
                 pm=1.0,
                 Pore_shape=1.0,
                 K0=1.0,
                 soilname='unknown',
                 username='unknown',
                 Npoint=1000,
                 NumSr=400,
                 MaxSuction=6,
                 MinSuction=-3,
                 implementation='fortran'):

        self.SimpleSWCC = SimpleSWCC
        self.stp = stp
        self.logDS = logDS
        self.logRS = logRS
        if Ccs is None:
            self.Ccs = SimpleSWCC.sl
        else:
            self.Ccs = Ccs
        if Css is None:
            self.Css = self.Ccs * 0.1
        else:
            self.Css = Css
        self.Ccd = Ccd
        self.Gs = Gs
        self.Assumption = Assumption
        self.beta = beta
        self.pm = pm
        self.Pore_shape = Pore_shape
        self.K0 = K0
        self.soilname = soilname
        self.username = username
        self.Npoint = Npoint
        self.NumSr = NumSr
        self.MaxSuction = MaxSuction
        self.MinSuction = MinSuction
        self.implementation = implementation

        self.Srdry = np.zeros(self.NumSr)
        self.Srwet = np.zeros(self.NumSr)


    def CVStress(self):
        """K0 to isotropic stress conversion factor

        Returns
        -------
        out : float
            (2 * K0 + 1) / 3

        """
        return (2 * self.K0 + 1) / 3


    def Wc(self, s):
        """Collapsible water content function

        Calls CurveFittingSWCC.Wc with self.Gs

        Parameters
        ----------
        s : float
            Suction.

        Returns
        -------
        out : float
            water content Wc

        See also
        --------
        CurveFittingSWCC.Wc : actual function called.

        """
        return self.SimpleSWCC.Wc(s, Gs=self.Gs)


    def DWc(self, s):
        """Derivative of collapsible water content function

        Calls CurveFittingSWCC.DWc with self.Gs

        Parameters
        ----------
        s : float
            Suction.

        Returns
        -------
        out : float
            Derivative of water content Wc w.r.t. suction

        See also
        --------
        CurveFittingSWCC.DWc : actual function called.

        """
        return self.SimpleSWCC.DWc(s, Gs=self.Gs)


    def DryPoreSize(self):
        """Set PoresizeDistribution self.f to slurry state


        Examples
        --------
        >>> SWCC = CurveFittingSWCC(wsat=0.262,
        ...                         a = 3.1e6,
        ...                         b = 3.377,
        ...                         wr = 0.128,
        ...                         sl = 0.115)
        >>> stp = StressPath([dict(ist=True, vl=20, name="1. Load to 20kPa")])
        >>> pdict=dict(
        ...      SimpleSWCC=SWCC,
        ...      stp=stp,
        ...      logDS=0.6,
        ...      logRS=2,
        ...      Css=0.019,
        ...      beta=0.1,
        ...      soilname='Artificial silt',
        ...      username='Hung Pham')
        >>> a=EPUS(**pdict)
        >>> a.DryPoreSize()
        >>> a.f.AEV[24]
        -2.78...
        >>> a.f.WEV[24]
        -8.09...
        >>> a.f.RVC[24]
        4.24...e-08
        >>> a.f.RV[24]
        4.24...e-08
        >>> a.f.Ccp[24]
        1.04...e-18
        >>> a.f.Csp[24]
        1.71...e-19
        >>> a.f.Ccd[24]
        0.0


        """


        b1 = self.SimpleSWCC.b
        a1 = self.SimpleSWCC.a
        Gs = self.Gs
        DWc = self.DWc
        Ccs = self.Ccs
        Css = self.Css
        Ccd = self.Ccd
        logRS = self.logRS
        logDS = self.logDS

        MaxSuction = self.MaxSuction
        MinSuction = self.MinSuction


        f = PoresizeDistribution(self.Npoint)

        # Total volume at completely dry
        # I don't think this Vtotal bit is used (so I have commented it out RTRW 18/6/2015)
#        Vtotal = 0
#        for i in range(f.Npoint):
#            Interval = (MaxSuction - MinSuction) / f.Npoint
#            s = 10 ** (f.AEV[i])
#            temp = (a1 - a1 * s ** b1 / (10 ** (6 * b1))) / (s ** b1 + a1)
#            Vtotal = Vtotal + (-Gs * DWc(s) * s * log(10) - temp * Ccs) * Interval


        for i in range(f.Npoint):
            Interval = (MaxSuction - MinSuction) / f.Npoint
            # Air Entry Value
            f.AEV[i] = (i) * Interval + MinSuction

            s = 10 ** (f.AEV[i])
            # Pore volume
            temp = (a1 - a1 * s ** b1 / (10 ** (6 * b1))) / (s ** b1 + a1)
            f.RV[i] = (-Gs * DWc(s) * s * log(10) - temp * Ccs) * Interval


            # Saturated compression indexes
            temp = -a1 * b1 * (s ** b1) * (a1 * 10 ** (-6 * b1) + 1) / (s * (s ** b1 + a1) ** 2)
#A-83

            f.Ccp[i] = -temp * Ccs * (10 ** (f.AEV[i] + Interval) - 10 ** (f.AEV[i]))
            f.Csp[i] = -temp * Css * (10 ** (f.AEV[i] + Interval) - 10 ** (f.AEV[i]))
            # Dried compression index
            f.Ccd[i] = f.Ccp[i] * (Ccd / Ccs)  #(f.RV[i] / Vtotal) * Ccd

            # Calculate wetting curve parameters
            bwt = (a1 / (10 ** logDS) ** b1) ** (1 / logRS)
            dwt = b1 / logRS

            # Calculate wetting soil suctions (assuming tails of the SWCC are linear)
            if f.AEV[i] > log10((2.7 * a1) ** (1 / b1)):# Then
              f.WEV[i] = f.AEV[i] - ((log10((2.7 * a1) ** (1 / b1)) - log10((2.7 * bwt) ** (1 / dwt))) * (6 - f.AEV[i]) / (6 - log10((2.7 * a1) ** (1 / b1))))
            else:
              f.WEV[i] = (1 / dwt) * (log10(bwt) + b1 * f.AEV[i] - log10(a1))
            f.RVC[i] = f.RV[i]

        self.f = f

    def SlurryPoreSize(self):
        """Set PoresizeDistribution self.f to slurry state

        Zero soil suction - Mean kPa net mean stress.


        Examples
        --------
        >>> SWCC = CurveFittingSWCC(wsat=0.262,
        ...                         a = 3.1e6,
        ...                         b = 3.377,
        ...                         wr = 0.128,
        ...                         sl = 0.115)
        >>> stp = StressPath([dict(ist=True, vl=20, name="1. Load to 20kPa")])
        >>> pdict=dict(
        ...      SimpleSWCC=SWCC,
        ...      stp=stp,
        ...      logDS=0.6,
        ...      logRS=2,
        ...      Css=0.019,
        ...      beta=0.1,
        ...      soilname='Artificial silt',
        ...      username='Hung Pham')
        >>> a=EPUS(**pdict)
        >>> a.SlurryPoreSize()
        >>> a.f.RV[24]
        4.24...e-08
        >>> a.f.RVC[24]
        4.24...e-08
        >>> a.f.AEVC[24]
        -2.78...
        >>> a.f.WEVC[24]
        -8.09...
        >>> a.f.YieldSt[24]
        0.0001
        >>> a.f.Airentrapped[24]
        False



        """


        MinSuction = self.MinSuction

        self.DryPoreSize()
        f = self.f
        for i in range(f.Npoint):
            f.RV[i] = f.RV[i] + (f.AEV[i] - MinSuction) * f.Ccp[i]
            f.RVC[i] = f.RV[i]
            f.AEVC[i] = f.AEV[i]
            f.WEVC[i] = f.WEV[i]
            f.YieldSt[i] = 0.0001
            f.Filled[i] = True
            f.Airentrapped[i] = False
        temp = 0
        for i in range(f.Npoint):
            temp = temp + f.RV[i]

        # Set current suction and stress to the slurry condition.
        self.Suction = 10 ** MinSuction           # Max suction
        self.Stress = (10 ** MinSuction) / 1000   # Min suction
        self.Curveon = 1                         # Starts from drying curve
        self.RefSr = 1                           # Initial degree of saturation = 1


    def ChangeSuction(self, MaxSumStress, Initialsuction):
        """

        ' = suction(0)/suction(p)

        Parameters
        ----------
        MaxSumStress : float
            Not sure.
        Initialsuction : float
            Not sure.

        Returns
        -------
        out : float
            not sure

        """

        SimpleSWCC = self.SimpleSWCC


        Gs = self.Gs

        Ccs = self.Ccs
        Css = self.Css
        Stress = self.Stress
        Pore_shape = self.Pore_shape

        # Set history parameter equal to 1 for all cases
        if (Initialsuction >= 1 * MaxSumStress):
            return 1


        #Check this condition... if it is collapsible pore group
        if Initialsuction >= (10 * SimpleSWCC.a) ** (1 / SimpleSWCC.b):
            return 1

        temp2 = (SimpleSWCC.wsat * Gs - Ccs * log10(Initialsuction) - SimpleSWCC.wr * Gs)
        if temp2 <= 0:
            return 1

        temp1 = (Ccs - Css) * log10(MaxSumStress) + Css * log10(Initialsuction + Stress) - Ccs * log10(Initialsuction)
        if ((1 - Pore_shape * (temp1 / (3 * temp2))) <= 0):
# A-84
          self.errocc = True
          return 1
        else:
          return (1 - Pore_shape * (temp1 / (3 * temp2)))


    def ChangeWetSuction(self, MaxSumStress, Airentryvalue, Waterentryvalue):
        """

        ' = suction(0)/suction(p)

        Parameters
        ----------
        MaxSumStress : float
            Not sure.
        Airentryvalue : float
            Not sure.
        Waterentryvalue : float
            Not sure.

        Returns
        -------
        out : float
            not sure

        """

        SimpleSWCC = self.SimpleSWCC


        Gs = self.Gs
        Ccs = self.Ccs
        Css = self.Css
        Stress = self.Stress
        Pore_shape = self.Pore_shape


        # Set history parameter equal to 1 for all cases
        if (Airentryvalue >= 1 * MaxSumStress):
            return 1

        #Check this condition...  if it is collapsible pore group
        if Airentryvalue >= (10 * SimpleSWCC.a) ** (1 / SimpleSWCC.b):
            return 1
        temp2 = (SimpleSWCC.wsat * Gs - Ccs * log10(Airentryvalue) - SimpleSWCC.wr * Gs)
        if temp2 < 0:
            return 1

        temp1 = (Ccs - Css) * log10(MaxSumStress) + Css * log10(Waterentryvalue + Stress) - Ccs * log10(Airentryvalue)
        if ((1 - Pore_shape * (temp1 / (3 * temp2))) <= 0):
            self.errocc = True
            return 1
        else:
            return (1 - Pore_shape * (temp1 / (3 * temp2)))


    def Changevolume(self, InitialVolume, Yieldstress, CurrentStress, CompIndex, UnloadIndex):

        MaxSuction = self.MaxSuction
        MinSuction = self.MinSuction

        temp = InitialVolume - (log10(Yieldstress) - MinSuction) * CompIndex + (log10(Yieldstress) - log10(CurrentStress)) * UnloadIndex
        if temp > 0:
            return temp
        else:
            return 0

    def Drying(self, ss_new):
        """
        'Increase soil suction at a certain net mean/vertical stress
        'If increase soil suction at a certain net mean stress. All pores that has air

        """

        RfSr_value = self.RfSr_value
        ChangeSuction = self.ChangeSuction
        Changevolume = self.Changevolume

        f = self.f
        Stress = self.Stress


        if self.Curveon==1:
            self.Curveon = 1
            self.RefSr = RfSr_value(1, ss_new)
        elif self.Curveon==2:
            if self.RefSr <= RfSr_value(1, ss_new):
                self.Curveon = 2
            else:
                self.Curveon = 1
                self.RefSr = RfSr_value(1, ss_new)
        elif self.Curveon==3:
            if self.RefSr <= RfSr_value(1, ss_new):
                self.Curveon = 2
            else:
                self.Curveon = 1
                self.RefSr = RfSr_value(1, ss_new)


#A-85


        for i in range(f.Npoint):
            s = 10 ** f.AEV[i]
            if Stress + s > f.YieldSt[i]:
                ys = Stress + s
            else:
                ys = f.YieldSt[i]
            if f.Filled[i]:
                if ChangeSuction(f.YieldSt[i], s) == 0:
                    pass
                    #sgBox "ssdsdd"
                    print("ssdsdd")

                if (s / ChangeSuction(f.YieldSt[i], s)) <= ss_new:
                    if (Stress + (s / ChangeSuction(ys, s))) > f.YieldSt[i]:
                        f.YieldSt[i] = (s / ChangeSuction(ys, s)) + Stress
                    f.RVC[i] = Changevolume(f.RV[i], f.YieldSt[i], (s / ChangeSuction(ys, s)) + Stress, f.Ccp[i], f.Csp[i])
                    f.Filled[i] = False
                    f.Airentrapped[i] = True
                else:
                    #For pores that have AEV>ss_new - filled with water and subject to the same stress
                    if Stress + ss_new > f.YieldSt[i]:
                        f.YieldSt[i] = Stress + ss_new

                f.RVC[i] = Changevolume(f.RV[i], f.YieldSt[i], Stress + ss_new, f.Ccp[i], f.Csp[i])

            else:
                # Do not care about the pore that is already dried
                pass
        self.Suction = ss_new     # Change current suction to the new soil suction.


    def Wetting(self, ss_new):
        """Decrease soil suction at a certain net mean/vertical Stress

        """

        RfSr_value = self.RfSr_value
        ChangeSuction = self.ChangeSuction
        Changevolume = self.Changevolume
        ChangeWetSuction = self.ChangeWetSuction
        CVStress = self.CVStress()

        f = self.f
        Stress = self.Stress
        Assumption = self.Assumption
        RefSr = self.RefSr
        pm = self.pm

        if self.Curveon==1:
            if self.RefSr >= RfSr_value(3, ss_new):
                self.Curveon = 2
            else:
                self.Curveon = 3
                self.RefSr = RfSr_value(3, ss_new)
        elif self.Curveon==2:
            if self.RefSr >= RfSr_value(3, ss_new):
                self.Curveon = 2
            else:
                self.Curveon = 3
                self.RefSr = RfSr_value(3, ss_new)
        elif self.Curveon==3:
            self.Curveon = 3
            self.RefSr = RfSr_value(3, ss_new)

        for i in range(f.Npoint):
            s = 10 ** f.WEV[i]
            if not f.Filled[i]:     # Check if the pore group is filled with water
                if Stress + s > f.YieldSt[i]:
                    ys = Stress + s
                else:
                    ys = f.YieldSt[i]
                if (s / ChangeWetSuction(ys, 10 ** f.AEV[i], 10 ** f.WEV[i])) >= ss_new:      # Check if the pore group is possible to be filled with water after suction decreased to ss_new
                    if (s / ChangeWetSuction(ys, 10 ** f.AEV[i], 10 ** f.WEV[i])) + Stress > f.YieldSt[i]:
                        #MsgBox "never happend:" + Str(f.YieldSt[i]) + "     <  " + Str((s / ChangeWetSuction(ys, 10 ** f.AEV[i], 10 ** f.WEV[i])) + Stress)
                        f.YieldSt[i] = (s / ChangeWetSuction(ys, 10 ** f.AEV[i], 10 ** f.WEV[i])) + Stress
#A-86
                    f.RVC[i] = Changevolume(f.RV[i], f.YieldSt[i], Stress + ss_new, f.Ccp[i], f.Csp[i])
                    f.Filled[i] = True
                else:
                    # Pores are continually empty (WEV < ss_new) ...those pores are dry and still dry after the wetting process
                    f.Filled[i] = False
                    if (Assumption == 1) and (f.Ccp[i] > 0):
                        tmp = 10 ** f.AEV[i]
                        if (Stress > 10 ** f.AEV[i]):
                            tmp = CVStress * 10 ** (((log10(Stress / 10 ** f.AEV[i]) * ((RefSr ** pm) * (f.Ccp[i] - f.Ccd[i]) + f.Ccd[i])) / f.Ccp[i]) + f.AEV[i])

                        if tmp > f.YieldSt[i]:
                            f.YieldSt[i] = tmp
                            f.RVC[i] = Changevolume(f.RV[i], f.YieldSt[i], f.YieldSt[i], f.Ccp[i], f.Csp[i])

            else:
                # For all pores that has WEV> current suction (..suction..) - AEV, WEV value are the same...pore increase in volume due to decrease in (suction+stress)
                # Yield stresses are the same
                if Stress + ss_new > f.YieldSt[i]:
                    f.YieldSt[i] = Stress + ss_new
                f.RVC[i] = Changevolume(f.RV[i], f.YieldSt[i], Stress + ss_new, f.Ccp[i], f.Csp[i])
        self.Suction = ss_new     # Change current suction to the new soil suction.


    def Loading(self, st_new):
        """Increase net mean stress at a certain soil suction"""

        RfSr_value = self.RfSr_value
        ChangeSuction = self.ChangeSuction
        Changevolume = self.Changevolume
        ChangeWetSuction = self.ChangeWetSuction

        f = self.f
        Stress = self.Stress
        Suction = self.Suction
        Assumption = self.Assumption
        RefSr = self.RefSr
        pm = self.pm
        CVStress = self.CVStress()

        for i in range(f.Npoint):
            s = 10 ** f.WEV[i]        # set variable s = water entry value of the pore on reference WPD

            if st_new + s > f.YieldSt[i]:
                ys = st_new + s
            else:
                ys = f.YieldSt[i]

            if f.Filled[i]:   # For pores that are currently filled with water
                if st_new + Suction > f.YieldSt[i]:
                    f.YieldSt[i] = st_new + Suction
                f.RVC[i] = Changevolume(f.RV[i], f.YieldSt[i], Suction + st_new, f.Ccp[i], f.Csp[i])

            else:  # If pores are not filled with water...
                if (s / ChangeWetSuction(ys, 10 ** f.AEV[i], s)) > Suction: # We have to check if WEV of the pore < current suction...then will be filled with water
                    f.YieldSt[i] = ys                                            # Group of pores must be wetted before apply a load
                    f.RVC[i] = Changevolume(f.RV[i], f.YieldSt[i], Suction + st_new, f.Ccp[i], f.Csp[i])

                    f.Filled[i] = True
                else:
                    # For all pores that has WEV> current suction (..suction..) - AEV, WEV value are the same...pore increase in volume due to decrease in (suction+stress)
                    # Yield stresses are the same
                    if (Assumption == 1) and (f.Ccp[i] > 0):
                        tmp = 10 ** f.AEV[i]
                        if Stress > 10 ** f.AEV[i]:
                            tmp = CVStress * 10 ** (((log10(st_new / 10 ** f.AEV[i]) * ((RefSr ** pm) * (f.Ccp[i] - f.Ccd[i]) + f.Ccd[i])) / f.Ccp[i]) + f.AEV[i])
                            #'If Stress < tmp Then
                            #'  MsgBox "Stress = " + Str(Stress) + "    Yield stress = " + Str(tmp)
                            #'End If
                        if tmp > f.YieldSt[i]:
                            f.YieldSt[i] = tmp
                            f.RVC[i] = Changevolume(f.RV[i], f.YieldSt[i], f.YieldSt[i], f.Ccp[i], f.Csp[i])
# A-87

                    #f.Filled[i] = False
                    #f.AEVC[i] = log10(10 ** f.AEV[i] / ChangeSuction(f.YieldSt[i], 10 ** f.AEV[i]))
                    #f.WEVC[i] = log10(10 ** f.WEV[i] / ChangeSuction(f.YieldSt[i], 10 ** f.AEV[i]))
        self.Stress = st_new # Change current suction to the new stress.

    def Unloading(self, st_new):
        """Decrease net mean stress at a certain soil suction"""


        Changevolume = self.Changevolume

        f = self.f
        Suction = self.Suction

        for i in range(f.Npoint):

            s = 10 ** f.WEV[i]
            ys = f.YieldSt[i]
            if f.Filled[i]: #For the pore that is filled with water
                f.RVC[i] = Changevolume(f.RV[i], f.YieldSt[i], Suction + st_new, f.Ccp[i], f.Csp[i])
#            End If
#            Next i
        self.Stress = st_new     # Change current suction to the new stress.

    def DegreeofsaturationSWCC(self):
        """This procedure is used to calculate Srdry and Srwet"""

        NumSr = self.NumSr
        MaxSuction = self.MaxSuction
        MinSuction = self.MinSuction
        Gs = self.Gs



        self.SlurryPoreSize()       # Reset the soil to initial slurry condition
        f = self.f
#        frmFlash.Show (vbModeless)
        ct = self.Stress          # cs = current stress
        cs = self.Suction         # ct = current suction
        intv = (MaxSuction - MinSuction) / NumSr  # Take equal interval in log scale
#        frmFlash.lbprogress.Caption = "Please wait, calculating Ref Drying S% SWCC..."
#        '''frmFlash.Refresh
        for i in range(NumSr): #' along the drying process
            cs = 10 ** (intv * (i - 0) + MinSuction)
            self.Drying(cs)
            tmp1 = 0
            tmp2 = 0
            for k in range(f.Npoint):
                tmp1 = tmp1 + f.RVC[k]
                if f.Filled[k]:
                    tmp2 = tmp2 + f.RVC[k]
            self.Srdry[i] = tmp2 / tmp1


#  frmFlash.lbprogress.Caption = "Please wait, calculating Ref Wetting S% SWCC..."
#  '''frmFlash.Refresh
        for i in range(NumSr): # along the wetting process
            cs = 10 ** (MaxSuction - intv * (i - 0))
            self.Wetting(cs)
            tmp1 = 0
            tmp2 = 0
            for k in range(f.Npoint):
                tmp1 = tmp1 + f.RVC[k]
                if f.Filled[k]:
                    tmp2 = tmp2 + (f.RVC[k] / Gs)
            self.Srwet[NumSr - 1 - i] = tmp2 * Gs / tmp1
#    '''frmFlash.Refresh
#      End Sub

    def RfSr_value(self, curvetype, ssvalue):
        """
        Parameters
        ----------
        Curvetype : int
            curvetype=1: drying; = 2: scanning; =3: wetting
        ssvalue : double
            Don't know

        Returns
        -------
        out : float
            Reference saturation value

        """



        NumSr = self.NumSr
        MaxSuction = self.MaxSuction
        MinSuction = self.MinSuction

        if curvetype == 1:
            return self.Srdry[int((log10(ssvalue) - MinSuction) / ((MaxSuction - MinSuction) / (NumSr - 1)))]
        if curvetype == 3:
            return self.Srwet[int((log10(ssvalue) - MinSuction) / ((MaxSuction - MinSuction) / (NumSr - 1)))]
#'A-88

    def _Calresults_scalar(self):

        stp = self.stp
        Drying = self.Drying
        Wetting = self.Wetting
        Loading = self.Loading
        Unloading = self.Unloading
        CVStress = self.CVStress()

        beta = self.beta
        Gs = self.Gs

        self.errocc = False
#        Initiate
        if self.Assumption == 1:
            self.DegreeofsaturationSWCC()
        else:
            #frmFlash.Show
            pass
        self.SlurryPoreSize()
        ct = self.Stress          #' = current stress
        cs = self.Suction         #' = current suction

        f = self.f
        stp.n = -1

        for i in range(stp.nsteps):
            if not stp.ist[i]:
                stp.n = stp.n + 1
                intv = (log10(stp.vl[i]) - log10(cs)) / (stp.npp[i] - 1) #' Take equal interval in log scale
                #stp.startpoint(stp.n) = datapoint.n + 1
                datapoint = stp.datapoints[i]
                datapoint.n = -1
                for j in range(stp.npp[i]):
                    datapoint.n = datapoint.n + 1
                    #frmFlash.lbprogress.Caption = "Calculating data point #" + Str[datapoint.n]
                    #'''frmFlash.Refresh
                    datapoint.ss[datapoint.n] = 10 ** ((j - 0) * intv + log10(cs))

                    datapoint.st[datapoint.n] = ct
                    if intv > 0:
                        Drying(datapoint.ss[datapoint.n])
                    else:
                        Wetting(datapoint.ss[datapoint.n])
                    tmp1 = 0
                    tmp2 = 0
                    for k in range(f.Npoint):
                        tmp1 = tmp1 + f.RVC[k]
                        if f.Filled[k]:
                            if f.Airentrapped[k]:
                                tmp2 = tmp2 + (f.RVC[k] / Gs) * (1 - beta)
                            else:
                                tmp2 = tmp2 + (f.RVC[k] / Gs)

                    datapoint.e[datapoint.n] = tmp1
                    datapoint.w[datapoint.n] = tmp2

                cs = stp.vl[i]

            if stp.ist[i]:
                stp.n = stp.n + 1
                intv = (log10(stp.vl[i]) - log10(ct)) / (stp.npp[i] - 1)

                #' Take equal interval in log scale
                #stp.startpoint(stp.n) = datapoint.n + 1
                datapoint = stp.datapoints[i]
                datapoint.n = -1
                for j in range(stp.npp[i]):
                    datapoint.n = datapoint.n + 1
                    #frmFlash.lbprogress.Caption = "         Calculating data point #" + Str[datapoint.n]
                    #'''frmFlash.Refresh
                    datapoint.ss[datapoint.n] = cs
                    datapoint.st[datapoint.n] = 10.0 ** ((j - 0) * intv + log10(ct))
                    if intv > 0:
                        Loading(datapoint.st[datapoint.n] * CVStress)

                    else:
                        Unloading(datapoint.st[datapoint.n] * CVStress)

                    tmp1 = 0
                    tmp2 = 0
                    for k in range(f.Npoint):
                        tmp1 = tmp1 + f.RVC[k]

#'A-89
                        if f.Filled[k]:
                            if f.Airentrapped[k]:
                                tmp2 = tmp2 + (f.RVC[k] / Gs) * (1 - beta)
                            else:
                                tmp2 = tmp2 + (f.RVC[k] / Gs)

                    datapoint.e[datapoint.n] = tmp1
                    datapoint.w[datapoint.n] = tmp2
                ct = stp.vl[i]
                #stp.endpoint(stp.n) = datapoint.n


        for datapoint in stp.datapoints:

            for i in range(datapoint.npts):
                if datapoint.ss[i] >= 999999: datapoint.ss[i] = 999998
                if datapoint.st[i] >= 999999: datapoint.st[i] = 999998
                datapoint.Sr[i] = datapoint.w[i] * Gs / datapoint.e[i]
                datapoint.vw[i] = datapoint.Sr[i] * datapoint.e[i] / (datapoint.e[i] + 1)

#            Unload frmFlash
#  Call output_datapoint(Application.ActiveSheet)
        if self.errocc:
            #MsgBox " Input data is not valid, please check the PORE-SHAPE PARAMETER"
            print("Input data is not valid, please check the PORE-SHAPE PARAMETER")

    def _Calresults_fortran(self):

        epus_ext.epus.dealloc()

        epus_ext.epus.nsteps = self.stp.nsteps
#        ALLOCATE(ist(1:nsteps))
#        ALLOCATE(vl(1:nsteps))
#        ALLOCATE(npp(1:nsteps))
#        ist(1:5) = (/.true., .true., .false., .false., .false./)
#        npp(1:5) = 100
#        vl(1:5) = (/ 20._DP, 1._DP, 1.E6_DP, 30._DP, 1500._DP/)
        epus_ext.epus.ist = self.stp.ist
        epus_ext.epus.vl = self.stp.vl
        epus_ext.epus.npp = self.stp.npp

        epus_ext.epus.wsat = self.SimpleSWCC.wsat
        epus_ext.epus.a = self.SimpleSWCC.a
        epus_ext.epus.b = self.SimpleSWCC.b
        epus_ext.epus.wr = self.SimpleSWCC.wr
        epus_ext.epus.sl = self.SimpleSWCC.sl
        epus_ext.epus.logds= self.logDS
        epus_ext.epus.logrs = self.logRS
        epus_ext.epus.ccs = self.Ccs
        epus_ext.epus.css = self.Css
        epus_ext.epus.ccd = self.Ccd
        epus_ext.epus.gs = self.Gs
        epus_ext.epus.assumption = self.Assumption
        epus_ext.epus.beta = self.beta
        epus_ext.epus.pm = self.pm
        epus_ext.epus.pore_shape = self.Pore_shape
        epus_ext.epus.k0 = self.K0
#        epus_ext.epus.soilname = self.soilname #fortran code *len messes everything up
#        epus_ext.epus.username = self.username #fortran code *len messes everything up
        epus_ext.epus.numsr = self.NumSr
        epus_ext.epus.npoint = self.Npoint
        epus_ext.epus.maxsuction = self.MaxSuction
        epus_ext.epus.minsuction = self.MinSuction

        epus_ext.epus.calresults()

        # put Fortran results (long 1d array) back into stp (multiple 1d arrays for each load step)
        for i in range(self.stp.nsteps):
            start = epus_ext.epus.startpoint[i] - 1 # Fortran indexing starts at 1
            end = epus_ext.epus.endpoint[i]

            self.stp.datapoints[i].ss[:] = epus_ext.epus.ss[start:end]
            self.stp.datapoints[i].st[:] = epus_ext.epus.st[start:end]
            self.stp.datapoints[i].e[:] = epus_ext.epus.ee[start:end]
            self.stp.datapoints[i].w[:] = epus_ext.epus.w[start:end]
            self.stp.datapoints[i].Sr[:] = epus_ext.epus.sr[start:end]
            self.stp.datapoints[i].vw[:] = epus_ext.epus.vw[start:end]


    def Calresults(self):
        """Calculate the results ie calculate the response to the stress path

        """

        if self.implementation == 'scalar':
            self._Calresults_scalar()
        else:
            if _SUCCESSFUL_FORTRAN_IMPORT:
                self._Calresults_fortran()
            else:
                self._Calresults_scalar()




class PoresizeDistribution(object):
    """Pore size distribution curve

    Basically a container for data.

    Parameters
    ----------
    Npoint : int, optional
        Number of data points along the pore size distribution curve.
        Default Npoint=1000

    Attributes
    ----------
    Npoint : int
        Number of data points along the pore size distrubution curve.
    AEV : 1d array of float
        Value of Air Entry Value that the function F represents.
    WEV : 1d array of float
        Value of Water Entry Value that the function F represents.
    Ccp : 1d array of float
        Virgin compression index of the group of pore at saturated.
    Csp : 1d array of float
        Unloading-Reloading index of the group of pore at saturated.
    Ccd : 1d array of float
        Virgin compression index of the group of pore at completely dry
        condition (10^6 kPa).
    RV : 1d array of float
        Ratio of Volume of the group of pore/ volume of solid phase.
    YieldSt : 1d array of float
        Equivalent maximum stress acted on the pores at water-filled state.
    Filled : 1d array of bool
        State of the pore - either filled or empty.
    RVC : 1d array of float
        Actual ratio of volume of the group of pore/ volume of solid phase.
    AEVC : 1d array of float
        Actual Air Entry Value of the group of pores [i] which has AEV at
        dried slurry = AEV[i].
    WEVC : 1d array of float
        Actual water entry value.
    Airentrapped : 1d array of bool
        If experienced dried state = True, if did not = False.


    """


    def __init__(self, Npoint=1000):
        self.Npoint = Npoint
        self.AEV = np.zeros(Npoint)
        self.WEV = np.zeros(Npoint)
        self.Ccp = np.zeros(Npoint)
        self.Csp = np.zeros(Npoint)
        self.Ccd = np.zeros(Npoint)
        self.RV = np.zeros(Npoint)
        self.YieldSt = np.zeros(Npoint)
        self.Filled = np.zeros(Npoint, dtype=bool)
        self.RVC = np.zeros(Npoint)
        self.AEVC = np.zeros(Npoint)
        self.WEVC = np.zeros(Npoint)
        self.Airentrapped = np.zeros(Npoint, dtype=bool)




class CurveFittingSWCC(object):
    """Curve fitting soil water characteristic curve for collapsible pores.
    
    i.e. wc(suction)

    Note that that when suction is less than 1, results are not meaningful.
    At low suction values we expect wc approx equal to wsat-wr, but the 
    sl * log10(suction) /Gs term gives large negative numbers for suctions 
    less than 1, whereas at suction =1 it dissapears as expected.
    
    
    Parameters
    ----------
    wsat : float
        Gravimetric water condent at 100% saturation.
    a : float
        Curve fitting parameter.
    b : float
        Curve fitting parameter.
    wr : float
        Residual gravimetric water content.
    sl : float
        Initial slope of SWCC*Gs. This is actually the saturated virgin 
        compressibility index Cc.

    Attributes
    ----------
    wsat : float
        Gravimetric water condent at 100% saturation.
    a : float
        Curve fitting parameter.
    b : float
        Curve fitting parameter.
    wr : float
        Residual gravimetric water content.
    sl : float
        Initial slope of SWCC.

    """

    def __init__(self, wsat, a, b, wr, sl):

        self.wsat = wsat
        self.a = a
        self.b = b
        self.wr = wr
        self.sl = sl


    def Wc(self, s, Gs=2.7):
        """Collapsible water content function

        Parameters
        ----------
        s : float
            Suction.
        Gs : float, optional
            Specific gravity of solids. Default Gs=2.7

        Returns
        -------
        out : float
            water content Wc

        Examples
        --------
        >>> SWCC = CurveFittingSWCC(wsat=0.262,
        ...                        a = 3.1 * 10 ** 6,
        ...                        b = 3.377,
        ...                        wr = 0.128,
        ...                        sl = 0.115)
        >>> SWCC.Wc(500, Gs=2.7)
        4.525...e-05


        """


        wsat1 = self.wsat
        wr1 = self.wr
        b1 = self.b
        a1 = self.a
        sl1 = self.sl
        Wc_ = ((wsat1 - sl1 * log10(s) / Gs - wr1) * a1 *
               (1 - s ** b1 / 10 ** (6 * b1)) / (s ** b1 + a1))
        return Wc_


    def DWc(self, s, Gs=2.7):
        """Derivative of collapsible water content function

        Parameters
        ----------
        s : float
            Suction.
        Gs : float, optional
            Specific gravity of solids. Default Gs=2.7

        Returns
        -------
        out : float
            derivative of water content Wc w.r.t. suction

        Examples
        --------
        >>> SWCC = CurveFittingSWCC(wsat=0.262,
        ...                        a = 3.1 * 10 ** 6,
        ...                        b = 3.377,
        ...                        wr = 0.128,
        ...                        sl = 0.115)
        >>> SWCC.DWc(500, Gs=2.7)
        -2.367...e-05


        """


        wsat1 = self.wsat
        wr1 = self.wr
        b1 = self.b
        a1 = self.a
        sl1 = self.sl
        temp1 = wsat1 - sl1 * log(s) / (log(10) * Gs) - wr1
        temp2 = -a1 * b1 * s ** b1 * (10 ** (-6 * b1) * a1 + 1) / (s * (s ** b1 + a1) ** 2)
        temp3 = (a1 - a1 * s ** b1 / (10 ** (6 * b1))) / (s ** b1 + a1)
        temp4 = sl1 / (s * log(10) * Gs)
        temp5 = 1 - (log(1 + s / a1 ** (1 / b1)) / log(1 + 10 ** 6 / a1 ** (1 / b1)))
        temp6 = a1 ** (1 / b1) * (1 + s / a1 ** (1 / b1)) * log(1 + 10 ** 6 / a1 ** (1 / b1))
        DWc_ = (temp1 * temp2 - temp3 * temp4) * temp5 - (temp3 * temp1 + wr1) / temp6

        return DWc_



class StressPath(object):
    """Stress paths to follow

    Parameters
    ----------
    path_dicts : list of dict
        Each dict describes a step in the stress path. Each dict in the list
        has the following.

        ==================  ============================================
        key                 description
        ==================  ============================================
        vl                  Value of net normal stress or suction
                            at end of stress path step.
        ist                 bool. Optional. If change in
                            stress...ist=True else ist=False. Default
                            ist=True.
        npp                 int. Optional. Number of points to subdivide
                            stress path step into. default npp=100.
        name                str. Optional. Description of step. Default
                            name='Step 0' etc.
        ==================  ============================================


    Attributes
    ----------
    ist : 1d array of bool
        Loading type for each step in stress path. True indicates a net
        normal stress change. False indicates a suction change.
    vl : 1d array of float
        Value of stress or suction at end of step.
    npp : 1d array of int
        Number of points to subdivide each stress path step into.
    name : list of str
        Name of each stress path step
    n : int
        Counter indicating current subdivision.
    nsteps : int
        Number of stress path steps. i.e. len(path_dicts)
    datapoints : list of DataResults objects
        Results data for each stress path step. (ss, st, e, w, sr, vw)
    datapoints_combined : DataResults object
        datapoints joined end to end. e.g. access ss from all steps using
        datapoints_combined.ss  (only exists after running combine_datapoints).

    """
    def __init__(self, path_dicts):
        self.n = 0

        self.nsteps = len(path_dicts)
        self.ist = np.empty(self.nsteps, dtype=bool)
        self.vl = np.empty(self.nsteps)
        self.npp = np.empty(self.nsteps, dtype=int)

        self.name = []
        for i, d in enumerate(path_dicts):
            self.ist[i] = d.get('ist', True)
            self.vl[i] = d['vl']
            self.npp[i] = d.get('npp', 100)

            name = d.get('name', None)
            if not name is None:
                self.name.append(name)
            else:
                if self.ist[i]:
                    name = ("Step #{:s}, stress change to "
                            "{:g} kPa.".format(str[i].zfill(3),
                                          self.vl[i]))

                else:
                    name = ("Step #{:s}, suction change to "
                            "{:g} kPa.".format(str[i].zfill(3),
                                          self.vl[i]))
                self.name.append(name)


        self.datapoints = [DataResults(npp) for npp in self.npp]


    def _print_steps(self):
        for i in range(self.nsteps):
            print(i, self.ist[i], self.vl[i], self.npp[i], self.name[i])

    def combine_datapoints(self):
        """Make datapoints_combined i.e. join all the datapoints"""

        self.datapoints_combined = sum(self.datapoints)


class DataResults(object):
    """Container for stress, suction, void ratio, water content, saturation.

    Parameters
    ----------
    npts : int
        Number of data points.

    Attributes
    ----------
    npts : int
        Numer of data points.
    n : int
        Counter
    st : 1d array of float
        Net normal stress. len is npts
    ss : 1d array of float
        Suction.
    e : 1d array of float
        Void Ratio
    w : 1d array of float
        gravimetric water content
    Sr : 1d array of float
        Degree of saturation
    vw : 1d array of float
        Volumetric water content.

    """

    def __init__(self, npts):

        self.npts = npts
        self.ss = np.zeros(npts, dtype=float)
        self.st = np.zeros(npts, dtype=float)
        self.e = np.zeros(npts, dtype=float)
        self.w = np.zeros(npts, dtype=float)
        self.Sr = np.zeros(npts, dtype=float)
        self.vw = np.zeros(npts, dtype=float)

        self._attr = ['ss', 'st', 'e', 'w', 'Sr', 'vw']

    def __add__(self, other):
        """Join two DataResults objects into one

        Parameters
        ----------
        other : DataResults object
            DataResults object to add onto existing object.

        Returns
        -------
        out : DataResults object
            out.npts = self.npts + other.npts
        """

        if not isinstance(other, DataResults):
            raise TypeError('other is not a DataResults object.')

        out = DataResults(self.npts + other.npts)



        for v in self._attr:
            getattr(out, v)[:self.npts] = getattr(self, v)
            getattr(out, v)[self.npts:] = getattr(other, v)

        return out

    def __radd__(self, other):

        if other == 0:
            return self
        else:
            return self.__add__(other)




class EpusProfile(object):
    """1D initial conditons (stress etc.) distribution for EOUS unsaturated
    soil model.


    Proceedure:

     - Determine pore water pressure, uw, at depth based on depth of
       water table, zw,  i.e. uw is negative above water table. This
       gives suction (psi) profile above water table.
     - Guess an initial stress (sig) distribution, including an initial
       surcharge of q0.
     - Assume a stress path starting from slurry to stress increase to
       sig and then suction to psi.  Use EPUS to calculate void ratio (e)
       and saturation (Sr).
     - Use Gs, e, Sr to calc unit weight (gamma) profile.
       Numerically integrate unit weight profile to with sig at z=0 equals q0
       to give new stress profile.
     - Check convergence of new stress vs old stress profile.  If not equal
       then iterate.
     - Once solutin converged, record stress, void ratio, saturation profiles.



    Note initial EPUS loading steps begin from 0 net normal stress (sig-ua).
    Below the water table there is a bouyant force so the sig-ua stress input
    is actually an effective stress.



    Parameters
    ----------
    epus_object : instance of EPUS object
        An EPUS object from which all relevant paramters can be taken.
        Use a dummy stress path for this; it will be ignored.  Also Npoint
        will be ignored.  Basically just using and EPUS object as a
        convienient container.
    H : float
        Height of soil profile.
    zw : float, optional
        Depth of water table.  Default zw=None, i.e. will be set to H.
    q0 : float, optional
        Existing surcharge.  Default q0=1.
    gamw : float, optional
        Unit weight of water.  Default gamw=10.
    nz : int, optional
        Final number of depth values in profile. default nz=10.
    Npoint : int, optional
        Number of data points along the pore size distribution curve.
        Default Npoint=1000
    npp : float, optional
        Number of points per stress path step.  Default npp=100
    nz_refine : list of float, optional
        Allows progressive refinement of the number of points in the profile.
        Initially stress profile will be calcuated at int(nz*nz_refine[0])
        evenly spaced depth values. Once convergence is reached
        a new stress profile will be interpolated at int(nz*nz_refine[1])
        and the convergence process is run again.  The refinement is repeated
        untill the final value of nz_refine (which should be 1) is used.
        Default nz_refine=[1] i.e. no successvie refinement.
    Npoint_refine : list of float, optional
        Similar to nz_refine.  Last value whould be 1. Sould be same length as
        nz_refine. Default Npoint_refine=[1]
    npp_refine : list of float
        Similar to nz_refine. Last value should be 1.  Should be same length as
        nz_refine.  Default npp_refine = [1]
    rtol, atol : float, optional
        Relative and absolute tolerance for checking convergence of stress.
        See numpy.allclose . Default atol=0.01, rtol=1e-6
    max_iter : int, optional
        Maxmum number of convergence iterations at each refinement step.
        Default max_iter=100 .
    initial_stress : tuple of two 1d arrays
        Initial stress to use to start of iteration.
        1st element of tuple is list/array of z values, 2nd element is
        list/array. Default intial_stress=None i.e. make up an initial guess
        using unit weight = 15.  Values will be interpolated.




    Attributes
    ----------
    _nz, _Npoint, _npp : int
        Current 'refined' value of nz, Npoint, and _npp.
    profile : DataResults object
        Contains information about each data point in the profile.
        In addition to normal DataResults attributes ( ss, st, e, w, Sr, vw)
        it has unit weight (gam), depth (z), pore water pressure (uw)
    _epus_dict : dictionary
        dict containing common keywords from the epus_object to initialize
        other epus objects at each depth.  Basically just need to add
        Npoints, and stp stress path objects at each depth and refinement.
    niter : list of int
        number of iterations to converge at each refinement level.

    Notes
    -----
    "stress" below water table is effective stress.  "stress" avbove water
    table is net normal stress.

    """



    def __init__(self,
                 epus_object,
                 H,
                 zw=None,
                 q0=1.0,
                 gamw=10,
                 nz=10,
                 Npoint=1000,
                 npp=100,
                 nz_refine=[1],
                 Npoint_refine=[1],
                 npp_refine=[1],
                 atol=0.01,
                 rtol=1e-6,
                 max_iter=100,
                 initial_stress=None):


        self.epus_object = epus_object
        self.H = H
        if zw is None:
            self.zw = self.H
        else:
            self.zw = zw
        self.q0 = q0
        self.gamw = gamw
        self.nz = nz
        self.Npoint = Npoint
        self.npp = npp
        self.nz_refine = nz_refine
        self.Npoint_refine = Npoint_refine
        self.npp_refine = npp_refine
        self.atol = atol
        self.rtol = rtol
        self.max_iter = max_iter

        self.niter=[]
        self.initial_stress = initial_stress
        names = ['SimpleSWCC',
                 #'stp',
                 'logDS',
                 'logRS',
                 'Ccs',
                 'Css',
                 'Ccd',
                 'Gs',
                 'Assumption',
                 'beta',
                 'pm',
                 'Pore_shape',
                 'K0',
                 'soilname',
                 'username',
                 #'Npoint',
                 'NumSr',
                 'MaxSuction',
                 'MinSuction',]

        self._epus_dict = dict()
        for v in names:
            self._epus_dict[v] = getattr(self.epus_object, v)


        pass

    def _initialize_stress(self):
        """Guess initial stress distribution to begin iterations from"""
        if self.initial_stress is None:
            gam = 15.
            self.profile.st[:]=self.profile.z * gam + self.q0
            s = self.profile.uw>0
            self.profile.st[s] -= self.profile.uw[s]
        else:
            zi, si = self.initial_stress
            self.profile.st[:] = np.interp(self.profile.z,
                                      zi,
                                      si)



    def _refine(self):
        """Refine numerical parameters"""
        pass

    def _stress_from_overburden(self):
        """update stress profile based on q0 and overbuden (integrate density)"""

        self.profile.st[:] = self.q0
        dz = np.diff(self.profile.z)
        gamavg = (self.profile.gam[1:] + self.profile.gam[:-1]) * 0.5
        dsig = dz*gamavg

        self.profile.st[1:] += np.cumsum(dsig)

        #adjust for bouyant force
        s = self.profile.uw>0
        self.profile.st[s]-=self.profile.uw[s]



    def _update_density(self):
        """update density based on Gs, e and Sr"""




        Gs = self.epus_object.Gs
        gamw = self.gamw
        e = self.profile.e
        Sr = self.profile.Sr
        self.profile.gam[:] = (Gs+Sr*e)/(1+e)*gamw


    def _e_and_Sr_from_EPUS(self):
        """Use EPUS and sig and psi distribution to calc e and Sr"""



        for i in range(self.profile.npts):

            ss = self.profile.ss[i]
            st = max(self.profile.st[i], (10 ** self.epus_object.MinSuction) / 1000)
            if ss <= 10**self.epus_object.MinSuction:
                #only need stress increase stresspath
                stp = StressPath(
                     [dict(ist=True, npp=self._npp, vl=st, name="1. Stress increase"),])
            else:
                #need stress increase and suction increase stress path
                stp = StressPath(
                     [dict(ist=True, npp=self._npp, vl=st, name="1. Stress increase"),
                      dict(ist=False, npp=self._npp, vl=ss, name="2. Suction increase")])
            self._epus_dict["Npoint"]= self._Npoint
            self._epus_dict["stp"] = stp

            a = EPUS(**self._epus_dict)
            a.Calresults()

            for v in a.stp.datapoints[0]._attr:
                getattr(self.profile, v)[i] = (
                    getattr(a.stp.datapoints[-1], v)[-1])

        #adjust Sr
        s = self.profile.Sr > 1
        self.profile.Sr[s] = 1

    def _blank_profile(self, nz):
        """Modify a DataResults object to contain info about the profile

        Add z, gam, uw
        Calc uw from water table
        calc suction ss from negative values of uw (set suctino to zero
        elsewhere).

        """

        profile = DataResults(npts=nz)
        profile.z = np.linspace(0, self.H, nz)
        profile.gam = np.zeros(nz, dtype=float)
        profile.uw = (np.linspace(0, self.H, nz) - self.zw) * self.gamw
        profile.ss[:] = 0.0
        s = profile.uw<0
        profile.ss[s] = -profile.uw[s]

        profile._attr.extend(['z', 'gam', 'uw'])

        return profile



    def calc(self):
        """do all the calculations"""

        first = True
        for fnz, fNpoint, fnpp in zip(self.nz_refine,
                                      self.Npoint_refine,
                                      self.npp_refine):

            self._nz = int(fnz * self.nz)
            self._Npoint = int(fnz * self.Npoint)
            self._npp = int(fnz * self.npp)


            new_profile = self._blank_profile(nz=self._nz)
            if first:
                self.profile = new_profile
                self._initialize_stress()
                first = False
            else:

                #self.profile already exists
                #interpolate stres from old profile
                new_profile.st[:] = np.interp(new_profile.z,
                                              self.profile.z,
                                              self.profile.st)

                self.profile = new_profile


            for j in range(self.max_iter):
                old_stress = self.profile.st[:].copy()


                self._e_and_Sr_from_EPUS()
                self._update_density()
                self._stress_from_overburden()

                if np.allclose(old_stress, self.profile.st,
                               atol=self.atol, rtol=self.rtol):
                    break


            if j>=self.max_iter:
                raise ValueError("Maximum iterations reached while nz={}".format(self._nz))
            self.niter.append(j+1)


    def plot_profile(self, fig=None):

        if fig is None:
            fig = plt.figure(figsize=(14,5))

        namemap=dict(vw="$\\theta$",
                     Sr="$S_r$",
                     ss="$\\psi$",
                     st="$\\sigma$",
                     gam="$\\gamma$",
                     uw="$u_w$",
                     m1s="$m_{1}^s$",
                     m2s="$m_{2}^s$",
                     m1w="$m_{1}^w$",
                     m2w="$m_{2}^w$",
                     m1a="$m_{1}^a$",
                     m2a="$m_{2}^a$",
                     kw="$k_w$",
                     ka="$k_a$")


        axlims = dict(Sr=dict(left=0, right=1),
                      )

        not_plot = ['m1s', 'm2s', 'm1w', 'm2w', 'm1a', 'm2a', 'kw', 'ka']
        attr = self.profile._attr[:]
        excl = ['z']
        for v in excl:
            if v in attr: attr.remove(v)

        n = len(attr)

        first=True
        for i, v in enumerate(attr):
            if first:
                ax = fig.add_subplot(1,n,i+1)
            else:
                ax = fig.add_subplot(1,n,i+1)
            ax.plot(getattr(self.profile, v), self.profile.z,
                    marker='o')
            if 'm1s' in self.profile._attr:
                if v not in not_plot:
                    ax.plot(getattr(self._dsig_profile, v), self.profile.z,
                        marker='s')
                    ax.plot(getattr(self._dpsi_profile, v), self.profile.z,
                        marker='h')


            ax.set_xlabel(namemap.get(v, v))
            ax.get_xticklabels()[-1].set_visible(False)
            if first:
                ax.set_ylabel('z')
                first=False
            else:
                plt.setp(ax.get_yticklabels(), visible=False)

            ax.invert_yaxis()
            ax.set_xlim(**axlims.get(v, dict()))
            plt.setp(ax.get_xticklabels(), rotation=-90, horizontalalignment='center')

        fig.subplots_adjust(left=0.05, right=0.95, wspace=0, top=0.95, bottom=0.3)
        return fig



    def save_profile_to_file(self, fpath):
        """Save profile data to a csv file

        Parameters
        ----------
        fpath : str
            path of file to save to

        """

        #make X, i.e. put dataresults in one big array
        x = np.zeros((self.profile.npts, len(self.profile._attr)))

        for i, v in enumerate(self.profile._attr):
            x[:, i] = getattr(self.profile, v)[:]


        header = ",".join(self.profile._attr)
        np.savetxt(fname=fpath, X=x, fmt="%g", header=header,
                   delimiter=",", comments="")


    def compression_indexes(self, dsig=1.0, dpsi=1.0):
        """Calculate the compression indexes

        m1s, m2s, m1w, m2w, m1a, m2a

        Calculation is volume change indexes after a) net normal stress
        change of dsig, and b) or suction change of dpsi.

        m1a + m1w = m1s
        m2a + m2w = m2s

        Parameters
        ----------
        dsig : float, optional
            Change in net normal stress (sig-ua),  Default dsig=1.0
        dpsi : float, optional
            Change in suction (ua-uw)


        """

        #

        from copy import deepcopy



        self.dsig = dsig
        self.dpsi = dpsi

        profile_backup = deepcopy(self.profile)


        #dsig
        for i in range(self.profile.npts):
            stp=[]
            ss = self.profile.ss[i]
            st = max(self.profile.st[i], (10 ** self.epus_object.MinSuction) / 1000)
            if ss <= 10**self.epus_object.MinSuction:
                #only need stress increase stresspath
                stp.extend(
                     [dict(ist=True, npp=self._npp, vl=st, name="1. Stress increase"),])
            else:
                #need stress increase and suction increase stress path
                stp.extend(
                     [dict(ist=True, npp=self._npp, vl=st, name="1. Stress increase"),
                      dict(ist=False, npp=self._npp, vl=ss, name="2. Suction increase")])

            #increase stress by dsig
            st += dsig
            stp.append(dict(ist=True, npp=self._npp, vl=st, name="3. Stress increase"))

            stp = StressPath(stp)
            self._epus_dict["Npoint"]= self._Npoint
            self._epus_dict["stp"] = stp

            a = EPUS(**self._epus_dict)
            a.Calresults()

            for v in a.stp.datapoints[0]._attr:
                getattr(self.profile, v)[i] = (
                    getattr(a.stp.datapoints[-1], v)[-1])

        #adjust Sr
        s = self.profile.Sr > 1
        self.profile.Sr[s] = 1

        self._dsig_profile = deepcopy(self.profile)



        #dpsi
        self.profile = deepcopy(profile_backup) # reset to intitial stress profile

        for i in range(self.profile.npts):
            stp=[]
            ss = self.profile.ss[i]
            st = max(self.profile.st[i], (10 ** self.epus_object.MinSuction) / 1000)
            if ss <= 10**self.epus_object.MinSuction:
                #only need stress increase stresspath
                stp.extend(
                     [dict(ist=True, npp=self._npp, vl=st, name="1. Stress increase"),])
            else:
                #need stress increase and suction increase stress path
                stp.extend(
                     [dict(ist=True, npp=self._npp, vl=st, name="1. Stress increase"),
                      dict(ist=False, npp=self._npp, vl=ss, name="2. Suction increase")])
            self._epus_dict["Npoint"]= self._Npoint
            self._epus_dict["stp"] = stp

            #increase suction by dpsi
            uw = self.profile.uw[i]

            if uw < 0:
                #uw starts unsaturated and ends in unsaturated; apply full suction
                stp.append(dict(ist=False, npp=self._npp, vl=ss+dpsi,
                                name="3. uw decrease--> suction increase"))
            else:
                if uw - dpsi >= (10 ** self.epus_object.MinSuction):
                    #uw starts saturated and ends in saturated
                    #increase stress by dpsi
                    stp.append(dict(ist=True, npp=self._npp, vl=st+dpsi,
                                    name="3. uw decrease--> stress increase"))

                else:
                    #uw starts saturated and ends unsaturated
                    #apply uw-0 to effective stress increase then
                    #apply dpsi-uw to suction
                    stp.append(dict(ist=True, npp=self._npp, vl=st+uw,
                                    name="3. uw decrease--> partial stress increase"))
                    stp.append(dict(ist=False, npp=self._npp, vl=dpsi-uw,
                                    name="3. uw decrease--> partial suction increase"))




            stp = StressPath(stp)
            self._epus_dict["Npoint"]= self._Npoint
            self._epus_dict["stp"] = stp


            a = EPUS(**self._epus_dict)
            a.Calresults()

            for v in a.stp.datapoints[0]._attr:
                getattr(self.profile, v)[i] = (
                    getattr(a.stp.datapoints[-1], v)[-1])

        #adjust Sr
        s = self.profile.Sr > 1
        self.profile.Sr[s] = 1

        self._dpsi_profile = deepcopy(self.profile)



        self.profile = deepcopy(profile_backup)
        #calc compression indexes

        for v in ['m1s', 'm2s', 'm1w', 'm2w', 'm1a', 'm2a']:
            if v not in self.profile._attr:
                self.profile._attr.append(v)
            setattr(self.profile, v, np.zeros(self.profile.npts, dtype=float))

        self.profile.m1s[:] = (self._dsig_profile.e - self.profile.e)/(1+self.profile.e) / dsig
        self.profile.m2s[:] = (self._dpsi_profile.e - self.profile.e)/(1+self.profile.e) / dpsi

        self.profile.m1w[:] = (self._dsig_profile.Sr*self._dsig_profile.e - self.profile.Sr*self.profile.e)/(1+self.profile.e) / dsig
        self.profile.m2w[:] = (self._dpsi_profile.Sr*self._dpsi_profile.e - self.profile.Sr*self.profile.e)/(1+self.profile.e) / dpsi

        self.profile.m1a[:] = self.profile.m1s-self.profile.m1w
        self.profile.m2a[:] = self.profile.m2s-self.profile.m2w


    def water_permeability(self, e_ksat=None, npp_swcc=500):
        """Calculate the water permeability
        
        First saturated permeability at depth is determined based on e_ksat.
        Then unsaturated water permeability is determined by a) working 
        out a volumetric water content vs suction curve (i.e. epus stress
        path of load to stress and then dry to get a SWCC) and b) use _[1] to 
        numerically integrate and determine relative permeability at the 
        relevatnt profile suction value.
        
        Parameters
        ----------
        e_ksat : PermeabilityVoidRatioRelationship object
            PermeabilityVoidRatioRelationship object defining dependence of
            saturated permebility on void ratio.  
            Default e_ksat=ConstantPermeabilityModel(ka=1) i.e. constant 
            saturated permeability with value one.
        npp_swcc : int, optional
            number of sterss path intervals to use when determining the 
            volumetric water content vs suction curve that will be used to 
            calculate the relative permeability. Defulat npp_swcc=500
        
                    
        See also
        --------
        geotecha.constitutive_models.void_ratio_permeability : void ratio
            saturated permeability relationship.
        
        References
        ----------
        .. [1] Fredlund, D.G., Anqing Xing, and Shangyan Huang. 
               "Predicting the Permeability Function for Unsaturated 
               Soils Using the Soil-Water Characteristic Curve. 
               Canadian Geotechnical Journal 31, no. 4 (1994): 533-46. 
               doi:10.1139/t94-062.
        
        """
        if e_ksat is None:
            e_ksat = void_ratio_permeability.ConstantPermeabilityModel(ka=1.0)
                
        # saturated water permeability
        ksat = e_ksat.k_from_e(self.profile.e)

        # water permeability relative to saturated
        kr = np.zeros_like(ksat)
        for i in range(self.profile.npts):
            stp=[]
            ss = self.profile.ss[i]
            st = max(self.profile.st[i], (10 ** self.epus_object.MinSuction) / 1000)
            if ss <= 10**self.epus_object.MinSuction:
                #soil is saturated
                kr[i] = 1.0                
            else:
                #need stress increase and then drying stress path
                stp.extend(
                     [dict(ist=True, npp=self._npp, vl=st, name="1. Stress increase"),
                      dict(ist=False, npp=npp_swcc, vl=1e6, name="2. Suction increase")])

                stp = StressPath(stp)
                self._epus_dict["Npoint"]= self._Npoint
                self._epus_dict["stp"] = stp
        
                a = EPUS(**self._epus_dict)
                a.Calresults()
                        
                #have the vw vs psi cureve (SWCC), now calc krel                        
                x = a.stp.datapoints[-1].ss 
                y = a.stp.datapoints[-1].vw 
                
                kr[i] = swcc.kwrel_from_discrete_swcc(ss, x, y)                    
        
        #add "kw" to the profile                               
        for v in ['kw']:
            if v not in self.profile._attr:
                self.profile._attr.append(v)
            setattr(self.profile, v, np.zeros(self.profile.npts, dtype=float))                                                
            
        self.profile.kw[:] = ksat * kr
                                                            
    def air_permeability(self, e_ksat=None, qfit=0.5):
        """Calculate the Air permeability based on saturation
        
        ka = kd * (1-Sr)**0.5*(1-Sr**(1/qfit))**(2*qfit)
        
        kd = fn(e)
        
        
        Parameters
        ----------
        e_ksat : PermeabilityVoidRatioRelationship object
            PermeabilityVoidRatioRelationship object defining dependence of
            dry air permeability on void ratio.  
            Default e_ksat=ConstantPermeabilityModel(ka=1) i.e. constant 
            saturated permeability with value one.
        qfit : float, optional
            fitting parameter.  Generally between between 0 and 1. 
            Default qfit=1
        
                    
        See also
        --------
        geotecha.constitutive_models.void_ratio_permeability : void ratio
            saturated permeability relationship.
        
        References
        ----------
        .. [1] Ba-Te, B., Limin Zhang, and Delwyn G. Fredlund. "A General 
               Air-Phase Permeability Function for Airflow through Unsaturated 
               Soils." In Proceedings of Geofrontiers 2005 Cngress, 
               2961-85. Austin, Tx: ASCE, 2005. doi:10.1061/40787(166)29.
               
        """

        if e_ksat is None:
            e_ksat = void_ratio_permeability.ConstantPermeabilityModel(ka=1.0)
                
        # saturated water permeability
        kd = e_ksat.k_from_e(self.profile.e)

        # air permeability relative to saturated
        kr = np.zeros_like(kd)
        Sr = self.profile.Sr
#        kr = (1 - Sr)**0.5*(1 - Sr**(1 / qfit))**(2 * qfit)
        kr = swcc.karel_air_from_saturation(Sr, qfit=qfit)

         #add "ka" to the profile                               
        for v in ['ka']:
            if v not in self.profile._attr:
                self.profile._attr.append(v)
            setattr(self.profile, v, np.zeros(self.profile.npts, dtype=float))                                                
            
        self.profile.ka[:] = kd * kr                                                            
                                                
if __name__ =="__main__":
    import nose
#    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest', '--doctest-options=+ELLIPSIS'])
    from geotecha.plotting.one_d import save_figure
    if 1: # this takes upwards of 
        import time
        from datetime  import timedelta
        start = time.time()
        SWCC = CurveFittingSWCC(wsat=0.262,
                                a = 3.1 * 10 ** 6,
                                b = 3.377,
                                wr = 0.128,
                                sl = 0.115, # note that sl is Cc * Gs see Equation 3.74 in Pham(2005 PhD)
                                )


        stp = StressPath([dict(ist=True, npp=5, vl=20, name="1. Load to 20kPa"),])

        pdict=dict(
             SimpleSWCC=SWCC,
             stp=stp,
             logDS=0.6,
             logRS=2,
             Css=0.019,
             beta=0.1,
             soilname='Artificial silt',
             username='Hung Pham',
             Npoint=400,
             NumSr=1000)

        epus_object = EPUS(**pdict)

        initial_stress=(np.array([  0. ,   2.5,   5. ,   7.5,  10. ]),
                         np.array([  10.        ,   62.11010239,  115.33445849,  169.3380806 ,
                                   223.80793771]))
#        a = EpusProfile(epus_object, H=10, zw=10, q0=10,
#                        nz=10, Npoint=1000, npp=10,
#                        nz_refine=[1],
#                        Npoint_refine=[0.25],
#                        npp_refine=[0.5],
#                        max_iter=15, atol=1, initial_stress=initial_stress)
        a = EpusProfile(epus_object, H=10, zw=5, q0=10,
                        nz=20, Npoint=500, npp=10,
                        nz_refine=[0.2, 1],
                        Npoint_refine=[0.4, 1],
                        npp_refine=[0.2, 1],
                        max_iter=15, atol=0.1, initial_stress=initial_stress, )


        a.calc()
#           fpath = "C:\\Users\\Rohan Walker\\Documents\\temp\\profile.csv"
        fpath = "C:\\Users\\rohanw\\Documents\\temp\\profile.csv"
#        print(repr(a.profile.z))
#        print(repr(a.profile.st))
        elapsed = (time.time() - start); print(str(timedelta(seconds=elapsed)))
        a.compression_indexes(dsig=20, dpsi=20)
        a.water_permeability()#e_ksat = void_ratio_permeability.ConstantPermeabilityModel
        a.air_permeability()
        a.save_profile_to_file(fpath=fpath)
        print(a.niter)
        elapsed = (time.time() - start); print(str(timedelta(seconds=elapsed)))


        fig=a.plot_profile()
        save_figure(fig, os.path.join(os.path.dirname(fpath), 'myfig'))
#        fig.tight_layout()

        plt.show()

