from __future__ import division, print_function
import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt

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
                 MinSuction=-3):

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

        return self.SimpleSWCC.Wc(s, Gs=self.Gs)


    def DWc(self, s):

        return self.SimpleSWCC.DWc(s, Gs=self.Gs)


    def DryPoreSize(self):
        """

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
        """' = suction(0)/suction(p)"""


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


    def Calresults(self):
        """Calculate the results ie calculate the response to the stress path

        """

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
    """Curve fitting soil water characteristic curve

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
        Initial slope of SWCC.

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

        attr = ['ss', 'st', 'e', 'w', 'Sr', 'vw']

        for v in attr:
            getattr(out, v)[:self.npts] = getattr(self, v)
            getattr(out, v)[self.npts:] = getattr(other, v)

        return out

    def __radd__(self, other):

        if other == 0:
            return self
        else:
            return self.__add__(other)





if __name__ =="__main__":
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest', '--doctest-options=+ELLIPSIS'])

