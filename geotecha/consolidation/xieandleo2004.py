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
Module implementing



Module implementing 'Analytical Solutions of
       One-Dimensional Large Strain Consolidation of Saturated and
       Homogeneous Clays' as per Xie and Leo (2004)[1]_.

References
----------
.. [1] Xie, K. H., and C. J. Leo. 2004. 'Analytical Solutions of
       One-Dimensional Large Strain Consolidation of Saturated and
       Homogeneous Clays'. Computers and Geotechnics 31 (4): 301-14.
       doi:10.1016/j.compgeo.2004.02.006.




"""
from __future__ import print_function, division

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy.integrate as integrate


class XieAndLeo2004(object):
    """Large strain analytical one-dimensional consolidation

    Parameters
    ----------
    qp : float
        existing load
    qp : float
        instant applied load
    H : float
        initial thickness of clay layer
    Hw : float
        Height of water surface above initial surface
    kv0 : float
        coefficient of vertical permeability
    mvl : float
        coefficeint of volume compressibility
    e00 : float
        initial void ratio at surface
    Gs : float
        specific gravity of solids
    gamw : float, optional
        unit weight of water. default gamw=10
    drn : [0,1]
        drainage condition.  drn=0 is PTPB, drn=1 is PTIB, default=0
    nterms : int, optional
        number of summation terms. default nterms=100


    Notes
    -----
    Basically initialize the XieAndLeo2004 object, then use individual methods
    of the data to extract data at particualr dpths and times.

    The most common error is if input data is not numpy arrays.

    """
    def __init__(self, qu, qp, H, Hw, kv0, mvl, e00, Gs, gamw=10, drn=0, nterms=100):

        self.qu = qu
        self.qp = qp
        self.H = H
        self.Hw = Hw
        self.kv0 = kv0
        self.mvl = mvl
        self.e00 = e00
        self.Gs = Gs
        self.gamw = gamw
        self.drn = drn
        self.nterms = nterms


        self._derived_parameters()

    def _derived_parameters(self):
        """calculate parameters that derive from the input parameters

        """
        self.M = np.pi * (np.arange(self.nterms) + 0.5)

#        if drn==1:
#            self.M = np.pi * (np.arange(self.nterms) + 0.5)
#        else:
#            self.M = np.pi * np.arange(1, self.nterms + 1.0)

        self.cv0 = self.kv0 / (self.mvl * self.gamw)

        self.dTv = self.cv0 / self.H**2

    def Tv(self, t):
        """Calculate vertical time factor

        Parameters
        ----------
        t : array-like of float
            time(s)

        Returns
        -------
        Tv : float
            Time factor

        """


        return self.dTv * t

    def e0(self, a):
        """Initial void ratio at depth

        Paramters
        ---------
        a : array like of float
            depth coord

        Returns
        -------
        e0 : float
            initial void ratio at depth a

        """

        e0 = self.e00 -self.mvl * self.gamw * (self.Gs - 1) * a

        return e0

    def efinal(self, a):
        """final void ration ratio at depth

        Paramters
        ---------
        a : array like of float
            depth coord

        Returns
        -------
        efinal : float
            final void ratio at depth a

        """

        mvl = self.mvl
        qu = self.qu

        e0 = self.e0(a)

        efinal = (1 + e0) * np.exp(-mvl * qu) - 1

        return efinal

    def settlement_final(self):
        """final settlement of clay layer"""
        return self.H * (1 - np.exp(-self.mvl * self.qu))

    def initial_effective_stress(self, a):
        """initial effective stress

        Paramters
        ---------
        a : array like of float
            depth coord

        Returns
        -------
        eff0 : float
            initial effective stress at depth a

        """

        qp = self.qp
        e00 = self.e00
        mvl = self.mvl
        Gs = self.Gs
        gamw = self.gamw

        f = qp + 1/mvl * np.log((1+e00)/(1+e00 - mvl * gamw * (Gs-1) * a))

        return f


    def u_PTIB(self, a, t):
        """pore pressure for PTIB drainage

        Paramters
        ---------
        a : array like of float
            depth coord
        t : array like of float
            time coord

        Returns
        -------
        u : float of size (len(a), len(t))
            excess pore pressure at depth a, time t

        """
#        a = np.atleast_1d(a)

        Tv = self.Tv(t)[None, :, None]
        a = (a/ self.H)[:, None, None]

        mvl = self.mvl
        qu = self.qu
        M = self.M[None, None, :]

        f = 2 / M * np.sin(M * a) * np.exp(-M**2 * Tv)
        f = np.sum(f, axis=2)
        f *= np.exp(mvl * qu) - 1
        f += 1
        np.log(f, out=f)
        f /= mvl

        return f
    def u_PTPB(self, a, t):
        """pore pressure for PTPB drainage

        Paramters
        ---------
        a : array like of float
            depth coord
        t : array like of float
            time coord

        Returns
        -------
        u : float of size (len(a), len(t))
            excess pore pressure at depth a, time t

        """

        Tv = self.Tv(t)[None, :, None]
        a = (a/ self.H)[:, None, None]

        mvl = self.mvl
        qu = self.qu
        M = self.M[None, None, :]

        f = 2 / M * np.sin(2 * M * a) * np.exp(-4 * M**2 * Tv)
        f = np.sum(f, axis=2)
        f *= np.exp(mvl * qu) - 1
        f += 1
        np.log(f, out=f)
        f /= mvl

        return f

    def settlement_PTIB(self, a, t):
        """settlement for PTIB drainage

        Paramters
        ---------
        a : array like of float
            depth coord
        t : array like of float
            time coord

        Returns
        -------
        settlement : float of size (len(a), len(t))
            settlement at depth a, time t

        """

        Tv = self.Tv(t)[None, :, None]
        a_ = (a /self.H)[:, None]
        a = (a/ self.H)[:, None, None]

        mvl = self.mvl
        qu = self.qu
        M = self.M[None, None, :]
        H = self.H

        f = 2 / M**2 * np.cos(M * a) * np.exp(-M**2 * Tv)
        f = -np.sum(f, axis=2)
        f += 1 - a_
        f *= 1 - np.exp(-mvl * qu)
        f *= H
        return f

    def settlement_PTPB(self, a, t):
        """settlement for PTPB drainage

        Paramters
        ---------
        a : array like of float
            depth coord
        t : array like of float
            time coord

        Returns
        -------
        settlement : float of size (len(a), len(t))
            settlement at depth a, time t

        """

        Tv = self.Tv(t)[None, :, None]
        a_ = (a /self.H)[:, None]
        a = (a/ self.H)[:, None, None]

        mvl = self.mvl
        qu = self.qu
        M = self.M[None, None, :]
        H = self.H

        f = 1 / M**2 * (1 + np.cos(2 * M * a)) * np.exp(-4 * M**2 * Tv)
        f = -np.sum(f, axis=2)
        f += 1 - a_
        f *= 1 - np.exp(-mvl * qu)
        f *= H
        return f

    def Us_PTIB(self, t):
        """Average degree of consolidation from settlement for PTIB drainage

        Paramters
        ---------
        t : array like of float
            time coord

        Returns
        -------
        Us : array of float of size len(t)
            settlement degree of consolidation at time t

        """

        Tv = self.Tv(t)[:, None]

        mvl = self.mvl
        qu = self.qu
        M = self.M[None, :]

        f = 2 / M**2 * np.exp(-M**2 * Tv)
        f = np.sum(f, axis=1)
        f*=-1
        f +=1

        return f

    def Us_PTPB(self, t):
        """Average degree of consolidation from settlement for PTPB drainage

        Paramters
        ---------
        t : array like of float
            time coord

        Returns
        -------
        Us : array of float of size len(t)
            settlement degree of consolidation at time t

        """

        Tv = self.Tv(t)[:, None]

        mvl = self.mvl
        qu = self.qu
        M = self.M[None, :]

        f = 2 / M**2 * np.exp(-4 * M**2 * Tv)
        f = np.sum(f, axis=1)
        f*=-1
        f +=1

        return f



    def Up_PTIB(self, t):
        """Average degree of consolidation from p.press for PTIB drainage

        Paramters
        ---------
        t : array like of float
            time coord

        Returns
        -------
        Up : array of float of size len(t)
            pore pressure average degree of consolidation at time t

        """

        def u(a, t):
            """wrapper for self.u_PTIB for scalar args"""
            a = np.atleast_1d(a)
            t = np.atleast_1d(t)
            return self.u_PTIB(a,t)[0, 0]

        qu = self.qu
        H = self.H

        f = np.empty(len(t), dtype=float)
        #TODO: replace call to quad with my own numerical integrations to avoid scipy license
        for i, t_ in enumerate(t):
            f[i] = 1 - 1.0 / (H * qu) * integrate.quad(u, 0, H, (t_,))[0]

        return f


    def Up_PTPB(self, t):
        """Average degree of consolidation from p.press for PTPB drainage

        Paramters
        ---------
        t : array like of float
            time coord

        Returns
        -------
        Up : array of float of size len(t)
            pore pressure average degree of consolidation at time t

        """

        def u(a, t):
            """wrapper for self.u_PTPB for scalar args"""
            a = np.atleast_1d(a)
            t = np.atleast_1d(t)
            return self.u_PTPB(a,t)[0, 0]

        qu = self.qu
        H = self.H

        f = np.empty(len(t), dtype=float)
        #TODO: replace call to quad with my own numerical integrations to avoid scipy license
        for i, t_ in enumerate(t):
            f[i] = 1 - 1.0 / (H * qu) * integrate.quad(u, 0, H, (t_,))[0]

        return f



    def effective_stress_PTIB(self, a, t):
        """Effective stress for PTIB drainage

        Paramters
        ---------
        a : array like of float
            depth coord
        t : array like of float
            time coord

        Returns
        -------
        effective_stress : float of size (len(a), len(t))
            effective stress at depth a, time t

        """

        u = self.u_PTIB(a,t)
        sig_0 = self.initial_effective_stress(a)[:, None]

        sig_ = sig_0 + qu - u

        return sig_

    def effective_stress_PTPB(self, a, t):
        """Effective stress for PTPB drainage

        Paramters
        ---------
        a : array like of float
            depth coord
        t : array like of float
            time coord

        Returns
        -------
        effective_stress : float of size (len(a), len(t))
            effective stress at depth a, time t

        """

        u = self.u_PTPB(a,t)
        sig_0 = self.initial_effective_stress(a)[:, None]

        sig_ = sig_0 + qu - u

        return sig_

    def total_stress_PTIB(self, a, t):
        """total stress for PTIB drainage

        Paramters
        ---------
        a : array like of float
            depth coord
        t : array like of float
            time coord

        Returns
        -------
        total_stress : float of size (len(a), len(t))
            total stress at depth a, time t

        """
        gamw = self.gamw
        Hw = self.Hw

        S = self.settlement_PTIB(a, t)
        sig_0 = self.initial_effective_stress(a)[:, None]
        a = a[:, None]

        sig = sig_0 + qu + gamw * (Hw + a + S)

        return sig

    def total_stress_PTPB(self, a, t):
        """total stress for PTPB drainage

        Paramters
        ---------
        a : array like of float
            depth coord
        t : array like of float
            time coord

        Returns
        -------
        total_stress : float of size (len(a), len(t))
            total stress at depth a, time t

        """
        gamw = self.gamw
        Hw = self.Hw

        S = self.settlement_PTPB(a, t)
        sig_0 = self.initial_effective_stress(a)[:, None]
        a = a[:, None]

        sig = sig_0 + qu + gamw * (Hw + a + S)

        return sig

    def total_pore_pressure_PTIB(self, a, t):
        """total pore pressure for PTIB drainage

        Paramters
        ---------
        a : array like of float
            depth coord
        t : array like of float
            time coord

        Returns
        -------
        total_pore_pressure : float of size (len(a), len(t))
            total pore pressure at depth a, time t

        """

        gamw = self.gamw
        Hw = self.Hw

        u = self.u_PTIB(a,t)
        S = self.settlement_PTIB(a, t)
        a = a[:, None]

        p = u + gamw * (Hw + a + S)
        return p

    def total_pore_pressure_PTPB(self, a, t):
        """total pore pressure for PTPB drainage

        Paramters
        ---------
        a : array like of float
            depth coord
        t : array like of float
            time coord

        Returns
        -------
        total_pore_pressure : float of size (len(a), len(t))
            total pore pressure at depth a, time t

        """

        gamw = self.gamw
        Hw = self.Hw

        u = self.u_PTPB(a,t)
        S = self.settlement_PTPB(a, t)
        a = a[:, None]

        p = u + gamw * (Hw + a + S)
        return p

    def e_PTIB(self, a, t):
        """void ration for PTIB drainage

        Paramters
        ---------
        a : array like of float
            depth coord
        t : array like of float
            time coord

        Returns
        -------
        e : float of size (len(a), len(t))
            void ration at depth a, time t

        """

        e0 = self.e0(a)[:, None]
        efinal = self.efinal(a)[:, None]

        Tv = self.Tv(t)[None, :, None]
        a = (a/ self.H)[:, None, None]

        M = self.M[None, None, :]

        f = 2 / M * np.sin(M * a) * np.exp(-M**2 * Tv)
        f = np.sum(f, axis=2)
        f *= e0 - efinal
        f += efinal

        return f

    def e_PTPB(self, a, t):
        """void ration for PTPB drainage

        Paramters
        ---------
        a : array like of float
            depth coord
        t : array like of float
            time coord

        Returns
        -------
        e : float of size (len(a), len(t))
            void ration at depth a, time t

        """

        e0 = self.e0(a)[:, None]
        efinal = self.efinal(a)[:, None]

        Tv = self.Tv(t)[None, :, None]
        a = (a/ self.H)[:, None, None]

        M = self.M[None, None, :]

        f = 2 / M * np.sin(2 * M * a) * np.exp(-4 * M**2 * Tv)
        f = np.sum(f, axis=2)
        f *= e0 - efinal
        f += efinal

        return f

    def vs_PTIB(self, a, t):
        """velocity of soil particles for PTIB drainage

        Paramters
        ---------
        a : array like of float
            depth coord
        t : array like of float
            time coord

        Returns
        -------
        vs : float of size (len(a), len(t))
            velocity of soil particles at depth a, time t

        """

        mvl = self.mvl
        qu = self.qu
        cv0 = self.cv0
        H = self.H

        Tv = self.Tv(t)[None, :, None]
        a = (a / self.H)[:, None, None]
        M = self.M[None, None, :]

        f = np.cos(M * a) * np.exp(-M**2 * Tv)
        f = np.sum(f, axis=2)
        f *= 1 - np.exp(-mvl * qu)
        f *= 2 * cv0/H

        return f

    def vs_PTPB(self, a, t):
        """velocity of soil particles for PTPB drainage

        Paramters
        ---------
        a : array like of float
            depth coord
        t : array like of float
            time coord

        Returns
        -------
        vs : float of size (len(a), len(t))
            velocity of soil particles at depth a, time t

        """

        mvl = self.mvl
        qu = self.qu
        cv0 = self.cv0
        H = self.H

        Tv = self.Tv(t)[None, :, None]
        a = (a / self.H)[:, None, None]
        M = self.M[None, None, :]

        f = (1 + np.cos(2 * M * a)) * np.exp(-4 * M**2 * Tv)
        f = np.sum(f, axis=2)
        f *= 1 - np.exp(-mvl * qu)
        f *= 4 * cv0/H

        return f

    def vw_PTIB(self, a, t):
        """velocity of fluid for PTIB drainage

        Paramters
        ---------
        a : array like of float
            depth coord
        t : array like of float
            time coord

        Returns
        -------
        vw : float of size (len(a), len(t))
            velocity of fluid at depth a, time t

        """

        mvl = self.mvl
        qu = self.qu
        cv0 = self.cv0
        H = self.H

        e = self.e_PTIB(a, t)
        Tv = self.Tv(t)[None, :, None]
        a = (a / self.H)[:, None, None]
        M = self.M[None, None, :]


        f = np.cos(M * a) * np.exp(-M**2 * Tv)
        f = np.sum(f, axis=2)
        f *= 1 - np.exp(-mvl * qu)
        f *= 2 * cv0 / H
        f /=e

        return f

    def vw_PTPB(self, a, t):
        """velocity of fluid for PTPB drainage

        Paramters
        ---------
        a : array like of float
            depth coord
        t : array like of float
            time coord

        Returns
        -------
        vw : float of size (len(a), len(t))
            velocity of fluid at depth a, time t

        """

        mvl = self.mvl
        qu = self.qu
        cv0 = self.cv0
        H = self.H

        e = self.e_PTPB(a, t)
        Tv = self.Tv(t)[None, :, None]
        a = (a / self.H)[:, None, None]
        M = self.M[None, None, :]


        f1 = np.exp(-4 * M**2 * Tv)
        f1 = np.sum(f1, axis=2)
        f1 *= 1 - np.exp(-mvl * qu)
        f1 *= 4 * cv0 / H
        f1=f1.ravel()[None,:]*(1+e)/e
#        f1 *= 1.0 + e
#        f1 /= e


        f2 = (1 + np.cos(2 * M * a)) * np.exp(-4 * M**2 * Tv)
        f2 = np.sum(f2, axis=2)
        f2 *= 1 - np.exp(-mvl * qu)
        f2 *= 2 * cv0/H
        f2 /=e

        return f1-f2

    def xi_PTIB(self, a,t):
        """convectove cordinate from lagrange coordinate

        Paramters
        ---------
        a : array like of float
            depth coord
        t : array like of float
            time coord

        Returns
        -------
        xi : float of size (len(a), len(t))
            convective coordinate at depth a, time t

        """

        S = self.settlement_PTIB(a,t)
        f = a[:,None] + S

        return f

    def xi_PTPB(self, a,t):
        """convectove cordinate from lagrange coordinate

        Paramters
        ---------
        a : array like of float
            depth coord
        t : array like of float
            time coord

        Returns
        -------
        xi : float of size (len(a), len(t))
            convective coordinate at depth a, time t

        """

        S = self.settlement_PTPB(a,t)
        f = a[:,None] + S

        return f


    def plot_all(self, a=None, t=None, figsize=(15,10)):

        if self.drn==1:
            u_ = self.u_PTIB
            settlement_ = self.settlement_PTIB
            Us_ = self.Us_PTIB
            Up_ = self.Up_PTIB
            effective_stress_ = self.effective_stress_PTIB
            total_stress_ = self.total_stress_PTIB
            total_pore_pressure_ = self.total_pore_pressure_PTIB
            e_ = self.e_PTIB
            vs_ = self.vs_PTIB
            vw_ = self.vw_PTIB
            xi_ = self.xi_PTIB
        else:
            u_ = self.u_PTPB
            settlement_ = self.settlement_PTPB
            Us_ = self.Us_PTPB
            Up_ = self.Up_PTPB
            effective_stress_ = self.effective_stress_PTPB
            total_stress_ = self.total_stress_PTPB
            total_pore_pressure_ = self.total_pore_pressure_PTPB
            e_ = self.e_PTPB
            vs_ = self.vs_PTPB
            vw_ = self.vw_PTPB
            xi_ = self.xi_PTPB


        t_interp = np.logspace(np.log10(0.0001/self.dTv),np.log10(3/self.dTv),100)
        Us_interp = Us_(t_interp)

        Tv_interp = self.Tv(t_interp)

        # determine times to plot
        if t is None:
            Us_plot = np.linspace(0,1,11)
            Us_plot[-1] = 0.99
            t = np.interp(Us_plot, Us_interp, t_interp)

        Tv = self.Tv(t)
        if a is None:
            a = np.linspace(0, self.H, 100)
        a = np.asarray(a)



        u = u_(a,t)
        settlement = settlement_(a, t)

        Up_interp = Up_(t_interp)

        effective_stress = effective_stress_(a, t)
        total_stress = total_stress_(a, t)
        total_pore_pressure = total_pore_pressure_(a, t)
        e = e_(a, t)
        vs = vs_(a, t)
        vw = vw_(a, t)
        xi = xi_(a, t)


        matplotlib.rcParams['font.size'] = 10
        fig = plt.figure(figsize = figsize)

        # Us and Up vs time
        ax = fig.add_subplot(2,4,1)
        ax.plot(Tv_interp, Us_interp, label="$U_s$")
        ax.plot(Tv_interp, Up_interp, label="$U_p$")
        ax.set_xlabel('$T_v,\, dT_v=${:6.2g}'.format(self.dTv))
        ax.set_ylabel('U')
        ax.set_ylim(0,1)
        ax.invert_yaxis()
        ax.set_xscale('log')
        ax.grid()
        leg = plt.legend(loc=3 )
        leg.draggable()

        #u vs depth
        ax = fig.add_subplot(2,4,2)
        ax.plot(u, xi)
        ax.set_xlabel("$u$")
        ax.set_ylabel(r'$\xi$')
        ax.set_ylim(0,self.H)
        ax.invert_yaxis()

        ax.grid()

        for line, t_  in zip(ax.get_lines(), t):
            Us = Us_(np.array([t_]))[0]
            plt.setp(line,
            label='$U_s={Us:6.3g}$\n$T_v={Tv:6.3g}$\n'
             '$t={t:6.3g}$'.format(Tv=self.dTv*t_, t=t_, Us=Us))
        loc = 'lower center'
        bbox_transform = fig.transFigure
        bbox_to_anchor = (0.5, 0)

        leg = fig.legend(ax.get_lines(),
                        [v.get_label() for v in ax.get_lines()], loc=loc, bbox_transform=bbox_transform,
                         bbox_to_anchor=bbox_to_anchor,
                         ncol=len(t))


        leg.draggable()

        #total pore pressure vs depth
        ax = fig.add_subplot(2,4,6)
        ax.plot(total_pore_pressure, xi)
        ax.set_xlabel("$p$")
        ax.set_ylabel(r'$\xi$')
        ax.set_ylim(0,self.H)
        ax.invert_yaxis()
        ax.grid()


        #effective stress vs depth
        ax = fig.add_subplot(2,4,3)
        ax.plot(effective_stress, xi)
        ax.set_xlabel("$\sigma'$")
        ax.set_ylabel(r'$\xi$')
        ax.set_ylim(0,self.H)
        ax.invert_yaxis()
        ax.grid()

        #total stress vs depth
        ax = fig.add_subplot(2,4,7)
        ax.plot(total_stress, xi)
        ax.set_xlabel("$\sigma$")
        ax.set_ylabel(r'$\xi$')
        ax.set_ylim(0,self.H)
        ax.invert_yaxis()
        ax.grid()

        #velocity of solids vs depth
        ax = fig.add_subplot(2,4,4)
        ax.plot(vs, xi)
        ax.set_xlabel("$v_s$")
        ax.set_ylabel(r'$\xi$')
        ax.set_ylim(0, self.H)
        ax.invert_yaxis()
        ax.grid()

        #velocity of water vs depth
        ax = fig.add_subplot(2,4,8)
        ax.plot(vw, xi)
        ax.set_xlabel("$v_w$")
        ax.set_ylabel(r'$\xi$')
        ax.set_ylim(0, self.H)
        ax.invert_yaxis()
        ax.grid()

        #void ratio vs depth
        ax = fig.add_subplot(2,4,5)
        ax.plot(e, xi)
        ax.set_xlabel("$e$")
        ax.set_ylabel(r'$\xi$')
        ax.set_ylim(0, self.H)
        ax.invert_yaxis()
        ax.grid()

#        fig.tight_layout()

#        fig.tight_layout()

#        bbox = leg.get_frame().get_bbox()
#        print(bbox)

#        plt.Figure.legend()
#        a=plt.getp(fig.legend, 'bbox')
#        print(a)
#        bbox = fig.legend.get_window_extent()
#        print(bbox)
#        bbox2 = bbox.transformed(fig.transFigure.inverted())
#        bbox2.width,bbox2.height
#        print(bbox2)
#



        fig.subplots_adjust(top=0.97, bottom=0.15, left=0.05, right=0.97)


        return fig




    def t_from_Us_PTIB(self, Us):
        """ back calculate t from Us

        Parameters
        ----------
        Us : 1d array
            values of degree of consolidation by settlement to calc the t at


        Returns
        -------
        t : 1d array
            times coresponding to Us

        """

        t_interp = np.logspace(np.log10(0.0001/self.dTv),np.log10(10/self.dTv), 500)
        Us_interp = self.Us_PTIB(t_interp)

        t = np.interp(Us, Us_interp, t_interp)



        return t
    def t_from_Us_PTPB(self, Us):
        """ back calculate t from Us

        Parameters
        ----------
        Us : 1d array
            values of degree of consolidation by settlement to calc the t at


        Returns
        -------
        t : 1d array
            times coresponding to Us

        """

        t_interp = np.logspace(np.log10(0.0001/self.dTv),np.log10(10/self.dTv), 500)
        Us_interp = self.Us_PTPB(t_interp)

        t = np.interp(Us, Us_interp, t_interp)

        return t



def xie_and_leo_2004_figure_4(ax=None):
    """reproduce fig 4 from article by Xie and Leo 2004
    pore pressure vs xi plot for various degrees of consolidation PTIB

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes object to plot on. If ax=None. plt.gca() will be used

    """

    qu=100
    qp=10
    H=10
    Hw=1.0
    kv0=1e-9
    mvl=4e-3
    e00=3
    Gs=2.75
    gamw=10 #N
    drn=1
    nterms=100

    obj = XieAndLeo2004(qu=qu, qp=qp, H=H, Hw=Hw,
                        kv0=kv0, mvl=mvl,e00=e00, Gs=Gs, gamw=gamw,
                        drn=drn, nterms=nterms)

    a = np.linspace(0,H, 100)

    Us = np.array([0.3, 0.5, 0.7, 0.9])

    t = obj.t_from_Us_PTIB(Us)
    u = obj.u_PTIB(a, t)
    xi = obj.xi_PTIB(a, t)

    if ax is None:
        ax = plt.gca()

    ax.plot(u, xi)
    ax.set_xlabel("$u$ Pore water pressure, PTPB")
    ax.set_ylabel(r'$\xi$, depth from initial top surface')
    ax.set_title("Figure 4 from Xie and Leo 2004")
    ax.set_ylim(0, H)
    ax.invert_yaxis()
    ax.grid()

    for line, t_, Us_  in zip(ax.get_lines(), t, Us):
        plt.setp(line,
        label='$U_s={Us:6.3g},\hspace{{0.5}}T_v={Tv:6.3g},\hspace{{0.5}}'
         't={t:6.3g}$'.format(Tv=obj.dTv*t_, t=t_, Us=Us_))
    leg = ax.legend(loc=1, labelspacing=0.0, fontsize=11)
    leg.draggable()
    return


def xie_and_leo_2004_figure_5(ax=None):
    """reproduce fig 5 from article by Xie and Leo 2004
    pore pressure vs xi plot for various degrees of consolidation PTPB

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes object to plot on. If ax=None. plt.gca() will be used

    """

    qu=100
    qp=10
    H=10
    Hw=1.0
    kv0=1e-9
    mvl=4e-3
    e00=3
    Gs=2.75
    gamw=10 #N
    drn=0
    nterms=100

    obj = XieAndLeo2004(qu=qu, qp=qp, H=H, Hw=Hw,
                        kv0=kv0, mvl=mvl,e00=e00, Gs=Gs, gamw=gamw,
                        drn=drn, nterms=nterms)

    a = np.linspace(0,H, 100)

    Us = np.array([0.3, 0.5, 0.7, 0.9])

    t = obj.t_from_Us_PTPB(Us)
    u = obj.u_PTPB(a, t)
    xi = obj.xi_PTPB(a, t)

    if ax is None:
        ax = plt.gca()

    ax.plot(u, xi)
    ax.set_xlabel("$u$ Pore water pressure, PTPB")
    ax.set_ylabel(r'$\xi$, depth from initial top surface')
    ax.set_title("Figure 5 from Xie and Leo 2004")
    ax.set_ylim(0, H)
    ax.invert_yaxis()
    ax.grid()

    for line, t_, Us_  in zip(ax.get_lines(), t, Us):
        plt.setp(line,
        label='$U_s={Us:6.3g},\hspace{{0.5}}T_v={Tv:6.3g},\hspace{{0.5}}'
         't={t:6.3g}$'.format(Tv=obj.dTv*t_, t=t_, Us=Us_))
    leg = ax.legend(loc=1, labelspacing=0.0, fontsize=11)
    leg.draggable()
    return



def xie_and_leo_2004_figure_6(ax=None):
    """reproduce fig 6 from article by Xie and Leo 2004
    settlement vs time and degree of consolidation vs time

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes object to plot on. If ax=None. plt.gca() will be used

    """


    qu=100
    qp=10
    H=10
    Hw=1.0
    kv0=1e-9
    mvl=4e-3
    e00=3
    Gs=2.75
    gamw=10 #N
    drn=0
    nterms=100

    obj = XieAndLeo2004(qu=qu, qp=qp, H=H, Hw=Hw,
                        kv0=kv0, mvl=mvl,e00=e00, Gs=Gs, gamw=gamw,
                        drn=drn, nterms=nterms)

    Tv = np.logspace(-3,np.log10(6), 200)
    t = Tv / obj.dTv

    Up_PTPB = obj.Up_PTPB(t)
    Us_PTPB = obj.Us_PTPB(t)
    S_PTPB = obj.settlement_PTPB(np.array([0.0]), t)

    Up_PTIB = obj.Up_PTIB(t)
    Us_PTIB = obj.Us_PTIB(t)
    S_PTIB = obj.settlement_PTIB(np.array([0.0]), t)

    if ax is None:
        ax = plt.gca()

    ax.plot(Tv, Us_PTPB, label="$U_s,\hspace{0.5}PTPB$", color='b', ls="-")
    ax.plot(Tv, Up_PTPB, label="$U_p,\hspace{0.5}PTPB$", color='b', ls="--")

    ax.plot(Tv, Us_PTIB, label="$U_s,\hspace{0.5}PTIB$", color='g', ls="-")
    ax.plot(Tv, Up_PTIB, label="$U_p,\hspace{0.5}PTIB$", color='g', ls="--")


    ax.set_xlabel("$T_v$")
    ax.set_ylabel(r'degree of consolidation')
    ax.set_title("Figure 6 from Xie and Leo 2004")
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_xscale('log')

    ticks11a = matplotlib.ticker.LinearLocator(11)
    ax.yaxis.set_major_locator(ticks11a)
#    ax.locator_params(axis='y', nbins=16)

    ax.grid(ls="-", which="major")
    ax.grid(which="minor")




    ax2 = ax.twinx()

    ax2.plot(Tv, S_PTPB[0], label="$settlement,\hspace{0.5}PTPB$",color='b', dashes = [3,2,6,2])
    ax2.plot(Tv, S_PTIB[0], label=r"$settlement,\hspace{0.5}PTIB$",color='g', dashes = [3,2,6,2])
    ax2.set_ylabel(r'Settlement (m)')
    ax2.set_ylim(0, 4)
    ax2.invert_yaxis()

    ticks11b = matplotlib.ticker.LinearLocator(11)
    ax2.yaxis.set_major_locator(ticks11b)

    lines=[]
    labels=[]
    for i in ax.get_lines():
        lines.append(i)
        labels.append(i.get_label())

    for i in ax2.get_lines():
        lines.append(i)
        labels.append(i.get_label())


    leg = ax.legend(lines, labels,loc=1, labelspacing=0.0, fontsize=12)
    leg.draggable()



    return
if __name__ == '__main__':




    if 0:
        xie_and_leo_2004_figure_4()
        plt.show()
    if 0:
        xie_and_leo_2004_figure_5()
        plt.show()
    if 0:
        xie_and_leo_2004_figure_6()
        plt.show()


    if 1:
        # plot all
        qu=100
        qp=10
        H=10
        Hw=1.0
        kv0=1e-9
        mvl=4e-3
        e00=3
        Gs=2.75
        gamw=10 #N
        drn=1
        nterms=100

        obj = XieAndLeo2004(qu=qu, qp=qp, H=H, Hw=Hw,
                            kv0=kv0, mvl=mvl,e00=e00, Gs=Gs, gamw=gamw,
                            drn=drn, nterms=nterms)

        fig = obj.plot_all()
        plt.show()




