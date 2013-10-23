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
Created on Wed Oct 16 16:08:19 2013

@author: Rohan Walker
"""
from __future__ import division, print_function
import geotecha.piecewise.piecewise_linear_1d as pwise
from geotecha.piecewise.piecewise_linear_1d import PolyLine
import geotecha.speccon.integrals as integ

import sys, imp
import textwrap
import numpy as np
import matplotlib.pyplot as plt

try:
    #for python 2 to 3 stuff see http://python3porting.com/stdlib.html
    #for using BytesIO instead of StringIO see http://stackoverflow.com/a/3423935/2530083
    from io import BytesIO as StringIO
except ImportError:
    from StringIO import StringIO

def make_module_from_text(reader):
    """make a module from file,StringIO, text etc
    
    Parameters
    ----------
    reader : file_like object
        object to get text from
    
    Returns
    -------
    m: module
        text as module
        
    """
    #for making module out of strings/files see http://stackoverflow.com/a/7548190/2530083    
    
    mymodule = imp.new_module('mymodule') #may need to randomise the name
    exec reader in mymodule.__dict__    
    return mymodule

class speccon1d(object):
    """
    speccon1d(reader)
    
    1d consolidation with:
    
    - vertical and radial drainage (radial drainage uses the eta method)
    - material properties that are constant in time but piecewsie linear with 
      depth
      
      - vertical permeability
      - horizontal permeability
      - lumped drain parameter eta
      - volume compressibilty  
      
    - surcharge and vacuum loading
    
      - distribution with depth does not change over time
      - magnitude varies piecewise linear with time
      - multiple loads can be superposed
      
    - pore pressure boundary conditions at top and bottom vary piecewise 
      linear with time
    - calculates
    
      - excess pore pressure at depth
      - average excess pore pressure between depths
      - settlement between depths
              
    
    
    
    
    Parameters
    ----------
    reader : object that can be run with exec to produce a module
        reader can be for examplestring, fileobject, StringIO.`reader` 
        should contain an statements such as H = 1, drn=0 corresponding to the 
        attributes listed below.  The user should pick an appropriate
        combination of attributes for their analysis.  e.g. don't put dTh=, 
        kh=, et=, if you are not modelling radial drainage        
            
    Attributes
    ----------
    H : float, optional
        total height of soil profile. default = 1.0
    drn : {0, 1}, optional
        drainage boundary condition. default = 0
        0 = Pervious top pervious bottom (PTPB)
        1 = Pervious top impoervious bottom (PTIB)
    dT : float, optional
        convienient time factor multiplier. default = 1.0
    neig: int, optional
        number of series terms to use in solution. default = 2
    dTv: float, optional
        vertical reference time factor multiplier.  dTv is calculated with 
        the chosen reference values of kz and mv: dTv = kz /(mv*gamw) / H ^ 2 
    dTh : float, optional
        horizontal reference time factor multiplier.  dTh is calculated with 
        the reference values of kh, et, and mv: dTh = kh / (mv * gamw) * et
    mv : PolyLine, optional
        normalised volume compressibility PolyLine(depth, mv)
    kh : PolyLine, optional
        normalised horizontal permeability PolyLine(depth, kh) 
    kz : PolyLine , optional
        normalised vertical permeability PolyLine(depth, kz) 
    et : PolyLine, optional
        normalised vertical drain parameter PolyLine(depth, et). 
        et = 2 / (mu * re^2) where mu is smear-zone/geometry parameter and re
        is radius of influence of vertical drain     
    surz : list of Polyline, optional
        surcharge variation with depth. PolyLine(depth, multiplier)
    vacz : list of Polyline, optinal
        vacuum variation with depth. PolyLine(depth, multiplier)
    surm : list of Polyline, optional
        surcharge magnitude variation with time. PolyLine(time, magnitude)
    vacm : list of Polyline, optional
        vacuum magnitude variation with time. Polyline(time, magnitude)
    topm : list of Polyline, optional
        top p.press variation with time. Polyline(time, magnitude)
    botm : list of Polyline, optional
        bottom p.press variation with time. Polyline(time, magnitude)    
    porz : list_like of float, optional
        normalised z to calc pore pressure at    
    avpz : list of two element list of float, optional
        nomalised zs to calc average pore pressure between
    setz : list of two element list of float, optional
        normalised depths to calculate normalised settlement between. 
        e.g. surface settlement would be [[0, 1]]
    outt : list of float
        times to calculate output at
        
    
    References
    ----------  
    All based on work by Dr Rohan Walker [1]_, [2]_, [3]_, [4]_
    
    .. [1] Walker, Rohan. 2006. 'Analytical Solutions for Modeling Soft Soil Consolidation by Vertical Drains'. PhD Thesis, Wollongong, NSW, Australia: University of Wollongong.
    .. [2] Walker, R., and B. Indraratna. 2009. 'Consolidation Analysis of a Stratified Soil with Vertical and Horizontal Drainage Using the Spectral Method'. Geotechnique 59 (5) (January): 439-449. doi:10.1680/geot.2007.00019.
    .. [3] Walker, Rohan, Buddhima Indraratna, and Nagaratnam Sivakugan. 2009. 'Vertical and Radial Consolidation Analysis of Multilayered Soil Using the Spectral Method'. Journal of Geotechnical and Geoenvironmental Engineering 135 (5) (May): 657-663. doi:10.1061/(ASCE)GT.1943-5606.0000075.
    .. [4] Walker, Rohan T. 2011. Vertical Drain Consolidation Analysis in One, Two and Three Dimensions'. Computers and Geotechnics 38 (8) (December): 1069-1077. doi:10.1016/j.compgeo.2011.07.006.

    """
    def __init__(self,reader=None):


        self._parameter_defaults = {'H': 1.0, 'drn': 0, 'dT': 1.0, 'neig': 2}
        #self._parameters = 'H drn dT neig dTh dTv mv kh kz et surz surm vacz vacm topm botm porz port avpz avpt setz sett'.split()        
        self._parameters = 'H drn dT neig dTh dTv mv kh kz et surz surm vacz vacm topm botm porz avpz setz outt'.split()        
        
        self._should_be_lists= 'surz surm vacz vacm topm botm'.split()
        self._should_have_same_z_limits = 'mv kz kh et surz vacz'.split()
        self._should_have_same_len_pairs = 'surz surm vacz vacm'.split() #pairs that should have the same length
        
        
#        #if you don't care about autocomplete for the parameters you can use this loop 
#        for v in self._parameters:
#            self.__setattr__(v, self._parameter_defaults.get(v, None))
                   
        self.H = self._parameter_defaults.get('H')               
        self.drn = self._parameter_defaults.get('drn')
        self.dT = self._parameter_defaults.get('dT')
        self.neig = self._parameter_defaults.get('neig')        
        self.dTh = None
        self.dTv = None
                    
        self.mv = None #normalised volume compressibility PolyLine(depth, mv)
        self.kh = None #normalised horizontal permeability PolyLine(depth, kh) 
        self.kz = None #normalised vertical permeability PolyLine(depth, kz) 
        self.et = None #normalised vertical drain parameter PolyLine(depth, et) 

        self.surz = None #surcharge list of PolyLine(depth, multiplier)
        self.vacz = None #vacuum list of PolyLine(depth, multiplier)
        self.surm = None #surcharge list of PolyLine(time, magnitude)
        self.vacm = None #vacuum list of Polyline(time, magnitude)
        self.topm = None #top p.press list of Polyline(time, magnitude)
        self.botm = None #bot p.press list of Polyline(time, magnitude)
        
        self.porz = None #normalised z to calc pore pressure at
        #self.port = None #times to calc pore pressure at
        self.avpz = None #nomalised zs to calc average pore pressure between
        #self.avpt = None #times to calc average pore pressure at
        self.setz = None #normalised depths to calculate normalised settlement between    
        #self.sett = None #times to calc normalised settlement at
        self.outt = None        
        
        self.text = None
        if not reader is None:
            self.grab_input_from_text(reader)
            
        #self._m = None            
        
    def make_m(self):
        """make the basis function eigenvalues
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        Notes
        -----
        
        .. math:: m_i =\\pi*\\left(i+1-drn/2\\right) 
        
        for :math:`i = 1\:to\:neig-1`
        
        """
        if sum(v is None for v in[self.neig, self.drn])!=0:
            raise ValueError('neig and/or drn is not defined')
        self.m = integ.m_from_sin_mx(np.arange(self.neig), self.drn)         
        return
          
    def check_for_input_errors(self):
        """checks for various input inconsistencies"""
        
        
        #bad material prop input errors  
        if self.mv is None:
            raise ValueError('No mv values given. '); sys.exit(0)    
        if self.dTh is None and self.dTv is None:
            raise ValueError('Neither dTv or dTh are defined. Need at least one'); sys.exit(0)
        if self.kh is None and self.kz is None:
            raise ValueError('Neither kh and kz are defined. Need at least one'); sys.exit(0)        
        if sum([v is None for v in [self.dTh, self.kh, self.et]]) in [1,2]:
            raise ValueError('dTh, kh, or et is not defined, need all three to model radial drainage/vacuum'); sys.exit(0)    
        if sum([v is None for v in [self.dTv, self.kz]])==1:
            raise ValueError('dTv or kz is not defined, need both to model vertical drainage.'); sys.exit(0)        
        
        
            
        #bad load input errors
        if sum([v is None for v in [self.surz, self.surm]]) == 1:
            raise ValueError('surz or surm is not defined, need both to model surcharge'); sys.exit(0)        
        if sum([v is None for v in [self.vacz, self.vacm]]) == 1:
            raise ValueError('vacz or vacm is not defined, need both to model vacuum'); sys.exit(0)
        if self.drn==1 and not (self.botm is None):
            raise ValueError('botm is meaningless when bottom is impervious (drn=1).  remove drn or change drn to 1'); sys.exit(0)
        if sum([v is None for v in [self.surz, self.surm, self.vacz, self.vacm, self.topm, self.botm]])==6:
            raise ValueError('surz, surm, vacz, vacm, topm, and botm not defined.  i.e. no loadings specified.'); sys.exit(0)        
        if (sum([v is None for v in [self.vacz, self.vacm]])==0) and (sum([v is None for v in [self.dTh, self.kh, self.et]])>0):
            raise ValueError('vacz, vacm, defined but one or more of dTh, kh, et not defined.  To model vacuum need dTh, kh and et'); sys.exit(0)        
            
        #bad output specifying errors    
#        if sum([v is None for v in [self.porz, self.port]])==1:
#            raise ValueError('porz or port is not defined, need both to output pore pressure at depth'); sys.exit(0)    
#        if sum([v is None for v in [self.avpz, self.avpt]])==1:
#            raise ValueError('avpz or avpt is not defined, need both to output average pore pressure'); sys.exit(0)            
#        if sum([v is None for v in [self.setz, self.sett]])==1:
#            raise ValueError('setz or sett is not defined, need both to output settlement'); sys.exit(0)        
#        if sum([v is None for v in [self.porz, self.port, self.avpz, self.avpt,self.setz, self.sett]])==6:
#            raise ValueError('porz, port, avpz, avpt, setz, and sett not specified.  i.e. no output specified.'); sys.exit(0)
        if self.outt is None:
            raise ValueError('outt not specified.  i.e. no output.'); sys.exit(0)                        
        if sum([v is None for v in [self.porz, self.avpz, self.setz]])==3:            
            raise ValueError('porz, avpz, setz and not specified.  i.e. no output specified.'); sys.exit(0)            
            
           
    def grab_input_from_text(self, reader):
        """grabs input parameters from fileobject, StringIO, text"""
        #self.text = reader.read()
        
        inp = make_module_from_text(reader)
        
        for v in self._parameters:
            self.__setattr__(v, inp.__dict__.get(v,self._parameter_defaults.get(v, None)))                
        
        self.check_for_input_errors()
        self.check_list_inputs(self._should_be_lists)        
        self.check_z_limits(self._should_have_same_z_limits)
        self.check_len_pairs(self._should_have_same_len_pairs)
        self.make_m()
        
        
    def check_list_inputs(self, check_list):                
        """puts non-lists in a list"""
        
        g = self.__getattribute__
        for v in check_list:
            if not g(v) is None:
                if not isinstance(g(v), list):
                    self.__setattr__(v,[g(v)])                
                
    def check_z_limits(self, check_list):
        """checks that members of check_list have same z limits"""
        g = self.__getattribute__
        
        #find first z values
        for v in check_list:
            if not g(v) is None:                
                if isinstance(g(v), list):
                    zcheck = np.array([g(v)[0].x[0], g(v)[0].x[-1]])                   
                    zstr = v
                    break
                else:
                    a = g(v).x[0]
                    
                    zcheck = np.array([g(v).x[0], g(v).x[-1]])
                    zstr = v
                    break
        
        for v in check_list:
            if not g(v) is None:                
                if isinstance(g(v), list):
                    for j, u in enumerate(g(v)):
                        if not np.allclose([u.x[0], u.x[-1]], zcheck):                             
                            raise ValueError('All upper and lower z limits must be the same.  Check ' + v + ' and ' + zstr + '.'); sys.exit(0)                                                                
                else:
                    if not np.allclose([g(v).x[0], g(v).x[-1]], zcheck):
                        raise ValueError('All upper and lower z limits must be the same.  Check ' + v + ' and ' + zstr + '.'); sys.exit(0)                                                                
                        
        
#        zs = [(g(v).x[0], g(v).x[-1]) for v in check_list if g(v) is not None]
#        print(zs)
#        if zs.count(zs[0])!=len(zs):
#            raise ValueError(','.join(check_list) + " must all have the same start and end z values (usually 0 and 1)."); sys.exit(0)
    def check_len_pairs(self,check_list):
        """checks pairs of parameters that hsould have eth same length"""

        g = self.__getattribute__
        
        # for iterating in chuncks see http://stackoverflow.com/a/434328/2530083
        for v1, v2 in [check_list[pos:pos + 2] for pos in xrange(0, len(check_list), 2)]:
            if not g(v1) is None and not g(v2) is None:
                if len(g(v1)) != len(g(v2)):
                    raise ValueError("%s has %d elements, %s has %d elements.  They should have the same number of elements." % (v1,len(g(v1)), v2, len(g(v2))))

    def make_m(self):
        """make the basis function eigenvalues
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        Notes
        -----
        
        .. math:: m_i =\\pi*\\left(i+1-drn/2\\right) 
        
        for :math:`i = 1\:to\:neig-1`
        
        """
        if sum(v is None for v in[self.neig, self.drn])!=0:
            raise ValueError('neig and/or drn is not defined')
        self.m = integ.m_from_sin_mx(np.arange(self.neig), self.drn)         
        return        
    def make_gam(self):
        """make the mv dependant gam matrix
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        Notes
        -----
        
        Creates the :math: `\Gam` matrix which occurs in the following equation:
        
        .. math:: \\mathbf{\\Gamma}\\mathbf{A}'=\\mathbf{\\Psi A}+loading\\:terms
        
        `self.gam`, :math:`\Gamma` is given by:          
        
        .. math:: \\mathbf{\Gamma}_{i,j}=\\int_{0}^1{{m_v\\left(z\\right)}{sin\\left({m_j}z\\right)}{sin\\left({m_i}z\\right)}\,dz}
        

        """
                  
        self.gam = integ.dim1sin_af_linear(self.m,self.mv.y1, self.mv.y2, self.mv.x1, self.mv.x2)
        self.gam[np.abs(self.gam)<1e-8]=0.0
        return
        
    def make_psi(self):
        """make kv, kh, et dependant psi matrix
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        Notes
        -----
        Creates the :math: `\Psi` matrix which occurs in the following equation:
        
        .. math:: \\mathbf{\\Gamma}\\mathbf{A}'=\\mathbf{\\Psi A}+loading\\:terms
        
        `self.psi`, :math:`\Psi` is given by:          
        
        .. math:: \\mathbf{\Psi}_{i,j}=dT_h\\mathbf{A}_{i,j}=\\int_{0}^1{{k_h\\left(z\\right)}{\eta\\left(z\\right)}\\phi_i\\phi_j\\,dz}-dT_v\\int_{0}^1{\\frac{d}{dz}\\left({k_z\\left(z\\right)}\\frac{d\\phi_j}{dz}\\right)\\phi_i\\,dz}
        
        """
        self.make_m()
        
        self.psi = np.zeros((self.neig, self.neig))           
        #kv part
        if sum([v is None for v in [self.kz, self.dTv]])==0:            
            self.psi -= self.dTv / self.dT * integ.dim1sin_D_aDf_linear(self.m, self.kz.y1, self.kz.y2, self.kz.x1, self.kz.x2)
        #kh & et part
        if sum([v is None for v in [self.kh, self.et, self.dTh]])==0:
            kh, et = pwise.polyline_make_x_common(self.kh, self.et)
            self.psi += self.dTh / self.dT * integ.dim1sin_abf_linear(self.m,kh.y1, kh.y2, et.y1, et.y2, kh.x1, kh.x2)
        self.psi[np.abs(self.psi)<1e-8]=0.0
        return        
        

    def make_eigs_and_v(self):
        """make Igam_psi, v and eigs, and Igamv

        Finds the eigenvalues, `self.eigs`, and eigenvectors, `self.v` of 
        inverse(gam)*psi.  Once found the matrix inverse(gamma*v), `self.Igamv`
        is determined.
        
        Notes
        -----
        From the original equation
        
        .. math:: \\mathbf{\\Gamma}\\mathbf{A}'=\\mathbf{\\Psi A}+loading\\:terms

        `self.eigs` and `self.v` are the eigenvalues and eigenvegtors of the matrix `self.Igam_psi`

        .. math:: \\left(\\mathbf{\\Gamma}^{-1}\\mathbf{\\Psi}\\right)        
        
        """
        
        #Igam_psi = gam.inverse()*psi
        Igam_psi = np.dot(np.linalg.inv(self.gam), self.psi)
        #self.Igam_psi = 
        
        self.eigs, self.v = np.linalg.eig(Igam_psi)
        
        self.Igamv=np.linalg.inv(np.dot(self.gam,self.v))                
        return

    def make_E_Igamv_the(self):
        """sum contributions from all loads
        
        Calculates all contributions to E*inverse(gam*v)*theta part of solution
        u=phi*vE*inverse(gam*v)*theta. i.e. surcharge, vacuum, top and bottom 
        pore pressure boundary conditions. `make_load_matrices will create 
        `self.E_Igamv_the`.  `self.E_Igamv_the`  is an array
        of size (neig, len(outt)). So the columns are the column array 
        E*inverse(gam*v)*theta calculated at each output time.  This will allow
        us later to do u = phi*v*self.E_Igamv_the 
        
        See also
        --------
        make_E_Igamv_the_surcharge :  surchage contribution
        make_E_Igamv_the_vacuum : vacuum contribution
        make_E_Igamv_the_top : top boundary pore pressure contribution
        make_E_Igamv_the_bot : bottom boundary pore pressure contribution        
        
        """
        self.E_Igamv_the = np.zeros((self.neig,len(self.outt)))
        if sum([v is None for v in [self.surz, self.surm]])==0:
            self.make_E_Igamv_the_surcharge()
            self.E_Igamv_the += self.E_Igamv_the_surcharge
        if sum([v is None for v in [self.vacz, self.vacm, self.et, self.kh,self.dTh]])==0:
            if self.dTh!=0:
                self.make_E_Igamv_the_vacuum()
                self.E_Igamv_the += self.E_Igamv_the_vacuum
        if not self.topm is None:
            self.make_E_Igamv_the_top()
            self.E_Igamv_the += self.E_Igamv_the_top
            if self.drn==1:
                self.E_Igamv_the += self.E_Igamv_the_bot
        
        if self.drn ==0 and not self.botm is None: #when drn==1 then self.E_Igamv_the_bot has already been created in make_E_Igamv_the_top()
            self.make_E_Igamv_the_bot()                    
            self.E_Igamv_the += self.E_Igamv_the_bot
            
        return
                    
            
        

    def make_E_Igamv_the_surcharge(self):
        """make the surcharge loading matrices
        
        Make the E*inverse(gam*v)*theta part of solution u=phi*vE*inverse(gam*v)*theta. 
        The contribution of each surcharge load is added and put in 
        `self.E_Igamv_the_surcharge`. `self.E_Igamv_the_surcharge` is an array
        of size (neig, len(outt)). So the columns are the column array 
        E*inverse(gam*v)*theta calculated at each output time.  This will allow
        us later to do u = phi*v*self.E_Igamv_the_surcharge
        
        Notes
        -----        
        Assuming the load are formulated as the product of separate time and depth 
        dependant functions:
        
        .. math:: \\sigma\\left({Z,t}\\right)=\\sigma\\left({Z}\\right)\\sigma\\left({t}\\right)
        
        the solution to the consolidation equation using the spectral method has 
        the form:
        
        .. math:: u\\left(Z,t\\right)=\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta} 
        
        `make_E_Igamv_the_surcharge` will create `self.E_Igamv_the_surcharge` which is 
        the :math:`\\mathbf{E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}` 
        part of the solution for all surcharge loads
        
        """
        
        self.E_Igamv_the_surcharge = np.zeros((self.neig,len(self.outt)))
        if sum([v is None for v in [self.surz, self.surm]])==0:
            for mag_vs_depth, mag_vs_time in zip(self.surz, self.surm):
                mv, mag_vs_depth = pwise.polyline_make_x_common(self.mv, mag_vs_depth)           
                
                theta = integ.dim1sin_ab_linear(self.m, mv.y1, mv.y2, mag_vs_depth.y1, mag_vs_depth.y2, mv.x1, mv.x2)
                Esur = integ.EDload_linear(mag_vs_time.x, mag_vs_time.y, self.eigs, self.outt, self.dT)

    
                #theta is 1d array, Igamv is nieg by neig array, np.dot(Igamv, theta) 
                #and np.dot(theta, Igamv) will give differetn 1d arrays.  
                #Basically np.dot(Igamv, theta) gives us what we want i.e. 
                #theta was treated as a column array.  The alternative 
                #np.dot(theta, Igamv) would have treated theta as a row vector.
                self.E_Igamv_the_surcharge += (Esur*np.dot(self.Igamv, theta)).T
        return

    def make_E_Igamv_the_vacuum(self):
        """make the vacuum loading matrices
        
        Make the E*inverse(gam*v)*theta part of solution u=phi*vE*inverse(gam*v)*theta. 
        The contribution of each vacuum load is added and put in 
        `self.E_Igamv_the_vacuum`. `self.E_Igamv_the_vacuum` is an array
        of size (neig, len(outt)). So the columns are the column array 
        E*inverse(gam*v)*theta calculated at each output time.  This will allow
        us later to do u = phi*v*self.E_Igamv_the_vacuum
        
        Notes
        -----        
        Assuming the load are formulated as the product of separate time and depth 
        dependant functions:
        
        .. math:: \\sigma\\left({Z,t}\\right)=\\sigma\\left({Z}\\right)\\sigma\\left({t}\\right)
        
        the solution to the consolidation equation using the spectral method has 
        the form:
        
        .. math:: u\\left(Z,t\\right)=\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta} 
        
        `make_E_Igamv_the_surcharge` will create `self.E_Igamv_the_surcharge` which is 
        the :math:`\\mathbf{E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}` 
        part of the solution for all surcharge loads
        
        """
        
        
        self.E_Igamv_the_vacuum = np.zeros((self.neig, len(self.outt)))
        
        if sum([v is None for v in [self.vacz, self.vacm, self.et, self.kh,self.dTh]])==0:
            if self.dTh!=0:
                for mag_vs_depth, mag_vs_time in zip(self.vacz, self.vacm):
                    et, kh, mag_vs_depth = pwise.polyline_make_x_common(self.et, self.kh, mag_vs_depth)           
                    
                    theta = integ.dim1sin_abc_linear(self.m, et.y1, et.y2, kh.y1, kh.y2, mag_vs_depth.y1, mag_vs_depth.y2, et.x1, et.x2)
                    Evac = integ.Eload_linear(mag_vs_time.x, mag_vs_time.y, self.eigs, self.outt, self.dT)        
                    
                    self.E_Igamv_the_vacuum += (Evac * np.dot(self.Igamv, theta)).T
        return

    def make_E_Igamv_the_top(self):
        """make the top pore pressure boundary condition loading matrices
        
        Make the E*inverse(gam*v)*theta part of solution u=phi*vE*inverse(gam*v)*theta. 
        The contribution of each top pore pressure boundary condition load is 
        added and put in 
        `self.E_Igamv_the_top`. `self.E_Igamv_the_top` is an array
        of size (neig, len(outt)). So the columns are the column array 
        E*inverse(gam*v)*theta calculated at each output time.  This will allow
        us later to do u = phi*v*self.E_Igamv_the_top.
        
        Note that if self.drn==1, PTIB then self.botm=self.topm so 
        `self.E_Igamv_the_top` will also be created here.
        
        Notes
        -----        
        Assuming the load are formulated as the product of separate time and depth 
        dependant functions:
        
        .. math:: \\sigma\\left({Z,t}\\right)=\\sigma\\left({Z}\\right)\\sigma\\left({t}\\right)
        
        the solution to the consolidation equation using the spectral method has 
        the form:
        
        .. math:: u\\left(Z,t\\right)=\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta} 
        
        `make_E_Igamv_the_top` will create `self.E_Igamv_the_top` which is 
        the :math:`\\mathbf{E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}` 
        part of the solution for all surcharge loads
        
        """
        
        self.E_Igamv_the_top = np.zeros((self.neig,len(self.outt)))
        if self.drn==1:
            self.E_Igamv_the_bot = np.zeros((self.neig,len(self.outt)))
            
        if not self.topm is None:
            for mag_vs_time in self.topm:                
                mv = self.mv
                #1. from mv*du/dt
                theta = integ.dim1sin_ab_linear(self.m, mv.y1, mv.y2, 1 - mv.x2, 1-mv.x1, mv.x1, mv.x2)                
                Etop = integ.EDload_linear(mag_vs_time.x, mag_vs_time.y, self.eigs, self.outt, self.dT)                            
                self.E_Igamv_the_top -= (Etop*np.dot(self.Igamv, theta)).T
                if self.drn==1:
                    theta = integ.dim1sin_ab_linear(self.m, mv.y1, mv.y2, mv.x1, mv.x2, mv.x1, mv.x2)                                        
                    self.E_Igamv_the_bot -= (Etop*np.dot(self.Igamv, theta)).T 
                
                #2. from dTh*et*kh*u
                if sum([v is None for v in [self.et, self.kh, self.dTh]])==0:
                    if self.dTh!=0:
                        et, kh = pwise.polyline_make_x_common(self.et, self.kh) 
                        theta = self.dTh / self.dT * integ.dim1sin_abc_linear(self.m, et.y1, et.y2, kh.y1, kh.y2, 1 - et.x2, 1 - et.x1, et.x1, et.x2)                
                        Etop = integ.Eload_linear(mag_vs_time.x, mag_vs_time.y, self.eigs, self.outt, self.dT)            
                        self.E_Igamv_the_top -= (Etop*np.dot(self.Igamv, theta)).T
                        if self.drn==1:
                            theta = self.dTh / self.dT * integ.dim1sin_abc_linear(self.m, et.y1, et.y2, kh.y1, kh.y2, et.x1, et.x2, et.x1, et.x2)                
                            self.E_Igamv_the_bot -= (Etop*np.dot(self.Igamv, theta)).T
                        
                #3. from dTv * D[kz * D[u,z], z]
                if sum([v is None for v in [self.kh, self.dTv]])==0:
                    if self.dTv!=0:                        
                        kv = self.kv
                        theta = self.dTv / self.dT * integ.dim1sin_D_aDb_linear(self.m, kv.y1, kv.y2, 1-kv.x2, 1-kv.x1,kv.x1, kv.x2)        
                        self.E_Igamv_the_top += (Etop*np.dot(self.Igamv, theta)).T                                    
                        if self.drn==1:
                            theta = self.dTv / self.dT * integ.dim1sin_D_aDb_linear(self.m, kv.y1, kv.y2, kv.x1, kv.x2,kv.x1, kv.x2)        
                            self.E_Igamv_the_top += (Etop*np.dot(self.Igamv, theta)).T                    
        return
        
    def make_E_Igamv_the_bot(self):
        """make the top pore pressure boundary condition loading matrices
        
        Make the E*inverse(gam*v)*theta part of solution u=phi*vE*inverse(gam*v)*theta. 
        The contribution of each top pore pressure boundary condition load is 
        added and put in 
        `self.E_Igamv_the_bot`. `self.E_Igamv_the_bot` is an array
        of size (neig, len(outt)). So the columns are the column array 
        E*inverse(gam*v)*theta calculated at each output time.  This will allow
        us later to do u = phi*v*self.E_Igamv_the_bot
        
        Note that if self.drn==1, PTIB then self.botm=self.topm so 
        `self.E_Igamv_the_top` calculated in `make_E_Igamv_the_top`.
        
        Notes
        -----        
        Assuming the load are formulated as the product of separate time and depth 
        dependant functions:
        
        .. math:: \\sigma\\left({Z,t}\\right)=\\sigma\\left({Z}\\right)\\sigma\\left({t}\\right)
        
        the solution to the consolidation equation using the spectral method has 
        the form:
        
        .. math:: u\\left(Z,t\\right)=\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta} 
        
        `make_E_Igamv_the_top` will create `self.E_Igamv_the_bot` which is 
        the :math:`\\mathbf{E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}` 
        part of the solution for all surcharge loads
        
        See also
        --------
        make_E_Igamv_the_top : if self.drn==1, PTIB then `self.botm`=`self.topm` so 
        `self.E_Igamv_the_top` is calculated in `make_E_Igamv_the_top`
        
        """
        
        self.E_Igamv_the_bot = np.zeros((self.neig,len(self.outt)))
        if not self.topm is None:
            for mag_vs_time in self.botm:                
                mv = self.mv
                #1. from mv*du/dt
                theta = integ.dim1sin_ab_linear(self.m, mv.y1, mv.y2, mv.x1, mv.x2, mv.x1, mv.x2)                
                Ebot = integ.EDload_linear(mag_vs_time.x, mag_vs_time.y, self.eigs, self.outt, self.dT)                            
                self.E_Igamv_the_top -= (Ebot*np.dot(self.Igamv, theta)).T
                
                #2. from dTh*et*kh*u
                if sum([v is None for v in [self.et, self.kh, self.dTh]])==0:
                    if self.dTh!=0:
                        et, kh = pwise.polyline_make_x_common(self.et, self.kh) 
                        theta = self.dTh / self.dT * integ.dim1sin_abc_linear(self.m, et.y1, et.y2, kh.y1, kh.y2, et.x1, et.x2, et.x1, et.x2)                
                        Ebot = integ.Eload_linear(mag_vs_time.x, mag_vs_time.y, self.eigs, self.outt, self.dT)            
                        self.E_Igamv_the_top -= (Ebot*np.dot(self.Igamv, theta)).T

                #3. from dTv * D[kz * D[u,z], z]   
                if sum([v is None for v in [self.kh, self.dTv]])==0:
                    if self.dTv!=0:             
                        kv = self.kv
                        theta = self.dTv / self.dT * integ.dim1sin_D_aDb_linear(self.m, kv.y1, kv.y2, kv.x2, kv.x2,kv.x1, kv.x2)                
                        self.E_Igamv_the_top += (Ebot*np.dot(self.Igamv, theta)).T                                    
        return   
    def make_output(self):
        """make all output"""
        
        if not self.porz is None:
            self.make_por()
        if not self.avpz is None:
            self.make_avp()
        if not self.set is None:
            self.set        
        return
        
    def make_por(self):
        """make the por pressure output"""

                
        phi = integ.dim1sin(self.m, self.porz)
        phi_v = np.dot(phi, self.v)


        self.por = np.dot(phi_v, self.E_Igamv_the)
        #top part                
        
        if hasattr(self,'E_Igamv_the_top'):
            for mag_vs_time in self.topm:
                
                self.por += pwise.interp_xa_ya_multipy_x1b_x2b_y1b_y2b(mag_vs_time.x, mag_vs_time.y, [0], [1], [1], [0], self.outt, self.porz)
                if self.drn==1:
                    #bottom part
                    self.por += pwise.interp_xa_ya_multipy_x1b_x2b_y1b_y2b(mag_vs_time.x, mag_vs_time.y, [0], [1], [0], [1], self.outt, self.porz)                                            
        #bot part           
        if self.drn==0 and hasattr(self,'E_Igamv_the_bot'):
            for mag_vs_time in self.botm:
                self.por += pwise.interp_xa_ya_multipy_x1b_x2b_y1b_y2b(mag_vs_time.x, mag_vs_time.y, [0], [1], [0], [1], self.outt, self.porz)
                            

def program(reader, writer):
    """run speccon1d_vert_radial_boundary program
    
    Parameters
    ----------
    reader : file-like object
        Object to read input from.  Can be string, file, StringIO
    writer : file-like object
        Object to write output to can be file or StringIO        
    """
    
    
    inp = make_module_from_text(reader)
    
        
    if inp.__dict__.get('echo', True):
        #maybe process here        
        writer.write(reader)
        
    writer.write(str(inp.a)+'\n')
    writer.write(str(inp.b)+'\n')
    writer.write(str(inp.c)+'\n')
    
    return




    
    
  
    
    
    
        
        
    
    
    
    
    
    
#def yyy(mymod):
#    print(isinstance(mymod, __builtins__.__class__)) #to check if it;s a moduel see http://stackoverflow.com/a/865619/2530083
#    print(type(mymod))
#    print(mymod.a)
#    print(mymod.b)
#    print(mymod.c) 
#
#yyy(mymodule)

if __name__ == '__main__':
    my_code = textwrap.dedent("""\
    from geotecha.piecewise.piecewise_linear_1d import PolyLine
    import numpy as np
    H = 1.0 
    drn = 0 
    dT = 1 
    #dTh = 1
    dTv = 0.1
    neig = 3
        
    mv = PolyLine([0,1], [1,1])
    #kh = PolyLine([0,1], [1,1])
    kz = PolyLine([0,1], [1,1])
    #et = PolyLine([0,1], [1,1])
    surz = PolyLine([0,1], [1,1]) 
    #vacz = PolyLine([0,1], [1,1])
    surm = PolyLine([0,0,3], [0,1,1])
    #vacm = PolyLine([0,0.4,3], [0,-0.2,-0.2])
    topm = PolyLine([0,0.0,3], [0,0.5,0.5])
    botm = PolyLine([0,0.00,3], [0,-0.1,-0.1])
    
    porz = np.linspace(0,1,20)
    #port = np.linspace(0,10,20)
    avpz = [[0,1],[0, 0.5]]
    #avpt = port
    setz = [[0,1],[0, 0.5]]
    #sett = port
    outt = np.linspace(0,3,10)
    """)    
#    writer = StringIO()
#    program(my_code, writer)
#    print('writer\n%s' % writer.getvalue())
    #a = calculate_normalised(my_code)
    a = speccon1d(my_code)
    a.make_gam()    
    #print(a.gam)
    a.make_psi()
    #print(a.psi)
    a.make_eigs_and_v()
    #print(a.eigs)        
    #print(a.v)
    a.make_E_Igamv_the()
    #print(a.E_Igamv_the)
    a.make_por()
    #print(a.por)
    print(len(a.outt))
    print(a.por.shape)
    plt.plot(a.por, a.porz)
    plt.xlabel('Pore pressure')
    plt.ylabel('Normalised depth, Z')
    plt.gca().invert_yaxis()
    plt.show()
#    for i, p in enumerate(a.por.T):
#    
#        print(a.outt[i])        
#        plt.plot(p, a.porz)
#        
#    plt.show()
    
    
    
    
#def main(argv = None):
#    
#    parser = argparse.ArgumentParser()
#    parser.add_argument('-f', '--files', metavar='files', type=str,
#                        nargs='+',
#                        help="files to process")
#
#
#                                
#    if isinstance(files, str):
#        files = [files] # convert single filename to list


    