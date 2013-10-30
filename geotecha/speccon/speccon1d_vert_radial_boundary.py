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
        should contain a statements such as H = 1, drn=0 corresponding to the 
        input attributes listed below.  The user should pick an appropriate
        combination of attributes for their analysis.  e.g. don't put dTh=, 
        kh=, et=, if you are not modelling radial drainage.  You do not have to
        initialize with `reader` but you should know what you are doing.
            
    Attributes
    ----------
    H : float, optional
        total height of soil profile. default = 1.0
    mvref : float, optional
        reference value of volume compressibility mv (used with `H` in 
        settlement calculations). default = 1.0
    kvref : float, optional
        reference value of vertical permeability kv (only used for pretty 
        output). default = 1.0
    khref : float, optional
        reference value of horizontal permeability kh (only used for 
        pretty output). default = 1.0
    etref : float, optional
        reference value of lumped drain parameter et (only used for pretty 
        output). default = 1.0        
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
        the chosen reference values of kv and mv: dTv = kv /(mv*gamw) / H ^ 2 
    dTh : float, optional
        horizontal reference time factor multiplier.  dTh is calculated with 
        the reference values of kh, et, and mv: dTh = kh / (mv * gamw) * et
    mv : PolyLine, optional
        normalised volume compressibility PolyLine(depth, mv)
    kh : PolyLine, optional
        normalised horizontal permeability PolyLine(depth, kh) 
    kv : PolyLine , optional
        normalised vertical permeability PolyLine(depth, kv) 
    et : PolyLine, optional
        normalised vertical drain parameter PolyLine(depth, et). 
        et = 2 / (mu * re^2) where mu is smear-zone/geometry parameter and re
        is radius of influence of vertical drain     
    surcharge_vs_depth : list of Polyline, optional
        surcharge variation with depth. PolyLine(depth, multiplier)
    vacuum_vs_depth : list of Polyline, optinal
        vacuum variation with depth. PolyLine(depth, multiplier)
    surcharge_vs_time : list of Polyline, optional
        surcharge magnitude variation with time. PolyLine(time, magnitude)
    vacuum_vs_time : list of Polyline, optional
        vacuum magnitude variation with time. Polyline(time, magnitude)
    top_vs_time : list of Polyline, optional
        top p.press variation with time. Polyline(time, magnitude)
    bot_vs_time : list of Polyline, optional
        bottom p.press variation with time. Polyline(time, magnitude).
        When drn=1, i.e. PTIB bot_vs_time is equivilent to saying D[u(1,t), Z] = bot_vs_time
    ppress_z : list_like of float, optional
        normalised z to calc pore pressure at    
    avg_ppress_z_pairs : list of two element list of float, optional
        nomalised zs to calc average pore pressure between
        e.g. average of all profile is [[0,1]]
    settlement_z_pairs : list of two element list of float, optional
        normalised depths to calculate normalised settlement between. 
        e.g. surface settlement would be [[0, 1]]
    tvals : list of float
        times to calculate output at
    por : ndarray, only present if ppress_z is input
        calculated pore pressure at depths coreespoinding to `ppress_z` and times corresponding
        to `tvals`.  This is an output array of size (len(ppress_z), len(tvals)).
        por : ndarray
    avp : ndarray, only present if avg_ppress_z_pairs is input
        calculated average pore pressure between depths coreespoinding to `avg_ppress_z_pairs` and 
        times corresponding to `tvals`.  This is an output array of size (len(avg_ppress_z_pairs), len(tvals)).
    set : ndarray, only present if settlement_z_pairs is input
        settlement between depths coreespoinding to `settlement_z_pairs` and 
        times corresponding to `tvals`.  This is an output array of size (len(avg_ppress_z_pairs), len(tvals))
    
    
    Notes
    -----

    governing equation:

    .. math     
    
    
    References
    ----------  
    All based on work by Dr Rohan Walker [1]_, [2]_, [3]_, [4]_
    
    .. [1] Walker, Rohan. 2006. 'Analytical Solutions for Modeling Soft Soil Consolidation by Vertical Drains'. PhD Thesis, Wollongong, NSW, Australia: University of Wollongong.
    .. [2] Walker, R., and B. Indraratna. 2009. 'Consolidation Analysis of a Stratified Soil with Vertical and Horizontal Drainage Using the Spectral Method'. Geotechnique 59 (5) (January): 439-449. doi:10.1680/geot.2007.00019.
    .. [3] Walker, Rohan, Buddhima Indraratna, and Nagaratnam Sivakugan. 2009. 'Vertical and Radial Consolidation Analysis of Multilayered Soil Using the Spectral Method'. Journal of Geotechnical and Geoenvironmental Engineering 135 (5) (May): 657-663. doi:10.1061/(ASCE)GT.1943-5606.0000075.
    .. [4] Walker, Rohan T. 2011. Vertical Drain Consolidation Analysis in One, Two and Three Dimensions'. Computers and Geotechnics 38 (8) (December): 1069-1077. doi:10.1016/j.compgeo.2011.07.006.

    """
    def __init__(self,reader=None):


        self._parameter_defaults = {'H': 1.0, 'drn': 0, 'dT': 1.0, 'neig': 2, 'mvref':1.0, 'kvref': 1.0, 'khref': 1.0, 'etref': 1.0 }
        #self._parameters = 'H drn dT neig dTh dTv mv kh kv et surcharge_vs_depth surcharge_vs_time vacuum_vs_depth vacuum_vs_time top_vs_time bot_vs_time ppress_z port avg_ppress_z_pairs avpt settlement_z_pairs sett'.split()        
        self._parameters = 'H drn dT neig mvref kvref khref etref dTh dTv mv kh kv et surcharge_vs_depth surcharge_vs_time vacuum_vs_depth vacuum_vs_time top_vs_time bot_vs_time ppress_z avg_ppress_z_pairs settlement_z_pairs tvals'.split()        
        
        self._should_be_lists= 'surcharge_vs_depth surcharge_vs_time vacuum_vs_depth vacuum_vs_time top_vs_time bot_vs_time'.split()
        self._should_have_same_z_limits = 'mv kv kh et surcharge_vs_depth vacuum_vs_depth'.split()
        self._should_have_same_len_pairs = 'surcharge_vs_depth surcharge_vs_time vacuum_vs_depth vacuum_vs_time'.split() #pairs that should have the same length
        
        
#        #if you don't care about autocomplete for the parameters you can use this loop 
#        for v in self._parameters:
#            self.__setattr__(v, self._parameter_defaults.get(v, None))
                   
        self.H = self._parameter_defaults.get('H')               
        self.drn = self._parameter_defaults.get('drn')
        self.dT = self._parameter_defaults.get('dT')
        self.neig = self._parameter_defaults.get('neig')        
        self.mvref = self._parameter_defaults.get('mvref')
        self.kvref = self._parameter_defaults.get('kvref')
        self.khref = self._parameter_defaults.get('khref')
        self.etref = self._parameter_defaults.get('etref')
        self.dTh = None
        self.dTv = None
                    
        self.mv = None #normalised volume compressibility PolyLine(depth, mv)
        self.kh = None #normalised horizontal permeability PolyLine(depth, kh) 
        self.kv = None #normalised vertical permeability PolyLine(depth, kv) 
        self.et = None #normalised vertical drain parameter PolyLine(depth, et) 

        self.surcharge_vs_depth = None #surcharge list of PolyLine(depth, multiplier)
        self.vacuum_vs_depth = None #vacuum list of PolyLine(depth, multiplier)
        self.surcharge_vs_time = None #surcharge list of PolyLine(time, magnitude)
        self.vacuum_vs_time = None #vacuum list of Polyline(time, magnitude)
        self.top_vs_time = None #top p.press list of Polyline(time, magnitude)
        self.bot_vs_time = None #bot p.press list of Polyline(time, magnitude)
        
        self.ppress_z = None #normalised z to calc pore pressure at
        #self.port = None #times to calc pore pressure at
        self.avg_ppress_z_pairs = None #nomalised zs to calc average pore pressure between
        #self.avpt = None #times to calc average pore pressure at
        self.settlement_z_pairs = None #normalised depths to calculate normalised settlement between    
        #self.sett = None #times to calc normalised settlement at
        self.tvals = None        
        
        self.text = None
        if not reader is None:
            self._grab_input_from_text(reader)
            
        #self._m = None            
    

    def check_all(self):
        """perform all checks
        
        See also
        --------                        
        self._check_for_input_errors : check for a variety of input errors
        self._check_list_inputs : for inputs that should be lists puts 
        non-lists in a list                
        self._check_z_limits : checks that all depth dependant matrices have 
        the same z-limits        
        self._check_len_pairs : checks that each load has a depth dependent 
        part and a corresponding time dependent part
        
        """        
        
        self._check_for_input_errors()
        self._check_list_inputs(self._should_be_lists)        
        self._check_z_limits(self._should_have_same_z_limits)
        self._check_len_pairs(self._should_have_same_len_pairs)
        
        return
        
    def make_all(self):
        """run checks, make all arrays, make output
        
        Generally run this after input is in place (either through 
        initializing the class with a reader/text/fileobject or 
        through some other means)
        
        See also
        --------
        check_all
        make_time_independent_arrays
        make_time_dependent_arrays
        make_output
        """
        self.check_all()
        self.make_time_independent_arrays()
        self.make_time_dependent_arrays()
        self.make_output()
        
        return
        
    def make_time_independent_arrays(self):
        """make all time independent arrays
        
        
        See also
        --------
        self._make_m : make the basis function eigenvalues
        self._make_gam : make the mv dependent gamma matrix
        self._make_psi : make the kv, kh, et dependent psi matrix
        self._make_eigs_and_v : make eigenvalues, eigenvectors and I_gamv
        
        """        
        
                
        
        
        
        self._make_m()
        self._make_gam()
        self._make_psi()
        self._make_eigs_and_v()
                        
        return

    def make_time_dependent_arrays(self):
        """make all time dependent arrays
        
        See also
        --------
        self.make_E_Igamv_the()
        
        """
        self.make_E_Igamv_the()
        self.v_E_Igamv_the=np.dot(self.v, self.E_Igamv_the)
        return
    def make_output(self):
        """make all output"""
        
        if not self.ppress_z is None:                                 
            self._make_por()
        if not self.avg_ppress_z_pairs is None:            
            self._make_avp()
        if not self.settlement_z_pairs is None:                      
            self._make_set()        
        return
        
        
    def _make_m(self):
        """make the basis function eigenvalues                
        
        Notes
        -----
        
        .. math:: m_i =\\pi*\\left(i+1-drn/2\\right) 
        
        for :math:`i = 1\:to\:neig-1`
        
        """
        if sum(v is None for v in[self.neig, self.drn])!=0:
            raise ValueError('neig and/or drn is not defined')
        self.m = integ.m_from_sin_mx(np.arange(self.neig), self.drn)         
        return
          
    def _check_for_input_errors(self):
        """checks for various input inconsistencies"""
        
        
        #bad material prop input errors  
        if self.mv is None:
            raise ValueError('No mv values given. '); sys.exit(0)    
        if self.dTh is None and self.dTv is None:
            raise ValueError('Neither dTv or dTh are defined. Need at least one'); sys.exit(0)
        if self.kh is None and self.kv is None:
            raise ValueError('Neither kh and kv are defined. Need at least one'); sys.exit(0)        
        if sum([v is None for v in [self.dTh, self.kh, self.et]]) in [1,2]:
            raise ValueError('dTh, kh, or et is not defined, need all three to model radial drainage/vacuum'); sys.exit(0)    
        if sum([v is None for v in [self.dTv, self.kv]])==1:
            raise ValueError('dTv or kv is not defined, need both to model vertical drainage.'); sys.exit(0)        
        
        
            
        #bad load input errors
        if sum([v is None for v in [self.surcharge_vs_depth, self.surcharge_vs_time]]) == 1:
            raise ValueError('surcharge_vs_depth or surcharge_vs_time is not defined, need both to model surcharge'); sys.exit(0)        
        if sum([v is None for v in [self.vacuum_vs_depth, self.vacuum_vs_time]]) == 1:
            raise ValueError('vacuum_vs_depth or vacuum_vs_time is not defined, need both to model vacuum'); sys.exit(0)
#        if self.drn==1 and not (self.bot_vs_time is None):
#            raise ValueError('bot_vs_time is meaningless when bottom is impervious (drn=1).  remove drn or change drn to 1'); sys.exit(0)
        if sum([v is None for v in [self.surcharge_vs_depth, self.surcharge_vs_time, self.vacuum_vs_depth, self.vacuum_vs_time, self.top_vs_time, self.bot_vs_time]])==6:
            raise ValueError('surcharge_vs_depth, surcharge_vs_time, vacuum_vs_depth, vacuum_vs_time, top_vs_time, and bot_vs_time not defined.  i.e. no loadings specified.'); sys.exit(0)        
        if (sum([v is None for v in [self.vacuum_vs_depth, self.vacuum_vs_time]])==0) and (sum([v is None for v in [self.dTh, self.kh, self.et]])>0):
            raise ValueError('vacuum_vs_depth, vacuum_vs_time, defined but one or more of dTh, kh, et not defined.  To model vacuum need dTh, kh and et'); sys.exit(0)        
            
        #bad output specifying errors    
#        if sum([v is None for v in [self.ppress_z, self.port]])==1:
#            raise ValueError('ppress_z or port is not defined, need both to output pore pressure at depth'); sys.exit(0)    
#        if sum([v is None for v in [self.avg_ppress_z_pairs, self.avpt]])==1:
#            raise ValueError('avg_ppress_z_pairs or avpt is not defined, need both to output average pore pressure'); sys.exit(0)            
#        if sum([v is None for v in [self.settlement_z_pairs, self.sett]])==1:
#            raise ValueError('settlement_z_pairs or sett is not defined, need both to output settlement'); sys.exit(0)        
#        if sum([v is None for v in [self.ppress_z, self.port, self.avg_ppress_z_pairs, self.avpt,self.settlement_z_pairs, self.sett]])==6:
#            raise ValueError('ppress_z, port, avg_ppress_z_pairs, avpt, settlement_z_pairs, and sett not specified.  i.e. no output specified.'); sys.exit(0)
        if self.tvals is None:
            raise ValueError('tvals not specified.  i.e. no output.'); sys.exit(0)                        
        if sum([v is None for v in [self.ppress_z, self.avg_ppress_z_pairs, self.settlement_z_pairs]])==3:            
            raise ValueError('ppress_z, avg_ppress_z_pairs, settlement_z_pairs and not specified.  i.e. no output specified.'); sys.exit(0)            
    
    
           
    def _grab_input_from_text(self, reader):
        """grabs input parameters from fileobject, StringIO, text"""
        #self.text = reader.read()
        
        inp = make_module_from_text(reader)
        
        for v in self._parameters:
            self.__setattr__(v, inp.__dict__.get(v,self._parameter_defaults.get(v, None)))                
                        
    def _check_list_inputs(self, check_list):                
        """puts non-lists in a list"""
        
        g = self.__getattribute__
        for v in check_list:
            if not g(v) is None:
                if not isinstance(g(v), list):
                    self.__setattr__(v,[g(v)])                
                
    def _check_z_limits(self, check_list):
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
    def _check_len_pairs(self, check_list):
        """checks pairs of parameters that hsould have eth same length"""

        g = self.__getattribute__
        
        # for iterating in chuncks see http://stackoverflow.com/a/434328/2530083
        for v1, v2 in [check_list[pos:pos + 2] for pos in xrange(0, len(check_list), 2)]:
            if not g(v1) is None and not g(v2) is None:
                if len(g(v1)) != len(g(v2)):
                    raise ValueError("%s has %d elements, %s has %d elements.  They should have the same number of elements." % (v1,len(g(v1)), v2, len(g(v2))))

    def _make_m(self):
        """make the basis function eigenvalues
        
        m in u = sin(m * Z)
        
        Notes
        -----
        
        .. math:: m_i =\\pi*\\left(i+1-drn/2\\right) 
        
        for :math:`i = 1\:to\:neig-1`
        
        """
        if sum(v is None for v in[self.neig, self.drn])!=0:
            raise ValueError('neig and/or drn is not defined')
        self.m = integ.m_from_sin_mx(np.arange(self.neig), self.drn)         
        return
        
    def _make_gam(self):
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
        
    def _make_psi(self):
        """make kv, kh, et dependant psi matrix    
        
        Notes
        -----
        Creates the :math: `\Psi` matrix which occurs in the following equation:
        
        .. math:: \\mathbf{\\Gamma}\\mathbf{A}'=\\mathbf{\\Psi A}+loading\\:terms
        
        `self.psi`, :math:`\Psi` is given by:          
        
        .. math:: \\mathbf{\Psi}_{i,j}=dT_h\\mathbf{A}_{i,j}=\\int_{0}^1{{k_h\\left(z\\right)}{\eta\\left(z\\right)}\\phi_i\\phi_j\\,dz}-dT_v\\int_{0}^1{\\frac{d}{dz}\\left({k_z\\left(z\\right)}\\frac{d\\phi_j}{dz}\\right)\\phi_i\\,dz}
        
        """        
        
        self.psi = np.zeros((self.neig, self.neig))           
        #kv part
        if sum([v is None for v in [self.kv, self.dTv]])==0:            
            self.psi -= self.dTv / self.dT * integ.dim1sin_D_aDf_linear(self.m, self.kv.y1, self.kv.y2, self.kv.x1, self.kv.x2)
        #kh & et part
        if sum([v is None for v in [self.kh, self.et, self.dTh]])==0:
            kh, et = pwise.polyline_make_x_common(self.kh, self.et)
            self.psi += self.dTh / self.dT * integ.dim1sin_abf_linear(self.m,kh.y1, kh.y2, et.y1, et.y2, kh.x1, kh.x2)
        self.psi[np.abs(self.psi)<1e-8]=0.0
        return        
        

    def _make_eigs_and_v(self):
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
        of size (neig, len(tvals)). So the columns are the column array 
        E*inverse(gam*v)*theta calculated at each output time.  This will allow
        us later to do u = phi*v*self.E_Igamv_the 
        
        See also
        --------
        _make_E_Igamv_the_surcharge :  surchage contribution
        _make_E_Igamv_the_vacuum : vacuum contribution
        _make_E_Igamv_the_BC : top boundary pore pressure contribution
        _make_E_Igamv_the_bot : bottom boundary pore pressure contribution        
        
        """
        
        self.E_Igamv_the = np.zeros((self.neig,len(self.tvals)))
        if sum([v is None for v in [self.surcharge_vs_depth, self.surcharge_vs_time]])==0:
            self._make_E_Igamv_the_surcharge()
            self.E_Igamv_the += self.E_Igamv_the_surcharge
        if sum([v is None for v in [self.vacuum_vs_depth, self.vacuum_vs_time, self.et, self.kh,self.dTh]])==0:
            if self.dTh!=0:
                self._make_E_Igamv_the_vacuum()
                self.E_Igamv_the += self.E_Igamv_the_vacuum                                                        
        if not self.top_vs_time is None or not self.bot_vs_time is None:
            self._make_E_Igamv_the_BC()
            self.E_Igamv_the += self.E_Igamv_the_BC
        
                       
        return
                    
            
        


    
    def _make_E_Igamv_the_surcharge(self):
        """make the surcharge loading matrices
        
        Make the E*inverse(gam*v)*theta part of solution u=phi*vE*inverse(gam*v)*theta. 
        The contribution of each surcharge load is added and put in 
        `self.E_Igamv_the_surcharge`. `self.E_Igamv_the_surcharge` is an array
        of size (neig, len(tvals)). So the columns are the column array 
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
        
        `_make_E_Igamv_the_surcharge` will create `self.E_Igamv_the_surcharge` which is 
        the :math:`\\mathbf{E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}` 
        part of the solution for all surcharge loads
        
        """
        
        #self.E_Igamv_the_surcharge = np.zeros((self.neig,len(self.tvals)))        
        self.E_Igamv_the_surcharge  = dim1sin_E_Igamv_the_aDmagDt_bilinear(self.m, self.eigs, self.mv, self.surcharge_vs_depth, self.surcharge_vs_time,self.tvals, self.Igamv, self.dT)        
        return

    def _make_E_Igamv_the_vacuum(self):
        """make the vacuum loading matrices
        
        Make the E*inverse(gam*v)*theta part of solution u=phi*vE*inverse(gam*v)*theta. 
        The contribution of each vacuum load is added and put in 
        `self.E_Igamv_the_vacuum`. `self.E_Igamv_the_vacuum` is an array
        of size (neig, len(tvals)). So the columns are the column array 
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
        
        `_make_E_Igamv_the_surcharge` will create `self.E_Igamv_the_surcharge` which is 
        the :math:`\\mathbf{E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}` 
        part of the solution for all surcharge loads
        
        """
        
        
        #self.E_Igamv_the_vacuum = np.zeros((self.neig, len(self.tvals)))
        self.E_Igamv_the_vacuum= self.dTh*dim1sin_E_Igamv_the_abmag_bilinear(self.m, self.eigs, self.kh, self.et, 
                                                                        self.vacuum_vs_depth, self.vacuum_vs_time, self.tvals, self.Igamv, self.dT)
       
        return
    def _make_E_Igamv_the_BC(self):
        
        self.E_Igamv_the_BC = np.zeros((self.neig, len(self.tvals)))        
        self.E_Igamv_the_BC -= dim1sin_E_Igamv_the_BC_aDfDt_linear(self.drn, self.m, self.eigs, self.mv, self.top_vs_time, self.bot_vs_time, self.tvals, self.Igamv, self.dT)        
        
        if sum([v is None for v in [self.et, self.kh,self.dTh]])==0:
            if self.dTh!=0:                
                self.E_Igamv_the_BC -= self.dTh / self.dT * dim1sin_E_Igamv_the_BC_abf_linear(self.drn, self.m, self.eigs, self.kh, self.et, self.top_vs_time, self.bot_vs_time, self.tvals, self.Igamv, self.dT)                
        if sum([v is None for v in [self.kv,self.dTv]])==0:
            if self.dTv!=0:                                
                self.E_Igamv_the_BC += self.dTv / self.dT * dim1sin_E_Igamv_the_BC_D_aDf_linear(self.drn, self.m, self.eigs, self.mv, self.top_vs_time, self.bot_vs_time, self.tvals, self.Igamv, self.dT)
        return          
    
        
    def _make_por(self):
        """make the pore pressure output
        
        makes `self.por`, the average pore pressure at depths corresponding to
        self.ppress_z and times corresponding to self.tvals.  `self.por`  has size
        (len(ppress_z), len(tvals)).
        
        Notes
        -----
        Solution to consolidation equation with spectral method for pore pressure at depth is :
        
        .. math:: u\\left(Z,t\\right)=\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}+u_{top}\\left({t}\\right)\\left({1-Z}\\right)+u_{bot}\\left({t}\\right)\\left({Z}\\right)
                
        For pore pressure :math:`\\Phi` is simply :math:`sin\\left({mZ}\\right)` for each value of m
        
        
        """
        

        self.por= dim1sin_f(self.m, self.ppress_z, self.tvals, self.v_E_Igamv_the, self.drn, self.top_vs_time, self.bot_vs_time)

                
        
    def _make_avp(self):                            
        """calculate average pore pressure
        
        makes `self.avp`, the average pore pressure at depths corresponding to
        self.avg_ppress_z_pairs and times corresponding to self.tvals.  `self.avp`  has size
        (len(ppress_z), len(tvals)).
        
        
        Notes
        -----
        The average pore pressure between Z1 and Z2 is given by:
        
        .. math:: \\overline{u}\\left(\\left({Z_1,Z_2}\\right),t\\right)=\\int_{Z_1}^{Z_2}{\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}+u_{top}\\left({t}\\right)\\left({1-Z}\\right)+u_{bot}\\left({t}\\right)\\left({Z}\\right)\,dZ}/\\left({Z_2-Z_1}\\right)
                                        
        """
        
        self.avp=dim1sin_avgf(self.m, self.avg_ppress_z_pairs, self.tvals, self.v_E_Igamv_the, self.drn, self.top_vs_time, self.bot_vs_time)
        
        

        
    def _make_set(self):        
        """calculate settlement
        
        makes `self.set`, the average pore pressure at depths corresponding to
        self.settlement_z_pairs and times corresponding to self.tvals.  `self.set`  has size
        (len(ppress_z), len(tvals)).
        
        
        Notes
        -----
        The average settlement between Z1 and Z2 is given by:
        
        .. math:: \\overline{\\rho}\\left(\\left({Z_1,Z_2}\\right),t\\right)=\\int_{Z_1}^{Z_2}{m_v\\left({Z}\\right)\\left({\\sigma\\left({Z,t}\\right)-u\\left({Z,t}\\right)}\\right)\\,dZ}
                
        
        .. math:: \\overline{\\rho}\\left(\\left({Z_1,Z_2}\\right),t\\right)=\\int_{Z_1}^{Z_2}{m_v\\left({Z}\\right)\\sigma\\left({Z,t}\\right)\\,dZ}+\\int_{Z_1}^{Z_2}{m_v\\left({Z}\\right)\\left({\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}+u_{top}\\left({t}\\right)\\left({1-Z}\\right)+u_{bot}\\left({t}\\right)\\left({Z}\\right)}\\right)\\,dZ}
                                        
        """
        z1 = np.asarray(self.settlement_z_pairs)[:,0]
        z2 = np.asarray(self.settlement_z_pairs)[:,1]
        self.set=-dim1sin_integrate_af(self.m, self.settlement_z_pairs, self.tvals, 
                                       self.v_E_Igamv_the, 
                                        self.drn, self.mv, self.top_vs_time, self.bot_vs_time)
        
        self.set+=pwise.pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between_super(self.surcharge_vs_time, self.surcharge_vs_depth, self.mv, self.tvals, z1,z2,achoose_max=True)            
        self.set *= self.H * self.mvref

                                      
        return


def dim1sin_f(m, outz, tvals, v_E_Igamv_the, drn, top_vs_time = None, bot_vs_time=None):
    """assemble output u(Z,t) = phi * v_E_Igam_v_the + utop(t) * (1-Z) + ubot(t)*Z
    
    Basically calculates the phi part for each tvals value, then dot product
    with v_E_Igamv_the.  Then account for non-zero boundary conditions by 
    adding utop(t)*(1-Z) and ubot(t)*Z parts for each outz, tvals pair
    
    
    Parameters
    ----------
        
    
    Notes
    -----
    
    
    """
    
    phi = integ.dim1sin(m, outz)
    u = np.dot(phi, v_E_Igamv_the)    
    #top part
    if not top_vs_time is None:        
        for mag_vs_time in top_vs_time:
            if drn==1: 
                u += pwise.pinterp_x_y(mag_vs_time, tvals, choose_max=True)                                            
            else:
                u += pwise.pinterp_xa_ya_multipy_x1b_x2b_y1b_y2b(mag_vs_time, PolyLine([0], [1], [1], [0]), tvals, outz, achoose_max=True)  
    #bot part           
    if not bot_vs_time is None:
        for mag_vs_time in bot_vs_time:
            u += pwise.pinterp_xa_ya_multipy_x1b_x2b_y1b_y2b(mag_vs_time, PolyLine([0], [1], [0], [1]), tvals, outz, achoose_max=True)          
    return u 

def dim1sin_avgf(m, z, tvals, v_E_Igamv_the, drn, top_vs_time = None, bot_vs_time=None):
    """Average u between Z1 and Z;: u(Z,t) = phi * v_E_Igam_v_the + utop(t) * (1-Z) + ubot(t)*Z
    
    Basically calculates the average phi part for each tvals value, then dot product
    with v_E_Igamv_the.  Then account for non-zero boundary conditions by 
    adding average of utop(t)*(1-Z) and average of ubot(t)*Z parts for each 
    avgz, tvals pair.
    
    
    Parameters
    ----------
        
    
    Notes
    -----
    
    
    """
    
    phi = integ.dim1sin_avg_between(m, z)
    

    avg = np.dot(phi, v_E_Igamv_the)
                   
    z1 = np.asarray(z)[:,0]
    z2 = np.asarray(z)[:,1]
    
    #top part                                 
    if not top_vs_time is None:
        for mag_vs_time in top_vs_time:                            
            if drn==1:
                #bottom part
                avg += pwise.pinterp_x_y(mag_vs_time, tvals, choose_max=True)
            else:                                                            
                avg += pwise.pxa_ya_multipy_avg_x1b_x2b_y1b_y2b_between(mag_vs_time,
                                                                        PolyLine([0], [1], [1], [0]),
                                                                        tvals, z1, z2, achoose_max=True)                                                                                                                
    #botom part           
    if not bot_vs_time is None:
        for mag_vs_time in bot_vs_time:
            avg += pwise.pxa_ya_multipy_avg_x1b_x2b_y1b_y2b_between(mag_vs_time,
                                                                        PolyLine([0], [1], [0], [1]),
                                                                        tvals, z1,z2, achoose_max=True)                                                                             

    return avg    




def dim1sin_integrate_af(m, z, tvals, v_E_Igamv_the, drn, a, top_vs_time = None, bot_vs_time=None):
    """Integrate between Z1 and Z2: a(Z)*(phi * v_E_Igam_v_the + utop(t) * (1-Z) + ubot(t)*Z)
        
    
    
    Parameters
    ----------
        
    
    Notes
    -----
    
    
    """
    
    
    z1 = np.array(z)[:,0]
    z2 = np.array(z)[:,1]            
    #a*u part
    phi = integ.pdim1sin_a_linear_between(m, a, z)
    
    out = np.dot(phi, v_E_Igamv_the)    
                                                   
    #top part                        
    if not top_vs_time is None:
        for mag_vs_time in top_vs_time:                            
            if drn==1:                                
                out += pwise.pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(
                    mag_vs_time,
                    a,
                    PolyLine(a.x1, a.x2, np.ones_like(a.x1), np.ones_like(a.x2)),
                    tvals, z1, z2, achoose_max=True)  
            else:
                out += pwise.pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(
                    mag_vs_time,
                    a,
                    PolyLine(a.x1, a.x2, 1-a.x1, 1-a.x2),
                    tvals, z1, z2, achoose_max=True)
                                                        
                                                                                                                                
    #bot part           
    if not bot_vs_time is None:
        for mag_vs_time in bot_vs_time:
            out += pwise.pxa_ya_multiply_integrate_x1b_x2b_y1b_y2b_multiply_x1c_x2c_y1c_y2c_between(
                    mag_vs_time,
                    a,
                    PolyLine(a.x1, a.x2, a.x1, a.x2),
                    tvals, z1, z2, achoose_max=True) 
    #self.set *= self.H * self.mvref                                
    return out
    
def dim1sin_E_Igamv_the_aDmagDt_bilinear(m, eigs, a, mag_vs_depth, mag_vs_time, tvals, Igamv, dT=1.0):
    """Loading dependant E_Igamv_the matrix for a(z)*D[mag(z, t), t] where mag is bilinear in depth and time
    
    Make the E*inverse(gam*v)*theta part of solution u=phi*v*E*inverse(gam*v)*theta. 
    The contribution of each `mag_vs_time`-`mag_vs_depth` pair are superposed. 
    The result is an array
    of size (neig, len(tvals)). So the columns are the column array 
    E*inverse(gam*v)*theta calculated at each output time.  This will allow
    us later to do u = phi*v*E_Igamv_the

    Uses sin(m*z) in the calculation of theta.
    
    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    eigs : 1d numpy.ndarray
        list of eigenvalues
    a : PolyLine
        Piewcewise linear function.  e.g. for 1d consolidation surcharge
        loading term is mv*D[sigma(z, t), t] so a would be mv.
    mag_vs_depth : list of PolyLine
        Piecewise linear magnitude  vs depth.
    mag_vs_time : list of PolyLine
        Piecewise linear magnitude vs time
    tvals : 1d numpy.ndarray`
        list of time values to calculate integral at
    dT : ``float``, optional
        time factor multiple (default = 1.0)        
    
    Returns
    -------
    E_Igamv_the: ndarray
        loading matrix        
    
    Notes
    -----        
    Assuming the loads are formulated as the product of separate time and depth 
    dependant functions: 
    
    .. math:: \\sigma\\left({Z,t}\\right)=\\sigma\\left({Z}\\right)\\sigma\\left({t}\\right)
    
    the solution to the consolidation equation using the spectral method has 
    the form:
    
    .. math:: u\\left(Z,t\\right)=\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta} 
    
    In this instance :math:`\\sigma\\left({Z}\\right)` 
    and :math:`\\sigma\\left({t}\\right)` are piecewise linear in depth and 
    time (hence the 'bilinear' in the function name).
    
    `dim1sin_E_Igamv_the_aDmagDt_bilinear` will calculate  
    :math:`\\mathbf{E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}` 
    for terms with the form:
    
    .. math:: a\\left({z}\\right)\\frac{\\partial\\sigma\\left({Z,t}\\right)}{\\partial t}
    
    where :math:`a\\left(z\\right)` is a piecewise linear function 
    w.r.t. :math:`z`
    
    """
    
    E_Igamv_the = np.zeros((len(m), len(tvals)))
    
    
    if sum([v is None for v in [mag_vs_depth, mag_vs_time]])==0:
        
        for mag_vs_t, mag_vs_z in zip(mag_vs_time, mag_vs_depth):
            a, mag_vs_z = pwise.polyline_make_x_common(a, mag_vs_z)                      
            theta = integ.pdim1sin_ab_linear(m, a, mag_vs_z) 
            E = integ.pEDload_linear(mag_vs_t, eigs, tvals, dT)                                   

            #theta is 1d array, Igamv is nieg by neig array, np.dot(Igamv, theta) 
            #and np.dot(theta, Igamv) will give differetn 1d arrays.  
            #Basically np.dot(Igamv, theta) gives us what we want i.e. 
            #theta was treated as a column array.  The alternative 
            #np.dot(theta, Igamv) would have treated theta as a row vector.
            E_Igamv_the += (E*np.dot(Igamv, theta)).T
            
    return E_Igamv_the


def dim1sin_E_Igamv_the_abmag_bilinear(m, eigs, a,b, mag_vs_depth, mag_vs_time, tvals, Igamv, dT=1.0):
    """Loading dependant E_Igamv_the matrix for a(z)*b(z)*D[mag(z, t), t] where mag is bilinear in depth and time
    
    Make the E*inverse(gam*v)*theta part of solution u=phi*v*E*inverse(gam*v)*theta. 
    The contribution of each `mag_vs_time`-`mag_vs_depth` pair are superposed. 
    The result is an array
    of size (neig, len(tvals)). So the columns are the column array 
    E*inverse(gam*v)*theta calculated at each output time.  This will allow
    us later to do u = phi*v*E_Igamv_the

    Uses sin(m*z) in the calculation of theta.
    
    Parameters
    ----------
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    eigs : 1d numpy.ndarray
        list of eigenvalues
    a : PolyLine
        Piewcewise linear function.  e.g. for 1d consolidation surcharge
        loading term is mv*D[sigma(z, t), t] so a would be mv.
    b : PolyLine
        Piewcewise linear function.  e.g. for 1d consolidation vacuum term is 
         kh*et*w(z,t) so a would be `kh`, `b` would be `et`
    mag_vs_depth : list of PolyLine
        Piecewise linear magnitude  vs depth.
    mag_vs_time : list of PolyLine
        Piecewise linear magnitude vs time
    tvals : 1d numpy.ndarray`
        list of time values to calculate integral at
    dT : ``float``, optional
        time factor multiple (default = 1.0)        
    
    Returns
    -------
    E_Igamv_the: ndarray
        loading matrix        
    
    Notes
    -----        
    Assuming the loads are formulated as the product of separate time and depth 
    dependant functions: 
    
    .. math:: \\sigma\\left({Z,t}\\right)=\\sigma\\left({Z}\\right)\\sigma\\left({t}\\right)
    
    the solution to the consolidation equation using the spectral method has 
    the form:
    
    .. math:: u\\left(Z,t\\right)=\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta} 
    
    In this instance :math:`\\sigma\\left({Z}\\right)` 
    and :math:`\\sigma\\left({t}\\right)` are piecewise linear in depth and 
    time (hence the 'bilinear' in the function name).
    
    `dim1sin_E_Igamv_the_abDmagDt_bilinear` will calculate  
    :math:`\\mathbf{E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}` 
    for terms with the form:
    
    .. math:: a\\left({z}\\right)b\\left({z}\\right)\\frac{\\partial\\sigma\\left({Z,t}\\right)}{\\partial t}
    
    where :math:`a\\left(z\\right)`, :math:`b\\left(z\\right)` are 
    piecewise linear functions w.r.t. :math:`z`.
    
    
    """
    
    E_Igamv_the = np.zeros((len(m), len(tvals)))
    
    
    if sum([v is None for v in [mag_vs_depth, mag_vs_time]])==0:
        
        for mag_vs_t, mag_vs_z in zip(mag_vs_time, mag_vs_depth):
            a, b , mag_vs_z = pwise.polyline_make_x_common(a, b, mag_vs_z)                      
            theta = integ.pdim1sin_abc_linear(m, a, b, mag_vs_z) 
            E = integ.pEload_linear(mag_vs_t, eigs, tvals, dT)                                   

            #theta is 1d array, Igamv is nieg by neig array, np.dot(Igamv, theta) 
            #and np.dot(theta, Igamv) will give differetn 1d arrays.  
            #Basically np.dot(Igamv, theta) gives us what we want i.e. 
            #theta was treated as a column array.  The alternative 
            #np.dot(theta, Igamv) would have treated theta as a row vector.
            E_Igamv_the += (E*np.dot(Igamv, theta)).T
            
    return E_Igamv_the

#def dim1sin_E_Igamv_the_aDfDt_bilinear(m, eigs, a, mag_vs_depth, mag_vs_time, tvals, Igamv, dT=1.0):
#    """Loading dependant E_Igamv_the matrix for a(z)*D[u(z, t), t] where mag is bilinear in depth and time
#    
#    Make the E*inverse(gam*v)*theta part of solution u=phi*v*E*inverse(gam*v)*theta. 
#    The contribution of each `mag_vs_time`-`mag_vs_depth` pair are superposed. 
#    The result is an array
#    of size (neig, len(tvals)). So the columns are the column array 
#    E*inverse(gam*v)*theta calculated at each output time.  This will allow
#    us later to do u = phi*v*E_Igamv_the
#
#    Uses sin(m*z) in the calculation of theta.
#    
#    Parameters
#    ----------
#    m : ``list`` of ``float``
#        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
#    eigs : 1d numpy.ndarray
#        list of eigenvalues
#    a : PolyLine
#        Piewcewise linear function.  e.g. for 1d consolidation surcharge
#        loading term is mv*D[sigma(z, t), t] so a would be mv.
#    mag_vs_depth : list of PolyLine
#        Piecewise linear magnitude  vs depth.
#    mag_vs_time : list of PolyLine
#        Piecewise linear magnitude vs time
#    tvals : 1d numpy.ndarray`
#        list of time values to calculate integral at
#    dT : ``float``, optional
#        time factor multiple (default = 1.0)        
#    
#    Returns
#    -------
#    E_Igamv_the: ndarray
#        loading matrix        
#    
#    Notes
#    -----        
#    Assuming the loads are formulated as the product of separate time and depth 
#    dependant functions: 
#    
#    .. math:: \\sigma\\left({Z,t}\\right)=\\sigma\\left({Z}\\right)\\sigma\\left({t}\\right)
#    
#    the solution to the consolidation equation using the spectral method has 
#    the form:
#    
#    .. math:: u\\left(Z,t\\right)=\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta} 
#    
#    In this instance :math:`\\sigma\\left({Z}\\right)` 
#    and :math:`\\sigma\\left({t}\\right)` are piecewise linear in depth and 
#    time (hence the 'bilinear' in the function name).
#    
#    `dim1sin_E_Igamv_the_aDmagDt_bilinear` will calculate  
#    :math:`\\mathbf{E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}` 
#    for terms with the form:
#    
#    .. math:: a\\left({z}\\right)\\frac{\\partial\\sigma\\left({Z,t}\\right)}{\\partial t}
#    
#    where :math:`a\\left(z\\right)` is a piecewise linear function 
#    w.r.t. :math:`z`
#    
#    """
#    
#    E_Igamv_the = np.zeros((len(m), len(tvals)))
#    
#    
#    if sum([v is None for v in [mag_vs_depth, mag_vs_time]])==0:
#        
#        for mag_vs_t, mag_vs_z in zip(mag_vs_time, mag_vs_depth):
#            a, mag_vs_z = pwise.polyline_make_x_common(a, mag_vs_z)                      
#            theta = integ.pdim1sin_ab_linear(m, a, mag_vs_z) 
#            E = integ.pEDload_linear(mag_vs_t, eigs, tvals, dT)                                   
#
#            #theta is 1d array, Igamv is nieg by neig array, np.dot(Igamv, theta) 
#            #and np.dot(theta, Igamv) will give differetn 1d arrays.  
#            #Basically np.dot(Igamv, theta) gives us what we want i.e. 
#            #theta was treated as a column array.  The alternative 
#            #np.dot(theta, Igamv) would have treated theta as a row vector.
#            E_Igamv_the += (E*np.dot(Igamv, theta)).T
#            
#    return E_Igamv_the

def dim1sin_E_Igamv_the_BC_aDfDt_linear(drn, m, eigs, a, top_vs_time, bot_vs_time, tvals, Igamv, dT=1.0):
    """Loading dependant E_Igamv_the matrix that arise from homogenising a(z)*D[u(z, t), t] for non_zero top and bottom boundary conditions
    
    When accounting for non-zero boundary conditions we homogenise the 
    governing equation by letting u(Z,t) = v(Z,t) + utop(t)*(1-Z) + ubot(t)*Z 
    and solving for v(Z, t).  This function calculates the 
    E*inverse(gam*v)*theta part of solution v(Z,t)=phi*v*E*inverse(gam*v)*theta. 
    For the terms that arise by subbing the BC's into terms like a(z)*D[u(Z,t), t]
    
    The contribution of each `mag_vs_time` are superposed. 
    The result is an array
    of size (neig, len(tvals)). So the columns are the column array 
    E*inverse(gam*v)*theta calculated at each output time.  This will allow
    us later to do v(Z,t) = phi*v*E_Igamv_the

    Uses sin(m*z) in the calculation of theta.
    
    Parameters
    ----------
    drn : [0,1]
        drainage condition,
        0 = Pervious top pervious bottom (PTPB)
        1 = Pervious top impoervious bottom (PTIB)
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    eigs : 1d numpy.ndarray
        list of eigenvalues
    a : PolyLine
        Piewcewise linear function.  e.g. for 1d consolidation surcharge
        loading term is mv*D[sigma(z, t), t] so a would be mv.
    top_vs_time : list of PolyLine
        Piecewise linear magnitude  vs time for the top boundary.
    bot_vs_time : list of PolyLine
        Piecewise linear magnitude vs time for the bottom boundary.
    tvals : 1d numpy.ndarray`
        list of time values to calculate integral at
    dT : ``float``, optional
        time factor multiple (default = 1.0)        
    
    Returns
    -------
    E_Igamv_the: ndarray
        loading matrix        
    
    Notes
    -----        
    Assuming the loads are formulated as the product of separate time and depth 
    dependant functions: 
    
    .. math:: \\sigma\\left({Z,t}\\right)=\\sigma\\left({Z}\\right)\\sigma\\left({t}\\right)
    
    the solution to the consolidation equation using the spectral method has 
    the form:
    
    .. math:: u\\left(Z,t\\right)=\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta} 
    
    
    when we consider non-zero boundary conditions, additional loading terms are
    created when we sub in the following into the original governing equation.
    
    .. math:: u\\left({Z,t}\\right)=v\\left({Z,t}\\right) + u_{top}\\left({t}\\right)\\left({1-Z}\\right)
    
    Two additional loading terms are created with each substitution, one 
    for the top boundary condition and one for the bottom boundary condition.
    
    This function calculates :math:`\\mathbf{E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}`
    when substitutions are made in 
    terms of the following form:
    
    
    .. math:: a\\left({z}\\right)\\frac{\\partial u}{\\partial t}
    
    It is assumed that :math:`u_{top}\\left({t}\\right)` and 
    :math:`u_{bot}\\left({t}\\right)` are piecewise linear
    in time, and that multiple functions are superposed.  Also :math:`a\\left(z\\right)` 
    is a piecewise linear function w.r.t. :math:`z`
        
    
    """
    
    E_Igamv_the = np.zeros((len(m), len(tvals)))
    
    
    
    
    if not a is None:
        if drn==1:
            zdist = PolyLine(a.x1,a.x2, np.ones_like(a.x1), np.ones_like(a.x2))
            #bot_vs_time=None
        else:
            zdist = PolyLine(a.x1,a.x2, 1-a.x1, 1-a.x2)
                            
        if not top_vs_time is None:       
            theta = integ.pdim1sin_ab_linear(m, a, zdist)            
            for top_vs_t in top_vs_time:                                        
                E = integ.pEDload_linear(top_vs_t, eigs, tvals, dT)                        
                E_Igamv_the += (E*np.dot(Igamv, theta)).T
                
                    
        if not bot_vs_time is None:       
            theta = integ.pdim1sin_ab_linear(m, a, PolyLine(a.x1,a.x2,a.x1,a.x2))                        
            for bot_vs_t in bot_vs_time:                                                        
                E = integ.pEDload_linear(bot_vs_t, eigs, tvals, dT)                        
                E_Igamv_the += (E*np.dot(Igamv, theta)).T
            
    #theta is 1d array, Igamv is nieg by neig array, np.dot(Igamv, theta) 
    #and np.dot(theta, Igamv) will give differetn 1d arrays.  
    #Basically np.dot(Igamv, theta) gives us what we want i.e. 
    #theta was treated as a column array.  The alternative 
    #np.dot(theta, Igamv) would have treated theta as a row vector.            
    return E_Igamv_the
  
def dim1sin_E_Igamv_the_BC_abf_linear(drn, m, eigs, a, b, top_vs_time, bot_vs_time, tvals, Igamv, dT=1.0):
    """Loading dependant E_Igamv_the matrix that arise from homogenising a(z)*b(z)u(z, t) for non_zero top and bottom boundary conditions
    
    When accounting for non-zero boundary conditions we homogenise the 
    governing equation by letting u(Z,t) = v(Z,t) + utop(t)*(1-Z) + ubot(t)*Z 
    and solving for v(Z, t).  This function calculates the 
    E*inverse(gam*v)*theta part of solution v(Z,t)=phi*v*E*inverse(gam*v)*theta. 
    For the terms that arise by subbing the BC's into terms like a(z)*b(z)*u(Z,t)
    
    The contribution of each `mag_vs_time` are superposed. 
    The result is an array
    of size (neig, len(tvals)). So the columns are the column array 
    E*inverse(gam*v)*theta calculated at each output time.  This will allow
    us later to do v(Z,t) = phi*v*E_Igamv_the

    Uses sin(m*z) in the calculation of theta.
    
    Parameters
    ----------
    drn : [0,1]
        drainage condition,
        0 = Pervious top pervious bottom (PTPB)
        1 = Pervious top impoervious bottom (PTIB)
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geoteca.speccon.m_from_sin_mx
    eigs : 1d numpy.ndarray
        list of eigenvalues
    a : PolyLine
        Piewcewise linear function.  e.g. for 1d consolidation surcharge
        radial draiange term is dTh*kh*et*U(Z,t) `a` would be kh.
    b : PolyLine
        Piewcewise linear function.  e.g. for 1d consolidation surcharge
        radial draiange term is dTh* kh*et*U(Z,t) so `b` would be et
    top_vs_time : list of PolyLine
        Piecewise linear magnitude  vs time for the top boundary.
    bot_vs_time : list of PolyLine
        Piecewise linear magnitude vs time for the bottom boundary.
    tvals : 1d numpy.ndarray`
        list of time values to calculate integral at
    dT : ``float``, optional
        time factor multiple (default = 1.0)        
    
    Returns
    -------
    E_Igamv_the: ndarray
        loading matrix        
    
    Notes
    -----        
    Assuming the loads are formulated as the product of separate time and depth 
    dependant functions: 
    
    .. math:: \\sigma\\left({Z,t}\\right)=\\sigma\\left({Z}\\right)\\sigma\\left({t}\\right)
    
    the solution to the consolidation equation using the spectral method has 
    the form:
    
    .. math:: u\\left(Z,t\\right)=\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta} 
    
    
    when we consider non-zero boundary conditions, additional loading terms are
    created when we sub in the following into the original governing equation.
    
    .. math:: u\\left({Z,t}\\right)=v\\left({Z,t}\\right) + u_{top}\\left({t}\\right)\\left({1-Z}\\right)
    
    Two additional loading terms are created with each substitution, one 
    for the top boundary condition and one for the bottom boundary condition.
    
    This function calculates :math:`\\mathbf{E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}`
    when substitutions are made in 
    terms of the following form:    
    
    .. math:: a\\left({z}\\right)b\\left({z}\\right)u\\left({Z,t}\\right)
    
    It is assumed that :math:`u_{top}\\left({t}\\right)` and 
    :math:`u_{bot}\\left({t}\\right)` are piecewise linear
    in time, and that multiple functions are superposed.  Also :math:`a\\left(z\\right)` 
    and :math:`b\\left(z\\right)` are piecewise linear functions w.r.t. :math:`z`.
          
    
    """
    
    E_Igamv_the = np.zeros((len(m), len(tvals)))
        
    if sum([v is None for v in [a, b]]) == 0:
        a, b = pwise.polyline_make_x_common(a, b)                
        if drn==1:
            zdist = PolyLine(a.x1,a.x2, np.ones_like(a.x1), np.ones_like(a.x2))
            #bot_vs_time=None
        else:
            zdist = PolyLine(a.x1,a.x2, 1-a.x1, 1-a.x2)
                            
        if not top_vs_time is None:       
            theta = integ.pdim1sin_abc_linear(m, a,b, zdist)            
            for top_vs_t in top_vs_time:                                        
                E = integ.pEload_linear(top_vs_t, eigs, tvals, dT)                        
                E_Igamv_the += (E*np.dot(Igamv, theta)).T
                                    
        if not bot_vs_time is None:       
            theta = integ.pdim1sin_abc_linear(m, a, b, PolyLine(a.x1,a.x2,a.x1,a.x2))            
            for bot_vs_t in bot_vs_time:                                        
                E = integ.pEload_linear(bot_vs_t, eigs, tvals, dT)                        
                E_Igamv_the += (E*np.dot(Igamv, theta)).T
            
    #theta is 1d array, Igamv is nieg by neig array, np.dot(Igamv, theta) 
    #and np.dot(theta, Igamv) will give differetn 1d arrays.  
    #Basically np.dot(Igamv, theta) gives us what we want i.e. 
    #theta was treated as a column array.  The alternative 
    #np.dot(theta, Igamv) would have treated theta as a row vector.            
    return E_Igamv_the
    
def dim1sin_E_Igamv_the_BC_D_aDf_linear(drn, m, eigs, a, top_vs_time, bot_vs_time, tvals, Igamv, dT=1.0):
    """Loading dependant E_Igamv_the matrix that arise from homogenising D[a(z)*D[u(z, t),z],z] for non_zero top and bottom boundary conditions
    
    When accounting for non-zero boundary conditions we homogenise the 
    governing equation by letting u(Z,t) = v(Z,t) + utop(t)*(1-Z) + ubot(t)*Z 
    and solving for v(Z, t).  This function calculates the 
    E*inverse(gam*v)*theta part of solution v(Z,t)=phi*v*E*inverse(gam*v)*theta. 
    For the terms that arise by subbing the BC's into terms like a(z)*b(z)*u(Z,t)
    
    The contribution of each `mag_vs_time` are superposed. 
    The result is an array
    of size (neig, len(tvals)). So the columns are the column array 
    E*inverse(gam*v)*theta calculated at each output time.  This will allow
    us later to do v(Z,t) = phi*v*E_Igamv_the

    Uses sin(m*z) in the calculation of theta.
    
    Parameters
    ----------
    drn : [0,1]
        drainage condition,
        0 = Pervious top pervious bottom (PTPB)
        1 = Pervious top impoervious bottom (PTIB)
    m : ``list`` of ``float``
        eigenvlaues of BVP. generate with geotecca.speccon.m_from_sin_mx
    eigs : 1d numpy.ndarray
        list of eigenvalues
    a : PolyLine
        Piewcewise linear function.  e.g. for 1d consolidation surcharge
        radial draiange term is D[kv(z)*D[u(Z,t), Z],Z] so `a` would be kv.    
    top_vs_time : list of PolyLine
        Piecewise linear magnitude  vs time for the top boundary.
    bot_vs_time : list of PolyLine
        Piecewise linear magnitude vs time for the bottom boundary.
    tvals : 1d numpy.ndarray`
        list of time values to calculate integral at
    dT : ``float``, optional
        time factor multiple (default = 1.0)        
    
    Returns
    -------
    E_Igamv_the: ndarray
        loading matrix        
    
    Notes
    -----        
    Assuming the loads are formulated as the product of separate time and depth 
    dependant functions: 
    
    .. math:: \\sigma\\left({Z,t}\\right)=\\sigma\\left({Z}\\right)\\sigma\\left({t}\\right)
    
    the solution to the consolidation equation using the spectral method has 
    the form:
    
    .. math:: u\\left(Z,t\\right)=\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta} 
    
    
    when we consider non-zero boundary conditions, additional loading terms are
    created when we sub in the following into the original governing equation.
    
    .. math:: u\\left({Z,t}\\right)=v\\left({Z,t}\\right) + u_{top}\\left({t}\\right)\\left({1-Z}\\right)
    
    Two additional loading terms are created with each substitution, one 
    for the top boundary condition and one for the bottom boundary condition.
    
    This function calculates :math:`\\mathbf{E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}`
    when substitutions are made in 
    terms of the following form:    
    
    .. math:: \\frac{\\partial}{\\partial Z}\\left({a\\left({Z}\\right)\\frac{\\partial u\\left({Z,t}\\right)}{\\partial Z}}\\right)
    
    It is assumed that :math:`u_{top}\\left({t}\\right)` and 
    :math:`u_{bot}\\left({t}\\right)` are piecewise linear
    in time, and that multiple functions are superposed.  Also :math:`a\\left(z\\right)` 
    is a piecewise linear functions w.r.t. :math:`z`.
          
    
    """
    
    E_Igamv_the = np.zeros((len(m), len(tvals)))
        
    if not a is None:                 
        if drn==1:
            zdist = PolyLine(a.x1,a.x2, np.ones_like(a.x1), np.ones_like(a.x2))
            #bot_vs_time=None
        else:
            zdist = PolyLine(a.x1,a.x2, 1-a.x1, 1-a.x2)
                            
        if not top_vs_time is None:       
            theta = integ.pdim1sin_D_aDb_linear(m, a, zdist)            
            for top_vs_t in top_vs_time:                                                        
                E = integ.pEload_linear(top_vs_t, eigs, tvals, dT)                        
                E_Igamv_the += (E*np.dot(Igamv, theta)).T
                                    
        if not bot_vs_time is None:       
            theta = integ.pdim1sin_D_aDb_linear(m, a, PolyLine(a.x1,a.x2,a.x1,a.x2))            
            for bot_vs_t in bot_vs_time:                                        
                E = integ.pEload_linear(bot_vs_t, eigs, tvals, dT)                        
                E_Igamv_the += (E*np.dot(Igamv, theta)).T
            
    #theta is 1d array, Igamv is nieg by neig array, np.dot(Igamv, theta) 
    #and np.dot(theta, Igamv) will give differetn 1d arrays.  
    #Basically np.dot(Igamv, theta) gives us what we want i.e. 
    #theta was treated as a column array.  The alternative 
    #np.dot(theta, Igamv) would have treated theta as a row vector.            
    return E_Igamv_the


         
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
    drn = 1 
    dT = 1 
    #dTh = 0#100
    dTv = 0.1
    neig = 20


    mvref = 1.0
    kvref = 1.0
    khref = 1.0
    etref = 1.0                
    
    mv = PolyLine([0,1], [1,1])
    #kh = PolyLine([0,1], [1,1])
    kv = PolyLine([0,1], [1,1])
    #et = PolyLine([0,0.48,0.48, 0.52, 0.52,1], [0, 0,1,1,0,0])
    surcharge_vs_depth = PolyLine([0,1], [1,1]) 
    surcharge_vs_time = PolyLine([0,0.1,3], [0,1,1])    
    #vacuum_vs_depth = PolyLine([0,1], [1,1])    
    #vacuum_vs_time = PolyLine([0,0.4,3], [0,-0.2,-0.2])
    #top_vs_time = PolyLine([0,0.0,3], [0,0.5,0.5])
    #bot_vs_time = PolyLine([0,0.0,3], [0,-0.2,-0.2])
    bot_vs_time = PolyLine([0,0.0,3], [0,-2,-2])
    
    ppress_z = np.linspace(0,1,20)    
    avg_ppress_z_pairs = [[0,1],[0, 0.2]]
    settlement_z_pairs = [[0,1],[0, 0.5]]
    #tvals = np.linspace(0,3,10)
    tvals = [0,0.05,0.1]+list(np.linspace(0.2,3,5))
    """)    
#    writer = StringIO()
#    program(my_code, writer)
#    print('writer\n%s' % writer.getvalue())
    #a = calculate_normalised(my_code)
    a = speccon1d(my_code)
    a.make_all()
#    a._make_gam()    
#    #print(a.gam)
#    a._make_psi()
#    #print(a.psi)
#    a._make_eigs_and_v()
#    #print(a.eigs)        
#    #print(a.v)
#    a._make_E_Igamv_the()
#    #print(a.E_Igamv_the)
#    a.make_por()
#    #print(a.por)
#    print(len(a.tvals))
    #print(a.por.shape)
    
    if True:
        plt.plot(a.por, a.ppress_z)
        plt.xlabel('Pore pressure')
        plt.ylabel('Normalised depth, Z')
        plt.gca().invert_yaxis()
        plt.show()
        
        plt.plot(a.tvals, a.avp.T)
        plt.xlabel('Time')
        plt.ylabel('Average pore pressure')
        #plt.gca().invert_yaxis()
        plt.show()
        print(a.set)
        plt.plot(a.tvals, a.set.T)
        plt.xlabel('Time')
        plt.ylabel('settlement')
        plt.gca().invert_yaxis()
        plt.show()
#    for i, p in enumerate(a.por.T):
#    
#        print(a.tvals[i])        
#        plt.plot(p, a.ppress_z)
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


    