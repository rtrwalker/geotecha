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
    """1d consolidation with vertical and radial drainage, surcharge and vacuum, varying boundary conditions"""
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
                    print(a)
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
            self.psi-= integ.dim1sin_D_aDf_linear(self.m, self.kz.y1, self.kz.y2, self.kz.x1, self.kz.x2)
        #kh & et part
        if sum([v is None for v in [self.kh, self.et, self.dTh]])==0:
            kh, et = pwise.polyline_make_x_common(self.kh, self.et)
            self.psi += integ.dim1sin_abf_linear(self.m,kh.y1, kh.y2, et.y1, et.y2, kh.x1, kh.x2)
        self.psi[np.abs(self.psi)<1e-8]=0.0
        return        
        

    def make_eigs_and_v(self):
        """make Igam_psi, v and eigs, and Igamv
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
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


    def make_emat_surcharge(self):
        """make the surcharge loading matrices"""
        
        self.Esurcharge = np.zeros((neigs,len(self.outt)))
        if not self.surz is None:
            for mag_vs_depth, mag_vs_time in zip(self.surz, self.surm):
                mv, mag_vs_depth = pwise.polyline_make_x_common(self.mv, self.surz)           
                
                theta = integ.dim1sin_ab_linear(self.m, mv.y1, mv.y2, mag_vs_depth.y1, mag_vs_depth.y2, mv.x1, mv.x2)
                Esur = integ.EDload_linear(mag_vs_time.x, mag_vs_time.y, self.eigs,self.outt, self.dT)
    
    
                #theta is 1d array, Igamv is nieg by neig array, np.dot(Igamv, theta) 
                #and np.dot(theta, Igamv) will give differetn 1d arrays.  
                #Basically np.dot(Igamv, theta) gives us what we want i.e. 
                #theta was treated as a column array.  The alternative 
                #np.dot(theta, Igamv) would have treated theta as a row vector.
                self.Esurcharge += (Esur*np.dot(self.Igamv, theta)).T
        return

    def make_emat_vacuum(self):
        """make the vacuum loading matrices"""
        
        self.Evacuum = np.zeros((neigs, len(self.outt)))
        
        if (not self.vacz is None) and (self.dTh != 0.0):
            for mag_vs_depth, mag_vs_time in zip(self.vacz, self.vacm):
                et, kh, mag_vs_depth = pwise.polyline_make_x_common(self.et, self.kh, self.vacz)           
                
                theta = integ.dim1sin_abc_linear(self.m, et.y1, et.y2, kh.y1, kh.y2, mag_vs_depth.y1, mag_vs_depth.y2, et.x1, et.x2)
                Evac = integ.Eload_linear(mag_vs_time.x, mag_vs_time.y, self.eigs, self.outt, self.dT)        
                
                self.Evacuum += (Evac * np.dot(self.Igamv, theta)).T
        self.make_gam()
        return



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
    dTv = 1
    neig = 3
        
    mv = PolyLine([0,1], [1,1])
    #kh = PolyLine([0,1], [1,1])
    kz = PolyLine([0,1], [1,1])
    #et = PolyLine([0,1], [1,1])
    surz = PolyLine([0,1], [1,1]) 
    #vacz = PolyLine([0,1], [1,1])
    surm = PolyLine([0,1,10], [0,1,1])
    #vacm = PolyLine([0,0.4,10], [0,-0.2,-0.2])
    topm = None
    botm = None
    
    porz = np.linspace(0,1,20)
    #port = np.linspace(0,10,20)
    avpz = [[0,1],[0, 0.5]]
    #avpt = port
    setz = [[0,1],[0, 0.5]]
    #sett = port
    outt = np.linspace(0,10,20)
    """)    
#    writer = StringIO()
#    program(my_code, writer)
#    print('writer\n%s' % writer.getvalue())
    #a = calculate_normalised(my_code)
    a = speccon1d(my_code)
    a.make_gam()    
    print(a.gam)
    a.make_psi()
    print(a.psi)
    a.make_eigs_and_v()
    print(a.eigs)        
    print(a.v)
    
    
    
    
    
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


    