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
"""Some test routines for the speccon module

Shows a few approaches to testing (so there may be some repeated test)

"""
from __future__ import division, print_function

from nose import with_setup
from nose.tools.trivial import assert_almost_equal
from nose.tools.trivial import assert_raises
from nose.tools.trivial import ok_

from math import pi
import numpy as np
from geotecha.piecewise.piecewise_linear_1d import PolyLine

from geotecha.speccon.integrals import m_from_sin_mx
from geotecha.speccon.integrals import dim1sin
from geotecha.speccon.integrals import dim1sin_avg_between

from geotecha.speccon.integrals import dim1sin_a_linear_between
from geotecha.speccon.integrals import dim1sin_af_linear
from geotecha.speccon.integrals import dim1sin_ab_linear
from geotecha.speccon.integrals import dim1sin_abc_linear
from geotecha.speccon.integrals import dim1sin_abf_linear
from geotecha.speccon.integrals import dim1sin_D_aDb_linear
from geotecha.speccon.integrals import dim1sin_D_aDf_linear
from geotecha.speccon.integrals import dim1sin_D_aDf_linear_v2
from geotecha.speccon.integrals import Eload_linear
from geotecha.speccon.integrals import EDload_linear
from geotecha.speccon.integrals import pdim1sin_a_linear_between
from geotecha.speccon.integrals import pdim1sin_af_linear
from geotecha.speccon.integrals import pdim1sin_ab_linear
from geotecha.speccon.integrals import pdim1sin_abc_linear
from geotecha.speccon.integrals import pdim1sin_abf_linear
from geotecha.speccon.integrals import pdim1sin_D_aDb_linear
from geotecha.speccon.integrals import pdim1sin_D_aDf_linear
from geotecha.speccon.integrals import pdim1sin_D_aDf_linear_v2
from geotecha.speccon.integrals import pEload_linear
from geotecha.speccon.integrals import pEDload_linear



#from spec1d.spec1d import m_func, make_gamma, make_psi, make_theta_two_prop

### start Method 1: global vars 
###     (not recommended as it uses global variables)
PTPB = None
PTIB = None

def setup():
    """setup fn for m_from_sin_mx tests using global variables"""
    global PTPB, PTIB
    PTPB = [3.14159, 6.28319, 9.42478, 12.56637, 15.70796, 18.84956, 21.99115]
    PTIB = [1.57080, 4.71239, 7.85398, 10.99557, 14.13717, 17.27876, 20.42035]

def teardown():
    """teardown fn for m_from_sin_mx tests using global variables"""
    global PTPB, PTIB
    PTPB = None
    PTIB = None

@with_setup(setup, teardown)    
def test_m_from_sin_mx_bc_0v1():
    """m_from_sin_mx tests using global vars, i = 0, boundary = 0"""       
        
    m0 = m_from_sin_mx(0, 0)    
    assert_almost_equal(m0, PTPB[0], 5)
    
@with_setup(setup)
def test_m_from_sin_mx_bc_1v1():
    """m_from_sin_mx tests using global vars, i = 0, boundary = 1"""        
    m0 = m_from_sin_mx(0, 1)    
    assert_almost_equal(m0, PTIB[0], 5)
### end Method 1

### start Method 2: self contained
###     (ok for very simply tests but will lead to repeated data)

def test_m_from_sin_mx_bc_0v2():
    """m_from_sin_mx tests, self contained, i = 0, boundary = 0"""        
    m0 = m_from_sin_mx(0, 0)    
    assert_almost_equal(m0, 3.14159, 5)
    
def test_m_from_sin_mx_bc_1v2():        
    """m_from_sin_mx tests, self contained, i = 0, boundary = 1"""
    m0 = m_from_sin_mx(0, 1)    
    assert_almost_equal(m0, 1.57080, 5)
### end Method 2


### start Method 3: classes
###     (better than 1 and 2 when fixtures are needed)
class test_m_from_sin_mx(object):
    """A suite of tests for m_from_sin_mx
    Shows two approaches: individual methods and looping through a list
        
    """
    
    def __init__(self):        
        self.PTPB = [3.14159, 6.28319, 9.42478, 12.56637, 
                     15.70796, 18.84956, 21.99115]
        self.PTIB = [1.57080, 4.71239, 7.85398, 10.99557, 
                     14.13717, 17.27876, 20.42035]
        
                     
        self.cases = [ #used for generator example
            [(0, 0), 3.14159],
            [(0, 1), 1.57080],
            [(1, 0), 6.28319],
            [(1, 1), 4.71239],
            [(np.array(range(7)), 0), self.PTPB],
            [(np.array(range(7)), 1), self.PTIB],
            ] #then you canjust add more cases
                
    def test_bc0(self):
        """test i = 0, boundary = 0"""
        m0 = m_from_sin_mx(0,0)    
        assert_almost_equal(m0, self.PTPB[0], 5)
        
    def test_bc1(self): 
        """test i = 0, boundary = 1"""
        m0 = m_from_sin_mx(0,1)    
        assert_almost_equal(m0, self.PTIB[0], 5)

    def test_numpy(self):
        """test a numpy array as input to i; i = range(7), boundary = 0"""
        x = np.array(range(7))
        y0 = m_from_sin_mx(x,0)
        assert np.allclose(y0,self.PTPB)
    
    ### a generator example
    def test_cases(self):
        """loop through and test m_from_sin_mx cases with numpy.allclose"""
        for fixture,result in self.cases:
            m = m_from_sin_mx(*fixture)            
            check = np.allclose(m, result)
            msg = """\
failed m_from_sin_mx.test_cases, case:
%s
m:
%s
expected:
%s""" % (fixture, m, result)
            yield ok_, check, msg
    
            
### end Method 3        

def test_m_from_sin_mx_bad_boundary():
    """m_from_sin_mx fail test, self contained, i = 2, boundary = 1.5"""
    assert_raises(ValueError, m_from_sin_mx , 2, 1.5)
    

class base_t_ester(object):
    """A template for testing make_functions"""
    #note the class name must be such that the nose test regex expression will
    #not find the base class.  Otherwise the base test will be run as a test.
    
    def __init__(self, fn, prefix = "base_tester"):
        self.fn = fn
        self.PTPB = np.array([ pi,  2.0 * pi])        
        self.PTIB = np.array([ pi / 2.0,  3.0 * pi / 2.0])        
        self.cases = []#each element should be [desc, fixture, result]
        self.prefix = prefix
    def test_cases(self):   
        """loop through and test make_fn cases with np.allclose"""
        for desc, fixture, result in self.cases:            
            res = self.fn(**fixture)
            check = np.allclose(res, result)
            msg = """\
failed test_%s, case: %s
%s
calculated:
%s
expected:
%s""" % (self.prefix + " " + self.fn.__name__, desc, fixture, res, result)
            yield ok_, check, msg 
            
            
class test_dim1sin_af_linear(base_t_ester):
    """loop through and test the dim1sin_af_linear cases with np.allclose"""
    def __init__(self):
        base_t_ester.__init__(self, dim1sin_af_linear, prefix = self.__class__.__name__)        
        self.gamma_isotropic = np.array([[0.5, 0], [0, 0.5]])
        
        self.cases = [            
            
            ['a const, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1]}, 
             self.gamma_isotropic],
            ['a const, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1]},
             self.gamma_isotropic],
            
            ['a const *2, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [2], 'ab': [2]}, 
             self.gamma_isotropic * 2],
            ['a const *2, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [2], 'ab': [2]},
             self.gamma_isotropic * 2],
             
            ['a const, 2 layers, PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 1], 'ab': [1, 1]}, 
             self.gamma_isotropic],
            ['a const, 2 layers, PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 1], 'ab': [1, 1]},
             self.gamma_isotropic], 
            
            ['a const*2, 2 layers, PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [2, 2], 'ab': [2, 2]}, 
             self.gamma_isotropic * 2],
            ['a const*2, 2 layers, PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [2, 2], 'ab': [2, 2]},
             self.gamma_isotropic * 2],
            
            ['a 2 layers, const within each layer, PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 2], 'ab': [1, 2]}, 
             np.array([[0.95136535, -0.10459088], [-0.10459088, 0.76881702]])],
            ['a 2 layers, const within each layer, PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 2], 'ab': [1, 2]},
             np.array([[0.84677446, -0.18254832], [-0.18254832, 0.76215866]])],
            
            
            #from speccon vba : debug.Print ("[[" & gammat(1,1) & ", " & gammat(1,2) & "],[" & gammat(2,1) & ", " & gammat(2,2) & "]]")
            ['a linear within one layer, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [2]}, 
             np.array([[0.851321183642337, -0.101321183642336],[-0.101321183642336, 0.761257909293592]])],
            ['a linear within one layer, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [2]},
             np.array([[0.750000000000001, -9.00632743487448E-02],[-9.00632743487448E-02, 0.750000000000001]])]
            ]
            
class test_pdim1sin_af_linear(base_t_ester):
    """loop through and test the pdim1sin_af_linear cases with np.allclose"""
    def __init__(self):
        base_t_ester.__init__(self, pdim1sin_af_linear, prefix = self.__class__.__name__)        
        self.gamma_isotropic = np.array([[0.5, 0], [0, 0.5]])
        
        self.cases = [                                    
            #from speccon vba : debug.Print ("[[" & gammat(1,1) & ", " & gammat(1,2) & "],[" & gammat(2,1) & ", " & gammat(2,2) & "]]")
            ['a linear within one layer, PTIB', 
             {'m': self.PTIB, 'a': PolyLine([0],[1],[1],[2])}, 
             np.array([[0.851321183642337, -0.101321183642336],[-0.101321183642336, 0.761257909293592]])]
            
            ]            



class test_dim1sin_abf_linear(base_t_ester):
    """loop through and test the dim1sin_abf_linear cases with np.allclose"""
    def __init__(self):
        base_t_ester.__init__(self, dim1sin_abf_linear, prefix = self.__class__.__name__)                
        self.iso = np.array([[0.5, 0], [0, 0.5]])        
        self.cases = [     
             
            #a
            ['a const, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1],
              'at': [1], 'ab':[1], 'bt': [1], 'bb': [1]},  
             self.iso], 
            ['a const, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1],
              'at': [1], 'ab':[1], 'bt': [1], 'bb': [1]},  
             self.iso], 
            
            ['a const*2, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1],
              'at': [2], 'ab':[2], 'bt': [1], 'bb': [1]},  
             self.iso*2], 
            ['a const*2, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1],
              'at': [2], 'ab':[2], 'bt': [1], 'bb': [1]},  
             self.iso*2],
             
            ['a const, 2 layers PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 
              'at': [1, 1], 'ab':[1, 1], 'bt': [1, 1], 'bb': [1, 1]},  
             self.iso], 
            ['a const, 2 layers, PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 
              'at': [1, 1], 'ab':[1, 1], 'bt': [1, 1], 'bb': [1, 1]},  
             self.iso],
             
            ['a 2 layers, const within each layer, PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 
              'at': [1, 2], 'ab':[1, 2], 'bt': [1, 1], 'bb': [1, 1]},  
             np.array([[ 0.951365345728, -0.104590881539], [-0.104590881539,  0.768817023874]])], 
            ['a 2 layers, const within each layer, PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 
              'at': [1, 2], 'ab':[1, 2], 'bt': [1, 1], 'bb': [1, 1]},  
             np.array([[ 0.846774464189, -0.182548321854], [-0.182548321854, 0.762158663568]])],             
            
            ['a linear within 1 layer, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1],
              'at': [1], 'ab':[2], 'bt': [1], 'bb': [1]},  
             np.array([[0.851321183642337, -0.101321183642336],[-0.101321183642336, 0.761257909293592]])], 
            ['a linear within 1 layer, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1],
              'at': [1], 'ab':[2], 'bt': [1], 'bb': [1]},  
             np.array([[0.750000000000001, -9.00632743487448E-02],[-9.00632743487448E-02, 0.750000000000001]])], 
            
            #b, originally Walker Phd a and b were lumped together.  
            #Therefore once separated varying one parameter while keeping 
            #the other constant, then changing which parameter alters should
            #yiueld the same result. Hence reusing the a results.
            ['b const, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1],
              'at': [1], 'ab':[1], 'bt': [1], 'bb': [1]},  
             self.iso], 
            ['b const, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1],
              'at': [1], 'ab':[1], 'bt': [1], 'bb': [1]},  
             self.iso], 
            
            ['b const*2, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1],
              'at': [2], 'ab':[2], 'bt': [1], 'bb': [1]},  
             self.iso*2], 
            ['b const*2, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1],
              'at': [1], 'ab':[1], 'bt': [2], 'bb': [2]},  
             self.iso*2],
            
            ['b const, 2 layers PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 
              'at': [1, 1], 'ab':[1, 1], 'bt': [1, 1], 'bb': [1, 1]},  
             self.iso], 
            ['b const, 2 layers, PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 
              'at': [1, 1], 'ab':[1, 1], 'bt': [1, 1], 'bb': [1, 1]},  
             self.iso],
             
            ['b 2 layers, const within each layer, PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 
              'at': [1, 1], 'ab':[1, 1], 'bt': [1, 2], 'bb': [1, 2]},  
             np.array([[ 0.951365345728, -0.104590881539], [-0.104590881539,  0.768817023874]])], 
            ['b 2 layers, const within each layer, PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 
              'at': [1, 1], 'ab':[1, 1], 'bt': [1, 2], 'bb': [1, 2]},  
             np.array([[ 0.846774464189, -0.182548321854], [-0.182548321854, 0.762158663568]])], 
             
            ['b linear within 1 layer, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1],
              'at': [1], 'ab':[1], 'bt': [1], 'bb': [2]},  
             np.array([[0.851321183642337, -0.101321183642336],[-0.101321183642336, 0.761257909293592]])], 
            ['b linear within 1 layer, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1],
              'at': [1], 'ab':[1], 'bt': [1], 'bb': [2]},  
             np.array([[0.750000000000001, -9.00632743487448E-02],[-9.00632743487448E-02, 0.750000000000001]])], 
            
            
            
            #mixed
            ['a const, b const cancel out, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1],
              'at': [0.5], 'ab':[0.5], 'bt': [2], 'bb': [2]},  
             self.iso], 
            ['a const, b const cancel out, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1],
              'at': [0.5], 'ab':[0.5], 'bt': [2], 'bb': [2]},  
             self.iso],
            
            ['a linear in one layer, b linear in one layer, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1],
              'at': [1], 'ab':[2], 'bt': [1], 'bb': [2]},  
             np.array([[1.4706302, -0.32929385], [-0.32929385, 1.2004404]])], 
            ['a linear in one layer, b linear in one layer, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1],
              'at': [1], 'ab':[2], 'bt': [1], 'bb': [2]},  
             np.array([[1.1413364, -0.27018982], [-0.27018982, 1.1603341]])],
             
            ]

class test_pdim1sin_abf_linear(base_t_ester):
    """loop through and test the pdim1sin_abf_linear cases with np.allclose"""
    def __init__(self):
        base_t_ester.__init__(self, pdim1sin_abf_linear, prefix = self.__class__.__name__)                
        self.iso = np.array([[0.5, 0], [0, 0.5]])        
        self.cases = [     
             
            
            
            ['a linear in one layer, b linear in one layer, PTIB', 
             {'m': self.PTIB, 'a': PolyLine([0],[1],[1],[2]),'b': PolyLine([0],[1], [1],[2])},  
             np.array([[1.4706302, -0.32929385], [-0.32929385, 1.2004404]])],
            
            ['a linear in one layer, b linear in one layer, PTIB', 
             {'m': self.PTIB, 'a': PolyLine([0,0.5],[0.5,1],[1, 1.5],[1.5, 2]),'b': PolyLine([0],[1], [1],[2])},  
             np.array([[1.4706302, -0.32929385], [-0.32929385, 1.2004404]])]
            
             
            ]

class test_dim1sin_D_aDf_linear(base_t_ester):
    """loop through and test the dim1sin_D_aDf_linear cases with np.allclose"""
    def __init__(self):
        base_t_ester.__init__(self, dim1sin_D_aDf_linear, prefix = self.__class__.__name__)                
        self.iso_PTIB = (-0.5) * np.array([[(np.pi/2.0)**2.0, 0], [0, (3.0*np.pi/2.0)**2.0]])
        self.iso_PTPB = (-0.5) * np.array([[(np.pi)**2.0, 0], [0, (2.0*np.pi)**2.0]])        
        
        self.cases = [     
            
            ['a const, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1]},
             self.iso_PTIB], 
            ['a const, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1]},
             self.iso_PTPB],
            
            ['a const*2, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [2], 'ab': [2]}, 
             self.iso_PTIB*2], 
            ['a const*2, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [2], 'ab': [2]},
             self.iso_PTPB*2],            
            
            ['a const, 2 layers, PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 1], 'ab': [1, 1]},
             self.iso_PTIB], 
            ['a const, 2 layers, PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 1], 'ab': [1, 1]},
             self.iso_PTPB],
            
            ['a const*2, 2 layers, PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [2, 2], 'ab': [2, 2]},
             self.iso_PTIB * 2], 
            ['a const*2, 2 layers, PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [2, 2], 'ab': [2, 2]},
             self.iso_PTPB * 2], 
            
            ['2 layers, a const within eachlayer, PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 2], 'ab': [1, 2]},
             np.array([[-1.60044185963,  1.466671155],[1.466671155, -18.4577561084]])], 
            ['2 layers, a const within eachlayer, PTPB',
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 2], 'ab': [1, 2]},
             np.array([[-7.43403806325, 2.37230488791], [2.37230488791, -33.0766501659]])], 
             
            # from speccon debug.Print ("[[" & psimat(1,1) & ", " & psimat(1,2) & "],[" & psimat(2,1) & ", " & psimat(2,2) & "]]") 
            ['a linear within one layer, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [2]},
             np.array([[-1.60055082520425, 0.749999999999957],[0.749999999999957, -16.4049574268382]])], 
            ['a linear within one layer, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [2]},
             np.array([[-7.40220330081701, 2.22222222222222],[2.22222222222222, -29.6088132032681]])],             

            ['2 layers, a linear within eachlayer, PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 1], 'ab': [1.4, 1.6]},
             np.array([[-1.4538543, 0.16333154], [0.16333154, -13.463177]])], 
            ['2 layers, a linear within eachlayer, PTPB',
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 1], 'ab': [1.4, 1.6]},
             np.array([[-6.4025090, 1.2733003], [1.2733003, -24.273837]])],
            ]
            
class test_pdim1sin_D_aDf_linear(base_t_ester):
    """loop through and test the pdim1sin_D_aDf_linear cases with np.allclose"""
    def __init__(self):
        base_t_ester.__init__(self, pdim1sin_D_aDf_linear, prefix = self.__class__.__name__)                
        self.iso_PTIB = (-0.5) * np.array([[(np.pi/2.0)**2.0, 0], [0, (3.0*np.pi/2.0)**2.0]])
        self.iso_PTPB = (-0.5) * np.array([[(np.pi)**2.0, 0], [0, (2.0*np.pi)**2.0]])        
        
        self.cases = [     
                        
            ['2 layers, a linear within eachlayer, PTIB', 
             {'m': self.PTIB, 'a': PolyLine([0, 0.4],[0.4, 1],[1, 1],[1.4, 1.6])},
             np.array([[-1.4538543, 0.16333154], [0.16333154, -13.463177]])]            
            ]
            
class test_dim1sin_D_aDf_linear_v2(base_t_ester):
    """loop through and test the dim1sin_D_aDf_linear_v2 cases with np.allclose"""
    def __init__(self):
        base_t_ester.__init__(self, dim1sin_D_aDf_linear_v2, prefix = self.__class__.__name__)                
        self.iso_PTIB = (-0.5) * np.array([[(np.pi/2.0)**2.0, 0], [0, (3.0*np.pi/2.0)**2.0]])
        self.iso_PTPB = (-0.5) * np.array([[(np.pi)**2.0, 0], [0, (2.0*np.pi)**2.0]])        
        
        self.cases = [     
            
            ['a const, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1]},
             self.iso_PTIB], 
            ['a const, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1]},
             self.iso_PTPB],
            
            ['a const*2, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [2], 'ab': [2]}, 
             self.iso_PTIB*2], 
            ['a const*2, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [2], 'ab': [2]},
             self.iso_PTPB*2],            
            
            ['a const, 2 layers, PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 1], 'ab': [1, 1]},
             self.iso_PTIB], 
            ['a const, 2 layers, PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 1], 'ab': [1, 1]},
             self.iso_PTPB],
            
            ['a const*2, 2 layers, PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [2, 2], 'ab': [2, 2]},
             self.iso_PTIB * 2], 
            ['a const*2, 2 layers, PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [2, 2], 'ab': [2, 2]},
             self.iso_PTPB * 2], 
            
            ['2 layers, a const within eachlayer, PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 2], 'ab': [1, 2]},
             np.array([[-1.60044185963,  1.466671155],[1.466671155, -18.4577561084]])], 
            ['2 layers, a const within eachlayer, PTPB',
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 2], 'ab': [1, 2]},
             np.array([[-7.43403806325, 2.37230488791], [2.37230488791, -33.0766501659]])], 
             
            # from speccon debug.Print ("[[" & psimat(1,1) & ", " & psimat(1,2) & "],[" & psimat(2,1) & ", " & psimat(2,2) & "]]") 
            ['a linear within one layer, PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [2]},
             np.array([[-1.60055082520425, 0.749999999999957],[0.749999999999957, -16.4049574268382]])], 
            ['a linear within one layer, PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [2]},
             np.array([[-7.40220330081701, 2.22222222222222],[2.22222222222222, -29.6088132032681]])],             

            ['2 layers, a linear within eachlayer, PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 1], 'ab': [1.4, 1.6]},
             np.array([[-1.4538543, 0.16333154], [0.16333154, -13.463177]])], 
            ['2 layers, a linear within eachlayer, PTPB',
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 1], 'ab': [1.4, 1.6]},
             np.array([[-6.4025090, 1.2733003], [1.2733003, -24.273837]])],
            ]   

class test_pdim1sin_D_aDf_linear_v2(base_t_ester):
    """loop through and test the pdim1sin_D_aDf_linear_v2 cases with np.allclose"""
    def __init__(self):
        base_t_ester.__init__(self, pdim1sin_D_aDf_linear_v2, prefix = self.__class__.__name__)                
        self.iso_PTIB = (-0.5) * np.array([[(np.pi/2.0)**2.0, 0], [0, (3.0*np.pi/2.0)**2.0]])
        self.iso_PTPB = (-0.5) * np.array([[(np.pi)**2.0, 0], [0, (2.0*np.pi)**2.0]])        
        
        self.cases = [     
                        
            ['2 layers, a linear within eachlayer, PTIB', 
             {'m': self.PTIB, 'a': PolyLine([0, 0.4],[0.4, 1],[1, 1],[1.4, 1.6])},
             np.array([[-1.4538543, 0.16333154], [0.16333154, -13.463177]])]            
            ] 
            
class test_dim1sin_ab_linear(base_t_ester):
    """loop through and test the dim1sin_ab_linear cases with np.allclose"""
    def __init__(self):
        base_t_ester.__init__(self, dim1sin_ab_linear, prefix = self.__class__.__name__)                
        
        self.iso_PTIB = np.array([2/pi, 2/(3*pi)])
        self.iso_PTPB = np.array([2/pi, 0.0])
        
        
        self.cases = [            
            
            ['a const, b const PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1], 'bt': [1], 'bb': [1]}, 
             self.iso_PTIB],
            ['a const, b const PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1], 'bt': [1], 'bb': [1]},
             self.iso_PTPB],
            
            ['a const*2,b const PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [2], 'ab': [2], 'bt': [1], 'bb': [1]}, 
             self.iso_PTIB * 2],
            ['a const*2, b const PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [2], 'ab': [2], 'bt': [1], 'bb': [1]},
             self.iso_PTPB * 2],
            
            ['a const, b const*2 PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1], 'bt': [2], 'bb': [2]}, 
             self.iso_PTIB * 2],
            ['a const, b const*2 PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1], 'bt': [2], 'bb': [2]},
             self.iso_PTPB * 2],
            
            ['a const*0.5, b const*2 PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [0.5], 'ab': [0.5], 'bt': [2], 'bb': [2]}, 
             self.iso_PTIB],
            ['a const*0.5, b const*2 PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [0.5], 'ab': [0.5], 'bt': [2], 'bb': [2]},
             self.iso_PTPB],
            
            ['a const within 2 layers, b const PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 2], 'ab': [1, 2], 'bt': [1, 1], 'bb': [1, 1]}, 
             np.array([1.15166, 0.146631])],
            ['a const within 2 layers, b const PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 2], 'ab': [1, 2], 'bt': [1, 1], 'bb': [1, 1]}, 
             np.array([1.05329, -0.287914])],
             
            ['a const, b const within 2 layers PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 1], 'ab': [1, 1], 'bt': [1, 2], 'bb': [1, 2]}, 
             np.array([1.15166, 0.146631])],
            ['a const, b const within 2 layers PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 1], 'ab': [1, 1], 'bt': [1, 2], 'bb': [1, 2]}, 
             np.array([1.05329, -0.287914])], 
             
            ['a linear within one layer, b const PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [2], 'bt': [1], 'bb': [1]}, 
             np.array([(2*(2+pi))/pi**2, (2*(3*pi-2))/(9*pi**2)])],
            ['a linear within one layer, b const PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [2], 'bt': [1], 'bb': [1]},
             np.array([3/pi, -1/(2*pi)])], 
             
            ['a const, b linear within one layer PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1], 'bt': [1], 'bb': [2]}, 
             np.array([(2*(2+pi))/pi**2, (2*(3*pi-2))/(9*pi**2)])],
            ['a const, b linear within one layer PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1], 'bt': [1], 'bb': [2]},
             np.array([3/pi, -1/(2*pi)])],  
             
            ['a and b linear witin one layer PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [2], 'bt': [1], 'bb': [2]}, 
             np.array([(2*(-8+8*pi+pi**2))/pi**3, (2*(-8-24*pi+9*pi**2))/(27*pi**3)])],
            ['a and b linear witin one layer PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [2], 'bt': [1], 'bb': [2]},
             np.array([(5*pi**2-4)/pi**3, -3/(2*pi)])],                                     
            ]

class test_pdim1sin_ab_linear(base_t_ester):
    """loop through and test the pdim1sin_ab_linear cases with np.allclose"""
    def __init__(self):
        base_t_ester.__init__(self, pdim1sin_ab_linear, prefix = self.__class__.__name__)                
        
        self.iso_PTIB = np.array([2/pi, 2/(3*pi)])
        self.iso_PTPB = np.array([2/pi, 0.0])
        
        
        self.cases = [            
            
            
            ['a and b linear witin one layer PTIB', 
             {'m': self.PTIB,'a': PolyLine([0],[1],[1],[2]),'b': PolyLine([0],[1],[1],[2])}, 
             np.array([(2*(-8+8*pi+pi**2))/pi**3, (2*(-8-24*pi+9*pi**2))/(27*pi**3)])],
            ['a and b linear witin one layer PTIB', 
             {'m': self.PTIB,'a': PolyLine([0,0.5],[0.5,1],[1, 1.5],[1.5,2]),'b': PolyLine([0],[1],[1],[2])}, 
             np.array([(2*(-8+8*pi+pi**2))/pi**3, (2*(-8-24*pi+9*pi**2))/(27*pi**3)])]                         
            ]

class test_dim1sin_abc_linear(base_t_ester):
    """A suite of tests for the dim1sin_abc_linear function"""
    def __init__(self):
        base_t_ester.__init__(self, dim1sin_abc_linear, prefix = self.__class__.__name__)                
        
        self.iso_PTIB = np.array([2/pi, 2/(3*pi)])
        self.iso_PTPB = np.array([2/pi, 0.0])
        
        
        self.cases = [                        
            #a
            ['a const, b const PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1], 'bt': [1], 'bb': [1], 'ct':[1], 'cb':[1]}, 
             self.iso_PTIB],
            ['a const, b const PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1], 'bt': [1], 'bb': [1], 'ct':[1], 'cb':[1]},
             self.iso_PTPB],
            
            ['a const*2,b const PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [2], 'ab': [2], 'bt': [1], 'bb': [1], 'ct':[1], 'cb':[1]}, 
             self.iso_PTIB * 2],
            ['a const*2, b const PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [2], 'ab': [2], 'bt': [1], 'bb': [1], 'ct':[1], 'cb':[1]},
             self.iso_PTPB * 2],
            
            ['a const, b const*2 PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1], 'bt': [2], 'bb': [2], 'ct':[1], 'cb':[1]}, 
             self.iso_PTIB * 2],
            ['a const, b const*2 PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1], 'bt': [2], 'bb': [2], 'ct':[1], 'cb':[1]},
             self.iso_PTPB * 2],
             
            ['a const, b const, c const*2 PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1], 'bt': [1], 'bb': [1], 'ct':[2], 'cb':[2]}, 
             self.iso_PTIB * 2],
            ['a const, b const, c const*2 PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1], 'bt': [1], 'bb': [1], 'ct':[2], 'cb':[2]},
             self.iso_PTPB * 2], 
            
            ['a const*0.4, b const*2, c const* 1.25 PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [0.4], 'ab': [0.4], 'bt': [2], 'bb': [2], 'ct':[1.25], 'cb':[1.25]}, 
             self.iso_PTIB],
            ['a const*0.5, b const*2 PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [0.4], 'ab': [0.4], 'bt': [2], 'bb': [2], 'ct':[1.25], 'cb':[1.25]},
             self.iso_PTPB],
            
            ['a const within 2 layers, b const PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 2], 'ab': [1, 2], 'bt': [1, 1], 'bb': [1, 1], 'ct':[1,1], 'cb':[1,1]}, 
             np.array([1.15166, 0.146631])],
            ['a const within 2 layers, b const PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 2], 'ab': [1, 2], 'bt': [1, 1], 'bb': [1, 1], 'ct':[1,1], 'cb':[1,1]}, 
             np.array([1.05329, -0.287914])],
             
            ['a const, b const within 2 layers PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 1], 'ab': [1, 1], 'bt': [1, 2], 'bb': [1, 2], 'ct':[1,1], 'cb':[1,1]}, 
             np.array([1.15166, 0.146631])],
            ['a const, b const within 2 layers PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 1], 'ab': [1, 1], 'bt': [1, 2], 'bb': [1, 2], 'ct':[1,1], 'cb':[1,1]}, 
             np.array([1.05329, -0.287914])], 
            
            ['a const, b const, c const within 2 layers PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 1], 'ab': [1, 1], 'bt': [1, 1], 'bb': [1, 1], 'ct':[1, 2], 'cb':[1, 2]}, 
             np.array([1.15166, 0.146631])],
            ['a const, b const, c const within 2 layers PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 1], 'ab': [1, 1], 'bt': [1, 1], 'bb': [1, 1], 'ct':[1, 2], 'cb':[1, 2]}, 
             np.array([1.05329, -0.287914])], 

            ['a linear within one layer, b const PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [2], 'bt': [1], 'bb': [1], 'ct':[1], 'cb':[1]}, 
             np.array([(2*(2+pi))/pi**2, (2*(3*pi-2))/(9*pi**2)])],
            ['a linear within one layer, b const PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [2], 'bt': [1], 'bb': [1], 'ct':[1], 'cb':[1]},
             np.array([3/pi, -1/(2*pi)])], 
             
            ['a const, b linear within one layer PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1], 'bt': [1], 'bb': [2], 'ct':[1], 'cb':[1]}, 
             np.array([(2*(2+pi))/pi**2, (2*(3*pi-2))/(9*pi**2)])],
            ['a const, b linear within one layer PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1], 'bt': [1], 'bb': [2], 'ct':[1], 'cb':[1]},
             np.array([3/pi, -1/(2*pi)])],

            ['a const, b const, c linear within one layer PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1], 'bt': [1], 'bb': [1], 'ct':[1], 'cb':[2]}, 
             np.array([1.0419045, 0.16717495])],
            ['a const, b const, c linear within one layer PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1], 'bt': [1], 'bb': [1], 'ct':[1], 'cb':[2]},
             np.array([0.95492966, -0.15915494])],    
             
             
            ['a and b and c linear witin one layer PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [2], 'bt': [1], 'bb': [2], 'ct':[1], 'cb':[2]}, 
             np.array([2.9664286, -0.37334203])],
            ['a and b linear witin one layer PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [2], 'bt': [1], 'bb': [2], 'ct':[1], 'cb':[2]},
             np.array([2.2842614, -1.0898960])],                                     
            ]
                                    
class test_pdim1sin_abc_linear(base_t_ester):
    """A suite of tests for the pdim1sin_abc_linear function"""
    def __init__(self):
        base_t_ester.__init__(self, pdim1sin_abc_linear, prefix = self.__class__.__name__)                
        
        self.iso_PTIB = np.array([2/pi, 2/(3*pi)])
        self.iso_PTPB = np.array([2/pi, 0.0])
        
        
        self.cases = [                        
             
            ['a and b and c linear witin one layer PTIB', 
             {'m': self.PTIB,'a': PolyLine([0],[1],[1],[2]),
             'b': PolyLine([0],[1],[1],[2]),
             'c': PolyLine([0],[1],[1],[2])}, 
             np.array([2.9664286, -0.37334203])],
            ['a and b and c linear witin one layer PTIB', 
             {'m': self.PTIB,'a': PolyLine([0, 0.5],[0.5,1],[1,1.5],[1.5,2]),
             'b': PolyLine([0],[1],[1],[2]),
             'c': PolyLine([0],[1],[1],[2])}, 
             np.array([2.9664286, -0.37334203])],                                     
            ]
                                    
                                    
                                    
class test_dim1sin_D_aDb_linear(base_t_ester):
    """A suite of tests for the make_thesig function"""
    def __init__(self):
        base_t_ester.__init__(self, dim1sin_D_aDb_linear, prefix = self.__class__.__name__)                
        self.zero = np.zeros(2)
        
        
        
        self.cases = [            
            
            ['a const, b const PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1], 'bt': [1], 'bb': [1]}, 
             self.zero],
            ['a const, b const PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1], 'bt': [1], 'bb': [1]},
             self.zero],
            
            ['a const, b linear within one layer PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1], 'bt': [1], 'bb': [2]}, 
             self.zero],
            ['a const, b linear within one layer PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [1], 'bt': [1], 'bb': [2]},
             self.zero],
                        
            ['a linear within one layer, b linear within one layer PTIB', 
             {'m': self.PTIB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [2], 'bt': [1], 'bb': [2]}, 
             np.array([0.63661977, 0.21220659])],
            ['a linear within one layer, b linear within one layer PTPB', 
             {'m': self.PTPB, 'zt': [0], 'zb': [1], 'at': [1], 'ab': [2], 'bt': [1], 'bb': [2]},
             np.array([0.63661977, 0])],
            
            ['a const within two layers, b linear accross both layers PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 2], 'ab': [1, 2], 'bt': [1,1.4], 'bb': [1.4, 2]},
             np.array([0.58778525, 0.95105652])],
            ['a const within two layers, b linear accross both layers PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 2], 'ab': [1, 2], 'bt': [1,1.4], 'bb': [1.4, 2]},
             np.array([0.95105652, 0.58778525])],
            
            ['a linear within two layers, b linear accross both layers PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 1], 'ab': [1.4, 1.6], 'bt': [1,1.4], 'bb': [1.4, 2]},
             np.array([0.40150567, -0.16821602])],
            ['a linear within two layers, b linear accross both layers PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 1], 'ab': [1.4, 1.6], 'bt': [1,1.4], 'bb': [1.4, 2]},
             np.array([0.25619717, -0.23511410])],

            ['a const, b linear  in two layers PTIB', 
             {'m': self.PTIB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 1], 'ab': [1, 1], 'bt': [1, 1], 'bb': [1.4, 2.2]},
             np.array([0.58778525, 0.95105652])],
            ['a const, b linear in two layers PTPB', 
             {'m': self.PTPB, 'zt': [0, 0.4], 'zb': [0.4, 1], 'at': [1, 1], 'ab': [1, 1], 'bt': [1, 1], 'bb': [1.4, 2.2]},
             np.array([0.95105652, 0.58778525])],
                                                
            ]                                    
            
class test_EDload_linear(base_t_ester):
    """A suite of tests for the make_thesig function"""
    def __init__(self):
        base_t_ester.__init__(self, EDload_linear, prefix = self.__class__.__name__)                
        self.eigs = np.array([2.46740110027, 22.2066099025])        
        self.eigs_dT2 = np.array([1.23370055014, 11.1033049512])
        #Eload_linear(loadtim, loadmag, eigs, tvals)
        
        
        self.cases = [            
            
            ['instantaneous load at 0', 
             {'loadtim': np.array([0,0]), 'loadmag': np.array([0, 100]), 'eigs': self.eigs, 'tvals': np.array([-1,0,1])}, 
             np.array([[0, 0], [100, 100], [8.48049725, 2.26877724e-08]])],
            
            ['instantaneous load at 0, dT=2', 
             {'loadtim': np.array([0,0]), 'loadmag': np.array([0, 100]), 'eigs': self.eigs_dT2, 'tvals': np.array([-1,0,1]), 'dT':2.0}, 
             np.array([[0, 0], [100, 100], [8.48049725, 2.26877724e-08]])],
            
            ['two ramp loads', 
             {'loadtim': np.array([0, 0.3, 0.5, 0.8]), 'loadmag': np.array([0, 40, 40, 80]), 'eigs': self.eigs, 'tvals': np.array([-1,0.1, 0.3, 0.4, 0.5, 0.7, 1])}, 
             np.array([[0, 0],
                       [11.8157397503, 5.35253639003],
                       [28.2613797105, 5.9965412459],
                       [22.0818518534, 0.650848653927],
                       [17.2535165046, 0.0706413835822],
                       [31.5811316808, 5.9343186486],
                       [22.2779636541, 0.0706424476152],                       
                       ])],
            
                                                
            ]             
            
class test_Eload_linear(base_t_ester):
    """A suite of tests for the make_thesig function"""
    def __init__(self):
        base_t_ester.__init__(self, Eload_linear, prefix = self.__class__.__name__)                
        self.eigs = np.array([2.46740110027, 22.2066099025])        
        self.eigs_dT2 = np.array([1.23370055014, 11.1033049512])
        #Eload_linear(loadtim, loadmag, eigs, tvals)
        
        
        self.cases = [            
            
            ['instantaneous load at 0', 
             {'loadtim': np.array([0,0,10]), 'loadmag': np.array([0, -100,-100]), 'eigs': self.eigs, 'tvals': np.array([-1,0,1])}, 
             np.array([[0, 0], [0, 0], [-37.0914569616, -4.50316371642]])],
            
            ['instantaneous load at 0, dT=2', 
             {'loadtim': np.array([0,0,10]), 'loadmag': np.array([0, -100,-100]), 'eigs': self.eigs_dT2, 'tvals': np.array([-1,0,1]), 'dT':2.0}, 
             np.array([[0, 0], [0, 0], [-37.0914569616, -4.50316371642]])],
            
            ['two ramp loads', 
             {'loadtim': [0, 0.3, 0.5, 0.8, 10], 'loadmag': [0, -40, -40, -80, -80], 'eigs': self.eigs, 'tvals': [-1,0.1, 0.3, 0.4, 0.5, 0.7, 1]}, 
              [[0, 0],
               [-0.615057512497, -0.359388352313],
               [-4.75748360824, -1.53123141729],
               [-7.26195191556, -1.77195670654],
               [-9.21880252584, -1.79808438982],
               [-14.219631734, -2.7348770607],
               [-23.3938601792, -3.59934982888],                       
               ]],
            
                                                
            ]
                                                
class test_dim1sin(base_t_ester):
    """A suite of tests for the dim1sin function"""
    def __init__(self):
        base_t_ester.__init__(self, dim1sin, prefix = self.__class__.__name__)                
                            
        self.cases = [            
            
            ['3 zs, 2 ms', 
             {'m': self.PTIB, 'z': [0,0.25, 1]}, 
             np.array([[0, 0], [0.382683, 0.9238795], [1, -1]])],
                                                     
            ]
            
class test_dim1sin_avg_between(base_t_ester):
    """A suite of tests for the dim1sin_avg_between function"""
    def __init__(self):
        base_t_ester.__init__(self, dim1sin_avg_between, prefix = self.__class__.__name__)                
                            
        self.cases = [            
            
            ['3 zs, 2 ms', 
             {'m': self.PTIB, 'z': [[0, 1], [0.1, 0.2], [0.5, 1]]}, 
             np.array([[0.636619772, 0.21220659], [0.023320543/0.1, 0.064345552/0.1], [0.450158158/0.5, -0.150052719/0.5]])],   

            ]         

class test_dim1sin_a_linear_between(base_t_ester):
    """A suite of tests for the dim1sin_a_linear_between function"""
    def __init__(self):
        base_t_ester.__init__(self, dim1sin_a_linear_between, prefix = self.__class__.__name__)                
                            
        self.cases = [            
            
            ['1 layer const a, 3 zs, 2 ms', 
             {'m': self.PTIB, 'at':[1], 'ab':[1],'zt':[0], 'zb':[1], 'z': [[0, 1], [0.1, 0.2], [0.5, 1]]}, 
             np.array([[0.636619772, 0.21220659], [0.023320543, 0.064345552], [0.450158158, -0.150052719]])],

            ['2 layer const a same in each layer, 3 zs, 2 ms', 
             {'m': self.PTIB, 'at':[1,1], 'ab':[1, 1],'zt':[0,0.4], 'zb':[0.4, 1], 'z': [[0, 1], [0.1, 0.2], [0.5, 1]]}, 
             np.array([[0.636619772, 0.21220659], [0.023320543, 0.064345552], [0.450158158, -0.150052719]])],   
            
            ['3 layers, a const = 1 betwn[0,0.4] 2 betw[0.4,0.6] 3 betw[0.6,1], z = [0.1, 0.3], PTIB', 
             {'m': self.PTIB, 'at':[1, 2, 3], 'ab':[1, 2, 3],'zt':[0,0.4,0.6], 'zb':[0.4,0.6,1], 'z': [0.1, 0.3]}, 
             np.array([0.0615495559529622, 0.155881032360824])],   
               
            ['3 layers, a const = 1 betwn[0,0.4] 2 betw[0.4,0.6] 3 betw[0.6,1], z = [0, 0.4], PTIB', 
             {'m': self.PTIB, 'at':[1, 2, 3], 'ab':[1, 2, 3],'zt':[0,0.4,0.6], 'zb':[0.4,0.6,1], 'z': [0, 0.4]}, 
             np.array([0.121583557567097, 0.277782033661425])],

            ['3 layers, a const = 1 betwn[0,0.4] 2 betw[0.4,0.6] 3 betw[0.6,1], z = [0.2, 0.5], PTIB', 
             {'m': self.PTIB, 'at':[1, 2, 3], 'ab':[1, 2, 3],'zt':[0,0.4,0.6], 'zb':[0.4,0.6,1], 'z': [0.2, 0.5]}, 
             np.array([0.220181281555903, 0.359261900351956])],

            ['3 layers, a linear = 1+x betwn[0,0.4] 0.6+x betw[0.4,0.6] 0.4+x betw[0.6,1], z = [0.2, 0.8], PTIB', 
             {'m': self.PTIB, 'at':[1, 1, 1], 'ab':[1.4, 1.2, 1.4],'zt':[0,0.4,0.6], 'zb':[0.4,0.6,1], 'z': [0.2, 0.8]}, 
             np.array([0.469837253756176, 0.359386748693161])],
 
            ['3 layers, a const = 1 betwn[0,0.4] 2 betw[0.4,0.6] 3 betw[0.6,1], z = [0.2, 0.8], PTIB', 
             {'m': self.PTIB, 'at':[1, 2, 3], 'ab':[1, 2, 3],'zt':[0,0.4,0.6], 'zb':[0.4,0.6,1], 'z': [0.2, 0.8]}, 
             np.array([0.904514325378385, 0.372372215462563])],
            ]           