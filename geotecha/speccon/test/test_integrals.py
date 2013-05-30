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

from geotecha.speccon.integrals import dim1sin_af_linear
from geotecha.speccon.integrals import dim1sin_abf_linear
from geotecha.speccon.integrals import m_from_sin_mx





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

