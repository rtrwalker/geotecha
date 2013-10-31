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

"""some input output stuff and checking stuff"""

from __future__ import division, print_function

import imp
import numpy as np
import textwrap

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
    

def copy_attributes_between_objects(from_object, to_object, attributes=[], defaults = dict(),  not_found_value = None):
    """Transfer object attributes between objects
    
    looks up `attributes` in `from_obj` and copies them to `to_object`.  If 
    not present in `from_object` the corresponding attribute in `to_object`
    will be set to `defaults` or `not_found_value` if not in `defaults`.
    
    Parameters
    ----------
    from_object: object or module
        object to copy attributes from
    to_object: object or module
        object to copy attributes to
    attributes: list of strings
        a list of attribute names to copy from `from_object` to `object_to`
    defaults: dict of string-value pairs
        dictionary specifying the default value of each attribute that is not 
        found in `from_object`.
    not_found_value: optional 
        default value to set attribute to if not found in `from_object` or in 
        `defaults`.  default = ``None``
        
    Returns
    -------
    None
    
    """    


    for v in attributes:
            to_object.__setattr__(v, from_object.__dict__.get(v, defaults.get(v, not_found_value)))

    return

def copy_attributes_from_text_to_object(reader,*args, **kwargs):
    """wrapper for `copy_attributes_between_objects` where `from_object` is fileobject, StringIO, text
    
    Parameters
    ----------
    reader : fileobject, StringIO, text
        text to turn into a module and pass as from_object
    
    See also
    --------
    copy_attributes_between_objects : see for args and kwargs input
    
    """  
    
    copy_attributes_between_objects(make_module_from_text(reader), *args, **kwargs)
    return


def check_attribute_is_list(obj, attributes=[], force_list=False):
    """check if objects attributes are lists

    if attribute is not a list and `force_list`=True then attribute will be
    placed in list.  e.g. say obj.a = 6, 6 is not a list so obj.a will be 
    changed to obj.a = [6].
    
    Parameters
    ----------
    obj: object
        object with attributes to check
    attributes: list of strings
        a list of attribute names to check
    force_list : bool, optional
        If True not-list attributes will be put in a list. If False then 
        non-list attributes will raise an error. Default=False
        
    Returns
    -------
    None
    
    """
    
    g = obj.__getattribute__
    for v in attributes:
        if not g(v) is None:
            if not isinstance(g(v), list):
                if force_list:
                    obj.__setattr__(v, [g(v)])    
                else:
                    raise ValueError('{0} should be a list. It is {1}'.format(v, type(g(v))))
    return
    

def check_attribute_PolyLines_have_same_x_limits(obj, attributes=[]):
    """check if objects attributes that are PolyLine have the same x values
    
    Each attribute can be a single instance of PolyLine or a list of PolyLines
    
    Parameters
    ----------
    obj: object
        object with attributes to check
    attributes: list of strings
        a list of attribute names to check
        
    Returns
    -------
    None
    
    """
    
    g = obj.__getattribute__
        
    #find first x values
    for v in attributes:
        if not g(v) is None:                
            if isinstance(g(v), list):
                xcheck = np.array([g(v)[0].x[0], g(v)[0].x[-1]])                   
                xstr = v
                break
            else:
                a = g(v).x[0]
                
                xcheck = np.array([g(v).x[0], g(v).x[-1]])
                xstr = v
                break
    #check against other x values
    for v in attributes:
        if not g(v) is None:                
            if isinstance(g(v), list):
                for j, u in enumerate(g(v)):
                    if not np.allclose([u.x[0], u.x[-1]], xcheck):                             
                        raise ValueError('All upper and lower x limits must be the same.  Check ' + v + ' and ' + xstr + '.')
            else:
                if not np.allclose([g(v).x[0], g(v).x[-1]], xcheck):
                    raise ValueError('All upper and lower x limits must be the same.  Check ' + v + ' and ' + xstr + '.')
    return


def check_attribute_pairs_have_equal_length(obj, attributes=[]):
    """check if attribute pairs ahave the same length    
    
    Parameters
    ----------
    obj: object
        object with attributes to check
    attributes: list of string
        list of attribute names to check.  Names will be considered in groups 
        of two e.g. attributes=['a', 'b', 'c', 'd',...] will compare a with b, c
        with d etc.
        
    Returns
    -------
    None
    
    """
    
    g = obj.__getattribute__
        
    # for iterating in chuncks see http://stackoverflow.com/a/434328/2530083
    for v1, v2 in [attributes[pos:pos + 2] for pos in xrange(0, len(attributes), 2)]:        
        if sum([v is None for v in [g(v1), g(v2)]])==0:
            if sum([hasattr(v,'__len__') for v in [g(v1), g(v2)]])!=2:
                raise TypeError("either {0} and {1} have no len attribute and so cannot be compared".format(v1,v2))                                
            if len(g(v1)) != len(g(v2)):
                raise ValueError("%s has %d elements, %s has %d elements.  They should have the same number of elements." % (v1,len(g(v1)), v2, len(g(v2))))
                
        elif sum([v is None for v in [g(v1), g(v2)]])==1:
            raise ValueError("Cannot compare length of {0} and {1} when one of them is None".format(v1,v2))
    return
    

def check_attribute_combinations(obj, zero_or_all=[], at_least_one=[], one_implies_others=[]):
    """check for incomplete combinations of attributes
    
    Raises ValueError if any combination fails
    
    Parameters
    ----------
    zero_or_all : list of list of string
        each element of `zero_or_all` is a list of attribute names that should
        either allbe None or all be not None.
    at_least_one : list of list of string
        each element of `at_least_one` is a list of attribute names of which at
        least one should not be None.
    one_implies_others: list of list of string
        each element of `one_implies_others` is a list of attribute names.  If
        the first attribute in the list is not None then all other members of 
        the list should also not be None.

    Returns
    -------
    None
    
    """ 
    
    g = obj.__getattribute__

    for check in zero_or_all:
        if sum([g(v) is None for v in check]) in range(1, len(check)):
            raise ValueError('Either zero or all of the following variables must be defined: ' + ', '.join(check))
            
    for check in at_least_one:
        if not any([not g(v) is None for v in check]):
            if len(check)==1:
                raise ValueError('Need the following variable: ' + ', '.join(check))
            else:
                raise ValueError('Need at least one of the following variables: ' + ', '.join(check))
    
    for i, check in enumerate(one_implies_others):
        if len(check)<=1:
            raise ValueError("each member of 'one_implies_others' must be a list with at least two elements.  Member {0} is {1}".format(i,', '.join(check)))
        if not g(check[0]) is None:
            if not all([not g(v) is None for v in check[1:]]):
                raise ValueError('If {0} is defined then the following variables must also be defined: '.format(check[0]) + ', '.join(check[1:]))
            
    return
    
            
def initialize_objects_attributes(obj, attributes=[], defaults = dict(),  not_found_value = None):
    """initialize an objects attributes
    
    for each attribute set it to the value found in `defaults` dictionary 
    or , if not found, set it to `not_found_value`.
    
    Parameters
    ----------
    obj: object 
        object to set attributes in
    
    attributes: list of strings
        a list of attribute names to set
    defaults: dict of string-value pairs
        dictionary specifying the default value of each attribute
    not_found_value: optional 
        default value to set attribute to if not found in `defaults`. 
        default = ``None``
        
    Returns
    -------
    None
    
    Notes
    -----
    If using this function to initialize attributes in a class then just
    be aware that the attributes will not be available for auto-complete 
    until an instance of the class is created.  This can be annoying when 
    coding the class itself because typing 'self.' will not have any 
    autocomplete options.  To work around this use a temporary explicit 
    assignment e.g. 'self.a = 6' and then later comment it out when coding of 
    the class is finsihed.
    
    See also
    --------
    code_for_explicit_attribute_initialization: use for temporary explicit
        attribute initialization to facilitate auto-complete, then comment 
        out when done        
    
    """    

    for v in attributes:
        obj.__setattr__(v, defaults.get(v, not_found_value))
    return
    
def code_for_explicit_attribute_initialization(
        attributes=[],
        defaults={},
        defaults_name = '_attribute_defaults',
        object_name = 'self',        
        not_found_value = None):
    """generate code to initialize an objects attributes

    Parameters
    ----------
    object_name: string , optional
        name of object to set attributes. default='self'    
    attributes: list of strings
        a list of attribute names to set
    defaults: dict of string-value pairs
        dictionary specifying the default value of each attribute
    defaults_name: string, optional
        If ``None`` then the actual values in `defaults` will be used
        to set attributes.  If not ``None`` then those attributes with 
        defaults will be initialized by pointing to a dictionary called 
        `defaults_name`.  default = '_attribute_defaults', 
    not_found_value: optional 
        default value to set attribute to if not found in `defaults`. 
        default = ``None``

    Returns
    -------
    out: string
        code that can be pasted into an object for initialization of 
        attributes

    See also
    --------
    initialize_objects_attributes: similar functionality with no copy paste
            
    """

    out = ""
    if defaults_name is None:
        for v in attributes:
            v2 = defaults.get(v, not_found_value)            
            if isinstance(v2, str):                
                v2 = "'{0}'".format(v2)
            
            out+='{0} = {1}\n'.format('.'.join([object_name,v]), v2)            
    else:
        for v in attributes:
            if v in defaults.keys():
                v2 = not_found_value
                if isinstance(v2, str):
                    v2 = "'{0}'".format(v2)
                out+='{0} = {1}.get({2}, {3})\n'.format('.'.join([object_name, v]), 
                    '.'.join([object_name, defaults_name]),
                     "'{}'".format(v), 
                     v2)
            else:
                v2 = not_found_value
                if isinstance(v2, str):
                    v2 = "'{0}'".format(v2)
                out+='{0} = {1}\n'.format('.'.join([object_name,v]), v2)
    return out           
                

            
if __name__=='__main__':
    b = {'H': 1.0, 'drn': 0, 'dT': 1.0, 'neig': 2, 'mvref':1.0, 'kvref': 1.0, 'khref': 1.0, 'etref': 'yes1.01' }
    a = 'H drn dT neig mvref kvref khref etref dTh dTv mv kh kv et surcharge_vs_depth surcharge_vs_time vacuum_vs_depth vacuum_vs_time top_vs_time bot_vs_time ppress_z avg_ppress_z_pairs settlement_z_pairs tvals'.split()                
    print(code_for_explicit_attribute_initialization(a,b, not_found_value=None))
        
        
#print(code_for_explicit_attribute_initialization(a,b, None, not_found_value='sally'))        
##def code_for_explicit_attribute_initialization(
##        attributes=[],
##        defaults={},
##        defaults_name = '_attribute_defaults',
##        object_name = 'self',        
##        not_found_value = None):
#a = 'a b c'.split
#b = {'a': 3,'b': 6}
#c = None
#d=self
#e = None
            
    




    
#class EmptyClass(object):
#    """empty class for assigning attributes fot object testing"""
#    def __init__(self):
#        pass
#
#if __name__=='__main__':
#    
#    a = EmptyClass()
#    
#    a.a = None
#    a.b = None
#    a.c = 1
#    a.d = 2
#    a.e = None
#    a.f = 5
#
#    #check_attribute_combinations(a, at_least_one=[['c','d']]) 
#    check_attribute_combinations(a, one_implies_others=[['a','b']])