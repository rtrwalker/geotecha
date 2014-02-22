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
"""Some test routines for the inputoutput module

"""

from __future__ import division, print_function
import ast
from nose import with_setup
from nose.tools.trivial import assert_almost_equal
from nose.tools.trivial import assert_raises
from nose.tools.trivial import ok_
from nose.tools.trivial import assert_equal
from nose.tools.trivial import assert_equals

import textwrap
from math import pi
import numpy as np
from geotecha.piecewise.piecewise_linear_1d import PolyLine

from geotecha.inputoutput.inputoutput import make_module_from_text
from geotecha.inputoutput.inputoutput import copy_attributes_between_objects
from geotecha.inputoutput.inputoutput import copy_attributes_from_text_to_object
from geotecha.inputoutput.inputoutput import check_attribute_is_list
from geotecha.inputoutput.inputoutput import check_attribute_PolyLines_have_same_x_limits
from geotecha.inputoutput.inputoutput import check_attribute_pairs_have_equal_length
from geotecha.inputoutput.inputoutput import check_attribute_combinations
from geotecha.inputoutput.inputoutput import initialize_objects_attributes
from geotecha.inputoutput.inputoutput import code_for_explicit_attribute_initialization
from geotecha.inputoutput.inputoutput import object_members
from geotecha.inputoutput.inputoutput import SyntaxChecker
from geotecha.inputoutput.inputoutput import force_attribute_same_len_if_none

class EmptyClass(object):
    """empty class for assigning attributes fot object testing"""
    def __init__(self):
        pass



def test_make_module_from_text():
    """test for make_module_from_text function"""
    #make_module_from_text(reader)
    reader = textwrap.dedent("""\
        a = 2
        """)

    ok_(isinstance(make_module_from_text(reader), type(textwrap)))

    assert_equal(make_module_from_text(reader).a, 2)

    assert_raises(SyntaxError,make_module_from_text,
                  reader,
                  syntax_checker=SyntaxChecker())




def test_copy_attributes_between_objects():
    """test for copy_attributes_between_objects function"""
    #copy_attributes_between_objects(from_object, to_object, attributes=[], defaults = dict(),  not_found_value = None)

    a = EmptyClass()

    from_object = EmptyClass()

    from_object.a = 2
    from_object.b = 3


    copy_attributes_between_objects(from_object,a,['a','b', 'aa', 'bb'], {'bb': 27})
    assert_equal([a.a, a.b, a.aa, a.bb], [2, 3, None, 27])

    copy_attributes_between_objects(from_object,a,['c'], not_found_value = 1000)
    assert_equal([a.c], [1000])

def test_copy_attributes_from_text_to_object():
    """test for copy_attributes_from_text_to_object function"""
    #copy_attributes_from_text_to_object(reader,*args, **kwargs)
    reader = textwrap.dedent("""\
        a = 2
        b = 3
        """)
    a = EmptyClass()
    copy_attributes_from_text_to_object(reader,a,['a','b', 'aa', 'bb'], {'bb': 27})
    assert_equal([a.a, a.b, a.aa, a.bb], [2, 3, None, 27])


def test_check_attribute_is_list():
    """test for check_attribute_is_list function"""
    #check_attribute_is_list(obj, attributes=[], force_list=False)

    a = EmptyClass()

    a.a = 2
    a.b = 4
    a.c = [8]
    a.d = [6,7]

    assert_raises(ValueError, check_attribute_is_list, a, attributes=['a','b','c'], force_list=False)

    check_attribute_is_list(a, attributes=['a','b','c'], force_list=True)

    assert_equal([a.a,a.b,a.c,a.d], [[2],[4],[8], [6,7]])

def test_check_attribute_PolyLines_have_same_x_limits():
    """test for check_attribute_PolyLines_have_same_x_limits function"""
    #check_attribute_PolyLines_have_same_x_limits(obj, attributes=[])

    a = EmptyClass()

    a.a = None
    a.b = PolyLine([0,4],[4,5])
    a.c = [PolyLine([0,4],[6,3]), PolyLine([0,5],[6,3])]
    a.d = PolyLine([0,2,4], [3,2,4])

    assert_raises(ValueError, check_attribute_PolyLines_have_same_x_limits, a,
                  attributes=['a','b','c','d'])

    assert_raises(ValueError, check_attribute_PolyLines_have_same_x_limits, a,
                  attributes=['c'])

    assert_equal(check_attribute_PolyLines_have_same_x_limits(a,
                  attributes=['a','b','d']), None)

def test_check_attribute_pairs_have_equal_length():
    """test for check_attribute_pairs_have_equal_length function"""
    #check_attribute_pairs_have_equal_length(obj, attributes=[])

    a = EmptyClass()

    a.a = None
    a.b = [7, 8]
    a.c = [8]
    a.d = [6,7]
    a.e = 8

#    assert_raises(ValueError, check_attribute_pairs_have_equal_length, a,
#                  attributes=[['a','b']])
    assert_raises(ValueError, check_attribute_pairs_have_equal_length, a,
                  attributes=[['b','c']])
    assert_raises(TypeError, check_attribute_pairs_have_equal_length, a,
                  attributes=[['b','e']])

    assert_equal(check_attribute_pairs_have_equal_length(a,
                  attributes=[['b','d']]), None)

def test_check_attribute_combinations():
    """test for check_attribute_combinations function"""
    #check_attribute_combinations(obj, zero_or_all=[], at_least_one=[], one_implies_others=[])

    a = EmptyClass()

    a.a = None
    a.b = None
    a.c = 1
    a.d = 2
    a.e = None
    a.f = 5

    assert_equal(check_attribute_combinations(a, zero_or_all=[['a','b']]), None)
    assert_equal(check_attribute_combinations(a, zero_or_all=[['c','d']]), None)
    assert_equal(check_attribute_combinations(a, zero_or_all=[['a','b'],['c','d']]), None)
    assert_raises(ValueError, check_attribute_combinations,a,  zero_or_all=[['a','c']])
    assert_raises(ValueError, check_attribute_combinations,a,  zero_or_all=[['a','b'], ['a','c']])

    assert_equal(check_attribute_combinations(a, at_least_one=[['a','c','e']]), None)
    assert_equal(check_attribute_combinations(a, at_least_one=[['c','d']]), None)
    assert_equal(check_attribute_combinations(a, at_least_one=[['a','c'],['c']]), None)
    assert_raises(ValueError, check_attribute_combinations, a, at_least_one=[['a','b','e']])
    assert_raises(ValueError, check_attribute_combinations, a, at_least_one=[['a','c'], ['a','b','e']])

    assert_equal(check_attribute_combinations(a, one_implies_others=[['c','d']]), None)
    assert_equal(check_attribute_combinations(a, one_implies_others=[['a','b']]), None)
    assert_equal(check_attribute_combinations(a, one_implies_others=[['a','b'],['c','d','f']]), None)
    assert_raises(ValueError, check_attribute_combinations, a,
                  one_implies_others=[['c','a']])
    assert_raises(ValueError, check_attribute_combinations, a,
                  one_implies_others=[['c','d','e']])
    assert_raises(ValueError, check_attribute_combinations, a,
                  one_implies_others=[['c','d'], ['c','d','e']])
def test_initialize_objects_attributes():
    """test for initialize_objects_attributes function"""
    #initialize_objects_attributes(obj, attributes=[], defaults = dict(),  not_found_value = None):

    a = EmptyClass()
    initialize_objects_attributes(a,attributes=['a','b'], defaults={'a': 6})
    assert_equal([a.a,a.b],[6,None])


def test_code_for_explicit_attribute_initialization():
    """test for code_for_explicit_attribute_initialization function"""
    #code_for_explicit_attribute_initialization(attributes=[], defaults={}, defaults_name = '_attribute_defaults', object_name = 'self', not_found_value = None)

    a = 'a b c'.split
    b = {'a': 3,'b': 6}
    c = None
    e = None

    assert_equal(code_for_explicit_attribute_initialization('a b c'.split(), {'a': 3,'b': 6}, None), 'self.a = 3\nself.b = 6\nself.c = None\n')
    assert_equal(code_for_explicit_attribute_initialization('a b c'.split(), {'a': 3,'b': 6}, None, not_found_value='sally'), "self.a = 3\nself.b = 6\nself.c = 'sally'\n")
    assert_equal(code_for_explicit_attribute_initialization('a b c'.split(), {'a': 3,'b': 6}), "self.a = self._attribute_defaults.get('a', None)\nself.b = self._attribute_defaults.get('b', None)\nself.c = None\n")

def test_force_attribute_same_len_if_none():
    """test for force_attribute_same_len_if_none"""
    #force_attribute_same_len_if_none(obj, same_len_attributes=[], value=None)

    a = EmptyClass
    a.a = [3,4]
    a.b = None
    a.c = [7,2,3]
    a.d = None

    force_attribute_same_len_if_none(a, same_len_attributes=[['a', 'b']], value=None)
    assert_equal(a.b,[None, None])
    force_attribute_same_len_if_none(a, same_len_attributes=[['d', 'c']], value=None)
    assert_equal(a.c, [7, 2, 3])

def test_object_members():
    """test for object_members function"""
    import math
    ok_(set(['acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh',
                 'ceil', 'copysign', 'cos', 'cosh', 'degrees', 'erf', 'erfc',
                 'exp', 'expm1', 'fabs', 'factorial', 'floor', 'fmod',
                 'frexp', 'fsum', 'gamma', 'hypot', 'isinf', 'isnan', 'ldexp',
                 'lgamma', 'log', 'log10', 'log1p', 'modf', 'pow', 'radians',
                 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'trunc']).issubset(
                 set(object_members(math, 'routine', join=False))))


def test_SyntaxChecker():
    """test for SytaxChecker class"""

    syntax_checker=SyntaxChecker(['ast','builtin','numpy','PolyLine'])


    assert_raises(SyntaxError, syntax_checker.visit,
                  ast.parse('import math', mode='exec'))

    assert_raises(SyntaxError, syntax_checker.visit,
                  ast.parse('from math import cos', mode='exec'))

    assert_raises(SyntaxError, syntax_checker.visit,
                  ast.parse('eval(44*2)', mode='exec'))

    assert_raises(SyntaxError, syntax_checker.visit,
                  ast.parse('exec "a=34"', mode='exec'))

    assert_raises(SyntaxError, syntax_checker.visit,
                  ast.parse("""[x for x in ().__class__.__bases__[0].__subclasses__()
           if x.__name__ == 'Popen'][0](['ls', '-la']).wait()""", mode='exec'))
