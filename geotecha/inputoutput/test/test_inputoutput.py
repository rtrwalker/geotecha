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

import unittest

from testfixtures import TempDirectory
import textwrap
from math import pi
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
import os


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
from geotecha.inputoutput.inputoutput import string_of_object_attributes
from geotecha.inputoutput.inputoutput import next_output_stem
from geotecha.inputoutput.inputoutput import make_array_into_dataframe
from geotecha.inputoutput.inputoutput import save_grid_data_to_file
from geotecha.inputoutput.inputoutput import GenericInputFileArgParser
from geotecha.inputoutput.inputoutput import working_directory
from geotecha.inputoutput.inputoutput import hms_string

from geotecha.inputoutput.inputoutput import fcode_one_large_expr



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


class test_string_of_object_attributes(unittest.TestCase):
    """ tests for string_of_object_attributes"""
#    string_of_object_attributes(obj, attributes=[], none_at_bottom=True,
#                                    numpy_array_prefix = "np."):

    def test_defaults(self):

        a=EmptyClass()
        a.a=None
        a.b=4
        a.c = np.array([1,2,3])
        a.d='happy'
        assert_equal(string_of_object_attributes(a, 'a b c d'.split()),
                     textwrap.dedent("""\
                     b = 4
                     c = np.array([          1,           2,           3])
                     d = 'happy'


                     a = None
                     """))

    def test_numpy_array_prefix_none(self):

        a=EmptyClass()
        a.a=None
        a.b=4
        a.c = np.array([1,2,3])
        a.d='happy'
        assert_equal(string_of_object_attributes(a, 'a b c d'.split(), numpy_array_prefix = None),
                     textwrap.dedent("""\
                     b = 4
                     c = array([       1,        2,        3])
                     d = 'happy'


                     a = None
                     """))
    def test_none_at_bottom(self):

        a=EmptyClass()
        a.a=None
        a.b=4
        a.d='happy'
        assert_equal(string_of_object_attributes(a, 'a b c d'.split(), none_at_bottom=False),
                     textwrap.dedent("""\
                     a = None
                     b = 4
                     c = None
                     d = 'happy'
                     """))


class test_next_output_stem(unittest.TestCase):
    """tests for next_output_stem"""
    #next_output_stem(prefix, path=None, start=1, inc=1, zfill=3,
    #       overwrite=False)

    def setUp(self):
        self.tempdir = TempDirectory()
        self.tempdir.write('a_004', b'some text a4')
        self.tempdir.write('a_005', b'some text a5')
        self.tempdir.write('b_002.txt', b'some text b2')
        self.tempdir.write('b_008.out', b'some text b8')
        self.tempdir.write(('c_010', 'por'), b'some text c5por')

    def tearDown(self):
        self.tempdir.cleanup()

#    @with_setup(setup=self.setup, teardown=self.teardown)
    def test_file(self):
        assert_equal(next_output_stem(prefix='a_', path=self.tempdir.path),
                     'a_006')
    def test_file2(self):
        assert_equal(next_output_stem(prefix='b_', path=self.tempdir.path),
                     'b_009')
    def test_directory(self):
        assert_equal(next_output_stem(prefix='c_', path=self.tempdir.path),
                     'c_011')
    def test_file_overwrite(self):
        assert_equal(next_output_stem(prefix='a_', path=self.tempdir.path,
                                      overwrite=True),
                     'a_005')
    def test_inc(self):
        assert_equal(next_output_stem(prefix='a_', path=self.tempdir.path,
                                      inc=3),
                     'a_008')
    def test_zfill(self):
        assert_equal(next_output_stem(prefix='a_', path=self.tempdir.path,
                                      zfill=5),
                     'a_00006')
    def test_does_not_exist(self):
        assert_equal(next_output_stem(prefix='g_', path=self.tempdir.path),
                     'g_001')
    def test_does_not_exist(self):
        assert_equal(next_output_stem(prefix='g_', path=self.tempdir.path,
                                      start=4),
                     'g_004')

class test_make_array_into_dataframe(unittest.TestCase):
    """tests for make_array_into_dataframe"""
#    make_array_into_dataframe(data, column_labels=None, row_labels=None,
#                                row_labels_label='item')

    def test_defaults(self):
        assert_frame_equal(make_array_into_dataframe(
                        data=np.arange(10).reshape((5,2))),
                 pd.DataFrame(data=np.arange(10).reshape((5,2))))
    def test_column_labels(self):
        assert_frame_equal(make_array_into_dataframe(
                        data=np.arange(10).reshape((5,2)),
                        column_labels=['a', 'b']),
                 pd.DataFrame(data=np.arange(10).reshape((5,2)),
                              columns=['a', 'b']))

    def test_row_labels(self):

        df = pd.DataFrame(data=np.arange(10).reshape((5,2)))
        s = pd.Series(['a', 'b', 'c', 'd', 'e'])
        df.insert(loc=0, column='item', value=s)

        assert_frame_equal(make_array_into_dataframe(
                        data=np.arange(10).reshape((5,2)),
                        row_labels=['a', 'b', 'c', 'd', 'e']),

                 df)
    def test_row_labels_label(self):

        df = pd.DataFrame(data=np.arange(10).reshape((5,2)))
        s = pd.Series(['a', 'b', 'c', 'd', 'e'])
        df.insert(loc=0, column='hey', value=s)

        assert_frame_equal(make_array_into_dataframe(
                        data=np.arange(10).reshape((5,2)),
                        row_labels=['a', 'b', 'c', 'd', 'e'],
                        row_labels_label='hey'),

                 df)





class test_save_grid_data_to_file(unittest.TestCase):
    """tests for save_grid_data_to_file"""
#    save_grid_data_to_file(directory=None, file_stem='out_000',
#                           create_directory=True, ext='.csv', *data_dicts)

    def setUp(self):
        self.tempdir = TempDirectory()


    def tearDown(self):
        self.tempdir.cleanup()

    def test_defaults(self):

        data = np.arange(6).reshape(3,2)
        save_grid_data_to_file({'data': data},
                               directory=self.tempdir.path)
        assert_equal(self.tempdir.read(('out_000','out_000.csv')).splitlines(),
                            textwrap.dedent("""\
                            ,0,1
                            0,0,1
                            1,2,3
                            2,4,5""").splitlines())
    def test_directory(self):

        data = np.arange(6).reshape(3,2)
        save_grid_data_to_file({'data': data},
                               directory=os.path.join(self.tempdir.path,'g'))
        assert_equal(self.tempdir.read(('g','out_000','out_000.csv')).splitlines(),
                            textwrap.dedent("""\
                            ,0,1
                            0,0,1
                            1,2,3
                            2,4,5""").splitlines())
    def test_file_stem(self):

        data = np.arange(6).reshape(3,2)
        save_grid_data_to_file({'data': data},
                               directory=self.tempdir.path,
                               file_stem="ppp")
        assert_equal(self.tempdir.read(('ppp','ppp.csv')).splitlines(),
                            textwrap.dedent("""\
                            ,0,1
                            0,0,1
                            1,2,3
                            2,4,5""").splitlines())
    def test_ext(self):

        data = np.arange(6).reshape(3,2)
        save_grid_data_to_file({'data': data},
                               directory=self.tempdir.path,
                               ext=".out")
        assert_equal(self.tempdir.read(('out_000','out_000.out')).splitlines(),
                            textwrap.dedent("""\
                            ,0,1
                            0,0,1
                            1,2,3
                            2,4,5""").splitlines())
    def test_create_directory(self):

        data = np.arange(6).reshape(3,2)
        save_grid_data_to_file({'data': data},
                               directory=self.tempdir.path,
                               create_directory=False)
        assert_equal(self.tempdir.read('out_000.csv').splitlines(),
                            textwrap.dedent("""\
                            ,0,1
                            0,0,1
                            1,2,3
                            2,4,5""").splitlines())
    def test_data_dict_header(self):

        data = np.arange(6).reshape(3,2)
        save_grid_data_to_file({'data': data, 'header':'hello header'},
                               directory=self.tempdir.path)
        assert_equal(self.tempdir.read(('out_000','out_000.csv')).splitlines(),
                            textwrap.dedent("""\
                            hello header
                            ,0,1
                            0,0,1
                            1,2,3
                            2,4,5""").splitlines())
    def test_data_dict_name(self):

        data = np.arange(6).reshape(3,2)
        save_grid_data_to_file({'data': data, 'name':'xx'},
                               directory=self.tempdir.path)
        assert_equal(self.tempdir.read(('out_000','out_000xx.csv')).splitlines(),
                            textwrap.dedent("""\
                            ,0,1
                            0,0,1
                            1,2,3
                            2,4,5""").splitlines())

    def test_data_dict_row_labels(self):

        data = np.arange(6).reshape(3,2)
        save_grid_data_to_file({'data': data, 'row_labels':[8,12,6]},
                               directory=self.tempdir.path)
        assert_equal(self.tempdir.read(('out_000','out_000.csv')).splitlines(),
                            textwrap.dedent("""\
                            ,item,0,1
                            0,8,0,1
                            1,12,2,3
                            2,6,4,5""").splitlines())
    def test_data_dict_row_labels_label(self):

        data = np.arange(6).reshape(3,2)
        save_grid_data_to_file({'data': data, 'row_labels':[8,12,6],
                                'row_labels_label':'yyy'},
                               directory=self.tempdir.path)
        assert_equal(self.tempdir.read(('out_000','out_000.csv')).splitlines(),
                            textwrap.dedent("""\
                            ,yyy,0,1
                            0,8,0,1
                            1,12,2,3
                            2,6,4,5""").splitlines())
    def test_data_dict_column_labels(self):

        data = np.arange(6).reshape(3,2)
        save_grid_data_to_file({'data': data, 'column_labels':['a', 'b']},
                               directory=self.tempdir.path)
        assert_equal(self.tempdir.read(('out_000','out_000.csv')).splitlines(),
                            textwrap.dedent("""\
                            ,a,b
                            0,0,1
                            1,2,3
                            2,4,5""").splitlines())
    def test_two_files(self):

        data = np.arange(6).reshape(3,2)
        save_grid_data_to_file([{'data': data, 'name':1},
                                {'data': 2*data, 'name':2}],
                               directory=self.tempdir.path,
                               file_stem="qqq")
        assert_equal(self.tempdir.read(('qqq','qqq1.csv')).splitlines(),
                            textwrap.dedent("""\
                            ,0,1
                            0,0,1
                            1,2,3
                            2,4,5""").splitlines())
        assert_equal(self.tempdir.read(('qqq','qqq2.csv')).splitlines(),
                            textwrap.dedent("""\
                            ,0,1
                            0,0,2
                            1,4,6
                            2,8,10""").splitlines())



class HelperForGenericInputFileArgParser(object):
    def __init__(self, path):
        self.path=path
        self.oname = os.path.join(os.path.dirname(path), 'out.zebra')

        with open(self.oname, 'a') as f:
            f.write(os.path.basename(path)+'\n')
        return

    def dog(self):
        with open(self.oname, 'a') as f:
            f.write('dog\n')



class test_GenericInputFileArgParser(unittest.TestCase):
    """tests GenericInputFileArgParser"""

    def setUp(self):
        self.tempdir = TempDirectory()
        self.tempdir.write('a1.py', "a1")
        self.tempdir.write('a2.py', "a2")
        self.tempdir.write('b1.txt', "b1")
        self.tempdir.write('b2.txt', "b2")

    def tearDown(self):
        self.tempdir.cleanup()

    def _abc_fobj(self, fobj):
        """print file contents to out.zebra"""
        with open(os.path.join(self.tempdir.path, 'out.zebra'), 'a') as f:
            f.write(fobj.read()+'\n')
        return

    def _abc_path(self, path):
        """print file basename out.zebra"""
        with open(os.path.join(self.tempdir.path, 'out.zebra'), 'a') as f:
            f.write(os.path.basename(path)+'\n')
        return
#    def dog(self):
#        with open(os.path.join(self.tempdir.path, 'out.zebra'), 'a') as f:
#            f.write('dog\n')

    def test_directory_with_path(self):

        a = GenericInputFileArgParser(self._abc_path, False)
        args = '-d {0} -p'.format(self.tempdir.path).split()
        print(args)
        a.main(argv=args)

        assert_equal(self.tempdir.read(('out.zebra')).splitlines(),
                            textwrap.dedent("""\
                            a1.py
                            a2.py""").splitlines())

    def test_directory_with_fobj(self):

        a = GenericInputFileArgParser(self._abc_fobj, True)
        args = '-d {0} -p'.format(self.tempdir.path).split()
#        print(args)
        a.main(argv=args)

        assert_equal(self.tempdir.read(('out.zebra')).splitlines(),
                            textwrap.dedent("""\
                            a1
                            a2""").splitlines())

    def test_pattern_with_path(self):

        a = GenericInputFileArgParser(self._abc_path, False)
        args = '-d {0} -p *.txt'.format(self.tempdir.path).split()
#        print(args)
        a.main(argv=args)

        assert_equal(self.tempdir.read(('out.zebra')).splitlines(),
                            textwrap.dedent("""\
                            b1.txt
                            b2.txt""").splitlines())

    def test_pattern_with_fobj(self):

        a = GenericInputFileArgParser(self._abc_fobj, True)
        args = '-d {0} -p *.txt'.format(self.tempdir.path).split()
#        print(args)
        a.main(argv=args)

        assert_equal(self.tempdir.read(('out.zebra')).splitlines(),
                            textwrap.dedent("""\
                            b1
                            b2""").splitlines())

    def test_filename_with_fobj(self):

        a = GenericInputFileArgParser(self._abc_fobj, True)
        args = '-f {0} {1}'.format(
            os.path.join(self.tempdir.path, 'a1.py'),
            os.path.join(self.tempdir.path, 'b1.txt')
                                ).split()
#        print(args)
        a.main(argv=args)

        assert_equal(self.tempdir.read(('out.zebra')).splitlines(),
                            textwrap.dedent("""\
                            a1
                            b1""").splitlines())


    def test_filename_with_path(self):

        a = GenericInputFileArgParser(self._abc_path, False)
        args = '-f {0} {1}'.format(
            os.path.join(self.tempdir.path, 'a1.py'),
            os.path.join(self.tempdir.path, 'b1.txt')
                                ).split()
#        print(args)
        a.main(argv=args)

        assert_equal(self.tempdir.read(('out.zebra')).splitlines(),
                            textwrap.dedent("""\
                            a1.py
                            b1.txt""").splitlines())



    def test_default_directory(self):

        a = GenericInputFileArgParser(self._abc_path, False)
        args = '-d -p'.format(self.tempdir.path).split()

        with working_directory(self.tempdir.path):
            a.main(argv=args)

        assert_equal(self.tempdir.read(('out.zebra')).splitlines(),
                            textwrap.dedent("""\
                            a1.py
                            a2.py""").splitlines())

    def test_methods_with_path(self):




        a = GenericInputFileArgParser(HelperForGenericInputFileArgParser,
                                      False, [('dog',[],{})])

        args = '-f {0} {1}'.format(
            os.path.join(self.tempdir.path, 'a1.py'),
            os.path.join(self.tempdir.path, 'b1.txt')
                                ).split()

        a.main(argv=args)

        assert_equal(self.tempdir.read(('out.zebra')).splitlines(),
                            textwrap.dedent("""\
                            a1.py
                            dog
                            b1.txt
                            dog
                            """).splitlines())

class test_hms_string(unittest.TestCase):
    """tests hms_string"""
    #hms_string(sec_elapsed)

    def test_20s(self):
        assert_equal(hms_string(20),"0:00:20.00")

    def test_130p5s(self):
        assert_equal(hms_string(130.5),"0:02:10.50")



class test_fcode_one_large_expr(unittest.TestCase):
    """tests for fcode_one_large_expr"""

    #fcode_one_large_expr(expr, prepend=None, **settings)

    import sympy
    from sympy import symbols, sin

    n= 8
    sss = symbols(','.join(['a%d' % v for v in range(n)]))

    e1 = 0
    for i in sss:
        e1+=sin(i)

    m = sympy.tensor.IndexedBase('m')
    j = sympy.tensor.Idx('j')
    e2 = m[j] + m[j+1]

    def test_line_wrap(self):
        assert_equal(fcode_one_large_expr(self.e1).splitlines(),
                     '      (sin(a0) + sin(a1) + sin(a2) + sin(a3) '
                     '+ sin(a4) + sin(a5) + sin(&\n      a6) + '
                     'sin(a7))'.splitlines())

    def test_prepend(self):
        assert_equal(fcode_one_large_expr(self.e1, prepend='k=').splitlines(),
                     '      k=(sin(a0) + sin(a1) + sin(a2) + sin(a3) + '
                     'sin(a4) + sin(a5) + sin&\n      (a6) + '
                     'sin(a7))'.splitlines())
    def test_settings(self):
        #note this only tests to see if setting s is passed correctly
        assert_equal(fcode_one_large_expr(self.e1,
                                          source_format="free").splitlines(),
                     '(sin(a0) + sin(a1) + sin(a2) + sin(a3) + sin(a4) + '
                     'sin(a5) + sin(a6) + sin(a7))'.splitlines())

    def test_parentheses(self):
        assert_equal(fcode_one_large_expr(self.e2),
                     '      (m(j + 1) + m(j))')


if __name__ == '__main__':

    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])