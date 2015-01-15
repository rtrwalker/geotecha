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

"""Input/output operations,attribute checking, proccess input files that
have python syntax.

"""

from __future__ import division, print_function
import __builtin__
import sys
import ast
import imp
import textwrap
import inspect
import numpy as np
from geotecha.piecewise.piecewise_linear_1d import PolyLine
from sympy.printing.fcode import FCodePrinter
import multiprocessing
import time
from StringIO import StringIO
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import wx
import fnmatch
import argparse
import logging
from contextlib import contextmanager
import pkg_resources
from geotecha.plotting.one_d import copy_dict
import pkgutil
import importlib

class SyntaxChecker(ast.NodeVisitor):
    """Check if a string contains only allowabel python syntax


    `SyntaxChecker` provides functionality to check syntax of string using the
    ast module.  Basically white-list allowable functions and attributes.  A
    SyntaxError will be raised if any code is found that is not allowed.

    Parameters
    ----------
    allow : list of str
        List of SyntaxChecker 'allow' methods to call. Default = [].
        e.g. allow=['ast', 'numpy'] will call SyntaxChecker.allow_ast(),
        SyntaxChecker.allow_numpy()

    Attributes
    ----------
    allowed_functions : dict
        A dictionary of allowable functions. e.g. allowed_functions['cos'] =
        math.cos would permit the use of 'cos' from the math module.
        allowed_functions['math'] = math would allow use of the math
        module (note to allow things like 'math.sin' you would also have to
        add 'sin' to the allowable_attributes).  Adding functions one by one
        can be cumbersome; see the 'allow_<some_name>' methods for bulk adding
        of common functionality.
    allowed_node_types : dict
        Dictionary of allowable ast node names.
        e.g. allowed_node_types['Name'] = ast.Name would allow ast.Name nodes.
        typically use SytaxChecker.allow_ast to allow a reasonable set of ast
        nodes
    allowed_attributes : set of string
        set of allowable attributes.  e.g. you have already allowed the math
        module with allowed_functions['math'] = math.
        allowable_attributes.add('tan') will allow use of math.tan(34) etc.
        Note that the attribute 'tan' could now be used as an attribute for
        any of the functions in allowed_functions even though this may not
        make any logical sense.  i.e. it will pass the SyntaxChecker but would
        fail if the code was executed. Adding attributes one by one
        can be cumbersome; see the 'allow_<some_name>' methods for bulk adding
        of common functionality.
    safe_names : dict
        Dictionary of safe names.
        Default safe_names={'True': True, 'False': False, 'None': None}.
    print_each_node : bool
        Print out each node when they are visited.
        Default print_each_node=False. Note that
        nodes may be printed twice, once for a generic visit and once for a
        specific visit such as visit_Name, visit_Attribute etc.

    Methods
    -------
    allow_ast
        Allow a subset of ast node types.
    allow_builtin
        Allow a subset of __builtins__ functions.
    allow_numpy
        Allow a subset of numpy functionality via np.funtion syntax.
    allow_PolyLine
        Allow PolyLine class from geotecha.piecewise.piecewise_linear_1d.

    See also
    --------
    ast.NodeVisitor : Parent class.  Descriptions of python syntax grammar.
    object_members : Easily print a string of an objects routines for use in
        a 'allow_<some_name>' method.

    Notes
    -----
    If subclassing new 'allow_<some_name>' methods can be written to bulk add
    allowable functions and attributes.

    Examples
    --------
    >>> syntax_checker = SyntaxChecker(allow=['ast', 'builtin', 'numpy'])
    >>> tree = ast.parse('a=np.cos(0.5)', mode='exec')
    >>> syntax_checker.visit(tree)

    >>> syntax_checker = SyntaxChecker(allow=['ast', 'builtin', 'numpy'])
    >>> tree = ast.parse('a=cos(0.5)', mode='exec')
    >>> syntax_checker.visit(tree)
    Traceback (most recent call last):
        ...
    SyntaxError: cos is not an allowed function!

    """
    #resources that helped in making this:
    #http://stackoverflow.com/questions/1515357/simple-example-of-how-to-use-ast-nodevisitor
    #http://stackoverflow.com/questions/10661079/restricting-pythons-syntax-to-execute-user-code-safely-is-this-a-safe-approach
    #http://stackoverflow.com/questions/12523516/using-ast-and-whitelists-to-make-pythons-eval-safe
    #http://docs.python.org/2/library/ast.html#abstract-grammar
    #http://eli.thegreenplace.net/2009/11/28/python-internals-working-with-python-asts/

    def __init__(self, allow=[]):
        """Initialize a SyntaxChecker object"""

        #super(SyntaxChecker, self).__init__() # not sure if I need this

        self.allowed_functions = dict()
        self.allowed_node_types = dict()
        self.allowed_attributes = set()
        self.safe_names = {'True': True, 'False': False, 'None': None}#dict()
        self.print_each_node = False
        for v in allow:
            s = 'allow_{0}'.format(v)
            if hasattr(self,s):
                getattr(self, s)()
            else:
                raise AttributeError("'SyntaxChecker' object has no attribute "
            "'{0}'. i.e. '{1}' is not a valid member "
            "of the allow list".format(s,v))

    def visit_Call(self, node):
        """Custom visit a 'Call' node"""

        if self.print_each_node:
            print('CALL:', ast.dump(node))

        if hasattr(node.func,'id'):
            if node.func.id not in self.allowed_functions:
                raise SyntaxError("{} is not an allowed function!".format(node.func.id))
            else:
                ast.NodeVisitor.generic_visit(self, node)
        else:
            ast.NodeVisitor.generic_visit(self, node)


    def visit_Name(self, node):
        """Custom visit a 'Name' node"""

        if self.print_each_node:
            print('NAME: ', ast.dump(node))

        if type(node.ctx).__name__=='Store':
            self.allowed_attributes.add(node.id)
        elif type(node.ctx).__name__=='Load':
            if node.id not in self.safe_names and node.id not in self.allowed_functions and node.id not in self.allowed_attributes:
                raise SyntaxError("cannot use {} as name, function, or attribute!".format(node.id))
                sys.exit(0)
        ast.NodeVisitor.generic_visit(self, node)#

    def visit_Attribute(self, node):
        """Custom visit an 'Attribute' node"""

        if self.print_each_node:
            print('ATTRIBUTE:', ast.dump(node))

        if node.attr not in self.allowed_functions and node.attr not in self.allowed_attributes:
            raise SyntaxError("{} is not an allowed attribute!".format(node.attr))
            sys.exit(0)
        ast.NodeVisitor.generic_visit(self, node)

    def generic_visit(self, node):
        """Custom visit a generic node"""
        if self.print_each_node:
            print('GENERIC: ', ast.dump(node))

        if type(node).__name__ not in self.allowed_node_types:
            raise SyntaxError("{} is not allowed!".format(type(node).__name__))
            sys.exit(0)
        else:
            ast.NodeVisitor.generic_visit(self, node)

    def _split_string_allow(self, s, add_to='attributes', fn_module=None):
        """Split a string and add items to allowed lists

        Adds items to self.allowed_attributes or self.allowed_functions

        Parameters
        ----------
        s : str
            String containing items to allow.
        add_to : ['attributes', 'functions']
            If add_to='attributes' items will be added to
            self.allowed_attributes.  If add_to='functions' the items will be
            added to self.allowed_functions.
        fn_module : module
            Module where functions are stored.  Only used when
            add_to='functions'.

        """

        in_list = ['attributes','functions']
        if add_to not in in_list:
            raise ValueError('add_to cannot be {}.  It must be one of '
                '[{}]'.format("'{}'".format(add_to),
                ", ".join(["'{}'".format(v) for v in in_list])))
        if add_to=='functions':
            for v in [a for a in s.split() if not a.startswith('#')]:
                self.allowed_functions[v] = getattr(fn_module, v)
            return
        if add_to=='attributes':
            for v in [a for a in s.split() if not a.startswith('#')]:
                self.allowed_attributes.add(v)

        return

    def allow_ast(self):
        """Allow subset of ast node types"""

        #object_members(ast, 'class')
        s=textwrap.dedent("""\
            #AST Add And Assert Assign Attribute AugAssign AugLoad
            AugStore BinOp BitAnd BitOr BitXor BoolOp Break Call
            #ClassDef Compare Continue Del Delete Dict DictComp Div
            Ellipsis Eq ExceptHandler #Exec Expr Expression ExtSlice
            FloorDiv For FunctionDef GeneratorExp Global Gt GtE If
            IfExp #Import #ImportFrom In Index Interactive Invert Is
            IsNot LShift Lambda List ListComp Load Lt LtE Mod Module
            Mult Name #NodeTransformer #NodeVisitor Not NotEq NotIn Num
            Or Param Pass Pow Print RShift Raise Repr Return Set
            SetComp Slice Store Str Sub Subscript Suite TryExcept
            TryFinally Tuple UAdd USub UnaryOp While With Yield #alias
            arguments boolop #cmpop comprehension #excepthandler #expr
            #expr_context keyword #mod #operator slice #stmt #unaryop
            """)

        for v in [a for a in s.split() if not a.startswith('#')]:
            self.allowed_node_types[v] = getattr(ast, v)

        return

    def allow_builtin(self):
        """Allow subset of __builtins__ functions"""

        #object_members(__builtin__ , 'routine')
        s=textwrap.dedent("""\
            #__import__ abs all any apply bin #callable chr cmp
            coerce compile #debugfile #delattr #dir divmod #eval
            #evalsc #execfile filter format #getattr #globals hasattr
            #hash hex id input #intern isinstance issubclass iter
            len locals map max min next oct #open #open_in_spyder
            ord pow print range raw_input reduce #reload repr
            round #runfile #setattr sorted sum unichr #vars zip
            """)
        self._split_string_allow(s, add_to='functions', fn_module=__builtin__)

        #object_members(__builtin__ , 'class')
        s=textwrap.dedent("""\
            basestring bool #buffer bytearray bytes #classmethod
            complex dict enumerate #file float frozenset int list
            long #memoryview #object #property reversed set slice
            #staticmethod str super tuple type unicode xrange
            """)
        self._split_string_allow(s, add_to='functions',fn_module=__builtin__)

        #object_members(complex , 'routine')
        s=textwrap.dedent("""\
            conjugate
            """)
        self._split_string_allow(s, add_to='attributes')

        #object_members(dict , 'routine')
        s=textwrap.dedent("""\
            copy fromkeys get has_key items iteritems iterkeys
            itervalues keys pop popitem setdefault update values
            viewitems viewkeys viewvalues
            """)
        self._split_string_allow(s, add_to='attributes')

        #object_members(list , 'routine')
        s=textwrap.dedent("""
            append count extend index
            insert pop remove reverse sort
            """)
        self._split_string_allow(s, add_to='attributes')

        #object_members(str , 'routine')
        s=textwrap.dedent("""
            capitalize center count decode encode endswith
            expandtabs find format index isalnum isalpha isdigit
            islower isspace istitle isupper join ljust lower
            lstrip partition replace rfind rindex rjust
            rpartition rsplit rstrip split splitlines startswith
            strip swapcase title translate upper zfill
            """)
        self._split_string_allow(s, add_to='attributes')

        #object_members(float , 'routine')
        s=textwrap.dedent("""
            as_integer_ratio conjugate fromhex hex
            is_integer
            """)
        self._split_string_allow(s, add_to='attributes')

        #object_members(set , 'routine')
        s=textwrap.dedent("""
            add clear copy difference
            difference_update discard intersection
            intersection_update isdisjoint issubset issuperset
            pop remove symmetric_difference
            symmetric_difference_update union update
            """)
        self._split_string_allow(s, add_to='attributes')

        #object_members(slice , 'routine')
        s=textwrap.dedent("""
            indices
            """)
        self._split_string_allow(s, add_to='attributes')

        #object_members(slice , 'routine')
        s=textwrap.dedent("""
            indices
            """)
        self._split_string_allow(s, add_to='attributes')

        return

    def allow_numpy(self):
        """Allow a subset of numpy functionality via np.attribute syntax"""

        self.allowed_functions['np'] = np

        #object_members(np , 'routine')
        s=textwrap.dedent("""
            add_docstring add_newdoc add_newdoc_ufunc alen all
            allclose alltrue alterdot amax amin angle any append
            apply_along_axis apply_over_axes arange argmax argmin
            argsort argwhere around array array2string
            array_equal array_equiv array_repr array_split
            array_str asanyarray asarray asarray_chkfinite
            ascontiguousarray asfarray asfortranarray asmatrix
            asscalar atleast_1d atleast_2d atleast_3d average
            bartlett base_repr bench binary_repr bincount
            blackman bmat broadcast_arrays busday_count
            busday_offset byte_bounds can_cast choose clip
            column_stack common_type compare_chararrays compress
            concatenate convolve copy copyto corrcoef correlate
            count_nonzero cov cross cumprod cumproduct cumsum
            datetime_as_string datetime_data delete deprecate
            deprecate_with_doc diag diag_indices
            diag_indices_from diagflat diagonal diff digitize
            disp dot dsplit dstack ediff1d einsum empty
            empty_like expand_dims extract eye
            fastCopyAndTranspose fill_diagonal find_common_type
            fix flatnonzero fliplr flipud frombuffer fromfile
            fromfunction fromiter frompyfunc fromregex fromstring
            fv genfromtxt get_array_wrap get_include
            get_numarray_include get_printoptions getbuffer
            getbufsize geterr geterrcall geterrobj gradient
            hamming hanning histogram histogram2d histogramdd
            hsplit hstack i0 identity imag in1d indices info
            inner insert int_asbuffer interp intersect1d ipmt irr
            is_busday isclose iscomplex iscomplexobj isfortran
            isneginf isposinf isreal isrealobj isscalar issctype
            issubclass_ issubdtype issubsctype iterable ix_
            kaiser kron lexsort linspace load loads loadtxt
            logspace lookfor mafromtxt mask_indices mat max
            maximum_sctype may_share_memory mean median meshgrid
            min min_scalar_type mintypecode mirr msort nan_to_num
            nanargmax nanargmin nanmax nanmin nansum ndfromtxt
            ndim nested_iters newbuffer nonzero nper npv
            obj2sctype ones ones_like outer packbits pad
            percentile piecewise pkgload place pmt poly polyadd
            polyder polydiv polyfit polyint polymul polysub
            polyval ppmt prod product promote_types ptp put
            putmask pv rank rate ravel ravel_multi_index real
            real_if_close recfromcsv recfromtxt repeat require
            reshape resize restoredot result_type roll rollaxis
            roots rot90 round round_ row_stack safe_eval save
            savetxt savez savez_compressed sctype2char
            searchsorted select set_numeric_ops set_printoptions
            set_string_function setbufsize setdiff1d seterr
            seterrcall seterrobj setxor1d shape show_config sinc
            size sometrue sort sort_complex source split squeeze
            std sum swapaxes take tensordot test tile trace
            transpose trapz tri tril tril_indices
            tril_indices_from trim_zeros triu triu_indices
            triu_indices_from typename union1d unique unpackbits
            unravel_index unwrap vander var vdot vsplit vstack
            where who zeros zeros_like pi
            """)
        self._split_string_allow(s, add_to='attributes')

        #object_members(np.ndarray , 'routine')
        s=textwrap.dedent("""
            all any argmax
            argmin argsort astype byteswap choose clip compress
            conj conjugate copy cumprod cumsum diagonal dot dump
            dumps fill flatten getfield item itemset max mean min
            newbyteorder nonzero prod ptp put ravel repeat
            reshape resize round searchsorted setfield setflags
            sort squeeze std sum swapaxes take tofile tolist
            tostring trace transpose var view
            """)
        self._split_string_allow(s, add_to='attributes')

        self.allowed_attributes.add('math')
        #object_members(np.math,'routine')
        s=textwrap.dedent("""
            acos acosh asin asinh atan atan2 atanh ceil copysign cos cosh
            degrees erf erfc exp expm1 fabs factorial floor fmod frexp fsum
            gamma hypot isinf isnan ldexp lgamma log log10 log1p modf pow
            radians sin sinh sqrt tan tanh trunc
            """)
        self._split_string_allow(s, add_to='attributes')

        #to find more numpy functionality to add try:
        #object_members(np.polynomial,'routine')
        #mem=object_members(np.polynomial,'class')
        #object_members(np.polynomial.polynomial,'routine')


    def allow_PolyLine(self):
        """Allow PolyLine class from geotecha.piecewise.piecewise_linear_1d"""

        self.allowed_functions['PolyLine']=PolyLine

        s=textwrap.dedent("""
            x y xy x1 x2 y1 y2 x1_x2_y1_y2
            """)
        self._split_string_allow(s, add_to='attributes')
        return


def object_members(obj, info='function', join=True):
    """Get a list of object members.

    Parameters
    ----------
    obj : object
        Object to get members of.
    info : string, optional
        Type of members to gather.  Members will be gathered according to
        inspect.is<member_type>. e.g. info='function' will check in object for
        inspect.isfunction. Other values include 'method' 'function'
        'routine'. Default info='function'. NOTE that it is safer to use
        'routine' when getting functions.
    join : bool, optional
        If join==True then list will be joined together into one large
        space separated string.

    Returns
    -------
    members : list of str, or str
        List of member names of specified types, or space separated names
        i single string.


    """
    #useful resources
    #http://www.andreas-dewes.de/en/2012/static-code-analysis-with-python/
    #isroutine vs isfunction: http://stackoverflow.com/a/22428966
    #pkgutil : http://stackoverflow.com/a/1310912

    if info=="module":
        # for some reason getmembers with modules does not allways return a
        # complete list.
        pkgpath = os.path.dirname(obj.__file__)
        members = [name for _, name, _ in pkgutil.iter_modules([pkgpath])]
    else:
        members = [i for i,j in
                    inspect.getmembers(obj,
                                       getattr(inspect,'is{}'.format(info)))]
    if join:
        members='\n'.join(textwrap.wrap(" ".join(members),
                                        break_long_words=False, width=65))
    return members


def make_module_from_text(reader, syntax_checker=None):
    """Make a module from file, StringIO, text etc.

    Parameters
    ----------
    reader : file_like object
        Object to get text from.
    syntax_checker : SyntaxChecker object, optional
        Specifies what syntax is allowed when executing the text
        within `reader`.  Default = None, which means that text will be
        executed with the all powerful all dangerous exec function.  Only use
        this option if you trust the input.

    Returns
    -------
    m : module
        text as module

    See also
    --------
    SyntaxChecker : allow certain syntax

    Notes
    -----
    I suspect it is best if `reader` is a string. i.e reader can be pickled.
    fileobjects may be cause issues if used with multiprocessing.process

    """

    #useful resources:
    #for making module out of strings/files see http://stackoverflow.com/a/7548190/2530083
    #http://old-blog.ooz.ie/2011/03/python-exec-module-in-namespace.html
    #http://stackoverflow.com/questions/7969949/whats-the-difference-between-globals-locals-and-vars
    #http://lucumr.pocoo.org/2011/2/1/exec-in-python/
    #   see down bottom  for config stuff
    #http://stackoverflow.com/a/2372145/2530083
    #   limit namespace


    mymodule = imp.new_module('mymodule') #may need to randomise the name; not sure

    if syntax_checker is None:
        exec reader in mymodule.__dict__
        return mymodule


    if not isinstance(syntax_checker, SyntaxChecker):
        raise TypeError('syntax_checker should be a SyntaxChecker instance.')

    tree = ast.parse(reader, mode='exec')
    syntax_checker.visit(tree)
    compiled = compile(tree, '<string>', "exec")
    mymodule.__dict__.update(syntax_checker.allowed_functions)
    exec compiled in mymodule.__dict__

    return mymodule

class print_all_nodes(ast.NodeVisitor):#http://stackoverflow.com/a/1515403/2530083
    """Simple ast.NodeVisitor sub class that prints each node when visited

    Examples
    --------
    >>> text="a=[3,2]*2"
    >>> x=print_all_nodes()
    >>> tree=ast.parse(text)
    >>> x.visit(tree)
         Module : Module(body=[Assign(targets=[Name(id='a', ctx=Store())], value=BinOp(left=List(elts=[Num(n=3), Num(n=2)], ctx=Load()), op=Mult(), right=Num(n=2)))])
         Assign : Assign(targets=[Name(id='a', ctx=Store())], value=BinOp(left=List(elts=[Num(n=3), Num(n=2)], ctx=Load()), op=Mult(), right=Num(n=2)))
         Name : Name(id='a', ctx=Store())
         Store : Store()
         BinOp : BinOp(left=List(elts=[Num(n=3), Num(n=2)], ctx=Load()), op=Mult(), right=Num(n=2))
         List : List(elts=[Num(n=3), Num(n=2)], ctx=Load())
         Num : Num(n=3)
         Num : Num(n=2)
         Load : Load()
         Mult : Mult()
         Num : Num(n=2)

    """

    #useful resources
    #http://stackoverflow.com/a/1515403/2530083
    def generic_visit(self, node):
        print(" "*4, type(node).__name__,':', ast.dump(node))
        ast.NodeVisitor.generic_visit(self, node)


def fcode_one_large_expr(expr, prepend=None, **settings):
    """Fortran friendly printing of sympy expression ignoring any loops/indexed

    The normal FcodePrinter.doprint method will try to figure out what fortran
    loops you need by looking for indexed expressions.  Sometimes you just want
    to change square brackets [] to parenteses () and wrap/indent the code for
    Fortran with appropriate line continuations. `fcode_one_large_expr`
    can do that for you. You will have to write your own fortran do loops.

    Parameters
    ----------
    expr : sympy expression
        A single large sympy expression.
    prepend : string, optional
        A string to prepend to your fortran code.  Assuming you are
        going to cut and paste into a fortran routine `prepend` should be
        correct fortran format.  (note you do not need an initial indent
        for your prepend it will be put in for you). e.g.
        prepend = 'a(i, i) = a(i, i) + '. Defualt prepend=None i.e. nothing
        prepended.
    settings : keyword arguments
        See `sympy.printing.fcode` docs, specifically FCodePrinter.__init__ .
        Note that 'assign_to will not work'.

    Returns
    -------
    out : str
        Fortran ready code that can be copy and pasted into a Fortran routine.

    See also
    --------
    sympy.printing.fcode : contains all the functionality.

    """


    printer = FCodePrinter(settings)

    if printer._settings['source_format'] == 'fixed':
        #FCodePrinter.indent_code uses ''.join to combine lines.  Should it be
        # '\n'.join ? This is my work around:
        printer._lead_cont = '&\n      ' #+ printer._lead_cont
    expr = printer.parenthesize(expr, 50)

    if not prepend is None:
        expr = prepend + expr

    return printer.indent_code(expr)


def copy_attributes_between_objects(from_object,
                                    to_object,
                                    attributes=[],
                                    defaults=dict(),
                                    not_found_value=None):
    """Copy attributes from one object to another

    Looks up `attributes` in `from_obj` and copies them to `to_object`.  If
    not present in `from_object` the corresponding attribute in `to_object`
    will be set to `defaults` or `not_found_value` if not in `defaults`.

    Parameters
    ----------
    from_object : object or module
        Object to copy attributes from.
    to_object : object or module
        Object to copy attributes to.
    attributes : list of strings
        A list of attribute names to copy from `from_object` to `object_to`.
    defaults : dict of string-value pairs
        Dictionary specifying the default value of each attribute that is not
        found in `from_object`.
    not_found_value : optional
        Default value to set attribute to if not found in `from_object` or in
        `defaults`.  Default not_found_value=None.

    Returns
    -------
    None

    """


    for v in attributes:
            to_object.__setattr__(v, from_object.__dict__.get(v, defaults.get(v, not_found_value)))

    return

def copy_attributes_from_text_to_object(reader, *args, **kwargs):
    """Wrapper for `copy_attributes_between_objects` where `from_object`
    is fileobject, StringIO, text

    Parameters
    ----------
    reader : fileobject, StringIO, text
        text to turn into a module and pass as from_object
    *args :
        Positional arguments that will be passed to
        `copy_attributes_between_objects`.
    **kwargs :
        Keyword arguments that will be passed to
        `copy_attributes_between_objects`. You can use a SyntaxChecker
        object to interpret the `reader` by using the keyword argument
        'syntax_checker'=<some SyntaxChecker object>.

    See also
    --------
    copy_attributes_between_objects : See for args and kwargs input.
    SyntaxChecker : Restrict the allowable syntax in the `reader`.

    Notes
    -----
    Put 'syntax_checker= ' in the kwargs to add a SyntaxChecker


    """

    syn_checker=kwargs.pop('syntax_checker', None)
    copy_attributes_between_objects(make_module_from_text(reader,
            syn_checker), *args, **kwargs)
    return


def check_attribute_is_list(obj, attributes=[], force_list=False):
    """Check if attributes of an object are lists

    If attribute is not a list and `force_list`=True then attribute will be
    placed in list.  e.g. say obj.a = 6, 6 is not a list so obj.a will be
    changed to obj.a = [6].  An error is raised if attribute is not a list.

    Parameters
    ----------
    obj : object
        Object with attributes to check.
    attributes : list of strings
        A list of attribute names to check.
    force_list : bool, optional
        If True not-list attributes will be put in a list. If False then
        non-list attributes will raise an error. Default force_list=False.

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
    """Check if attributes of object that are PolyLine have the same x values

    Each attribute can be a single instance of PolyLine or a list of PolyLines.
    An error is raised if x-values are different.

    Parameters
    ----------
    obj : object
        Object with attributes to check.
    attributes : list of list of strings
        Each sublist is a list of attribute names to check.

    See Also
    --------
    geotecha.piecewise.piecewise_linear_1d.PolyLine : PolyLine class

    """

    g = obj.__getattribute__

    for sublist in attributes:
        #find first x values
        for v in sublist:
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
        for v in sublist:
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
    """Check if attribute pairs have the same length

    Compares pairs only if both pairs are not None.  Raises error if pair items
    have unequal lenghth.

    Parameters
    ----------
    obj : object
        Object with attributes to check.
    attributes : list of list of two string
        List of attribute names to check. Each sub-list should have two
        elements.

    """

    g = obj.__getattribute__

    # for iterating in chuncks see http://stackoverflow.com/a/434328/2530083
    #for v1, v2 in [attributes[pos:pos + 2] for pos in xrange(0, len(attributes), 2)]:

    for pair in attributes:
        if len(pair)>2:
            raise ValueError('Can only compare two items. you have {}'.format(str(pair)))
        v1, v2 = pair
        if sum([v is None for v in [g(v1), g(v2)]])==0:
            if sum([hasattr(v,'__len__') for v in [g(v1), g(v2)]])!=2:
                raise TypeError("either {0} and {1} have no len attribute and so cannot be compared".format(v1,v2))
            if len(g(v1)) != len(g(v2)):
                raise ValueError("{} has {:d} elements, {} has {:d} "
                    "elements.  They should have the same number of "
                    "elements.".format(v1, len(g(v1)), v2, len(g(v2))))

#        elif sum([v is None for v in [g(v1), g(v2)]])==1:
#            raise ValueError("Cannot compare length of {0} and {1} when one of them is None".format(v1,v2))
    return


def check_attribute_combinations(obj, zero_or_all=[], at_least_one=[], one_implies_others=[]):
    """Check for incomplete combinations of attributes

    Raises ValueError if any combination fails.

    Parameters
    ----------
    zero_or_all : list of list of string
        Each element of `zero_or_all` is a list of attribute names that should
        either all be None or all be not None.
    at_least_one : list of list of string
        Each element of `at_least_one` is a list of attribute names of which at
        least one should not be None.
    one_implies_others : list of list of string
        Each element of `one_implies_others` is a list of attribute names.  If
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



def force_attribute_same_len_if_none(obj, same_len_attributes=[], value=None):
    """Make list of None with len the same as an existing attribute

    If attributes after the first are None then make those attributes a list
    of len(first_attribute) filled with value.  This is useful when you want
    to zip each member of two attributes. If the second attribute is None,
    then any zipping is impossible.  However, if the second attribute
    is set to a list of None then zipping is possible.

    Parameters
    ----------
    obj : object
        Object that has the attributes.
    same_len_attributes : list of list of string
        For each group of attribute names, if attributes after the first are
        None then those attributes will be made a list of len(first_attribute)
        filled with `value`.
    value : obj, optional
        Value to fill each list.  Default value=None.


    """

    for group in same_len_attributes:
        a1 =getattr(obj, group[0])
        if a1!=None:
            for v in group[1:]:
                if getattr(obj, v) is None:
                    setattr(obj, v, [value] * len(a1))



def initialize_objects_attributes(obj,
                                  attributes=[],
                                  defaults=dict(),
                                  not_found_value=None):
    """Initialize an objects attributes

    For each attribute, set it to the value found in `defaults` dictionary
    or, if the attribute is not found, set it to `not_found_value`.

    Parameters
    ----------
    obj : object
        Object to set attributes in.
    attributes : list of strings
        A list of attribute names to set.
    defaults : dict of string-value pairs, optional
        Dictionary specifying the default value of each attribute.
        Default defautls=dict() i.e. no explicit default value.
    not_found_value : optional
        Default value to set attribute to if not found in `defaults` dict.
        Default not_found_value=None.



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
    code_for_explicit_attribute_initialization : Use for temporary explicit
        attribute initialization to facilitate auto-complete, then comment
        out when done.

    """

    for v in attributes:
        obj.__setattr__(v, defaults.get(v, not_found_value))
    return

def code_for_explicit_attribute_initialization(attributes=[],
                                               defaults={},
                                               defaults_name='_attribute_defaults',
                                               object_name='self',
                                               not_found_value=None):
    """Generate code to initialize an objects attributes

    Parameters
    ----------
    object_name : string , optional
        Name of object to set attributes. default='self'.
    attributes : list of strings
        A list of attribute names to set.
    defaults : dict of string-value pairs
        Dictionary specifying the default value of each attribute.
    defaults_name : string, optional
        If ``None`` then the actual values in `defaults` will be used
        to set attributes.  If not ``None`` then those attributes with
        defaults will be initialized by pointing to a dictionary called
        `defaults_name`.  Default defaults_name='_attribute_defaults',
    not_found_value : optional
        Default value to set attribute to if not found in `defaults`.
        default not_found_value=``None``

    Returns
    -------
    out : string
        Code that can be pasted into an object (usually in the __init__
        methos) to explicitly initialize attributes, so the attributes appear
        in autocomplete.

    See also
    --------
    initialize_objects_attributes : Similar functionality with no copy paste.

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


class InputFileLoaderCheckerSaver(object):
    """A class for accepting an input file, running checks, saveing data/plots.

    Options to:

     - Accept an inputfile with python syntax
     - Copy attribute value from a text file (i.e. text such as 'b = 34' in
       the input file will be result in self.b=34)
     - Run checks on the attributes.
     - Save data to file
     - Save figures to file.
     - Common attributes along with their defaults are available to all
       classes that sub-class InputFileLoaderCheckerSaver.


    Parameters
    ----------
    reader : file_like object
        Object to get text from. file, StringIO, text etc.
    pkg_for_version : string, optional
        The version attribute will be determined by the version number of
        the `pkg_for_version` package.  Default pkg_for_version='geotecha'.
        If for some reason pkg_for_version has not been set, or the package
        cannot be found, then version will be 'unknown'.


    Attributes
    ----------
    _attributes : list of string
        User defined list of attributes. Must be defined in the sub-class.
        A number of common attributes will be added to the user defined list.
        See the Notes section for the common attributes and their default
        values.
    _attribute_defaults : dict, optional
        User defined dict of attribute defaults. May be defined in the
        sub-class. Defaults for the common attributes will will be added
        to the user defined list.
        When no default value is given , None will be used.
        Default _attribute_defaults=dict() i.e. all user defaults are None.
        See the Notes section for the common attributes and their default
        values.
    _attributes_that_should_be_lists : list of string, optional
        list of attributes that should be of type ``list``. Those attributes
        that are not None and not lists will be put in a list.
        May be defined in the sub-class.
    _attributes_that_should_have_same_x_limits : list of list of str, optional
        Each sublist is a list of attribute names to check.
        May be defined in the sub-class.
    _attributes_that_should_have_same_len_pairs : list of list of two string
        List of attribute names to check. Each sub-list should have two
        elements.
        May be defined in the sub-class.
    _zero_or_all : list of list of string
        Each element of `zero_or_all` is a list of attribute names that should
        either all be None or all be not None.
        May be defined in the sub-class.
    _at_least_one : list of list of string
        Each element of `at_least_one` is a list of attribute names of which at
        least one should not be None.
        May be defined in the sub-class.
    _one_implies_others : list of list of string
        Each element of `one_implies_others` is a list of attribute names.  If
        the first attribute in the list is not None then all other members of
        the list should also not be None.
        May be defined in the sub-class.
    _input_text : string
        str of the input file.  Will be None if no reader is passed to the
        __init__ method
    _debug : True/False
        For use with debugging. Default _debug=False
    _figures : list of matplotlib figures
        Define in sub-class. Each figure should have a label which
        will be used to save figures.
    _grid_data_dicts : list of dict
        Define in sub-class. For saving grid_data to file
        using `save_grid_data_to_file`.
    _file_stem: string
        path including file name prefix with number for all output files.
        default='out_000'.  If
        create_directory=True then files will be put in a folder named
        `file_stem`


    Notes
    -----
    For the _attributes and _attribute_defaults  attribute the common
    attributes and their defualts are:

    =========================== ========================================
    attribute name              description
    =========================== ========================================
    plot_properties             dict of dict.  Each subdict will be
                                used to pass propeties to plotting
                                routines.  Contents plot_properties
                                should be defined in each sub-class
                                Default = dict().
    save_data_to_file           True/False. If True data will be
                                saved to file.
                                Need to write routines to save data
                                Default = False.
    save_figures_to_file        True/False. If True then figures will
                                be saved to file.  Need to write
                                routine to plot figures.
                                Default = False.
    show_figures'              True/False. If True the after
                                calculation figures will be shown
                                on screen. Need plt.show somewhere in
                                routines.
                                Default = False.
    prefix                      string. Filename prefix for all
                                output files. Default = 'out'.
    directory                   string. Path to directory where files
                                should be stored.
                                Default directory=None which
                                will use the current working directory.
                                Note if you keep getting
                                directory does not exist errors then
                                try putting an r before the string
                                definition. i.e.
                                directory = r'C:\\Users\\...'.
    overwrite                   True/False. If True then existing
                                files will be overwritten.
                                Default = False.
    create_directory            True/Fase. If True a new sub-folder
                                with name based on  `prefix` and an
                                incremented number will contain the
                                output. Default = True.
    data_ext                    string. File extension for data files.
                                Default = '.csv'.
    input_ext                   string. File extension for original and
                                parsed input files. Default = '.py'.
    figure_ext                  string. File extension for figures.
                                Can be any valid matplotlib option for
                                savefig. Default = ".eps". Others
                                include 'pdf', 'png'.
    author                      string. Author of analysis.
                                Default = 'unknown'
    title                       string. A title for the input file.
                                This will appear at the top of data
                                files. Default = None, i.e. no title.
    version                     Default = version number from
                                `pkg_for_version paramter`
    =========================== ========================================



    """

    def __init__(self, reader = None, pkg_for_version='geotecha'):

        self._debug = False
        self._setup()

        self.pkg_for_version = pkg_for_version
        if hasattr(self, '_attributes'):
            self._initialize_attributes()


        self._input_text = None
        if not reader is None:
            if isinstance(reader, str):
                self._input_text = reader
            else:
                self._input_text = reader.read()

            self._transfer_attributes_from_inputfile()



    def _transfer_attributes_from_inputfile(self):
        """Transfer attributes from input file using restricted syntax.

        For safety reasons the inputfile syntax is restricted.  If you wish
        to remove these restrictions then overwrite this method with
        syn_checker=None.  You will then have to do explicit imports.
        Existing input files used when syntax_checker was not None may not
        work because some modules (e.g. Numpy) were implicitly imported,
        'import' commands are often banned in default syntax_checker.

        """

        if not hasattr(self, '_attributes'):
            raise ValueError("No 'self._attributes' defined in object.")

        attribute_defaults = getattr(self, '_attribute_defaults', dict())

        syn_checker=SyntaxChecker(['ast','builtin','numpy','PolyLine'])
#        syntax_checker=None


        copy_attributes_from_text_to_object(self._input_text,
            self,
            self._attributes, attribute_defaults,
            not_found_value = None, syntax_checker=syn_checker)


    def _initialize_attributes(self):
        """Initialize attributes

        Relies on presence of self._attributes
        """

        if not hasattr(self, '_attributes'):
            raise ValueError("No 'self._attributes' defined in object.")

        common_attributes = ('plot_properties '
            'save_data_to_file prefix directory overwrite '
            'create_directory data_ext input_ext figure_ext '
            'save_figures_to_file show_figures '
            'author '
            'version '
            'title').split()

        try:
            version_str = pkg_resources.require(self.pkg_for_version)[0].version
        except (pkg_resources.DistributionNotFound, AttributeError):
            version_str = 'unknown'

        common_attribute_defaults = {
            'plot_properties': dict(),
            'save_data_to_file': False,
            'save_figures_to_file': False,
            'show_figures': False,
            'prefix': 'out',
            'overwrite': False,
            'create_directory': True,
            'data_ext': '.csv',
            'input_ext': '.py',
            'figure_ext': '.eps',
            'author': 'unknown',
            'version': version_str}

        attribute_defaults = getattr(self, '_attribute_defaults', dict())


        [self._attributes.append(v) for v in common_attributes if
                v not in self._attributes]

        self._attribute_defaults = copy_dict(common_attribute_defaults,
                                        attribute_defaults)


        initialize_objects_attributes(self,
                                      self._attributes,
                                      self._attribute_defaults,
                                      not_found_value = None)

    def _determine_output_stem(self):
        """Determine the stem for outputting files"""
        prefix = getattr(self, 'prefix', None)
        if prefix is None:
            prefix='out'
        directory = getattr(self, 'directory', None)
        if directory is None:
            directory = os.curdir
        overwrite = getattr(self, 'overwrite', None)
        if overwrite is None:
            overwrite = False

        self._file_stem = next_output_stem(
            prefix=prefix, path=directory, start=1, inc=1, zfill=4,
                     overwrite=overwrite)

        create_directory = getattr(self, 'create_directory', True)
        if create_directory:
            self._file_stem = os.path.join(directory,
                                           self._file_stem,
                                           self._file_stem)
        else:
            self._file_stem = os.path.join(directory,
                                           self._file_stem)
        if not os.path.isdir(os.path.dirname(self._file_stem)):
            os.mkdir(os.path.dirname(self._file_stem))

    def _save_figures(self):
        """
        You need the following defined:

         - self._figures: a list of figures to save.  Each figure should
           have a label


        """
        if not getattr(self, 'save_figures_to_file', False):
            return

        if getattr(self, '_figures', None) is None:
            return

        if getattr(self, '_file_stem', None) is None:
            self._determine_output_stem()

        if not isinstance(self._figures, list):
            figures = [self._figures]
        else:
            figures = self._figures

#        directory = getattr(self, 'directory', None)
#        if directory is None:
#            directory = os.curdir
#
#        create_directory = getattr(self, 'create_directory', True)


        for i,fig in enumerate(self._figures):
            if len(fig.get_label()) > 0:
                figname = fig.get_label()
            else:
                figname = "Figure_{}".format(str(i).zfill(2))
            filename = self._file_stem +'_' + figname + getattr(self, 'figure_ext',
                                                               '.eps')
            fig.savefig(filename, dpi=fig.dpi, bbox_inches='tight')

    def _save_data(self):

        if not getattr(self, 'save_data_to_file', False):
            return

        if getattr(self, '_file_stem', None) is None:
            self._determine_output_stem()

        #original input
        if not getattr(self, '_input_text', None) is None:
            filename = (self._file_stem + "_input_original" +
                getattr(self, 'input_ext', '.py'))

            with open(filename,'w') as myfile:
                myfile.write(self._input_text)

        #parsed input
        if hasattr(self, '_attributes'):
            filename = (self._file_stem + "_input_parsed" +
                    getattr(self, 'input_ext', '.py'))
            text = string_of_object_attributes(obj=self,
                            attributes=self._attributes, none_at_bottom=True,
                                        numpy_array_prefix = "np.")
            with open(filename, 'w') as myfile:
                myfile.write(text)


        #grid data
        if not getattr(self, '_grid_data_dicts', None) is None: #no data
            save_grid_data_to_file(data_dicts=self._grid_data_dicts,
                   directory=os.path.dirname(self._file_stem),
                   file_stem=os.path.basename(self._file_stem),
                   create_directory=False,# already in file stem getattr(self, 'create_directory', True),
                   ext=getattr(self, 'data_ext', '.csv'))




    def _setup(self):
        """Set up attributes for checking and initializing

        To be overridden in subclasses

        Basically define:
         - self._attribute_defaults
         - self._attributes
         - self._attributes_that_should_be_lists
         - self._attributes_to_force_same_len
         - self._attributes_that_should_have_same_x_limits
         - self._attributes_that_should_have_same_len_pairs
         - self._zero_or_all
         - self._at_least_one
         - self._one_implies_others

        To make coding easier (i.e. autocomplete) once you have a
        self.attributes defined,
        `use code_for_explicit_attribute_initialization`
        and paste the resulting code into your `_setup` routine

        """

        pass
        return

    def check_input_attributes(self):
        """Perform checks on attributes

        Notes
        -----

        See also
        --------
        check_attribute_combinations
        check_attribute_is_list
        force_attribute_same_len_if_none
        check_attribute_PolyLines_have_same_x_limits
        check_attribute_pairs_have_equal_length

        """


        zero_or_all = getattr(self, '_zero_or_all', [])
        at_least_one = getattr(self, '_at_least_one', [])
        one_implies_others = getattr(self, '_one_implies_others', [])
        attributes_that_should_be_lists = getattr(self,
            '_attributes_that_should_be_lists',[])
        attributes_to_force_same_len = getattr(self,
             '_attributes_to_force_same_len', [])
        attributes_that_should_have_same_x_limits = getattr(self,
            '_attributes_that_should_have_same_x_limits', [])
        attributes_that_should_have_same_len_pairs = getattr(self,
            '_attributes_that_should_have_same_len_pairs', [])




        check_attribute_combinations(self,
                                     zero_or_all,
                                     at_least_one,
                                     one_implies_others)

        check_attribute_is_list(self,
                            attributes_that_should_be_lists, force_list=True)

        force_attribute_same_len_if_none(self,
                                     attributes_to_force_same_len, value=None)

        check_attribute_PolyLines_have_same_x_limits(self,
                         attributes=attributes_that_should_have_same_x_limits)
        check_attribute_pairs_have_equal_length(self,
                        attributes=attributes_that_should_have_same_len_pairs)

        return





def string_of_object_attributes(obj, attributes=[], none_at_bottom=True,
                                    numpy_array_prefix = "np."):
    """Put repr of objects attributes into one string

    The repr of each attribute will be put in a line within the output string.
    If the attribute is not found then None will be given.

    Parameters
    ----------
    obj : object
        Object containing attributes.
    attributes : list of string, optional
        List of attribute names.
    none_at_bottom : True/False
        If True then any attribute that is None will be placed at the end
        of the output string.
    numpy_array_prefix : string, optional
        repr of numpy arrays by default do not print with anything in front of
        'array'.  with `numpy_array_prefix` wou can add a 'np.' etc.  default=
        'np.' Use `None` for no prefix.

    Returns
    -------
    out : str
        String

    """

    def f(v):
        if isinstance(v, str):
            return "'"
        else:
            return ""

    if not numpy_array_prefix is None:
        a = PrefixNumpyArrayString(numpy_array_prefix)
    else:
        a = PrefixNumpyArrayString('')
    a.turn_on()



    writer = ""
    if none_at_bottom:
        writer+='\n'.join(['{0} = {1}{2}{1}'.format(v, f(getattr(obj, v, None)),
            getattr(obj, v, None))
            for v in attributes if not getattr(obj, v, None) is None])
        writer+='\n\n\n'
        writer+='\n'.join(['{0} = {1}{2}{1}'.format(v, f(getattr(obj, v, None)),
            getattr(obj, v, None))
            for v in attributes if getattr(obj, v, None) is None])
    else:
        writer+='\n'.join(['{0} = {1}{2}{1}'.format(v, f(getattr(obj, v, None)),
            getattr(obj, v, None))
            for v in attributes])

    writer+='\n'
    a.turn_off()
    return writer



class PrefixNumpyArrayString(object):
    """When printing a numpy array string prefix 'array'

    If you use `repr` to print a numpy array it will print as 'array([...])'.
    If you have imported numpy with 'import numpy.array as array' or some such
    import then you can paste the printed output straight back into Python.
    However, I use 'import numpy as np' so I have to manually put in the 'np.'
    before I can use the printed output in my code.  Using
    PrefixNumpyArrayString can put in the 'np.' automatically when using the
    __str__ of the numpy array. i.e. __repr__ remains unchanged but __str__
    will have the prefix

    Attributes
    ----------
    prefix : string, optional
        string to prefix with. Default prefix='np.'.


    Examples
    --------

    >>> print(np.arange(3))
    [0 1 2]
    >>> a=PrefixNumpyArrayString('numpy.')
    >>> a.turn_on()
    >>> print(np.arange(3))
    numpy.array([             0,              1,              2])
    >>> a.turn_off()
    >>> print(np.arange(3))
    [0 1 2]

    """

    def __init__(self, prefix = 'np.'):

        self.prefix = prefix

    def _pprint(self, arr):
        """Function to pass to np.set_string_function"""
        return '{}{}'.format(self.prefix, repr(arr))

    def turn_on(self):
        """Turn_on printing numpy array with prefix"""
        np.set_string_function(self._pprint, False)

    def turn_off(self):
        """Turn_on printing numpy array with prefix"""
        np.set_string_function(None, False)


def next_output_stem(prefix, path=None, start=1, inc=1, zfill=3,
                     overwrite=False):
    """Find next unique prefix-number in sequence of numbered files/folders

    Looks in folder of `path` for files/folders with prefix-number combos
    of the form 'prefixdddxyz.ext'.  If found
    then the ddd is incremented by `inc` and the new 'prefixddd' is returned.
    E.g. if file 'prefix004_exact.out' exists you will get 'prefix_005' as
    the next output stem.

    Parameters
    ----------
    prefix : string
        Start of file/folder name to search for.
    path : string, optional
        Folder to seach in.  Default path=None i.e. search in current working
        directory.
    start : int, optional
        If no existing files/folders match the numbering will start at `start`.
        Default start=1.
    inc : int, optional
        If matching files/folders are found then the number will increment
        by inc. Default inc=1.
    zfill : int, optional
        Fill number with zeros to the left. eg. if `zfill` is 3 and the number
        is 8, then the number will be output as '008'.
    overwrite : True/False
        When True the prefix-number combo will not be incremented and the
        highest exisiting prefix-number combo will be returned.

    Returns
    -------
    stem : string
        Next output stem.

    """


    if path is None:
        cwd = os.curdir
    else:
        if not os.path.isdir(path):
            raise ValueError('folder does not exist: ' + path)
        cwd = path


    pattern = re.compile(prefix + "(?P<x>\d+).*")

    nums = []
    for nm in os.listdir(cwd):
        match = pattern.match(nm)
        if match:
            nums.append(int(match.group('x')))
    if len(nums)>0:

        num = max(nums)
        if not overwrite:
            num += inc
    else:
        num = start

    return prefix+""+str(num).zfill(zfill)


def make_array_into_dataframe(data, column_labels=None, row_labels=None,
                                row_labels_label='item'):
    """Make an array into a pandas dataframe

    The data frame will have no index. Use df.set_index().

    Parameters
    ----------
    data : 2d array
        Data to be put int
    column_labels :  list or 1d array, optional
        Column lables for data. Default column_labels=None which will give
        column numbers 0,1,2, etc.
    row_labels : list or 1d array, optional
        Default row_labels=None which will use row numbers, 0,1,2,etc.
    row_labels_label : string, optional
        Column label for the row_labels column.  Default
        row_labels_label='item'.

    Returns
    -------
    df : pandas dataframe
        dDtaframe of data with column and row labels added


    """

    if isinstance(data, pd.DataFrame):
        return data

    if column_labels is None:
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame(data=data, columns=column_labels)

    if not row_labels is None:
        s = pd.Series(row_labels)
        df.insert(loc=0, column=row_labels_label, value = row_labels)

    return df





def save_grid_data_to_file(data_dicts, directory=None, file_stem='out_000',
                           create_directory=True, ext='.csv'):
    """Save grid data to files using a common filename stem

    Saves grid data in `directory`.  Each data filename will begin with
    `file_stem`.  Grid data is basically 2d data with rows and columns.


    Parameters
    ----------
    data_dicts : list of dict or single dict
        Each data_dict contains info for outputing a piece of data to a file.
        Although the data_dicts parameter is listed last, each data_dict
        should appear at the start of the argument list when calling the
        function.

        ==================  ============================================
        data_dict key       description
        ==================  ============================================
        name                String to be added to end of stem to create
                            file_name.
        data                array of data to save to file.  If data
                            is a pandas dataframe then row_labels and
                            column labels will be ignored (i.e. it will
                            be assumed that the dataframe already has
                            row labels and column labels).
        row_labels          List of row labels for data. default = None
                            which will give no row labels.
        row_labels_label    If there are row_labels then the column
                            label for the row_label column.
                            Default = None which will give 'item'.
        column_labels       Column lables for data.
                            Default = None which will give nothing.
        header              String to appear before data.
                            Default = None.
        df_kwargs           dict of kwargs to pass to
                            pandas.DataFrame.to_csv(). Default ={}.
        ==================  ============================================
    directory : string, optional
        Path of directory to save data in.  Default directory=None which will
        save to the current working directory.
    file_stem : string, optional
        beginning of file name, Default file_stem='out_000'. Filename will be
        made from `file_stem` + [`data_dict`['name']] + `ext`.
    ext : string, optional
        File extension.  Default ext=".csv".
    create_directory : True/False, optional
        If True, all data files will be placed in a directory named file_stem.
        So you might get files
        path_to_directory\\out_000\\out_000_cow.csv etc.
        If False, all data files will be placed in `directory`.  So you would
        get files path_to_directory\\out_000_cow.csv etc. Default
        create_directory=True.


    """

    if not isinstance(data_dicts, list):
        data_dicts=[data_dicts]

    if len(data_dicts)==0:
        return


    if not all([isinstance(v, dict) for v in data_dicts]):
        raise ValueError('all elements of data_dicts must be a dict.')

    if directory is None:
        directory = os.path.curdir
    if not os.path.isdir(directory):
        os.mkdir(directory)


    if create_directory:
        dirname = os.path.join(directory, file_stem)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
    else:
        dirname = directory


    for d in data_dicts:
        data = d.pop('data', None)
        name = d.pop('name',"")
        row_labels = d.pop('row_labels', None)
        row_labels_label = d.pop('row_labels_label', 'item')
        column_labels = d.pop('column_labels', None)
        header = d.pop('header', None)
        df_kwargs = d.pop('df_kwargs',{})

        if data is None:
            continue
        df = make_array_into_dataframe(data=data,
                column_labels=column_labels, row_labels=row_labels,
                row_labels_label=row_labels_label)


        filename = os.path.join(dirname, str(file_stem) + str(name) + ext )
        with open(filename,'w') as myfile:
            if not header is None:
                myfile.write(header+'\n')
            df.to_csv(myfile, **df_kwargs)



def hms_string(sec_elapsed):
    """Convert seconds to human readable h:min:seconds

    Parameters
    ----------
    sec_elapsed : float
        Elapsed seconds

    References
    ----------
    Copied verbatim from http://arcpy.wordpress.com/2012/04/20/146/

    """

    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


class GenericInputFileArgParser(object):
    """Pass input files to a class/function, then call methods on the object

    see self.main for use in scripts.

    The working directory will be changed to the location of each input file

    Basically the path of each input file is passed to the method self.process.


    Parameters
    ----------
    obj : object/callable
        Object to call with input file as argument.
    methods : sequence of 3 element tuples, optional
        Each element of methods is a 3 element tuple (method_name, args,
        kwargs). After `obj` has been initialized the obj's method called
        'method_name' will be called with unpacked args and kwargs. i.e.
        obj(filename).method_name(*args, **kwargs). Default methods=[] i.e.
        no methods called.
    pass_open_file : True/False, optional
        If True then input files will be passed to obj as open file objects.
        If False then paths of the input files will be passed to obj.
        Default pass_open_file=True
    prog : string, optional
        Part of usage method for argparse, the name of the program.
        Default prog=None which uses sys.argv[0] i.e. the first command line
        paramter.
    description : string, optional
        Part of usage method for argparse.
        Default description="Process some input files".
    epilog : string, optional
        Part of usage method for argparse. Default epilog='If no options are
        specifyied then the user will be prompted to select files.'



    Notes
    -----
    For basic usage, in a file to be used as a script put::

        if __name__=='__main__':
            a = GenericInputFileArgParser(obj_to_call)
            a.main()


    See Also
    --------
    argparse : better description of some of the variables.

    """

    def __init__(self, obj,
                 pass_open_file=True,
                 methods=[],
                 prog = None,
                 description = "Process some input files",
                 epilog = ('If no options are specifyied then the user '
                           'will be prompted to select files.')):


        self.obj = obj
        self.pass_open_file = pass_open_file
        self.methods = methods
        self._prog = prog
        self._description = description
        self._epilog = epilog

    def process(self, path):
        """Process the input file path

        The default behaviour of self.process depends on `self.pass_open_file`.
        If True then `path` will be opened and then self.obj will be called
        with the opened file as the argument. If False then then `self.obj`
        will be called with the path itself as argument.

        If you need more complicated behaviour to process your input file
        (e.g. initalize self.obj with a path and then call one of it's methods)
        then over write this method.

        Parameters
        ----------
        path : string
            Absolute path to input file.

        """

        if self.pass_open_file:
            with open(path,'r') as f:
                a = self.obj(f)
        else:
            a = self.obj(path)

        for s, args, kwargs in self.methods:
            getattr(a, s)(*args, **kwargs)


    def main(self, argv=None):
        """Accept command line arguments and pass files/paths to self.obj

        Allows users to specify files via the command line in various ways:

        1. If no options are specifyied a dialog box will open for the user
        to select files.
        2. A list of file paths can be specified with the --filename option
        3. The --directory and --pattern options can be used together to
        search a particular directory for files matching a certain pattern.
        If no actual directory is specified then the current working directory
        will be used.  The default pattern to look for is "*.py".  The
        script file itself will not be processed if it matches the pattern
        itself.

        Parameters
        ----------
        argv : list of str, optional
            List of command line arguments. Default argv=None which will grab
            arguments from sys.argv.  Used mainly for testing or bypassing
            script based command line arguments.


        """

        parser = argparse.ArgumentParser(prog=self._prog,
                                         description=self._description,
                                         epilog = self._epilog)

        parser.add_argument('-f', '--filename', type=str,#argparse.FileType('r'),
                            nargs='+',
                            help="Path(s) to input file(s).")

        parser.add_argument('-d', '--directory', type=str,
                            nargs='*',
                            help="Path(s) to directory containing input "
                            "files. Files in specified directory(s) that "
                            "match the pattern given with the --pattern "
                            "flag will be used.  If the --directory flag is "
                            "used without specifying a directory then the "
                            "current working directroy will be used.")

        parser.add_argument('-p', '--pattern', type=str,
                            nargs='?', const="*.py",
                            help="pattern to match files in the "
                            "directory specified using the --directory flag. "
                            "If the --pattern flag is used without "
                            "specifying a pattern then '*.py' will be used.")




        if argv == None:
            ns = parser.parse_args()
        else:
            ns = parser.parse_args(argv)



        if len([x for x in (ns.pattern, ns.directory) if x is not None]) == 1:
            parser.error('--directory and --pattern must be given together. ')


        if len([x for x in (ns.filename, ns.directory)
                    if x is not None]) == 2:
            parser.error('--filenames and --directory cannot be used '
                         'together.')


        ns.directory = [os.getcwd()] if ns.directory==[] else ns.directory

        if (not ns.directory is None) and (not ns.pattern is None):
            #find all files that match pattern
            filenames=[]
            for i,d in enumerate(ns.directory):
                if not os.path.isdir(d):
                    parser.error('directory #{0} in command line is not a '
                    'valid directory: {1}'.format(i, d))

                [filenames.append(os.path.join(os.path.abspath(d), v)) for v in
                    fnmatch.filter(os.listdir(d), ns.pattern) if v!=os.path.basename(__file__)]

            if len(filenames)==0:
                raise IOError("no files that match '{0}' can be found in the "
                              "specified directories".format(ns.pattern))

        elif not ns.filename is None:
            filenames = [os.path.abspath(v) for v in ns.filename]

        else:
            #prompt for files
            filenames = get_filepaths(wildcard="")

        if filenames is None:
            return

        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.INFO)
        logging.info(time.strftime("%Y-%m-%d %H:%M:%S") +
            ' Start processing files')
        start_time= time.time()
        for i, path in enumerate(filenames):

            try:

#                time.sleep(0.5)
                if i>0:
                    time_remaining = ((time.time() - start_time)/(i+1) *
                                        (len(filenames)-i))

                    time_remaining = (', estimated. time remaining ' +
                                    hms_string(time_remaining))
                else:
                    time_remaining= "."
                logging.info("{0}, processing {1}, File {2} of {3}{4}".format(
                                time.strftime("%Y-%m-%d %H:%M:%S"),
                                path,
                                i+1,
                                len(filenames),
                                time_remaining))

                with working_directory(os.path.dirname(path)):

                    self.process(path)

            except Exception:
                logging.exception("problem with file {0}:\n".format(path))
            finally:

                pass
        logging.info(time.strftime("%Y-%m-%d %H:%M:%S") +
            ' End processing files')
        logging.getLogger().setLevel(level)







@contextmanager
def working_directory(path):
    """Change working directory for duration of with statement

    Parameters
    ----------
    path : str
        Path to new current working directory

    Notes
    -----
    e.g. with working_directory(\data\stuff):

    References
    ----------
    Taken straight from http://stackoverflow.com/a/3012921/2530083

    """

    current_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(current_dir)



def get_filepaths(wildcard=""):
    """wx dialog box to select files

    Opens a dialog box from which the user can select files

    Parameters
    ----------
    wildcard : string, optional
        Only show file types of this kind. Default wildcard="" which means
        no file type filtering.
        e.g. "BMP files (*.bmp)|*.bmp|GIF files (*.gif)|*.gif"

    Returns
    -------
    filepaths: list of paths
        List of user selected file paths.  Will return None if the cancel
        button was selected.

    """

    # wx idea from http://stackoverflow.com/a/9319832/2530083
    app = wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE
    dialog = wx.FileDialog(None, 'Open', wildcard=wildcard, style=style)
    if dialog.ShowModal() == wx.ID_OK:
        filepaths = dialog.GetPaths()
    else:
        filepaths = None
    dialog.Destroy()
    return filepaths


def modules_in_package(package_name, exclude=['test']):
    """List of modules in a package

    Parameters
    ----------
    package_name : string
        Name of package to import and get modules from.
    exclude : list of string, optional
        Module names to exclude.  Default exclude=['test']

    Returns
    -------
    module_list : lsit of string
        List of modules in package_name ommiting any names in the
        exclude list.

    """

    pkg = importlib.import_module(package_name)

    pkgpath = os.path.dirname(pkg.__file__)
    return [name for _, name, _ in pkgutil.iter_modules([pkgpath])
            if name not in exclude]


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])
