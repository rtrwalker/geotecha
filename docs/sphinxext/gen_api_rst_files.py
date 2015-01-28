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
"""Some routines to generate some documentation

"""

from __future__ import division, print_function
import textwrap
import os
import functools
import importlib
import pkgutil
import inspect
from geotecha.inputoutput.inputoutput import object_members
from geotecha.inputoutput.inputoutput import modules_in_package


def rst_heading(heading_text, underline="=", overline=False):
    """add underline and overline to heading for use in an rst ffile

    Parameters
    ----------
    heading_text : string
        text to underline
    underline : string, optional
        character to use as underline for the heading.  Default='='.
    overline : False/True Optional
        if True then the underline character will be repeated above the
        heading.

    Returns
    -------
    formatted_heading : string
        heading with underline, and overline where appropriate


    Examples
    --------
    >>> rst_heading('cat').split()
    ['cat', '===']
    >>> rst_heading('cat', overline=True).split()
    ['===', 'cat', '===']
    >>> rst_heading('puma', underline="+").split()
    ['puma', '++++']


    """

    n = len(heading_text)
    uline = (underline * n)[0:n]

    if overline:
        formatted_heading = "\n".join([uline, heading_text, uline])
    else:
        formatted_heading = "\n".join([heading_text, uline])

    return formatted_heading


class document_package(object):
    """Some crudeFunctionality to generate some docs for a package

    Parameters
    ----------
    package_name : string, optional
        Name of package.  Must be importable, default = 'geotecha'
    docs_path : string, optional
        path where doc files are generated.  default = "..\api"
    heading_underlines : list of string, optional
        characters for heading uncerlines.
        Default=["*", "=", "-", "+"]
    heading_overlines : list of True/False, optional
        If True, then corresponding heading level will be overlined.
        Default=[True, False, False, False]),
    ext : s, optional
        file extension for documents (without the '.').  default = "rst"


    Attributes
    ----------
    heading_underlines : list of string
        See parameters above
    heading_overlines : list of string
        See parameters above

    """

    def __init__(self,
                 package_name='geotecha',
                 docs_path=None,
                 heading_underlines=["*", "=", "-", "+"],
                 heading_overlines=[True, False, False, False],
                 ext='rst'):


        self.package_name = package_name

        if docs_path is None:
            self.docs_path = os.path.abspath(os.path.join('..', 'api'))
        else:
            self.docs_path = docs_path

        if not os.path.isdir(self.docs_path):
            os.mkdir(self.docs_path)

        self.heading_underlines = heading_underlines
        self.heading_overlines = heading_overlines

        self.ext = ext

    def heading(self, text, level):

        return rst_heading(text,
                           underline=self.heading_underlines[level],
                           overline=self.heading_overlines[level])


    def level0_template(self):
        """geotecha.rst"""
        text = textwrap.dedent("""\
        {heading:s}

        .. automodule:: {module_name:s}
            :members:
            :undoc-members:
            :show-inheritance:

        Sub-packages summary:

        .. currentmodule:: {current_module:s}

        .. autosummary::
           :toctree:

        {autosummary_items:s}


        """)
        return text#os.linesep.join(text.splitlines())

    def level1_template(self):
        """geotecha.<sub-package>.rst"""
        text = textwrap.dedent("""\
        {heading:s}

        .. automodule:: {module_name:s}
            :members:
            :undoc-members:
            :show-inheritance:


        Modules:

        .. currentmodule:: {current_module:s}

        .. autosummary::
           :toctree:

           {autosummary_items:s}



        """)
        return text#os.linesep.join(text.splitlines())

    def module_template(self):
        """geotecha.<sub-package>.<module-name>.rst"""
        text = textwrap.dedent("""\
        {heading:s}


        {class_summary:s}
        {function_summary:s}
        Module listing
        ++++++++++++++

        .. automodule:: {current_module:s}
           :members:
           :undoc-members:
           :show-inheritance:



        """)
        return text#os.linesep.join(text.splitlines())



    def class_summary(self):
        """text snippet for autosummary of classes in a module"""
        text = textwrap.dedent("""\
        Class summary
        +++++++++++++

        .. currentmodule:: {current_module:s}

        .. autosummary::

        {class_list:s}


        """)
        return text#os.linesep.join(text.splitlines())

    def function_summary(self):
        """text snippet for autosummary of functions in a module"""
        text = textwrap.dedent("""\
        Function summary
        ++++++++++++++++

        .. currentmodule:: {current_module:s}

        .. autosummary::

        {function_list:s}


        """)
        return text#os.linesep.join(text.splitlines())






    def make_rst_files(self):
        """Generate the rst files for use with sphinx"""


        level0 = importlib.import_module(self.package_name)
        # list of sub_packages in geotecha
#        member_list = object_members(level0, "module", join=False)
        member_list = modules_in_package(self.package_name)
        member_list_dotted = [".".join([self.package_name, v]) for v
                            in member_list]
        member_list_text = "\n".join([" "*3 + v for v in member_list])
        file_list = [ v +'.' + self.ext for v in member_list_dotted]
        file_list_indented = [" "*3 + v for v in file_list]
        file_list_text = "\n".join(file_list_indented)



        # geotecha.rst
        fpath = os.path.join(self.docs_path,
             self.package_name + "." + self.ext)

        d = dict(heading=self.heading(self.package_name, 0),
                 toc_tree_files=file_list_text,
                 module_name=self.package_name,
                 current_module=self.package_name,
                 autosummary_items=member_list_text)
#        print(fpath)
#        print(file_list_text)

        with open(fpath,'w') as f:
            f.write(self.level0_template().format(**d))

        #sub_package.rst
        for sub_package, heading in zip(member_list_dotted, member_list):
            level1 = importlib.import_module(sub_package)
#            sub_package_list = object_members(level1, "module", join=False)
            sub_package_list = modules_in_package(sub_package)
            sub_package_list_dotted = [".".join([sub_package, v]) for v
                                  in sub_package_list]
            sub_package_list_text ="\n".join([" "*3 + v for v in sub_package_list])
            file_list = [ v +'.' + self.ext for v in sub_package_list_dotted]
            file_list_indented = [" "*3 + v for v in file_list]
            file_list_text = "\n".join(file_list_indented)

            fpath = os.path.join(self.docs_path,
                                 sub_package + "." + self.ext)

            d = dict(heading=self.heading(heading, 1),
                     module_name=sub_package,
                     current_module=sub_package,
                     autosummary_items=sub_package_list_text,

                     )
#            print(fpath)
#            print(file_list_text)
            with open(fpath,'w') as f:
                f.write(self.level1_template().format(**d))




            #modules
            for module, heading in zip(sub_package_list_dotted,
                                       sub_package_list):

                level2 = importlib.import_module(module)
#                class_list = object_members(level2, "class", join=False)
#                class_list = [i for i, j in
#                                 inspect.getmembers(level2,
#                                                    inspect.isclass)]
                class_list = [i for i, j in
                                 inspect.getmembers(level2,
                                                    predicate =
                    lambda x: inspect.isclass(x) and x.__module__ == level2.__name__)]


                if len(class_list) > 0:
                    class_list_indented = [" "*3 + v for v in class_list]
                    class_list_text = "\n".join(class_list_indented)
                    d = dict(current_module=module,
                             class_list=class_list_text)
                    class_summary_text = self.class_summary().format(**d)
                else:
                    class_summary_text = ""

#                function_list = object_members(level2, "routine", join=False)
#                function_list = [i for i, j in
#                                 inspect.getmembers(level2,
#                                                    inspect.isroutine)]
                function_list = [i for i, j in
                                 inspect.getmembers(level2,
                                                    predicate =
                    lambda x: inspect.isroutine(x) and x.__module__ == level2.__name__)]
                if len(function_list) > 0:
                    function_list_indented = [" "*3 + v for v in function_list]
                    function_list_text = "\n".join(function_list_indented)
                    d = dict(current_module=module,
                             function_list=function_list_text)
                    function_summary_text = self.function_summary().format(**d)
                else:
                    function_summary_text = ""

                fpath = os.path.join(self.docs_path,
                                     module + "." + self.ext)

                d = dict(heading=self.heading(heading, 2),
                         current_module=module,
                         function_summary=function_summary_text,
                         class_summary=class_summary_text)
#                print('xxx', fpath)
#                print(class_list_text)
#                print(function_list_text)
                with open(fpath,'w') as f:

                    f.write(self.module_template().format(**d))

def generate_api_rst(app):
    rootdir = os.path.join(app.builder.srcdir, 'api')
    print('made it here')
    a = document_package(package_name='geotecha', docs_path=rootdir)
    print('made it here2')
    a.make_rst_files()

def setup(app):

    app.connect('builder-inited', generate_api_rst)


