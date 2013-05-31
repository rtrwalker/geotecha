# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

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
"""Latex scratch pad using IPython Notebook

This is just a scratch pad for displaying latex expressions.  
It might show some examples/reminders that are useful.

To display the equations open IPython Notebook and then import 
the file.

"""

#in Ipython Notebook you will have to evaluate this cell, SHIFT + ENTER, before any other cell
from IPython.display import display, Math, Latex
import re
def d(s):
    """display a string as latex math"""
    display(Math(s))
    
def unraw(s):
    """Escape any back slashes in a raw string

    If you print the result you can then copy the output 
    and use it as a plain string and it should produce the 
    same Latex image.  Useful for including equations in 
    docstrings when you don't want to make the whole docstring
    raw.
    
    Parameters
    ----------
    s : ``str``
        string to do replacements in

    Returns
    -------
    out : ``str``
        string with all backslashes replaced with two backslashes
    """
    
    return s.replace("\\", "\\\\") 

# <codecell>

#example
s = r'F(k) = \int_{-\infty}^{\infty} f(x) e^{2\pi i h} dx'
s2 = 'F(k) = \\int_{-\\infty}^{\\infty} f(x) e^{2\\pi i h} dx'
print('s is a raw string: \n'+ s)
print('display s:')
d(s)
print('s2 is a plain string: \n'+ s)
print('display s2:')
d(s)
print('unraw s: \n' + unraw(s))

# <codecell>

s = r'\mathbf{\Psi}_{i,j}=\int_{0}^1{\frac{d}{dZ}\left({a\left(z\right)}\frac{d}{dZ}\left({sin\left({m_j}z\right)}\right)\right){sin\left({m_i}z\right)}\,dZ}'
d(s)

# <codecell>

s = []
s += [r'\mathbf{A}_{i,j}=\int_{0}^1{{a\left(z\right)}\phi_i\phi_j\,dz}']
d(s[-1])
s += [r'\phi_i\left(z\right)=\sin\left({m_i}z\right)']
d(s[-1])
s += [r'a\left(z\right) = a_t+\frac{a_b-a_t}{z_b-z_t}\left(z-z_t\right)']
d(s[-1])
print('%s\n' * len(s) % tuple(map(unraw,s)))

# <codecell>

s = []
s += [r'\mathbf{A}_{i,j}=\int_{0}^1{{a\left(z\right)}{b\left(z\right)}\phi_i\phi_j\,dz}']
d(s[-1])
s += [r'\phi_i\left(z\right)=\sin\left({m_i}z\right)']
d(s[-1])
s += [r'a\left(z\right) = a_t+\frac{a_b-a_t}{z_b-z_t}\left(z-z_t\right)']
d(s[-1])
s += [r'b\left(z\right) = b_t+\frac{b_b-b_t}{z_b-z_t}\left(z-z_t\right)']
d(s[-1])
print('%s\n' * len(s) % tuple(map(unraw,s)))

# <codecell>

s = []
s += [r'\mathbf{A}_{i,j}=\int_{0}^1{{a\left(z\right)}\phi_i\phi_j\,dz}']
d(s[-1])
s += [r'\phi_i\left(z\right)=\sin\left({m_i}z\right)']
d(s[-1])
s += [r'a\left(z\right) = a_t+\frac{a_b-a_t}{z_b-z_t}\left(z-z_t\right)']
d(s[-1])
print('%s\n' * len(s) % tuple(map(unraw,s)))

# <codecell>

s = []
s += [r'\mathbf{A}_{i,j}=\int_{0}^1{\frac{d}{dz}\left({a\left(z\right)}\frac{d\phi_j}{dz}\right)\phi_i\,dz}']
d(s[-1])
s += [r'\mathbf{A}_{i,j}=\int_{0}^1{{a\left(z\right)}\frac{d^2\phi_j}{dZ^2}\phi_i\,dZ}+\int_{0}^1{\frac{d{a\left(z\right)}}{dZ}\frac{d\phi_j}{dZ}\phi_i\,dZ}']
d(s[-1])
s += [r'\mathbf{A}_{i,j,\text{layer}}=\int_{z_t}^{z_b}{{a\left(z\right)}\frac{d^2\phi_j}{dz^2}\phi_i\,dZ}+\int_{z_t}^{z_b}{\frac{d{a\left(z\right)}}{dz}\frac{d\phi_j}{dz}\phi_i\,dz}+\int_{0}^{1}{{a\left(z\right)}\delta\left(z-z_t\right)\frac{d\phi_j}{dz}\phi_i\,dz}-\int_{0}^{1}{{a\left(z\right)}\delta\left(z-z_b\right)\frac{d\phi_j}{dz}\phi_i\,dz}']
d(s[-1])
s += [r'\mathbf{A}_{i,j,\text{layer}}=\int_{z_t}^{z_b}{{a\left(z\right)}\frac{d^2\phi_j}{dz^2}\phi_i\,dZ}+\int_{z_t}^{z_b}{\frac{d{a\left(z\right)}}{dz}\frac{d\phi_j}{dz}\phi_i\,dz}+\left.{a\left(z\right)}\frac{d\phi_j}{dz}\phi_i\right|_{z=z_t}-\left.{a\left(z\right)}\frac{d\phi_j}{dz}\phi_i\right|_{z=z_b}']
d(s[-1])
s += [r'F\left(z\right)=\int{{a\left(z\right)}\frac{d^2\phi_j}{dz^2}\phi_i\,dZ}+\int{\frac{d{a\left(z\right)}}{dz}\frac{d\phi_j}{dz}\phi_i\,dz}-\left.{a\left(z\right)}\frac{d\phi_j}{dz}\phi_i\right|_{z=z}']
d(s[-1])
s += [r'\mathbf{A}_{i,j,\text{layer}}=F\left(z_b\right)-F\left(z_t\right)']
d(s[-1])
print('%s\n' * len(s) % tuple(map(unraw,s)))

# <codecell>


# <codecell>

print '\\mathbf{\\Psi}_{i,j,layer}=\\frac{dT_h}{dT}\\int_{Z_t}^{Z_b}{\\frac{k_h}{\\overline{k}_h}\\frac{\\eta}{\\overline{\\eta}}\\phi_j\\phi_i\\,dZ}-\\frac{dT_v}{dT}\\int_{Z_t}^{Z_b}{\\frac{k_v}{\\overline{k}_v}\\frac{d^2\\phi_j}{dZ^2}\\phi_i\\,dZ}-\\frac{dT_v}{dT}\\int_{Z_t}^{Z_b}{\\frac{d}{dZ}\\left(\\frac{k_v}{\\overline{k}_v}\\right)\\frac{d\\phi_j}{dZ}\\phi_i\\,dZ}-\\frac{dT_v}{dT}\\left.\\frac{k_v}{\\overline{k}_v}\\frac{d\\phi_j}{dZ}\\phi_i\\right|_{Z=Z_t}+\\frac{dT_v}{dT}\\left.\\frac{k_v}{\\overline{k}_v}\\frac{d\\phi_j}{dZ}\\phi_i\\right|_{Z=Z_b}'

# <codecell>

d('\mathbf{\Psi}_')

# <codecell>

d('\\mathbf{A}_{i,j,\\text{layer}}=\\int_{z_t}^{z_b}{{a\\left(z\\right)}\\frac{d^2\\phi_j}{dz^2}\\phi_i\\,dZ}+\\int_{z_t}^{z_b}{\\frac{d{a\\left(z\\right)}}{dz}\\frac{d\\phi_j}{dz}\\phi_i\\,dz}+\\int_{0}^{1}{{a\\left(z\\right)}\\delta\\left(z-z_t\\right)\\frac{d\\phi_j}{dz}\\phi_i\\,dz}-\\int_{0}^{1}{{a\\left(z\\right)}\\delta\\left(z-z_b\\right)\\frac{d\\phi_j}{dz}\\phi_i\\,dz}')

# <codecell>


