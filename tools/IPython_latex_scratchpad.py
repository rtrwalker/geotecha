# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# geotecha - A software suite for geotechncial engineering
# Copyright (C) 2018  Rohan T. Walker (rtrwalker@gmail.com)
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

s = []
s += [r'\mathbf{A}_{i,j}=\int_{0}^1{\frac{d}{dz}\left({a\left(z\right)}\frac{d}{dz}{b\left(z\right)}\right)\phi_i\,dz}']
d(s[-1])

print('%s\n' * len(s) % tuple(map(unraw,s)))

# <codecell>

d('\mathbf{\Psi}_')

# <codecell>

d('\\mathbf{A}_{i,j,\\text{layer}}=\\int_{z_t}^{z_b}{{a\\left(z\\right)}\\frac{d^2\\phi_j}{dz^2}\\phi_i\\,dZ}+\\int_{z_t}^{z_b}{\\frac{d{a\\left(z\\right)}}{dz}\\frac{d\\phi_j}{dz}\\phi_i\\,dz}+\\int_{0}^{1}{{a\\left(z\\right)}\\delta\\left(z-z_t\\right)\\frac{d\\phi_j}{dz}\\phi_i\\,dz}-\\int_{0}^{1}{{a\\left(z\\right)}\\delta\\left(z-z_b\\right)\\frac{d\\phi_j}{dz}\\phi_i\\,dz}')

# <codecell>

d("\\mathbf{\\Gamma}\\mathbf{A}'=\\mathbf{\\Psi A}+loading\\:terms")#\: for space
s += [r"\mathbf{\Gamma}\mathbf{A}'=\mathbf{\Psi A}+loading\:terms"]
d(s[-1])
s += ['\\mathbf{\Gamma}_{i,j}=\\int_{0}^1{{m_v\\left(z\\right)}{\sin\\left({m_j}z\\right)}{\sin\\left({m_i}z\\right)}\,dz}']
d(s[-1])
s += ['\\mathbf{\Psi}_{i,j}=dT_h\\mathbf{A}_{i,j}=\\int_{0}^1{{k_h\\left(z\\right)}{\eta\\left(z\\right)}\\phi_i\\phi_j\\,dz}-dT_v\\int_{0}^1{\\frac{d}{dz}\\left({k_z\\left(z\\right)}\\frac{d\\phi_j}{dz}\\right)\\phi_i\\,dz}']
d(s[-1])
s+= ['\\left(\\mathbf{\\Gamma}^{-1}\\mathbf{\\Psi}\\right)']
d(s[-1])
s+= [r'u\left(Z,t\right)=\mathbf{\Phi v E}\left(\mathbf{\Gamma v}\right)^{-1}\mathbf{\theta}']
d(s[-1])
s+= [r'\mathbf{E}_{i,i}=\int_{0}^t{{\sigma\left(\tau\right)}{\exp\left({(dT\left(t-\tau\right)\lambda_i}\right)}\,d\tau}']
d(s[-1])
s+= [r'\mathbf{E}_{i,i}=\int_{0}^t{\frac{d{\sigma\left(\tau\right)}}{d\tau}{\exp\left({(dT\left(t-\tau\right)\lambda_i}\right)}\,d\tau}']
d(s[-1])
s+= [r'\sigma\left({Z,t}\right)=\sigma\left({Z}\right)\sigma\left({t}\right)']
d(s[-1])

print('%s\n' * len(s) % tuple(map(unraw,s)))

# <codecell>

s = []
s+=['\\mathbf{E}_{i,i}=\\int_{t_s}^{t_f}{\\frac{d{\\sigma\\left(\\tau\\right)}}{d\\tau}\\exp\\left({dT\\left(t-\\tau\\right)*\\lambda_i}\\right)\\,d\\tau}']
d(s[-1])    

# <codecell>

s = []
s+=['\\mathbf{A}=\\left(\\begin{matrix}E_{0,0}(t_0)&E_{1,1}(t_0)& \cdots & E_{neig-1,neig-1}(t_0)\\\ E_{0,0}(t_1)&E_{1,1}(t_1)& \\cdots & E_{neig-1,neig-1}(t_1)\\\ \\vdots&\\vdots&\\ddots&\\vdots \\\ E_{0,0}(t_m)&E_{1,1}(t_m)& \cdots & E_{neig-1,neig-1}(t_m)\\end{matrix}\\right)']
d(s[-1])

# <codecell>

s = []
s+=['t_s = \\min\\left(t,t_{increment\\:start}\\right)']
d(s[-1])

# <codecell>

s = []
s+=['\\mathbf{E}_{i,i}=\\int_{t_s}^{t_f}{{\\sigma\\left(\\tau\\right)}\\exp\\left({dT\\left(t-\\tau\\right)*\\lambda_i}\\right)\\,d\\tau}']
d(s[-1])

# <codecell>

s = []
s+=['u\\left({Z,t}\\right)=\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}+u_{top}\\left({t}\\right)\\left({1-Z}\\right)+u_{bot}\\left({t}\\right)\\left({Z}\\right)']
d(s[-1])


# <codecell>

s = []
s+=['\\overline{u}\\left(\\left({Z_1,Z_2}\\right),t\\right)=\\int_{Z_1}^{Z_2}{\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}+u_{top}\\left({t}\\right)\\left({1-Z}\\right)+u_{bot}\\left({t}\\right)\\left({Z}\\right)\,dZ}']
d(s[-1])

# <codecell>

s = []
s+=['\\overline{\\rho}\\left(\\left({Z_1,Z_2}\\right),t\\right)=\\int_{Z_1}^{Z_2}{m_v\\left({Z}\\right)\\left({\\sigma\\left({Z,t}\\right)-u\\left({Z,t}\\right)}\\right)\\,dZ}']
d(s[-1])
    s+=['\\overline{\\rho}\\left(\\left({Z_1,Z_2}\\right),t\\right)=\\int_{Z_1}^{Z_2}{m_v\\left({Z}\\right)\\sigma\\left({Z,t}\\right)\\,dZ}+\\int_{Z_1}^{Z_2}{m_v\\left({Z}\\right)\\left({\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}+u_{top}\\left({t}\\right)\\left({1-Z}\\right)+u_{bot}\\left({t}\\right)\\left({Z}\\right)}\\right)\\,dZ}']
d(s[-1])
s+=['\\int_{Z_1}^{Z_2}{m_v\\left({Z}\\right)\\left({\\mathbf{\\Phi v E}\\left(\\mathbf{\\Gamma v}\\right)^{-1}\\mathbf{\\theta}+u_{top}\\left({t}\\right)\\left({1-Z}\\right)+u_{bot}\\left({t}\\right)\\left({Z}\\right)}\\right)\\,dZ}']
d(s[-1])

# <codecell>

s = []
s+=['\\frac{m_v}{\\overline{m_v}}'
'\\frac{\\partial u}{\\partial t}='
'-\\left({dT_h \\frac{\\eta}{\\overline{\\eta}}}\\frac{k_h}{\\overline{k_h}}u'
'-dT_v \\frac{\\partial}{\\partial Z}\\left({\\frac{k_v}{\\overline{k_v}}\\frac{\\partial u}{\\partial Z}}\\right)\\right)' 
'+\\sum_{i=1}^{n_{sur}}\\frac{m_v}{\\overline{m_v}}\\frac{\\partial \\sigma_i}{\\partial t}'
'+\\sum_{i=1}^{n_{vac}}dT_h \\frac{m_v}{\\overline{m_v}}\\frac{\\partial w_i}{\\partial t}']
d(s[-1])
s+=['u\\left({Z,t}\\right) = v\\left({Z,t}\\right)'
'+\\left({1-Z}\\right)\\sum_{i=1}^{n_T}u_T\\left({t}\\right)'
'+Z\\sum_{i=1}^{n_B}u_B\\left({t}\\right)']
d(s[-1])

# <codecell>

s = []
s+=['u\\left({Z,t}\\right)=v\\left({Z,t}\\right) + u_{top}\\left({t}\\right)\\left({1-Z}\\right)+u_{bot}\\left({t}\\right)Z']
d(s[-1])
#s+=['\\frac{\\partial}{\\partial Z}\\left({\\frac{k_v}{\\overline{k_v}}\\frac{\\partial u}{\\partial Z}}\\right)\\right)']
s+=['\\frac{\\partial}{\\partial Z}\\left({a\\left({Z}\\right)\\frac{\\partial u\\left({Z,t}\\right)}{\\partial Z}}\\right)']
d(s[-1])

# <codecell>

s = []
s+=['\\sigma\\left({Z,t}\\right)=\\sigma\\left({Z}\\right)\\sigma\\left({t}\\right)\\cos\\left(\\omega t + \\phi\\right)']
d(s[-1])
s+=['\\mathbf{E}_{i,i}=\\int_{0}^t{\\frac{d{\\sigma\\left(\\tau\\right)}}{d\\tau}{\exp\\left({(dT\\left(t-\\tau\\right)\\lambda_i}\\right)}\\,d\\tau}']
d(s[-1])
s+=['\\mathbf{E}_{i,i}=\\int_{t_s}^{t_f}{\\frac{d{\\sigma\\left(\\tau\\right)}}{d\\tau}\\exp\\left({dT\\left(t-\\tau\\right)*\\lambda_i}\\right)\\,d\\tau}']
d(s[-1])


# <codecell>


