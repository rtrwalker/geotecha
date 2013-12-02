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

"""some one dimesional plottig stuff
one d data is basically x-y data .
    
"""

from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import brewer2mpl
from itertools import cycle


def rgb_shade(rgb, factor=1, scaled=True):
    """apply shade (darken) to an rgb triplet
    
    If rgba tuple is given the 'a' value is not altered.
    
    Paramters
    ---------
    rgb: tuple
        triplet of rgb values.  Note can be an rgba value
    factor: float 
        between 0 and 1.
        factor by which to shade.  Default=1   
    scaled: bool
        if True then assumes RGB values aare scaled between 0, 1. If False 
        rgb values are between 0 and 255.  default = True.
        
    Returns
    -------
    
    rgb_new: tuple
        new tuple of rgb ( or rgba)
        
    """
    
    
    
    #calc from http://stackoverflow.com/a/6615053/2530083    
    if factor<0 or factor>1:
        raise ValueError('factor must be between 0 and 1.  You have %g' % factor)
        
    rgb_new = [v * factor for v in rgb[:3]]
    if len(rgb)==4:
        rgb_new.append(rgb[3])
        
    return tuple(rgb_new)

def rgb_tint(rgb, factor=0, scaled=True):
    """apply tint (lighten) to an rgb triplet
    
    If rgba tuple is given the 'a' value is not altered.
    
    Paramters
    ---------
    rgb: tuple
        triplet of rgb values.  Note can be an rgba value
    factor: float 
        between 0 and 1.
        factor by which to tint. default=0   
    scaled: bool
        if True then assumes RGB values aare scaled between 0, 1. If False 
        rgb values are between 0 and 255.  default = True.
        
    Returns
    -------
    
    rgb_new: tuple
        new tuple of rgb ( or rgba)
        
    """
        
    #calc from http://stackoverflow.com/a/6615053/2530083    
    if factor<0 or factor>1:
        raise ValueError('factor must be between 0 and 1.  You have %g' % factor)
    
    if scaled:
        black=1.0
    else:
        black=255
        
    rgb_new = [v + (black-v) * factor for v in rgb[:3]]
    if len(rgb)==4:
        rgb_new.append(rgb[3])
        
    return tuple(rgb_new)
    

    
def copy_dict(source_dict, diffs):
    """Returns a copy of source_dict, updated with the new key-value
       pairs in diffs."""
       #http://stackoverflow.com/a/5551706/2530083
       
    result=dict(source_dict) # Shallow copy, see addendum below
    result.update(diffs)
    return result

    
class MarkersDashesColors(object):
    """Nice looking markers, dashed lines, and colors for matplotlib line plots
    
    Use this object to create a list of style dictionaries.  Each style dict 
    can be unpacked when passed to the matplotlib.plot command.  each dict 
    contains the appropriate keywords to set the marker, dashes, and color 
    properties.  You can turn the list into a cycle using functools.cycle
    To see the marker, dashes, and color options run the `demo_options' method. 
    To choose a subset of styles see the `__call__` method.  To view what the 
    chosen styles actually look like run the `construct_styles' and the 
    `demo_styles` method
    
    Parameters
    ----------
    kwargs:
        key value pairs to override the default_marker. e.g. color=(1.0,0,0) 
        would change the default line and marker color to red. markersize=8 
        would dchange all marker sizes to 8. Most likely value to change are:
        color, markersize, alpha.
    
    Attributes
    ----------
    almost_black: str
        hex string representing the default color, almost black '#262626'
    color: matplotlib color
        default color of markers and lines.  Default = almost_black
    dashes: list of tuples
        list of on, off tuples for use with matplotlib dashes.  
        See `construct_dashes` method.
    colors: list of tuples
        list of rgb tuples describing colors.  see `construct_dashes`
    markers: list of dict
        list of dict describing marker properties to pass to matplotlib.plot
        command.  see `construct_markers` method.
    default_marker: dict
        default properties for markers.  These defaults will be overridden by
        the values in `markers`
                
    """
    
    almost_black = '#262626'
    def __init__(self, **kwargs):
        """initialization of MarkersDashesColors object"""
        
        self.color = kwargs.get('color',self. almost_black)        
        self.default_marker={#'marker':'o',
                             'markersize':5,
                             'markeredgecolor': self.color,
                             'markeredgewidth':1,
                             'markerfacecolor': self.color,
                             'alpha':0.9,
                             'color': self.color
                             }
        self.default_marker.update(kwargs)                    
                        
        self.construct_dashes()
        self.construct_colors()
        
        self.construct_markers()        
        self.merge_default_markers()        
        self.styles=[]

        
        
    def __call__(self, markers=None,
                 dashes=None,
                 marker_colors=None, 
                 line_colors=None):
        """list of styles to unpack in matplotlib.plot for pleasing markers
        
        If `markers`, `dashes`, `marker_colors`, and `line_colors` are all
        ``None`` then styles will be a cycle through combos of markers, colors,
        and lines such that each default marker, line, color appears at least
        once.
        
        Parameters
        ----------
        markers: sequence
            list of ``int`` specifying  index of self.markers.  Default=None 
            i.e. no markers
        dashes: sequence
            list of ``int`` specifying index of self.dashes.  Default=None i.e.
            no line.
        marker_colors: sequence
            list of ``int`` specifying index of self.colors to apply to 
            each marker. Default=None i.e. use self.color for all markers.
        line_colors: sequence
            list of ``int`` specifying index of self.colors.  Default=None i.e.
            use self.color for lines.
            
            
        """

        n = 0
        try:
            n=max([len(v) for v in [markers, dashes, marker_colors, line_colors] if v is not None])
        except ValueError:
            pass        

        if n==0:
            markers = range(len(self.markers))
            dashes = range(len(self.dashes))
            marker_colors = range(len(self.colors))
            line_colors = range(len(self.colors))
            n = max([len(v) for v in [markers, dashes, marker_colors, line_colors] if v is not None])
                
        if markers is None: # no markers
            markers=cycle([None])
        else:
            markers=cycle(markers)
        if dashes is None: # no lines
            dashes=cycle([None])
        else:
            dashes=cycle(dashes)
        if marker_colors is None: #default color
            marker_colors=cycle([None])
        else:
            marker_colors = cycle(marker_colors)

        if line_colors is None: #defult color
            line_colors = cycle([None])
        else:
            line_colors = cycle(line_colors)
                        
        styles=[dict() for i in range(n)]           
        for i in range(n):             
            m = markers.next()
            mc = marker_colors.next()
            if m is None:
                styles[i]['marker'] = 'none'
            else:
                styles[i].update(self.markers[m])
                if mc is None:
                    styles[i].update({'markeredgecolor': self.color,
                               'markerfacecolor': self.color})
                else:                    
                    if self.markers[m]['markeredgecolor'] != 'none':
                        styles[i]['markeredgecolor']= self.colors[mc]
                    if self.markers[m]['markerfacecolor'] != 'none':
                        styles[i]['markerfacecolor'] = self.colors[mc]
                                    
            d = dashes.next()
            if d is None:
                styles[i]['linestyle'] = 'None'
            else:            
                styles[i]['dashes'] = self.dashes[d]                 
                                                          
            lc = line_colors.next()         
            if lc is None:
                styles[i]['color'] = self.color
            else:            
                styles[i]['color'] = self.colors[lc]

        

        return styles 
                     
            
        
        
    def merge_default_markers(self):
        """merge self.default_marker dict with each dict in self.markers"""
        self.markers=[copy_dict(self.default_marker, d) for d in self.markers]
        return
    def construct_dashes(self):
        """list of on, off tuples for use with matplotlib dashes"""
        
        self.dashes=[(None, None), 
                     (10,3), 
                     (3,4,10,4), 
                     (2,2), 
                     (10,4,3,4,3,4), 
                     (10,10),
                    ]#, (7,3,7,3,7,3,2,3),(20,20) ,(5,5)]                
        return
    def tint_colors(self, factor=1):
        "Lighten all colors by factor"
        
        self.colors = [rgb_tint(v, factor) for v in self.colors]
        return
    def shade_colors(self, factor=1):
        "Darken all colors by factor"
        
        self.colors = [rgb_shade(v, factor) for v in self.colors]
        return            
    def construct_colors(self):  
        """populate self.colors: a list of rgb colors"""
        
              
        # These are from brewer2mpl.get_map('Set2', 'qualitative', 8).mpl_colors
            #see http://colorbrewer2.org/
        self.colors = [(0.4, 0.7607843137254902, 0.6470588235294118), #turqoise
                       (0.9882352941176471, 0.5529411764705883, 0.3843137254901961), #orange
                       (0.5529411764705883, 0.6274509803921569, 0.796078431372549), #blue
                       (0.9058823529411765, 0.5411764705882353, 0.7647058823529411), #pink
                       (0.6509803921568628, 0.8470588235294118, 0.32941176470588235), #lime green
#                       (1.0, 0.8509803921568627, 0.1843137254901961), # yellow
                       (0.8980392156862745, 0.7686274509803922, 0.5803921568627451), #light brown                       
                       (0.7019607843137254, 0.7019607843137254, 0.7019607843137254)] #grey
                       
        self.shade_colors(0.8)        

    def construct_markers(self):
        """populate self.markers: a list of dict that define a matplotlib marker
        
        run `self.merge_default_markers` after running 
        `self.construct_markers` so that default properties are applied 
        properly.
        
        """
        
        #marker (numsides,style, angle) # theses markers cannot have fillstyle
        ngon=0
        star=1
        asterisk=2
        circle=3
        #fillstyle, 'full', 'left', 'right', 'bottom', 'top', 'none' 

        #anything in self.makers
        self.markers=[        
            #mix of matplotlib, named latex, and STIX font unicode 
                # http://www.stixfonts.org/allGlyphs.html
            {'marker': 'o'},
            {'marker': 's',
             'fillstyle': 'bottom'},
            {'marker': 'D',
             'markerfacecolor': 'none'},
            {'marker': '^'},
            {'marker': 'o',
             'fillstyle': 'left'},
            {'marker': 's',
             'markerfacecolor': 'none'},
            {'marker': 'D',
             'fillstyle': 'top'},
            {'marker': '^',
             'fillstyle': 'left'},
            {'marker': 'o',
             'markerfacecolor': 'none'},              
            {'marker': 's'}, 
            {'marker': 'D',
             'fillstyle': 'left'},                       
            {'marker': 'v',
             'markerfacecolor': 'none'},
            {'marker': r'$\boxplus$'},
            {'marker': 'D'},
            {'marker': 'v',
             'fillstyle': 'bottom'},
            {'marker': '^',
             'markerfacecolor': 'none'},
            {'marker': u'$\u25E9$'},
            {'marker': u'$\u2b2d$'},             
            {'marker': 'h'},                  
            {'marker': '^',
             'fillstyle': 'bottom'},
            {'marker': r'$\otimes$'},              
            {'marker': 'v'},
            {'marker': 'h',
             'fillstyle': 'right'},
            {'marker': 'o',
             'fillstyle': 'bottom'},
            {'marker': 's',
             'fillstyle': 'left'},            
            {'marker': 'h',
             'markerfacecolor': 'none'},                        
            {'marker': 'H',
             'fillstyle': 'top'},           
            {'marker': (6, asterisk, 0)},
            {'marker': u'$\u29bf$'},
            {'marker': u'$\u29c7$'}, 
            {'marker': u'$\u29fe$'},                                   
            {'marker': u'$\u27E1$'},            
            ]
            
        
            
            

 
#            #didn't make the cut
#            {'marker': 'H',
#             'fillstyle': 'bottom'},
#            {'marker': 'D',
#             'markerfacecolor': 'none',
#             'fillstyle': 'top'},
#            {'marker': 'o',
#             'markerfacecolor': 'none',
#             'fillstyle': 'top'},
#            {'marker': 's',
#             'markerfacecolor': 'none',
#             'fillstyle': 'top'}, 
#            {'marker': 'h',
#             'markerfacecolor': 'none',
#             'fillstyle': 'top'},
#            {'marker': '^',
#             'markerfacecolor': 'none',
#             'fillstyle': 'top'},
#             
#            {'marker': '^',
#             'markerfacecolor': 'none',
#             'fillstyle': 'right'}, 
#            {'marker': 's',
#             'markerfacecolor': 'none',
#             'fillstyle': 'right'},
#            {'marker': 'o',
#             'markerfacecolor': 'none',
#             'fillstyle': 'right'},
#            {'marker': 'h',
#             'markerfacecolor': 'none',
#             'fillstyle': 'right'},
#            {'marker': 'D',
#             'markerfacecolor': 'none',
#             'fillstyle': 'right'},
#                         
#             #default matplotlib markers that didn't make the cut
#            {'marker': 0},
#            {'marker': 1},
#            {'marker': 2},
#            {'marker': 3},
#            {'marker': 4},            
#            {'marker': 6},
#            {'marker': 7},            
#            {'marker': '|'},
#            {'marker': ''},
#            {'marker': 'None'},
#            {'marker': None},
#            {'marker': 'x'},
#            {'marker': 5},
#            {'marker': '_'},            
#            {'marker': ' '},
#            {'marker': 'd'},
#            {'marker': 'd',
#             'fillstyle': 'bottom'},
#            {'marker': 'd',
#             'fillstyle': 'left'},            
#            {'marker': '+'},
#            {'marker': '*'},
#            {'marker': ','},            
#            {'marker': '.'},
#            {'marker': '1'},
#            {'marker': 'p'},
#            {'marker': '3'},
#            {'marker': '2'},
#            {'marker': '4'},
#            {'marker': 'H'},                        
#            {'marker': 'v',
#             'fillstyle': 'left'},
#            {'marker': '8'},
#            {'marker': '<'},
#            {'marker': '>'},            
#            {'marker': (6, star, 0)},
            
#            #named latex that didn't make the cut
#            {'marker': r'$\boxtimes$'},            
#            {'marker': r'$\boxdot$'},            
#            {'marker': r'$\oplus$'},
#            {'marker': r'$\odot$'},
#            {'marker': u'$\smile$'},                        
            
            #STIX FONTS unicode that didn't make the cut
#            {'marker': u'$\u29b8$'},
#            {'marker': u'$\u29bb$'},
#            {'marker': u'$\u29be$'},
#            {'marker': u'$\u29c4$'},            
#            {'marker': u'$\u29c5$'}, 
#            {'marker': u'$\u29c6$'}, 
#            {'marker': u'$\u29c8$'}, 
#            {'marker': u'$\u29d0$'}, 
#            {'marker': u'$\u29d1$'}, 
#            {'marker': u'$\u29d2$'}, 
#            {'marker': u'$\u29d3$'},             
#            {'marker': u'$\u29d6$'}, 
#            {'marker': u'$\u29d7$'}, 
#            {'marker': u'$\u29e8$'}, 
#            {'marker': u'$\u29e9$'},
#            {'marker': u'$\u2b12$'}, 
#            {'marker': u'$\u2b13$'}, 
#            {'marker': u'$\u2b14$'}, 
#            {'marker': u'$\u2b15$'}, 
#            {'marker': u'$\u2b16$'}, 
#            {'marker': u'$\u2b17$'}, 
#            {'marker': u'$\u2b18$'}, 
#            {'marker': u'$\u2b19$'}, 
#            {'marker': u'$\u2b1f$'}, 
#            {'marker': u'$\u2b20$'}, 
#            {'marker': u'$\u2b21$'}, 
#            {'marker': u'$\u2b22$'}, 
#            {'marker': u'$\u2b23$'}, 
#            {'marker': u'$\u2b24$'}, 
#            {'marker': u'$\u2b25$'}, 
#            {'marker': u'$\u2b26$'}, 
#            {'marker': u'$\u2b27$'}, 
#            {'marker': u'$\u2b28$'}, 
#            {'marker': u'$\u2b29$'}, 
#            {'marker': u'$\u2b2a$'}, 
#            {'marker': u'$\u2b2b$'}, 
#            {'marker': u'$\u2b2c$'}, 
#            {'marker': u'$\u2b2e$'}, 
#            {'marker': u'$\u2b2f$'},
#            {'marker': u'$\u272a$'},                        
#            {'marker': u'$\u2736$'},             
#            {'marker': u'$\u273d$'},                     
#            {'marker': u'$\u27c1$'},             
#            {'marker': u'$\u25A3$'}, 
#            {'marker': u'$\u25C8$'},
#            {'marker': u'$\u25D0$'},#strange straight ine on upper right
#            {'marker': u'$\u25D1$'},#strange straight ine on upper right
#            {'marker': u'$\u25D2$'},#strange straight ine on upper right
#            {'marker': u'$\u25D3$'},#strange straight ine on upper right
#            {'marker': u'$\u25E7$'},
#            {'marker': u'$\u25E8$'},            
#            {'marker': u'$\u25EA$'},
#            {'marker': u'$\u25EC$'},
#            {'marker': u'$\u27D0$'},            
#            {'marker': u'$\u2A39$'},
#            {'marker': u'$\u2A3B$'},
#            {'marker': u'$\u22C7$'},
#            {'marker': u'$\u$'},


    #matplotlib markers
    #marker 	description
    #"." 	point
    #"," 	pixel
    #"o" 	circle
    #"v" 	triangle_down
    #"^" 	triangle_up
    #"<" 	triangle_left
    #">" 	triangle_right
    #"1" 	tri_down
    #"2" 	tri_up
    #"3" 	tri_left
    #"4" 	tri_right
    #"8" 	octagon
    #"s" 	square
    #"p" 	pentagon
    #"*" 	star
    #"h" 	hexagon1
    #"H" 	hexagon2
    #"+" 	plus
    #"x" 	x
    #"D" 	diamond
    #"d" 	thin_diamond
    #"|" 	vline
    #"_" 	hline
    #TICKLEFT 	tickleft
    #TICKRIGHT 	tickright
    #TICKUP 	tickup
    #TICKDOWN 	tickdown
    #CARETLEFT 	caretleft
    #CARETRIGHT 	caretright
    #CARETUP 	caretup
    #CARETDOWN 	caretdown
    #"None" 	nothing
    #None 	nothing
    #" " 	nothing
    #"" 	nothing    


        return
            
        
        
    def construct_styles(self, markers=None, dashes=None, marker_colors=None, 
                         line_colors=None):
        """Calls self.__call__ and assigns results to self.styles"""
                             
        self.styles = self(markers, dashes, marker_colors, line_colors)        
        
        return                      
    def demo_styles(self):
        """show a figure of all the styles in self.styles"""

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, frame_on=True)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([]) 
            
        if len(self.styles)==0:
            ax.set_title("no styles set.  Use the 'construct_styles' method")
        else:        
            for i, style in enumerate(self.styles):
                x = (i % 5)*2
                y = i//5
                ax.plot(np.array([x,x+1]), np.array([y,y+0.3]), **style)            
                s=i
                ax.annotate(s, xy=(x+1,y+0.3), 
                             xytext=(18, 0), textcoords='offset points',
                             horizontalalignment='left', verticalalignment='center')
            ax.set_xlim(-1, 11)
            ax.set_ylim(-1, y+1)
            ax.set_xlabel('styles')
            
        plt.show()            
        return
        
    def demo_options(self):
        """show a figure with all the marker, dashes, color options available"""
        
        #markers   
        gs = gridspec.GridSpec(3,1)
        fig=plt.figure(num=1,figsize=(12,10))
        
        
        ax1=fig.add_subplot(gs[0])
        for i, m in enumerate(self.markers):
            x = (i % 5)*2
            y = i//5
            ax1.plot(np.array([x,x+1]), np.array([y,y+0.3]), **m)
            s=repr(m['marker']).replace('$','').replace("u'\\u","")
            s=i
            ax1.annotate(s, xy=(x+1,y+0.3), 
                         xytext=(18, 0), textcoords='offset points',
                         horizontalalignment='left', verticalalignment='center')
        ax1.set_xlim(-1, 11)
        ax1.set_ylim(-1, y+1)
        ax1.set_xlabel('Markers')
        ax1.get_xaxis().set_ticks([])
        ax1.get_yaxis().set_ticks([])        
        ax1.set_title('Options for MarkersDashesColors object')

        #dashes        
        ax2=fig.add_subplot(gs[1])
        for i, d in enumerate(self.dashes):
            x = 0
            y = i
            ax2.plot(np.array([x,x+1]), np.array([y,y+0.3]), dashes=d,color=self.almost_black)
            s=i
            ax2.annotate(s, xy=(x+1,y+0.3), 
                         xytext=(18, 0), textcoords='offset points',
                         horizontalalignment='left', verticalalignment='center',
                         color = self.almost_black
                         )
        ax2.set_xlim(-0.1,1.2)
        ax2.set_ylim(-1,y+1)
        ax2.set_xlabel('Dashes')
        ax2.get_xaxis().set_ticks([])
        ax2.get_yaxis().set_ticks([])  
        
        #colors        
        ax3 = fig.add_subplot(gs[2])
        for i, c in enumerate(self.colors):
            x = 0
            y = i
            ax3.plot(np.array([x,x+1]), np.array([y,y+0.3]), color=c,ls='-')
            s=i
            ax3.annotate(s, xy=(x+1,y+0.3), 
                         xytext=(18, 0), textcoords='offset points',
                         horizontalalignment='left', verticalalignment='center',
                         color=self.almost_black
                         )
        ax3.set_xlim(-0.1,1.2)
        ax3.set_ylim(-1,y+1)
        ax3.set_xlabel('colors')
        ax3.get_xaxis().set_ticks([])
        ax3.get_yaxis().set_ticks([])
                     
        plt.show()
        return


def pleasing_defaults():
    """alter some matplotlib rcparams defaults to be visually more appealing


    References
    ----------
    Many of the ideas/defaults/code come from Olga Botvinnik [1]_, and her 
    prettyplot package [2]_.
    
    .. [1] http://blog.olgabotvinnik.com/post/58941062205/prettyplotlib-painlessly-create-beautiful-matplotlib
    .. [2] http://olgabot.github.io/prettyplotlib/  
    
    """
    

    # Get Set2 from ColorBrewer, a set of colors deemed colorblind-safe and
    # pleasant to look at by Drs. Cynthia Brewer and Mark Harrower of Pennsylvania
    # State University. These colors look lovely together, and are less
    # saturated than those colors in Set1. For more on ColorBrewer, see:
    # - Flash-based interactive map:
    #     http://colorbrewer2.org/
    # - A quick visual reference to every ColorBrewer scale:
    #     http://bl.ocks.org/mbostock/5577023
    set2 = brewer2mpl.get_map('Set2', 'qualitative', 8).mpl_colors
    
    # Another ColorBrewer scale. This one has nice "traditional" colors like
    # reds and blues
    set1 = brewer2mpl.get_map('Set1', 'qualitative', 9).mpl_colors
    mpl.rcParams['axes.color_cycle'] = set2
    
    # Set some commonly used colors
    almost_black = '#262626'
    light_grey = np.array([float(248) / float(255)] * 3)
    
    reds = mpl.cm.Reds
    reds.set_bad('white')
    reds.set_under('white')
    
    blues_r = mpl.cm.Blues_r
    blues_r.set_bad('white')
    blues_r.set_under('white')
    
    # Need to 'reverse' red to blue so that blue=cold=small numbers,
    # and red=hot=large numbers with '_r' suffix
    blue_red = brewer2mpl.get_map('RdBu', 'Diverging', 11,
                                  reverse=True).mpl_colormap
    
    # Default "patches" like scatterplots
    mpl.rcParams['patch.linewidth'] = 0.75     # edge width in points
    
    # Default empty circle with a colored outline
    mpl.rcParams['patch.facecolor'] = 'none'
    mpl.rcParams['patch.edgecolor'] = set2[0]
    
    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    mpl.rcParams['axes.edgecolor'] = almost_black
    mpl.rcParams['axes.labelcolor'] = almost_black
    mpl.rcParams['axes.linewidth'] = 0.5
    
    # Make the default grid be white so it "removes" lines rather than adds
    mpl.rcParams['grid.color'] = 'white'
    
    # change the tick colors also to the almost black
    mpl.rcParams['ytick.color'] = almost_black
    mpl.rcParams['xtick.color'] = almost_black
    
    # change the text colors also to the almost black
    mpl.rcParams['text.color'] = almost_black    

    
#def plot_common_x(x_y_for_each_plot,                  
#                  
#          x_axis_label='x', y_axis_labels=None, legend_labels=None,          
#          hspace=0.1, height_ratios=None,
#          plot_type='plot',
#          kwargs_figure={}):                            
#    """
#    Create a column of subplots with a common x-axis.
#    
#    plots are 'numbered' upward from zero (at bottom)
#    
#    Parameters
#    ----------
#    x_y_for_each_plot: list
#        tuple of x,y data for each plot. Data may be nested
#        e.g. x_y_for_each_plot=[([x0,y0]), ([x1,y1]), ([x2_1, y2_1], [x_2_2, y2_2]))
#    x_axis_label: str, optional
#        x-axis label. Default='x'
#    y_axis_labels: sequence of str, optional
#        y-axis labels.  Default=None i.e no y axis labels.  Use None in the 
#        sequence to turn off a particular label
#        e.g. y_axis_labels=('y0','y1','y2')
#    legend_labels: sequence of sequence of str, optional
#        legend labels for each line in plot. default=None i.e. no line labels
#        e.g. legend_labels=(['a1'], ['b1'], ['c1','c2'])   
#    line_labels: sequence of sequece of str, optional
#        label to annotate each line in plot.  
#    hspace: float, optional
#        vertical space between each subplot. default=0.1
#    height_ratios: list, optional
#        height ratios of plots. default=None i.e. all subplots have the same 
#        height.
#    plot_type: str, optional
#        matplotlib.pyplot method to use.  default='plot' i.e. x-y plot.  e.g. 
#        plot_type='scatter' gives a scatter plot.
#    kwargs_figure: dict, optional
#        dictionary of keyword arguments that wll be passed to the plt.figure 
#        e.g. kwars_figure=dict(figsize=(8, 6), dpi=80, facecolor='w', 
#        edgecolor='k')
#        
#        
#    """
#    
#    n = len(x_y_for_each_plot)
#    
#    fig = plt.figure(**kwargs_figure)
#    
#    if height_ratios is None: 
#        height_ratios = [1 for i in range(n)]
#        
#    gs=gridspec.GridSpec(n,1,
#                height_ratios=height_ratios[::-1])
#    ax=[]
#    line_objects=[]                
#    for i, x_y in enumerate(x_y_for_each_plot):
#        if i==0:        
#            ax.append(fig.add_subplot(gs[n-1-i]))
#            
#            if not x_axis_label is None:
#                ax[i].set_xlabel(x_axis_label)
#        else:
#            ax.append(fig.add_subplot(gs[n-1-i], sharex=ax[0]))
#            plt.setp(ax[i].get_xticklabels(), visible=False)
#            
#        if not y_axis_labels is None:                    
#            if not y_axis_labels[i] is None:                
#                ax[i].set_ylabel(y_axis_labels[i])                
#        
#        for j, (x,y) in enumerate(x_y):
#            ax[i].plot
#            #line_objects[i].append(getattr(ax[i], plot_type)(x,y)) #http://stackoverflow.com/a/3071/2530083
#            line_objects.append(getattr(ax[i], plot_type)(x,y)) #http://stackoverflow.com/a/3071/2530083
#
#    if not legend_labels is None:                    
#        for i, lines in enumerate(line_objects):
#            if not legend_labels[i] is None:
#                for j, line in enumerate(lines):
#                    if not legend_labels[i][j] is None:
#                        line.set_label(legend_labels[i][j]) #consider using '{0:.3g}' for numbers
#            
#    legends=[v.legend() for v in ax]                
#    [leg.draggable(True) for leg in legends] #http://stackoverflow.com/questions/2539477/how-to-draggable-legend-in-matplotlib
#            
#    return fig




    
    




def iterable_method_call(iterable, method, unpack, *args):
    """call a method on each element of an iterable
    
    
    iterable[i].method(arg)
    
    Parameters
    ----------
    iterable: sequence etc.
        iterable whos members will have thier attribute changed
    method: string
        method to call
    unpack: bool
        if True then each member of args will be unpacked before passing to 
        method.
    args: value or sequence of values
        if a single value then all members of `iterable` will have there 
        method called with the same arguments.  If a sequence of arguments 
        then each
        member[0].method will be called with args[0], 
        member[1].method will be set to args[1] etc.
        skip elements by having corresponding value of args=None
        
    """
    
    if len(args)==0:
        for i in iterable:
            getattr(i, method)()
        return
    if len(args)==1:
        if not args[0] is None:
            if unpack:                
                for i in iterable:
                    getattr(i, method)(*args[0])
            else:
                for i in iterable:
                    getattr(i, method)(args[0])                
        return
    if len(args)>1:
        for i, a in enumerate(args):
            if unpack:
                if not a is None:
                    getattr(iterable[i],method)(*a)
            else:
                if not a is None:
                    getattr(iterable[i],method)(a)
        return            

        
    

def xylabel_subplots(fig, y_axis_labels=None, x_axis_labels=None):
    """set x-axis label and y-axis label for each sub plot in figure
    
    Note: labels axes in the order they were created, which is not always the
    way they appear in the figure.
    
    Parameters
    ----------
    fig: matplotlib.Figure
        figure to apply labels to
    y_axis_labels: sequence
        label to place on y-axis of each subplot.  Use None to skip a subplot
    x_axis_labels: sequence
        label to place on x-axis of each subplot.  Use None to skip a subplot        
        
    Returns
    -------
    None
    
    """
    
    if not y_axis_labels is None:     
        for i, label in enumerate(y_axis_labels):    
            if not label is None:
                fig.axes[i].set_ylabel(label)
    if not x_axis_labels is None:     
        for i, label in enumerate(x_axis_labels):    
            if not label is None:
                fig.axes[i].set_xlabel(label)
    
    return
    

        
    
def plot_common_x(x_y_for_each_plot,                  
                  
          x_axis_label='x', y_axis_labels=None, legend_labels=None,          
          hspace=0.1, height_ratios=None,
          plot_type='plot',
          kwargs_figure={}):                            
    """
    Create a column of subplots with a common x-axis.
    
    plots are 'numbered' upward from zero (at bottom)
    
    Parameters
    ----------
    x_y_for_each_plot: list
        tuple of x,y data for each plot. Data may be nested
        e.g. x_y_for_each_plot=[([x0,y0]), ([x1,y1]), ([x2_1, y2_1], [x_2_2, y2_2]))
    x_axis_label: str, optional
        x-axis label. Default='x'
    y_axis_labels: sequence of str, optional
        y-axis labels.  Default=None i.e no y axis labels.  Use None in the 
        sequence to turn off a particular label
        e.g. y_axis_labels=('y0','y1','y2')
    legend_labels: sequence of sequence of str, optional
        legend labels for each line in plot. default=None i.e. no line labels
        e.g. legend_labels=(['a1'], ['b1'], ['c1','c2'])   
    line_labels: sequence of sequece of str, optional
        label to annotate each line in plot.  
    hspace: float, optional
        vertical space between each subplot. default=0.1
    height_ratios: list, optional
        height ratios of plots. default=None i.e. all subplots have the same 
        height.
    plot_type: str, optional
        matplotlib.pyplot method to use.  default='plot' i.e. x-y plot.  e.g. 
        plot_type='scatter' gives a scatter plot.
    kwargs_figure: dict, optional
        dictionary of keyword arguments that wll be passed to the plt.figure 
        e.g. kwars_figure=dict(figsize=(8, 6), dpi=80, facecolor='w', 
        edgecolor='k')
        
        
    """
    
    n = len(x_y_for_each_plot)
    
    fig = plt.figure(**kwargs_figure)
    
    if height_ratios is None: 
        height_ratios = [1 for i in range(n)]
        
    gs=gridspec.GridSpec(n,1,
                height_ratios=height_ratios[::-1])
    ax=[]
    line_objects=[]                
    for i, x_y in enumerate(x_y_for_each_plot):
        if i==0:        
            ax.append(fig.add_subplot(gs[n-1-i]))
            
            if not x_axis_label is None:
                ax[i].set_xlabel(x_axis_label)
        else:
            ax.append(fig.add_subplot(gs[n-1-i], sharex=ax[0]))
            plt.setp(ax[i].get_xticklabels(), visible=False)
            
        if not y_axis_labels is None:                    
            if not y_axis_labels[i] is None:                
                ax[i].set_ylabel(y_axis_labels[i])                
        
        for j, (x,y) in enumerate(x_y):
            ax[i].plot
            #line_objects[i].append(getattr(ax[i], plot_type)(x,y)) #http://stackoverflow.com/a/3071/2530083
            line_objects.append(getattr(ax[i], plot_type)(x,y)) #http://stackoverflow.com/a/3071/2530083

    if not legend_labels is None:                    
        for i, lines in enumerate(line_objects):
            if not legend_labels[i] is None:
                for j, line in enumerate(lines):
                    if not legend_labels[i][j] is None:
                        line.set_label(legend_labels[i][j]) #consider using '{0:.3g}' for numbers
            
    legends=[v.legend() for v in ax]                
    [leg.draggable(True) for leg in legends] #http://stackoverflow.com/questions/2539477/how-to-draggable-legend-in-matplotlib
            
    return fig






def row_major_order_reverse_map(shape, index_steps=None, transpose=False):
    """map an index to a position in a row-major ordered array by reversing dims
    
    ::
     
         e.g. shape=(3,3)                               
         |2 1 0|      |0 1 2|
         |5 4 3| -->  |3 4 5|         
         |8 7 6|      |6 7 8|
         need 0-->2, 1-->1, 2-->0. i.e. [2 1 0 5 4 3 8 7 6].
         Use row_major_order_reverse_map((3,3), (1,-1))
    
    Parameters
    ----------
    shape: tuple
        shape of array, e.g. (rows, columns)
    index_steps: list of 1 or -1, optional
        travese each array dimension in steps f `index_steps`. Default=None 
        i.e. all dims traversed in normal order. e.g. for 3 d array, 
        index_steps=(1,-1, 1) would mean 2nd dimension would be reversed.        
    transpose: bppl, optional
        when True, transposes indexes (final operation). Default=False
        
    Returns
    -------
    pos : 1d ndarray
        array that maps index to position in row-major ordered array
                    
        
    """
    shape=np.asarray(shape)
    if index_steps is None:
        index_steps=np.ones_like(shape,dtype=int)        
    
    
    pos=np.arange(np.product(shape)).reshape(shape)            
    a=[slice(None,None,i) for i in index_steps]
    pos[...]=pos[a]
                
    if transpose:
        return pos.T.flatten()
    else:                
        return pos.flatten()

#shape=(3,3)
#index_steps=(1,-1)
#transpose_axes=(0)
#print(row_major_order_reverse_map(shape=shape, index_steps=None, transpose=False))
#print(row_major_order_reverse_map(shape=shape, index_steps=(-1,1), transpose=False))
#print(row_major_order_reverse_map(shape=shape, index_steps=(1,-1), transpose=False))
#print(row_major_order_reverse_map(shape=shape, index_steps=(-1,-1), transpose=False))
#print(row_major_order_reverse_map(shape=shape, index_steps=None, transpose=False))        

    return



def split_sequence_into_dict_and_nondicts(*args):    
    """Separate dict and non-dict items.  Combine merge dicts and merge non-dicts

    elements are combined in teh order that they appear.  i.e. non-dict items 
    will be appended to a combined list as they are encounterd.  repeated dict
    keys will be overridded by the latest value.
    
    Parameters
    ----------
    args: tuple or list of tuples
        tuple containing mixture of dict and non-dict

    Returns
    -------
    merged_non_dict: list
        list of non dictionary items
    merged_dict: dict
        merged dictionary
        
    """
    
    
    merged_non_dict=[]
    merged_dict=dict()
    
    for tup in args:
        for v in tup:
            if isinstance(v,dict):
                merged_dict.update(v) #http://stackoverflow.com/a/39437/2530083
            else:
                merged_non_dict.append(v)
    return merged_non_dict, merged_dict








def plot_data_in_grid(fig, data, gs, 
                       gs_index=None,                      
                       sharex=None, sharey=None):
    """make a subplot for each set of data 
    
    Parameters
    ----------
    fig : matplotlib.Figure
        figure to create subplots in             
    data: sequence of sequence of 2 element sequence        
        data[i] = Data for the ith subplot.
        data[i][j] = jth (x, y) data set for the ith subplot.
        Each set of (x,y) data will be plotted using matplotlib.plot fn
        e.g. data=[([x0,y0]), ([x1,y1]), ([x2a, y2a], [x2b, x2b]))        
        Note that data[i][j] will be split into list of all the non-dict items
        and a merged dict of all the dict items.  Both the list and the merged
        dict will be unpacked and passed to the `plot_type` function.  This
        allows passing of keyword arguments. If one of the dict keys is 
        'plot_type' it's value should be a string indicating a method of 
        matplotlib.Axes that can be used to create the subplot.  If 
        'plot_type' is not found then the default matplotlib.Axes.plot will 
        be used.
    gs: matplotlib.gridspec.GridSpec instance
        defines the grid in which subplots will be created
    gs_index: list of int or slice, optional
        Specifies the position within gs that each data set will be plotted
        Positions can be specified by 1) an integer which will correspond to 
        row-major ordering in the grid (e.g. for a 3x3 grid, index 3 will be
        second row, first column), or 2) a tuple of (row,column), or 3) a slice
        (e.g. index of np.s_[:1,:1] will span from first row, first column to 
        second row, second column)
        Default=None subplots are added in row-major ordering   
    sharex: sequence of int
        subplot index to share x-axis with. Default=None i.e. no sharing.  
        To skip a  subplot put None as the corresponding element of sharex. 
        If only one value is given and ther is more than one data set then 
        all subplots will share the given axis.  Note that the axis to share 
        must already have been created.
    sharey: sequence of int
        subplot index to share y-axis with. Default=None i.e. no sharing.  
        To skip a  subplot put None as the corresponding element of sharey.
        If only one value is given and ther is more than one data set then 
        all subplots will share the given axis.  Note that the axis to share 
        must already have been created.
    
    
        
        
    Returns
    -------
    ax: list of :class:`matplotlib.pyplot.Axes` instances.  
    
    
    
    """
    
#    gridspec_prop: dict
#        dictionary of keyword arguments to pass to matplotlib.gridspec.GridSpec
#        object. Any attribute will correspond to the convential positioning, 
#        i.e. gs_index will be ignored. Default=dict(). 
#        e.g. gridspec_prop=dict(width_ratios=[1,2], 
#        height_ratios=[4,1], left=0.55, right=0.98, hspace=0.05, wspace=0.02)       
#    plot_type: sequence of str, optional
#        list of matplotlib.pyplot methods to use for each data set.  
#        default=None which uses 'plot'
#        i.e. x-y plot.  e.g. plot_type='scatter' gives a scatter plot.
        
    if gs_index is None:
        gs_index = np.arange(len(data))                            
        
#    if plot_type is None:
#        plot_type = ['plot' for i in data]
#    elif len(plot_type) == 1:
#        plot_type = [plot_type[0] for i in data]
        
        
    if sharex is None:
        sharex = [None for i in data]
    elif len(sharex) == 1:
        i = sharex[0]        
        sharex = [i for j in data]
        sharex[i] = None
        
    if sharey is None:
        sharey = [None for i in data]    
    elif len(sharey) == 1:
        i = sharey[0]        
        sharey = [i for j in data]
        sharey[i] = None
                        
    ax = []            
    for i, sublot_data in enumerate(data):
        j = gs_index[i] 
        if sharex[i] is None:
            shx=None
        else:
            shx=ax[sharex[i]]
        
        if sharey[i] is None:
            shy=None
        else:
            shy=ax[sharey[i]]

        ax.append(fig.add_subplot(gs[j], sharex=shx, sharey=shy))
        
        for j, xy_etc in enumerate(sublot_data):
            args_, kwargs_ = split_sequence_into_dict_and_nondicts(xy_etc)
            plot_type = kwargs_.pop('plot_type', 'plot')
            
            getattr(ax[-1], plot_type)(*args_,**kwargs_) #http://stackoverflow.com/a/3071/2530083 
                
        
                
        
        
#        if suplot_data is None:
#            ax[-1].plot()
##            getattr(ax[-1], plot_type[i])()            
##            ax[-1].axis('off')#removes axes instance            
#            ax[-1].set_axis_bgcolor('none')
#            ax[-1].set_frame_on(False)
#            ax[-1].get_xaxis().set_ticks([])#http://stackoverflow.com/a/2176591/2530083
#            ax[-1].get_yaxis().set_ticks([])
#        else:                                              
#                   
##        ax[-1].set_ylabel(i) #use for debugging
    return ax


    
if __name__ == '__main__':
    a=MarkersDashesColors()
    a.demo_options()
    a.construct_styles()
    a.construct_styles(markers=[0,5,6,2])
    a.construct_styles(markers=[0,5,6,2], dashes=[0,3])
    a.construct_styles(markers=[0,5,6,2], dashes=[0,3], marker_colors=[0,1], line_colors=[2,3])
    a.demo_styles()
    
    
    #plot_data_in_grid(None,[(1,2),(4,5)],[(3,5)])

    #flat = [x for sublist in nested for x in sublist] #http://stackoverflow.com/a/2962856/2530083
    if 0:
        pleasing_defaults()
        fig=plt.figure()
        x = np.linspace(-np.pi,np.pi,100)
        d2 = np.linspace(0,1,4)
        d3 = np.linspace(0,1,2)
        label1= ['{0:.3g}'.format(i) for i in d2]
        y1 = np.sin(x)
        y2 = 1000*np.cos(x[:,np.newaxis]+d2)
        y3 = 1e-3*np.sin(x[:,np.newaxis]+d3)


        data= [[[x,y2]],[[2*x,y1,dict(plot_type='scatter')]],[[1.5*x,y3]]]
        y_axis_labels=['0','1', '2']
        x_axis_labels=['Time',None, None]                
        
        #gs = gridspec.GridSpec(shape, hspace=0.08, wspace=0.1)
        shape=(3,3)
        gs = gridspec.GridSpec(*shape)
        transpose = True
        index_steps=(1,-1)
        sharey=None        
        sharex=[None,0,None]
        
        #plot_type=['plot','plot','scatter','plot']
        gs_index = row_major_order_reverse_map(shape=shape,index_steps=index_steps, transpose=transpose)
        gs_index = [np.s_[:2,:2],2,8]
        #gs_index=[1,2,3,0]
        ax=plot_data_in_grid(fig, data=data, gs=gs,
                              gs_index=gs_index,
                              sharex=sharex, sharey=sharey)
                           
        
        
        
        
        

        [plt.setp(ax[i], ylabel=label) 
            for i,label in enumerate(y_axis_labels) if not label is None]
        [plt.setp(ax[i], xlabel=label) 
            for i,label in enumerate(x_axis_labels) if not label is None]
        [plt.setp(ax[i].get_xticklabels(), visible=value) 
            for i, value in enumerate([True,False,False]) if not value is None]
        #gs.tight_layout(fig)
        fig.tight_layout()
#            
##Note: axes.flat
#            
##        [fig.axes[i]._shared_x_axes.join(fig.axes[i], value) for 
##            i,value in enumerate(sharex) if not value is None]
#            
#            
#        [print(sorted(map(tuple, fig.axes[i]._shared_x_axes))) for 
#            i,value in enumerate(sharex) if not value is None]    
#        #print(sorted(map(tuple, fig.axes[i]._shared_x_axes)))
#        
#        #make the joins        
#        [fig.axes[i]._shared_x_axes.join(fig.axes[i], value) for 
#            i,value in enumerate(sharex) if not value is None]
#            
#        [print(sorted(map(tuple, fig.axes[i]._shared_x_axes))) for 
#            i,value in enumerate(sharex) if not value is None]                
#        #print(sorted(map(tuple, fig.axes[i]._shared_x_axes)))    
##        print([fig.axes[i] in fig.axes[i]._shared_x_axes for 
##            i,value in enumerate(sharex) if not value is None]            )
##        
#        [fig.axes[i].apply_aspect() for 
#            i,value in enumerate(sharex) if not value is None]
#        plt.Axes.get_shared_x_axes()
#        matplot
#        self._shared_x_axes.join(self, sharex)            
            
#        print(plt.getp(fig.axes[i], 'sharex'))                
#        [plt.setp(fig.axes[i], sharex=value) 
#            for i, value in enumerate(sharex) if not value is None]                
                  
#        iterable_method_call(fig.axes, 'set_ylabel', *y_axis_labels)
#        iterable_method_call(fig.axes, 'set_xlabel', *x_axis_labels)
#        iterable_method_call(fig.axes, 'set_xlabel', *x_axis_labels)
#        xylabel_subplots(fig, y_axis_labels,x_axis_labels)
        
        plt.show()
    if 0:
        x = np.linspace(-np.pi,np.pi,100)
        d = np.linspace(0,1,4)
        
        label1= ['{0:.3g}'.format(i) for i in d]
        y1 = np.sin(x)
        y2 = np.cos(x[:,np.newaxis]+d)
        y3 = np.sin(x[:,np.newaxis]+d)
            
        a= plot_common_x([[[x,y2]],[[x,y1]],[[x,y3]]],                  
                          
                  x_axis_label='x', y_axis_labels=['$\sigma$', 'load', None], legend_labels=[label1,['surcharge'],label1],          
                  hspace=0.1, height_ratios=[2,1,1],
                  plot_type='plot',
                  kwargs_figure=dict(num=3, figsize=(10,10)))
        
        plt.show()
    
    
    
    
#    bbox_args = None#dict(boxstyle="round,pad=0.4", fc="yellow",alpha=0.3)
#    bbox_args2 = dict(boxstyle="round,pad=0.6", fc=None, alpha=0)
#    arrow_args = dict(arrowstyle="->")
#    fig=plt.figure()
#    ax = fig.add_subplot(111)
#    np.random.seed(2)
#    x=np.random.randn(100)
#    y=np.random.randn(100)
#    scatter = ax.scatter(x, y, label='h')    
#    
#    legend=ax.legend()
#    legend.draggable(True)        
#    
#    anp = ax.annotate('$\hspace{1}$', xy=(x[0], y[0]),  xycoords='data',
#                   xytext=None,  textcoords=None,
#                   ha="center", va="center",
#                   bbox=bbox_args2,                         
#                   )
#    
#    ant = ax.annotate('Drag me 1', xy=(0.5, 0.5),  xycoords=anp,
#                   xytext=(15,0.5),  textcoords=anp,#'offset points',
#                   ha="left", va="center",
#                   bbox=bbox_args,
#                    arrowprops=dict(
#                                   #patchB=anp.get_bbox_patch(),
#                                   connectionstyle="arc3,rad=0.2",
#                                   **arrow_args)
#                   )
#
#    
#    anp.draggable()
#    ant.draggable()
#
#    
#    plt.show()    