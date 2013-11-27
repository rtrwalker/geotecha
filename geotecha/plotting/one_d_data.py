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

"""some one dimesional plottig stuff"""

from __future__ import division, print_function

import numpy as np
from sympy.printing.fcode import FCodePrinter
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
    #plot_data_in_grid(None,[(1,2),(4,5)],[(3,5)])

    #flat = [x for sublist in nested for x in sublist] #http://stackoverflow.com/a/2962856/2530083
    if 1:
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
                          
                  x_axis_label='x', y_axis_labels=['$\sigma$', 'load',None], legend_labels=[label1,['surcharge'],label1],          
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