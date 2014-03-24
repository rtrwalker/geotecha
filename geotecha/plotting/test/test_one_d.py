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
"""Some test routines for the one_d module

"""
from __future__ import division, print_function

from nose import with_setup
from nose.tools.trivial import assert_almost_equal

from nose.tools.trivial import assert_equal
from nose.tools.trivial import assert_raises
from nose.tools.trivial import ok_

import unittest
from math import pi
import numpy as np
from geotecha.piecewise.piecewise_linear_1d import PolyLine
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.testing.decorators import cleanup
from numpy.testing import assert_allclose
try:
    from matplotlib.testing.decorators import CleanupTestCase
    temp_cls = CleanupTestCase
except ImportError:
    temp_cls = unittest.TestCase


from geotecha.plotting.one_d import rgb_shade
from geotecha.plotting.one_d import rgb_tint
from geotecha.plotting.one_d import copy_dict
from geotecha.plotting.one_d import MarkersDashesColors
from geotecha.plotting.one_d import apply_dict_to_object
from geotecha.plotting.one_d import split_sequence_into_dict_and_nondicts
from geotecha.plotting.one_d import row_major_order_reverse_map
from geotecha.plotting.one_d import xylabel_subplots
from geotecha.plotting.one_d import iterable_method_call
from geotecha.plotting.one_d import apply_markevery_to_sequence_of_lines
from geotecha.plotting.one_d import plot_vs_time
from geotecha.plotting.one_d import plot_vs_depth

def test_rgb_shade():
    """some tests for rgb_shade"""
#    rgb_shade(rgb, factor=1, scaled=True)
    ok_(np.allclose(rgb_shade((0.4, 0.8, 0.2)), (0.4, 0.8, 0.2)))

    ok_(np.allclose(rgb_shade((0.4, 0.8, 0.2), factor=0.5),
                        (0.2, 0.4, 0.1)))
    ok_(np.allclose(rgb_shade((0.4, 0.8, 0.2, 0.8), factor=0.5),
                        (0.2, 0.4, 0.1,0.8)))
#    assert_equal(rgb_shade((0.4, 0.8, 0.2)), (0.4, 0.8, 0.2))

    assert_raises(ValueError, rgb_shade,(0.5,0.5,0.5),factor=1.5)
    assert_raises(ValueError, rgb_shade,(0.5,0.5,0.5),factor=-0.5)

def test_rgb_tint():
    """some tests for rgb_tint"""
#    rgb_tint(rgb, factor=1, scaled=True)
    ok_(np.allclose(rgb_tint((0.4, 0.8, 0.2)), (0.4, 0.8, 0.2)))

    ok_(np.allclose(rgb_tint((0.4, 0.8, 0.2), factor=0.5),
                        (0.7, 0.9, 0.6)))

#    assert_equal(rgb_tint((0.4, 0.8, 0.2)), (0.4, 0.8, 0.2))

    assert_raises(ValueError, rgb_tint,(0.5,0.5,0.5),factor=1.5)
    assert_raises(ValueError, rgb_tint,(0.5,0.5,0.5),factor=-0.5)

    ok_(np.allclose(rgb_tint((155, 205, 55), factor=0.5, scaled=False),
                        (205, 230, 155)))
    ok_(np.allclose(rgb_tint((155, 205, 55, 0.5), factor=0.5, scaled=False),
                        (205, 230, 155, 0.5)))

def test_copy_dict():
    """test for copy_dict"""
    #copy_dict(source_dict, diffs)
    ok_(copy_dict({'a':7, 'b':12}, {'c':13})=={'a':7, 'b':12, 'c':13})

    ok_(copy_dict({'a':7, 'b':12}, {'a':21, 'c':13})=={'a':21, 'b':12, 'c':13})

def test_MarkersDashesColors():
    """tests for MarkersDashesColors class"""


    a = MarkersDashesColors()
    a.color=(0.5, 0, 0)
    a.default_marker={       'markersize':5,
                             'markeredgecolor': a.color,
                             'markeredgewidth':1,
                             'markerfacecolor': a.color,
                             'alpha':0.9,
                             'color': a.color
                             }

    a.markers=[{'marker': 'o', 'markerfacecolor': 'none'},
               {'marker': 's'},
               {'marker': '^'},]
    a.colors=[(0, 0.5, 0),(0, 0, 0.5)]

    a.dashes=[(None, None), [4, 4] ]

    a.merge_default_markers()

    a.construct_styles()

    assert_equal(a.styles[0]['marker'], 'o')
    assert_equal(a.styles[0]['markersize'], 5)
    assert_equal(a.styles[0]['markeredgecolor'], (0,0.5,0))
    assert_equal(a.styles[0]['markeredgewidth'], 1)
    assert_equal(a.styles[0]['markerfacecolor'], 'none')
    assert_equal(a.styles[0]['alpha'], 0.9)
    assert_equal(a.styles[0]['color'], (0,0.5,0))
    assert_equal(a.styles[0]['dashes'], (None, None))

    assert_equal(a.styles[1]['marker'], 's')
    assert_equal(a.styles[1]['markersize'], 5)
    assert_equal(a.styles[1]['markeredgecolor'], (0,0,0.5))
    assert_equal(a.styles[1]['markeredgewidth'], 1)
    assert_equal(a.styles[1]['markerfacecolor'], (0,0,0.5))
    assert_equal(a.styles[1]['alpha'], 0.9)
    assert_equal(a.styles[1]['color'], (0,0,0.5))
    assert_equal(a.styles[1]['dashes'], [4, 4])


    assert_equal(a.styles[2]['marker'], '^')
    assert_equal(a.styles[2]['markersize'], 5)
    assert_equal(a.styles[2]['markeredgecolor'], (0,0.5,0))
    assert_equal(a.styles[2]['markeredgewidth'], 1)
    assert_equal(a.styles[2]['markerfacecolor'], (0,0.5,0))
    assert_equal(a.styles[2]['alpha'], 0.9)
    assert_equal(a.styles[2]['color'], (0,0.5,0))
    assert_equal(a.styles[2]['dashes'], (None, None))


    styles = a(markers=[1,1])
    assert_equal(styles[0]['marker'], 's')
    assert_equal(styles[0]['markersize'], 5)
    assert_equal(styles[0]['markeredgecolor'], (0.5,0,0))
    assert_equal(styles[0]['markeredgewidth'], 1)
    assert_equal(styles[0]['markerfacecolor'], (0.5,0,0))
    assert_equal(styles[0]['alpha'], 0.9)
    assert_equal(styles[0]['color'], (0.5,0,0))
    assert_equal(styles[0]['linestyle'], 'None')

    assert_equal(styles[1]['marker'], 's')
    assert_equal(styles[1]['markersize'], 5)
    assert_equal(styles[1]['markeredgecolor'], (0.5,0,0))
    assert_equal(styles[1]['markeredgewidth'], 1)
    assert_equal(styles[1]['markerfacecolor'], (0.5,0,0))
    assert_equal(styles[1]['alpha'], 0.9)
    assert_equal(styles[1]['color'], (0.5,0,0))
    assert_equal(styles[1]['linestyle'],'None')


    styles=a(markers=[1, 1], dashes=None, marker_colors=[0])
    assert_equal(styles[0]['marker'], 's')
    assert_equal(styles[0]['markersize'], 5)
    assert_equal(styles[0]['markeredgecolor'], (0,0.5,0))
    assert_equal(styles[0]['markeredgewidth'], 1)
    assert_equal(styles[0]['markerfacecolor'], (0,0.5,0))
    assert_equal(styles[0]['alpha'], 0.9)
    assert_equal(styles[0]['color'], (0.5,0,0))
    assert_equal(styles[0]['linestyle'], 'None')

    assert_equal(styles[1]['marker'], 's')
    assert_equal(styles[1]['markersize'], 5)
    assert_equal(styles[1]['markeredgecolor'], (0,0.5,0))
    assert_equal(styles[1]['markeredgewidth'], 1)
    assert_equal(styles[1]['markerfacecolor'], (0,0.5,0))
    assert_equal(styles[1]['alpha'], 0.9)
    assert_equal(styles[1]['color'], (0.5,0,0))
    assert_equal(styles[1]['linestyle'],'None')

    styles=a(markers=[1, 1], dashes=[1, 0], marker_colors=[0])
    assert_equal(styles[0]['marker'], 's')
    assert_equal(styles[0]['markersize'], 5)
    assert_equal(styles[0]['markeredgecolor'], (0,0.5,0))
    assert_equal(styles[0]['markeredgewidth'], 1)
    assert_equal(styles[0]['markerfacecolor'], (0,0.5,0))
    assert_equal(styles[0]['alpha'], 0.9)
    assert_equal(styles[0]['color'], (0.5,0,0))
    assert_equal(styles[0]['dashes'], [4, 4])

    assert_equal(styles[1]['marker'], 's')
    assert_equal(styles[1]['markersize'], 5)
    assert_equal(styles[1]['markeredgecolor'], (0,0.5,0))
    assert_equal(styles[1]['markeredgewidth'], 1)
    assert_equal(styles[1]['markerfacecolor'], (0,0.5,0))
    assert_equal(styles[1]['alpha'], 0.9)
    assert_equal(styles[1]['color'], (0.5,0,0))
    assert_equal(styles[1]['dashes'], (None, None))

    styles=a(markers=[1, 1], dashes=[1, None], marker_colors=[0])
    assert_equal(styles[0]['marker'], 's')
    assert_equal(styles[0]['markersize'], 5)
    assert_equal(styles[0]['markeredgecolor'], (0,0.5,0))
    assert_equal(styles[0]['markeredgewidth'], 1)
    assert_equal(styles[0]['markerfacecolor'], (0,0.5,0))
    assert_equal(styles[0]['alpha'], 0.9)
    assert_equal(styles[0]['color'], (0.5,0,0))
    assert_equal(styles[0]['dashes'], [4, 4])

    assert_equal(styles[1]['marker'], 's')
    assert_equal(styles[1]['markersize'], 5)
    assert_equal(styles[1]['markeredgecolor'], (0,0.5,0))
    assert_equal(styles[1]['markeredgewidth'], 1)
    assert_equal(styles[1]['markerfacecolor'], (0,0.5,0))
    assert_equal(styles[1]['alpha'], 0.9)
    assert_equal(styles[1]['color'], (0.5,0,0))
    assert_equal(styles[1]['linestyle'], 'None')

    styles=a(markers=[1, 1], dashes=[1, None], marker_colors=[0], line_colors=[1])
    assert_equal(styles[0]['marker'], 's')
    assert_equal(styles[0]['markersize'], 5)
    assert_equal(styles[0]['markeredgecolor'], (0,0.5,0))
    assert_equal(styles[0]['markeredgewidth'], 1)
    assert_equal(styles[0]['markerfacecolor'], (0,0.5,0))
    assert_equal(styles[0]['alpha'], 0.9)
    assert_equal(styles[0]['color'], (0,0,0.5))
    assert_equal(styles[0]['dashes'], [4, 4])

    assert_equal(styles[1]['marker'], 's')
    assert_equal(styles[1]['markersize'], 5)
    assert_equal(styles[1]['markeredgecolor'], (0,0.5,0))
    assert_equal(styles[1]['markeredgewidth'], 1)
    assert_equal(styles[1]['markerfacecolor'], (0,0.5,0))
    assert_equal(styles[1]['alpha'], 0.9)
    assert_equal(styles[1]['color'], (0,0,0.5))
    assert_equal(styles[1]['linestyle'], 'None')


    a.colors=[(0.4, 0.8, 0.2), (0.4, 0.8, 0.2)]
    a.shade_colors(factor=0.5)
    ok_(np.allclose(a.colors[0], (0.2, 0.4, 0.1)))
    ok_(np.allclose(a.colors[1], (0.2, 0.4, 0.1)))

    a.colors=[(0.4, 0.8, 0.2), (0.4, 0.8, 0.2)]
    a.tint_colors(factor=0.5)
    ok_(np.allclose(a.colors[0], (0.7, 0.9, 0.6)))
    ok_(np.allclose(a.colors[1], (0.7, 0.9, 0.6)))

@cleanup
def test_apply_dict_to_object():
    """test for apply_dict_to_object"""
    #apply_dict_to_object(obj, dic)
    fig = plt.figure(num=1)
    ax = fig.add_subplot(111, frame_on=True)
    for i in range(6):
        ax.plot([1,2,3],[3+i,7+i,6+i])

    obj = ax.get_lines()
    apply_dict_to_object(obj[0],{'marker':'^'})
    assert_equal(obj[0].get_marker(), '^')

    obj = ax.get_lines()[:2]
    apply_dict_to_object(obj,[{'marker':'s'}, {'marker':'h'}])
    assert_equal(obj[0].get_marker(), 's')
    assert_equal(obj[1].get_marker(), 'h')

def test_split_sequence_into_dict_and_nondicts():
    """test for split_sequence_into_dict_and_nondicts"""
    #split_sequence_into_dict_and_nondicts(*args)

    assert_equal(split_sequence_into_dict_and_nondicts({'a': 2, 'b': 3},
                                   4,
                                   {'a':8, 'c':5},
                                   5),
                                   ([4,5], {'a': 8, 'b': 3, 'c':5}))


def test_row_major_order_reverse_map():
    """test for row_major_order_reverse_map"""
    #row_major_order_reverse_map(shape, index_steps=None, transpose=False)
    ok_(np.allclose(row_major_order_reverse_map(shape=(3, 3), index_steps=None, transpose=False),
                 np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])))
    ok_(np.allclose(row_major_order_reverse_map(shape=(3, 3), index_steps=(-1,1), transpose=False),
                 np.array([6, 7, 8, 3, 4, 5, 0, 1, 2])))
    ok_(np.allclose(row_major_order_reverse_map(shape=(3, 3), index_steps=(1,-1), transpose=False),
                 np.array([2, 1, 0, 5, 4, 3, 8, 7, 6])))
    ok_(np.allclose(row_major_order_reverse_map(shape=(3, 3), index_steps=(-1,-1), transpose=False),
                 np.array([8, 7, 6, 5, 4, 3, 2, 1, 0])))
    ok_(np.allclose(row_major_order_reverse_map(shape=(3, 3), index_steps=None, transpose=True),
                 np.array([0, 3, 6, 1, 4, 7, 2, 5, 8])))






class test_xylabel_subplots(temp_cls):
    """tests for xylabel_subplots"""
    #xylabel_subplots(fig, y_axis_labels=None, x_axis_labels=None):


#    @cleanup
    def test_defaults(self):
        fig = plt.figure()
        ax1=fig.add_subplot('211')
        ax1.plot([1],[2])
        ax2=fig.add_subplot('212')
        ax2.plot([3],[4])

        xylabel_subplots(fig)

        assert_equal(fig.axes[0].get_xlabel(),"")
        assert_equal(fig.axes[1].get_xlabel(),"")
        assert_equal(fig.axes[0].get_ylabel(),"")
        assert_equal(fig.axes[1].get_ylabel(),"")

    def test_x_y_lables(self):
        fig = plt.figure()
        ax1=fig.add_subplot('211')
        ax1.plot([1],[2])
        ax2=fig.add_subplot('212')
        ax2.plot([3],[4])

        xylabel_subplots(fig,['y1','y2'],['x1','x2'])

        assert_equal(fig.axes[0].get_xlabel(),"x1")
        assert_equal(fig.axes[1].get_xlabel(),"x2")
        assert_equal(fig.axes[0].get_ylabel(),"y1")
        assert_equal(fig.axes[1].get_ylabel(),"y2")

class test_iterable_method_call(temp_cls):
    """tests for iterable_method_call"""
    #iterable_method_call(iterable, method, unpack, *args)


#    iterable_method_call(fig.axes, 'set_ylabel', *y_axis_labels)

    def test_one_to_one(self):
        fig = plt.figure()
        ax1=fig.add_subplot('311')
        ax1.plot([1],[2])
        ax2=fig.add_subplot('312')
        ax2.plot([3],[4])
        ax3=fig.add_subplot('313')
        ax3.plot([5],[6])

        iterable_method_call(fig.axes,
                             'set_xlabel',
                             False,
                             *['x1', None, 'x3'])

        assert_equal(fig.axes[0].get_xlabel(),"x1")
        assert_equal(fig.axes[1].get_xlabel(),"")
        assert_equal(fig.axes[2].get_xlabel(),"x3")

    def test_single_arg(self):
        fig = plt.figure()
        ax1=fig.add_subplot('311')
        ax1.plot([1],[2])
        ax2=fig.add_subplot('312')
        ax2.plot([3],[4])
        ax3=fig.add_subplot('313')
        ax3.plot([5],[6])

        iterable_method_call(fig.axes,
                             'set_xlabel',
                             False,
                             'xx')

        assert_equal(fig.axes[0].get_xlabel(),"xx")
        assert_equal(fig.axes[1].get_xlabel(),"xx")
        assert_equal(fig.axes[2].get_xlabel(),"xx")


    def test_unpack(self):

        fig = plt.figure()
        ax1=fig.add_subplot('311')
        ax1.plot([1],[2])
        ax2=fig.add_subplot('312')
        ax2.plot([3],[4])
        ax3=fig.add_subplot('313')
        ax3.plot([5],[6])

        #this is a bad test because with xlim expect the same answer from unpack =True or unpack=False
        iterable_method_call(fig.axes,
                             'set_xlim',
                             True,
                             (5,8), None, (1, 6))

        assert_equal(fig.axes[0].get_xlim(),(5,8))
#        assert_equal(fig.axes[1].get_xlim(),None)
        assert_equal(fig.axes[2].get_xlim(),(1,6))

class test_apply_markevery_to_sequence_of_lines(temp_cls):
    """tests for apply_markevery_to_sequence_of_lines"""
    #apply_markevery_to_sequence_of_lines(lines, markevery=None ,
#                                         random_start=True, seed=None)

    def test_defaults(self):

        fig=plt.figure()
        ax = fig.add_subplot('111')

        ax.plot([3,4], [8,6], markevery=4)
        ax.plot([3,5], [8,7], markevery=4)
        ax.plot([3,6], [8,8], markevery=4)

        apply_markevery_to_sequence_of_lines(ax.get_lines(), markevery=None ,
                                         random_start=True, seed=None)

        assert_equal(ax.get_lines()[0].get_markevery(), None)
        assert_equal(ax.get_lines()[1].get_markevery(), None)
        assert_equal(ax.get_lines()[2].get_markevery(), None)

    def test_int(self):


        fig=plt.figure()
        ax = fig.add_subplot('111')

        ax.plot([3,4], [8,6])
        ax.plot([3,5], [8,7])
        ax.plot([3,6], [8,8])

        apply_markevery_to_sequence_of_lines(ax.get_lines(), markevery=8,
                                         random_start=False, seed=None)

        assert_equal(ax.get_lines()[0].get_markevery(), 8)
        assert_equal(ax.get_lines()[1].get_markevery(), 8)
        assert_equal(ax.get_lines()[2].get_markevery(), 8)

    def test_int_random_start(self):

        fig=plt.figure()
        ax = fig.add_subplot('111')

        ax.plot([3,4], [8,6])
        ax.plot([3,5], [8,7])
        ax.plot([3,6], [8,8])

        apply_markevery_to_sequence_of_lines(ax.get_lines(), markevery=8,
                                         random_start=True, seed=1)

        assert_equal(ax.get_lines()[0].get_markevery(), (1,8))
        assert_equal(ax.get_lines()[1].get_markevery(), (7,8))
        assert_equal(ax.get_lines()[2].get_markevery(), (6,8))

    def test_int(self):


        fig=plt.figure()
        ax = fig.add_subplot('111')

        ax.plot([3,4], [8,6], markevery=4)
        ax.plot([3,5], [8,7], markevery=4)
        ax.plot([3,6], [8,8], markevery=4)

        apply_markevery_to_sequence_of_lines(ax.get_lines(), markevery=0.1,
                                         random_start=False, seed=None)

        try:

            from matplotlib.lines import _mark_every_path
            # the above import should fail for matplotlib <1.4.
            # However, since it was my addition to matplotlib I may have hardwired
            # all the markevery code it in to my own matplotlib.lines file.
            assert_equal(ax.get_lines()[0].get_markevery(), 0.1)
            assert_equal(ax.get_lines()[1].get_markevery(), 0.1)
            assert_equal(ax.get_lines()[2].get_markevery(), 0.1)
        except ImportError:
            assert_equal(ax.get_lines()[0].get_markevery(), 4)
            assert_equal(ax.get_lines()[1].get_markevery(), 4)
            assert_equal(ax.get_lines()[2].get_markevery(), 4)



    def test_float_random_start(self):

        fig=plt.figure()
        ax = fig.add_subplot('111')

        ax.plot([3,4], [8,6], markevery=4)
        ax.plot([3,5], [8,7], markevery=4)
        ax.plot([3,6], [8,8], markevery=4)

        apply_markevery_to_sequence_of_lines(ax.get_lines(), markevery=0.1,
                                         random_start=True, seed=1)


        try:
            from matplotlib.lines import _mark_every_path
            # the above import should fail for matplotlib <1.4.
            # However, since it was my addition to matplotlib I may have hardwired
            # all the markevery code it in to my own matplotlib.lines file.
            assert_allclose(ax.get_lines()[0].get_markevery(), (0.0134364244112, 0.1))
            assert_allclose(ax.get_lines()[1].get_markevery(), (0.0847433736937, 0.1))
            assert_allclose(ax.get_lines()[2].get_markevery(), (0.07637746189766, 0.1))
        except ImportError:
            assert_equal(ax.get_lines()[0].get_markevery(), 4)
            assert_equal(ax.get_lines()[1].get_markevery(), 4)
            assert_equal(ax.get_lines()[2].get_markevery(), 4)



class test_plot_vs_time(temp_cls):
    """tests for plot_vs_time"""

    #plot_vs_time(t, y, line_labels, prop_dict={})
    t = np.array([ 0.,  2.,  4.,  6.,  8.])
    y = np.array([[  1.,   2.,   3.],
                  [  3.,   4.,   5.],
                  [  5.,   6.,   7.],
                  [  7.,   8.,   9.],
                  [  9.,  10.,  11.]])

    def test_defaults(self):
        fig = plot_vs_time(self.t, self.y, None)
        ax = fig.get_axes()[0]

        assert_allclose(ax.get_lines()[0].get_xydata(),
                     np.array([[ 0.,  1.],
                               [ 2.,  3.],
                               [ 4.,  5.],
                               [ 6.,  7.],
                               [ 8.,  9.]]))
        assert_allclose(ax.get_lines()[1].get_xydata(),
                     np.array([[  0.,   2.],
                               [  2.,   4.],
                               [  4.,   6.],
                               [  6.,   8.],
                               [  8.,  10.]]))
        assert_allclose(ax.get_lines()[2].get_xydata(),
                     np.array([[  0.,   3.],
                               [  2.,   5.],
                               [  4.,   7.],
                               [  6.,   9.],
                               [  8.,  11.]]))

        assert_equal(ax.get_lines()[0].get_label(), '_line0')
        assert_equal(ax.get_lines()[1].get_label(), '_line1')
        assert_equal(ax.get_lines()[2].get_label(), '_line2')

        assert_equal(ax.get_xlabel(), 'Time, t')
        assert_equal(ax.get_ylabel(), 'y')

        assert_equal(ax.get_legend(), None)

    def test_line_labels(self):
        fig = plot_vs_time(self.t,self.y, ['a', 'b', 'c'])
        ax = fig.get_axes()[0]

        assert_allclose(ax.get_lines()[0].get_xydata(),
                     np.array([[ 0.,  1.],
                               [ 2.,  3.],
                               [ 4.,  5.],
                               [ 6.,  7.],
                               [ 8.,  9.]]))
        assert_allclose(ax.get_lines()[1].get_xydata(),
                     np.array([[  0.,   2.],
                               [  2.,   4.],
                               [  4.,   6.],
                               [  6.,   8.],
                               [  8.,  10.]]))
        assert_allclose(ax.get_lines()[2].get_xydata(),
                     np.array([[  0.,   3.],
                               [  2.,   5.],
                               [  4.,   7.],
                               [  6.,   9.],
                               [  8.,  11.]]))

        assert_equal(ax.get_lines()[0].get_label(), 'a')
        assert_equal(ax.get_lines()[1].get_label(), 'b')
        assert_equal(ax.get_lines()[2].get_label(), 'c')
        ok_(not ax.get_legend() is None)
        assert_equal(ax.get_legend().get_title().get_text(), 'Depth interval:')

    def test_propdict_xylabels(self):


        fig = plot_vs_time(self.t, self.y, ['a', 'b', 'c'],
                           prop_dict={'xlabel':'xxx', 'ylabel':'yyy'})
        ax = fig.get_axes()[0]

        assert_equal(ax.get_xlabel(), 'xxx')
        assert_equal(ax.get_ylabel(), 'yyy')

    def test_propdict_has_legend(self):


        fig = plot_vs_time(self.t, self.y, ['a', 'b', 'c'],
                           prop_dict={'has_legend': False})
        ax = fig.get_axes()[0]

        assert_equal(ax.get_legend(), None)


    def test_propdict_fig_prop_figsize(self):
        fig = plot_vs_time(self.t, self.y, ['a', 'b', 'c'],
                           prop_dict={'fig_prop':{'figsize':(8,9)}})
        assert_allclose(fig.get_size_inches(), (8,9))

    def test_propdict_legend_prop_title(self):
        fig = plot_vs_time(self.t, self.y, ['a', 'b', 'c'],
                           prop_dict={'legend_prop':{'title':'abc'}})
        ax = fig.get_axes()[0]
        assert_equal(ax.get_legend().get_title().get_text(), 'abc')




class test_plot_vs_depth(temp_cls):
    """tests for plot_vs_depth"""
    #plot_vs_depth(x, z, line_labels=None, H = 1.0, RLzero=None, prop_dict={})

    z = np.array([ 0.,  2.,  4.,  6.,  8.])
    x = np.array([[  1.,   2.,   3.],
                  [  3.,   4.,   5.],
                  [  5.,   6.,   7.],
                  [  7.,   8.,   9.],
                  [  9.,  10.,  11.]])

    def test_defaults(self):
        fig = plot_vs_depth(self.x, self.z, None)
        ax = fig.get_axes()[0]

        assert_allclose(ax.get_lines()[0].get_xydata(),
                     np.array([[ 1.,  0.],
                               [ 3.,  2.],
                               [ 5.,  4.],
                               [ 7.,  6.],
                               [ 9.,  8.]]))
        assert_allclose(ax.get_lines()[1].get_xydata(),
                     np.array([[  2.,   0.],
                               [  4.,   2.],
                               [  6.,   4.],
                               [  8.,   6.],
                               [ 10.,   8.]]))
        assert_allclose(ax.get_lines()[2].get_xydata(),
                     np.array([[  3.,   0.],
                               [  5.,   2.],
                               [  7.,   4.],
                               [  9.,   6.],
                               [ 11.,   8.]]))

        assert_equal(ax.get_lines()[0].get_label(), '_line0')
        assert_equal(ax.get_lines()[1].get_label(), '_line1')
        assert_equal(ax.get_lines()[2].get_label(), '_line2')

        assert_equal(ax.get_xlabel(), 'x')
        assert_equal(ax.get_ylabel(), 'Depth, z')

        assert_equal(ax.get_legend(), None)

    def test_line_labels(self):
        fig = plot_vs_depth(self.x,self.z, ['a', 'b', 'c'])
        ax = fig.get_axes()[0]

        assert_allclose(ax.get_lines()[0].get_xydata(),
                     np.array([[ 1.,  0.],
                               [ 3.,  2.],
                               [ 5.,  4.],
                               [ 7.,  6.],
                               [ 9.,  8.]]))
        assert_allclose(ax.get_lines()[1].get_xydata(),
                     np.array([[  2.,   0.],
                               [  4.,   2.],
                               [  6.,   4.],
                               [  8.,   6.],
                               [ 10.,   8.]]))
        assert_allclose(ax.get_lines()[2].get_xydata(),
                     np.array([[  3.,   0.],
                               [  5.,   2.],
                               [  7.,   4.],
                               [  9.,   6.],
                               [ 11.,   8.]]))

        assert_equal(ax.get_lines()[0].get_label(), 'a')
        assert_equal(ax.get_lines()[1].get_label(), 'b')
        assert_equal(ax.get_lines()[2].get_label(), 'c')
        ok_(not ax.get_legend() is None)
        assert_equal(ax.get_legend().get_title().get_text(), 'time:')

    def test_propdict_xylabels(self):


        fig = plot_vs_depth(self.x, self.z, ['a', 'b', 'c'],
                           prop_dict={'xlabel':'xxx', 'ylabel':'yyy'})
        ax = fig.get_axes()[0]

        assert_equal(ax.get_xlabel(), 'xxx')
        assert_equal(ax.get_ylabel(), 'yyy')

    def test_propdict_has_legend(self):


        fig = plot_vs_depth(self.x, self.z, ['a', 'b', 'c'],
                           prop_dict={'has_legend': False})
        ax = fig.get_axes()[0]

        assert_equal(ax.get_legend(), None)


    def test_propdict_fig_prop_figsize(self):
        fig = plot_vs_depth(self.x, self.z, ['a', 'b', 'c'],
                           prop_dict={'fig_prop':{'figsize':(8,9)}})
        assert_allclose(fig.get_size_inches(), (8,9))

    def test_propdict_legend_prop_title(self):
        fig = plot_vs_depth(self.x, self.z, ['a', 'b', 'c'],
                           prop_dict={'legend_prop':{'title':'abc'}})
        ax = fig.get_axes()[0]
        assert_equal(ax.get_legend().get_title().get_text(), 'abc')

    def test_H(self):
        fig = plot_vs_depth(self.x, self.z, None, H=2.0)
        ax = fig.get_axes()[0]

        assert_allclose(ax.get_lines()[0].get_xydata(),
                     np.array([[ 1.,  0.],
                               [ 3.,  4.],
                               [ 5.,  8.],
                               [ 7.,  12.],
                               [ 9.,  16.]]))
        assert_allclose(ax.get_lines()[1].get_xydata(),
                     np.array([[  2.,   0.],
                               [  4.,   4.],
                               [  6.,   8.],
                               [  8.,   12.],
                               [ 10.,   16.]]))
        assert_allclose(ax.get_lines()[2].get_xydata(),
                     np.array([[  3.,   0.],
                               [  5.,   4.],
                               [  7.,   8.],
                               [  9.,   12.],
                               [ 11.,   16.]]))

        assert_equal(ax.get_lines()[0].get_label(), '_line0')
        assert_equal(ax.get_lines()[1].get_label(), '_line1')
        assert_equal(ax.get_lines()[2].get_label(), '_line2')

        assert_equal(ax.get_xlabel(), 'x')
        assert_equal(ax.get_ylabel(), 'Depth, z')

        assert_equal(ax.get_legend(), None)

    def test_Rlzero(self):
        fig = plot_vs_depth(self.x, self.z, None, H=2.0, RLzero=0)
        ax = fig.get_axes()[0]

        assert_allclose(ax.get_lines()[0].get_xydata(),
                     np.array([[ 1.,  0.],
                               [ 3.,  -4.],
                               [ 5.,  -8.],
                               [ 7.,  -12.],
                               [ 9.,  -16.]]))
        assert_allclose(ax.get_lines()[1].get_xydata(),
                     np.array([[  2.,   0.],
                               [  4.,   -4.],
                               [  6.,   -8.],
                               [  8.,   -12.],
                               [ 10.,   -16.]]))
        assert_allclose(ax.get_lines()[2].get_xydata(),
                     np.array([[  3.,   0.],
                               [  5.,   -4.],
                               [  7.,   -8.],
                               [  9.,   -12.],
                               [ 11.,   -16.]]))

        assert_equal(ax.get_lines()[0].get_label(), '_line0')
        assert_equal(ax.get_lines()[1].get_label(), '_line1')
        assert_equal(ax.get_lines()[2].get_label(), '_line2')

        assert_equal(ax.get_xlabel(), 'x')
        assert_equal(ax.get_ylabel(), 'RL')

        assert_equal(ax.get_legend(), None)
if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])
#    plt.show()