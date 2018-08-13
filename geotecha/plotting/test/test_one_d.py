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
from mock import patch
import random
#from matplotlib.testing.decorators import cleanup
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
from geotecha.plotting.one_d import plot_single_material_vs_depth
from geotecha.plotting.one_d import plot_generic_loads
from geotecha.plotting.one_d import plot_data_in_grid


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
#    ok_(copy_dict({'a':7, 'b':12}, {'c':13})=={'a':7, 'b':12, 'c':13})
    assert_equal(copy_dict({'a':7, 'b':12}, {'c':13}),
                 {'a':7, 'b':12, 'c':13})
#    ok_(copy_dict({'a':7, 'b':12}, {'a':21, 'c':13})=={'a':21, 'b':12, 'c':13})
    assert_equal(copy_dict({'a':7, 'b':12}, {'a':21, 'c':13}),
                 {'a':21, 'b':12, 'c':13})

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

#@cleanup
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

    @patch.object(random, 'randint')
    def test_int_random_start(self,
                              mock_randint):

        mock_randint.return_value=2
        fig=plt.figure()
        ax = fig.add_subplot('111')

        ax.plot([3,4], [8,6])
        ax.plot([3,5], [8,7])
        ax.plot([3,6], [8,8])

        apply_markevery_to_sequence_of_lines(ax.get_lines(), markevery=8,
                                         random_start=True, seed=1)
        # using mock here is a bit of a cop out but randint, even
        # with the same seed, gives different answeres in
        # py34 vs py27
        assert_equal(ax.get_lines()[0].get_markevery(), (2,8))
        assert_equal(ax.get_lines()[1].get_markevery(), (2,8))
        assert_equal(ax.get_lines()[2].get_markevery(), (2,8))

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


    def test_prop_dict_styles(self):
        fig = plot_vs_time(self.t, self.y, ['a', 'b', 'c'],
                           prop_dict={'styles':[{'markersize': 12,
                                                 'marker': '^'},
                                                 {'markersize': 3,
                                                 'marker': 's'}]})

        ax = fig.get_axes()[0]
        line1= ax.get_lines()[0]
        line2= ax.get_lines()[1]
        line3= ax.get_lines()[2]



        assert_equal(line1.get_marker(),'^')
        assert_equal(line1.get_markersize(),12)
        assert_equal(line2.get_marker(),'s')
        assert_equal(line2.get_markersize(),3)
        assert_equal(line3.get_marker(),'^')
        assert_equal(line3.get_markersize(),12)

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

    def test_prop_dict_styles(self):
        fig = plot_vs_depth(self.x, self.z, None,
                            prop_dict={'styles':[{'markersize': 12,
                                                 'marker': '^'},
                                                 {'markersize': 3,
                                                 'marker': 's'}]})
        ax = fig.get_axes()[0]
        line1= ax.get_lines()[0]
        line2= ax.get_lines()[1]
        line3= ax.get_lines()[2]



        assert_equal(line1.get_marker(),'^')
        assert_equal(line1.get_markersize(),12)
        assert_equal(line2.get_marker(),'s')
        assert_equal(line2.get_markersize(),3)
        assert_equal(line3.get_marker(),'^')
        assert_equal(line3.get_markersize(),12)

class test_plot_single_material_vs_depth(temp_cls):
    """tests for plot_single_material_vs_depth"""

#    plot_single_material_vs_depth(z_x, xlabels, H = 1.0, RLzero=None,
#                    prop_dict={})

    z1 = np.array([ 0. ,  0.5,  1. ])
    x1 = np.array([ 1. ,  1.5,  2. ])
    x2 = np.array([ 2. ,  2.5,  3. ])

    a = PolyLine(z1, x1)
    b = PolyLine(z1, x2)

    xlabels = ['a','b']

    def test_defaults(self):
        fig = plot_single_material_vs_depth((self.a, self.b), self.xlabels)

        assert_equal(len(fig.get_axes()), 2)
        ax1 = fig.get_axes()[0]
        line1= ax1.get_lines()[0]

        ax2 = fig.get_axes()[1]
        line2 = ax2.get_lines()[0]

        assert_allclose(line1.get_xydata()[0],
                     np.array([ 1., 0.]))
        assert_allclose(line1.get_xydata()[-1],
                     np.array([ 2., 1]))

        assert_allclose(line2.get_xydata()[0],
                     np.array([ 2., 0.]))
        assert_allclose(line2.get_xydata()[-1],
                     np.array([ 3., 1]))

        assert_equal(ax1.get_xlabel(), 'a')
        assert_equal(ax2.get_xlabel(), 'b')

        assert_equal(ax1.get_ylabel(), 'Depth, z')
        assert_equal(ax2.get_ylabel(), '')

    def test_prop_dict_ylabel(self):
        fig = plot_single_material_vs_depth((self.a, self.b),
                                            self.xlabels,
                                            prop_dict={'ylabel': 'hello'})
        ax1 = fig.get_axes()[0]

        assert_equal(ax1.get_ylabel(), 'hello')

    def test_H(self):

        fig = plot_single_material_vs_depth((self.a, self.b), self.xlabels,
                                            H=2)

        ax1 = fig.get_axes()[0]
        line1= ax1.get_lines()[0]

        ax2 = fig.get_axes()[1]
        line2 = ax2.get_lines()[0]

        assert_allclose(line1.get_xydata()[0],
                     np.array([ 1., 0.]))
        assert_allclose(line1.get_xydata()[-1],
                     np.array([ 2., 2]))

        assert_allclose(line2.get_xydata()[0],
                     np.array([ 2., 0.]))
        assert_allclose(line2.get_xydata()[-1],
                     np.array([ 3., 2]))
    def test_RLzero(self):

        fig = plot_single_material_vs_depth((self.a, self.b), self.xlabels,
                                            H=2, RLzero=1)

        ax1 = fig.get_axes()[0]
        line1= ax1.get_lines()[0]

        ax2 = fig.get_axes()[1]
        line2 = ax2.get_lines()[0]

        assert_allclose(line1.get_xydata()[0],
                     np.array([ 1., 1]))
        assert_allclose(line1.get_xydata()[-1],
                     np.array([ 2., -1]))

        assert_allclose(line2.get_xydata()[0],
                     np.array([ 2., 1]))
        assert_allclose(line2.get_xydata()[-1],
                     np.array([ 3., -1]))

        assert_equal(ax1.get_ylabel(), 'RL')

    def test_propdict_fig_prop_figsize(self):
        fig = plot_single_material_vs_depth((self.a, self.b), self.xlabels,

                           prop_dict={'fig_prop':{'figsize':(8,9)}})
        assert_allclose(fig.get_size_inches(), (8,9))



    def test_prop_dict_styles(self):
        fig = plot_single_material_vs_depth((self.a, self.b), self.xlabels,

                           prop_dict={'styles':[{'markersize': 12,
                                                 'marker': '^'}]})
        ax1 = fig.get_axes()[0]
        line1= ax1.get_lines()[0]

        ax2 = fig.get_axes()[1]
        line2 = ax2.get_lines()[0]

        assert_equal(line1.get_marker(),'^')
        assert_equal(line1.get_markersize(),12)
        assert_equal(line2.get_marker(),'^')
        assert_equal(line2.get_markersize(),12)

class test_plot_generic_loads(temp_cls):
    """tests for plot_generic_loads"""
#    plot_generic_loads(load_triples, load_names, ylabels=None,
#                        trange = None, H = 1.0, RLzero=None, prop_dict={})

    vs_time1 = PolyLine(np.array([0,10]), np.array([0,1]))
    vs_depth1 = PolyLine(np.array([0,0.4,1]), np.array([1.0,1.0,0.5]))
    omega_phase1 = (0.5, 0.3)
    triple1 = (vs_time1, vs_depth1, omega_phase1)

    vs_time2 = PolyLine(np.array([0,9]), np.array([0,2]))
    vs_depth2 = PolyLine(np.array([0,0.4,1]), np.array([1.0,1.0,0.8]))
    omega_phase2 = None
    triple2 = (vs_time2, vs_depth2, omega_phase2)

    load_names=['a', 'b']

    def test_defaults(self):

        fig = plot_generic_loads([[self.triple1], [self.triple2]],
                                  load_names=self.load_names)

        assert_equal(len(fig.get_axes()), 4)

        ax1 = fig.get_axes()[0]
        line1= ax1.get_lines()[0]
        ax2 = fig.get_axes()[1]
        line2 = ax2.get_lines()[0]
        ax3 = fig.get_axes()[2]
        line3 = ax3.get_lines()[0]
        ax4 = fig.get_axes()[3]
        line4 = ax4.get_lines()[0]

        #first row of charts
        assert_allclose(line1.get_xydata()[0],
                     np.array([ 0, 0]))
        assert_allclose(line1.get_xydata()[-1],
                     np.array([ 10, 1*np.cos(0.5*10+0.3)]))

        assert_allclose(line2.get_xydata()[0],
                     np.array([ 1, 0]))
        assert_allclose(line2.get_xydata()[-1],
                     np.array([ 0.5, 1]))

        #2nd row of charts
        assert_allclose(line3.get_xydata()[0],
                     np.array([ 0, 0]))
        assert_allclose(line3.get_xydata()[-1],
                     np.array([ 9, 2]))

        assert_allclose(line4.get_xydata()[0],
                     np.array([ 1, 0]))
        assert_allclose(line4.get_xydata()[-1],
                     np.array([ 0.8, 1]))


        assert_equal(ax1.get_xlabel(), '')
        assert_equal(ax1.get_ylabel(), 'y0')
        assert_equal(ax2.get_xlabel(), '')
        assert_equal(ax2.get_ylabel(), 'Depth, z')

        assert_equal(ax3.get_xlabel(), 'Time')
        assert_equal(ax3.get_ylabel(), 'y1')
        assert_equal(ax4.get_xlabel(), 'Load factor')
        assert_equal(ax4.get_ylabel(), 'Depth, z')

        assert_equal(line1.get_label(), 'a0')
        assert_equal(line2.get_label(), 'a0')
        assert_equal(line3.get_label(), 'b0')
        assert_equal(line4.get_label(), 'b0')

        ok_(not ax1.get_legend() is None)
        ok_(not ax3.get_legend() is None)

    def test_ylabels(self):
        fig = plot_generic_loads([[self.triple1], [self.triple2]],
                                  load_names=self.load_names,
                                  ylabels = ['one', 'two'])


        ax1 = fig.get_axes()[0]
        ax3 = fig.get_axes()[2]

        assert_equal(ax1.get_ylabel(), 'one')
        assert_equal(ax3.get_ylabel(), 'two')

    def test_trange(self):
        fig = plot_generic_loads([[self.triple1], [self.triple2]],
                                  load_names=self.load_names,
                                  trange = (2,3))

        ax1 = fig.get_axes()[0]
        ax3 = fig.get_axes()[2]

        assert_allclose(ax1.get_xlim(), (2,3))
        assert_allclose(ax3.get_xlim(), (2,3))

    def test_H(self):

        fig = plot_generic_loads([[self.triple1], [self.triple2]],
                                  load_names=self.load_names,
                                  H=2.0)

        ax1 = fig.get_axes()[0]
        line1= ax1.get_lines()[0]
        ax2 = fig.get_axes()[1]
        line2 = ax2.get_lines()[0]
        ax3 = fig.get_axes()[2]
        line3 = ax3.get_lines()[0]
        ax4 = fig.get_axes()[3]
        line4 = ax4.get_lines()[0]

        #first row of charts
        assert_allclose(line1.get_xydata()[0],
                     np.array([ 0, 0]))
        assert_allclose(line1.get_xydata()[-1],
                     np.array([ 10, 1*np.cos(0.5*10+0.3)]))

        assert_allclose(line2.get_xydata()[0],
                     np.array([ 1, 0]))
        assert_allclose(line2.get_xydata()[-1],
                     np.array([ 0.5, 2]))

        #2nd row of charts
        assert_allclose(line3.get_xydata()[0],
                     np.array([ 0, 0]))
        assert_allclose(line3.get_xydata()[-1],
                     np.array([ 9, 2]))

        assert_allclose(line4.get_xydata()[0],
                     np.array([ 1, 0]))
        assert_allclose(line4.get_xydata()[-1],
                     np.array([ 0.8, 2]))

    def test_RLzero(self):

        fig = plot_generic_loads([[self.triple1], [self.triple2]],
                                  load_names=self.load_names,
                                  H=2.0, RLzero=1)

        ax1 = fig.get_axes()[0]
        line1= ax1.get_lines()[0]
        ax2 = fig.get_axes()[1]
        line2 = ax2.get_lines()[0]
        ax3 = fig.get_axes()[2]
        line3 = ax3.get_lines()[0]
        ax4 = fig.get_axes()[3]
        line4 = ax4.get_lines()[0]

        #first row of charts
        assert_allclose(line1.get_xydata()[0],
                     np.array([ 0, 0]))
        assert_allclose(line1.get_xydata()[-1],
                     np.array([ 10, 1*np.cos(0.5*10+0.3)]))

        assert_allclose(line2.get_xydata()[0],
                     np.array([ 1, 1]))
        assert_allclose(line2.get_xydata()[-1],
                     np.array([ 0.5, -1]))

        #2nd row of charts
        assert_allclose(line3.get_xydata()[0],
                     np.array([ 0, 0]))
        assert_allclose(line3.get_xydata()[-1],
                     np.array([ 9, 2]))

        assert_allclose(line4.get_xydata()[0],
                     np.array([ 1, 1]))
        assert_allclose(line4.get_xydata()[-1],
                     np.array([ 0.8, -1]))


        assert_equal(ax2.get_ylabel(), 'RL')
        assert_equal(ax4.get_ylabel(), 'RL')

    def test_propdict_fig_prop_figsize(self):
        fig = plot_generic_loads([[self.triple1], [self.triple2]],
                                  load_names=self.load_names,
                           prop_dict={'fig_prop':{'figsize':(8,9)}})

        assert_allclose(fig.get_size_inches(), (8,9))


    def test_propdict_has_legend(self):


        fig = plot_generic_loads([[self.triple1], [self.triple2]],
                                  load_names=self.load_names,
                           prop_dict={'has_legend': False})
        ax1 = fig.get_axes()[0]
        ax3 = fig.get_axes()[2]

        assert_equal(ax1.get_legend(), None)
        assert_equal(ax3.get_legend(), None)

    def test_prop_dict_styles(self):
        fig = plot_generic_loads([[self.triple1], [self.triple2]],
                                  load_names=self.load_names,
                           prop_dict={'styles':[{'markersize': 12,
                                                 'marker': '^'},
                                                 {'markersize': 3,
                                                 'marker': 's'}]})

        ax1 = fig.get_axes()[0]
        line1= ax1.get_lines()[0]
        ax2 = fig.get_axes()[1]
        line2 = ax2.get_lines()[0]
        ax3 = fig.get_axes()[2]
        line3 = ax3.get_lines()[0]
        ax4 = fig.get_axes()[3]
        line4 = ax4.get_lines()[0]

        assert_equal(line1.get_marker(),'^')
        assert_equal(line1.get_markersize(),12)
        assert_equal(line2.get_marker(),'^')
        assert_equal(line2.get_markersize(),12)

        assert_equal(line3.get_marker(),'s')
        assert_equal(line3.get_markersize(),3)
        assert_equal(line4.get_marker(),'s')
        assert_equal(line4.get_markersize(),3)

    def test_prop_dict_time_axis_label(self):
        fig = plot_generic_loads([[self.triple1], [self.triple2]],
                                  load_names=self.load_names,
                           prop_dict={'time_axis_label': 'hello'})

        ax1 = fig.get_axes()[0]
        ax3 = fig.get_axes()[2]
        assert_equal(ax1.get_xlabel(), '')
        assert_equal(ax3.get_xlabel(), 'hello')

    def test_prop_dict_depth_axis_label(self):
        fig = plot_generic_loads([[self.triple1], [self.triple2]],
                                  load_names=self.load_names,
                           prop_dict={'depth_axis_label': 'hello'})

        ax2 = fig.get_axes()[1]
        ax4 = fig.get_axes()[3]
        assert_equal(ax2.get_xlabel(), '')
        assert_equal(ax4.get_xlabel(), 'hello')

    def test_propdict_legend_prop_title(self):
        fig = plot_generic_loads([[self.triple1], [self.triple2]],
                                  load_names=self.load_names,
                           prop_dict={'legend_prop':{'title':'abc'}})
        ax1 = fig.get_axes()[0]
        ax3 = fig.get_axes()[2]
        assert_equal(ax1.get_legend().get_title().get_text(), 'abc')
        assert_equal(ax3.get_legend().get_title().get_text(), 'abc')


class test_plot_data_in_grid(temp_cls):
    """tests for plot_data_in_grid"""


    xa1 = np.array([1,2])
    ya1 = np.array([3,4])
    xa2 = np.array([4,5])
    ya2 = np.array([5,6])

    xb1 = np.array([6,7])
    yb1 = np.array([[7,9],
                    [8,4]])

    gs = mpl.gridspec.GridSpec(3,3)


    def test_defaults(self):
        fig = plt.figure()
        data = [([self.xa1, self.ya1], [self.xa2, self.ya2]),
                ([self.xb1, self.yb1],)]
        plot_data_in_grid(fig, data, self.gs)

        assert_equal(len(fig.get_axes()), 2)
        ax1 = fig.get_axes()[0]
        line_a1 = ax1.get_lines()[0]
        line_a2 = ax1.get_lines()[1]
        ax2 = fig.get_axes()[1]
        line_b1 = ax2.get_lines()[0]
        line_b2 = ax2.get_lines()[1]

        assert_allclose(line_a1.get_xydata(),
                     np.array([[1, 3],
                               [2, 4]]))
        assert_allclose(line_a2.get_xydata(),
                     np.array([[4, 5],
                               [5, 6]]))
        assert_allclose(line_b1.get_xydata(),
                     np.array([[6, 7],
                               [7, 8]]))
        assert_allclose(line_b2.get_xydata(),
                     np.array([[6, 9],
                               [7, 4]]))

        assert_equal(ax1.get_subplotspec().get_geometry(), (3,3,0, 0))
        assert_equal(ax2.get_subplotspec().get_geometry(), (3,3,1, 1))


    def test_plot_args_scatter(self):
        fig = plt.figure()
        data = [([self.xa1, self.ya1,{'plot_type': 'scatter'}], [self.xa2, self.ya2]),
                ([self.xb1, self.yb1],)]

        plot_data_in_grid(fig, data, self.gs)

        assert_equal(len(fig.get_axes()), 2)
        ax1 = fig.get_axes()[0]

        ax2 = fig.get_axes()[1]

        #I don't know how to explicitly test for scatter so...
        assert_equal(len(ax1.get_lines()), 1)
        assert_equal(len(ax2.get_lines()), 2)

    def test_plot_args_marker(self):
        fig = plt.figure()
        data = [([self.xa1, self.ya1,{'marker': 's'}], [self.xa2, self.ya2]),
                ([self.xb1, self.yb1],)]

        plot_data_in_grid(fig, data, self.gs)

        assert_equal(len(fig.get_axes()), 2)
        ax1 = fig.get_axes()[0]
        line1 = ax1.get_lines()[0]
        assert_equal(line1.get_marker(), 's')



    def test_gs_index(self):
        fig = plt.figure()
        data = [([self.xa1, self.ya1], [self.xa2, self.ya2]),
                ([self.xb1, self.yb1],)]
        gs_index=[3,0]
        plot_data_in_grid(fig, data, self.gs, gs_index)

        ax1 = fig.get_axes()[0]
        ax2 = fig.get_axes()[1]
        #get_geometry give (nrows, ncols, start_index, end_index)
        # end_index is None if the gs doesn't span any positions.
        assert_equal(ax1.get_subplotspec().get_geometry(), (3,3,3, 3))
        assert_equal(ax2.get_subplotspec().get_geometry(), (3,3,0, 0))

    def test_gs_index_span(self):
        fig = plt.figure()
        data = [([self.xa1, self.ya1], [self.xa2, self.ya2]),
                ([self.xb1, self.yb1],)]
        gs_index=[slice(5, 9), 2]
        plot_data_in_grid(fig, data, self.gs, gs_index)

        ax1 = fig.get_axes()[0]
        ax2 = fig.get_axes()[1]

        assert_equal(ax1.get_subplotspec().get_geometry(), (3, 3, 5, 8))
        assert_equal(ax2.get_subplotspec().get_geometry(), (3,3, 2, 2))




if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])
#    plt.show()