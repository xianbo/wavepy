#!/usr/bin/env python
# -*- coding: utf-8 -*-
# #########################################################################
# Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

"""
Utility functions to help.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm  # progress bar

__authors__ = "Walan Grizolli"
__copyright__ = "Copyright (c) 2016, Affiliation"
__version__ = "0.1.0"
__docformat__ = "restructuredtext en"
__all__ = ['function_01',
           'function_02']


def color_print(message, color='red',
                highlights='on_white', attrs=''):
    """
        Print with colored characters. It is only a alias for colored print in the package termcolor

        Parameters
        ----------
        message : str
            Message to print.
        color, highlights: str
            see options at https://pypi.python.org/pypi/termcolor
        attrs: list
    """
    import termcolor
    print((termcolor.colored(message, color, highlights, attrs=attrs)))


def color_print_red(message):
    """
        Print with red characters. It is only a alias for colored print in the package termcolor

        Parameters
        ----------
        message : str
            Message to print.
    """
    import termcolor
    print((termcolor.colored(message, color='red')))


def color_print_blue(message):
    """
        Print with red characters. It is only a alias for colored print in the package termcolor

        Parameters
        ----------
        message : str
            Message to print.
    """
    import termcolor
    print((termcolor.colored(message, 'blue')))


############
# Plot Tools
############


def _fwhm_xy(xvalues, yvalues):
    """
    Calculate FWHM of a vector  y(x)

    Parameters
    ----------
    xvalues : ndarray
        vector with the values of x
    yvalues : ndarray
        vector with the values of x

    Returns
    -------
    list
        list of values x and y(x) at half maximum in the format
        [[fwhm_x1, fwhm_x2],[fwhm_y1, fwhm_y2]]
    """

    from scipy.interpolate import UnivariateSpline
    spline = UnivariateSpline(xvalues, yvalues - np.min(yvalues) / 2 - np.max(yvalues) / 2, s=0)
    # find the roots and return

    xvalues = spline.roots().tolist()
    yvalues = (spline(spline.roots()) + np.min(yvalues) / 2 + np.max(yvalues) / 2).tolist()

    if len(xvalues) == 2:
        return [xvalues, yvalues]
    else: return[[], []]


def plot_profile(xmatrix, ymatrix, zmatrix,
                 xlabel='x', ylabel='y', zlabel='z', title='Title', xo=None, yo=None,
                 xunit='', yunit='', do_fwhm=True,
                 arg4main=None, arg4top=None, arg4side=None):
    """
        Plot contourf in the main graph plus profiles over vertical and horizontal line defined by mouse.

        Parameters
        ----------

    """

    if arg4side is None:
        arg4side = {}
    if arg4top is None:
        arg4top = {}
    if arg4main is None:
        arg4main = {}
    from matplotlib.widgets import Cursor

    z_min, z_max = float(np.nanmin(zmatrix)), float(np.nanmax(zmatrix))

    fig = plt.figure(figsize=(12., 10.))
    fig.suptitle(title, fontsize=14, weight='bold')

    # Main contourf plot
    main_subplot = plt.subplot2grid((4, 5), (1, 1), rowspan=3, colspan=3)
    ax_main = fig.gca()
    ax_main.minorticks_on()
    plt.grid(True)
    ax_main.get_yaxis().set_tick_params(which='both', direction='out')
    ax_main.get_xaxis().set_tick_params(which='both', direction='out')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    main_plot = main_subplot.contourf(xmatrix, ymatrix, zmatrix, 256, **arg4main)

    #    main_subplot.contour(xmatrix, ymatrix, zmatrix, 256, cmap='jet', lw=0.1)

    colorbar_subplot = plt.subplot2grid((4, 20), (1, 0), rowspan=3, colspan=1)
    plt.colorbar(main_plot, cax=colorbar_subplot)

    # Top graph, horizontal profile. Empty, wait data from cursor on the graph.
    top_subplot = plt.subplot2grid((4, 5), (0, 1), rowspan=1, colspan=3)
    ax_top = fig.gca()
    ax_top.set_xticklabels([])
    plt.minorticks_on()
    plt.grid(True, which='both', axis='both')
    plt.ylabel(zlabel)
    plt.yticks(np.linspace(z_min, z_max, 3))
    plt.ylim(z_min, 1.05 * z_max)

    # Side graph, vertical profile. Empty, wait data from cursor on the graph.
    ax_side = side_subplot = plt.subplot2grid((4, 5), (1, 4), rowspan=3, colspan=1)
    ax_side.set_yticklabels([])
    plt.minorticks_on()
    plt.grid(True, which='both', axis='both')
    plt.xlabel(zlabel)
    ax_side.xaxis.set_label_position('top')
    plt.xticks(np.linspace(z_min, z_max, 3), rotation=-90)
    plt.xlim(z_min, 1.05 * z_max)

    def onclick(event):
        if event.xdata is not None and event.ydata is not None and event.button == 2:
            return plot_profiles_at(event.xdata, event.ydata)

        if event.button == 3:
            for j in [main_subplot, top_subplot, side_subplot]:
                j.lines = []
                j.legend_ = None

            plt.draw()

    def plot_profiles_at(_xo, _yo):

        # catch the x and y position to draw the profile
        _xo = xmatrix[1, np.argmin(np.abs(xmatrix[1, :] - _xo))]
        _yo = ymatrix[np.argmin(np.abs(ymatrix[:, 1] - _yo)), 1]
        # print('xo: %.4f, yo: %.4f' % (xo, yo))

        # plot the vertical and horiz. profiles that pass at xo and yo
        lines = top_subplot.plot(xmatrix[ymatrix == _yo], zmatrix[ymatrix == _yo],
                                 lw=2, drawstyle='steps-mid', **arg4top)

        side_subplot.plot(zmatrix[xmatrix == _xo],
                          ymatrix[xmatrix == _xo],
                          lw=2, drawstyle='steps-mid', **arg4side)

        # plot the vertical and horz. lines in the main graph
        last_color = lines[0].get_color()
        main_subplot.axhline(_yo, ls='--', lw=2, color=last_color)
        main_subplot.axvline(_xo, ls='--', lw=2, color=last_color)

        message = r'$x_o = %.4g %s$' % (xo, xunit)
        message = message + '\n' + r'$y_o = %.4g %s$' % (yo, yunit)

        main_subplot_x_min, main_subplot_x_max = main_subplot.get_xlim()
        main_subplot_y_min, main_subplot_y_max = main_subplot.get_ylim()

        # calculate and plot the FWHM
        _delta_x = None
        _delta_y = None

        if do_fwhm:
            [fwhm_top_x, fwhm_top_y] = _fwhm_xy(xmatrix[(ymatrix == _yo) &
                                                (xmatrix > main_subplot_x_min) &
                                                (xmatrix < main_subplot_x_max)],
                                                zmatrix[(ymatrix == _yo) &
                                                (xmatrix > main_subplot_x_min) &
                                                (xmatrix < main_subplot_x_max)])

            [fwhm_side_x, fwhm_side_y] = _fwhm_xy(ymatrix[(xmatrix == _xo) &
                                                  (ymatrix > main_subplot_y_min) &
                                                  (ymatrix < main_subplot_y_max)],
                                                  zmatrix[(xmatrix == _xo) &
                                                  (ymatrix > main_subplot_y_min) &
                                                  (ymatrix < main_subplot_y_max)])

            if len(fwhm_top_x) == 2:
                _delta_x = abs(fwhm_top_x[0] - fwhm_top_x[1])
                print('fwhm_x: %.4f' % _delta_x)
                message = message + '\n' + r'$FWHM_x = %.4g %s' % (_delta_x, xunit) + '$'
                top_subplot.plot(fwhm_top_x, fwhm_top_y, 'r--+', lw=1.5, ms=15, mew=1.4)

            if len(fwhm_side_x) == 2:
                _delta_y = abs(fwhm_side_x[0] - fwhm_side_x[1])
                print('fwhm_y: %.4f\n' % _delta_y)
                message = message + '\n' + r'$FWHM_y = {0:.4g} {1:s}'.format(_delta_y, yunit) + '$'
                side_subplot.plot(fwhm_side_y, fwhm_side_x, 'r--+', lw=1.5, ms=15, mew=1.4)

        # adjust top and side graphs to the zoom of the main graph

        top_subplot.set_xlim(main_subplot_x_min, main_subplot_x_max)
        side_subplot.set_ylim(main_subplot_y_min, main_subplot_y_max)

        plt.gcf().texts = []
        plt.gcf().text(.8, .75, message, fontsize=14, va='bottom',
                       bbox=dict(facecolor=last_color, alpha=0.5))

        fig.suptitle(title, fontsize=14, weight='bold')

        plt.show()

        return [_delta_x, _delta_y]

    [delta_x, delta_y] = [None, None]
    if xo is None and yo is None:
        # cursor on the main graph
        Cursor(ax_main, useblit=True, color='red', linewidth=2)
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show(block=True)
    else:
        [delta_x, delta_y] = plot_profiles_at(xo, yo)

    return [ax_main, ax_top, ax_side, delta_x, delta_y]


# %%%%%%%%%%%%%%%%%%%%
# files manipulation
# %%%%%%%%%%%%%%%%%%%%

def select_file(pattern='*', n_levels=5, message_to_print=None):

    list_files = ls_files(pattern, n_levels)

    if len(list_files) == 1:
        color_print("Only one option. Loading " + list_files[0])
        return list_files[0]
    elif len(list_files) == 0:
        color_print("\n\n\n#WG: ================== ERROR ==========================#")
        color_print("No files with pattern '" + pattern + "'")
    else:

        if message_to_print is None:
            print("\n\n\n#===================================================#")
            print('Enter the number of the file to be loaded:\n')
        else:
            print(message_to_print)

        for nOption, _ in enumerate(list_files):
            print(str(nOption) + ': ' + list_files[nOption])

        print('Any value different of the above raises GeneratorExit\n')

        try:
            return list_files[int(input())]
        except ValueError:
            print('\nSelected value does not correspond to any option.')
            print('WG: raise GeneratorExit!\n')
            raise GeneratorExit


def select_dir(n_levels=5, message_to_print=None):
    if message_to_print is None:
        print("\n\n\n#===================================================#")
        print('Enter the number of the directory to be loaded:\n')
    else:
        print(message_to_print)

    return select_file(pattern='', n_levels=n_levels, message_to_print='')


def ls_files(pattern='*', n_levels=1):
    """
    emulates Unix ls function

    Parameters
    ----------
    pattern : string
        pattern for ls
    n_levels : integer
        number of sub directories to run ls

    Returns
    -------
    list
        list of files inside the directories
    """
    import glob

    files_list = []

    for i in range(n_levels):
        files_list = files_list + glob.glob(pattern)
        pattern = '*/' + pattern

    return files_list.sort()


def _choose_one_of_this_options(header=None, list_of_options=None):
    """
    Plot contourf in the main graph plus profiles over vertical and horizontal line defined by mouse.

    Parameters
    ----------

    """
    for whatToPrint in header:
        print(whatToPrint)

    for optionChar, optionDescription in list_of_options:
        print(optionChar + ': ' + optionDescription)

    entry = input()

    if entry == '!':
        raise GeneratorExit

    return entry


# Tools for masking/croping

def nan_mask_threshold(array, threshold=0.0):
    """
    Calculate a square mask for array above OR below a threshold


    Parameters
    ----------
    array : ndarray
        vector with the values of x
    threshold: float
        threshold for masking. If real (imaginary) value, values below(above) the threshold are set to NAN

    Returns
    -------
    ndarray
        array with values either equal to 1 or NAN. To use as a mask for array use:
        mask = nan_mask_threshold(array, threshold)
        masked_array = array*mask

        (note that array[mask] will return only the values where mask == 1)
    """

    mask_intensity = np.ones(array.shape)

    if np.isreal(threshold):
        mask_intensity[array <= threshold] = float('nan')
    else:
        mask_intensity[array >= threshold] = float('nan')

    return mask_intensity


def index_square_mask_threshold(array, threshold=0.0):
    """
    Calculate the corner indexes i_min, i_max, j_min and j_max] for a square mask

    Parameters
    ----------
    array : ndarray
        vector with the values of x
    threshold : float
        see TODO: how do I link?

    Returns
    -------
    list
        list of indexes [i_min, i_max, j_min, j_max]
    """

    i_min, i_max, j_min, j_max = None, None, None, None
    mask_intensity = nan_mask_threshold(array, threshold)

    for i in range(mask_intensity.shape[0]):
        if any(~np.isnan(val) for val in mask_intensity[i, :]):
            i_min = i
            break

    for i in range(mask_intensity.shape[0] - 1, 0, -1):
        if any(~np.isnan(val) for val in mask_intensity[i, :]):
            i_max = i
            break

    for i in range(mask_intensity.shape[1]):
        if any(~np.isnan(val) for val in mask_intensity[:, i]):
            j_min = i
            break

    for i in range(mask_intensity.shape[1] - 1, 0, -1):
        if any(~np.isnan(val) for val in mask_intensity[:, i]):
            j_max = i
            break

    return [i_min, i_max, j_min, j_max]


def nan_square_mask_at_threshold(input_matrix, threshold=0.0):

    [i_min, i_max, j_min, j_max] = index_square_mask_threshold(input_matrix, threshold)

    square_mask = np.ones(input_matrix.shape)

    square_mask[:, :j_min] = float('NaN')
    square_mask[:, j_max:] = float('NaN')
    square_mask[:i_min, :] = float('NaN')
    square_mask[i_max:, :] = float('NaN')

    return square_mask


def crop_matrix_at_indexes(input_matrix, list_of_indexes):
    """
    Returns a copy of inputMatrix[i_min:i_max, j_min:j_max]

    Parameters
    ----------
    input_matrix : ndarray
        vector with the values of x
    list_of_indexes: list
        list in the format [i_min, i_max, j_min, j_max]

    Returns
    -------
    ndarray
        copy of subregion of the inputMatrix. ATTENTION: Note the difference of copy and view in Numpy
        inputMatrix[i_min:i_max, j_min:j_max]
    """

    return np.copy(input_matrix[list_of_indexes[0]:list_of_indexes[1],
                   list_of_indexes[2]:list_of_indexes[3]])


def crop_matrix_at_thresholds(input_matrix, threshold=0.0):
    """ listOfIndexes = [i_min, i_max, j_min, j_max]  """

    list_of_indexes = index_square_mask_threshold(input_matrix, threshold)

    return np.copy(input_matrix[list_of_indexes[0]:list_of_indexes[1],
                   list_of_indexes[2]:list_of_indexes[3]])


def find_nearest_value(array, value):
    return array[np.argmin(np.abs(array - value))]


def find_nearest_value_index(localarray, value):

    return np.int(np.where(localarray == find_nearest_value(localarray, value))[0])


def find_nearest_value_matrix(localarray, value):
    return find_nearest_value(find_nearest_value(localarray, value), value)


def find_nearest_value_matrix_index(localarray, value):

    return np.int(np.where(localarray == find_nearest_value_matrix(localarray, value)))


def dummy_images(imagetype='None', size=(100, 100), **kwargs):

    if imagetype is None:
        imagetype = 'Noise'

    if imagetype == 'Noise':
        return np.random.random(size)

    elif imagetype == 'Stripes':
        if 'nLinesH' in kwargs:
            nLinesH = kwargs['nLinesH']
            return np.kron([[1, 0] * nLinesH],
                           np.ones((size[0], size[1] / 2 / nLinesH)))
        elif 'nLinesV':
            nLinesV = kwargs['nLinesV']
            return np.kron([[1], [0]] * nLinesV,
                           np.ones((size[0] / 2 / nLinesV, size[1])))
        else:
            return np.kron([[1], [0]] * 10, np.ones((size[0] / 2 / 10, size[1])))

    elif imagetype == 'Checked':

        if 'nLinesH' in kwargs:
            nLinesH = kwargs['nLinesH']

        else:
            nLinesH = 1

        if 'nLinesV' in kwargs:
            nLinesV = kwargs['nLinesV']
        else:
            nLinesV = 1

        return np.kron([[1, 0] * nLinesH, [0, 1] * nLinesH] * nLinesV,
                       np.ones((size[0] / 2 / nLinesV, size[1] / 2 / nLinesH)))
        # Note that the new dimension is int(size/p)*p !!!

    elif imagetype == 'SumOfHarmonics':

        if 'harmAmpl' in kwargs:
            harmAmpl = kwargs['harmAmpl']
        else:
            harmAmpl = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]

        sumArray = np.zeros(size)
        iGrid, jGrid = np.mgrid[-1:1:1j * size[0], -1:1:1j * size[1]]

        for i in range(len(harmAmpl)):
            for j in range(len(harmAmpl[0])):
                sumArray += harmAmpl[i][j] * np.cos(2 * np.pi * iGrid * i) \
                            * np.cos(2 * np.pi * jGrid * j)

        return sumArray

    elif imagetype == 'Shapes':

        dx, dy = int(size[0] / 10), int(size[1] / 10)
        square = np.ones((dx * 2, dy * 2))
        triangle = np.tril(square)

        array = np.random.rand(size[0], size[1]) * .5

        array[1 * dx:3 * dx, 2 * dy:4 * dy] += triangle
        array[5 * dx:7 * dx, 1 * dy:3 * dy] += triangle * -1

        array[2 * dx:4 * dx, 7 * dy:9 * dy] += np.tril(square, +1)

        array[6 * dx:8 * dx, 5 * dy:7 * dy] += square
        array[7 * dx:9 * dx, 6 * dy:8 * dy] += square * -1

        return array


# noinspection PyClassHasNoInit,PyShadowingNames
def graphical_roi_idx(zmatrix, arg4graph=None, verbose=False):
    if arg4graph is None:
        arg4graph = {}
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    mutable_object_ROI = {'ROI_j_lim': [0, -1],
                          'ROI_i_lim': [0, -1]}

    def onselect(eclick, erelease):
        """eclick and erelease are matplotlib events at press and release"""

        ROI_j_lim = np.sort([eclick.xdata, erelease.xdata]).astype(int).tolist()
        ROI_i_lim = np.sort([eclick.ydata, erelease.ydata]).astype(int).tolist()
        # this round method has
        # an error of +-1pixel

        # if verbose: print(type(eclick.xdata))

        mutable_object_ROI['ROI_j_lim'] = [ROI_j_lim[0], ROI_j_lim[1]]
        mutable_object_ROI['ROI_i_lim'] = [ROI_i_lim[0], ROI_i_lim[1]]

        if verbose:
            print('\nSelecting ROI:')
            print(' lower position : (%d, %d)' % (ROI_j_lim[0], ROI_i_lim[0]))
            print(' higher position   : (%d, %d)' % (ROI_j_lim[1], ROI_i_lim[1]))
            print(' width x and y: (%d, %d)' %
                  (ROI_j_lim[1] - ROI_j_lim[0], ROI_i_lim[1] - ROI_i_lim[0]))

        if eclick.button == 1:

            delROIx = ROI_j_lim[1] - ROI_j_lim[0]
            delROIy = ROI_i_lim[1] - ROI_i_lim[0]

            plt.xlim(ROI_j_lim[0] - .2 * delROIx,
                     ROI_j_lim[1] + .2 * delROIx)
            plt.ylim(ROI_i_lim[1] + .2 * delROIy,
                     ROI_i_lim[0] - .2 * delROIy)

        elif eclick.button == 2:
            plt.xlim(0, np.shape(zmatrix)[1])
            plt.ylim(np.shape(zmatrix)[0], 0)

        elif eclick.button == 3:

            delROIx = ROI_j_lim[1] - ROI_j_lim[0]
            delROIy = ROI_i_lim[1] - ROI_i_lim[0]

            plt.xlim(ROI_j_lim[0] - 5 * delROIx,
                     ROI_j_lim[1] + 5 * delROIx)
            plt.ylim(ROI_i_lim[1] + .5 * delROIy,
                     ROI_i_lim[0] - .5 * delROIy)

    class MyRectangleSelector(RectangleSelector):
        def release(self, event):
            super(MyRectangleSelector, self).release(event)
            self.to_draw.set_visible(True)
            self.canvas.draw()

    def toggle_selector(event):
        if verbose: print(' Key pressed.')
        if event.key in ['Q', 'q'] and toggle_selector.RS.active:
            if verbose: print(' RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a'] and not toggle_selector.RS.active:
            if verbose: print(' RectangleSelector activated.')
            toggle_selector.RS.set_active(True)

    fig = plt.figure(facecolor="white",
                     figsize=(10, 8))

    surface = plt.imshow(zmatrix,  # origin='lower',
                         cmap='spectral', **arg4graph)

    plt.xlabel('Pixels')
    plt.ylabel('Pixels')
    plt.title('CHOOSE ROI, CLOSE WHEN DONE\n'
              'Middle Click: Reset, \n' +
              'Right Click: select ROI - zoom in,\n' +
              'Left Click: select ROI - zoom out',
              fontsize=16, color='r', weight='bold')
    plt.colorbar(surface)

    toggle_selector.RS = MyRectangleSelector(plt.gca(), onselect,
                                             drawtype='box',
                                             rectprops=dict(facecolor='purple',
                                                            edgecolor='black',
                                                            alpha=0.5,
                                                            fill=True))

    fig.canvas.mpl_connect('key_press_event', toggle_selector)
    plt.show(block=True)

    if verbose: print(mutable_object_ROI['ROI_i_lim'] + \
                      mutable_object_ROI['ROI_j_lim'])

    return mutable_object_ROI['ROI_i_lim'] + \
           mutable_object_ROI['ROI_j_lim']  # Note that the + signal concatenates the two lists


def crop_graphic(xvec, yvec, zmatrix, verbose=False):
    idx = graphical_roi_idx(zmatrix, verbose=verbose)

    return xvec[idx[2]:idx[3]], \
           yvec[idx[0]:idx[1]], \
           crop_matrix_at_indexes(zmatrix, idx), idx


def choose_unit(array):


    max_abs = np.max(np.abs(array))

    if 2e0 < max_abs <= 2e3:
        factor = 1.0
        unit = ''
    elif 2e-9 < max_abs <= 2e-6:
        factor = 1.0e9
        unit = 'n'
    elif 2e-6 < max_abs <= 2e-3:
        factor = 1.0e6
        unit = r'\mu'
    elif 2e-3 < max_abs <= 2e0:
        factor = 1.0e3
        unit = 'm'
    elif 2e3 < max_abs <= 2e6:
        factor = 1.0e-3
        unit = 'K'
    elif 2e6 < max_abs <= 2e9:
        factor = 1.0e-6
        unit = 'M'
    else:
        factor = 1.0
        unit = ''

    return factor, unit


### time functions

def datetime_now_str():
    from time import strftime

    return strftime("%Y%m%d_%H%M%S")


def time_now_str():
    from time import strftime

    return strftime("%H%M%S")


def date_now_str():
    from time import strftime

    return strftime("%Y%m%d")


# coordinates in real and kspace.

def realcoordvec(npoints, delta):
    """
    Build a vector with real space coordinates based on the number of points and bin (pixels) size.

    Alias for np.mgrid[-npoints/2*delta:npoints/2*delta-delta:npoints*1j]

    Parameters
    ----------
    npoints : int
        vector with the values of x
    delta : float
        vector with the values of x

    Returns
    -------
    ndarray
        2 vectors (1D array) with real coordinates
    """
    return np.mgrid[-npoints/2*delta:npoints/2*delta-delta:npoints*1j]


def realcoordmatrix_fromvec(xvec, yvec):
    """
    Alias for np.meshgrid(xvec, yvec)

    Parameters
    ----------
    xvec, yvec : ndarray
        vector (1D array) with real coordinates

    Returns
    -------
    ndarray
        2 matrices (1D array) with real coordinates
    """
    return np.meshgrid(xvec, yvec)

def realcoordmatrix(npointsx, deltax, npointsy, deltay):
    """
    Build a matrix (2D array) with real space coordinates based on the number of points and bin (pixels) size.

    Alias for realcoordmatrix_fromvec(realcoordvec(nx, delx), realcoordvec(ny, dely))

    Parameters
    ----------
    npointsx, npointsy : int
        vector with the values of x
    deltax, deltay : float
        vector with the values of x

    Returns
    -------
    ndarray
        2 matrices (1D array) with real coordinates
    """
    return realcoordmatrix_fromvec(realcoordvec(npointsx, deltax),
                                   realcoordvec(npointsy, deltay))

def fouriercoordvec(npoints, delta):

    return np.mgrid[-1/2/delta:1/2/delta-1/npoints/delta:npoints*1j]


def fouriercoordmatrix(nx, delx, ny, dely):
    return np.meshgrid(fouriercoordvec(nx, delx),
                       fouriercoordvec(ny, dely))




### h5 tools

def h5_list_of_groups(h5file):
    list_of_goups = []
    h5file.visit(list_of_goups.append)

    return list_of_goups


if __name__ == '__main__':
    color_print_blue('Oi')
    color_print('Oi', 'white', 'on_green', ['bold'])


# Progress bar

def progress_bar4pmap(res):
    while (True):

        print(len(res._value))

        remaining = res._number_left/len(res._value)
        pbar = str('int {0:2} of {1} '.format(len(res._value) - res._number_left,
                                              len(res._value)))

        pbar += '['+'*'*(30-int(remaining*30)) + ' '*int(remaining*30) + '] '
        pbar += '{0:5.2f}% tasks completed...\r'.format(100 - remaining*100)
        print(pbar, end='')

        if (res.ready()):
            print('')
            break
        time.sleep(0.5)

def progress_bar4pmap2(res,sleep_time=1.0):

    res_size = len(res._value)

    print()

    pbar = tqdm(total=res_size)

    old_res_n_left = res_size

    while res._number_left > 0:
        if old_res_n_left != res._number_left:
            pbar.update(old_res_n_left - res._number_left)
            old_res_n_left = res._number_left
        time.sleep(sleep_time)

    pbar.close()
    print('')