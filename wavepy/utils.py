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

import configparser
import os.path

__authors__ = "Walan Grizolli"
__copyright__ = "Copyright (c) 2016, Affiliation"
__version__ = "0.1.0"
__docformat__ = "restructuredtext en"
__all__ = ['print_color', 'print_red', 'print_blue', 'plot_profile',
           'select_file', 'select_dir', 'nan_mask_threshold',
           'crop_matrix_at_indexes',
           'find_nearest_value', 'find_nearest_value_index',
           'dummy_images', 'graphical_roi_idx', 'crop_graphic', 'choose_unit',
           'datetime_now_str', 'time_now_str', 'date_now_str',
           'realcoordvec', 'realcoordmatrix_fromvec', 'realcoordmatrix',
           'fouriercoordvec', 'fouriercoordmatrix',
           'h5_list_of_groups',
           'progress_bar4pmap', 'load_ini_file']



def print_color(message, color='red',
                highlights='on_white', attrs=''):
    """
    Print with colored characters. It is only a alias for colored print using
    the package :py:mod:`termcolor` and equals to::

        print(termcolor.colored(message, color, highlights, attrs=attrs))


    See options at https://pypi.python.org/pypi/termcolor

    Parameters
    ----------
    message : str
        Message to print.
    color, highlights: str

    attrs: list

    """
    import termcolor
    print(termcolor.colored(message, color, highlights, attrs=attrs))


def print_red(message):
    """
    Print with colored characters. It is only a alias for colored print using
    the package :py:mod:`termcolor` and equals to::

            print(termcolor.colored(message, color='red'))

    Parameters
    ----------
    message : str
        Message to print.
    """
    import termcolor
    print(termcolor.colored(message, color='red'))


def print_blue(message):
    """
    Print with colored characters. It is only a alias for colored print using
    the package :py:mod:`termcolor` and equals to::

            print(termcolor.colored(message, color='blue'))

    Parameters
    ----------
    message : str
        Message to print.
    """
    import termcolor
    print(termcolor.colored(message, 'blue'))


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
                 xlabel='x', ylabel='y', zlabel='z', title='Title',
                 xo=None, yo=None,
                 xunit='', yunit='', do_fwhm=True,
                 arg4main=None, arg4top=None, arg4side=None):
    """
    Plot contourf in the main graph plus profiles over vertical and horizontal
    lines defined with mouse.




    Parameters
    ----------
    xmatrix, ymatrix: ndarray
        `x` and `y` matrix coordinates generated with :py:func:`numpy.meshgrid`

    zmatrix: ndarray
        Matrix with the data. Note that ``xmatrix``, ``ymatrix`` and ``zmatrix``
        must have the same shape

    xlabel, ylabel, zlabel: str, optional

    title: str, optional
        title for the main graph #BUG: sometimes this title disappear

    xo, yo: float, optional
        if equal to ``None``, it allows to use the mouse to choose the vertical and
        horizontal lines for the profile. If not ``None``, the profiles lines are
        are centered at ``(xo,yo)``

    xunit, yunit: str, optional
        String to be shown after the values in the small text box

    do_fwhm: Boolean, optional
        Calculate and print the FWHM in the figure. The script to calculate the
        FWHM is not very robust, it works well if only one well defined peak is
        present. Turn this off by setting this var to ``False``

    *arg4main:
        `*args` for the main graph

    *arg4top:
        `*args` for the top graph

    *arg4side:
        `*args` for the side graph

    Returns
    -------

    ax_main, ax_top, ax_side: matplotlib.axes
        return the axes in case one wants to modify them.

    delta_x, delta_y: float

    Example
    -------

    >>> import numpy as np
    >>> import wavepy.utils as wpu
    >>> xx, yy = np.meshgrid(np.linspace(-1, 1, 101), np.linspace(-1, 1, 101))
    >>> wpu.plot_profile(xx, yy, np.exp(-(xx**2+yy**2)/.2))

    Animation of the example above:

    .. image:: img/plot_profile_animation.gif

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

        message = r'$x_o = %.4g %s$' % (_xo, xunit)
        message = message + '\n' + r'$y_o = %.4g %s$' % (_yo, yunit)

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
        cursor = Cursor(ax_main, useblit=True, color='red', linewidth=2)
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show(block=True)
    else:
        [delta_x, delta_y] = plot_profiles_at(xo, yo)

    return [ax_main, ax_top, ax_side, delta_x, delta_y]


# %%%%%%%%%%%%%%%%%%%%
# files manipulation
# %%%%%%%%%%%%%%%%%%%%

def select_file(pattern='*', message_to_print=None):
    """
    List files under the subdirectories of the current working directory, and expected the user to choose one of them.

    The list of files is of the form ``number: filename``. The user choose the file by typing the number of the desired filename.


    Parameters
    ----------

    pattern: str
        list only files with this patter. Similar to pattern in the linux comands ls, grep, etc
    message_to_print: str, optional

    Returns
    -------

    filename: str
        path and name of the file

    Example
    -------

    >>>  select_file('*.dat')

    """

    import glob

    list_files = glob.glob(pattern, recursive=True)

    if len(list_files) == 1:
        print_color("Only one option. Loading " + list_files[0])
        return list_files[0]
    elif len(list_files) == 0:
        print_color("\n\n\n# ================== ERROR ==========================#")
        print_color("No files with pattern '" + pattern + "'")
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
            choice = int(input())
            print('Selected file ' + list_files[choice])
            return list_files[choice]
        except ValueError:
            print('\nSelected value does not correspond to any option.')
            print('raise GeneratorExit!\n')
            raise GeneratorExit


def select_dir(message_to_print=None):
    """

    List subdirectories of the current working directory, and expected the user to choose one of them.

    The list of files is of the form ``number: filename``. The user choose the file by typing the number of the desired filename.

    Similar to :py:func:`wavepy.utils.select_file`

    Parameters
    ----------

    message_to_print:str, optional

    Returns
    -------

    str
        directory path

    See Also
    --------
    :py:func:`wavepy.utils.select_file`

    """
    if message_to_print is None:
        print("\n\n\n#===================================================#")
        print('Enter the number of the directory to be loaded:\n')
    else:
        print(message_to_print)

    return select_file(pattern='', message_to_print='')



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

def nan_mask_threshold(input_matrix, threshold=0.0):
    """
    Calculate a square mask for array above OR below a threshold


    Parameters
    ----------
    input_matrix : ndarray
        2 dimensional (or n-dimensional?) numpy.array to be masked
    threshold: float
        threshold for masking. If real (imaginary) value, values below(above) the threshold are set to NAN

    Returns
    -------
    ndarray
        array with values either equal to 1 or NAN.


    Example
    -------

        To use as a mask for array use:

        >>> mask = nan_mask_threshold(input_array, threshold)
        >>> masked_array = input_array*mask

    Notes
    -----

        * Note that ``array[mask]`` will return only the values where ``mask == 1``.

        * Also note that this is NOT the same as :py:mod:`numpy.ma`, the `masked arrays <http://docs.scipy.org/doc/numpy/reference/maskedarray.html>`_ in numpy.

    """

    mask_intensity = np.ones(input_matrix.shape)

    if np.isreal(threshold):
        mask_intensity[input_matrix <= threshold] = float('nan')
    else:
        mask_intensity[input_matrix >= threshold] = float('nan')

    return mask_intensity


def crop_matrix_at_indexes(input_matrix, list_of_indexes):
    """
    Alias for ``np.copy(inputMatrix[i_min:i_max, j_min:j_max])``

    Parameters
    ----------
    input_matrix : ndarray
        2 dimensional array
    list_of_indexes: list
        list in the format ``[i_min, i_max, j_min, j_max]``

    Returns
    -------
    ndarray
        copy of the sub-region ``inputMatrix[i_min:i_max, j_min:j_max]`` of the inputMatrix.

    Warning
    -------
        Note the `difference of copy and view in Numpy <http://scipy-cookbook.readthedocs.io/items/ViewsVsCopies.html>`_.
    """

    return np.copy(input_matrix[list_of_indexes[0]:list_of_indexes[1],
                   list_of_indexes[2]:list_of_indexes[3]])


# def find_nearest_value_index(input_array, value):
#     """
#
#     Similar to :py:func:`wavepy.utils.find_nearest_value`, but returns the index of the nearest value (instead of the value itself)
#
#     Parameters
#     ----------
#
#     input_array : ndarray
#     value : float
#
#     Returns
#     -------
#
#         int
#
#     """
#
#     return np.int(np.where(input_array == find_nearest_value(input_array, value))[0])


def find_nearest_value(input_array, value):
    """

    Alias for ``input_array.flatten()[np.argmin(np.abs(input_array.flatten() - value))]``

    In a array of float numbers, due to the precision, it is impossible to find exact values. For instance something like ``array1[array2==0.0]`` might fail because the zero values in the float array ``array2`` are actually something like 0.0004324235 (fictious value).

    This function will return the value in the array that is the nearest to the parameter ``value``.

    Parameters
    ----------

    input_array: ndarray
    value: float

    Returns
    -------
    ndarray

    Example
    -------

    >>> foo = dummy_images('NormalDist')
    >>> find_nearest_value(foo, 0.5000)
    0.50003537554879007

    See Also
    --------

    :py:func:`wavepy:utils:find_nearest_value_index`

    """

    return input_array.flatten()[np.argmin(np.abs(input_array.flatten() - value))]


def find_nearest_value_index(input_array, value):
    """

    Similar to :py:func:`wavepy.utils.find_nearest_value`, but returns the index of the nearest value (instead of the value itself)

    Parameters
    ----------

    input_array : ndarray
    value : float

    Returns
    -------

    tuple of ndarray:
        each array have the index of the nearest value in each dimension

    Note
    ----
    In principle it has no limit of the number of dimensions.


    Example
    -------

    >>> foo = dummy_images('NormalDist')
    >>> find_nearest_value(foo, 0.5000)
    0.50003537554879007
    >>> (i_index, j_index) = find_nearest_value_index(foo, 0.500)
    >>> foo[i_index[:], j_index[:]]
    array([ 0.50003538,  0.50003538,  0.50003538,  0.50003538])

    See Also
    --------
    :py:func:`wavepy:utils:find_nearest_value`


    """



    return np.where(input_array == find_nearest_value(input_array, value))


def dummy_images(imagetype='None', shape=(100, 100), **kwargs):
    """

    Dummy images for simple tests.


    Parameters
    ----------

    imagetype: str
        See options Below
    shape: tuple
        Shape of the image. Similar to :py:mod:`numpy.shape`
    kwargs:
        keyword arguments depending on the image type.


    Image types
        * Noise (default):    alias for ``np.random.random(shape)``

        * Stripes:            ``kwargs: nLinesH, nLinesV``

        * SumOfHarmonics: image is defined by:
         .. math:: \sum_{ij} Amp_{ij} \cos (2 \pi i y) \cos (2 \pi j x).

         The keyword ``kwargs: harmAmpl`` is a 2D list that can be used to set the values for Amp_ij, see **Examples**

        * Shapes: see **Examples**. ``kwargs=noise``, amplitude of noise to be \
          added to the image

        * NormalDist: Normal distribution where it is assumed that ``x`` and ``y`` are in the interval `[-1,1]`.
          ``keywords: FWHM_x, FWHM_y``


    Returns
    -------
        2D ndarray


    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(dummy_images())

    is the same than

    >>> plt.imshow(dummy_images('Noise'))


    .. image:: img/dummy_image_Noise.png
       :width: 350px


    >>> plt.imshow(dummy_images('Stripes', nLinesV=5))

    .. image:: img/dummy_image_stripe_V5.png
       :width: 350px


    >>> plt.imshow(dummy_images('Stripes', nLinesH=8))

    .. image:: img/dummy_image_stripe_H8.png
       :width: 350px


    >>> plt.imshow(dummy_images('Checked', nLinesH=8, nLinesV=5))

    .. image:: img/dummy_image_checked_v5_h8.png
       :width: 350px


    >>> plt.imshow(dummy_images('SumOfHarmonics', harmAmpl=[[1,0,1],[0,1,0]]))

    .. image:: img/dummy_image_harmonics_101_010.png
       :width: 350px

    >>> plt.imshow(dummy_images('Shapes', noise = 1))

    .. image:: img/dummy_image_shapes_noise_1.png
       :width: 350px

    >>> plt.imshow(dummy_images('NormalDist', FWHM_x = .5, FWHM_y=1.0))

    .. image:: img/dummy_image_NormalDist.png
       :width: 350px



    """

    if imagetype is None:
        imagetype = 'Noise'

    if imagetype == 'Noise':
        return np.random.random(shape)

    elif imagetype == 'Stripes':
        if 'nLinesH' in kwargs:
            nLinesH = kwargs['nLinesH']
            return np.kron([[1, 0] * nLinesH],
                           np.ones((shape[0], shape[1] / 2 / nLinesH)))
        elif 'nLinesV':
            nLinesV = kwargs['nLinesV']
            return np.kron([[1], [0]] * nLinesV,
                           np.ones((shape[0] / 2 / nLinesV, shape[1])))
        else:
            return np.kron([[1], [0]] * 10, np.ones((shape[0] / 2 / 10, shape[1])))

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
                       np.ones((shape[0] / 2 / nLinesV, shape[1] / 2 / nLinesH)))
        # Note that the new dimension is int(shape/p)*p !!!

    elif imagetype == 'SumOfHarmonics':

        if 'harmAmpl' in kwargs:
            harmAmpl = kwargs['harmAmpl']
        else:
            harmAmpl = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]

        sumArray = np.zeros(shape)
        iGrid, jGrid = np.mgrid[-1:1:1j * shape[0], -1:1:1j * shape[1]]

        for i in range(len(harmAmpl)):
            for j in range(len(harmAmpl[0])):
                sumArray += harmAmpl[i][j] * np.cos(2 * np.pi * iGrid * i) \
                            * np.cos(2 * np.pi * jGrid * j)

        return sumArray

    elif imagetype == 'Shapes':

        if 'noise' in kwargs:
            noiseAmp = kwargs['noise']
        else:
            noiseAmp = 0.0

        dx, dy = int(shape[0] / 10), int(shape[1] / 10)
        square = np.ones((dx * 2, dy * 2))
        triangle = np.tril(square)

        array = np.random.rand(shape[0], shape[1]) * noiseAmp

        array[1 * dx:3 * dx, 2 * dy:4 * dy] += triangle
        array[5 * dx:7 * dx, 1 * dy:3 * dy] += triangle * -1

        array[2 * dx:4 * dx, 7 * dy:9 * dy] += np.tril(square, +1)

        array[6 * dx:8 * dx, 5 * dy:7 * dy] += square
        array[7 * dx:9 * dx, 6 * dy:8 * dy] += square * -1

        return array


    elif imagetype == 'NormalDist':

        FWHM_x, FWHM_y = 1.0, 1.0

        if 'FWHM_x' in kwargs:
            FWHM_x = kwargs['FWHM_x']
        if 'FWHM_y' in kwargs:
            FWHM_y = kwargs['FWHM_y']

        x, y = np.mgrid[-1:1:1j * shape[0], -1:1:1j * shape[1]]

        return np.exp(-((x/FWHM_x*2.3548200)**2 +
                        (y/FWHM_y*2.3548200)**2)/2)  # sigma for FWHM = 1

    else:
        print_color("ERROR: image type invalid: " + str(imagetype))

        return np.random.random(shape)


# noinspection PyClassHasNoInit,PyShadowingNames
def graphical_roi_idx(zmatrix, verbose=False, **kargs4graph):
    """
    Function to define a rectangular region of interest (ROI) in an image.

    The image is plotted and, using the mouse, the user select the region of interest (ROI). The ROI is ploted as an transparent rectangular region. When the image is closed the function returns the indexes ``[i_min, i_max, j_min,_j_max]`` of the ROI.

    Parameters
    ----------

    input_array : ndarray
    verbose : Boolean
        In the verbose mode it is printed some additional infomations, like the ROI indexes, as the user select different ROI's
    **kargs4graph : float
        Options for the main graph. **WARNING:** not tested very well

    Returns
    -------

    list:
        indexes of the crop ``[i_min, i_max, j_min,_j_max]``. Useful when the same crop must be applies to other images

    Note
    ----
    In principle it has no limit of the number of dimensions.


    Example
    -------
    See example at :py:func:`wavepy:utils:crop_graphic`


    See Also
    --------
    :py:func:`wavepy:utils:crop_graphic`
    """


    if kargs4graph is None:
        kargs4graph = {}

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
                         cmap='spectral', **kargs4graph)

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
    """

    Function to crop an image to the ROI selected using the mouse.

    :py:func:`wavepy.utils.graphical_roi_idx` is first used to plot and select the ROI. The function then returns the croped version of the matrix, the cropped coordinate vectors ``x`` and  ``y``, and the indexes ``[i_min, i_max, j_min,_j_max]``

    Parameters
    ----------
    xvec, yvec: 1D ndarray
        vector with the coordinates ``x`` and ``y``
    zmatrix: 2D numpy array
        image to be croped, as an 2D ndarray

    Returns
    -------
    1D ndarray, 1D ndarray:
        cropped coordinate vectors ``x`` and  ``y``
    2D ndarray:
        cropped image
    list:
        indexes of the crop ``[i_min, i_max, j_min,_j_max]``. Useful when the same crop must be applies to other images

    Examples
    --------

    >>> import numpy as np
    >>> import matplotlib as plt
    >>> xVec = np.arange(0.,101)
    >>> yVec = np.arange(0.,101)
    >>> img = dummy_images('Shapes', size=(101,101), FWHM_x = .5, FWHM_y=1.0)
    >>> (xVecCroped, yVecCroped, imgCroped, idx4crop) = crop_graphic(xVec, yVec, img)
    >>> plt.imshow(imgCroped, cmap='Spectral')


    .. image:: img/graphical_roi_idx_in_action.gif
       :width: 350px

    See Also
    --------
    :py:func:`wavepy.utils.graphical_roi_idx`
    """
    idx = graphical_roi_idx(zmatrix, verbose=verbose)

    return xvec[idx[2]:idx[3]], \
           yvec[idx[0]:idx[1]], \
           crop_matrix_at_indexes(zmatrix, idx), idx


def choose_unit(array):
    """

    Script to choose good(best) units in engineering notation for a ``ndarray``.

    For a given input array, the function returns ``factor`` and ``unit`` according to

    .. math:: 10^{n} < \max(array) < 10^{n + 3}

    +------------+----------------------+------------------------+
    |     n      |    factor (float)    |        unit(str)       |
    +============+======================+========================+
    |     0      |    1.0               |   ``''`` empty string  |
    +------------+----------------------+------------------------+
    |     -9     |    10^-9             |        ``n``           |
    +------------+----------------------+------------------------+
    |     -6     |    10^-6             |     ``r'\mu'``         |
    +------------+----------------------+------------------------+
    |     -3     |    10^-9             |        ``m``           |
    +------------+----------------------+------------------------+
    |     +3     |    10^-6             |        ``k``           |
    +------------+----------------------+------------------------+
    |     +6     |    10^-9             |        ``M``           |
    +------------+----------------------+------------------------+
    |     +9     |    10^-6             |        ``G``           |
    +------------+----------------------+------------------------+

    ``n=-6`` returns ``\mu`` since this is the latex syntax for micro. See Example.


    Parameters
    ----------
    array : ndarray

    Returns
    -------
    float :
    unit :

    Example
    -------

    >>> array1 = np.linspace(0,100e-6,101)
    >>> array2 = array1*1e10
    >>> factor1, unit1 = choose_unit(array1)
    >>> factor2, unit2 = choose_unit(array2)
    >>> plt.plot(array1*factor1,array2*factor2)
    >>> plt.xlabel(r'${0} m$'.format(unit1))
    >>> plt.ylabel(r'${0} m$'.format(unit2))

    The syntax ``r'$ string $ '`` is necessary to use latex commands in the :py:mod:`matplotlib` labels.

    """


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
        unit = 'k'
    elif 2e6 < max_abs <= 2e9:
        factor = 1.0e-6
        unit = 'M'
    elif 2e9 < max_abs <= 2e12:
        factor = 1.0e-6
        unit = 'G'
    else:
        factor = 1.0
        unit = ''

    return factor, unit


### time functions

def datetime_now_str():
    """
    Returns the current date and time as a string in the format YYmmDD_HHMMSS. Alias for ``time.strftime("%Y%m%d_%H%M%S")``.

    Return
    ------
    str

    """

    from time import strftime

    return strftime("%Y%m%d_%H%M%S")


def time_now_str():
    """
    Returns the current time as a string in the format HHMMSS. Alias for ``time.strftime("%H%M%S")``.

    Return
    ------
    str

    """
    from time import strftime

    return strftime("%H%M%S")


def date_now_str():
    """
    Returns the current date as a string in the format YYmmDD. Alias for ``time.strftime("%Y%m%d")``.

    Return
    ------
    str

    """
    from time import strftime

    return strftime("%Y%m%d")


# coordinates in real and kspace.

def realcoordvec(npoints, delta):
    """
    Build a vector with real space coordinates based on the number of points and bin (pixels) size.

    Alias for ``np.mgrid[-npoints/2*delta:npoints/2*delta-delta:npoints*1j]``

    Parameters
    ----------
    npoints : int
        number of points
    delta : float
        vector with the values of x

    Returns
    -------
    ndarray
        vector (1D array) with real coordinates

    Example
    -------
    >>> realcoordvec(10,1)
    array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])

    See Also
    --------
    :py:func:`wavepy.utils.realcoordmatrix_fromvec`
    :py:func:`wavepy.utils.realcoordmatrix`

    """
    return np.mgrid[-npoints/2*delta:npoints/2*delta-delta:npoints*1j]


def realcoordmatrix_fromvec(xvec, yvec):
    """
    Alias for ``np.meshgrid(xvec, yvec)``

    Parameters
    ----------
    xvec, yvec : ndarray
        vector (1D array) with real coordinates

    Returns
    -------
    ndarray
        2 matrices (1D array) with real coordinates

    Example
    -------

    >>> vecx = realcoordvec(3,1)
    >>> vecy = realcoordvec(4,1)
    >>> realcoordmatrix_fromvec(vecx, vecy)
    [array([[-1.5, -0.5,  0.5], [-1.5, -0.5,  0.5], [-1.5, -0.5,  0.5],
    [-1.5, -0.5,  0.5]]), array([[-2., -2., -2.], [-1., -1., -1.],
    [ 0.,  0.,  0.], [ 1.,  1.,  1.]])]


    See Also
    --------
    :py:func:`wavepy.utils.realcoordvec`
    :py:func:`wavepy.utils.realcoordmatrix`

    """
    return np.meshgrid(xvec, yvec)


def realcoordmatrix(npointsx, deltax, npointsy, deltay):
    """
    Build a matrix (2D array) with real space coordinates based on the number of points and bin (pixels) size.

    Alias for ``realcoordmatrix_fromvec(realcoordvec(nx, delx), realcoordvec(ny, dely))``

    Parameters
    ----------
    npointsx, npointsy : int
        number of points in the x and y directions
    deltax, deltay : float
        step size in the x and y directions

    Returns
    -------
    ndarray, ndarray
        2 matrices (2D array) with real coordinates

    Example
    -------

    >>> realcoordmatrix(3,1,4,1)
    [array([[-1.5, -0.5,  0.5], [-1.5, -0.5,  0.5], [-1.5, -0.5,  0.5],
    [-1.5, -0.5,  0.5]]), array([[-2., -2., -2.], [-1., -1., -1.],
    [ 0.,  0.,  0.], [ 1.,  1.,  1.]])]

    See Also
    --------
    :py:func:`wavepy.utils.realcoordvec`
    :py:func:`wavepy.utils.realcoordmatrix_fromvec`

    """
    return realcoordmatrix_fromvec(realcoordvec(npointsx, deltax),
                                   realcoordvec(npointsy, deltay))


def fouriercoordvec(npoints, delta):
    r"""

    Create coordinates in the (spatial) frequency domain based on the number of
    points ``n`` and the step (binning) ``\Delta x`` in the **REAL SPACE**. It
    returns a vector of frequencies with values in the interval


    .. math:: f = \left[ \frac{-1}{2 \Delta x} : \frac{1}{2 \Delta x} - \frac{1}{n \Delta x} \right]

    with the same number of points

    Parameters
    ----------
    npoints : int
        number of points
    delta : float
        step size in the **REAL SPACE**

    Returns
    -------
    ndarray


    Example
    -------

    >>> fouriercoordvec(10,1e-3)
    array([-500., -400., -300., -200., -100.,    0.,  100.,  200.,  300.,  400.])

    See Also
    --------
    :py:func:`wavepy.utils.realcoordvec`
    :py:func:`wavepy.utils.fouriercoordmatrix`

    """

    return np.mgrid[-1/2/delta:1/2/delta-1/npoints/delta:npoints*1j]


def fouriercoordmatrix(npointsx, deltax, npointsy, deltay):
    """

    Similar to :py:func:`wavepy.utils.fouriercoordvec`, but for matrices (2D arrays).

    Parameters
    ----------
    npointsx, npointsy : int
        number of points in the x and y directions
    deltax, deltay : float
        step size in the x and y directions

    Returns
    -------
    ndarray, ndarray
        2 matrices (2D array) with coordinates in the frequencies domain

    Example
    -------

    >>> fouriercoordmatrix(5, 1e-3, 4, 1e-3)
    [array([[-500., -300., -100.,  100.,  300.],
    [-500., -300., -100.,  100.,  300.],
    [-500., -300., -100.,  100.,  300.],
    [-500., -300., -100.,  100.,  300.]]),
    array([[-500., -500., -500., -500., -500.],
    [-250., -250., -250., -250., -250.],
    [   0.,    0.,    0.,    0.,    0.],
    [ 250.,  250.,  250.,  250.,  250.]])]

    See Also
    --------
    :py:func:`wavepy.utils.realcoordmatrix`
    :py:func:`wavepy.utils.fouriercoordvec`
    """
    return np.meshgrid(fouriercoordvec(npointsx, deltax),
                       fouriercoordvec(npointsy, deltay))

# h5 tools


def h5_list_of_groups(h5file):
    """

    Get the names of all groups and subgroups in a hdf5 file.

    Parameters
    ----------
    h5file : h5py file


    Return
    ------
    list
        list of strings with group names

    Example
    -------


    >>> fh5 = h5py.File(filename,'r')
    >>> listOfGoups = h5_list_of_groups(fh5)
    >>> for group in listOfGoups: print(group)

    """

    list_of_goups = []
    h5file.visit(list_of_goups.append)

    return list_of_goups


if __name__ == '__main__':
    pass


# Progress bar


def progress_bar4pmap(res, sleep_time=1.0):
    """
    Progress bar from :py:mod:`tqdm` to be used with the function      :py:func:`multiprocessing.starmap_async`.

    It holds the program in a loop waiting      :py:func:`multiprocessing.starmap_async` to finish


    Parameters
    ----------

    res: result object of the :py:class:`multiprocessing.Pool` class
    sleep_time:


    Example
    -------

    >>> from multiprocessing import Pool
    >>> p = Pool()
    >>> res = p.starmap_async(...)  # use your function inside brackets
    >>> p.close()  # No more work
    >>> progress_bar4pmap(res)

    """

    old_res_n_left = res._number_left
    pbar = tqdm(total= old_res_n_left)

    while res._number_left > 0:
        if old_res_n_left != res._number_left:
            pbar.update(old_res_n_left - res._number_left)
            old_res_n_left = res._number_left
        time.sleep(sleep_time)

    pbar.close()
    print('')

def load_ini_file(inifname):
    """

    This function make use of `configparser
    <https://docs.python.org/3.5/library/configparser.html>`_ to set default
    option in a ``*.ini`` file.

    In fact this function only update the ``ini`` file. The way to use is to run
    ``load_ini_file`` at the begining of the script and then load the
    parameters from the file. See **Examples**.

    The ``ini`` file must contain two sections: ``Files`` and ``Parameters``.
    The ``Files`` section list all files to be loaded. If you don't accept the
    default     value that it is offered, it will run
    :py:func:`wavepy.utils..select_file` to select other file.

    The section ``Parameters`` can contain anything, in any format, but keep in
    mind that they are passed as string.


    Parameters
    ----------
    inifname : str
        name of the ``*.ini`` file.


    Examples
    --------

    Example of ``ini`` file::

        [Files]
        image_filename = file1.tif
        ref_filename = file2.tif

        [Parameters]
        par1 = 10.5e-5
        par2 = 10, 100, 500, 600
        par can have long name = 25
        par3 = the value can be anything



    Note that ``load_ini_file`` first set/update the parameters in the file, and
    we need to load each parameters afterwards:

    >>> ini_pars, ini_file_list = load_ini_file('configfile.ini')
    >>> par1 = float(ini_pars.get('par1'))
    >>> par2 = list(map(int, ini_pars.get('par2').split(',')))

    """

    if not os.path.isfile(inifname):
        raise Exception("File " + inifname + " doesn't exist. You must " +
                         "create your init file first.")

    config = configparser.ConfigParser()
    config.read(inifname)

    print('\nAll sections:')
    for sections in config.sections(): print(sections)

    ini_pars = config['Parameters']

    ini_file_list = config['Files']

    print('\nAll keys:')
    for key in ini_pars: print(key + ':\t ' +  ini_pars.get(key))

    use_last_value = input('\nUse last values? [Y/n]: ')

    if use_last_value.lower() == 'n':

        for ftype in ini_file_list:
            kb_input = input('\nUse ' + ini_file_list.get(ftype) + 'as ' \
                               + ftype + '? [Y/n]')
            if kb_input.lower() == 'n':
                _filename = select_file('*/*.tif')
                ini_file_list[ftype] = os.getcwd() + '/' + _filename

        for key in ini_pars:
            kb_input = input('\nEnter ' + key + ' value [' \
                              + ini_pars.get(key) + '] : ')
            if kb_input != '': ini_pars[key] = kb_input

    with open(inifname, 'w') as configfile:
      config.write(configfile)

    return ini_pars, ini_file_list