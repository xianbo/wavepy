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

'''
Author: Walan Grizolli

'''

import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase
import dxchange

import wavepy.utils as wpu
import wavepy.grating_interferometry as wgi

import easygui_qt as easyqt

from scipy.interpolate import splrep, splev, sproot
from scipy import constants


rad2deg = np.rad2deg(1)
deg2rad = np.deg2rad(1)
NAN = float('Nan')  # not a number alias
hc = constants.value('inverse meter-electron volt relationship')  # hc


def intial_setup():

    [list_sample_files,
     list_ref_files,
     list_dark_files] = wpu.gui_list_data_phase_stepping()

    for fname in list_sample_files + list_ref_files + list_dark_files:
        wpu.print_blue('MESSAGE: Loading ' + fname.rsplit('/')[-1])

    pixelSize = easyqt.get_float("Enter Pixel Size [um]",
                                 title='Experimental Values',
                                 default_value=.65)*1e-6

    stepSize = easyqt.get_float("Enter scan step size [um]",
                                title='Experimental Values',
                                default_value=.2)*1e-6

    return (list_sample_files, list_ref_files, list_dark_files,
            pixelSize, stepSize)


def files_to_array(list_sample_files, list_ref_files, list_dark_files,
                   idx4crop=[0, -1, 0, -1]):

    img = wpu.crop_matrix_at_indexes(dxchange.read_tiff(list_sample_files[0]),
                                     idx4crop)

    (nlines, ncolums) = img.shape

    img_stack = np.zeros((len(list_sample_files), nlines, ncolums))
    ref_stack = img_stack*0.0

    dark_im = img_stack[0, :, :]*0.0

    for i in range(len(list_dark_files)):

        dark_im += wpu.crop_matrix_at_indexes(dxchange.read_tiff(list_dark_files[i]),
                                              idx4crop)

    for i in range(len(list_sample_files)):

        img_stack[i, :, :] = wpu.crop_matrix_at_indexes(dxchange.read_tiff(list_sample_files[i]),
                                                  idx4crop) - dark_im


        ref_stack[i, :, :] = wpu.crop_matrix_at_indexes(dxchange.read_tiff(list_ref_files[i]),
                                                        idx4crop) - dark_im

    return img_stack, ref_stack


def period_estimation_spline(signal_one_pixel, stepSize):

    signal_one_pixel -= np.mean(signal_one_pixel)

    nsteps = np.size(signal_one_pixel)

    xg = np.mgrid[0:(nsteps-1)*stepSize:nsteps*1j]
    xg2 = np.mgrid[0:(nsteps-1)*stepSize:nsteps*10j]

    tck = splrep(xg, signal_one_pixel)
    y2 = splev(xg2, tck)

    estimated_period = np.mean(np.diff(sproot(tck)))*2

    plt.figure()
    plt.plot(xg*1e6, signal_one_pixel, '-o', xg2*1e6, y2, '--.')

    plt.annotate(r'period = {:.3} $\mu m$'.format(estimated_period*1e6),
                 xy=(.80, .90), xycoords='axes fraction',
                 xytext=(-20, 20), textcoords='offset pixels', fontsize=16,
                 bbox=dict(boxstyle="round", fc="0.9"))

    plt.legend(['data', 'spline'], loc=4)
    plt.xlabel(r'$\mu m$')
    plt.ylabel('Counts')
    plt.grid()
    plt.show(block=False)

    return estimated_period


# %%

if __name__ == '__main__':

    # ==========================================================================
    # Experimental parameters
    # ==========================================================================

    (list_sample_files, list_ref_files, list_dark_files,
     pixelSize, stepSize) = intial_setup()

    # ==========================================================================
    # % % Load one image and crop
    # ==========================================================================

    img = dxchange.read_tiff(list_sample_files[0])

    colorlimit,
    cmap = wpu.plot_slide_colorbar(img, title='Raw Image',
                                   xlabel=r'x [$\mu m$ ]',
                                   ylabel=r'y [$\mu m$ ]',
                                   extent=wpu.extent_func(img, pixelSize)*1e6)

    img_croped, idx4crop = wpu.crop_graphic(zmatrix=img, verbose=True,
                                            kargs4graph={'cmap': cmap,
                                                         'vmin': colorlimit[0],
                                                         'vmax': colorlimit[1]})


    # ==========================================================================
    # %% Load tiff files to numpy array
    # ==========================================================================

    img_stack, ref_stack = files_to_array(list_sample_files,
                                          list_ref_files,
                                          list_dark_files,
                                          idx4crop=idx4crop)

    nimages, nlines, ncolumns = ref_stack.shape

    # ==========================================================================
    # %% use data to determine grating period
    # ==========================================================================

    period_estimated = period_estimation_spline(ref_stack[:, nlines//4,
                                                          nlines//4],
                                                stepSize)

    period_estimated += period_estimation_spline(ref_stack[:, 3*nlines//4,
                                                           3*nlines//4],
                                                 stepSize)

    period_estimated /= 2.0

    wpu.print_red('MESSAGE: Pattern Period from the ' +
                  'data: {:.4f}'.format(period_estimated*1e6))

    # ==========================================================================
    # %% do your thing
    # ==========================================================================

    (intensity,
     dk_field,
     dpc_1d,
     chi2) = wgi.stepping_grating_1Danalysis(img_stack, ref_stack,
                                             period_estimated, stepSize)

    # Intensity

    wpu.plot_slide_colorbar(intensity,
                            title='Intensity',
                            xlabel=r'x [$\mu m$]',
                            ylabel=r'y [$\mu m$]',
                            cmin_o=wpu.mean_plus_n_sigma(intensity, -3),
                            cmax_o=wpu.mean_plus_n_sigma(intensity, 3),
                            extent=wpu.extent_func(dpc_1d, pixelSize)*1e6)

    # Dark Field
    wpu.plot_slide_colorbar(dk_field, title='Dark Field',
                            xlabel=r'x [$\mu m$]',
                            ylabel=r'y [$\mu m$]',
                            extent=wpu.extent_func(dpc_1d, pixelSize)*1e6)

    # DPC
    wpu.plot_slide_colorbar(dpc_1d/np.pi/2.0,
                            title=r'DPC [$\pi rad$]',
                            xlabel=r'x [$\mu m$]',
                            ylabel=r'y [$\mu m$]',
                            extent=wpu.extent_func(dpc_1d, pixelSize)*1e6)

    # %% chi2

    plt.figure()
    hist = plt.hist(chi2[np.where(chi2<10*np.std(chi2))], 100, log=False)
    plt.title(r'$\chi^2$', fontsize=14, weight='bold')
    plt.show(block=False)

    chi2_copy = np.copy(chi2)

    chi2_copy[np.where(chi2>200)] = np.nan



    wpu.plot_slide_colorbar(chi2_copy, title=r'$\chi^2$ sample',
                            xlabel=r'x [$\mu m$ ]',
                            ylabel=r'y [$\mu m$ ]',
                            extent=wpu.extent_func(chi2, pixelSize)*1e6)


    # %% mask by chi2

    dpc_1d[np.where(np.abs(dpc_1d) < 1*np.std(dpc_1d))] = 0.0

    masked_plot = dpc_1d*1.0


    masked_plot[np.where(chi2>50)] = 0.0

    wpu.plot_slide_colorbar(masked_plot, title='DPC',
                            xlabel=r'x [$\mu m$ ]',
                            ylabel=r'y [$\mu m$ ]',
                            extent=wpu.extent_func(masked_plot, pixelSize)*1e6)
