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


Grating interferometry
----------------------


This library contain the function to analyse data from grating
interferometry experiments.

There are several different layouts for a grating interferometry experiments,
where one could use: one dimensional, two-dimensional (checked board) or
circular gratings; phase or absorption gratings; and, in experimetns with more
than one grating, we can have combination of different gratings.

For this reason, it is very difficult to write a function that covers all the
possibilities and (at least initally) we need a function for each particular
case.




"""

# import itertools
# import numpy as np
# import time
# from tqdm import tqdm
#
# from skimage.feature import register_translation
#
# from multiprocessing import Pool, cpu_count
#
# import wavepy.utils as wpu
#
# from wavepy.cfg import *


import numpy as np
import matplotlib.pyplot as plt
import wavepy.utils as wpu


from unwrap import unwrap


__authors__ = "Walan Grizolli"
__copyright__ = "Copyright (c) 2016, Affiliation"
__version__ = "0.1.0"
__docformat__ = "restructuredtext en"
__all__ = ['extract_harmonic', 'plot_harmonic_grid']




def _idxPeak_ij(i, j, nRows, nColumns, periodVert, periodHor):
    return [nRows // 2 + i * periodVert, nColumns // 2 + j * periodHor]


def extract_harmonic(img, harmonicPeriod,
                     harmonic_ij='00', searchRegion = 10, isFFT=False,
                     plotFlag=False, verbose=True):

    '''
    Function to extract one harmonic image of the FFT of single grating
    Talbot imaging.


    The function use the provided value of period to search for the harmonics
    peak. The search is done in a rectangle of size
    ``periodVert*periodHor/searchRegion**2``. The final result is a rectagle of
    size ``periodVert x periodHor`` centered at
    ``(harmonic_Vertical*periodVert x harmonic_Horizontal*periodHor)``


    Parameters
    ----------
    imgFFT : 	ndarray – Data (data_exchange format)
        FFT of the Experimental image, whith proper blank image, crop and rotation already
        applied.


    harmonicPeriod : list of integers in the format [periodVert, periodHor]
        ``periodVert`` and ``periodVert`` are the period of the harmonics in
        the reciprocal space in pixels. For the checked board grating,
        periodVert = sqrt(2) * pixel Size / grating Period * number of
        rows in the image. For 1D grating, set one of the values to negative or
        zero (it will set the period to number of rows or colunms).

    harmonic_ij : string or list of string
        string with the harmonic to extract, for instance '00', '01', '10'
        or '11'. In this notation negative harmonics are not allowed.

        Alternativelly, it accepts a list of string
        ``harmonic_ij=[harmonic_Vertical, harmonic_Horizontal]``, for instance
        ``harmonic_ij=['0', '-1']``

        Note that since the original image contain only real numbers (not
        complex), then negative and positive harmonics are symetric
        related to zero.

    searchRegion: int
        search for the peak will be in a region of harmonicPeriod/searchRegion
        around the theoretical peak position

    plotFlag: boolean

    verbose: boolean
        verbose mode


    Returns
    -------
    2D ndarray
        Copped Images of the harmonics ij


    This functions crops a rectagle of size ``periodVert x periodHor`` centered
    at ``(harmonic_Vertical*periodVert x harmonic_Horizontal*periodHor)`` from
    the provided FFT image.


    Note
    ----
        * Note that it is the FFT of the image that is required.
        * When the peak is found, the only extra operation tis to re-center
          the harmonic image. It is be possible to recalculate an
          experimental period, but this has almost no effect and it is not done.

    **Q: Why not the real image??**

    **A:** Because FFT can be time consuming. If we use the real image, it will
    be necessary to run FFT for each harmonic. It is encourage to wrap this
    function within a function that do the FFT, extract the harmonics, and
    return the real space harmonic image.


    See Also
    --------
    :py:func:`wavepy.grating_interferometry.plot_harmonic_grid`

    '''

    (nRows, nColumns) = img.shape

    harV = int(harmonic_ij[0])
    harH = int(harmonic_ij[1])

    periodVert = harmonicPeriod[0]
    periodHor = harmonicPeriod[1]


    # adjusts for 1D grating
    if periodVert <= 0 or periodVert is None:
        periodVert = nRows

    if periodHor <= 0 or periodHor is None:
        periodHor = nColumns

    # Check if full harmonic image is within the main image

    if (harV + .5)*periodVert > nRows / 2:
        return np.ones((nRows, periodHor))*np.nan

    if (harH + .5)*periodHor > nColumns / 2:
        return np.ones((periodVert, nColumns))*np.nan



    if not isFFT:
        imgFFT = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img)))
    else:
        imgFFT = img


    intensity = (np.abs(imgFFT))




    #  Estimate harmonic positions

    idxPeak_ij = _idxPeak_ij(harV, harH, nRows, nColumns, periodVert, periodHor)

    maskSearchRegion = np.zeros((nRows, nColumns))

    maskSearchRegion[idxPeak_ij[0] - periodVert//searchRegion:idxPeak_ij[0] + periodVert//searchRegion,
                     idxPeak_ij[1] - periodHor//searchRegion:idxPeak_ij[1] + periodHor//searchRegion] = 1.0


    # correct idxPeak_ij to the experimental value

    idxPeak_ij_exp = np.where(intensity * maskSearchRegion == np.max(np.abs(imgFFT * maskSearchRegion)))

    idxPeak_ij_exp = [idxPeak_ij_exp[0][0], idxPeak_ij_exp[1][0]]

    # convert to list

    if verbose:
        print("MESSAGE: extract_harmonic:" +
              " harmonic peak " + harmonic_ij[0] + harmonic_ij[1] +
              " is misplaced by:")
        print("MESSAGE: {:d} pixels in vertical, {:d} pixels in hor".format(
               (idxPeak_ij_exp[0]-idxPeak_ij[0]),
               (idxPeak_ij_exp[1]-idxPeak_ij[1])))

    if ((np.abs(idxPeak_ij_exp[0]-idxPeak_ij[0]) > periodVert // searchRegion // 2) or
        (np.abs(idxPeak_ij_exp[1]-idxPeak_ij[1]) > periodHor // searchRegion // 2)):

        wpu.print_red("ATTENTION: Harmonic Peak " + harmonic_ij[0] +
                      harmonic_ij[1] + " is too far from theoretical value.")
        wpu.print_red("ATTENTION: {:d} pixels in vertical, {:d} pixels in hor".format(
                      (idxPeak_ij_exp[0]-idxPeak_ij[0]),
                      (idxPeak_ij_exp[1]-idxPeak_ij[1])))

    if plotFlag:

        from matplotlib.patches import Rectangle
        plt.figure()
        plt.imshow(np.log10(intensity), cmap='Spectral_r')

        plt.gca().add_patch(Rectangle((idxPeak_ij_exp[1] - periodHor//2,
                                       idxPeak_ij_exp[0] - periodVert//2),
                                        periodHor, periodVert,
                                        lw=2, ls='--', color='red',
                                        fill=None, alpha=1))

        plt.title('Selected Region ' + harmonic_ij[0] + harmonic_ij[1],
                  fontsize=18, weight='bold')
        plt.show()

    return imgFFT[idxPeak_ij_exp[0] - periodVert//2:
                  idxPeak_ij_exp[0] + periodVert//2,
                  idxPeak_ij_exp[1] - periodHor//2:
                  idxPeak_ij_exp[1] + periodHor//2]




def plot_harmonic_grid(img, harmonicPeriod=None, isFFT=False):

    '''
    Takes the FFT of single 2D grating Talbot imaging and plot the grid from
    where we extract the harmonic in a image of the



    Parameters
    ----------
    img : 	ndarray – Data (data_exchange format)
        Experimental image, whith proper blank image, crop and rotation already
        applied.

    harmonicPeriod : integer or list of integers
        If list, it must be in the format ``[periodVert, periodHor]``. If
        integer, then [periodVert = periodHor``.
        ``periodVert`` and ``periodVert`` are the period of the harmonics in
        the reciprocal space in pixels. For the checked board grating,
        ``periodVert = sqrt(2) * pixel Size / grating Period * number of
        rows in the image``

    isFFT : boolean
        if True, then imf is the FFT of the desired image. Used to avoid an
        extra FFT operation, which can be time consuming

    '''


    if not isFFT:
        imgFFT = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img)))
    else:
        imgFFT = img

    (nRows, nColumns) = img.shape

    periodVert = harmonicPeriod[0]
    periodHor = harmonicPeriod[1]


    # adjusts for 1D grating
    if periodVert <= 0 or periodVert is None:
        periodVert = nRows

    if periodHor <= 0 or periodHor is None:
        periodHor = nColumns


    plt.figure()
    plt.imshow(np.log10(np.abs(imgFFT)), cmap='Spectral_r')

    harV_min = -(nRows + 1) // 2 // periodVert
    harV_max = (nRows + 1) // 2 // periodVert

    harH_min = -(nColumns + 1) // 2 // periodHor
    harH_max = (nColumns + 1) // 2 // periodHor

    for harV in range(harV_min + 1, harV_max + 2):

       idxPeak_ij = _idxPeak_ij(harV, 0, nRows, nColumns, periodVert, periodHor)

       plt.axhline(idxPeak_ij[0] - periodVert//2, lw=2, color='r')

    for harH in range(harH_min + 1, harH_max + 2):

        idxPeak_ij = _idxPeak_ij(0, harH, nRows, nColumns, periodVert, periodHor)
        plt.axvline(idxPeak_ij[1] - periodHor // 2, lw=2, color='r')

    for harV in range(harV_min, harV_max + 1):
        for harH in range(harH_min, harH_max + 1):

            idxPeak_ij = _idxPeak_ij(harV, harH,
                                     nRows, nColumns,
                                     periodVert, periodHor)


            plt.plot(idxPeak_ij[1], idxPeak_ij[0],
                    'ko', mew=2, mfc="None", ms=5 )

            plt.annotate('{:d}{:d}'.format(harV, harH),
                         (idxPeak_ij[1], idxPeak_ij[0]),
                          color='red', fontsize=20)

    plt.xlim(0, nColumns)
    plt.ylim(nRows,0)
    plt.title('log scale FFT magnitude, Hamonics Subsets and Indexes',
              fontsize=16, weight='bold')




def single_grating_harmonic_images(img, harmonicPeriod,
                                   searchRegion=10,
                                   plotFlag=False, verbose=False):

    '''
    Auxiliar function to process the data of single 2D grating Talbot imaging. It
    obtain the (real space) harmonic images  00, 01 and 10.

    Parameters
    ----------
    img : 	ndarray – Data (data_exchange format)
        Experimental image, whith proper blank image, crop and rotation already
        applied.

    harmonicPeriod : list of integers in the format [periodVert, periodHor]
        ``periodVert`` and ``periodVert`` are the period of the harmonics in
        the reciprocal space in pixels. For the checked board grating,
        periodVert = sqrt(2) * pixel Size / grating Period * number of
        rows in the image. For 1D grating, set one of the values to negative or
        zero (it will set the period to number of rows or colunms).

    searchRegion: int
        search for the peak will be in a region of harmonicPeriod/searchRegion
        around the theoretical peak position. See also
        `:py:func:`wavepy.grating_interferometry.plot_harmonic_grid`

    plotFlag: boolean

    verbose: boolean
        verbose mode

    Returns
    -------
    three 2D ndarray data
        Images obtained from the harmonics 00, 01 and 10.

    '''


    #  FFT image
    imgFFT = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img)))


    if plotFlag:
        plot_harmonic_grid(imgFFT, harmonicPeriod=harmonicPeriod, isFFT=True)
        plt.show(block=False)



    imgFFT00 = extract_harmonic(imgFFT,
                                    harmonicPeriod=harmonicPeriod,
                                    harmonic_ij='00',
                                    searchRegion=searchRegion,
                                    isFFT=True,
                                    plotFlag=plotFlag,
                                    verbose=verbose)

    imgFFT01 = extract_harmonic(imgFFT,
                                    harmonicPeriod=harmonicPeriod,
                                    harmonic_ij='01',
                                    searchRegion=searchRegion,
                                    isFFT=True,
                                    plotFlag=plotFlag,
                                    verbose=verbose)

    imgFFT10 = extract_harmonic(imgFFT,
                                    harmonicPeriod=harmonicPeriod,
                                    harmonic_ij=['1','0'],
                                    searchRegion=searchRegion,
                                    isFFT=True,
                                    plotFlag=plotFlag,
                                    verbose=verbose)



    #  Plot Fourier image (intensity)
    if plotFlag:

        # Intensity is Fourier Space
        intFFT00 = np.log10(np.abs(imgFFT00))
        intFFT01 = np.log10(np.abs(imgFFT01))
        intFFT10 = np.log10(np.abs(imgFFT10))

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14,4))
        for dat, ax, textTitle in zip([intFFT00, intFFT01, intFFT10], axes.flat,
                                      ['FFT 00', 'FFT 01', 'FFT 10']):

            # The vmin and vmax arguments specify the color limits
            im = ax.imshow(dat, cmap='Spectral_r', vmin=np.min(intFFT00),
                           vmax=np.max(intFFT00))

            ax.set_title(textTitle)


        # Make an axis for the colorbar on the right side
        cax = fig.add_axes([0.92, 0.1, 0.03, 0.8])
        fig.colorbar(im, cax=cax)
        plt.suptitle('FFT subsets - Intensity', fontsize=18, weight='bold')
        plt.show(block=True)


    # non existing harmonics will return NAN, so here we check NAN

    img00 = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(imgFFT00)))

    if np.all(np.isfinite(imgFFT01)):
        img01 = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(imgFFT01)))
    else:
        img01 = imgFFT01

    if np.all(np.isfinite(imgFFT10)):
        img10 = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(imgFFT10)))
    else:
        img10 = imgFFT10

    return (img00, img01, img10)


def single_2Dgrating_analyses(img, img_ref, harmonicPeriod,
                              unwrapFlag=True, plotFlag=True):

    '''
    Function to process the data of single 2D grating Talbot imaging. It
    wraps other functions in order to make all the process transparent

    '''


    # Obtain Harmonic images
    h_img = single_grating_harmonic_images(img, harmonicPeriod,
                                           plotFlag=plotFlag)

    h_img_ref = single_grating_harmonic_images(img_ref, harmonicPeriod,
                                               plotFlag=plotFlag)

    int00 = np.abs(h_img[0])/np.abs(h_img_ref[0])
    int01 = np.abs(h_img[1])/np.abs(h_img_ref[1])
    int10 = np.abs(h_img[2])/np.abs(h_img_ref[2])

    darkField01 = int01*int00
    darkField10 = int10*int00

    diffPhase01 = np.arctan2(np.real(h_img[1]/h_img_ref[1]),
                             np.imag(h_img[1]/h_img_ref[1]))
    diffPhase10 = np.arctan2(np.real(h_img[2]/h_img_ref[2]),
                             np.imag(h_img[2]/h_img_ref[2]))

    if unwrapFlag:

        diffPhase01 = unwrap(diffPhase01,
                             wrap_around_axis_0=True,
                             wrap_around_axis_1=False,
                             wrap_around_axis_2=False)

        diffPhase10 = unwrap(diffPhase10,
                             wrap_around_axis_0=False,
                             wrap_around_axis_1=True,
                             wrap_around_axis_2=False)

        diffPhase01 -= np.mean(diffPhase01)
        diffPhase10 -= np.mean(diffPhase10)

    return [int00, int01, int10,
            darkField01, darkField10,
            diffPhase01, diffPhase10]


def visib_1st_harmonics(img, harmonicPeriod, searchRegion=20):
    '''
    This function obtain the visibility in a grating imaging experiment by the
    ratio of the amplitudes of the first and zero harmonics. See
    https://doi.org/10.1364/OE.22.014041 .

    Note
    ----
    Note that the absolute visibility also depends on the higher harmonics, and
    for a absolute value of visibility all of them must be considered.


    Parameters
    ----------
    img : 	ndarray – Data (data_exchange format)
        Experimental image, whith proper blank image, crop and rotation already
        applied.

    harmonicPeriod : list of integers in the format [periodVert, periodHor]
        ``periodVert`` and ``periodVert`` are the period of the harmonics in
        the reciprocal space in pixels. For the checked board grating,
        periodVert = sqrt(2) * pixel Size / grating Period * number of
        rows in the image. For 1D grating, set one of the values to negative or
        zero (it will set the period to number of rows or colunms).

    searchRegion: int
        search for the peak will be in a region of harmonicPeriod/searchRegion
        around the theoretical peak position. See also
        `:py:func:`wavepy.grating_interferometry.plot_harmonic_grid`


    Returns
    -------
    (float, float)
        horizontal and vertical visibilities respectivelly from
        harmonics 01 and 10


    '''

    (imgFFT00,
     imgFFT01,
     imgFFT10) = single_grating_harmonic_images(img,
                                                [harmonicPeriod[0],
                                                 harmonicPeriod[1]],
                                                searchRegion=searchRegion,
                                                plotFlag=False)
    img00 = np.fft.ifft(imgFFT00)
    img01 = np.fft.ifft(imgFFT01)
    img10 = np.fft.ifft(imgFFT10)

    return (np.sum(np.abs(img01))/np.sum(np.abs(img00)),
            np.sum(np.abs(img10))/np.sum(np.abs(img00)))


#
