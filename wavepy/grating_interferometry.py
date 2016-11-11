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


__authors__ = "Walan Grizolli"
__copyright__ = "Copyright (c) 2016, Affiliation"
__version__ = "0.1.0"
__docformat__ = "restructuredtext en"
__all__ = ['frankotchellappa', 'error_integration']




def _idxPeak_ij(i, j, nRows, nColumns, periodVert, periodHor):
    return [nRows // 2 + i * periodVert, nColumns // 2 + j * periodHor]

#def extract_harmonic(imgFFT, pixelSize, gratingPeriod, harmonic_ij='00',
#                     plotFlag=False, verbose=True):

def extract_harmonic(imgFFT, harmonicPeriod,
                     harmonic_ij='00',
                     plotFlag=False, verbose=True):

    '''
    Function to extract one harmonic image of the FFT of single 2D grating Talbot imaging.

    Note that it is the FFT of the real image tis required.

    ``Q: Why not the real image??``.
    A: Because FFT can be time consuming. If we use the real image, it will
    be necessary to run FFT for each harmonic. It is encourage to wrap this function within
    a function that do the FFT, extract the harmonics, and return the real space harmonic image.

    Parameters
    ----------
    imgFFT : 	ndarray – Data (data_exchange format)
        FFT of the Experimental image, whith proper blank image, crop and rotation already
        applied.


    harmonicPeriod : list of integers in the format [periodVert, periodHor]
        ``periodVert`` and ``periodVert`` are the period of the harmonics in
        the reciprocal space in pixels. For the checked board grating,
        periodVert = sqrt(2) * pixel Size / grating Period * number of
        rows in the image

    harmonic_ij : string or list of string
        string with the harmonic to extract, for instance '00', '01', '10'
        or '11'. In this notation negative harmonics are not allowed.

        Alternativelly, it accepts a list of string
        ``harmonic_ij=[harmonic_Vertical, harmonic_Horizontal]``, for instance
        ``harmonic_ij=['0', '-1']``

        Note that since the original image contain only real numbers (not
        complex), then negative and positive harmonics are symetric
        related to zero.

    Returns
    -------
    img00, img01, img10: three 2D ndarray data
        Images obtained from the harmonics 00, 01 and 10.

    '''


    #  information from the image

    (nRows, nColumns) = imgFFT.shape

    periodVert = harmonicPeriod[0]
    periodHor = harmonicPeriod[1]

    intensity = (np.abs(imgFFT))
    
    harV = int(harmonic_ij[0])
    harH = int(harmonic_ij[1])



    #  Estimate harmonic positions

    idxPeak_ij = _idxPeak_ij(harV, harH, nRows, nColumns, periodVert, periodHor)
    
    maskHarmRegion = np.zeros((nRows, nColumns))

    maskHarmRegion[idxPeak_ij[0] - periodVert//2:idxPeak_ij[0] + periodVert//2,
                   idxPeak_ij[1] - periodHor//2:idxPeak_ij[1] + periodHor//2] = 1.0


    # correct idxPeak_ij to the experimental value
    idxPeak_ij_exp = np.where(intensity*maskHarmRegion == np.max(np.abs(imgFFT*maskHarmRegion)))


    idxPeak_ij_exp = [idxPeak_ij_exp[0][0], idxPeak_ij_exp[1][0]]
    # convert to list

    if verbose:
        print("MESSAGE: extract_harmonic:" +
              " harmonic peak " + harmonic_ij[0] + harmonic_ij[1] +
              " is misplaced by:")
        print("MESSAGE: {:d} pixels in vertical, {:d} pixels in hor".format(
               (idxPeak_ij_exp[0]-idxPeak_ij[0]),
               (idxPeak_ij_exp[1]-idxPeak_ij[1])))
               
    if plotFlag:
        
        from matplotlib.patches import Rectangle
        plt.figure()
        plt.imshow(np.log10(intensity))
        
        plt.gca().add_patch(Rectangle((idxPeak_ij_exp[1] - periodHor//2,
                                       idxPeak_ij_exp[0] - periodVert//2),
                                        periodHor, periodVert,
                                        lw=2, ls='--', color='red',
                                        fill=None, alpha=1))

        plt.title('Selected Region', fontsize=18, weight='bold')
        plt.show(block=True)               

    return imgFFT[idxPeak_ij_exp[0] - periodVert//2:
                  idxPeak_ij_exp[0] + periodVert//2,
                  idxPeak_ij_exp[1] - periodHor//2:
                  idxPeak_ij_exp[1] + periodHor//2]




def plot_harmonic_grid(imgFFT, harmonicPeriod):

    '''
    Takes the FFT of single 2D grating Talbot imaging and plot the grid from
    where we extract the harmonic in a image of the



    Parameters
    ----------
    imgFFT : 	ndarray – Data (data_exchange format)
        FFT of the Experimental image, whith proper blank image, crop and rotation already
        applied.


    pixelSize : float or list of float
        Detector pixel size in meters. If pixel is not square, use
        ``pixelSize=[pixelSizeVertical, pixelSizeHorizontal]``

    gratingPeriod : float or list of float
        Grating line period in lines per meters. If period is the same in both directions,
        use ``gratingPeriod=[gratingPeriodVertical, gratingPeriodHorizontal]``

    harmonic_ij : string
        string with the harmonic to extract, for instance '00', '01', '10' or '11'. Note
        that negative harmonics are not allowed, but since the original image contain
        only real numbers (not complex), then negative and positive harmonics are symetric
        related to zero


    '''

    (nRows, nColumns) = imgFFT.shape

    periodVert = harmonicPeriod[0]
    periodHor = harmonicPeriod[1]


    plt.figure()
    plt.imshow(np.log10(np.abs(imgFFT)))

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
    plt.title('Hamonics Subset and indexes', fontsize=18, weight='bold')
    plt.show()





