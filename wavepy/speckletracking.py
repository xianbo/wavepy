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
Functions for speckle tracking analises
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from skimage.feature import register_translation

from multiprocessing import Pool, cpu_count

import wavepy.utils as utils

__authors__ = "Walan Grizolli"
__copyright__ = "Copyright (c) 2016, Affiliation"
__version__ = "0.1.0"
__docformat__ = "restructuredtext en"
__all__ = ['speckleDisplacement',
           'speckleDisplacementMulticore',
           'function_03']





def speckleDisplacement(image, image_ref, halfsubwidth=10,
                        stride=1, npoints=None, subpixelResolution = 1):



    if npoints is not None:
        stride = int(image.shape[0] / npoints) - 1
        if stride <= 0: stride = 1  # note that this is not very precise

    irange = np.arange(halfsubwidth,
                       image.shape[0] - halfsubwidth + 1,
                       stride)
    jrange = np.arange(halfsubwidth,
                       image.shape[1] - halfsubwidth + 1,
                       stride)

    pbar = tqdm(total=np.size(irange))  # progress bar

    sx = np.ones(image.shape)*NAN
    sy = np.ones(image.shape)*NAN
    error = np.ones(image.shape)*NAN


    for (i, j) in itertools.product(irange, jrange):

        interrogation_window = image_ref[i - halfsubwidth:i + halfsubwidth + 1,
                                         j - halfsubwidth:j + halfsubwidth + 1]

        sub_image = image[i - halfsubwidth:i + halfsubwidth + 1,
                          j - halfsubwidth:j + halfsubwidth + 1]

        shift, error_ij, _ = register_translation(sub_image,
                                                  interrogation_window,
                                                  subpixelResolution)

        sx[i, j] = shift[1]
        sy[i, j] = shift[0]
        error[i, j] = error_ij

        if j == jrange[-1]: pbar.update()  # update progress bar

    print(" ")

    return (sx[halfsubwidth:-halfsubwidth:stride,
               halfsubwidth:-halfsubwidth:stride],
            sy[halfsubwidth:-halfsubwidth:stride,
               halfsubwidth:-halfsubwidth:stride],
            error[halfsubwidth:-halfsubwidth:stride,
                  halfsubwidth:-halfsubwidth:stride],
            stride)


#==============================================================================
# %% Data Analysis Multicore Loop
#==============================================================================



def _func(args1 ,parList):
#    time.sleep(0.5)

    i = args1[0]
    j = args1[1]
    image = parList[0]
    image_ref = parList[1]
    halfsubwidth = parList[2]
    subpixelResolution = parList[3]

    #    print(i, j ,halfsubwidth, subpixelResolution)

    interrogation_window = image_ref[i - halfsubwidth:i + halfsubwidth + 1,
                                     j - halfsubwidth:j + halfsubwidth + 1]

    sub_image = image[i - halfsubwidth:i + halfsubwidth + 1,
                      j - halfsubwidth:j + halfsubwidth + 1]

    shift, error_ij, _ = register_translation(sub_image,
                                              interrogation_window,
                                                  subpixelResolution)

#    for _ in range(100000000): pass
#    print('i,j : {0}, {1}'.format(i,j))
    return shift[1], shift[0], error_ij


def speckleDisplacementMulticore(image, image_ref, halfsubwidth=10,
                        stride=1, npoints=None, subpixelResolution = 1,
                        ncores=1/2):



    if npoints is not None:
        stride = int(image.shape[0] / npoints) - 1
        if stride <= 0: stride = 1  # note that this is not very precise

    irange = np.arange(halfsubwidth,
                       image.shape[0] - halfsubwidth + 1,
                       stride)
    jrange = np.arange(halfsubwidth,
                       image.shape[1] - halfsubwidth + 1,
                       stride)

    print(stride)
    print(npoints)

    print("%d cpu's available" % cpu_count())
    p = Pool(processes=int(cpu_count()*ncores))
    print("Using %d cpu's" % p._processes)


    parList = [image, image_ref, halfsubwidth,
               subpixelResolution]


    res = p.starmap_async(_func,
                          zip(itertools.product(irange, jrange),
                              itertools.repeat(parList)),
                          chunksize=1)


    p.close() # No more work


    utils.progressBar4pmap2(res)
    #    print(np.array(res.get()))

    return np.array(res.get())[:,0].reshape(len(irange),len(jrange))




