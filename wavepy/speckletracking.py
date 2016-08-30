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

import wavepy.utils as wpu

from wavepy.cfg import *

__authors__ = "Walan Grizolli"
__copyright__ = "Copyright (c) 2016, Affiliation"
__version__ = "0.1.0"
__docformat__ = "restructuredtext en"
__all__ = ['speckleDisplacement']

# TODO: Remove debug library
from wavepy._my_debug_tools import *


def _speckleDisplacementSingleCore(image, image_ref, halfsubwidth,
                                   stride, subpixelResolution,):

    irange = np.arange(halfsubwidth,
                       image.shape[0] - halfsubwidth + 1,
                       stride)
    jrange = np.arange(halfsubwidth,
                       image.shape[1] - halfsubwidth + 1,
                       stride)

    pbar = tqdm(total=np.size(irange))  # progress bar

    sx = np.ones(image.shape) * NAN
    sy = np.ones(image.shape) * NAN
    error = np.ones(image.shape) * NAN

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


# ==============================================================================
# %% Data Analysis Multicore
# ==============================================================================



def _func_4_starmap_async(args, parList):
    i = args[0]
    j = args[1]
    image = parList[0]
    image_ref = parList[1]
    halfsubwidth = parList[2]
    subpixelResolution = parList[3]


    interrogation_window = image_ref[i - halfsubwidth:i + halfsubwidth + 1,
                           j - halfsubwidth:j + halfsubwidth + 1]

    sub_image = image[i - halfsubwidth:i + halfsubwidth + 1,
                j - halfsubwidth:j + halfsubwidth + 1]

    shift, error_ij, _ = register_translation(sub_image,
                                              interrogation_window,
                                              subpixelResolution)

    return shift[1], shift[0], error_ij


def _speckleDisplacementMulticore(image, image_ref, halfsubwidth,
                                  stride, subpixelResolution,
                                  ncores, taskPerCore):

    print('MESSAGE: _speckleDisplacementMulticore:')
    print("MESSAGE: %d cpu's available" % cpu_count())
    p = Pool(processes=int(cpu_count() * ncores))
    print("MESSAGE: Using %d cpu's" % p._processes)

    irange = np.arange(halfsubwidth,
                       image.shape[0] - halfsubwidth + 1,
                       stride)
    jrange = np.arange(halfsubwidth,
                       image.shape[1] - halfsubwidth + 1,
                       stride)

    parList = [image, image_ref, halfsubwidth,
               subpixelResolution]

    ntasks = np.size(irange) * np.size(jrange)

    chunksize = int(ntasks / p._processes / taskPerCore + 1)

    # DEBUG_print_var("chunksize", chunksize)
    # DEBUG_print_var("ntasks", ntasks)
    # DEBUG_print_var("np.size(irange)", np.size(irange))

    # DEBUG_print_var("np.size(jrange)", np.size(jrange))

    res = p.starmap_async(_func_4_starmap_async,
                          zip(itertools.product(irange, jrange),
                              itertools.repeat(parList)),
                          chunksize=chunksize)

    p.close()  # No more work

    wpu.progress_bar4pmap(res)  # Holds the program in a loop waiting
                                 # starmap_async to finish

    sx = np.array(res.get())[:, 0].reshape(len(irange), len(jrange))
    sy = np.array(res.get())[:, 1].reshape(len(irange), len(jrange))
    error = np.array(res.get())[:, 2].reshape(len(irange), len(jrange))

    return (sx, sy, error, stride)


def speckleDisplacement(image, image_ref, halfsubwidth=10,
                        stride=1, npointsmax=None, subpixelResolution=1,
                        ncores=1/2, taskPerCore=100, verbose=False):


    if npointsmax is not None:
        npoints = int((image.shape[0] - 2 * halfsubwidth) / stride)
        # DEBUG_print_var("npoints", npoints)
        if npoints > npointsmax:
            stride = int((image.shape[0] - 2 * halfsubwidth) / npointsmax)
            # DEBUG_print_var("stride", stride)
        if stride <= 0: stride = 1  # note that this is not very precise

    if verbose:
        print('MESSAGE: speckleDisplacement:')
        print("MESSAGE: stride =  %d" % stride)
        print("MESSAGE: npoints =  %d" %
                int((image.shape[0] - 2 * halfsubwidth) / stride))

    if ncores < 0 or ncores > 1: ncores = 1

    if int(cpu_count() * ncores) <= 1:

        res = _speckleDisplacementSingleCore(image, image_ref,
                                             halfsubwidth=halfsubwidth,
                                             stride=stride, subpixelResolution=1)

    else:

        res = _speckleDisplacementMulticore(image, image_ref,
                                            halfsubwidth=halfsubwidth,
                                            stride=stride, subpixelResolution=1,
                                            ncores=ncores, taskPerCore=taskPerCore)

    return res
