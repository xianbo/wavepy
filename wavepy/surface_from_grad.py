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


Function to obtain the surface from gradient data
-------------------------------------------------

In many x-ray imaging techniques we obtain the differential phase of the
wavefront in two directions. This is the same as to say that we measured
the gradient of the phase. Therefore to obtain the phase we need a method to
integrate the differential data.

This is not as straight forward as it looks. The main reason is that, to be
able to calculate the function from its gradient, the partial derivatives must
be integrable (that is, the two cross partial derivative of the function must
be equal). However, due to experimental errors and noises, there are no
guarantees that the data is integrable (very likely they are not).


Different methods have been developed in other areas of science, in special
computer vision, where this problem is refered as "Surface Reconstruction
from Gradient Fields", and for consistense we will (try to) use this term.


The simplest method is the so-called Frankot-Chelappa method. The idea
behind this method is to search (calculate) an integrable gradient field
that best fits the data. Luckly, Frankot Chelappa were able in they article
to find a simple single (non interective) equation for that. We are
even luckier since this equation makes use of FFT, which is very
computationally efficient.

However it is  is well known that Frankot-Chelappa is not the best method. More
advanced alghorithms are available, where it is used more complex math and
interactive methods. Unfortunatelly, these algorothims are only available for
MATLAB.


There are plans to make use of such methods in the future, but for now we
only provide the Frankot-Chelappa. It is advised to use some kind of check
for the integration, for instance by calculating the gradient from the result
and comparing with the original gradient.


"""

# import itertools
# import numpy as np
# import matplotlib.pyplot as plt
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


__authors__ = "Walan Grizolli"
__copyright__ = "Copyright (c) 2016, Affiliation"
__version__ = "0.1.0"
__docformat__ = "restructuredtext en"
__all__ = ['frankotchellappa']


def frankotchellappa(del_f_del_x, del_f_del_y, reflec_pad=True):
    """

    Frankt-Chellappa Algorithm

    Parameters
    ----------

    del_f_del_x, del_f_del_y : ndarrays
        2 dimensional gradient data

    reflec_pad: bool
       This flag pad the gradient field in order to obtain a 2-dimensional
       reflected function. See more in the Notes below.

    Returns
    -------
    ndarray
        Integrated data, as provided by the Frankt-Chellappa Algorithm.


    Notes
    -----


    * Padding

        Frankt-Chellappa makes intensive use of the Discrete Fourier
        Transform (DFT), and due to the periodicity property of the DFT, the
        result of the integration will also be periodic (even though we
        only get one period of the answer). This property can result in a
        discontinuity at the edges, and Frankt-Chellappa method is badly
        affected by discontinuity,

        In this sense the idea of this padding is that by reflecting the
        function at the edges we avoid discontinuity. This was inspired by
        the code of the function
        `DfGBox
        <https://www.mathworks.com/matlabcentral/fileexchange/45269-dfgbox>`_,
        available in the MATLAB File Exchange website.

        Note that, since we only have the gradient data, we need to consider
        how a reflection at the edges will affect the partial derivatives. We
        show it below without proof (but it is easy to see).

        First lets consider the data for the :math:`x` direction derivative
        :math:`\\Delta_x = \\dfrac{\\partial f}{\\partial x}` consisting of a
        2D array of size :math:`N \\times M`. The padded matrix
        will be given by:

        .. math::
            \\left[
            \\begin{matrix}
              \\Delta_x(x, y) & -\\Delta_x(N-x, y) \\\\
              \\Delta_x(x, M-y) & -\\Delta_x(N-x, M-y)
            \\end{matrix}
            \\right]

        and for the for the y direction derivative
        :math:`\\Delta_y = \\dfrac{\\partial f}{\\partial y}` we have

        .. math::
            \\left[
            \\begin{matrix}
              \\Delta_y(x, y) & \\Delta_y(N-x, y) \\\\
              -\\Delta_y(x, M-y) & -\\Delta_y(N-x, M-y)
            \\end{matrix}
            \\right]

        Note that this padding increases the number of points from
        :math:`N \\times M` to :math:`2Mx2N`. However, **the function only
        returns the** :math:`N \\times M` **result**, since the other parts are
        only a repetion of the result. In other words, the padding is done
        only internally.


    See Also
    --------
        `Original Frankt-Chellappa Algorithm
        paper <http://dx.doi.org/10.1109/34.3909l>`_.

    """

    from numpy.fft import fft2, ifft2, fftfreq

    if reflec_pad:
        del_f_del_x, del_f_del_y = _reflec_pad_grad_fields(del_f_del_x,
                                                           del_f_del_y)

    NN, MM = del_f_del_x.shape
    wx, wy = np.meshgrid(fftfreq(MM) * 2 * np.pi,
                         fftfreq(NN) * 2 * np.pi, indexing='xy')

    numerator = -1j * wx * fft2(del_f_del_x) - 1j * wy * fft2(del_f_del_y)

    denominator = (wx) ** 2 + (wy) ** 2 + np.finfo(float).eps

    res = ifft2(numerator / denominator)
    res -= np.mean(np.real(res))

    if reflec_pad:
        return _one_forth_of_array(res)
    else:
        return res


def _reflec_pad_grad_fields(del_func_x, del_func_y):
    """

    This fucntion pad the gradient field in order to obtain a 2-dimensional
    reflected function. The idea is that, by having an reflected function,
    we avoid discontinuity at the edges.


    This was inspired by the code of the function DfGBox, available in the
    MATLAB File Exchange website:
    https://www.mathworks.com/matlabcentral/fileexchange/45269-dfgbox

    """

    del_func_x_c1 = np.concatenate((del_func_x,
                                    del_func_x[::-1, :]), axis=0)

    del_func_x_c2 = np.concatenate((-del_func_x[:, ::-1],
                                    -del_func_x[::-1, ::-1]), axis=0)

    del_func_x = np.concatenate((del_func_x_c1, del_func_x_c2), axis=1)

    del_func_y_c1 = np.concatenate((del_func_y,
                                    -del_func_y[::-1, :]), axis=0)

    del_func_y_c2 = np.concatenate((del_func_y[:, ::-1],
                                    -del_func_y[::-1, ::-1]), axis=0)

    del_func_y = np.concatenate((del_func_y_c1, del_func_y_c2), axis=1)

    return del_func_x, del_func_y


def _one_forth_of_array(array):
    """
    Undo for the function
    :py:func:`wavepy:surface_from_grad:_reflec_pad_grad_fields`

    """

    array, _ = np.array_split(array, 2, axis=0)
    return np.array_split(array, 2, axis=1)[0]
