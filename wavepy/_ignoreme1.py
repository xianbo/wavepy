# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 15:19:12 2016

@author: grizolli
"""

import numpy as np
import skimage.io

xgrid, ygrid = np.mgrid[-3:3:501j,-3:3:501j]

foo1 = np.sinc(xgrid**2 + ygrid**2) # + np.random.rand(501,501)*.1
#foo2 = np.sinc(xgrid**2/1.10 + ygrid**2/1.105) # + np.random.rand(501,501)*.1
foo2 = foo1*0.0
foo2[-490:,-490:] = foo1[0:490,0:490]


foo1 = (foo1 - np.min(foo1)) / np.max(foo1 - np.min(foo1))
foo2 = (foo2 - np.min(foo2)) / np.max(foo2 - np.min(foo2))


foo1 = ((foo1*2**8)-1).astype(int)
foo2 = ((foo2*2**8)-1).astype(int)


skimage.io.imsave('foo1.tif', foo1, plugin='pil')
skimage.io.imsave('foo2.tif', foo2, plugin='pil')





