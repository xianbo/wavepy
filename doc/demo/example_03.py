#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script 
"""

from __future__ import print_function
import dxchange
import matplotlib.pyplot as plt

if __name__ == '__main__':


    fname = '/Users/decarlo/Downloads/raw_data_TIFF/singleshot/Be1x50um_8keV_10s_T3_001/Be1x50um_in_8keV_10s_T3_001.tif'
    img = dxchange.read_tiff(fname)
    plt.imshow(img, cmap='Greys_r')
    plt.show()