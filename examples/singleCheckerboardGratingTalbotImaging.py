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

import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import dxchange

import wavepy.utils as wpu
import wavepy.grating_interferometry as wgi

import easygui_qt as easyqt
import xraylib

rad2deg = np.rad2deg(1)
deg2rad = np.deg2rad(1)
hc = wpu.hc  # hc

def main_single_gr_Talbot(img, imgRef,
                          phenergy, pixelsize, distDet2sample,
                          period_harm, saveFileSuf,
                          unwrapFlag=True,
                          plotFlag=True,
                          saveFigFlag=False):

    global inifname  # name of .ini file

    [period_harm_Vert, period_harm_Hor] = period_harm


    #    img, imgRef = wpu.align_two_images(img, imgRef)

    # Crop

    img_size_o = np.shape(img)

    # take index from ini file
    idx4crop = list(map(int, (wpu.get_from_ini_file(inifname, 'Parameters',
                                                    'Crop').split(','))))

    # Plot Real Image wiht default crop

    tmpImage = wpu.crop_matrix_at_indexes(img, idx4crop)

    plt.figure()
    plt.imshow(tmpImage,
               cmap='viridis',
               extent=wpu.extent_func(tmpImage, pixelsize)*1e6)
    plt.xlabel(r'$[\mu m]$')
    plt.ylabel(r'$[\mu m]$')
    plt.colorbar()
    plt.title('Raw Image with initial Crop', fontsize=18, weight='bold')

    plt.pause(.5)
    # ask if the crop need to be changed
    newCrop = easyqt.get_yes_or_no('New Crop?')

    if saveFigFlag and not newCrop:
        wpu.save_figs_with_idx(saveFileSuf + '_Talbot_image')
    plt.close(plt.gcf())

    if newCrop:

        colorlimit = [wpu.mean_plus_n_sigma(img, -4),
                      wpu.mean_plus_n_sigma(img, 4)]

        [colorlimit,
         cmap] = wpu.plot_slide_colorbar(img,
                                         title='SELECT COLOR SCALE,\n' +
                                         'Raw Image, No Crop',
                                         xlabel=r'x [$\mu m$ ]',
                                         ylabel=r'y [$\mu m$ ]',
                                         min_slider_val=colorlimit[0],
                                         max_slider_val=colorlimit[1],
                                         extent=wpu.extent_func(img,
                                                                pixelsize)*1e6)

        idx4crop = wpu.graphical_roi_idx(img, verbose=True,
                                         kargs4graph={'cmap': cmap,
                                                      'vmin': colorlimit[0],
                                                      'vmax': colorlimit[1]})
        wpu.set_at_ini_file(inifname, 'Parameters', 'Crop',
                            '{}, {}, {}, {}'.format(idx4crop[0], idx4crop[1],
                                                    idx4crop[2], idx4crop[3]))

        img = wpu.crop_matrix_at_indexes(img, idx4crop)

        # Plot Real Image AFTER crop

        plt.imshow(img, cmap='viridis',
                   extent=wpu.extent_func(img, pixelsize)*1e6)
        plt.xlabel(r'$[\mu m]$')
        plt.ylabel(r'$[\mu m]$')
        plt.colorbar()
        plt.title('Raw Image with New Crop', fontsize=18, weight='bold')

        if saveFigFlag:
            wpu.save_figs_with_idx(saveFileSuf + '_Talbot_image')
        plt.show(block=True)

    else:
        img = tmpImage

    imgRef = wpu.crop_matrix_at_indexes(imgRef, idx4crop)

    # calculate harmonic position after crop

    period_harm_Vert = int(period_harm_Vert*(idx4crop[1] - idx4crop[0]) /
                           img_size_o[0])
    period_harm_Hor = int(period_harm_Hor*(idx4crop[3] - idx4crop[2]) /
                          img_size_o[1])

    # Obtain harmonic periods from images

    (period_harm_Vert,
     period_harm_Hor) = wgi.exp_harm_period(img, [period_harm_Vert,
                                            period_harm_Hor],
                                            harmonic_ij=['1', '1'],
                                            searchRegion=10,
                                            isFFT=False, verbose=True)

    # Calculate everything

    harmPeriod = [period_harm_Vert, period_harm_Hor]

    [int00, int01, int10,
     darkField01, darkField10,
     phaseFFT_01,
     phaseFFT_10] = wgi.single_2Dgrating_analyses(img, imgRef,
                                                  harmonicPeriod=harmPeriod,
                                                  plotFlag=plotFlag,
                                                  unwrapFlag=unwrapFlag,
                                                  verbose=True)

    virtual_pixelsize = [0, 0]
    virtual_pixelsize[0] = pixelsize[0]*img.shape[0]/int00.shape[0]
    virtual_pixelsize[1] = pixelsize[1]*img.shape[1]/int00.shape[1]

    diffPhase01 = phaseFFT_01*virtual_pixelsize[1]/distDet2sample/hc*phenergy
    diffPhase10 = phaseFFT_10*virtual_pixelsize[0]/distDet2sample/hc*phenergy

    return [int00, int01, int10,
            darkField01, darkField10,
            diffPhase01, diffPhase10,
            virtual_pixelsize]


# %%
def _intial_gui_setup(argvzero):

    global inifname  # name of .ini file
    pwd, inifname = argvzero.rsplit('/', 1)
    inifname = pwd + '/.' + inifname.replace('.py', '.ini')

    defaults = wpu.load_ini_file(inifname)

    if defaults is None:
        p1, p2, p3, p4, p5, p6 = [0, 0, 0, 0, 0, 0]

    else:

        p1 = float(defaults['Parameters'].get('Pixel Size'))
        p2 = float(defaults['Parameters'].get('Chekerboard Grating Period'))
        p3 = defaults['Parameters'].get('Pattern')
        p4 = float(defaults['Parameters'].get('Distance Detector to Gr'))
        p5 = float(defaults['Parameters'].get('Photon Energy'))
        p6 = float(defaults['Parameters'].get('Source Distance'))

    title = 'Select Sample, Reference and Dark Images. '
    title += 'Press ESC to repeat last run.'
    fname1, fname2, fname3 = wpu.gui_load_data_ref_dark_filenames(title=title)

    if fname1 is None:
        fname1 = defaults['Files'].get('Sample')
        fname2 = defaults['Files'].get('Reference')
        fname3 = defaults['Files'].get('Blank')

    else:
        defaults['Files']['Sample'] = fname1
        defaults['Files']['Reference'] = fname2
        defaults['Files']['Blank'] = fname3

    wpu.print_blue('MESSAGE: loading dark: ' + fname3)
    wpu.print_blue('MESSAGE: loading Reference: ' + fname2)
    wpu.print_blue('MESSAGE: loading Image: ' + fname1)

    pixelsize = easyqt.get_float("Enter Pixel Size [um]",
                                 title='Experimental Values',
                                 default_value=p1*1e6)*1e-6

    gratingPeriod = easyqt.get_float("Enter CB Grating Period [um]",
                                     title='Experimental Values',
                                     default_value=p2*1e6)*1e-6

    if p3 == 'Diagonal half pi':
        choices = ['Diagonal half pi', 'Edge pi']
    else:
        choices = ['Edge pi', 'Diagonal half pi']

    pattern = easyqt.get_choice(message='Select CB Grating Pattern',
                                title='Title',
                                choices=choices)

    distDet2sample = easyqt.get_float("Enter Distance Sample - Detector [mm]",
                                      title='Experimental Values',
                                      default_value=p4*1e3)*1e-3

    phenergy = easyqt.get_float("Enter Photon Energy [KeV]",
                                title='Experimental Values',
                                default_value=p5*1e-3)*1e3

    sourceDistance = easyqt.get_float("Enter Distance to Source [m]",
                                      title='Experimental Values',
                                      default_value=p6)

    defaults['Parameters']['Pixel Size'] = str(pixelsize)
    defaults['Parameters']['Chekerboard Grating Period'] = str(gratingPeriod)
    defaults['Parameters']['Pattern'] = pattern
    defaults['Parameters']['Distance Detector to Gr'] = str(distDet2sample)
    defaults['Parameters']['Photon Energy'] = str(phenergy)
    defaults['Parameters']['Source Distance'] = str(sourceDistance)

    with open(inifname, 'w') as configfile:
            defaults.write(configfile)

    return (fname1, fname2, fname3,
            pixelsize, gratingPeriod, pattern, distDet2sample,
            phenergy, sourceDistance)


# %%
def _load_experimental_pars(argv):

    if len(argv) == 10:

        fname_img, fname_imgRef, fname_blank = argv[1:4]

        pixelsize = float(argv[4])*1e-6
        gratingPeriod = float(argv[5])*1e-6
        pattern = argv[6]
        distDet2sample = float(argv[7])*1e-3
        phenergy = float(argv[8])*1e3
        sourceDistance = float(argv[9])

        img, imgRef, blank = (dxchange.read_tiff(fname_img),
                              dxchange.read_tiff(fname_imgRef),
                      dxchange.read_tiff(fname_blank))

    elif len(argv) == 1:

        (fname_img, fname_imgRef, fname_blank,
         pixelsize, gratingPeriod, pattern, distDet2sample,
         phenergy, sourceDistance) = _intial_gui_setup(argv[0])

        img, imgRef, blank = (dxchange.read_tiff(fname_img),
                              dxchange.read_tiff(fname_imgRef),
                      dxchange.read_tiff(fname_blank))

    else:
        print('ERROR: wrong number of inputs: {} \n'.format(len(argv)-1) +
              'Usage: \n'
              '\n'
              'singleGratingTalbotImaging.py : (no inputs) load dialogs \n'
              '\n'
              'singleGratingTalbotImaging.py [args] \n'
              '\n'
              'arg1: file name main image\n'
              'arg2: file name reference image\n'
              'arg3: file name dark image\n'
              'arg4: pixel size [um]\n'
              'arg5: Check Board grating period [um]\n'
              "arg6: pattern, 'Edge pi' or 'Diagonal half pi' \n"
              'arg7: distance detector to CB Grating [mm]\n'
              'arg8: Photon Energy [KeV]\n'
              'arg9: Distance to the source [m], to correct for beam\n'
              '      divergence (use 1e5 to ignore this, which means\n'
              '      source at infinity and zero divergence)\n'
              '\n')

        exit(-1)

    img = img - blank
    imgRef = imgRef - blank

    pixelsize = [pixelsize, pixelsize]
    # change here if you need rectangular pixel

    if pattern == 'Diagonal half pi':
        gratingPeriod *= 1.0/np.sqrt(2.0)
        phaseShift = 'halfPi'

    elif pattern == 'Edge pi':
        gratingPeriod *= 1.0/2.0
        phaseShift = 'Pi'

    saveFileSuf = 'cb{:.2f}um_'.format(gratingPeriod*1e6)
    saveFileSuf += phaseShift
    saveFileSuf += '_d{:.0f}mm_'.format(distDet2sample*1e3)
    saveFileSuf += '{:.1f}KeV'.format(phenergy*1e-3)
    saveFileSuf = saveFileSuf.replace('.', 'p')

    return (img, imgRef, saveFileSuf,
            pixelsize, gratingPeriod, pattern,
            distDet2sample,
            phenergy, sourceDistance)

# %%
def _get_delta_gui(phenergy):

    choices = ['Diamond, 3.525g/cm^3',
               'Be, 1.848 g/cm^3',
               'Manual Input']

    menu_choices = [choices[0], choices[1], choices[2]]  # Change order here!

    choice = easyqt.get_choice(message='Select Sample Material',
                               title='Title',
                               choices=menu_choices)

    if choice is None:
        choice = menu_choices[0]

    if choice == choices[0]:
        # delta Diamond, density from wikipedia:
        # delta at 8KeV: 1.146095341e-05
        delta = 1 - xraylib.Refractive_Index_Re("C", phenergy/1e3, 3.525)
        material = 'Diamond'

    elif choice == choices[1]:
        # delta at 8KeV = 5.3265E-06
        delta = 1 - xraylib.Refractive_Index_Re("Be", phenergy/1e3,
                                                xraylib.ElementDensity(4))
        material = 'Beryllium'

    elif choice == 'Manual Input':
        # delta Diamond, density from wikipedia:
        material = easyqt.get_string('Enter symbol of material ' +
                                     '(if compounds, you need to' +
                                     ' provide the density):',
                                     title='Thickness Calculation',
                                     default_response='C')

        elementZnumber = xraylib.SymbolToAtomicNumber(material)
        density = xraylib.ElementDensity(elementZnumber)

        density = easyqt.get_float('Density [g/cm^3] ' +
                                   '(Enter for default value)',
                                   title='Thickness Calculation',
                                   default_value=density)

        delta = 1 - xraylib.Refractive_Index_Re(material,
                                                phenergy/1e3, density)

    else:
        wpu.print_red('ERROR: unknown option')

    return delta, material

# %%
def correct_zero_from_unwrap(angleArray):

    pi_jump = int(np.round(angleArray / np.pi))

    j_o, i_o = wpu.graphical_select_point_idx(pi_jump)

    if j_o is not None:
        angleArray -= pi_jump[i_o, j_o]*np.pi

    return angleArray, pi_jump[i_o, j_o]


def _plot_profile(data, pixelsize, title, arg4main={'cmap': 'viridis'}):

    xxGrid, yyGrid = wpu.grid_coord(data, pixelsize)

    wpu.plot_profile(xxGrid*1e6, yyGrid*1e6, data[::-1, :],
                     xlabel=r'$x [\mu m]$', ylabel=r'$y [\mu m]$',
                     title=title,
                     xunit='\mu m', yunit='\mu m',
                     arg4main=arg4main)


# %%
def _default_plot_for_pickle(data, pixelsize, patternforpickle='graph',
                             title='', xlabel=r'$x$', ylabel=r'$y$', ctitle='',
                             removeSpark=True, cmap='viridis'):

    if removeSpark:
        vmin = wpu.mean_plus_n_sigma(data, -6)
        vmax = wpu.mean_plus_n_sigma(data, 6)
    else:
        vmin = np.min(data)
        vmax = np.max(data)

    #    vmax = np.max((np.abs(vmin), np.abs(vmax)))
    #    vmin = -vmax

    fig = plt.figure(figsize=(12, 9.5))

    plt.imshow(data,
               extent=wpu.extent_func(data, virtual_pixelsize)*1e6,
               cmap=cmap, vmin=vmin, vmax=vmax)

    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)

    cbar = plt.colorbar(shrink=0.9)
    cbar.ax.set_title(ctitle, y=1.01)

    plt.title(title, fontsize=24, weight='bold', y=1.01)

    wpu.save_figs_with_idx_pickle(fig, patternforpickle)

    plt.show(block=True)


# %%
def correct_zero_DPC(dpc01, dpc10,
                     pixelsize, distDet2sample, phenergy, saveFileSuf):

    title = ['Angle displacement of fringes 01',
             'Angle displacement of fringes 10']

    factor = distDet2sample*hc/phenergy

    angle = [dpc01/pixelsize[0]*factor, dpc10/pixelsize[1]*factor]
    dpc = [dpc01, dpc10]

    pi_jump = [0, 0]

    iamhappy = False
    while not iamhappy:

        pi_jump[0] = int(np.round(np.mean(angle[0])/np.pi))
        pi_jump[1] = int(np.round(np.mean(angle[1])/np.pi))

        plt.figure()
        plt.hist(angle[0].flatten()/np.pi, 201,
                 histtype='step')
        plt.hist(angle[1].flatten()/np.pi, 201,
                 histtype='step')

        plt.xlabel(r'Angle [$\pi$rad]')

        plt.title('Correct DPC\n' + r'Angle displacement of fringes $[\pi$ rad]' +
                  '\n' + r'Calculated jumps $x$ and $y$ : ' +
                  '{:d}, {:d} $\pi$'.format(pi_jump[1], pi_jump[0]))

        plt.legend(('DPC x', 'DPC y'))
        plt.show(block=False)
        plt.pause(.5)

        if easyqt.get_yes_or_no('Subtract mean value of DPC?'):
            plt.close(plt.gcf())
            plt.close(plt.gcf())

            angle[0] -= pi_jump[0]*np.pi
            angle[1] -= pi_jump[1]*np.pi

            dpc01 = angle[0]*pixelsize[0]/factor
            dpc10 = angle[1]*pixelsize[1]/factor
            dpc = [dpc01, dpc10]

            wgi.plot_DPC(dpc01, dpc10,
                         virtual_pixelsize, saveFigFlag=True,
                         saveFileSuf=saveFileSuf)
        else:
            plt.close(plt.gcf())
            plt.close(plt.gcf())
            iamhappy = True



    if easyqt.get_yes_or_no('Correct DPC center?'):

        for i in [0, 1]:

            iamhappy = False
            while not iamhappy:

                angle[i], pi_jump[i] = correct_zero_from_unwrap(angle[i])

                wpu.print_blue('VALUES: pi jump ' +
                               '{:}: {:} pi'.format(i, pi_jump[i]))
                plt.figure()
                plt.hist(angle[i].flatten() / np.pi, 101)
                plt.title(r'Angle displacement of fringes $[\pi$ rad]')
                plt.show()

                plt.figure()

                vlim = np.max((np.abs(wpu.mean_plus_n_sigma(angle[i]/np.pi,
                                                            -5)),
                               np.abs(wpu.mean_plus_n_sigma(angle[i]/np.pi,
                                                            5))))

                plt.imshow(angle[i] / np.pi,
                           cmap='RdGy',
                           vmin=-vlim, vmax=vlim)

                plt.colorbar()
                plt.title(title[i] + r' [$\pi$ rad],')
                plt.xlabel('Pixels')
                plt.ylabel('Pixels')

                plt.pause(.5)

                iamhappy = easyqt.get_yes_or_no('Happy?')
                plt.close(plt.gcf())
                plt.close(plt.gcf())

            dpc[i] = angle[i]*pixelsize[i]/factor

    return dpc


# %%
# =============================================================================
# %% Main
# =============================================================================

if __name__ == '__main__':

    # ==========================================================================
    # %% Experimental parameters
    # ==========================================================================

    (img, imgRef, saveFileSuf,
     pixelsize, gratingPeriod, pattern,
     distDet2sample,
     phenergy, sourceDistance) = _load_experimental_pars(sys.argv)

    wavelength = hc/phenergy
    kwave = 2*np.pi/wavelength

    # calculate the theoretical position of the hamonics
    period_harm_Vert = np.int(pixelsize[0]/gratingPeriod*img.shape[0] /
                              (sourceDistance + distDet2sample)*sourceDistance)
    period_harm_Hor = np.int(pixelsize[1]/gratingPeriod*img.shape[1] /
                             (sourceDistance + distDet2sample)*sourceDistance)

    # ==========================================================================
    # %% do the magic
    # ==========================================================================

    result = main_single_gr_Talbot(img, imgRef,
                                   phenergy, pixelsize, distDet2sample,
                                   period_harm=[period_harm_Vert,
                                                period_harm_Hor],
                                   saveFileSuf=saveFileSuf,
                                   unwrapFlag=True,
                                   plotFlag=False,
                                   saveFigFlag=True)

    [int00, int01, int10,
     darkField01, darkField10,
     diffPhase01, diffPhase10,
     virtual_pixelsize] = result

    # due to beam divergence, the image will expand when propagating.
    # The script uses the increase of the pattern period compared to
    # the theoretical period, and apply the same factor to the pixel size.
    # Note that, since the hor and vert divergences can be different, the
    # virtual pixel size can also be different for hor and vert directions
    wpu.print_blue('VALUES: virtual pixelsize i, j: ' +
                   '{:.4f}um, {:.4f}um'.format(virtual_pixelsize[0]*1e6,
                                               virtual_pixelsize[1]*1e6))

    # %%
    # ==========================================================================
    # % Plot
    # ==========================================================================

    #    # %% plot Intensities
    #
    #    wgi.plot_intensities_harms(int00, int01, int10,
    #                               virtual_pixelsize, saveFigFlag=False,
    #                               saveFileSuf=saveFileSuf)
    #
    #    # %% plot dark field
    #
    #    wgi.plot_dark_field(darkField01, darkField10,
    #                        virtual_pixelsize, saveFigFlag=False,
    #                        saveFileSuf=saveFileSuf)

    # %% plot DPC

    wgi.plot_DPC(diffPhase01, diffPhase10,
                 virtual_pixelsize, saveFigFlag=True,
                 saveFileSuf=saveFileSuf)

    plt.pause(.5)

    # %% remove linear component of DPC

    removeLinearFromDPC = easyqt.get_yes_or_no('Remove Linear ' +
                                               'component from DPC?')
    plt.close(plt.gcf())

    if removeLinearFromDPC:

        wpu.log_this('%%% COMMENT: Removed Linear Component from DPC')

        def _fit_lin_surface(zz, pixelsize):

            from numpy.polynomial import polynomial

            xx, yy = wpu.grid_coord(zz, pixelsize)

            f = zz.flatten()
            deg = np.array([1, 1])
            vander = polynomial.polyvander2d(xx.flatten(), yy.flatten(), deg)
            vander = vander.reshape((-1, vander.shape[-1]))
            f = f.reshape((vander.shape[0],))
            c = np.linalg.lstsq(vander, f)[0]

            print(c)

            return polynomial.polyval2d(xx, yy, c.reshape(deg+1))

        linfitDPC01 = _fit_lin_surface(diffPhase01, virtual_pixelsize)
        linfitDPC10 = _fit_lin_surface(diffPhase10, virtual_pixelsize)

        diffPhase01 -= linfitDPC01
        diffPhase10 -= linfitDPC10

        wgi.plot_DPC(diffPhase01, diffPhase10,
                     virtual_pixelsize,
                     titleStr='\n(removed linear DPC component)',
                              saveFigFlag=True, saveFileSuf=saveFileSuf)

    # %% correct DPC
    [diffPhase01,
     diffPhase10] = correct_zero_DPC(diffPhase01, diffPhase10,
                                     virtual_pixelsize,
                                     distDet2sample, phenergy, saveFileSuf)

    vlim01 = np.max(np.abs(diffPhase01))
    vlim10 = np.max(np.abs(diffPhase10))

    #    np.savetxt('dpc_x.dat', diffPhase01, '%.8g',
    #               header='values in meter, pixel size i,j = ' +
    #                      '{:.6g} meters, '.format(virtual_pixelsize[0]) +
    #                      '{:.6g} meters'.format(virtual_pixelsize[1]))
    #
    #    np.savetxt('dpc_y.dat', diffPhase10, '%.8g',
    #               header='values in meter, pixel size i,j = ' +
    #                      '{:.6g} meters, '.format(virtual_pixelsize[0]) +
    #                      '{:.6g} meters'.format(virtual_pixelsize[1]))

    #    _plot_profile(diffPhase01, virtual_pixelsize, title='DPC x',
    #                  arg4main={'cmap': 'RdGy', 'vmin': -vlim01, 'vmax': vlim01})
    #
    #    _plot_profile(diffPhase10, virtual_pixelsize, title='DPC y',
    #                  arg4main={'cmap': 'RdGy', 'vmin': -vlim10, 'vmax': vlim10})

    # ==========================================================================
    # %% Integration
    # ==========================================================================

    if easyqt.get_yes_or_no('New Crop for Integration? ' +
                            "(If 'No', the values from last run are used.)"):

        idx4crop = ''

    else:

        idx4crop = list(map(int,
                            (wpu.get_from_ini_file(inifname,
                                                   'Parameters',
                                                   'crop integration').split(','))))

    phase, idx4crop = wgi.dpc_integration(diffPhase01, diffPhase10,
                                          virtual_pixelsize,
                                          idx4crop=idx4crop,
                                          plotErrorIntegration=True,
                                          shifthalfpixel=True)

    wpu.set_at_ini_file(inifname, 'Parameters', 'crop integration',
                        '{}, {}, {}, {}'.format(idx4crop[0], idx4crop[1],
                                                idx4crop[2], idx4crop[3]))

    phase -= np.min(phase)  # apply here your favorite offset

    #
    #    wgi.plot_integration(1/2/np.pi*phase*wavelength*1e9,
    #                         virtual_pixelsize,
    #                         titleStr=r'WF $[nm]$', saveFigFlag=True,
    #                         saveFigFlag=True, saveFileSuf=saveFileSuf)

    # %%
    ax = wgi.plot_integration(1/2/np.pi*phase, virtual_pixelsize,
                              titleStr=r'WF $[\lambda$ units $]$',
                              saveFigFlag=True, saveFileSuf=saveFileSuf)

    ax.view_init(elev=30, azim=60)

    makeAnimation = False
    if makeAnimation is True:
        wpu.rocking_3d_figure(ax, saveFileSuf + '.ogv',
                              elevOffset=0, azimOffset=60,
                              elevAmp=45, azimAmpl=45, dpi=80, npoints=200)

    plt.show(block=True)

    #    wgi.plot_integration(phase/2/np.pi, virtual_pixelsize,
    #                         titleStr=r'Phase/2$\pi$ [ radians]',
    #                         saveFigFlag=True, saveFileSuf=saveFileSuf)

    # ==========================================================================
    # %% Thickness
    # ==========================================================================

    delta, material = _get_delta_gui(phenergy)

    # %%
    thickness = (phase - np.min(phase))/kwave/delta

    wgi.plot_integration(thickness*1e6,
                         virtual_pixelsize, titleStr=r'Material: ' + material +
                         ', Thickness $[\mu m]$',
                         saveFigFlag=True, saveFileSuf=saveFileSuf)

    # %% save thicknes txt

    saveFileSuf += '_thickness_' + material
    np.savetxt(saveFileSuf + '.dat', thickness,
               header='values in meter, pixel size i,j = ' +
                      '{:.6g} meters, '.format(virtual_pixelsize[0]) +
                      '{:.6g} meters'.format(virtual_pixelsize[1]))

    # %% Plot thickness and save as pickle

    #    fig = plt.figure(figsize=(12, 9.5))
    #
    #    plt.imshow(thickness*1e6,
    #               extent=wpu.extent_func(thickness, virtual_pixelsize)*1e6,
    #               cmap='viridis')
    #
    #    plt.xlabel(r'$x$ [$\mu m$]', fontsize=24)
    #    plt.ylabel(r'$y$ [$\mu m$]', fontsize=24)
    #
    #    cbar = plt.colorbar(shrink=0.9)
    #    cbar.ax.set_title(r'$[nm]$', y=1.01)
    #
    #    plt.title(,
    #              fontsize=24, weight='bold', y=1.01)
    #
    #    wpu.save_figs_with_idx_pickle(fig, )
    #
    #    plt.show(block=True)

    titleStr = r'Material: ' + material + ', Thickness $[\mu m]$'
    _default_plot_for_pickle(thickness*1e6, pixelsize,
                             patternforpickle=saveFileSuf,
                             title=titleStr, xlabel=r'$x$', ylabel=r'$y$',
                             ctitle=r'$[\mu m]$',
                             removeSpark=False, cmap='viridis')

    wpu.log_this('Material = ' + material)
    wpu.log_this('delta = ' + str('{:.5g}'.format(delta)))
    wpu.log_this('wavelength [m] = ' + str('{:.5g}'.format(wavelength)))

    thickSensitivy100 = virtual_pixelsize[0]**2/distDet2sample/delta/100
    # the 100 means that I arbitrarylly assumed the angular error in
    #  fringe displacement to be 2pi/100 = 3.6 deg
    wpu.log_this('Thickness Sensitivy 100 [m] = ' +
                 str('{:.5g}'.format(thickSensitivy100)))

    wpu.log_this('', inifname=inifname)

    # =============================================================================
    # %% sandbox to play
    # =============================================================================

    # %%
    # Plot WF in nm
    #    fig = plt.figure(figsize=(12, 9.5))
    #    plt.imshow(1/2/np.pi*phase*wavelength*1e9,
    #               extent=wpu.extent_func(phase, virtual_pixelsize)*1e6,
    #               cmap='viridis')
    #
    #    plt.xlabel(r'$x$ [$\mu m$]', fontsize=24)
    #    plt.ylabel(r'$y$ [$\mu m$]', fontsize=24)
    #
    #    cbar = plt.colorbar(shrink=0.9)
    #    cbar.ax.set_title(r'$[nm]$',
    #                      fontsize=24, weight='bold', y=1.01)
    #
    #    plt.title(r'Wavefront', y=1.01)
    #
    #    wpu.save_figs_with_idx_pickle(fig, saveFileSuf + '_WF')
    #
    #    plt.show(block=True)
    #
    #    # %% Plot thickness as contourf
    #    fig = plt.figure(figsize=(12, 9.5))
    #    plt.imshow(thickness*1e6,
    #               extent=wpu.extent_func(phase, virtual_pixelsize)*1e6,
    #               cmap='viridis')
    #
    #    plt.xlabel(r'$x$ [$\mu m$]', fontsize=24)
    #    plt.ylabel(r'$y$ [$\mu m$]', fontsize=24)
    #
    #    cbar = plt.colorbar(shrink=0.9)
    #    cbar.ax.set_title(r'$[\mu m]$', y=1.01)
    #
    #    plt.title(r'Thickness $[\mu m]$',
    #              fontsize=24, weight='bold', y=1.01)
    #
    #    wpu.save_figs_with_idx_pickle(fig, saveFileSuf + '_thickness')
    #
    #    plt.show(block=True)
