# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 16:00:19 2016

@author: wcgrizolli
"""

#==============================================================================
# %%
#==============================================================================
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


import h5py as h5


import wavepy.utils as wpu

import itertools



# %%
#==============================================================================
# preamble
#==============================================================================


# Flags
saveFigFlag = True

# useful constants
rad2deg = np.rad2deg(1)
deg2rad = np.deg2rad(1)
NAN = float('Nan')  # not a number alias

from scipy import constants
hc = constants.value('inverse meter-electron volt relationship') # hc

figCount = itertools.count()  # itera
next(figCount)


def mpl_settings_4_nice_graphs():
    # Settings for latex fonts in the graphs
    # ATTENTION: This will make the program slow because it will compile all
    # latex text. This means that you also need latex and any latex package
    # you want to use. I suggest to only use this when you want produce final
    # graphs to publish or to make public. The latex dependecies are not taken
    # care  by the installation scripts. You are by your own to solve these
    # dependencies.

    plt.style.use('default')
    #Direct input
    plt.rcParams['text.latex.preamble']=[r"\usepackage[utopia]{mathdesign}"]
    ##Options
    params = {'text.usetex' : True,
              'font.size' : 16,
              'font.family' : 'utopia',
              'text.latex.unicode': True,
              'figure.facecolor' : 'white'
              }
    plt.rcParams.update(params)

# mpl_settings_4_nice_graphs()




#==============================================================================
# %% Load files
#==============================================================================

fname = wpu.select_file('**/*.h5')
# %%

f = h5.File(fname,'r')

#print(wpu.h5ListOfGroups(f))

#==============================================================================
# %% parameters
#==============================================================================

delta = 5.3265E-06 # real part refractive index Be

pixelsize = f['raw'].attrs['Pixel Size [m]']
distDet2sample = f['raw'].attrs['Distance Detector to Sample [m]']
phenergy = f['raw'].attrs['Photon Energy [eV]']

wavelength = hc/phenergy
kwave = 2*np.pi/wavelength

print('MESSAGE: Comments from hdf5 files')
print('MESSAGE: '+ f['raw'].attrs['Comments'])


# %%

sx_raw = np.array(f['displacement/displacement_x'])
sy_raw = np.array(f['displacement/displacement_y'])
error_raw = np.array(f['displacement/error'])

xVec_raw =  np.array(f['displacement/xvec'])
yVec_raw =  np.array(f['displacement/yvec'])

#==============================================================================
# %% Crop
#==============================================================================

(xVec, yVec, _, idx4crop) = wpu.crop_graphic(xVec_raw, yVec_raw,
                                              np.sqrt(sx_raw**2 + sy_raw**2),
                                              verbose=True)



sx = wpu.crop_matrix_at_indexes(sx_raw, idx4crop)
sy = wpu.crop_matrix_at_indexes(sy_raw, idx4crop)
error = wpu.crop_matrix_at_indexes(error_raw, idx4crop)


#==============================================================================
# %% Undersampling
#==============================================================================
step = int(np.ceil(sx.shape[0]/201))



sx = sx[::step, ::step]
sy = sy[::step, ::step]
error = error[::step, ::step]

xVec = xVec[::step]
yVec = yVec[::step]

xmatrix, ymatrix = np.meshgrid(xVec, yVec)


#==============================================================================
# %% Mask
#==============================================================================
maskGood = np.ones(error.shape, dtype='Bool')*NAN
maskGood[error < 1.200] = 1.0



#==============================================================================
# %% Calculations of physical quantities
#==============================================================================


totalS = np.sqrt(sx**2 + sy**2)

# Differenctial Thickness
dTx = 1.0/delta*np.arctan2(sx*pixelsize, distDet2sample)
dTy = 1.0/delta*np.arctan2(sy*pixelsize, distDet2sample)


# %%
#==============================================================================
# integration
#==============================================================================

def fourier_integration(del_f_del_x, del_f_del_y,
                                xvec, yvec):

    delx = xvec[1] - xvec[0]
    dely = yvec[1] - yvec[0]


    fx, fy = wpu.fouriercoordmatrix(xvec.size, delx, yvec.size, dely)

    fx = fx - np.min(fx)*2
    fy = fy - np.min(fy)*2


    denominator = 1j*fx - fy
    denominator[np.abs(denominator) < 1e-10] = NAN

    return np.fft.ifft2(np.fft.fftshift(np.fft.fft2(del_f_del_x +
                                                    1j*del_f_del_y))
                                        /denominator)


integration_res = fourier_integration(dTx, dTy, xVec, yVec)

thickness = np.abs(integration_res)

thickness = thickness - np.min(thickness)


# %%
#==============================================================================
# Plot
#==============================================================================

def mySaveFig(figname = None):

    if figname is None:
        figname = str('output/graph_{0:02d}.png'.format(plt.gcf().number))

    plt.savefig(figname)
    print(figname + ' SAVED')

def mySimplePlot(array, title=''):

    plt.figure(next(figCount))
    plt.imshow(array, cmap='spectral', interpolation='none')
    plt.title(title)
    plt.colorbar()
    if saveFigFlag: mySaveFig()
    plt.show(block=True)



def plotsidebyside(array1, array2, title1='', title2='', maintitle=''):

    fig = plt.figure(next(figCount),figsize=(14, 5))
    fig.suptitle(maintitle, fontsize=14)

    vmax = np.max([array1, array2])
    vmin = np.min([array1, array2])

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)

    im1 = ax1.imshow(array1, cmap='RdGy',
                     interpolation='none',
                     vmin=vmin, vmax=vmax)
    ax1.set_title(title1, fontsize=22)
    ax1.set_adjustable('box-forced')
    fig.colorbar(im1, ax=ax1, shrink=.8, aspect=20)

    im2 = ax2.imshow(array2, cmap='RdGy',
                     interpolation='none',
                     vmin=vmin, vmax=vmax)
    ax2.set_title(title2, fontsize=22)
    ax2.set_adjustable('box-forced')
    fig.colorbar(im2, ax=ax2, shrink=.8, aspect=20)


    if saveFigFlag: mySaveFig()
    plt.show(block=True)

    #    plt.tight_layout()




# %%

plotsidebyside(sx, sy, r'Displacement $S_x$ [pixels]',
                         r'Displacement $S_y$ [pixels]')

# %%
mySimplePlot(totalS, title=r'Displacement Module $|\vec{S}|$ [pixels]')

# %%


fig = plt.figure(next(figCount),figsize=(14, 5))

ax1 = plt.subplot(121)
ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)

ax1.plot(sx.flatten(),error.flatten(),'.')
ax1.set_xlabel('Sy [pixel]')
ax1.set_title('Error vs Sx', fontsize=22)
ax1.set_adjustable('box-forced')


ax2.plot(sy.flatten(),error.flatten(),'.')
ax2.set_xlabel('Sy [pixel]')
ax2.set_title('Error vs Sy', fontsize=22)
ax2.set_adjustable('box-forced')


if saveFigFlag: mySaveFig()
plt.show(block=True)


#==============================================================================
# %% Quiver graph
#==============================================================================

stride = int(np.ceil(sx.shape[0]/15)) # undersampling to reduce the number of arrows

plt.figure(next(figCount),figsize=(10, 8))

Q = plt.quiver(xmatrix[::stride, ::stride]*1e6,
               ymatrix[::stride, ::stride]*1e6,
               sx[::stride, ::stride], sy[::stride, ::stride],
               totalS[::stride, ::stride], minlength=3,
               cmap='gist_heat_r')

plt.title(r'$\vec{S}$ [pixels] (undersampled)')



plt.xlim(1.1*xmatrix[0,0]*1e6,1.1*xmatrix[-1,-1]*1e6)
plt.ylim(1.1*ymatrix[0,0]*1e6,1.1*ymatrix[-1,-1]*1e6)
plt.colorbar()
if saveFigFlag: mySaveFig()
plt.show(block=True)

#==============================================================================
# %% Quiver graph overlap contourf
#==============================================================================


stride = int(np.ceil(sx.shape[0]/15)) # undersampling

plt.figure(next(figCount),figsize=(10, 8))


Q = plt.contourf(xmatrix*1e6, ymatrix*1e6, totalS, 101,
               cmap='spectral')

Q = plt.colorbar()

Q = plt.quiver(xmatrix[::stride, ::stride]*1e6,
               ymatrix[::stride, ::stride]*1e6,
               sx[::stride, ::stride], sy[::stride, ::stride],
               totalS[::stride, ::stride],
               cmap='gist_heat_r', minlength=3)

plt.title(r'$\vec{S}$ [pixels] (undersampled)')
if saveFigFlag: mySaveFig()
plt.show(block=True)

#==============================================================================
# %% Histograms to evaluate data quality
#==============================================================================


fig = plt.figure(next(figCount),figsize=(14, 5))
fig.suptitle('Histograms to evaluate data quality', fontsize=16)

ax1 = plt.subplot(121)
ax1 = plt.hist(sx.flatten(), 51)
ax1 = plt.title(r'$S_x$ [pixels]', fontsize=16)

ax1 = plt.subplot(122)
ax2 = plt.hist(sy.flatten(), 51)
ax2 = plt.title(r'$S_y$ [pixels]', fontsize=16)


if saveFigFlag: mySaveFig()
plt.show(block=True)

#==============================================================================
# %% Total displacement
#==============================================================================

plt.figure(next(figCount))
plt.hist(totalS.flatten(), 51)[0]
plt.title(r'Total displacement $|\vec{S}|$ [pixels]', fontsize=16)
if saveFigFlag: mySaveFig()
plt.show(block=True)


#==============================================================================
# %% Integration Real and Imgainary part
#==============================================================================


fig = plt.figure(next(figCount),figsize=(14, 5))
fig.suptitle('Histograms to evaluate data quality', fontsize=16)

ax1 = plt.subplot(121)
ax1 = plt.hist(np.real(integration_res).flatten()*1e6, 51)
ax1 = plt.title(r'Integration Real part', fontsize=16)

ax1 = plt.subplot(122)
ax2 = plt.hist(np.imag(integration_res).flatten()*1e6, 51)
ax2 = plt.title(r'Integration Imag part', fontsize=16)

if saveFigFlag: mySaveFig()
plt.show(block=True)


#==============================================================================
# %% Thickness
#==============================================================================

stride = 1

plt.figure(next(figCount))

plt.contourf(xmatrix[::stride, ::stride]*1e6,
             ymatrix[::stride, ::stride]*1e6,
             (maskGood*thickness)[::stride, ::stride]*1e6, 101, cmap='spectral')



plt.xlabel('[um]')
plt.ylabel('[um]')

plt.title('Thickness')


plt.colorbar()
if saveFigFlag: mySaveFig()
plt.show(block=True)
#plt.close()


# %%

stride = 1

wpu.plot_profile(xmatrix[::stride, ::stride]*1e6,
                ymatrix[::stride, ::stride]*1e6,
                (maskGood*thickness)[::stride, ::stride]*1e6,
                title='Thickness', xlabel='[um]', ylabel='[um]') #, xo=0.0, yo=0.0)
plt.show(block=True)


# %% Crop Result



(xVec_croped1, yVec_croped1,
 thickness_croped1, idx) = wpu.crop_graphic(xVec, yVec,
                                                     thickness*1e6,
                                                     verbose=True)

thickness_croped1 *= 1e-6

xmatrix_croped1, ymatrix_croped1 = wpu.realcoordmatrix_fromvec(xVec_croped1,
                                                               yVec_croped1)


maskGood_croped1 = wpu.crop_matrix_at_indexes(maskGood, idx)


#

fig = plt.figure(next(figCount),figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

stride = 1


surf = ax.plot_surface(xmatrix_croped1*1e6,
                       ymatrix_croped1*1e6,
                       -maskGood_croped1*thickness_croped1*1e6,
                        rstride=stride, cstride=stride,
                        #vmin=-120, vmax=0,
                       cmap='spectral', linewidth=0.1)

plt.xlabel('[um]')
plt.ylabel('[um]')

plt.title('Thickness [um]', fontsize=18, weight='bold')
plt.colorbar(surf, shrink=.8, aspect=20)

plt.tight_layout()
if saveFigFlag: mySaveFig()
plt.show(block=True)
