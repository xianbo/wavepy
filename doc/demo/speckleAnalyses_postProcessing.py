# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 16:00:19 2016

@author: wcgrizolli
"""

#==============================================================================
# %%
#==============================================================================
import numpy as np

from numpy.fft import fft2, ifft2, fftfreq
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

f = h5.File(fname,'r')

#print(wpu.h5ListOfGroups(f))

#==============================================================================
# %% parameters
#==============================================================================

delta = 5.3265E-06 # real part refractive index Be

pixelsize = f['raw'].attrs['Pixel Size Processed images [m]']
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

#sx_raw = np.array(f['displacement/displacement_y']).T
#sy_raw = np.array(f['displacement/displacement_x']).T
#error_raw = np.array(f['displacement/error']).T
#
#xVec_raw =  np.array(f['displacement/yvec'])
#yVec_raw =  np.array(f['displacement/xvec'])

#==============================================================================
# %% Crop
#==============================================================================

idx4crop = wpu.graphical_roi_idx(np.sqrt(sx_raw**2 + sy_raw**2), verbose=True)



sx = wpu.crop_matrix_at_indexes(sx_raw, idx4crop)
sy = wpu.crop_matrix_at_indexes(sy_raw, idx4crop)
error = wpu.crop_matrix_at_indexes(error_raw, idx4crop)


#==============================================================================
# %% Undersampling
#==============================================================================
step = 1 #int(np.ceil(sx.shape[0]/201))



sx = sx[::step, ::step]
sy = sy[::step, ::step]
error = error[::step, ::step]

xVec = wpu.realcoordvec(sx.shape[1], xVec_raw[1]-xVec_raw[0])
yVec = wpu.realcoordvec(sx.shape[0], yVec_raw[1]-yVec_raw[0])

xmatrix, ymatrix = np.meshgrid(xVec, yVec)




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


def fourier_integration(del_f_del_x, delx, del_f_del_y, dely, zeroPaddingWidth=0):


    if zeroPaddingWidth is not 0:
        del_f_del_x = np.pad(del_f_del_x,zeroPaddingWidth,
                             'constant', constant_values=0.0)

        del_f_del_y = np.pad(del_f_del_y,zeroPaddingWidth,
                             'constant', constant_values=0.0)

    fx, fy = np.meshgrid(np.fft.fftfreq(del_f_del_x.shape[1], delx),
                         np.fft.fftfreq(del_f_del_x.shape[0], dely))

    xx, yy = wpu.realcoordmatrix(del_f_del_x.shape[1], delx,
                                 del_f_del_x.shape[0], dely)

    fo_x = - np.abs(fx[0,1]/2) # shift fx value
    fo_y = - np.abs(fy[1,0]/2) # shift fy value


    phaseShift = np.exp(2*np.pi*1j*(fo_x*xx + fo_y*yy))  # exp factor for shift

    mult_factor = 1/(2*np.pi*1j)/(fx - fo_x + 1j*fy - 1j*fo_y)


    bigGprime = fft2((del_f_del_x + 1j*del_f_del_y)*phaseShift)
    bigG = bigGprime*mult_factor

    func_g = ifft2(bigG) / phaseShift


    if zeroPaddingWidth is not 0:
        func_g = func_g[zeroPaddingWidth:-zeroPaddingWidth,
                        zeroPaddingWidth:-zeroPaddingWidth]


    func_g -= func_g[0]  # since the integral have and undefined constant,
                         # here it is applied an arbritary offset

    return func_g



integration_res = fourier_integration(dTx, pixelsize, dTy, pixelsize,
                                      zeroPaddingWidth=200)

thickness = np.real(integration_res)

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

#==============================================================================
# %% Plot Sx and Sy
#==============================================================================


fig = plt.figure(next(figCount),figsize=(14, 5))
fig.suptitle('Displacements', fontsize=14)


ax1 = plt.subplot(121)
ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)

ax1.plot(xVec*1e6, sx[sx.shape[1]//4,:],'-o')
ax1.plot(xVec*1e6, sx[sx.shape[1]//2,:],'-o')
ax1.plot(xVec*1e6, sx[sx.shape[1]//4*3,:],'-o')
ax1.set_title('Sx [pixel]', fontsize=22)
ax1.set_adjustable('box-forced')


ax2.plot(yVec*1e6, sy[:,sy.shape[0]//4],'-o')
ax2.plot(yVec*1e6, sy[:,sy.shape[0]//2],'-o')
ax2.plot(yVec*1e6, sy[:,sy.shape[0]//4*3],'-o')
ax2.set_title('Sy [pixel]', fontsize=22)
ax2.set_adjustable('box-forced')



if saveFigFlag: mySaveFig()
plt.show(block=True)


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


##==============================================================================
## %% Quiver graph
##==============================================================================
#
#stride = int(np.ceil(sx.shape[0]/15)) # undersampling to reduce the number of arrows
#
#plt.figure(next(figCount),figsize=(10, 8))
#
#Q = plt.quiver(xmatrix[::stride, ::stride]*1e6,
#               ymatrix[::stride, ::stride]*1e6,
#               sx[::stride, ::stride], sy[::stride, ::stride],
#               totalS[::stride, ::stride], minlength=3,
#               cmap='gist_heat_r')
#
#plt.title(r'$\vec{S}$ [pixels] (undersampled)')
#
#
#
#plt.xlim(1.1*xmatrix[0,0]*1e6,1.1*xmatrix[-1,-1]*1e6)
#plt.ylim(1.1*ymatrix[0,0]*1e6,1.1*ymatrix[-1,-1]*1e6)
#plt.colorbar()
#if saveFigFlag: mySaveFig()
#plt.show(block=True)
#
##==============================================================================
## %% Quiver graph overlap contourf
##==============================================================================
#
#
#stride = int(np.ceil(sx.shape[0]/15)) # undersampling
#
#plt.figure(next(figCount),figsize=(10, 8))
#
#
#Q = plt.contourf(xmatrix*1e6, ymatrix*1e6, totalS, 101,
#               cmap='spectral')
#
#Q = plt.colorbar()
#
#Q = plt.quiver(xmatrix[::stride, ::stride]*1e6,
#               ymatrix[::stride, ::stride]*1e6,
#               sx[::stride, ::stride], sy[::stride, ::stride],
#               totalS[::stride, ::stride],
#               cmap='gist_heat_r', minlength=3)
#
#plt.title(r'$\vec{S}$ [pixels] (undersampled)')
#if saveFigFlag: mySaveFig()
#plt.show(block=True)

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

##==============================================================================
## %% Total displacement
##==============================================================================
#
#plt.figure(next(figCount))
#plt.hist(totalS.flatten(), 51)[0]
#plt.title(r'Total displacement $|\vec{S}|$ [pixels]', fontsize=16)
#if saveFigFlag: mySaveFig()
#plt.show(block=True)


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

wpu.plot_profile(xmatrix[::stride, ::stride]*1e6,
                ymatrix[::stride, ::stride]*1e6,
                thickness[::stride, ::stride]*1e6,
                title='Thickness', xlabel='[um]', ylabel='[um]',
                arg4main={'cmap':'spectral'}) #, xo=0.0, yo=0.0)
plt.show(block=True)


#==============================================================================
# %% Error
#==============================================================================

stride = 1

wpu.plot_profile(xmatrix[::stride, ::stride]*1e6,
                ymatrix[::stride, ::stride]*1e6,
                error[::stride, ::stride],
                title='Error', xlabel='[um]', ylabel='[um]',
                arg4main={'cmap':'spectral'}) #, xo=0.0, yo=0.0)
plt.show(block=True)


# %% Crop Result and plot surface



(xVec_croped1, yVec_croped1,
 thickness_croped, _) = wpu.crop_graphic(xVec, yVec,
                                                     thickness*1e6,
                                                     verbose=True)

thickness_croped *= 1e-6

xmatrix_croped1, ymatrix_croped1 = wpu.realcoordmatrix_fromvec(xVec_croped1,
                                                               yVec_croped1)



# %%

offset =  100e-6

#

fig = plt.figure(next(figCount),figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

stride = 5

thickness_croped = np.where(thickness_croped >offset,
                            thickness_croped - offset,0)

surf = ax.plot_surface(xmatrix_croped1*1e6,
                       ymatrix_croped1*1e6,
                       thickness_croped*1e6,
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

#exit()

# %%
from scipy.optimize import curve_fit




# %% 1D Fit


thickness_croped -= np.max(thickness_croped)

line = xmatrix_croped1.shape[1]//2
col = ymatrix_croped1.shape[1]//2

lim = 100

plt.figure()
plt.plot(xmatrix_croped1[line,lim:-lim]*1e6,
         thickness_croped[line,lim:-lim]*1e6,
         '-ko', markersize=5, label='1D data')


#

def _1Dparabol_4_fit(x, a, b, c):
    return (x-b)**2/a + c

popt, pcov = curve_fit(_1Dparabol_4_fit, xmatrix_croped1[line,lim:-lim],
                       thickness_croped[line,lim:-lim],
                       p0=[-100e-6, 1e-9, 520e-6])

print(popt)
fitted = _1Dparabol_4_fit(xmatrix_croped1[line,lim:-lim], popt[0], popt[1], popt[2])


plt.plot(xmatrix_croped1[line,lim:-lim]*1e6, fitted*1e6,
         '-+r', label='Fir parabolic')

#
lim = 100
def _1DSphere_4_fit(x, amp, xo, R):

    return -amp * ((x-xo)/np.sin(np.arctan2((x-xo),R)) - R)
#    return np.sqrt(a**2 - (x - b)**2) + c

popt2, pcov2 = curve_fit(_1DSphere_4_fit, xmatrix_croped1[line,lim:-lim],
                                    thickness_croped[line,lim:-lim],
                                     p0=[1.0909, -1e-5, 50.9e-6])


print(popt2)
fitted2 = _1DSphere_4_fit(xmatrix_croped1[line,lim:-lim],
                       popt2[0], popt2[1], popt2[2])


plt.plot(xmatrix_croped1[line,lim:-lim]*1e6, fitted2*1e6,
         '-xg', label='Fit Sphere')

plt.legend()
plt.show()

# %%
plt.figure()
plt.plot(xmatrix_croped1[line,lim:-lim]*1e6,
         thickness_croped[line,lim:-lim]*1e6 - fitted*1e6,
         '-+', markersize=5, label='f')

plt.show()

# %% 2D Fit

r2 = np.sqrt(xmatrix_croped1**2 + ymatrix_croped1**2)

args4fit = np.where( r2.flatten() < 100e-6)

data2fit = thickness_croped.flatten()[args4fit]

xxfit = xmatrix_croped1.flatten()[args4fit]
yyfit = ymatrix_croped1.flatten()[args4fit]

xyfit = [xxfit, yyfit]

# %%



def _2Dparabol_4_fit(xy, ax, ay, xo, yo, offset):

    x, y = xy
    return (x - xo)**2/ax + (y - yo)**2/ay + offset
thickness_croped


popt, pcov = curve_fit(_2Dparabol_4_fit, xyfit, data2fit,
                       p0=[-100e-6, -100e-6,  1e-9, 1e-9, 5.43e-6],
                       bounds=([-1., -1., -1e-5, -1e-5, -1e-4],
                               [0., 0., 1e-5, 1e-5, 1e-4]))

print(popt*1e6)
fitted = _2Dparabol_4_fit( [xmatrix_croped1, ymatrix_croped1],
                          popt[0], popt[1], popt[2], popt[3], popt[4])

#fitted = np.reshape(fitted, xmatrix_croped1.shape)

# %%
stride = 1
lim = 1

errorThickness = thickness_croped - fitted

wpu.plot_profile(xmatrix_croped1[lim:-lim,lim:-lim]*1e6,
                 ymatrix_croped1[lim:-lim,lim:-lim]*1e6,
                 fitted[lim:-lim,lim:-lim]*1e6,
                 title='FIT', xlabel='[um]', ylabel='[um]',
                 arg4main={'cmap':'spectral'}) #, xo=0.0, yo=0.0)




if saveFigFlag: mySaveFig()
plt.show(block=True)

# %%
stride = 1
lim = 40


wpu.plot_profile(xmatrix_croped1[lim:-lim,lim:-lim]*1e6,
                 ymatrix_croped1[lim:-lim,lim:-lim]*1e6,
                 errorThickness[lim:-lim,lim:-lim]*1e6,
                 title='Error Be1x50um_in_d345_8keV_10s_gr2500_001_20160922_175544Thickness', xlabel='[um]', ylabel='[um]',
                 arg4main={'cmap':'spectral'}) #, xo=0.0, yo=0.0)


# %%


lim = 100

fig = plt.figure(figsize=plt.figaspect(3.0/4.0), facecolor="white")
ax = fig.gca(projection='3d')
plt.tight_layout(pad=2.5)

ax.scatter(xxfit*1e6, yyfit*1e6, data2fit*1e6, c='b', marker='+', alpha=0.8)

if saveFigFlag: mySaveFig()
plt.show(block=True)

# %%


lim = 30
stride = 5

fig = plt.figure(figsize=plt.figaspect(3.0/4.0), facecolor="white")
ax = fig.gca(projection='3d')
plt.tight_layout(pad=2.5)

ax.scatter(xmatrix_croped1[lim:-lim:stride,lim:-lim:stride]*1e6,
           ymatrix_croped1[lim:-lim:stride,lim:-lim:stride]*1e6,
           errorThickness[lim:-lim:stride,lim:-lim:stride]*1e6, c='m', marker='+', alpha=0.8)


if saveFigFlag: mySaveFig()
plt.show(block=True)

# %%
plt.figure()


lim = 100
line = 105

plt.plot(xmatrix_croped1[line,lim:-lim]*1e6,
         thickness_croped[line,lim:-lim]*1e6,
         '--+', markersize=5, label='f')
plt.plot(xmatrix_croped1[line,lim:-lim]*1e6,
         fitted[line,lim:-lim]*1e6, '--')


if saveFigFlag: mySaveFig()
plt.show(block=True)
