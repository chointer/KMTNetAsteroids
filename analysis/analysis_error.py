import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import sys
sys.path.append("/home/astro/PycharmProjects/KMTNet_git/KMTNet/MyModule")
from ModuleCommon import err_Add, err_Mul, linfit_sigclip, sigma_boundary
from ModulePCA import *


def func_gauss(x, a, mu, sig):
    return a * np.exp(-(((x - mu) / sig)**2)/2) / (sig * (2 * np.pi)**0.5)


def errorhist(err, errlim=0.12, ftsize=15, titlename='', ylim=0):
    mask_large_err = (err >= errlim)
    print(np.max(err[np.logical_not(mask_large_err)]))
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)

    # histogram
    n1, bins1, _ = ax.hist(err[np.logical_not(mask_large_err)], bins=24, range=(0.0, errlim),
                           color='goldenrod', alpha=0.95)
    if sum(mask_large_err) > 0:
        for i in range(sum(mask_large_err)):
            ax.scatter(errlim, 50 + 25 * i, marker='>', color='goldenrod')
            print('[%s] large error : %s' % (titlename, err[mask_large_err][i]))
    ax.set_xlim(-0.005, errlim + 0.005)
    ax.set_title(titlename, fontsize=ftsize)
    ax.set_xlabel('Error', fontsize=ftsize)
    ax.set_ylabel('#', fontsize=ftsize)
    ax.tick_params(axis='x', labelsize=ftsize)
    ax.tick_params(axis='y', labelsize=ftsize)
    if ylim != 0:
        ax.set_ylim(0, ylim)

    # accumulated plot
    ax2 = ax.twinx()
    n2, bins2, _ = ax2.hist(err, bins=26, range=(0.0, errlim * 26/24), color='orangered',
                            cumulative=True, histtype='step', linewidth=1.2, density=True, alpha=0.95)
    cens = (bins2[1:] + bins2[:-1]) / 2

    tag_y = [0.25, 0.50, 0.75]
    tag_x = np.interp(tag_y, n2, cens)
    ax2.vlines(tag_x, ymin=0, ymax=0.05, linewidth=1.2, colors='orangered', linestyles='--')
    for k in range(len(tag_x)):
        ax2.text(tag_x[k] - errlim * 0.045, 0.055, '%.4f' % tag_x[k])
    ax2.tick_params(axis='y', labelsize=ftsize)

    # gaussfit
    popt, pcov = curve_fit(func_gauss, (bins1[1:] + bins1[:-1]) / 2, n1)
    ax.plot(np.linspace(0, errlim, 100), func_gauss(np.linspace(0, errlim, 100), popt[0], popt[1], popt[2]),
            linewidth=1.2, c='mediumblue')
    tag_gauss = [popt[1] - popt[2], popt[1], popt[1] + popt[2]]
    ax.vlines(tag_gauss, ymin=ylim * 0.95, ymax=ylim * 1.05, linewidth=1.2, colors='mediumblue', linestyles='--')
    for k in range(len(tag_gauss)):
        ax.text(tag_gauss[k] - errlim * 0.045, ylim * 0.915, '%.4f' % tag_gauss[k])






### data ###
dat_griz = np.genfromtxt('/home/astro/PycharmProjects/KMTNet_git/KMTNet/photometry/' + 'AstList_griz.cat', dtype=float)
PCA_griz = PCA(dat_griz[:, 3:11], ['g', 'r', 'i', 'z'], constant=False)
fg, fge, fr, fre, fi, fie, fz, fze = dat_griz[:, 3:11].T
maxerr_griz = np.max(np.array([fge, fre, fie, fze]), axis=0)

dat_BVRI = np.genfromtxt('/home/astro/PycharmProjects/KMTNet_git/KMTNet/photometry/' + 'AstList_BVRI.cat', dtype=float)
PCA_BVRI = PCA(dat_BVRI[:, 3:11], ['B', 'V', 'R', 'I'], constant=False)
fB, fBe, fV, fVe, fR, fRe, fI, fIe = dat_BVRI[:, 3:11].T
maxerr_BVRI = np.max(np.array([fBe, fVe, fRe, fIe]), axis=0)

### histogram plot ###
# max error #
#errorhist(maxerr_griz, titlename='griz', ylim=500)
#errorhist(maxerr_BVRI, titlename='BVRI', ylim=500)
"""
# f3 - f1 #
errorhist(err_Add([fie, fge]), titlename='g - i', ylim=900, errlim=0.14)
plt.savefig('hist_err_gi.png')
errorhist(err_Add([fRe, fBe]), titlename='R - B', ylim=900, errlim=0.14)
plt.savefig('hist_err_BR.png')

# f3 - f4 #
errorhist(err_Add([fze, fie]), titlename='i - z', ylim=900, errlim=0.14)
plt.savefig('hist_err_zi.png')
errorhist(err_Add([fIe, fRe]), titlename='R - I', ylim=900, errlim=0.14)
plt.savefig('hist_err_IR.png')

# PCA #
errorhist(PCA_griz.a_star_err, titlename='PCA_griz', ylim=900, errlim=0.14)
plt.savefig('hist_err_PCIgriz.png')
errorhist(PCA_BVRI.a_star_err, titlename='PCA_BVRI', ylim=900, errlim=0.14)
plt.savefig('hist_err_PCIBVRI.png')

# integrate #
errorhist(err_Add([3 * fge, fre, fie, fze]), titlename='griz_integrated', ylim=900, errlim=0.4)
plt.savefig('hist_err_Integgriz.png')
errorhist(err_Add([3 * fBe, fVe, fRe, fIe]), titlename='BVRI_integrated', ylim=900, errlim=0.4)
plt.savefig('hist_err_IntegBVRI.png')

#plt.show()
#plt.close('all')
"""
### Each Error Plot ###
errlim = 0.12
ftsize = 14
cols = ['tomato', 'goldenrod', 'green', 'navy']

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(1, 1, 1)
labels = ['g', 'r', 'i', 'z']
for j in range(4):
    mask_large_err = dat_griz[:, 4 + 2 * j] >= errlim
    ax.hist(dat_griz[:, 4 + 2 * j][np.logical_not(mask_large_err)], bins=24, range=(0.0, errlim),
            color=cols[j], alpha=0.9, histtype='step', linewidth=3, label=labels[j])
    if sum(mask_large_err) > 0:
        for i in range(sum(mask_large_err)):
            ax.scatter(errlim, 50 + 25 * i, marker='>', color=cols[j])
            print('large error : %s' % dat_griz[:, 4 + 2 * j][mask_large_err][i])
    ax.set_xlim(-0.005, errlim + 0.005)
    ax.set_title('griz', fontsize=ftsize)
    ax.set_xlabel('Max Error', fontsize=ftsize)
    ax.set_ylabel('#', fontsize=ftsize)
    ax.tick_params(axis='x', labelsize=ftsize)
    ax.tick_params(axis='y', labelsize=ftsize)
    ax.legend()
plt.savefig('hist_err_griz.png')

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(1, 1, 1)
labels = ['B', 'V', 'R', 'I']
for j in range(4):
    mask_large_err = dat_BVRI[:, 4 + 2 * j] >= errlim
    ax.hist(dat_BVRI[:, 4 + 2 * j][np.logical_not(mask_large_err)], bins=24, range=(0.0, errlim),
            color=cols[j], alpha=0.9, histtype='step', linewidth=3, label=labels[j])
    if sum(mask_large_err) > 0:
        for i in range(sum(mask_large_err)):
            ax.scatter(errlim, 50 + 25 * i, marker='>', color=cols[j])
            print('large error : %s' % dat_BVRI[:, 4 + 2 * j][mask_large_err][i])
    ax.set_xlim(-0.005, errlim + 0.005)
    ax.set_title('BVRI', fontsize=ftsize)
    ax.set_xlabel('Error', fontsize=ftsize)
    ax.set_ylabel('#', fontsize=ftsize)
    ax.tick_params(axis='x', labelsize=ftsize)
    ax.tick_params(axis='y', labelsize=ftsize)
    ax.legend()
plt.savefig('hist_err_BVRI.png')
assert False
#"""


"""
### PCA plot ###
# 0.01 / 0.02, 0.04, 0.06, 0.08, 0.10
plot_err_range = 
xrange = [-0.6, 0.55]
yrange = [-0.7, 0.5]

fig = plt.figure(figsize=(15, 12))
fig.suptitle('SDSS', fontsize=17)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.90, wspace=0., hspace=0.)
for i in range(6):
    ax = fig.add_subplot(2, 3, i + 1)
    ax.set_xlim(xrange[0] - 0.05, xrange[1] + 0.05)
    ax.set_ylim(yrange[0] - 0.05, yrange[1] + 0.05)
    if i % 3 != 0:
        ax.axes.yaxis.set_visible(False)
    else:
        ax.axes.yaxis.set_ticks(np.arange(yrange[0], yrange[1]+0.02, 0.3))
    if i < 3:
        ax.axes.xaxis.set_visible(False)
    else:
        ax.axes.xaxis.set_ticks(np.arange(xrange[0], xrange[1]+0.02, 0.3))

    # draw
    if i == 0:
        idx = np.where(maxerr_griz < 0.01)[0]
        ax.text(xrange[0] - 0.01, yrange[1] - 0.06, '$err_{max}$ < 0.01', fontsize=15)
    else:
        idx = np.where((maxerr_griz >= plot_err_range[i-1]) & (maxerr_griz < plot_err_range[i]))[0]
        ax.text(xrange[0] - 0.01, yrange[1] - 0.06, '%.2f <= $err_{max}$ < %.2f' % (plot_err_range[i-1], plot_err_range[i]),
                fontsize=15)
    drawPCA_simple(ax, dat_griz[:, 3], dat_griz[:, 4], dat_griz[:, 5], dat_griz[:, 6],
                   dat_griz[:, 7], dat_griz[:, 8], dat_griz[:, 9], dat_griz[:, 10],
                   PCA_griz.a_star, PCA_griz.a_star_err, col='gray', alp=0.5, err=False)
    drawPCA_simple(ax, dat_griz[idx, 3], dat_griz[idx, 4], dat_griz[idx, 5], dat_griz[idx, 6],
                   dat_griz[idx, 7], dat_griz[idx, 8], dat_griz[idx, 9], dat_griz[idx, 10],
                   PCA_griz.a_star[idx], PCA_griz.a_star_err[idx], col='navy', alp=0.5, err=False)


### BVRI ###
xrange = [-0.6, 0.5]
yrange = [-0.1, 0.7]
fig = plt.figure(figsize=(15, 12))
fig.suptitle('Johnson-Cousins', fontsize=17)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.90, wspace=0., hspace=0.)
for i in range(6):
    ax = fig.add_subplot(2, 3, i + 1)
    ax.set_xlim(xrange[0] - 0.05, xrange[1] + 0.05)
    ax.set_ylim(yrange[0] - 0.05, yrange[1] + 0.05)
    #ax.set_xlim()
    if i % 3 != 0:
        ax.axes.yaxis.set_visible(False)
    else:
        ax.axes.yaxis.set_ticks(np.arange(yrange[0], yrange[1]+0.02, 0.2))
    if i < 3:
        ax.axes.xaxis.set_visible(False)
    else:
        ax.axes.xaxis.set_ticks(np.arange(xrange[0], xrange[1]+0.02, 0.3))

    # draw
    if i == 0:
        idx = np.where(maxerr_BVRI < 0.01)[0]
        ax.text(xrange[0] - 0.01, yrange[1] - 0.03, '$err_{max}$ < 0.01', fontsize=15)
    else:
        idx = np.where((maxerr_BVRI >= plot_err_range[i-1]) & (maxerr_BVRI < plot_err_range[i]))[0]
        ax.text(xrange[0] - 0.01, yrange[1] - 0.03, '%.2f <= $err_{max}$ < %.2f' % (plot_err_range[i-1], plot_err_range[i]),
                fontsize=15)
    drawPCA_simple(ax, dat_BVRI[:, 3], dat_BVRI[:, 4], dat_BVRI[:, 5], dat_BVRI[:, 6],
                   dat_BVRI[:, 7], dat_BVRI[:, 8], dat_BVRI[:, 9], dat_BVRI[:, 10],
                   PCA_BVRI.a_star, PCA_BVRI.a_star_err, col='gray', alp=0.5, err=False)
    drawPCA_simple(ax, dat_BVRI[idx, 3], dat_BVRI[idx, 4], dat_BVRI[idx, 5], dat_BVRI[idx, 6],
                   dat_BVRI[idx, 7], dat_BVRI[idx, 8], dat_BVRI[idx, 9], dat_BVRI[idx, 10],
                   PCA_BVRI.a_star[idx], PCA_BVRI.a_star_err[idx], col='navy', alp=0.5, err=False)
"""

### accumulative PCA plot ###
plot_err_lim = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
xrange = [0.2, 0.85]
yrange = [-0.45, 0.30]

fig = plt.figure(figsize=(18, 15))
fig.suptitle('griz [accumulation]', fontsize=30)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.90, wspace=0., hspace=0.)
for i in range(len(plot_err_lim)):
    ax = fig.add_subplot(3, 3, i + 1)
    ax.set_xlim(xrange[0] - 0.05, xrange[1] + 0.05)
    ax.set_ylim(yrange[0] - 0.05, yrange[1] + 0.05)
    if i % 3 != 0:
        ax.axes.yaxis.set_visible(False)
    else:
        ax.axes.yaxis.set_ticks([-0.4, -0.2, 0, 0.2])#np.arange(yrange[0], yrange[1]+0.02, 0.3))
    if i < 6:
        ax.axes.xaxis.set_visible(False)
    else:
        ax.axes.xaxis.set_ticks([0.2, 0.4, 0.6, 0.8])#np.arange(xrange[0], xrange[1]+0.02, 0.3))
    if i == 7:
        ax.set_xlabel('PCA', fontsize=25)
    if i == 3:
        ax.set_ylabel('i - z', fontsize=25)
    # draw

    idx = np.where(maxerr_griz < plot_err_lim[i])[0]
    ax.text(xrange[0] - 0.01, yrange[1] - 0.06, '$err_{max}$ < %.2f' % plot_err_lim[i], fontsize=22)
    drawPCA_simple(ax, dat_griz[:, 3], dat_griz[:, 4], dat_griz[:, 5], dat_griz[:, 6],
                   dat_griz[:, 7], dat_griz[:, 8], dat_griz[:, 9], dat_griz[:, 10],
                   PCA_griz.a_star, PCA_griz.a_star_err, col='gray', alp=0.2, err=False, fsize=22)
    drawPCA_simple(ax, dat_griz[idx, 3], dat_griz[idx, 4], dat_griz[idx, 5], dat_griz[idx, 6],
                   dat_griz[idx, 7], dat_griz[idx, 8], dat_griz[idx, 9], dat_griz[idx, 10],
                   PCA_griz.a_star[idx], PCA_griz.a_star_err[idx], col='navy', alp=0.5, err=False, fsize=22)
fig.savefig('PCA_accum_griz.png', dpi=200)

### BVRI ###
xrange = [0.5, 1.23]
yrange = [0.13, 0.63]
fig = plt.figure(figsize=(18, 15))
fig.suptitle('BVRI [accumulation]', fontsize=30)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.90, wspace=0., hspace=0.)
for i in range(len(plot_err_lim)):
    ax = fig.add_subplot(3, 3, i + 1)
    ax.set_xlim(xrange[0] - 0.05, xrange[1] + 0.05)
    ax.set_ylim(yrange[0] - 0.05, yrange[1] + 0.05)
    if i % 3 != 0:
        ax.axes.yaxis.set_visible(False)
    else:
        ax.axes.yaxis.set_ticks([0.1, 0.3, 0.5])#np.arange(yrange[0], yrange[1]+0.02, 0.2))
    if i < 6:
        ax.axes.xaxis.set_visible(False)
    else:
        ax.axes.xaxis.set_ticks([0.5, 0.7, 0.9, 1.1])#np.arange(xrange[0], xrange[1]+0.02, 0.3))
    if i == 7:
        ax.set_xlabel('PCA', fontsize=25)
    if i == 3:
        ax.set_ylabel('R - I', fontsize=25)

    # draw

    idx = np.where(maxerr_BVRI < plot_err_lim[i])[0]
    ax.text(xrange[0] - 0.01, yrange[1] - 0.03, '$err_{max}$ < %.2f' % plot_err_lim[i], fontsize=22)
    drawPCA_simple(ax, dat_BVRI[:, 3], dat_BVRI[:, 4], dat_BVRI[:, 5], dat_BVRI[:, 6],
                   dat_BVRI[:, 7], dat_BVRI[:, 8], dat_BVRI[:, 9], dat_BVRI[:, 10],
                   PCA_BVRI.a_star, PCA_BVRI.a_star_err, col='gray', alp=0.2, err=False, fsize=22)
    drawPCA_simple(ax, dat_BVRI[idx, 3], dat_BVRI[idx, 4], dat_BVRI[idx, 5], dat_BVRI[idx, 6],
                   dat_BVRI[idx, 7], dat_BVRI[idx, 8], dat_BVRI[idx, 9], dat_BVRI[idx, 10],
                   PCA_BVRI.a_star[idx], PCA_BVRI.a_star_err[idx], col='navy', alp=0.5, err=False, fsize=22)
fig.savefig('PCA_accum_BVRI.png', dpi=200)

#plt.show()
#"""