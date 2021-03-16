# 20210228? err_Mul
#           -> modify for the case of '0'
# 20210316  func_linear, func_loss, sigma_boundary, linfit_sigclip, linfit_chisigclip
#           -> add sigma clipping linear fit functions
import numpy as np
from scipy.optimize import leastsq

# deg transformation
def hms2deg_header(hms):
    # only for a single
    hms = np.char.split(hms, sep=':').tolist()
    deg = 15 * (float(hms[0]) + float(hms[1]) / 60 + float(hms[2]) / 3600)
    return deg

def dms2deg_header(dms):
    # only for a single
    dms0 = np.char.split(dms, sep=':').tolist()
    if dms0[0][0] == '-' : sd = [-1, dms0[0][1:]]              # sd means 'sign & deg'
    else : sd = [1, dms0[0][1:]]
    deg = sd[0] * (float(sd[1]) + float(dms0[1])/60 + float(dms0[2])/3600)
    return deg

def hms2deg_SkyBoT(hms):
    # only for multiple
    hms = np.array(np.char.split(hms).tolist())
    deg = 15 * (hms[:, 0].astype(float) + hms[:, 1].astype(float)/60 + hms[:, 2].astype(float)/3600)
    return deg

def dms2deg_SkyBoT(dms):
    # only for multiple
    dms0 = np.char.split(dms).tolist()
    sd = np.ones((len(dms0), 2))                # sd means 'sign & deg'
    for temp in range(len(dms0)):
        if dms0[temp][0][0] == '-' : sd[temp][0] = -1
        sd[temp][1] = int(dms0[temp][0][1:])
    dms0 = np.array(dms0)
    deg = sd[:, 0] * (sd[:, 1] + dms0[:, 1].astype(float)/60 + dms0[:, 2].astype(float)/3600)
    return deg


# error propagation
def err_Add(errors):
    errors = np.array(errors)
    return (np.sum(errors**2, axis=0))**0.5

def err_Mul(elements, errors):
    #x = np.prod(elements, axis=0)
    elements = np.array(elements)
    errors = np.array(errors)
    if elements.shape[0] != 2:
        assert False
    #xerr = x * (np.sum((errors/elements)**2, axis=0))**0.5
    xerr = ((elements[0, :] * errors[1, :])**2 + (elements[1, :] * errors[0, :])**2)**0.5
    return xerr


# linear fit
def func_linear(x, a, b): return a+b*x

def func_loss(coe, x, y, xerr, yerr):
    return (y - func_linear(x, coe[0], coe[1]))/((yerr**2 + coe[1]**2*xerr**2)**0.5)

def sigma_boundary(x, a, b, aerr, berr, sigmaclip):
    table = np.zeros((len(x), 4))
    A = np.array([a-sigmaclip*aerr, a+sigmaclip*aerr, a-sigmaclip*aerr, a+sigmaclip*aerr])
    B = np.array([b-sigmaclip*berr, b-sigmaclip*berr, b+sigmaclip*berr, b+sigmaclip*berr])
    for i in range(4):
        table[:, i] = func_linear(x, A[i], B[i])
    boundary_max = np.max(table, axis=1)
    boundary_min = np.min(table, axis=1)
    return boundary_min, boundary_max

def linfit_sigclip(x, y, xerr, yerr, coe0=np.array([0, 1]), sigmaclip=3.0, maxiter=10, img=False):
    # 1. fit / 2. clip / 3. iter
    # return final mask, coe
    # !!! Final(maxiter) clipping is not conducted !!!
    fit = np.array([coe0])
    mask = np.full(len(x), True)
    mask = (mask == True) * np.logical_not(np.isnan(xerr)) * np.logical_not(np.isnan(yerr))
    Niter = 0
    for i in range(maxiter):
        Niter += 1
        # fit
        fit = leastsq(func_loss, fit[0], args=(x[mask], y[mask], xerr[mask], yerr[mask]), full_output=True)
        if fit[4] > 4: assert False, 'fit flag : %s' % fit[4]
        # clipping
        a_temp, b_temp = fit[0]
        aerr_temp, berr_temp = np.diag(fit[1]) ** 0.5
        boundary = sigma_boundary(x, a_temp, b_temp, aerr_temp, berr_temp, sigmaclip=sigmaclip)
        mask_new = (mask == True) * (y >= boundary[0]) * (y <= boundary[1])
        if np.array_equiv(mask, mask_new): break
        elif i == maxiter - 1: break       # [reach maxiter] final clipping should not be conducted
        else: mask = mask_new

    return mask, fit, Niter


def linfit_chisigclip(x, y, xerr, yerr, coe0=np.array([0, 1]), sigmaclip=3.0, maxiter=10):
    # 1. fit / 2. clip / 3. iter
    # return final mask, coe
    # !!! Final(maxiter) clipping is not conducted !!!
    fit = np.array([coe0])
    mask = np.full(len(x), True)
    Niter = 0
    for i in range(maxiter):
        Niter += 1
        # fit
        fit = leastsq(func_loss, fit[0], args=(x[mask], y[mask], xerr[mask], yerr[mask]), full_output=True)
        if fit[4] > 4: assert False, 'fit flag : %s' % fit[4]

        # clipping
        chi = func_loss(fit[0], x, y, xerr, yerr)
        chi_avg = np.mean(chi[mask])
        chi_std = np.std(chi[mask])
        mask_new = (mask == True) * (chi <= chi_avg + sigmaclip * chi_std) * (chi >= chi_avg - sigmaclip * chi_std)
        if np.array_equiv(mask, mask_new): break
        elif i == maxiter - 1 : break       # [reach maxiter] final clipping should not be conducted
        else: mask = mask_new

    return mask, fit, Niter