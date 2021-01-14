import numpy as np

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


def err_Add(errors):
    errors = np.array(errors)
    return (np.sum(errors**2, axis=0))**0.5

def err_Mul(elements, errors):
    assert False # check zero exception !
    x = np.prod(elements, axis=0)
    elements = np.array(elements)
    errors = np.array(errors)
    xerr = x * (np.sum((errors/elements)**2, axis=0))**0.5
    return xerr
