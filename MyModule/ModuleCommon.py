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

