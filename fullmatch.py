from glob import glob

import numpy as np
import pandas as pd

import utils.transutils as tu
from utils.transutils import match_stars

# get lists of files
fl1 = sorted(glob('firstfull/*flc.csv'))
fl2 = sorted(glob('secondfull/*flc.csv'))

# read first/reference table, then cut that fn from the list
ref = pd.read_csv(fl1[0])
fl1 = fl1[1:]

# cut reference table to only have useful range of values
# base this on the max magnitude of the galaxies (faintest)
galmax = pd.read_csv('../gallist.csv', usecols=['M'])['M'].max()
qmax = 1.0
# qmax = 99999
# galmax = -10
ref = ref[(ref['M'] < galmax + 3.0) & (ref['q'] < qmax)]
ref['id'] = ref.index.values
ref = ref.reset_index(drop=True)
ref['Vega_M'] = ref['M']
ref = ref[['id', 'X', 'Y', 'M', 'r', 'd', 'x', 'y', 'Vega_M']]
refnp = ref.to_numpy(copy=True)
reflennan = np.full(len(ref), np.nan, dtype=np.float64)
ypix = refnp[:, 7]

mcols = [
    'id',
    'X',
    'Y',
    'M',
    'r',
    'd',
    'x',
    'y',
    'Vega_M',
    'dX',
    'dM',
    'last_X',
    'last_Y',
]
rd = True
ps = 10 * 1.389e-5
ms = 5

x_arr1 = np.full((len(fl1) + 1, len(ref)), reflennan)
y_arr1 = np.full((len(fl1) + 1, len(ref)), reflennan)
m_arr1 = np.full((len(fl1) + 1, len(ref)), reflennan)

x_arr1[0] = refnp[:, 1]
y_arr1[0] = refnp[:, 2]
m_arr1[0] = refnp[:, 3]
for i, fn in enumerate(fl1):
    print(fn)
    df = pd.read_csv(fn)
    df = df[(df['M'] < galmax + 1.5) & (df['q'] < qmax)]
    df['id'] = df.index.values
    df = df.reset_index(drop=True)
    df['Vega_M'] = df['M']
    df['dX'] = np.nan
    df['dM'] = np.nan
    df['last_X'] = np.nan
    df['last_Y'] = np.nan
    df = df[mcols]
    mid, mX, mY, mVM, mr, md, _, __ = match_stars(
        refnp, df.to_numpy(copy=True), ps, ms, radec=rd, debug=False, pref='d'
    )
    x_arr1[i + 1] = mX
    y_arr1[i + 1] = mY
    m_arr1[i + 1] = mVM

x_arr2 = np.full((len(fl2) + 1, len(ref)), reflennan)
y_arr2 = np.full((len(fl2) + 1, len(ref)), reflennan)
m_arr2 = np.full((len(fl2) + 1, len(ref)), reflennan)

for i, fn in enumerate(fl2):
    print(fn)
    df = pd.read_csv(fn)
    df = df[(df['M'] < galmax + 1.5) & (df['q'] < qmax)]
    df['id'] = df.index.values
    df = df.reset_index(drop=True)
    df['Vega_M'] = df['M']
    df['dX'] = np.nan
    df['dM'] = np.nan
    df['last_X'] = np.nan
    df['last_Y'] = np.nan
    df = df[mcols]
    mid, mX, mY, mVM, mr, md, _, __ = match_stars(
        refnp, df.to_numpy(copy=True), ps, ms, radec=rd, debug=False, pref='d'
    )
    x_arr2[i + 1] = mX
    y_arr2[i + 1] = mY
    m_arr2[i + 1] = mVM

x_arr1 = x_arr1.T
y_arr1 = y_arr1.T
m_arr1 = m_arr1.T
x_arr2 = x_arr2.T
y_arr2 = y_arr2.T
m_arr2 = m_arr2.T

x1 = np.full((len(x_arr1), 3), np.nan, dtype=np.float64)
y1 = np.full((len(x_arr1), 3), np.nan, dtype=np.float64)
m1 = np.full((len(x_arr1), 3), np.nan, dtype=np.float64)

x2 = np.full((len(x_arr1), 3), np.nan, dtype=np.float64)
y2 = np.full((len(x_arr1), 3), np.nan, dtype=np.float64)
m2 = np.full((len(x_arr1), 3), np.nan, dtype=np.float64)

for i in range(len(x_arr1)):
    x = np.nanmedian(x_arr1[i])
    xc = np.count_nonzero(~np.isnan(x_arr1[i]))
    xe = np.nanstd(x_arr1[i]) / np.sqrt(xc)
    y = np.nanmedian(y_arr1[i])
    yc = np.count_nonzero(~np.isnan(y_arr1[i]))
    ye = np.nanstd(y_arr1[i]) / np.sqrt(yc)
    m = np.nanmedian(m_arr1[i])
    mc = np.count_nonzero(~np.isnan(m_arr1[i]))
    me = np.nanstd(m_arr1[i]) / np.sqrt(mc)

    x1[i] = np.array([x, xe, xc])
    y1[i] = np.array([y, ye, yc])
    m1[i] = np.array([m, me, mc])

    x = np.nanmedian(x_arr2[i])
    xc = np.count_nonzero(~np.isnan(x_arr2[i]))
    xe = np.nanstd(x_arr2[i]) / np.sqrt(xc)
    y = np.nanmedian(y_arr2[i])
    yc = np.count_nonzero(~np.isnan(y_arr2[i]))
    ye = np.nanstd(y_arr2[i]) / np.sqrt(yc)
    m = np.nanmedian(m_arr2[i])
    mc = np.count_nonzero(~np.isnan(m_arr2[i]))
    me = np.nanstd(m_arr2[i]) / np.sqrt(mc)

    x2[i] = np.array([x, xe, xc])
    y2[i] = np.array([y, ye, yc])
    m2[i] = np.array([m, me, mc])

cond = np.where(
    (x1[:, 2] >= int(0.5 * len(fl1) + 1)) & (x2[:, 2] >= int(0.5 * len(fl2)))
)
x1 = x1[cond]
y1 = y1[cond]
m1 = m1[cond]
x2 = x2[cond]
y2 = y2[cond]
m2 = m2[cond]
ypix = ypix[cond]

dX = np.zeros(len(x1), dtype=np.float64)
dXe = np.zeros(len(x1), dtype=np.float64)
dY = np.zeros(len(x1), dtype=np.float64)
dYe = np.zeros(len(x1), dtype=np.float64)

for i in range(len(x1)):
    x1_i, x1e_i, _ = x1[i]
    y1_i, y1e_i, _ = y1[i]

    x2_i, x2e_i, _ = x2[i]
    y2_i, y2e_i, _ = y2[i]

    dX[i] = x2_i - x1_i
    dXe[i] = np.sqrt(x1e_i**2 + x2e_i**2)
    dY[i] = y2_i - y1_i
    dYe[i] = np.sqrt(y1e_i**2 + y2e_i**2)

matched = pd.DataFrame(
    data={
        'x1': x1[:, 0],
        'y1': y1[:, 0],
        'x2': x2[:, 0],
        'y2': y2[:, 0],
        'dx': dX,
        'dxe': dXe,
        'dy': dY,
        'dye': dYe,
        'm1': m1[:, 0],
        'm2': m2[:, 0],
        'ypix': ypix,
    }
)
matched.to_csv('./output/fullmatched_forloc.csv', index=False)
