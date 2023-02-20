import pickle
import sys

import numpy as np
import pandas as pd
from acstools import acszpt


def cmd_cut(df, matchfn, low, high, exp):
    cols = ['r_M']
    for i in range(0, int(exp)):
        cols += [f'm_M{i+1}']
    dfM = df[cols]
    dfM_med = dfM.T.median()
    df['M_med'] = dfM_med

    df2 = pd.read_csv(matchfn)
    df['m_r'] = np.full(len(df), np.nan)
    df['m_d'] = np.full(len(df), np.nan)
    df['m_Vega_M'] = np.full(len(df), np.nan)
    for i in range(len(df)):
        row = df.loc[i]
        r, d, m = row[['r_r', 'r_d', 'M_med']]
        df2['s'] = np.sqrt((df2.r - r) ** 2 + (df2.d - d) ** 2)
        dfs = df2.sort_values(by='s')
        row2 = dfs.iloc[0]
        try:
            if row2.s < 2.0 * 2.8e-4:
                df.loc[i, 'm_r'] = row2.r
                df.loc[i, 'm_d'] = row2.d
                df.loc[i, 'm_Vega_M'] = row2.Vega_M
        except:
            continue
    mask = (df.M_med - df.m_Vega_M > low) & (df.M_med - df.m_Vega_M < high)
    return mask


def get_zpt(filt, date, detector='WFC'):
    """
    Run a query given the filter, date, and detector for an image, and
    return the corresponding magnitude zeropoint.
    """
    print(filt)
    q = acszpt.Query(date=date, detector=detector, filt=filt)
    filt_zpt = q.fetch()
    return filt_zpt['VEGAmag'][0].value


with open('utils/555adjust.sav', 'rb') as f:
    adj555 = pickle.load(f)


def calc_vega(mag, filt, date='1997-11-25', t=1000.0):
    """
    Calculate the Vega magnitude for a source given its instrumental magnitude,
    the filter and date of the observation. t=1000.0 for the normalized
    instrumental magnitudes from hst1pass.

    By default, F555W magnitudes are adjusted to conform more closely to
    corresponding F606W magnitudes.
    """
    zpt = get_zpt(filt, date)
    if filt == 'F555W':
        mag = mag + 2.5 * np.log10(t) + zpt
        mag = [[i] for i in mag]
        return adj555.predict(mag)
    else:
        mag = mag + 2.5 * np.log10(t) + zpt
        return mag


def match_stars(ref, mch, d, M_tol, radec=True, debug=False, pref='d'):
    """
    Matches stars from 2 different images on position.
    For the first match, use RA and Dec.
    """
    pind = 9
    if pref == 'M':
        pind = 10
    # [id, X, Y, M, r, d, x, y, Vega_M, dX, dM, last_X, last_Y]
    # [ 0, 1, 2, 3, 4, 5, 6, 7,      8, 9,  10,     11,     12]
    mid = np.full(len(ref), np.nan)
    mX = np.full(len(ref), np.nan)
    mY = np.full(len(ref), np.nan)
    mVM = np.full(len(ref), np.nan)
    mr = np.full(len(ref), np.nan)
    md = np.full(len(ref), np.nan)
    lX = np.full(len(ref), np.nan)
    lY = np.full(len(ref), np.nan)
    for i in range(len(ref)):
        if radec:
            x = ref[:, 4][i]
            y = ref[:, 5][i]
            M = ref[:, 8][i]
            cd = np.cos(y * np.pi / 180)
            mch[:, 9] = np.sqrt(
                ((mch[:, 4] - x) * cd) ** 2 + (mch[:, 5] - y) ** 2
            )
            mch[:, 10] = np.abs(mch[:, 8] - M)
            mch = mch[np.argsort(mch[:, pind])]
            mask = (mch[:, 9] < d) & (np.abs(mch[:, 8] - M) < M_tol)
            res = mch[mask]
        else:
            x = ref[:, 1][i]
            y = ref[:, 2][i]
            M = ref[:, 8][i]
            mch[:, 9] = np.sqrt((mch[:, 1] - x) ** 2 + (mch[:, 2] - y) ** 2)
            mch[:, 10] = np.abs(mch[:, 8] - M)
            mch = mch[np.argsort(mch[:, pind])]
            mask = (mch[:, 9] < d) & (np.abs(mch[:, 8] - M) < M_tol)
            res = mch[mask]
        if len(res) == 0:
            continue
        else:
            res = res[0]
            mid[i] = int(res[0])
            if debug:
                mX[i] = res[6]
                mY[i] = res[7]
            else:
                mX[i] = res[1]
                mY[i] = res[2]
            mVM[i] = res[8]
            mr[i] = res[4]
            md[i] = res[5]
            lX[i] = res[11]
            lY[i] = res[12]
    return [mid, mX, mY, mVM, mr, md, lX, lY]


def maskradec(df, ra_mask, dec_mask, radius=10, gt=True):
    atd = 1.389e-5   # 0.05 arcsec to degree
    for r, d in zip(ra_mask, dec_mask):
        cd = np.cos(d * np.pi / 180)
        sep = np.sqrt(((df.r - r) * cd) ** 2 + (df.d - d) ** 2)
        if gt:
            df = df[sep > radius * atd].reset_index(drop=True)
        else:
            df = df[sep < radius * atd].reset_index(drop=True)
    return df
