from glob import glob

import numpy as np
import pandas as pd
from astropy.io import fits

with fits.open('./utils/milliquas.fits') as hdu:
    data = hdu[1].data

d = {
    'ra': data['ra'].byteswap().newbyteorder(),
    'dec': data['dec'].byteswap().newbyteorder(),
    'name': data['name'].byteswap().newbyteorder(),
    'R': data['R'].byteswap().newbyteorder(),
    'B': data['B'].byteswap().newbyteorder(),
    'type': data['type'].byteswap().newbyteorder(),
}
qso_tot = pd.DataFrame(data=d)
qso_tot['uid'] = qso_tot.index.values
qso_tot['uid'] = qso_tot.uid.astype(int)
qso_tot = qso_tot[(qso_tot.R == '-') | (qso_tot.B == '-')].reset_index(
    drop=True
)
# qso_tot = qso_tot[(qso_tot.R != '1000') & (qso_tot.B != '1')].reset_index(drop=True)
# qso_tot = qso_tot[qso_tot.type == 'Q'].reset_index(drop=True)

fn1 = glob('./firstcsv/*fl?.csv')
fn2 = glob('./secondcsv/*fl?.csv')

for fn in fn1:
    df = pd.read_csv(fn)
    df = df[df.q < 0.4].reset_index(drop=True)
    rave = df.r.mean()
    dave = df.d.mean()
    qso = qso_tot
    qso = qso[
        (np.abs(qso.ra - rave) * np.cos(np.radians(dave)) < 1)
        & (np.abs(qso.dec - dave) < 1)
    ].reset_index(drop=True)

    df['qso'] = np.nan
    df['msep'] = np.nan

    for i in qso.index:
        r, d = qso.loc[i, ['ra', 'dec']]
        df['sep'] = np.sqrt(
            (df.r - r) ** 2 * np.cos(np.radians(d)) ** 2 + (df.d - d) ** 2
        )
        df2 = df.sort_values(by='sep')
        df2 = df2[df2.sep < 3 * 5.56e-5]
        if len(df2) > 0:
            j = df2.index.values[0]
            df.loc[j, 'qso'] = qso.loc[i, 'uid']
            df.loc[j, 'msep'] = df.loc[j, 'sep']

    df = df[df.qso.notna()].reset_index(drop=True)
    df['des'] = df.qso
    df['gpmr'] = 0.0
    df['gpmr_e'] = 0.0
    df['gpmd'] = 0.0
    df['gpmd_e'] = 0.0
    df['gGmag'] = 100
    df = df.drop(
        columns=[
            'qso',
            'msep',
        ]
    )

    df.to_csv(f"./qso1/{fn.replace('firstcsv/','')}", index=False)

for fn in fn2:
    df = pd.read_csv(fn)
    df = df[df.q < 0.4].reset_index(drop=True)
    rave = df.r.mean()
    dave = df.d.mean()
    qso = qso_tot
    qso = qso[
        (np.abs(qso.ra - rave) * np.cos(np.radians(dave)) < 1)
        & (np.abs(qso.dec - dave) < 1)
    ].reset_index(drop=True)

    df['qso'] = np.nan
    df['msep'] = np.nan

    for i in qso.index:
        r, d = qso.loc[i, ['ra', 'dec']]
        df['sep'] = np.sqrt(
            (df.r - r) ** 2 * np.cos(np.radians(d)) ** 2 + (df.d - d) ** 2
        )
        df2 = df.sort_values(by='sep')
        df2 = df2[df2.sep < 3 * 5.56e-5]
        if len(df2) > 0:
            j = df2.index.values[0]
            df.loc[j, 'qso'] = qso.loc[i, 'uid']
            df.loc[j, 'msep'] = df.loc[j, 'sep']

    df = df[df.qso.notna()].reset_index(drop=True)
    df['des'] = df.qso
    df['gpmr'] = 0.0
    df['gpmr_e'] = 0.0
    df['gpmd'] = 0.0
    df['gpmd_e'] = 0.0
    df['gGmag'] = 100
    df = df.drop(
        columns=[
            'qso',
            'msep',
        ]
    )

    df.to_csv(f"./qso2/{fn.replace('secondcsv/','')}", index=False)
