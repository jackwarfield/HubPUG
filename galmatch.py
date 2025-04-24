from glob import glob

import numpy as np
import pandas as pd
from astropy.io import fits

config = pd.read_json('./config.json')

qso_tot = pd.read_csv(config.output.gallist)

fn1 = glob('./firstcsv/*fl?.csv')
fn2 = glob('./secondcsv/*fl?.csv')

fn1 = glob(f'{config.epoch1.galcsv}/*fl?.csv')
fn2 = glob(f'{config.epoch2.galcsv}/*fl?.csv')

sepval = 6.94e-5 * 5

for fn in fn1:
    df = pd.read_csv(fn)
    # df = df[df.q > config.epoch1.gaiaqcut].reset_index(drop=True)
    # df = df[df.q < 0.5].reset_index(drop=True)
    rave = df.r.mean()
    dave = df.d.mean()
    qso = qso_tot

    df['qso'] = np.full(len(df), np.nan, dtype='<U50')
    df['msep'] = np.nan

    for i in qso.index:
        r, d = qso.loc[i, ['ra', 'dec']]
        df['sep'] = np.sqrt(
            (df.r - r) ** 2 * np.cos(np.radians(d)) ** 2 + (df.d - d) ** 2
        )
        df2 = df.sort_values(by='sep')
        df2 = df2[df2.sep < sepval]
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

    df.to_csv(
        f"./qso1/{fn.replace(f'{config.epoch1.galcsv}/','')}", index=False
    )

for fn in fn2:
    df = pd.read_csv(fn)
    # df = df[df.q > config.epoch2.gaiaqcut].reset_index(drop=True)
    rave = df.r.mean()
    dave = df.d.mean()
    qso = qso_tot
    qso = qso[
        (np.abs(qso.ra - rave) * np.cos(np.radians(dave)) < 1)
        & (np.abs(qso.dec - dave) < 1)
    ].reset_index(drop=True)

    df['qso'] = np.full(len(df), np.nan, dtype='<U50')
    df['msep'] = np.nan

    for i in qso.index:
        r, d = qso.loc[i, ['ra', 'dec']]
        df['sep'] = np.sqrt(
            (df.r - r) ** 2 * np.cos(np.radians(d)) ** 2 + (df.d - d) ** 2
        )
        df2 = df.sort_values(by='sep')
        df2 = df2[df2.sep < sepval * 3 / 2]
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

    df.to_csv(
        f"./qso2/{fn.replace(f'{config.epoch2.galcsv}/','')}", index=False
    )
