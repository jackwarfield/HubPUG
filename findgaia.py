import argparse as ap
from glob import glob

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astroquery.gaia import Gaia

from utils.transutils import get_zpt

### use Gaia eDR3
Gaia.MAIN_GAIA_TABLE = 'gaiaedr3.gaia_source'
Gaia.ROW_LIMIT = int(1e6)


def main(args):
    config = pd.read_json(args.config)
    csvloc1 = config.epoch1.csvloc
    csvloc2 = config.epoch2.csvloc
    gaialoc1 = config.epoch1.gaia
    gaialoc2 = config.epoch2.gaia
    fitsloc1 = config.epoch1.fitsloc
    fitsloc2 = config.epoch2.fitsloc
    prefix1 = config.epoch1.prefix
    prefix2 = config.epoch2.prefix
    gq1 = config.epoch1.gaiaqcut
    gq2 = config.epoch2.gaiaqcut

    ### bring in original CSV files as an array, then make an array of pandas DFs
    fns_1 = sorted(glob(f'{csvloc1}/{prefix1}*fl?.csv'))
    fns_2 = sorted(glob(f'{csvloc2}/{prefix2}*fl?.csv'))
    dfs_1 = [pd.read_csv(fn) for fn in fns_1]
    dfs_2 = [pd.read_csv(fn) for fn in fns_2]
    ### fits files
    fits1 = sorted(glob(f'{fitsloc1}/{prefix1}*fl?.fits'))[0]
    fits2 = sorted(glob(f'{fitsloc2}/{prefix2}*fl?.fits'))[0]

    ### set some general parameters
    t0 = 2016.0
    with fits.open(fits1) as hdu:
        t1 = (hdu[0].header['expstart'] - 51544) / 365.25 + 2000
        d1 = hdu[0].header['date-obs']
    with fits.open(fits2) as hdu:
        t2 = (hdu[0].header['expstart'] - 51544) / 365.25 + 2000
        d2 = hdu[0].header['date-obs']
    baseline = t2 - t1
    pix = config.general.gaiapix
    zpt1 = get_zpt(config.epoch1.filt, d1)
    zpt2 = get_zpt(config.epoch2.filt, d2)
    print(zpt1, zpt2)
    Mcutraw = config.general.gaia_Mcut
    Mcut = Mcutraw - 2.5 * np.log10(1000.0)
    width = pix * 1.38889e-5   # degrees
    cols = [
        'designation',
        'ra',
        'dec',
        'pmra',
        'pmra_error',
        'pmdec',
        'pmdec_error',
        'phot_g_mean_mag',
        'phot_rp_mean_mag',
    ]

    # take the average position in the first image, find all Gaia sources in 1.5x
    # that area
    ra, dec = dfs_1[0][['r', 'd']].mean()

    rerunquery = True
    if rerunquery == True:
        q = f"""
         SELECT TOP {int(1e6)}
         designation,ra,dec,pmra,pmra_error,pmdec,pmdec_error,phot_g_mean_mag,
         phot_rp_mean_mag,parallax,parallax_error,ruwe,phot_bp_mean_mag,
         COORD1(
           EPOCH_PROP_POS(ra,dec,parallax,pmra,pmdec,radial_velocity,
                          {t0},{t1})
           ) as RA_e1,
         COORD1(
           EPOCH_PROP_POS(ra,dec,parallax,pmra,pmdec,radial_velocity,
                          {t0},{t2})
           ) as RA_e2,
         COORD2(
           EPOCH_PROP_POS(ra,dec,parallax,pmra,pmdec,radial_velocity,
                          {t0},{t1})
           ) as DEC_e1,
         COORD2(
           EPOCH_PROP_POS(ra,dec,parallax,pmra,pmdec,radial_velocity,
                          {t0},{t2})
           ) as DEC_e2
         FROM gaiadr3.gaia_source
         WHERE
         CONTAINS(
           POINT('ICRS',ra,dec),
           CIRCLE('ICRS',{ra},{dec},0.0833333)
           )=1
         AND ruwe < 1.1
         AND ipd_gof_harmonic_amplitude <= 0.2
         AND visibility_periods_used >= 9
         AND astrometric_excess_noise_sig <= 2
         AND parallax_error < 0.7
         """
        # AND pmra_error < 1 AND pmdec_error < 1
        job = Gaia.launch_job(q)
        g = job.get_results().to_pandas()
        g.columns = [s.lower() for s in g.columns]
        _ = g.to_csv('output/fullgaiastamp.csv', index=False)
        # g = g[g.phot_g_mean_mag < Mcutraw-1].reset_index(drop=True)
    else:
        g = pd.read_csv('output/fullgaiastamp.csv')
        g = g[g.phot_g_mean_mag < Mcutraw - 1].reset_index(drop=True)

    for i in range(len(dfs_1)):
        print(f'Matching Gaia sources for {fns_1[i]}')
        df = dfs_1[i].copy()
        df = df[df.M < Mcut - zpt1].reset_index(drop=True)
        df = df[df.q < gq1].reset_index(drop=True)
        df['des'] = np.full(len(df), np.nan)
        df['gr'] = np.full(len(df), np.nan)
        df['gd'] = np.full(len(df), np.nan)
        df['gpmr'] = np.full(len(df), np.nan)
        df['gpmr_e'] = np.full(len(df), np.nan)
        df['gpmd'] = np.full(len(df), np.nan)
        df['gpmd_e'] = np.full(len(df), np.nan)
        df['gGmag'] = np.full(len(df), np.nan)
        df['gRPmag'] = np.full(len(df), np.nan)
        df['gBPmag'] = np.full(len(df), np.nan)
        df['gr_e1'] = np.full(len(df), np.nan)
        df['gd_e1'] = np.full(len(df), np.nan)
        df['gr_e2'] = np.full(len(df), np.nan)
        df['gd_e2'] = np.full(len(df), np.nan)

        ### loop through the rows
        for j in range(len(df)):
            ra = df.loc[j, 'r']
            dec = df.loc[j, 'd']
            cd = np.cos(dec * np.pi / 180)
            g['sep'] = (
                ((g.ra_e1 - ra) * cd) ** 2 + (g.dec_e1 - dec) ** 2
            ) ** 0.5
            gf = g.copy().sort_values(by='sep')
            gf = gf[gf.sep < width].reset_index(drop=True)
            if len(gf) > 0:
                df.loc[j, 'des'] = gf.loc[0, 'designation']
                df.loc[j, 'gr'] = gf.loc[0, 'ra']
                df.loc[j, 'gd'] = gf.loc[0, 'dec']
                df.loc[j, 'gpmr'] = gf.loc[0, 'pmra']
                df.loc[j, 'gpmr_e'] = gf.loc[0, 'pmra_error']
                df.loc[j, 'gpmd'] = gf.loc[0, 'pmdec']
                df.loc[j, 'gpmd_e'] = gf.loc[0, 'pmdec_error']
                df.loc[j, 'gGmag'] = gf.loc[0, 'phot_g_mean_mag']
                df.loc[j, 'gRPmag'] = gf.loc[0, 'phot_rp_mean_mag']
                df.loc[j, 'gBPmag'] = gf.loc[0, 'phot_bp_mean_mag']
                df.loc[j, 'gr_e1'] = gf.loc[0, 'ra_e1']
                df.loc[j, 'gd_e1'] = gf.loc[0, 'dec_e1']
                df.loc[j, 'gr_e2'] = gf.loc[0, 'ra_e2']
                df.loc[j, 'gd_e2'] = gf.loc[0, 'dec_e2']

        df['gr'] = df['gr_e1']
        df['gd'] = df['gd_e1']
        df = df[df.des.notna()].reset_index(drop=True)
        df = df.sort_values(by='m')
        df = df.drop_duplicates(keep='first').reset_index(drop=True)
        dfs_1[i] = df
        fn = fns_1[i].replace(csvloc1, gaialoc1)
        _ = df.to_csv(fn, index=False)

    for i in range(len(dfs_2)):
        print(f'Matching Gaia sources for {fns_2[i]}')
        df = dfs_2[i].copy()
        df = df[df.M < Mcut - zpt2].reset_index(drop=True)
        df = df[df.q < gq2].reset_index(drop=True)
        df['des'] = np.full(len(df), np.nan)
        df['gr'] = np.full(len(df), np.nan)
        df['gd'] = np.full(len(df), np.nan)
        df['gpmr'] = np.full(len(df), np.nan)
        df['gpmr_e'] = np.full(len(df), np.nan)
        df['gpmd'] = np.full(len(df), np.nan)
        df['gpmd_e'] = np.full(len(df), np.nan)
        df['gGmag'] = np.full(len(df), np.nan)
        df['gRPmag'] = np.full(len(df), np.nan)
        df['gBPmag'] = np.full(len(df), np.nan)
        df['gr_e1'] = np.full(len(df), np.nan)
        df['gd_e1'] = np.full(len(df), np.nan)
        df['gr_e2'] = np.full(len(df), np.nan)
        df['gd_e2'] = np.full(len(df), np.nan)

        ### loop through the rows
        for j in range(len(df)):
            ra = df.loc[j, 'r']
            dec = df.loc[j, 'd']
            cd = np.cos(dec * np.pi / 180)
            g['sep'] = (
                ((g.ra_e2 - ra) * cd) ** 2 + (g.dec_e2 - dec) ** 2
            ) ** 0.5
            gf = g.copy().sort_values(by='sep')
            gf = gf[gf.sep < width].reset_index(drop=True)
            if len(gf) > 0:
                df.loc[j, 'des'] = gf.loc[0, 'designation']
                df.loc[j, 'gr'] = gf.loc[0, 'ra']
                df.loc[j, 'gd'] = gf.loc[0, 'dec']
                df.loc[j, 'gpmr'] = gf.loc[0, 'pmra']
                df.loc[j, 'gpmr_e'] = gf.loc[0, 'pmra_error']
                df.loc[j, 'gpmd'] = gf.loc[0, 'pmdec']
                df.loc[j, 'gpmd_e'] = gf.loc[0, 'pmdec_error']
                df.loc[j, 'gGmag'] = gf.loc[0, 'phot_g_mean_mag']
                df.loc[j, 'gRPmag'] = gf.loc[0, 'phot_rp_mean_mag']
                df.loc[j, 'gBPmag'] = gf.loc[0, 'phot_bp_mean_mag']
                df.loc[j, 'gr_e1'] = gf.loc[0, 'ra_e1']
                df.loc[j, 'gd_e1'] = gf.loc[0, 'dec_e1']
                df.loc[j, 'gr_e2'] = gf.loc[0, 'ra_e2']
                df.loc[j, 'gd_e2'] = gf.loc[0, 'dec_e2']

        df['gr'] = df['gr_e2']
        df['gd'] = df['gd_e2']
        df = df[df.des.notna()].reset_index(drop=True)
        df = df.sort_values(by='m')
        df = df.drop_duplicates(keep='first').reset_index(drop=True)
        dfs_2[i] = df
        fn = fns_2[i].replace(csvloc2, gaialoc2)
        _ = df.to_csv(fn, index=False)


if __name__ == '__main__':
    parser = ap.ArgumentParser(
        description='Transform all images into the same\
                                          image frame.'
    )
    _ = parser.add_argument(
        '-c',
        '--config',
        help='Name of the config json file.\
                               (Default: config.json)',
        default='config.json',
        type=str,
    )
    args = parser.parse_args()

    _ = main(args)
