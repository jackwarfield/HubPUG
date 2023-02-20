import argparse as ap
from glob import glob

import numpy as np
import pandas as pd
from astropy.io import fits


class gaia_pm:
    def __init__(self, g_fn, s_fn, fits1_fn, fits2_fn, include_Sat=True):
        """
        g        | str | filename for file with Gaia sources from combineproducts
        s        | str | filename for file with galaxy sources from transform.py
        fits1_fn | str | filepath for the reference fits image
        fits2_fn | str | filepath for the first e2 fits image
        """
        ### will determine whether you can ask the object for the results
        self.allPM = False
        ### pix->mas for ACS/WFC
        self.SCL = 50.0

        ### read in Gaia source list. by default, keep saturated stars
        self.g = pd.read_csv(g_fn)
        if not include_Sat:
            self.g = self.g[(self.g.q_e1 > 0) & (self.g.q_e2 > 0)]
            self.g = self.g.reset_index(drop=True)
        ### read in galaxy source list, find error contribution from transformation
        self.s = pd.read_csv(s_fn)
        l = np.sqrt(len(self.s))
        self.errgrandx = self.s.PMx.std() / l
        self.errgrandy = self.s.PMy.std() / l

        ### read in exposure dates, find baseline
        with fits.open(fits1_fn) as hdu:
            self.MJD1 = hdu[0].header['expstart']
        with fits.open(fits2_fn) as hdu:
            self.MJD2 = hdu[0].header['expstart']
        self.baseline = (self.MJD2 - self.MJD1) / 365.25
        print(f'baseline: {self.baseline}')

        tsig = (
            np.mean(np.sqrt(self.errgrandx**2 + self.errgrandy**2))
            * 50
            / self.baseline
        )
        print(f'trans sigma: {tsig}')

        ### calculate the total displacement over baseline from star pm
        self.g['gdr'] = self.g['pmra'] * self.baseline
        self.g['gdr_e'] = self.g['pmra_error'] * self.baseline
        self.g['gdd'] = self.g['pmdec'] * self.baseline
        self.g['gdd_e'] = self.g['pmdec_error'] * self.baseline
        ### create empty columns to store pm,pm_e later
        self.g['dN'] = np.full(len(self.g), np.nan)
        self.g['dN_e'] = np.full(len(self.g), np.nan)
        self.g['dE'] = np.full(len(self.g), np.nan)
        self.g['dE_e'] = np.full(len(self.g), np.nan)
        self.g['nogaia_dN'] = np.full(len(self.g), np.nan)
        self.g['nogaia_dN_e'] = np.full(len(self.g), np.nan)
        self.g['nogaia_dE'] = np.full(len(self.g), np.nan)
        self.g['nogaia_dE_e'] = np.full(len(self.g), np.nan)

    def single_gaia(self, i):
        """
        Find the proper motion for a single Gaia star in reference to the mean of
        the galaxy's star field
        """
        SCL = self.SCL
        # SCL = 3600/1000
        bl = self.baseline
        egx, egy = (self.errgrandx, self.errgrandy)

        ### choose Gaia star from row i, populate variables for each epoch
        g = self.g.loc[i]
        g_x1, g_xe1, g_y1, g_ye1 = g[['X_e1', 'X_e_e1', 'Y_e1', 'Y_e_e1']]
        g_x2, g_xe2, g_y2, g_ye2 = g[['X_e2', 'X_e_e2', 'Y_e2', 'Y_e_e2']]
        g_dd, g_dd_e, g_dr, g_dr_e = g[['gdd', 'gdd_e', 'gdr', 'gdr_e']]
        rp = g['rp']   # Gaia auto-correlation

        dx = (g_x2 - g_x1) * SCL
        # dx_e = SCL*((g_xe1**2 + g_xe2**2 + rp**2)**0.5 + egx)
        dx_e = SCL * ((g_xe1**2 + g_xe2**2 + rp**2 + egx**2) ** 0.5)
        dy = (g_y2 - g_y1) * SCL
        # dy_e = SCL*((g_ye1**2 + g_ye2**2 + rp**2)**0.5 + egy)
        dy_e = SCL * ((g_ye1**2 + g_ye2**2 + rp**2 + egy**2) ** 0.5)

        dE = -dx
        dN = dy
        dE_e = dx_e
        dN_e = dy_e

        self.g.loc[i, 'nogaia_dN'] = dN / bl
        self.g.loc[i, 'nogaia_dN_e'] = dN_e / bl
        self.g.loc[i, 'nogaia_dE'] = dE / bl
        self.g.loc[i, 'nogaia_dE_e'] = dE_e / bl

        dNf = dN - g_dd
        dNf_e = (dN_e**2 + (g_dd_e) ** 2) ** 0.5
        dEf = dE - g_dr
        dEf_e = (dE_e**2 + (g_dr_e) ** 2) ** 0.5

        self.g.loc[i, 'dN'] = dNf / bl
        self.g.loc[i, 'dN_e'] = np.abs(dNf_e / bl)
        self.g.loc[i, 'dE'] = dEf / bl
        self.g.loc[i, 'dE_e'] = np.abs(dEf_e / bl)

    def calcweight(self):
        df = self.g.copy()
        weights = 1 / df.dN_e.values**2
        waN, ws = np.average(df.dN.values, weights=weights, returned=True)
        waN_e = 0
        for w, e in zip(weights, df.dN_e.values):
            waN_e += (w * e / ws) ** 2
        waN_e = waN_e**0.5
        weights = 1 / df.dE_e.values**2
        waE, ws = np.average(df.dE.values, weights=weights, returned=True)
        waE_e = 0
        for w, e in zip(weights, df.dE_e.values):
            waE_e += (w * e / ws) ** 2
        waE_e = waE_e**0.5
        return -waN, waN_e, -waE, waE_e

    def calcPMs(self):
        """
        Loop through and calculate proper motion for all stars in the table. Then
        calculate weighted means, and stores the reflex.
        """
        for i in range(len(self.g)):
            _ = self.single_gaia(i)
        self.g = self.g[(self.g.dN.notna()) & (self.g.dE.notna())]
        self.g = self.g.reset_index(drop=True)
        self.dN, self.dN_e, self.dE, self.dE_e = self.calcweight()
        self.allPM = True

    def getPM(self):
        """
        Must run calcPMs first.
        Returns [dE,dN,dE_e,dN_E]
        """
        if self.allPM:
            return [self.dE, self.dN, self.dE_e, self.dN_e]
        else:
            print('You need to run calcPMs first!')
            return None

    def getTable(self):
        if self.allPM:
            return self.g
        else:
            print('You should probably run calcPMs first!')
            return self.g


def main(args):
    config = pd.read_json(args.config)
    g_fn = 'output/allgaia_list.csv'
    s_fn = 'output/finalMaT.csv'
    fits1_fn = f'{config.epoch1.fitsloc}/{config.epoch1.prefix}*fl?.fits'
    fits1_fn = sorted(glob(fits1_fn))[0]
    fits2_fn = f'{config.epoch2.fitsloc}/{config.epoch2.prefix}*fl?.fits'
    fits2_fn = sorted(glob(fits2_fn))[0]

    pmobj = gaia_pm(g_fn, s_fn, fits1_fn, fits2_fn, include_Sat=True)
    _ = pmobj.calcPMs()

    # dE,dN,dE_e,dN_e = pmobj.getPM()
    # resStr = f"dN = {dN:.3f} +/- {dN_e:.3f}\n"\
    #         f"dE = {dE:.3f} +/- {dE_e:.3f}"
    # print(resStr)
    # with open("output/result.txt", "w") as txtfile:
    #  print(resStr, file=txtfile)
    df = pmobj.getTable()
    _ = df.to_csv('output/resultsTable.csv', index=False)
    return 0


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

    raise SystemExit(main(args))
