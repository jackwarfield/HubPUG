import argparse as ap
import subprocess as sp
from os.path import exists

import pandas as pd


def main(args):
    config = pd.read_json(args.config)

    if not exists('utils/milliquas.fits'):
        sp.call('wget https://quasars.org/milliquas.fits.zip', shell=True)
        sp.call('unzip milliquas.fits.zip -d utils/', shell=True)
        sp.call('rm -f milliquas.fits.zip', shell=True)

    sp.call(['mkdir', config.epoch1.csvloc])
    sp.call(['mkdir', config.epoch2.csvloc])
    sp.call(['mkdir', config.epoch1.fullloc])
    sp.call(['mkdir', config.epoch2.fullloc])
    sp.call(['mkdir', 'qso1'])
    sp.call(['mkdir', 'qso2'])
    sp.call(['mkdir', config.epoch1.gaia])
    sp.call(['mkdir', config.epoch2.gaia])
    sp.call(['mkdir', 'output'])

    sp.call(f'rm -f {config.epoch1.csvloc}/*', shell=True)
    sp.call(f'rm -f {config.epoch2.csvloc}/*', shell=True)
    sp.call(f'rm -f {config.epoch1.fullloc}/*', shell=True)
    sp.call(f'rm -f {config.epoch2.fullloc}/*', shell=True)
    sp.call(f'rm -f qso1/*', shell=True)
    sp.call(f'rm -f qso2/*', shell=True)
    sp.call(f'rm -f {config.epoch1.gaia}/*', shell=True)
    sp.call(f'rm -f {config.epoch2.gaia}/*', shell=True)

    sp.call(
        f'cp {config.epoch1.fullcsv}/{config.epoch1.prefix}* firstcsv/',
        shell=True,
    )
    sp.call(
        f'cp {config.epoch2.fullcsv}/{config.epoch2.prefix}* secondcsv/',
        shell=True,
    )

    sp.call('python3 findgaia.py', shell=True)

    # for full trans
    sp.call(
        f'mv firstcsv/*flc.csv firstfull/',
        shell=True,
    )
    sp.call(
        f'mv secondcsv/*flc.csv secondfull/',
        shell=True,
    )

    sp.call(
        f'cp {config.epoch1.galcsv}/{config.epoch1.prefix}* firstcsv/',
        shell=True,
    )
    sp.call(
        f'cp {config.epoch2.galcsv}/{config.epoch2.prefix}* secondcsv/',
        shell=True,
    )

    sp.call('python3 galmatch.py', shell=True)

    sp.call(
        f'cp {config.epoch1.origcsv}/{config.epoch1.prefix}* firstcsv/',
        shell=True,
    )
    sp.call(
        f'cp {config.epoch2.origcsv}/{config.epoch2.prefix}* secondcsv/',
        shell=True,
    )

    if eval(config.general.trimmer):
        sp.call('python3 trimmer.py', shell=True)
    # sp.call('python3 transform.py', shell=True)
    sp.call('python3 transform_w_full.py', shell=True)
    sp.call('python3 combineproducts.py', shell=True)
    sp.call('python3 galcombine.py', shell=True)
    if eval(config.general.fullmatch):
        sp.call('python3 fullmatch.py', shell=True)
    sp.call('python3 findpm.py', shell=True)
    # sp.call('python3 oldfindpm.py', shell=True)
    sp.call('python3 plotresult.py', shell=True)
    # sp.call('python3 cdfplotresult.py', shell=True)

    return 0


if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Run HubPUG.')
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
