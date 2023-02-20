import glob
import random
import sys

import pandas as pd


def sortreduce(filenames, sortby):
    fl = sorted(glob.glob(filenames))
    cols = ['id', 'X', 'Y', 'M', 'r', 'd', 'x', 'y', 'k']
    for fn in fl:
        df = pd.read_csv(fn)
        df['id'] = random.sample(range(int(1e6)), len(df))
        df = df.sort_values(by=sortby, ascending=True)
        df[cols].to_csv(fn + 'sortred.csv', index=False)
    fl = [fn + 'sortred.csv' for fn in fl]
    return fl


if __name__ == '__main__':
    _ = sortreduce(sys.argv[1], 'M')
