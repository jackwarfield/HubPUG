import subprocess
from glob import glob

import numpy as np
import pandas as pd

config = pd.read_json('./config.json')
qlim1 = config.epoch1.qcut
qlim2 = config.epoch2.qcut
trimpix = config.general.gaiapix


fns = sorted(glob('**/*flc.csv'))

for fn in fns:
    df = pd.read_csv(fn)
    df = df[df.y < 2048]
    df.to_csv(fn, index=False)
