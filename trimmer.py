import glob
import subprocess

import numpy as np
import pandas as pd

config = pd.read_json("./config.json")
qlim1 = config.epoch1.qcut
qlim2 = config.epoch2.qcut
trimpix = config.general.gaiapix

def gaiacut(df, gdf):
  for i in range(len(gdf)):
    row = gdf.loc[i]
    x,y = row[['X','Y']].values
    df['sep'] = np.sqrt((x-df.X)**2 + (y-df.Y)**2)
    df = df[df.sep > trimpix]
  return df.reset_index(drop=True)

print("trimmer.py")

#print("cut dolphot cat")
#dp = pd.read_csv("./ngc147ss.csv",)
#dp = dp[(dp.F606W_SNR>4) & (dp.F606W_CROWD<0.75) & (dp.F606W_SHARP**2<0.21) &\
#    (dp.F606W_ROUND<3)].reset_index(drop=True)
#dp = dp[(dp.F814W_SNR>4) & (dp.F814W_CROWD<0.75) & (dp.F814W_SHARP**2<0.21) &\
#    (dp.F814W_ROUND<3)].reset_index(drop=True)
#td = looppart(dp.X.values, dp.Y.values, dp.index.values)
#dp = dp.drop(index=td).reset_index(drop=True)

print("first")
fl = sorted(glob.glob("./firstcsv/*fl?.csv"))
gfl = sorted(glob.glob("./gaia1/*"))

for fn,gfn in zip(fl,gfl):
  print(fn)
  df = pd.read_csv(fn)
  df = df[df.q < qlim1].reset_index(drop=True)
  gdf = pd.read_csv(gfn)
  df = gaiacut(df.copy(), gdf.copy())
  df.to_csv(fn, index=False,)

qlim = config.epoch2.qcut

print("second")
fl = sorted(glob.glob("./secondcsv/*fl?.csv"))
gfl = sorted(glob.glob("./gaia2/*"))

for fn,gfn in zip(fl,gfl):
  print(fn)
  df = pd.read_csv(fn)
  df = df[df.q < qlim2].reset_index(drop=True)
  gdf = pd.read_csv(gfn)
  df = gaiacut(df.copy(), gdf.copy())
  df.to_csv(fn, index=False,)
