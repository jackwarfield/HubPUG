import argparse as ap
from glob import glob

import numpy as np
import pandas as pd

import utils.bailer_jones_dist as bjd

def correlate(dX1a, dX2a, dY1a, dY2a):
  rp = []
  for i in range(len(dX1a)):
    dX1 = np.array(eval(dX1a[i]))
    dY1 = np.array(eval(dY1a[i]))
    dX2 = np.array(eval(dX2a[i]))
    dY2 = np.array(eval(dY2a[i]))
    deltaX = np.append(dX1,dX2)
    deltaY = np.append(dY1,dY2)
    rp += [np.corrcoef(deltaX, deltaY)[0,1]]
  return np.array(rp)

def main(args):
  config = pd.read_json(args.config)

  ### get lists of the Gaia files, read in DataFrames
  e1 = f"{config.epoch1.gaia}/{config.epoch1.prefix}*flc.csv"
  e2 = f"{config.epoch2.gaia}/{config.epoch2.prefix}*flc.csv"
  fle1 = sorted(glob(e1))
  fle2 = sorted(glob(e2))
  dfse1 = [pd.read_csv(fn) for fn in fle1]
  dfse2 = [pd.read_csv(fn) for fn in fle2]

  ### drop stars that don't have Gaia pmra and pmdec values, if there are
  ### any duplicates, keep the star closest to the Gaia star in ra,dec
  for i in range(len(dfse1)):
    df = dfse1[i].copy()
    df = df[(df.gpmr.notna()) & (df.gpmd.notna())].reset_index(drop=True)
    cd = np.cos(df.d*np.pi/180)
    df['gsep'] = (((df.r-df.gr)*cd)**2 + (df.d-df.gd)**2)**0.5
    df = df.sort_values(by='gsep', ignore_index=True)
    df = df.drop_duplicates(subset='des', keep='first', ignore_index=True)
    dfse1[i] = df
  for i in range(len(dfse2)):
    df = dfse2[i].copy()
    df = df[(df.gpmr.notna()) & (df.gpmd.notna())].reset_index(drop=True)
    cd = np.cos(df.d*np.pi/180)
    df['gsep'] = (((df.r-df.gr)*cd)**2 + (df.d-df.gd)**2)**0.5
    df = df.sort_values(by='gsep', ignore_index=True)
    df = df.drop_duplicates(subset='des', keep='first', ignore_index=True)
    dfse2[i] = df

  df1 = dfse1[0]
  df2 = dfse2[0]
  for i in range(1,len(dfse1)):
    df = dfse1[i].copy()
    df1 = pd.merge(df1, df,
                   how="left", on="des",
                   suffixes=(None,f"{i}"))
  for i in range(1,len(dfse2)):
    df = dfse2[i].copy()
    df2 = pd.merge(df2, df,
                   how="left", on="des",
                   suffixes=(None,f"{i}"))

  keepcols = ['X','X_e','Y','Y_e','m','M','q','x','y','k','r','r_e','d','d_e',
              'des','gr','gd','gpmr','gpmr_e','gpmd','gpmd_e','gGmag','gRPmag',
              'gBPmag','gsep',]

  etol = 0.1

  Xcols = ['X']
  Ycols = ['Y']
  scols = ['gsep']
  rcols = ['r']
  dcols = ['d']
  for i in range(1,len(dfse1)):
    Xcols += [f'X{i}']
    Ycols += [f'Y{i}']
    scols += [f'gsep{i}']
    rcols += [f'r{i}']
    dcols += [f'd{i}']
  threshf = config.general.gthresh
  thresh = threshf*(len(Xcols)+len(Ycols))
  df1 = df1.dropna(subset=Xcols+Ycols, thresh=thresh)
  df1 = df1.reset_index(drop=True)
  df1['X'] = df1[Xcols].T.median()
  df1['Y'] = df1[Ycols].T.median()
  df1['X_e'] = (np.abs(df1['X']-df1[Xcols].T)).median()*1.4826
  df1['Y_e'] = (np.abs(df1['Y']-df1[Ycols].T)).median()*1.4826
  df1 = df1[(df1.X_e < etol) & (df1.Y_e < etol)].reset_index(drop=True)
  df1['gsep'] = df1[scols].T.median()
  df1['r'] = df1[rcols].T.median()
  df1['d'] = df1[dcols].T.median()
  df1['r_e'] = (np.abs(df1['r']-df1[rcols].T)).median()*1.4826
  df1['d_e'] = (np.abs(df1['d']-df1[dcols].T)).median()*1.4826
  dX1 = df1[Xcols].to_numpy()
  dY1 = df1[Ycols].to_numpy()
  for i in range(len(dX1)):
    dX1[i] = dX1[i] - df1.X.values[i]
    dY1[i] = dY1[i] - df1.Y.values[i]
  dX1 = [str(tuple(row)) for row in dX1]
  dY1 = [str(tuple(row)) for row in dY1]
  df1['dX1'] = dX1
  df1['dY1'] = dY1
  df1_f = df1[keepcols+['dX1']+['dY1']]
  _= df1.to_csv('output/ge1_full.csv', index=False)
  
  Xcols = ['X']
  Ycols = ['Y']
  scols = ['gsep']
  rcols = ['r']
  dcols = ['d']
  for i in range(1,len(dfse2)):
    Xcols += [f'X{i}']
    Ycols += [f'Y{i}']
    scols += [f'gsep{i}']
    rcols += [f'r{i}']
    dcols += [f'd{i}']
  thresh = threshf*(len(Xcols)+len(Ycols))
  df2 = df2.dropna(subset=Xcols+Ycols, thresh=thresh)
  df2 = df2.reset_index(drop=True)
  df2['X'] = df2[Xcols].T.median()
  df2['Y'] = df2[Ycols].T.median()
  df2['X_e'] = (np.abs(df2['X']-df2[Xcols].T)).median()*1.4826
  df2['Y_e'] = (np.abs(df2['Y']-df2[Ycols].T)).median()*1.4826
  df2 = df2[(df2.X_e < etol) & (df2.Y_e < etol)].reset_index(drop=True)
  df2['gsep'] = df2[scols].T.median()
  df2['r'] = df2[rcols].T.median()
  df2['d'] = df2[dcols].T.median()
  df2['r_e'] = (np.abs(df2['r']-df2[rcols].T)).median()*1.4826
  df2['d_e'] = (np.abs(df2['d']-df2[dcols].T)).median()*1.4826
  dX2 = df2[Xcols].to_numpy()
  dY2 = df2[Ycols].to_numpy()
  for i in range(len(dX2)):
    dX2[i] = dX2[i] - df2.X.values[i]
    dY2[i] = dY2[i] - df2.Y.values[i]
  dX2 = [str(tuple(row)) for row in dX2]
  dY2 = [str(tuple(row)) for row in dY2]
  df2['dX2'] = dX2
  df2['dY2'] = dY2
  df2_f = df2[keepcols+['dX2']+['dY2']]
  _= df2.to_csv('output/ge2_full.csv', index=False)

  df = pd.merge(df1_f, df2_f,
                how="inner", on="des",
                suffixes=("_e1","_e2"))
  #df['rp'] = correlate(df.dX1.values, df.dX2.values,
  #                     df.dY1.values, df.dY2.values)
  df['rp'] = 0

  g_e2_cols = ['gr_e2','gd_e2','gpmr_e2','gpmr_e_e2','gpmd_e2','gpmd_e_e2',
               'gGmag_e2','gRPmag_e2','gBPmag_e2','gsep_e2',]
  for cn in g_e2_cols:
    df[cn.replace("_e2","")] = df[cn]
  keepcols = ['X_e1','X_e_e1','Y_e1','Y_e_e1','m_e1','M_e1','q_e1',
              'x_e1','y_e1','k_e1','r_e1','r_e_e1','d_e1','d_e_e1','des',
              'X_e2','X_e_e2','Y_e2','Y_e_e2','m_e2','M_e2','q_e2',
              'x_e2','y_e2','k_e2','r_e2','r_e_e2','d_e2','d_e_e2','rp',
              'gr_e1','gd_e1',
              'gr','gd','gpmr','gpmr_e','gpmd','gpmd_e','gGmag','gRPmag',
              'gBPmag','gsep',]
  df = df[keepcols]

  g = pd.read_csv("output/fullgaiastamp.csv")
  df = pd.merge(df,g,
                how="left", left_on="des", right_on="designation",
                suffixes=(None,"_fg"))
  df['bjdist'] = bjd.main(1.2, df.parallax.values, df.parallax_error.values)
  
  _= df.to_csv("output/allgaia_list.csv", index=False)
  return df

if __name__ == '__main__':
  parser = ap.ArgumentParser(description="Transform all images into the same\
                                          image frame.")
  _= parser.add_argument("-c", "--config",
                        help="Name of the config json file.\
                              (Default: config.json)",
                        default="config.json", type=str)
  args = parser.parse_args()

  df = main(args)
