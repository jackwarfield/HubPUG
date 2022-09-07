### import things
import argparse as ap
import subprocess
from glob import glob
from warnings import simplefilter

import numpy as np
import pandas as pd
from astropy.io import fits

import utils.paultrans as pt
import utils.transutils as tu
from utils.linear6d import test_linear
from utils.reducecsv import sortreduce

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

def main(args):
  ### Read in config file
  print(f"\nReading in {args.config} file.\n")
  config = pd.read_json(args.config)

  ### Create lists for the filenames of the fits file
  ### Take into account upper and lower limits on exposure time
  print("Reading in fits filenames.")

  cfn1 = f"{config.epoch1.csvloc}/{config.epoch1.prefix}*fl?.csv"
  cfn2 = f"{config.epoch2.csvloc}/{config.epoch2.prefix}*fl?.csv"
  csv1_1 = sorted(glob(cfn1))
  csv2_1 = sorted(glob(cfn2))
  fits1_1 = [fn.replace(f"{config.epoch1.csvloc}/{config.epoch1.prefix}",\
                        f"{config.epoch1.fitsloc}/{config.epoch1.prefix}") for
             fn in csv1_1]
  fits1_1 = np.array([fn.replace(".csv",".fits") for fn in fits1_1])
  fits2_1 = [fn.replace(f"{config.epoch2.csvloc}/{config.epoch2.prefix}",\
                        f"{config.epoch2.fitsloc}/{config.epoch2.prefix}") for
             fn in csv2_1]
  fits2_1 = np.array([fn.replace(".csv",".fits") for fn in fits2_1])
  
  print("Reading in filenames for Gaia lists.")
  gfn1 = f"{config.epoch1.gaia}/{config.epoch1.prefix}*fl?.csv"
  gfn2 = f"{config.epoch2.gaia}/{config.epoch2.prefix}*fl?.csv"
  gaia1_1 = sorted(glob(gfn1))
  gaia2_1 = sorted(glob(gfn2))

  qfn1 = f"qso1/{config.epoch1.prefix}*fl?.csv"
  qfn2 = f"qso2/{config.epoch2.prefix}*fl?.csv"
  qso1_1 = sorted(glob(qfn1))
  qso2_1 = sorted(glob(qfn2))

  print("Reading fits files and cutting based on exposure times.\n")
  fits1 = []; fits2 = []
  date1 = []; date2 = []
  csv1  = []; csv2  = []
  gaia1 = []; gaia2 = []
  for f1,c1 in zip(fits1_1,csv1_1):
    with fits.open(f1) as hdu:
      exp = hdu[0].header['exptime']
      if ((exp > config.epoch1.expLlim) & (exp < config.epoch1.expUlim)):
        print(f"{f1} - {exp}s")
        fits1 += [f1]
        csv1 += [c1]
        date1 += [hdu[0].header['date-obs']]
  fits1 = np.array(fits1); csv1 = np.array(csv1)
  for f2,c2 in zip(fits2_1,csv2_1):
    with fits.open(f2) as hdu:
      exp = hdu[0].header['exptime']
      if ((exp > config.epoch2.expLlim) & (exp < config.epoch2.expUlim)):
        print(f"{f2} - {exp}s")
        fits2 += [f2]
        csv2 += [c2]
        date2 += [hdu[0].header['date-obs']]
  fits2 = np.array(fits2); csv2 = np.array(csv2)

  ### Create sorted and reduced version of the tables, then convert
  ### magnitudes to their respective Vega magnitudes.
  ### Apply the rotation so that y->N, x->W
  print("\nCutting source lists based on fit quality, magnitude.")
  qcut = config.epoch1.qcut
  for fn in csv1:
    df = pd.read_csv(fn)
    df = df[df.q < qcut]
    # could do position-based cuts here e.g. around saturated stars
    _= df.to_csv(f"{fn}qred.csv", index=False)
  qcut = config.epoch2.qcut
  for fn in csv2:
    df = pd.read_csv(fn)
    df = df[df.q < qcut]
    # could do position-based cuts here e.g. around saturated stars
    _= df.to_csv(f"{fn}qred.csv", index=False)

  print("Sorting tables and calculating Vega magnitudes for the 1st epoch.")
  fpath = f"{config.epoch1.csvloc}/{config.epoch1.prefix}*fl?.csvqred.csv"
  csv1 = sortreduce(fpath, "M")
  for fn,f1,d1 in zip(csv1,fits1,date1):
    df = pt.full_process(fn,f1)
    inmag = df.M.values
    Vmag = tu.calc_vega(inmag,config.epoch1.filt,d1)
    df["Vega_M"] = Vmag
    _= df.to_csv(fn, index=False)
    # rotate Gaia star lists
    gfn_ = fn.replace(f"{config.epoch1.csvloc}/",f"{config.epoch1.gaia}/")
    gfn_ = gfn_.replace("qred.csvsortred.csv","")
    gdf = pt.full_process(gfn_,f1)
    _= gdf.to_csv(gfn_, index=False)
    qfn_ = fn.replace(f"{config.epoch1.csvloc}/",f"qso1/")
    qfn_ = qfn_.replace("qred.csvsortred.csv","")
    qdf = pt.full_process(qfn_,f1)
    _= qdf.to_csv(qfn_, index=False)

  print("Sorting tables and calculating Vega magnitudes for the 2nd epoch.\n")
  fpath = f"{config.epoch2.csvloc}/{config.epoch2.prefix}*fl?.csvqred.csv"
  csv2 = sortreduce(fpath, "M")
  for fn,f2,d2 in zip(csv2,fits2,date2):
    df = pt.full_process(fn,f2)
    inmag = df.M.values
    Vmag = tu.calc_vega(inmag,config.epoch2.filt,d2)
    df["Vega_M"] = Vmag
    _= df.to_csv(fn, index=False)
    # rotate Gaia star lists
    gfn_ = fn.replace(f"{config.epoch2.csvloc}/",f"{config.epoch2.gaia}/")
    gfn_ = gfn_.replace("qred.csvsortred.csv","")
    gdf = pt.full_process(gfn_,f2)
    _= gdf.to_csv(gfn_, index=False)
    qfn_ = fn.replace(f"{config.epoch2.csvloc}/",f"qso2/")
    qfn_ = qfn_.replace("qred.csvsortred.csv","")
    qdf = pt.full_process(qfn_,f2)
    _= qdf.to_csv(qfn_, index=False)

  ### Read in files and match with reference objects, then get list of best.
  print("Preparing for first match and transformation.")
  allcsv = np.append(csv1,csv2)
  refcsv = allcsv[0]
  othercsv = allcsv[1:]
  ocsv_match = np.array([i+"_match.csv" for i in othercsv])
  gfns = [fn.replace(f"{config.epoch1.csvloc}/{config.epoch1.prefix}",
                     f"{config.epoch1.gaia}/{config.epoch1.prefix}") for
          fn in othercsv]
  gfns = [fn.replace(f"{config.epoch2.csvloc}/{config.epoch2.prefix}",
                     f"{config.epoch2.gaia}/{config.epoch2.prefix}") for \
          fn in gfns]
  gfns = [fn.replace(".csvqred.csvsortred","") for fn in gfns]
  gfns = np.array(gfns)

  qfns = [fn.replace(f"{config.epoch1.csvloc}/{config.epoch1.prefix}",
                     f"qso1/{config.epoch1.prefix}") for
          fn in othercsv]
  qfns = [fn.replace(f"{config.epoch2.csvloc}/{config.epoch2.prefix}",
                     f"qso2/{config.epoch2.prefix}") for \
          fn in qfns]
  qfns = [fn.replace(".csvqred.csvsortred","") for fn in qfns]
  qfns = np.array(qfns)

  print("Running first match and transformation.")
  goldnum = int(1000)
  Mlim_u = config.general.uppermagcut1
  Mlim_l = config.general.lowermagcut

  for i,(fn,nfn) in enumerate(zip(othercsv,ocsv_match)):
    print(f"\tFile:\t{fn}", end="")
    ref = pd.read_csv(refcsv)
    ref = ref[(ref.Vega_M > Mlim_u) & (ref.Vega_M < Mlim_l)]
    ref = ref.head(goldnum).reset_index(drop=True)
    df = pd.read_csv(fn)
    df_c = df[df.Vega_M > Mlim_u].reset_index(drop=True)
    df_c['dX'] = np.full(len(df_c),np.nan)
    df_c['dM'] = np.full(len(df_c),np.nan)
    df_c['last_X'] = np.full(len(df_c),np.nan)
    df_c['last_Y'] = np.full(len(df_c),np.nan)
    ref = ref[['id','X','Y','M','r','d','x','y','Vega_M',]]
    df_c = df_c[['id','X','Y','M','r','d','x','y','Vega_M','dX','dM',
                 'last_X','last_Y',]]
    rd = eval(config.general.match1_radec)
    ps = config.general.match1_pixsep
    if rd:
      ps *= 1.389e-5
    Ms = config.general.match1_Msep
    mid,mX,mY,mVM,mr,md,_,__ = tu.match_stars(ref.to_numpy(), df_c.to_numpy(),
                                              ps, Ms, radec=rd, debug=False,
                                              pref="d")
    ref['match_id'] = mid
    ref['match_X'] = mX
    ref['match_Y'] = mY
    ref['match_Vega_M'] = mVM
    ref['match_r'] = mr
    ref['match_d'] = md
    _= ref.to_csv(nfn, index=False)

    ref_m = ref[ref.match_X.notna()].reset_index(drop=True)
    cd = np.cos(ref_m.d)
    ref_m['sep'] = (cd*cd*(ref_m.r-ref_m.match_r)**2 +\
                          (ref_m.d-ref_m.match_d)**2)**0.5
    ref_m = ref_m.sort_values(by='sep').reset_index(drop=True)
    ref_m = ref_m.head(1000)
    wgts = np.ones(len(ref_m))
    mch_new,all_new = test_linear(ref_m.match_X.values, ref_m.match_Y.values,
                                  ref_m.X.values, ref_m.Y.values,
                                  wgts, wgts,
                                  df.X.values, df.Y.values)
    ref_m['new_X'] = mch_new[:,0]
    ref_m['new_Y'] = mch_new[:,1]
    ref_m['new_sep'] = ((ref_m.X-ref_m.new_X)**2 +\
                        (ref_m.Y-ref_m.new_Y)**2)**0.5
    avesep = ref_m.new_sep.mean()
    print(f"\t{avesep:.6f}")
    df['last_X'] = df['X']
    df['last_Y'] = df['Y']
    df['X'] = all_new[:,0]
    df['Y'] = all_new[:,1]
    _= df.to_csv(fn, index=False)

    g = pd.read_csv(gfns[i])
    _,g_new = test_linear(ref_m.match_X.values, ref_m.match_Y.values,
                          ref_m.X.values, ref_m.Y.values,
                          wgts, wgts,
                          g.X.values, g.Y.values)
    g['X'] = g_new[:,0]
    g['Y'] = g_new[:,1]
    _= g.to_csv(gfns[i], index=False)
    print(f"\t\t{gfns[i]}")
    q = pd.read_csv(qfns[i])
    _,q_new = test_linear(ref_m.match_X.values, ref_m.match_Y.values,
                          ref_m.X.values, ref_m.Y.values,
                          wgts, wgts,
                          q.X.values, q.Y.values)
    q['X'] = q_new[:,0]
    q['Y'] = q_new[:,1]
    _= q.to_csv(qfns[i], index=False)
    print(f"\t\t{qfns[i]}")

  print("\nRunning second match.")

  cols = []
  for i in range(len(othercsv)):
    cols += [f'm_sep{i+1}']

  Mlim_u = config.general.uppermagcut2
  Mlim_l = config.general.lowermagcut
  ref = pd.read_csv(refcsv)
  ref = ref[(ref.Vega_M > Mlim_u) & (ref.Vega_M < Mlim_l)]
  ref = ref.reset_index(drop=True)
  smp = pd.DataFrame()
  smp['r_id'] = ref.id.values
  smp['r_X'] = ref.X.values
  smp['r_Y'] = ref.Y.values
  smp['r_M'] = ref.Vega_M.values
  smp['r_r'] = ref.r.values
  smp['r_d'] = ref.d.values

  for i,(fn,nfn) in enumerate(zip(othercsv, ocsv_match)):
    print(f"\tFile:\t{fn}   ", end="\r")
    ref = pd.read_csv(refcsv)
    ref = ref[(ref.Vega_M > Mlim_u) & (ref.Vega_M < Mlim_l)]
    ref = ref.reset_index(drop=True)
    df = pd.read_csv(fn)
    df_c = df[(df.Vega_M > Mlim_u) & (df.Vega_M < Mlim_l)]
    df_c = df_c.reset_index(drop=True)
    df_c['dX'] = np.full(len(df_c), np.nan)
    df_c['dM'] = np.full(len(df_c), np.nan)
    ref = ref[['id','X','Y','M','r','d','x','y','Vega_M',]]
    df_c = df_c[['id','X','Y','M','r','d','x','y','Vega_M','dX','dM',
                 'last_X','last_Y',]]
    rd = eval(config.general.match2_radec)
    ps = config.general.match2_pixsep
    if rd:
      ps *= 1.389e-5
    Ms = config.general.match2_Msep
    mid,mX,mY,mVM,mr,md,lX,lY = tu.match_stars(ref.to_numpy(), df_c.to_numpy(),
                                               ps, Ms, radec=rd, debug=False,
                                               pref='d')
    ref['match_id'] = mid
    ref['match_X'] = mX
    ref['match_Y'] = mY
    ref['match_Vega_M'] = mVM
    ref['match_r'] = mr
    ref['match_d'] = md
    ref['last_X'] = lX
    ref['last_Y'] = lY
    _= ref.to_csv(nfn, index=False)
    ref['sep'] = ((ref.X-ref.match_X)**2 +\
                  (ref.Y-ref.match_Y)**2)**0.5
    #ref['sep'] = ((ref_m.X-ref_m.new_X)**2 +\
    #              (ref_m.Y-ref_m.new_Y)**2)**0.5
    #ref.to_csv("./lookatthis.csv")

    smp[f'm_id{i+1}'] = ref.match_id.values
    smp[f'm_X{i+1}'] = ref.match_X.values
    smp[f'm_Y{i+1}'] = ref.match_Y.values
    smp[f'm_r{i+1}'] = ref.match_r.values
    smp[f'm_d{i+1}'] = ref.match_d.values
    smp[f'm_M{i+1}'] = ref.match_Vega_M.values
    smp[f'm_lX{i+1}'] = ref.last_X.values
    smp[f'm_lY{i+1}'] = ref.last_Y.values
    smp[f'm_sep{i+1}'] = ref.sep.values

  ### make sure a star is seen in a number of images corresponding to thresh
  thresh = config.general.thresh
  thresh = int(len(cols)*thresh)
  smp_cut = smp.dropna(axis=0,
      thresh=thresh,
      subset=cols,
      ).reset_index(drop=True)

  ### apply cuts to source list based on average "proper motion" of stars
  ### between images
  pmtol = config.general.pmtol_start
  ss = smp_cut[cols]
  avestd = ss.T.std().mean()
  avesep = ss.T.mean().mean()
  for i in range(len(smp_cut)):
    std = ss.loc[i].std()
    objsep = ss.loc[i].mean()
    if ((objsep-avesep > pmtol) | (std-avestd > pmtol)):
      smp_cut = smp_cut.drop(i, axis=0)
  smp_cut = smp_cut.reset_index(drop=True)

  smp_cut = smp_cut.sort_values(by='r_M', ignore_index=True)
  smp_cut = smp_cut.head(100)

  print("\nTransforming.")
  for i,(fn,nfn) in enumerate(zip(othercsv,ocsv_match)):
    print(f"{i+1}\tFile:\t{fn}", end="")
    df = pd.read_csv(fn)
    smp_c = smp_cut[['r_X','r_Y',f'm_X{i+1}',f'm_Y{i+1}',f'm_sep{i+1}']]
    smp_c.columns = ['X','Y','m_X','m_Y','sep']
    smp_c = smp_c.dropna(axis=0).reset_index(drop=True)
    avesep = smp_c.sep.mean()
    print(f"\t{avesep:.4f} ", end="")
    wgts = np.ones(len(smp_c))

    # try statement here, maybe
    smpc_new,all_new = test_linear(smp_c.m_X.values, smp_c.m_Y.values,
                                   smp_c.X.values, smp_c.Y.values,
                                   wgts, wgts,
                                   df.X.values, df.Y.values)
    _,smpcut_new = test_linear(smp_c.m_X.values, smp_c.m_Y.values,
                               smp_c.X.values, smp_c.Y.values,
                               wgts, wgts,
                               smp_cut[f'm_X{i+1}'].values,
                               smp_cut[f'm_Y{i+1}'].values)
    smp_c['new_X'] = smpc_new[:,0]
    smp_c['new_Y'] = smpc_new[:,1]
    smp_cut[f'm_X{i+1}'] = smpcut_new[:,0]
    smp_cut[f'm_Y{i+1}'] = smpcut_new[:,1]
    smp_cut[f'm_sep{i+1}'] = ((smp_cut.r_X-smp_cut[f'm_X{i+1}'])**2+\
                              (smp_cut.r_Y-smp_cut[f'm_Y{i+1}'])**2)**0.5
    df['last_X'] = df['X']
    df['last_Y'] = df['Y']
    df['X'] = all_new[:,0]
    df['Y'] = all_new[:,1]
    _= df.to_csv(fn, index=False)

    smp_c['new_sep'] = ((smp_c.X-smp_c.new_X)**2 +\
                        (smp_c.Y-smp_c.new_Y)**2)**0.5
    enddf = smp_c.copy()
    avesep = smp_c.new_sep.mean()

    g = pd.read_csv(gfns[i])
    _,g_new = test_linear(smp_c.m_X.values, smp_c.m_Y.values,
                          smp_c.X.values, smp_c.Y.values,
                          wgts, wgts,
                          g.X.values, g.Y.values)
    g['X'] = g_new[:,0]
    g['Y'] = g_new[:,1]
    _ = g.to_csv(gfns[i], index=False)

    q = pd.read_csv(qfns[i])
    _,q_new = test_linear(smp_c.m_X.values, smp_c.m_Y.values,
                          smp_c.X.values, smp_c.Y.values,
                          wgts, wgts,
                          q.X.values, q.Y.values)
    q['X'] = q_new[:,0]
    q['Y'] = q_new[:,1]
    _ = q.to_csv(qfns[i], index=False)

    print(f"-> {avesep:.4f}")

  xcols=[]; ycols=[]
  for i in range(0,len(othercsv)):
    xcols += [f'm_X{i+1}']
    ycols += [f'm_Y{i+1}']

  matchnum = 3
  loop = 5
  while loop > 0:
    print(f"\nTransform round {matchnum}.")
    if pmtol > config.general.pmtol_end:
      pmtol = pmtol/config.general.pmtol_speed

    print("Apply cuts to source list.")
    smp = smp_cut.copy()
    smp_cut = smp.dropna(axis=0, thresh=thresh).reset_index(drop=True)
    xave=[]; yave=[]
    for i in range(len(smp_cut)):
      x = smp_cut.loc[i,xcols].mean()
      y = smp_cut.loc[i,ycols].mean()
      xave += [x]
      yave += [y]
    smp_cut['m_Xave'] = xave
    smp_cut['m_Yave'] = yave
    smp_cut['PMx'] = smp_cut.m_Xave - smp_cut.r_X
    smp_cut['PMy'] = smp_cut.m_Yave - smp_cut.r_Y

    ss = smp_cut[cols]
    avestd = np.mean(np.std(ss.T))
    avesep = np.mean(np.mean(ss.T))
    for i in range(len(smp_cut)):
      std = np.std(ss.loc[i])
      objsep = np.mean(ss.loc[i])
      PMx = smp_cut.loc[i,'PMx']
      PMy = smp_cut.loc[i,'PMy']
      if ((abs(objsep-avesep) > pmtol) | (abs(std-avestd) > pmtol) |\
          (abs(PMx) > pmtol) | (abs(PMy) > pmtol)):
        smp_cut = smp_cut.drop(i,axis=0)
    smp_cut = smp_cut.reset_index(drop=True)
    print(f"{len(smp_cut)} stars left.")
    
    print("Transforming.")
    for i,(fn,nfn) in enumerate(zip(othercsv,ocsv_match)):
      print(f"{i+1}\tFile\t{fn}", end="")
      df = pd.read_csv(fn)
      smp_c = smp_cut[['r_X','r_Y',f'm_X{i+1}',f'm_Y{i+1}']]
      smp_c.columns = ['X','Y','m_X','m_Y']
      smp_c = smp_c.dropna(axis=0)
      smp_c['sep'] = ((smp_c.X-smp_c.m_X)**2 +\
                      (smp_c.Y-smp_c.m_Y)**2)**0.5
      avesep = smp_c.sep.mean()
      print(f"\t{avesep:.4f} ", end="")
      wgts = np.ones(len(smp_c))
      
      smpc_new,all_new = test_linear(smp_c.m_X.values, smp_c.m_Y.values,
                                     smp_c.X.values, smp_c.Y.values,
                                     wgts, wgts,
                                     df.X.values, df.Y.values)
      _,smpcut_new = test_linear(smp_c.m_X.values, smp_c.m_Y.values,
                                 smp_c.X.values, smp_c.Y.values,
                                 wgts, wgts,
                                 smp_cut[f'm_X{i+1}'].values,
                                 smp_cut[f'm_Y{i+1}'].values)
      smp_c['new_X'] = smpc_new[:,0]
      smp_c['new_Y'] = smpc_new[:,1]
      smp_cut[f'm_X{i+1}'] = smpcut_new[:,0]
      smp_cut[f'm_Y{i+1}'] = smpcut_new[:,1]
      smp_cut[f'm_sep{i+1}'] = ((smp_cut.r_X-smp_cut[f'm_X{i+1}'])**2 +\
                                (smp_cut.r_Y-smp_cut[f'm_Y{i+1}'])**2)**0.5
      df['last_X'] = df['X']
      df['last_Y'] = df['Y']
      df['X'] = all_new[:,0]
      df['Y'] = all_new[:,1]
      _= df.to_csv(fn, index=False)

      smp_c['new_sep'] = ((smp_c.X-smp_c.new_X)**2 +\
                          (smp_c.Y-smp_c.new_Y)**2)**0.5
      enddf = smp_c.copy()
      avesep = smp_c.new_sep.mean()

      g = pd.read_csv(gfns[i])
      _,g_new = test_linear(smp_c.m_X.values, smp_c.m_Y.values,
                            smp_c.X.values, smp_c.Y.values,
                            wgts, wgts,
                            g.X.values, g.Y.values)
      g['X'] = g_new[:,0]
      g['Y'] = g_new[:,1]
      _= g.to_csv(gfns[i], index=False)

      print(f"-> {avesep:.4f}")
    matchnum += 1
    if pmtol < config.general.pmtol_end:
      loop -= 1

  _= smp_cut.to_csv('output/finalMaT.csv', index=False)

  return smp_cut

if __name__ == '__main__':
  parser = ap.ArgumentParser(description="Transform all images into the same\
                                          image frame.")
  _= parser.add_argument("-c", "--config",
                        help="Name of the config json file.\
                              (Default: config.json)",
                        default="config.json", type=str)
  args = parser.parse_args()
  #_= subprocess.call("make --silent clean", shell=True)

  df = main(args)
