import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rc('xtick', top=True, direction='in', labelsize=15)
plt.rc('ytick', right=True, direction='in', labelsize=15)
plt.rc('font', family='Arial')
plt.rc('errorbar', capsize=5)
scarlet = '#bb0000'
grey = '#666666'
rotorange = '#E57200'
jeffblue = '#232D4B'

def calcweight(df):
  l = np.sqrt(len(df))
  wmlim = int(l**2/8)-1
  wmlim = 2
  df = df.sort_values(by='dN', ascending=True).reset_index(drop=True)
  weights = 1/df.dN_e.values**2
  wt = np.sum(weights)
  w,i = 0,-1
  while w < 0.15865:
    w += weights[(i := i+1)]/wt
  lower = []
  l_e = []
  for j in range(i-wmlim,i+wmlim+1):
    lower += [df.dN.values[j]]
    l_e += [df.dN_e.values[j]]
  l_e = 1/np.array(l_e)**2
  lower = np.average(lower, weights=l_e, returned=False)
  while w < 0.5:
    w += weights[(i := i+1)]/wt
  waN = df.dN.values[i-1]
  waN = []
  waN_e = []
  for j in range(i-wmlim,i+wmlim+1):
    waN += [df.dN.values[j]]
    waN_e += [df.dN_e.values[j]]
  waN_e = 1/np.array(waN_e)**2
  waN = np.average(waN, weights=waN_e, returned=False)
  while w < 0.84135:
    w += weights[(i := i+1)]/wt
  upper = df.dN.values[i-1]
  upper = []
  u_e = []
  for j in range(i-wmlim,i+wmlim+1):
    upper += [df.dN.values[j]]
    u_e += [df.dN_e.values[j]]
  u_e = 1/np.array(u_e)**2
  upper = np.average(upper, weights=u_e, returned=False)
  waN_e = [[(waN-lower)/l], [(upper-waN)/l]]

  df = df.sort_values(by='dE', ascending=True).reset_index(drop=True)
  weights = 1/df.dE_e.values**2
  wt = np.sum(weights)
  w,i = 0,-1
  while w < 0.15865:
    w += weights[(i := i+1)]/wt
  lower = []
  l_e = []
  for j in range(i-wmlim,i+wmlim+1):
    lower += [df.dE.values[j]]
    l_e += [df.dE_e.values[j]]
  l_e = 1/np.array(l_e)**2
  lower = np.average(lower, weights=l_e, returned=False)
  while w < 0.5:
    w += weights[(i := i+1)]/wt
  waE = df.dE.values[i-1]
  waE = []
  waE_e = []
  for j in range(i-wmlim,i+wmlim+1):
    waE += [df.dE.values[j]]
    waE_e += [df.dE_e.values[j]]
  waE_e = 1/np.array(waE_e)**2
  waE = np.average(waE, weights=waE_e, returned=False)
  while w < 0.84135:
    w += weights[(i := i+1)]/wt
  upper = df.dE.values[i-1]
  upper = []
  u_e = []
  for j in range(i-wmlim,i+wmlim+1):
    upper += [df.dE.values[j]]
    u_e += [df.dE_e.values[j]]
  u_e = 1/np.array(u_e)**2
  upper = np.average(upper, weights=u_e, returned=False)
  waE_e = [[(waE-lower)/l], [(upper-waE)/l]]

  #print(f"E: {waE} + {waE_e[1][0]} - {waE_e[0][0]}")
  #print(f"N: {waN} + {waN_e[1][0]} - {waN_e[0][0]}")
  return waN,waN_e,waE,waE_e

def calcweight2(df):
  l = len(df)
  weights = 1/df.dN_e.values**2
  waN, ws = np.average(df.dN.values, weights=weights, returned=True)
  waN_e = 0
  for w,e in zip(weights, df.dN_e.values):
    waN_e += (w*e/ws)**2
  waN_e = np.sqrt(waN_e)
  waN_e = [[waN_e],[waN_e]]
  weights = 1/df.dE_e.values**2
  waE, ws = np.average(df.dE.values, weights=weights, returned=True)
  waE_e = 0
  for w,e in zip(weights, df.dE_e.values):
    waE_e += (w*e/ws)**2
  waE_e = np.sqrt(waE_e)
  waE_e = [[waE_e],[waE_e]]
  l = np.sqrt(len(df))
  return waN,waN_e,waE,waE_e

config = pd.read_json("config.json")
wmean = eval(config.output.wmean)
if wmean:
  calcweight = calcweight2

df = pd.read_csv('output/resultsTable.csv')

df['dN'] = -df.dN
df['dE'] = -df.dE

df = df.sort_values(by='phot_g_mean_mag', ascending=False,).reset_index(drop=True)
dffull = df.copy()

l0,l = len(df),0
waN,waN_e,waE,waE_e = calcweight(df.copy())
if eval(config.output.sigmaclip):
  for i in range(10):
    l0 = len(df)
    df = df[(np.abs(df.dN-waN) < 3.0*df.dN_e)]
    df = df[(np.abs(df.dE-waE) < 3.0*df.dE_e)]
    waN,waN_e,waE,waE_e = calcweight(df.copy())
    l = len(df)
  df = df[(np.abs(df.dN-waN) < 3.0*df.dN_e)]
  df = df[(np.abs(df.dE-waE) < 3.0*df.dE_e)]
  waN,waN_e,waE,waE_e = calcweight(df.copy())
keep = df.index.values

print('mu_alpha: ', waE, waE_e)
print('mu_delta: ', waN, waN_e)

fig = plt.figure(figsize=(15,7))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.grid(alpha=0.5), ax2.grid(alpha=0.5)
fig.suptitle(config.output.targetname, fontsize=25, y=0.93)
fig.subplots_adjust(wspace=0.0)

j = 0
for i in range(1,len(dffull)+1):
  j += 0.25
  dN,dNe = dffull.loc[i-1][['dN','dN_e']].values
  dE,dEe = dffull.loc[i-1][['dE','dE_e']].values
  if dffull.loc[i-1,'q_e2'] == 0:
    mkr = 'x'
  else:
    mkr = 'o'
  if i-1 in keep:
    ax1.errorbar(dE, j, xerr=dEe, fmt=mkr, color=rotorange)
    ax2.errorbar(dN, j, xerr=dNe, fmt=mkr, color=rotorange)
  else:
    ax1.errorbar(dE, j, xerr=dEe, fmt=mkr, color=rotorange, alpha=0.2)
    ax2.errorbar(dN, j, xerr=dNe, fmt=mkr, color=rotorange, alpha=0.2)

i = j
s = 'Individual Reflex Gaia'
s = 'Individual Gaia\nstellar reflex PMs'
fs = 14
fs2 = 17
ax2.text(1.0, i, s, color=rotorange, ha='right', va='center', fontsize=fs)
i+=1-0.25+0.25
i
if wmean:
  s = 'Weighted Mean'
else:
  s = 'Weighted Median'
ax1.errorbar(waE, i, xerr=waE_e, fmt='|', color=jeffblue, alpha=0.7)
ax2.errorbar(waN, i, xerr=waN_e, fmt='|', color=jeffblue, alpha=0.7)
ax2.text(1.0, i, s, color=jeffblue, alpha=0.7, ha='right', va='center', fontsize=fs)
if wmean:
  ax1.text(-1.0, i, f'{waE:.3f}'+'$\\pm$'+f'{np.mean(waE_e):.3f}',
      color=jeffblue, alpha=0.7, ha='left', va='center', fontsize=fs2,)
  ax2.text(-1.0, i, f'{waN:.3f}'+'$\\pm$'+f'{np.mean(waN_e):.3f}',
      color=jeffblue, alpha=0.7, ha='left', va='center', fontsize=fs2,)
else:
  ax1.text(-1.0, i, f'{waE:.3f}'+r'$\pm^{%.4f}_{%.4f}$' %(waE_e[1][0],waE_e[0][0]),
      color=jeffblue, alpha=0.7, ha='left', va='center',
      fontsize=fs2,)
  ax2.text(-1.0, i, f'{waN:.3f}'+r'$\pm^{%.4f}_{%.4f}$' %(waN_e[1][0],waN_e[0][0]),
      color=jeffblue, alpha=0.7, ha='left', va='center',
      fontsize=fs2,)

ax1.axvline(0, ls="-", lw=1, color='k', alpha=0.5, zorder=0.005)
ax1.axvline(waE, ls="--", lw=1, color=jeffblue, alpha=0.8, zorder=0.005)
ax2.axvline(0, ls="-", lw=1, color='k', alpha=0.5, zorder=0.005)
ax2.axvline(waN, ls="--", lw=1, color=jeffblue, alpha=0.8, zorder=0.005)

ax1.set_xlabel(r'$\mu_{\alpha}^*$ [mas/yr]', fontsize=20)
ax2.set_xlabel(r'$\mu_{\delta}$ [mas/yr]', fontsize=20)
ax1.set_yticks([]), ax2.set_yticks([])
ax1.set_xlim([-1.1,1.1]), ax2.set_xlim([-1.1,1.1])

fig.savefig(f'{config.output.targetname.replace(" ","_")}_summary.pdf',
    dpi=800, bbox_inches='tight',
    )

