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
    wmlim = int(l**2 / 8) - 1
    wmlim = 0
    df = df.sort_values(by='dN', ascending=True).reset_index(drop=True)
    weights = 1 / df.dN_e.values**2
    wt = np.sum(weights)
    w, i = 0, -1
    while w < 0.15865:
        w += weights[(i := i + 1)] / wt
    lower = []
    l_e = []
    for j in range(i - wmlim, i + wmlim + 1):
        lower += [df.dN.values[j]]
        l_e += [df.dN_e.values[j]]
    l_e = 1 / np.array(l_e) ** 2
    lower = np.average(lower, weights=l_e, returned=False)
    while w < 0.5:
        w += weights[(i := i + 1)] / wt
    waN = df.dN.values[i - 1]
    waN = []
    waN_e = []
    for j in range(i - wmlim, i + wmlim + 1):
        waN += [df.dN.values[j]]
        waN_e += [df.dN_e.values[j]]
    waN_e = 1 / np.array(waN_e) ** 2
    waN = np.average(waN, weights=waN_e, returned=False)
    while w < 0.84135:
        w += weights[(i := i + 1)] / wt
    upper = df.dN.values[i - 1]
    upper = []
    u_e = []
    for j in range(i - wmlim, i + wmlim + 1):
        upper += [df.dN.values[j]]
        u_e += [df.dN_e.values[j]]
    u_e = 1 / np.array(u_e) ** 2
    upper = np.average(upper, weights=u_e, returned=False)
    waN_e = [[(waN - lower) / l], [(upper - waN) / l]]

    df = df.sort_values(by='dE', ascending=True).reset_index(drop=True)
    weights = 1 / df.dE_e.values**2
    wt = np.sum(weights)
    w, i = 0, -1
    while w < 0.15865:
        w += weights[(i := i + 1)] / wt
    lower = []
    l_e = []
    for j in range(i - wmlim, i + wmlim + 1):
        lower += [df.dE.values[j]]
        l_e += [df.dE_e.values[j]]
    l_e = 1 / np.array(l_e) ** 2
    lower = np.average(lower, weights=l_e, returned=False)
    while w < 0.5:
        w += weights[(i := i + 1)] / wt
    waE = df.dE.values[i - 1]
    waE = []
    waE_e = []
    for j in range(i - wmlim, i + wmlim + 1):
        waE += [df.dE.values[j]]
        waE_e += [df.dE_e.values[j]]
    waE_e = 1 / np.array(waE_e) ** 2
    waE = np.average(waE, weights=waE_e, returned=False)
    while w < 0.84135:
        w += weights[(i := i + 1)] / wt
    upper = df.dE.values[i - 1]
    upper = []
    u_e = []
    for j in range(i - wmlim, i + wmlim + 1):
        upper += [df.dE.values[j]]
        u_e += [df.dE_e.values[j]]
    u_e = 1 / np.array(u_e) ** 2
    upper = np.average(upper, weights=u_e, returned=False)
    waE_e = [[(waE - lower) / l], [(upper - waE) / l]]

    # print(f"E: {waE} + {waE_e[1][0]} - {waE_e[0][0]}")
    # print(f"N: {waN} + {waN_e[1][0]} - {waN_e[0][0]}")
    return waN, waN_e, waE, waE_e


def calcweight2(df):
    l = len(df)
    weights = 1 / df.dN_e.values**2
    waN, ws = np.average(df.dN.values, weights=weights, returned=True)
    waN_e = 0
    for w, e in zip(weights, df.dN_e.values):
        waN_e += (w * e / ws) ** 2
    waN_e = np.sqrt(waN_e)
    waN_e = [[waN_e], [waN_e]]
    weights = 1 / df.dE_e.values**2
    waE, ws = np.average(df.dE.values, weights=weights, returned=True)
    waE_e = 0
    for w, e in zip(weights, df.dE_e.values):
        waE_e += (w * e / ws) ** 2
    waE_e = np.sqrt(waE_e)
    waE_e = [[waE_e], [waE_e]]
    l = np.sqrt(len(df))
    return waN, waN_e, waE, waE_e


def calcweight3(df, iters=1e3):
    dNmeds, dEmeds = [], []
    dN = df.dN.to_numpy(copy=True)
    dN_e = df.dN_e.to_numpy(copy=True)
    dE = df.dE.to_numpy(copy=True)
    dE_e = df.dE_e.to_numpy(copy=True)
    for _ in range(int(iters)):
        dNarr = np.random.normal(
            df.dN.to_numpy(copy=True), df.dN_e.to_numpy(copy=True)
        )
        dNarr = dNarr[np.random.randint(len(dNarr))]
        dEarr = np.random.normal(
            df.dE.to_numpy(copy=True), df.dE_e.to_numpy(copy=True)
        )
        dEarr = dEarr[np.random.randint(len(dEarr))]
        dNmeds += [np.median(dNarr)]
        dEmeds += [np.median(dEarr)]

    waN = np.median(dNmeds)
    waE = np.median(dEmeds)
    waN_e = [
        [np.std(dNmeds) / np.sqrt(iters)],
        [np.std(dNmeds) / np.sqrt(iters)],
    ]
    waE_e = [
        [np.std(dEmeds) / np.sqrt(iters)],
        [np.std(dEmeds) / np.sqrt(iters)],
    ]

    return waN, waN_e, waE, waE_e


config = pd.read_json('config.json')
wmean = eval(config.output.wmean)
# wmean = False
if wmean:
    calcweight = calcweight2

df = pd.read_csv('output/resultsTable.csv')
# df = pd.concat(
#    [df, pd.read_csv('output/topResultsTable.csv')], ignore_index=True
# )
# df = df[df.pmra == 0].reset_index(drop=True)
# df = df[df.pmra != 0].reset_index(drop=True)

df['dN'] = -df.dN
df['dE'] = -df.dE

maxstarerr_N = df.loc[df.pmra != 0, 'dN_e'].max()
maxstarerr_E = df.loc[df.pmra != 0, 'dE_e'].max()
multi = 1
# df = df[(df.dN_e < maxstarerr_N * multi) & (df.dE_e < maxstarerr_E * multi)]

# df.loc[df.pmra == 0, 'dN'] -= 1.429674e-01
## df.loc[df.pmra == 0, 'dN'] -= 1.455307e-01
# df.loc[df.pmra == 0, 'dE'] -= -6.227710e-01
## df.loc[df.pmra == 0, 'dE'] -= -6.772535e-01

###ang = -np.pi * 142.4484564279773 / 180
###ang = np.pi * (-118.828) / 180
###SA = np.sin(ang)
###CA = np.cos(ang)
###dx, dx_e = -df.dE.values, df.dE_e.values
###dy, dy_e = df.dN.values, df.dN_e.values
###dN = -(-SA * dx - CA * dy)
###dE = -(CA * dx - SA * dy)
#### dN = dx * CA - dy * SA
#### dE = dx * SA + dy * CA
###dN_e = np.sqrt(
###    dx_e * dy_e * (SA**2 * (dx_e / dy_e) + CA**2 * (dy_e / dx_e))
###)
###dE_e = np.sqrt(
###    dx_e * dy_e * (CA**2 * (dx_e / dy_e) + SA**2 * (dy_e / dx_e))
###)
# df['dN'] = dN
# df['dE'] = dE
# df['dN_e'] = dN_e
# df['dE_e'] = dE_e

rapel, rapele = 1.030, 0.0685
decpel, decpele = 0.889, 0.0745
# rapel, rapele = 0.032, 0.017
# decpel, decpele = 0.033, 0.018
###dx, dx_e = -rapel, rapele
###dy, dy_e = decpel, decpele
###dN = -(-SA * dx - CA * dy)
###dE = -(+CA * dx - SA * dy)
#### dN = dx * CA - dy * SA
#### dE = dx * SA + dy * CA
###dN_e = np.sqrt(
###    dx_e * dy_e * (SA**2 * (dx_e / dy_e) + CA**2 * (dy_e / dx_e))
###)
###dE_e = np.sqrt(
###    dx_e * dy_e * (CA**2 * (dx_e / dy_e) + SA**2 * (dy_e / dx_e))
###)
# rapel, rapele = -dE, dE_e
# decpel, decpele = dN, dN_e

# rapel = rapel * np.cos(ang) - decpel * np.sin(ang)
# decpel = rapel * np.sin(ang) + decpel * np.cos(ang)


# df = df[df.pmra == 0]
# df.loc[(df.pmra == 0) & (df.pmdec == 0), 'dE'] -= -6.778269e-1
# df.loc[(df.pmra == 0) & (df.pmdec == 0), 'dN'] -= 1.432394e-1
####df.loc[df.pmra == 0, 'dE'] -= -6.282814e-1
####df.loc[df.pmra == 0, 'dN'] -= 1.897436e-1

# df = df[np.abs(df.dN) < 10].reset_index(drop=True)
# df = df[np.abs(df.dE) < 10].reset_index(drop=True)

df = df.sort_values(by='m_e1', ascending=False).reset_index(drop=True)
df['toterr'] = np.sqrt(df.dN_e**2 + df.dE_e**2)
df = df.sort_values(by='toterr', ascending=True, ignore_index=True)

df = df.drop_duplicates(subset='des', keep='first', ignore_index=True)
df = df[df.des.notna()]

df = df.sort_values(by='m_e1', ascending=False).reset_index(drop=True)
# df = df.sort_values(by='dE_e', ascending=False).reset_index(drop=True)
# df = df.head(len(df) - 1).reset_index(drop=True)
dffull = df.copy()

l0, l = len(df), 0
waN, waN_e, waE, waE_e = calcweight(df.copy())
# waN = decpel
# waE = rapel
if eval(config.output.sigmaclip):
    dfstar = df[df.pmra != 0].reset_index(drop=True)
    waN, waN_e, waE, waE_e = calcweight(dfstar.copy())
    for i in range(10):
        l0 = len(df)
        # df = df[(np.abs(df.dN - waN) < config.output.sigmaval * df.dN_e)]
        # df = df[(np.abs(df.dE - waE) < config.output.sigmaval * df.dE_e)]
        q = np.sqrt(
            ((df.dN - waN) / df.dN_e) ** 2 + ((df.dE - waE) / df.dE_e) ** 2
        )
        df = df[q < config.output.sigmaval * 1.52]
        # df = df[(np.abs(df.dN - waN) < 1.0) & (np.abs(df.dE - waE) < 1.0)]
        # waTOT = np.sqrt(waN**2 + waE**2)
        # df = df[np.sqrt((df.dN - waN) ** 2 + (df.dE - waE) ** 2) < waTOT * 1.0]
        waN, waN_e, waE, waE_e = calcweight(df.copy())
        l = len(df)
        if l == l0:
            break
    # df = df[(np.abs(df.dN - waN) < config.output.sigmaval * df.dN_e)]
    # df = df[(np.abs(df.dE - waE) < config.output.sigmaval * df.dE_e)]
    # q = np.sqrt(
    #    ((df.dN - waN) / df.dN_e) ** 2 + ((df.dE - waE) / df.dE_e) ** 2
    # )
    # df = df[q < config.output.sigmaval * 1.52]
    waN, waN_e, waE, waE_e = calcweight(df.copy())
keep = df.index.values
print(df.des.values)

print('mu_alpha: ', waE, waE_e)
print('mu_delta: ', waN, waN_e)

# fig = plt.figure(figsize=(15, 14))
fig = plt.figure(figsize=(21, 21))
fig = plt.figure(figsize=(21, 30))
# fig = plt.figure(figsize=(15, 7))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.grid(alpha=0.5), ax2.grid(alpha=0.5)
# fig.suptitle(config.output.targetname, fontsize=25, y=0.93)
# fig.suptitle('Dr2 all chips at once, median', fontsize=25, y=0.93)
fig.suptitle(config.output.targetname, fontsize=35, y=0.93)
fig.subplots_adjust(wspace=0.0)

j = 0
for i in range(1, len(dffull) + 1):
    j += 0.25
    dN, dNe = dffull.loc[i - 1][['dN', 'dN_e']].values
    dE, dEe = dffull.loc[i - 1][['dE', 'dE_e']].values
    # if dffull.loc[i-1,'q_e2'] == 0:
    if dffull.loc[i - 1, 'pmra'] == 0:
        mkr = 'x'
    else:
        mkr = 'o'
    if i - 1 in keep:
        ax1.errorbar(dE, j, xerr=dEe, fmt=mkr, color=rotorange)
        ax2.errorbar(dN, j, xerr=dNe, fmt=mkr, color=rotorange)
    else:
        ax1.errorbar(dE, j, xerr=dEe, fmt=mkr, color=rotorange, alpha=0.2)
        ax2.errorbar(dN, j, xerr=dNe, fmt=mkr, color=rotorange, alpha=0.2)

i = j + 0.25
s = 'Individual Reflex Gaia'
s = 'Individual Gaia\nstellar reflex PMs'
s = 'Individual Gaia star (o) and\nbackground galaxy (x) reflex PMs'
fs = 14
fs2 = 17
ax2.text(
    1.0 + waN, i, s, color=rotorange, ha='right', va='center', fontsize=fs
)
i += 1 - 0.25 + 0.25
i
if wmean:
    s = 'Weighted Mean' + f' ({len(df)} stars)'
else:
    s = 'Weighted Median'
ax1.errorbar(waE, i, xerr=waE_e, fmt='|', color=jeffblue, alpha=0.7)
ax2.errorbar(waN, i, xerr=waN_e, fmt='|', color=jeffblue, alpha=0.7)
ax2.text(
    1.0 + waN,
    i,
    s,
    color=jeffblue,
    alpha=0.7,
    ha='right',
    va='center',
    fontsize=fs,
)
if wmean:
    ax1.text(
        -1.0 + waE,
        i,
        f'{waE:.3f}' + '$\\pm$' + f'{np.mean(waE_e):.3f}',
        color=jeffblue,
        alpha=0.7,
        ha='left',
        va='center',
        fontsize=fs2,
    )
    ax2.text(
        -1.0 + waN,
        i,
        f'{waN:.3f}' + '$\\pm$' + f'{np.mean(waN_e):.3f}',
        color=jeffblue,
        alpha=0.7,
        ha='left',
        va='center',
        fontsize=fs2,
    )
else:
    ax1.text(
        -1.0 + waE,
        i,
        f'{waE:.3f}' + r'$\pm^{%.4f}_{%.4f}$' % (waE_e[1][0], waE_e[0][0]),
        color=jeffblue,
        alpha=0.7,
        ha='left',
        va='center',
        fontsize=fs2,
    )
    ax2.text(
        -1.0 + waN,
        i,
        f'{waN:.3f}' + r'$\pm^{%.4f}_{%.4f}$' % (waN_e[1][0], waN_e[0][0]),
        color=jeffblue,
        alpha=0.7,
        ha='left',
        va='center',
        fontsize=fs2,
    )

rapel, rapele = 1.030, 0.0685
decpel, decpele = 0.889, 0.0745
i += 0.25 + 0.25
ax1.errorbar(rapel, i, xerr=rapele, fmt='|', color=jeffblue, alpha=0.7)
ax2.errorbar(decpel, i, xerr=decpele, fmt='|', color=jeffblue, alpha=0.7)
ax2.text(
    1.0 + waN,
    i,
    'Pace, Erkal, & Li 2020',
    color=jeffblue,
    alpha=0.7,
    ha='right',
    va='center',
    fontsize=fs,
)

ax1.text(
    -1.0 + waE,
    i,
    f'{rapel:.3f}' + '$\\pm$' + f'{rapele:.3f}',
    color=jeffblue,
    alpha=0.7,
    ha='left',
    va='center',
    fontsize=fs2,
)
ax2.text(
    -1.0 + waN,
    i,
    f'{decpel:.3f}' + '$\\pm$' + f'{decpele:.3f}',
    color=jeffblue,
    alpha=0.7,
    ha='left',
    va='center',
    fontsize=fs2,
)

rapel, rapele = 1.12, 0.09
decpel, decpele = 0.91, 0.10
i += 0.25 + 0.25
ax1.errorbar(rapel, i, xerr=rapele, fmt='|', color=jeffblue, alpha=0.7)
ax2.errorbar(decpel, i, xerr=decpele, fmt='|', color=jeffblue, alpha=0.7)
ax2.text(
    1.0 + waN,
    i,
    'Battaglia+ 2020',
    color=jeffblue,
    alpha=0.7,
    ha='right',
    va='center',
    fontsize=fs,
)

ax1.text(
    -1.0 + waE,
    i,
    f'{rapel:.3f}' + '$\\pm$' + f'{rapele:.3f}',
    color=jeffblue,
    alpha=0.7,
    ha='left',
    va='center',
    fontsize=fs2,
)
ax2.text(
    -1.0 + waN,
    i,
    f'{decpel:.3f}' + '$\\pm$' + f'{decpele:.3f}',
    color=jeffblue,
    alpha=0.7,
    ha='left',
    va='center',
    fontsize=fs2,
)

rapel, rapele = 0.85, 0.04
decpel, decpele = 0.82, 0.03
i += 0.25 + 0.25
ax1.errorbar(rapel, i, xerr=rapele, fmt='|', color=jeffblue, alpha=0.7)
ax2.errorbar(decpel, i, xerr=decpele, fmt='|', color=jeffblue, alpha=0.7)
ax2.text(
    1.0 + waN,
    i,
    'Kevin BP3M',
    color=jeffblue,
    alpha=0.7,
    ha='right',
    va='center',
    fontsize=fs,
)

ax1.text(
    -1.0 + waE,
    i,
    f'{rapel:.3f}' + '$\\pm$' + f'{rapele:.3f}',
    color=jeffblue,
    alpha=0.7,
    ha='left',
    va='center',
    fontsize=fs2,
)
ax2.text(
    -1.0 + waN,
    i,
    f'{decpel:.3f}' + '$\\pm$' + f'{decpele:.3f}',
    color=jeffblue,
    alpha=0.7,
    ha='left',
    va='center',
    fontsize=fs2,
)

ax1.axvline(0, ls='-', lw=1, color='k', alpha=0.5, zorder=0.005)
ax1.axvline(waE, ls='--', lw=1, color=jeffblue, alpha=0.8, zorder=0.005)
ax2.axvline(0, ls='-', lw=1, color='k', alpha=0.5, zorder=0.005)
ax2.axvline(waN, ls='--', lw=1, color=jeffblue, alpha=0.8, zorder=0.005)

ax1.set_xlabel(r'$\mu_{\alpha}^*$ [mas/yr]', fontsize=20)
ax2.set_xlabel(r'$\mu_{\delta}$ [mas/yr]', fontsize=20)
ax1.set_yticks([]), ax2.set_yticks([])
# ax1.set_xlim([-1.1,1.1]), ax2.set_xlim([-1.1,1.1])
ax1.set_xlim([waE - 1.1, waE + 1.1]), ax2.set_xlim([waN - 1.1, waN + 1.1])

fig.savefig(
    f'{config.output.targetname.replace(" ","_")}_summary.png',
    # f'dr2_allchip.png',
    dpi=400,
    bbox_inches='tight',
)
