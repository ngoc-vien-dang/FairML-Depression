"""
File name: plot.py
Author: ngocviendang
Date created: October 26, 2022

This file contains functions for plotting figures.
"""
import numpy as np
import matplotlib.pyplot as plt

color_cols = {'0–20':'#2ca02c', 
'20–40':'#ff7f0e',
 '40–60': '#9467bd',
 '60–80':'#bcbd22', 
 '>80':'#7f7f7f',
 'missing':'#1f77b4',
 'l0':'#1f77b4', 
 'l1':'#2ca02c',
 'l2': '#ff7f0e',
 'l3':'#bcbd22', 
 'l4':'#d62728',
 'l5': '#9467bd',
 'l6': '#8c564b',
 'high': '#ff7f0e',
 'low' : '#2ca02c',
 'mexican':'#7f7f7f', 
 'black':'#ff7f0e',
 'hispanic': '#bcbd22',
 'asian': '#8c564b',
 'white':'#9467bd', 
 'other':'#2ca02c',
 'female':'#d62728', 
 'male':'#1f77b4',
 'presence':'#d62728', 
 'absence':'#1f77b4',
 'french':'#1f77b4',
 'foreigner':'#d62728'}
 
def plot_sim2(tukey, xlabel='True positive rate', xticks=None,n_g=1,m_g=None):
    # Reference https://github.com/irenetrampoline/mimic-disparities
    tukey._simultaneous_ci()
    means = tukey._multicomp.groupstats.groupmean
    minrange = [means[i] - tukey.halfwidths[i] for i in range(len(means))]
    maxrange = [means[i] + tukey.halfwidths[i] for i in range(len(means))]
    fig, ax = plt.subplots()
    ax.margins(0.05)
    x_vals = range(len(means))
    protected = tukey.groupsunique
    if n_g == 2:
      protected = tukey.groupsunique
      protected = protected.tolist()
      protected.remove(m_g)
    else:
      protected = tukey.groupsunique
    errN = dict()
    meansN = dict()
    mksN = dict()
    colorsN = dict()
    for pix, p in enumerate(protected):
        errN[p] = tukey.halfwidths[pix]
        meansN[p] = means[pix]
        colorsN[p] = color_cols[p.lower()]
        mksN[p] = 's'
    sort_prot = sorted(protected)
    for pix2, p in enumerate(sort_prot):
        ax.errorbar(meansN[p], len(protected) - 1 -x_vals[pix2], marker='s', markersize=20, xerr=errN[p],\
        lw=2, capsize=10, capthick=5, color=colorsN[p], elinewidth=5)
    ax.xaxis.grid()
    r = np.max(maxrange) - np.min(minrange)
    if n_g == 2:
      ax.set_ylim([-1, tukey._multicomp.ngroups-1])
    else:
      ax.set_ylim([-1, tukey._multicomp.ngroups])
    ax.set_xlim([np.min(minrange) - r / 10., np.max(maxrange) + r / 10.])
    
    def clean_p(x):
        spltx = x.split('_')
        if len(spltx) == 1:
            return x
        else:
            return spltx[1]
        
    sort_prot2 = sorted([clean_p(i) for i in protected], reverse=True)
    ax.set_yticklabels(sort_prot2, fontsize=16)
    ax.set_yticks(np.arange(0, len(means)))
    plt.setp(ax.get_xticklabels(), fontsize=12)
    if xticks != None:
        plt.xticks(xticks)
    plt.tight_layout()
    plt.xlabel(xlabel, fontsize=16)

def plot_tradeoff(orig,sup,rw,dir,cpp,st,title,imgpath):
    plt.errorbar(orig['EOD'].mean(), orig['bacc_test'].mean(), xerr=orig['EOD'].std(),\
             yerr=orig['bacc_test'].std(), marker='s',color = 'gray',ms=7, 
            ecolor = 'gray') 
    plt.errorbar(sup['EOD'].mean(), sup['bacc_test'].mean(), xerr=orig['EOD'].std(),\
             yerr=sup['bacc_test'].std(), marker='s',color = '#d62728',ms=7, 
            ecolor = '#d62728') 
    plt.errorbar(rw['EOD'].mean(), rw['bacc_test'].mean(), xerr=rw['EOD'].std(),\
             yerr=rw['bacc_test'].std(), marker='s',color = '#ff7f0e', ms=7,
            ecolor = '#ff7f0e') 
    plt.errorbar(dir['EOD'].mean(), dir['bacc_test'].mean(), xerr=dir['EOD'].std(),\
             yerr=dir['bacc_test'].std(),marker='s',color = '#9467bd', ms=7,
            ecolor = '#9467bd') 
    plt.errorbar(cpp['EOD'].mean(), cpp['bacc_test'].mean(), xerr=cpp['EOD'].std(),\
             yerr=cpp['bacc_test'].std(), marker='s',color = '#1f77b4', ms=7,
            ecolor = '#1f77b4') 
    plt.errorbar(st['EOD'].mean(), st['bacc_test'].mean(), xerr=st['EOD'].std(),\
             yerr=st['bacc_test'].std(), marker='s',color = '#2ca02c', ms=7,
            ecolor = '#2ca02c') 
    plt.ylabel('Balanced accuracy', fontsize=16)
    plt.xlabel('Equal opportunity difference', fontsize=16)
    #plt.legend(legend,loc='best',fancybox=True)
    plt.title(title,fontsize=16)
    plt.savefig(imgpath, dpi=1200,bbox_inches='tight')