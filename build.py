
# =============================================================================
# Packages
# =============================================================================
import os
import numpy as np
import pandas as pd
import csv
from scipy import linalg as ls
import functions as core


import matplotlib.pyplot as plt
import matplotlib as mpl
from fractions import Fraction
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, LogLocator, NullFormatter,MultipleLocator, AutoMinorLocator, FuncFormatter
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

import scipy.linalg as la
import networkx as nx 
from functools import reduce
from tqdm import tqdm

# ----------------------------- Matplotlib style ------------------------------

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

mpl.rcParams['font.family'] = 'serif' #'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 1


def style_axes(ax):

    ax.tick_params(axis='both', which='both', direction='in', top=False, right=False)

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))   # 4 minors between majors

    ax.yaxis.set_minor_locator(AutoMinorLocator(5))   # 4 minors between majors


def indian_flag_cmap(name="indian_flag", with_chakra=False):
    saffron = "#FF7A00"
    white   = "#FFFFFF"
    green   = "#138808"
    chakra  = "#000088"   # Ashoka Chakra navy (optional)

    if not with_chakra:
        stops = [(0.00, saffron), (0.50, white), (1.00, green)]
    else:
        # narrow navy band at the midpoint
        stops = [(0.00, saffron), (0.45, white), (0.50, chakra),
                 (0.55, white), (1.00, green)]

    return LinearSegmentedColormap.from_list(name, stops)

def make_log_label_formatter(ticks_to_label):
    ticks_to_label = np.array(ticks_to_label, dtype=float)

    def _formatter(val, pos):
        if np.any(np.isclose(val, ticks_to_label)):
            return r"$10^{%d}$" % np.log10(val)
        return ""  # no label for other ticks

    return ticker.FuncFormatter(_formatter)




# =============================================================================
# Figure 2a — Generate CSV
# =============================================================================
def generate_figure2a_csv(out_csv, save = True): 


    r_vals = np.linspace(-1.0, 1.0, 1201)
    s_vals = np.linspace(-1.0, 1.0, 1201)

    R, S = np.meshgrid(r_vals, s_vals)
    Alpha = np.empty_like(R)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            Alpha[i, j] = core.alpha_of_rs_scalar(R[i, j], S[i, j])

    
    # Build data for CSV
    x_flat = R.ravel(order='C')
    y_flat = S.ravel(order ='C')
    z_flat = Alpha.ravel(order = 'C')


    data = np.column_stack([x_flat, y_flat,z_flat])

    # Header row 
    header = "rbar, s, alpha"

    # Save CSV
    if save:
        np.savetxt(out_csv, data, delimiter=",", header=header, comments="", fmt="%.16g")
    else:
        return data

# =============================================================================
# Figure 2a — Plot from CSV
# =============================================================================
def plot_figure2a_from_csv(
    csv_path,
    out_fig,
    dpi=600, save = True
):

    df = pd.read_csv(csv_path, comment="#")
    x_flat = df.iloc[:, 0].values  # alpha list
    y_flat = df.iloc[:, 1].values  # time
    z_flat = df.iloc[:, 2].values  # S(t)

    x_vals = np.unique(x_flat)   # unique rbar
    y_vals = np.unique(y_flat)   # unique s

    R, S = np.meshgrid(x_vals, y_vals)
    Alpha = z_flat.reshape(len(y_vals), len(x_vals))

    #PRL width and height
    width_mm = 59.94 
    width_in = width_mm / 25.4
    panel_h = 2.1238582677165354
    fig, ax = plt.subplots(figsize=(width_in,panel_h),dpi =200)
    
    style_axes(ax)
    for sp in ax.spines.values():
        sp.set_zorder(50)

    # plotting the special critical star (chakra) at (0, 0.5)
    chakra  = "#000088" 
    ax.plot(0.0, 0.5, marker='*',color= chakra, markersize=8, zorder=8)
    ax.text(-0.45, 0.47, r'$(0,\,\frac{1}{2})$', color=chakra, fontsize=8)

    #plot the points for Fig.2b
    ax.plot(-0.5, -0.5, marker='o', color= chakra, markersize=6,markeredgecolor ='k', zorder=8)
    ax.plot(-0.9, -1.0, marker='s', color='crimson', markersize=7,markeredgecolor ='k', zorder=8)
    ax.plot(-0.45, -0.1, marker='d', color='gold', markersize=8,markeredgecolor ='k', zorder=8)
    ax.plot(-0.7, -0.6, marker='v', color='violet', markersize=8,markeredgecolor ='k', zorder=8)

    #labelling markers for Fig. 2b
    label_positions = [

        (-0.1, 0.7),   # A1
        (-0.1,0.15),    #A2
        (0.7,0.8),    # B 
        (0.4, -0.5),# C
        (-0.8, 0)     # D
    ]


    letters = ["A1", "A2", "B", "C", "D"]

    for letter, (lr, ls) in zip(letters, label_positions):

        label_text = f"({letter})" #+ a_val

        # place text with a white box for readability
        ax.text(lr, ls, label_text, fontsize=8, ha='left', va='center',
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="0.8", alpha=0.85))

    #contour plot
    Alpha_sneg = np.ma.masked_where(S >= 0, Alpha)  # mask out s>=0
    cp = ax.contourf(R, S, Alpha_sneg, levels=np.arange(0.0,1.01,0.01),cmap = indian_flag_cmap(with_chakra=False))
    cbar = plt.colorbar(cp,ax=ax, orientation="vertical", fraction=0.08,shrink = 0.85)
    
    # line across the colorbar at alpha = 0.5
    cbar.ax.axhline(0.5, color=chakra, lw=1.6)
    ticks = [0,  0.25, 0.5, 0.75, 1.0]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([0,r'$\frac{1}{4}$', r'$\frac{1}{2}$',r'$\frac{3}{4}$',1])
    cbar.ax.text(0.5,1.05, r'$\alpha$', transform=cbar.ax.transAxes,ha='center', va='center')

    #plot the critical line at rbar = 0
    s_crit = np.arange(0.0, 1.01,0.01)
    r_crit = np.zeros(len(s_crit))


    alpha = np.array([core.alpha_of_rs_scalar(0.0,s_crit[i]) for i in range(len(s_crit))])

    # Create segments for LineCollection
    points = np.array([r_crit, s_crit]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create the line collection
    norm = Normalize(vmin=0.0, vmax=1)


    lc = LineCollection(
        segments,
        cmap=indian_flag_cmap(),
        norm=norm,
        array=alpha,
        linewidth=3
    )

    line = ax.add_collection(lc)

    ax.contour(R, S, Alpha, levels=[0.5], colors=chakra, linewidths=1.6, linestyles='--')

    ax.set_xlabel(r'$\bar{r}$',labelpad =1,size = 8 )
    ax.set_ylabel(r'$s$',labelpad =-6,size = 8)

    # draw thin axes guide lines separating different regions
    ax.axhline(0.0,xmin = 0.5, ls=":", color="black", linewidth=1)
    ax.axvline(0.0,ymin = 0.5, ls=":", color="black", linewidth=1)
    ax.plot(x_vals[x_vals<0],y_vals[y_vals<0], ls=":", color="black", linewidth=1)


    ax.text(-0.5, -0.4, r"$s=\bar{r}$", fontsize=8, color="k", rotation=45)
    ax.text(-0.66, -0.95, r"$s= 2\bar{r} +\frac{1}{2}$", fontsize=8, color=chakra, rotation=65)
    ax.text(-0.42, -0.95, r"$s= 2\bar{r}$", fontsize=8, color = '#138808', rotation=65)



    plt.yticks(fontsize =8)
    plt.xticks(fontsize =8)


    if save:
        plt.savefig(out_fig, dpi=dpi, bbox_inches="tight")
        plt.show()
    else:
        plt.show()

# =============================================================================
# Figure 2b — Generate CSV
# =============================================================================
def generate_figure2b_csv(
    out_csv,dt = 0.01,threshold = 0.001, save = True
):  

    #Time in logspace
    Tstep = np.logspace(1,15.6,2500,dtype=int)

    #List of system size
    Nlist = np.logspace(1,5,50,dtype=int) 
    
    #rbar-s list
    rbars_list = [
        (-0.9,-1.0),
        (-0.45,-0.1),
        (-0.7,-0.6),
        (-0.5,-0.5)
    ]

    results = []

    i=0
    for (rbar, s) in rbars_list:

        results.append([])
        for N,Nval in enumerate(Nlist):

            gamma = Nval**(-(rbar+1))
            kappa = Nval**(-s)

            s_nr = core.surv_prob_theory_total_logspace(Tstep, dt, Nval, gamma, kappa)
            results[i].append(core.find_transition_point(s_nr,Tstep, dt,threshold))
        i = i + 1


    # --- build data matrix: Nlist + results columns ---
    cols = [np.asarray(results[j], dtype=float) for j in range(len(results))]

    data = np.column_stack([Nlist] + cols)

    # Header row (no leading '#', clean commas)
    header = (
        "N,"
        "tau (rbar=-0.9+s=-1.0),"
        "tau (rbar=-0.45+s=-0.1),"
        "tau (rbar=-0.7+s=-0.6),"
        "tau (rbar=-0.5+s=-0.5),"        
    )

    # Save / return
    if save:
        np.savetxt(out_csv, data, delimiter=",", header=header, comments="", fmt="%.16g")
    else:
        return data

# =============================================================================
# Figure 2b — Plot from CSV
# =============================================================================

def _y_at_xcut_loglog(x, y, xcut):
    x = np.asarray(x); y = np.asarray(y)
    order = np.argsort(x)
    x = x[order]; y = y[order]

    # clip to valid range
    if xcut <= x[0]:  return y[0]
    if xcut >= x[-1]: return y[-1]

    lx = np.log10(x)
    ly = np.log10(y)
    ycut_log = np.interp(np.log10(xcut), lx, ly)
    return 10**ycut_log

def plot_faded_after(ax, x, y, xcut, marker, color,
                     ms=4, edgecolor='k', edgewidth=0.7,
                     alpha_after=0.5, annotate=True, text=None):
    """Plot (x,y) with full alpha up to xcut, and faded after xcut."""
    x = np.asarray(x)
    y = np.asarray(y)

    m1 = x <= xcut
    m2 = x > xcut

    # main part
    ax.plot(x[m1], y[m1], marker,
            markersize=ms, color=color,
            markeredgecolor=edgecolor, markeredgewidth=edgewidth,
            alpha=1.0)

    # faded tail
    if np.any(m2):
        ax.plot(x[m2], y[m2], marker,
                markersize=ms, color=color,
                markeredgecolor=edgecolor, markeredgewidth=edgewidth,
                alpha=alpha_after)

    # reference line
    if annotate and (x.min() <= xcut <= x.max()):
        ycut = _y_at_xcut_loglog(x, y, xcut)

        # from bottom of plot (in log scale, this is the "x-axis line" visually)
        ymin = ax.get_ylim()[0]
        ymax = ax.get_ylim()[1]
        ax.vlines(xcut, ymin=ymin, ymax=ycut, color=color, lw=0.8, ls=':', alpha=0.9)



def plot_line_faded_after(ax, x, y, xcut, color, ls='--', lw=1,
                          alpha_before=0.8, alpha_after=0.25):
    x = np.asarray(x)
    y = np.asarray(y)

    m1 = x <= xcut
    m2 = x > xcut

    ax.plot(x[m1], y[m1], color=color, ls=ls, lw=lw, alpha=alpha_before)
    if np.any(m2):
        ax.plot(x[m2], y[m2], color=color, ls=ls, lw=lw, alpha=alpha_after)
 


def plot_figure2b_from_csv(
    csv_path,
    out_fig,
    dpi=600, save = True
):

    df = pd.read_csv(csv_path, comment="#")
    Nlist = df.iloc[:, 0].values
    results_0 = df.iloc[:, 1].values  # rbar=-0.9,s=-1.0
    results_1 = df.iloc[:, 2].values  # rbar=-0.45,s=-0.1
    results_2 = df.iloc[:, 3].values  # rbar= -0.7,s=-0.6
    results_3 = df.iloc[:, 4].values  # rbar= -0.5,s=-0.5

    dt =0.01
    N1 = (dt)**(1.0/(-1.0))
    N2 = (dt)**(1.0/(-0.45))
    N3 = (dt)**(1.0/(-0.7))
    N4 = (dt)**(1.0/(-0.5))

    #PRL width and height
    width_mm = 28.77
    width_in = width_mm / 25.4
    panel_h = 2.1238582677165354
    fig, ax2 = plt.subplots(figsize=(width_in,panel_h),dpi =200)
    style_axes(ax2)

   
    chakra  = "#000088" 
    # build the theory curves
    y_th0 = 4.4 * Nlist**0.2
    y_th1 = 3.0 * Nlist**0.9
    y_th2 = 4.0 * Nlist**0.4
    y_th3 = 6.0 * Nlist**0.5

    # plot theory curves faded after their corresponding Ni
    plot_line_faded_after(ax2, Nlist, y_th0, N1, color='crimson', ls='--', lw=1)
    plot_line_faded_after(ax2, Nlist, y_th1, N2, color='gold',    ls='--', lw=1)
    plot_line_faded_after(ax2, Nlist, y_th2, N3, color='violet',  ls='--', lw=1)
    plot_line_faded_after(ax2, Nlist, y_th3, N4, color=chakra,    ls='--', lw=1)

    # Plotting every 6 points (same as your code)
    x6 = Nlist[::4]

    plot_faded_after(ax2, x6, results_0[::4], N1, 's', 'crimson', alpha_after=0.25)
    plot_faded_after(ax2, x6, results_1[::4], N2, 'd', 'gold',    alpha_after=0.25)
    plot_faded_after(ax2, x6, results_2[::4], N3, 'v', 'violet',  alpha_after=0.25)
    plot_faded_after(ax2, x6, results_3[::4], N4, 'o', chakra,    alpha_after=0.25)

    
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_ylim(bottom=1)
    ax2.set_xlabel(r'$N$',labelpad = 0,size =8)
    ax2.set_ylabel(r'$\tau$',labelpad = 0,size = 10)

    nticks = 9
    min_loc = ticker.LogLocator(subs='all', numticks=nticks)
    ax2.xaxis.set_minor_locator(min_loc)
    ax2.yaxis.set_minor_locator(min_loc)


    ax2.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax2.yaxis.set_minor_formatter(ticker.NullFormatter())

    ax2.set_xticks([10, 1000, 100000])

    ax2.text(14, 2, r"$N_1^*$", fontsize=7, color='crimson', rotation=0)
    ax2.text(1.2e2, 2, r"$N_2^*$", fontsize=7, color='violet', rotation=0)
    ax2.text(1.5e3, 2, r"$N_3^*$", fontsize=7, color=chakra, rotation=0)
    ax2.text(3e4, 2, r"$N_4^*$", fontsize=7, color='gold', rotation=0)

    if save:
        plt.savefig(out_fig, dpi=dpi, bbox_inches="tight")
        plt.show()
    else:
        plt.show()






# =============================================================================
# Figure 3a — Generate CSV
# =============================================================================
def generate_figure3a_csv(
    out_csv, save = True
):  

    r_vals = np.linspace(-1.0, 1.0, 1201)
    s_vals = np.linspace(-1.0, 1.0, 1201)

    R, S = np.meshgrid(r_vals, s_vals)
    Alpha = np.empty_like(R)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            Alpha[i, j] = core.alpha_of_rs_scalar_reset(R[i, j], S[i, j])

    
    # Build data for CSV
    x_flat = R.ravel(order='C')
    y_flat = S.ravel(order ='C')
    z_flat = Alpha.ravel(order = 'C')


    data = np.column_stack([x_flat, y_flat,z_flat])

    # Header row 
    header = "rbar, s, alpha"

    # Save CSV
    if save:
        np.savetxt(out_csv, data, delimiter=",", header=header, comments="", fmt="%.16g")
    else:
        return data



# =============================================================================
# Figure 3a — Plot from CSV
# =============================================================================
def plot_figure3a_from_csv(
    csv_path,
    out_fig,
    dpi=600, save = True
):

    df = pd.read_csv(csv_path, comment="#")
    x_flat = df.iloc[:, 0].values  # alpha list
    y_flat = df.iloc[:, 1].values  # time
    z_flat = df.iloc[:, 2].values  # S(t)

    x_vals = np.unique(x_flat)   # unique rbar
    y_vals = np.unique(y_flat)   # unique s

    R, S = np.meshgrid(x_vals, y_vals)
    Alpha = z_flat.reshape(len(y_vals), len(x_vals))

    #PRL width and height
    width_mm = 59.94 
    width_in = width_mm / 25.4
    panel_h = 2.1238582677165354
    fig, ax = plt.subplots(figsize=(width_in,panel_h),dpi =200)
    
    style_axes(ax)
    for sp in ax.spines.values():
        sp.set_zorder(50)

    # plotting the special critical star (chakra) at (0, 0.5)
    chakra  = "#000088" 
    ax.plot(0.0, 0.5, marker='*',color= chakra, markersize=8, zorder=8)
    ax.text(-0.45, 0.47, r'$(0,\,\frac{1}{2})$', color=chakra, fontsize=8)

    #labelling markers for Fig. 2b
    label_positions = [

        (-0.15, 0.75),   # A1
        (-0.15,0.25),    #A2
        (0.7,0.8),    # B 
        (0.4, -0.5),# C
        (-0.8, 0)     # D
    ]

    letters = ["A1", "A2", "B", "C", "D"]
    for letter, (lr, ls) in zip(letters, label_positions):

        label_text = f"({letter})" #+ a_val

        # place text with a white box for readability
        ax.text(lr, ls, label_text, fontsize=8, ha='left', va='center',
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="0.8", alpha=0.85))

    #contour plot
    Alpha_sneg = np.ma.masked_where(S >= 0, Alpha)  # mask out s>=0
    cp = ax.contourf(R, S, Alpha_sneg, levels=np.arange(0.0,1.01,0.01),cmap = indian_flag_cmap(with_chakra=False))
    cbar = plt.colorbar(cp,ax=ax, orientation="vertical", fraction=0.08,shrink = 0.85)
    
    # line across the colorbar at alpha = 0.5
    cbar.ax.axhline(0.5, color=chakra, lw=1.6)
    ticks = [0,  0.25, 0.5, 0.75, 1.0]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([0,r'$\frac{1}{4}$', r'$\frac{1}{2}$',r'$\frac{3}{4}$',1])
    cbar.ax.text(0.5,1.05, r'$\alpha$', transform=cbar.ax.transAxes,ha='center', va='center')

    #plot the critical line at rbar = 0
    s_crit = np.arange(0.0, 1.01,0.01)
    r_crit = np.zeros(len(s_crit))


    alpha = np.array([core.alpha_of_rs_scalar(0.0,s_crit[i]) for i in range(len(s_crit))])

    # Create segments for LineCollection
    points = np.array([r_crit, s_crit]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create the line collection
    norm = Normalize(vmin=0.0, vmax=1)


    lc = LineCollection(
        segments,
        cmap=indian_flag_cmap(),
        norm=norm,
        array=alpha,
        linewidth=3
    )

    line = ax.add_collection(lc)

    ax.contour(R, S, Alpha, levels=[0.5], colors=chakra, linewidths=1.6, linestyles='--')

    ax.set_xlabel(r'$\bar{r}$',labelpad =1,size = 8 )
    ax.set_ylabel(r'$s$',labelpad =-6,size = 8)

    # draw thin axes guide lines separating different regions
    ax.axhline(0.0,xmin = 0.5, ls=":", color="black", linewidth=1)
    ax.axvline(0.0,ymin = 0.5, ls=":", color="black", linewidth=1)
    ax.plot(x_vals[x_vals<0],y_vals[y_vals<0], ls=":", color="black", linewidth=1)


    ax.text(-0.5, -0.4, r"$s=\bar{r}$", fontsize=8, color="k", rotation=45)

    #plot the points for Fig.3b
    ax.plot(0.0, -0.9, marker='s', color= 'white', markersize=6,markeredgecolor ='k', zorder=8)

    plt.yticks(fontsize =8)
    plt.xticks(fontsize =8)


    if save:
        plt.savefig(out_fig, dpi=dpi, bbox_inches="tight")
        plt.show()
    else:
        plt.show()





# =============================================================================
# Figure 3b — Generate CSV
# =============================================================================
def generate_figure3b_csv(
    out_csv,dt = 0.01,threshold = 0.001, save = True
):  

    #Time in logspace
    Tstep = np.logspace(1,15.6,2500,dtype=int)

    #List of system size
    Nlist = np.logspace(1,3,50,dtype=int) 
    
    #rbar-s list
    rbar = 0.0
    s = -0.9

    non_reset_time = np.zeros(len(Nlist))
    reset_time = np.zeros(len(Nlist))
    for N,Nval in enumerate(Nlist):

        gamma = Nval**(-(rbar+1))
        kappa = Nval**(-s)

        s_nr = core.surv_prob_theory_total_logspace(Tstep, dt, Nval, gamma, kappa)
        non_reset_time[N] = core.find_transition_point(s_nr,Tstep, dt,threshold)


    for N,Nval in enumerate(Nlist):
        T_R = Nval**(s)
        if T_R >= dt:
            S_T = core.surv_prob_theory_T(T_R, Nval, gamma, kappa)

            if S_T >=threshold:
                m = np.log(threshold)/np.log(S_T)
                reset_time[N] = m*T_R
            else:
                reset_time[N] = non_reset_time[N]
        else:
            S_T = core.surv_prob_theory_T(dt, Nval, gamma, kappa)

            if S_T >=threshold:
                m = np.log(threshold)/np.log(S_T)
                reset_time[N] = m*dt
            else:
                reset_time[N] = non_reset_time[N]

    data = np.column_stack([Nlist] + [non_reset_time] + [reset_time])

    # Header row (no leading '#', clean commas)
    header = (
        "N,"
        "Non Reset tau (rbar=0.0+s=-0.9),"
        "Reset tau (rbar=0.0+s=-0.9)"
    )

    # Save / return
    if save:
        np.savetxt(out_csv, data, delimiter=",", header=header, comments="", fmt="%.16g")
    else:
        return data
    

# =============================================================================
# Figure 3b — Plot from CSV
# =============================================================================
def plot_figure3b_from_csv(
    csv_path,
    out_fig,
    dpi=600, save = True
):

    df = pd.read_csv(csv_path, comment="#")
    Nlist = df.iloc[:, 0].values
    non_reset_time = df.iloc[:, 1].values 
    reset_time = df.iloc[:, 2].values 
 


    #PRL width and height
    width_mm = 28.77
    width_in = width_mm / 25.4
    panel_h = 2.1238582677165354
    fig, ax2 = plt.subplots(figsize=(width_in,panel_h),dpi =200)
    style_axes(ax2)

    dt =0.01
    s = -0.9
    Nbar = int((dt)**(1.0/s))

    #scaling as predicted from the theory (See table 1)
    chakra  = "#000088" 
    ax2.plot(Nlist,4*Nlist**(1.9) ,color = 'crimson',ls ='--',lw = 1)
    ax2.plot(Nlist,6*Nlist**(0.1),color = 'violet',ls ='--',lw = 1)

    ax2.axvline(Nbar,color = 'k',ls ='--',lw = 1,alpha = 0.5)

   # ---- grey shaded region to the right of Nbar ----
    x1 = ax2.get_xlim()[1]          # current right x-limit
    ax2.axvspan(Nbar, x1, color='grey', alpha=0.25, zorder=0)


    #Plotting every 6 points for ease of visualization
    ax2.plot(Nlist[::6], (np.pi/2)*np.sqrt(Nlist[::6]),color = chakra,label = r"$\tau_{G}$")
    ax2.plot(Nlist[::4], non_reset_time[::4],'s',markersize = 3,color = 'crimson',markeredgecolor ='k',markeredgewidth= 0.7,label = r"$\tau$" )
    ax2.plot(Nlist[::4], reset_time[::4],'s',markersize = 3,color = 'violet',markeredgecolor ='k',markeredgewidth= 0.7, label = r"$\tau_R$")


    ax2.legend(
    loc='upper left',
    frameon=False,
    labelspacing=0.2,   # vertical space between legend rows (default ~0.5)
    handletextpad=0.4,  # space between handle (line/marker) and text
    handlelength=1.2,   # length of the line in the legend
    borderpad=0.1,      # padding inside the legend box (even if frameon=False)
    markerscale=0.9     # scale legend markers a bit (optional)
    )

    ax2.set_yscale('log')
    ax2.set_xscale('log')

    ax2.set_xlabel(r'$N$',labelpad = 0,size =8)

    nticks = 9
    min_loc = ticker.LogLocator(subs='all', numticks=nticks)
    ax2.xaxis.set_minor_locator(min_loc)
    ax2.yaxis.set_minor_locator(min_loc)


    ax2.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax2.yaxis.set_minor_formatter(ticker.NullFormatter())

    
    if save:
        plt.savefig(out_fig, dpi=dpi, bbox_inches="tight")
        plt.show()
    else:
        plt.show()


# ===========================Supplementary Material============================

# =============================================================================
# Figure S1 — Generate CSV
# =============================================================================
def generate_figureS1_csv(out_csv, N=100, save=True):
    """
    Generate data for the real and imaginary parts of the eigenvalues of the
    effective system Hamiltonian and store all results in a CSV file. We fix gamma
    to be at the exceptional point and fix N, varying monitoring strength kappa.

    The CSV contains:
    - kappa (rescaled by sqrt(N))
    - eigenvalue 1, real part
    - eigenvalue 2, real part
    - eigenvalue 1, imaginary part
    - eigenvalue 2, imaginary part
    """
    gamma_EP = 1.0 / (N-2)
    kappa_EP_plus = 2 * gamma_EP * np.sqrt(N-1)  # one branch
    gamma = gamma_EP # Fix gamma to value at the exceptional point
    kappa_vals = np.linspace(0, kappa_EP_plus + 0.6, 501) # list of kappa values

    data_rows = []
    for j, kappa in enumerate(kappa_vals):
        H = core.Heff_matrix(gamma, kappa, N)
        lam, _, _ = core.eig_biorth(H)
        idx = np.argsort(np.real(lam) + 1e-6*np.imag(lam))
        lam = lam[idx]
        data_rows.append([
                    float(kappa*np.sqrt(N)),
                    float(lam[0].real),
                    float(lam[1].real),
                    float(lam[0].imag),
                    float(lam[1].imag),
                ])
    if not save:
        return data_rows

    with open(out_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "kappa",
            "Re(eigenvalue 1)",
            "Re(eigenvalue 2)",
            "Im(eigenvalue 1)",
            "Im(eigenvalue 2)",
        ])
        for row in data_rows:
            row_out = [
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
            ]
            writer.writerow(row_out)

# =============================================================================
# Figure S1 — Plot from CSV
# =============================================================================
def plot_figureS1_from_csv(csv_path, fig_w=3.4, fig_h=4.74, dpi=600, save=False, out_fig=None):
    """
    Read CSV written by generate_figureS1_csv, and plot the figure as in the article.
    """
    kappaRescaled = []
    ev1Real = []
    ev2Real = []
    ev1Im = []
    ev2Im = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kappaRescaled.append(float(row["kappa"]))
            ev1Real.append(float(row["Re(eigenvalue 1)"]))
            ev2Real.append(float(row["Re(eigenvalue 2)"]))
            ev1Im.append(float(row["Im(eigenvalue 1)"]))
            ev2Im.append(float(row["Im(eigenvalue 2)"]))
    N = 100
    gamma_EP = 1.0 / (N-2)
    kappa_EP_plus = 2 * gamma_EP * np.sqrt(N-1)

    def style_axes(ax):
        # Ticks inward on all sides
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        # Major ticks: every 2 on x-axis
        ax.xaxis.set_major_locator(MultipleLocator(2))
        # Minor ticks: 4 between each major (i.e., 5 subdivisions)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
    gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0.07)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    # (a) Re(λ)
    ax1.plot(kappaRescaled, ev1Real, label=r'$\mathrm{Re}(\lambda_-)$', color="blue")
    ax1.plot(kappaRescaled, ev2Real, label=r'$\mathrm{Re}(\lambda_+)$', ls="-.", color="red")
    ax1.set_ylabel(r'$\mathrm{Re}(\lambda)$')
    ax1.axvline(kappa_EP_plus*np.sqrt(N), ls="--", color="grey", label="EP")
    style_axes(ax1)
    ax1.legend(frameon=False, loc='best')
    ax1.text(0.01, 0.93, r"$(a)$", transform=ax1.transAxes, ha="left", va="top")
    ax1.set_xlabel(None)
    ax1.tick_params(axis='x', which='both', labelbottom=False)

    # (b) Im(λ)
    ax2.plot(kappaRescaled, ev1Im, label=r'$\mathrm{Im}(\lambda_-)$', color="blue")
    ax2.plot(kappaRescaled, ev2Im, label=r'$\mathrm{Im}(\lambda_+)$', ls="-.", color="red")
    ax2.plot(kappaRescaled, np.array(ev1Im) + np.array(ev2Im), label=r'$\mathrm{Im}(\lambda_++\lambda_-)$', ls=":", color="green")
    ax2.set_xlabel(r'$\kappa\sqrt{N}$')
    ax2.set_ylabel(r'$\mathrm{Im}(\lambda)$')
    ax2.axvline(kappa_EP_plus*np.sqrt(N), ls="--", color="grey", label="EP")
    style_axes(ax2)
    ax2.legend(frameon=False, loc='best')
    # Panel label
    ax2.text(0.01, 0.93, r"$(b)$", transform=ax2.transAxes, ha="left", va="top")

    # ---------- make the inner axes (framed region) the same size ----------
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    def widest_ytick_width(ax):
        ticks = [t for t in ax.get_yticklabels() if t.get_text() != ""]
        if not ticks:
            return 0.0
        return max(t.get_window_extent(renderer=renderer).width for t in ticks)
    max_w = max(widest_ytick_width(ax1), widest_ytick_width(ax2))
    fig_w_px = fig.get_figwidth() * fig.dpi
    left_pad_px = 8.0 # Add a padding (in pixels) between the tick labels and the axes
    left = (max_w + left_pad_px) / fig_w_px
    fig.subplots_adjust(left=left, right=0.995, hspace=0.25) # Apply a common left margin so both axes share identical inner width

    # Save or show
    if save:
        if out_fig is None:
            raise ValueError("out_fig must be provided when save=True")
        plt.savefig(out_fig, dpi=dpi, bbox_inches="tight")
        plt.show()
    else:
        plt.show()

        
# =============================================================================
# Figure S2 — Generate CSV
# =============================================================================
def generate_figureS2_csv(out_csv, N=100, save=True):
    """
    Generate data for the real and imaginary parts of the eigenvalues of the
    effective system Hamiltonian and store all results in a CSV file. We fix kappa
    to be at the exceptional point (where kappa>0) and fix N, varying hopping parameter gamma.

    The CSV contains:
    - gamma (rescaled by N)
    - eigenvalue 1, real part
    - eigenvalue 2, real part
    - eigenvalue 1, imaginary part
    - eigenvalue 2, imaginary part
    """
    gamma_EP = 1.0 / (N-2)
    kappa_EP_plus = 2 * gamma_EP * np.sqrt(N-1)  # one branch (positive kappa)
    kappa = kappa_EP_plus # Fix kappa to value at the exceptional point
    gamma_vals = np.linspace(0, gamma_EP + 0.02, 501) # list of gamma values

    data_rows = []
    for j, gamma in enumerate(gamma_vals):
        H = core.Heff_matrix(gamma, kappa, N)
        lam, _, _ = core.eig_biorth(H)
        idx = np.argsort(np.real(lam) + 1e-6*np.imag(lam))
        lam = lam[idx]
        data_rows.append([
                    float(gamma*N),
                    float(lam[0].real),
                    float(lam[1].real),
                    float(lam[0].imag),
                    float(lam[1].imag),
                ])
    if not save:
        return data_rows

    with open(out_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "gamma",
            "Re(eigenvalue 1)",
            "Re(eigenvalue 2)",
            "Im(eigenvalue 1)",
            "Im(eigenvalue 2)",
        ])
        for row in data_rows:
            row_out = [
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
            ]
            writer.writerow(row_out)

# =============================================================================
# Figure S2 — Plot from CSV
# =============================================================================
def plot_figureS2_from_csv(csv_path, fig_w=3.4, fig_h=4.74, dpi=600, save=False, out_fig=None):
    """
    Read CSV written by generate_figureS1_csv, and plot the figure as in the article.
    """
    gammaRescaled = []
    ev1Real = []
    ev2Real = []
    ev1Im = []
    ev2Im = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gammaRescaled.append(float(row["gamma"]))
            ev1Real.append(float(row["Re(eigenvalue 1)"]))
            ev2Real.append(float(row["Re(eigenvalue 2)"]))
            ev1Im.append(float(row["Im(eigenvalue 1)"]))
            ev2Im.append(float(row["Im(eigenvalue 2)"]))
    N = 100
    gamma_EP = 1.0 / (N-2)
    kappa_EP_plus = 2 * gamma_EP * np.sqrt(N-1)

    def style_axes(ax):
        # Ticks inward on all sides
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        # Major ticks: every 2 on x-axis
        ax.xaxis.set_major_locator(MultipleLocator(2))
        # Minor ticks: 4 between each major (i.e., 5 subdivisions)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
    gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0.07)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    # (a) Re(λ)
    ax1.plot(gammaRescaled, ev1Real, label=r'$\mathrm{Re}(\lambda_-)$', color="blue")
    ax1.plot(gammaRescaled, ev2Real, label=r'$\mathrm{Re}(\lambda_+)$', ls="-.", color="red")
    ax1.set_xlabel(r'$\gamma N$')
    ax1.set_ylabel(r'$\mathrm{Re}(\lambda)$')
    ax1.axvline(gamma_EP*N, ls="--", color="grey", label="EP")

    style_axes(ax1)
    ax1.legend(frameon=False, loc='best')
    # Panel label
    ax1.text(0.01, 0.93, r"$(a)$", transform=ax1.transAxes, ha="left", va="top")
    ax1.set_xlabel(None)
    ax1.tick_params(axis='x', which='both', labelbottom=False)

    # (b) Im(λ)
    ax2.plot(gammaRescaled, ev1Im, label=r'$\mathrm{Im}(\lambda_-)$', color="blue")
    ax2.plot(gammaRescaled, ev2Im, label=r'$\mathrm{Im}(\lambda_+)$', ls="-.", color="red")
    ax2.plot(gammaRescaled, np.array(ev1Im) + np.array(ev2Im), label=r'$\mathrm{Im}(\lambda_++\lambda_-)$', ls=":", color="green")
    ax2.set_xlabel(r'$\gamma N$')
    ax2.set_ylabel(r'$\mathrm{Im}(\lambda)$')
    ax2.axvline(gamma_EP*N, ls="--", color="grey", label="EP")
    style_axes(ax2)
    ax2.legend(frameon=False, loc='best')
    # Panel label
    ax2.text(0.01, 0.93, r"$(b)$", transform=ax2.transAxes, ha="left", va="top")

    # ---------- make the inner axes (framed region) the same size ----------
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    def widest_ytick_width(ax):
        ticks = [t for t in ax.get_yticklabels() if t.get_text() != ""]
        if not ticks:
            return 0.0
        return max(t.get_window_extent(renderer=renderer).width for t in ticks)
    max_w = max(widest_ytick_width(ax1), widest_ytick_width(ax2))
    fig_w_px = fig.get_figwidth() * fig.dpi
    left_pad_px = 8.0 # Add a padding (in pixels) between the tick labels and the axes
    left = (max_w + left_pad_px) / fig_w_px
    fig.subplots_adjust(left=left, right=0.995, hspace=0.25) # Apply a common left margin so both axes share identical inner width

    # Save or show
    if save:
        if out_fig is None:
            raise ValueError("out_fig must be provided when save=True")
        plt.savefig(out_fig, dpi=dpi, bbox_inches="tight")
        plt.show()
    else:
        plt.show()

# =============================================================================
# Figure S3a — Generate CSV
# =============================================================================
def generate_figureS3a_csv(
    out_csv, save = True
):  

    r_vals = np.linspace(-1.0, 1.0, 1201)
    s_vals = np.linspace(-1.0, 1.0, 1201)

    R, S = np.meshgrid(r_vals, s_vals)
    Alpha = np.empty_like(R)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            Alpha[i, j] = core.alpha_of_rs_scalar(R[i, j], S[i, j])

    
    # Build data for CSV
    x_flat = R.ravel(order='C')
    y_flat = S.ravel(order ='C')
    z_flat = Alpha.ravel(order = 'C')


    data = np.column_stack([x_flat, y_flat,z_flat])

    # Header row 
    header = "rbar, s, alpha"

    # Save CSV
    if save:
        np.savetxt(out_csv, data, delimiter=",", header=header, comments="", fmt="%.16g")
    else:
        return data

# =============================================================================
# Figure S3a — Plot from CSV
# =============================================================================
def plot_figureS3a_from_csv(
    csv_path,
    out_fig,
    dpi=600, save = True
):

    df = pd.read_csv(csv_path, comment="#")
    x_flat = df.iloc[:, 0].values  # alpha list
    y_flat = df.iloc[:, 1].values  # time
    z_flat = df.iloc[:, 2].values  # S(t)

    x_vals = np.unique(x_flat)   # unique rbar
    y_vals = np.unique(y_flat)   # unique s

    R, S = np.meshgrid(x_vals, y_vals)
    Alpha = z_flat.reshape(len(y_vals), len(x_vals))

    #PRL width and height
    width_mm = 59.94 
    width_in = width_mm / 25.4
    panel_h = 2.1238582677165354
    fig, ax = plt.subplots(figsize=(width_in,panel_h),dpi =200)
    
    style_axes(ax)
    for sp in ax.spines.values():
        sp.set_zorder(50)

    # plotting the special critical star (chakra) at (0, 0.5)
    chakra  = "#000088" 
    ax.plot(0.0, 0.5, marker='*',color= chakra, markersize=8, zorder=8)
    ax.text(-0.45, 0.47, r'$(0,\,\frac{1}{2})$', color=chakra, fontsize=8)

    #plot the points for Fig.2b
    ax.plot(0.25, 0.25, marker='o', color= 'gray', markersize=8,markeredgecolor ='k', zorder=8)
    ax.plot(-0.9, -1.0, marker='s', color='crimson', markersize=7,markeredgecolor ='k', zorder=8)
    ax.plot(-0.5, 0.0, marker='d', color='gold', markersize=8,markeredgecolor ='k', zorder=8)
    ax.plot(0.0, 0.85, marker='v', color='violet', markersize=8,markeredgecolor ='k', zorder=8)

    #labelling markers for Fig. 2b
    label_positions = [

        (-0.1, 0.7),   # A1
        (-0.1,0.15),    #A2
        (0.7,0.8),    # B 
        (0.4, -0.5),# C
        (-0.8, 0)     # D
    ]


    letters = ["A1", "A2", "B", "C", "D"]

    for letter, (lr, ls) in zip(letters, label_positions):

        label_text = f"({letter})" #+ a_val

        # place text with a white box for readability
        ax.text(lr, ls, label_text, fontsize=8, ha='left', va='center',
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="0.8", alpha=0.85))

    #contour plot
    Alpha_sneg = np.ma.masked_where(S >= 0, Alpha)  # mask out s>=0
    cp = ax.contourf(R, S, Alpha_sneg, levels=np.arange(0.0,1.01,0.01),cmap = indian_flag_cmap(with_chakra=False))
    cbar = plt.colorbar(cp,ax=ax, orientation="vertical", fraction=0.08,shrink = 0.85)
    
    # line across the colorbar at alpha = 0.5
    cbar.ax.axhline(0.5, color=chakra, lw=1.6)
    ticks = [0,  0.25, 0.5, 0.75, 1.0]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([0,r'$\frac{1}{4}$', r'$\frac{1}{2}$',r'$\frac{3}{4}$',1])
    cbar.ax.text(0.5,1.05, r'$\alpha$', transform=cbar.ax.transAxes,ha='center', va='center')

    #plot the critical line at rbar = 0
    s_crit = np.arange(0.0, 1.01,0.01)
    r_crit = np.zeros(len(s_crit))


    alpha = np.array([core.alpha_of_rs_scalar(0.0,s_crit[i]) for i in range(len(s_crit))])

    # Create segments for LineCollection
    points = np.array([r_crit, s_crit]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create the line collection
    norm = Normalize(vmin=0.0, vmax=1)


    lc = LineCollection(
        segments,
        cmap=indian_flag_cmap(),
        norm=norm,
        array=alpha,
        linewidth=3
    )

    line = ax.add_collection(lc)

    ax.contour(R, S, Alpha, levels=[0.5], colors=chakra, linewidths=1.6, linestyles='--')

    ax.set_xlabel(r'$\bar{r}$',labelpad =1,size = 8 )
    ax.set_ylabel(r'$s$',labelpad =-6,size = 8)

    # draw thin axes guide lines separating different regions
    ax.axhline(0.0,xmin = 0.5, ls=":", color="black", linewidth=1)
    ax.axvline(0.0,ymin = 0.5, ls=":", color="black", linewidth=1)
    ax.plot(x_vals[x_vals<0],y_vals[y_vals<0], ls=":", color="black", linewidth=1)


    ax.text(-0.5, -0.4, r"$s=\bar{r}$", fontsize=8, color="k", rotation=45)
    ax.text(-0.66, -0.95, r"$s= 2\bar{r} +\frac{1}{2}$", fontsize=8, color=chakra, rotation=65)
    ax.text(-0.42, -0.95, r"$s= 2\bar{r}$", fontsize=8, color = '#138808', rotation=65)



    plt.yticks(fontsize =8)
    plt.xticks(fontsize =8)


    if save:
        plt.savefig(out_fig, dpi=dpi, bbox_inches="tight")
        plt.show()
    else:
        plt.show()
    

# =============================================================================
# Figure S3b — Generate CSV
# =============================================================================
def generate_figureS3b_csv(
    out_csv,dt = 0.01,threshold = 0.001, save = True
):  

    #Time in logspace
    Tstep = np.logspace(1,15.6,2500,dtype=int)

    #List of system size
    Nlist = np.logspace(2,5,50,dtype=int) 
    
    #rbar-s list
    rbars_list = [
        (-0.9,-1.0),
        (-0.5,0.0),
        (0.0,0.5),
        (0.0,0.85),
        (0.25,0.25)
    ]

    results = []

    i=0
    for (rbar, s) in rbars_list:

        results.append([])
        for N,Nval in enumerate(Nlist):

            gamma = Nval**(-(rbar+1))
            kappa = Nval**(-s)

            s_nr = core.surv_prob_theory_total_logspace(Tstep, dt, Nval, gamma, kappa)
            results[i].append(core.find_transition_point(s_nr,Tstep, dt,threshold))
        i = i + 1


    # --- build data matrix: Nlist + results columns ---
    cols = [np.asarray(results[j], dtype=float) for j in range(len(results))]

    data = np.column_stack([Nlist] + cols)

    # Header row (no leading '#', clean commas)
    header = (
        "N,"
        "tau (rbar=-0.9+s=-1.0),"
        "tau (rbar=-0.5+s=0.0),"
        "tau (rbar=0.0+s=0.5),"
        "tau (rbar=0.0+s=0.85),"
        "tau (rbar=0.25+s=0.25)"
    )

    # Save / return
    if save:
        np.savetxt(out_csv, data, delimiter=",", header=header, comments="", fmt="%.16g")
    else:
        return data
    

# =============================================================================
# Figure S3b — Plot from CSV
# =============================================================================
def plot_figureS3b_from_csv(
    csv_path,
    out_fig,
    dpi=600, save = True
):

    df = pd.read_csv(csv_path, comment="#")
    Nlist = df.iloc[:, 0].values
    results_0 = df.iloc[:, 1].values  # rbar=-0.9,s=-1.0
    results_1 = df.iloc[:, 2].values  # rbar=-0.5,s=0.0
    results_2 = df.iloc[:, 3].values  # rbar=0.0,s=0.5
    results_3 = df.iloc[:, 4].values  # rbar=0.0,s=0.85
    results_4 = df.iloc[:, 5].values  # rbar=0.25,s=0.25


    #PRL width and height
    width_mm = 28.77
    width_in = width_mm / 25.4
    panel_h = 2.1238582677165354
    fig, ax2 = plt.subplots(figsize=(width_in,panel_h),dpi =200)
    style_axes(ax2)

    #scaling as predicted from the theory (See table 1)
    chakra  = "#000088" 
    ax2.plot(Nlist,4*Nlist**(0.2) ,color = 'crimson',ls ='--',lw = 1)
    ax2.plot(Nlist,6*Nlist**(1/2) ,color = chakra,ls ='--',lw = 1)
    ax2.plot(Nlist,3*Nlist**(1.75) ,color = 'grey',ls ='--',lw = 1)
    ax2.plot(Nlist,4*Nlist**(1),color = 'gold',ls ='--',lw = 1)
    ax2.plot(Nlist,7.5*Nlist**(0.85),color = 'violet',ls ='--',lw = 1)

    #Plotting every 6 points for ease of visualization
    ax2.plot(Nlist[::6], results_0[::6],'s',markersize = 4,color = 'crimson',markeredgecolor ='k',markeredgewidth= 0.7)
    ax2.plot(Nlist[::6], results_1[::6],'d',markersize = 4,color = 'gold',markeredgecolor ='k',markeredgewidth= 0.7)
    ax2.plot(Nlist[::6], results_2[::6],'*',markersize = 4,color = chakra)
    ax2.plot(Nlist[::6], results_3[::6],'v',markersize = 4,color = 'violet',markeredgecolor ='k',markeredgewidth= 0.7)
    ax2.plot(Nlist[::6], results_4[::6],'o',markersize = 4,color = 'grey',markeredgecolor ='k',markeredgewidth= 0.7)

    ax2.set_yscale('log')
    ax2.set_xscale('log')

    ax2.set_xlabel(r'$N$',labelpad = 0,size =8)
    ax2.set_ylabel(r'$\tau$',labelpad = 0,size = 10)

    nticks = 9
    min_loc = ticker.LogLocator(subs='all', numticks=nticks)
    ax2.xaxis.set_minor_locator(min_loc)
    ax2.yaxis.set_minor_locator(min_loc)


    ax2.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax2.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax2.text(2e4, 1e9, r"B", fontsize=8, color='grey', rotation=0)
    ax2.text(2e4, 5e5, r"D", fontsize=8, color='gold', rotation=0)
    ax2.text(2e4, 1e4, r"A1", fontsize=8, color='violet', rotation=0)
    ax2.text(2e4, 0.7e1, r"C", fontsize=8, color='crimson', rotation=0)


    ax2.text(4.3e3, 1e2, r"$\Theta(\sqrt{N})$", fontsize=8, color=chakra, rotation=0)


    if save:
        plt.savefig(out_fig, dpi=dpi, bbox_inches="tight")
        plt.show()
    else:
        plt.show()



# =============================================================================
# Figure S4 — Generate CSV
# =============================================================================
def generate_figureS4_csv(
    out_csv,
    Ns,
    panels, save = True
):
    """
    Generate data for the smallest absolute imaginary part of the eigenvalues
    and store all results in a CSV file.

    The CSV contains:
    - panel index
    - panel title
    - dataset index
    - parameters gamma bar, r exponent, kappa bar, s exponent
    - system size N
    - inverse imaginary part of the slowest eigenvalue
    """

    with open(out_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        # CSV header
        writer.writerow([
            "panel_index",
            "panel_title",
            "dataset_index",
            "gbar",
            "r",
            "kbar",
            "s",
            "N",
            "inv_im_lambda_s"
        ])

        for p_idx, panel in enumerate(panels):
            title = panel["title"]
            for d_idx, (gbar, r, kbar, s) in enumerate(panel["datasets"]):
                for N in Ns:
                    gamma = gbar / (N ** r)
                    kappa = kbar / (N ** s)

                    H = core.Heff_matrix(gamma, kappa, N)
                    lam, VR, VL = core.eig_biorth(H)

                    im_vals = np.abs(np.imag(lam))
                    im_s = np.min(im_vals)

                    inv_im_s = 1.0 / max(im_s, 1e-300)

                    writer.writerow([
                        p_idx,
                        title,
                        d_idx,
                        gbar,
                        r,
                        kbar,
                        s,
                        N,
                        inv_im_s
                    ])


# =============================================================================
# Figure S4 — Plot from CSV
# =============================================================================
def plot_figureS4_from_csv(
    csv_path,
    out_fig,
    fig_w=6.8, 
    fig_h=4.54,
    dpi=600,
    save=True,
):
    panels = [
        { 'title': 'Exceptional point',
        'analytic_alphas': [0.5, 1.5, 1.5],
        'analytic_label': None},
        { 'title': 'Regime A1',
        'analytic_alphas': None,
        'analytic_label': r'$\sim N^{s}$',
        'analytic_alpha_func': lambda s: s},
        { 'title': 'Regime A2',
        'analytic_alphas': None,
        'analytic_label': r'$\sim N^{1-s}$',
        'analytic_alpha_func': lambda s: 1 - s},
        { 'title': 'Regime B',
        'analytic_alphas': None,
        'analytic_label': r'$\sim N^{2r+s-1}$',
        'analytic_alpha_func': lambda r,s: 2*r + s - 1},
        { 'title': 'Regime C',
        'analytic_alphas': None,
        'analytic_label': r'$\sim N^{2r-s-1}$',
        'analytic_alpha_func': lambda r,s: 2*r - s - 1},
        { 'title': 'Regime D',
        'analytic_alphas': None,
        'analytic_label': r'$\sim N^{s+1}$',
        'analytic_alpha_func': lambda s: 1 + s}
    ]
    # ------------------------------------------------------------
    # Read CSV into nested dictionary keyed by (panel_index, dataset_index)
    # ------------------------------------------------------------
    data = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p_idx = int(row["panel_index"])
            d_idx = int(row["dataset_index"])
            key = (p_idx, d_idx)

            if key not in data:
                data[key] = {
                    "params": (
                        float(row["gbar"]),
                        float(row["r"]),
                        float(row["kbar"]),
                        float(row["s"])
                    ),
                    "N": [],
                    "y": [],
                    "title": str(row["panel_title"]),
                }
            data[key]["N"].append(int(row["N"]))
            data[key]["y"].append(float(row["inv_im_lambda_s"]))

    # Convert lists to numpy arrays and ensure sorted by N
    for entry in data.values():
        order = np.argsort(entry["N"])
        entry["N"] = np.array(entry["N"])[order]
        entry["y"] = np.array(entry["y"])[order]

    # ------------------------------------------------------------
    # Figure setup, marker styles and colours
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(fig_w, fig_h))
    plt.subplots_adjust(wspace=0.25, hspace=0.45, left=0.07, right=0.98, top=0.95, bottom=0.08)

    markers = ["o", "s", "d"]
    cmap = plt.get_cmap("viridis")
    base_colors = [cmap(0.12), cmap(0.5), cmap(0.88)]

    colors_class = {
        "D": ["#800000", "#FF0000", "#F78D66"],
        "B": ["#000080", "#607EDB", "#E6ADFA"],
        "C": ["#315C3F", "#37C468", "#A2BB76"]
    }

    # ------------------------------------------------------------
    # Plot each panel
    # ------------------------------------------------------------
    for p_idx, panel in enumerate(panels):
        ax = fig.add_subplot(2, 3, p_idx + 1)
        ax.set_title(data[(p_idx,0)]["title"])
        header_label = r'$(\bar\gamma,r,\bar\kappa,s)$'
        legend_handles = []
        legend_labels = []
        h_header = Line2D([0], [0], linestyle="None", marker="None", label=header_label)
        legend_handles.append(h_header)
        legend_labels.append(header_label)

        for d_idx in [0,1,2]:
            key = (p_idx, d_idx)
            if key not in data:
                raise ValueError(f"Missing CSV data for panel {p_idx}, dataset {d_idx}")
            entry = data[key]
            N_vals = entry["N"]
            y_vals = entry["y"]
            color = base_colors[d_idx % len(base_colors)]
            if p_idx > 2:
                class_idx = p_idx % 3
                class_label = ["B", "C", "D"][class_idx]
                color = colors_class[class_label][d_idx]

            mk = markers[d_idx % len(markers)]
            ax.loglog(N_vals, y_vals, marker=mk, linestyle='', markerfacecolor=color,
                      markeredgecolor='k', color=color)

            gbar, r, kbar, s = entry["params"]
            lab = f'({gbar},{r},{kbar},{s})'
            h = Line2D([0], [0], marker=mk, linestyle='None', markerfacecolor=color, markeredgecolor='k')
            legend_handles.append(h)
            legend_labels.append(lab)

        if panel.get("analytic_alphas") is not None:
            for j, alpha in enumerate(panel["analytic_alphas"]):
                key = (p_idx, j)
                if key not in data:
                    # skip if numeric data missing
                    continue
                # Base point
                N0 = data[key]["N"][0]
                y0 = data[key]["y"][0]
                # analytic reference
                analytic_line = y0 * (N_vals / N0) ** (alpha)
                color = base_colors[j % len(base_colors)]
                if p_idx > 2:
                    class_idx = p_idx % 3
                    class_label = ["B", "C", "D"][class_idx]
                    color = colors_class[class_label][j]

                ax.loglog(N_vals, analytic_line, linestyle='--', linewidth=1.5, color=color)

            h_dot = Line2D([0, 1], [0, 1], linestyle='--', color='black')
            legend_handles.append(h_dot)
            legend_labels.append('Analytic')

        elif panel.get("analytic_alpha_func") is not None and panel.get("analytic_label") is not None:
            for j in [0,1,2]:
                key = (p_idx, j)
                gbar0, r0, kbar0, s0 = data[key]["params"]
                try:
                    alpha_val = panel["analytic_alpha_func"](s0)
                except TypeError:
                    try:
                        alpha_val = panel["analytic_alpha_func"](r0, s0)
                    except Exception:
                        alpha_val = None

                if alpha_val is None:
                    continue

                if key not in data:
                    continue

                N0 = data[key]["N"][0]
                y0 = data[key]["y"][0]

                analytic_line = y0 * (N_vals / N0) ** (alpha_val)
                color = base_colors[j % len(base_colors)]
                if p_idx > 2:
                    class_idx = p_idx % 3
                    class_label = ["B", "C", "D"][class_idx]
                    color = colors_class[class_label][j]

                ax.loglog(N_vals, analytic_line, linestyle='--', linewidth=1.5, color=color, label=panel['analytic_label'])

                if j == 0:
                    h_dot = Line2D([0, 1], [0, 1], linestyle='--', color='black')
                    legend_handles.append(h_dot)
                    legend_labels.append(panel['analytic_label'])

        # --------------------------------------------------------
        # Axes formatting
        # --------------------------------------------------------
        ax.set_xlabel(r'$N$')

        if p_idx in [0, 3]:
            ax.set_ylabel(r'$1/| \mathrm{Im}\,\lambda_s |$')
        else:
            ax.set_ylabel('')

        ax.set_yscale("log")
        ax.grid(False)

        ax.legend(legend_handles, legend_labels, handletextpad=0.4, borderpad=0.2,
                  labelspacing=0.2, loc='upper left', frameon=True)

        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=12))
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax.tick_params(axis='y', which='major')
        ax.tick_params(axis='y', which='minor')

    # ------------------------------------------------------------
    # Save or show figure
    # ------------------------------------------------------------
    if save:
        if out_fig is None:
            raise ValueError("out_fig must be provided when save=True")
        plt.savefig(out_fig, dpi=dpi, bbox_inches="tight")
        plt.show()
    else:
        plt.show()
 

# =============================================================================
# Figure S5 — Generate CSV
# =============================================================================
def generate_figureS5_csv(
    out_csv,
    Ns,
    panels, save = True
):
    """
    Generate data for the largest absolute imaginary part of the eigenvalues
    and store all results in a CSV file.

    The CSV contains:
    - panel index
    - panel title
    - dataset index
    - parameters gamma bar, r exponent, kappa bar, s exponent
    - system size N
    - inverse imaginary part of the fastest eigenvalue
    """

    with open(out_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        # CSV header
        writer.writerow([
            "panel_index",
            "panel_title",
            "dataset_index",
            "gbar",
            "r",
            "kbar",
            "s",
            "N",
            "inv_im_lambda_f"
        ])

        for p_idx, panel in enumerate(panels):
            title = panel["title"]
            for d_idx, (gbar, r, kbar, s) in enumerate(panel["datasets"]):
                for N in Ns:
                    gamma = gbar / (N ** r)
                    kappa = kbar / (N ** s)

                    H = core.Heff_matrix(gamma, kappa, N)
                    lam, VR, VL = core.eig_biorth(H)

                    im_vals = np.abs(np.imag(lam))
                    im_f = np.max(im_vals)

                    inv_im_f = 1.0 / max(im_f, 1e-300)

                    writer.writerow([
                        p_idx,
                        title,
                        d_idx,
                        gbar,
                        r,
                        kbar,
                        s,
                        N,
                        inv_im_f
                    ])

# =============================================================================
# Figure S5 — Plot from CSV
# =============================================================================
def plot_figureS5_from_csv(
    csv_path,
    out_fig,
    fig_w=6.8, 
    fig_h=4.54,
    dpi=600,
    save=True
):
    # ------------------------------------------------------------
    # Read CSV into nested dictionary keyed by (panel_index, dataset_index)
    # ------------------------------------------------------------
    data = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p_idx = int(row["panel_index"])
            d_idx = int(row["dataset_index"])
            key = (p_idx, d_idx)

            if key not in data:
                data[key] = {
                    "params": (
                        float(row["gbar"]),
                        float(row["r"]),
                        float(row["kbar"]),
                        float(row["s"])
                    ),
                    "N": [],
                    "y": [],
                    "title": str(row["panel_title"]),
                }
            data[key]["N"].append(int(row["N"]))
            data[key]["y"].append(float(row["inv_im_lambda_f"]))

    # Convert lists to numpy arrays and ensure sorted by N
    for entry in data.values():
        order = np.argsort(entry["N"])
        entry["N"] = np.array(entry["N"])[order]
        entry["y"] = np.array(entry["y"])[order]

    # ------------------------------------------------------------
    # Figure setup, marker styles and colours
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(fig_w, fig_h))
    plt.subplots_adjust(wspace=0.25, hspace=0.45, left=0.07, right=0.98, top=0.95, bottom=0.08)

    markers = ["o", "s", "d"]
    cmap = plt.get_cmap("viridis")
    base_colors = [cmap(0.12), cmap(0.5), cmap(0.88)]

    colors_class = {
        "D": ["#800000", "#FF0000", "#F78D66"],
        "B": ["#000080", "#607EDB", "#E6ADFA"],
        "C": ["#315C3F", "#37C468", "#A2BB76"]
    }

    # ------------------------------------------------------------
    # Plot each panel
    # ------------------------------------------------------------
    for p_idx in range(6):
        ax = fig.add_subplot(2, 3, p_idx + 1)
        ax.set_title(data[(p_idx,0)]["title"])
        header_label = r'$(\bar\gamma,r,\bar\kappa,s)$'
        legend_handles = []
        legend_labels = []
        h_header = Line2D([0], [0], linestyle="None", marker="None", label=header_label)
        legend_handles.append(h_header)
        legend_labels.append(header_label)

        for d_idx in [0,1,2]:
            key = (p_idx, d_idx)
            if key not in data:
                raise ValueError(f"Missing CSV data for panel {p_idx}, dataset {d_idx}")
            entry = data[key]
            N_vals = entry["N"]
            y_vals = entry["y"]
            color = base_colors[d_idx % len(base_colors)]
            if p_idx > 2:
                class_idx = p_idx % 3
                class_label = ["B", "C", "D"][class_idx]
                color = colors_class[class_label][d_idx]

            mk = markers[d_idx % len(markers)]
            ax.loglog(N_vals, y_vals, marker=mk, linestyle='', markerfacecolor=color,
                      markeredgecolor='k', color=color)

            gbar, r, kbar, s = entry["params"]
            lab = f'({gbar},{r},{kbar},{s})'
            h = Line2D([0], [0], marker=mk, linestyle='None', markerfacecolor=color, markeredgecolor='k')
            legend_handles.append(h)
            legend_labels.append(lab)

            analytic_line = y_vals[0] * (N_vals / N_vals[0]) ** (s)
            ax.loglog(N_vals, analytic_line, linestyle='--', linewidth=1.5, color=color)
            h_dot = Line2D([0,1],[0,1], linestyle='--', color='black')
            if d_idx==2: 
                legend_handles.append(h_dot)
                legend_labels.append(r'$\sim N^{s}$')

        # --------------------------------------------------------
        # Axes formatting
        # --------------------------------------------------------
        ax.set_xlabel(r'$N$')

        if p_idx in [0, 3]:
            ax.set_ylabel(r'$1/| \mathrm{Im}\,\lambda_f |$')
        else:
            ax.set_ylabel('')

        ax.set_yscale("log")
        ax.grid(False)

        if p_idx in [4,5]:
            ax.legend(legend_handles, legend_labels, handletextpad=0.4, borderpad=0.2, labelspacing=0.2, loc='lower left', frameon=True)
        else:
            ax.legend(legend_handles, legend_labels, handletextpad=0.4, borderpad=0.2, labelspacing=0.2, loc='upper left', frameon=True)

        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=12))
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax.tick_params(axis='y', which='major')
        ax.tick_params(axis='y', which='minor')

    # ------------------------------------------------------------
    # Save or show figure
    # ------------------------------------------------------------
    if save:
        if out_fig is None:
            raise ValueError("out_fig must be provided when save=True")
        plt.savefig(out_fig, dpi=dpi, bbox_inches="tight")
        plt.show()
    else:
        plt.show()





# =============================================================================
# Figure S6 — Generate CSV
# =============================================================================
def generate_figureS6_csv(
    out_csv,dt =0.01, save = True
):

    rs_list = [(1.1,0.25, 8.6),(-1.0,-0.5,4.6),(-1.0,-1.0,3.6)]

    Nlist = np.array([100,1000,10000])

    results = []
    for r,s,m in rs_list:
        Tstep = np.unique(np.logspace(1,m,2000,dtype=int))
        for N,Nval in enumerate(Nlist):
            gamma = Nval**(-r)
            kappa = Nval**(-s)

            s_nr = core.surv_prob_theory_total_logspace(Tstep, dt, Nval, gamma, kappa)
            results.append(s_nr)

    T1 = np.unique(np.logspace(1, rs_list[0][2], 2000).astype(int))*dt
    T2 = np.unique(np.logspace(1, rs_list[1][2], 2000).astype(int))*dt
    T3 = np.unique(np.logspace(1, rs_list[2][2], 2000).astype(int))*dt

    
    # Pad columns to same length so column_stack works
    def pad_to(arr, L):
        out = np.full(L, np.nan, dtype=float)
        out[:len(arr)] = np.asarray(arr, dtype=float)
        return out

    L = max(len(T1), len(T2), len(T3))


    # Data
    data = np.column_stack([
        pad_to(T1, L), pad_to(results[0], L), pad_to(results[1], L), pad_to(results[2], L),
        pad_to(T2, L), pad_to(results[3], L), pad_to(results[4], L), pad_to(results[5], L),
        pad_to(T3, L), pad_to(results[6], L), pad_to(results[7], L), pad_to(results[8], L),
    ])


    # Header:
    header = (
        "T [r = 1.1 + s = 0.25],"
        "P(t)(N=100)[r = 1.1 + s = 0.25],"
        "P(t) (N=1000)[r = 1.1 + s = 0.25],"
        "P(t) (N=10000)[r = 1.1 + s = 0.25],"
        "T [r = -1.0 + s = -0.5],"
        "P(t)(N=100)[r = -1.0 + s = -0.5],"
        "P(t) (N=1000)[r = -1.0 + s = -0.5],"
        "P(t) (N=10000)[r = -1.0 + s = -0.5],"
        "T [r = -1.0 + s = -1.0],"
        "P(t)(N=100) [r = -1.0 + s = -1.0],"
        "P(t) (N=1000) [r = -1.0 + s = -1.0],"
        "P(t) (N=10000) [r = -1.0 + s = -1.0]"        
    )

  
    # Save / return
    if save:
        np.savetxt(out_csv, data, delimiter=",", header=header, comments="", fmt="%.16g")
    else:
        return data


# =============================================================================
# Figure S6 — Plot from CSV
# =============================================================================
def plot_figureS6_from_csv(
    csv_path,
    out_fig,
    dpi=600, save = True
):
    df = pd.read_csv(csv_path, comment="#")
    results_0 = df.iloc[:, 1].values  # r = 1.1, s = 0.25, N = 100
    results_1 = df.iloc[:, 2].values  # r = 1.1, s = 0.25, N = 1000
    results_2 = df.iloc[:, 3].values  # r = 1.1, s = 0.25, N = 10000
    results_3 = df.iloc[:, 5].values  # r = -1.0, s = -0.5, N = 100
    results_4 = df.iloc[:, 6].values  # r = -1.0, s = -0.5, N = 1000
    results_5 = df.iloc[:, 7].values  # r = -1.0, s = -0.5, N = 10000
    results_6 = df.iloc[:, 9].values  # r = -1.0, s = -1.0, N = 100
    results_7 = df.iloc[:, 10].values  # r = -1.0, s = -1.0, N = 1000
    results_8 = df.iloc[:, 11].values  # r = -1.0, s = -1.0, N = 10000


    def style_axes_log(ax):

        ax.tick_params(axis='both', which='both', direction='in', top=False, right=False)


        # X minors
        if ax.get_xscale() == 'log':
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2,10)*0.1, numticks=100))
            ax.xaxis.set_minor_formatter(NullFormatter())  # keep labels only on majors
        else:
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        # Y minors (do the same logic; optional if you only need x)
        if ax.get_yscale() == 'log':
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2,10)*0.1, numticks=100))
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))


    width_mm = 179.83 
    width_in = width_mm / 25.4
    panel_aspect = 2/8  
    panel_h = width_in * panel_aspect
    fig, axs = plt.subplots(1, 3, figsize=(width_in,panel_h), sharey=True,dpi =200)
    # Apply styling to each axes
    for ax in axs.ravel():
        style_axes_log(ax)

    Ns_labels = [r'$N = 100$', r'$N = 1000$', r'$N = 10000$']

    Tstep =  df.iloc[:, 0].values
    axs[0].plot(Tstep, results_0,'-', lw=1.2)
    axs[0].plot(Tstep, results_1,'--' , lw=1.2)
    axs[0].plot(Tstep, results_2,':', lw=1.2)
    axs[0].set_xscale('log')
    axs[0].set_ylabel(r'$P(t)$')
    axs[0].set_xlabel(r'$t$')
    axs[0].text(0.1,0.85 , r"(a)", fontsize=9, color="k", rotation=0)
    

    Tstep = df.iloc[:, 4].values
    axs[1].plot(Tstep, results_3,'-', lw=1.2)
    axs[1].plot(Tstep, results_4,'--', lw=1.2)
    axs[1].plot(Tstep, results_5,':', lw=1.2)
    axs[1].set_xscale('log')
    axs[1].set_xlabel(r'$t$')
    axs[1].text(0.1,0.85 , r"(b)", fontsize=9, color="k", rotation=0)

    Tstep =  df.iloc[:, 8].values
    axs[2].plot(Tstep, results_6,'-', label=Ns_labels[0], lw=1.2)
    axs[2].plot(Tstep, results_7,'--',label=Ns_labels[1], lw=1.2)
    axs[2].plot(Tstep, results_8,':', label=Ns_labels[2], lw=1.2)
    axs[2].set_xscale('log')
    axs[2].set_xlabel(r'$t$')
    axs[2].legend(frameon=False, fontsize=9)
    axs[2].text(0.1,0.85 , r"(c)", fontsize=9, color="k", rotation=0)


    if save:
        plt.savefig(out_fig, dpi=dpi, bbox_inches="tight")
        plt.show()
    else:
        plt.show()

# =============================================================================
# Figure S7 — Generate CSV
# =============================================================================
def generate_figureS7_csv(out_csv, save=True):
    """
    Generate data for the overlaps of the no-click probability
    and store all results in a CSV file.

    The CSV contains:
    - panel index
    - panel title
    - dataset index
    - parameters gamma bar, r exponent, kappa bar, s exponent
    - system size N
    - numeric overlap for fast eigenmode
    - numeric overlap for slow eigenmode
    - analytic overlap for fast eigenmode
    - analytic overlap for slow eigenmode
    """
    Ns = np.logspace(1.5, 5, num=10, dtype=int)
    panels = [
        { 'title': 'Regime B',
        'datasets': [(1,1.15,1,0.1),(1,1.2,1,1.25),(1,1.4,1,0.7)],
        'analytic_alphas': [-1.0,-1.0,-1.0],
        'analytic_label': r'$\sim N^{-1}$',
        'analytic_alpha_func': None },
        { 'title': 'Regime C',
        'datasets': [(1,1.25,1,-0.5),(1,-0.2,1,-1.5),(1,0.5,1,-1.0)],
        'analytic_alphas': [-1.0,-1.0,-1.0],
        'analytic_label': r'$\sim N^{-1}$',
        'analytic_alpha_func': None },
        { 'title': 'Regime D',
        'datasets': [(1,0.1,1,-0.05),(1,-0.5,1,-0.2),(1,-0.5,1,-1.0)],
        'analytic_alphas': None,
        'analytic_label': r'$\sim N^{2r-2s-3}$',
        'analytic_alpha_func': lambda r,s: 2*r-2*s-3 },
        { 'title': 'Regime B',
        'datasets': [(1,1.15,1,0.1),(1,1.2,1,1.25),(1,1.4,1,0.7)],
        'analytic_alphas': None,
        'analytic_label': r'$\mathcal{O}(1)$',
        'analytic_alpha_func': None },
        { 'title': 'Regime C',
        'datasets': [(1,1.25,1,-0.5),(1,-0.2,1,-1.5),(1,0.5,1,-1.0)],
        'analytic_alphas': None,
        'analytic_label': r'$\mathcal{O}(1)$',
        'analytic_alpha_func': None },
        { 'title': 'Regime D',
        'datasets': [(1,0.1,1,-0.05),(1,-0.5,1,-0.2),(1,-0.5,1,-1.0)],
        'analytic_alphas': None,
        'analytic_label': r'$\mathcal{O}(1)$',
        'analytic_alpha_func': None }
    ]
    data_rows = []

    for p_idx, panel in enumerate(panels):
        title = panel.get("title", "")
        for d_idx, params in enumerate(panel.get("datasets", [])):
            gbar, r, kbar, s = params
            for N in Ns:
                gamma = gbar / (N ** r)
                kappa = kbar / (N ** s)
                H = core.Heff_matrix(gamma, kappa, N)
                lam, VR, VL = core.eig_biorth(H)
                Of, Os, Opm = core.overlaps_from_bi(VR, VL, N)
                OfA, OsA, _, _ = core.overlaps_analytic(gbar, r, kbar, s, N)

                data_rows.append([
                    int(p_idx),
                    title,
                    int(d_idx),
                    float(gbar),
                    float(r),
                    float(kbar),
                    float(s),
                    int(N),
                    float(Of),
                    float(Os),
                    float(OfA) if np.isfinite(OfA) else "",
                    float(OsA) if np.isfinite(OsA) else ""
                ])

    if not save:
        return data_rows

    with open(out_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "panel_index",
            "panel_title",
            "dataset_index",
            "gbar",
            "r",
            "kbar",
            "s",
            "N",
            "Of",
            "Os",
            "OfA",
            "OsA"
        ])
        for row in data_rows:
            # format floats with sufficient precision
            row_out = [
                int(row[0]),
                row[1],
                int(row[2]),
                f"{row[3]:.16g}",
                f"{row[4]:.16g}",
                f"{row[5]:.16g}",
                f"{row[6]:.16g}",
                int(row[7]),
                f"{row[8]:.16g}",
                f"{row[9]:.16g}",
                f"{row[10]:.16g}",
                f"{row[11]:.16g}"
            ]
            writer.writerow(row_out)


# =============================================================================
# Figure S7 — Plot from CSV
# =============================================================================
def plot_figureS7_from_csv(csv_path, fig_w=6.8, fig_h=4.54, dpi=600, save=False, out_fig=None):
    """
    Read CSV written by generate_figureS6_csv, and plot the figure.
    """
    panels = [
        { 'title': 'Regime B',
        'analytic_alphas': [-1.0,-1.0,-1.0],
        'analytic_label': r'$\sim N^{-1}$',
        'analytic_alpha_func': None },
        { 'title': 'Regime C',
        'analytic_alphas': [-1.0,-1.0,-1.0],
        'analytic_label': r'$\sim N^{-1}$',
        'analytic_alpha_func': None },
        { 'title': 'Regime D',
        'analytic_alphas': None,
        'analytic_label': r'$\sim N^{2r-2s-3}$',
        'analytic_alpha_func': lambda r,s: 2*r-2*s-3 },
        { 'title': 'Regime B',
        'analytic_alphas': None,
        'analytic_label': r'$\mathcal{O}(1)$',
        'analytic_alpha_func': None },
        { 'title': 'Regime C',
        'analytic_alphas': None,
        'analytic_label': r'$\mathcal{O}(1)$',
        'analytic_alpha_func': None },
        { 'title': 'Regime D',
        'analytic_alphas': None,
        'analytic_label': r'$\mathcal{O}(1)$',
        'analytic_alpha_func': None }
    ]
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        data = {}
        for row in reader:
            p_idx = int(row["panel_index"])
            d_idx = int(row["dataset_index"])
            key = (p_idx, d_idx)
            if key not in data:
                data[key] = {
                    "params": (
                        float(row["gbar"]),
                        float(row["r"]),
                        float(row["kbar"]),
                        float(row["s"])
                    ),
                    "N": [],
                    "Of": [],
                    "Os": [],
                    "OfA": [],
                    "OsA": [],
                    "title": str(row.get("panel_title"))
                }
            data[key]["N"].append(int(row["N"]))
            data[key]["Of"].append(float(row["Of"]) if row["Of"].strip() != "" else np.nan)
            data[key]["Os"].append(float(row["Os"]) if row["Os"].strip() != "" else np.nan)
            data[key]["OfA"].append(float(row["OfA"]) if row.get("OfA", "").strip() != "" else np.nan)
            data[key]["OsA"].append(float(row["OsA"]) if row.get("OsA", "").strip() != "" else np.nan)

    # Convert lists -> numpy arrays and sort by N
    for entry in data.values():
        order = np.argsort(entry["N"])
        entry["N"] = np.array(entry["N"])[order]
        entry["Of"] = np.array(entry["Of"])[order]
        entry["Os"] = np.array(entry["Os"])[order]
        entry["OfA"] = np.array(entry["OfA"])[order]
        entry["OsA"] = np.array(entry["OsA"])[order]

    fig = plt.figure(figsize=(fig_w, fig_h))
    plt.subplots_adjust(wspace=0.25, hspace=0.12, left=0.07, right=0.98, top=0.95, bottom=0.08)

    colors_class = {
        'D': ['#800000', '#FF0000', "#F78D66"],
        'B': ['#000080', "#607EDB", "#E6ADFA"],
        'C': ["#315C3F", "#37C468", "#A2BB76"]
    }

    stored_legends = [None, None, None]
    bottom_markers = ['o', 's', '*']

    for i, panel in enumerate(panels):
        ax = fig.add_subplot(2, 3, i + 1)
        class_idx = i % 3
        class_label = ['B', 'C', 'D'][class_idx]
        class_colors = colors_class[class_label]

        header_label = r'$(\bar\gamma,r,\bar\kappa,s)$'
        legend_handles = []
        legend_labels = []
        h_header = Line2D([0], [0], linestyle='None', marker='None', label=header_label)
        legend_handles.append(h_header)
        legend_labels.append(header_label)

        for j in [0,1,2]:
            key = (i, j)
            if key not in data:
                raise ValueError(f"Missing data for panel {i}, dataset {j} in CSV")

            gbar, r, kbar, s = data[key]["params"]
            Ns_data = data[key]["N"]
            Of_list = data[key]["Of"]
            Os_list = data[key]["Os"]
            OfA_list = data[key]["OfA"]
            OsA_list = data[key]["OsA"]

            color = class_colors[j % 3]

            if i < 3:
                mk = bottom_markers[j % len(bottom_markers)]
                ax.loglog(Ns_data, Of_list, marker=mk, linestyle='', markerfacecolor='none', markeredgecolor=color, color=color)
                # analytic overlap from analytic routine (solid line same color) if available
                if not np.all(np.isnan(OfA_list)):
                    ax.loglog(Ns_data, OfA_list, ls='-', color=color)

                # analytic power-law (dashed black) using analytic_alphas or analytic_alpha_func
                if panel.get('analytic_alphas') is not None:
                    alpha = panel['analytic_alphas'][j]
                    analytic_line = np.array(Ns_data, dtype=float) ** float(alpha)
                    ax.loglog(Ns_data, analytic_line, linestyle='--', linewidth=1.2, color='k')
                elif panel.get('analytic_alpha_func') is not None:
                    try:
                        alpha = panel["analytic_alpha_func"](s)
                    except TypeError:
                        try:
                            alpha = panel["analytic_alpha_func"](r, s)
                        except Exception:
                            alpha = None
                    if alpha is not None:
                        analytic_line = np.array(Ns_data, dtype=float) ** float(alpha)
                        ax.loglog(Ns_data, analytic_line, linestyle='--', linewidth=1.2, color='k')

                lab = f'({gbar},{r},{kbar},{s})'
                h = Line2D([0], [0], marker=mk, linestyle='None', markerfacecolor='none', markeredgecolor=color)
                legend_handles.append(h)
                legend_labels.append(lab)

            else:
                mk = bottom_markers[j % len(bottom_markers)]
                ax.plot(Ns_data, Os_list, marker=mk, linestyle='', markerfacecolor='none', markeredgecolor=color, color=color)
                #lab = f'({gbar},{r},{kbar},{s})'
                #h = Line2D([0], [0], marker=mk, linestyle='None', markerfacecolor='none', markeredgecolor=color)
                #legend_handles.append(h)
                #legend_labels.append(lab)

        if i < 3 and (panel.get('analytic_alphas') is not None or panel.get('analytic_alpha_func') is not None):
            h_dot = Line2D([0, 1], [0, 1], linestyle='--', color='black')
            legend_handles.append(h_dot)
            legend_labels.append(panel.get('analytic_label'))

        if i < 3:
            stored_legends[i] = (legend_handles, legend_labels)

        ax.set_xlabel(r'$N$')
        if i in [0, 1, 2]:
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_title(data[(i,0)]["title"])
            ax.set_xlabel('')
            ax.set_xticklabels([])
            ax.yaxis.set_major_locator(LogLocator(base=10.0))
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=12))
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.set_xscale("log")
            ax.axhline(1, color="gray", ls=":")
            ax.set_ylim(0.96, 1.005)
            top_idx = i - 3
            if stored_legends[top_idx] is not None:
                handles, labels = stored_legends[top_idx]
                ax.legend(handles, labels, handletextpad=0.4, borderpad=0.2, labelspacing=0.12, loc='lower right', frameon=True, fontsize=9)
            else:
                ax.legend(legend_handles, legend_labels, handletextpad=0.4, borderpad=0.2, labelspacing=0.12, loc='lower right', frameon=True, fontsize=9)

        if i == 0:
            ax.set_ylabel(r'$O_f$')
        elif i == 3:
            ax.set_ylabel(r'$O_s$')
        else:
            ax.set_ylabel('')

        ax.grid(False)

    # Save or show
    if save:
        if out_fig is None:
            raise ValueError("out_fig must be provided when save=True")
        plt.savefig(out_fig, dpi=dpi, bbox_inches="tight")
        plt.show()
    else:
        plt.show()



# =============================================================================
# Figure S8 — Generate CSV
# =============================================================================
def generate_figureS8_csv(
    out_csv,dt =0.01,  N = 1000000, Tcutoff = 10000, r = 1.0, s = 0.5,  save = True
):
    time_array = np.arange(0,Tcutoff,dt)

    gamma = N**(-r)
    kappa = N**(-s)  

    s_nr = np.real(core.surv_prob_theory_total(Tcutoff, dt, N, gamma, kappa))
    ub = 2*np.exp(-time_array/np.sqrt(N)) #upper bound
    lb = (2/3)*np.exp(-time_array/np.sqrt(N)) #lower bound


    data = np.column_stack([time_array,s_nr,ub,lb])
    
    # Header:
    header = (
        "T,"
        "P(t),"
        "Upper Bound,"
        "Lower Bound"  
    )

    # Save / return
    if save:
        np.savetxt(out_csv, data, delimiter=",", header=header, comments="", fmt="%.16g")
    else:
        return data

# =============================================================================
# Figure S8 — Plot from CSV
# =============================================================================
def plot_figureS8_from_csv(
    csv_path,
    out_fig,
    dpi=600, save = True
):
    
    df = pd.read_csv(csv_path, comment="#")
    time_array = df.iloc[:, 0].values #t
    s_nr = df.iloc[:, 1].values  #P(t)
    ub = df.iloc[:, 2].values  #Upper Bound
    lb = df.iloc[:, 3].values  #Lower bound
 

    width_mm = 179.832/2
    width_in = width_mm / 25.4
    panel_aspect = 3/4   
    panel_h = width_in * panel_aspect
    fig, ax = plt.subplots(figsize=(width_in,panel_h),dpi=200)
    style_axes(ax)

    ax.plot(time_array, s_nr, label = r'Exact: Eq. S15')
    ax.plot(time_array,ub,'-.', label = r'Upper Bound: Eq. S72')
    ax.plot(time_array,lb,'--', label = r'Lower Bound: Eq. S72')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.legend()
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$P\,(t)$')

    if save:
        plt.savefig(out_fig, dpi=dpi, bbox_inches="tight")
        plt.show()
    else:
        plt.show()

# =============================================================================
# Figure S9 — Generate CSV
# =============================================================================
def generate_figureS9_csv(
    out_csv,dt =0.01,gbar =0.9,rbar =0.0,kbar =1.0,s =0.5,threshold =0.001, save = True
):  
    #time in logspace to compute the non-reset time
    T_step = np.logspace(1,7,2500)
    Nlist = [1000,5000,10000]

    #time in linear space to see the effect of resetting
    T = np.arange(10,100,0.01)

    reset_time = np.zeros((len(Nlist),len(T)))
    non_reset_time = np.zeros(len(Nlist))

    for nn, N_val in enumerate(tqdm(Nlist)):
        gamma = gbar*N_val**(-(rbar + 1))
        kappa = kbar*N_val**(-s)  

        #non-reset time
        s_nr = np.real(core.surv_prob_theory_total_logspace(T_step, dt, N_val, gamma, kappa))
        non_reset_time[nn] = core.find_transition_point(s_nr,T_step, dt,threshold)    

        for tt,T_val in enumerate(T):
            # Computing no-click probability at time T and then computing m according to the main text.
            # See the section on "Impact of Resetting"
            S_T = np.real(core.surv_prob_theory_T(T_val, N_val, gamma, kappa))
 
            if S_T >=threshold:
                m = np.round(np.log(threshold)/np.log(S_T))
                reset_time[nn,tt] = m*T_val
            else:
                reset_time[nn,tt] = non_reset_time[nn]

    # Last column: only 3 values, rest NaN
    non_reset_col = np.full(T.shape, np.nan, dtype=float)
    non_reset_col[:len(non_reset_time)] = non_reset_time 

    # Data
    data = np.column_stack([T, reset_time[0], reset_time[1], reset_time[2], non_reset_col])

    # Header:
    header = (
        "Epoch Time T, "
        "reset_time (N=1000), "
        "reset_time (N=5000), "
        "reset_time (N=10000), "
        "non_reset_time (only first 3 rows)"
    )

 
    # Save / return
    if save:
        np.savetxt(out_csv, data, delimiter=",", header=header, comments="", fmt="%.16g")
    else:
        return data

# =============================================================================
# Figure S9 — Plot from CSV
# =============================================================================
def plot_figureS9_from_csv(
    csv_path,
    out_fig,
    dpi=600, save = True
):

    df = pd.read_csv(csv_path, comment="#")
    T = df.iloc[:, 0].values
    reset_time_0 = df.iloc[:, 1].values  # N = 1000
    reset_time_1 = df.iloc[:, 2].values  # N = 5000
    reset_time_2 = df.iloc[:, 3].values  # N = 10000
    non_reset_time = df.iloc[:, 4].values  

    width_mm = 179.832/2
    width_in = width_mm / 25.4
    panel_aspect = 3/8   
    panel_h = width_in * panel_aspect
    fig, ax = plt.subplots(figsize=(width_in,panel_h),dpi=200)
    style_axes(ax)

    ax.plot(T,reset_time_0/non_reset_time[0],'-',label = r'$N = 1000$')
    ax.plot(T,reset_time_1/non_reset_time[1],'--',label = r'$N = 5000$')
    ax.plot(T,reset_time_2/non_reset_time[2],'-.',label = r'$N = 10000$')
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$\tau_R/\tau$')
    ax.legend(frameon = False)
    ax.axhline(1,color = 'k', ls ='--')

    if save:
        plt.savefig(out_fig, dpi=dpi, bbox_inches="tight")
        plt.show()
    else:
        plt.show()

# =============================================================================
# Figure S10 — Generate CSV
# =============================================================================
def generate_figureS10_csv(
    out_csv,dt =0.001, r = 1.0, s = -1.0, threshold = 0.001, save = True
):

    Tstep = np.logspace(1,12,2000,dtype=int)

    Nlist = [100,1000,10000]
    
    # Epoch Time Exponent
    beta = np.arange(s, core.alpha_of_rs_scalar(r-1,s) + 0.1,0.1)  # rbar  = r -1 (See SM)

    reset_time = np.zeros((len(Nlist),len(beta)))
    non_reset_time =  np.zeros(len(Nlist))

    for nn,N in enumerate(Nlist):

        gamma = N**(-r)
        kappa = N**(-s)

        T_fast = N**(s)
        T_slow = N**(core.alpha_of_rs_scalar(r - 1,s)) # rbar  = r -1 (See SM)

        
        #Non Reset Time
        s_nr = core.surv_prob_theory_total_logspace(Tstep, dt, N, gamma, kappa)
        non_reset_time[nn] = core.find_transition_point(s_nr,Tstep, dt,threshold)

        for bb, beta_val in enumerate(beta):
            T = N**(beta_val)
            S_T = core.surv_prob_theory_T(T, N, gamma, kappa)
    
            if S_T >=10e-8:
                m = np.log(threshold)/np.log(S_T)
                reset_time[nn,bb] = m*T
            else:
                reset_time[nn,bb] = non_reset_time[nn]



    diff_N0 = np.log(reset_time[0]/non_reset_time[0])/np.log(Nlist[0])
    diff_N1 = np.log(reset_time[1]/non_reset_time[1])/np.log(Nlist[1])
    diff_N2 = np.log(reset_time[2]/non_reset_time[2])/np.log(Nlist[2])

    data = np.column_stack([beta , diff_N0, diff_N1, diff_N2])
  
   # Header:
    header = (
        "beta,"
        "Change in Exponent(N = 100),"
        "Change in Exponent(N = 1000),"
        "Change in Exponent(N = 10000)"  
    )

    # Save / return
    if save:
        np.savetxt(out_csv, data, delimiter=",", header=header, comments="", fmt="%.16g")
    else:
        return data


# =============================================================================
# Figure S10 — Plot from CSV
# =============================================================================
def plot_figureS10_from_csv(
    csv_path,
    out_fig,
    dpi=600, save = True
):

    df = pd.read_csv(csv_path, comment="#")
    beta = df.iloc[:, 0].values #beta
    diff_N0 = df.iloc[:, 1].values  #N = 100
    diff_N1 = df.iloc[:, 2].values  #N = 1000
    diff_N2 = df.iloc[:, 3].values  #N = 10000

    width_mm = 179.832/2
    width_in = width_mm / 25.4
    panel_aspect = 3/4   
    panel_h = width_in * panel_aspect
    fig, ax = plt.subplots(figsize=(width_in,panel_h),dpi=200)
    style_axes(ax)

    ax.plot(beta, diff_N2,'C3d',markerfacecolor='None',markersize = 3,label =f'$N = 10000$')
    ax.plot(beta, diff_N1,'C0s',markerfacecolor='None',markersize = 3,label =f'$N = 1000$')
    ax.plot(beta, diff_N0,'ko',markerfacecolor='None',markersize = 3,label =f'$N = 100$')
  
    ax.set_xlabel(r'Epoch Time exponent $\beta$')
    ax.set_ylabel(r'$\frac{\log\, (\tau_R/\tau)}{\log\, N}$')

    legend = ax.legend( loc='lower right',bbox_to_anchor=(0.96, 0.04),fontsize=6.5,frameon=True,fancybox=True,framealpha=0.9, borderpad=0.3,handlelength=1.2,handleheight=1.2,labelspacing=0.2)
    legend.get_frame().set_edgecolor('0.2')
    legend.get_frame().set_linewidth(0.8)


    if save:
        plt.savefig(out_fig, dpi=dpi, bbox_inches="tight")
        plt.show()
    else:
        plt.show()