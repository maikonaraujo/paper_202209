"""
Copyright 2022 Maikon Araujo.
MIT License

This module implements the main program which generates the tables
and figures from article:
    Brazilian listed options with discrete dividends and the fast Laplace transform. (Araujo, Maikon)
"""
import argparse
import matplotlib.pyplot as plt
from matplotlib import gridspec
from fltdiv import call_eu, call_eu_strike_adj
from relatedwork import thakoor_bhuruth_call_eu
from blackscholes import callbs
import scipy.fft as fft
import itertools as it
from timeit import Timer
import numpy as np
import pandas as pd
import seaborn as sn

RESET = u'\u001b[0m'
BLUE = u'\u001b[36m'
CYAN = u'\u001b[36;1m'

plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.family': 'Times New Roman'})
plt.rcParams.update({'font.stretch': 'ultra-expanded'})
plt.rcParams.update({'mathtext.fontset': 'dejavuserif'})
plt.rcParams.update({'axes.spines.top': True})
plt.rcParams.update({'axes.spines.right': True})


def D(f, h):
    return lambda x: (f(x+h) - f(x-h))/h/2


def _bs_vanilla_column(y, bs, la, axe, axl, l, has_legend):
    c = sn.color_palette("Paired")

    err = la - bs

    ler,  = axe.plot(y, err, color=c[5], lw=1.5, label='Error')
    lbs,  = axl.plot(y, bs, color=c[0], lw=1.5, label='Black Scholes')
    lb,  = axl.plot(y, la,  'x', color=c[3], lw=1.5, ms=6, label='FLT')

    plt.setp(axl.get_xticklabels(), visible=False)
    if has_legend:
        axl.legend((lb, lbs, ler), (f'FLT', f'Black-Scholes',
                   f'Error'), loc='best', ncol=1, frameon=False)

    axl.set_title(f'{l}')
    return (lb, lbs, ler)


def fig_bs_vanilla(r, yb, sig, T, m, fname=None):
    y = np.linspace(yb[0], yb[1], 25)
    S = 100
    K = S * np.exp(-y)
    N = 2**m

    laplace = call_eu(S,
                      K, r,
                      np.array([sig]), T,
                      np.array([]),
                      np.array([]),
                      N, 1e-3)

    blacks = callbs(S, K, r, sig, T)

    fig = plt.figure(figsize=(11, 5), tight_layout=True)
    gs = gridspec.GridSpec(2, 3, height_ratios=[4, 1])
    ax = []
    for i in range(3):
        axv = plt.subplot(gs[0, i])
        axe = plt.subplot(gs[1, i], sharex=axv)
        axe.set_xlabel('moneyness $y = ln(S/K)$')
        ax = [*ax, [axv, axe]]

    h = [_bs_vanilla_column(y, blacks[i]/s, laplace[i]/s, ax[i][1], ax[i][0], l, hl)
         for i, l, s, hl in
         zip(range(3), [r'$C(y)$', r'$\Delta(y)$', r'$\Gamma(y)$'], [1, 1, 1], [True, False, False])]

    if fname:
        plt.savefig(fname, dpi=600)
    else:
        plt.show()


def fig_accuracy_tb_1(t, d, K, S, r, vol, T, tol, ax):
    def next_flt(N):
        while True:
            N = fft.next_fast_len(N+4, True)
            while N % 2 == 1:
                N = fft.next_fast_len(N+4, True)
            yield N

    def run_time(fun, target_secs):
        t = Timer(fun)
        N, n = target_secs
        times = t.repeat(N, n)
        return min(times)/n, max(times)/n

    for i, K in enumerate(K):
        p_flt_target = call_eu_strike_adj(
            S, np.array([K]), r, vol, T, t, d, 2**15)[0][0]

        def cond_flt(v): return np.abs(v-p_flt_target) > tol
        p_tb_target = thakoor_bhuruth_call_eu(
            S, K - np.sum(d), r, vol, T, t, d, 1000, 6.5)

        def cond_tb(v): return np.abs(v-p_tb_target) > tol
        print((i, K, p_flt_target, np.abs(p_flt_target - p_tb_target),
              np.abs(p_flt_target - p_tb_target) < tol/10))

        p_flt = np.fromiter(it.takewhile(cond_flt, (call_eu_strike_adj(S, np.array(
            [K]), r, vol, T, t, d, N)[0][0] for N in next_flt(2))), dtype=float)
        p_tb = np.fromiter(it.takewhile(cond_tb, (thakoor_bhuruth_call_eu(
            S, K - np.sum(d), r, vol, T, t, d, N, 6.5) for N in it.count(10, 2))), dtype=float)
        N_flt = np.array(list(v for i, v in it.takewhile(
            lambda i: i[0] < p_flt.shape[-1], enumerate(next_flt(3)))))
        N_tb = np.array(list(v for i, v in it.takewhile(
            lambda i: i[0] < p_tb.shape[-1], enumerate(it.count(10)))))
        t_flt = [run_time(lambda: call_eu_strike_adj(S, np.array(
            [K]), r, vol, T, t, d, N)[0], (10, 25)) for N in N_flt]
        t_tb = [run_time(lambda: thakoor_bhuruth_call_eu(
            S, K - np.sum(d), r, vol, T, t, d, N, 6.5), (10, 25)) for N in N_tb]

        e_flt = np.abs(p_flt - p_flt_target)
        x_flt = np.array([mi for mi, _ in t_flt])*1e3
        s_flt = np.array([5*mi/ma for mi, ma in t_flt])**2
        ax[i].scatter(x_flt[e_flt != 0], -np.log(e_flt[e_flt != 0]),
                      color=sn.color_palette("Paired")[3], alpha=0.5, s=s_flt[e_flt != 0], label='FLT')

        e_tb = np.abs(p_tb - p_tb_target)
        x_tb = np.array([mi for mi, _ in t_tb])*1e3
        s_tb = np.array([5*mi/ma for mi, ma in t_tb])**2
        ax[i].scatter(x_tb[e_tb != 0], -np.log(e_tb[e_tb != 0]),
                      color=sn.color_palette("Paired")[9], alpha=0.5, s=s_tb[e_tb != 0], label='TB')


def fig_accuracy_tb(tds, tol, fname):
    K = np.array([70, 100, 130])
    fig, ax = plt.subplots(nrows=len(tds), ncols=len(K), figsize=(11, 5))
    fig.tight_layout()
    r = [fig_accuracy_tb_1(np.array(t), np.array(d), K, 100, 0.06, np.array(
        [.3]), 1, tol, ax[i]) for i, (t, d) in enumerate(tds)]
    for i, _ in enumerate(tds):
        for j, _ in enumerate(K):
            ax[i, j].legend(loc='best')

    for u in ax[-1]:
        u.set_xlabel('miliseconds')
    for i, k in enumerate(K):
        ax[0, i].set_title(f'K = {k:.0f}')

    from string import ascii_lowercase
    for i, (l, _) in enumerate(zip(ascii_lowercase, tds)):
        ax[i, 0].set_ylabel(f'({l}) accuracy')

    if fname:
        plt.savefig(fname, dpi=600)
    else:
        plt.show()


def figure_cmd(figs, fnames):
    cmds = {'1': lambda f: fig_bs_vanilla(0.06, [-0.005, 0.005], 0.01, 5/360, 10, f),
            '2': lambda f: fig_accuracy_tb([([0.2, 0.4, 0.6, .8], [4, 5, 6, 3]), ([.2, .6], [9, 9])], 1e-9, f)
            }

    def error_msg(fig): return lambda f: print(f'Figure {fig} not available.')

    if fnames is None:
        [cmds.get(f, error_msg(f))(None) for f in figs]
        return

    if len(fnames) != len(figs):
        comp = 'Less' if len(figs) < len(fnames) else 'More'
        print(f'{comp} figures then files to generate.')
        return

    print('Saving figures:')
    for f, o in zip(figs, fnames):
        print(f'\t{BLUE}{f} -> {o}{RESET}')
        cmds.get(f, error_msg(f))(o)


def table_t_d_k_one_dividend(t, d, K, S, r, s, T, N):
    cols = ['t', 'D',
            *[f'L{k:.0f}' for k in K],
            *[f'TB{k:.0f}' for k in K]]

    df = pd.DataFrame(data=[
        [ti, di,
         *call_eu_strike_adj(S, K, r, s, T,
                             np.array([ti]), np.array([di]), N)[0],
         *[thakoor_bhuruth_call_eu(S, k-di, r, s[0], T, np.array([ti]), np.array([di]), 5000, 6.5) for k in K]]
        for di in d for ti in t], columns=cols)

    N = len(K)
    for k in K:
        df[f'E{k:.0f}'] = np.abs(df[f'L{k:.0f}'] - df[f'TB{k:.0f}'])

    cols = df.columns

    df = df[['t', 'D', *[u for sub in [cols[i+2::N]
                                       for i in range(N)] for u in sub]]]

    fmt = {**{c: '{:.2E}' for c in [f'E{k:.0f}' for k in K]},
           **{c: '{:.4f}' for c in [f'L{k:.0f}' for k in K]},
           **{c: '{:.4f}' for c in [f'TB{k:.0f}' for k in K]},
           't': '{:.4f}', 'D': '{:.0f}'}

    return df, df.style.format(fmt).hide(axis='index')


def table_greeks_1(t, d, K, S, r, vol, T, N):
    v = call_eu_strike_adj(S, K, r, vol, T, t, d, N)[:3]
    delta_num = D(lambda S: call_eu_strike_adj(
        S, K, r, vol, T, t, d, N)[0], S*1e-4)(S)
    gamma_num = D(lambda S: call_eu_strike_adj(
        S, K, r, vol, T, t, d, N)[1], S*1e-4)(S)

    cols = ['K', 'P', 'D', 'G', 'Dn', 'Gn']

    df = pd.DataFrame(data=np.vstack(
        (K, *v, delta_num, gamma_num)).T, columns=cols)

    df['D'] *= 100
    df['Dn'] *= 100
    df['G'] *= 10000
    df['Gn'] *= 10000

    df = df[['K', 'P', 'D', 'Dn', 'G', 'Gn']]

    return df


def table_greeks(tds):
    df = pd.concat((table_greeks_1(np.array(t), np.array(d),  np.array(
        [70, 100, 130]), 100, 0.06, np.array([.3]), 1, 1024) for t, d in tds), axis=1)
    fmt = {'K': '{:.0f}', 'P': '{:.4f}', 'D': '{:.2f}',
           'G': '{:.2f}', 'Dn': '{:.2f}', 'Gn': '{:.2f}'}
    return df, df.style.format(fmt).hide(axis='index')


def table_cmd(tbl, fmt):
    cmds = {'1': lambda: table_t_d_k_one_dividend([1e-4, 0.5, 1 - 1e-4], [7.0, 20, 50], np.array(
            [70.0, 100, 130]), 100.0, 0.06, np.array([0.30]), 1, 2**13),
            '2': lambda: table_greeks([([0.2, 0.4, 0.6, .8], [4, 5, 6, 3]), ([.2, .6], [9, 9])])
            }

    def error_msg(t): return lambda: (print(f'Table {t} not available.'), None)

    if fmt is None:
        [print(cmds.get(f, error_msg(f))()[0]) for f in tbl]
        return

    if len(tbl) != len(fmt):
        comp = 'Less' if len(tbl) < len(fmt) else 'More'
        print(f'{comp} tables then fmt have been specified.')
        return

    for t, f in zip(tbl, fmt):
        df, s = cmds.get(t, error_msg(t))()
        if f == 'table':
            print(df)
        elif f == 'latex':
            print(s.to_latex())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=f"""
        Results program for the paper:
        {BLUE}Brazilian listed options with discrete dividends and the fast Laplace transform.{RESET}
        Author: {CYAN}Maikon Araujo{RESET} 2022.
        """)

    figgrp = parser.add_argument_group('figures')
    figgrp.add_argument('--figure', metavar='N', type=str,
                        nargs='+', help="Shows a figure from the paper.")
    figgrp.add_argument('--output', metavar='F', type=str, nargs='+',
                        help="Output files to save, list one for each figure in '--figure' option.")

    tblgrp = parser.add_argument_group('tables')
    tblgrp.add_argument('--table', metavar='N', type=str,
                        nargs='+', help="Shows a table from the paper.")

    fmt_choices = ['table', 'latex']
    tblgrp.add_argument('--fmt', metavar='fmt', type=str, choices=['table', 'latex'],
                        nargs='+', help=f"Format to display table, can be {fmt_choices}.")

    args = parser.parse_args()

    if args.figure is not None:
        figure_cmd(args.figure, args.output)
        parser.exit()

    if args.table is not None:
        table_cmd(args.table, args.fmt)
        parser.exit()

    parser.print_help()
