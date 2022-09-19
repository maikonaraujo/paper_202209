"""
Copyright 2022 Maikon Araujo.
MIT License

This module implements the routines to price call options using the fast Laplace transform method:

Functions:
---------
    call_eu_strike: Vanilla call options
    call_eu_strike_adj: Vanilla call options with strike dividend adjustment.
"""
from typing import Tuple
import numpy as np
import scipy.interpolate as spl
from blackscholes import callbs
import scipy.fft as fft
import flt


def call_eu_strike_adj(S: float, K: np.array, r: float, vol: np.array, T: float, t: np.array, D: np.array, N: int = 2**12, eps: float = 1e-3) -> Tuple[np.array, np.array, np.array, np.array]:
    """    
        Returns the call option price and its Greeks for each strike/vol in np.array vol.
        It calls call_eu, replacing the strike by: K - np.sum(D).

        Returns: (price, delta, gamma, theta)

        Parameters
        -----------
            S : initial spot price.
            K : np.array of strikes to calculate the call price.
            r : risk free interest rate.
            vol: np.array of volatility for each strike.
            T: time to maturity.
            t : np.array with ex-dates for each dividend.
            D: np.array with discrete dividend for each ex-date in t.
            N: discretization size for Laplace transform (should be a power of 2)
            eps: convergence ratio for Laplace dump such that: exp(-lx) -> eps as x -> infinity.

        Notes
        ------
        K and vol: should have the same length or one of them should be of length 1.
    """
    return call_eu(S, K - np.sum(D), r, vol, T, t, D, N, eps)


def call_eu(S: float, K: np.array, r: float, vol: np.array, T: float, t: np.array, D: np.array, N: int = 2**12, eps: float = 1e-3) -> Tuple[np.array, np.array, np.array, np.array]:
    """    
        Returns the call option price and its Greeks for each strike/vol in np.array vol.

        Returns: (price, delta, gamma, theta)

        Parameters
        -----------
            S : initial spot price.
            K : np.array of strikes to calculate the call price.
            r : risk free interest rate.
            vol: np.array of volatility for each strike.
            T: time to maturity.
            t : np.array with ex-dates for each dividend.
            D: np.array with discrete dividend for each ex-date in t.
            N: discretization size for Laplace transform (should be a power of 2)
            eps: convergence ratio for Laplace dump such that: exp(-lx) -> eps as x -> infinity.

        Notes
        ------
        K and vol: should have the same length or one of them should be of length 1.
    """
    N = fft.next_fast_len(N, True)
    while N % 2:
        N = fft.next_fast_len(N+1, True)

    K = K.reshape(len(K), 1)
    vol = vol.reshape(len(vol), 1)
    lnS, lnK = np.log(S), np.log(K)
    L = 2 * (7.5) * np.max(vol) * np.sqrt(T)
    dx = L / N
    vol2 = vol*vol
    rvol2 = r-vol2/2
    x0 = lnS - lnK + rvol2*T
    x = np.arange(-N/2, N/2) * dx + x0

    D = np.flip(D)
    tau = np.flip(T - t)

    fx = K * np.maximum(np.exp(x) - 1, 0.0)

    l = _lamb_max_min(fx, K, N, eps)
    s = flt.rfltfreq(l, N, dx)
    svol2 = (s*s) * vol2
    expl = flt.exp_l(l, N)

    def solvepde(t, fx0):
        fs0 = flt.rflt(expl, fx0, expl=True)
        fs = fs0 * np.exp(svol2 * t/2)
        return flt.irflt(expl, fs, expl=True), fs

    tprev = 0
    for i, (d, t) in enumerate(zip(D, tau)):
        dt, tprev = t - tprev, t
        St = K * np.exp(x - rvol2*t)
        fx = solvepde(dt, fx)[0] if i > 0 else callbs(
            St, K, r, vol, dt, False)*np.exp(r*t)
        xeps = np.log1p(d/St)
        fx = _splinterp(x, x + xeps, fx)

    fx, fs = solvepde(T - tprev, fx)
    sfs = s*fs
    s2fs = s*sfs
    B = np.exp(-r*T)
    dfdx = flt.irflt(expl, sfs, expl=True)
    d2fdx2 = flt.irflt(expl, s2fs, expl=True)
    price = fx[..., N//2] * B
    delta = dfdx[..., N//2]/S * B
    gamma = (d2fdx2[..., N//2] - dfdx[..., N//2])/S/S * B
    theta = price * r - (S*S*vol2.ravel())*gamma/2 - delta*S*r
    return price, delta, gamma, theta


def _lamb_max_min(fx, K, N, eps):
    vd = fx[..., 0].reshape(len(K), 1)
    vu = fx[..., -1].reshape(len(K), 1)
    vd[vd < eps] = eps
    return np.log(vu/vd) * N/(N-1)


def _splinterp(x, xb, fb):
    if xb.ndim == fb.ndim == 1:
        return spl.splev(x, spl.splrep(xb, fb), ext=1)

    if x.shape[0] == xb.shape[0] == fb.shape[0]:
        return np.array([spl.splev(xi, spl.splrep(xbi, fbi), ext=1) for xi, xbi, fbi in zip(x, xb, fb)])

    if xb.shape == fb.shape:
        return np.array([spl.splev(x, spl.splrep(xbi, fbi), ext=1) for xbi, fbi in zip(xb, fb)])

    if x.shape[0] == fb.shape[0]:
        return np.array([spl.splev(xp, spl.splrep(xb, fbi), ext=1) for xp, fbi in zip(x, fb)])

    return np.array([spl.splev(x, spl.splrep(xb, fbi), ext=1) for fbi in fb])
