"""
MIT License

This module implements the vanilla call options from Black Scholes model.
"""
from typing import Union, Tuple
import numpy as np
from scipy.stats import norm


def optionbs(S: Union[float, np.array], K: Union[float, np.array], r: float, vol: Union[float, np.array], T: float, phi: float, greeks: bool = True) -> Union[np.array, Tuple[np.array, np.array, np.array]]:
    N = norm.cdf
    B = np.exp(-r*T)
    F = S / B
    w = vol*np.sqrt(T)
    d1 = np.log(F/K)/w + w/2
    d2 = d1 - w
    delta = phi*N(phi*d1)
    p = S*delta - phi*K * B * N(phi*d2)
    if not greeks:
        return p
    return p, delta, norm.pdf(d1)/S/w


def callbs(S, K, r, vol, T, greeks=True):
    return optionbs(S, K, r, vol, T, 1, greeks)


def putbs(S, K, r, vol, T, greeks=True):
    return optionbs(S, K, r, vol, T, -1, greeks)
