"""
MIT License

This module implements the vanilla call option's pricer from:
    Thakoor, D., Bhuruth, M., 2018. Fast quadrature methods for options with discrete dividends. Journal of Computational
    and Applied Mathematics. doi:https://doi.org/10.1016/j.cam.2017.08.006
"""

import numpy as np
from blackscholes import callbs


def thakoor_bhuruth_call_eu(S: float, K: float, r: float, vol: float, T: float, t: np.array, D: np.array, N: int, Xi: float) -> float:
    mu = r - vol*vol/2
    m = len(t)
    time = np.array([0, *t, T])
    tau = np.diff(time)
    vecN = np.arange(N)
    thetai = (vecN + .5)/N*np.pi
    xi = np.cos(thetai)
    wi = np.zeros(N)
    x = N//2 + N % 2
    p = int(np.round((N-1)/2))
    h = np.arange(1, p+1)
    
    wi[:x] = [2 / N*(1 - 2 * np.sum(np.cos(2*h*theta)/(4*h**2-1)))
              for theta in thetai[:x]]
    wi[x:] = wi[N//2-1::-1]

    U = S * np.exp(mu*T + Xi*vol*np.sqrt(T))
    L = np.maximum(S*np.exp(mu*T - Xi*vol*np.sqrt(T)), np.min(D))
    xiLU = 1/2*(U-L)*xi+1/2*(U+L)

    xiLUplus = xiLU - D[-1]
    V_E = callbs(xiLUplus, K, r, vol, tau[-1])[0]
    for k in range(m-1, 0, -1):
        xiLUplus = xiLU - D[k-1]
        fnxkLU = np.array([(V_E/np.sqrt(2*np.pi*vol**2*tau[k])/xiLU*np.exp(-(
            np.log(xiLU/a) - mu*tau[k])**2/(2*vol**2*tau[k]))).dot(wi) for a in xiLUplus])
        V_E = np.exp(-r*tau[k])/2*(U-L)*fnxkLU

    V_E = V_E/(xiLU*np.sqrt(2*np.pi*vol**2 *
               tau[0]))*np.exp(-(np.log(xiLU/S) - mu*tau[0])**2/(2*vol**2*tau[0]))
    return np.exp(-r*tau[0])/2*(U-L)*V_E.dot(wi)
