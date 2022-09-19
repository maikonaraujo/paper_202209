"""
Copyright 2022 Maikon Araujo.
MIT License

This module implements the routines for the fast Laplace transform:

Functions:
---------
    flt: Fast Laplace transform.
    iflt Inverse fast Laplace transform.
    fltfreq: Frequencies s_k for the fast Laplace transform.
    rflt: Real fast Laplace transform.
    irflt Inverse real fast Laplace transform.
    rfltfreq: Frequencies s_k for the real fast Laplace transform.

"""
from typing import Union
import numpy as np
import scipy.fft as fft


def exp_l(l: float, N: int) -> np.array:
    """
        Convenient method to avoid calculating exp(l * arange(N)/N) in
        all fast Laplace Transform methods.
    """
    return np.exp(l*np.arange(N)/N)


def rflt(l: Union[float, np.array], f: np.array, expl: bool = False, axis: int = -1) -> np.array:
    """
    Real fast Laplace transform of function: f[k], where k = Imag(s)/(2*pi).

    Parameters
    ----------
        l: lambda dump factor such that l = Real(s).
        f: f in time domain f[n].
        expl: If True, assumes that l is the result of exp_l function, defaults to False.
    """
    N = f.shape[axis]
    if expl:
        return fft.rfft(f/l, axis=axis, overwrite_x=True)
    return fft.rfft(np.exp(-l * np.arange(N)/N) * f, axis=axis, overwrite_x=True)


def rfltfreq(l: float, n: int, dx: float = 1) -> np.array:
    """
    Returns the s Laplace argument, such that:
        s = (l + 2j * pi* k)/(n*dx), 
    for the real transform rflt.

    Parameters
    ----------
        l: lambda dump factor such that l = Real(s).
        n: number of points in f
        dx: samples size.
    """
    e = fft.rfftfreq(n)
    w = 2j * np.pi * e
    return (l/n + w)/dx


def irflt(l: Union[float, np.array], fs: np.array, expl: bool = False, axis: int = -1) -> np.array:
    """
    Real inverse fast Laplace transform of function: $f[n]$.

    Parameters
    ----------
        l: lambda dump factor such that l = Real(s).
        f: f in frequency domain f[k].
        expl: If True, assumes that l is the result of exp_l function, defaults to False.
    """
    fh = fft.irfft(fs, axis=axis)
    N = fh.shape[axis]
    if expl:
        return fh * l
    return fh * np.exp(l * np.arange(N)/N)


def flt(l: Union[float, np.array], f: np.array, expl: bool = False, axis: int = -1) -> np.array:
    """
    Fast Laplace transform of function: f[k], where k = Imag(s)/(2*pi).

    Parameters
    ----------
        l: lambda dump factor such that l = Real(s).
        f: f in time domain f[n].
        expl: If True, assumes that l is the result of exp_l function, defaults to False.
    """
    N = f.shape[axis]
    if expl:
        return fft.fft(f/l, axis=axis, overwrite_x=True)
    return fft.fft(np.exp(-l * np.arange(N)/N) * f, axis=axis, overwrite_x=True)


def fltfreq(l: float, n: int, dx: float = 1) -> np.array:
    """
    Returns the s Laplace argument, such that:
        s = (l + 2j * pi* k)/(n*dx), 
    for the transform flt.

    Parameters
    ----------
        l: lambda dump factor such that l = Real(s).
        n: number of points in f
        dx: samples size.
    """
    e = fft.fftfreq(n)
    w = 2j * np.pi * e
    return (l/n + w)/dx


def iflt(l: Union[float, np.array], fs: np.array, expl: bool = False, axis: int = -1) -> np.array:
    """
    Inverse fast Laplace transform of function: $f[n]$.

    Parameters
    ----------
        l: lambda dump factor such that l = Real(s).
        f: f in frequency domain f[k].
        expl: If True, assumes that l is the result of exp_l function, defaults to False.
    """
    fh = fft.ifft(fs, axis=axis)
    N = fh.shape[axis]
    if expl:
        return fh * l
    return fh * np.exp(l * np.arange(N)/N)
