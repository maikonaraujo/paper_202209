#  Brazilian listed options with discrete dividends and the fast Laplace transform

This project contains the source code to generate the tables and figures presented in the 
article "Brazilian listed options with discrete dividends and the fast Laplace transform".

## Code structure

```python
fltdiv.py:
"""
This module implements the routines to price call options using the fast Laplace transform method:
  
  Functions:
    - call_eu_strike: Vanilla call options
    - call_eu_strike_adj: Vanilla call options with strike dividend adjustment.
"""

flt.py:
"""
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

relatedwork.py:
"""
This module implements the vanilla call option's pricer from:
    Thakoor, D., Bhuruth, M., 2018. Fast quadrature methods for options with discrete dividends. Journal of Computational
    and Applied Mathematics. doi:https://doi.org/10.1016/j.cam.2017.08.006
"""

blackscholes.py: 
"""
This module implements the vanilla call options from Black Scholes model.
"""


results.py:
"""
This module implements the main program which generates the tables
and figures from article:
    Brazilian listed options with discrete dividends and the fast Laplace transform. (Araujo, Maikon)
"""
```

## Windows configuration

To setup a python virtual environment, run:

```
python -m venv venv
.\venv\Scripts\activate
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

## Linux configuration

To setup a python virtual environment, run:

```
python -m venv venv
source ./venv/bin/activate
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

## Showing figures

Note that figure 2 (accuracy plots) takes some minutes to run.

Run:
```
python results.py --figure 1 
python results.py --figure 2
```
or 
```
python results.py --figure 1 2 
```
if you wish to generate a pdf file with the images run:

```
python results.py --figure 1 2  --output fig1.pdf fig2.pdf
```

## Showing tables

Run:
```
python results.py --table 1 
python results.py --table 2
python results.py --table 1 2
```

## The usage output the results.py program

```
>> python results.py --help

usage: results.py [-h] [--figure N [N ...]] [--output F [F ...]]
                  [--table N [N ...]] [--fmt fmt [fmt ...]]

        Results program for the paper:
        Brazilian listed options with discrete dividends and the fast Laplace transform.
        Author: Maikon Araujo 2022.
        

options:
  -h, --help           show this help message and exit

figures:
  --figure N [N ...]   Shows a figure from the paper.
  --output F [F ...]   Output files to save, list one for each figure in '--
                       figure' option.

tables:
  --table N [N ...]    Shows a table from the paper.
  --fmt fmt [fmt ...]  Format to display table, can be ['table', 'latex'].
```