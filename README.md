## Empirical Dynamic Modeling (EDM)
---
This package provides a Python/Pandas DataFrame interface to the [cppEDM](https://github.com/SugiharaLab/cppEDM) library for [EDM analysis](http://deepeco.ucsd.edu/nonlinear-dynamics-research/edm/).  [Documentation](https://github.com/SugiharaLab/pyEDM/blob/master/doc/pyEDM.pdf) is available at [pyEDM](https://github.com/SugiharaLab/pyEDM).

Functionality includes:
* Simplex projection (Sugihara and May 1990)
* Sequential Locally Weighted Global Linear Maps (S-map) (Sugihara 1994)
* Multivariate embeddings (Dixon et. al. 1999)
* Convergent cross mapping (Sugihara et. al. 2012)
* Multiview embedding (Ye and Sugihara 2016)

---
## Installation

### Python Package Index (PyPI)
Certain Mac OSX and Windows platforms are supported with prebuilt binary distributions and can  be installed using the Python pip module.  The module is located at [pypi.org/project/pyEDM](https://pypi.org/project/pyEDM/).

Installation can be executed as: `python -m pip install pyEDM`


### Manual Install
Unfortunately, we do not have the resources to provide pre-built binary distributions for all computer platforms.  In this case the user is required to first build the cppEDM library on their machine, and then install the Python package using pip.  On OSX and Linux this requires g++, on Windows, Microsoft Visual Studio Compiler (MSVC) which can be obtained from `Build Tools for Visual Studio 2019`. Only the Windows SDK is needed.

Note that the [LAPACK](http://www.netlib.org/lapack/explore-html/index.html) library is required to build cppEDM.

#### OSX and Linux
1. Download pyEDM: `git clone https://github.com/SugiharaLab/pyEDM`
2. Build cppEDM library: `cd pyEDM/cppEDM/src; make`
3. Build and install package: `cd ../..; python -m pip install . --user --trusted-host pypi.org`

#### Windows
1. Download pyEDM: `git clone https://github.com/SugiharaLab/pyEDM`
2. Build cppEDM library: `cd pyEDM\cppEDM\src; nmake /f makefile.windows`
3. Build and install package: `cd ..\..; python -m pip install . --user --trusted-host pypi.org`

---
## Usage
Example usage at the python prompt:
```python
>>> import pyEDM
>>> pyEDM.Examples()
```

---
### References
Sugihara G. and May R. 1990.  Nonlinear forecasting as a way of distinguishing 
chaos from measurement error in time series. Nature, 344:734–741.

Sugihara G. 1994. Nonlinear forecasting for the classification of natural 
time series. Philosophical Transactions: Physical Sciences and 
Engineering, 348 (1688) : 477–495.

Dixon, P. A., M. Milicich, and G. Sugihara, 1999. Episodic fluctuations in larval supply. Science 283:1528–1530.

Sugihara G., May R., Ye H., Hsieh C., Deyle E., Fogarty M., Munch S., 2012.
Detecting Causality in Complex Ecosystems. Science 338:496-500.

Ye H., and G. Sugihara, 2016. Information leverage in interconnected 
ecosystems: Overcoming the curse of dimensionality. Science 353:922–925.
