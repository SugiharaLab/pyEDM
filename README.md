## Empirical Dynamic Modeling (EDM)
---
This package provides a Python/Pandas DataFrame toolset for [EDM analysis](http://deepeco.ucsd.edu/nonlinear-dynamics-research/edm/ "EDM @ Sugihara Lab").  Introduction and documentation are are avilable [online](https://sugiharalab.github.io/EDM_Documentation/ "EDM Docs"), or in the package [API docs](https://github.com/SugiharaLab/pyEDM/blob/master/doc/pyEDM.pdf "pyEDM API"). A Jupyter notebook interface is available at [jpyEDM](https://github.com/SugiharaLab/jpyEDM#empirical-dynamic-modeling-edm-jupyter-notebook).

Functionality includes:
* Simplex projection ([Sugihara and May 1990](https://www.nature.com/articles/344734a0))
* Sequential Locally Weighted Global Linear Maps (S-Map) ([Sugihara 1994](https://royalsocietypublishing.org/doi/abs/10.1098/rsta.1994.0106))
* Multivariate embeddings ([Dixon et. al. 1999](https://science.sciencemag.org/content/283/5407/1528))
* Convergent cross mapping ([Sugihara et. al. 2012](https://science.sciencemag.org/content/338/6106/496))
* Multiview embedding ([Ye and Sugihara 2016](https://science.sciencemag.org/content/353/6302/922))

---
## Installation

### Python Package Index (PyPI)
Certain MacOS, Linux and Windows platforms are supported with prebuilt binary distributions hosted on PyPI [pyEDM](https://pypi.org/project/pyEDM/) and can be installed with the Python pip module: `python -m pip install pyEDM`

---
## Usage
Examples can be executed in the python command line:
```python
>>> import pyEDM
>>> pyEDM.Examples()
```

---
### References
Sugihara G. and May R. 1990.  Nonlinear forecasting as a way of distinguishing 
chaos from measurement error in time series. [Nature, 344:734–741](https://www.nature.com/articles/344734a0).

Sugihara G. 1994. Nonlinear forecasting for the classification of natural 
time series. [Philosophical Transactions: Physical Sciences and 
Engineering, 348 (1688) : 477–495](https://royalsocietypublishing.org/doi/abs/10.1098/rsta.1994.0106).

Dixon, P. A., M. Milicich, and G. Sugihara, 1999. Episodic fluctuations in larval supply. [Science 283:1528–1530](https://science.sciencemag.org/content/283/5407/1528).

Sugihara G., May R., Ye H., Hsieh C., Deyle E., Fogarty M., Munch S., 2012.
Detecting Causality in Complex Ecosystems. [Science 338:496-500](https://science.sciencemag.org/content/338/6106/496).

Ye H., and G. Sugihara, 2016. Information leverage in interconnected 
ecosystems: Overcoming the curse of dimensionality. [Science 353:922–925](https://science.sciencemag.org/content/353/6302/922).
