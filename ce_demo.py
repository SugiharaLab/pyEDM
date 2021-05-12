# This file is just for hackish demonstration of conditional embeddings, do not
# include when merging to main branch. 

import pyEDM
import pdb
f=pdb.set_trace

import numpy as np
from pandas import DataFrame

circle = pyEDM.sampleData['circle']
sardine = pyEDM.sampleData['sardine_anchovy_sst']

validLib = np.ones(200)
validLib[20:50] = 0

pyEDM.Simplex(dataFrame=circle,columns="x y",target="x",E=5,lib="1 100",
              pred="101 198",validLib=None,embedded=True)
