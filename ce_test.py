# This file is just for hackish demonstration of conditional embeddings, do not
# include when merging to main branch. 

import pyEDM
import pdb
f=pdb.set_trace

import numpy as np
from pandas import DataFrame

circle = pyEDM.sampleData['circle']
sardine = pyEDM.sampleData['sardine_anchovy_sst']

ce = [
      (circle.eval("10<index<100"),
       circle.eval("not(1<index<100)")),

      (circle.eval("10<index<101"),
       circle.eval("not(1<index<100)"))
     ]

pyEDM.Simplex(dataFrame=circle,columns="x y",target="x",E=5,lib="1 100",
              pred="101 198")

#pyEDM.SMap(dataFrame=circle,columns="x y",target="x",E=5,lib="1 100",
#        embedded=True, pred="101 198")

#df = pyEDM.CCM( "", "",
#                  sardine, "./", "",
#                  3, 0, 0, -1, 0, "anchovy", "np_sst",
#                  "10 75 5", 1, False, False, 0, False, False)


