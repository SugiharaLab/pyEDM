'''Python tools for EDM'''
# import EDM functions
from .API      import MakeBlock, Embed, Simplex, SMap, CCM, Multiview
from .API      import EmbedDimension, PredictInterval, PredictNonlinear
from .API      import PredictExclusionRadius
from .AuxFunc  import Examples, PlotObsPred, PlotCoeff, ComputeError
from .AuxFunc  import SurrogateData
from .LoadData import sampleData
# EDM applications
from .apps.CrossMap_Matrix  import CrossMap_Matrix
from .apps.CCM_Matrix       import CCM_Matrix
from .apps.EmbedDim_Columns import EmbedDim_Columns

__version__     = "2.5.5"
__versionDate__ = "2026-07-10"
