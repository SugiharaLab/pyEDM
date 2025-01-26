'''Python tools for EDM'''
# import EDM functions
from .API      import MakeBlock, Embed, Simplex, SMap, CCM, Multiview
from .API      import EmbedDimension, PredictInterval, PredictNonlinear
from .AuxFunc  import Examples, PlotObsPred, PlotCoeff, ComputeError
from .AuxFunc  import SurrogateData
from .LoadData import sampleData

__version__     = "2.2.0"
__versionDate__ = "2025-01-27"
