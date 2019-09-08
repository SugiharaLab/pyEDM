'''Loading of example data.'''

import pkg_resources # Get data file pathnames from EDM package

from pandas import read_csv

dataFileNames = [ ("TentMap_rEDM.csv",        "TentMap"),
                  ("TentMapNoise_rEDM.csv",   "TentMapNoise"),
                  ("circle.csv",              "circle"),
                  ("block_3sp.csv",           "block_3sp"),
                  ("sardine_anchovy_sst.csv", "sardine_anchovy_sst") ]

# Dictionary of module DataFrames so user can access sample data
sampleData = {}

for fileName, dataName in dataFileNames:

    filePath = "data/" + fileName

    if pkg_resources.resource_exists( __name__, filePath ):
        sampleData[ dataName ] = \
            read_csv( pkg_resources.resource_filename( __name__, filePath ) )
    else :
        raise Warning( "pyEDM: Failed to find sample data file " + \
                       fileName + " in pyEDM package." )
