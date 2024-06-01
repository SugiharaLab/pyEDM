'''Loading of example data.'''

import importlib.resources # Get data file pathnames from EDM package

from pandas import read_csv

dataFileNames = [ ("TentMap.csv",             "TentMap"),
                  ("TentMapNoise.csv",        "TentMapNoise"),
                  ("circle.csv",              "circle"),
                  ("circle_noise.csv",        "circleNoise"),
                  ("circle_noTime.csv",       "circle_noTime"),
                  ("columnNameSpace.csv",     "columnNameSpace"),
                  ("block_3sp.csv",           "block_3sp"),
                  ("sardine_anchovy_sst.csv", "sardine_anchovy_sst"),
                  ("LorenzData1000.csv",      "Lorenz5D"),
                  ("S12CD-S333-SumFlow_1980-2005.csv", "SumFlow_1980-2005") ]

# Dictionary of module DataFrames so user can access sample data
sampleData = {}

for fileName, dataName in dataFileNames:

    filePath = "data/" + fileName

    ref = importlib.resources.files('pyEDM') / filePath

    with importlib.resources.as_file( ref ) as filePath_ :
        sampleData[ dataName ] = read_csv( filePath_ )

if not len( sampleData ) :
    raise Warning( "pyEDM: Failed to find sample data in pyEDM package." )
