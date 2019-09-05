# file to load in the sample data

import pkg_resources # Get data file pathnames from EDM package
import pandas as pd

dataFile_names = [ ["TentMap_rEDM.csv","TentMap"],
                   ["TentMapNoise_rEDM.csv","TentMapNoise"],
                   ["circle.csv","circle"],
                   ["block_3sp.csv","block_3sp"],
                   ["sardine_anchovy_sst.csv","sardine_anchovy_sst"] ]

# Create map of module datafile DataFrames in Files so uer can use sample data

sample_data = {}

for filename, dataname in dataFile_names:

    filepath = "data/" + filename

    if pkg_resources.resource_exists( __name__, filepath ):

        sample_data[ dataname ] = \
            pd.read_csv( pkg_resources.resource_filename( __name__, filepath ) )
    else :
        raise Warning( "pyEDM: Failed to find sample data file " + \
                         dataname + " in pyEDM package" )
