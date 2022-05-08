#! /usr/bin/env python3

import pyEDM as EDM
import pkg_resources # Get data file names from EDM package

#------------------------------------------------------------
#------------------------------------------------------------
def main():
    '''EDM package examples.
       df denotes a Pandas DataFrame.
       S and M denote dictionaries of Pandas DataFrames'''

    dataFiles = [ "TentMap_rEDM.csv",
                  "TentMapNoise_rEDM.csv",
                  "circle.csv",
                  "block_3sp.csv",
                  "sardine_anchovy_sst.csv" ]
    
    # Create map of module dataFiles pathnames in Files
    Files = {}

    for file in dataFiles:
        filename = "data/" + file
        if pkg_resources.resource_exists( EDM.__name__, filename ):
            Files[ file ] = \
            pkg_resources.resource_filename( EDM.__name__, filename )
        else :
            raise Exception( "examples.py: Failed to find data file " + \
                             file + " in EDM package" )

    # Note the path argument is empty "", file path is in Files{}
    df = EDM.EmbedDimension( pathIn = "", dataFile = Files["TentMap_rEDM.csv"],
                             lib = "1 100", pred = "201 500", maxE = 10,
                             Tp = 1, tau = -1, exclusionRadius = 0,
                             columns = "TentMap", target = "TentMap",
                             validLib = [], numThreads = 4 )
    
    df = EDM.PredictInterval( pathIn = "", dataFile = Files["TentMap_rEDM.csv"],
                              lib = "1 100", pred = "201 500", maxTp = 10,
                              E = 2, tau = -1, exclusionRadius = 0,
                              columns = "TentMap", target = "TentMap",
                              validLib = [], numThreads = 4 );

    df = EDM.PredictNonlinear( pathIn = "",
                               dataFile = Files[ "TentMapNoise_rEDM.csv" ],
                               lib = "1 100", pred = "201 500", E = 2,
                               Tp = 1, knn = 0, tau = -1,
                               columns = "TentMap", target = "TentMap",
                               validLib = [], numThreads = 4 )
    
    # Tent map simplex : specify multivariable columns embedded = True
    S = EDM.Simplex( pathIn = "", dataFile = Files[ "block_3sp.csv" ],
                     lib = "1 99", pred = "100 198", E = 3, Tp = 1,
                     knn = 0, tau = -1, exclusionRadius = 0,
                     columns = "x_t y_t z_t", target = "x_t",
                     embedded = True, showPlot = True,
                     validLib = [], generateSteps = 0, parameterList = False )

    # Tent map simplex : Embed column x_t to E=3, embedded = False
    S = EDM.Simplex( pathIn = "", dataFile = Files[ "block_3sp.csv" ],
                     lib = "1 99", pred = "100 198", E = 3, Tp = 1,
                     knn = 0, tau = -1, exclusionRadius = 0,
                     columns = "x_t y_t z_t", target = "x_t",
                     embedded = False, showPlot = True,
                     validLib = [], generateSteps = 0, parameterList = False )

    M = EDM.Multiview( pathIn = "", dataFile = Files[ "block_3sp.csv" ],
                       lib = "1 100", pred = "101 198",
                       D = 0, E = 3, Tp = 1, knn = 0, tau = -1,
                       multiview = 0, exclusionRadius = 0,
                       columns = "x_t y_t z_t", target = "x_t",
                       trainLib = True, excludeTarget = False,
                       numThreads = 4, showPlot = True )

    # S-map circle : specify multivariable columns embedded = True
    S = EDM.SMap( pathIn = "", dataFile = Files[ "circle.csv" ], 
                  lib = "1 100", pred = "101 198",
                  E = 2, Tp = 1, knn = 0, tau = -1,
                  theta = 4, exclusionRadius = 0,
                  columns = "x y", target = "x",
                  solver = None, embedded = True,
                  validLib = [], generateSteps = 0, parameterList = False )
    
    CM = EDM.CCM( pathIn = "", dataFile = Files[ "sardine_anchovy_sst.csv" ],
                  E = 3, Tp = 0, knn = 0, tau = -1, exclusionRadius = 0,
                  columns = "anchovy", target = "np_sst",
                  libSizes = "10 70 10", sample = 100, random = True,
                  replacement = False, seed = 0, includeData = False,
                  parameterList = False, verbose = False, showPlot = True )

#------------------------------------------------------------
#------------------------------------------------------------
if __name__ == '__main__':
    main()
