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
    df = EDM.EmbedDimension( "", Files[ "TentMap_rEDM.csv" ], None, "./", "",
                             "1 100", "201 500", 10, 1, -1,
                             "TentMap", "", False, False, 4 )
    
    df = EDM.PredictInterval( "", Files[ "TentMap_rEDM.csv" ], None, "./", "",
                              "1 100", "201 500", 10, 2, -1,
                              "TentMap", "", False, False, 4 );

    df = EDM.PredictNonlinear( "", Files[ "TentMapNoise_rEDM.csv" ], None,
                               "./", "", "1 100", "201 500", "", 2, 1, 0, -1,
                               "TentMap", "", False, False, 4 )
    
    # Tent map simplex : specify multivariable columns embedded = True
    df = EDM.Simplex( "", Files[ "block_3sp.csv" ], None, "./", "", 
                      "1 99", "100 198", 3, 1, 0, -1, 0,
                      "x_t y_t z_t", "x_t", True, False, True, True )

    # Tent map simplex : Embed column x_t to E=3, embedded = False
    df = EDM.Simplex( "", Files[ "block_3sp.csv" ], None, "./", "", 
                      "1 99", "100 195", 3, 1, 0, -1, 0,
                      "x_t", "x_t", False, False, True, True )

    M = EDM.Multiview( "", Files[ "block_3sp.csv" ], None, "./", "", 
                       "1 100", "101 198", 0, 3, 1, 0, -1,
                       "x_t y_t z_t", "x_t", 0, 0, True, False, False, 4, True )

    # S-map circle : specify multivariable columns embedded = True
    S = EDM.SMap( "", Files[ "circle.csv" ], None, "./", "", 
                  "1 100", "101 198", 2, 1, 0, -1, 4, 0,
                  "x y", "x", "", "", None, True, False, True, True )
    
    df = EDM.CCM( "", Files[ "sardine_anchovy_sst.csv" ], None, "./", "", 
                  3, 0, 0, -1, 0, "anchovy", "np_sst",
                  "10 70 10", 100, True, False, 0, False, True, True )

#------------------------------------------------------------
#------------------------------------------------------------
if __name__ == '__main__':
    main()
