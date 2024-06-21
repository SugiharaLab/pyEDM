#! /usr/bin/env python3

import time, argparse
from   multiprocessing import Pool
from   itertools       import repeat

from pandas import DataFrame, read_csv, concat
from pyEDM  import Simplex, sampleData
from matplotlib import pyplot as plt

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def CrossMap_Columns( data, target = None, E = 0,
                      Tp = 1, tau = -1, exclusionRadius = 0,
                      lib = None, pred = None, cores = 5,
                      outputFile = None, noTime = False,
                      verbose = False, plot = False ):

    '''Use multiprocessing Pool to process parallelise Simplex.
       The target (-t) argument specifies a column against which all
       other columns are cross mapped.
       Return dict '{columns[i]}:{target}' : Simplex DataFrame
    '''

    startTime = time.time()

    if not target :
        raise( RuntimeError( 'target required' ) )

    # If no lib and pred, create from full data span
    if not lib :
        lib = [ 1, data.shape[0] ]
    if not pred :
        pred = [ 1, data.shape[0] ]

    # Ignore first column, convert to list
    if not noTime :
        columns = data.columns[ 1 : len(data.columns) ].to_list()

    # Dictionary of arguments for Pool : SimplexFunc
    argsD = { 'lib' : lib, 'pred' : pred, 'E' : E,
              'exclusionRadius' : exclusionRadius, 'Tp' : Tp,
              'tau' : tau, 'target' : target, 'noTime' : noTime }

    # Create iterable for Pool.starmap, use repeated copies of argsD, data
    poolArgs = zip( columns, repeat( argsD ), repeat( data ) )

    # Use pool.starmap to distribute among cores
    with Pool( processes = cores ) as pool :
        CMList = pool.starmap( SimplexFunc, poolArgs )

    # Load CMList results into dictionary
    D = {}
    for i in range( len( columns ) ) :
        D[ f'{columns[i]}:{target}' ] = CMList[ i ]

    if verbose :
        print( "Elapsed time:", round( time.time() - startTime, 2 ) )

        print( D.keys() )

    if plot :
        df = D[ list( D.keys() )[-1] ]
        df.plot( df.columns[0], df.columns[1:3], linewidth = 3,
                 ylabel = list( D.keys() )[-1] )
        plt.show()

    if outputFile :
        df.to_csv( outputFile )

    return D
    
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def SimplexFunc( column, argsD, data ):
    '''Call pyEDM Simplex using the column, args, and data'''

    df = Simplex( dataFrame       = data,
                  columns         = column,
                  target          = argsD['target'],
                  lib             = argsD['lib'],
                  pred            = argsD['pred'],
                  E               = argsD['E'],
                  exclusionRadius = argsD['exclusionRadius'],
                  Tp              = argsD['Tp'],
                  tau             = argsD['tau'],
                  noTime          = argsD['noTime'],
                  showPlot        = False )

    return df

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def CrossMap_Columns_CmdLine():
    '''Wrapper for CrossMap_Columns with command line parsing'''

    args = ParseCmdLine()

    # Read data
    # If -i input file: load it, else look for inputData in sampleData
    if args.inputFile:
        dataFrame = read_csv( args.inputFile )
    elif args.inputData:
        dataFrame = sampleData[ args.inputData ]
    else:
        raise RuntimeError( "Invalid inputFile or inputData" )

    # Call CrossMap_Columns()
    df = CrossMap_Columns( data = dataFrame, target = args.target,
                           E = args.E, Tp = args.Tp, tau = args.tau,
                           exclusionRadius = args.exclusionRadius,
                           lib = args.lib, pred = args.pred,
                           cores = args.cores, noTime = args.noTime,
                           outputFile = args.outputFile,
                           verbose = args.verbose, plot = args.Plot )

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def ParseCmdLine():
    
    parser = argparse.ArgumentParser( description = 'CrossMap Columns' )
    
    parser.add_argument('-i', '--inputFile',
                        dest    = 'inputFile', type = str, 
                        action  = 'store',
                        default = None,
                        help    = 'Input data file.')
    
    parser.add_argument('-d', '--inputData',
                        dest    = 'inputData', type = str, 
                        action  = 'store',
                        default = 'Lorenz5D',
                        help    = 'Input data frame name.')
    
    parser.add_argument('-o', '--outputFile',
                        dest    = 'outputFile', type = str, 
                        action  = 'store',
                        default = None,
                        help    = 'Output file.')

    parser.add_argument('-E', '--E',
                        dest    = 'E', type = int, 
                        action  = 'store',
                        default = 0,
                        help    = 'E.')

    parser.add_argument('-x', '--exclusionRadius',
                        dest    = 'exclusionRadius', type = int, 
                        action  = 'store',
                        default = 0,
                        help    = 'Exclusion Radius.')

    parser.add_argument('-T', '--Tp',
                        dest    = 'Tp', type = int, 
                        action  = 'store',
                        default = 1,
                        help    = 'Tp.')

    parser.add_argument('-tau', '--tau',
                        dest    = 'tau', type = int, 
                        action  = 'store',
                        default = -1,
                        help    = 'tau.')

    parser.add_argument('-t', '--target',
                        dest    = 'target', type = str, 
                        action  = 'store',
                        default = 'V5',
                        help    = 'Data target name.')
    
    parser.add_argument('-l', '--lib', nargs = '*',
                        dest    = 'lib', type = int, 
                        action  = 'store',
                        default = [],
                        help    = 'library indices.')

    parser.add_argument('-p', '--pred', nargs = '*',
                        dest    = 'pred', type = int, 
                        action  = 'store',
                        default = [],
                        help    = 'prediction indices.')

    parser.add_argument('-C', '--cores',
                        dest    = 'cores', type = int, 
                        action  = 'store',
                        default = 5,
                        help    = 'Multiprocessing cores.')

    parser.add_argument('-nT', '--noTime',
                        dest    = 'noTime',
                        action  = 'store_true',
                        default = False,
                        help    = 'noTime.')

    parser.add_argument('-v', '--verbose',
                        dest    = 'verbose',
                        action  = 'store_true',
                        default = False,
                        help    = 'verbose.')

    parser.add_argument('-P', '--Plot',
                        dest    = 'Plot',
                        action  = 'store_true',
                        default = False,
                        help    = 'Plot.')

    args = parser.parse_args()

    return args

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading
if __name__ == "__main__":
    CrossMap_Columns_CmdLine()
