#! /usr/bin/env python3

import time, argparse
from   multiprocessing import Pool
from   itertools       import repeat

from pandas import DataFrame, read_csv, concat
from pyEDM  import PredictNonlinear, sampleData
from matplotlib import pyplot as plt

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def PredictNL_Columns( data, target = None, E = 2,
                       Tp = 1, tau = -1, exclusionRadius = 0,
                       lib = None, pred = None, theta = None,
                       numThreads = 4, cores = 5,
                       outputFile = None, noTime = False,
                       verbose = False, plot = False ):
    
    '''Use multiprocessing Pool to process parallelise PredictNonlinear.
       The target (-t) argument specifies a column against which all
       other columns will be cross mapped.
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

    # Dictionary of arguments for Pool : PredictNLFunc
    argsD = { 'lib' : lib, 'pred' : pred, 'E' : E,
              'exclusionRadius' : exclusionRadius, 'Tp' : Tp,
              'tau' : tau, 'target' : target, 'theta' : theta,
              'numThreads' : numThreads, 'noTime' : noTime }

    # Create iterable for Pool.starmap, use repeated copies of argsD, data
    poolArgs = zip( columns, repeat( argsD ), repeat( data ) )

    # Use pool.starmap to distribute among cores
    pool = Pool( processes = cores )
    
    # starmap: elements of the iterable argument are iterables
    #          that are unpacked as arguments
    NL_List = pool.starmap( PredictNLFunc, poolArgs )

    # Load EDList results into dictionary : DataFrame
    D = {}

    D[ 'theta' ] = NL_List[0][ 'Theta' ]
    for i in range( len( columns ) ) :
        D[ f'{columns[i]}:{target}' ] = NL_List[ i ]['rho']

    df = DataFrame( D )

    print( "Elapsed time:", round( time.time() - startTime, 2 ) )

    if verbose :
        print( df )

    if plot :
        df.plot( 'theta', df.columns[1:], linewidth = 3,
                 ylabel = 'Prediction ρ'  )
        plt.show()

    if outputFile :
        df.to_csv( outputFile )

    return df
    
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def PredictNLFunc( column, argsD, data ):
    '''Call pyEDM PredictNonlinear using the column, args, and data'''

    nl = PredictNonlinear( dataFrame       = data,
                           lib             = argsD['lib'],
                           pred            = argsD['pred'],
                           E               = argsD['E'],
                           exclusionRadius = argsD['exclusionRadius'],
                           Tp              = argsD['Tp'],
                           tau             = argsD['tau'],
                           columns         = column,
                           target          = argsD['target'],
                           theta           = argsD['theta'],
                           numThreads      = argsD['numThreads'],
                           noTime          = argsD['noTime'],
                           showPlot        = False )

    return nl

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def PredictNL_Columns_CmdLine():
    '''Wrapper for PredictNL_Columns with command line parsing'''

    args = ParseCmdLine()

    # Read data
    # If -i input file: load it, else look for inputData in sampleData
    if args.inputFile:
        dataFrame = read_csv( args.inputFile )
    elif args.inputData:
        dataFrame = sampleData[ args.inputData ]
    else:
        raise RuntimeError( "Invalid inputFile or inputData" )

    # Call PredictNL_Columns()
    df = PredictNL_Columns( data = dataFrame, target = args.target,
                            E = args.E, Tp = args.Tp, tau = args.tau,
                            exclusionRadius = args.exclusionRadius,
                            lib = args.lib, pred = args.pred, theta = args.theta,
                            numThreads = args.numThreads, cores = args.cores,
                            outputFile = args.outputFile, noTime = args.noTime,
                            verbose = args.verbose, plot = args.Plot )

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def ParseCmdLine():
    
    parser = argparse.ArgumentParser( description = 'EmbedDimension MultiProc' )
    
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
                        default = 2,
                        help    = 'Embedding dimension.')

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

    parser.add_argument('-th', '--theta', nargs = '*',
                        dest    = 'theta', type = float, 
                        action  = 'store',
                        default = [0,0.1,0.25,0.5,1,1.5,2,3,4,5,8],
                        help    = 'List of theta.')

    parser.add_argument('-n', '--numThreads',
                        dest    = 'numThreads', type = int, 
                        action  = 'store',
                        default = 5,
                        help    = 'EmbedDimension numThreads.')

    parser.add_argument('-C', '--cores',
                        dest    = 'cores', type = int, 
                        action  = 'store',
                        default = 4,
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
    PredictNL_Columns_CmdLine()
