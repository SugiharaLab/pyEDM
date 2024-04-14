#! /usr/bin/env python3

import time, argparse
from   multiprocessing import Pool
from   itertools       import repeat

from pandas import DataFrame, read_csv, concat
from pyEDM  import EmbedDimension, sampleData
from matplotlib import pyplot as plt

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def EmbedDim_Columns( data, target = None, maxE = 20,
                      Tp = 1, tau = -1, exclusionRadius = 0,
                      lib = None, pred = None, numThreads = 4, cores = 5,
                      outputFile = None, noTime = False,
                      verbose = False, plot = False ):
    
    '''Use multiprocessing Pool to process parallelise EmbedDimension.
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

    # Dictionary of arguments for Pool : EmbedDimFunc
    argsD = { 'lib' : lib, 'pred' : pred, 'maxE' : maxE,
              'exclusionRadius' : exclusionRadius, 'Tp' : Tp,
              'tau' : tau, 'target' : target,
              'numThreads' : numThreads, 'noTime' : noTime }

    # Create iterable for Pool.starmap, use repeated copies of argsD, data
    poolArgs = zip( columns, repeat( argsD ), repeat( data ) )

    # Use pool.starmap to distribute among cores
    pool = Pool( processes = cores )
    
    # starmap: elements of the iterable argument are iterables
    #          that are unpacked as arguments
    EDList = pool.starmap( EmbedDimFunc, poolArgs )

    # Load EDList results into dictionary : DataFrame
    D = {}
    D[ 'E' ] = EDList[0]['E'].astype( int )
    for i in range( len( columns ) ) :
        D[ f'{columns[i]}:{target}' ] = EDList[ i ]['rho']

    df = DataFrame( D )

    print( "Elapsed time:", round( time.time() - startTime, 2 ) )

    if verbose :
        print( df )

    if plot :
        df.plot( 'E', df.columns[1:], linewidth = 3,
                 ylabel = 'Prediction ρ'  )
        plt.show()

    if outputFile :
        df.to_csv( outputFile )

    return df
    
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def EmbedDimFunc( column, argsD, data ):
    '''Call pyEDM EmbedDimension using the column, args, and data'''

    ed = EmbedDimension( dataFrame       = data,
                         lib             = argsD['lib'],
                         pred            = argsD['pred'],
                         maxE            = argsD['maxE'],
                         exclusionRadius = argsD['exclusionRadius'],
                         Tp              = argsD['Tp'],
                         tau             = argsD['tau'],
                         columns         = column,
                         target          = argsD['target'],
                         numThreads      = argsD['numThreads'],
                         noTime          = argsD['noTime'],
                         showPlot        = False )

    return ed

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def EmbedDim_Columns_CmdLine():
    '''Wrapper for EmbedDim_Columns with command line parsing'''

    args = ParseCmdLine()

    # Read data
    # If -i input file: load it, else look for inputData in sampleData
    if args.inputFile:
        dataFrame = read_csv( args.inputFile )
    elif args.inputData:
        dataFrame = sampleData[ args.inputData ]
    else:
        raise RuntimeError( "Invalid inputFile or inputData" )

    # Call EmbedDim_Columns()
    df = EmbedDim_Columns( data = dataFrame, target = args.target,
                           maxE = args.maxE, Tp = args.Tp, tau = args.tau,
                           exclusionRadius = args.exclusionRadius,
                           lib = args.lib, pred = args.pred,
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

    parser.add_argument('-E', '--maxE',
                        dest    = 'maxE', type = int, 
                        action  = 'store',
                        default = 20,
                        help    = 'Maximum embedding dimension.')

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

    parser.add_argument('-n', '--numThreads',
                        dest    = 'numThreads', type = int, 
                        action  = 'store',
                        default = 3,
                        help    = 'EmbedDimension numThreads.')

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
    EmbedDim_Columns_CmdLine()
