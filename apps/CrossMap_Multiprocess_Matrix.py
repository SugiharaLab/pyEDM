#! /usr/bin/env python3

import time, argparse
from   multiprocessing import Pool
from   itertools       import repeat, product

from numpy  import full, nan
from pandas import DataFrame, read_csv, concat
from pyEDM  import Simplex, sampleData, ComputeError
from matplotlib import pyplot as plt

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def CrossMap_Matrix( data, E = 0, Tp = 1,
                     tau = -1, exclusionRadius = 0,
                     lib = None, pred = None, cores = 5,
                     outputFile = None, noTime = False,
                     verbose = False, plot = False ):

    '''Use multiprocessing Pool to process parallelize Simplex.
       All dataFrame columns are cross mapped to all others.
    '''

    startTime = time.time()

    # If no lib and pred, create from full data span
    if not lib :
        lib = [ 1, data.shape[0] ]
    if not pred :
        pred = [ 1, data.shape[0] ]

    # Ignore first column, convert to list
    if not noTime :
        columns = data.columns[ 1 : len(data.columns) ].to_list()

    numCols = N = len( columns )

    # Allocate matrix for cross map rho
    CM_mat = full ( ( N, N ), nan )

    # Iterable of all columns x columns
    allPairs = list( product( columns, columns ) )
    # Group column tuples into matrix column sets of N
    matColumns = [ allPairs[ i:(i+N) ] for i in range(0, len(allPairs), N) ]

    # Dictionary of arguments for Pool : SimplexFunc
    argsD = { 'lib' : lib, 'pred' : pred, 'E' : E,
              'exclusionRadius' : exclusionRadius, 'Tp' : Tp,
              'tau' : tau, 'noTime' : noTime }

    # Create iterable for Pool.starmap, use repeated copies of argsD, data
    poolArgs = zip( matColumns, repeat( argsD ), repeat( data ) )

    # Use pool.starmap to distribute among cores
    with Pool( processes = cores ) as pool :
        CMList = pool.starmap( SimplexFunc, poolArgs )

    # Load CMList results into dictionary
    D = {}
    for i in range( len( matColumns ) ) :
        D[ i ] = CMList[ i ]

        CM_mat[ i, : ] = CMList[ i ]

    print( "Elapsed time:", round( time.time() - startTime, 2 ) )

    if verbose :
        print( D.keys() )
        print( D.values() )
        print( CM_mat )

    if plot :
        PlotMatrix( CM_mat, columns, figsize = (5,5), dpi = 150,
                    title = None, plot = True, plotFile = None )

    if outputFile :
        df.to_csv( outputFile )

    return D

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def SimplexFunc( matColumns, argsD, data ):
    '''Call pyEDM EmbedDimension using the column, args, and data
       Return prediction correlation

       Each matColumn is a 1xN list of column tuples : for the matrix columns
    '''

    N = len( matColumns )
    rhoCol = full( N, nan )

    for i in range( N ) :
        column, target = matColumns[i]

        df = Simplex( dataFrame       = data,
                      columns         = column,
                      target          = target,
                      lib             = argsD['lib'],
                      pred            = argsD['pred'],
                      E               = argsD['E'],
                      exclusionRadius = argsD['exclusionRadius'],
                      Tp              = argsD['Tp'],
                      tau             = argsD['tau'],
                      noTime          = argsD['noTime'],
                      showPlot        = False )

        rho = ComputeError( df['Observations'], df['Predictions'] )['rho']
        rhoCol[ i ] = rho

    return rhoCol

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def CrossMap_Matrix_CmdLine():
    '''Wrapper for CrossMap_Matrix with command line parsing'''

    args = ParseCmdLine()

    # Read data
    # If -i input file: load it, else look for inputData in sampleData
    if args.inputFile:
        dataFrame = read_csv( args.inputFile )
    elif args.inputData:
        dataFrame = sampleData[ args.inputData ]
    else:
        raise RuntimeError( "Invalid inputFile or inputData" )

    # Call CrossMap_Matrix()
    df = CrossMap_Matrix( data = dataFrame,
                          E = args.E, Tp = args.Tp, tau = args.tau,
                          exclusionRadius = args.exclusionRadius,
                          lib = args.lib, pred = args.pred,
                          cores = args.cores, noTime = args.noTime,
                          outputFile = args.outputFile,
                          verbose = args.verbose, plot = args.Plot )

#--------------------------------------------------------------
#--------------------------------------------------------------
def PlotMatrix( xm, columns, figsize = (5,5), dpi = 150, title = None,
                plot = True, plotFile = None ):
    ''' '''

    fig = plt.figure( figsize = figsize, dpi = dpi )
    ax  = fig.add_subplot()

    #fig.suptitle( title )
    ax.set( title = f'{title}' )
    ax.xaxis.set_ticks( [x for x in range( len(columns) )] )
    ax.yaxis.set_ticks( [x for x in range( len(columns) )] )
    ax.set_xticklabels(columns, rotation = 90)
    ax.set_yticklabels(columns)

    cax = ax.matshow( xm )
    fig.colorbar( cax )

    if plotFile :
        fname = f'{plotFile}'
        plt.savefig( fname, dpi = 'figure', format = 'png' )

    if plot :
        plt.show()

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
    CrossMap_Matrix_CmdLine()
