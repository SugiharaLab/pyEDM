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
                     returnValue = 'matrix', # or 'dict'
                     outputFile = None, noTime = False,
                     verbose = False, plot = False ):

    '''Use multiprocessing Pool to process parallelize Simplex.
       All dataFrame columns are cross mapped to all others.
       E is a vector of embedding dimension for each column.
       if E is single value it is repeated for all columns.
       if returnValue == 'matrix' : NxN ndarray
       if returnValue == 'dict'   : { column i : vector column rho }
    '''

    startTime = time.time()

    # If no lib and pred, create from full data span
    if not lib :
        lib =  [ 1, data.shape[0] ]
    if not pred :
        pred = [ 1, data.shape[0] ]

    if noTime :
        columns = data.columns.to_list()
    else :
        # Ignore first column
        columns = data.columns[ 1 : len(data.columns) ].to_list()

    N = len( columns )

    if isinstance( E, int ) :
        E = [ e for e in repeat( E, len( columns ) ) ]
    elif len( E ) == 1 :
        E = [ e for e in repeat( E[0], len( columns ) ) ]
    if len( E ) != N :
        msg = 'CrossMap_Matrix() E must be scalar or length of data columns.'
        raise RuntimeError( msg )

    if 'matrix' in returnValue :
        # Allocate matrix for cross map rho
        CM_mat = full ( ( N, N ), nan )
    else :
        CM_mat = None

        if plot :
            msg = "CrossMap_Matrix() set returnValue = 'matrix' for plot."
            raise RuntimeError( msg )

    # Iterable of all columns x columns
    allPairs = list( product( columns, columns ) )
    # Group column tuples into matrix column sets of N
    matColumns = [ allPairs[ i:(i+N) ] for i in range(0, len(allPairs), N) ]

    # Dictionary of arguments for Pool : SimplexFunc
    argsD = { 'lib' : lib, 'pred' : pred,
              'exclusionRadius' : exclusionRadius, 'Tp' : Tp,
              'tau' : tau, 'noTime' : noTime }

    # Create iterable for Pool.starmap, use repeated copies of argsD, data
    poolArgs = zip( matColumns, E, repeat( argsD ), repeat( data ) )

    # Use pool.starmap to distribute among cores
    with Pool( processes = cores ) as pool :
        CMList = pool.starmap( SimplexFunc, poolArgs )

    # Load CMList results into dictionary and matrix
    D = {}
    for i in range( len( matColumns ) ) :
        if 'dict' in returnValue :
            D[ i ] = CMList[ i ]
        else :
            CM_mat[ i, : ] = CMList[ i ]

    if verbose :
        print( "Elapsed time:", round( time.time() - startTime, 2 ) )

    if plot :
        PlotMatrix( CM_mat, columns, figsize = (5,5), dpi = 150,
                    title = None, plot = True, plotFile = None )

    if outputFile :
        if 'dict' in returnValue :
            df = DataFrame( D, index = columns )
        else :
            df = DataFrame( CM_mat, index = columns, columns = columns )
        df.to_csv( outputFile, index_label = 'variable' )

    if 'matrix' in returnValue :
        return CM_mat
    else :
        return D

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def SimplexFunc( matColumns, E, argsD, data ):
    '''Call pyEDM Simplex using the column, args, and data
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
                      E               = E,
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
                          cores = args.cores, returnValue = args.returnValue,
                          noTime = args.noTime, outputFile = args.outputFile,
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

    parser.add_argument('-E', '--E', nargs = '*',
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

    parser.add_argument('-r', '--returnValue',
                        dest    = 'returnValue',
                        action  = 'store',
                        default = 'matrix',
                        help    = 'returnValue.')

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
