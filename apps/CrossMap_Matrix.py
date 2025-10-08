#! /usr/bin/env python3

from argparse        import ArgumentParser
from datetime        import datetime
from multiprocessing import get_context
from itertools       import repeat, product, islice
from sys             import platform
from warnings        import filterwarnings

from numpy  import full, nan, fill_diagonal
from pandas import DataFrame, read_feather, read_csv, concat
from pyEDM  import Simplex, sampleData, ComputeError
from matplotlib import pyplot as plt

# numpy/lib/_function_base_impl.py:3000:
# RuntimeWarning: invalid value encountered in divide  c /= stddev[None, :]
filterwarnings("ignore", category=RuntimeWarning)

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def CrossMap_Matrix( data, E = 0, Tp = 1,
                     tau = -1, exclusionRadius = 0,
                     lib = None, pred = None,
                     threshold = None,
                     cores = 5, mpMethod = None, chunksize = 1,
                     returnValue = 'matrix', # or 'dataframe'
                     outputFile = None, noTime = False,
                     verbose = False, plot = False, title = None,
                     figsize = (5,5), dpi = 150 ):

    '''Use multiprocessing Pool to process parallelize Simplex.
       All dataFrame columns are cross mapped to all others.
       E is a vector of embedding dimension for each column.
       if E is single value it is repeated for all columns.

       Return: NxN ndarray or Dataframe
    '''

    startTime = datetime.now()

    if 'dataframe' not in returnValue and 'matrix' not in returnValue :
        msg = "returnValue must be 'matrix' or 'dataframe'"
        raise( RuntimeError( msg ) )

    if outputFile :
        if '.csv' not in outputFile[-4:] and '.feather' not in outputFile[-8:] :
            msg = f'outputFile {outputFile} must be .csv or .feather'
            raise( RuntimeError( msg ) )

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

    if verbose :
        print( startTime )
        print( 'DataFrame:', data.shape, ':', columns[:4],
               '...', columns[-4:] )
        print( 'lib:', lib, ' pred:', pred )

    N = len( columns )

    if isinstance( E, int ) :
        E = [ e for e in repeat( E, len( columns ) ) ]
    elif len( E ) == 1 :
        E = [ e for e in repeat( E[0], len( columns ) ) ]
    if len( E ) != N :
        msg = 'CrossMap_Matrix() E must be scalar or length of data columns.'
        raise RuntimeError( msg )

    # Allocate matrix for cross map rho
    CM_mat = full ( ( N, N ), nan )

    # iterable of all columns x columns with length NxN
    allPairs = list( product( columns, columns ) )

    # Group column tuples into matrix column sets of N
    # This would be used with SimplexColumnsFunc() to process N at time
    # For big data this is memory intensive... instead use block_i generator
    # matColumns = [ allPairs[ i:(i+N) ] for i in range(0, len(allPairs), N) ]

    # islice iterator of allPairs into N items corresponding to N rows
    block_i = blockGenerate( allPairs, N )

    # Static dictionary of arguments for Pool : SimplexFunc
    argsD = { 'lib' : lib, 'pred' : pred,
              'exclusionRadius' : exclusionRadius,
              'Tp' : Tp, 'tau' : tau, 'noTime' : noTime }

    if verbose :
        print( f'Multiprocess loop @  {datetime.now()}', flush = True )
        mpContext = get_context( mpMethod )
        print( f'multiprocessing: {mpContext._name} ' +\
               f'using {cores} of {mpContext.cpu_count()} available CPU' )

    # Loop over allPairs blocks : parallelize block calls to SimplexFunc
    j = 0 # matrix row index
    for i_ in block_i :
        blockPairs = [_ for _ in i_]

        # Iterable for Pool.starmap, use repeated copies of argsD, data
        poolArgs = zip( blockPairs, E, repeat( argsD ), repeat( data ) )

        mpContext = get_context( mpMethod )

        # Use pool.starmap to distribute among cores
        with mpContext.Pool( processes = cores ) as pool :
            CMList = pool.starmap( SimplexFunc, poolArgs, chunksize = chunksize )

        # Load CMList results into matrix
        for i in range( N ) :
            CM_mat[ j, : ] = CMList
        j = j + 1

        if verbose :
            if j % int(N/5) == 0 :
                print(  f'    Multiprocess {100*j/N:,.0f}%  {datetime.now()}',
                        flush = True )

    if verbose :
        print( f'Finished {datetime.now()} ET: {datetime.now() - startTime}' )

    # Replace values below threshold & diagonal with Nan
    fill_diagonal( CM_mat, nan )

    if not threshold is None :
        CM_mat[ CM_mat < threshold ] = nan

    df = None # Created if outputFile or returnValue = dataframe

    if outputFile :
        df = DataFrame( CM_mat, index = columns, columns = columns )
        if '.csv' in outputFile[-4:] :
            df.to_csv( outputFile, index_label = 'variable' )
        elif '.feather' in outputFile[-8:] :
            df.to_feather( outputFile )

    if plot :
        PlotMatrix( CM_mat, columns, figsize = figsize, dpi = dpi,
                    title = title, plot = True, plotFile = None )

    if 'matrix' in returnValue :
        return CM_mat
    else :
        if df is None:
            df = DataFrame( CM_mat, index = columns, columns = columns )
        return df

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def blockGenerate(iterable, N):
    '''Given iterable of NxN items, generate slices corresponding
       to N rows of the NxN product matrix'''
    N_  = len(iterable)
    k_s = 0 # block start index
    while k_s < N_ :
        yield islice( iterable, k_s, k_s + N )
        k_s = k_s + N

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def SimplexFunc( blockPairs, E, argsD, data ):
    '''Call pyEDM Simplex using the column, args, and data
       Return prediction correlation
    '''

    column, target = blockPairs

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

    return rho

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# def SimplexColumnsFunc( matColumns, E, argsD, data ):
#     '''Call pyEDM Simplex using the column, args, and data
#        Return prediction correlation
#
#        Each matColumn is a 1xN list of column tuples : for the matrix columns
#     '''
#
#     N = len( matColumns )
#     rhoCol = full( N, nan )
#
#     for i in range( N ) :
#         column, target = matColumns[i]
#
#         df = Simplex( dataFrame       = data,
#                       columns         = column,
#                       target          = target,
#                       lib             = argsD['lib'],
#                       pred            = argsD['pred'],
#                       E               = E,
#                       exclusionRadius = argsD['exclusionRadius'],
#                       Tp              = argsD['Tp'],
#                       tau             = argsD['tau'],
#                       noTime          = argsD['noTime'],
#                       showPlot        = False )
#
#         rho = ComputeError( df['Observations'], df['Predictions'] )['rho']
#         rhoCol[ i ] = rho
#
#     return rhoCol

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def CrossMap_Matrix_CmdLine():
    '''Wrapper for CrossMap_Matrix with command line parsing'''

    args = ParseCmdLine()

    # Read data
    # If -i input file: load it, else look for inputData in pyEDM sampleData
    if args.inputFile:
        if '.csv' in args.inputFile[-4:] :
            dataFrame = read_csv( args.inputFile )
        elif '.feather' in args.inputFile[-8:] :
            dataFrame = read_feather( args.inputFile )
        else :
            msg = f'Input file {args.inputFile} must be csv or feather'
            raise( RuntimeError( msg ) )
    elif args.inputData:
        from pyEDM import sampleData
        dataFrame = sampleData[ args.inputData ]
    else:
        raise RuntimeError( "Invalid inputFile or inputData" )

    # Call CrossMap_Matrix()
    df = CrossMap_Matrix( data = dataFrame,
                          E = args.E, Tp = args.Tp, tau = args.tau,
                          exclusionRadius = args.exclusionRadius,
                          lib = args.lib, pred = args.pred,
                          cores = args.cores, mpMethod = args.mpMethod,
                          chunksize = args.chunksize,
                          returnValue = args.returnValue,
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

    parser = ArgumentParser( description = 'CrossMap Matrix' )

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

    parser.add_argument('-mp', '--mpMethod',
                        dest    = 'mpMethod', type = str,
                        action  = 'store',
                        default = None,
                        help    = 'Multiprocessing start method')

    parser.add_argument('-cz', '--chunksize',
                        dest   = 'chunksize', type = int,
                        action = 'store', default = 1,
                        help = 'ProcessPoolExecutor.map chunksize')

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
