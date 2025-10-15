#! /usr/bin/env python3

from argparse        import ArgumentParser
from datetime        import datetime
from multiprocessing import get_context
from itertools       import repeat
from pickle          import dump

from pandas import DataFrame, concat, read_feather, read_csv
from pyEDM  import ComputeError, Simplex, sampleData
from numpy  import array
from matplotlib import pyplot as plt

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def CrossMap_Columns( data, target = None, E = 0, Evec = None,
                      Tp = 1, tau = -1, exclusionRadius = 0,
                      lib = None, pred = None, cores = 5,
                      mpMethod = None, chunksize = 1,
                      returnError = False, noTime = False,
                      outputFile = None, verbose = False,
                      errorPlot = 'rho', plot = False ):

    '''Use multiprocessing Pool to process parallelise Simplex.
       The target (-t) argument specifies a column against which all
       other columns are cross mapped. If target is None each column
       is mapped against itself.

       If lib and pred are not provided set to full data span

       If Evec is None use global E,
       else Evec is a vector of E for each column in data

       Return
          dict '{columns[i]}:{target}' : Simplex DataFrame
       if returnError True :
          dict '{columns[i]}:{target}' : ComputeError DataFrame
    '''

    startTime = datetime.now()

    if verbose :
        print( f"CrossMap_Columns : {startTime}" )

    # If no lib and pred, create from full data span
    if not lib :
        lib = [ 1, data.shape[0] ]
    if not pred :
        pred = [ 1, data.shape[0] ]

    columns = data.columns.to_list()
    if not noTime :
        columns = columns[1:] # Ignore first column

    if target is None :
        targets = columns # univariate mapping
    else :
        if target not in columns :
            raise RuntimeError( f'target {target} not found in data' )
        # Cross map each column to the same target
        targets = [ t for t in repeat( target, len(columns) ) ]

    # If Evec not provided create Evec
    if Evec is None :
        if E < 1 :
            raise RuntimeError('E must be positive, no EDim provided')
        Evec = array( [E] * len( columns ) )
    else :
        if len( Evec ) != len( columns ) :
            raise RuntimeError(f'Evec length {len(Evec)} does not match columns')

    # Dictionary of static arguments for Pool : SimplexFunc
    argsD = { 'lib' : lib, 'pred' : pred,
              'exclusionRadius' : exclusionRadius, 'Tp' : Tp,
              'tau' : tau, 'noTime' : noTime, 'returnError' : returnError }

    # Create iterable for Pool.starmap, use repeated copies of argsD, data
    poolArgs = zip( columns, targets, Evec, repeat( argsD ), repeat( data ) )

    # Use pool.starmap to distribute among cores
    mpContext = get_context( mpMethod )
    with mpContext.Pool( processes = cores ) as pool :
        CMList = pool.starmap( SimplexFunc, poolArgs, chunksize = chunksize )

    # Load CMList results into dictionary
    D = {}
    for i in range( len( columns ) ) :
        D[ f'{columns[i]}:{targets[i]}' ] = CMList[ i ]

    if verbose :
        print( D.keys() )
        print( f'Elapsed time: {datetime.now() - startTime}' )

    if outputFile :
        if not '.pkl' in outputFile[-4:] :
            outputFile = outputFile + '.pkl'

        with open( outputFile, 'wb' ) as f :
            dump( D, f )

        if verbose :
            print( f'Wrote D to pickle file: {outputFile}' )

    if plot :
        if returnError :
            df = concat( D, ignore_index = True )
            df.index = D.keys()
            df.plot( y = errorPlot, linewidth=2, ylabel=errorPlot )
            
        else :
            lastD = list( D.keys() )[-1]
            print( f'Plotting last item in D: {lastD}' )
            df = D[ lastD ]
            df.plot( df.columns[0], df.columns[1:3], linewidth=3, ylabel=lastD )

        plt.tight_layout()
        plt.show()

    return D
    
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def SimplexFunc( column, target, E, argsD, data, returnError = False ):
    '''Call pyEDM Simplex using the column, args, and data'''

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

    if argsD['returnError'] :
        D_err = ComputeError( df['Observations'], df['Predictions'] )
        df = DataFrame( D_err, index = [ f'{column}:{target}' ] )

    return df

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def CrossMap_Columns_CmdLine():
    '''Wrapper for CrossMap_Columns with command line parsing'''

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

    # Optional node specific E DataFrame : pyEDM/apps/EmbedDim_Columns.py
    Evec = None
    if args.EDimFile:
        if '.csv' in args.EDimFile[-4:] :
            EDim = read_csv( args.EDimFile )
        elif '.feather' in args.EDimFile[-8:] :
            EDim = read_feather( args.EDimFile )
        else:
            msg = f'Input file {args.EDimFile} must be csv or feather'
            raise RuntimeError( msg )

        # Get EDim['E'] values into Evec according to dataFrame.columns
        columns = dataFrame.columns
        nodesInEDim = set( columns ).issubset( set( EDim['column'] ) ) # bool
        if not nodesInEDim:
            raise RuntimeError( "Failed to find data column in EDim['column']" )

        mask = EDim['column'].isin( columns )
        EDim = EDim[ mask ] # subset to rows with gmnNodes
        # Get Evec corresponding to sorted( data_.columns )
        EDim.set_index( 'column', inplace = True )
        Evec = EDim.loc[ columns, 'E' ].to_numpy()

    # Call CrossMap_Columns()
    df = CrossMap_Columns( data = dataFrame, target = args.target, Evec = Evec,
                           E = args.E, Tp = args.Tp, tau = args.tau,
                           exclusionRadius = args.exclusionRadius,
                           lib = args.lib, pred = args.pred,
                           cores = args.cores, mpMethod = args.mpMethod,
                           chunksize = args.chunksize,
                           returnError = args.returnError, noTime = args.noTime,
                           outputFile = args.outputFile, verbose = args.verbose,
                           errorPlot = args.errorPlot, plot = args.Plot )

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def ParseCmdLine():

    parser = ArgumentParser( description = 'CrossMap Columns' )

    parser.add_argument('-i', '--inputFile',
                        dest    = 'inputFile', type = str, 
                        action  = 'store',
                        default = None,
                        help    = 'Input data file.')

    parser.add_argument('-d', '--inputData',
                        dest    = 'inputData', type = str, 
                        action  = 'store',
                        default = 'Lorenz5D',
                        help    = 'pyEDM sampleData DataFrame name.')

    parser.add_argument('-o', '--outputFile',
                        dest    = 'outputFile', type = str, 
                        action  = 'store',
                        default = None,
                        help    = 'Output file.')

    parser.add_argument('-e', '--EDimFile',
                        dest    = 'EDimFile', type = str,
                        action  = 'store',
                        default = None,
                        help    = 'EDimFile DataFrame[column,E]')

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

    parser.add_argument('-mp', '--mpMethod',
                        dest    = 'mpMethod', type = str,
                        action  = 'store',
                        default = None,
                        help    = 'Multiprocessing start method')

    parser.add_argument('-cz', '--chunksize',
                        dest   = 'chunksize', type = int,
                        action = 'store', default = 1,
                        help = 'ProcessPool chunksize')

    parser.add_argument('-re', '--returnError',
                        dest    = 'returnError',
                        action  = 'store_true',
                        default = False,
                        help    = 'returnError instead of cross map data.')

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

    parser.add_argument('-ep', '--errorPlot',
                        dest    = 'errorPlot', type = str,
                        action  = 'store',
                        default = 'rho',
                        help    = 'errorPlot metric')

    args = parser.parse_args()

    return args

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading
if __name__ == "__main__":
    CrossMap_Columns_CmdLine()
