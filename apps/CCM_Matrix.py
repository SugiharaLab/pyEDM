#! /usr/bin/env python3

# Python distribution modules
import time, argparse
from   itertools          import repeat
from   concurrent.futures import ProcessPoolExecutor

# Community modules
from pyEDM  import CCM
from numpy  import array, nan_to_num, round, zeros
from pandas import DataFrame, read_csv
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def CCM_Matrix( data,
                E,
                libSizes        = [],
                pLibSizes       = [10,20,80,90],
                sample          = 30,
                Tp              = 0,
                tau             = -1,
                exclusionRadius = 0,
                ignoreNan       = True,
                noTime          = False,
                validLib        = [],
                cores           = 5,
                verbose         = False,
                debug           = False,
                outputFileName  = None,
                outputFileType  = 'csv',
                includeCCM      = False,
                plot            = False,
                title           = "",
                figSize         = (5,5),
                dpi             = 150 ) :

    '''Use ProcessPoolExecutor to process parallelize CCM.
       All dataFrame columns are cross mapped to all others.

       E is a vector of embedding dimension for each column.
       if E is a single value it is repeated for all columns.

       tau is a vector of embedding delay for each column.
       if tau is a single value it is repeated for all columns.

       Note CCM is already multiprocessing with two processes.
       The number of cores used : -C args.cores should be less
       than os.process_cpu_count() / 2

       The slope of ccm rho(libSizes) is computed based on a [0,1]
       normalization of libSizes.

       If libSizes are not provided they are computed from pLibSizes:
       a vector of percentiles of the data length. pLibSizes can also
       be specified directly.

       return : { 'ccm rho' : DataFrame, 'ccm slope' : DataFrame }

       if args.includeCCM return:
          { 'ccm rho':DataFrame, 'ccm slope':DataFrame, 'ccm results':list }
    '''

    startTime = time.time()

    if outputFileName :
        if 'csv' not in outputFileType and 'feather' not in outputFileType :
            msg = f'outputFileType {outputFileType} must be csv or feather'
            raise( RuntimeError( msg ) )

    # If no libSizes create from pLibSizes percentiles of data len
    if not len( libSizes ) :
        libSizes = [ int( data.shape[0] * (p/100) ) for p in pLibSizes ]
        if verbose :
            print( f'CCM_Matrix(): libSizes set to {libSizes}\n' )

    if noTime :
        columns = data.columns.to_list()
    else :
        # Ignore first column
        columns = data.columns[ 1 : len(data.columns) ].to_list()

    # Empty DataFrame for CCM & slope
    CCM_DF   = DataFrame( columns = columns, index = columns )
    slope_DF = DataFrame( columns = columns, index = columns )

    # Create dictionary of columns : E
    # If E is scalar use E for all columns
    if isinstance( E, int ) :
        E = [ e for e in repeat( E, len( columns ) ) ]
    elif len( E ) == 1 :
        E = [ e for e in repeat( E[0], len( columns ) ) ]
    if len( E ) != len( columns ) :
        msg = 'CCM_Matrix() E must be scalar or length of columns.'
        raise RuntimeError( msg )
    D_E = dict( zip( columns, E ) )

    # Create dictionary of columns : tau
    # If tau is scalar use tau for all columns
    if isinstance( tau, int ) :
        tau = [ t for t in repeat( tau, len( columns ) ) ]
    elif len( tau ) == 1 :
        tau = [ t for t in repeat( E[0], len( columns ) ) ]
    if len( tau ) != len( columns ) :
        msg = 'CCM_Matrix() tau must be scalar or length of columns.'
        raise RuntimeError( msg )
    D_tau = dict( zip( columns, tau ) )

    #----------------------------------------------------------------
    def UpperDiagonalProduct( iterable ):
        '''Manually generate upper triangular elements of iterable product'''
        for i, x in enumerate( iterable ):
            for y in iterable[ i+1: ]:
                yield(x, y)

    # List of tuples of upper triangular columns x columns
    # Remove degenerate pairs from diagonal with same columns
    # Remove pairs with reversed order since CCM computes both V1:V2, V2:V1
    upperDiagPairs = [ _ for _ in UpperDiagonalProduct( columns ) ]
    N = len( upperDiagPairs )

    if debug :
        print( upperDiagPairs )
        print()

    # Dictionary of static arguments for Pool : CCMFunc
    argsD = { 'D_E'             : D_E,
              'libSizes'        : libSizes,
              'Tp'              : Tp,
              'D_tau'           : D_tau,
              'sample'          : sample,
              'exclusionRadius' : exclusionRadius,
              'validLib'        : validLib,
              'noTime'          : noTime,
              'ignoreNan'       : ignoreNan,
              'verbose'         : verbose }

    # ProcessPoolExecutor has no starmap(). Pass argument lists directly.
    with ProcessPoolExecutor( max_workers = cores ) as exe :
        ccm_ = exe.map( CCMFunc,
                        upperDiagPairs,
                        repeat( data,  N ),
                        repeat( argsD, N ) )

    # ccm_ is a generator of dictionaries from CCMFunc
    ccmD_ = [ _ for _ in ccm_ ]

    if debug :
        for ccmD in ccmD_ :
            print( ccmD )
            print()

    # Load ccmD_ results into DataFrame
    for ccmD in ccmD_ :
        CCM_DF.loc[ ccmD['column'], ccmD['target'] ] = ccmD['col_tgt'][-1]
        CCM_DF.loc[ ccmD['target'], ccmD['column'] ] = ccmD['tgt_col'][-1]

        slope_DF.loc[ ccmD['column'], ccmD['target'] ] = ccmD['col_tgt_slope']
        slope_DF.loc[ ccmD['target'], ccmD['column'] ] = ccmD['tgt_col_slope']

    if debug :
        print( 'CCM_Matrix: ccm rho' )
        print( CCM_DF )
        print()
        print( 'CCM_Matrix: ccm slope' )
        print( slope_DF )
        print()

    if verbose :
        print( "Elapsed time:", round( time.time() - startTime, 2 ) )

    if plot :
        PlotMatrix( CCM_DF.to_numpy( dtype = float ), columns,
                    figsize = figSize, dpi = dpi,
                    title = title, plot = True, plotFile = None )

    if outputFileName :
        if outputFileType == 'csv' :
            CCM_DF.to_csv( outputFileName + '_CCM.csv',
                           index_label = 'variable' )
            slope_DF.to_csv( outputFileName + '_Slope.csv',
                             index_label = 'variable' )
        elif outputFileType == 'feather' :
            CCM_DF.to_feather( outputFileName + '_CCM.feather' )
            slope_DF.to_feather( outputFileName + '_Slope.feather' )

    D = { 'ccm rho' : CCM_DF, 'ccm slope' : slope_DF }
    if includeCCM :
        D['ccm results'] = ccmD_
    return D

#----------------------------------------------------------------------------
def CCMFunc( column_target, df, args ):
    '''pyEDM CCM
       Compute CCM for forward and reverse mapping, and, linear regression
       slope of CCM rho against [0,1] normalized libSizes.

       Return dict: { 'target':, 'column':, 'libSize':, 'col_tgt':, 'tgt_col':,
                      'col_tgt_slope':, 'tgt_col_slope': }
    '''

    column, target = column_target
    E   = args['D_E']  [column]
    tau = args['D_tau'][column]

    try :
        ccm_ = CCM( dataFrame = df,
                    columns   = column,
                    target    = target,
                    libSizes  = args['libSizes'],
                    sample    = args['sample'],
                    E         = E,
                    tau       = tau,
                    Tp        = args['Tp'],
                    seed      = 0 )
    except :
        if args['verbose'] :
            print( f'\tCCMFunc() CCM Error: column {column} target {target}.' )

        D = { 'column'  : column,
              'target'  : target,
              'E'       : E,
              'tau'     : tau,
              'libSize' : args['libSizes'],
              'col_tgt' : zeros( len( args['libSizes'] ) ),
              'tgt_col' : zeros( len( args['libSizes'] ) ),
              'col_tgt_slope' : 0.,
              'tgt_col_slope' : 0. }

        return D

    col_target = f'{column}:{target}'
    target_col = f'{target}:{column}'

    ccm_col_tgt = ccm_[ ['LibSize', col_target] ]
    ccm_tgt_col = ccm_[ ['LibSize', target_col] ]

    # libSizes ndarray for CCM convergence slope esimate
    libSizesVec = array( args['libSizes'], dtype = float ).reshape( -1, 1 )
    # normalize [0,1]
    libSizesVec = libSizesVec / libSizesVec[ -1 ]

    # Slope of linear fit to rho(libSizes)
    lm_col_tgt = LinearRegression().fit(libSizesVec,
                                        nan_to_num(ccm_col_tgt[ col_target ]))
    lm_tgt_col = LinearRegression().fit(libSizesVec,
                                        nan_to_num(ccm_tgt_col[ target_col ]))

    slope_col_tgt = lm_col_tgt.coef_[0]
    slope_tgt_col = lm_tgt_col.coef_[0]

    D = { 'column'  : column,
          'target'  : target,
          'E'       : E,
          'tau'     : tau,
          'libSize' : args['libSizes'],
          'col_tgt' : round( ccm_col_tgt[ col_target ].to_numpy(), 5 ),
          'tgt_col' : round( ccm_tgt_col[ target_col ].to_numpy(), 5 ),
          'col_tgt_slope' : round( slope_col_tgt, 5 ),
          'tgt_col_slope' : round( slope_tgt_col, 5 ) }

    return D

#---------------------------------------------------------------------
def CCM_Matrix_CmdLine():
    '''Wrapper for CCM_Matrix with command line parsing'''

    args = ParseCmdLine()

    # Read data
    # If -i input file: load it, else look for inputData in pyEDM sampleData
    if args.inputFile:
        dataFrame = read_csv( args.inputFile )
    elif args.inputData:
        from pyEDM import sampleData
        dataFrame = sampleData[ args.inputData ]
    else:
        raise RuntimeError( "Invalid inputFile or inputData" )

    # Call CCM_Matrix()
    df = CCM_Matrix( data = dataFrame, E = args.E,
                     libSizes = args.libSizes, pLibSizes = args.pLibSizes,
                     sample = args.sample, Tp = args.Tp, tau = args.tau,
                     exclusionRadius = args.exclusionRadius,
                     ignoreNan = args.ignoreNan, noTime = args.noTime,
                     validLib = args.validLib, cores = args.cores,
                     verbose = args.verbose, debug = args.debug,
                     outputFileName = args.outputFileName,
                     outputFileType = args.outputFileType,
                     includeCCM = args.includeCCM,
                     plot = args.Plot, title = args.plotTitle,
                     figSize = args.figureSize, dpi = args.dpi )

#---------------------------------------------------------------------
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

#----------------------------------------------------------------------
def ParseCmdLine():

    parser = argparse.ArgumentParser( description = 'CCM Matrix' )

    parser.add_argument('-i', '--inputFile',
                        dest    = 'inputFile', type = str,
                        action  = 'store',
                        default = None,
                        help    = 'Input data file.')

    parser.add_argument('-d', '--inputData',
                        dest    = 'inputData', type = str,
                        action  = 'store',
                        default = 'Lorenz5D',
                        help    = 'pyEDM sampleData name')

    parser.add_argument('-of', '--outputFileName',
                        dest    = 'outputFileName', type = str,
                        action  = 'store',
                        default = None,
                        help    = 'CCM matrix output .csv file')

    parser.add_argument('-ot', '--outputFileType',
                        dest    = 'outputFileType', type = str,
                        action  = 'store',
                        default = 'csv',
                        help    = 'CCM matrix output .csv file')

    parser.add_argument('-iCCM', '--includeCCM',
                        dest    = 'includeCCM',
                        action  = 'store_true',
                        default = False,
                        help    = 'Include CCM results in return dictionary')

    parser.add_argument('-E', '--E', nargs = '*',
                        dest    = 'E', type = int,
                        action  = 'store',
                        default = [],
                        help    = 'E')

    parser.add_argument('-x', '--exclusionRadius',
                        dest    = 'exclusionRadius', type = int,
                        action  = 'store',
                        default = 0,
                        help    = 'Exclusion Radius')

    parser.add_argument('-T', '--Tp',
                        dest    = 'Tp', type = int,
                        action  = 'store',
                        default = 0,
                        help    = 'Tp')

    parser.add_argument('-tau', '--tau', nargs = '*',
                        dest    = 'tau', type = int,
                        action  = 'store',
                        default = [-1],
                        help    = 'tau')

    parser.add_argument('-l', '--libSizes', nargs = '*',
                        dest    = 'libSizes', type = int,
                        action  = 'store',
                        default = [],
                        help    = 'library sizes')

    parser.add_argument('-p', '--pLibSizes', nargs = '*',
                        dest    = 'pLibSizes', type = int,
                        action  = 'store',
                        default = [10,20,80,90],
                        help    = 'percentile library sizes')

    parser.add_argument('-s', '--sample',
                        dest    = 'sample', type = int,
                        action  = 'store',
                        default = 30,
                        help    = 'CCM sample')

    parser.add_argument('-V', '--validLib', nargs = '*',
                        dest    = 'validLib', type = int,
                        action  = 'store',
                        default = [],
                        help    = 'validLib indices')

    parser.add_argument('-C', '--cores',
                        dest    = 'cores', type = int,
                        action  = 'store',
                        default = 5,
                        help    = 'Multiprocessing cores')

    parser.add_argument('-nT', '--noTime',
                        dest    = 'noTime',
                        action  = 'store_true',
                        default = False,
                        help    = 'set noTime True')

    parser.add_argument('-in', '--ignoreNan',
                        dest    = 'ignoreNan',
                        action  = 'store_true',
                        default = False,
                        help    = 'set ignoreNan True')

    parser.add_argument('-v', '--verbose',
                        dest    = 'verbose',
                        action  = 'store_true',
                        default = False,
                        help    = 'verbose')

    parser.add_argument('-P', '--Plot',
                        dest    = 'Plot',
                        action  = 'store_true',
                        default = False,
                        help    = 'Plot CCM matrix')

    parser.add_argument('-title', '--title',
                        dest    = 'plotTitle', type = str,
                        action  = 'store',
                        default = "",
                        help    = 'CCM matrix plot title.')

    parser.add_argument('-fs', '--figureSize', nargs = 2,
                        dest    = 'figureSize', type = float,
                        action  = 'store',
                        default = [5,5],
                        help    = 'CCM matrix figure size.')

    parser.add_argument('-dpi', '--dpi',
                        dest    = 'dpi', type = int,
                        action  = 'store',
                        default = 150,
                        help    = 'CCM matrix figure dpi.')

    parser.add_argument('-D', '--debug',
                        dest    = 'debug',
                        action  = 'store_true',
                        default = False,
                        help    = 'debug.')

    args = parser.parse_args()

    return args

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading
if __name__ == "__main__":
    CCM_Matrix_CmdLine()
