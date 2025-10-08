#! /usr/bin/env python3

# Python distribution modules
from datetime           import datetime
from argparse           import ArgumentParser
from itertools          import repeat
from concurrent.futures import ProcessPoolExecutor
from multiprocessing    import get_context

# Community modules
from pyEDM  import CCM
from numpy  import array, exp, full, nan, nan_to_num, round, zeros
from pandas import DataFrame, read_csv, read_feather
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
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
                expConverge     = False,
                ignoreNan       = True,
                noTime          = False,
                validLib        = [],
                cores           = 5,
                mpMethod        = None,
                chunksize       = 1,
                verbose         = False,
                debug           = False,
                outputFile      = None,
                includeCCM      = False,
                plot            = False,
                title           = "",
                figSize         = (6,6),
                dpi             = 150 ) :

    '''Use ProcessPoolExecutor to process parallelize CCM.
       All dataFrame columns are cross mapped against all others.

       E is a vector of embedding dimension for each column.
       if E is a single value it is repeated for all columns.

       tau is a vector of embedding delay for each column.
       if tau is a single value it is repeated for all columns.

       Note CCM() invokes multiprocessing with two processes.
       The number of cores used : -C args.cores should be less
       than os.process_cpu_count() / 2

       If libSizes are not provided they are computed from pLibSizes:
       a vector of percentiles of the data length. pLibSizes can also
       be specified directly.

       The slope of CCM rho(libSizes) is computed based on a [0,1]
       normalization of libSizes.

       if expConverge = True a nonlinear convergence function is fit
       to rho(libSizes) : y0 + b * ( 1 - exp(-a * x) )
         Fit parameters [a,b,y0] returned in DataFrame D['ccm converge']
         DataFrame of just a returned in D['ccm converge a']

       return : D = { 'ccm rho' : DataFrame, 'ccm slope' : DataFrame }

       if args.expConverge add to D:
          { 'ccm converge' : DataFrame, 'ccm converge a' : DataFrame }

       if args.includeCCM add to D:
          { 'ccm results':list }
    '''

    startTime = datetime.now()
    
    if verbose :
        print( f'CCM_Matrix: {startTime}' )

    if outputFile :
        if '.csv' not in outputFile[-4:] and '.feather' not in outputFile[-8:] :
            msg = f'outputFile {outputFile} must be .csv or .feather'
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
    if expConverge :
        expConverge_DF = DataFrame( columns = columns, index = columns )
    else :
        expConverge_DF = DataFrame() # empty

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
              'expConverge'     : expConverge,
              'validLib'        : validLib,
              'noTime'          : noTime,
              'ignoreNan'       : ignoreNan,
              'verbose'         : verbose }

    mpContext = get_context( mpMethod )

    if verbose :
        print( f'multiprocessing: {mpContext._name} ' +\
               f'using {cores} of {mpContext.cpu_count()} available CPU' )

    # ProcessPoolExecutor has no starmap(). Pass argument lists directly.
    with ProcessPoolExecutor( max_workers=cores, mp_context=mpContext ) as exe :
        ccm_ = exe.map( CCMFunc,
                        upperDiagPairs,
                        repeat( data,  N ),
                        repeat( argsD, N ),
                        chunksize = chunksize )

    # ccm_ is a generator of dictionaries from CCMFunc
    ccmD_ = [ _ for _ in ccm_ ]

    if debug :
        for ccmD in ccmD_ :
            print( ccmD )
            print()

    # Load ccmD_ results into DataFrame
    for ccmD in ccmD_ :
        column_ = ccmD['column']
        target_ = ccmD['target']

        CCM_DF.loc[ column_, target_ ] = ccmD['col_tgt'][-1]
        CCM_DF.loc[ target_, column_ ] = ccmD['tgt_col'][-1]

        slope_DF.loc[ column_, target_ ] = ccmD['col_tgt_slope']
        slope_DF.loc[ target_, column_ ] = ccmD['tgt_col_slope']

        if expConverge :
            expConverge_DF.loc[ column_, target_ ] = ccmD['col_tgt_expConverge']
            expConverge_DF.loc[ target_, column_ ] = ccmD['tgt_col_expConverge']

    if debug :
        print( 'CCM_Matrix: ccm rho' )
        print( CCM_DF )
        print()
        print( 'CCM_Matrix: ccm slope' )
        print( slope_DF )
        print()
        print( 'CCM_Matrix: ccm expConverge' )
        print( expConverge_DF )
        print()

    if verbose :
        print( f'Finished {datetime.now()}' )
        print( f'Elapsed time: {datetime.now() - startTime}' )

    if plot :
        PlotMatrix( CCM_DF.to_numpy( dtype = float ), columns,
                    figsize = figSize, dpi = dpi,
                    title = title, plot = True, plotFile = None )

    if outputFile :
        if '.csv' in outputFile[-4:] :
            CCM_DF.to_csv( 'CCM_' + outputFile, index_label = 'variable' )
            slope_DF.to_csv( 'Slope_' + outputFile, index_label = 'variable' )
            expConverge_DF.to_csv( 'expConverge_' + outputFile,
                                   index_label = 'variable' )

        elif '.feather' in outputFile[-8:] :
            CCM_DF.to_feather( 'CCM_' + outputFile )
            slope_DF.to_feather( 'Slope_' + outputFile )
            expConverge_DF.to_feather( 'expConverge_' + outputFile )

    # Output dict
    D = { 'ccm rho' : CCM_DF, 'ccm slope' : slope_DF }

    if expConverge :
          D['ccm converge']   = expConverge_DF
          D['ccm converge a'] = expConverge_DF.apply(lambda x: x.str[0])

    if includeCCM :
        D['ccm results'] = ccmD_

    return D

#----------------------------------------------------------------------------
def CCMFunc( column_target, df, args ):
    '''pyEDM CCM
       Compute CCM for forward and reverse mapping, and, linear regression
       slope of CCM rho against [0,1] normalized libSizes. if args.expConverge
       fit CCM_rho_L_fit() function to rho(libSizes)

       Return dict: { 'target':, 'column':, 'libSize':, 'col_tgt':, 'tgt_col':,
                      'col_tgt_slope':, 'tgt_col_slope':,
                      'expConverge_col_tgt':, 'expConverge_tgt_col': }
    '''
    #------------------------------------------------------------------
    def CCM_rho_L_fit( x = 0, a = 2, b = 1, y0 = 0.1 ):
        '''CCM rho(L) curve for L = [0:1] to optimize in curve_fit()'''
        return ( y0 + b * ( 1 - exp(-a * x) ) ).flatten()

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
              'tgt_col_slope' : 0.,
              'col_tgt_expConverge' : 0.,
              'tgt_col_expConverge' : 0.}

        return D

    col_target = f'{column}:{target}'
    target_col = f'{target}:{column}'

    ccm_col_tgt = ccm_[ ['LibSize', col_target] ]
    ccm_tgt_col = ccm_[ ['LibSize', target_col] ]

    # libSizes ndarray for CCM convergence slope estimate
    libSizesVec = array( args['libSizes'], dtype = float ).reshape( -1, 1 )
    # normalize L ∈ [0,1]
    libSizesVec = libSizesVec / df.shape[0]

    # Slope of linear fit to rho(libSizes)
    lm_col_tgt = LinearRegression().fit(libSizesVec,
                                        nan_to_num(ccm_col_tgt[ col_target ]))
    lm_tgt_col = LinearRegression().fit(libSizesVec,
                                        nan_to_num(ccm_tgt_col[ target_col ]))

    slope_col_tgt = lm_col_tgt.coef_[0]
    slope_tgt_col = lm_tgt_col.coef_[0]

    expConverge_col_tgt = None
    expConverge_tgt_col = None

    if args['expConverge'] :
        try:
            ydata = ccm_[ f'{column}:{target}' ]
            popt_c_t, pcov_c_t, infodict_c_t, msg_c_t, ier_c_t = \
                curve_fit( CCM_rho_L_fit,
                           xdata  = libSizesVec,
                           ydata  = ydata,
                           p0     = [2,1,0.1],
                           bounds = ( [0,0,0], [100,1,1] ),
                           method = 'dogbox',
                           full_output = True )
        except Exception as e:
            if args['verbose'] :
                print( e )
            popt_c_t = full( 3, nan )

        try:
            ydata = ccm_[ f'{target}:{column}' ]
            popt_t_c, pcov_t_c, infodict_t_c, msg_t_c, ier_t_c = \
                curve_fit( CCM_rho_L_fit,
                           xdata  = libSizesVec,
                           ydata  = ydata,
                           p0     = [2,1,0.1],
                           bounds = ( [0,0,0], [100,1,1] ),
                           method = 'dogbox',
                           full_output = True )
        except Exception as e:
            if args['verbose'] :
                print( e )
            popt_t_c = full( 3, nan )

        expConverge_col_tgt = popt_c_t # 3 parameters
        expConverge_tgt_col = popt_t_c # 3 parameters

    D = { 'column'  : column,
          'target'  : target,
          'E'       : E,
          'tau'     : tau,
          'libSize' : args['libSizes'],
          'col_tgt' : round( ccm_col_tgt[ col_target ].to_numpy(), 5 ),
          'tgt_col' : round( ccm_tgt_col[ target_col ].to_numpy(), 5 ),
          'col_tgt_slope' : round( slope_col_tgt, 5 ),
          'tgt_col_slope' : round( slope_tgt_col, 5 ),
          'col_tgt_expConverge' : expConverge_col_tgt,
          'tgt_col_expConverge' : expConverge_tgt_col }

    return D

#---------------------------------------------------------------------
def CCM_Matrix_CmdLine():
    '''Wrapper for CCM_Matrix with command line parsing'''

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

    # Call CCM_Matrix()
    df = CCM_Matrix( data = dataFrame, E = args.E,
                     libSizes = args.libSizes, pLibSizes = args.pLibSizes,
                     sample = args.sample, Tp = args.Tp, tau = args.tau,
                     exclusionRadius = args.exclusionRadius,
                     expConverge = args.expConverge,
                     ignoreNan = args.ignoreNan, noTime = args.noTime,
                     validLib = args.validLib,
                     cores = args.cores, mpMethod = args.mpMethod,
                     chunksize = args.chunksize,
                     verbose = args.verbose, debug = args.debug,
                     outputFile = args.outputFile,
                     includeCCM = args.includeCCM,
                     plot = args.Plot, title = args.plotTitle,
                     figSize = args.figureSize, dpi = args.dpi )

#---------------------------------------------------------------------
def PlotMatrix( xm, columns, figsize = (5,5), dpi = 150, title = None,
                plot = True, plotFile = None, cmap = None, norm = None,
                aspect = None, vmin = None, vmax = None, colorBarShrink = 1. ):
    '''Generic function to plot numpy matrix

    Examples of DataFrame returned in dict of CCM_Matrix:
       CM = CCM_Matrix( df, 5, expConverge = True )
       PlotMatrix(CM['ccm slope'].to_numpy(dtype=float),CM['ccm slope'].columns)
       PlotMatrix(CM['ccm converge a'].to_numpy(dtype=float),
                  CM['ccm converge a'].columns)
    '''

    fig = plt.figure( figsize = figsize, dpi = dpi )
    ax  = fig.add_subplot()

    #fig.suptitle( title )
    ax.set( title = f'{title}' )
    ax.xaxis.set_ticks( [x for x in range( len(columns) )] )
    ax.yaxis.set_ticks( [x for x in range( len(columns) )] )
    ax.set_xticklabels(columns, rotation = 90)
    ax.set_yticklabels(columns)

    cax = ax.matshow( xm, cmap = cmap, norm = norm,
                      aspect = aspect, vmin = vmin, vmax = vmax )
    fig.colorbar( cax, shrink = colorBarShrink )

    plt.tight_layout()

    if plotFile :
        fname = f'{plotFile}'
        plt.savefig( fname, dpi = 'figure', format = 'png' )

    if plot :
        plt.show()

#----------------------------------------------------------------------
def ParseCmdLine():

    parser = ArgumentParser( description = 'CCM Matrix' )

    parser.add_argument('-i', '--inputFile',
                        dest    = 'inputFile', type = str,
                        action  = 'store',
                        default = None,
                        help    = 'Input data file .csv or .feather')

    parser.add_argument('-d', '--inputData',
                        dest    = 'inputData', type = str,
                        action  = 'store',
                        default = 'Lorenz5D',
                        help    = 'pyEDM sampleData name')

    parser.add_argument('-of', '--outputFile',
                        dest    = 'outputFile', type = str,
                        action  = 'store',
                        default = None,
                        help    = 'CCM/slope matrix output file name')

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

    parser.add_argument('-ec', '--expConverge',
                        dest    = 'expConverge',
                        action  = 'store_true',
                        default = False,
                        help    = 'Compute exp convergence')

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

    parser.add_argument('-mp', '--mpMethod',
                        dest    = 'mpMethod', type = str,
                        action  = 'store',
                        default = None,
                        help    = 'Multiprocessing start method')

    parser.add_argument('-cz', '--chunksize',
                        dest   = 'chunksize', type = int,
                        action = 'store', default = 1,
                        help = 'ProcessPool chunksize')

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
