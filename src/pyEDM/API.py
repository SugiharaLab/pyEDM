'''Interface to Empirical Dynamic Modeling (EDM) pyEDM'''

# python modules
from multiprocessing import Pool
from itertools       import repeat

# package modules
from pandas import DataFrame, concat, read_csv
from matplotlib.pyplot import show, axhline

# local modules
from .AuxFunc   import IsIterable, PlotObsPred, PlotCoeff, ComputeError
from .Simplex   import Simplex   as SimplexClass
from .SMap      import SMap      as SMapClass
from .CCM       import CCM       as CCMClass
from .Multiview import Multiview as MultiviewClass

import pyEDM.PoolFunc as PoolFunc

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def MakeBlock( dataFrame,
               E             = 0,
               tau           = -1,
               columnNames   = [],
               deletePartial = False ):

    raise RuntimeError( "MakeBlock() deprecated. Use Embed()." )

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def Embed( dataFrame     = None,
           E             = 0,
           tau           = -1,
           columns       = "",
           includeTime   = False,
           pathIn        = "./",
           dataFile      = None ):
           # deletePartial = False ):
    '''Takens time-delay embedding on columns via pandas DataFrame.shift()
       if includeTime True : insert dataFrame column 0 in first column
       nan will be present in |tau| * (E-1) rows.'''

    if E < 1 :
        raise RuntimeError( 'Embed(): E must be positive.' )
    if tau == 0 :
        raise RuntimeError( 'Embed(): tau must be non-zero.' )
    if not columns :
        raise RuntimeError( 'Embed(): columns required.' )
    if dataFile is not None:
        dataFrame = read_csv( pathIn + dataFile )
    if not isinstance( dataFrame, DataFrame ) :
        raise RuntimeError('Embed(): dataFrame is not a Pandas DataFrame.')

    if not IsIterable( columns ) :
        columns = columns.split() # Convert string to []

    for column in columns :
        if column not in dataFrame.columns :
            raise RuntimeError(f'Embed(): {column} not in dataFrame.')

    # Setup period shift vector for DataFrame.shift()
    # Note that DataFrame.shift() indices are opposite the tau convention
    shiftVec = [ i for i in range( 0, int( E * (-tau) ), -tau ) ]

    df = dataFrame[ columns ].shift( periods = shiftVec ).copy()

    # Replace shifted column names x with x(t-0), x(t-1)...
    # DataFrame.shift() appends _0, _1... or _0, _-1 ... to column names
    # Use rsplit to split the DataFrame.shift() names on the last _
    colNamePairs = [ s.rsplit( '_', 1 ) for s in df.columns ]
    if tau > 0 :
        newColNames = [ ''.join( [s[0],'(t+',s[1].replace('-',''),')'] ) \
                        for s in colNamePairs ]
    else :
        newColNames = [ ''.join( [s[0],'(t-',s[1],')'] ) for s in colNamePairs ]

    df.columns = newColNames

    if includeTime :
        # First column of time/index merged into df
        df = dataFrame.iloc[ :, [0] ].join( df )

    return df

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def Simplex( dataFrame       = None,
             columns         = "",
             target          = "",
             lib             = "",
             pred            = "",
             E               = 0,
             Tp              = 1,
             knn             = 0,
             tau             = -1,
             exclusionRadius = 0,
             embedded        = False,
             validLib        = [],
             noTime          = False,
             generateSteps   = 0,
             generateConcat  = False,
             verbose         = False,
             showPlot        = False,
             ignoreNan       = True,
             returnObject    = False ):
    '''Simplex prediction of dataFrame target from columns.'''

    # Instantiate SimplexClass object
    #    Constructor assigns dataFrame to self, calls Validate(),
    #    CreateIndices(), and assigns targetVec, time
    S = SimplexClass( dataFrame       = dataFrame,
                      columns         = columns,
                      target          = target,
                      lib             = lib,
                      pred            = pred,
                      E               = E,
                      Tp              = Tp,
                      knn             = knn,
                      tau             = tau,
                      exclusionRadius = exclusionRadius,
                      embedded        = embedded,
                      validLib        = validLib,
                      noTime          = noTime,
                      generateSteps   = generateSteps,
                      generateConcat  = generateConcat,
                      ignoreNan       = ignoreNan,
                      verbose         = verbose )

    if generateSteps :
        S.Generate()
    else :
        S.Run()

    if showPlot :
        if embedded :
            if IsIterable( columns ) :
                E = len( columns )
            else :
                E = len( columns.split() )
        PlotObsPred( S.Projection, "", S.E, S.Tp )

    if returnObject :
        return S
    else :
        return S.Projection

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def SMap( dataFrame       = None,
          columns         = "",
          target          = "",
          lib             = "",
          pred            = "",
          E               = 0,
          Tp              = 1,
          knn             = 0,
          tau             = -1,
          theta           = 0,
          exclusionRadius = 0,
          solver          = None,
          embedded        = False,
          validLib        = [],
          noTime          = False,
          generateSteps   = 0,
          generateConcat  = False,
          ignoreNan       = True,
          showPlot        = False,
          verbose         = False,
          returnObject    = False ):
    '''S-Map prediction of dataFrame target from columns.'''

    # Validate solver if one was provided
    if solver is not None :
        supportedSolvers = [ 'function',  'lstsq',
                             'LinearRegression', 'SGDRegressor',
                             'Ridge',      'RidgeCV',
                             'Lasso',      'LassoCV',
                             'Lars',       'LarsCV',
                             'LassoLars',  'LassoLarsCV', 'LassoLarsIC',
                             'ElasticNet', 'ElasticNetCV',
                             'OrthogonalMatchingPursuit',
                             'OrthogonalMatchingPursuitCV' ]
        if not solver.__class__.__name__ in supportedSolvers :
            msg = f'SMap(): Invalid solver {solver.__name__}.\n' +\
                  f'Supported solvers: {supportedSolvers}'
            raise Exception( msg )

    # Instantiate SMapClass object
    #    Constructor assigns dataFrame to self, calls Validate(),
    #    CreateIndices(), and assigns targetVec, time
    S = SMapClass( dataFrame       = dataFrame,
                   columns         = columns,
                   target          = target,
                   lib             = lib,
                   pred            = pred,
                   E               = E,
                   Tp              = Tp,
                   knn             = knn,
                   tau             = tau,
                   theta           = theta,
                   exclusionRadius = exclusionRadius,
                   solver          = solver,
                   embedded        = embedded,
                   validLib        = validLib,
                   noTime          = noTime,
                   generateSteps   = generateSteps,
                   generateConcat  = generateConcat,
                   ignoreNan       = ignoreNan,
                   verbose         = verbose )

    if generateSteps :
        S.Generate()
    else :
        S.Run()

    if showPlot :
        if embedded :
            if IsIterable( columns ) :
                E = len( columns )
            else :
                E = len( columns.split() )
        PlotObsPred( S.Projection,   "", S.E, S.Tp )
        PlotCoeff  ( S.Coefficients, "", S.E, S.Tp )

    if returnObject :
        return S
    else :
        SMapDict = { 'predictions'    : S.Projection,
                     'coefficients'   : S.Coefficients,
                     'singularValues' : S.SingularValues }
        return SMapDict

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def CCM( dataFrame        = None,
         columns          = "",
         target           = "",
         libSizes         = "",
         sample           = 0,
         E                = 0,
         Tp               = 0,
         knn              = 0,
         tau              = -1,
         exclusionRadius  = 0,
         seed             = None,
         embedded         = False,
         validLib         = [],
         includeData      = False,
         noTime           = False,
         ignoreNan        = True,
         verbose          = False,
         showPlot         = False,
         returnObject     = False ) :
    '''Convergent Cross Mapping.'''

    # Instantiate CCMClass object
    # __init__ creates .FwdMap & .RevMap
    C = CCMClass( dataFrame       = dataFrame,
                  columns         = columns,
                  target          = target,
                  E               = E,
                  Tp              = Tp,
                  knn             = knn,
                  tau             = tau,
                  exclusionRadius = exclusionRadius,
                  libSizes        = libSizes,
                  sample          = sample,
                  seed            = seed,
                  includeData     = includeData,
                  embedded        = embedded,
                  validLib        = validLib,
                  noTime          = noTime,
                  ignoreNan       = ignoreNan,
                  verbose         = verbose )

    # Embedding of Forward & Reverse mapping
    C.FwdMap.EmbedData()
    C.FwdMap.RemoveNan()
    C.RevMap.EmbedData()
    C.RevMap.RemoveNan()

    C.Project()

    if showPlot :
        title = f'E = {C.E}'
        if C.libMeans.shape[1] == 3 :
            # CCM of two different variables
            ax = C.libMeans.plot(
                'LibSize', [ C.libMeans.columns[1], C.libMeans.columns[2] ],
                title = title, linewidth = 3 )
        elif C.libMeans.shape[1] == 2 :
            # CCM of degenerate columns : target
            ax = C.libMeans.plot( 'LibSize', C.libMeans.columns[1],
                                  title = title, linewidth = 3 )

        ax.set( xlabel = "Library Size", ylabel = "CCM ρ" )
        axhline( y = 0, linewidth = 1 )
        show()

    if returnObject :
        return C
    else :
        if includeData :
            return { 'LibMeans'      : C.libMeans,
                     'PredictStats1' : C.PredictStats1,
                     'PredictStats2' : C.PredictStats2 }
        else :
            return C.libMeans

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def Multiview( dataFrame       = None,
               columns         = "",
               target          = "",
               lib             = "",
               pred            = "",
               D               = 0,
               E               = 1, 
               Tp              = 1,
               knn             = 0,
               tau             = -1,
               multiview       = 0,
               exclusionRadius = 0,
               trainLib        = True,
               excludeTarget   = False,
               ignoreNan       = True,
               verbose         = False,
               numProcess      = 4,
               showPlot        = False,
               returnObject    = False ):
    '''Multiview prediction on path/file.'''

    # Instantiate MultiviewClass object
    # __init__ creates .Simplex_, calls Validate(), Setup()
    M = MultiviewClass( dataFrame       = dataFrame,
                        columns         = columns,
                        target          = target,
                        lib             = lib,
                        pred            = pred,
                        D               = D,
                        E               = E,
                        Tp              = Tp,
                        knn             = knn,
                        tau             = tau,
                        multiview       = multiview,
                        exclusionRadius = exclusionRadius,
                        trainLib        = trainLib,
                        excludeTarget   = excludeTarget,
                        ignoreNan       = ignoreNan,
                        verbose         = verbose,
                        numProcess      = numProcess,
                        returnObject    = returnObject )

    M.Rank()
    M.Project()

    # multiview averaged prediction
    df = concat( M.topRankProjections.values(), axis = 1 )
    df = df[ 'Predictions' ]
    multiviewPredict = df.mean( axis = 1 )

    # Get a Simplex returned DataFrame for Time and Observations
    df_pred = iter( M.topRankProjections.values() ).__next__()

    df = DataFrame( { 'Time'        : df_pred['Time'],
                      'Observations': df_pred['Observations'],
                      'Predictions' : multiviewPredict } )

    M.Projection = df

    # View DataFrame
    colCombos = list( M.topRankProjections.keys() )
    dfCombos  = DataFrame( {'Columns': colCombos } )

    topRankStats = {}
    for combo in colCombos :
        df_ = M.topRankProjections[ combo ]
        topRankStats[ combo ] = ComputeError( df_['Observations'],
                                              df_['Predictions'] )
    M.topRankStats = topRankStats
    M.View         = concat( [ dfCombos, DataFrame( M.topRankStats.values() ) ],
                             axis = 1 )

    if showPlot :
        PlotObsPred( M.Projection, "", M.D, M.Tp )

    if returnObject :
        return M
    else :
        return { 'Predictions' : M.Projection, 'View' : M.View }

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def EmbedDimension( dataFrame       = None,
                    columns         = "",
                    target          = "",
                    maxE            = 10,
                    lib             = "",
                    pred            = "",
                    Tp              = 1,
                    tau             = -1,
                    exclusionRadius = 0,
                    embedded        = False,
                    validLib        = [],
                    noTime          = False,
                    ignoreNan       = True,
                    verbose         = False,
                    numProcess      = 4,
                    showPlot        = True ):
    '''Estimate optimal embedding dimension [1:maxE].'''

    # Setup Pool
    Evals = [ E for E in range( 1, maxE + 1 ) ]
    args = { 'columns'         : columns,
             'target'          : target,
             'lib'             : lib,
             'pred'            : pred,
             'Tp'              : Tp,
             'tau'             : tau,
             'exclusionRadius' : exclusionRadius,
             'embedded'        : embedded,
             'validLib'        : validLib,
             'noTime'          : noTime,
             'ignoreNan'       : ignoreNan }

    # Create iterable for Pool.starmap, use repeated copies of data, args
    poolArgs = zip( Evals, repeat( dataFrame ), repeat( args ) )

    # Multiargument starmap : EmbedDimSimplexFunc in PoolFunc
    with Pool( processes = numProcess ) as pool :
        rhoList = pool.starmap( PoolFunc.EmbedDimSimplexFunc, poolArgs )

    df = DataFrame( {'E':Evals, 'rho':rhoList} )

    if showPlot :
        title = "Tp=" + str(Tp)
        ax = df.plot( 'E', 'rho', title = title, linewidth = 3 )
        ax.set( xlabel = "Embedding Dimension",
                ylabel = "Prediction Skill ρ" )
        show()

    return df

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def PredictInterval( dataFrame       = None,
                     columns         = "",
                     target          = "",
                     lib             = "",
                     pred            = "",
                     maxTp           = 10,
                     E               = 1,
                     tau             = -1,
                     exclusionRadius = 0,
                     embedded        = False,
                     validLib        = [],
                     noTime          = False,
                     ignoreNan       = True,
                     verbose         = False,
                     numProcess      = 4,
                     showPlot        = True ):
    '''Estimate optimal prediction interval [1:maxTp]'''

    # Setup Pool
    Evals = [ Tp for Tp in range( 1, maxTp + 1 ) ]
    args = { 'columns'         : columns,
             'target'          : target,
             'lib'             : lib,
             'pred'            : pred,
             'E'               : E,
             'tau'             : tau,
             'exclusionRadius' : exclusionRadius,
             'embedded'        : embedded,
             'validLib'        : validLib,
             'noTime'          : noTime,
             'ignoreNan'       : ignoreNan }

    # Create iterable for Pool.starmap, use repeated copies of data, args
    poolArgs = zip( Evals, repeat( dataFrame ), repeat( args ) )

    # Multiargument starmap : EmbedDimSimplexFunc in PoolFunc
    with Pool( processes = numProcess ) as pool :
        rhoList = pool.starmap( PoolFunc.PredictIntervalSimplexFunc, poolArgs )

    df = DataFrame( {'Tp':Evals, 'rho':rhoList} )

    if showPlot :
        if embedded :
            if IsIterable( columns ) :
                E = len( columns )
            else :
                E = len( columns.split() )
        title = "E=" + str( E )
        ax = df.plot( 'Tp', 'rho', title = title, linewidth = 3 )
        ax.set( xlabel = "Forecast Interval",
                ylabel = "Prediction Skill ρ" )
        show()

    return df

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def PredictNonlinear( dataFrame       = None,
                      columns         = "",
                      target          = "",
                      theta           = None,
                      lib             = "",
                      pred            = "",
                      E               = 1,
                      Tp              = 1,
                      knn             = 0,
                      tau             = -1,
                      exclusionRadius = 0,
                      solver          = None,
                      embedded        = False,
                      validLib        = [],
                      noTime          = False,
                      ignoreNan       = True,
                      verbose         = False,
                      numProcess      = 4,
                      showPlot        = True ):
    '''Estimate S-map localisation over theta.'''

    if theta is None :
        theta = [ 0.01, 0.1, 0.3, 0.5, 0.75, 1,
                  1.5, 2, 3, 4, 5, 6, 7, 8, 9 ]
    elif not IsIterable( theta ) :
        theta = [float(t) for t in theta.split()]

    # Setup Pool
    args = { 'columns'         : columns,
             'target'          : target, 
             'lib'             : lib,
             'pred'            : pred,
             'E'               : E,
             'Tp'              : Tp,
             'tau'             : tau,
             'exclusionRadius' : exclusionRadius,
             'solver'          : solver,
             'embedded'        : embedded,
             'validLib'        : validLib,
             'noTime'          : noTime,
             'ignoreNan'       : ignoreNan }

    # Create iterable for Pool.starmap, use repeated copies of data, args
    poolArgs = zip( theta, repeat( dataFrame ), repeat( args ) )

    # Multiargument starmap : EmbedDimSimplexFunc in PoolFunc
    with Pool( processes = numProcess ) as pool :
        rhoList = pool.starmap( PoolFunc.PredictNLSMapFunc, poolArgs )

    df = DataFrame( {'theta':theta, 'rho':rhoList} )

    if showPlot :
        if embedded :
            if IsIterable( columns ) :
                E = len( columns )
            else :
                E = len( columns.split() )
        title = "E=" + str( E )

        ax = df.plot( 'theta', 'rho', title = title, linewidth = 3 )
        ax.set( xlabel = "S-map Localisation (θ)",
                ylabel = "Prediction Skill ρ" )
        show()

    return df
