'''Python interface to Empirical Dynamic Modeling (EDM) C++ library (cppEDM)
   https://github.com/SugiharaLab/cppEDM.'''

from pandas            import DataFrame
from matplotlib.pyplot import show, axhline

from os import name as osName # Windog cannot use LAPACK/BLAS
if osName == 'nt':
    from multiprocessing import Pool
    from itertools       import repeat
    from pandas          import read_csv
    from numpy           import zeros
    from sklearn.linear_model import LinearRegression

import pyBindEDM
import pyEDM.AuxFunc

#------------------------------------------------------------------------
# 
#------------------------------------------------------------------------
def MakeBlock( dataFrame,
               E             = 0, 
               tau           = -1,
               columnNames   = [],
               deletePartial = False ):
    '''Takens time-delay embedding on columnNames in Pandas DataFrame.
       NaN will be present in tau * (E-1) rows if deletePartial False.'''

    if not isinstance( dataFrame, DataFrame ) :
        raise Exception( "MakeBlock(): dataFrame is not a Pandas DataFrame." )

    DF = pyEDM.AuxFunc.PandasDataFrametoDF( dataFrame )

    # D is a Python dict from pybind11 < cppEDM Embed
    D = pyBindEDM.MakeBlock( DF,
                             E, 
                             tau,
                             columnNames,
                             deletePartial )

    df = DataFrame( D ) # Convert to pandas DataFrame

    return df

#------------------------------------------------------------------------
# 
#------------------------------------------------------------------------
def Embed( pathIn    = "./",
           dataFile  = "",
           dataFrame = None,
           E         = 0, 
           tau       = -1,
           columns   = "",
           verbose   = False ):
    '''Takens time-delay embedding on path/file.
       Embed DataFrame columns (subset) in E dimensions.
       Calls MakeBlock() after validation and column subset selection.'''

    # Establish DF struct as empty list, or from Pandas DataFrame
    DF = pyEDM.AuxFunc.GetDF( pathIn, dataFile, dataFrame, False, "Embed" )

    # Format columns as string for cppEDM
    columns = pyEDM.AuxFunc.ArgTo_cppEDM_String( columns )

    # D is a Python dict from pybind11 < cppEDM Embed
    D = pyBindEDM.Embed( pathIn,
                         dataFile,
                         DF,
                         E, 
                         tau,
                         columns,
                         verbose )

    df = DataFrame( D ) # Convert to pandas DataFrame

    return df

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def Simplex( pathIn          = "./",
             dataFile        = "",
             dataFrame       = None,
             pathOut         = "./",
             predictFile     = "",
             lib             = "",
             pred            = "",
             E               = 0, 
             Tp              = 1,
             knn             = 0,
             tau             = -1,
             exclusionRadius = 0,
             columns         = "",
             target          = "", 
             embedded        = False,
             verbose         = False,
             const_pred      = False,
             showPlot        = False,
             validLib        = [],
             generateSteps   = 0,
             generateLibrary = False,
             parameterList   = False,
             noTime          = False
             ):
    '''Simplex prediction on path/file.'''

    # Establish DF struct as empty list, or from Pandas DataFrame
    DF = pyEDM.AuxFunc.GetDF( pathIn, dataFile, dataFrame, noTime, "Simplex" )
    if dataFile and noTime :
        dataFile = '' # DF was created in GetDF(), disable dataFile read

    # If lib, pred, columns are not string, but iterable, convert to string
    if pyEDM.AuxFunc.IsIterable( lib ) :
        lib = ' '.join( map( str, lib ) )
    if pyEDM.AuxFunc.IsIterable( pred ) :
        pred = ' '.join( map( str, pred ) )

    # target is string; format for cppEDM if has space
    if ' ' in target :
        target = target + ','

    # Format columns as string for cppEDM
    columns = pyEDM.AuxFunc.ArgTo_cppEDM_String( columns )

    # D is a Python dict from pybind11 < cppEDM Simplex:
    #  { "predictions" : {}, ["parameters" : {}] }
    D = pyBindEDM.Simplex( pathIn,
                           dataFile,
                           DF,
                           pathOut,
                           predictFile,
                           lib,
                           pred,
                           E, 
                           Tp,
                           knn,
                           tau,
                           exclusionRadius,
                           columns,
                           target, 
                           embedded,
                           const_pred,
                           verbose,
                           validLib,
                           generateSteps,
                           generateLibrary,
                           parameterList )

    df = DataFrame( D['predictions'] ) # Convert to pandas DataFrame

    if showPlot :
        if embedded :
            E = len( columns.split() )
        pyEDM.AuxFunc.PlotObsPred( df, dataFile, E, Tp )

    if parameterList :
        SDict = { 'predictions' : df }
        SDict[ 'parameters' ] = D[ 'parameters' ]
        return SDict
    else :
        return df

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def SMap( pathIn          = "./",
          dataFile        = "",
          dataFrame       = None,
          pathOut         = "./",
          predictFile     = "",
          lib             = "",
          pred            = "",
          E               = 0, 
          Tp              = 1,
          knn             = 0,
          tau             = -1,
          theta           = 0,
          exclusionRadius = 0,
          columns         = "",
          target          = "",
          smapCoefFile    = "",
          smapSVFile      = "",
          solver          = None,
          embedded        = False,
          verbose         = False,
          const_pred      = False,
          showPlot        = False,
          validLib        = [],
          ignoreNan       = True,
          generateSteps   = 0,
          generateLibrary = False,
          parameterList   = False,
          noTime          = False
          ):
    '''S-Map prediction on path/file.'''

    # Establish DF struct as empty list, or from Pandas DataFrame
    DF = pyEDM.AuxFunc.GetDF( pathIn, dataFile, dataFrame, noTime, "SMap" )
    if dataFile and noTime :
        dataFile = '' # DF was created in GetDF(), disable dataFile read

    # If lib, pred, columns are not string, but iterable, convert to string
    if pyEDM.AuxFunc.IsIterable( lib ) :
        lib = ' '.join( map( str, lib ) )
    if pyEDM.AuxFunc.IsIterable( pred ) :
        pred = ' '.join( map( str, pred ) )

    # target is string; format for cppEDM if has space
    if ' ' in target :
        target = target + ',' # space in target: add , for cppEDM

    # Format columns as string for cppEDM
    columns = pyEDM.AuxFunc.ArgTo_cppEDM_String( columns )

    # Validate the solver if one was passed in
    if solver :
        supportedSolvers = [ 'LinearRegression',
                             'Ridge',   'Lasso',   'ElasticNet',
                             'RidgeCV', 'LassoCV', 'ElasticNetCV' ]
        if not solver.__class__.__name__ in supportedSolvers :
            raise Exception( "SMap(): Invalid solver." )

    elif osName == 'nt':
        # No solver was specified. Default to LinearRegression/SVD
        # since supporting BLAS/LAPACK on Windog is a fools errand.
        solver = LinearRegression()

    # D is a Python dict from pybind11 < cppEDM SMap:
    #  { "predictions" : {}, "coefficients" : {},
    #    "singularValues : {}, ["parameters" : {}] }
    D = pyBindEDM.SMap( pathIn,
                        dataFile,
                        DF,
                        pathOut,
                        predictFile,
                        lib,
                        pred,
                        E, 
                        Tp,
                        knn,
                        tau,
                        theta,
                        exclusionRadius,
                        columns,
                        target,
                        smapCoefFile,
                        smapSVFile,
                        solver,
                        embedded,
                        const_pred,
                        verbose,
                        validLib,
                        ignoreNan,
                        generateSteps,
                        generateLibrary,
                        parameterList )

    df_pred = DataFrame( D['predictions']    ) # Convert to pandas DataFrame
    df_coef = DataFrame( D['coefficients']   ) # Convert to pandas DataFrame
    df_SV   = DataFrame( D['singularValues'] ) # Convert to pandas DataFrame

    SMapDict = { 'predictions'    : df_pred,
                 'coefficients'   : df_coef,
                 'singularValues' : df_SV }

    if parameterList :
        SMapDict[ 'parameters' ] = D[ 'parameters' ]

    if showPlot :
        if embedded :
            E = len( columns.split() )
        pyEDM.AuxFunc.PlotObsPred( df_pred, dataFile, E, Tp, False )
        pyEDM.AuxFunc.PlotCoeff  ( df_coef, dataFile, E, Tp )

    return SMapDict

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def Multiview( pathIn          = "./",
               dataFile        = "",
               dataFrame       = None,
               pathOut         = "./",
               predictFile     = "",
               lib             = "",
               pred            = "",
               D               = 0, 
               E               = 1, 
               Tp              = 1,
               knn             = 0,
               tau             = -1,
               columns         = "",
               target          = "",
               multiview       = 0,
               exclusionRadius = 0,
               trainLib        = True,
               excludeTarget   = False,
               parameterList   = False,
               verbose         = False,
               numThreads      = 4,
               showPlot        = False,
               noTime          = False ):
    '''Multiview prediction on path/file.'''

    # Establish DF struct as empty list, or from Pandas DataFrame
    DF = pyEDM.AuxFunc.GetDF( pathIn, dataFile, dataFrame, noTime, "Multiview" )
    if dataFile and noTime :
        dataFile = '' # DF was created in GetDF(), disable dataFile read

    # If lib, pred, columns are not string, but iterable, convert to string
    if pyEDM.AuxFunc.IsIterable( lib ) :
        lib = ' '.join( map( str, lib ) )
    if pyEDM.AuxFunc.IsIterable( pred ) :
        pred = ' '.join( map( str, pred ) )

    # target is string; format for cppEDM if has space
    if ' ' in target :
        target = target + ',' # space in target: add , for cppEDM

    # Format columns as string for cppEDM
    columns = pyEDM.AuxFunc.ArgTo_cppEDM_String( columns )

    # D is a Python dict from pybind11 < cppEDM Multiview:
    #  { "View" : {}, "Predictions" : {}, ColumnNames : {} }
    D = pyBindEDM.Multiview( pathIn,
                             dataFile,
                             DF,
                             pathOut,
                             predictFile,
                             lib,
                             pred,
                             D, 
                             E, 
                             Tp,
                             knn,
                             tau,
                             columns,
                             target,
                             multiview,
                             exclusionRadius,
                             trainLib,
                             excludeTarget,
                             parameterList,
                             verbose,
                             numThreads )

    df_pred = DataFrame( D['Predictions'] ) # Convert to pandas DataFrame
    view    = DataFrame( D['View'] )

    # Add columnNames to view DataFrame
    for key in D['ColumnNames'].keys() :
        view[ key ] = D['ColumnNames'][ key ]
    
    MV = { 'Predictions' : df_pred, 'View' : view }

    if parameterList :
        MV[ 'parameters' ] = D[ 'parameters' ]

    if showPlot :
        pyEDM.AuxFunc.PlotObsPred( df_pred, dataFile, E, Tp )

    return MV

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def CCM( pathIn           = "./",
         dataFile         = "",
         dataFrame        = None,
         pathOut          = "./",
         predictFile      = "",
         E                = 0, 
         Tp               = 0,
         knn              = 0,
         tau              = -1,
         exclusionRadius  = 0,
         columns          = "",
         target           = "",
         libSizes         = "",
         sample           = 0,
         random           = True,
         replacement      = False,
         seed             = 0,
         embedded         = False,
         includeData      = False,
         parameterList    = False,
         verbose          = False,
         showPlot         = False,
         noTime           = False ) :
    '''Convergent Cross Mapping on path/file.'''

    # Establish DF struct as empty list, or from Pandas DataFrame
    DF = pyEDM.AuxFunc.GetDF( pathIn, dataFile, dataFrame, noTime, "CCM" )
    if dataFile and noTime :
        dataFile = '' # DF was created in GetDF(), disable dataFile read

    # If libSizes are not string, but iterable convert to string
    if pyEDM.AuxFunc.IsIterable( libSizes ) :
        libSizes = ' '.join( map( str, libSizes ) )

    # Format target and columns as string for cppEDM
    target  = pyEDM.AuxFunc.ArgTo_cppEDM_String( target )
    columns = pyEDM.AuxFunc.ArgTo_cppEDM_String( columns )
    
    # D is a Python dict from pybind11 < cppEDM CCM
    D = pyBindEDM.CCM( pathIn,
                       dataFile,
                       DF,
                       pathOut,
                       predictFile,
                       E, 
                       Tp,
                       knn,
                       tau,
                       exclusionRadius,
                       columns,
                       target,
                       libSizes,
                       sample,
                       random,
                       replacement,
                       seed,
                       embedded,
                       includeData,
                       parameterList,
                       verbose )

    # D has { "LibMeans" : DF }
    # and if includeData has : { PredictStats1 : DF, PredictStats2 : DF }
    libMeans = DataFrame( D[ "LibMeans" ] ) # Convert to pandas DataFrame

    # If includeData, create dict with means and individual prediction stats
    if includeData :
        CM = { 'LibMeans'      : libMeans,
               'PredictStats1' : DataFrame( D[ "PredictStats1" ] ),
               'PredictStats2' : DataFrame( D[ "PredictStats2" ] ) }

    if parameterList and includeData :
        CM[ 'parameters' ] = D[ 'parameters' ]

    if showPlot :
        title = dataFile + "\nE=" + str(E)

        ax = libMeans.plot( 'LibSize',
                            [ libMeans.columns[1], libMeans.columns[2] ],
                            title = title, linewidth = 3 )
        ax.set( xlabel = "Library Size", ylabel = "Correlation ρ" )
        axhline( y = 0, linewidth = 1 )
        show()

    if includeData :
        return CM
    else :
        return libMeans

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def EmbedDimension( pathIn          = "./",
                    dataFile        = "",
                    dataFrame       = None,
                    pathOut         = "./",
                    predictFile     = "",
                    lib             = "",
                    pred            = "",
                    maxE            = 10,
                    Tp              = 1,
                    tau             = -1,
                    exclusionRadius = 0,
                    columns         = "",
                    target          = "",
                    embedded        = False,
                    verbose         = False,
                    validLib        = [],
                    numThreads      = 4,
                    showPlot        = True,
                    noTime          = False ):
    '''Estimate optimal embedding dimension [1:maxE].'''

    # Establish DF struct as empty list, or from Pandas DataFrame
    DF = pyEDM.AuxFunc.GetDF( pathIn, dataFile, dataFrame, noTime,
                              "EmbedDimension" )
    if dataFile and noTime :
        dataFile = '' # DF was created in GetDF(), disable dataFile read

    # If lib, pred, columns are not string, but iterable, convert to string
    # NOTE: columns joined on ',' to enable names with whitespace in cppEDM
    if pyEDM.AuxFunc.IsIterable( lib ) :
        lib = ' '.join( map( str, lib ) )
    if pyEDM.AuxFunc.IsIterable( pred ) :
        pred = ' '.join( map( str, pred ) )

    # target is string; format for cppEDM if has space
    if ' ' in target :
        target = target + ',' # space in target: add , for cppEDM

    # Format columns as string for cppEDM
    columns = pyEDM.AuxFunc.ArgTo_cppEDM_String( columns )

    # D is a Python dict from pybind11 < cppEDM CCM
    D = pyBindEDM.EmbedDimension( pathIn,
                                  dataFile,
                                  DF,
                                  pathOut,
                                  predictFile,
                                  lib,
                                  pred, 
                                  maxE,
                                  Tp,
                                  tau,
                                  exclusionRadius,
                                  columns,
                                  target,
                                  embedded,
                                  verbose,
                                  validLib,
                                  numThreads )

    df = DataFrame( D ) # Convert to pandas DataFrame

    if showPlot :
        title = dataFile + "\nTp=" + str(Tp)
    
        ax = df.plot( 'E', 'rho', title = title, linewidth = 3 )
        ax.set( xlabel = "Embedding Dimension",
                ylabel = "Prediction Skill ρ" )
        show()

    return df

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def PredictInterval( pathIn          = "./",
                     dataFile        = "",
                     dataFrame       = None,
                     pathOut         = "./",
                     predictFile     = "",
                     lib             = "",
                     pred            = "",
                     maxTp           = 10,
                     E               = 1,
                     tau             = -1,
                     exclusionRadius = 0,
                     columns         = "",
                     target          = "",
                     embedded        = False,
                     verbose         = False,
                     validLib        = [],
                     numThreads      = 4,
                     showPlot        = True,
                     noTime          = False ):
    '''Estimate optimal prediction interval [1:maxTp]'''

    # Establish DF struct as empty list, or from Pandas DataFrame
    DF = pyEDM.AuxFunc.GetDF( pathIn, dataFile, dataFrame, noTime,
                              "PredictInterval" )
    if dataFile and noTime :
        dataFile = '' # DF was created in GetDF(), disable dataFile read

    # If lib, pred, columns are not string, but iterable, convert to string
    # NOTE: columns joined on ',' to enable names with whitespace in cppEDM
    if pyEDM.AuxFunc.IsIterable( lib ) :
        lib = ' '.join( map( str, lib ) )
    if pyEDM.AuxFunc.IsIterable( pred ) :
        pred = ' '.join( map( str, pred ) )

    # target is string; format for cppEDM if has space
    if ' ' in target :
        target = target + ',' # space in target: add , for cppEDM

    # Format columns as string for cppEDM
    columns = pyEDM.AuxFunc.ArgTo_cppEDM_String( columns )

    # D is a Python dict from pybind11 < cppEDM PredictInterval
    D = pyBindEDM.PredictInterval( pathIn,
                                   dataFile,
                                   DF,
                                   pathOut,
                                   predictFile,
                                   lib,
                                   pred, 
                                   maxTp,
                                   E,
                                   tau,
                                   exclusionRadius,
                                   columns,
                                   target,
                                   embedded,
                                   verbose,
                                   validLib,
                                   numThreads )

    df = DataFrame( D ) # Convert to pandas DataFrame

    if showPlot :
        if embedded :
            E = len( columns.split() )
        title = dataFile + "\nE=" + str( E )

        ax = df.plot( 'Tp', 'rho', title = title, linewidth = 3 )
        ax.set( xlabel = "Forecast Interval",
                ylabel = "Prediction Skill ρ" )
        show()

    return df

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def PredictNonlinear( pathIn          = "./",
                      dataFile        = "",
                      dataFrame       = None,
                      pathOut         = "./",
                      predictFile     = "",
                      lib             = "",
                      pred            = "",
                      theta           = "",
                      E               = 1,
                      Tp              = 1,
                      knn             = 0,
                      tau             = -1,
                      exclusionRadius = 0,
                      columns         = "",
                      target          = "",
                      embedded        = False,
                      verbose         = False,
                      validLib        = [],
                      ignoreNan       = True,
                      numThreads      = 4,
                      showPlot        = True,
                      noTime          = False ):
    '''Estimate S-map localisation over theta.

       This function has two implementations since Windog does not use
       the BLAS SVD solver dgelss but sklearn LinearSolver by default.
       Supporting OpenBLAS on Windog is not feasible.

       If the platform is Windog, perform the computations "manually"
       using SMap above in a multiprocess wrapper. This allows
       specification of the SMap solver from sklearn.linear_model.

       If the platform is not Windog, call the pybind11 : cppEDM function.
    '''

    if osName == 'nt':
        # Manually implement PredictNonlinear with the sklearn solver
        # Return Pandas DataFrame: 'Theta', 'rho'
        df = PredictNonlinearInternal( pathIn,
                                       dataFile,
                                       dataFrame,
                                       pathOut,
                                       predictFile,
                                       lib,
                                       pred,
                                       theta,
                                       E,
                                       Tp,
                                       knn,
                                       tau,
                                       exclusionRadius,
                                       columns,
                                       target,
                                       embedded,
                                       verbose,
                                       validLib,
                                       ignoreNan,
                                       numThreads )
    else:
        # Call pybind11 : cppEDM PredictNonlinear
        # Establish DF struct as empty list, or from Pandas DataFrame
        DF = pyEDM.AuxFunc.GetDF( pathIn, dataFile, dataFrame, noTime,
                                  "PredictNonlinear" )
        if dataFile and noTime :
            dataFile = '' # DF was created in GetDF(), disable dataFile read

        # If lib,pred,columns,theta are not string, but iterable, convert to str
        # NOTE: columns joined on ',' to enable names with whitespace in cppEDM
        if pyEDM.AuxFunc.IsIterable( lib ) :
            lib = ' '.join( map( str, lib ) )
        if pyEDM.AuxFunc.IsIterable( pred ) :
            pred = ' '.join( map( str, pred ) )

        if pyEDM.AuxFunc.IsIterable( theta ) :
            theta = ' '.join( map( str, theta ) )
    
        # target is string; format for cppEDM if has space
        if ' ' in target :
            target = target + ',' # space in target: add , for cppEDM

        # Format columns as string for cppEDM
        columns = pyEDM.AuxFunc.ArgTo_cppEDM_String( columns )
        
        # Call pybind11 : cppEDM PredictNonlinear
        # D is a Python dict from pybind11 < cppEDM PredictNonlinear
        D = pyBindEDM.PredictNonlinear( pathIn,
                                        dataFile,
                                        DF,
                                        pathOut,
                                        predictFile,
                                        lib,
                                        pred,
                                        theta,
                                        E,
                                        Tp,
                                        knn,
                                        tau,
                                        exclusionRadius,
                                        columns,
                                        target,
                                        embedded,
                                        verbose,
                                        validLib,
                                        ignoreNan,
                                        numThreads )

        df = DataFrame( D ) # Convert to pandas DataFrame

    if showPlot :
        if embedded :
            E = len( columns.split() )
        title = dataFile + "\nE=" + str( E )

        ax = df.plot( 'Theta', 'rho', title = title, linewidth = 3 )
        ax.set( xlabel = "S-map Localisation (θ)",
                ylabel = "Prediction Skill ρ" )
        show()

    return df

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def PredictNonlinearInternal( pathIn          = "./",
                              dataFile        = "",
                              dataFrame       = None,
                              pathOut         = "./",
                              predictFile     = "",
                              lib             = "",
                              pred            = "",
                              theta           = "",
                              E               = 1,
                              Tp              = 1,
                              knn             = 0,
                              tau             = -1,
                              exclusionRadius = 0,
                              columns         = "",
                              target          = "",
                              embedded        = False,
                              verbose         = False,
                              validLib        = [],
                              ignoreNan       = True,
                              numThreads      = 4,
                              showPlot        = True,
                              noTime          = False ) :

    '''If the platform is Windog, perform computations "manually"
       using SMap above in a multiprocess wrapper. This allows
       specification of the SMap solver from sklearn.linear_model.
    '''

    # If pandas DataFrame not passed in, create one from dataFile
    if dataFrame is None :
        dataFrame = read_csv( pathIn + dataFile )

    if not isinstance( dataFrame, DataFrame ) :
        raise Exception( "PredictNonlinearInternal(): Failed to get dataFrame." )

    if pyEDM.AuxFunc.IsIterable( theta ) :
        # theta is a list
        if len( theta ) > 0 :
            thetaValues = theta
        else :
            raise Exception( "PredictNonlinearInternal(): Empty theta list." )
    else :
        # theta is a string
        if len( theta ) == 0 : # string is empty
            thetaValues = [ 0.01, 0.1, 0.3, 0.5, 0.75, 1,
                            1.5, 2, 3, 4, 5, 6, 7, 8, 9 ]
        else :
            # Convert string to list
            thetaValues = [ float(x) for x in theta.split() ]

    # Create args dictionary for SMapFunc
    args = { 'lib':lib, 'pred':pred, 'E':E, 'Tp':Tp, 'tau':tau,
             'exclusionRadius':exclusionRadius, 'columns':columns,
             'target':target, 'embedded':embedded, 'validLib':validLib,
             'ignoreNan':ignoreNan, 'noTime':noTime }

    # Create iterable for Pool.starmap, use repeated copies of args, data
    poolArgs = zip( thetaValues, repeat( args ), repeat( dataFrame ) )

    # Use pool.starmap to distribute among cores
    pool = Pool( processes = numThreads )

    # starmap: elements of the iterable argument are iterables
    #          that are unpacked as arguments
    SMapList = pool.starmap( SMapFunc, poolArgs )

    # SMapList is a list of SMap dictionaries, process rho values
    rho = zeros( len( SMapList ) )
    for i in range( len( SMapList ) ):
        SM           = SMapList[ i ]
        SM_df        = SM['predictions']
        predictions  = SM_df['Predictions']
        observations = SM_df['Observations']
        rho[ i ]     = pyBindEDM.ComputeError(observations, predictions)['rho']

    # Return pandas DataFrame of Theta : rho
    df = DataFrame( {'Theta':thetaValues, 'rho':rho} )

    return df

#----------------------------------------------------------------------------
def SMapFunc( theta, args, data ):
    '''Call pyEDM SMap using theta, args, and data'''

    sm = SMap( dataFrame       = data,
               lib             = args['lib'],
               pred            = args['pred'],
               E               = args['E'],
               Tp              = args['Tp'],
               tau             = args['tau'],
               theta           = theta,
               exclusionRadius = args['exclusionRadius'],
               columns         = args['columns'],
               target          = args['target'],
               embedded        = args['embedded'],
               validLib        = args['validLib'],
               ignoreNan       = args['ignoreNan'],
               noTime          = args['noTime'],
               showPlot        = False )

    return sm
