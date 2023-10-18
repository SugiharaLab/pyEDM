'''Python interface to Empirical Dynamic Modeling (EDM) C++ library (cppEDM)
   https://github.com/SugiharaLab/cppEDM.'''

from pandas            import DataFrame
from matplotlib.pyplot import show, axhline

import pyBindEDM
import pyEDM.AuxFunc

#------------------------------------------------------------------------
# 
#------------------------------------------------------------------------
def GetDF( dataFile, dataFrame, noTime, func ):
    '''Establish DF as empty list or create from Pandas DataFrame 
       DF is a C++ struct to hold cppEDM DataFrame values 
       see: src/bindings/PyBind.h
       If the API function is passed dataFile, dataFile is
       passed directly to cppEDM API, no need to create DF.
    '''
    if dataFile :
        DF = pyBindEDM.DF() 
    elif isinstance( dataFrame, DataFrame ) :
        if dataFrame.empty :
            raise Exception( func + "(): dataFrame is empty." )
        DF = pyEDM.AuxFunc.PandasDataFrametoDF( dataFrame, noTime )
    else :
        raise Exception( func + "(): Invalid data input." )

    return DF

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

    # Establish DF as empty list or Pandas DataFrame for Embed()
    DF = GetDF( dataFile, dataFrame, False, "Embed" )

    # If columns are not string, but iterable, convert to string
    if pyEDM.AuxFunc.NotStringIterable( columns ) :
        columns = ' '.join( map( str,columns   ) )

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

    # Establish DF as empty list or Pandas DataFrame for Simplex()
    DF = GetDF( dataFile, dataFrame, noTime, "Simplex" )

    # If lib, pred, columns are not string, but iterable, convert to string
    if pyEDM.AuxFunc.NotStringIterable( lib ) :
        lib = ' '.join( map( str, lib ) )
    if pyEDM.AuxFunc.NotStringIterable( pred ) :
        pred = ' '.join( map( str, pred ) )
    if pyEDM.AuxFunc.NotStringIterable( columns ) :
        columns = ' '.join( map( str, columns ) )

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

    # Establish DF as empty list or Pandas DataFrame for SMap()
    DF = GetDF( dataFile, dataFrame, noTime, "SMap" )

    # If lib, pred, columns are not string, but iterable, convert to string
    if pyEDM.AuxFunc.NotStringIterable( lib ) :
        lib = ' '.join( map( str, lib ) )
    if pyEDM.AuxFunc.NotStringIterable( pred ) :
        pred = ' '.join( map( str, pred ) )
    if pyEDM.AuxFunc.NotStringIterable( columns ) :
        columns = ' '.join( map( str, columns ) )

    # Validate the solver if one was passed in
    if solver :
        supportedSolvers = [ 'LinearRegression',
                             'Ridge',   'Lasso',   'ElasticNet',
                             'RidgeCV', 'LassoCV', 'ElasticNetCV' ]
        if not solver.__class__.__name__ in supportedSolvers :
            raise Exception( "SMap(): Invalid solver." )

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

    # Establish DF as empty list or Pandas DataFrame for Multiview()
    DF = GetDF( dataFile, dataFrame, noTime, "Multiview" )

    # If lib, pred, columns are not string, but iterable, convert to string
    if pyEDM.AuxFunc.NotStringIterable( lib ) :
        lib = ' '.join( map( str, lib ) )
    if pyEDM.AuxFunc.NotStringIterable( pred ) :
        pred = ' '.join( map( str, pred ) )
    if pyEDM.AuxFunc.NotStringIterable( columns ) :
        columns = ' '.join( map( str, columns ) )

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

    # Establish DF as empty list or Pandas DataFrame for CCM()
    DF = GetDF( dataFile, dataFrame, noTime, "CCM" )

    # If columns, libSizes, target are not string, but iterable convert to string
    if pyEDM.AuxFunc.NotStringIterable( columns ) :
        columns = ' '.join( map( str, columns ) )
    if pyEDM.AuxFunc.NotStringIterable( libSizes ) :
        libSizes = ' '.join( map( str, libSizes ) )
    if pyEDM.AuxFunc.NotStringIterable( target ) :
        target = ' '.join( map( str, target ) )

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

    # Establish DF as empty list or Pandas DataFrame for EmbedDimension()
    DF = GetDF( dataFile, dataFrame, noTime, "EmbedDimension" )

    # If lib, pred, columns are not string, but iterable, convert to string
    if pyEDM.AuxFunc.NotStringIterable( lib ) :
        lib = ' '.join( map( str, lib ) )
    if pyEDM.AuxFunc.NotStringIterable( pred ) :
        pred = ' '.join( map( str, pred ) )
    if pyEDM.AuxFunc.NotStringIterable( columns ) :
        columns = ' '.join( map( str, columns ) )

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

    # Establish DF as empty list or Pandas DataFrame for PredictInterval()
    DF = GetDF( dataFile, dataFrame, noTime, "PredictInterval" )
    
    # If lib, pred, columns are not string, but iterable, convert to string
    if pyEDM.AuxFunc.NotStringIterable( lib ) :
        lib = ' '.join( map( str, lib ) )
    if pyEDM.AuxFunc.NotStringIterable( pred ) :
        pred = ' '.join( map( str, pred ) )
    if pyEDM.AuxFunc.NotStringIterable( columns ) :
        columns = ' '.join( map( str, columns ) )

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
                      numThreads      = 4,
                      showPlot        = True,
                      noTime          = False ):
    '''Estimate S-map localisation over theta.'''

    # Establish DF as empty list or Pandas DataFrame for PredictNonlinear()
    DF = GetDF( dataFile, dataFrame, noTime, "PredictNonlinear" )

    # If lib,pred,columns,theta are not string, but iterable, convert to string
    if pyEDM.AuxFunc.NotStringIterable( lib ) :
        lib = ' '.join( map( str, lib ) )
    if pyEDM.AuxFunc.NotStringIterable( pred ) :
        pred = ' '.join( map( str, pred ) )
    if pyEDM.AuxFunc.NotStringIterable( columns ) :
        columns = ' '.join( map( str, columns ) )
    if pyEDM.AuxFunc.NotStringIterable( theta ) :
        theta = ' '.join( map( str, theta ) )

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
