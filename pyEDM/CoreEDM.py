'''Python interface to Empirical Dynamic Modeling (EDM) C++ library (cppEDM)
   https://github.com/SugiharaLab/cppEDM.'''

from pandas            import DataFrame
from matplotlib.pyplot import show, axhline

import pyBindEDM
import pyEDM.AuxFunc
 
#------------------------------------------------------------------------
# 
#------------------------------------------------------------------------
def MakeBlock( dataFrame,
               E           = 0, 
               tau         = 1,
               columnNames = "",
               verbose     = False ):
    '''Takens time-delay embedding on columnNames in Pandas DataFrame.
       Truncates the timeseries by tau * (E-1) rows.'''

    if not isinstance( dataFrame, DataFrame ) :
        raise Exception( "MakeBlock(): dataFrame is not a Pandas DataFrame." )
    
    DF = pyEDM.AuxFunc.PandasDataFrametoDF( dataFrame )
    
    # D is a Python dict from pybind11 < cppEDM Embed
    D = pyBindEDM.MakeBlock( DF,
                             E, 
                             tau,
                             columnNames,
                             verbose )

    df = DataFrame( D ) # Convert to pandas DataFrame

    return df

#------------------------------------------------------------------------
# 
#------------------------------------------------------------------------
def Embed( pathIn    = "./",
           dataFile  = "",
           dataFrame = None,
           E         = 0, 
           tau       = 1,
           columns   = "",
           verbose   = False ):
    '''Takens time-delay embedding on path/file.
       Embed DataFrame columns (subset) in E dimensions.
       Calls MakeBlock() after validation and column subset selection.'''

    # Establish DF as empty list or Pandas DataFrame for Embed()
    if dataFile :
        DF = pyBindEDM.DF() 
    elif isinstance( dataFrame, DataFrame ) :
        if dataFrame.empty :
            raise Exception( "Embed(): dataFrame is empty." )
        DF = pyEDM.AuxFunc.PandasDataFrametoDF( dataFrame )
    else :
        raise Exception( "Embed(): Invalid data input." )
    
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
def Simplex( pathIn       = "./",
             dataFile     = "",
             dataFrame    = None,
             pathOut      = "./",
             predictFile  = "",
             lib          = "",
             pred         = "",
             E            = 0, 
             Tp           = 1,
             knn          = 0,
             tau          = 1,
             exclusionRadius = 0,
             columns      = "",
             target       = "", 
             embedded     = False,
             verbose      = False,
             const_pred   = False,
             showPlot     = False ):
    '''Simplex prediction on path/file.'''

    # Establish DF as empty list or Pandas DataFrame for Simplex()
    if dataFile :
        DF = pyBindEDM.DF()
    elif isinstance( dataFrame, DataFrame ) :
        if dataFrame.empty :
            raise Exception( "Simplex(): dataFrame is empty." )
        DF = pyEDM.AuxFunc.PandasDataFrametoDF( dataFrame )
    else :
        raise Exception( "Simplex(): Invalid data input." )
    
    # D is a Python dict from pybind11 < cppEDM Simplex 
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
                           verbose  )

    df = DataFrame( D ) # Convert to pandas DataFrame

    if showPlot :
        pyEDM.AuxFunc.PlotObsPred( df, dataFile, E, Tp )
    
    return df

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def SMap( pathIn       = "./",
          dataFile     = "",
          dataFrame    = None,
          pathOut      = "./",
          predictFile  = "",
          lib          = "",
          pred         = "",
          E            = 0, 
          Tp           = 1,
          knn          = 0,
          tau          = 1,
          theta        = 0,
          exclusionRadius = 0,
          columns      = "",
          target       = "",
          smapFile     = "",
          jacobians    = "",
          embedded     = False,
          verbose      = False,
          const_pred   = False,
          showPlot     = False ):
    '''S-Map prediction on path/file.'''

    # Establish DF as empty list or Pandas DataFrame for SMap()
    if dataFile :
        DF = pyBindEDM.DF() 
    elif isinstance( dataFrame, DataFrame ) :
        if dataFrame.empty :
            raise Exception( "SMap(): dataFrame is empty." )
        DF = pyEDM.AuxFunc.PandasDataFrametoDF( dataFrame )
    else :
        raise Exception( "SMap(): Invalid data input." )
    
    # D is a Python dict from pybind11 < cppEDM SMap:
    #  { "predictions" : {}, "coefficients" : {} }
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
                        smapFile,
                        jacobians,
                        embedded,
                        const_pred,
                        verbose  )

    df_pred = DataFrame( D['predictions']  ) # Convert to pandas DataFrame
    df_coef = DataFrame( D['coefficients'] ) # Convert to pandas DataFrame

    if showPlot :
        pyEDM.AuxFunc.PlotObsPred( df_pred, dataFile, E, Tp, False )
        pyEDM.AuxFunc.PlotCoeff  ( df_coef, dataFile, E, Tp )

    SMapDict = { 'predictions' : df_pred, 'coefficients' : df_coef }
    
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
               E               = 0, 
               Tp              = 1,
               knn             = 0,
               tau             = 1,
               columns         = "",
               target          = "",
               multiview       = 0,
               exclusionRadius = 0,
               verbose         = False,
               numThreads      = 4,
               showPlot        = False ):
    '''Multiview prediction on path/file.'''

    # Establish DF as empty list or Pandas DataFrame for Multiview()
    if dataFile :
        DF = pyBindEDM.DF() 
    elif isinstance( dataFrame, DataFrame ) :
        if dataFrame.empty :
            raise Exception( "Multiview(): dataFrame is empty." )
        DF = pyEDM.AuxFunc.PandasDataFrametoDF( dataFrame )
    else :
        raise Exception( "Multiview(): Invalid data input." )
    
    # D is a Python dict from pybind11 < cppEDM Multiview:
    #  { "View" : < vector< string >, "Predictions" : {} }
    D = pyBindEDM.Multiview( pathIn,
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
                             columns,
                             target,
                             multiview,
                             exclusionRadius,
                             verbose,
                             numThreads )
    
    df_pred = DataFrame( D['Predictions'] ) # Convert to pandas DataFrame
    view    = DataFrame( D['View'] )

    if showPlot :
        pyEDM.AuxFunc.PlotObsPred( df_pred, dataFile, E, Tp )

    MV = { 'Predictions' : df_pred, 'View' : view }
    
    return MV

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def CCM( pathIn       = "./",
         dataFile     = "",
         dataFrame    = None,
         pathOut      = "./",
         predictFile  = "",
         E            = 0, 
         Tp           = 0,
         knn          = 0,
         tau          = 1,
         columns      = "",
         target       = "",
         libSizes     = "",
         sample       = 0,
         random       = True,
         replacement  = False,
         seed         = 0,
         verbose      = False,
         showPlot     = False ):
    '''Convergent Cross Mapping on path/file.'''


    # Establish DF as empty list or Pandas DataFrame for CCM()
    if dataFile :
        DF = pyBindEDM.DF()
    elif isinstance( dataFrame, DataFrame ) :
        if dataFrame.empty :
            raise Exception( "CCM(): dataFrame is empty." )
        DF = pyEDM.AuxFunc.PandasDataFrametoDF( dataFrame )
    else :
        raise Exception( "CCM(): Invalid data input." )
    
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
                       columns,
                       target,
                       libSizes,
                       sample,
                       random,
                       replacement,
                       seed,
                       verbose )

    df = DataFrame( D ) # Convert to pandas DataFrame

    if showPlot :
        title = dataFile + "\nE=" + str(E)
    
        ax = df.plot( 'LibSize', [ df.columns[1], df.columns[2] ],
                      title = title, linewidth = 3 )
        ax.set( xlabel = "Library Size", ylabel = "Correlation ρ" )
        axhline( y = 0, linewidth = 1 )
        show()
    
    return df

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def EmbedDimension( pathIn       = "./",
                    dataFile     = "",
                    dataFrame    = None,
                    pathOut      = "./",
                    predictFile  = "",
                    lib          = "",
                    pred         = "",
                    maxE         = 10,
                    Tp           = 1,
                    tau          = 1,
                    columns      = "",
                    target       = "",
                    embedded     = False,
                    verbose      = False,
                    numThreads   = 4,
                    showPlot     = True ):
 
    '''Estimate optimal embedding dimension [1:maxE].'''

    # Establish DF as empty list or Pandas DataFrame for EmbedDimension()
    if dataFile :
        DF = pyBindEDM.DF()
    elif isinstance( dataFrame, DataFrame ) :
        if dataFrame.empty :
            raise Exception( "EmbedDimension(): dataFrame is empty." )
        DF = pyEDM.AuxFunc.PandasDataFrametoDF( dataFrame )
    else :
        raise Exception( "EmbedDimension(): Invalid data input." )
    
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
                                  columns,
                                  target,
                                  embedded,
                                  verbose,
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
def PredictInterval( pathIn       = "./",
                     dataFile     = "",
                     dataFrame    = None,
                     pathOut      = "./",
                     predictFile  = "",
                     lib          = "",
                     pred         = "",
                     maxTp        = 10,
                     E            = 1,
                     tau          = 1,
                     columns      = "",
                     target       = "",
                     embedded     = False,
                     verbose      = False,
                     numThreads   = 4,
                     showPlot     = True ):
    '''Estimate optimal prediction interval [1:maxTp]'''

    # Establish DF as empty list or Pandas DataFrame for PredictInterval()
    if dataFile :
        DF = pyBindEDM.DF()
    elif isinstance( dataFrame, DataFrame ) :
        if dataFrame.empty :
            raise Exception( "PredictInterval(): dataFrame is empty." )
        DF = pyEDM.AuxFunc.PandasDataFrametoDF( dataFrame )
    else :
        raise Exception( "PredictInterval(): Invalid data input." )
    
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
                                   columns,
                                   target,
                                   embedded,
                                   verbose,
                                   numThreads )

    df = DataFrame( D ) # Convert to pandas DataFrame

    if showPlot :
        title = dataFile + "\nE=" + str( E )
    
        ax = df.plot( 'Tp', 'rho', title = title, linewidth = 3 )
        ax.set( xlabel = "Forecast Interval",
                ylabel = "Prediction Skill ρ" )
        show()
    
    return df

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def PredictNonlinear( pathIn       = "./",
                      dataFile     = "",
                      dataFrame    = None,
                      pathOut      = "./",
                      predictFile  = "",
                      lib          = "",
                      pred         = "",
                      theta        = "",
                      E            = 1,
                      Tp           = 1,
                      tau          = 1,
                      columns      = "",
                      target       = "",
                      embedded     = False,
                      verbose      = False,
                      numThreads   = 4,
                      showPlot     = True ):
    '''Estimate S-map localisation over theta.'''

    # Establish DF as empty list or Pandas DataFrame for PredictNonlinear()
    if dataFile :
        DF = pyBindEDM.DF()
    elif isinstance( dataFrame, DataFrame ) :
        if dataFrame.empty :
            raise Exception( "PredictNonlinear(): dataFrame is empty." )
        DF = pyEDM.AuxFunc.PandasDataFrametoDF( dataFrame )
    else :
        raise Exception( "PredictNonlinear(): Invalid data input." )

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
                                    tau,
                                    columns,
                                    target,
                                    embedded,
                                    verbose,
                                    numThreads )

    df = DataFrame( D ) # Convert to pandas DataFrame

    if showPlot :
        title = dataFile + "\nE=" + str( E )
    
        ax = df.plot( 'Theta', 'rho', title = title, linewidth = 3 )
        ax.set( xlabel = "S-map Localisation (θ)",
                ylabel = "Prediction Skill ρ" )
        show()
    
    return df
