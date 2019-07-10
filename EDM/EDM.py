'''Python interface to Empirical Dynamic Modeling (EDM) C++ library (cppEDM)
   https://github.com/SugiharaLab/cppEDM.
'''

import EDM_pybind
from   pandas import DataFrame
from   matplotlib.pyplot import show, axhline

import pkg_resources # Get data file pathnames from EDM package

#------------------------------------------------------------------------
# 
#------------------------------------------------------------------------
def Examples():
    '''Run examples'''

    dataFiles = [ "TentMap_rEDM.csv",
                  "TentMapNoise_rEDM.csv",
                  "circle.csv",
                  "block_3sp.csv",
                  "sardine_anchovy_sst.csv" ]
    
    # Create map of module dataFiles pathnames in Files
    Files = {}

    for file in dataFiles:
        filename = "data/" + file
        if pkg_resources.resource_exists( __name__, filename ):
            Files[ file ] = \
            pkg_resources.resource_filename( __name__, filename )
        else :
            raise Exception( "Examples(): Failed to find data file " + \
                             file + " in EDM package" )

    print( "-----------------------------------------------------------" )
    print( "EDM Examples: Package data location: " )
    print( pkg_resources.resource_filename( __name__, "data/" ) )
    print( "-----------------------------------------------------------\n" )

    # Note the path argument is empty "", file path is in Files{}
    #---------------------------------------------------------------
    print( '''\nEmbedDimension( "./", TentMap_rEDM.csv", None, "", "",''' +\
           '''"1 100", "201 500", 1, 1, "TentMap", "", False, False, 4 )''' )
    
    df = EmbedDimension( "", Files[ "TentMap_rEDM.csv" ], None, "./", "",
                         "1 100", "201 500", 1, 1,
                         "TentMap", "", False, False, 4 )
    
    #---------------------------------------------------------------
    print( '''\nPredictInterval( "./", "TentMap_rEDM.csv", None, "./", "",''' +\
           '''"1 100", "201 500", 2, 1,"TentMap", "", False, False, 4 )''' )
    
    df = PredictInterval( "", Files[ "TentMap_rEDM.csv" ], None, "./", "",
                          "1 100", "201 500", 2, 1,
                          "TentMap", "", False, False, 4 );

    #---------------------------------------------------------------
    print( '''\nPredictNonlinear( "./", "TentMapNoise_rEDM.csv", None,''' +\
           '''"./", "", "1 100", "201 500", ''' +\
           '''2, 1, 1,"TentMap", "", False, False, 4 )''' )
    
    df = PredictNonlinear( "", Files[ "TentMapNoise_rEDM.csv" ], None, "./", "",
                           "1 100", "201 500", 2, 1, 1,
                           "TentMap", "", False, False, 4 )
    
    #---------------------------------------------------------------
    # Tent map simplex : specify multivariable columns embedded = True
    print( '''\nSimplex( "./", "block_3sp.csv", None, "./", "", ''' +\
           '''"1 99", "100 198", 3, 1, 0, 1, ''' +\
           '''"x_t y_t z_t", "x_t", True, True, True )''' )

    df = Simplex( "", Files[ "block_3sp.csv" ], None, "./", "", 
                  "1 99", "100 198", 3, 1, 0, 1, 0,
                  "x_t y_t z_t", "x_t", True, False, True, True )

    #---------------------------------------------------------------
    # Tent map simplex : Embed column x_t to E=3, embedded = False
    print( '''\nSimplex( "./", "block_3sp.csv", None, "./", "", ''' +\
           '''"1 99", "100 195", 3, 1, 0, 1,''' +\
           '''"x_t", "x_t", False, True, True )''' )

    df = Simplex( "", Files[ "block_3sp.csv" ], None, "./", "", 
                  "1 99", "100 195", 3, 1, 0, 1, 0,
                  "x_t", "x_t", False, False, True, True )

    #---------------------------------------------------------------
    print( '''\nMultiview( "", "block_3sp.csv", None, "./", "", ''' +\
           '''"1 100", "101 198", 3, 1, 0, 1, ''' +\
           '''"x_t y_t z_t", "x_t, 0, False, 4, True )"''' )

    M = Multiview( "", Files[ "block_3sp.csv" ], None, "./", "", 
                   "1 100", "101 198", 3, 1, 0, 1,
                   "x_t y_t z_t", "x_t", 0, False, 4, True )

    #---------------------------------------------------------------
    # S-map circle : specify multivariable columns embedded = True
    print('''\nSMap( "", "circle.csv", None, "./", "", "1 100", "101 198", '''+\
           '''2, 1, 0, 1, 4, "x y", "x", "", "", True, True, True )''' )
           
    S = SMap( "", Files[ "circle.csv" ], None, "./", "", 
              "1 100", "101 198", 2, 1, 0, 1, 4, 0,
              "x y", "x", "", "", True, False, True, True )

    #---------------------------------------------------------------
    print( '''\nCCM( "./", "sardine_anchovy_sst.csv", None, "./", "",''' +\
           ''' 3, 0, 0, 1, "anchovy", "np_sst" ''' +\
           '''"10 80 10", 100, True, 0, True, True )",''' )
           
    df = CCM( "", Files[ "sardine_anchovy_sst.csv" ], None, "./", "", 
              3, 0, 0, 1, "anchovy", "np_sst",
              "10 80 10", 100, True, 0, True, True )
    
#------------------------------------------------------------------------
# 
#------------------------------------------------------------------------
def PlotObsPred( df, dataFile = None, E = None, Tp = None, block = True ):
    '''Plot observations and predictions'''
    
    # stats: {'MAE': 0., 'RMSE': 0., 'rho': 0. }
    stats = EDM_pybind.ComputeError( df['Observations'].tolist(),
                                     df['Predictions' ].tolist() )

    title = dataFile + "\nE=" + str(E) + " Tp=" + str(Tp) +\
            "  ρ="   + str( round( stats['rho'],  2 ) )   +\
            " RMSE=" + str( round( stats['RMSE'], 2 ) )

    if "time" in df.columns :
        time_col = "time"
    elif "Time" in df.columns :
        time_col = "Time"
    else :
        raise RuntimeError( "PlotObsPred() Time column not found." )
    
    df.plot( time_col, ['Observations', 'Predictions'],
             title = title, linewidth = 3 )
    
    show( block = block )
    
#------------------------------------------------------------------------
# 
#------------------------------------------------------------------------
def PlotCoeff( df, dataFile = None, E = None, Tp = None, block = True ):
    '''Plot S-Map coefficients'''
    
    title = dataFile + "\nE=" + str(E) + " Tp=" + str(Tp) +\
            "  S-Map Coefficients"
    
    # Coefficient columns can be in any column
    coef_cols = [ x for x in df.columns if "Time" not in x ]

    df.plot( "Time", coef_cols, title = title, linewidth = 3,
             subplots = True )
    
    show( block = block )
    
#------------------------------------------------------------------------
# pybind C++  DF = list< pair< string, valarray<double> > >
# pybind Py        [ ( string, array ),  ]
#------------------------------------------------------------------------
def PandasDataFrametoDF( df ):
    '''Convert Pandas DataFrame to list of tuples.'''

    if df is None :
        raise RuntimeError( "PandasDataFrametoDF() empty DataFrame" )
    
    #DF = []
    #for column in df.columns :
    #    DF.append( ( column, df.get( column ).tolist() ) )

    DF = EDM_pybind.DF()
    # time and timeName?
    for column in df.columns :
        DF.dataList.append( ( column, df.get( column ).tolist() ) )
    
    
    return DF
             
#------------------------------------------------------------------------
# 
#------------------------------------------------------------------------
def ComputeError( obs, pred ):
    '''Pearson rho, RMSE, MAE.'''

    D = EDM_pybind.ComputeError( obs, pred )

    return D
             
#------------------------------------------------------------------------
# 
#------------------------------------------------------------------------
def ReadDataFrame( path, file ):
    '''Read path/file into DataFrame.'''

    D = EDM_pybind.ReadDataFrame( path, file )
    
    df = DataFrame( D ) # Convert to pandas DataFrame

    return df
             
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
    
    DF = PandasDataFrametoDF( dataFrame )
    
    # D is a Python dict from pybind11 < cppEDM Embed
    D = EDM_pybind.MakeBlock( DF,
                              E, 
                              tau,
                              columnNames,
                              verbose )

    df = DataFrame( D ) # Convert to pandas DataFrame

    return df

#------------------------------------------------------------------------
# 
#------------------------------------------------------------------------
def Embed( path      = "./",
           dataFile  = None,
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
        DF = EDM_pybind.DF() 
    elif isinstance( dataFrame, DataFrame ) :
        if dataFrame.empty :
            raise Exception( "Embed(): dataFrame is empty." )
        DF = PandasDataFrametoDF( dataFrame )
    else :
        raise Exception( "Embed(): Invalid data input." )
    
    # D is a Python dict from pybind11 < cppEDM Embed
    D = EDM_pybind.Embed( path,
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
             dataFile     = None,
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
        DF = EDM_pybind.DF()
    elif isinstance( dataFrame, DataFrame ) :
        if dataFrame.empty :
            raise Exception( "Simplex(): dataFrame is empty." )
        DF = PandasDataFrametoDF( dataFrame )
    else :
        raise Exception( "Simplex(): Invalid data input." )
    
    # D is a Python dict from pybind11 < cppEDM Simplex 
    D = EDM_pybind.Simplex( pathIn,
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
        PlotObsPred( df, dataFile, E, Tp )
    
    return df

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def SMap( pathIn       = "./",
          dataFile     = None,
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
        DF = EDM_pybind.DF() 
    elif isinstance( dataFrame, DataFrame ) :
        if dataFrame.empty :
            raise Exception( "SMap(): dataFrame is empty." )
        DF = PandasDataFrametoDF( dataFrame )
    else :
        raise Exception( "SMap(): Invalid data input." )
    
    # D is a Python dict from pybind11 < cppEDM SMap:
    #  { "predictions" : {}, "coefficients" : {} }
    D = EDM_pybind.SMap( pathIn,
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

    # JP --------------------------------------------------------
    # SMap coef df.time is not the right size?
    # print( "Pred: ", df_pred.shape, "\n", df_pred.head(5) )
    # print( "Coef: ", df_coef.shape, "\n", df_coef.head(5) )
    # JP --------------------------------------------------------
    
    if showPlot :
        PlotObsPred( df_pred, dataFile, E, Tp, False )
        PlotCoeff  ( df_coef, dataFile, E, Tp )

    SMapDict = { 'predictions' : df_pred, 'coefficients' : df_coef }
    
    return SMapDict

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def Multiview( pathIn       = "./",
               dataFile     = None,
               dataFrame    = None,
               pathOut      = "./",
               predictFile  = "",
               lib          = "",
               pred         = "",
               E            = 0, 
               Tp           = 1,
               knn          = 0,
               tau          = 1,
               columns      = "",
               target       = "",
               multiview    = 0,
               verbose      = False,
               numThreads   = 4,
               showPlot     = False ):
    '''Multiview prediction on path/file.'''

    # Establish DF as empty list or Pandas DataFrame for Multiview()
    if dataFile :
        DF = EDM_pybind.DF() 
    elif isinstance( dataFrame, DataFrame ) :
        if dataFrame.empty :
            raise Exception( "Multiview(): dataFrame is empty." )
        DF = PandasDataFrametoDF( dataFrame )
    else :
        raise Exception( "Multiview(): Invalid data input." )
    
    # D is a Python dict from pybind11 < cppEDM Multiview:
    #  { "Combo_rho" : {}, "Predictions" : {} }
    D = EDM_pybind.Multiview( pathIn,
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
                              verbose,
                              numThreads )
    
    df_pred = DataFrame( D['Predictions'] ) # Convert to pandas DataFrame
    df_rho  = DataFrame( D['Combo_rho']   ) # Convert to pandas DataFrame

    if showPlot :
        PlotObsPred( df_pred, dataFile, E, Tp )

    MV = { 'Predictions' : df_pred, 'Combo_rho' : df_rho }
    
    return MV

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def CCM( pathIn       = "./",
         dataFile     = None,
         dataFrame    = None,
         pathOut      = "./",
         predictFile  = "",
         E            = 0, 
         Tp           = 1,
         knn          = 0,
         tau          = 1,
         columns      = "",
         target       = "",
         libSizes     = "",
         sample       = 0,
         random       = True,
         seed         = 0,
         verbose      = False,
         showPlot     = False ):
    '''Convergent Cross Mapping on path/file.'''


    # Establish DF as empty list or Pandas DataFrame for CCM()
    if dataFile :
        DF = EDM_pybind.DF()
    elif isinstance( dataFrame, DataFrame ) :
        if dataFrame.empty :
            raise Exception( "CCM(): dataFrame is empty." )
        DF = PandasDataFrametoDF( dataFrame )
    else :
        raise Exception( "CCM(): Invalid data input." )
    
    # D is a Python dict from pybind11 < cppEDM CCM
    D = EDM_pybind.CCM( pathIn,
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
                    dataFile     = None,
                    dataFrame    = None,
                    pathOut      = "./",
                    predictFile  = "",
                    lib          = "",
                    pred         = "",
                    Tp           = 1,
                    tau          = 1,
                    columns      = "",
                    target       = "",
                    embedded     = False,
                    verbose      = False,
                    numThreads   = 4,
                    showPlot     = True ):
    '''Estimate optimal embedding dimension [1,10] on path/file.'''

    # Establish DF as empty list or Pandas DataFrame for EmbedDimension()
    if dataFile :
        DF = EDM_pybind.DF()
    elif isinstance( dataFrame, DataFrame ) :
        if dataFrame.empty :
            raise Exception( "EmbedDimension(): dataFrame is empty." )
        DF = PandasDataFrametoDF( dataFrame )
    else :
        raise Exception( "EmbedDimension(): Invalid data input." )
    
    # D is a Python dict from pybind11 < cppEDM CCM
    D = EDM_pybind.EmbedDimension( pathIn,
                                   dataFile,
                                   DF,
                                   pathOut,
                                   predictFile,
                                   lib,
                                   pred, 
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
                     dataFile     = None,
                     dataFrame    = None,
                     pathOut      = "./",
                     predictFile  = "",
                     lib          = "",
                     pred         = "",
                     E            = 1,
                     tau          = 1,
                     columns      = "",
                     target       = "",
                     embedded     = False,
                     verbose      = False,
                     numThreads   = 4,
                     showPlot     = True ):
    '''Estimate optimal prediction interval [1,10] on path/file.'''

    # Establish DF as empty list or Pandas DataFrame for PredictInterval()
    if dataFile :
        DF = EDM_pybind.DF()
    elif isinstance( dataFrame, DataFrame ) :
        if dataFrame.empty :
            raise Exception( "PredictInterval(): dataFrame is empty." )
        DF = PandasDataFrametoDF( dataFrame )
    else :
        raise Exception( "PredictInterval(): Invalid data input." )
    
    # D is a Python dict from pybind11 < cppEDM PredictInterval
    D = EDM_pybind.PredictInterval( pathIn,
                                    dataFile,
                                    DF,
                                    pathOut,
                                    predictFile,
                                    lib,
                                    pred, 
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
                      dataFile     = None,
                      dataFrame    = None,
                      pathOut      = "./",
                      predictFile  = "",
                      lib          = "",
                      pred         = "",
                      E            = 0,
                      Tp           = 1,
                      tau          = 1,
                      columns      = "",
                      target       = "",
                      embedded     = False,
                      verbose      = False,
                      numThreads   = 4,
                      showPlot     = True ):
    '''Estimate S-map localisation on theta in [0.01,9] on path/file.'''

    # Establish DF as empty list or Pandas DataFrame for PredictNonlinear()
    if dataFile :
        DF = EDM_pybind.DF()
    elif isinstance( dataFrame, DataFrame ) :
        if dataFrame.empty :
            raise Exception( "PredictNonlinear(): dataFrame is empty." )
        DF = PandasDataFrametoDF( dataFrame )
    else :
        raise Exception( "PredictNonlinear(): Invalid data input." )
    
    # D is a Python dict from pybind11 < cppEDM PredictNonlinear
    D = EDM_pybind.PredictNonlinear( pathIn,
                                     dataFile,
                                     DF,
                                     pathOut,
                                     predictFile,
                                     lib,
                                     pred, 
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
