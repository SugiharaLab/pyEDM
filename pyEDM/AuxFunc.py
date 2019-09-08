'''Examples and graphical functions pyEDM interface to cppEDM 
   github.com/SugiharaLab/cppEDM.'''

import pkg_resources

from pandas            import DataFrame
from matplotlib.pyplot import show, axhline

import pyBindEDM
import pyEDM.CoreEDM
from   pyEDM.LoadData import sampleData

#------------------------------------------------------------------------
# 
#------------------------------------------------------------------------
def Examples():

    def RunEDM ( cmd ):
        print(cmd)
        print()
        df = eval( 'pyEDM.' + cmd )
        return df

    sampleDataNames = \
        ["TentMap","TentMapNoise","circle","block_3sp","sardine_anchovy_sst"]

    for dataName in sampleDataNames :
        if dataName not in sampleData:
            raise Exception( "Examples(): Failed to find sample data " + \
                             dataName + " in EDM package" )

    #---------------------------------------------------------------
    cmd = str().join(['EmbedDimension( dataFrame = sampleData["TentMap"],',
                       ' lib = "1 100", pred = "201 500",',
                       ' columns = "TentMap", target = "TentMap") '])
    RunEDM( cmd )
    
    #---------------------------------------------------------------
    cmd = str().join(['PredictInterval( dataFrame = sampleData["TentMap"],',
                       ' lib = "1 100", pred = "201 500", E = 2,',
                       ' columns = "TentMap", target = "TentMap") '])
    RunEDM( cmd )
    
    #---------------------------------------------------------------
    cmd = str().join(
        ['PredictNonlinear( dataFrame = sampleData["TentMapNoise"],',
         ' lib = "1 100", pred = "201 500", E = 2,',
         ' columns = "TentMap", target = "TentMap" ) '])
    RunEDM( cmd )
    
    #---------------------------------------------------------------
    # Tent map simplex : specify multivariable columns embedded = True
    cmd = str().join(['Simplex( dataFrame = sampleData["block_3sp"],',
                      ' lib = "1 99", pred = "100 195", ',
                      ' E = 3, embedded = True, showPlot = True,',
                      ' const_pred = True,',
                      ' columns="x_t y_t z_t", target="x_t") '])
    RunEDM( cmd )
    
    #---------------------------------------------------------------
    # Tent map simplex : Embed column x_t to E=3, embedded = False
    cmd = str().join(['Simplex( dataFrame = sampleData["block_3sp"],',
                      ' lib = "1 99", pred = "105 190", ',
                      ' E = 3, showPlot = True, const_pred = True,',
                      ' columns = "x_t", target = "x_t") '])
    RunEDM( cmd )
    
    #---------------------------------------------------------------
    cmd = str().join(['Multiview( dataFrame = sampleData["block_3sp"],',
                      ' lib = "1 99", pred = "105 190", ',
                      ' E = 3, columns = "x_t y_t z_t", target = "x_t",',
                      ' showPlot = True) '])
    RunEDM( cmd )
    
    #---------------------------------------------------------------
    # S-map circle : specify multivariable columns embedded = True
    cmd = str().join(['SMap( dataFrame = sampleData["circle"],',
                      ' lib = "1 100", pred = "110 190", theta = 4, E = 2,',
                      ' verbose = True, showPlot = True, embedded = True,',
                      ' columns = "x y", target = "x") '])
    RunEDM( cmd )
    
    #---------------------------------------------------------------
    cmd = str().join(['CCM( dataFrame = sampleData["sardine_anchovy_sst"],',
                      ' E = 3, Tp = 0, columns = "anchovy", target = "np_sst",',
                      ' libSizes = "10 80 10", sample = 100, verbose = True, ',
                      ' showPlot = True) '])
    RunEDM( cmd )

#------------------------------------------------------------------------
# 
#------------------------------------------------------------------------
def PlotObsPred( df, dataFile = None, E = None, Tp = None, block = True ):
    '''Plot observations and predictions'''
    
    # stats: {'MAE': 0., 'RMSE': 0., 'rho': 0. }
    stats = pyBindEDM.ComputeError( df['Observations'].tolist(),
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
    
    # Need to create lists separately since pybind list append not working ??
    timeName = 'Time'
    time     = []
    dataList = []

    if df.columns[0] is timeName : 
        time = df[ timeName ].tolist()
    else : 
        # Create time column if doesn't exist
        time = [ str(i) for i in list( range( df.shape[0] ) ) ]

    # Add time series data
    for column in df.columns :
        dataList.append( ( column, df.get( column ).tolist() ) )
    
    # cppEDM DF struct
    DF          = pyBindEDM.DF()
    DF.timeName = timeName
    DF.time     = time
    DF.dataList = dataList
    
    return DF
             
#------------------------------------------------------------------------
# 
#------------------------------------------------------------------------
def ComputeError( obs, pred ):
    '''Pearson rho, RMSE, MAE.'''

    D = pyBindEDM.ComputeError( obs, pred )

    return D
             
#------------------------------------------------------------------------
# 
#------------------------------------------------------------------------
def ReadDataFrame( path, file ):
    '''Read path/file into DataFrame.'''

    D = pyBindEDM.ReadDataFrame( path, file )
    
    df = DataFrame( D ) # Convert to pandas DataFrame

    return df


