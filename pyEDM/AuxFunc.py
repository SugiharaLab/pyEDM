'''Examples and graphical functions pyEDM interface to cppEDM 
   github.com/SugiharaLab/cppEDM.'''

import pkg_resources

from math   import floor, pi, sqrt, cos
from cmath  import exp
from random import sample, uniform, normalvariate

from numpy             import mean, std, fft, arange, zeros, ptp
from scipy.interpolate import UnivariateSpline
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
                      ' libSizes = "10 70 10", sample = 100, verbose = True, ',
                      ' showPlot = True) '])
    RunEDM( cmd )

#------------------------------------------------------------------------
# 
#------------------------------------------------------------------------
def PlotObsPred( df, dataFile = None, E = None, Tp = None, block = True ):
    '''Plot observations and predictions'''
    
    # stats: {'MAE': 0., 'RMSE': 0., 'rho': 0. }
    stats = pyBindEDM.ComputeError( df['Observations'],
                                    df['Predictions' ] )

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
    
    if "time" in df.columns :
        time_col = "time"
    elif "Time" in df.columns :
        time_col = "Time"
    else :
        raise RuntimeError( "PlotCoeff() Time column not found." )
    
    # Coefficient columns can be in any column
    coef_cols = [ x for x in df.columns if time_col not in x ]

    df.plot( time_col, coef_cols, title = title, linewidth = 3,
             subplots = True )
    
    show( block = block )
    
#------------------------------------------------------------------------
# pybind C++  DF = struct { string timeName, vector<string> time, DataList }
#             DataList = list< pair< string, valarray > >
#
# pybind Py   [ ( string, array ), ..., ]
#------------------------------------------------------------------------
def PandasDataFrametoDF( df ):
    '''Convert Pandas DataFrame to DF struct.'''

    if df is None :
        raise RuntimeError( "PandasDataFrametoDF() empty DataFrame" )

    # Here is a fundamental problem/incompatability between cppEDM and pyEDM
    # cppEDM DataFrame stores time vector as strings, and data as valarray
    # time values are not strictly required DataFrame( noTime = true )
    # Here, we don't have a way to know if the Pandas dataframe passed in
    # will have a time vector or not... So... We will just require/assume
    # that the first column is ALWAYS a time or index vector.
    # Validate that at least 2 columns are provided
    if df.shape[1] < 2:
        raise RuntimeError( "PandasDataFrametoDF() DataFrame must have"
                            " at least 2 columns. First column is time." )
    timeName = df.columns[0]
    timeVec  = df.get( timeName )
    time     = [ str( x ) for x in timeVec ] # convert to list of strings

    # Also require data homogeneity : all numeric, no mixed-data
    # but allow a Time first column that is an object...
    # in .dtypes: non-numerics are converted to dtype "object"
    if any( df.dtypes[1:] == "object" ) :
        print( df.dtypes )
        raise RuntimeError( "PandasDataFrametoDF() non-numeric data is not"
                            " allowed in a DataFrame." )
    
    dataList = []

    # Add time series data, Skipping the first column!!!
    for column in df.columns[1:] :
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
def ReadDataFrame( path, file, noTime = False ):
    '''Read path/file into DataFrame.'''

    D = pyBindEDM.ReadDataFrame( path, file, noTime )
    
    df = DataFrame( D ) # Convert to pandas DataFrame

    return df

#------------------------------------------------------------------------
# Is an object iterable?
#------------------------------------------------------------------------
def Iterable( obj ):
    try:
        it = iter( obj )
    except TypeError: 
        return False
    return True

#------------------------------------------------------------------------
# Is an object iterable and not a string?
#------------------------------------------------------------------------
def NotStringIterable( obj ):
    if Iterable( obj ) :
        if isinstance( obj, str ) :
            return False
        else :
            return True
    return False

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def SurrogateData( dataFrame     = None,
                   column        = None,
                   method        = 'ebisuzaki',
                   numSurrogates = 10,
                   alpha         = None,
                   smooth        = 0.8,
                   outputFile    = None ):
    '''Three methods:

    random_shuffle :
      Sample the data with a uniform distribution.

    ebisuzaki :
      Journal of Climate. A Method to Estimate the Statistical Significance
      of a Correlation When the Data Are Serially Correlated.
      https://doi.org/10.1175/1520-0442(1997)010<2147:AMTETS>2.0.CO;2

      Presumes data are serially correlated with low pass coherence. It is:
      "resampling in the frequency domain. This procedure will not preserve
      the distribution of values but rather the power spectrum (periodogram).
      The advantage of preserving the power spectrum is that resampled series
      retains the same autocorrelation as the original series."

    seasonal :
      Presume a smoothing spline represents the seasonal trend.
      Each surrogate is a summation of the trend, resampled residuals,
      and possibly additive Gaussian noise. Default noise has a standard
      deviation that is the data range / 5.
    '''

    if dataFrame is None :
        raise RuntimeError( "SurrogateData() empty DataFrame." )

    if column is None :
        raise RuntimeError( "SurrogateData() must specify column." )

    # New dataFrame with initial time column
    df = DataFrame( dataFrame.iloc[ :,0 ] )

    if method.lower() == "random_shuffle" :
        for s in range( numSurrogates ) : # use pandas sample here...
            surr = dataFrame[ column ].sample( n = dataFrame.shape[0] ).values
            df[ s + 1 ] = surr

    elif method.lower() == "ebisuzaki" :
        data          = dataFrame[ column ].values
        n             = dataFrame.shape[0]
        n2            = floor( n/2 )
        mu            = mean   ( data )
        sigma         = std    ( data )
        a             = fft.fft( data )
        amplitudes    = abs( a )
        amplitudes[0] = 0

        for s in range( numSurrogates ) :
            thetas      = [ 2 * pi * uniform( 0, 1 ) for x in range( n2 - 1 )]
            revThetas   = thetas[::-1]
            negThetas   = [ -x for x in revThetas ]
            angles      = [0] + thetas + [0] + negThetas
            surrogate_z = [ A * exp( complex( 0, theta ) )
                            for A, theta in zip( amplitudes, angles ) ]

            if n % 2 == 0 : # even length
                surrogate_z[-1] = complex( sqrt(2) * amplitudes[-1] *
                                           cos( 2 * pi * uniform(0,1) ) )

            ifft = fft.ifft( surrogate_z ) / n

            realifft = [ x.real for x in ifft ]
            sdevifft = std( realifft )

            # adjust variance of surrogate time series to match original
            scaled = [ sigma * x / sdevifft for x in realifft ]

            df[ s + 1 ] = scaled

    elif method.lower() == "seasonal" :
        y = dataFrame[ column ].values
        n = dataFrame.shape[0]

        # Presume a spline captures the seasonal cycle
        x      = arange( n )
        spline = UnivariateSpline( x, y )
        spline.set_smoothing_factor( smooth )
        y_spline = spline( x )

        # Residuals of the smoothing
        residual = list( y - y_spline )

        # spline plus shuffled residuals plus Gaussian noise
        noise = zeros( n )

        # If no noise specified, set std dev to data range / 5
        if alpha is None :
            alpha = ptp( y ) / 5

        for s in range( numSurrogates ) :
            noise = [ normalvariate( 0, alpha ) for z in range( n ) ]

            df[ s + 1 ] = y_spline + sample( residual, n ) + noise

    else :
        raise RuntimeError( "SurrogateData() invalid method." )

    df = df.round( 8 ) # Should be a parameter

    # Rename columns
    columnNames = [ column + "_" + str( c + 1 )
                    for c in range( numSurrogates ) ]

    columnNames.insert( 0, df.columns[0] ) # insert time column name

    df.columns = columnNames

    if outputFile :
        df.to_csv( outputFile, index = False )

    return df
