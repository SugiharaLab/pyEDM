'''Examples, plot functions, IsIterable, ComputeError, SurrogateData'''

# python modules
from math   import floor, pi, sqrt, cos
from cmath  import exp
from random import sample, uniform, normalvariate

# package modules
from numpy             import any, arange, corrcoef, fft, isfinite, mean
from numpy             import max, nan, ptp, std, sqrt, zeros
from pandas            import DataFrame, read_csv
from scipy.interpolate import UnivariateSpline
from matplotlib.pyplot import show, axhline

import pyEDM.API as EDM
from .LoadData import sampleData

#------------------------------------------------------------------------
#------------------------------------------------------------------------
def Examples():

    def RunEDM ( cmd ):
        print(cmd)
        print()
        df = eval( 'EDM.' + cmd )
        return df

    sampleDataNames = \
        ["TentMap","TentMapNoise","circle","block_3sp","sardine_anchovy_sst"]

    for dataName in sampleDataNames :
        if dataName not in sampleData:
            raise Exception( "Examples(): Failed to find sample data " + \
                             dataName + " in EDM package" )

    #---------------------------------------------------------------
    cmd = str().join(['EmbedDimension( dataFrame = sampleData["TentMap"],',
                      ' columns = "TentMap", target = "TentMap",',
                      ' lib = [1, 100], pred = [201, 500] )'])
    RunEDM( cmd )

    #---------------------------------------------------------------
    cmd = str().join(['PredictInterval( dataFrame = sampleData["TentMap"],',
                      ' columns = "TentMap", target = "TentMap",'
                      ' lib = [1, 100], pred = [201, 500], E = 2 )'])
    RunEDM( cmd )

    #---------------------------------------------------------------
    cmd = str().join(
        ['PredictNonlinear( dataFrame = sampleData["TentMapNoise"],',
         ' columns = "TentMap", target = "TentMap", '
         ' lib = [1, 100], pred = [201, 500], E = 2 )'])
    RunEDM( cmd )

    #---------------------------------------------------------------
    # Tent map simplex : specify multivariable columns embedded = True
    cmd = str().join(['Simplex( dataFrame = sampleData["block_3sp"],',
                      ' columns="x_t y_t z_t", target="x_t",'
                      ' lib = [1, 99], pred = [100, 195],',
                      ' E = 3, embedded = True, showPlot = True )'])
    RunEDM( cmd )

    #---------------------------------------------------------------
    # Tent map simplex : Embed column x_t to E=3, embedded = False
    cmd = str().join(['Simplex( dataFrame = sampleData["block_3sp"],',
                      ' columns = "x_t", target = "x_t",',
                      ' lib = [1, 99], pred = [105, 190],',
                      ' E = 3, showPlot = True )'])
    RunEDM( cmd )

    #---------------------------------------------------------------
    cmd = str().join(['Multiview( dataFrame = sampleData["block_3sp"],',
                      ' columns = "x_t y_t z_t", target = "x_t",',
                      ' lib = [1, 100], pred = [101, 198],',
                      ' D = 0, E = 3, Tp = 1, multiview = 0,',
                      ' trainLib = False, showPlot = True ) '])
    RunEDM( cmd )

    #---------------------------------------------------------------
    # S-map circle : specify multivariable columns embedded = True
    cmd = str().join(['SMap( dataFrame = sampleData["circle"],',
                      ' columns = ["x", "y"], target = "x",'
                      ' lib = [1, 100], pred = [110, 190], theta = 4, E = 2,',
                      ' verbose = False, showPlot = True, embedded = True )'])
    RunEDM( cmd )

    #---------------------------------------------------------------
    cmd = str().join(['CCM( dataFrame = sampleData["sardine_anchovy_sst"],',
                      ' columns = "anchovy", target = "np_sst",',
                      ' libSizes = [10, 70, 10], sample = 50,',
                      ' E = 3, Tp = 0, verbose = False, showPlot = True )'])
    RunEDM( cmd )

#------------------------------------------------------------------------
#------------------------------------------------------------------------
def PlotObsPred( df, dataName = "", E = 0, Tp = 0, block = True ):
    '''Plot observations and predictions'''

    # stats: {'MAE': 0., 'RMSE': 0., 'rho': 0. }
    stats = ComputeError( df['Observations'], df['Predictions' ] )

    title = dataName + "\nE=" + str(E) + " Tp=" + str(Tp) +\
            "  ρ="   + str( round( stats['rho'],  3 ) )   +\
            " RMSE=" + str( round( stats['RMSE'], 3 ) )

    time_col = df.columns[0]

    df.plot( time_col, ['Observations', 'Predictions'],
             title = title, linewidth = 3 )

    show( block = block )

#------------------------------------------------------------------------
#------------------------------------------------------------------------
def PlotCoeff( df, dataName = "", E = 0, Tp = 0, block = True ):
    '''Plot S-Map coefficients'''

    title = dataName + "\nE=" + str(E) + " Tp=" + str(Tp) +\
            "  S-Map Coefficients"

    time_col = df.columns[0]

    # Coefficient columns can be in any column
    coef_cols = [ x for x in df.columns if time_col not in x ]

    df.plot( time_col, coef_cols, title = title, linewidth = 3,
             subplots = True )

    show( block = block )

#------------------------------------------------------------------------
#------------------------------------------------------------------------
def ComputeError( obs, pred, digits = 6 ):
    '''Pearson rho, RMSE, MAE
       Remove nan from obs, pred for corrcoeff.
    '''

    notNan = isfinite( pred )
    if any( ~notNan ) :
        pred = pred[ notNan ]
        obs  = obs [ notNan ]

    notNan = isfinite( obs )
    if any( ~notNan ) :
        pred = pred[ notNan ]
        obs  = obs [ notNan ]

    if len( pred ) < 5 :
        msg = f'ComputeError(): Not enough data ({len(pred)}) to ' +\
               ' compute error statistics.'
        print( msg )
        return { 'rho' : nan, 'MAE' : nan, 'RMSE' : nan }

    rho  = round( corrcoef( obs, pred )[0,1], digits )
    err  = obs - pred
    MAE  = round( max( err ), digits )
    RMSE = round( sqrt( mean( err**2 ) ), digits )

    D = { 'rho' : rho, 'MAE' : MAE, 'RMSE' : RMSE }

    return D

#------------------------------------------------------------------------
#------------------------------------------------------------------------
def Iterable( obj ):
    '''Is an object iterable?'''

    try:
        it = iter( obj )
    except TypeError: 
        return False
    return True

#------------------------------------------------------------------------
#------------------------------------------------------------------------
def IsIterable( obj ):
    '''Is an object iterable and not a string?'''

    if Iterable( obj ) :
        if isinstance( obj, str ) :
            return False
        else :
            return True
    return False

#------------------------------------------------------------------------
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
        for s in range( numSurrogates ) : # use pandas sample
            surr = dataFrame[ column ].sample(
                n = dataFrame.shape[0] ).to_numpy()
            df[ s + 1 ] = surr

    elif method.lower() == "ebisuzaki" :
        data          = dataFrame[ column ].to_numpy()
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
        y = dataFrame[ column ].to_numpy()
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
