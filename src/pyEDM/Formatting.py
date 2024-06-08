# python modules
import datetime as dt

# package modules
from pandas import DataFrame
from numpy  import append, array, floating, full, integer, nan

#--------------------------------------------------------------------
# EDM Methods
#-------------------------------------------------------------------
def FormatProjection( self ) :
#-------------------------------------------------------------------
    '''Create Projection, Coefficients, SingularValues DataFrames
       FillTimes() attempts to extend forecast time
    '''
    if self.verbose:
        print( f'{self.name}: FormatProjection()' )

    N_dim        = self.E + 1
    N_pred       = len( self.pred_i_all )
    Tp_magnitude = abs( self.Tp )
    outSize      = N_pred + Tp_magnitude

    # Get timeOut vector with additional Tp points from FillTimes()
    if self.Tp != 0 :
        timeOut = self.FillTimes( Tp_magnitude, outSize )
    else :
        timeOut = self.allTime.to_numpy()[ self.pred_i_all ]

    #----------------------------------------------------
    # Observations: Insert target data in observations
    #----------------------------------------------------
    observations      = full( outSize, nan )
    startObservations = 0

    if self.Tp > -1 : # Positive Tp
        startTarget = self.pred_i_all[0]
    else:
        startTarget = self.pred_i_all[0] - Tp_magnitude

    if startTarget < 0 :
        startObservations = abs( startTarget )
        startTarget = 0

    t = startTarget
    N_targetVec = len( self.targetVec )
    for i in range( startObservations, outSize ) :
        if t < N_targetVec :
            observations[ i ] = self.targetVec[ t, 0 ]
        else :
            break
        t = t + 1

    #----------------------------------------------------
    # Projections & variance
    #----------------------------------------------------
    # ndarray init to nan
    projectionOut = full( outSize, nan )
    varianceOut   = full( outSize, nan )
    
    if self.Tp > -1 : # Positive Tp
        predOut_i = self.pred_i - self.pred_i[0] + self.Tp
        
    else :            # Negative Tp
        predOut_i = self.pred_i - self.pred_i[0]

    # fill projectionOut, varianceOut with projected values
    projectionOut[ predOut_i ] = self.projection_
    varianceOut  [ predOut_i ] = self.variance

    self.Projection = DataFrame(
        { 'Time'          : timeOut,
          'Observations'  : observations,
          'Predictions'   : projectionOut,
          'Pred_Variance' : varianceOut } )

    #----------------------------------------------------
    # SMap coefficients and singular values
    #----------------------------------------------------
    if self.name == 'SMap' :
        # ndarray init to nan
        coefOut = full( (outSize, N_dim), nan )
        SVOut   = full( (outSize, N_dim), nan )
        # fill coefOut, SVOut with projected values
        coefOut[ predOut_i, : ] = self.coefficients
        SVOut  [ predOut_i, : ] = self.singularValues

        timeDF = DataFrame( { 'Time' : timeOut } )

        colNames  = [ 'C0' ]
        coefNames = [f'∂{self.target[0]}/∂{e}' for e in self.Embedding.columns]
        for c in coefNames :
            colNames.append( c )
        coefDF = DataFrame( coefOut, columns = colNames )

        colNames = [ f'C{i}' for i in range( N_dim ) ]
        SVDF = DataFrame( SVOut, columns = colNames )

        self.Coefficients   = timeDF.join( coefDF )
        self.SingularValues = timeDF.join( SVDF )

#-------------------------------------------------------------------
def FillTimes( self, Tp_magnitude, outSize ) :
#-------------------------------------------------------------------
    '''Provide some utility for parsing time, date or datetime in
    data.allTime extending the range for output at Tp != 0
    ISO 8601 formats are supported in the time & datetime modules.

    return timeOut vector with additional Tp points
    '''
    if self.verbose:
        print( f'{self.name}: FillTimes()' )

    # Local copy of allTime
    times = self.allTime.to_numpy().copy() # ndarray from Series
    time0 = times[0]

    # If times are strings, try to parse into time or datetime
    # If success, replace times with parsed time or datetime list
    if isinstance( time0, str ) :
        # Assume date or datetime to be converted from string
        try :
            t0 = dt.time.fromisoformat( time0 )
            # parsed t0 into dt.time. Parse the whole vector
            times = array( [ dt.time.fromisoformat( t ) for t in times ] )
        except ValueError :
            try:
                t0 = dt.datetime.fromisoformat( time0 )
                # parsed t0 into dt.datetime. Parse the whole vector
                times = array([dt.datetime.fromisoformat(t) for t in times])
            except ValueError :
                msg = f'{self.name} FillTimes(): Time values are strings '+\
                      'but are not ISO 8601 recognized time or datetime.'
                raise RuntimeError( msg )

    # if times were string they have been converted to time or datetime
    time0  = times[0] # reset time0
    deltaT = None

    # Ensure times can be numerically manipulated, compute deltaT
    if isinstance( time0, int )     or isinstance( time0, integer )  or \
       isinstance( time0, float )   or isinstance( time0, floating ) or \
       isinstance( time0, dt.time ) or isinstance( time0, dt.datetime ) :

        deltaT = times[1] - time0

    if deltaT is None :
        msg = f'{self.name} FillTimes(): Time values not recognized.' +\
            ' Accepted values are int, float, or string of ISO 8601 ' +\
            ' compliant time or datetime.'
        raise RuntimeError( msg )

    # Now we have times that can be numerically manipulated...
    min_pred_i = self.pred_i_all[0]
    max_pred_i = self.pred_i_all[-1]
    timeOut    = None

    if self.Tp > 0 :
        if max_pred_i + self.Tp < len( times ) :
            # All times are present in allTime, copy them
            timeOut = times[ min_pred_i : (min_pred_i + outSize) ]
        else :
            # Tp introduces time values beyond the range of allTime
            # First, fill timeOut with times in allTime
            timeOut = times[ min_pred_i : (max_pred_i + 1) ]

            # Generate future times
            newTimes = [ times[ max_pred_i ] ] * (self.Tp)
            for i in range( self.Tp ) :
                newTimes[ i ] = newTimes[ i ] + (i+1) * deltaT

            timeOut = append( timeOut, newTimes )

    else : # Tp < 0
        if min_pred_i + self.Tp >= 0 :
            # All times are present in allTime, copy them
            timeOut = times[ min_pred_i + self.Tp :
                             (min_pred_i + self.Tp + outSize ) ]
        else :
            # Tp introduces time values before the range of allTime
            # First, fill timeOut with times in allTime
            timeOut = times[ min_pred_i : (max_pred_i + 1) ]

            # Generate past times
            newTimes = [ times[ min_pred_i ] ] * Tp_magnitude
            for i in range( Tp_magnitude - 1, -1, -1 ) :
                newTimes[ i ] = timeOut[ 0 ] - (i+1) * deltaT

            newTimes.reverse()
            timeOut = append( newTimes, timeOut )

    if timeOut is None :
        msg = f'{self.name} FillTimes(): Failed to pre/append new times.' +\
            ' Accepted values are int, float, or string of ISO 8601 ' +\
            ' compliant time or datetime. Time vector of indices provided.'
        print( msg, flush = True )

        timeOut = array( [i for i in range( 1, outSize + 1 )], dtype = int )

    return timeOut
