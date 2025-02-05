# python modules
import datetime as dt
from warnings import warn

# package modules
from pandas import DataFrame
from numpy  import append, array, empty, floating, full, integer, nan

#--------------------------------------------------------------------
# EDM Methods
#-------------------------------------------------------------------
def FormatProjection( self ) :
#-------------------------------------------------------------------
    '''Create Projection, Coefficients, SingularValues DataFrames
       AddTime() attempts to extend forecast time if needed

       NOTE: self.pred_i had all nan removed for KDTree by RemoveNan().
             self.predList only had leading/trailing embedding nan removed.
             Here we want to include any nan observation rows so we
             process predList & pred_i_all, not self.pred_i.
    '''
    if self.verbose:
        print( f'{self.name}: FormatProjection()' )

    if len( self.pred_i ) == 0 :
        msg = f'{self.name}: FormatProjection() No valid prediction indices.'
        warn( msg )

        self.Projection = DataFrame({ 'Time': [],        'Observations':  [],
                                      'Predictions': [], 'Pred_Variance': [] })
        return

    N_dim        = self.E + 1
    Tp_magnitude = abs( self.Tp )

    #----------------------------------------------------
    # Observations: Insert target data in observations
    #----------------------------------------------------
    outSize = 0

    # Create array of indices into self.targetVec for observations
    obs_i = array( [], dtype = int )

    # Process each pred segment in self.predList
    for pred_i in self.predList :
        N_pred    = len( pred_i )
        outSize_i = N_pred + Tp_magnitude
        outSize   = outSize + outSize_i
        append_i  = array( [], dtype = int )

        if N_pred == 0 :
             # No prediction made for this pred segment
            if self.verbose :
                msg = f'{self.name} FormatProjection(): No prediction made ' +\
                      f'for empty pred in {self.predList}. ' +\
                      'Examine pred, E, tau, Tp parameters and/or nan.'
                print( msg )
            continue

        if self.Tp == 0 : # Tp = 0
            append_i = pred_i.copy()

        elif self.Tp > 0 : # Positive Tp
            if pred_i[-1] + self.Tp < self.targetVec.shape[0] :
                # targetVec data available before end of targetVec
                append_i = append( append_i, pred_i )
                Tp_i     = [ i for i in range( append_i[-1] + 1,
                                               append_i[-1] + self.Tp + 1) ]
                append_i = append( append_i, array( Tp_i, dtype = int ) )
            else :
                # targetVec data not available at prediction end
                append_i = append( append_i, pred_i )

        else : # Negative Tp
            if pred_i[0] + self.Tp > -1 :
                # targetVec data available after begin of pred_i[0]
                append_i = append( append_i, pred_i )
                Tp_i     = [ i for i in range( pred_i[0] + self.Tp,
                                               pred_i[0] ) ]
                append_i = append( array( Tp_i, dtype = int ), append_i )
            else :
                # targetVec data not available before pred_i[0]
                append_i = append( append_i, pred_i )

        obs_i = append( obs_i, append_i )

    observations = self.targetVec[ obs_i, 0 ]

    #----------------------------------------------------
    # Projections & variance
    #----------------------------------------------------
    # Define array's of indices predOut_i, obsOut_i for DataFrame vectors
    predOut_i = array( [], dtype = int )
    predOut_0 = 0

    # Process each pred segment in self.predList for predOut_i
    for pred_i in self.predList :
        N_pred    = len( pred_i )
        outSize_i = N_pred + Tp_magnitude

        if N_pred == 0 :
            # No prediction made for this pred segment
            continue

        if self.Tp == 0 :
            Tp_i      = [i for i in range( predOut_0, predOut_0 + N_pred )]
            predOut_i = append( predOut_i, array( Tp_i, dtype = int ) )
            predOut_0 = predOut_i[-1] + 1

        elif self.Tp > 0 : # Positive Tp
            Tp_i = [i for i in range( predOut_0 + self.Tp,
                                      predOut_0 + self.Tp + N_pred ) ]
            predOut_i = append( predOut_i, array( Tp_i, dtype = int ) )
            predOut_0 = predOut_i[-1] + 1

        else : # Negative Tp
            Tp_i = [i for i in range(predOut_0, predOut_0 + N_pred)]
            predOut_i = append( predOut_i, array( Tp_i, dtype = int ) )
            predOut_0 = predOut_i[-1] + Tp_magnitude + 1

    # If nan are present, the foregoing can be wrong since it is not
    # known before prediction what lib vectors will produce pred
    # If len( pred_i ) != len( predOut_i ), nan resulted in missing pred
    # Create a map between pred_i_all : predOut_i to create a new/shorter
    # predOut_i mapping pred_i to the output vector predOut_i
    if len( self.pred_i ) != len( predOut_i ) :
        # Map the last predOut_i values since embed shift near data begining
        # can have (E-1)*tau na, but still listed in pred_i_all
        N_ = len( predOut_i )

        if self.tau < 0 :
            D  = dict( zip( self.pred_i_all[ -N_: ], predOut_i ) )
        else :
            D  = dict( zip( self.pred_i_all[ :N_ ], predOut_i ) )

        # Reset predOut_i
        predOut_i = [ D[i] for i in self.pred_i ]

    # Create obsOut_i indices for output vectors in DataFrame
    if self.Tp > 0 : # Positive Tp
        if obs_i[-1] + self.Tp > self.Data.shape[0] - 1 :
            # Edge case of end of data with positive Tp
            obsOut_i = [ i for i in range( len(obs_i) ) ]
        else :
            obsOut_i = [ i for i in range( outSize ) ]

    elif self.Tp < 1 : # Negative or Zero Tp
        if self.pred_i[0] + self.Tp < 0 :
            # Edge case of start of data with negative Tp
            obsOut_i = [ i for i in range( len(obs_i) ) ]

            # Shift obsOut_i values based on leading nan
            shift    = Tp_magnitude - self.pred_i[0]
            obsOut_i = obsOut_i + shift
        else :
            obsOut_i = [ i for i in range( len(obs_i) ) ]

    obsOut_i = array( obsOut_i, dtype = int )

    # ndarray init to nan
    observationOut = full( outSize, nan )
    projectionOut  = full( outSize, nan )
    varianceOut    = full( outSize, nan )

    # fill *Out with observed & projected values
    observationOut[ obsOut_i ] = observations
    projectionOut[ predOut_i ] = self.projection
    varianceOut  [ predOut_i ] = self.variance

    #----------------------------------------------------
    # Time
    #----------------------------------------------------
    self.ConvertTime()

    if self.Tp == 0 or \
       (self.Tp > 0 and (self.pred_i_all[-1] + self.Tp) < len(self.time)) or \
       (self.Tp < 0 and (self.pred_i[0] + self.Tp >= 0)) :
        # All times present in self.time, copy them
        timeOut = empty( outSize, dtype = self.time.dtype )
        timeOut[ obsOut_i ] = self.time[ obs_i ]
    else :
        # Need to pre/append additional times
        timeOut = self.AddTime( Tp_magnitude, outSize, obs_i, obsOut_i )

    #----------------------------------------------------
    # Output DataFrame
    #----------------------------------------------------
    self.Projection = DataFrame(
        { 'Time'          : timeOut,
          'Observations'  : observationOut,
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
def ConvertTime( self ) :
#-------------------------------------------------------------------
    '''Replace self.time with ndarray numerically operable values
       ISO 8601 formats are supported in the time & datetime modules
    '''
    if self.verbose:
        print( f'{self.name}: ConvertTime()' )

    time0 = self.time[0]

    # If times are numerically operable, nothing to do.
    if isinstance( time0, int )     or isinstance( time0, float )    or \
       isinstance( time0, integer ) or isinstance( time0, floating ) or \
       isinstance( time0, dt.time ) or isinstance( time0, dt.datetime ) :
        return

    # Local copy of time
    time_ = self.time.copy() # ndarray

    # If times are strings, try to parse into time or datetime
    # If success, replace time with parsed time or datetime array
    if isinstance( time0, str ) :
        try :
            t0 = dt.time.fromisoformat( time0 )
            # Parsed t0 into dt.time OK. Parse the whole vector
            time_ = array( [ dt.time.fromisoformat(t) for t in time_ ] )
        except ValueError :
            try:
                t0 = dt.datetime.fromisoformat( time0 )
                # Parsed t0 into dt.datetime OK. Parse the whole vector
                time_ = array([ dt.datetime.fromisoformat(t) for t in time_ ])
            except ValueError :
                msg = f'{self.name} ConvertTime(): Time values are strings '+\
                      'but are not ISO 8601 recognized time or datetime.'
                raise RuntimeError( msg )

    # If times were string they have been converted to time or datetime
    # Ensure times can be numerically manipulated, compute deltaT
    try :
        deltaT = time_[1] - time_[0]
    except TypeError :
        msg = f'{self.name} ConvertTime(): Time values not recognized.' +\
            ' Accepted values are int, float, or string of ISO 8601'    +\
            ' compliant time or datetime.'
        raise RuntimeError( msg )

    # Replace DataFrame derived time with converted time_
    self.time = time_

#-------------------------------------------------------------------
def AddTime( self, Tp_magnitude, outSize, obs_i, obsOut_i ) :
#-------------------------------------------------------------------
    '''Prepend or append time values to self.time if needed
       Return timeOut vector with additional Tp points
    '''
    if self.verbose:
        print( f'{self.name}: AddTime()' )

    min_pred_i     = self.pred_i[0]
    max_pred_i_all = self.pred_i_all[-1]
    deltaT         = self.time[1] - self.time[0]

    # First, fill timeOut with times in time
    # timeOut should not be int (np.integer) since they cannot be nan
    time0 = self.time[0]
    if isinstance( time0, int ) or isinstance( time0, integer ) :
        time_dtype = float
    else :
        time_dtype = type( time0 )

    timeOut = full( outSize, nan, dtype = time_dtype )

    timeOut[ obsOut_i ] = self.time[ obs_i ]

    newTimes = full( Tp_magnitude, nan, dtype = time_dtype )

    if self.Tp > 0 :
        # Tp introduces time values beyond the range of time
        # Generate future times
        lastTime    = self.time[ max_pred_i_all ]
        newTimes[0] = lastTime + deltaT

        for i in range( 1, self.Tp ) :
            newTimes[ i ] = newTimes[ i-1 ] + deltaT

        timeOut[ -self.Tp : ] = newTimes

    else :
        # Tp introduces time values before the range of time
        # Generate past times
        newTimes[0] = self.time[ min_pred_i ] - deltaT
        for i in range( 1, Tp_magnitude ) :
            newTimes[ i ] = newTimes[ i-1 ] - deltaT

        newTimes = newTimes[::-1] # Reverse

        # Shift timeOut values based on leading nan
        shift = Tp_magnitude - self.pred_i[0]
        timeOut[ : Tp_magnitude ] = newTimes

    return timeOut
