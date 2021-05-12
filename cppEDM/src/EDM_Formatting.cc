
#include "EDM.h"
#include "DateTime.h"

//----------------------------------------------------------
// Validate dataFrameIn rows against lib and pred indices
//----------------------------------------------------------
void EDM::CheckDataRows( std::string call )
{
    // parameters.prediction & library have been zero-offset in Validate()
    // to convert from user specified data row to array indicies
    size_t prediction_max_i = parameters.prediction.back();
    size_t library_max_i    = parameters.library.back();

    if ( not parameters.embedded ) {
        if ( parameters.E < 1 ) {
            std::stringstream errMsg;
            errMsg << "CheckDataRows(): E = " << parameters.E
                   << " is invalid.\n" ;
            throw std::runtime_error( errMsg.str() );
        }
    }

    if ( data.NRows() <= prediction_max_i ) {
        std::stringstream errMsg;
        errMsg << "CheckDataRows(): " << call
               << ": The prediction index " << prediction_max_i + 1
               << " exceeds the number of data rows "
               << data.NRows();
        throw std::runtime_error( errMsg.str() );
    }

    if ( data.NRows() <= library_max_i ) {
        std::stringstream errMsg;
        errMsg << "CheckDataRows(): " << call
               << ": The library index " << library_max_i + 1
               << " exceeds the number of data rows "
               << data.NRows();
        throw std::runtime_error( errMsg.str() );
    }
}
//----------------------------------------------------------
// Validate dataFrameIn rows against lib and pred indices
//----------------------------------------------------------
void EDM::CheckValidLib( std::string call )
{

    // validLib should have at last |lib| rows long. But for simplicity we will
    // require validLib spans the entire dataset. Reduce this restriction later.
    if ( parameters.validLib.size() < data.NRows() ){
        std::stringstream errMsg;
        errMsg << "CheckValidLib(): " << call
               << ": The number of elements in validLib " 
               << parameters.validLib.size()
               << " is less than the number of data rows "
               << data.NRows();
        throw std::runtime_error( errMsg.str() );
    }

}

//----------------------------------------------------------
// Common code for Simplex and Smap output generation
//----------------------------------------------------------
void EDM::FormatOutput() {
    //----------------------------------------------------
    // TimeOut vector with additional Tp points
    //----------------------------------------------------
    size_t N_time       = data.Time().size(); // Used here for data.time bool
    size_t N_row        = parameters.prediction.size();
    size_t Tp_magnitude = abs( parameters.Tp );
    size_t outSize      = N_row + Tp_magnitude;

    std::vector< std::string > timeOut( outSize );

    // Populate timeOut vector with strings for output
    if ( N_time ) {
        FillTimes( std::ref( timeOut ) );
    }

    //----------------------------------------------------
    // Observations: Insert target data in observations
    //----------------------------------------------------
    std::valarray< double > observations( NAN, outSize );

    int startObservations = 0;
    int startTarget;

    if ( parameters.Tp > -1 ) { // Positive Tp
        startTarget = parameters.prediction[ 0 ] - embedShift;
    }
    else {                      // Negative Tp
        startTarget = parameters.prediction[ 0 ] - embedShift - Tp_magnitude;
    }
    if ( startTarget < 0 ) {
        startObservations = std::abs( startTarget );
        startTarget = 0;
    }

    size_t t = startTarget;
    for ( size_t o = startObservations; o < outSize; o++ ) {
        if ( t < target.size() ) {
            observations[ o ] = target[ t ];
        } else { break; }
        t++;
    }

    //---------------------------------------------------------------------
    // Predictions & variance
    //---------------------------------------------------------------------
    std::valarray< double > predictionsOut     ( NAN, outSize );
    std::valarray< double > constPredictionsOut( NAN, outSize );
    std::valarray< double > varianceOut        ( NAN, outSize );

    if ( parameters.Tp > -1 ) {  // Positive Tp ---------------------------
        std::slice predOut_i = std::slice( parameters.Tp, N_row, 1 );

        predictionsOut[ predOut_i ] = predictions;
        varianceOut   [ predOut_i ] = variance;

        if ( parameters.const_predict ) {
            constPredictionsOut[ predOut_i  ] = const_predictions;
        }
    }
    else {  // Negative Tp --------------------------------------------
        std::slice predOut_i = std::slice( 0, N_row - Tp_magnitude, 1 );
        std::slice predIn_i  = std::slice( 0, N_row, 1 );

        predictionsOut[ predOut_i ] = predictions[ predIn_i ];
        varianceOut   [ predOut_i ] = variance   [ predIn_i ];

        if ( parameters.const_predict ) {
            constPredictionsOut[ predOut_i ] = const_predictions[ predIn_i ];
        }
    }

    //----------------------------------------------------
    // Output DataFrame
    //----------------------------------------------------
    size_t dataFrameColumms = parameters.const_predict ? 4 : 3;

    projection = DataFrame< double >( N_row + Tp_magnitude, dataFrameColumms );

    if ( parameters.const_predict ) {
        projection.ColumnNames() = { "Observations", "Predictions", 
                                     "Pred_Variance", "Const_Predictions" };
    }
    else {
        projection.ColumnNames()={"Observations","Predictions","Pred_Variance"};
    }

    if ( N_time ) {
        projection.TimeName() = data.TimeName();
        projection.Time()     = timeOut;
    }

    projection.WriteColumn( 0, observations   );
    projection.WriteColumn( 1, predictionsOut );
    projection.WriteColumn( 2, varianceOut    );

    if ( parameters.const_predict ) {
        projection.WriteColumn( 3, constPredictionsOut );
    }

#ifdef DEBUG_ALL
    std::cout << "EDM::FormatOutput() time " << timeOut.size()
              << " pred " << predictionsOut.size()
              << " obs " << observations.size() << std::endl;
    std::cout << "FormatOutput() projection -------------------" << std::endl;
    std::cout << projection;
#endif
}

//----------------------------------------------------------
// Copy strings of time values into timeOut.
// If prediction times exceed times from the data,
// create new entries for the additional times. 
//----------------------------------------------------------
void EDM::FillTimes( std::vector< std::string > & timeOut )
{
    size_t N_time = allTime.size();

    if ( not N_time ) { return; }

    size_t N_row       = parameters.prediction.size();
    size_t max_pred_i  = parameters.prediction.back();
    size_t min_pred_i  = parameters.prediction.front();
    size_t TpMagnitude = abs( parameters.Tp );

    if ( timeOut.size() != N_row + TpMagnitude ) {
        std::stringstream errMsg;
        errMsg << "FillTimes(): timeOut vector length " << timeOut.size()
               << " is not equal to the number of predictions + Tp "
               << N_row + TpMagnitude << std::endl;
        throw std::runtime_error( errMsg.str() );
    }

    bool TpPositive = parameters.Tp > -1 ? true : false;

    if ( TpPositive ) {
        if ( max_pred_i - embedShift + parameters.Tp < N_time ) {
            // All times are present in allTime
            for ( size_t i = 0; i < N_row + TpMagnitude; i++ ) {
                int t_i = min_pred_i + i - embedShift;
                timeOut[ i ] = allTime[ t_i ];
            }
        }
        else {
            // Tp introduces time values beyond the range of time
            bool timeFormatWarningPrinted = false;

            // Times need to be generated beyond allTime
            // First, fill in times that are in allTime
            for ( size_t i = 0; i < N_row; i++ ) {
                int t_i = min_pred_i + i - embedShift;
                timeOut[ i ] = allTime[ t_i ];
            }

            // Now, generate future times
            // Try to parse the last time vector string as a date or datetime
            // if dtinfo.unrecognized_fmt = true; it is not a date or datetime
            datetime_info dtinfo = ParseDatetime( data.Time()[ max_pred_i ] );

            for ( int i = 0; i < parameters.Tp; i++ ) {
                std::stringstream tss;

                if ( dtinfo.unrecognized_fmt ) {
                    // Numeric so add Tp
                    double time_delta = std::stof( allTime[ 1 ] ) -
                                        std::stof( allTime[ 0 ] );

                    double newTime = std::stof( allTime[ max_pred_i ] ) +
                                     ( i + 1 - embedShift ) * time_delta;
                    tss << newTime;
                }
                else {
                    int time_delta = i + 1;
                    // Last two datetimes to compute time diff to add time delta
                    std::string time_new( data.Time()[ max_pred_i     ] );
                    std::string time_old( data.Time()[ max_pred_i - 1 ] );
                    std::string new_time =
                        IncrementDatetime( time_old, time_new, time_delta );

                    // Add +ti if not recognized format(datetime util returns "")
                    if ( new_time.size() ) {
                        tss << new_time;
                    }
                    else {
                        tss << data.Time()[ max_pred_i ] << " +" << i + 1;

                        if ( not timeFormatWarningPrinted ) {
                            std::cout << "FillTimes(): "
                                      << "time column unrecognized time format."
                                      << "\n\tManually adding + tp to the last"
                                      << " time column available." << std::endl;
                            timeFormatWarningPrinted = true;
                        }
                    }
                }

                timeOut[ N_row + i ] = tss.str();
            }
        }
    }
    else { // Tp Negative
        if ( int( min_pred_i ) - embedShift + parameters.Tp >= 0 ) {
            // All times are present in allTime
            for ( size_t i = 0; i < N_row + TpMagnitude; i++ ) {
                int t_i = min_pred_i + i + parameters.Tp - embedShift;
                timeOut[ i ] = allTime[ t_i ];
            }
        }
        else {
            // Tp introduces time values before the range of time
            bool timeFormatWarningPrinted = false;

            // Times need to be generated before allTime
            // First, fill in times that are in allTime
            for ( size_t i = 0; i < N_row; i++ ) {
                int t_i = min_pred_i + i - embedShift;
                timeOut[ i + TpMagnitude ] = allTime[ t_i ];
            }

            // Now, generate past times
            // Try to parse the first time vector string as a date or datetime
            // if dtinfo.unrecognized_fmt = true; it is not a date or datetime
            datetime_info dtinfo = ParseDatetime( data.Time()[ 0 ] );

            for ( int i = (int) TpMagnitude; i > 0; i-- ) {
                std::stringstream tss;

                if ( dtinfo.unrecognized_fmt ) {
                    // Numeric so subtract i * Tp
                    double time_delta = std::stof( allTime[ 1 ] ) -
                                        std::stof( allTime[ 0 ] );

                    double newTime = std::stof( allTime[ min_pred_i ] ) -
                                     ( i + embedShift ) * time_delta;
                    tss << newTime;
                }
                else {
                    int time_delta = i - 1;
                    // Get first two datetimes to compute time diff
                    // to add time delta
                    std::string time_new( data.Time()[ 1 ] );
                    std::string time_old( data.Time()[ 0 ] );
                    std::string new_time =
                        IncrementDatetime( time_old, time_new, time_delta );

                    // Subtract +ti if not a recognized format
                    // (datetime util returns "")
                    if ( new_time.size() ) {
                        tss << new_time;
                    }
                    else {
                        tss << data.Time()[ max_pred_i ] << " -" << i + 1;

                        if ( not timeFormatWarningPrinted ) {
                            std::cout << "FillTimes(): "
                                      << "time column unrecognized time format."
                                      << "\n\tManually adding - tp to the first"
                                      << " time column available." << std::endl;
                            timeFormatWarningPrinted = true;
                        }
                    }
                } // else not dtinfo.unrecognized_fmt

                timeOut[ (int) TpMagnitude - i ] = tss.str();

            } // for ( size_t i = 0; i < Tp_magnitude; i++ ) 
        }
    } // Tp Negative
}
