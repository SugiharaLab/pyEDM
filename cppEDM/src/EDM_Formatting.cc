
#include "EDM.h"
#include "DateTime.h"

//----------------------------------------------------------
// Clip data & target rows to match the embedding
//----------------------------------------------------------
void EDM::RemovePartialData()
{
    // NOTE : Not thread safe : Call needs mutex wrap

    if ( data.PartialDataRowsDeleted() ) {
        std::cout << "RemovePartialData(): Partial data rows have "
            "already been deleted." << std::endl;
        return;
    }

    data.PartialDataRowsDeleted() = true;

    int shift = abs( parameters.tau ) * ( parameters.E - 1 );
    
    // Delete data rows corresponding to embedding partial data rows
    data.DeletePartialDataRows( shift, parameters.tau );

    GetTarget();
    
    // Adjust parameters.library and parameters.prediction vectors of indices
    if ( shift > 0 ) {
        parameters.DeleteLibPred();
    }
}

//----------------------------------------------------------
// Validate dataFrameIn rows against lib and pred indices
//----------------------------------------------------------
void EDM::CheckDataRows( std::string call )
{
    // parameters.prediction & library have been zero-offset in Validate()
    // to convert from user specified data row to array indicies
    size_t prediction_max_i =
        parameters.prediction[ parameters.prediction.size() - 1 ];

    size_t library_max_i =
        parameters.library[ parameters.library.size() - 1 ];

    size_t shift;
    if ( parameters.embedded ) {
        shift = 0;
    }
    else {
        if ( parameters.E < 1 ) {
            std::stringstream errMsg;
            errMsg << "CheckDataRows(): E = " << parameters.E
                   << " is invalid.\n" ;
            throw std::runtime_error( errMsg.str() );
        }

        shift = abs( parameters.tau ) * ( parameters.E - 1 );
    }

    if ( data.NRows() <= prediction_max_i ) {
        std::stringstream errMsg;
        errMsg << "CheckDataRows(): " << call
               << ": The prediction index "
               << prediction_max_i + 1
               << " exceeds the number of data rows "
               << data.NRows();
        throw std::runtime_error( errMsg.str() );
    }

    // Tweak for CCM that sets lib = pred = [1, NRow]
    if ( parameters.method == Method::CCM ) { shift = 0; }
    
    if ( data.NRows() <= library_max_i + shift ) {
        std::stringstream errMsg;
        errMsg << "CheckDataRows(): " << call
               << ": The library index " << library_max_i + 1
               <<  " + tau(E-1) " << shift << " = "
               << library_max_i + 1 + shift
               << " exceeds the number of data rows "
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
    size_t N_time       = data.Time().size();
    size_t N_row        = parameters.prediction.size();
    size_t Tp_magnitude = abs( parameters.Tp );

    std::vector< std::string > timeOut( N_row + Tp_magnitude );

    // Populate timeOut vector with strings for output
    if ( N_time ) {
        FillTimes( std::ref( timeOut ) );
    }

    //----------------------------------------------------
    // Observations: Insert data; add Tp nan at end/start
    //----------------------------------------------------
    std::valarray< double > observations( N_row + Tp_magnitude );

    if ( parameters.Tp > -1 ) {  // Positive Tp ---------------------------
        std::slice pred_i = std::slice( parameters.prediction[0], N_row, 1 );
    
        observations[ std::slice( 0, N_row, 1 ) ] =
            ( std::valarray< double > ) target[ pred_i ];
    
        for ( size_t i = N_row; i < N_row + parameters.Tp; i++ ) {
            observations[ i ] = NAN;  // assign nan at end
        }
    }
    else {  // Negative Tp -------------------------------------------
        std::slice pred_i;

        if ( parameters.prediction[0] >= Tp_magnitude ) {
            pred_i = std::slice( parameters.prediction[ 0 ] - Tp_magnitude,
                                 N_row + Tp_magnitude, 1 );
    
            observations[ std::slice( 0, N_row + Tp_magnitude, 1 ) ] =
                ( std::valarray< double > ) target[ pred_i ];
        }
        else {
            // Edge case where -Tp preceeds available record pred
            pred_i = std::slice( 0, N_row + Tp_magnitude, 1 );

            observations[std::slice( Tp_magnitude, N_row, 1 )] =
                ( std::valarray< double > ) target[ pred_i ];

            for ( size_t i = 0; i < Tp_magnitude; i++ ) {
                observations[ i ] = NAN;  // assign nan at start
            }
        }
    }

    //------------------------------------------------------------------
    // Predictions & variance: Assign values; insert Tp nan at start/end
    //------------------------------------------------------------------
    std::valarray< double > predictionsOut     ( N_row + Tp_magnitude );
    std::valarray< double > constPredictionsOut( N_row + Tp_magnitude );
    std::valarray< double > varianceOut        ( N_row + Tp_magnitude );

    if ( parameters.Tp > -1 ) {  // Positive Tp ---------------------------
        std::slice predOut_i = std::slice( parameters.Tp, N_row, 1 );

        for ( int i = 0; i < parameters.Tp; i++ ) {
            predictionsOut[ i ] = NAN;  // assign nan at start
            varianceOut   [ i ] = NAN;  // assign nan at start
        }
        predictionsOut[ predOut_i ] = predictions;
        varianceOut   [ predOut_i ] = variance;

        if ( parameters.const_predict ) {
            for ( int i = 0; i < parameters.Tp; i++ ) {
                constPredictionsOut[ i ] = NAN;  // assign nan at start
            }
            constPredictionsOut[ predOut_i  ] = const_predictions;
        }
    }
    else {  // Negative Tp --------------------------------------------
        std::slice predOut_i = std::slice( 0, N_row - Tp_magnitude, 1 );
        std::slice predIn_i  = std::slice( 0, N_row, 1 );

        predictionsOut[ predOut_i ] = predictions[ predIn_i ];
        varianceOut   [ predOut_i ] = variance   [ predIn_i ];

        for ( size_t i = N_row; i < N_row + Tp_magnitude; i++ ) {
            predictionsOut[ i ] = NAN;  // assign nan at end
            varianceOut   [ i ] = NAN;  // assign nan at end
        } 

        if ( parameters.const_predict ) {
            constPredictionsOut[ predOut_i ] = const_predictions[ predIn_i ];

            for ( size_t i = N_row; i < N_row + Tp_magnitude; i++ ) {
                constPredictionsOut[ i ] = NAN;  // assign nan at end
            }
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
    size_t N_time       = data.Time().size();
    size_t N_row        = parameters.prediction.size();
    size_t max_pred_i   = parameters.prediction[ N_row - 1 ];
    size_t min_pred_i   = parameters.prediction[ 0 ];
    size_t Tp_magnitude = abs( parameters.Tp );

    if ( max_pred_i >= N_time ) {
        // If tau > 0 end rows were deleted. max_pred_i might exceed time bounds
        max_pred_i = N_time - 1;
    }

    if ( timeOut.size() != N_row + Tp_magnitude ) {
        std::stringstream errMsg;
        errMsg << "FillTimes(): timeOut vector length " << timeOut.size()
               << " is not equal to the number of predictions + Tp "
               << N_row + Tp_magnitude << std::endl;
        throw std::runtime_error( errMsg.str() );
    }

    // Positive Tp -----------------------------------------------------
    if ( parameters.Tp > -1 ) {
        // Fill in times guaranteed to be in parameters.prediction indices
        for ( size_t i = 0; i < N_row; i++ ) {
            size_t pred_i = parameters.prediction[ i ];
            if ( pred_i < N_time ) {
                timeOut[ i ] = data.Time()[ pred_i ];
            }
        }

        // Now fill in times beyond parameters.prediction indices
        if ( max_pred_i + parameters.Tp < N_time ) {
            // All prediction times are available in time, get the rest
            for ( int i = 0; i < parameters.Tp; i++ ) {
                timeOut[ N_row + i ] = data.Time()[ max_pred_i + i + 1 ];
            }
        }
        else {
            // Tp introduces time values beyond the range of time
            bool timeFormatWarningPrinted = false;

            // Try to parse the last time vector string as a date or datetime
            // if dtinfo.unrecognized_fmt = true; it is not a date or datetime
            datetime_info dtinfo = ParseDatetime( data.Time()[ max_pred_i ] );

            for ( int i = 0; i < parameters.Tp; i++ ) {
                std::stringstream tss;
            
                if ( dtinfo.unrecognized_fmt ) {
                    // Numeric so add Tp
                    tss << std::stod( data.Time()[ max_pred_i ] ) + i + 1;
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
    // Negative Tp -----------------------------------------------------
    else {
        // Fill in times guaranteed to be in parameters.prediction indices
        for ( size_t i = 0; i < N_row; i++ ) {
            size_t pred_i = parameters.prediction[ i ];
            if ( pred_i < N_time ) {
                // parameters.Tp is negative, start at timeOut[0 - parameters.Tp]
                // timeOut is shifted forward to accomodate the preceeding Tp
                timeOut[ i + Tp_magnitude ] = data.Time()[ pred_i ];
            }
        }

        // Now fill in times before parameters.prediction indices
        if ( (int) min_pred_i + parameters.Tp >= 0 ) {
            // All prediction times are available in time, get the rest
            for ( size_t i = 0; i < Tp_magnitude; i++ ) {
                timeOut[ i ] = data.Time()[ parameters.prediction[ i ] -
                                            Tp_magnitude ];
            }
        }
        else {
            // Tp introduces time values before the range of time
            bool timeFormatWarningPrinted = false;
            
            // Try to parse the first time vector string as a date or datetime
            // if dtinfo.unrecognized_fmt = true; it is not a date or datetime
            datetime_info dtinfo = ParseDatetime( data.Time()[ 0 ] );

            for ( size_t i = 0; i < Tp_magnitude; i++ ) {
                std::stringstream tss;
            
                if ( dtinfo.unrecognized_fmt ) {
                    // Numeric so subtract i Tp
                    tss << std::stod( data.Time()[Tp_magnitude - 1] ) - (i + 1);
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
                
                timeOut[ i ] = tss.str();

            } // for ( size_t i = 0; i < Tp_magnitude; i++ ) 
        } // else Tp introduces time values before the range of time
    } // else Negative Tp ------------------------------------------------
}
