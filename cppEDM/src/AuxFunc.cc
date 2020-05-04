
#include "AuxFunc.h"
#include "DateTime.h"

namespace EDM_AuxFunc {
    std::mutex mtx;
}

//---------------------------------------------------------------
// Common code for Simplex and Smap:
//   1) Extract or Embed() data into dataBlock
//   2) Get target (library) vector
//   3) DeletePartialDataRows()
//   4) Adjust param.library and param.prediction indices
//   5) FindNeighbors()
//
// NOTE: time column is not returned in the embedding dataBlock.
//
// NOTE: If data is embedded by Embed(), the returned dataBlock
//       has tau * (E-1) fewer rows than data. Since data is
//       included in the returned DataEmbedNN struct, the first
//       (or last) tau * (E-1) data rows are deleted to match
//       dataBlock.  The target vector is also reduced.
//
// NOTE: If rows are deleted, then the library and prediction
//       vectors in Parameters are updated to reflect this. 
//---------------------------------------------------------------
DataEmbedNN EmbedNN( DataFrame<double> *data,
                     Parameters        &param,
                     bool               checkDataRows )
{
    DataFrame<double> &dataIn = std::ref( *data );
    
    if ( checkDataRows ) {
        CheckDataRows( param, dataIn, "EmbedNN: Input data" );
    }
    
    //----------------------------------------------------------
    // Extract or embed dataIn into dataBlock
    //----------------------------------------------------------
    DataFrame<double> dataBlock; // Multivariate or embedded DataFrame

    if ( param.embedded ) {
        // dataIn is a multivariable block, no embedding needed
        // Select the specified columns into dataBlock
        if ( param.columnNames.size() ) {
            dataBlock = dataIn.DataFrameFromColumnNames( param.columnNames );
        }
        else if ( param.columnIndex.size() ) {
            dataBlock = dataIn.DataFrameFromColumnIndex( param.columnIndex );
        }
        else {
            throw std::runtime_error( "EmbedNN(): colNames and "
                                      " colIndex are empty.\n" );
        }
    }
    else {
        // embedded = false: Create the embedding dataBlock via Embed()
        // dataBlock will have tau * (E-1) fewer rows than dataIn
        dataBlock = Embed( dataIn, param.E, param.tau,
                           param.columns_str, param.verbose );
    }

    //----------------------------------------------------------
    // Get target (library) vector
    //----------------------------------------------------------
    std::valarray<double> targetIn;
    if ( param.targetIndex ) {
        targetIn = dataIn.Column( param.targetIndex );
    }
    else if ( param.targetName.size() ) {
        targetIn = dataIn.VectorColumnName( param.targetName );
    }
    else {
        // Default to first column
        targetIn = dataIn.Column( 0 );
    }
    
    //------------------------------------------------------------
    // embedded = false: Embed() was called on dataIn
    // Remove target, dataIn rows as needed
    // Adjust param.library and param.prediction indices
    //------------------------------------------------------------
    if ( not param.embedded ) {

        if ( param.E < 1 ) {
            std::stringstream errMsg;
            errMsg << "EmbedNN(): E = " << param.E << " is invalid.\n" ;
            throw std::runtime_error( errMsg.str() );
        }
        
        size_t shift = abs( param.tau ) * ( param.E - 1 );

        // Copy targetIn excluding partial data into targetEmbed
        std::valarray<double> targetEmbed( dataIn.NRows() - shift );
        
        // Bogus cast to ( std::valarray<double> ) for MSVC
        // as it doesn't export its own slice_array applied to []
        if ( param.tau < 0 ) {
            targetEmbed = ( std::valarray<double> )
                targetIn[ std::slice( shift, targetIn.size() - shift, 1 ) ];
        }
        else {
            targetEmbed = ( std::valarray<double> )
                targetIn[ std::slice( 0, targetIn.size() - shift, 1 ) ];
        }
        
        // Resize targetIn to ignore partial data rows
        targetIn.resize( targetEmbed.size() );
        
        // Copy target without partial data into resized targetIn
        std::slice targetEmbed_i  = std::slice( 0, targetEmbed.size(), 1 );
        targetIn[ targetEmbed_i ] = ( std::valarray<double> )
            targetEmbed[ targetEmbed_i ];

        // Delete dataIn top or bottom rows of partial data
        if ( not dataIn.PartialDataRowsDeleted() ) {
            // Not thread safe
            std::lock_guard<std::mutex> lck( EDM_AuxFunc::mtx );
            
            dataIn.DeletePartialDataRows( shift, param.tau );
        }

        // Adjust param.library and param.prediction vectors of indices
        if ( shift > 0 ) {
            param.DeleteLibPred();
        }

        // Check boundaries again since rows were removed
        if ( checkDataRows ) {
            CheckDataRows( param, dataIn, "EmbedNN: Embedded data" );
        }
    }
    
    //----------------------------------------------------------
    // Nearest neighbors
    //----------------------------------------------------------
    Neighbors neighbors = FindNeighbors( dataBlock, param );

    // Create struct to return the objects
    DataEmbedNN dataEmbedNN = DataEmbedNN( &dataIn,  dataBlock, 
                                           targetIn, neighbors );
    return dataEmbedNN;
}

//----------------------------------------------------------
// Common code for Simplex and Smap output generation
//----------------------------------------------------------
DataFrame<double> FormatOutput( Parameters               param,
                                std::valarray<double>    predictions,
                                std::valarray<double>    const_predictions,
                                std::valarray<double>    variance,
                                std::valarray<double>    target_vec,
                                std::vector<std::string> time,
                                std::string              timeName )
{
    //----------------------------------------------------
    // TimeOut vector with additional Tp points
    //----------------------------------------------------
    size_t N_time       = time.size();
    size_t N_row        = predictions.size();
    size_t Tp_magnitude = abs( param.Tp );

    std::vector<std::string> timeOut( N_row + Tp_magnitude );

    // Populate timeOut vector with strings for output
    if ( N_time ) {
        FillTimes( param, time, std::ref( timeOut ) );
    }
    
    //----------------------------------------------------
    // Observations: Insert data; add Tp nan at end/start
    //----------------------------------------------------
    std::valarray<double> observations( N_row + Tp_magnitude );
    
    if ( param.Tp > -1 ) {  // Positive Tp ---------------------------
        std::slice pred_i = std::slice( param.prediction[0], N_row, 1 );
    
        observations[ std::slice( 0, N_row, 1 ) ] =
            ( std::valarray<double> ) target_vec[ pred_i ];
    
        for ( size_t i = N_row; i < N_row + param.Tp; i++ ) {
            observations[ i ] = NAN;  // assign nan at end
        }
    }
    else {  // Negative Tp -------------------------------------------
        std::slice pred_i;

        if ( param.prediction[0] >= Tp_magnitude ) {
            pred_i = std::slice( param.prediction[ 0 ] - Tp_magnitude,
                                 N_row + Tp_magnitude, 1 );
    
            observations[ std::slice( 0, N_row + Tp_magnitude, 1 ) ] =
                ( std::valarray<double> ) target_vec[ pred_i ];
        }
        else {
            // Edge case where -Tp preceeds available record pred
            pred_i = std::slice( 0, N_row + Tp_magnitude, 1 );
            
            observations[std::slice( Tp_magnitude, N_row + Tp_magnitude, 1 )] =
                ( std::valarray<double> ) target_vec[ pred_i ];

            for ( size_t i = 0; i < Tp_magnitude; i++ ) {
                observations[ i ] = NAN;  // assign nan at start
            }
       }
    }

    //------------------------------------------------------------------
    // Predictions & variance: Assign values; insert Tp nan at start/end
    //------------------------------------------------------------------
    std::valarray<double> predictionsOut     ( N_row + Tp_magnitude );
    std::valarray<double> constPredictionsOut( N_row + Tp_magnitude );
    std::valarray<double> varianceOut        ( N_row + Tp_magnitude );
    
    if ( param.Tp > -1 ) {  // Positive Tp ---------------------------
        std::slice predOut_i = std::slice( param.Tp, N_row, 1 );
        
        for ( size_t i = 0; i < param.Tp; i++ ) {
            predictionsOut[ i ] = NAN;  // assign nan at start
            varianceOut   [ i ] = NAN;  // assign nan at start
        } 
        predictionsOut[ predOut_i ] = predictions;
        varianceOut   [ predOut_i ] = variance;
        
        if ( param.const_predict ) {
            for ( size_t i = 0; i < param.Tp; i++ ) {
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
        
        if ( param.const_predict ) {
            constPredictionsOut[ predOut_i ] = const_predictions[ predIn_i ];

            for ( size_t i = N_row; i < N_row + Tp_magnitude; i++ ) {
                constPredictionsOut[ i ] = NAN;  // assign nan at end
            }
        }
    }

    //----------------------------------------------------
    // Create output DataFrame
    //----------------------------------------------------
    size_t dataFrameColumms = param.const_predict ? 4 : 3;
    
    DataFrame<double> dataFrame( N_row + Tp_magnitude, dataFrameColumms );
    
    if ( param.const_predict ) {
        dataFrame.ColumnNames() = { "Observations", "Predictions", 
                                    "Pred_Variance", "Const_Predictions" };
    }
    else {
        dataFrame.ColumnNames() = {"Observations","Predictions","Pred_Variance"};
    }

    if ( N_time ) {
        dataFrame.TimeName() = timeName;
        dataFrame.Time()     = timeOut;
    }

    dataFrame.WriteColumn( 0, observations   );
    dataFrame.WriteColumn( 1, predictionsOut );
    dataFrame.WriteColumn( 2, varianceOut    );
    
    if ( param.const_predict ) {
        dataFrame.WriteColumn( 3, constPredictionsOut );
    }

#ifdef DEBUG_ALL
    std::cout << "FormatOutput() time " << timeOut.size()
              << " pred " << predictionsOut.size()
              << " obs " << observations.size() << std::endl;
    std::cout << "FormatOutput() dataFrame -------------------" << std::endl;
    std::cout << dataFrame;
#endif
    
    return dataFrame;
}

//----------------------------------------------------------
// Copy strings of time values into timeOut.
// If prediction times exceed times from the data,
// create new entries for the additional times. 
//----------------------------------------------------------
void FillTimes( Parameters                param,
                std::vector<std::string>  time,
                std::vector<std::string> &timeOut )
{
    size_t N_time       = time.size();
    size_t N_row        = param.prediction.size();
    size_t max_pred_i   = param.prediction[ N_row - 1 ];
    size_t min_pred_i   = param.prediction[ 0 ];
    size_t Tp_magnitude = abs( param.Tp );

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
    if ( param.Tp > -1 ) {
        // Fill in times guaranteed to be in param.prediction indices
        for ( size_t i = 0; i < N_row; i++ ) {
            size_t pred_i = param.prediction[ i ];
            if ( pred_i < N_time ) {
                timeOut[ i ] = time[ pred_i ];
            }
        }

        // Now fill in times beyond param.prediction indices
        if ( max_pred_i + param.Tp < N_time ) {
            // All prediction times are available in time, get the rest
            for ( size_t i = 0; i < param.Tp; i++ ) {
                timeOut[ N_row + i ] = time[ max_pred_i + i + 1 ];
            }
        }
        else {
            // Tp introduces time values beyond the range of time
            bool time_format_warning_printed = false;

            // Try to parse the last time vector string as a date or datetime
            // if dtinfo.unrecognized_fmt = true; it is not a date or datetime
            datetime_info dtinfo = parse_datetime( time[ max_pred_i ] );

            for ( size_t i = 0; i < param.Tp; i++ ) {
                std::stringstream tss;
            
                if ( dtinfo.unrecognized_fmt ) {
                    // Numeric so add Tp
                    tss << std::stod( time[ max_pred_i ] ) + i + 1;
                }
                else {
                    int time_delta = i + 1;
                    // Last two datetimes to compute time diff to add time delta
                    std::string time_new( time[ max_pred_i     ] );
                    std::string time_old( time[ max_pred_i - 1 ] );
                    std::string new_time =
                        increment_datetime_str( time_old, time_new, time_delta );
                
                    // Add +ti if not recognized format(datetime util returns "")
                    if ( new_time.size() ) {
                        tss << new_time;
                    }
                    else {
                        tss << time[ max_pred_i ] << " +" << i + 1;

                        if ( not time_format_warning_printed ) {
                            std::cout << "FillTimes(): "
                                      << "time column unrecognized time format."
                                      << "\n\tManually adding + tp to the last"
                                      << " time column available." << std::endl;
                            time_format_warning_printed = true;
                        }
                    }
                }
            
                timeOut[ N_row + i ] = tss.str();
            }
        }
    }
    // Negative Tp -----------------------------------------------------
    else {
        // Fill in times guaranteed to be in param.prediction indices
        for ( size_t i = 0; i < N_row; i++ ) {
            size_t pred_i = param.prediction[ i ];
            if ( pred_i < N_time ) {
                // param.Tp is negative, start at timeOut[ 0 - param.Tp ]
                // timeOut is shifted forward to accomodate the preceeding Tp
                timeOut[ i + Tp_magnitude ] = time[ pred_i ];
            }
        }

        // Now fill in times before param.prediction indices
        if ( (int) min_pred_i + param.Tp >= 0 ) {
            // All prediction times are available in time, get the rest
            for ( size_t i = 0; i < Tp_magnitude; i++ ) {
                timeOut[ i ] = time[ param.prediction[ i ] - Tp_magnitude ];
            }
        }
        else {
            // Tp introduces time values before the range of time
            bool time_format_warning_printed = false;
            
            // Try to parse the first time vector string as a date or datetime
            // if dtinfo.unrecognized_fmt = true; it is not a date or datetime
            datetime_info dtinfo = parse_datetime( time[ 0 ] );

            for ( size_t i = 0; i < Tp_magnitude; i++ ) {
                std::stringstream tss;
            
                if ( dtinfo.unrecognized_fmt ) {
                    // Numeric so subtract i Tp
                    tss << std::stod( time[ Tp_magnitude - 1 ] ) - (i + 1);
                }
                else {
                    int time_delta = i - 1;
                    // Get first two datetimes to compute time diff
                    // to add time delta
                    std::string time_new( time[ 1 ] );
                    std::string time_old( time[ 0 ] );
                    std::string new_time =
                        increment_datetime_str( time_old, time_new, time_delta );
                
                    // Subtract +ti if not a recognized format
                    // (datetime util returns "")
                    if ( new_time.size() ) {
                        tss << new_time;
                    }
                    else {
                        tss << time[ max_pred_i ] << " -" << i + 1;

                        if ( not time_format_warning_printed ) {
                            std::cout << "FillTimes(): "
                                      << "time column unrecognized time format."
                                      << "\n\tManually adding - tp to the first"
                                      << " time column available." << std::endl;
                            time_format_warning_printed = true;
                        }
                    }
                } // else not dtinfo.unrecognized_fmt
                
                timeOut[ i ] = tss.str();
                
            } // for ( size_t i = 0; i < Tp_magnitude; i++ ) 
        } // else Tp introduces time values before the range of time
    } // else Negative Tp ------------------------------------------------
}

//----------------------------------------------------------
// Validate dataFrameIn rows against lib and pred indices
//----------------------------------------------------------
void CheckDataRows( Parameters         param,
                    DataFrame<double> &dataFrameIn,
                    std::string        call )
{
    // param.prediction & library have been zero-offset in Validate()
    // to convert from user specified data row to array indicies
    size_t prediction_max_i = param.prediction[ param.prediction.size() - 1 ];
    size_t library_max_i    = param.library   [ param.library.size()    - 1 ];

    size_t shift;
    if ( param.embedded ) {
        shift = 0;
    }
    else {
        if ( param.E < 1 ) {
            std::stringstream errMsg;
            errMsg << "CheckDataRows(): E = " << param.E << " is invalid.\n" ;
            throw std::runtime_error( errMsg.str() );
        }
    
        shift = abs( param.tau ) * ( param.E - 1 );
    }

    if ( dataFrameIn.NRows() <= prediction_max_i ) {
        std::stringstream errMsg;
        errMsg << "CheckDataRows(): " << call
               << ": The prediction index "
               << prediction_max_i + 1
               << " exceeds the number of data rows "
               << dataFrameIn.NRows();
        throw std::runtime_error( errMsg.str() );
    }
    
    if ( dataFrameIn.NRows() <= library_max_i + shift ) {
        std::stringstream errMsg;
        errMsg << "CheckDataRows(): " << call
               << ": The library index " << library_max_i + 1
               <<  " + tau(E-1) " << shift << " = "
               << library_max_i + 1 + shift
               << " exceeds the number of data rows "
               << dataFrameIn.NRows();
        throw std::runtime_error( errMsg.str() );
    }
}
