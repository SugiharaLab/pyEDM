
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
//       tau * (E-1) data rows are deleted to match dataBlock.
//       The target vector is also reduced.
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
        CheckDataRows( param, dataIn, "EmbedNN" );
    }
    
    //----------------------------------------------------------
    // Extract or embed dataIn into dataBlock
    //----------------------------------------------------------
    DataFrame<double> dataBlock; // Multivariate or embedded DataFrame

    if ( param.embedded ) {
        // dataIn is a multivariable block, no embedding needed
        // Select the specified columns into dataBlock
        if ( param.columnNames.size() ) {
            dataBlock = dataIn.DataFrameFromColumnNames(param.columnNames);
        }
        else if ( param.columnIndex.size() ) {
            dataBlock = dataIn.DataFrameFromColumnIndex(param.columnIndex);
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
        // If we support negative tau, this will change
        // For now, assume only positive tau is allowed
        size_t shift = std::max( 0, param.tau * (param.E - 1) );

        // Copy targetIn excluding partial data into targetEmbed
        std::valarray<double> targetEmbed( dataIn.NRows() - shift );
        // Bogus cast to ( std::valarray<double> ) for MSVC
        // as it doesn't export its own slice_array applied to []
        targetEmbed = ( std::valarray<double> )
            targetIn[ std::slice( shift, targetIn.size() - shift, 1 ) ];
        
        // Resize targetIn to ignore partial data rows
        targetIn.resize( targetEmbed.size() );
        
        // Copy target without partial data into resized targetIn
        std::slice targetEmbed_i  = std::slice( 0, targetEmbed.size(), 1 );
        targetIn[ targetEmbed_i ] = ( std::valarray<double> )
            targetEmbed[ targetEmbed_i ];

        // Delete dataIn top rows of partial data
        if ( not dataIn.PartialDataRowsDeleted() ) {
            // Not thread safe
            std::lock_guard<std::mutex> lck( EDM_AuxFunc::mtx );
            
            dataIn.DeletePartialDataRows( shift );
        }

        // Adjust param.library and param.prediction vectors of indices
        if ( shift > 0 ) {
            param.DeleteLibPred( shift );
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
                                std::valarray<double>    target_vec,
                                std::vector<std::string> time,
                                std::string              timeName )
{

#ifdef DEBUG_ALL
    std::cout << "FormatOutput() param.prediction.size: "
              << param.prediction.size() << " >>> ";
    for( auto i = 0; i < param.prediction.size(); i++ ) {
        std::cout << param.prediction[i] << ",";
    } std::cout << std::endl;
    std::cout << "FormatOutput() time.size: " << time.size() << " >>> ";
    for( auto i = 0; i < time.size(); i++ ) {
        std::cout << time[i] << ",";
    } std::cout << std::endl;
#endif

    //----------------------------------------------------
    // Time vector with additional Tp points
    //----------------------------------------------------
    size_t N_time = time.size();
    size_t N_row  = predictions.size();

    // Populate vector of time strings for output
    std::vector<std::string> timeOut( N_row + param.Tp );

    if ( N_time ) {
        FillTimes( param, time, std::ref( timeOut ) );
    }
    
    //----------------------------------------------------
    // Observations: add Tp nan at end
    //----------------------------------------------------
    std::valarray<double> observations( N_row + param.Tp );
    std::slice pred_i = std::slice( param.prediction[0], N_row, 1 );
    
    observations[ std::slice( 0, N_row, 1 ) ] =
        ( std::valarray<double> ) target_vec[ pred_i ];
    
    for ( size_t i = N_row; i < N_row + param.Tp; i++ ) {
        observations[ i ] = NAN;
    }

    //----------------------------------------------------
    // Predictions: insert Tp nan at start
    //----------------------------------------------------
    std::valarray<double> predictionsOut( N_row + param.Tp );
    for ( size_t i = 0; i < param.Tp; i++ ) {
        predictionsOut[ i ] = NAN;
    }
    predictionsOut[ std::slice(param.Tp, N_row, 1) ] = predictions;

    std::valarray<double> constPredictionsOut( N_row + param.Tp );
    if ( param.const_predict ) {
        for ( size_t i = 0; i < param.Tp; i++ ) {
            constPredictionsOut[ i ] = NAN;
        }
        constPredictionsOut[ std::slice(param.Tp, N_row, 1) ] =
            const_predictions;
    }
    
    //----------------------------------------------------
    // Create output DataFrame
    //----------------------------------------------------
    size_t dataFrameColumms = param.const_predict ? 3 : 2;
    
    DataFrame<double> dataFrame( N_row + param.Tp, dataFrameColumms );
    
    if ( param.const_predict ) {
        dataFrame.ColumnNames() = { "Observations",
                                    "Predictions", "Const_Predictions" };
    }
    else {
        dataFrame.ColumnNames() = { "Observations", "Predictions" };
    }

    if ( N_time ) {
        dataFrame.TimeName() = timeName;
        dataFrame.Time()     = timeOut;
    }
    
    dataFrame.WriteColumn( 0, observations );
    dataFrame.WriteColumn( 1, predictionsOut );
    
    if ( param.const_predict ) {
        dataFrame.WriteColumn( 2, constPredictionsOut );
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
    size_t N_time     = time.size();
    size_t N_row      = param.prediction.size();
    size_t max_pred_i = param.prediction[ N_row - 1 ];

    if ( timeOut.size() != N_row + param.Tp ) {
        std::stringstream errMsg;
        errMsg << "FillTimes(): timeOut vector length " << timeOut.size()
               << " is not equal to the number of predictions + Tp "
               << N_row + param.Tp << std::endl;
        throw std::runtime_error( errMsg.str() );
    }
    
    // Fill in times guaranteed to be in param.prediction indices
    for ( auto i = 0; i < N_row; i++ ) {
        timeOut[ i ] = time[ param.prediction[ i ] ];
    }
    
    // Now fill in times beyond param.prediction indices
    if ( max_pred_i + param.Tp < N_time ) {
        // All prediction times are available in time, get the rest
        for ( auto i = 0; i < param.Tp; i++ ) {
            timeOut[ N_row + i ] = time[ max_pred_i + i + 1 ];
        }
    }
    else {
        bool time_format_warning_printed = false;
        
        // Tp introduces time values beyond the range of time
        for ( auto i = N_row; i < N_row + param.Tp; i++ ) {
            std::stringstream tss;
            
            if ( OnlyDigits( time[ max_pred_i ], false ) ) {
                // Numeric so add Tp
                tss << std::stod( time[ max_pred_i ] ) + i - N_row + 1;
            }
            else {
                int time_delta = i - N_row + 1;
                // Get last two datetimes to compute time diff to add time delta
                std::string time_new( time[ max_pred_i     ] );
                std::string time_old( time[ max_pred_i - 1 ] );
                std::string new_time =
                    increment_datetime_str( time_old, time_new, time_delta );
                
                // Add +ti if not a recognized format (datetime util returns "")
                if ( new_time.size() ) {
                    tss << new_time;
                }
                else {
                    tss << time[ max_pred_i ] << " +" << i - N_row + 1;

                    if ( not time_format_warning_printed ) {
                        std::cout << "FillTimes(): "
                                  << "time column is unrecognized time format."
                                  << "\n\tManually adding + tp to the last"
                                  << " time column available." << std::endl;
                        time_format_warning_printed = true;
                    }
                }
            }
            
            timeOut[ i ] = tss.str();
        }
    }
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
        shift = std::max( 0, param.tau * (param.E - 1) );
    }

    if ( dataFrameIn.NRows() <= prediction_max_i + shift ) {
        std::stringstream errMsg;
        errMsg << "CheckDataRows(): " << call
               << ": The prediction index + tau(E-1) "
               << prediction_max_i + shift
               << " equals or exceeds the number of data rows "
               << dataFrameIn.NRows();
        throw std::runtime_error( errMsg.str() );
    }
    
    if ( dataFrameIn.NRows() <= library_max_i + shift ) {
        std::stringstream errMsg;
        errMsg << "CheckDataRows(): " << call
               << ": The library index + tau(E-1) "
               << library_max_i + shift
               << " equals or exceeds the number of data rows "
               << dataFrameIn.NRows();
        throw std::runtime_error( errMsg.str() );
    }
}
