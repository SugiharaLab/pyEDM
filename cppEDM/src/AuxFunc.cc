
#include "AuxFunc.h"

//---------------------------------------------------------------
// Common code for Simplex and Smap that embeds, extracts
// the target vector and computes neighbors.
//
// NOTE: time column is not returned in the embedding dataBlock.
//
// NOTE: If dataIn is embedded by Embed(), the returned dataBlock
//       has tau * (E-1) fewer rows than dataIn. Since dataIn is
//       included in the returned DataEmbedNN struct, the first
//       tau * (E-1) dataIn rows are deleted from dataIn.  The
//       target vector is also reduced.
//
// NOTE: If rows are deleted, then the library and prediction
//       vectors in Parameters are updated to reflect this. 
//---------------------------------------------------------------
DataEmbedNN EmbedNN( DataFrame<double>  dataIn,
                     Parameters        &param,
                     bool               checkDataRows )
{
    if ( checkDataRows ) {
        CheckDataRows( param, dataIn, "EmbedNN" );
    }
    
    //----------------------------------------------------------
    // Extract or embedd data block
    //----------------------------------------------------------
    DataFrame<double> dataBlock; // Multivariate or embedded DataFrame

    if ( param.embedded ) {
        // dataIn is multivariable block, no embedding needed
        // Select the specified columns 
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
        // embedded = false: Create the embedding block via Embed()
        // dataBlock will have tau * (E-1) fewer rows than dataIn
        dataBlock = Embed( dataIn, param.E, param.tau,
                           param.columns_str, param.verbose );
    }
    
    //----------------------------------------------------------
    // Get target (library) vector
    //----------------------------------------------------------
    std::valarray<double> target_vec;
    if ( param.targetIndex ) {
        target_vec = dataIn.Column( param.targetIndex );
    }
    else if ( param.targetName.size() ) {
        target_vec = dataIn.VectorColumnName( param.targetName );
    }
    else {
        // Default to first column, column i=0 is time
        target_vec = dataIn.Column( 1 );
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
        
        std::valarray<double> target_vec_embed( dataIn.NRows() - shift );
        // Bogus cast to ( std::valarray<double> ) for MSVC
        // as it doesn't export its own slice_array applied to []
        target_vec_embed = ( std::valarray<double> )
            target_vec[ std::slice( shift, target_vec.size() - shift, 1 ) ];
        
        target_vec = target_vec_embed;

        DataFrame<double> dataInEmbed( dataIn.NRows() - shift,
                                       dataIn.NColumns(),
                                       dataIn.ColumnNames() );
        
        for ( size_t row = 0; row < dataInEmbed.NRows(); row++ ) {
            dataInEmbed.WriteRow( row, dataIn.Row( row + shift ) );
        }

        // dataIn was passed in by value, so copy constructed.
        // JP Is it OK to reassign it here, then copy into dataEmbedNN?
        dataIn = dataInEmbed;

        // Adust param.library and param.prediction vectors of indices
        if ( shift > 0 ) {
            size_t library_len    = param.library.size();
            size_t prediction_len = param.prediction.size();

            // If 0, 1, ... shift are in library or prediction
            // those rows were deleted, delete these elements.
            // First, create a vector of indices to delete
            std::vector< size_t > deleted_elements( shift, 0 );
            std::iota( deleted_elements.begin(), deleted_elements.end(), 0 );

            // erase elements of row indices that were deleted
            for ( auto element =  deleted_elements.begin();
                       element != deleted_elements.end(); element++ ) {

                std::vector< size_t >::iterator it;
                it = std::find( param.library.begin(),
                                param.library.end(), *element );

                if ( it != param.library.end() ) {
                    param.library.erase( it );
                }
                
                it = std::find( param.prediction.begin(),
                                param.prediction.end(), *element );

                if ( it != param.prediction.end() ) {
                    param.prediction.erase( it );
                }
            }
            
            // Now offset all values by shift so that vectors indices
            // in library and prediction refer to the same data rows
            // before the deletion/shift.
            for ( auto li =  param.library.begin();
                       li != param.library.end(); li++ ) {
                *li = *li - shift;
            }
            for ( auto pi =  param.prediction.begin();
                       pi != param.prediction.end(); pi++ ) {
                *pi = *pi - shift;
            }
        } // if ( shift > 0 )
    }
    
    //----------------------------------------------------------
    // Nearest neighbors
    //----------------------------------------------------------
    Neighbors neighbors = FindNeighbors( dataBlock, param );

    // Create struct to return the objects
    DataEmbedNN dataEmbedNN = DataEmbedNN( dataIn, dataBlock, 
                                           target_vec, neighbors );
    return dataEmbedNN;
}

//----------------------------------------------------------
// Common code to Simplex and Smap for output generation
//----------------------------------------------------------
DataFrame<double> FormatOutput( Parameters            param,
                                size_t                N_row,
                                std::valarray<double> predictions,
                                std::valarray<double> const_predictions,
                                DataFrame<double>     dataFrameIn,
                                std::valarray<double> target_vec,
                                bool                  checkDataRows )
{
    if ( checkDataRows ) {
        CheckDataRows( param, dataFrameIn, "FormatOutput" );
    }
    
    std::slice pred_i = std::slice( param.prediction[0], N_row, 1 );
    
    //----------------------------------------------------
    // Time vector with additional Tp points
    //----------------------------------------------------
    std::valarray<double> time( N_row + param.Tp );
    
    // Insert times from prediction. Time is the 1st column
    time[ std::slice( 0, N_row, 1 ) ] =
        ( std::valarray<double> ) dataFrameIn.Column( 0 )[ pred_i ];

#ifdef DEBUG_ALL
    std::cout << "FormatOutput() >>>> " << time.size() << " >>> ";
    for( auto i = 0; i < time.size(); i++ ) {
        std::cout << time[i] << ",";
    } std::cout << std::endl;
#endif
    
    // Insert Tp times at end
    for ( size_t i = N_row; i < N_row + param.Tp; i++ ) {
        time[ i ] = time[ i - 1 ] + param.Tp;
    }

    //----------------------------------------------------
    // Observations: add Tp nan at end
    //----------------------------------------------------
    std::valarray<double> observations( N_row + param.Tp );
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
    size_t dataFrameColumms = param.const_predict ? 4 : 3;
    
    DataFrame<double> dataFrame( N_row + param.Tp, dataFrameColumms );
    
    if ( param.const_predict ) {
        dataFrame.ColumnNames() = { "Time", "Observations",
                                    "Predictions", "Const_Predictions" };
    }
    else {
        dataFrame.ColumnNames() = { "Time", "Observations", "Predictions" };
    }
    
    dataFrame.WriteColumn( 0, time );
    dataFrame.WriteColumn( 1, observations );
    dataFrame.WriteColumn( 2, predictionsOut );
    if ( param.const_predict ) {
        dataFrame.WriteColumn( 3, constPredictionsOut );
    }    
    return dataFrame;
}

//----------------------------------------------------------
// 
//----------------------------------------------------------
void CheckDataRows( Parameters        param,
                    DataFrame<double> dataFrameIn,
                    std::string       call )
{
    //-----------------------------------------------------------------
    // Validate the dataFrameIn rows against the lib and pred indices
    //-----------------------------------------------------------------
    // param.prediction has been zero-offset in Validate() : +1
    size_t prediction_max_i = param.prediction[param.prediction.size()-1] + 1;
    size_t library_max_i    = param.library   [param.library.size()   -1] + 1;

    if ( dataFrameIn.Column( 0 ).size() < prediction_max_i ) {
        std::stringstream errMsg;
        errMsg << call << "(): The prediction index "
               << prediction_max_i
               << " exceeds the number of data rows "
               << dataFrameIn.Column( 0 ).size();
        throw std::runtime_error( errMsg.str() );
    }
    if ( dataFrameIn.Column( 0 ).size() < library_max_i ) {
        std::stringstream errMsg;
        errMsg << call << "(): The library index " << library_max_i
               << " exceeds the number of data rows "
               << dataFrameIn.Column( 0 ).size();
        throw std::runtime_error( errMsg.str() );
    }
}
