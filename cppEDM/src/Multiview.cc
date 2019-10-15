
//--------------------------------------------------------------------
// Data input requires columns to specify timeseries columns
// that will be embedded by Embed(), and target for predictions.
// 
// E represents the number of variables to combine for each
// assessment, as well as the number of time delays to create in 
// Embed() for each variable.
// 
// multiview is the number of top-ranked E-dimensional predictions
// to "average" for the final prediction. Corresponds to parameter
// k in Ye & Sugihara with default k = sqrt(m) where m is the
// number of combinations C(n,E) available from the n = nColumns * E
// columns taken E at-a-time. 
//
// Parameters.Validate() with Method::Simplex sets knn equal to E+1
// if knn not specified.
// 
// Ye H., and G. Sugihara, 2016. Information leverage in
// interconnected ecosystems: Overcoming the curse of dimensionality.
// Science 353:922–925.
//--------------------------------------------------------------------

#include <thread>
#include <atomic>
#include <mutex>
#include <queue>

#include "Common.h"
#include "AuxFunc.h"

namespace EDM_Multiview {
    // Thread Work Queue : Vector of combos indices
    typedef std::vector< int > WorkQueue;
    
    // Thread exception_ptr queue
    std::queue< std::exception_ptr > exceptionQ;
    
    // atomic counter for all threads
    std::atomic<std::size_t> eval_i(0); // initialize to 0
    
    std::mutex mtx;
    std::mutex q_mtx;
}

//----------------------------------------------------------------
// forward declarations
//----------------------------------------------------------------
std::vector< std::vector< size_t > > Combination( int n, int k );

DataFrame<double> SimplexProjection( Parameters  param,
                                     DataEmbedNN embedNN,
                                     bool        checkDataRows = true );

void EvalComboThread( Parameters                            param,
                      EDM_Multiview::WorkQueue              workQ,
                      std::vector< std::vector< size_t > >  combos,
                      DataFrame< double >                  &embedding,
                      std::valarray< double >              &targetVec,
                      DataFrame< double >                  &combos_rho,
                      std::vector< DataFrame< double > >   &prediction );

std::vector< std::string > ComboRhoTable( DataFrame<double>          combosRho,
                                          std::vector< std::string > colNames );

//----------------------------------------------------------------
// Multiview() : Evaluate Simplex rho vs. dimension E
// API Overload 1: Explicit data file path/name
//     Implemented as a wrapper to API Overload 2:
//----------------------------------------------------------------
MultiviewValues Multiview( std::string pathIn,
                           std::string dataFile,
                           std::string pathOut,
                           std::string predictFile,
                           std::string lib,
                           std::string pred,
                           int         E,
                           int         Tp,
                           int         knn,
                           int         tau,
                           std::string columns,
                           std::string target,
                           int         multiview,
                           int         exclusionRadius,
                           bool        verbose,
                           unsigned    nThreads ) {

    // Create DataFrame (constructor loads data)
    DataFrame< double > dataFrameIn( pathIn, dataFile );
    
    MultiviewValues result = Multiview( dataFrameIn,
                                        pathOut,
                                        predictFile,
                                        lib,
                                        pred,
                                        E,
                                        Tp,
                                        knn,
                                        tau,
                                        columns,
                                        target,
                                        multiview,
                                        exclusionRadius,
                                        verbose,
                                        nThreads );
    return result;
}

//----------------------------------------------------------------
// Multiview()
// API Overload 2: DataFrame provided
//----------------------------------------------------------------
MultiviewValues  Multiview( DataFrame< double > data,
                            std::string         pathOut,
                            std::string         predictFile,
                            std::string         lib,
                            std::string         pred,
                            int                 E,
                            int                 Tp,
                            int                 knn,
                            int                 tau,
                            std::string         columns,
                            std::string         target,
                            int                 multiview,
                            int                 exclusionRadius,
                            bool                verbose,
                            unsigned            nThreads ) {

    // Create local Parameters struct. Note embedded = true
    Parameters param = Parameters( Method::Simplex, "", "",
                                   pathOut, predictFile,
                                   lib, pred, E, Tp, knn, tau, 0,
                                   exclusionRadius, columns, target,
                                   true, false, verbose,
                                   "", "", "", 0, 0, 0, multiview );

    if ( not param.columnNames.size() ) {
        throw std::runtime_error( "Multiview() requires column names." );
    }
    if ( not param.targetName.size() ) {
        throw std::runtime_error( "Multiview() requires target name." );
    }
    if ( param.E < 1 ) {
        throw std::runtime_error( "Multiview() E is required." );
    }
    // Ensure that params are validated so columnNames are populated
    if ( not param.validated ) {
        throw std::runtime_error( "Multiview() params not validated." );        
    }

    // Validate that columns & target are in data
    for ( auto colName : param.columnNames ) {
        auto ci = find( data.ColumnNames().begin(),
                        data.ColumnNames().end(), colName );
        
        if ( ci == data.ColumnNames().end() ) {
            std::stringstream errMsg;
            errMsg << "Multiview(): Failed to find column "
                   << colName << " in dataFrame with columns: [ ";
            for ( auto col : data.ColumnNames() ) {
                errMsg << col << " ";
            } errMsg << " ]\n";
            throw std::runtime_error( errMsg.str() );
        }
    }
    auto ti = find( data.ColumnNames().begin(),
                    data.ColumnNames().end(), param.targetName );
    if ( ti == data.ColumnNames().end() ) {
        std::stringstream errMsg;
        errMsg << "Multiview(): Failed to find target "
               << param.targetName << " in dataFrame with columns: [ ";
        for ( auto col : data.ColumnNames() ) {
            errMsg << col << " ";
        } errMsg << " ]\n";
        throw std::runtime_error( errMsg.str() );
    }

    // Validate data rows against lib and pred indices
    CheckDataRows( param, std::ref( data ), "Multiview()" );

    //------------------------------------------------------------
    // Generate embedding on param.columns_str
    // embedding will have tau * (E-1) fewer rows than data
    //------------------------------------------------------------
    DataFrame< double > embedding = Embed( data,
                                           param.E,
                                           param.tau,
                                           param.columns_str,
                                           param.verbose );

    size_t shift = std::max( 0, param.tau * (param.E - 1) );
    
    // Delete data top rows of partial data
    if ( not data.PartialDataRowsDeleted() ) {
        // Not thread safe
        std::lock_guard<std::mutex> lck( EDM_Multiview::mtx );
        
        data.DeletePartialDataRows( shift );
    }
        
    // Adjust param.library and param.prediction vectors of indices
    if ( shift > 0 ) {
        param.DeleteLibPred( shift );
    }

    // Get target
    std::valarray< double > targetVec = data.VectorColumnName(param.targetName);
    
    // Save param.predictOutputFile and reset so Simplex() does not write 
    std::string outputFile = param.predictOutputFile;
    param.predictOutputFile = "";
    
    // Combinations of possible embedding variables (columns), E at-a-time
    // Note that these combinations are not zero-offset, i.e.
    // Combination( 3, 2 ) = [(1, 2), (1, 3), (2, 3)]
    // These correspond to column indices +1
    size_t nVar = param.columnNames.size();
    std::vector< std::vector< size_t > > combos =
        Combination( nVar * param.E, param.E );

#ifdef DEBUG_ALL
    std::cout << "Multiview(): " << combos.size() << " combos:\n";
    for ( auto i = 0; i < combos.size(); i++ ) {
        std::vector< size_t > combo_i = combos[i];
        std::cout << "[";
        for ( auto j = 0; j < combo_i.size(); j++ ) {
            std::cout << combo_i[j] << ",";
        }
        std::cout << "] ";
    } std::cout << std::endl;
#endif
    
    // Establish number of ensembles if not specified
    if ( not param.MultiviewEnsemble ) {
        // Ye & Sugihara suggest sqrt( m ) as the number of embeddings to avg
        param.MultiviewEnsemble = std::max(2, (int) std::sqrt(combos.size()));
        
        std::stringstream msg;
        msg << "Multiview() Set view sample size to "
            << param.MultiviewEnsemble << std::endl;
        std::cout << msg.str();
    }

    //---------------------------------------------------------------
    // Evaluate variable combinations.
    // Note that this is done within the library itself (in-sample).
    //---------------------------------------------------------------
    // Save a copy of the specified prediction observation rows.
    std::vector<size_t> prediction = param.prediction;
    
    // Override the param.prediction for in-sample forecast skill evaluation
    param.prediction = param.library;

    // Create column names for the results DataFrame
    // One row for each combo: E columns (a combo), rho, MAE, RMSE
    // JP: NOTE DataFrame is based on valarray, so we can't store non
    //     numeric values (column names) in the results DataFrame.
    std::stringstream header;
    for ( auto i = 1; i <= param.E; i++ ) {
        header << "Col_" << i << " ";
    }
    header << "rho MAE RMSE";

    // Results Data Frame: E columns (a combo), rho, mae, rmse
    DataFrame<double> combos_rho( combos.size(), param.E + 3, header.str() );
    
    // Results vector of DataFrame's with prediction results
    std::vector< DataFrame< double > > combos_prediction( combos.size() );
        
    // Build work queue
    EDM_Multiview::WorkQueue workQ( combos.size() );

    // Insert combos index into work queue
    for ( auto i = 0; i < combos.size(); i++ ) {
        workQ[ i ] = i;
    }

    unsigned maxThreads = std::thread::hardware_concurrency();
    if ( maxThreads < nThreads ) { nThreads = maxThreads; }
    
    // thread container
    std::vector< std::thread > threads;
    for ( unsigned i = 0; i < nThreads; ++i ) {
        threads.push_back( std::thread( EvalComboThread,
                                        param,
                                        workQ,
                                        combos,
                                        std::ref( embedding ),
                                        std::ref( targetVec ),
                                        std::ref( combos_rho ),
                                        std::ref( combos_prediction ) ) );
    }
    
    // join threads
    for ( auto &thrd : threads ) {
        thrd.join();
    }

    // If thread threw exception, get from queue and rethrow
    if ( not EDM_Multiview::exceptionQ.empty() ) {
        std::lock_guard<std::mutex> lck( EDM_Multiview::q_mtx );

        // Take the first exception in the queue
        std::exception_ptr exceptionPtr = EDM_Multiview::exceptionQ.front();

        // Unroll all other exception from the thread/loops
        while( not EDM_Multiview::exceptionQ.empty() ) {
            // JP When do these exception_ptr get deleted? Is it a leak?
            EDM_Multiview::exceptionQ.pop();
        }
        std::rethrow_exception( exceptionPtr );
    }

    //-----------------------------------------------------------------
    // Rank in-sample (library) forecasts
    //-----------------------------------------------------------------
    // Make pairs of row indices and rho
    std::valarray< double > rho = combos_rho.VectorColumnName( "rho" );
    // vector of indices
    std::valarray< size_t > indices( rho.size() );
    std::iota( begin( indices ), end( indices ), 0 );

    // Ensure that rho is the first of the pair so sort will work
    std::vector< std::pair< double, int > > combo_sort( rho.size() );
    for ( size_t i = 0; i < rho.size(); i++ ) {
        combo_sort[ i ] = std::make_pair( rho[i], indices[i] );
    }

    // sort pairs and reverse for largest rho first
    std::sort   ( combo_sort.begin(), combo_sort.end() );
    std::reverse( combo_sort.begin(), combo_sort.end() );

#ifdef DEBUG_ALL
    std::cout << "Multiview(): combos:\n" << combos_rho << std::endl;
    std::cout << "Ranked combos:\n";
    for ( auto i = 0; i < combo_sort.size(); i++ ) {
        std::cout << "(";
        std::pair< double, int > combo_pair = combo_sort[ i ];
        std::cout << combo_pair.first << ","
                  << combo_pair.second << ") ";
    } std::cout << std::endl;
#endif

    // ---------------------------------------------------------------
    // Perform predictions with the top library multiview embeddings
    // ---------------------------------------------------------------
    // Reset the user specified prediction vector
    param.prediction = prediction;

    // Get top param.MultiviewEnsemble combos
    std::vector< std::pair< double, int > >
        combo_best( combo_sort.begin(),
                    combo_sort.begin() + param.MultiviewEnsemble );
   
#ifdef DEBUG_ALL
    std::cout << "Multiview(): Best combos:\n";
    for ( auto i = 0; i < combo_best.size(); i++ ) {
        std::pair< double, int > combo_pair = combo_best[ i ];
        std::vector< size_t >    this_combo = combos[ combo_pair.second ];
        std::cout << "(" << combo_pair.first << " [";
        for ( auto j = 0; j < this_combo.size(); j++ ) {
            std::cout << this_combo[j] << ",";
        } std::cout << "]) ";
    } std::cout << std::endl;
#endif

    // Create combos_best (vector of column numbers) from combo_best
    std::vector< std::vector< size_t > > combos_best( param.MultiviewEnsemble );
    for ( auto i = 0; i < combo_best.size(); i++ ) {
        std::pair< double, int > combo_pair = combo_best[ i ];
        std::vector< size_t >    this_combo = combos[ combo_pair.second ];
        combos_best[ i ] = this_combo;
    }
    
    // Results Data Frame: E columns (a combo), and rho mae rmse
    DataFrame<double> combos_rho_pred( param.MultiviewEnsemble,
                                       param.E + 3, header.str() );
    
    // Results vector of DataFrame's with prediction results
    // Used to compute the multiview ensemble average prediction
    std::vector< DataFrame< double > >
        combos_rho_prediction( param.MultiviewEnsemble );

    // Build work queue
    EDM_Multiview::WorkQueue workQ_pred( param.MultiviewEnsemble );

    // Insert combos index into work queue
    for ( auto i = 0; i < param.MultiviewEnsemble; i++ ) {
        workQ_pred[ i ] = i;
    }

    // thread container
    std::vector< std::thread > threads_pred;
    for ( unsigned i = 0; i < nThreads; ++i ) {
        threads_pred.push_back( std::thread( EvalComboThread,
                                             param,
                                             workQ_pred,
                                             combos_best,
                                             std::ref( embedding ),
                                             std::ref( targetVec ),
                                             std::ref( combos_rho_pred ),
                                             std::ref( combos_rho_prediction)));
    }
    
    // join threads
    for ( auto &thrd : threads_pred ) {
        thrd.join();
    }
    
    // If thread threw exception, get from queue and rethrow
    if ( not EDM_Multiview::exceptionQ.empty() ) {
        std::lock_guard<std::mutex> lck( EDM_Multiview::q_mtx );

        // Take the first exception in the queue
        std::exception_ptr exceptionPtr = EDM_Multiview::exceptionQ.front();

        // Unroll all other exception from the thread/loops
        while( not EDM_Multiview::exceptionQ.empty() ) {
            // JP When do these exception_ptr get deleted? Is it a leak?
            EDM_Multiview::exceptionQ.pop();
        }
        std::rethrow_exception( exceptionPtr );
    }
    
#ifdef DEBUG_ALL
    for ( auto cpi =  combos_rho_prediction.begin();
               cpi != combos_rho_prediction.end(); ++cpi ) {
        std::cout << *cpi;
    }
    std::cout << combos_rho_pred;
#endif
    
    //----------------------------------------------------------
    // Compute Multiview averaged prediction
    // combos_rho_prediction is a vector of DataFrames with
    // columns [ Observations, Predictions ]
    //----------------------------------------------------------
    // Get copy of Observations
    std::valarray< double >
        Obs = combos_rho_prediction[0].VectorColumnName( "Observations" );

    // Create ensemble average prediction vector
    std::valarray< double > Predictions( 0., Obs.size() );

    // Compute ensemble prediction (this seems clunky...)
    // combos_rho_prediction is a vector of DataFrames with prediction
    // from each combo
    for ( auto i = 0; i < param.MultiviewEnsemble; i++ ) {
        std::valarray< double > prediction_i =
            combos_rho_prediction[ i ].VectorColumnName( "Predictions" );

        // Accumulate prediction values
        for ( auto j = 0; j < Predictions.size(); j++ ) {
            Predictions[ j ] += prediction_i[ j ];
        }
    }
    // Mean of prediction values
    for ( auto i = 0; i < Predictions.size(); i++ ) {
        Predictions[ i ] /= param.MultiviewEnsemble;
    }

    // Error of ensemble prediction
    VectorError ve = ComputeError( Obs, Predictions );

    // Output Prediction DataFrame
    DataFrame< double > Prediction( Predictions.size(), 2,
                                    "Observations  Predictions" );
    // Output time vector
    std::vector< std::string > predTime( param.prediction.size() + param.Tp );
    
    FillTimes( param, data.Time(), std::ref( predTime ) );
    
    Prediction.Time()     = predTime;
    Prediction.TimeName() = data.TimeName();
    Prediction.WriteColumn( 0, Obs  );
    Prediction.WriteColumn( 1, Predictions );

    if ( outputFile.size() ) {
        Prediction.WriteData( param.pathOut, outputFile );
    }

    // Create combos_rho table with column names
    std::vector< std::string > comboTable =
        ComboRhoTable( combos_rho_pred, embedding.ColumnNames() );

    if ( param.verbose ) {
        std::cout << "Multiview(): rho " << ve.rho
                  << "  MAE " << ve.MAE << "  RMSE " << ve.RMSE << std::endl;
        std::cout << std::endl << "Multiview Combinations:" << std::endl;
        for ( auto tableRow : comboTable ) {
            std::cout << tableRow << std::endl;
        } std::cout << std::endl; 
   }

    struct MultiviewValues MV( combos_rho_pred, Prediction, comboTable );
    
    return MV;
}

//----------------------------------------------------------------
// Worker thread
// Output: Write rho to combos_rho DataFrame,
//         Simplex results to combos_prediction
//----------------------------------------------------------------
void EvalComboThread( Parameters                            param,
                      EDM_Multiview::WorkQueue              workQ,
                      std::vector< std::vector< size_t > >  combos,
                      DataFrame< double >                  &embedding,
                      std::valarray< double >              &targetVec,
                      DataFrame< double >                  &combos_rho,
                      std::vector< DataFrame< double > >   &combos_prediction )
{
    // atomic_fetch_add(): Adds val to the contained value and returns
    // the value it had immediately before the operation.
    std::size_t eval_i =
        std::atomic_fetch_add( &EDM_Multiview::eval_i, std::size_t(1) );

    while( eval_i < workQ.size() ) {
        
        // WorkQueue stores combo index in combos
        size_t combo_i = workQ[ eval_i ];
      
        // Get the combo for this thread
        std::vector< size_t > combo = combos[ combo_i ];

        try {
        
        // Local copy with combo column indices (zero-offset)
        std::vector< size_t > combo_cols( combo );

        // Zero offset combo column indices for dataFrame
        for ( auto ci = combo_cols.begin(); ci != combo_cols.end(); ++ci ) {
            *ci = *ci - 1;
        }

#ifdef DEBUG_ALL
        {
            std::lock_guard<std::mutex> lck( EDM_Multiview::mtx );
            std::cout << "EvalComboThread() Thread ["
                      << std::this_thread::get_id() << "] ";
            std::cout << "combo: [";
            for ( auto i = 0; i < combo.size(); i++ ) {
                std::cout << combo[i] << ",";
            } std::cout << "]  rho = ";
        }
#endif

        // Select combo columns from the data
        DataFrame<double> comboData =
            embedding.DataFrameFromColumnIndex(combo_cols);

        // Compute neighbors on comboData
        Neighbors neighbors = FindNeighbors( comboData, param );

        // Pack embedding, target, neighbors for SimplexProjection
        DataEmbedNN embedNN = DataEmbedNN( &embedding, comboData,
                                            targetVec, neighbors );

        // combo prediction
        DataFrame<double> S = SimplexProjection( param, embedNN );

        // Write combo prediction DataFrame
        combos_prediction[ eval_i ] = S;

        // Evaluate combo prediction
        VectorError ve = ComputeError( S.VectorColumnName( "Observations" ),
                                       S.VectorColumnName( "Predictions"  ) );

#ifdef DEBUG_ALL
        {
            std::lock_guard<std::mutex> lck( EDM_Multiview::mtx );
            std::cout << ve.rho << std::endl;
            std::cout << "-------------- embedding -------------------\n";
            std::cout << embedding;
            std::cout << "-------------- comboData -------------------\n";
            std::cout << comboData;
        }
#endif

        // Write combo and rho to the Data Frame
        // E columns (a combo), E column names, rho, MAE, RMSE
        std::valarray< double > combo_row( combo.size() + 3 );
        for ( auto i = 0; i < combo.size(); i++ ) {
            combo_row[ i ] = combo[ i ];
        }
        combo_row[ combo.size()     ] = ve.rho;
        combo_row[ combo.size() + 1 ] = ve.MAE;
        combo_row[ combo.size() + 2 ] = ve.RMSE;
        
        combos_rho.WriteRow( eval_i, combo_row );

        } // try
        catch(...) {
            // push exception pointer onto queue for main thread to catch
            std::lock_guard<std::mutex> lck( EDM_Multiview::q_mtx );
            EDM_Multiview::exceptionQ.push( std::current_exception() );
        }
    
        eval_i = std::atomic_fetch_add(&EDM_Multiview::eval_i, std::size_t(1));
    }
    
    // Reset counter
    std::atomic_store( &EDM_Multiview::eval_i, std::size_t(0) );
}

//----------------------------------------------------------------
// Return combinations C(n,k) as a vector of vectors
//----------------------------------------------------------------
std::vector< std::vector< size_t > > Combination( int n, int k ) {

    std::vector< bool > v(n);
    for ( size_t i = 0; i < n; ++i ) {
        v[i] = ( i >= (n - k) );
    }

    std::vector< std::vector< size_t > > combos;
    
    do {
        std::vector< size_t > this_combo( k );
        size_t j = 0;
        for ( size_t i = 0; i < n; ++i ) {
            if ( v[i] ) {
                this_combo[ j ] = i + 1;
                j++;
            }
        }
        // insert this tuple in the combos vector
        combos.push_back( this_combo );
        
    } while ( std::next_permutation( v.begin(), v.end() ) );

    return combos;
}

//----------------------------------------------------------------
// Return combos_rho_pred DataFrame as a vector of strings
// with column names. 
//----------------------------------------------------------------
std::vector< std::string > ComboRhoTable(
    DataFrame<double>          combos_rho_pred,
    std::vector< std::string > columnNames )
{

    // combos_rho_pred has E + 3 columns: Col_1, ... Col_E, rho, MAE, RMSE
    size_t nCol = combos_rho_pred.NColumns() - 3; // JP Hardcoded silliness!

    if ( nCol > columnNames.size() ) {
        std::stringstream errMsg;
        errMsg << "ComboRhoTable(): Combos_rho has " << nCol
               << " columns, but the data embedding has "
               << columnNames.size() << " elements.";
        throw std::runtime_error( errMsg.str() );
    }

    std::vector< std::string > table;
    
    // Header
    std::stringstream header;
    for ( size_t col = 0; col < nCol; col++ ) {  // column indices
        header << "col_" << col + 1 << ", ";
    }
    for ( size_t col = 0; col < nCol; col++ ) {  // column names
        header << "name_" << col + 1 << ", ";
    }
    header << "rho, MAE, RMSE";
    table.push_back( header.str() );

    // Process each row of combos_rho_pred
    for ( size_t row = 0; row < combos_rho_pred.NRows(); row++ ) {
        std::stringstream rowsstring;
        rowsstring.precision( 4 );
        
        std::valarray< double > rowValues = combos_rho_pred.Row( row );
        
        for ( size_t col = 0; col < nCol; col++ ) {
            rowsstring << std::setw(4) << rowValues[ col ] <<  ", ";
        }
        for ( size_t col = 0; col < nCol; col++ ) {
            size_t col_i = (size_t) rowValues[ col ];
            rowsstring << columnNames[ col_i - 1 ] <<  ", ";
        }

        rowsstring << std::setw(6) << rowValues[ nCol     ] <<  ", "; // rho
        rowsstring << std::setw(6) << rowValues[ nCol + 1 ] <<  ", "; // MAE
        rowsstring << std::setw(6) << rowValues[ nCol + 2 ];          // RMSE
        
        table.push_back( rowsstring.str() );
    }
    
    return table;
}
