
//--------------------------------------------------------------------
// Data input requires columns to specify timeseries columns
// that will be embedded by Embed(), and target for predictions.
// 
// D represents the number of variables to combine for each
// assessment, if not specified, it is the number of columns.
// E is the embedding dimension of each variable. 
// If E = 1, no time delay embedding is done, but the variables
// in the embedding are named X(t-0), Y(t-0)...
// 
// Parameters.Validate() with Method::Simplex sets knn equal to E+1
// if knn not specified, so we need to explicitly set knn to D + 1.
// 
// multiviewEnsemble is the number of top-ranked D-dimensional
// predictions to "average" for the final prediction. Corresponds
// to parameter k in Ye & Sugihara with default k = sqrt(m) where
// m is the number of combinations C(n,D) available from the n = D * E
// columns taken D at-a-time. 
//
// Ye H., and G. Sugihara, 2016. Information leverage in
// interconnected ecosystems: Overcoming the curse of dimensionality.
// Science 353:922–925.
//--------------------------------------------------------------------
//--------------------------------------------------------------------
// NOTE: Multiview evaluates the top projections using in-sample
//       library predictions.  It can be shown that highly accurate
//       in-sample predictions can be made from arbitrary non-
//       constant, non-oscillatory vectors. Therefore, some attention
//       may be warranted to filter prospective embedding vectors.
//       The multiviewTrainLib flag disables this default behavior
//       so that the top k evalutions are done using the specified
//       lib and pred. 
//--------------------------------------------------------------------

#include "Multiview.h"

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

void EvalComboThread( MultiviewClass                      & MV,
                      EDM_Multiview::WorkQueue              workQ,
                      std::vector< std::vector< size_t > >& combos,
                      DataFrame< double >                 & combosRho,
                      std::vector< DataFrame< double > >  & comboPrediction );

std::vector< std::string > ComboRhoTable( DataFrame< double >        combosRho,
                                          std::vector< std::string > colNames );

//----------------------------------------------------------------
// Constructor
// data & parameters initialise EDM::SimplexClass parent, and, 
// both mapping objects to the same, initial parameters.
//----------------------------------------------------------------
MultiviewClass::MultiviewClass (
    DataFrame< double > & data, 
    Parameters          & parameters ) :
    SimplexClass{ data, parameters },  // base class initialise
    predictOutputFileIn( parameters.predictOutputFile )
{}

//----------------------------------------------------------------
// Project : Polymorphic implementation
//----------------------------------------------------------------
void MultiviewClass::Project( unsigned nThreads ) {

    CheckParameters();

    // Set embedded false so E-dimensional embedding is computed
    parameters.embedded = false;

    PrepareEmbedding(); // EmbedData(). target, data, lib/pred adjust

    SetupParameters();  // Requires valid embedding

    // Set embedded true for subset columns in Simplex()
    parameters.embedded = true;

    Multiview( nThreads ); // Simplex() and output
}

//----------------------------------------------------------------
// Multiview algorithm
//----------------------------------------------------------------
void MultiviewClass::Multiview ( unsigned nThreads ) {
    // Create column names for the results DataFrame
    // One row for each combo: D columns (a combo), rho, MAE, RMSE
    // JP: NOTE DataFrame is based on valarray, so we can't store non
    //     numeric values (column names) in the results DataFrame.
    std::stringstream header;
    for ( auto i = 1; i <= parameters.multiviewD; i++ ) {
        header << "Col_" << i << " ";
    }
    header << "rho MAE RMSE";

    // Combinations of possible embedding variables, D at-a-time
    // Note that these combinations are not zero-offset, i.e.
    // Combination( 3, 2 ) = [(1, 2), (1, 3), (2, 3)]
    // These correspond to column indices +1
    std::vector< std::vector< size_t > > combos =
        Combination( embedding.NColumns(), parameters.multiviewD );

#ifdef DEBUG_ALL
    std::cout << "Multiview(): " << combos.size() << " combos:\n";
    for ( size_t i = 0; i < combos.size(); i++ ) {
        std::vector< size_t > combo_i = combos[i];
        std::cout << "[";
        for ( size_t j = 0; j < combo_i.size(); j++ ) {
            std::cout << combo_i[j] << ",";
        }
        std::cout << "] ";
    } std::cout << std::endl;
#endif

    // Establish number of ensembles if not specified
    if ( not parameters.multiviewEnsemble ) {
        // Ye & Sugihara suggest sqrt( m ) as the number of embeddings to avg
        parameters.multiviewEnsemble = std::max(2,(int)std::sqrt(combos.size()));

        std::stringstream msg;
        msg << "Multiview() Set view sample size to "
            << parameters.multiviewEnsemble << std::endl;
        std::cout << msg.str();
    }

    // Validate number of combinations
    if ( parameters.multiviewEnsemble > (int) combos.size() ) {
        std::stringstream msg;
        msg << "WARNING: Multiview(): multiview ensembles "
            << parameters.multiviewEnsemble
            << " exceeds the number of available combinations: "
            << combos.size() << ".   Set to " << combos.size() << std::endl;
        std::cout << msg.str();

        parameters.multiviewEnsemble = combos.size();
    }

    // Results Data Frame: D columns (a combo), rho, mae, rmse
    DataFrame< double > combosRho( combos.size(),
                                   parameters.multiviewD + 3, header.str() );

    // Results vector of DataFrame's with prediction results
    std::vector< DataFrame< double > > combosPrediction( combos.size() );

    // Build work queue
    EDM_Multiview::WorkQueue workQ( combos.size() );

    // Insert combos index into work queue
    for ( size_t i = 0; i < combos.size(); i++ ) {
        workQ[ i ] = i;
    }

    unsigned maxThreads = std::thread::hardware_concurrency();
    if ( maxThreads < nThreads ) { nThreads = maxThreads; }

    //---------------------------------------------------------------
    // Evaluate variable combinations.
    //---------------------------------------------------------------
    // thread container
    std::vector< std::thread > threads;
    for ( unsigned i = 0; i < nThreads; ++i ) {
        threads.push_back( std::thread( EvalComboThread,
                                        std::ref( *this ),
                                        workQ,
                                        std::ref( combos ),
                                        std::ref( combosRho ),
                                        std::ref( combosPrediction ) ) );
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
            EDM_Multiview::exceptionQ.pop();
        }
        std::rethrow_exception( exceptionPtr );
    }

    //-----------------------------------------------------------------
    // Rank forecasts. If trainLib true these are in-sample (library)
    //-----------------------------------------------------------------
    // Make pairs of row indices and rho
    std::valarray< double > rho = combosRho.VectorColumnName( "rho" );
    // vector of indices
    std::valarray< size_t > indices( rho.size() );
    std::iota( begin( indices ), end( indices ), 0 );

    // Ensure that rho is the first of the pair so sort will work
    std::vector< std::pair< double, int > > comboSort( rho.size() );
    for ( size_t i = 0; i < rho.size(); i++ ) {
        comboSort[ i ] = std::make_pair( rho[i], indices[i] );
    }

    // sort pairs and reverse for largest rho first
    std::sort   ( comboSort.begin(), comboSort.end() );
    std::reverse( comboSort.begin(), comboSort.end() );

#ifdef DEBUG_ALL
    std::cout << "Multiview(): combos:\n" << combosRho << std::endl;
    std::cout << "Ranked combos:\n";
    for ( size_t i = 0; i < comboSort.size(); i++ ) {
        std::cout << "(";
        std::pair< double, int > comboPair = comboSort[ i ];
        std::cout << comboPair.first << ","
                  << comboPair.second << ") ";
    } std::cout << std::endl;
#endif

    // ---------------------------------------------------------------
    // Perform predictions with the top multiview embeddings
    // ---------------------------------------------------------------
    if ( parameters.multiviewTrainLib ) {
        // Reset the user specified prediction vector
        parameters.prediction = predictionIn;
    }

    // Get top param.MultiviewEnsemble combos
    size_t nEnsemble = std::min( (int) comboSort.size(),
                                 parameters.multiviewEnsemble );

    std::vector< std::pair< double, int > >
        comboBest( comboSort.begin(), comboSort.begin() + nEnsemble );

#ifdef DEBUG_ALL
    std::cout << "Multiview(): Best combos:\n";
    for ( size_t i = 0; i < comboBest.size(); i++ ) {
        std::pair< double, int > comboPair = comboBest[ i ];
        std::vector< size_t >    thisCombo = combos[ comboPair.second ];
        std::cout << "(" << comboPair.first << " [";
        for ( size_t j = 0; j < thisCombo.size(); j++ ) {
            std::cout << thisCombo[j] << ",";
        } std::cout << "]) ";
    } std::cout << std::endl;
#endif

    // Create combosBest (vector of column numbers) from comboBest
    std::vector< std::vector< size_t > >
        combosBest( parameters.multiviewEnsemble );

    for ( size_t i = 0; i < comboBest.size(); i++ ) {
        std::pair< double, int > comboPair = comboBest[ i ];
        std::vector< size_t >    thisCombo = combos[ comboPair.second ];
        combosBest[ i ] = thisCombo;
    }

    // Results Data Frame: D columns (a combo), and rho mae rmse
    DataFrame<double> combosRhoPred( parameters.multiviewEnsemble,
                                     parameters.multiviewD + 3, header.str() );

    // Results vector of DataFrame's with prediction results
    // Used to compute the multiview ensemble average prediction
    std::vector< DataFrame< double > >
        combosRhoPrediction( parameters.multiviewEnsemble );

    //--------------------------------------------------------------------
    // If trainLib false, no need to compute these projections
    //--------------------------------------------------------------------
    if ( parameters.multiviewTrainLib ) {
        // Build work queue
        EDM_Multiview::WorkQueue workQPred( parameters.multiviewEnsemble );

        // Insert combos index into work queue
        for ( auto i = 0; i < parameters.multiviewEnsemble; i++ ) {
            workQPred[ i ] = i;
        }

        // thread container
        std::vector< std::thread > threadsPred;
        for ( unsigned i = 0; i < nThreads; ++i ) {
            threadsPred.push_back(
                std::thread( EvalComboThread,
                             std::ref( *this ),
                             workQPred,
                             std::ref( combosBest ),
                             std::ref( combosRhoPred ),
                             std::ref( combosRhoPrediction) ) );
        }

        // join threads
        for ( auto &thrd : threadsPred ) {
            thrd.join();
        }

        // If thread threw exception, get from queue and rethrow
        if ( not EDM_Multiview::exceptionQ.empty() ) {
            std::lock_guard<std::mutex> lck( EDM_Multiview::q_mtx );
            
            // Take the first exception in the queue
            std::exception_ptr exceptionPtr = EDM_Multiview::exceptionQ.front();
            
            // Unroll all other exception from the thread/loops
            while( not EDM_Multiview::exceptionQ.empty() ) {
                EDM_Multiview::exceptionQ.pop();
            }
            std::rethrow_exception( exceptionPtr );
        }
#ifdef DEBUG_ALL
    for ( auto cpi =  combosRhoPrediction.begin();
               cpi != combosRhoPrediction.end(); ++cpi ) {
        std::cout << *cpi;
    }
    std::cout << combosRhoPred;
#endif
    } // if ( trainLib )
    //-------------------------------------------------------------------
    // else: trainLib = false
    //   Projections were initally made with lib != pred
    //-------------------------------------------------------------------
    else {
        // Insert top prediction results into combos_rho_pred
        // Insert top predictions into combos_rho_prediction
        int row;
        for ( int row_i = 0; row_i < parameters.multiviewEnsemble; row_i++ ) {
            row = comboSort[ row_i ].second; // row index of best rho
            combosRhoPred.WriteRow( row_i, combosRho.Row( row ) );
            combosRhoPrediction[ row_i ] = combosPrediction[ row ];
        }
    }

    //----------------------------------------------------------
    // Compute Multiview averaged prediction
    // combos_rho_prediction is a vector of DataFrames with
    // columns [ Observations, Predictions ]
    //----------------------------------------------------------
    // Get copy of Observations
    std::valarray< double >
        Obs = combosRhoPrediction[0].VectorColumnName( "Observations" );

    // Create ensemble average prediction vector
    std::valarray< double > Predictions( 0., Obs.size() );

    // Compute ensemble prediction (this seems clunky...)
    // combosRhoPrediction is a vector of DataFrames with prediction
    // from each combo
    for ( auto i = 0; i < parameters.multiviewEnsemble; i++ ) {
        std::valarray< double > prediction_i =
            combosRhoPrediction[ i ].VectorColumnName( "Predictions" );

        // Accumulate prediction values
        for ( size_t j = 0; j < Predictions.size(); j++ ) {
                Predictions[ j ] += prediction_i[ j ];
        }
    }
    // Mean of prediction values
    for ( size_t i = 0; i < Predictions.size(); i++ ) {
        Predictions[ i ] /= parameters.multiviewEnsemble;
    }

    // Allocate output Prediction DataFrame
    DataFrame< double > Prediction ( Predictions.size(), 2,
                                     "Observations  Predictions" );
    // Output time vector
    std::vector< std::string > predTime( parameters.prediction.size() +
                                         abs( parameters.Tp ) );

    FillTimes( std::ref( predTime ) );

    Prediction.Time()     = predTime;
    Prediction.TimeName() = data.TimeName();
    Prediction.WriteColumn( 0, Obs  );
    Prediction.WriteColumn( 1, Predictions );

    if ( predictOutputFileIn.size() ) {
        Prediction.WriteData( parameters.pathOut, predictOutputFileIn );
    }

    // Create combos_rho table with column names
    std::vector< std::string > comboTable =
        ComboRhoTable( combosRhoPred, embedding.ColumnNames() );

    if ( parameters.verbose ) {
        // Error of ensemble prediction
        VectorError ve = ComputeError( Obs, Predictions );

        std::cout << "Multiview(): rho " << ve.rho
                  << "  MAE " << ve.MAE << "  RMSE " << ve.RMSE << std::endl;
        std::cout << std::endl << "Multiview Combinations:" << std::endl;

        for ( auto tableRow : comboTable ) {
            std::cout << tableRow << std::endl;
        } std::cout << std::endl; 
    }

    // Allocate output 
    MultiviewValues MVout;
    MVout.ComboRho      = combosRhoPred;
    MVout.Predictions   = Prediction;
    MVout.ComboRhoTable = comboTable;

    MVvalues = MVout; // Assign to Multiview Object
}

//----------------------------------------------------------------
// Worker thread
// Output: Write rho to combosRho DataFrame,
//         Simplex results to combosPrediction
//----------------------------------------------------------------
void EvalComboThread( MultiviewClass                       & MV,
                      EDM_Multiview::WorkQueue               workQ,
                      std::vector< std::vector< size_t > > & combos,
                      DataFrame< double >                  & combosRho,
                      std::vector< DataFrame< double > >   & combosPrediction )
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
        std::vector< size_t > comboCols( combo );

        // Zero offset combo column indices for dataFrame
        for ( auto ci = comboCols.begin(); ci != comboCols.end(); ++ci ) {
            *ci = *ci - 1;
        }

        // Embedded column names are column names + (t-0), (t-1),...
        std::vector< std::string > comboColumnNames;
        for ( size_t i = 0; i < comboCols.size(); i++ ) {
            comboColumnNames.push_back(
                MV.embedding.ColumnNames()[ comboCols[ i ] ] );
        }

#ifdef DEBUG_ALL
        {
            std::lock_guard<std::mutex> lck( EDM_Multiview::mtx );
            std::cout << "EvalComboThread() Thread ["
                      << std::this_thread::get_id() << "] ";
            std::cout << "combo: [";
            for ( size_t i = 0; i < combo.size(); i++ ) {
                std::cout << combo[i] << ",";
            } std::cout << "]  rho = ";
        }
#endif

        // Target has been embedded too... add "(t-0)"
        std::stringstream threadTarget;
        threadTarget << MV.parameters.targetName << "(t-0)";

        // Find target column in embedding
        size_t targetColumn =
            MV.embedding.ColumnNameToIndex()[ threadTarget.str() ];

        // Add target column to comboCols for DataFrame subset
        comboCols.push_back( targetColumn );

        // Select combo columns from the data : columns and target
        DataFrame< double > comboData =
            MV.embedding.DataFrameFromColumnIndex( comboCols );

        // Must use thread local Parameters since columns have been embedded
        // and are different than base Parameters. x_t -> x_t(t-0)...
        Parameters threadParameters( MV.parameters );

        // Replace base copied columnNames and targetName with embedded ones
        threadParameters.columnNames = comboColumnNames;
        threadParameters.targetName  = threadTarget.str();

        // Simplex
        SimplexClass S( std::ref( comboData ), std::ref( threadParameters ) );

        // This is an embedded = true, E = D columns prediction
        S.Project();

        // Write combo prediction DataFrame
        combosPrediction[ eval_i ] = S.projection;

        // Evaluate combo prediction
        VectorError ve =
            ComputeError( S.projection.VectorColumnName( "Observations" ),
                          S.projection.VectorColumnName( "Predictions"  ) );
#ifdef DEBUG_ALL
        {
            std::lock_guard<std::mutex> lck( EDM_Multiview::mtx );
            std::cout << ve.rho << std::endl;
            std::cout << "-------------- embedding -------------------\n";
            std::cout << S.embedding;
            std::cout << "-------------- comboData -------------------\n";
            std::cout << comboData;
        }
#endif

        // Write combo and rho to the Data Frame
        // D columns (a combo), rho, MAE, RMSE
        std::valarray< double > comboRow( combo.size() + 3 );
        for ( size_t i = 0; i < combo.size(); i++ ) {
            comboRow[ i ] = combo[ i ];
        }
        comboRow[ combo.size()     ] = ve.rho;
        comboRow[ combo.size() + 1 ] = ve.MAE;
        comboRow[ combo.size() + 2 ] = ve.RMSE;

        combosRho.WriteRow( eval_i, comboRow );

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

//-----------------------------------------------------------------
// Populate EDM::MultiviewClass Parameters objects
//-----------------------------------------------------------------
void MultiviewClass::SetupParameters() {
    // Clear parameters.predictOutputFile so Simplex() does not write 
    // predictOutputFile copied to member predictOutputFileIn in constructor
    parameters.predictOutputFile = "";

    // Establish the state-space dimension D
    // default to the number of input columns (not embedding columns)
    if ( parameters.multiviewD == 0 ) {
        parameters.multiviewD = parameters.columnNames.size();
    }
    if ( parameters.multiviewD > (int) embedding.NColumns() ) {
        std::stringstream msg;
        msg << "WARNING: Multiview(): D = " << parameters.multiviewD
            << " exceeds the number of columns in the embedding: "
            << embedding.NColumns() << ".  D set to "
            << embedding.NColumns() << std::endl;
        std::cout << msg.str();
        
        parameters.multiviewD = (int) embedding.NColumns();
    }

    // Save a copy of the specified prediction observation rows.
    predictionIn = parameters.prediction;

    if ( parameters.multiviewTrainLib ) {
        // Override parameters.prediction for in-sample forecast skill evaluation
        parameters.prediction = parameters.library;
    }

    // This is not a good implementation...
    // Replace parameters.E with the number of dimensions, recall embbeded = true
    parameters.E = parameters.multiviewD;
}

//-----------------------------------------------------------------
// 
//-----------------------------------------------------------------
void MultiviewClass::CheckParameters() {
    // Require at least E = 1
    if ( parameters.E < 1 ) {
        std::stringstream errMsg;
        errMsg << " Multiview(): E = " << parameters.E << " is invalid.\n" ;
        throw std::runtime_error( errMsg.str() );
    }

    if ( not parameters.columnNames.size() ) {
        throw std::runtime_error( "Multiview() requires column names." );
    }
    if ( not parameters.targetName.size() ) {
        throw std::runtime_error( "Multiview() requires target name." );
    }
    // Ensure that params are validated so columnNames are populated
    if ( not parameters.validated ) {
        throw std::runtime_error( "Multiview() params not validated." );        
    }

    // Validate that columns & target are in data
    for ( auto colName : parameters.columnNames ) {
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
                    data.ColumnNames().end(), parameters.targetName );
    if ( ti == data.ColumnNames().end() ) {
        std::stringstream errMsg;
        errMsg << "Multiview(): Failed to find target "
               << parameters.targetName << " in dataFrame with columns: [ ";
        for ( auto col : data.ColumnNames() ) {
            errMsg << col << " ";
        } errMsg << " ]\n";
        throw std::runtime_error( errMsg.str() );
    }

    // Validate data rows against lib and pred indices
    CheckDataRows( "Multiview()" );
}

//----------------------------------------------------------------
// Return combinations C(n,k) as a vector of vectors
//----------------------------------------------------------------
std::vector< std::vector< size_t > > Combination( int n, int k ) {

    std::vector< bool > v(n);
    for ( int i = 0; i < n; ++i ) {
        v[i] = ( i >= (n - k) );
    }

    std::vector< std::vector< size_t > > combos;

    do {
        std::vector< size_t > this_combo( k );
        size_t j = 0;
        for ( int i = 0; i < n; ++i ) {
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
