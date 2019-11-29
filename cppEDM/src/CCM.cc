
#include <cstdlib>
#include <random>
#include <unordered_set>
#include <chrono>
#include <queue>

#ifdef CCM_THREADED // Defined in makefile
// Two explicit CrossMap() threads are invoked. One for forward mapping, 
// one for inverse mapping.  The call signature of CrossMap() is
// dependent on which path is used.  This should probably be unified 
// to use the same signature.
#include <thread>
#endif

#include "Common.h"
#include "Embed.h"
#include "AuxFunc.h"

namespace EDM_CCM {
    std::mutex mtx;
    std::mutex q_mtx;
    std::queue< std::exception_ptr > exceptionQ;
}

//----------------------------------------------------------------
// forward declarations
//----------------------------------------------------------------
#ifdef CCM_THREADED
void CrossMap(       Parameters           param,
                     DataFrame< double >  dataFrameIn,
               const DataFrame< double > &LibStats );
#else
DataFrame< double > CrossMap( Parameters          param,
                              DataFrame< double > dataFrameIn );
#endif

DataFrame< double > CCMDistances( const DataFrame< double > &dataBlock,
                                        Parameters           param );

Neighbors CCMNeighbors( const DataFrame< double >   &Distances,
                              std::vector< size_t >  lib_i,
                              Parameters             param );

DataFrame<double> SimplexProjection( Parameters  param,
                                     DataEmbedNN embedNN,
                                     bool        checkDataRows );

//----------------------------------------------------------------
// API Overload 1: Explicit data file path/name
//   Implemented as a wrapper to API Overload 2:
//   which is a wrapper for CrossMap()
//----------------------------------------------------------------
DataFrame <double > CCM( std::string pathIn,
                         std::string dataFile,
                         std::string pathOut,
                         std::string predictFile,
                         int         E,
                         int         Tp,
                         int         knn,
                         int         tau,
                         std::string columns,
                         std::string target,
                         std::string libSizes_str,
                         int         sample,
                         bool        random,
                         bool        replacement,
                         unsigned    seed,
                         bool        verbose )
{
    //----------------------------------------------------------
    // Load data to dataFrameIn
    //----------------------------------------------------------
    DataFrame< double > dataFrameIn( pathIn, dataFile );

    DataFrame <double > PredictLibRho = CCM( dataFrameIn,
                                             pathOut,
                                             predictFile,
                                             E,
                                             Tp,
                                             knn,
                                             tau,
                                             columns,
                                             target,
                                             libSizes_str,
                                             sample,
                                             random,
                                             replacement,
                                             seed,
                                             verbose );
    return PredictLibRho;
}

//----------------------------------------------------------------
// API Overload 2: DataFrame passed in
//   Implemented a wrapper for CrossMap()
//----------------------------------------------------------------
DataFrame <double > CCM( DataFrame< double > dataFrameIn,
                         std::string         pathOut,
                         std::string         predictFile,
                         int                 E,
                         int                 Tp,
                         int                 knn,
                         int                 tau,
                         std::string         columns,
                         std::string         target,
                         std::string         libSizes_str,
                         int                 sample,
                         bool                random,
                         bool                replacement,
                         unsigned            seed,
                         bool                verbose )
{
    if ( not columns.size() ) {
        throw std::runtime_error("CCM() must specify the column to embed.");
    }
    if ( not target.size() ) {
        throw std::runtime_error("CCM() must specify the target.");
    }
    if ( not libSizes_str.size() ) {
        throw std::runtime_error("CCM() must specify library sizes.");
    }

    Parameters param = Parameters( Method::CCM,
                                   "",           // pathIn
                                   "",           // dataFile
                                   pathOut,      // 
                                   predictFile,  // 
                                   "",           // lib_str
                                   "",           // pred_str
                                   E,            // 
                                   Tp,           // 
                                   knn,          // 
                                   tau,          // 
                                   0,            // theta
                                   0,            // exclusionRadius
                                   columns,      // 
                                   target,       // 
                                   false,        // embedded
                                   false,        // const_predict
                                   verbose,      // 
                                   "",           // SmapFile
                                   "",           // blockFile
                                   "",           // derivatives_str
                                   0,            // svdSig
                                   0,            // tikhonov
                                   0,            // elasticNet
                                   0,            // multi
                                   libSizes_str, // 
                                   sample,       // 
                                   random,       // 
                                   replacement,  // 
                                   seed );       // 

    if ( param.columnNames.size() > 1 ) {
        std::cout << "WARNING: CCM() Only the first column will be mapped.\n";
    }

    // Create Parameters object that switches column[0] and target
    // for the inverse mapping
    Parameters  inverseParam( param ); // copy constructor
    std::string newTarget( param.columns_str );
    inverseParam.columns_str = param.target_str;
    inverseParam.target_str  = newTarget;
    
    // Validate converts column_str & target_str to columnNames, targetName
    inverseParam.Validate();
    
#ifdef DEBUG_ALL
    std::cout << "CCM() params:\n";
    std::cout << param;
    std::cout << "CCM() inverseParams:\n";
    std::cout << inverseParam;
#endif

#ifdef CCM_THREADED
    DataFrame<double> col_to_target( param.librarySizes.size(), 4,
                                     "LibSize rho RMSE MAE" );
    
    DataFrame<double> target_to_col( param.librarySizes.size(), 4,
                                     "LibSize rho RMSE MAE" );

    std::thread CrossMapColTarget( CrossMap, param, dataFrameIn,
                                   std::ref( col_to_target ) );
    
    std::thread CrossMapTargetCol( CrossMap, inverseParam, dataFrameIn,
                                   std::ref( target_to_col ) );

    CrossMapColTarget.join();
    CrossMapTargetCol.join();

    // If thread threw exception, get from queue and rethrow
    if ( not EDM_CCM::exceptionQ.empty() ) {
        std::lock_guard<std::mutex> lck( EDM_CCM::q_mtx );

        // Take the first exception in the queue
        std::exception_ptr exceptionPtr = EDM_CCM::exceptionQ.front();

        // Unroll all other exception from the thread/loops
        while( not EDM_CCM::exceptionQ.empty() ) {
            // JP When do these exception_ptr get deleted? Is it a leak?
            EDM_CCM::exceptionQ.pop();
        }
        std::rethrow_exception( exceptionPtr );
    }
    
#else
    DataFrame< double > col_to_target = CrossMap( param, dataFrameIn );

    DataFrame< double > target_to_col = CrossMap( inverseParam, dataFrameIn );
#endif
    
    //-----------------------------------------------------------------
    // Output
    //-----------------------------------------------------------------
    // Create column names of output DataFrame
    std::stringstream libRhoNames;
    libRhoNames << "LibSize "
                << param.columnNames[0] << ":" << param.targetName << " "
                << param.targetName     << ":" << param.columnNames[0];

    // Output DataFrame
    DataFrame<double> PredictLibRho( param.librarySizes.size(), 3,
                                     libRhoNames.str() );

    PredictLibRho.WriteColumn( 0, col_to_target.Column( 0 ) );
    PredictLibRho.WriteColumn( 1, col_to_target.Column( 1 ) );
    PredictLibRho.WriteColumn( 2, target_to_col.Column( 1 ) );

    if ( param.predictOutputFile.size() ) {
        // Write to disk
        PredictLibRho.WriteData( param.pathOut, param.predictOutputFile );
    }
    
    return PredictLibRho;
}

//----------------------------------------------------------------
// CrossMap()
// Worker function for CCM.
// Return DataFrame of rho, RMSE, MAE values for param.librarySizes
//
// NOTE: This is a bit of a kludge at the moment... would be nice
//       to pass in a reference to dataFrameIn, however... Embed()
//       returns a dataBlock with (E-1)*tau fewer rows since
//       partial data vectors are removed. To maintain proper
//       alignment with the target in the dataFrameIn,
//       dataFrameIn.DeletePartialDataRows() is called.  If we use
//       a reference to dataFrameIn, then the second thread will
//       receive a reference to dataFrameIn that has rows deleted
//       and Embed() will then not be correct.  So at the moment
//       we'll pass a copy of the dataFrameIn to each thread. 
//----------------------------------------------------------------
#ifdef CCM_THREADED
void CrossMap(       Parameters           paramCCM,
                     DataFrame< double >  dataFrameIn,
               const DataFrame< double > &LibStatsIn ) {
    
DataFrame< double > &LibStats =
    const_cast< DataFrame< double > & >( LibStatsIn );
#else
DataFrame< double > CrossMap( Parameters          paramCCM,
                              DataFrame< double > dataFrameIn ) {
// Output DataFrame
DataFrame<double> LibStats( paramCCM.librarySizes.size(), 4,
                            "LibSize rho RMSE MAE" );
#endif
    
    if ( paramCCM.verbose ) {
        std::stringstream msg;
        msg << "CrossMap(): Simplex cross mapping from "
            << paramCCM.columnNames[0]
            << " to " << paramCCM.targetName << "  E=" << paramCCM.E
            << "  knn=" << paramCCM.knn << "  Library range: ["
            << paramCCM.libSizes_str << "] ";
        for ( size_t i = 0; i < paramCCM.librarySizes.size(); i++ ) {
            msg << paramCCM.librarySizes[ i ] << " ";
        } msg << std::endl << std::endl;
        std::cout << msg.str();
    }

    try {
    //------------------------------------------------------------
    // Generate embedding on data to be cross mapped (-c column)
    // dataBlock will have tau * (E-1) fewer rows than dataFrameIn
    // JP: Should this be allocated on the heap?
    //------------------------------------------------------------
    DataFrame<double> dataBlock = Embed( dataFrameIn,
                                         paramCCM.E,
                                         paramCCM.tau,
                                         paramCCM.columnNames[0],
                                         paramCCM.verbose );
    
    size_t N_row = dataBlock.NRows();

    // NOTE: No need to adjust param.library and param.prediction indices
    //       [param.DeleteLibPred( shift );] since pred will be created
    //       below based on N_row of dataBlock.
    
    //--------------------------------------------------------------
    // Remove dataFrameIn rows to match embedded dataBlock with
    // partial data rows ignored: CrossMap() -> Embed() -> MakeBlock()
    // This removal of partial data rows is also done in EmbedNN()
    //--------------------------------------------------------------
    // If we support negative tau, this will change
    // For now, assume only positive tau is allowed
    size_t shift = std::max( 0, paramCCM.tau * (paramCCM.E - 1) );
    {
        std::lock_guard<std::mutex> lck( EDM_CCM::mtx );
        if ( not dataFrameIn.PartialDataRowsDeleted() ) {
            // Not thread safe.
            dataFrameIn.DeletePartialDataRows( shift );
        }
    }
    
#ifdef DEBUG_ALL
    {
    std::lock_guard<std::mutex> lck( EDM_CCM::mtx );
    std::cout << ">>>> CrossMap() dataFrameIn-----------------------\n";
    std::cout << dataFrameIn;
    std::cout << "<<<< dataFrameIn----------------------------------\n";
    std::cout << ">>>> dataBlock------------------------------------\n";
    std::cout << dataBlock;
    std::cout << "<<<< dataBlock------------------------------------\n";
    }
#endif
    
    //-----------------------------------------------------------------
    // Create Parameters for SimplexProjection
    // Add library and prediction indices for the entire library
    //-----------------------------------------------------------------
    std::stringstream ss;
    ss << "1 " << N_row;
    paramCCM.lib_str  = ss.str();
    paramCCM.pred_str = ss.str();
    // Validate converts lib_str, pred_str to library & prediction vectors
    paramCCM.Validate();

    //-----------------------------------------------------------------
    // Set number of samples
    //-----------------------------------------------------------------
    size_t maxSamples;
    if ( paramCCM.randomLib ) {
        // Random samples from library
        maxSamples = paramCCM.subSamples;
    }
    else {
        // Contiguous samples up to the size of the library
        maxSamples = 1;
    }

    //-----------------------------------------------------------------
    // Create random number generator: DefaultRandEngine
    //-----------------------------------------------------------------
    if ( paramCCM.randomLib ) {
        if ( paramCCM.seed == 0 ) {
            // Select a random seed
            typedef std::chrono::high_resolution_clock CCMclock;
            CCMclock::time_point beginning = CCMclock::now();
            CCMclock::duration   duration  = CCMclock::now() - beginning;
            paramCCM.seed = duration.count();
        }
    }
    std::default_random_engine DefaultRandomEngine( paramCCM.seed );

    //-----------------------------------------------------------------
    // Distance for all possible pred : lib E-dimensional vector pairs
    // Distances is a square Matrix of all row to to row distances
    //-----------------------------------------------------------------
    DataFrame< double > Distances = CCMDistances( std::ref( dataBlock ),
                                                  paramCCM );

#ifdef DEBUG_ALL
    {
    std::lock_guard<std::mutex> lck( EDM_CCM::mtx );
    std::cout << "CrossMap() " << paramCCM.columnNames[0] << " to "
              << paramCCM.targetName << " Distances: " << Distances.NRows()
              << " x " << Distances.NColumns() << std::endl;
    }
#endif
    
    //----------------------------------------------------------
    // Predictions
    //----------------------------------------------------------
    // Loop for library sizes
    for ( size_t lib_size_i = 0;
                 lib_size_i < paramCCM.librarySizes.size(); lib_size_i++ ) {

        size_t lib_size = paramCCM.librarySizes[ lib_size_i ];

        // Create random RNG sampler for this lib_size
        std::uniform_int_distribution<size_t> distribution( 0, N_row - 1 );
        
#ifdef DEBUG_ALL
    {
    std::lock_guard<std::mutex> lck( EDM_CCM::mtx );
        std::cout << "lib_size: " << lib_size
                  << " ------------------------------------------\n";
    }
#endif
            
        std::valarray< double > rho ( maxSamples );
        std::valarray< double > RMSE( maxSamples );
        std::valarray< double > MAE ( maxSamples );

        // Loop for subsamples
        for ( size_t n = 0; n < maxSamples; n++ ) {
            
            // Vector of row indices to include in this lib_size evaluation
            std::vector< size_t > lib_i( lib_size );

            if ( paramCCM.randomLib ) {
                // Uniform random sample of rows
                
                if ( paramCCM.replacement ) {
                    // With replacement
                    for ( size_t i = 0; i < lib_size; i++ ) {
                        lib_i[ i ] = distribution( DefaultRandomEngine );
                    }
                }
                else {
                    // Without replacement lib_size elements from [0, N_row-1]
                    // Robert W. Floyd's algorithm
                    // NOTE: c++17 has the sample() function in <algorithm>
                    if ( lib_size >= N_row ) {
                        std::stringstream errMsg;
                        errMsg << "CrossMap(): lib_size=" << lib_size
                               << " must be less than N_row=" << N_row
                               << " for random sample without replacement.";
                        throw std::runtime_error( errMsg.str() );
                    }
                    
                    // unordered set to store samples
                    std::unordered_set<size_t> samples;
                    
                    // Sample and insert values into samples
                    for ( size_t r = N_row - lib_size; r < N_row; r++ ) {
                        size_t v = distribution( DefaultRandomEngine );
                        if ( not samples.insert( v ).second ) {
                            samples.insert( r );
                        }
                    }
                    
                    // Copy samples into result
                    std::vector<size_t> result(samples.begin(), samples.end());
                    
                    // Shuffle result
                    std::shuffle( result.begin(), result.end(),
                                  DefaultRandomEngine  );

                    lib_i = result; // Copy result to lib_i
                }
            }
            else {
                // Not random samples, contiguous samples increasing size
                if ( lib_size >= N_row ) {
                    // library size exceeded, back down
                    lib_i.resize( N_row );
                    std::iota( lib_i.begin(), lib_i.end(), 0 );
                    lib_size = N_row;
                    
                    if ( paramCCM.verbose ) {
                        std::stringstream msg;
                        msg << "CCM(): Sequential library samples,"
                            << " max lib_size is " << N_row
                            << ", lib_size has been limited.\n";
                        std::cout << msg.str();
                    }
                }
                else {
                    // Contiguous blocks up to N_rows = maxSamples
                    if ( n + lib_size < N_row ) {
                        std::iota( lib_i.begin(), lib_i.end(), n );
                    }
                    else {
                        // n + lib_size > N_row, wrap around to data origin
                        std::vector< size_t > lib_start( N_row - n );
                        std::iota( lib_start.begin(), lib_start.end(), n );
                        
                        size_t max_i = std::min( lib_size-(N_row - n), N_row );
                        std::vector< size_t > lib_wrap( max_i );
                        std::iota( lib_wrap.begin(), lib_wrap.end(), 0 );

                        // Build new lib_i
                        lib_i = std::vector< size_t > ( lib_start );
                        lib_i.insert( lib_i.end(),
                                      lib_wrap.begin(),
                                      lib_wrap.end() );
                        lib_size = lib_i.size();
                    }
                }
            }
            
#ifdef DEBUG_ALL
            {
            std::lock_guard<std::mutex> lck( EDM_CCM::mtx );
            std::cout << "lib_i: (" << lib_i.size() << ") ";
            for ( size_t i = 0; i < lib_i.size(); i++ ) {
                std::cout << lib_i[i] << " ";
            } std::cout << std::endl;
            }
#endif

            //----------------------------------------------------------
            // Nearest neighbors : Local CCMNeighbors() function
            //----------------------------------------------------------
            Neighbors neighbors = CCMNeighbors( Distances, lib_i, paramCCM );

            //----------------------------------------------------------
            // Subset dataFrameIn to lib_i
            //----------------------------------------------------------
            DataFrame< double > dataFrameLib_i( lib_i.size(),
                                                dataFrameIn.NColumns(),
                                                dataFrameIn.ColumnNames() );
            
            for ( size_t i = 0; i < lib_i.size(); i++ ) {
                dataFrameLib_i.WriteRow( i, dataFrameIn.Row( lib_i[ i ] ) ) ;
            }

            std::valarray<double> targetVec =
                dataFrameLib_i.VectorColumnName( paramCCM.targetName );
            
#ifdef DEBUG_ALL
            {
            std::lock_guard<std::mutex> lck( EDM_CCM::mtx );
            std::cout << "dataFrameLib_i -------------------------------\n";
            std::cout << dataFrameLib_i;
            }
#endif
            
            //----------------------------------------------------------
            // Pack embedding, target, neighbors for SimplexProjection
            //----------------------------------------------------------
            DataEmbedNN embedNN = DataEmbedNN( &dataFrameLib_i, dataBlock,
                                                targetVec,      neighbors );

            //----------------------------------------------------------
            // Simplex Projection: lib_str & pred_str set from N_row
            //----------------------------------------------------------
            DataFrame<double> S = SimplexProjection( paramCCM, embedNN, false );

            VectorError ve = ComputeError(
                S.VectorColumnName( "Observations" ),
                S.VectorColumnName( "Predictions"  ) );
            
#ifdef DEBUG_ALL
            {
            std::lock_guard<std::mutex> lck( EDM_CCM::mtx );
            std::cout << "CCM Simplex ---------------------------------\n";
            S.MaxRowPrint() = S.NRows();
            std::cout << S;
            std::cout << "rho " << ve.rho << "  RMSE " << ve.RMSE
                      << "  MAE " << ve.MAE << std::endl;
            }
#endif
            
            rho [ n ] = ve.rho;
            RMSE[ n ] = ve.RMSE;
            MAE [ n ] = ve.MAE;
            
        } // for ( n = 0; n < maxSamples; n++ )

        std::valarray< double > statVec( 4 );
        statVec[ 0 ] = lib_size;
        statVec[ 1 ] = rho.sum()  / maxSamples;
        statVec[ 2 ] = RMSE.sum() / maxSamples;
        statVec[ 3 ] = MAE.sum()  / maxSamples;

        LibStats.WriteRow( lib_size_i, statVec );
    } // for ( lib_size : param.librarySizes ) 

    } // try 
    catch(...) {
        // push exception pointer onto queue for main thread to catch
        std::lock_guard<std::mutex> lck( EDM_CCM::q_mtx );
        EDM_CCM::exceptionQ.push( std::current_exception() );
    }
    
#ifndef CCM_THREADED
    return LibStats;
#endif
}

//--------------------------------------------------------------------- 
// Note that for CCM the library and prediction rows are the same.
// Note that dataBlock does NOT have the time in column 0.
//
// Return Distances: a square matrix with distances.
// Matrix elements D[i,j] hold the distance between the E-dimensional
// phase space point (vector) between rows (observations) i and j.
//---------------------------------------------------------------------
DataFrame< double > CCMDistances( const DataFrame< double > &dataBlock,
                                        Parameters           param ) {

    size_t N_row = dataBlock.NRows();

    size_t E = param.E;

    DataFrame< double > D = DataFrame< double >( N_row, N_row );

    // Initialise D to DISTANCE_MAX to avoid sort() : Add init constructor?
    std::valarray< double > row_init( DISTANCE_MAX, N_row );
    for ( size_t row = 0; row < N_row; row++ ) {
        D.WriteRow( row, row_init );
    }

    for ( size_t row = 0; row < N_row; row++ ) {
        // Get E-dimensional vector from this library row
        std::valarray< double > v1_ = dataBlock.Row( row );
        // The first column (i=0) is NOT time, use it
        std::valarray< double > v1 = v1_[ std::slice( 0, E, 1 ) ];

        // Only compute upper triangular D, the diagonal and
        // lower left are redundant: (col < N_row - 1); row >= col
        for ( size_t col = 0; col < N_row - 1; col++ ) {
            // Avoid redundant computations
            if ( row >= col ) {
                continue; // Computed in upper triangle, copied below
            }
            
            // Find distance between vector (v) and other library vector
            std::valarray< double > v2_ = dataBlock.Row( col );
            // The first column (i=0) is NOT time, use it
            std::valarray< double > v2 = v2_[ std::slice( 0, E, 1 ) ];
            
            D( row, col ) = Distance( v1, v2, DistanceMetric::Euclidean );
            
            // Insert degenerate values since D[i,j] = D[j,i]
            D( col, row ) = D( row, col );
        }
    }
    return D;
}

//--------------------------------------------------------------------- 
// Return Neighbors { neighbors, distances }. neighbors is a matrix of 
// row indices in the library matrix. Each neighbors row represents one 
// prediction vector. Columns are the indices of knn nearest neighbors 
// for the prediction vector (phase-space point) in the library matrix.
// distances is a matrix with the same shape as neighbors holding the 
// corresponding distance values in each row.
//
// Note that the indices in neighbors are not the original indices in
// the libraryMatrix rows (observations), but are with respect to the
// distances subset defined by the list of rows lib_i, and so have values
// from 0 to len(lib_i)-1.
//
//---------------------------------------------------------------------
Neighbors CCMNeighbors( const DataFrame< double >   &DistancesIn,
                              std::vector< size_t >  lib_i,
                              Parameters             param ) {

    size_t N_row = lib_i.size();
    size_t knn   = param.knn;

#ifdef DEBUG_ALL
    {
    std::lock_guard<std::mutex> lck( EDM_CCM::mtx );
    std::cout << "CCMNeighbors Distances\n";
    for ( size_t r = 0; r < 5; r++ ) {
        for ( int c = 0; c < 5; c++ ) {
            std::cout << DistancesIn(r,c) << "  ";
        } std::cout << std::endl;
    }
    std::cout << "lib_i N_row: " << N_row
              << "  DistancesIn NRow: " << DistancesIn.NRows() << std::endl;
    }
#endif

    // Matrix to hold libraryMatrix row indices
    // One row for each prediction vector, knn columns for each index
    DataFrame< size_t > neighbors( N_row, knn );

    // Matrix to hold libraryMatrix knn distance values
    // One row for each prediction vector, k_NN columns for each index
    DataFrame< double > distances( N_row, knn );

    // For each prediction vector (row in predictionMatrix) find the list
    // of library indices that are the closest knn points
    size_t row = 0;
    std::valarray< double > knn_distances( knn );
    std::valarray< size_t > knn_neighbors( knn );

#ifdef DEBUG_ALL
    {
    std::lock_guard<std::mutex> lck( EDM_CCM::mtx );
    std::cout << "CCMNeighbors lib_i: ";
    for ( size_t i = 0; i < lib_i.size(); i++ ) {
        std::cout << lib_i[i] << " ";
    } std::cout << std::endl << std::flush;
    }
#endif    

    for ( auto row_i : lib_i ) {
  
        // Take Distances( row, col ) a row at a time
        // col represent the other row distance
        std::valarray< double > dist_row = DistancesIn.Row( row_i );
        
        // These new column indices are with respect to the lib_i vector
        // not the original Distances with all other columns
        
        // Reset the neighbor and distance vectors for this pred row
        for ( size_t i = 0; i < knn; i++ ) {
            knn_neighbors[ i ] = 0;
            // This avoids the need to sort the distances of this row
            knn_distances[ i ] = DISTANCE_MAX;
        }

        for ( size_t col_i = 0; col_i < N_row; col_i++ ) {
            double d_i = dist_row[ lib_i[col_i] ];
            // If d_i is less than values in knn_distances, add to list
            auto max_it = std::max_element( begin( knn_distances ),
                                            end( knn_distances ) );
            if ( d_i < *max_it ) {
                size_t max_i = std::distance( begin(knn_distances), max_it );

                if ( col_i >= N_row - param.tau * param.E ) {
                    continue;
                }
                
                knn_neighbors[ max_i ] = col_i;  // Save the index
                knn_distances[ max_i ] = d_i;    // Save the value
            }
        }
        
        neighbors.WriteRow( row, knn_neighbors );
        distances.WriteRow( row, knn_distances );

        row = row + 1;
    }

    Neighbors ccmNeighbors = Neighbors();
    ccmNeighbors.neighbors = neighbors;
    ccmNeighbors.distances = distances;

#ifdef DEBUG_ALL
    {
    std::lock_guard<std::mutex> lck( EDM_CCM::mtx );
    std::cout << "CCMNeighbors knn_neighbors\n";
    for ( size_t r = 0; r < 5; r++ ) {
        for ( int c = 0; c < ccmNeighbors.neighbors.NColumns(); c++ ) {
            std::cout << ccmNeighbors.neighbors(r,c) << "  ";
        } std::cout << std::endl;
    }
    }
#endif

    return ccmNeighbors;
}
