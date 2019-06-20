
#include <cstdlib>
#include <random>
#include <chrono>

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

//----------------------------------------------------------------
// forward declarations
//----------------------------------------------------------------
#ifdef CCM_THREADED
void CrossMap( Parameters p, DataFrame< double > df,
               const DataFrame< double > & LibStats );
#else
DataFrame< double > CrossMap( Parameters p, DataFrame< double > df );
#endif

DataFrame< double > CCMDistances( DataFrame< double > dataBlock,
                                  Parameters param );

Neighbors CCMNeighbors( DataFrame< double > Distances,
                        std::vector< size_t > lib_i, Parameters param );

DataFrame<double> SimplexProjection( Parameters  param,
                                     DataEmbedNN embedNN,
                                     bool        checkDataRows = true );

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
                                             seed,
                                             verbose );
    return PredictLibRho;
}

//----------------------------------------------------------------
// API Overload 2: DataFrame passed in
//   Implemented a wrapper for CrossMap()
//----------------------------------------------------------------
DataFrame <double > CCM( DataFrame< double > dataFrameIn,
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
                         unsigned    seed,
                         bool        verbose )
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

    Parameters param = Parameters( Method::Simplex, "", "",
                                   pathOut, predictFile,
                                   "", "", E, Tp, knn, tau, 0, 0,
                                   columns, target, false, false, verbose,
                                   "", "", "", 0, 0, 0, 0,
                                   libSizes_str, sample, random, seed );

    if ( param.columnNames.size() > 1 ) {
        std::cout << "WARNING: CCM() Only the first column will be mapped.\n";
    }

    // Create an Parameters object that switches column[0] and target
    // for the inverse mapping
    Parameters inverseParam( param ); // copy constructor
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
//----------------------------------------------------------------
#ifdef CCM_THREADED
void CrossMap( Parameters paramCCM,
               DataFrame< double > dataFrameIn,
               const DataFrame< double > & LibStatsIn ) {
    
    DataFrame< double > & LibStats =
        const_cast<DataFrame< double > &>(LibStatsIn);
#else
DataFrame< double > CrossMap( Parameters paramCCM,
                              DataFrame< double > dataFrameIn ) {
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
    
    //----------------------------------------------------------
    // Generate embedding on data to be cross mapped (-c column)
    //----------------------------------------------------------
    DataFrame<double> dataBlock;

    dataBlock = Embed( dataFrameIn, paramCCM.E, paramCCM.tau,
                       paramCCM.columnNames[0], paramCCM.verbose );

    size_t N_row = dataBlock.NRows();

    // JP: This removal of partial data rows is also done in EmbedNN()
    //     Should investigate how to avoid this duplication
    //----------------------------------------------------------
    // Remove dataFrameIn, target rows as needed
    //----------------------------------------------------------
    // If we support negtive tau, this will change
    // For now, assume only positive tau is allowed
    size_t shift = std::max(0, paramCCM.tau * (paramCCM.E - 1) );
    
    DataFrame<double> dataInEmbed( dataFrameIn.NRows() - shift,
                                   dataFrameIn.NColumns(),
                                   dataFrameIn.ColumnNames() );
    
    for ( size_t row = 0; row < dataInEmbed.NRows(); row++ ) {
        dataInEmbed.WriteRow( row, dataFrameIn.Row( row + shift ) );
    }

#ifdef DEBUG_ALL
    std::cout << ">>>> CrossMap() dataInEmbed-----------------------\n";
    std::cout << dataInEmbed;
    std::cout << "<<<< dataInEmbed----------------------------------\n";
    std::cout << ">>>> dataBlock------------------------------------\n";
    std::cout << dataBlock;
    std::cout << "<<<< dataBlock------------------------------------\n";
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
    // Set the number of samples
    //-----------------------------------------------------------------
    size_t maxSamples;
    if ( paramCCM.randomLib ) {
        // Random samples from library with replacement
        maxSamples = paramCCM.subSamples;
    }
    else {
        // Contiguous samples up to the size of the library
        maxSamples = 1;
    }

    //-----------------------------------------------------------------
    // Create random number generator
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
    std::default_random_engine generator( paramCCM.seed );
            
    //-----------------------------------------------------------------
    // Distance for all possible pred : lib E-dimensional vector pairs
    // Distances is a square Matrix of all row to to row distances
    //-----------------------------------------------------------------
    DataFrame< double > Distances = CCMDistances( dataBlock, paramCCM );

#ifdef DEBUG_ALL
    std::cout << "CrossMap() " << paramCCM.columnNames[0] << " to "
              << paramCCM.targetName << " Distances: " << Distances.NRows()
              << " x " << Distances.NColumns() << std::endl;
#endif
    
    //----------------------------------------------------------
    // Predictions
    //----------------------------------------------------------
#ifndef CCM_THREADED
    // Output DataFrame
    DataFrame<double> LibStats( paramCCM.librarySizes.size(), 4,
                                "LibSize rho RMSE MAE" );
#endif
    
    // Loop for library sizes
    for ( size_t lib_size_i = 0;
          lib_size_i < paramCCM.librarySizes.size(); lib_size_i++ ) {

        size_t lib_size = paramCCM.librarySizes[ lib_size_i ];

        // Create RNG sampler for this lib_size
        std::uniform_int_distribution<size_t> distribution( 0, N_row - 1 );
        
#ifdef DEBUG_ALL
        std::cout << "lib_size: " << lib_size
                  << " ------------------------------------------\n";
#endif
            
        std::valarray< double > rho ( maxSamples );
        std::valarray< double > RMSE( maxSamples );
        std::valarray< double > MAE ( maxSamples );

        // Loop for subsamples
        for ( size_t n = 0; n < maxSamples; n++ ) {
            
            // Vector of row indices to include in this lib_size evaluation
            std::vector< size_t > lib_i( lib_size );

            if ( paramCCM.randomLib ) {
                // Uniform random sample of rows, with replacement
                for ( size_t i = 0; i < lib_size; i++ ) {
                    lib_i[ i ] = distribution( generator );
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
                        msg << "CCM(): max lib_size is " << N_row
                            << " lib_size has been limited.\n";
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
            std::cout << "lib_i: (" << lib_i.size() << ") ";
            for ( size_t i = 0; i < lib_i.size(); i++ ) {
                std::cout << lib_i[i] << " ";
            } std::cout << std::endl;
#endif

            //----------------------------------------------------------
            // Nearest neighbors : Local CCMNeighbors() function
            //----------------------------------------------------------
            Neighbors neighbors = CCMNeighbors( Distances, lib_i, paramCCM );

            //----------------------------------------------------------
            // Subset dataInEmbed to lib_i
            //----------------------------------------------------------
            DataFrame< double > dataFrameLib_i( lib_i.size(),
                                                dataInEmbed.NColumns(),
                                                dataInEmbed.ColumnNames() );
            
            for ( size_t i = 0; i < lib_i.size(); i++ ) {
                dataFrameLib_i.WriteRow( i, dataInEmbed.Row( lib_i[ i ] ) ) ;
            }

            std::valarray<double> targetVec =
                dataFrameLib_i.VectorColumnName( paramCCM.targetName );
            
            //----------------------------------------------------------
            // Pack embedding, target, neighbors for SimplexProjection
            //----------------------------------------------------------
            DataEmbedNN embedNN = DataEmbedNN( dataFrameLib_i, dataBlock,
                                               targetVec,  neighbors );

            //----------------------------------------------------------
            // Simplex Projection: lib_str & pred_str set from N_row
            //----------------------------------------------------------
            DataFrame<double> S = SimplexProjection( paramCCM, embedNN, false );

            VectorError ve = ComputeError(
                S.VectorColumnName( "Observations" ),
                S.VectorColumnName( "Predictions"  ) );
            
#ifdef DEBUG_ALL
            std::cout << "CCM Simplex ---------------------------------\n";
            std::cout << "rho " << ve.rho << "  RMSE " << ve.RMSE
                      << "  MAE " << ve.MAE << std::endl;
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
DataFrame< double > CCMDistances( DataFrame< double > dataBlock,
                                  Parameters param ) {
    
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
Neighbors CCMNeighbors( DataFrame< double >   DistancesIn,
                        std::vector< size_t > lib_i,
                        Parameters            param ) {

    size_t N_row = lib_i.size();
    size_t knn   = param.knn;

#ifdef DEBUG_ALL
    std::cout << "CCMNeighbors Distances\n";
    for ( size_t r = 0; r < 5; r++ ) {
        for ( int c = 0; c < 5; c++ ) {
            std::cout << DistancesIn(r,c) << "  ";
        } std::cout << std::endl;
    }
    std::cout << "lib_i N_row: " << N_row
              << "  DistancesIn NRow: " << DistancesIn.NRows() << std::endl;
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
    std::cout << "CCMNeighbors lib_i: ";
    for ( size_t i = 0; i < lib_i.size(); i++ ) {
        std::cout << lib_i[i] << " ";
    } std::cout << std::endl << std::flush;
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
    std::cout << "CCMNeighbors knn_neighbors\n";
    for ( size_t r = 0; r < 5; r++ ) {
        for ( int c = 0; c < ccmNeighbors.neighbors.NColumns(); c++ ) {
            std::cout << ccmNeighbors.neighbors(r,c) << "  ";
        } std::cout << std::endl;
    }        
#endif

    return ccmNeighbors;
}
