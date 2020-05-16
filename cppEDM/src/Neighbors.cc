
#include "Neighbors.h"

//----------------------------------------------------------------
Neighbors:: Neighbors(): anyTies(false) {}
Neighbors::~Neighbors() {}

namespace EDM_Neighbors {
    // Define the initial maximum distance for neigbors to avoid sort()
    // DBL_MAX is a Macro equivalent to: std::numeric_limits<double>::max()
    double DistanceMax   = std::numeric_limits<double>::max();
    double DistanceLimit = std::numeric_limits<double>::max() / ( 1 + 1E-9 );
}

//----------------------------------------------------------------
// It is assumed that the data frame has only columns of data for
// which knn will be computed.  The (time) column is not present.
//----------------------------------------------------------------
Neighbors FindNeighbors(
    DataFrame<double> dataFrame,
    Parameters        parameters )
{

#ifdef DEBUG_ALL
    PrintDataFrameIn( dataFrame, parameters );
#endif

    if ( not parameters.validated ) {
        std::string errMsg("FindNeighbors(): Parameters not validated." );
        throw( std::runtime_error( errMsg ) );
    }

    if ( parameters.embedded and parameters.E > dataFrame.NColumns() ) {
        std::stringstream errMsg;
        errMsg << "WARNING: FindNeighbors() Multivariate data "
               << "(embedded = true): The number of dataFrame columns ("
               << dataFrame.NColumns() << ") is less than the embedding "
               << "dimension E (" << parameters.E << ")\n";
        std::cout << errMsg.str();
    }

    size_t N_library_rows    = parameters.library.size();
    size_t N_prediction_rows = parameters.prediction.size();
    size_t N_columns         = dataFrame.NColumns();
    
    auto max_lib_it = std::max_element( parameters.library.begin(),
                                        parameters.library.end() );
    size_t max_lib_index = *max_lib_it;
    
    // Maximum column index.
    // We assume that the dataFrame has been selected to the proper columns
    size_t maxCol_i = N_columns - 1;

    if ( parameters.verbose ) {
        // Identify degenerate library : prediction points by
        // set_intersection() of lib & pred indices, needs a result vector
        std::vector< double > result( N_library_rows + N_prediction_rows, 0 );
        
        std::vector< double >::iterator ii = set_intersection (
            parameters.prediction.begin(), parameters.prediction.end(),
            parameters.library.begin(),    parameters.library.end(), 
            result.begin() );
        
        if ( ii != result.begin() ) {
            // Overlapping indices exist
            std::stringstream msg;
            msg << "WARNING: FindNeighbors(): Degenerate library and "
                << " prediction data found. Overlap indices: ";
            for ( auto ri = result.begin(); ri != ii; ++ri ) {
                msg << *ri << " ";
            } msg << std::endl;
            std::cout << msg.str();
        }
    }

    // Compute distances of all pred : lib vectors
    // DistLib.distances is DataFrame of pred rows x lib columns with
    // distances for each pred row to all lib rows
    Neighbors DistLib = Distances( std::ref(dataFrame), parameters );

    // DistLib.neighbors are the lib row indices, 1 row x lib columns
    std::valarray< size_t > rowLib = DistLib.neighbors.Row( 0 );

    // Pair the distances and library row indices for sort on distance
    // Each predPairs element correponds to a prediction row and
    // holds a vector of < distance, lib_row > pairs for each lib_row
    std::vector< std::vector< std::pair< double, size_t > > >
        predPairs( N_prediction_rows );
    
    for ( size_t pred_row = 0; pred_row < N_prediction_rows; pred_row++ ) {
        std::valarray< double > rowDist = DistLib.distances.Row( pred_row );

        std::vector< std::pair<double, size_t> > rowPairs( rowDist.size() );
        for ( size_t i = 0; i < rowDist.size(); i++ ) {
            rowPairs[ i ] = std::make_pair( rowDist[i], rowLib[i] );
        }
        // insert into predPairs
        predPairs[ pred_row ] = rowPairs;
    }
    
#ifdef DEBUG_ALL
    std::cout << DistLib.neighbors;
    std::cout << DistLib.distances;
    for ( size_t pred_row = 0; pred_row < predPairs.size(); pred_row++ ) {
        std::vector< std::pair<double, size_t> > rowPair = predPairs[ pred_row ];
        for ( size_t i = 0; i < rowPair.size(); i++ ) {
            std::pair<double, size_t> thisPair = rowPair[ i ];
            std::cout << "[" << thisPair.first << ", "
                      << thisPair.second << "] ";
        } std::cout << std::endl;
    } std::cout << std::endl;
#endif
    
    // Neighbors: struct on local stack to be returned by copy
    Neighbors neighbors = Neighbors();
    neighbors.neighbors = DataFrame<size_t>(N_prediction_rows, parameters.knn);
    neighbors.distances = DataFrame<double>(N_prediction_rows, parameters.knn);

    // To be inserted in neighbors struct below
    std::vector< bool > ties( N_prediction_rows, false );
    std::vector< std::vector< std::pair< double, size_t > > >
        tiePairs( N_prediction_rows );
    
    //-------------------------------------------------------------------
    // For each prediction vector (row in prediction DataFrame) find the
    // list of library indices that are within k_NN points
    //-------------------------------------------------------------------
    for ( size_t pred_row = 0; pred_row < predPairs.size(); pred_row++ ) {

        // rowPair is a vector of pairs of length library rows
        // Get the rowPair for this prediction row
        std::vector< std::pair<double, size_t> > rowPair = predPairs[ pred_row ];

        // sort < distance, lib_row > pairs for this pred_row
        // distance must be .first
        std::sort( rowPair.begin(), rowPair.end(), DistanceCompare );

        // Insert knn distance / library row index into knn vectors
        std::valarray< double > knnDistances( parameters.knn );
        std::valarray< size_t > knnLibRows  ( parameters.knn );

        size_t lib_row_i = 0;
        size_t k         = 0;
        while ( k < parameters.knn ) {
            double distance = rowPair[ lib_row_i ].first;
            size_t lib_row  = rowPair[ lib_row_i ].second;

            if ( not parameters.noNeighborLimit ) {
                // Reach exceeding grasp : forecast point is outside library
                if ( lib_row + parameters.Tp > max_lib_index or
                     lib_row + parameters.Tp < 0 ) {
                    lib_row_i++;
                    continue; // keep looking 
                }
            }

            // Exclusion radius: units are data rows, not time
            if ( parameters.exclusionRadius ) {
                int xrad = (int) lib_row - (int) pred_row;
                if ( std::abs( xrad ) <= parameters.exclusionRadius ) {
                    lib_row_i++;
                    continue; // skip this neighbor
                }
            }

            knnDistances[ k ] = distance;
            knnLibRows  [ k ] = lib_row;
            lib_row_i++;
            k++;
        }

        neighbors.distances.WriteRow( pred_row, knnDistances );
        neighbors.neighbors.WriteRow( pred_row, knnLibRows   );

        // Check for ties 1.18e−38 is float 32-bit min
        if ( k < rowPair.size() ) {
            if ( rowPair[ k ].first <= rowPair[ k-1 ].first ) {
                // At least one tie...
                std::vector< std::pair< double, size_t > > rowTiePairs;

                while( k < rowPair.size() and rowPair[ k ].first > 0 and
                       rowPair[ k ].first <= rowPair[ k-1 ].first ) {
                    
                    // Set flag in ties and store tie pairs in tiePairs
                    ties[ pred_row ] = true;
                    
                    rowTiePairs.push_back(std::make_pair( rowPair[ k ].first,
                                                          rowPair[ k ].second ));
                    k++;
                }

                if ( find( ties.begin(), ties.end(), true ) != ties.end() ) {
                    neighbors.anyTies = true;
                    tiePairs[ pred_row ] = rowTiePairs;
                }
            }
        }
    } // for ( pred_row = 0; pred_row < predPairs.size(); pred_row++ )

    neighbors.ties     = ties;
    neighbors.tiePairs = tiePairs;

#ifdef DEBUG_ALL
    for ( size_t i = 0; i < neighbors.ties.size(); i++ ) {
        if ( neighbors.ties[ i ] ) {
            std::vector< std::pair< double, size_t > > rowTiePairs =
                neighbors.tiePairs[ i ];
            std::cout << "Ties at pred_i " << i << ": ";
            for ( size_t j = 0; j < rowTiePairs.size(); j++ ) {
                double dist = rowTiePairs[ j ].first;
                size_t prow = rowTiePairs[ j ].second;
                std::cout << "[ " << dist << ", " << prow << "] ";
            } std::cout << std::endl;
        }
    }
    
    const Neighbors &neigh = neighbors;
    PrintNeighborsOut( neigh );
#endif
    
    return neighbors;
}

//--------------------------------------------------------------------- 
// Compute all prediction row : library row distances.
// Note that dataBlock does NOT have the time in column 0.
//
// Hijack a Neighbors struct to return two DataFrames:
// distances: pred rows x lib columns matrix with distances.
//            distance(i,j) hold distance between the E-dimensional
//            phase space point prediction row i and library row j.
// neighbors: 1 row x lib cols matrix with lib rows
//---------------------------------------------------------------------
Neighbors Distances( const DataFrame< double > &dataBlock,
                           Parameters           param ) {
    
    size_t N_pred = param.prediction.size();
    size_t N_lib  = param.library.size();

    // Output distance matrix
    DataFrame< double > D = DataFrame< double >( N_pred, N_lib );
    DataFrame< size_t > N = DataFrame< size_t >( 1,     N_lib );

    // Initialise D to DistanceMax
    std::valarray< double > row_init( EDM_Neighbors::DistanceMax, N_lib );
    for ( size_t row = 0; row < N_pred; row++ ) {
        D.WriteRow( row, row_init );
    }

    // Set lib indices into neighbors
    for ( size_t col = 0; col < N_lib; col++ ) {
        N( 0, col ) = param.library[ col ];
    }

    // Compute all prediction row : library row distances
    for ( size_t row = 0; row < N_pred; row++ ) {
        // Get E-dimensional vector from this prediction row
        std::valarray< double > v1 = dataBlock.Row( param.prediction[ row ] );

        for ( size_t col = 0; col < N_lib; col++ ) {
            // Find distance between vector (v1) and library vector v2
            std::valarray< double > v2 = dataBlock.Row( param.library[ col ] );
            
            D( row, col ) = Distance( v1, v2, DistanceMetric::Euclidean );
        }
    }

    Neighbors DistLib = Neighbors();
    DistLib.distances = D;
    DistLib.neighbors = N;
    
    return DistLib;
}

//----------------------------------------------------------------
// 
//----------------------------------------------------------------
double Distance( const std::valarray<double> &v1,
                 const std::valarray<double> &v2,
                 DistanceMetric metric )
{
    double distance = 0;

    // For efficiency sake, we forego the usual validation of v1 & v2.

    if ( metric == DistanceMetric::Euclidean ) {
        double sum   = 0;
        double delta = 0;
        for ( size_t i = 0; i < v1.size(); i++ ) {
            delta = v2[i] - v1[i];
            sum  += delta * delta; // avoid call to pow()
        }
        distance = sqrt( sum );

        // Note: this implicit implementation is slower
        // std::valarray<double> delta = v2 - v1;
        // distance = sqrt( (delta * delta).sum() );
    }
    else if ( metric == DistanceMetric::Manhattan ) {
        double sum = 0;
        for ( size_t i = 0; i < v1.size(); i++ ) {
            sum += abs( v2[i] - v1[i] );
        }
        distance = sum;
    }
    else {
        std::stringstream errMsg;
        errMsg << "Distance() Invalid DistanceMetric: "
               << static_cast<size_t>( metric );
        throw std::runtime_error( errMsg.str() );
    }

    return distance;
}

#ifdef DEBUG_ALL
//----------------------------------------------------------------
// 
//----------------------------------------------------------------
void PrintDataFrameIn( const DataFrame<double> &dataFrame,
                       const Parameters        &parameters )
{
    std::cout << "FindNeighbors(): library:" << std::endl;
    for ( size_t row = 0; row < parameters.library.size(); row++ ) {
        size_t row_i = parameters.library[row];
        std::cout << "row " << row_i << " : ";
        for ( size_t col = 0; col < dataFrame.NColumns(); col++ ) {
            std::cout << dataFrame(row_i,col) << " "; 
        } std::cout << std::endl;
    }
    std::cout << "FindNeighbors(): prediction:" << std::endl;
    for ( size_t row = 0; row < parameters.prediction.size(); row++ ) {
        size_t row_i = parameters.prediction[row];
        std::cout << "row " << row_i << " : ";
        for ( size_t col = 0; col < dataFrame.NColumns(); col++ ) {
            std::cout << dataFrame(row_i,col) << " "; 
        } std::cout << std::endl;
    }
}

//----------------------------------------------------------------
// 
//----------------------------------------------------------------
void PrintNeighborsOut( const Neighbors &neighbors )
{
    std::cout << "FindNeighbors(): neighbors:distances" << std::endl;
    for ( size_t i = 0; i < neighbors.neighbors.NRows(); i++ ) {
        std::cout << "Row " << i << " | ";
        for ( size_t j = 0; j < neighbors.neighbors.NColumns(); j++ ) {
            std::cout << neighbors.neighbors( i, j ) << " ";
        } std::cout << "   : ";
        for ( size_t j = 0; j < neighbors.neighbors.NColumns(); j++ ) {
            std::cout << neighbors.distances( i, j ) << " ";
        } std::cout << std::endl;
    }
}
#endif
