
#include "Neighbors.h"

//----------------------------------------------------------------
Neighbors:: Neighbors() {}
Neighbors::~Neighbors() {}

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

    if ( parameters.embedded and parameters.E != dataFrame.NColumns() ) {
        std::stringstream errMsg;
        errMsg << "FindNeighbors(): The number of dataFrame columns ("
               << dataFrame.NColumns() << ") does not match the embedding "
               << "dimension E (" << parameters.E << ")\n";
        throw std::runtime_error( errMsg.str() );
    }

    size_t N_library_rows    = parameters.library.size();
    size_t N_prediction_rows = parameters.prediction.size();
    size_t N_columns         = dataFrame.NColumns();
    
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

    // Neighbors: struct on local stack to be returned by copy
    Neighbors neighbors = Neighbors();
    neighbors.neighbors = DataFrame<size_t>(N_prediction_rows, parameters.knn);
    neighbors.distances = DataFrame<double>(N_prediction_rows, parameters.knn);

    // Vectors to hold indices and values from each comparison
    std::valarray<size_t> k_NN_neighbors( parameters.knn );
    std::valarray<double> k_NN_distances( parameters.knn );

    //-------------------------------------------------------------------
    // For each prediction vector (row in prediction DataFrame) find the
    // list of library indices that are within k_NN points
    //-------------------------------------------------------------------
    for ( size_t row_i = 0; row_i < parameters.prediction.size(); row_i++ ) {
        // Get the prediction vector for this pred_row index
        size_t pred_row = parameters.prediction[ row_i ];
        std::valarray<double> pred_vec = dataFrame.Row( pred_row );
        
        // Reset the neighbor and distance vectors for this pred row
        for ( size_t i = 0; i < parameters.knn; i++ ) {
            k_NN_neighbors[ i ] = 0;
            // JP: Used to avoid sort()
            k_NN_distances[ i ] = DISTANCE_MAX;
        }

        //--------------------------------------------------------------
        // Library Rows
        //--------------------------------------------------------------
        for ( size_t row_j = 0; row_j < parameters.library.size(); row_j++ ) {
            // Get the library vector for this lib_row index
            size_t lib_row = parameters.library[ row_j ];
            std::valarray<double> lib_vec = dataFrame.Row( lib_row );
            
            // If the library point is degenerate with the prediction,
            // ignore it.
            if ( lib_row == pred_row ) {
#ifdef DEBUG_ALL
                if ( parameters.verbose ) {
                    std::stringstream msg;
                    msg << "FindNeighbors(): Ignoring degenerate lib_row "
                        << lib_row << " and pred_row " << pred_row << std::endl;
                    std::cout << msg.str();
                }
#endif
                continue;
            }

            // Apply temporal exclusion radius: units are data rows, not time
            if ( parameters.exclusionRadius ) {
                int xrad = (int) lib_row - pred_row;
                if ( std::abs( xrad ) <= parameters.exclusionRadius ) {
                    continue;
                }
            }
                
            // If this lib_row + args.Tp >= library_N_row, then this neighbor
            // would be outside the library, keep looking if noNeighborLimit
            if ( not parameters.noNeighborLimit ) {
                if ( lib_row + parameters.Tp >= N_library_rows ) {
                    continue;
                }
            }
            
            // Find distance between the prediction vector
            // and each of the library vectors
            // The 1st column (j=0) of Time has been excluded above
            double d_i = Distance( lib_vec, pred_vec,
                                   DistanceMetric::Euclidean );

            // If d_i is less than values in k_NN_distances, add to list
            auto max_it = std::max_element( begin( k_NN_distances ),
                                            end( k_NN_distances ) );
            if ( d_i < *max_it ) {
                size_t max_i = std::distance( begin(k_NN_distances), max_it );
                k_NN_neighbors[ max_i ] = lib_row;  // Save the index
                k_NN_distances[ max_i ] = d_i;      // Save the value
            }
        } // for ( row_j = 0; row_j < library.size(); row_j++ )
        
        if ( *std::max_element( begin( k_NN_distances ),
                                end  ( k_NN_distances ) ) > DISTANCE_LIMIT ) {
            std::stringstream errMsg;
            errMsg << "FindNeighbors(): Library is too small to resolve "
                   << parameters.knn << " knn neighbors." << std::endl;
            throw std::runtime_error( errMsg.str() );
        }

        // Check for ties.  JP: Need to address this, not just warning
        // First sort a copy of k_NN_neighbors so unique() will work
        std::valarray<size_t> k_NN_neighborCopy( k_NN_neighbors );
        std::sort( begin( k_NN_neighborCopy ), end( k_NN_neighborCopy ) );
        
        // ui is iterator to first non unique element
        auto ui = std::unique( begin( k_NN_neighborCopy ),
                               end  ( k_NN_neighborCopy ) );
        
        if ( std::distance( begin( k_NN_neighborCopy ), ui ) !=
             k_NN_neighborCopy.size() ) {
            std::cout << "WARNING: FindNeighbors(): Degenerate neighbors./n";
        }

        // Write the neighbor indices and distance values
        neighbors.neighbors.WriteRow( row_i, k_NN_neighbors );
        neighbors.distances.WriteRow( row_i, k_NN_distances );
        
    } // for ( row_i = 0; row_i < predictionRows->size(); row_i++ )

#ifdef DEBUG_ALL
    const Neighbors &neigh = neighbors;
    PrintNeighborsOut( neigh );
#endif
    
    return neighbors;
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
    //for ( size_t i = 0; i < neighbors.neighbors.NRows(); i++ ) {
    for ( size_t i = 0; i < 5; i++ ) {
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
