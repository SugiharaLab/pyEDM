
#include "EDM_Neighbors.h"

namespace EDM_Neighbors_Lock {
    std::mutex mtx;
}

//----------------------------------------------------------------
// Common code for Simplex and Smap:
//   0) CheckDataRows()
//   1) Extract or Embed() data into embedding
//   2) Get target vector
//   3) AdjustLibPred() :
//      Adjust parameters.library and parameters.prediction indices
//
// NOTE: time column is not returned in the embedding dataBlock.
//
// NOTE: If data is embedded by Embed(), the returned dataBlock
//       has tau * (E-1) rows with partial data vectors. 
//
// NOTE: If data is embedded, then the library and prediction
//       vectors in Parameters are updated to reflect this.
//----------------------------------------------------------------
void EDM::PrepareEmbedding( bool checkDataRows ) {

    if ( checkDataRows ) {
        CheckDataRows( "PrepareEmbedding" );
    }

    if ( parameters.validLib.size() ){
        CheckValidLib( "PrepareEmbedding" );
    }

    // 1) Extract or Embed() data into embedding
    if ( parameters.embedded ) {
        // data is a multivariable block, no embedding needed
        // Select the specified columns into embedding
        if ( parameters.columnNames.size() ) {
            embedding = data.DataFrameFromColumnNames( parameters.columnNames );
        }
        else {
            throw std::runtime_error( "PrepareEmbedding(): "
                                      " columnNames are empty.\n" );
        }
    }
    else {
        // embedded = false: Create time-delay embedding via EmbedData()
        // tau * (E-1) rows will be partial vectors
        EmbedData();
    }

    GetTarget(); // 2) Get target vector

    allTime = data.Time(); // Original, full record of time

    //------------------------------------------------------------
    // embedded = false: Embed() was called on data
    // Adjust parameters.library and parameters.prediction indices
    //------------------------------------------------------------
    if ( not parameters.embedded ) {
        // 3) AdjustLibPred()
        parameters.AdjustLibPred(); // JP still needed?
    }
}

//----------------------------------------------------------------
// Required that EDM::Distances() has been called.
//
// Writes to EDM object:
//   knn_distances  :  sorted knn distances
//   knn_neighbors  :  library neighbor rows of knn_distances
//   ties           :  pred row vector of bool, true if tie
//   tiePairs       :  pred row vector of < distance, libRow > pairs
//----------------------------------------------------------------
void EDM::FindNeighbors() {

#ifdef DEBUG_ALL
    PrintDataFrameIn();
#endif

    if ( not parameters.validated ) {
        std::string errMsg( "FindNeighbors(): Parameters not validated." );
        throw( std::runtime_error( errMsg ) );
    }

    if ( parameters.embedded and parameters.E != (int) embedding.NColumns() ) {
        std::stringstream errMsg;
        errMsg << "WARNING: FindNeighbors() Multivariate data "
               << "(embedded = true): The number of embedding columns ("
               << embedding.NColumns() << ") does not equal the specified "
               << "dimension E (" << parameters.E << ")\n";
        std::cout << errMsg.str();
    }

    size_t N_library_rows    = parameters.library.size();
    size_t N_prediction_rows = parameters.prediction.size();

    //-----------------------------------------------------------------
    // Identify degenerate library : prediction points by
    // set_intersection() of lib & pred indices, needs a result vector
    //-----------------------------------------------------------------
    if ( parameters.verbose and parameters.method != Method::CCM ) {
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

    // Identify maximum library index to compare against libRow + Tp
    // to avoid asking for neighbors outside the library
    auto max_lib_it = std::max_element( parameters.library.begin(),
                                        parameters.library.end() );
    int max_lib_index = *max_lib_it;

    // allLibRows are the library row indices, 1 row x lib columns
    std::valarray< size_t > rowLib = allLibRows.Row( 0 );

    //-----------------------------------------------------------------
    // Pair the distances and library row indices for sort on distance.
    // Filter out library nn that are "invalid".
    // Output predPairs is a "matrix" of prediction : library rows.
    // Each predPairs vector member correponds to a prediction row.
    // Each member is a vector of < distance, libRow (nn) > pairs.
    //-----------------------------------------------------------------
    std::vector< std::vector< std::pair< double, size_t > > >
        predPairs( N_prediction_rows );

    for ( size_t pred_row = 0; pred_row < N_prediction_rows; pred_row++ ) {

        std::valarray< double > rowDist = allDistances.Row( pred_row );

        size_t predictionRow = parameters.prediction[ pred_row ];

        // The library < distance, libRow (nn) > pairs for each pred_row
        std::vector< std::pair< double, size_t > > rowPairs;

        //------------------------------------------------------
        // Process all library distance, nn for this pred_row
        //------------------------------------------------------
        for ( size_t i = 0; i < rowDist.size(); i++ ) {
            size_t libRow   = rowLib[ i ];
            int    libRowTp = (int) libRow + parameters.Tp;

            //--------------------------------------------------
            // Exclude library rows : don't inlcude in rowPairs
            //--------------------------------------------------
            // "Leave-one-out"
            if ( libRow == predictionRow ) {
                continue; // keep looking
            }

            // Reach exceeding grasp : forecast point is outside library
            if ( libRowTp > max_lib_index ) {
               continue; // keep looking
            }
            if ( libRowTp < 0 ) {
                if ( parameters.embedded ) {
                    continue; // keep looking
                }
                else {
                    if ( libRowTp < parameters.tau * ( parameters.E - 1 ) ) {
                        continue; // keep looking
                    }
                }
            }

            // Exclusion radius: units are data rows, not time
            if ( parameters.exclusionRadius ) {
                int delta_i = std::abs( (int) predictionRow - (int) libRow );
                if ( delta_i <= parameters.exclusionRadius ) {
                    continue; // keep looking
                }
            }

            // Conditional embedding : check row is valid lib if supplied
            if ( parameters.validLib.size() ) {
                if ( not parameters.validLib[ libRow ] ) {
                    continue; // keep looking
                }
            }

            // Add this distance, libRow (nn) to the rowPairs
            rowPairs.push_back( std::make_pair( rowDist[ i ], rowLib[ i ] ) );
        }

        // Insert into predPairs
        predPairs[ pred_row ] = rowPairs;
    }

    // Allocate objects in EDM class
    // JP Put on heap & destructor, or use smart pointers
    knn_neighbors = DataFrame  < size_t >( N_prediction_rows, parameters.knn );
    knn_distances = DataFrame  < double >( N_prediction_rows, parameters.knn );

    ties          = std::vector< bool   >( N_prediction_rows, false );
    tieFirstIndex = std::vector< size_t >( N_prediction_rows, 0     );
    tiePairs      = std::vector< std::vector< std::pair< double, size_t > > >
                    ( N_prediction_rows );

    knnSmap = std::vector< size_t > ( N_prediction_rows, parameters.knn );

    //-------------------------------------------------------------------
    // For each prediction vector (row in prediction DataFrame) find
    // library indices that are within knn points
    //-------------------------------------------------------------------
    for ( size_t predPair_i = 0; predPair_i < predPairs.size(); predPair_i++ ) {

        // The actual prediction row specified by user (zero offset)
        size_t predictionRow = parameters.prediction[ predPair_i ];

        // rowPair is a vector of <distance, lib> pairs of length library rows
        // Get the rowPair for this prediction row
        std::vector< std::pair<double, size_t> > rowPair = predPairs[predPair_i];

        int rowPairSize = (int) rowPair.size();

        // sort < distance, libRow > pairs for this predPair_i
        // distance must be .first
        std::sort( rowPair.begin(), rowPair.end(), DistanceCompare );

        //----------------------------------------------------------------
        // Insert knn distance / library row index into knn vectors
        //----------------------------------------------------------------
        std::valarray< double > knnDistances( nan("knn"), parameters.knn );
        std::valarray< size_t > knnLibRows  ( (size_t) 0, parameters.knn );

        int k = 0;

        while ( k < parameters.knn ) {

            // Check for failure to find knn neighbors
            if ( k >= rowPairSize ) {

                if ( parameters.method == Method::SMap ) {
                    knnSmap[ predPair_i ] = k;  // Save this reduced knn
                }

                if ( parameters.verbose ) {
                    std::stringstream errMsg;
                    if ( k == 0) {
                        errMsg << "WARNING: FindNeighbors(): No neighbors found"
                               << " for prediction row " << predictionRow
                               << std::endl;
                    }
                    else {
                        errMsg << "WARNING: FindNeighbors(): "
                               << "knn search failed to find " << parameters.knn
                               << " neighbors in the library at prediction row "
                               << predictionRow << ". Found "
                               << k << "." << std::endl;
                    }
                    std::cout << errMsg.str();
                }

                break; // Continue to next predictionRow
            }

            knnDistances[ k ] = rowPair[ k ].first;  // distance
            knnLibRows  [ k ] = rowPair[ k ].second; // libRow
            k++;
        }

        knn_distances.WriteRow( predPair_i, knnDistances );
        knn_neighbors.WriteRow( predPair_i, knnLibRows   );

        //----------------------------------------------------------------
        // Check for ties.
        // Set EDM class ties[predPair_i] = true; anyTies = true if found.
        // Store all tied { nn, distance } pairs in EDM :: tiePairs vector.
        // Note: A tie exists only if the k-th nn has distance equal
        //       to the k+1 nn. Multiple ties can exist beyond k+1. 
        //----------------------------------------------------------------
        if ( parameters.method == Method::Simplex and k > 0 ) {

            // Is there a tie?  A quick check.
            // Note k was post incremented in loop above
            bool   knnDistanceTie       = false;
            double tieDistance          = knnDistances[ parameters.knn - 1 ];
            size_t tieNNindex           = knnLibRows  [ parameters.knn - 1 ];
            size_t rowPairFirstTieIndex = 0;

            // First, find the NN index in rowPair[].second that matches
            // that of the terminal knn value, store in rowPairFirstTieIndex
            for ( size_t i = 0; i < rowPair.size(); i++ ) {
                if ( rowPair[ i ].second == tieNNindex ) {
                    rowPairFirstTieIndex = i;
                    break;
                }
            }

            if ( tieDistance == rowPair[ rowPairFirstTieIndex + 1 ].first ) {
                knnDistanceTie = true;
            }

            // If there is a tie, populate tiePairs for Simplex
            if ( knnDistanceTie ) {
                // At least one tie... find the first tie in knn
                size_t firstTieIndex = 0;
                for ( size_t i = 0; i < knnDistances.size(); i++ ) {
                    if ( knnDistances[ i ] == tieDistance ) {
                        firstTieIndex = i;
                        break;
                    }
                }

                // Save knn firstTieIndex for Simplex
                tieFirstIndex[ predPair_i ] = firstTieIndex;

                // List of rowTiePairs for Simplex
                std::vector< std::pair< double, size_t > > rowTiePairs;

                // Start looking at rowPairFirstTieIndex
                size_t kk = rowPairFirstTieIndex;
                while( kk < rowPair.size() - 1 and
                       rowPair[ kk ].first == rowPair[ kk + 1 ].first ) {

                    // Set ties flag and store tie pairs in tiePairs for Simplex
                    ties[ predPair_i ] = true;

                    std::pair< double, size_t > thisPair =
                        std::make_pair( rowPair[ kk ].first,
                                        rowPair[ kk ].second );

                    rowTiePairs.push_back( thisPair );
                    kk++;
                }

                // Add the final tie since the above loop is pairs
                rowTiePairs.push_back( std::make_pair( rowPair[ kk ].first,
                                                       rowPair[ kk ].second ) );

                if ( find( ties.begin(), ties.end(), true ) != ties.end() ) {
                    anyTies = true;
                    tiePairs[ predPair_i ] = rowTiePairs;
                }
            } // if ( knnDistanceTie )
        } // if ( parameters.method == Method::Simplex ) {
    } // for ( predPair_i = 0; predPair_i < predPairs.size(); predPair_i++ )

#ifdef DEBUG_ALL
    for ( size_t i = 0; i < tiePairs.size(); i++ ) {
        size_t predictionRow = parameters.prediction[ i ];
        std::vector< std::pair< double, size_t > > rowTiePairs = tiePairs[ i ];

        if ( rowTiePairs.size() ) {
            std::cout << "Ties at pred " << predictionRow << " ";
            for ( size_t j = 0; j < rowTiePairs.size(); j++ ) {
                double dist = rowTiePairs[ j ].first;
                size_t prow = rowTiePairs[ j ].second;
                std::cout << "[" << prow << " : " <<  dist << "] ";
            } std::cout << std::endl;
        }
    }
    PrintNeighbors();
#endif
}

//--------------------------------------------------------------------- 
// Compute all prediction row : library row distances.
// Note that embedding does NOT have the time in column 0.
// 
// Writes EDM objects:
// allDistances: pred rows x lib columns matrix with distances.
//               distance(i,j) is distance between the E-dimensional
//               phase space point prediction row i and library row j.
// allLibRows  : 1 row x lib cols matrix with lib rows
//
// JP Note: All prediction x library distances are computed.
//          This is not optimal since some values are degenerate.
//          Ideally, degenerate values are computed once, then copied.
//          This should be addressed.  However, it's sticky since the
//          allDistances matrix is indexed from [0:Npred] rows x
//          [0:Nlib] columns, whereas the actual library or prediction
//          rows are not required to be in [0:Npred], [0:Nlib].
//          Therefore, allDistances D(i,j) != D(j,i) unless lib = pred.
//          Rather, actual rows are parameters.prediction[ predRow ],
//          parameters.library[ libRow ] with predRow in [0:Npred]...
//---------------------------------------------------------------------
void EDM::Distances () {

    // Validate library and prediction rows are in embedding
    auto max_it = std::max_element( parameters.prediction.begin(),
                                    parameters.prediction.end() );
    size_t maxPredIndex = (size_t) *max_it;

    max_it = std::max_element( parameters.library.begin(),
                               parameters.library.end() );
    size_t maxLibIndex = (size_t) *max_it;

    if ( maxPredIndex >= embedding.NRows() or
         maxLibIndex  >= embedding.NRows() ) {
        std::stringstream errMsg;
        errMsg << "Distances() library or prediction index exceeds embedding "
               << "rows: " << embedding.NRows();
        throw std::runtime_error( errMsg.str() );
    }

    size_t Npred = parameters.prediction.size();
    size_t Nlib  = parameters.library.size();

    // Allocate output distance matrix and libRows list in EDM object
    allDistances = DataFrame< double >( Npred, Nlib );
    allLibRows   = DataFrame< size_t >( 1,     Nlib );

    // Initialise D to DistanceMax
    std::valarray< double > row_init( EDM_Distance::DistanceMax, Nlib );
    for ( size_t row = 0; row < Npred; row++ ) {
        allDistances.WriteRow( row, row_init );
    }

    // Set lib indices into allLibRows
    for ( size_t col = 0; col < Nlib; col++ ) {
        allLibRows( 0, col ) = parameters.library[ col ];
    }

    // Compute all prediction row : library row distances
    for ( size_t predRow = 0; predRow < Npred; predRow++ ) {

        size_t predictionRow = parameters.prediction[ predRow ];

        // Get E-dimensional vector from this prediction row
        std::valarray< double > v1 = embedding.Row( predictionRow );

        for ( size_t libRow = 0; libRow < Nlib; libRow++ ) {

            size_t libraryRow = parameters.library[ libRow ];

            if ( predictionRow == libraryRow ) {
                continue;  // degenerate pred & lib : default DistanceMax
            }

            // Find distance between vector (v1) and library vector v2
            std::valarray< double > v2 = embedding.Row( libraryRow );

            allDistances( predRow, libRow ) =
                Distance( v1, v2, DistanceMetric::Euclidean );
        }
    }
}

//----------------------------------------------------------------
// 
//----------------------------------------------------------------
double Distance( const std::valarray< double > & v1,
                 const std::valarray< double > & v2,
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
            sum += fabs( v2[i] - v1[i] );
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
void EDM::PrintDataFrameIn()
{
    std::cout << "FindNeighbors(): library:" << std::endl;
    for ( size_t row = 0; row < parameters.library.size(); row++ ) {
        size_t row_i = parameters.library[row];
        std::cout << "row " << row_i << " : ";
        for ( size_t col = 0; col < data.NColumns(); col++ ) {
            std::cout << data(row_i,col) << " "; 
        } std::cout << std::endl;
    }
    std::cout << "FindNeighbors(): prediction:" << std::endl;
    for ( size_t row = 0; row < parameters.prediction.size(); row++ ) {
        size_t row_i = parameters.prediction[row];
        std::cout << "row " << row_i << " : ";
        for ( size_t col = 0; col < data.NColumns(); col++ ) {
            std::cout << data(row_i,col) << " "; 
        } std::cout << std::endl;
    }
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void EDM::PrintNeighbors()
{
    std::cout << "EDM::FindNeighbors(): neighbors:distances" << std::endl;
    size_t predictionRow;
    for ( size_t i = 0; i < knn_neighbors.NRows(); i++ ) {
        predictionRow = parameters.prediction[ i ];
        std::cout << "pred " << predictionRow << " | ";
        for ( size_t j = 0; j < knn_neighbors.NColumns(); j++ ) {
            std::cout << knn_neighbors( i, j ) << " ";
        } std::cout << "   : ";
        for ( size_t j = 0; j < knn_neighbors.NColumns(); j++ ) {
            std::cout << knn_distances( i, j ) << " ";
        } std::cout << std::endl;
    }
}
#endif
