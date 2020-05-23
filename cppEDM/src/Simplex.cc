
#include "Common.h"
#include "Parameter.h"
#include "Neighbors.h"
#include "Embed.h"
#include "AuxFunc.h"

// Forward declaration
DataFrame<double> SimplexProjection( Parameters  param,
                                     DataEmbedNN embedNN,
                                     bool        checkDataRows = true );

//----------------------------------------------------------------
// API Overload 1: Explicit data file path/name
//   Implemented as a wrapper to API Overload 2:
//----------------------------------------------------------------
DataFrame<double> Simplex( std::string pathIn,
                           std::string dataFile,
                           std::string pathOut,
                           std::string predictFile,
                           std::string lib,
                           std::string pred,
                           int         E,
                           int         Tp,
                           int         knn,
                           int         tau,
                           int         exclusionRadius,
                           std::string columns,
                           std::string target,
                           bool        embedded,
                           bool        const_predict,
                           bool        verbose ) {
    
    // DataFrame constructor loads data
    DataFrame< double > *dataFrameIn =
        new DataFrame< double > ( pathIn, dataFile );

    // Pass data frame to Simplex 
    DataFrame< double > S = Simplex( std::ref( *dataFrameIn ),
                                     pathOut,
                                     predictFile,
                                     lib,
                                     pred,
                                     E,
                                     Tp,
                                     knn,
                                     tau,
                                     exclusionRadius,
                                     columns,
                                     target,
                                     embedded,
                                     const_predict,
                                     verbose );
    delete dataFrameIn;
    
    return S;
}

//----------------------------------------------------------------
// API Overload 2: DataFrame provided
//----------------------------------------------------------------
DataFrame<double> Simplex( DataFrame< double > &data,
                           std::string pathOut,
                           std::string predictFile,
                           std::string lib,
                           std::string pred,
                           int         E,
                           int         Tp,
                           int         knn,
                           int         tau,
                           int         exclusionRadius,
                           std::string columns,
                           std::string target,
                           bool        embedded,
                           bool        const_predict,
                           bool        verbose ) {

    Parameters param = Parameters( Method::Simplex, "", "",
                                   pathOut, predictFile,
                                   lib, pred, E, Tp, knn, tau, 0,
                                   exclusionRadius,
                                   columns, target, embedded,
                                   const_predict, verbose );

    //----------------------------------------------------------
    // Embed, compute Neighbors
    //----------------------------------------------------------
    DataEmbedNN embedNN = EmbedNN( &data, std::ref( param ) );

    DataFrame<double> S = SimplexProjection( param, embedNN );

    return S;
}

//----------------------------------------------------------------
// Simplex Projection
//----------------------------------------------------------------
DataFrame<double> SimplexProjection( Parameters  param,
                                     DataEmbedNN embedNN,
                                     bool        checkDataRows ) {

    // Unpack the data, (embedding dataBlock not used), target & neighbors
    DataFrame<double>    *dataIn     = embedNN.dataIn;  // used for output
    std::valarray<double> target_vec = embedNN.targetVec;
    Neighbors             neighbors  = embedNN.neighbors;

    size_t library_N_row = param.library.size();
    size_t N_row         = neighbors.neighbors.NRows();

    auto max_lib_it = std::max_element( param.library.begin(),
                                        param.library.end() );
    int max_lib_index = *max_lib_it; // int for compare to libRow int

#ifdef DEBUG_ALL
    std::cout << "SimplexProjection -------------------------\n";
    std::cout << "Neighbors: (" << neighbors.neighbors.NRows() << "x"
              << neighbors.neighbors.NColumns() << ")\n";
    std::cout << neighbors.neighbors;
    std::cout << "Target: (" << target_vec.size() << ")\n";
    for ( size_t row = 0; row < 10; row++ ) {
        std::cout << target_vec[ row ] << " ";
    } std::cout << std::endl;
    std::cout << "-------------------------------------------\n\n";
#endif
    
    if ( N_row != neighbors.distances.NRows() ) {
        std::stringstream errMsg;
        errMsg << "Simplex(): Number of neighbor rows " << N_row
               << " doesn't match the number of distances rows "
               << neighbors.distances.NRows() << std::endl;
        throw std::runtime_error( errMsg.str() );
    }

    double minWeight = 1.E-6;
    std::valarray<double> predictions( 0., N_row );
    std::valarray<double> variance   ( 0., N_row );

    // Process each prediction row in neighbors
    for ( size_t row = 0; row < N_row; row++ ) {

        std::valarray<double> distanceRow = neighbors.distances.Row( row );
        
        // Establish exponential weight reference, the 'distance scale'
        double minDistance = distanceRow.min();

        // Compute weight (vector) for each k_NN
        std::valarray<double> weightedDistances( minWeight, param.knn );
        
        if ( minDistance == 0 ) {
            // Handle cases of distanceRow = 0 : can't divide by minDistance
            for ( size_t i = 0; i < param.knn; i++ ) {
                if ( distanceRow[i] > 0 ) {
                    weightedDistances[i] = exp( -distanceRow[i] / minDistance );
                }
                else {
                    // Setting weight = 1 implies that the corresponding
                    // library target vector is the same as the observation
                    // so it will be given full-weight in the prediction.
                    weightedDistances[i] = 1;
                }
            }
        }
        else {
            // exp() is a valarray<> overload (vectorized?)
            weightedDistances = exp( -distanceRow / minDistance );
        }

        // weight vector
        std::valarray<double> weights( param.knn );
        for  ( size_t i = 0; i < param.knn; i++ ) {
            weights[i] = std::max( weightedDistances[i], minWeight );
        }

        // target library vector, one element for each knn
        std::valarray<double> libTarget( param.knn );

        for ( size_t k = 0; k < param.knn; k++ ) {
            int libRow = neighbors.neighbors( row, k ) + param.Tp;

            if ( libRow > max_lib_index ) {
                // The k_NN index + Tp is outside the library domain
                // Can only happen if noNeighborLimit = true is used.
                if ( param.verbose ) {
                    std::stringstream msg;
                    msg << "Simplex() in row " << row << " libRow " << libRow
                        << " exceeds library domain.\n";
                    std::cout << msg.str();
                }
                // Use the neighbor at the 'base' of the trajectory
                libTarget[ k ] = target_vec[ libRow - abs( param.Tp ) ];
            }
            else if ( libRow < 0 ) {
                if ( param.verbose ) {
                    std::stringstream msg;
                    msg << "Simplex() in row " << row << " libRow " << libRow
                        << " precedes library domain.\n";
                    std::cout << msg.str();
                }
                // Use the neighbor at the 'base' of the trajectory
                libTarget[ k ] = target_vec[ 0 ];
            }
            else {
                libTarget[ k ] = target_vec[ libRow ];
            }
        }

        //------------------------------------------------------------------
        // If ties, expand & adjust libTarget & weights
        //------------------------------------------------------------------
        if ( neighbors.anyTies ) {
            if ( neighbors.ties[ row ] ) {
                std::vector< std::pair< double, size_t > > rowTiePairs =
                    neighbors.tiePairs[ row ];
                
                double tieDistance = rowTiePairs[ 0 ].first; // all dist same...
                size_t numTies     = rowTiePairs.size();
                double tieFactor   = 1;
                if ( numTies ) {
                    tieFactor = 1 / double( numTies );
                }

                // resize libTarget
                std::valarray< double > libTargetCopy( libTarget );
                libTarget.resize( param.knn + numTies ); // destroys contents

                // Copy original libTarget knn values
                libTarget[ std::slice( 0, param.knn, 1 ) ] = libTargetCopy;

                // Add numTies values
                for ( size_t k2 = 0; k2 < rowTiePairs.size(); k2++ ) {
                    int libRow = rowTiePairs[ k2 ].second + param.Tp;
                    if ( libRow > max_lib_index ) {
                        libTarget[ k2 + param.knn ] =
                            target_vec[ libRow - abs( param.Tp ) ];
                    }
                    else if ( libRow < 0 ) {
                        libTarget[ k2 + param.knn ] = target_vec[ 0 ];
                    }
                    else {
                        libTarget[ k2 + param.knn ] = target_vec[ libRow ];
                    }
                }

                // Weights
                // Resize distanceRow 
                std::valarray<double> distanceRowCopy( distanceRow );
                distanceRow.resize( param.knn + numTies ); // destroys contents
                
                // Copy original distanceRow knn values
                distanceRow[ std::slice( 0, param.knn, 1 ) ] = distanceRowCopy;
                
                // Add numTies values
                for ( size_t k2 = 0; k2 < rowTiePairs.size(); k2++ ) {
                    distanceRow[ k2 + param.knn ] = rowTiePairs[ k2 ].first;
                }
                minDistance = distanceRow.min();

                // Resize weightedDistances
                std::valarray<double> weightedDistancesCopy( weightedDistances );
                weightedDistances.resize( param.knn + numTies ); // destroys
                
                if ( minDistance == 0 ) {
                    // Handle cases of distanceRow = 0
                    for ( size_t i = 0; i < param.knn + numTies; i++ ) {
                        if ( distanceRow[i] > 0 ) {
                            weightedDistances[i] = exp( -distanceRow[i] /
                                                        minDistance );
                        }
                        else {
                            weightedDistances[i] = 1;
                        }
                    }
                }
                else {
                    weightedDistances = exp( -distanceRow / minDistance );
                }

                // Resize weights
                std::valarray<double> weightsCopy( weights );
                weights.resize( param.knn + numTies ); // destroys
                
                // Copy original knn weight values
                weights[ std::slice( 0, param.knn, 1 ) ] = weightsCopy;

                // Apply weight adjusment for ties
                for ( size_t k2 = param.knn; k2 < weights.size(); k2++ ) {
                    weights[k2] = tieFactor *
                        std::max( weightedDistances[k2], minWeight );
                }
            }

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
#endif
        }
        //------------------------------------------------------------------
        //------------------------------------------------------------------

        // Prediction is average of weighted library projections
        predictions[ row ] = ( weights * libTarget ).sum() / weights.sum();

        // "Variance" estimate assuming weights are probabilities
        std::valarray< double > deltaSqr =
            std::pow(libTarget - predictions[ row ], 2);
        variance[ row ] = ( weights * deltaSqr ).sum() / weights.sum();

    } // for ( row = 0; row < N_row; row++ )

    // non "predictions" X(t+1) = X(t) if const_predict specified
    std::valarray< double > const_predictions( 0., N_row );
    if ( param.const_predict ) {
        std::slice pred_slice =
            std::slice( param.prediction[ 0 ], param.prediction.size(), 1 );
        
        const_predictions = target_vec[ pred_slice ];
    }

    //----------------------------------------------------
    // Ouput
    //----------------------------------------------------
    DataFrame<double> dataFrame = FormatOutput( param,
                                                predictions,
                                                const_predictions,
                                                variance,
                                                target_vec,
                                                dataIn->Time(),
                                                dataIn->TimeName() );

    if ( param.predictOutputFile.size() ) {
        // Write to disk
        dataFrame.WriteData( param.pathOut, param.predictOutputFile );
    }

#ifdef DEBUG_ALL
    std::cout << dataFrame;
    VectorError ve = ComputeError(
        dataFrame.VectorColumnName( "Observations" ),
        dataFrame.VectorColumnName( "Predictions"  ) );
    std::cout << "-------------------------------------------\n";
    std::cout << "rho " << ve.rho << "  RMSE " << ve.RMSE
              << "  MAE " << ve.MAE << std::endl;
    std::cout << "-------------------------------------------\n";
#endif

    return dataFrame;
}
