
#include "Simplex.h"

//----------------------------------------------------------------
// Constructor
//----------------------------------------------------------------
SimplexClass::SimplexClass (
    DataFrame< double > & data, 
    Parameters          & parameters ):
    EDM{ data, parameters } {
}

//----------------------------------------------------------------
// Project : Polymorphic implementation
//----------------------------------------------------------------
void SimplexClass::Project () {
    
    PrepareEmbedding();
    
    Distances(); // all pred : lib vector distances into allDistances
    
    FindNeighbors();

    Simplex();

    FormatOutput();

    WriteOutput();
}

//----------------------------------------------------------------
// Simplex algorithm
//----------------------------------------------------------------
void SimplexClass::Simplex () {

    // Allocate output vectors to populate EDM class projections DataFrame.
    // Must be after FindNeighbors()
    size_t Npred = knn_neighbors.NRows();
    
    predictions       = std::valarray< double > ( 0., Npred );
    const_predictions = std::valarray< double > ( 0., Npred );
    variance          = std::valarray< double > ( 0., Npred );
    
    auto maxLibit = std::max_element( parameters.library.begin(),
                                      parameters.library.end() );
    int maxLibIndex = *maxLibit; // int for compare to libRow int
    
    double minWeight = 1.E-6;

    // Process each prediction row in neighbors : distances
    for ( size_t row = 0; row < Npred; row++ ) {

        std::valarray< double > distanceRow = knn_distances.Row( row );
        
        // Establish exponential weight reference, the 'distance scale'
        double minDistance = distanceRow.min();

        // Compute weight vector for each k_NN
        std::valarray< double > weightedDistances( minWeight, parameters.knn );

        if ( minDistance == 0 ) {
            // Handle cases of distanceRow = 0 : can't divide by minDistance
            for ( int i = 0; i < parameters.knn; i++ ) {
                if ( distanceRow[i] > 0 ) {
                    weightedDistances[i] = exp( -distanceRow[i] / minDistance );
                }
                else {
                    // Setting weight = 1 implies that the corresponding
                    // library target vector is the same as the observation
                    // so it will be given full-weight in the prediction.
                    weightedDistances[ i ] = 1;
                }
            }
        }
        else {
            // exp() is a valarray<> overload (vectorized?)
            weightedDistances = exp( -distanceRow / minDistance );
        }

        // weight vector
        std::valarray< double > weights( parameters.knn );
        for  ( int i = 0; i < parameters.knn; i++ ) {
            weights[i] = std::max( weightedDistances[i], minWeight );
        }

        // target library vector, one element for each knn
        std::valarray< double > libTarget( parameters.knn );

        for ( int k = 0; k < parameters.knn; k++ ) {
            int libRow = knn_neighbors( row, k ) + parameters.Tp;

            if ( libRow > maxLibIndex ) {
                // The k_NN index + Tp is outside the library domain
                // Can only happen if noNeighborLimit = true is used.
                if ( parameters.verbose ) {
                    std::stringstream msg;
                    msg << "Simplex() in row " << row << " libRow " << libRow
                        << " exceeds library domain.\n";
                    std::cout << msg.str();
                }
                // Use the neighbor at the 'base' of the trajectory
                libTarget[ k ] = target[ libRow - abs( parameters.Tp ) ];
            }
            else if ( libRow < 0 ) {
                if ( parameters.verbose ) {
                    std::stringstream msg;
                    msg << "Simplex() in row " << row << " libRow " << libRow
                        << " precedes library domain.\n";
                    std::cout << msg.str();
                }
                // Use the neighbor at the 'base' of the trajectory
                libTarget[ k ] = target[ 0 ];
            }
            else {
                libTarget[ k ] = target[ libRow ];
            }
        }

        //------------------------------------------------------------------
        // If ties, expand & adjust libTarget & weights
        //------------------------------------------------------------------
        if ( anyTies ) {
            if ( ties[ row ] ) {
                std::vector< std::pair< double, size_t > > rowTiePairs =
                    tiePairs[ row ];
                
                size_t numTies   = rowTiePairs.size();
                double tieFactor = 1;
                if ( numTies ) {
                    tieFactor = 1 / double( numTies );
                }

                // resize libTarget
                std::valarray< double > libTargetCopy( libTarget );
                libTarget.resize( parameters.knn + numTies );// destroys contents

                // Copy original libTarget knn values
                libTarget[ std::slice( 0, parameters.knn, 1 ) ] = libTargetCopy;

                // Add numTies values
                for ( size_t k2 = 0; k2 < rowTiePairs.size(); k2++ ) {
                    int libRow = rowTiePairs[ k2 ].second + parameters.Tp;
                    if ( libRow > maxLibIndex ) {
                        libTarget[ k2 + parameters.knn ] =
                            target[ libRow - abs( parameters.Tp ) ];
                    }
                    else if ( libRow < 0 ) {
                        libTarget[ k2 + parameters.knn ] = target[ 0 ];
                    }
                    else {
                        libTarget[ k2 + parameters.knn ] = target[ libRow ];
                    }
                }

                // Weights
                // Resize distanceRow 
                std::valarray<double> distanceRowCopy( distanceRow );
                distanceRow.resize(parameters.knn + numTies);// destroys contents
                
                // Copy original distanceRow knn values
                distanceRow[ std::slice( 0, parameters.knn, 1 ) ] =
                    distanceRowCopy;
                
                // Add numTies values
                for (size_t k2 = 0; k2 < rowTiePairs.size(); k2++ ) {
                    distanceRow[ k2 + parameters.knn ] = rowTiePairs[ k2 ].first;
                }
                minDistance = distanceRow.min();

                // Resize weightedDistances
                std::valarray<double> weightedDistancesCopy( weightedDistances );
                weightedDistances.resize( parameters.knn + numTies ); // destroys
                
                if ( minDistance == 0 ) {
                    // Handle cases of distanceRow = 0
                    for ( int i = 0; i < parameters.knn + (int) numTies; i++ ) {
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
                weights.resize( parameters.knn + numTies ); // destroys
                
                // Copy original knn weight values
                weights[ std::slice( 0, parameters.knn, 1 ) ] = weightsCopy;
                
                // Apply weight adjusment for ties
                for ( size_t k2 = parameters.knn; k2 < weights.size(); k2++ ) {
                    weights[k2] = tieFactor *
                        std::max( weightedDistances[k2], minWeight );
                }
            } // if ( ties[ row ] )
        } // if ( anyTies )
        //------------------------------------------------------------------
        //------------------------------------------------------------------

        // Prediction is average of weighted library projections
        predictions[ row ] = ( weights * libTarget ).sum() / weights.sum();

        // "Variance" estimate assuming weights are probabilities
        std::valarray< double > deltaSqr =
            std::pow(libTarget - predictions[ row ], 2);
        variance[ row ] = ( weights * deltaSqr ).sum() / weights.sum();

    } // for ( row = 0; row < Npred; row++ )

    // non "predictions" X(t+1) = X(t) if const_predict specified
    const_predictions = std::valarray< double > ( 0., Npred );
    if ( parameters.const_predict ) {
        std::slice pred_slice =
            std::slice( parameters.prediction[ 0 ],
                        parameters.prediction.size(), 1 );
        
        const_predictions = target[ pred_slice ];
    }
}

//----------------------------------------------------------------
// 
//----------------------------------------------------------------
void SimplexClass::WriteOutput () {
    if ( parameters.predictOutputFile.size() ) {
        projection.WriteData( parameters.pathOut,
                              parameters.predictOutputFile );
    }
}
