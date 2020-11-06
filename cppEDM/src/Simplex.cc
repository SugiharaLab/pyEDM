
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
    
    double minWeight = 1.E-6;

    // Process each prediction row in neighbors : distances
    for ( size_t row = 0; row < Npred; row++ ) {

        std::valarray< double > distanceRow = knn_distances.Row( row );
        
        // Establish exponential weight reference, the 'distance scale'
        double minDistance = distanceRow.min();

        // Compute weightedDistances vector for each k_NN
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

        // weights vector is weightedDistances > minWeight
        std::valarray< double > weights( parameters.knn );
        for  ( int i = 0; i < parameters.knn; i++ ) {
            weights[i] = std::max( weightedDistances[i], minWeight );
        }

        // target library vector, one element for each knn
        std::valarray< double > libTarget( parameters.knn );

        for ( int k = 0; k < parameters.knn; k++ ) {
            int libRow     = knn_neighbors( row, k ) + parameters.Tp;
            libTarget[ k ] = target[ libRow ];
        }

        //------------------------------------------------------------------
        // If ties, expand & adjust libTarget & weights
        //------------------------------------------------------------------
        if ( anyTies ) {

            if ( ties[ row ] ) {

                std::vector< std::pair< double, size_t > >
                    rowTiePairs = tiePairs[ row ];

                size_t tieFirstIdx = tieFirstIndex[ row ];
                size_t numTies     = rowTiePairs.size();
                size_t knnSize     = tieFirstIdx + numTies;

                double tieFactor = double( numTies + parameters.knn - knnSize )/
                                   double( numTies );

                double tieWeight = *( end( weights ) - 1 );

                // resize libTarget
                std::valarray< double > libTargetCopy( libTarget );
                libTarget.resize( knnSize ); // destroys contents

                // Copy original libTarget values
                libTarget[ std::slice( 0, parameters.knn, 1 ) ] = libTargetCopy;

                // Copy expanded nn target values
                size_t p = 1;
                for ( size_t k = tieFirstIdx + 1; k < knnSize; k++ ) {

                    if ( p > rowTiePairs.size() - 1 ) {
                        std::string errMsg( "Simplex(): Tie index error.\n" );
                        throw std::runtime_error( errMsg );
                    }

                    int libRow = (int) rowTiePairs[ p ].second + parameters.Tp;
                    p++;

                    libTarget[ k ] = target[ libRow ];
                }

                // Resize weights
                std::valarray<double> weightsCopy( weights );
                weights.resize( knnSize ); // destroys contents

                // Copy original knn weight values
                weights[ std::slice( 0, parameters.knn, 1 ) ] = weightsCopy;

                // Copy weight ties at end
                std::slice knnExpandSlice =
                    std::slice( parameters.knn - 1,
                                knnSize - parameters.knn + 1, 1 );
                weights[ knnExpandSlice ] = tieWeight;

                // Apply weight adjusment to ties
                for ( size_t i = tieFirstIdx; i < weights.size(); i++ ) {
                    weights[i] = tieFactor * weights[i];
                }
            } // if ( ties[ row ] )
        } // if ( anyTies )
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
