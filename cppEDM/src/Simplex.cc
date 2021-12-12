
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
    size_t Npred      = knn_neighbors.NRows();
    predictions       = std::valarray< double > ( 0., Npred );
    const_predictions = std::valarray< double > ( 0., Npred );
    variance          = std::valarray< double > ( 0., Npred );

    int    targetSize = (int) target.size();
    double minWeight  = 1.E-6;

    // Process each prediction row in neighbors : distances
    for ( size_t row = 0; row < Npred; row++ ) {

        std::valarray< double > distanceRow = knn_distances.Row( row );

        // Establish exponential weight reference, the 'distance scale'
        double minDistance = distanceRow.min();

        // Compute weightedDistances vector for each k_NN
        std::valarray< double > weightedDistances( minWeight, parameters.knn );

        if ( minDistance == 0 ) {
            // Handle cases of distanceRow = 0
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
        std::valarray< double > libTarget( 0., parameters.knn );
        int targetLibRowOffset = parameters.Tp - embedShift;
        for ( int k = 0; k < parameters.knn; k++ ) {
            int libRow = knn_neighbors( row, k ) + targetLibRowOffset;
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
                size_t numTies     = ( parameters.knn - 1 ) - tieFirstIdx +
                                     rowTiePairs.size();
                size_t knnSize     = tieFirstIdx + numTies;
                size_t tiesFound   = 0;

                if ( (int) knnSize > parameters.knn ) {

                    double tieFactor =
                        double( numTies + parameters.knn - knnSize ) /
                        double( numTies );

                    double tieWeight = *( end( weights ) - 1 );

                    // Copies of libTarget & weights for resize
                    std::valarray< double > libTargetCopy( libTarget );
                    std::valarray< double > weightsCopy  ( weights );

                    // resize libTarget & weights : destroys contents, init 0
                    libTarget.resize( (size_t) knnSize, 0. );
                    weights.resize  ( (size_t) knnSize, 0. );

                    // Copy original knn libTarget & weights values
                    libTarget[std::slice(0, parameters.knn, 1)] = libTargetCopy;
                    weights  [std::slice(0, parameters.knn, 1)] = weightsCopy;

                    // Copy expanded nn target values
                    size_t p = 1;
                    for ( size_t k = parameters.knn; k < knnSize; k++ ) {

                        if ( p >= rowTiePairs.size() ) {
                            std::string errMsg("Simplex(): Tie index error.\n");
                            throw std::runtime_error( errMsg );
                        }

                        int libRow = (int) rowTiePairs[p].second +
                                           targetLibRowOffset;
                        p++;

                        if ( libRow >= targetSize or libRow < 0 ) {
                            continue; // no target lib
                        }

                        libTarget[ k ] = target[ libRow ];
                        weights  [ k ] = tieWeight;

                        tiesFound++;
                    }

                    // Apply weight adjusment to ties
                    if ( tiesFound ) {
                        for ( size_t i = tieFirstIdx; i < weights.size(); i++ ) {
                            weights[i] = tieFactor * weights[i];
                        }
                    }
                } // if ( (int) knnSize > parameters.knn )
            } // if ( ties[ row ] )
        } // if ( anyTies )
        //------------------------------------------------------------------

        // Prediction is average of weighted library projections
        predictions[ row ] = ( weights * libTarget ).sum() / weights.sum();

        // "Variance" estimate assuming weights are probabilities
        std::valarray< double > deltaSqr =
            std::pow( libTarget - predictions[ row ], 2 );
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
// Generate : Recursively generate n = generateSteps predictions
//
//       This should be a base EDM method for Simplex & SMap.
//       That requires virtual class members/accessor since
//       the SimplexClass::EDM object calls EDM::Generate().
//       virtual methods & runtime deference are not splendid.
//
// NOTE: The EDM::SimplexClass DataFrame "data" is a reference 
//       that was instantiated in the Simplex() overload 
//       in API.cc (if data filepath provided), or, passed
//       into Simplex() overload in API.h by the pyEDM or
//       rEDM wrappers, or direct call from cppEDM API.
//
//       The DataFrame contains numeric data in a valarray.
//       Since valarray does not have push_back, a new DataFrame is
//       constructed for each iteration of the generative projection.
//
//       Only implemented for univariate data with embedded = false.
//       Not for multivariate data with embedded = true.
//----------------------------------------------------------------
void SimplexClass::Generate() {

#ifdef DEBUG_ALL
    std::cout << ">>>> SimplexClass:: Generate() "
              << parameters.generateSteps << std::endl;
    std::cout << "     data.NRows " << data.NRows() << std::endl << "     ";
    for ( unsigned i = 0; i < data.NColumns(); i++ ) {
        std::cout << data.ColumnNames()[i] << " ";
    } std::cout << std::endl;
    std::cout << "     data.Time() end: " << data.Time().back() << std::endl;
#endif

    // Override prediction to have max( 2,Tp ) points at end of data.
    // We need Tp points if Tp > 1 to prevent nan gaps in prediction.
    // prediction & library are zero-offset in Parameters::Validate()
    size_t nPrediction = std::max( 2, parameters.Tp );

    if ( nPrediction >= data.NRows() ) {
        std::string errMsg("SimplexClass::Generate(): Tp too large.\n");
        throw std::runtime_error( errMsg );
    }

    size_t predStart = data.NRows() - nPrediction;

    if ( predStart < 1 ) {
        std::string errMsg("SimplexClass::Generate(): "
                           "prediction index too low.\n");
        throw std::runtime_error( errMsg );
    }

    // Override prediction to have max(2,Tp) points at end of data
    parameters.prediction.clear();
    for ( size_t i = 0; i < nPrediction; i++ ) {
        parameters.prediction.push_back( predStart + i );
    }

    std::cout << "NOTE: SimplexClass::Generate(): "
              << "prediction indices overriden to "
              << parameters.prediction.front() + 1 << " "
              << parameters.prediction.back()  + 1 << std::endl;

    // Output DataFrame to replace projections
    // This function over-rides the prediction indices to ensure no
    // nan data gaps at the next time-step, even with Tp > 1.
    // Project() returns a DataFrame with 3 rows if Tp = 1,
    // or Tp * 2 rows if Tp > 1. The final generated DataFrame
    // will have the original rows, plus generateSteps rows.
    size_t nOutRows = parameters.Tp == 1 ?
        parameters.generateSteps + 2 :
        parameters.generateSteps + parameters.Tp;
    DataFrame< double > generated( nOutRows, 3,
                                   "Observations Predictions Pred_Variance" );
    // Output time vector
    std::vector< std::string > generatedTime;

    // Get univariate column data into columnData vector for push_back addition.
    // At each iteration, the prediction is added to a new DataFrame
    // that replaces the SimplexClass::data object for the next Project()
    std::valarray< double >
        valarrayData = data.VectorColumnName( parameters.columnNames.front() );

    std::vector< double > columnData;
    columnData.assign( std::begin( valarrayData ), std::end( valarrayData ) );

    // Local copy of complete data time vector for new data DataFrame
    std::vector< std::string > dataTime( data.Time() );

    //-------------------------------------------------------------------
    // Loop for each feedback generation step
    //-------------------------------------------------------------------
    for ( int step = 0; step < (int) parameters.generateSteps; step++ ) {

        // 1) Generate prediction --------------------------------------
        Project();

        std::valarray< double > newPredictions =
            projection.VectorColumnName( "Predictions" );

        double      newPrediction = newPredictions   [ nPrediction ];
        std::string newTime       = projection.Time()[ nPrediction ];

#ifdef DEBUG_ALL
        projection.MaxRowPrint() = 10;
        std::cout << "+++++++ newTime " << newTime
                  << " newPredict " << newPrediction << " +++++++" << std::endl;
        std::cout << "+++++++ projection " << step << " +++++++" << std::endl;
        std::cout << projection;
#endif

        // 2) Save prediction in generated -----------------------------
        if ( step == 0 ) {
            // Existing obervations
            for ( size_t j = 0; j < nPrediction; j++ ) {
                generated.WriteRow( j, projection.Row( j ) );
                generatedTime.push_back( projection.Time()[ j ] );
            }
        }
        // The 1-step ahead prediction
        generated.WriteRow( nPrediction + step,
                            projection.Row( nPrediction ) );

        generatedTime.push_back( newTime );

        // 3) Increment library by adding another row index ------------
        parameters.library.push_back( parameters.library.back() + 1 );

        // 4) Increment prediction indices -----------------------------
        for ( auto pi  = parameters.prediction.begin();
                   pi != parameters.prediction.end(); pi++ ) {
            *pi = *pi + 1;
        }

        //--------------------------------------------------------------
        // 5) Add 1-step ahead projection to data for next Project()
        //    Create newDF DataFrame with columnData.size + 1 rows, 1 column
        DataFrame< double > newDF( columnData.size() + 1, 1,
                                   parameters.columnNames.front() );

        // Add prediction time
        dataTime.push_back( newTime );
        newDF.Time()     = dataTime;
        newDF.TimeName() = data.TimeName();

        // Append projection to columnData
        columnData.push_back( newPrediction );
        // Convert to valarray and write to newData DataFrame
        std::valarray< double > newData( columnData.data(), columnData.size() );
        newDF.WriteColumn( 0, newData );

        //--------------------------------------------------------------
        // 6) Replace SimplexClass.data with newDF for next Project()
        this->data = newDF; // JP is this a leak?

#ifdef DEBUG_ALL
        std::cout << "+++++++ newDF " << step << " +++++++" << std::endl;
        newDF.MaxRowPrint() = 5;
        std::cout << newDF;
#endif
    }

    // 7) Replace SimplexClass.projection with generated
    generated.Time()     = generatedTime;
    generated.TimeName() = data.TimeName();

    projection = generated; // JP is this a leak?

#ifdef DEBUG_ALL
    std::cout << "+++++++ generated +++++++" << std::endl;
    generated.MaxRowPrint() = generated.NRows();
    std::cout << generated;
    std::cout << "<<<< SimplexClass:: Generate()" << std::endl;
#endif
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
