
#include "SMap.h"

// NOTE: Contains SMapClass method implementations, AND:
//       General SVD functions: SVD, Lapack_SVD, dgelss_

//----------------------------------------------------------------
// Constructor
//----------------------------------------------------------------
SMapClass::SMapClass (
    DataFrame< double > & data, 
    Parameters          & parameters ):
    EDM{ data, parameters } {
}

//----------------------------------------------------------------
// Project : Polymorphic implementation
//----------------------------------------------------------------
void SMapClass::Project ( Solver solver ) {

    PrepareEmbedding();

    Distances(); // all pred : lib vector distances into allDistances

    FindNeighbors();

    SMap( solver );

    FormatOutput();  // Common formatting

    WriteOutput();   // SMap specific formatting & output
}

//----------------------------------------------------------------
// SMap algorithm
//----------------------------------------------------------------
void SMapClass::SMap ( Solver solver ) {

    // Allocate output vectors to populate EDM class projections DataFrame.
    // Must be after FindNeighbors()
    size_t Npred = knn_neighbors.NRows();

    predictions       = std::valarray< double > ( 0., Npred );
    const_predictions = std::valarray< double > ( 0., Npred );
    variance          = std::valarray< double > ( 0., Npred );

    // Init coefficients to NAN ?
    // Allocate +Tp rows.  Coefficients/nan will be shifted in Project()
    coefficients = DataFrame< double >( Npred + abs( parameters.Tp ),
                                        parameters.E + 1 );

    // Process each prediction row in neighbors : distances
    for ( size_t row = 0; row < Npred; row++ ) {

        size_t knn = knnSmap[ row ]; // knn is variable...

        // Average distance for knn
        double Dsum = 0;
        for ( size_t i = 0; i < knn_distances.Row( row ).size(); i++ ) {
            if ( std::isnan( knn_distances( row, i ) ) ) {
                break; // Presume first nan is contiguous at end
            }
            Dsum += knn_distances( row, i );
        }
        double Davg = Dsum / knn;

        // Weight vector w
        std::valarray< double > w = std::valarray< double >( knn );
        if ( parameters.theta > 0 ) {
            double Dscale = parameters.theta / Davg;
            for ( size_t k = 0; k < knn; k++ ) {
                w[ k ] = std::exp( -Dscale * knn_distances( row, k ) );
            }
        }
        else {
            w = std::valarray< double >( 1, knn );
        }

        // Allocate work space for solver, and target (B_noWeight)
        DataFrame< double > A = DataFrame< double >( knn, parameters.E + 1 );
        std::valarray< double > B          = std::valarray< double >( knn );
        std::valarray< double > B_noWeight = std::valarray< double >( knn );

        // Populate matrix A (exp weighted future prediction), and
        // vector B (target BC's) for this row (observation).
        int    libRow;
        size_t libRowBase;
        int    targetLibRowOffset = parameters.Tp - embedShift;

        for ( size_t k = 0; k < knn; k++ ) {
            libRowBase = knn_neighbors( row, k );
            libRow     = libRowBase + targetLibRowOffset;
            
            B[ k ] = target[ libRow ];

            //---------------------------------------------------------------
            // Linear system coefficient matrix
            //---------------------------------------------------------------
            // NOTE: The matrix A has a (weighted) constant (1) first column
            //       to enable a linear intercept/bias term.
            // NOTE: The embedding does not have a time vector, and only
            //       has columns from the embedding.  So the coefficient
            //       matrix A has E+1 columns, while the embedding has E.
            //---------------------------------------------------------------
            A( k, 0 ) = w[ k ]; // Intercept bias terms in column 0 (weighted)

            for ( int j = 1; j < parameters.E + 1; j++ ) {
                A( k, j ) = w[ k ] * embedding( libRowBase, j - 1 );
            }
        }

        B_noWeight = B; // Copy target vector for "variance" estimate

        B = w * B;      // Weight target/boundary condition vector for solver

        // Estimate linear mapping of predictions A onto target B
        std::valarray < double > C = solver( A, B );

        // Prediction is local linear projection
        double prediction = C[ 0 ]; // C[ 0 ] is the bias term

        for ( int e = 1; e < parameters.E + 1; e++ ) {
            prediction = prediction +
                         C[ e ] * embedding( parameters.prediction[ row ], e-1 );
        }

        predictions[ row ] = prediction;
        coefficients.WriteRow( row, C );

        // "Variance" estimate assuming weights are probabilities
        std::valarray< double > deltaSqr =
            std::pow( B_noWeight - predictions[ row ], 2);

        variance[ row ] = ( w * deltaSqr ).sum() / w.sum();

    } // for ( row = 0; row < Npred; row++ )

    // non "predictions" X(t+1) = X(t) if const_predict specified
    const_predictions = std::valarray< double >( 0., Npred );
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
//       the SMapClass::EDM object calls EDM::Generate().
//       virtual methods & runtime deference are not splendid.
//
// NOTE: The EDM::SMapClass DataFrame "data" is a reference 
//       that was instantiated in the SMap() overload 
//       in API.cc (if data filepath provided), or, passed
//       into SMap() overload in API.h by the pyEDM or
//       rEDM wrappers, or direct call from cppEDM API.
//
//       The DataFrame contains numeric data in a valarray.
//       Since valarray does not have push_back, a new DataFrame is
//       constructed for each iteration of the generative projection.
//
//       Only implemented for univariate data with embedded = false.
//       Not for multivariate data with embedded = true.
//
//----------------------------------------------------------------
void SMapClass::Generate( Solver solver ) {

#ifdef DEBUG_ALL
    std::cout << ">>>> SMapClass:: Generate() "
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
        std::string errMsg("SMapClass::Generate(): Tp too large.\n");
        throw std::runtime_error( errMsg );
    }

    size_t predStart = data.NRows() - nPrediction;

    if ( predStart < 1 ) {
        std::string errMsg("SMapClass::Generate(): "
                           "prediction index too low.\n");
        throw std::runtime_error( errMsg );
    }

    // Override prediction to have max(2,Tp) points at end of data
    parameters.prediction.clear();
    for ( size_t i = 0; i < nPrediction; i++ ) {
        parameters.prediction.push_back( predStart + i );
    }

    std::cout << "NOTE: SMapClass::Generate(): "
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
    // Output DataFrame for coefficients
    DataFrame< double > generatedCoef( nOutRows, parameters.E + 1 );
    
    // Output time vector
    std::vector< std::string > generatedTime;

    // Get univariate target data into columnData vector for push_back addition.
    // At each iteration, the prediction is added to a new DataFrame
    // that replaces the SMapClass::data object for the next Project()
    std::valarray< double >
        valarrayData = data.VectorColumnName( parameters.targetName );

    std::vector< double > columnData;
    columnData.assign( std::begin( valarrayData ), std::end( valarrayData ) );

    // Local copy of complete data time vector for new data DataFrame
    std::vector< std::string > dataTime( data.Time() );

    //-------------------------------------------------------------------
    // Loop for each feedback generation step
    //-------------------------------------------------------------------
    for ( int step = 0; step < (int) parameters.generateSteps; step++ ) {

        // 1) Generate prediction --------------------------------------
        Project( solver );

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
        std::cout << "+++++++ coefficients " << step << " +++++++" << std::endl;
        std::cout << coefficients;
#endif

        // 2) Save prediction in generated -----------------------------
        if ( step == 0 ) {
            // Existing obervations
            for ( size_t j = 0; j < nPrediction; j++ ) {
                generated.WriteRow( j, projection.Row( j ) );
                generatedTime.push_back( projection.Time()[ j ] );

                generatedCoef.WriteRow( j, coefficients.Row( j ) );
            }
        }
        // The 1-step ahead prediction
        generated.WriteRow( nPrediction + step,
                            projection.Row( nPrediction ) );

        generatedCoef.WriteRow( nPrediction + step,
                                coefficients.Row( nPrediction ) );

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
        // 6) Replace SMapClass.data with newDF for next Project()
        this->data = newDF; // JP is this a leak?

#ifdef DEBUG_ALL
        std::cout << "+++++++ newDF " << step << " +++++++" << std::endl;
        newDF.MaxRowPrint() = 5;
        std::cout << newDF;
#endif
    }

    // 7) Replace SMapClass.projection, coefficients with generated
    generated.Time()            = generatedTime;
    generated.TimeName()        = data.TimeName();
    generatedCoef.Time()        = generatedTime;
    generatedCoef.TimeName()    = data.TimeName();
    generatedCoef.ColumnNames() = coefficients.ColumnNames();
    generatedCoef.BuildColumnNameIndex();

    projection   = generated;     // JP is this a leak?
    coefficients = generatedCoef; // JP is this a leak?

#ifdef DEBUG_ALL
    std::cout << "+++++++ generated +++++++" << std::endl;
    generated.MaxRowPrint() = generated.NRows();
    std::cout << generated;
    std::cout << "<<<< SMapClass:: Generate()" << std::endl;
#endif
}

//----------------------------------------------------------------
// 
//----------------------------------------------------------------
void SMapClass::WriteOutput () {

    // Process SMap coefficients output
    // Set time from: data -> FormatOutput() -> FillTimes() -> projection
    if ( projection.Time().size() ) {
        coefficients.Time()     = projection.Time();
        coefficients.TimeName() = projection.TimeName();
    }
    // else { throw ? }  JP

    // coefficients column names
    std::vector<std::string> coefNames;
    if ( parameters.columnNames.size() and parameters.targetName.size() ) {
        coefNames.push_back( "C0" );

        if ( parameters.embedded ) {
            for ( auto colName : parameters.columnNames ) {
                std::stringstream coefName;
                coefName << "∂" << colName << "/∂" << parameters.targetName;
                coefNames.push_back( coefName.str() );
            }
        }
        else {
            for ( auto colName : embedding.ColumnNames() ) {
                std::stringstream coefName;
                coefName << "∂" << colName << "/∂" << parameters.targetName;
                coefNames.push_back( coefName.str() );
            }
        }
    }
    else {
        // Default: C0, C1, C2, ...
        for ( size_t col = 0; col < coefficients.NColumns(); col++ ) {
            std::stringstream coefName;
            coefName << "C" << col;
            coefNames.push_back( coefName.str() );
        }
    }
    coefficients.ColumnNames() = coefNames;

    // coefficients has Npred + Tp rows, but coef were written in first Npred
    // Create coefficient column vector with Tp nan rows at the
    // beginning/end of coefficients as in FormatOutput()
    std::valarray< double > coefColumnVec( NAN, coefficients.NRows() );

    // Copy/shift coefficients vectors
    std::slice slice_in = std::slice( 0, knn_neighbors.NRows(), 1 );
    std::slice slice_out;
    if ( parameters.Tp > -1 ) {
        slice_out = std::slice( parameters.Tp, knn_neighbors.NRows(), 1 );
    }
    else {
        slice_out = std::slice( 0, knn_neighbors.NRows() + parameters.Tp, 1 );
    }
    for ( size_t col = 0; col < coefficients.NColumns(); col++ ) {
        coefColumnVec[ slice_out ] = coefficients.Column( col )[ slice_in ];
        coefficients.WriteColumn( col, coefColumnVec );
    }

    if ( parameters.predictOutputFile.size() ) {
        projection.WriteData( parameters.pathOut,
                              parameters.predictOutputFile );
    }
    if ( parameters.SmapOutputFile.size() ) {
        coefficients.WriteData( parameters.pathOut, parameters.SmapOutputFile );
    }
}

//----------------------------------------------------------------
// Singular Value Decomposition : wrapper for Lapack_SVD()
//----------------------------------------------------------------
std::valarray < double > SVD( DataFrame    < double > A,
                              std::valarray< double > B ) {

    // NOTE: A elements are Row Major format
    // Convert A to column major for LAPACK dgelss()
    // a is the memory start location pointer to colMajorElements
    std::valarray < double > colMajorElements = A.ColumnMajorData();
    double *a = &( colMajorElements[0] );

    double *b = &( B[0] );

    std::valarray < double > C =
        Lapack_SVD( A.NRows(),     // number of rows
                    A.NColumns(),  // number of columns
                    a,             // A
                    b,             // b
                    1.E-9 );       // rcond

#ifdef DEBUG_ALL
    std::cout << "SVD------------------------\n";
    std::cout << "A ----------\n";
    std::cout << A << std::endl;
#endif

    return C;
}

//-------------------------------------------------------------------------
// subroutine dgelss() : LAPACK function call in Lapack_SVD()
//-----------------------------------------------------------------------
// DGELSS computes the minimum norm solution to a real linear least
// squares problem:
// 
// Minimize 2-norm(| b - A*x |).
// 
// using the singular value decomposition (SVD) of A. A is an M-by-N
// matrix which may be rank-deficient.
// 
// Several right hand side vectors b and solution vectors x can be
// handled in a single call; they are stored as the columns of the
// M-by-NRHS right hand side matrix B and the N-by-NRHS solution matrix X.
// 
// The effective rank of A is determined by treating as zero those
// singular values which are less than RCOND times the largest singular
// value.
// 
// INFO is INTEGER
//   = 0:  successful exit
//   < 0:  if INFO = -i, the i-th argument had an illegal value.
//   > 0:  the algorithm for computing the SVD failed to converge;
//         if INFO = i, i off-diagonal elements of an intermediate
//         bidiagonal form did not converge to zero.
//-----------------------------------------------------------------------
//-------------------------------------------------------------------------
// DOUBLE PRECISION = REAL*8 = c++ double
//-------------------------------------------------------------------------
extern "C" {

    void dgelss_( int    *M,
                  int    *N,
                  int    *NRHS,
                  double *A,
                  int    *LDA,
                  double *B,
                  int    *LDB,
                  double *S,
                  double *RCOND,
                  int    *RANK,
                  double *WORK,
                  int    *LWORK,
                  int    *INFO );
}

//-----------------------------------------------------------------------
// Wrapper for LAPACK dgelss_()
//-----------------------------------------------------------------------
std::valarray< double > Lapack_SVD( int     m, // rows in matrix
                                    int     n, // columns in matrix
                                    double *a, // ptr to top-left
                                    double *b,
                                    double  rcond )
{
    int N_SingularValues = m < n ? m : n;

    // s to hold singular values
    // MSVC BS: Have to use static const int size, or new
    double *s = new double[ N_SingularValues ];

    int lda  = m; // LDA >= max(1,M)
    int ldb  = m; // LDB >= max(1,max(M,N))
    int nrhs = 1;

    // Workspace and info variables:
    // MSVC BS: Have to use static const int size, or new
    int *iwork = new int[ 8 * N_SingularValues ];

    double workSize = 0;  // To query optimal work size
    int    lwork    = -1; // To query optimal work size
    int    info     = 0;  // return code
    int    rank     = 0;

#ifdef DEBUG_ALL
    std::cout << "m.row=" << m << " n.col=" << n << " lda=" << lda
              << " s.n=" << N_SingularValues << std::endl;

    for ( int i = 0; i < m*n; i++ ) {
        std::cout << a[i] << " ";
    } std::cout << std::endl;
#endif

    // Call dgelss with lwork = -1 to query optimal workspace size:
    dgelss_( &m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond,
             &rank, &workSize, &lwork, &info );

    if ( info ) {
        throw std::runtime_error( "Lapack_SVD(): dgelss failed on query.\n" );
    }

#ifdef DEBUG_ALL
    std::cout << "Optimal work size is " << workSize << std::endl;
#endif

    // Optimal workspace size is returned in workSize.
    // MSVC BS: Have to use static const int size, or new
    double *work = new double[ (size_t) workSize ];

    lwork = (int) workSize;

    // Call dgelss for SVD solution using lwork workSize:
    dgelss_( &m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond,
             &rank, work, &lwork, &info );

    if ( info ) {
        throw std::runtime_error( "Lapack_SVD(): dgelss failed.\n" );
    }

#ifdef DEBUG_ALL
    std::cout << "Solution: [ ";
    for ( auto i = 0; i < N_SingularValues; i++ ) {
        std::cout << b[i] << " ";
    } std::cout << "]" << std::endl;
#endif

    // Copy solution vector in b to C
    std::valarray< double > C( b, N_SingularValues );

    delete[] s;
    delete[] work;
    delete[] iwork;

    return C;
}
