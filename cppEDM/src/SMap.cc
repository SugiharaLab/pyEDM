
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
    
    auto maxLibit = std::max_element( parameters.library.begin(),
                                      parameters.library.end() );
    int maxLibIndex = *maxLibit; // int for compare to libRow int
    
    // Process each prediction row in neighbors : distances
    for ( size_t row = 0; row < Npred; row++ ) {
        
        double D_avg = knn_distances.Row( row ).sum() / parameters.knn;

        // Compute weight vector 
        std::valarray< double > w = std::valarray< double >( parameters.knn );
        if ( parameters.theta > 0 ) {
            w = std::exp( (-parameters.theta / D_avg) * knn_distances.Row(row) );
        }
        else {
            w = std::valarray< double >( 1, parameters.knn );
        }

        DataFrame< double > A = DataFrame< double >( parameters.knn,
                                                     parameters.E + 1 );
        std::valarray< double > B = std::valarray< double >( parameters.knn );

        // Populate matrix A (exp weighted future prediction), and
        // vector B (target BC's) for this row (observation).
        int    libRow;
        size_t libRowBase;
        
        for ( int k = 0; k < parameters.knn; k++ ) {
            libRowBase = knn_neighbors( row, k );
            libRow     = libRowBase + parameters.Tp;
            
            if ( libRow > maxLibIndex ) {
                // The knn index + Tp is outside the library domain
                // Can only happen if noNeighborLimit = true is used.
                if ( parameters.verbose ) {
                    std::stringstream msg;
                    msg << "SMap() in row " << row << " libRow " << libRow
                        << " exceeds library domain.\n";
                    std::cout << msg.str();
                }                
                // Use the neighbor at the 'base' of the trajectory
                B[ k ] = target[ libRowBase ];
            }
            else if ( libRow < 0 ) {
                B[ k ] = target[ 0 ];
            }
            else {
                B[ k ] = target[ libRow ];
            }

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

        B = w * B; // Weighted target vector

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
            std::pow( B - predictions[ row ], 2);
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
    
    // coefficients column names: C0, C1, C2, ...
    std::vector<std::string> coefNames;
    for ( size_t col = 0; col < coefficients.NColumns(); col++ ) {
        std::stringstream coefName;
        coefName << "C" << col;
        coefNames.push_back( coefName.str() );
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
