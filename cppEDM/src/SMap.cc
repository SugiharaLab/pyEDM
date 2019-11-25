
#include "Common.h"
#include "Parameter.h"
#include "Embed.h"
#include "Neighbors.h"
#include "AuxFunc.h"

// forward declarations
std::valarray < double > SVD( DataFrame    < double > A,
                              std::valarray< double > B );

std::valarray< double >  Lapack_SVD( int     m,
                                     int     n,
                                     double *a,
                                     double *b,
                                     double  rcond );

//----------------------------------------------------------------
// Overload 1: Explicit data file path/name
//   Implemented as a wrapper to API Overload 2:
//----------------------------------------------------------------
SMapValues SMap( std::string pathIn,
                 std::string dataFile,
                 std::string pathOut,
                 std::string predictFile,
                 std::string lib,
                 std::string pred,
                 int         E,
                 int         Tp,
                 int         knn,
                 int         tau,
                 double      theta,
                 int         exclusionRadius,
                 std::string columns,
                 std::string target,
                 std::string smapFile,
                 std::string derivatives,
                 bool        embedded,
                 bool        const_predict,
                 bool        verbose )
{
    // DataFrame constructor loads data
    DataFrame< double > dataFrameIn( pathIn, dataFile );
    
    SMapValues SMapOutput = SMap( dataFrameIn, pathOut, predictFile,
                                  lib, pred, E, Tp, knn, tau, theta,
                                  exclusionRadius,
                                  columns, target, smapFile, derivatives, 
                                  embedded, const_predict, verbose );
    return SMapOutput;
}

//----------------------------------------------------------------
// Overload 2: DataFrame provided
//----------------------------------------------------------------
SMapValues SMap( DataFrame< double > &data,
                 std::string pathOut,
                 std::string predictFile,
                 std::string lib,
                 std::string pred,
                 int         E,
                 int         Tp,
                 int         knn,
                 int         tau,
                 double      theta,
                 int         exclusionRadius,
                 std::string columns,
                 std::string target,
                 std::string smapFile,
                 std::string derivatives,
                 bool        embedded,
                 bool        const_predict,
                 bool        verbose )
{

    Parameters param = Parameters( Method::SMap, "", "",
                                   pathOut, predictFile,
                                   lib, pred, E, Tp, knn, tau, theta,
                                   exclusionRadius, columns, target,
                                   embedded, const_predict, verbose,
                                   smapFile, "", derivatives );

    //----------------------------------------------------------
    // Load data, Embed, compute Neighbors
    //----------------------------------------------------------
    DataEmbedNN dataEmbedNN = EmbedNN( &data, std::ref( param ) );

    // Unpack the dataEmbedNN for convenience
    DataFrame<double>    *dataInRef  = dataEmbedNN.dataIn;
    DataFrame<double>     dataBlock  = dataEmbedNN.dataFrame;
    std::valarray<double> target_vec = dataEmbedNN.targetVec;
    Neighbors             neighbors  = dataEmbedNN.neighbors;
    
    DataFrame<double> &dataIn = std::ref( *dataInRef );

    //----------------------------------------------------------
    // SMap projection
    //----------------------------------------------------------
    size_t library_N_row = param.library.size();
    size_t predict_N_row = param.prediction.size();
    size_t N_row         = neighbors.neighbors.NRows();

    if ( predict_N_row != N_row ) {
        std::stringstream errMsg;
        errMsg << "SMap(): Number of prediction rows (" << predict_N_row
               << ") does not match the number of neighbor rows ("
               << N_row << ").\n";
        throw std::runtime_error( errMsg.str() );
    }
    if ( neighbors.distances.NColumns() != param.knn ) {
        std::stringstream errMsg;
        errMsg << "SMap(): Number of neighbor columns ("
               << neighbors.distances.NColumns()
               << ") does not match knn (" << param.knn << ").\n";
        throw std::runtime_error( errMsg.str() );        
    }
    
    std::valarray< double > predictions = std::valarray< double >( N_row );

    // Init coefficients to NAN ?
    DataFrame< double > coefficients = DataFrame< double >( N_row,
                                                            param.E + 1 );
    DataFrame< double > derivative;
    DataFrame< double > tangents;

    //------------------------------------------------------------
    // Process each prediction row
    //------------------------------------------------------------
    for ( size_t row = 0; row < N_row; row++ ) {
        
        double D_avg = neighbors.distances.Row( row ).sum() / param.knn;

        // Compute weight vector 
        std::valarray< double > w = std::valarray< double >( param.knn );
        if ( param.theta > 0 ) {
            w = std::exp( (-param.theta/D_avg) * neighbors.distances.Row(row) );
        }
        else {
            w = std::valarray< double >( 1, param.knn );
        }

        DataFrame< double >     A = DataFrame< double >(param.knn, param.E + 1);
        std::valarray< double > B = std::valarray< double >( param.knn );

        // Populate matrix A (exp weighted future prediction), and
        // vector B (target BC's) for this row (observation).
        size_t lib_row;
        size_t lib_row_base;
        
        for ( size_t k = 0; k < param.knn; k++ ) {
            lib_row_base = neighbors.neighbors( row, k );
            lib_row      = lib_row_base + param.Tp;
            
            if ( lib_row > library_N_row ) {
                // The knn index + Tp is outside the library domain
                // Can only happen if noNeighborLimit = true is used.
                if ( param.verbose ) {
                    std::stringstream msg;
                    msg << "SMap() in row " << row << " libRow " << lib_row
                        << " exceeds library domain.\n";
                    std::cout << msg.str();
                }                
                // Use the neighbor at the 'base' of the trajectory
                B[ k ] = target_vec[ lib_row_base ];
            }
            else {
                B[ k ] = target_vec[ lib_row ];
            }

            //---------------------------------------------------------------
            // Linear system coefficient matrix
            //---------------------------------------------------------------
            // NOTE: The matrix A has a (weighted) constant (1) first column
            //       to enable a linear intercept/bias term.
            // NOTE: The dataBlock does not have a time vector, and only
            //       has columns from the embedding.  So the coefficient
            //       matrix A has E+1 columns, while the dataBlock has E.
            //---------------------------------------------------------------
            A( k, 0 ) = w[ k ]; // Intercept bias terms in column 0 (weighted)
            
            for ( size_t j = 1; j < param.E + 1; j++ ) {
                A( k, j ) = w[k] * dataBlock( lib_row_base, j-1 );
            }
        }

        B = w * B; // Weighted target vector

        // Estimate linear mapping of predictions A onto target B
        std::valarray < double > C = SVD( A, B );

        // Prediction is local linear projection
        double prediction = C[ 0 ]; // C[ 0 ] is the bias term
        
        for ( size_t e = 1; e < param.E + 1; e++ ) {
            prediction = prediction +
                         C[ e ] * dataBlock( param.prediction[ row ], e-1 );
        }

        predictions[ row ] = prediction;
        coefficients.WriteRow( row, C );

    } // for ( row = 0; row < predict_N_row; row++ )

    // non "predictions" X(t+1) = X(t) if const_predict specified
    std::valarray< double > const_predictions( 0., N_row );
    if ( param.const_predict ) {
        std::slice pred_slice =
            std::slice( param.prediction[ 0 ], param.prediction.size(), 1 );
        
        const_predictions = target_vec[ pred_slice ];
    }
    
    //-----------------------------------------------------
    // Derivatives
    //-----------------------------------------------------


    //----------------------------------------------------
    // Ouput
    //----------------------------------------------------
    // Observations & predictions: Adjust rows/nan for Tp
    DataFrame<double> dataOut = FormatOutput( param,
                                              predictions,
                                              const_predictions,
                                              target_vec,
                                              dataIn.Time(),
                                              dataIn.TimeName() );
    
    // Coefficient output DataFrame: N_row + Tp rows
    DataFrame< double > coefOut = DataFrame< double >( dataOut.NRows(),
                                                       param.E + 1 );
    
    // Set time from: dataIn -> FormatOutput() -> FillTimes() -> dataOut
    if ( dataOut.Time().size() ) {
        coefOut.Time()     = dataOut.Time();
        coefOut.TimeName() = dataOut.TimeName();
    }
    // else { throw ? }  JP
    
    // Populate coefOut column names: C0, C1, C2, ...
    std::vector<std::string> coefNames;
    for ( size_t col = 0; col < coefficients.NColumns(); col++ ) {
        std::stringstream coefName;
        coefName << "C" << col;
        coefNames.push_back( coefName.str() );
    }
    coefOut.ColumnNames() = coefNames;

    // coefficients have N_row's; coefOut has N_row + Tp
    // Create coefficient column vector with Tp nan rows at the
    // beginning of coefOut as in FormatOutput()
    std::valarray<double> coefColumnVec( NAN, dataOut.NRows() );
    
    // Copy coefficients vectors into coefOut
    std::slice coef_i = std::slice( param.Tp, N_row, 1 );
    for ( size_t col = 0; col < coefOut.NColumns(); col++ ) {
        coefColumnVec[ coef_i ] = coefficients.Column( col );
        coefOut.WriteColumn( col, coefColumnVec );
    }
    
    if ( param.predictOutputFile.size() ) {
        // Write predictions to disk
        dataOut.WriteData( param.pathOut, param.predictOutputFile );
    }
    if ( param.SmapOutputFile.size() ) {
        // Write Smap coefficients to disk
        coefOut.WriteData( param.pathOut, param.SmapOutputFile );
    }
    
    SMapValues values = SMapValues();
    values.predictions  = dataOut;
    values.coefficients = coefOut;

    return values;
}

//----------------------------------------------------------------
// Singular Value Decomposition
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
// subroutine dgelss()
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
//
//-----------------------------------------------------------------------
std::valarray< double > Lapack_SVD( int     m, // number of rows in matrix
                                    int     n, // number of columns in matrix
                                    double *a, // pointer to top-left corner
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

    for ( size_t i = 0; i < m*n; i++ ) {
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

    delete s;
    delete work;
    delete iwork;
    
    return C;
}
