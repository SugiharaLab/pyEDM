
//-------------------------------------------------------------------------
// Eigen can internally call LAPACKE instead of Eigen::JacobiSVD()
// https://eigen.tuxfamily.org/dox/TopicUsingBlasLapack.html
//-------------------------------------------------------------------------
// #define EIGEN_USE_LAPACKE_STRICT
// EIGEN_USE_LAPACKE: Enables the use of external Lapack routines via
// the Lapacke C interface to Lapack (compatible with F77 LAPACK interface).
// EIGEN_USE_LAPACKE_STRICT: Same as EIGEN_USE_LAPACKE but algorithms of
// lower numerical robustness are disabled. This currently concerns only
// JacobiSVD which otherwise would be replaced by gesvd that is less robust
// than Jacobi rotations.
//
//-------------------------------------------------------------------------
// The Lapack SVD routine dgesdd_() can also be called directly without
// need for Eigen. An example is in etc/lapack_dgesdd.cc
//-------------------------------------------------------------------------

#include <Eigen/Dense>

#include "Common.h"
#include "Parameter.h"
#include "Embed.h"
#include "Neighbors.h"
#include "AuxFunc.h"

// forward declaration
std::valarray < double > SVD( DataFrame    < double > A,
                              std::valarray< double > B );

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
SMapValues SMap( DataFrame< double > data,
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
    DataEmbedNN dataEmbedNN = EmbedNN( data, std::ref( param ) );

    // Unpack the dataEmbedNN for convenience
    DataFrame<double>     dataIn     = dataEmbedNN.dataIn;
    DataFrame<double>     dataBlock  = dataEmbedNN.dataFrame;
    std::valarray<double> target_vec = dataEmbedNN.targetVec;
    Neighbors             neighbors  = dataEmbedNN.neighbors;
    
    // target_vec spans the entire dataBlock, subset targetLibVector
    // to library for row indexing used below:
    std::slice lib_i = std::slice( param.library[0], param.library.size(), 1 );
    std::valarray<double> targetLibVector = target_vec[ lib_i ];

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
        
        for ( size_t k = 0; k < param.knn; k++ ) {
            lib_row = neighbors.neighbors( row, k ) + param.Tp;
            
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
                B[ k ] = targetLibVector[ lib_row - param.Tp ];
            }
            else {
                B[ k ] = targetLibVector[ lib_row ];
            }

            A( k, 0 ) = w[ k ];
            for ( size_t j = 1; j < param.E + 1; j++ ) {
                A( k, j ) = w[k] * dataBlock( param.prediction[ row ], j );
            }
        }

        B = w * B; // Weighted target vector

        // Estimate linear mapping of predictions A onto target B
        std::valarray < double > C = SVD( A, B );

        // Prediction is local linear projection
        double prediction = C[ 0 ]; // C[ 0 ] is the bias term

        for ( size_t e = 1; e < param.E + 1; e++ ) {
            prediction = prediction + C[ e ] *
                dataBlock( param.prediction[ row ], e );
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
    // Observations & predictions
    DataFrame<double> dataOut = FormatOutput( param,
                                              predictions,
                                              const_predictions,
                                              target_vec,
                                              dataIn.Time(),
                                              dataIn.TimeName() );
    
    // Insert coefficients DataFrame column names: C0, C1, C2, ...
    for ( size_t col = 0; col < coefficients.NColumns(); col++ ) {
        std::stringstream coefName;
        coefName << "C" << col;
        coefficients.ColumnNames().push_back( coefName.str() );
    }

    // Coefficient output DataFrame
    DataFrame< double > coefOut = DataFrame< double >( N_row, param.E + 2 );
    
    // Prediction times
    std::vector<std::string> predTime( param.prediction.size() + param.Tp );
    
    if ( dataIn.Time().size() ) {
        FillTimes( param, dataIn.Time(), std::ref( predTime ) );
        coefOut.Time()     = predTime;
        coefOut.TimeName() = dataIn.TimeName();
    }
    // else { throw ? }  JP
    
    // Populate coefOut column names
    std::vector<std::string> coefNames;
    coefNames.push_back( "Time" );
    for ( size_t col = 0; col < coefficients.ColumnNames().size(); col++ ) {
        coefNames.push_back( coefficients.ColumnNames()[ col ] );
    }
    coefOut.ColumnNames() = coefNames;

    // Prediction row slice
    std::slice pred_i = std::slice( param.prediction[0], N_row, 1 );

    // Write coefficients to columns
    for ( size_t col = 1; col < coefOut.NColumns(); col++ ) {
        coefOut.WriteColumn( col, coefficients.Column( col - 1 ) );
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
// Singular Value Decomposition using Eigen C++ template library
//----------------------------------------------------------------
std::valarray < double >  SVD( DataFrame    < double > A_,
                               std::valarray< double > B_ ) {

    // Eigen::Map<> allows "raw" initialization from a pointer
    // The Map (A) is then a mapping to the memory location pointer (a)
    double *a = &(A_.Elements()[0]);

    // Eigen defaults to storing in column-major: use RowMajor flag
    Eigen::Map< Eigen::Matrix< double,
                               Eigen::Dynamic,
                               Eigen::Dynamic,
                               Eigen::RowMajor> > A( a,
                                                     A_.NRows(),
                                                     A_.NColumns() );

    double *b = &(B_[0]);
    Eigen::VectorXd B = Eigen::Map< Eigen::VectorXd >( b, B_.size() );

    //-------------------------------------------------------------------
    // https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
    //-------------------------------------------------------------------
    // The recommended method is the BDCSVD class, which scale well for
    // large problems and automatically fall-back to the JacobiSVD class
    // for smaller problems.
    //
    // JacobiSVD implements two-sided Jacobi iterations that are
    // numerically very accurate, fast for small matrices, but very
    // slow for larger ones.
    //
    // BDCSVD implements a recursive divide & conquer strategy on top of
    // an upper-bidiagonalization which remains fast for large problems.
    // Warning:  This algorithm is unlikely to provide accurate result when
    // compiled with unsafe math optimizations. For instance, this concerns
    // Intel's compiler (ICC), which perfroms such optimization by default
    // unless you compile with the -fp-model precise option. Likewise,
    // the -ffast-math option of GCC or clang will significantly degrade
    // the accuracy.
    // NOTE gcc: -ffast-math is not turned on by any -O option besides -Ofast
    
    Eigen::VectorXd C =
        A.bdcSvd( Eigen::ComputeThinU | Eigen::ComputeThinV ).solve( B );

    // Extract fit coefficients from Eigen::VectorXd to valarray<>
    std::valarray < double > C_( C.data(), A_.NColumns() );

#ifdef DEBUG_ALL
        double relative_error = (A*C - B).norm() / B.norm(); // L2 norm
        std::cout << "SVD relative error: " << relative_error << std::endl;
#endif

#ifdef DEBUG_ALL
    std::cout << "SVD------------------------\n";
    std::cout << "Eigen A ----------\n";
    std::cout << A << std::endl;
    std::cout << "Eigen B ----------\n";
    std::cout << B << std::endl;
#endif
    
    return C_;
}
