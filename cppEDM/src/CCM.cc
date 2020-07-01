
#include "CCM.h"

namespace EDM_CCM_Lock {
    std::mutex mtx;
    std::mutex q_mtx;
    std::queue< std::exception_ptr > exceptionQ;
}

//----------------------------------------------------------------
// forward declaration
//----------------------------------------------------------------
void CrossMap( SimplexClass   & S,         // input
               CrossMapValues & values );  // output

//----------------------------------------------------------------
// Constructor
// Initialise EDM::SimplexClass parent, and, 
// both mapping objects to the same initial data & parameters.
//----------------------------------------------------------------
CCMClass::CCMClass (
    DataFrame< double > & data, 
    Parameters          & parameters ) :
    SimplexClass( data, parameters ), // base class initialise
    colToTarget ( data, parameters ),
    targetToCol ( data, parameters )
{
    // Set targetToCol reverse mapping in SetupParameters()
}

//----------------------------------------------------------------
// Project : Polymorphic implementation
//----------------------------------------------------------------
void CCMClass::Project () {

    SetupParameters(); // Forward and reverse mapping objects

    // Compute all distances in SimplexClass objects
    // CrossMap() will take subsets of these for each library size
    // with calls to FindNeighbors(), Simplex() in CrossMap()
    colToTarget.PrepareEmbedding(); // embedding, target, RemovePartialData()
    targetToCol.PrepareEmbedding(); // embedding, target, RemovePartialData()

    colToTarget.Distances();        // allDistances, allLibRows
    targetToCol.Distances();        // allDistances, allLibRows

    CCM();                          // FindNeighbors(), Simplex()

    FormatOutput();
    WriteOutput();
}

//----------------------------------------------------------------
// CCM 
// To accomodate two threads running forward & inverse mapping
// CrossMap() is called with separate Simplex and CrossMapValues:
//     SimpexClass colToTarget;  column to target mapping
//     SimpexClass targetToCol;  target to column mapping
// These fill the respective EDM object CrossMapValues structs:
//     CrossMapValues colToTargetValues; 
//     CrossMapValues targetToColValues;
//----------------------------------------------------------------
void CCMClass::CCM () {

    if ( parameters.columnNames.size() > 1 ) {
        std::cout << "WARNING: CCM() Only the first column will be mapped.\n";
    }

#ifdef CCM_THREADED
    std::thread CrossMapColTarget( CrossMap,
                                   std::ref( colToTarget ),
                                   std::ref( colToTargetValues ) );

    std::thread CrossMapTargetCol( CrossMap,
                                   std::ref( targetToCol ),
                                   std::ref( targetToColValues ) );
    CrossMapColTarget.join();
    CrossMapTargetCol.join();

    // If thread threw exception, get from queue and rethrow
    if ( not EDM_CCM_Lock::exceptionQ.empty() ) {
        std::lock_guard<std::mutex> lck( EDM_CCM_Lock::q_mtx );

        // Take the first exception in the queue
        std::exception_ptr exceptionPtr = EDM_CCM_Lock::exceptionQ.front();

        // Unroll all other exception from the thread/loops
        while( not EDM_CCM_Lock::exceptionQ.empty() ) {
            EDM_CCM_Lock::exceptionQ.pop();
        }
        std::rethrow_exception( exceptionPtr );
    }
#else
    CrossMap( std::ref( colToTarget ), std::ref( colToTargetValues ) );
    CrossMap( std::ref( targetToCol ), std::ref( targetToColValues ) );
#endif
}

//----------------------------------------------------------------
// CrossMap()
// Thread worker function for CCM.
//----------------------------------------------------------------
void CrossMap( SimplexClass   & S,
               CrossMapValues & values )
{
    if ( S.parameters.verbose ) {
        std::lock_guard<std::mutex> lck( EDM_CCM_Lock::mtx );
        std::stringstream msg;
        msg << "CrossMap(): Simplex cross mapping from "
            << S.parameters.columnNames[0]
            << " to " << S.parameters.targetName << "  E=" << S.parameters.E
            << "  knn=" << S.parameters.knn << "  Library range: ["
            << S.parameters.libSizes_str << "] ";
        for ( size_t i = 0; i < S.parameters.librarySizes.size(); i++ ) {
            msg << S.parameters.librarySizes[ i ] << " ";
        } msg << std::endl << std::endl;
        std::cout << msg.str();
    }

    try {
    if ( S.parameters.E < 1 ) {
        std::lock_guard<std::mutex> lck( EDM_CCM_Lock::mtx );
        std::stringstream errMsg;
        errMsg << "CrossMap(): E = " << S.parameters.E << " is invalid.\n";
        throw std::runtime_error( errMsg.str() );
    }

    int shift = abs( S.parameters.tau ) * ( S.parameters.E - 1 );

    if ( shift >= (int) S.data.NRows() ) {
        std::lock_guard<std::mutex> lck( EDM_CCM_Lock::mtx );
        std::stringstream errMsg;
        errMsg << "CrossMap(): Number of data rows " << S.data.NRows()
               << " is not sufficient for tau*(E-1) = " << shift << ".\n";
        throw std::runtime_error( errMsg.str() );
    }

    size_t N_row = S.embedding.NRows();

    //-----------------------------------------------------------------
    // Set number of samples
    //-----------------------------------------------------------------
    size_t maxSamples;
    if ( S.parameters.randomLib ) {
        // Random samples from library
        maxSamples = S.parameters.subSamples;
    }
    else {
        // Contiguous samples up to the size of the library
        maxSamples = 1;
    }

    //-----------------------------------------------------------------
    // Create random number generator: DefaultRandomEngine
    //-----------------------------------------------------------------
    if ( S.parameters.randomLib ) {
        if ( S.parameters.seed == 0 ) {
            // Select a random seed
            typedef std::chrono::high_resolution_clock CCMclock;
            CCMclock::time_point beginning = CCMclock::now();
            CCMclock::duration   duration  = CCMclock::now() - beginning;
            S.parameters.seed = duration.count();
        }
    }
    std::default_random_engine DefaultRandomEngine( S.parameters.seed );

    //----------------------------------------------------------
    // Predictions
    //----------------------------------------------------------
    size_t predictionCount = 0;

    //----------------------------------------------------------
    // Loop for library sizes
    //----------------------------------------------------------
    for ( size_t libSize_i = 0;
                 libSize_i < S.parameters.librarySizes.size(); libSize_i++ ) {

        size_t libSize = S.parameters.librarySizes[ libSize_i ];

        // Create random RNG sampler for this libSize out of N_row
        std::uniform_int_distribution< size_t > distribution( 0, N_row - 1 );

#ifdef DEBUG_ALL
        {
        std::lock_guard<std::mutex> lck( EDM_CCM_Lock::mtx );
        std::cout << "N_row: " << N_row << " maxSamples: " << maxSamples
                  << " libSize: " << libSize
                  << " ------------------------------------------\n";
        }
#endif
        // Output statistics vectors
        std::valarray< double > rho ( maxSamples );
        std::valarray< double > RMSE( maxSamples );
        std::valarray< double > MAE ( maxSamples );

        //----------------------------------------------------------
        // Loop for subsamples
        //----------------------------------------------------------
        for ( size_t n = 0; n < maxSamples; n++ ) {

            //------------------------------------------------------
            // Generate library row indices for this subsample
            //------------------------------------------------------
            std::vector< size_t > lib_i( libSize );

            if ( S.parameters.randomLib ) {
                // Uniform random sample of rows
                if ( S.parameters.replacement ) {
                    // With replacement
                    for ( size_t i = 0; i < libSize; i++ ) {
                        lib_i[ i ] = distribution( DefaultRandomEngine );
                    }
                }
                else {
                    // Without replacement libSize elements from [0, N_row-1]
                    // NOTE: c++17 has the sample() function in <algorithm>
                    if ( libSize >= N_row ) {
                        std::stringstream errMsg;
                        errMsg << "CrossMap(): libSize=" << libSize
                               << " must be less than N_row=" << N_row
                               << " for random sample without replacement.";
                        throw std::runtime_error( errMsg.str() );
                    }

                    // unordered set to store samples
                    std::unordered_set< size_t > samples;
 
                    // Sample and insert unique values into samples
                    while( samples.size() < libSize ) {
                        size_t v = distribution( DefaultRandomEngine );
                        samples.insert( v );
                    }

                    // Copy samples into result
                    std::vector<size_t> result( samples.begin(), samples.end() );

                    lib_i = result; // Copy result to lib_i
                }
                // std::sort( lib_i.begin(), lib_i.end() ); // JP Why?
            }
            else {
                // Not random samples, contiguous samples increasing size
                if ( libSize >= N_row ) {
                    // library size exceeded, back down
                    lib_i.resize( N_row );
                    std::iota( lib_i.begin(), lib_i.end(), 0 );
                    libSize = N_row;

                    if ( S.parameters.verbose ) {
                        std::stringstream msg;
                        msg << "CCM(): Sequential library samples,"
                            << " max libSize is " << N_row
                            << ", libSize has been limited.\n";
                        std::cout << msg.str();
                    }
                }
                else {
                    // Contiguous blocks up to N_rows = maxSamples
                    if ( n + libSize < N_row ) {
                        std::iota( lib_i.begin(), lib_i.end(), n );
                    }
                    else {
                        // n + libSize > N_row, wrap around to data origin
                        std::vector< size_t > lib_start( N_row - n );
                        std::iota( lib_start.begin(), lib_start.end(), n );

                        size_t max_i = std::min( libSize-(N_row - n), N_row );
                        std::vector< size_t > lib_wrap( max_i );
                        std::iota( lib_wrap.begin(), lib_wrap.end(), 0 );

                        // Build new lib_i
                        lib_i = std::vector< size_t > ( lib_start );
                        lib_i.insert( lib_i.end(),
                                      lib_wrap.begin(),
                                      lib_wrap.end() );
                        libSize = lib_i.size();
                    }
                }
            }

#ifdef DEBUG_ALL
            {
            std::lock_guard<std::mutex> lck( EDM_CCM_Lock::mtx );
            std::cout << "lib_i: (" << lib_i.size() << ") ";
            for ( size_t i = 0; i < lib_i.size(); i++ ) {
                std::cout << lib_i[i] << " ";
            } std::cout << std::endl;
            }
#endif
            //----------------------------------------------------------
            // Local SimplexClass object for mapping
            //    Uses subset of CCMClass SimplexClass object
            //----------------------------------------------------------
            SimplexClass Simplex_( S.data, S.parameters );

            //----------------------------------------------------------
            // Subset Distances and lib row indices to lib_i
            //----------------------------------------------------------
            Simplex_.allLibRows = 
                S.allLibRows.DataFrameFromColumnIndex( lib_i );
            
            Simplex_.allDistances =
                S.allDistances.DataFrameFromColumnIndex( lib_i );            

            //----------------------------------------------------------
            // Cross mapping
            //----------------------------------------------------------
            Simplex_.GetTarget();
            Simplex_.FindNeighbors();
            Simplex_.Simplex();
            Simplex_.FormatOutput();

            VectorError ve = ComputeError(
                Simplex_.projection.VectorColumnName( "Observations" ),
                Simplex_.projection.VectorColumnName( "Predictions"  ) );

#ifdef DEBUG_ALL
            {
            std::lock_guard<std::mutex> lck( EDM_CCM_Lock::mtx );
            std::cout << "CCM Simplex -------- Column: ";
            std::cout << Simplex_.parameters.columnNames[0] << "  :  Target: "
                      << Simplex_.parameters.targetName << " --------\n";
            // Simplex_.projection.MaxRowPrint() = Simplex_.projection.NRows();
            // std::cout << Simplex_.projection;
            std::cout << "    rho " << ve.rho << "  RMSE " << ve.RMSE
                      << "  MAE " << ve.MAE << std::endl;
            }
#endif
            // Record values for these lib_i samples
            rho [ n ] = ve.rho;
            RMSE[ n ] = ve.RMSE;
            MAE [ n ] = ve.MAE;

            if ( S.parameters.includeData ) {
                // Save stats for this prediction
                std::valarray< double > predOutVec( 8 );
                predOutVec[ 0 ] = predictionCount + 1; // N
                predOutVec[ 1 ] = S.parameters.E;      // E
                predOutVec[ 2 ] = S.parameters.knn;    // nn
                predOutVec[ 3 ] = S.parameters.tau;    // tau
                predOutVec[ 4 ] = libSize;             // LibSize
                predOutVec[ 5 ] = ve.rho;              // rho
                predOutVec[ 6 ] = ve.RMSE;             // RMSE
                predOutVec[ 7 ] = ve.MAE;              // MAE

                values.PredictStats.WriteRow( predictionCount,
                                              predOutVec );
                // Save predictions
                values.Predictions.push_front( S.projection );
            }

            predictionCount++;
        } // for ( n = 0; n < maxSamples; n++ )

        std::valarray< double > statVec( 4 );
        statVec[ 0 ] = libSize;
        statVec[ 1 ] = rho.sum()  / maxSamples;
        statVec[ 2 ] = RMSE.sum() / maxSamples;
        statVec[ 3 ] = MAE.sum()  / maxSamples;

        values.LibStats.WriteRow( libSize_i, statVec );
    } // for ( libSize_i < parameters.librarySizes )
    } // try 
    catch(...) {
#ifdef CCM_THREADED
        // push exception pointer onto queue for main thread to catch
        std::lock_guard<std::mutex> lck( EDM_CCM_Lock::q_mtx );
        EDM_CCM_Lock::exceptionQ.push( std::current_exception() );
#else
        throw std::rethrow_exception( std::current_exception() );
#endif
    }
}

//-----------------------------------------------------------------
// Populate EDM::SimplexClass Parameters objects for CrossMap calls
//-----------------------------------------------------------------
void CCMClass::SetupParameters() {

    // Swap column : target in targetToCol.parameters
    targetToCol.parameters.columns_str = parameters.targetName;
    targetToCol.parameters.target_str  = parameters.columnNames[0];
    targetToCol.parameters.Validate();

    //------------------------------------------------------------------
    // DataFrames for output CrossMapValues structs in EDM object
    //------------------------------------------------------------------
    size_t maxSamples;
    if ( parameters.randomLib ) {
        // Random samples from library
        maxSamples = parameters.librarySizes.size() * parameters.subSamples;
    }
    else {
        // Contiguous samples up to the size of the library
        maxSamples = parameters.librarySizes.size();
    }

    DataFrame< double > PredictionStats1( maxSamples, 8,
                                          "N E nn tau LibSize rho RMSE MAE" );
    DataFrame< double > PredictionStats2( maxSamples, 8,
                                          "N E nn tau LibSize rho RMSE MAE" );

    DataFrame< double > LibStats1( parameters.librarySizes.size(), 4,
                                   "LibSize rho RMSE MAE" );
    DataFrame< double > LibStats2( parameters.librarySizes.size(), 4,
                                   "LibSize rho RMSE MAE" );

    // Instantiate Simplex CrossMapValues output structs and insert DataFrames
    colToTargetValues = CrossMapValues();
    targetToColValues = CrossMapValues();

    colToTargetValues.LibStats = LibStats1;
    targetToColValues.LibStats = LibStats2;

    if ( parameters.includeData ) {
        colToTargetValues.PredictStats = PredictionStats1;
        targetToColValues.PredictStats = PredictionStats2;
    }
}

//----------------------------------------------------------------
// 
//----------------------------------------------------------------
void CCMClass::FormatOutput () {
    // Create unified column names of output DataFrame
    std::stringstream libRhoNames;
    libRhoNames << "LibSize "
                << parameters.columnNames[0] <<":"<< parameters.targetName << " "
                << parameters.targetName     <<":"<< parameters.columnNames[0];

    // Allocate unified LibStats output DataFrame in EDM object
    allLibStats = DataFrame< double >( parameters.librarySizes.size(), 3,
                                       libRhoNames.str() );

    allLibStats.WriteColumn( 0, colToTargetValues.LibStats.Column( 0 ) );
    allLibStats.WriteColumn( 1, colToTargetValues.LibStats.Column( 1 ) );
    allLibStats.WriteColumn( 2, targetToColValues.LibStats.Column( 1 ) );
}

//----------------------------------------------------------------
// 
//----------------------------------------------------------------
void CCMClass::WriteOutput () {
    if ( parameters.predictOutputFile.size() ) {
        // Write to disk
        allLibStats.WriteData( parameters.pathOut,
                               parameters.predictOutputFile );
    }
}
