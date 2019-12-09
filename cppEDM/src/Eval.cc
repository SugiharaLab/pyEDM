
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>

#include "Common.h"

namespace EDM_Eval {
    // Thread Work Queue : Vector of int
    typedef std::vector< int > WorkQueue;
    
    // Thread exception_ptr queue
    std::queue< std::exception_ptr > embedDimExceptQ;
    std::queue< std::exception_ptr > predictIntExceptQ;
    std::queue< std::exception_ptr > predictNLExceptQ;
    
    // atomic counters for all threads
    std::atomic<std::size_t> tp_count_i   (0); // initialize to 0
    std::atomic<std::size_t> embed_count_i(0); // initialize to 0
    std::atomic<std::size_t> smap_count_i (0); // initialize to 0

    std::mutex mtx;
    std::mutex q_mtx;
}

//----------------------------------------------------------------
// Forward declaration:
// Worker thread for EmbedDimension()
//----------------------------------------------------------------
void EmbedThread( EDM_Eval::WorkQueue &workQ,
                  DataFrame< double > &data,
                  DataFrame< double > &E_rho,
                  std::string          lib,
                  std::string          pred,
                  int                  Tp,
                  int                  tau,
                  std::string          colNames,
                  std::string          targetName,
                  bool                 embedded,
                  bool                 verbose );

//----------------------------------------------------------------
// Forward declaration:
// Worker thread for PredictInterval()
//----------------------------------------------------------------
void PredictIntervalThread( EDM_Eval::WorkQueue &workQ,
                            DataFrame< double > &data,
                            DataFrame< double > &Tp_rho,
                            std::string          lib,
                            std::string          pred,
                            int                  E,
                            int                  tau,
                            std::string          colNames,
                            std::string          targetName,
                            bool                 embedded,
                            bool                 verbose );

//----------------------------------------------------------------
// Forward declaration:
// Worker thread for PredictNonLinear()
//----------------------------------------------------------------
void SMapThread( EDM_Eval::WorkQueue   &workQ,
                 DataFrame< double >   &data,
                 DataFrame< double >   &Theta_rho,
                 std::vector<double>    ThetaValues,
                 std::string            lib,
                 std::string            pred,
                 int                    E,
                 int                    Tp,
                 int                    knn,
                 int                    tau,
                 std::string            colNames,
                 std::string            targetName,
                 bool                   embedded,
                 bool                   verbose );

//----------------------------------------------------------------
// EmbedDimension() : Evaluate Simplex rho vs. dimension E
// API Overload 1: Explicit data file path/name
//     Implemented as a wrapper to API Overload 2:
//----------------------------------------------------------------
DataFrame<double> EmbedDimension( std::string pathIn,
                                  std::string dataFile,
                                  std::string pathOut,
                                  std::string predictFile,
                                  std::string lib,
                                  std::string pred,
                                  int         maxE,
                                  int         Tp,
                                  int         tau,
                                  std::string colNames,
                                  std::string targetName,
                                  bool        embedded,
                                  bool        verbose,
                                  unsigned    nThreads ) {

    // Create DataFrame (constructor loads data)
    DataFrame< double > *dataFrameIn =
        new DataFrame< double > ( pathIn, dataFile );
    
    DataFrame<double> E_rho = EmbedDimension( std::ref( *dataFrameIn ),
                                              pathOut,
                                              predictFile,
                                              lib,
                                              pred,
                                              maxE,
                                              Tp,
                                              tau,
                                              colNames,
                                              targetName,
                                              embedded,
                                              verbose,
                                              nThreads );

    delete dataFrameIn;
    
    return E_rho;
}

//----------------------------------------------------------------
// EmbedDimension() : Evaluate Simplex rho vs. dimension E
// API Overload 2: DataFrame provided
//----------------------------------------------------------------
DataFrame<double> EmbedDimension( DataFrame< double > &data,
                                  std::string          pathOut,
                                  std::string          predictFile,
                                  std::string          lib,
                                  std::string          pred,
                                  int                  maxE,
                                  int                  Tp,
                                  int                  tau,
                                  std::string          colNames,
                                  std::string          targetName,
                                  bool                 embedded,
                                  bool                 verbose,
                                  unsigned             nThreads ) {
    
    // Container for results
    DataFrame<double> E_rho( maxE, 2, "E rho" );

    // Build work queue
    EDM_Eval::WorkQueue workQ( maxE );

    // Insert dimension values into work queue
    for ( auto i = 0; i < maxE; i++ ) {
        workQ[ i ] = i + 1;
    }

    unsigned maxThreads = std::thread::hardware_concurrency();
    if ( maxThreads < nThreads ) { nThreads = maxThreads; }
    if ( nThreads > maxE       ) { nThreads = maxE;       }
    
    // thread container
    std::vector< std::thread > threads;
    for ( unsigned i = 0; i < nThreads; i++ ) {
    
        threads.push_back( std::thread( EmbedThread,
                                        std::ref( workQ ),
                                        std::ref( data ),
                                        std::ref( E_rho ),
                                        lib,
                                        pred,
                                        Tp,
                                        tau,
                                        colNames,
                                        targetName,
                                        embedded,
                                        verbose ) );
    }
    
    // join threads
    for ( auto &thrd : threads ) {
        thrd.join();
    }
    
    // If thread threw exception, get from queue and rethrow
    if ( not EDM_Eval::embedDimExceptQ.empty() ) {
        std::lock_guard<std::mutex> lck( EDM_Eval::q_mtx );

        // Take the first exception in the queue
        std::exception_ptr exceptionPtr = EDM_Eval::embedDimExceptQ.front();

        // Unroll all other exception from the thread/loops
        while( not EDM_Eval::embedDimExceptQ.empty() ) {
            // JP When do these exception_ptr get deleted? Is it a leak?
            EDM_Eval::embedDimExceptQ.pop();
        }
        std::rethrow_exception( exceptionPtr );
    }
    
    if ( predictFile.size() ) {
        E_rho.WriteData( pathOut, predictFile );
    }
    
    return E_rho;
}

//----------------------------------------------------------------
// Worker thread for EmbedDimension()
//----------------------------------------------------------------
void EmbedThread( EDM_Eval::WorkQueue &workQ,
                  DataFrame< double > &data,
                  DataFrame< double > &E_rho,
                  std::string          lib,
                  std::string          pred,
                  int                  Tp,
                  int                  tau,
                  std::string          colNames,
                  std::string          targetName,
                  bool                 embedded,
                  bool                 verbose )
{
    
    std::size_t i = std::atomic_fetch_add( &EDM_Eval::embed_count_i,
                                           std::size_t(1) );
    
    while( i < workQ.size() ) {
        
        // WorkQueue stores E
        int E = workQ[ i ];

        // Simplex() -> EmbedNN() -> Embed() -> DeletePartialDataRows()
        // In a multthreaded application we need to pass a unique copy of
        // the data so that DeletePartialDataRows() is not recursively
        // applied to the same data frame.
        DataFrame< double > localData( data );

        try {
            DataFrame<double> S = Simplex( std::ref( localData ),
                                           "",          // pathOut,
                                           "",          // predictFile,
                                           lib,
                                           pred,
                                           E,
                                           Tp,
                                           0,           // knn
                                           tau,
                                           0,           // exclusionRadius
                                           colNames,
                                           targetName,
                                           embedded,
                                           false,       // const_predict
                                           verbose );
            
            VectorError ve = ComputeError( S.VectorColumnName("Observations"),
                                           S.VectorColumnName("Predictions"));

            E_rho.WriteRow( i, std::valarray<double>({ (double) E, ve.rho }));
        
            if ( verbose ) {
                std::lock_guard<std::mutex> lck( EDM_Eval::mtx );
                std::cout << "EmbedThread() workQ[" << workQ[i] << "]  E " << E 
                          << "  rho " << ve.rho << "  RMSE " << ve.RMSE
                          << "  MAE " << ve.MAE << std::endl << std::endl;
            }
        }
        catch(...) {
            // push exception pointer onto queue for main thread to catch
            std::lock_guard<std::mutex> lck( EDM_Eval::q_mtx );
            EDM_Eval::embedDimExceptQ.push( std::current_exception() );
        }
        
        i = std::atomic_fetch_add(&EDM_Eval::embed_count_i, std::size_t(1));
    }
    
    // Reset counter
    std::atomic_store( &EDM_Eval::embed_count_i, std::size_t(0) );
}

//-----------------------------------------------------------------
// PredictInterval() : Evaluate Simplex rho vs. predict interval Tp
// API Overload 1: Explicit data file path/name
//     Implemented as a wrapper to API Overload 2: 
//-----------------------------------------------------------------
DataFrame<double> PredictInterval( std::string pathIn,
                                   std::string dataFile,
                                   std::string pathOut,
                                   std::string predictFile,
                                   std::string lib,
                                   std::string pred,
                                   int         maxTp,
                                   int         E,
                                   int         tau,
                                   std::string colNames,
                                   std::string targetName,
                                   bool        embedded,
                                   bool        verbose,
                                   unsigned    nThreads ) {
    
    // Create DataFrame (constructor loads data)
    DataFrame< double > *dataFrameIn =
        new DataFrame< double > ( pathIn, dataFile );
    
    DataFrame<double> Tp_rho = PredictInterval( std::ref( *dataFrameIn ),
                                                pathOut,
                                                predictFile,
                                                lib,
                                                pred,
                                                maxTp,
                                                E,
                                                tau,
                                                colNames,
                                                targetName,
                                                embedded,
                                                verbose );
    delete dataFrameIn;
    
    return Tp_rho;
}

//-----------------------------------------------------------------
// PredictInterval() : Evaluate Simplex rho vs. predict interval Tp
// API Overload 2: DataFrame provided
//-----------------------------------------------------------------
DataFrame<double> PredictInterval( DataFrame< double > &data,
                                   std::string          pathOut,
                                   std::string          predictFile,
                                   std::string          lib,
                                   std::string          pred,
                                   int                  maxTp,
                                   int                  E,
                                   int                  tau,
                                   std::string          colNames,
                                   std::string          targetName,
                                   bool                 embedded,
                                   bool                 verbose,
                                   unsigned             nThreads ) {
    
    // Container for results
    DataFrame<double> Tp_rho( maxTp, 2, "Tp rho" );

    // Build work queue
    EDM_Eval::WorkQueue workQ( maxTp );

    // Insert Tp values into work queue
    for ( auto i = 0; i < maxTp; i++ ) {
        workQ[ i ] = i + 1;
    }

    unsigned maxThreads = std::thread::hardware_concurrency();
    if ( maxThreads < nThreads ) { nThreads = maxThreads; }
    if ( nThreads   > maxTp    ) { nThreads = maxTp;      }
    
    // thread container
    std::vector< std::thread > threads;
    for ( unsigned i = 0; i < nThreads; ++i ) {
        threads.push_back( std::thread( PredictIntervalThread,
                                        std::ref( workQ ),
                                        std::ref( data ),
                                        std::ref( Tp_rho ),
                                        lib,
                                        pred,
                                        E,
                                        tau,
                                        colNames,
                                        targetName,
                                        embedded,
                                        verbose ) );
    }
    
    // join threads
    for ( auto &thrd : threads ) {
        thrd.join();
    }
    
    // If thread threw exception, get from queue and rethrow
    if ( not EDM_Eval::predictIntExceptQ.empty() ) {
        std::lock_guard<std::mutex> lck( EDM_Eval::q_mtx );

        // Take the first exception in the queue
        std::exception_ptr exceptionPtr = EDM_Eval::predictIntExceptQ.front();

        // Unroll all other exception from the thread/loops
        while( not EDM_Eval::predictIntExceptQ.empty() ) {
            // JP When do these exception_ptr get deleted? Is it a leak?
            EDM_Eval::predictIntExceptQ.pop();
        }
        std::rethrow_exception( exceptionPtr );
    }
    
    if ( predictFile.size() ) {
        Tp_rho.WriteData( pathOut, predictFile );
    }

    return Tp_rho;
}

//----------------------------------------------------------------
// Worker thread for PredictInterval()
//----------------------------------------------------------------
void PredictIntervalThread( EDM_Eval::WorkQueue &workQ,
                            DataFrame< double > &data,
                            DataFrame< double > &Tp_rho,
                            std::string          lib,
                            std::string          pred,
                            int                  E,
                            int                  tau,
                            std::string          colNames,
                            std::string          targetName,
                            bool                 embedded,
                            bool                 verbose )
{
    std::size_t i = std::atomic_fetch_add( &EDM_Eval::tp_count_i,
                                           std::size_t(1) );
    
    while( i < workQ.size() ) {
        
        // WorkQueue stores Tp
        int Tp = workQ[ i ];
        
        // In a multthreaded application we need to pass a unique copy of
        // the data so that DeletePartialDataRows() is not recursively
        // applied to the same data frame.
        DataFrame< double > localData( data );

        try {
            DataFrame<double> S = Simplex( std::ref( localData ),
                                           "",          // pathOut,
                                           "",          // predictFile,
                                           lib,
                                           pred,
                                           E,
                                           Tp,
                                           0,           // knn
                                           tau,
                                           0,           // exclusionRadius
                                           colNames,
                                           targetName,
                                           embedded,
                                           false,       // const_pred
                                           verbose );
            
            VectorError ve = ComputeError( S.VectorColumnName("Observations"),
                                           S.VectorColumnName("Predictions"));

            Tp_rho.WriteRow( i, std::valarray<double>({ (double) Tp, ve.rho }));
        
            if ( verbose ) {
                std::lock_guard<std::mutex> lck( EDM_Eval::mtx );
                std::cout << "PredictIntervalThread() workQ[" << workQ[i]
                          << "]  Tp " << Tp 
                          << "  rho " << ve.rho << "  RMSE " << ve.RMSE
                          << "  MAE " << ve.MAE << std::endl << std::endl;
            }
        }
        catch(...) {
            // push exception pointer onto queue for main thread to catch
            std::lock_guard<std::mutex> lck( EDM_Eval::q_mtx );
            EDM_Eval::predictIntExceptQ.push( std::current_exception() );
        }
        
        i = std::atomic_fetch_add( &EDM_Eval::tp_count_i, std::size_t(1) );
    }
    
    // Reset counter
    std::atomic_store( &EDM_Eval::tp_count_i, std::size_t(0) );    
}

//----------------------------------------------------------------
// PredictNonlinear() : Smap rho vs. localisation parameter theta
// API Overload 1: Explicit data file path/name
//     Implemented as a wrapper to API Overload 2: 
//----------------------------------------------------------------
DataFrame<double> PredictNonlinear( std::string pathIn,
                                    std::string dataFile,
                                    std::string pathOut,
                                    std::string predictFile,
                                    std::string lib,
                                    std::string pred,
                                    std::string theta,
                                    int         E,
                                    int         Tp,
                                    int         knn,
                                    int         tau,
                                    std::string colNames,
                                    std::string targetName,
                                    bool        embedded,
                                    bool        verbose,
                                    unsigned    nThreads ) {
    
    // Create DataFrame (constructor loads data)
    DataFrame< double > *dataFrameIn =
        new DataFrame< double > ( pathIn, dataFile );
    
    DataFrame< double > Theta_rho = PredictNonlinear( std::ref( *dataFrameIn ),
                                                      pathOut,
                                                      predictFile,
                                                      lib,
                                                      pred,
                                                      theta,
                                                      E,
                                                      Tp,
                                                      knn,
                                                      tau,
                                                      colNames,
                                                      targetName,
                                                      embedded,
                                                      verbose );
    delete dataFrameIn;
    
    return Theta_rho;
}

//----------------------------------------------------------------
// PredictNonlinear() : Smap rho vs. localisation parameter theta
// API Overload 2: DataFrame provided
//----------------------------------------------------------------
DataFrame<double> PredictNonlinear( DataFrame< double > &data,
                                    std::string          pathOut,
                                    std::string          predictFile,
                                    std::string          lib,
                                    std::string          pred,
                                    std::string          theta,
                                    int                  E,
                                    int                  Tp,
                                    int                  knn,
                                    int                  tau,
                                    std::string          colNames,
                                    std::string          targetName,
                                    bool                 embedded,
                                    bool                 verbose,
                                    unsigned             nThreads ) {

    std::vector<double> ThetaValues( { 0.01, 0.1, 0.3, 0.5, 0.75, 1,
                                       1.5, 2, 3, 4, 5, 6, 7, 8, 9 } );

   if ( theta.size() ) {
       // Use theta values passed in as parameter string
       ThetaValues.clear();
        
       std::vector<std::string> theta_vec = SplitString( theta, " \t,\n" );
       
       try {
           for ( auto ci =  theta_vec.begin(); ci != theta_vec.end(); ++ci ) {
               ThetaValues.push_back( std::stod( *ci ) );
           }
       }
       catch ( const std::invalid_argument &ia ) {
           std::stringstream errMsg;
           errMsg << "PredictNonlinear(): Unable to convert theta ["
                  << ia.what() << "] to numeric.";
           throw std::runtime_error( errMsg.str() );
       }
   }
   
    // Container for results
    DataFrame<double> Theta_rho( ThetaValues.size(), 2, "Theta rho" );

    // Build work queue
    EDM_Eval::WorkQueue workQ( ThetaValues.size() );

    // Insert ThetaValues indexes into work queue
    for ( auto i = 0; i < ThetaValues.size(); i++ ) {
        workQ[ i ] = i;
    }

    unsigned maxThreads = std::thread::hardware_concurrency();
    if ( maxThreads < nThreads           ) { nThreads = maxThreads;         }
    if ( nThreads   > ThetaValues.size() ) { nThreads = ThetaValues.size(); }
    
    // thread container
    std::vector< std::thread > threads;
    for ( unsigned i = 0; i < nThreads; ++i ) {
        threads.push_back( std::thread( SMapThread,
                                        std::ref( workQ ),
                                        std::ref( data ),
                                        std::ref( Theta_rho ),
                                        ThetaValues,
                                        lib,
                                        pred,
                                        E,
                                        Tp,
                                        knn,
                                        tau,
                                        colNames,
                                        targetName,
                                        embedded,
                                        verbose ) );
    }
    
    // join threads
    for ( auto &thrd : threads ) {
        thrd.join();
    }

    // If thread threw exception, get from queue and rethrow
    if ( not EDM_Eval::predictNLExceptQ.empty() ) {
        std::lock_guard<std::mutex> lck( EDM_Eval::q_mtx );

        // Take the first exception in the queue
        std::exception_ptr exceptionPtr = EDM_Eval::predictNLExceptQ.front();

        // Unroll all other exception from the thread/loops
        while( not EDM_Eval::predictNLExceptQ.empty() ) {
            // JP When do these exception_ptr get deleted? Is it a leak?
            EDM_Eval::predictNLExceptQ.pop();
        }
        std::rethrow_exception( exceptionPtr );
    }
    
    if ( predictFile.size() ) {
        Theta_rho.WriteData( pathOut, predictFile );
    }
    
    return Theta_rho;
}

//----------------------------------------------------------------
// Worker thread for PredictNonlinear()
//----------------------------------------------------------------
void SMapThread( EDM_Eval::WorkQueue   &workQ,
                 DataFrame< double >   &data,
                 DataFrame< double >   &Theta_rho,
                 std::vector<double>    ThetaValues,
                 std::string            lib,
                 std::string            pred,
                 int                    E,
                 int                    Tp,
                 int                    knn,
                 int                    tau,
                 std::string            colNames,
                 std::string            targetName,
                 bool                   embedded,
                 bool                   verbose )
{
    
    std::size_t i =
        std::atomic_fetch_add( &EDM_Eval::smap_count_i, std::size_t(1) );

    while( i < workQ.size() ) {
        
        double theta = ThetaValues[ workQ[ i ] ];  
        
        // In a multthreaded application we need to pass a unique copy of
        // the data so that DeletePartialDataRows() is not recursively
        // applied to the same data frame.
        DataFrame< double > localData( data );

        try {
            SMapValues S = SMap( std::ref( localData ),
                                 "",
                                 "",        // predictFile
                                 lib,
                                 pred,
                                 E,
                                 Tp,
                                 knn,
                                 tau,
                                 theta,
                                 0,         // exclusionRadius
                                 colNames,
                                 targetName,
                                 "",        // smapFile
                                 "",        // derivatives
                                 embedded,
                                 false,     // const_predict
                                 verbose );
            
            DataFrame< double > predictions  = S.predictions;
            DataFrame< double > coefficients = S.coefficients;
            
            VectorError ve = ComputeError(
                predictions.VectorColumnName( "Observations" ),
                predictions.VectorColumnName( "Predictions"  ) );
            
            Theta_rho.WriteRow( i, std::valarray<double>({ theta, ve.rho }));
            
            if ( verbose ) {
                std::lock_guard<std::mutex> lck( EDM_Eval::mtx );
                std::cout << "Theta " << theta
                          << "  rho " << ve.rho << "  RMSE " << ve.RMSE
                          << "  MAE " << ve.MAE << std::endl << std::endl;
            }
        }
        catch(...) {
            // push exception pointer onto queue for main thread to catch
            std::lock_guard<std::mutex> lck( EDM_Eval::q_mtx );
            EDM_Eval::predictNLExceptQ.push( std::current_exception() );
        }
        
        i = std::atomic_fetch_add( &EDM_Eval::smap_count_i, std::size_t(1) );
    }
    
    // Reset counter
    std::atomic_store( &EDM_Eval::smap_count_i, std::size_t(0) );
}
