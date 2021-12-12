
#include "Parameter.h"

//----------------------------------------------------------------
// Constructor
// Default values set in Parameter.h declaration
//----------------------------------------------------------------
Parameters::Parameters(
    Method      method,
    std::string pathIn,
    std::string dataFile,
    std::string pathOut,
    std::string predictOutputFile,

    std::string lib_str,
    std::string pred_str,

    int         E,
    int         Tp,
    int         knn,
    int         tau,
    double      theta,
    int         exclusionRadius,

    std::string columns_str,
    std::string target_str,

    bool        embedded,
    bool        const_predict,
    bool        verbose,

    std::vector<bool> validLib,

    int         generateSteps,
    bool        parameterList,
    
    std::string SmapOutputFile,
    std::string blockOutputFile,

    int         multiviewEnsemble,
    int         multiviewD,
    bool        multiviewTrainLib,
    bool        multiviewExcludeTarget,

    std::string libSizes_str,
    int         subSamples,
    bool        randomLib,
    bool        replacement,
    unsigned    seed,
    bool        includeData
    ) :
    // Variable initialization from Parameters arguments
    method           ( method ),
    pathIn           ( pathIn ),
    dataFile         ( dataFile ),
    pathOut          ( pathOut ),
    predictOutputFile( predictOutputFile ),

    lib_str          ( lib_str ),
    pred_str         ( pred_str ),

    E                ( E ),
    Tp               ( Tp ),
    knn              ( knn ),
    tau              ( tau ),
    theta            ( theta ),
    exclusionRadius  ( exclusionRadius ),

    columns_str      ( columns_str ),
    target_str       ( target_str ),

    embedded         ( embedded ),
    const_predict    ( const_predict ),
    verbose          ( verbose ),

    validLib         ( validLib ),

    generateSteps    ( generateSteps ),
    parameterList    ( parameterList ),

    SmapOutputFile   ( SmapOutputFile ),
    blockOutputFile  ( blockOutputFile ),

    multiviewEnsemble     ( multiviewEnsemble ),
    multiviewD            ( multiviewD ),
    multiviewTrainLib     ( multiviewTrainLib ),
    multiviewExcludeTarget( multiviewExcludeTarget ),

    libSizes_str     ( libSizes_str ),
    subSamples       ( subSamples ),
    randomLib        ( randomLib ),
    replacement      ( replacement ),
    seed             ( seed ),
    includeData      ( includeData ),

    // Set validated flag and instantiate Version
    validated        ( false ),
    version          ( 1, 10, 0, "2021-12-11" )
{
    // Constructor code
    if ( method != Method::None ) {

        Validate();
        FillMap();

        if ( verbose ) {
            version.ShowVersion();
        }
    }
}

//----------------------------------------------------------------
// Destructor
//----------------------------------------------------------------
Parameters::~Parameters() {}

//----------------------------------------------------------------
// Generate library and prediction indices.
// Parameter validation and initialization.
//----------------------------------------------------------------
void Parameters::Validate() {

    validated = true;

    if ( not embedded and tau == 0 ) {
        std::string errMsg( "Parameters::Validate(): "
                            "tau must be non-zero.\n" );
        throw std::runtime_error( errMsg );
    }

    //--------------------------------------------------------------
    // Convert multi argument parameters from string to vectors
    //--------------------------------------------------------------

    //--------------------------------------------------------------
    // columns : Fill in vector<string> columnNames
    //--------------------------------------------------------------
    if ( columns_str.size() ) {

        std::vector<std::string> columns_vec = SplitString( columns_str,
                                                            " \t,\n" );
        columnNames = columns_vec;
    }

    if ( not columnNames.size() ) {
        std::stringstream errMsg;
        errMsg << "Parameters::Validate(): Simplex/CCM: "
               << " No columns parsed." << std::endl;
        throw std::runtime_error( errMsg.str() );
    }

    //--------------------------------------------------------------
    // target
    //--------------------------------------------------------------
    if ( target_str.size() ) {
        targetName = target_str;
    }

    //--------------------------------------------------------------
    // CCM sample not 0 if random is true
    //--------------------------------------------------------------
    if ( method == Method::CCM ) {
        if ( randomLib ) {
            if ( subSamples < 1 ) {
                std::string errMsg( "Parameters::Validate(): "
                                    "CCM samples must be > 0.\n" );
                throw std::runtime_error( errMsg );
            }
        }
    }

    //--------------------------------------------------------------
    // CCM librarySizes
    //   1) 3 arguments : start stop increment
    //      if increment < stop generate the library sequence.
    //      if increment > stop presume list of 3 library sizes.
    //   2) Otherwise: "x y ..." : list of library sizes.
    //--------------------------------------------------------------
    if ( libSizes_str.size() > 0 ) {
        std::vector<std::string> libsize_vec = SplitString(libSizes_str," \t,");

        bool libSizeSequence = false;
        int  start;
        int  stop;
        int  increment;

        if ( libsize_vec.size() == 3 ) {
            // Presume ( start, stop, increment ) sequence arguments
            start     = std::stoi( libsize_vec[0] );
            stop      = std::stoi( libsize_vec[1] );
            increment = std::stoi( libsize_vec[2] );

            // However, it might just be 3 library sizes...
            // If increment < stop, presume start : stop : increment
            if ( increment < stop ) {
                libSizeSequence = true; // Generate the library sizes
            }
        }

        if ( libSizeSequence ) {
            if ( increment < 1 ) {
                std::stringstream errMsg;
                errMsg << "Parameters::Validate(): "
                       << "CCM librarySizes increment " << increment
                       << " is invalid.\n";
                throw std::runtime_error( errMsg.str() );
            }

            if ( start > stop ) {
                std::stringstream errMsg;
                errMsg << "Parameters::Validate(): "
                       << "CCM librarySizes start " << start
                       << " stop " << stop  << " are invalid.\n";
                throw std::runtime_error( errMsg.str() );
            }

            if ( (int) start < E ) {
                std::stringstream errMsg;
                errMsg << "Parameters::Validate(): "
                       << "CCM librarySizes start < E = " << E << "\n";
                throw std::runtime_error( errMsg.str() );
            }
            else if ( (int) start < 3 ) {
                std::string errMsg( "Parameters::Validate(): "
                                    "CCM librarySizes start < 3.\n" );
                throw std::runtime_error( errMsg );
            }

            // Allocate the librarySizes vector
            int N_lib = ( stop - start ) / increment + 1;
            librarySizes = std::vector< size_t >( N_lib, 0 );

            // Fill in the sizes
            size_t libSize = start;
            for ( size_t i = 0; i < librarySizes.size(); i++ ) {
                librarySizes[i] = libSize;
                libSize = libSize + increment;
            }
        }
        else {
            // Presume a list of lib sizes
            librarySizes = std::vector< size_t >( libsize_vec.size(), 0 );
            for ( size_t i = 0; i < librarySizes.size(); i++ ) {
                size_t libSize;
                std::stringstream libStringStream( libsize_vec[i] );
                libStringStream >> libSize;
                librarySizes[i] = libSize;
            }
        }
    }

    //--------------------------------------------------------------------
    // Simplex: embedded true     : E set to number columns
    //          knn not specified : knn set to E+1
    //--------------------------------------------------------------------
    if ( method == Method::Simplex or method == Method::CCM ) {

        // embedded = true: Set E to number of columns if not already set
        if ( embedded ) {
            if ( columnNames.size() ) {
                if ( E != (int) columnNames.size() ) {
                    E = columnNames.size();
                }
            }
        }

        if ( knn < 1 ) {
            knn = E + 1;

            if ( verbose ) {
                std::stringstream msg;
                msg << "Parameters::Validate(): Set knn = " << knn
                    << " (E+1) for Simplex. " << std::endl;
                std::cout << msg.str();
            }
        }

        if ( knn < 1 ) {
            std::stringstream errMsg;
            errMsg << "Parameters::Validate(): Simplex knn less than 1."
                   << std::endl;
            throw std::runtime_error( errMsg.str() );
        }
    }

    //--------------------------------------------------------------------
    // SMap
    // embedded true : E set to size( columns ) for output processing
    // knn check deferred until after library is created
    //--------------------------------------------------------------------
    else if ( method == Method::SMap ) {
        if ( embedded and columnNames.size() > 1 ) {
            if ( columnNames.size() ) {
                E = columnNames.size();
            }
        }

        if ( not embedded and columnNames.size() > 1 ) {
            std::stringstream errMsg( "Parameters::Validate(): "
                                      "Multivariable S-Map must use "
                                      "embedded = true to ensure "
                                      "data/dimension correspondance.\n" );
            throw std::runtime_error( errMsg.str() );
        }
    }
    else if ( method == Method::Embed ) {
        // no-op
    }
    else {
        throw std::runtime_error( "Parameters::Validate() "
                                  "Prediction method error.\n" );
    }

    if ( method == Method::Simplex or method == Method::SMap ) {
        if ( E < 1 ) {
            std::stringstream errMsg;
            errMsg << "Parameters::Validate() E = " << E
                   << " is invalid with embedded = true.\n" ;
            throw std::runtime_error( errMsg.str() );
        }
    }

    //--------------------------------------------------------------
    // Generate library
    //--------------------------------------------------------------
    if ( lib_str.size() ) {
        // Parse lib_str into vector of strings
        std::vector<std::string> lib_vec = SplitString( lib_str, " \t," );
        if ( lib_vec.size() % 2 != 0 ) {
            std::string errMsg( "Parameters::Validate(): "
                                "library must be even number of integers.\n" );
            throw std::runtime_error( errMsg );
        }

        // Generate libPairs vector of start, stop index pairs
        std::vector< std::pair< size_t, size_t > > libPairs; // numeric bounds
        for ( size_t i = 0; i < lib_vec.size(); i = i + 2 ) {
            libPairs.emplace_back( std::make_pair( std::stoi( lib_vec[i] ),
                                                   std::stoi( lib_vec[i+1] ) ) );
        }

        // Validate end > start
        for ( auto thisPair : libPairs ) {
            size_t lib_start = thisPair.first;
            size_t lib_end   = thisPair.second;

            // Validate end > stop indices
            if ( method == Method::Simplex or method == Method::SMap ) {
                // Don't check if method == None, Embed or CCM since default
                // of "1 1" is used.
                if ( lib_start >= lib_end ) {
                    std::stringstream errMsg;
                    errMsg << "Parameters::Validate(): library start "
                           << lib_start << " exceeds end " << lib_end << ".\n";
                    throw std::runtime_error( errMsg.str() );
                }
            }
            // Disallow indices < 0, the user may have specified 0 start
            if ( lib_start < 1 or lib_end < 1 ) {
                std::stringstream errMsg;
                errMsg << "Parameters::Validate(): Library indices less than "
                       << "1 not allowed.\n";
                throw std::runtime_error( errMsg.str() );
            }
        }

        // Create library of indices
        library = std::vector< size_t >(); // Per thread

        int NPartial = abs( tau ) * (E - 1); // embedding shift

        // Loop over each lib pair
        // Add rows for library segments, disallowing vectors
        // in disjoint library "gap" accomodating embedding and Tp
        for ( size_t i = 0; i < libPairs.size(); i++ ) {
            std::pair< size_t, size_t > pair = libPairs[ i ];
            int start = pair.first;
            int stop  = pair.second;

            if ( tau < 0 ) {
                if ( not embedded ) { start = start + NPartial; }
            }
            else {
                if ( not embedded ) { stop  = stop - NPartial;  }
            }

            if ( Tp < 0 ) {
                start = std::max( start, start + abs( Tp ) - 1 );
            }
            else {
                if ( i != libPairs.size() - 1 ) {
                    stop = stop - Tp;
                }
            }

            for ( int j = start; j <= stop; j++ ) {
                library.push_back( j - 1 ); // apply zero-offset
            }
        }

        // Validate lib: E, tau, Tp combination
        if ( method == Method::Simplex or
             method == Method::SMap or method == Method::CCM ) {
            int vectorStart  = std::max( (E - 1) * tau, 0 );
            vectorStart      = std::max( vectorStart, Tp );
            int vectorEnd    = std::min( (E - 1) * tau, Tp );
            vectorEnd        = std::min( vectorEnd, 0 );
            int vectorLength = std::abs( vectorStart - vectorEnd ) + 1;

            int maxLibrarySegment = 0;
            for ( auto thisPair : libPairs ) {
                int libPairSpan = thisPair.second - thisPair.first + 1;
                maxLibrarySegment = std::max( maxLibrarySegment, libPairSpan );
            }

            if ( vectorLength > maxLibrarySegment ) {
                std::stringstream errMsg;
                errMsg << "Parameters::Validate(): Combination of E = "
                       << E << " Tp = " << Tp << " tau = " << tau
                       << " is invalid.\n";
                throw std::runtime_error( errMsg.str() );
            }
        }
    }

    //--------------------------------------------------------------
    // Generate prediction
    //--------------------------------------------------------------
    if ( pred_str.size() ) {
        // Parse pred_str into vector of strings
        std::vector<std::string> pred_vec = SplitString( pred_str, " \t," );
        if ( pred_vec.size() % 2 != 0 ) {
            std::string errMsg( "Parameters::Validate(): "
                                "prediction must be even number of integers.\n");
            throw std::runtime_error( errMsg );
        }

        // Generate vector of start, stop index pairs
        std::vector< std::pair< size_t, size_t > > predPairs;
        for ( size_t i = 0; i < pred_vec.size(); i = i + 2 ) {
            predPairs.emplace_back( std::make_pair( std::stoi( pred_vec[i] ),
                                                    std::stoi( pred_vec[i+1])));
        }

        size_t nPred = 0; // Count of pred items

        // Get number of pred indices, validate end > start
        for ( auto thisPair : predPairs ) {
            size_t pred_start = thisPair.first;
            size_t pred_end   = thisPair.second;

            nPred += pred_end - pred_start + 1;

            // Validate end > stop indices
            if ( method == Method::Simplex or method == Method::SMap ) {
                // Don't check if method == None, Embed or CCM since default
                // of "1 1" is used.
                if ( pred_start >= pred_end ) {
                    std::stringstream errMsg;
                    errMsg << "Parameters::Validate(): prediction start "
                           << pred_start << " exceeds end " << pred_end << ".\n";
                    throw std::runtime_error( errMsg.str() );
                }
            }
            // Disallow indices < 0, the user may have specified 0 start
            if ( pred_start < 1 or pred_end < 1 ) {
                std::stringstream errMsg;
                errMsg << "Parameters::Validate(): Prediction indices less than "
                       << "1 not allowed.\n";
                throw std::runtime_error( errMsg.str() );
            }
        }

        // Create prediction vector of indices
        prediction = std::vector< size_t >( nPred );
        size_t i = 0;
        for ( auto thisPair : predPairs ) {
            for ( size_t li = thisPair.first; li <= thisPair.second; li++ ) {
                prediction[ i ] = li - 1; // apply zero-offset
                i++;
            }
        }
        // Require sequential ordering
        for ( size_t i = 1; i < prediction.size(); i++ ) {
            if ( prediction[ i ] <= prediction[ i - 1 ] ) {
                std::stringstream errMsg;
                errMsg << "Parameters::Validate(): Prediction indices are "
                       << "not strictly increasing.\n";
                throw std::runtime_error( errMsg.str() );
            }
        }
        // Warn about disjoint prediction sets: not fully supported
        if ( predPairs.size() > 1 ) {
            std::stringstream msg;
            msg << "WARNING: Validate(): Disjoint prediction sets "
                << " are not fully supported. Use with caution." << std::endl;
            std::cout << msg.str();
        }
    }

    //--------------------------------------------------------------
    // Validate library & prediction
    // Set SMap knn default based on E and library size
    //--------------------------------------------------------------
    if ( method == Method::Simplex or method == Method::SMap ) {
        if ( not library.size() ) {
            std::string errMsg( "Parameters::Validate(): "
                                "library indices not found.\n" );
            throw std::runtime_error( errMsg );
        }
        if ( not prediction.size() ) {
            std::string errMsg( "Parameters::Validate(): "
                                "prediction indices not found.\n" );
            throw std::runtime_error( errMsg );
        }

        if ( method == Method::SMap ) {
            if ( knn == 0 ) { // default knn = 0, set knn value
                knn = library.size();

                if ( verbose ) {
                    std::stringstream msg;
                    msg << "Parameters::Validate(): Set knn = " << knn
                        << " for SMap. " << std::endl;
                    std::cout << msg.str();
                }
            }
            if ( knn < 2 ) {
                std::stringstream errMsg;
                errMsg<<"Parameters::Validate() S-Map knn must be at least 2.\n";
                throw std::runtime_error( errMsg.str() );
            }
        }
    }
    else {
        // Defaults if Method is None, Embed or CCM
        if ( not library.size() ) {
            library = std::vector<size_t>( 1, 0 );
        }
        if ( not prediction.size() ) {
            prediction = std::vector<size_t>( 1, 0 );
        }
    }

    //--------------------------------------------------------------
    // Generative modes
    // 
    //--------------------------------------------------------------
    if ( generateSteps != 0 and
         ( method == Method::Simplex or method == Method::SMap ) ) {
        
        if ( generateSteps < 0 ) {
            std::string errMsg( "Parameters::Validate(): "
                                "generateSteps must be non-negative.\n" );
            throw std::runtime_error( errMsg );
        }
        if ( embedded ) {
            std::string errMsg("Parameters::Validate(): "
                               "generateSteps embedded data not supported.\n");
            throw std::runtime_error( errMsg );
        }
        
        if ( Tp < 0 ) {
            std::string errMsg("Parameters::Validate(): "
                               "generateSteps negative Tp invalid.\n");
            throw std::runtime_error( errMsg );
        }
        
        if ( columnNames.front() != targetName ) {
            std::string errMsg("Parameters::Validate(): generateSteps "
                               "columns and target disjoint. "
                               "Only univariate data allowed.\n");
            throw std::runtime_error( errMsg );
        }
        
        if ( columnNames.size() > 1 ) {
            std::cout << "NOTE: Parameters::Validate(): generateSteps: "
                      << "Multiple columns found. "
                      << "Only univariate time-delay embedded data allowed. "
                      << columnNames.front() << " used as column data."
                      << std::endl;
            columnNames.clear();
            columnNames.push_back( targetName );
        }
    }

#ifdef DEBUG_ALL
    PrintIndices( library, prediction );
#endif

}

//---------------------------------------------------------------
// Adjust lib/pred concordant with Embed() tau(E-1) partial rows
//---------------------------------------------------------------
void Parameters::AdjustLibPred() {

    size_t shift          = abs( tau ) * ( E - 1 );
    size_t library_len    = library.size();
    size_t prediction_len = prediction.size();

    // If [0, 1, ... shift]  (negative tau) or
    // [N-shift, ... N-1, N] (positive tau) are in library or prediction
    // delete these index elements.

    // First, create vectors of indices to delete.
    std::vector< size_t > deleted_pred_elements( shift, 0 );
    std::vector< size_t > deleted_lib_elements ( shift, 0 );

    int predStart = 0;
    int libStart  = 0;

    if ( tau > 0 ) {
        predStart = std::max( 0, (int) prediction_len - (int) shift );
        libStart  = std::max( 0, (int) library_len    - (int) shift );
    }
    std::iota( deleted_pred_elements.begin(),
               deleted_pred_elements.end(), predStart );
    std::iota( deleted_lib_elements.begin(),
               deleted_lib_elements.end(),  libStart  );

    // Now that we have the indices of partial embedding vectors,
    // check to see if any are in lib and pred
    bool deleteLibIndex = false;
    std::vector< size_t >::iterator it;
    for ( auto element  = deleted_lib_elements.begin();
               element != deleted_lib_elements.end(); element++ ) {
        it = std::find( library.begin(), library.end(), *element );
        if ( it != library.end() ) {
            deleteLibIndex = true;
            break;
        }
    }

    bool deletePredIndex = false;
    for ( auto element  = deleted_pred_elements.begin();
               element != deleted_pred_elements.end(); element++ ) {
        it = std::find( prediction.begin(), prediction.end(), *element );
        if ( it != prediction.end() ) {
            deletePredIndex = true;
            break;
        }
    }

    // Erase elements of row indices that were deleted
    if ( deleteLibIndex ) {
        for ( auto element  = deleted_lib_elements.begin();
                   element != deleted_lib_elements.end(); element++ ) {

            std::vector< size_t >::iterator it;
            it = std::find( library.begin(), library.end(), *element );

            if ( it != library.end() ) {
                library.erase( it );
            }
        }
    }

    if ( deletePredIndex ) {
        for ( auto element  = deleted_pred_elements.begin();
              element != deleted_pred_elements.end(); element++ ) {

            std::vector< size_t >::iterator it;
            it = std::find( prediction.begin(),
                            prediction.end(), *element );

            if ( it != prediction.end() ) {
                prediction.erase( it );
            }
        }
    }

    return;
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void Parameters::FillMap() {
    std::stringstream ss;

    ss << version.Major << "." << version.Minor
       << "." << version.Micro << " " << version.Date;
    Map[ "version" ] = ss.str();
    ss.str( std::string() ); // clear the ss string

    if      ( method == Method::Simplex ) { ss <<  "Simplex"; }
    else if ( method == Method::SMap    ) { ss <<  "SMap";    }
    else if ( method == Method::CCM     ) { ss <<  "CCM";     }
    else if ( method == Method::None    ) { ss <<  "None";    }
    else if ( method == Method::Embed   ) { ss <<  "Embed";   }
    Map[ "method" ] = ss.str();
    ss.str( std::string() );

    ss << columns_str;
    Map[ "columns" ] = ss.str();
    ss.str( std::string() );

    ss << target_str;
    Map[ "target" ] = ss.str();
    ss.str( std::string() );

    ss << lib_str;
    Map[ "lib" ] = ss.str();
    ss.str( std::string() );

    ss << pred_str;
    Map[ "pred" ] = ss.str();
    ss.str( std::string() );

    ss << E;
    Map[ "E" ] = ss.str();
    ss.str( std::string() );

    ss << Tp;
    Map[ "Tp" ] = ss.str();
    ss.str( std::string() );

    ss << knn;
    Map[ "knn" ] = ss.str();
    ss.str( std::string() );

    ss << tau;
    Map[ "tau" ] = ss.str();
    ss.str( std::string() );

    ss << theta;
    Map[ "theta" ] = ss.str();
    ss.str( std::string() );

    ss << exclusionRadius;
    Map[ "exclusionRadius" ] = ss.str();
    ss.str( std::string() );

    ss << libSizes_str;
    Map[ "libSizes" ] = ss.str();
    ss.str( std::string() );

    ss << subSamples;
    Map[ "subSamples" ] = ss.str();
    ss.str( std::string() );

    ss << randomLib;
    Map[ "randomLib" ] = ss.str();
    ss.str( std::string() );

    ss << replacement;
    Map[ "replacement" ] = ss.str();
    ss.str( std::string() );

    ss << seed;
    Map[ "seed" ] = ss.str();
    ss.str( std::string() );

    ss << includeData;
    Map[ "includeData" ] = ss.str();
    ss.str( std::string() );

    ss << multiviewEnsemble;
    Map[ "multiviewEnsemble" ] = ss.str();
    ss.str( std::string() );

    ss << multiviewD;
    Map[ "multiviewD" ] = ss.str();
    ss.str( std::string() );

    ss << multiviewTrainLib;
    Map[ "multiviewTrainLib" ] = ss.str();
    ss.str( std::string() );

    ss << multiviewExcludeTarget;
    Map[ "multiviewExcludeTarget" ] = ss.str();
    ss.str( std::string() );

    ss << embedded;
    Map[ "embedded" ] = ss.str();
    ss.str( std::string() );

    ss << "[ ";
    for ( auto v : validLib ) { ss << v << " "; }
    ss << "]";
    Map[ "validLib" ] = ss.str();
    ss.str( std::string() );

    ss << const_predict;
    Map[ "const_predict" ] = ss.str();
    ss.str( std::string() );

    ss << generateSteps;
    Map[ "generateSteps" ] = ss.str();
    ss.str( std::string() );

    ss << parameterList;
    Map[ "parameterList" ] = ss.str();
    ss.str( std::string() );

    ss << verbose;
    Map[ "verbose" ] = ss.str();
    ss.str( std::string() );

    ss << pathIn;
    Map[ "pathIn" ] = ss.str();
    ss.str( std::string() );

    ss << dataFile;
    Map[ "dataFile" ] = ss.str();
    ss.str( std::string() );

    ss << pathOut;
    Map[ "pathOut" ] = ss.str();
    ss.str( std::string() );

    ss << predictOutputFile;
    Map[ "predictOutputFile" ] = ss.str();
    ss.str( std::string() );

    ss << SmapOutputFile;
    Map[ "SmapOutputFile" ] = ss.str();
    ss.str( std::string() );

    ss << blockOutputFile;
    Map[ "blockOutputFile" ] = ss.str();
    ss.str( std::string() );
}

//------------------------------------------------------------------
// Overload << to output to ostream
//------------------------------------------------------------------
std::ostream& operator<< ( std::ostream &os, Parameters &p ) {

    os << "Parameters: -------------------------------------------\n";

    std::string method("Unknown");
    if      ( p.method == Method::Simplex ) { method = "Simplex"; }
    else if ( p.method == Method::SMap    ) { method = "SMap";    }
    else if ( p.method == Method::CCM     ) { method = "CCM";     }
    else if ( p.method == Method::None    ) { method = "None";    }
    else if ( p.method == Method::Embed   ) { method = "Embed";   }

    os << "Method: " << method
       << " E=" << p.E << " Tp=" << p.Tp
       << " knn=" << p.knn << " tau=" << p.tau << " theta=" << p.theta
       << std::endl;

    if ( p.columnNames.size() ) {
        os << "Column Names : [ ";
        for ( auto ci = p.columnNames.begin();
              ci != p.columnNames.end(); ++ci ) {
            os << *ci << " ";
        } os << "]" << std::endl;
    }

    if ( p.targetName.size() ) {
        os << "Target: " << p.targetName << std::endl;
    }

    os << "Library: [" << p.library[0] << " : "
       << p.library[ p.library.size() - 1 ] << "]  "
       << "Prediction: [" << p.prediction[0] << " : "
       << p.prediction[ p.prediction.size() - 1 ]
       << "] " << std::endl;

    os << "-------------------------------------------------------\n";

    return os;
}

#ifdef DEBUG_ALL
//------------------------------------------------------------------
// 
//------------------------------------------------------------------
void Parameters::PrintIndices( std::vector<size_t> library,
                               std::vector<size_t> prediction )
{
    std::cout << "Parameters(): library: ";
    for ( auto li = library.begin(); li != library.end(); ++li ) {
        std::cout << *li << " ";
    } std::cout << std::endl;
    std::cout << "Parameters(): prediction: ";
    for ( auto pi = prediction.begin(); pi != prediction.end(); ++pi ) {
        std::cout << *pi << " ";
    } std::cout << std::endl;
}
#endif
