
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

    std::string SmapOutputFile,
    std::string blockOutputFile,

    int         multiviewEnsemble,
    int         multiviewD,
    bool        multiviewTrainLib,

    std::string libSizes_str,
    int         subSamples,
    bool        randomLib,
    bool        replacement,
    unsigned    seed,
    bool        includeData,
    bool        noNeighborLimit
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
    targetIndex      ( 0 ),

    embedded         ( embedded ),
    const_predict    ( const_predict ),
    verbose          ( verbose ),

    SmapOutputFile   ( SmapOutputFile ),
    blockOutputFile  ( blockOutputFile ),

    multiviewEnsemble( multiviewEnsemble ),
    multiviewD       ( multiviewD ),
    multiviewTrainLib( multiviewTrainLib ),

    libSizes_str     ( libSizes_str ),
    subSamples       ( subSamples ),
    randomLib        ( randomLib ),
    replacement      ( replacement ),
    seed             ( seed ),
    includeData      ( includeData ),
    noNeighborLimit  ( noNeighborLimit ),

    // Set validated flag and instantiate Version
    validated        ( false ),
    version          ( 1, 5, 0, "2020-07-01" )
{
    // Constructor code
    if ( method != Method::None ) {

        Validate();

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
// Index offsets, generate library and prediction indices,
// and parameter validation
//----------------------------------------------------------------
void Parameters::Validate() {

    validated = true;

    if ( not embedded and tau == 0 ) {
        std::string errMsg( "Parameters::Validate(): "
                            "tau must be non-zero.\n" );
        throw std::runtime_error( errMsg );
    }

    //--------------------------------------------------------------
    // Generate library indices: Apply zero-offset
    //--------------------------------------------------------------
    if ( lib_str.size() ) {
        // Parse lib_str into vector of strings
        std::vector<std::string> lib_vec = SplitString( lib_str, " \t," );
        if ( lib_vec.size() % 2 != 0 ) {
            std::string errMsg( "Parameters::Validate(): "
                                "library must be even number of integers.\n" );
            throw std::runtime_error( errMsg );
        }

        // Generate vector of start, stop index pairs
        std::vector< std::pair< size_t, size_t > > libPairs;
        for ( size_t i = 0; i < lib_vec.size(); i = i + 2 ) {
            libPairs.emplace_back( std::make_pair( std::stoi( lib_vec[i] ),
                                                   std::stoi( lib_vec[i+1] ) ) );
        }

        size_t nLib = 0; // Count of lib items

        // Get number of lib indices, validate end > start
        for ( auto thisPair : libPairs ) {
            size_t lib_start = thisPair.first;
            size_t lib_end   = thisPair.second;

            nLib += lib_end - lib_start + 1;

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
        }

        // Create library vector of indices
        library = std::vector< size_t >( nLib );
        size_t i = 0;
        for ( auto thisPair : libPairs ) {
            for ( size_t li = thisPair.first; li <= thisPair.second; li++ ) {
                library[ i ] = li - 1; // apply zero-offset
                i++;
            }
        }
    }

    //--------------------------------------------------------------
    // Generate prediction indices: Apply zero-offset
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
    }

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

#ifdef DEBUG_ALL
    PrintIndices( library, prediction );
#endif

    //--------------------------------------------------------------
    // Convert multi argument parameters from string to vectors
    //--------------------------------------------------------------

    // Columns
    // If columns are purely integer, then populate vector<size_t> columnIndex
    // Otherwise fill in vector<string> columnNames
    if ( columns_str.size() ) {

        std::vector<std::string> columns_vec = SplitString( columns_str,
                                                            " \t,\n" );

        bool onlyDigits = false;

        for ( auto ci = columns_vec.begin(); ci != columns_vec.end(); ++ci ) {
            onlyDigits = OnlyDigits( *ci, true );
            
            if ( not onlyDigits ) { break; }
        }

        if ( onlyDigits ) {
            for ( auto ci =  columns_vec.begin();
                       ci != columns_vec.end(); ++ci ) {
                columnIndex.push_back( std::stoi( *ci ) );
            }
        }
        else {
            columnNames = columns_vec;
        }
    }

    if ( not columnIndex.size() and not columnNames.size() ) {
        std::stringstream errMsg;
        errMsg << "Parameters::Validate(): Simplex/CCM: "
               << " No valid columns found." << std::endl;
        throw std::runtime_error( errMsg.str() );
    }

    // target
    if ( target_str.size() ) {
        bool onlyDigits = OnlyDigits( target_str, true );
        if ( onlyDigits ) {
            targetIndex = std::stoi( target_str );
        }
        else {
            targetName = target_str;
        }
    }

    //--------------------------------------------------------------------
    // CCM sample not 0 if random is true
    if ( method == Method::CCM ) {
        if ( randomLib ) {
            if ( subSamples < 1 ) {
                std::string errMsg( "Parameters::Validate(): "
                                    "CCM samples must be > 0.\n" );
                throw std::runtime_error( errMsg );
            }
        }
    }

    // CCM librarySizes
    if ( libSizes_str.size() > 0 ) {
        std::vector<std::string> libsize_vec = SplitString(libSizes_str," \t,");

        bool   libSizeSequence = false;
        size_t start;
        size_t stop;
        size_t increment;
        
        if ( libsize_vec.size() == 3 ) {
            // Presume ( start, stop, increment ) sequence arguments
            start     = std::stoi( libsize_vec[0] );
            stop      = std::stoi( libsize_vec[1] );
            increment = std::stoi( libsize_vec[2] );

            // However, it might just be 3 library sizes...
            // If increment < stop, then presume 3 sequence arguments
            if ( increment < stop ) {
                libSizeSequence = true;
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

            size_t N_lib = std::floor((stop-start)/increment + 1/increment) + 1;

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
    // Simplex and knn not specified: not embedded : knn set to E+1
    //                                    embedded : knn set to size( columns )
    // S-Map require knn > E + 1, default is all neighbors.
    if ( method == Method::Simplex or method == Method::CCM ) {
        if ( knn < 1 ) {
            if ( not embedded ) {
                knn = E + 1;
                if ( verbose ) {
                    std::stringstream msg;
                    msg << "Parameters::Validate(): Set knn = " << knn
                        << " (E+1) for Simplex. " << std::endl;
                    std::cout << msg.str();
                }
            }
            else { // embedded = true
                if ( columnIndex.size() ) {
                    knn = columnIndex.size() + 1;
                }
                else if ( columnNames.size() ) {
                    knn = columnNames.size() + 1;
                }
                
                if ( verbose ) {
                    std::stringstream msg;
                    msg << "Parameters::Validate(): Set knn = " << knn
                        << " for Simplex (embedded = true). " << std::endl;
                    std::cout << msg.str();
                }
            }
        }
        if ( knn < E + 1 ) {
            std::stringstream errMsg;
            errMsg << "Parameters::Validate(): Simplex knn of " << knn
                   << " is less than E+1 = " << E + 1 << std::endl;
            throw std::runtime_error( errMsg.str() );
        }
    }
    else if ( method == Method::SMap ) {
        if ( knn > 0 ) {
            if ( knn < E + 1 ) {
                std::stringstream errMsg;
                errMsg << "Parameters::Validate() S-Map knn must be at least "
                          " E+1 = " << E + 1 << ".\n";
                throw std::runtime_error( errMsg.str() );
            }
        }
        else {
            // default knn = 0, set knn value
            knn = library.size() - abs( Tp ) * (E + 1);
            if ( verbose ) {
                std::stringstream msg;
                msg << "Parameters::Validate(): Set knn = " << knn
                    << " for SMap. " << std::endl;
                std::cout << msg.str();
            }
        }
        if ( not embedded and columnNames.size() > 1 ) {
            std::string msg( "Parameters::Validate() WARNING:  "
                             "Multivariable S-Map should use "
                             "-e (embedded) data input to ensure "
                             "data/dimension correspondance.\n" );
            std::cout << msg;
        }
    }
    else if ( method == Method::Embed ) {
        // no-op
    }
    else {
        throw std::runtime_error( "Parameters::Validate() "
                                  "Prediction method error.\n" );
    }
}

//------------------------------------------------------------
// Adjust lib/pred concordant with Embed() removal of tau(E-1)
// rows, and DeletePartialDataRow()
//------------------------------------------------------------
void Parameters::DeleteLibPred() {

    size_t shift          = abs( tau ) * ( E - 1 );
    size_t library_len    = library.size();
    size_t prediction_len = prediction.size();

    // If [0, 1, ... shift]  (negative tau) or
    // [N-shift, ... N-1, N] (positive tau) are in library or prediction
    // those rows were deleted, delete these index elements.
    // First, create vectors of indices to delete.
    std::vector< size_t > deleted_pred_elements( shift, 0 );
    std::vector< size_t > deleted_lib_elements ( shift, 0 );

    if ( tau < 0 ) {
        std::iota(deleted_pred_elements.begin(), deleted_pred_elements.end(),0);
        std::iota(deleted_lib_elements.begin(),  deleted_lib_elements.end(), 0);
    }
    else {
        std::iota( deleted_pred_elements.begin(),
                   deleted_pred_elements.end(), prediction_len - shift );
        std::iota( deleted_lib_elements.begin(),
                   deleted_lib_elements.end(), library_len - shift );
    }

    // Now that we have the indices that could have been deleted,
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

#ifdef DEBUG_ALL
    std::cout << "DeleteLibPred(): shift: " << shift
              << " deleteLibIndex: " << deleteLibIndex
              << " deletePredIndex: " << deletePredIndex << std::endl;
    std::cout << " deleted_lib_elements: ";
    for ( auto element  = deleted_lib_elements.begin();
               element != deleted_lib_elements.end(); element++ ) {
        std::cout << *element << ", ";
    } std::cout << std::endl << " deleted_pred_elements: ";
    for ( auto element  = deleted_pred_elements.begin();
               element != deleted_pred_elements.end(); element++ ) {
        std::cout << *element << ", ";
    } std::cout << std::endl;
#endif

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

    // Now offset all values by shift so that vectors indices
    // in library and prediction refer to the same data rows
    // before the deletion/shift.
    if ( tau < 0 ) {
        for ( auto li = library.begin(); li != library.end(); li++ ) {
            *li = *li - shift;
        }
        for ( auto pi = prediction.begin(); pi != prediction.end(); pi++ ) {
            *pi = *pi - shift;
        }
    }
    // tau > 0  : Forward shifting: no adjustment needed from origin

    return;
}

//------------------------------------------------------------------
// Overload << to output to ostream
//------------------------------------------------------------------
std::ostream& operator<< ( std::ostream &os, Parameters &p ) {

    // print info about the dataframe
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
