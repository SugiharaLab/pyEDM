
#include "Parameter.h"

//----------------------------------------------------------------
// Constructor
// Default values set in Parameter.h
//----------------------------------------------------------------
Parameters::Parameters(
    Method      method,
    std::string pathIn,
    std::string dataFile,
    std::string pathOut,
    std::string predictFile,    
    std::string lib_str,
    std::string pred_str,
    int         E,
    int         Tp,
    int         knn,
    int         tau,
    float       theta,
    int         exclusionRadius,

    std::string columns_str,
    std::string target_str,
    
    bool        embedded,
    bool        const_predict,
    bool        verbose,
    
    std::string SmapFile,
    std::string blockFile,
    std::string derivatives_str,
    
    float       svdSig,
    float       tikhonov,
    float       elasticNet,
    
    int         multi,
    std::string libSizes_str,
    int         sample,
    bool        random,
    bool        replacement,
    unsigned    rseed,
    bool        noNeigh,
    bool        fwdTau
    ) :
    // Variable initialization from Parameters arguments
    method           ( method ),
    pathIn           ( pathIn ),
    dataFile         ( dataFile ),
    pathOut          ( pathOut ),
    predictOutputFile( predictFile ),    
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
    
    SmapOutputFile   ( SmapFile ),
    blockOutputFile  ( blockFile ),

    SVDSignificance  ( svdSig ),
    derivatives_str  ( derivatives_str ),
    TikhonovAlpha    ( tikhonov ),
    ElasticNetAlpha  ( elasticNet ),
    
    MultiviewEnsemble( multi ),
    libSizes_str     ( libSizes_str ),
    subSamples       ( sample ),
    randomLib        ( random ),
    replacement      ( replacement ),
    seed             ( rseed ),
    noNeighborLimit  ( noNeigh ),
    forwardTau       ( fwdTau ),

    // Set validated flag and instantiate Version
    validated        ( false ),
    version          ( 1, 0, 1, "2019-12-12" )
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
// 
//----------------------------------------------------------------
void Parameters::Load() {}

//----------------------------------------------------------------
// Index offsets, generate library and prediction indices,
// and parameter validation
//----------------------------------------------------------------
void Parameters::Validate() {

    validated = true;

    //--------------------------------------------------------------
    // Generate library indices: Apply zero-offset
    //--------------------------------------------------------------
    if ( lib_str.size() ) {
        std::vector<std::string> lib_vec = SplitString( lib_str, " \t," );
        if ( lib_vec.size() != 2 ) {
            std::string errMsg( "Parameters::Validate(): "
                                "library must be two integers.\n" );
            throw std::runtime_error( errMsg );
        }
        int lib_start = std::stoi( lib_vec[0] );
        int lib_end   = std::stoi( lib_vec[1] );

        if ( method == Method::Simplex or method == Method::SMap ) {
            // Don't check if None, Embed or CCM since default of "1 1" is used.
            if ( lib_start >= lib_end ) {
                std::stringstream errMsg;
                errMsg << "Parameters::Validate(): library start "
                       << lib_start << " exceeds end " << lib_end << ".\n";
                throw std::runtime_error( errMsg.str() );
            }
        }
        
        library = std::vector<size_t>( lib_end - lib_start + 1 );
        std::iota ( library.begin(), library.end(), lib_start - 1 );
    }

    //--------------------------------------------------------------
    // Generate prediction indices: Apply zero-offset
    //--------------------------------------------------------------
    if ( pred_str.size() ) {
        std::vector<std::string> pred_vec = SplitString( pred_str, " \t," );
        if ( pred_vec.size() != 2 ) {
            std::string errMsg( "Parameters::Validate(): "
                                "prediction must be two integers.\n");
            throw std::runtime_error( errMsg );
        }
        int pred_start = std::stoi( pred_vec[0] );
        int pred_end   = std::stoi( pred_vec[1] );
        
        if ( method == Method::Simplex or method == Method::SMap ) {
            // Don't check if None, Embed or CCM since default of "1 1" is used.
            if ( pred_start >= pred_end ) {
                std::stringstream errMsg;
                errMsg << "Parameters::Validate(): prediction start "
                       << pred_start << " exceeds end " << pred_end << ".\n";
                throw std::runtime_error( errMsg.str() );
            }
        }
        
        prediction = std::vector<size_t>( pred_end - pred_start + 1 );
        std::iota ( prediction.begin(), prediction.end(), pred_start - 1 );
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
    
    // Derivatives
    if ( derivatives_str.size() > 0 ) {
        std::vector<std::string> der_vec = SplitString(derivatives_str," \t,");
        if ( der_vec.size() < 2 ) {
            std::string errMsg( "Parameters::Validate(): "
                                "derivatives must be integer pairs.\n");
            throw std::runtime_error( errMsg );
        }
        derivatives = std::vector<size_t>( der_vec.size() );
        for ( size_t i = 0; i < der_vec.size(); i++ ) {
            derivatives.push_back( std::stoi( der_vec[i] ) );
        }
    }

    // CCM librarySizes
    if ( libSizes_str.size() > 0 ) {
        std::vector<std::string> libsize_vec = SplitString(libSizes_str," \t,");
        if ( libsize_vec.size() != 3 ) {
            std::string errMsg( "Parameters::Validate(): "
                                "CCM librarySizes must be three integers.\n" );
            throw std::runtime_error( errMsg );
        }
        
        size_t start     = std::stoi( libsize_vec[0] );
        size_t stop      = std::stoi( libsize_vec[1] );
        size_t increment = std::stoi( libsize_vec[2] );
        size_t N_lib     = std::floor((stop-start)/increment + 1/increment)+1;

        if ( start < E ) {
            std::stringstream errMsg;
            errMsg << "Parameters::Validate(): "
                   << "CCM librarySizes start < E = " << E << "\n";
            throw std::runtime_error( errMsg.str() );
        }
        else if ( start < 3 ) {
            std::string errMsg( "Parameters::Validate(): "
                                "CCM librarySizes start < 3.\n" );
            throw std::runtime_error( errMsg );
        }

        // Create the librarySizes vector
        librarySizes = std::vector<size_t>( N_lib, 0 );

        // Fill in the sizes
        size_t libSize = start;
        for ( size_t i = 0; i < librarySizes.size(); i++ ) {
            librarySizes[i] = libSize;
            libSize = libSize + increment;
        }
    }

    //--------------------------------------------------------------------
    // If Simplex and knn not specified, knn set to E+1
    // If S-Map require knn > E + 1, default is all neighbors.
    if ( method == Method::Simplex or method == Method::CCM ) {
        if ( knn < 1 ) {
            knn = E + 1;
            if ( verbose ) {
                std::stringstream msg;
                msg << "Parameters::Validate(): Set knn = " << knn
                    << " (E+1) for Simplex. " << std::endl;
                std::cout << msg.str();
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
            // knn = 0
            knn = library.size() - Tp * (E + 1);
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

        // S-Map coefficient columns for derivatives start at 1 since the 0th
        // column is the S-Map linear prediction bias term
        if ( derivatives.size() > 1 ) {
            std::vector<size_t>::iterator it = std::find( derivatives.begin(),
                                                          derivatives.end(), 0);
            if ( it != derivatives.end() ) {
                std::string errMsg( "Parameters::Validate() S-Map coefficient "
                            " columns for derivatives can not use column 0.\n");
                throw std::runtime_error( errMsg );
            }
            if ( derivatives.size() % 2 ) {
                std::string errMsg( "Parameters::Validate() S-Map coefficient "
                            " columns for derivatives must be in pairs.\n");
                throw std::runtime_error( errMsg );                
            }
        }

        // Tikhonov and ElasticNet are mutually exclusive
        if ( TikhonovAlpha and ElasticNetAlpha ) {
            std::string errMsg( "Parameters::Validate() Multiple S-Map solve "
                                "methods specified.  Use one or none of: "
                                "tikhonov,   elasticNet.\n");
            throw std::runtime_error( errMsg );                
        }

        // Very small alphas don't make sense in elastic net
        if ( ElasticNetAlpha < 0.01 ) {
            std::cout << "Parameters::Validate() ElasticNetAlpha too small."
                         " Setting to 0.01.";
            ElasticNetAlpha = 0.01;
        }
        if ( ElasticNetAlpha > 1 ) {
            std::cout << "Parameters::Validate() ElasticNetAlpha too large."
                         " Setting to 1.";
            ElasticNetAlpha = 1;
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
// If we support negative tau, this will change
// For now, assume only positive tau is allowed
//------------------------------------------------------------
void Parameters::DeleteLibPred( size_t shift ) {
    
    size_t library_len    = library.size();
    size_t prediction_len = prediction.size();

    // If 0, 1, ... shift are in library or prediction
    // those rows were deleted, delete these elements.
    // First, create a vector of indices to delete
    std::vector< size_t > deleted_elements( shift, 0 );
    std::iota( deleted_elements.begin(), deleted_elements.end(), 0 );

    // erase elements of row indices that were deleted
    for ( auto element =  deleted_elements.begin();
          element != deleted_elements.end(); element++ ) {

        std::vector< size_t >::iterator it;
        it = std::find( library.begin(), library.end(), *element );

        if ( it != library.end() ) {
            library.erase( it );
        }
                
        it = std::find( prediction.begin(),
                        prediction.end(), *element );

        if ( it != prediction.end() ) {
            prediction.erase( it );
        }
    }
            
    // Now offset all values by shift so that vectors indices
    // in library and prediction refer to the same data rows
    // before the deletion/shift.
    for ( auto li =  library.begin();
          li != library.end(); li++ ) {
        *li = *li - shift;
    }
    for ( auto pi =  prediction.begin();
          pi != prediction.end(); pi++ ) {
        *pi = *pi - shift;
    }
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
