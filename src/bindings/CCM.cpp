
#include "PyBind.h"

//-----------------------------------------------------------
// 
//-----------------------------------------------------------
py::dict CCM_pybind( std::string pathIn, 
                     std::string dataFile,
                     DF          dataList,
                     std::string pathOut,
                     std::string predictFile,
                     int         E,
                     int         Tp,
                     int         knn,
                     int         tau, 
                     std::string columns,
                     std::string target,
                     std::string libSizes,
                     int         sample,
                     bool        random,
                     unsigned    seed, 
                     bool        verbose ) {
    
    DataFrame< double > ccmOutput;

    if ( dataFile.size() ) {
        // dataFile specified, dispatch overloaded CCM, ignore dataList
        
        ccmOutput = CCM( pathIn,
                         dataFile,
                         pathOut,
                         predictFile,
                         E, 
                         Tp,
                         knn,
                         tau,
                         columns,
                         target, 
                         libSizes,
                         sample,
                         random,
                         seed,
                         verbose );
    }
    else if ( dataList.size() ) {
        DataFrame< double > dataFrame = DFToDataFrame( dataList );
        
        ccmOutput = CCM( dataFrame,
                         pathOut,
                         predictFile,
                         E, 
                         Tp,
                         knn,
                         tau,
                         columns,
                         target, 
                         libSizes,
                         sample,
                         random,
                         seed,
                         verbose );
    }
    else {
        throw std::runtime_error( "CCM_pybind(): Invalid input.\n" );
    }

    DF       df = DataFrameToDF( ccmOutput );
    py::dict D  = DFtoDict( df );
    
    return D;
}
