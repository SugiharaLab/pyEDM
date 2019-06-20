
#include "PyBind.h"

//-------------------------------------------------------------
// 
//-------------------------------------------------------------
py::dict Simplex_pybind( std::string pathIn,
                         std::string dataFile,
                         DF          dataList,
                         std::string pathOut,
                         std::string predictFile,
                         std::string lib,
                         std::string pred, 
                         int         E,
                         int         Tp,
                         int         knn,
                         int         tau,
                         int         exclusionRadius,
                         std::string columns,
                         std::string target,
                         bool        embedded,
                         bool        const_predict,
                         bool        verbose ) {

    DataFrame< double > S;
    
    if ( dataFile.size() ) {
        // dataFile specified, dispatch overloaded Simplex, ignore dataList
        S = Simplex( pathIn,
                     dataFile,
                     pathOut,
                     predictFile,
                     lib,
                     pred,
                     E, 
                     Tp,
                     knn,
                     tau,
                     exclusionRadius,
                     columns,
                     target, 
                     embedded,
                     const_predict,
                     verbose );
    }
    else if ( dataList.size() ) {
        DataFrame< double > dataFrame = DFToDataFrame( dataList );
        
        S = Simplex( dataFrame,
                     pathOut,
                     predictFile,
                     lib,
                     pred,
                     E, 
                     Tp,
                     knn,
                     tau,
                     exclusionRadius,
                     columns,
                     target, 
                     embedded,
                     const_predict,
                     verbose );
    }
    else {
        throw std::runtime_error( "Simplex_pybind(): Invalid input.\n" );
    }
    
    DF       df = DataFrameToDF( S );
    py::dict D  = DFtoDict( df );
    
    return D;
}
