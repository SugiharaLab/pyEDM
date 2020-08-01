
#include "PyBind.h"

//-------------------------------------------------------------
// 
//-------------------------------------------------------------
//py::dict Simplex_pybind( std::string pathIn,
// JP for GMN return a dictionary, not just a data frame
std::map< std::string, py::dict > Simplex_pybind( std::string pathIn,
                         std::string dataFile,
                         DF          df,
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

    // JP DataFrame< double > S;
    SimplexValues S;
    
    if ( dataFile.size() ) {
        // dataFile specified, dispatch overloaded Simplex, ignore df.dataList
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
    else if ( df.dataList.size() ) {
        DataFrame< double > dataFrame = DFToDataFrame( df );
        
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
    
    // JP  DF       dfout = DataFrameToDF( S );
    // JP  py::dict D     = DFtoDict( dfout );

    DF  df_pred  = DataFrameToDF( S.predictions   );
    DFI df_neigh = DataFrameToDFI( S.knn_neighbors );
    DFI df_lib   = DataFrameToDFI( S.knn_library   );
 
    std::map< std::string, py::dict > D;

    D["predictions" ]  = DFtoDict( df_pred  );
    D["knn_neighbors"] = DFItoDict( df_neigh );
    D["knn_library"]   = DFItoDict( df_lib   );
    
    return D;
}
