
#include "PyBind.h"

//----------------------------------------------------------
// 
//----------------------------------------------------------
std::map< std::string, py::dict > SMap_pybind( std::string pathIn, 
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
                                               double      theta,
                                               std::string columns,
                                               std::string target,
                                               std::string smapFile,
                                               std::string jacobians,
                                               bool        embedded,
                                               bool        verbose ) {
    SMapValues SM;
    
    if ( dataFile.size() ) {
        // dataFile specified, dispatch overloaded SMap, ignore dataList
        
        SM = SMap( pathIn,
                   dataFile,
                   pathOut,
                   predictFile,
                   lib,
                   pred,
                   E, 
                   Tp,
                   knn,
                   tau,
                   theta,
                   columns, 
                   target,
                   smapFile,
                   jacobians,
                   embedded,
                   verbose);
    }
    else if ( dataList.size() ) {
        DataFrame< double > dataFrame = DFToDataFrame( dataList );
        
        SM = SMap( dataFrame,
                   pathOut,
                   predictFile,
                   lib,
                   pred,
                   E, 
                   Tp,
                   knn,
                   tau,
                   theta,
                   columns, 
                   target,
                   smapFile,
                   jacobians,
                   embedded,
                   verbose);
    }
    else {
        throw std::runtime_error( "SMap_pybind(): Invalid input.\n" );
    }
    
    DF df_pred = DataFrameToDF( SM.predictions  );
    DF df_coef = DataFrameToDF( SM.coefficients );

    std::map< std::string, py::dict > SMap_;

    SMap_["predictions" ] = DFtoDict( df_pred );
    SMap_["coefficients"] = DFtoDict( df_coef );

    return SMap_;
}
