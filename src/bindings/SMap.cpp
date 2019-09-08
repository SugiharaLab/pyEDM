
#include "PyBind.h"

//----------------------------------------------------------
// 
//----------------------------------------------------------
std::map< std::string, py::dict > SMap_pybind( std::string pathIn, 
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
                                               double      theta,
                                               int         exclusionRadius,
                                               std::string columns,
                                               std::string target,
                                               std::string smapFile,
                                               std::string derivatives,
                                               bool        embedded,
                                               bool        const_predcit,
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
                   exclusionRadius,
                   columns, 
                   target,
                   smapFile,
                   derivatives,
                   embedded,
                   const_predcit,
                   verbose);
    }
    else if ( df.dataList.size() ) {
        DataFrame< double > dataFrame = DFToDataFrame( df );
        
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
                   exclusionRadius,
                   columns, 
                   target,
                   smapFile,
                   derivatives,
                   embedded,
                   const_predcit,
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
