
#include "PyBind.h"

//---------------------------------------------------------------
// 
//---------------------------------------------------------------
py::dict EmbedDimension_pybind( std::string pathIn,
                                std::string dataFile,
                                DF          df,
                                std::string pathOut,
                                std::string predictFile,
                                std::string lib,
                                std::string pred,
                                int         maxE,
                                int         Tp,
                                int         tau,
                                std::string columns,
                                std::string target,
                                bool        embedded,
                                bool        verbose,
                                unsigned    numThreads ) {
    
    DataFrame< double > EmbedDimDF;

    if ( dataFile.size() ) {
        // dataFile specified, dispatch overloaded EmbedDimension,
        // ignore dataList
        EmbedDimDF = EmbedDimension( pathIn,
                                     dataFile,
                                     pathOut,
                                     predictFile,
                                     lib,
                                     pred,
                                     maxE,
                                     Tp,
                                     tau,
                                     columns,
                                     target,
                                     embedded,
                                     verbose,
                                     numThreads );
    }
    else if ( df.dataList.size() ) {
        DataFrame< double > dataFrame = DFToDataFrame( df );
        
        EmbedDimDF = EmbedDimension( dataFrame,
                                     pathOut,
                                     predictFile,
                                     lib,
                                     pred,
                                     maxE,
                                     Tp,
                                     tau,
                                     columns,
                                     target,
                                     embedded,
                                     verbose,
                                     numThreads );
    }
    else {
        throw std::runtime_error( "EmbedDimension_pybind(): Invalid input.\n" );
    }
    
    DF       dfout = DataFrameToDF( EmbedDimDF );
    py::dict D     = DFtoDict( dfout );
    
    return D;
}
