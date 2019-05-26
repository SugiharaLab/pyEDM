
#include "PyBind.h"

//---------------------------------------------------------------
// 
//---------------------------------------------------------------
py::dict EmbedDimension_pybind( std::string pathIn,
                                std::string dataFile,
                                DF          dataList,
                                std::string pathOut,
                                std::string predictFile,
                                std::string lib,
                                std::string pred,
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
                                     Tp,
                                     tau,
                                     columns,
                                     target,
                                     embedded,
                                     verbose,
                                     numThreads );
    }
    else if ( dataList.size() ) {
        DataFrame< double > dataFrame = DFToDataFrame( dataList );
        
        EmbedDimDF = EmbedDimension( dataFrame,
                                     pathOut,
                                     predictFile,
                                     lib,
                                     pred,
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
    
    DF       df = DataFrameToDF( EmbedDimDF );
    py::dict D  = DFtoDict( df );
    
    return D;
}
