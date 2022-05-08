
#include "PyBind.h"

//---------------------------------------------------------------
// Input data path and file
//---------------------------------------------------------------
py::dict PredictInterval_pybind( std::string       pathIn,
                                 std::string       dataFile,
                                 DF                df,
                                 std::string       pathOut,
                                 std::string       predictFile,
                                 std::string       lib,
                                 std::string       pred,
                                 int               maxTp,
                                 int               E,
                                 int               tau,
                                 int               exclusionRadius,
                                 std::string       columns,
                                 std::string       target,
                                 bool              embedded,
                                 bool              verbose,
                                 std::vector<bool> validLib,
                                 unsigned          numThreads ) {

    DataFrame< double > PredictDF;

    if ( dataFile.size() ) {
        // dataFile specified, dispatch overloaded PredictInterval,
        // ignore dataList
        PredictDF = PredictInterval( pathIn,
                                     dataFile,
                                     pathOut,
                                     predictFile,
                                     lib,
                                     pred,
                                     maxTp,
                                     E,
                                     tau,
                                     exclusionRadius,
                                     columns,
                                     target,
                                     embedded,
                                     verbose,
                                     validLib,
                                     numThreads );
    }
    else if ( df.dataList.size() ) {
        DataFrame< double > dataFrame = DFToDataFrame( df );
        
        PredictDF = PredictInterval( dataFrame,
                                     pathOut,
                                     predictFile,
                                     lib,
                                     pred,
                                     maxTp,
                                     E,
                                     tau,
                                     exclusionRadius,
                                     columns,
                                     target,
                                     embedded,
                                     verbose,
                                     validLib,
                                     numThreads );
    }
    else {
        throw std::runtime_error("PredictInterval_pybind(): Invalid input.\n");
    }

    DF       dfout = DataFrameToDF( PredictDF );
    py::dict D     = DFtoDict( dfout );
    
    return D;
}
