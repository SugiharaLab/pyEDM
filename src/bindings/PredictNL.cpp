
#include "PyBind.h"

//---------------------------------------------------------------
// Input data path and file
//---------------------------------------------------------------
py::dict PredictNonlinear_pybind( std::string       pathIn,
                                  std::string       dataFile,
                                  DF                df,
                                  std::string       pathOut,
                                  std::string       predictFile,
                                  std::string       lib,
                                  std::string       pred,
                                  std::string       theta,
                                  int               E,
                                  int               Tp,
                                  int               knn,
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
        // dataFile specified, dispatch overloaded PredictNonlinear,
        // ignore dataList
        PredictDF  = PredictNonlinear( pathIn,
                                       dataFile,
                                       pathOut,
                                       predictFile,
                                       lib,
                                       pred,
                                       theta,
                                       E,
                                       Tp,
                                       knn,
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
        
        PredictDF  = PredictNonlinear( dataFrame,
                                       pathOut,
                                       predictFile,
                                       lib,
                                       pred,
                                       theta,
                                       E,
                                       Tp,
                                       knn,
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
        throw std::runtime_error("PredictNonlinear_pybind(): Invalid input.\n");
    }

    DF       dfout = DataFrameToDF( PredictDF );
    py::dict D     = DFtoDict( dfout );
    
    return D;
}
