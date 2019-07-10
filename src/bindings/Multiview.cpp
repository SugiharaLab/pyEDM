
#include "PyBind.h"

//--------------------------------------------------------------
// 
//--------------------------------------------------------------
std::map< std::string, py::dict > Multiview_pybind (
    std::string  pathIn,
    std::string  dataFile,
    DF           df,
    std::string  pathOut,
    std::string  predictFile,
    std::string  lib,
    std::string  pred,
    int          E,
    int          Tp,
    int          knn,
    int          tau, 
    std::string  columns,
    std::string  target,
    int          multiview,
    bool         verbose,
    unsigned int numThreads ) {

    MultiviewValues MV;

    if ( dataFile.size() ) {
        // dataFile specified, dispatch overloaded Multiview, ignore dataList
        
        MV = Multiview( pathIn,
                        dataFile,
                        pathOut,
                        predictFile,
                        lib,
                        pred, 
                        E,
                        Tp,
                        knn,
                        tau,
                        columns,
                        target,
                        multiview,
                        verbose,
                        numThreads );
    }
    else if ( df.dataList.size() ) {
        DataFrame< double > dataFrame = DFToDataFrame( df );
        
        MV = Multiview( dataFrame,
                        pathOut,
                        predictFile,
                        lib,
                        pred, 
                        E,
                        Tp,
                        knn,
                        tau,
                        columns,
                        target,
                        multiview,
                        verbose,
                        numThreads );
    }
    else {
        throw std::runtime_error( "Multiview_pybind(): Invalid input.\n" );
    }

    DF df_combo_rho   = DataFrameToDF( MV.Combo_rho   );
    DF df_predictions = DataFrameToDF( MV.Predictions );

    std::map< std::string, py::dict > MV_;

    MV_["Combo_rho"  ] = DFtoDict( df_combo_rho   );
    MV_["Predictions"] = DFtoDict( df_predictions );

    return MV_;
}
