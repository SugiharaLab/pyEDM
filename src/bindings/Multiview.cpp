
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
    int          D,
    int          E,
    int          Tp,
    int          knn,
    int          tau, 
    std::string  columns,
    std::string  target,
    std::vector<std::vector<size_t>> embeddingAssignments,
    std::vector<std::vector<bool>> conditionalEmbeddings,
    int          multiview,
    int          exclusionRadius,
    bool         trainLib,
    bool         excludeTarget,
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
                        D,
                        E,
                        Tp,
                        knn,
                        tau,
                        columns,
                        target,
                        embeddingAssignments,
                        conditionalEmbeddings,
                        multiview,
                        exclusionRadius,
                        trainLib,
                        excludeTarget,
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
                        D,
                        E,
                        Tp,
                        knn,
                        tau,
                        columns,
                        target,
                        embeddingAssignments,
                        conditionalEmbeddings,
                        multiview,
                        exclusionRadius,
                        trainLib,
                        excludeTarget,
                        verbose,
                        numThreads );
    }
    else {
        throw std::runtime_error( "Multiview_pybind(): Invalid input.\n" );
    }

    DF predictions = DataFrameToDF( MV.Predictions );
    DF combo_rho   = DataFrameToDF( MV.ComboRho    );

    std::map< std::string, py::dict > MV_;

    MV_["View"  ]      = DFtoDict( combo_rho   );
    MV_["Predictions"] = DFtoDict( predictions );

    return MV_;
}
