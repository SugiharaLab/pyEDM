
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
    int          multiview,
    int          exclusionRadius,
    bool         trainLib,
    bool         excludeTarget,
    bool         parameterList,
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
                        multiview,
                        exclusionRadius,
                        trainLib,
                        excludeTarget,
                        parameterList,
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
                        multiview,
                        exclusionRadius,
                        trainLib,
                        excludeTarget,
                        parameterList,
                        verbose,
                        numThreads );
    }
    else {
        throw std::runtime_error( "Multiview_pybind(): Invalid input.\n" );
    }

    DF predictions = DataFrameToDF( MV.Predictions );
    DF combo_rho   = DataFrameToDF( MV.ComboRho    );

    // Convert MV.ColumnNames to py::dict
    py::dict columnNames;
    for ( auto ci = MV.ColumnNames.begin(); ci != MV.ColumnNames.end(); ci++ ) {
        // Why is cast required for string key but not valarray?
        columnNames[ py::str( ci->first ) ] = ci->second;
    }

    std::map< std::string, py::dict > MV_;

    MV_["View" ]       = DFtoDict( combo_rho   );
    MV_["Predictions"] = DFtoDict( predictions );
    MV_["ColumnNames"] = columnNames;

    if ( parameterList ) {
        MV_["parameters"] = ParamMaptoDict( MV.parameterMap );
    }

    return MV_;
}
