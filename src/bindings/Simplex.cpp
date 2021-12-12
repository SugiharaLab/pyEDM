
#include "PyBind.h"

//-------------------------------------------------------------
// 
//-------------------------------------------------------------
std::map< std::string, py::dict >
    Simplex_pybind( std::string pathIn,
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
                    bool        verbose,
                    std::vector<bool> validLib,
                    int         generateSteps,
                    bool        parameterList ) {

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
                     verbose,
                     validLib,
                     generateSteps,
                     parameterList );
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
                     verbose,
                     validLib,
                     generateSteps,
                     parameterList );
    }
    else {
        throw std::runtime_error( "Simplex_pybind(): Invalid input.\n" );
    }

    DF df_pred = DataFrameToDF( S.predictions );

    std::map< std::string, py::dict > S_;

    S_["predictions"] = DFtoDict( df_pred );
    
    if ( parameterList ) {
        S_["parameters"]  = ParamMaptoDict( S.parameterMap );
    }

    return S_;
}
