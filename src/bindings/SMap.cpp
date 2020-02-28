
#include "PyBind.h"
#include <pybind11/numpy.h>

// have to store the potential elastic net model since only interface with
// smap is the function pointer

py::object elasticModel = pybind11::cast<pybind11::none>(Py_None);

//----------------------------------------------------------
// wrapper for the elastic net solver
//----------------------------------------------------------
std::valarray<double> ElasticNetSolver (DataFrame<double> A,
        std::valarray<double> B){

    // remove first col since first col bias term and model includes bias term 

    std::vector<size_t> non_bias_cols (A.NColumns()-1);
    std::iota( non_bias_cols.begin(), non_bias_cols.end(), 1 );
    A = A.DataFrameFromColumnIndex( non_bias_cols );

    // convert DataFrame to pd df for sklearn 

    py::dict embeddingAsDict = DFtoDict(DataFrameToDF(A));
    py::object pdDfCons      = py::module::import("pandas").attr("DataFrame");
    py::object embeddingAsPd = pdDfCons( embeddingAsDict );

    // fit elastic net model and then setup {intercept, coefs} solution

    elasticModel.attr("fit")( embeddingAsPd, B );

    std::vector<double> coeffs = elasticModel.attr("coef_").cast<std::vector<double>>();
    double intercept           = elasticModel.attr("intercept_").cast<float>();

    coeffs.insert( coeffs.begin(), intercept );

    std::valarray<double> solutionValArr(coeffs.data(),coeffs.size());

    return solutionValArr;
};


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
                                               py::object  ElasticNetModel,
                                               bool        embedded,
                                               bool        const_predcit,
                                               bool        verbose ) {
    elasticModel = ElasticNetModel;


    // set solver as SVD or elastic net based on whether elastic net null or not 


    auto solver = elasticModel.is(pybind11::cast<pybind11::none>(Py_None)) ? 
                    &SVD : &ElasticNetSolver;

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
                   solver,
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
                   solver,
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

    // release python object since done with it now
    elasticModel.release();


    return SMap_;
}
