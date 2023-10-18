
#include "PyBind.h"
#include <pybind11/numpy.h>

// Instantiate a default object for the SMap solver
// This will contain the python reference to the scikit solver object
// if one was created by the user and passed in the solver parameter.
py::object SmapSolverObject = pybind11::cast<pybind11::none>(Py_None);

//----------------------------------------------------------
// Wrapper for the SMap solver from sklearn.linear_model
//----------------------------------------------------------
SVDValues SmapSolver( DataFrame< double >     A,
                      std::valarray< double > B ) {

    // The construction of coefficient marix A in cppEDM::SMap inserts
    // a unity vector in the first column to create a bias (intercept)
    // term in the LAPACK solver.  sklearn.linear_model's include an
    // intercept term by default.  Remove first column.
    std::vector< size_t > coeff( A.NColumns() - 1 );
    std::iota( coeff.begin(), coeff.end(), 1 );
    A = A.DataFrameFromColumnIndex( coeff );

    // Convert DataFrame to pandas DataFrame / numpy array for sklearn 
    py::dict A_pydict = DFtoDict( DataFrameToDF( A ) );
    py::object A_df   = py::module::import("pandas").attr("DataFrame");
    py::object A_     = A_df( A_pydict );

    // Fit linear model
    SmapSolverObject.attr( "fit" )( A_, B );

    // Get solver results
    std::vector<double> coeffs =
        SmapSolverObject.attr( "coef_" ).cast< std::vector< double > >();

    double intercept = SmapSolverObject.attr( "intercept_" ).cast< float >();

    // Insert intercept bias vector as leading coefficient as in cppEDM::SMap
    coeffs.insert( coeffs.begin(), intercept );

    std::valarray<double> coefficients( coeffs.data(), coeffs.size() );

    // Only LinearRegression has singular_
    std::vector<double> SV;
    bool singularFound = false;
    // JP There must be a better way to check if the generic Python object
    // has attribute singular_. pybind11 does not have a c++ equivalent of
    // python builtin hasattr()
    try {    
        SV = SmapSolverObject.attr("singular_").cast< std::vector<double> >();
        singularFound = true;
    }
    catch ( const std::exception& e ) {
        singularFound = false;
        //std::cout << e.what() << std::endl;
    }

    std::valarray<double> singularValues;
    if ( singularFound ) {
        // Since sklearn.linear_model's include an intercept term by default
        // the first column in A was removed (intercept term in cppEDM),
        // so LinearRegression returns E, not E+1 singular values
        // cppEDM expects E+1. Add an E+1 entry at the intercept (first) col
        SV.insert( SV.begin(), nan("SMap") );
        singularValues = std::valarray<double>( SV.data(), SV.size() );
    }
    else {
        // insert nan for singularValues
        singularValues.resize( coeffs.size(), nan("SMap") );
    }

    SVDValues SVD_;
    SVD_.coefficients   = coefficients;
    SVD_.singularValues = singularValues;
    
    return SVD_;
};

//----------------------------------------------------------
// 
//----------------------------------------------------------
std::map< std::string, py::dict >
    SMap_pybind( std::string pathIn, 
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
                 std::string smapCoefFile,
                 std::string smapSVFile,
                 py::object  solver,
                 bool        embedded,
                 bool        const_predcit,
                 bool        verbose,
                 std::vector<bool> validLib,
                 bool        ignoreNan,
                 int         generateSteps,
                 bool        generateLibrary,
                 bool        parameterList
 ) {
    SmapSolverObject = solver;

    // Select solver
    auto solver_ =
        SmapSolverObject.is( pybind11::cast<pybind11::none>(Py_None) ) ? 
        &SVD : &SmapSolver;

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
                   smapCoefFile,
                   smapSVFile,
                   solver_,
                   embedded,
                   const_predcit,
                   verbose,
                   validLib,
                   ignoreNan,
                   generateSteps,
                   generateLibrary,
                   parameterList );
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
                   smapCoefFile,
                   smapSVFile,
                   solver_,
                   embedded,
                   const_predcit,
                   verbose,
                   validLib,
                   ignoreNan,
                   generateSteps,
                   generateLibrary,
                   parameterList );
    }
    else {
        throw std::runtime_error( "SMap_pybind(): Invalid input.\n" );
    }

    DF df_pred = DataFrameToDF( SM.predictions    );
    DF df_coef = DataFrameToDF( SM.coefficients   );
    DF df_SV   = DataFrameToDF( SM.singularValues );

    std::map< std::string, py::dict > SMap_;

    SMap_["predictions" ]   = DFtoDict( df_pred );
    SMap_["coefficients"]   = DFtoDict( df_coef );
    SMap_["singularValues"] = DFtoDict( df_SV   );

    if ( parameterList ) {
        SMap_["parameters"] = ParamMaptoDict( SM.parameterMap );
    }

    // Release python object since done with it now
    SmapSolverObject.release();

    return SMap_;
}
