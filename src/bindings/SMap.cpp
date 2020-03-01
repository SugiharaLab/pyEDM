
#include "PyBind.h"
#include <pybind11/numpy.h>

// Instantiate a default object for the SMap solver
// This will contain the python reference to the scikit solver object
// if one was created by the user and passed in the solver parameter.
py::object SmapSolverObject = pybind11::cast<pybind11::none>(Py_None);

//----------------------------------------------------------
// Wrapper for the SMap solver from sklearn.linear_model
//----------------------------------------------------------
std::valarray<double> SmapSolver( DataFrame< double >     A,
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

    std::valarray<double> solutionValArr( coeffs.data(), coeffs.size() );

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
                                               py::object  solver,
                                               bool        embedded,
                                               bool        const_predcit,
                                               bool        verbose ) {
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
                   smapFile,
                   derivatives,
                   solver_,
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
                   solver_,
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

    // Release python object since done with it now
    SmapSolverObject.release();

    return SMap_;
}
