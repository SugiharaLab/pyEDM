
#include "PyBind.h"

//----------------------------------------------------------------
// Compute Error Wrapper method
// @param vec1      : the first vector to compare
// @param vec2      : the second vector to compare
// @return          : map/dictionary with the rho, mae, rmse
//----------------------------------------------------------------
std::map< std::string, double > ComputeError_pybind (
    std::valarray<double> vec1, 
    std::valarray<double> vec2 ) {

    VectorError vecErr = ComputeError( vec1, vec2 );

    // Setup as map instead of vecErr struct
    std::map<std::string, double> VE;

    VE[ "MAE"  ]  = vecErr.MAE; 
    VE[ "rho"  ]  = vecErr.rho; 
    VE[ "RMSE" ]  = vecErr.RMSE;

    return VE;
}
