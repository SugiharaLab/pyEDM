#ifndef EDM_COMMON_H
#define EDM_COMMON_H

#include <iostream>
#include <sstream>
#include <vector>
#include <valarray>
#include <map>
#include <forward_list>
#include <cctype>
#include <cmath>
#include <functional> // std::ref 

#ifdef _MSC_VER
#include <ciso646> // macro constants for MSVC C++ operators not in ISO646
#endif

// Enumerations
enum class Method         { None, Embed, Simplex, SMap, CCM };
enum class DistanceMetric { Euclidean, Manhattan };

#include "DataFrame.h"

//---------------------------------------------------------
// Data structs
//---------------------------------------------------------
struct VectorError {
    double rho;
    double RMSE;
    double MAE;
};

struct SMapValues {
    DataFrame< double > predictions;
    DataFrame< double > coefficients;
};

// Return object for CrossMap() worker function
struct CrossMapValues {
    DataFrame< double > LibStats;     // mean libsize, rho, RMSE, MAE
    DataFrame< double > PredictStats; // each predict libsize, rho, RMSE, MAE
    std::forward_list< DataFrame< double > > Predictions;
};

// Return object for CCM() with two CrossMapValues
struct CCMValues {
    DataFrame< double > AllLibStats;  // unified mean libsize, rho, RMSE, MAE
    CrossMapValues CrossMap1;
    CrossMapValues CrossMap2;
};

struct MultiviewValues {
    DataFrame< double > ComboRho;             // col_i..., rho, MAE, RMSE
    DataFrame< double > Predictions;
    std::vector< std::string > ComboRhoTable; // includes column names
};

//-------------------------------------------------------------
// Prototypes
//-------------------------------------------------------------
std::string ToLower   ( std::string str );
bool        OnlyDigits( std::string str, bool integerOnly );

std::vector<std::string> SplitString( std::string inString, 
                                      std::string delimeters );

VectorError ComputeError( std::valarray< double > obs,
                          std::valarray< double > pred );

std::string increment_datetime_str( std::string datetime1, 
                                    std::string datetime2,
                                    int         tp );
#endif
