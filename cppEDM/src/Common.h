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
enum class Method         { None, Embed, Simplex, SMap, CCM, Multiview };
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

struct SimplexValues {
    DataFrame< double >                  predictions;
    std::map< std::string, std::string > parameterMap;
};

struct SMapValues {
    DataFrame< double >                  predictions;
    DataFrame< double >                  coefficients;
    DataFrame< double >                  singularValues;
    std::map< std::string, std::string > parameterMap;
};

struct SVDValues {
    std::valarray< double > coefficients;
    std::valarray< double > singularValues;
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
    std::map< std::string, std::string > parameterMap;
};

struct MultiviewValues {
    DataFrame< double > ComboRho;            // col_i..., rho, MAE, RMSE
    DataFrame< double > Predictions;
    // Vectors of column names <- col_i
    std::map< std::string, std::vector< std::string > > ColumnNames;
    std::map< std::string, std::string > parameterMap;
};

//-------------------------------------------------------------
// Prototypes
//-------------------------------------------------------------
std::string ToLower( std::string str );

std::vector<std::string> SplitString( std::string inString, 
                                      std::string delimeters,
                                      bool        removeWhitespace );

VectorError ComputeError( std::valarray< double > obs,
                          std::valarray< double > pred );

std::string increment_datetime_str( std::string datetime1, 
                                    std::string datetime2,
                                    int         tp );
#endif
