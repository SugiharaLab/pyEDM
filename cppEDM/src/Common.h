#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <sstream>
#include <vector>
#include <valarray>
#include <map>
#include <cctype>
#include <cmath>

#ifdef _MSC_VER
#include <ciso646> // macro constants for MSVC C++ operators not in ISO646
#endif

#include "DataFrame.h" // has #include Common.h

// Normally, macros are eschewed
// Define the initial maximum distance for neigbor distances to avoid sort()
// Note: std::numeric_limits<double>::max() ~1E308
//       std::numeric_limits<float> ::max() ~1E38
//       std::numeric_limits<int64> ::max() ~1E18
//       std::numeric_limits<int>   ::max() ~1E9
#define DISTANCE_MAX   1E300
#define DISTANCE_LIMIT 1E299  // Must be less than DISTANCE_MAX, but large

// Enumerations
enum class Method         { None, Embed, Simplex, SMap, CCM };
enum class DistanceMetric { Euclidean, Manhattan };

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

struct MultiviewValues {
    DataFrame< double > Combo_rho;              // col_i..., rho, MAE, RMSE
    DataFrame< double > Predictions;
    std::vector< std::string > Combo_rho_table; // includes column names

#ifdef MULTIVIEW_VALUES_OVERLOAD
    // Don't define constructors for the setuptools module build on Windows
    // The MSVC compiler with pybind11 does not handle overloads easily...
    // https://pybind11.readthedocs.io/en/stable/classes.html
    
    // Constructors
    MultiviewValues();

    MultiviewValues( DataFrame< double >        combo_rho,
                     DataFrame< double >        predictions,
                     std::vector< std::string > combo_rho_table ):
        Combo_rho( combo_rho ), Predictions( predictions ),
        Combo_rho_table( combo_rho_table ) {}
#endif
};

//-------------------------------------------------------------
// Prototypes
// Primary API functions generally have two call-signatures.
// The first takes a (path, file name) pair specifying the data
// file image on disk to be loaded and converted to a data frame.
// The second replaces these two arguments with a DataFrame object.
//
// NOTE: These are the first declarations seen by the compiler
//       for the API and provide default argument values
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

// API functions Embed() and MakeBlock() are in Embed.h Embed.cc

DataFrame<double> Simplex( std::string pathIn          = "./data/",
                           std::string dataFile        = "",
                           std::string pathOut         = "./",
                           std::string predictFile     = "",
                           std::string lib             = "",
                           std::string pred            = "",
                           int         E               = 0,
                           int         Tp              = 1,
                           int         knn             = 0,
                           int         tau             = 1,
                           int         exclusionRadius = 0,
                           std::string colNames        = "",
                           std::string targetName      = "",
                           bool        embedded        = false,
                           bool        const_predict   = false,
                           bool        verbose         = true );

DataFrame<double> Simplex( DataFrame< double > &dataFrameIn,
                           std::string pathOut         = "./",
                           std::string predictFile     = "",
                           std::string lib             = "",
                           std::string pred            = "",
                           int         E               = 0,
                           int         Tp              = 1,
                           int         knn             = 0,
                           int         tau             = 1,
                           int         exclusionRadius = 0,
                           std::string colNames        = "",
                           std::string targetName      = "",
                           bool        embedded        = false,
                           bool        const_predict   = false,
                           bool        verbose         = true );

SMapValues SMap( std::string pathIn          = "./data/",
                 std::string dataFile        = "",
                 std::string pathOut         = "./",
                 std::string predictFile     = "",
                 std::string lib             = "",
                 std::string pred            = "",
                 int         E               = 0,
                 int         Tp              = 1,
                 int         knn             = 0,
                 int         tau             = 1,
                 double      theta           = 0,
                 int         exclusionRadius = 0,
                 std::string columns         = "",
                 std::string target          = "",
                 std::string smapFile        = "",
                 std::string derivatives     = "",
                 bool        embedded        = false,
                 bool        const_predict   = false,
                 bool        verbose         = true );

SMapValues SMap( DataFrame< double > &dataFrameIn,
                 std::string pathOut         = "./",
                 std::string predictFile     = "",
                 std::string lib             = "",
                 std::string pred            = "",
                 int         E               = 0,
                 int         Tp              = 1,
                 int         knn             = 0,
                 int         tau             = 1,
                 double      theta           = 0,
                 int         exclusionRadius = 0,
                 std::string columns         = "",
                 std::string target          = "",
                 std::string smapFile        = "",
                 std::string derivatives     = "",
                 bool        embedded        = false,
                 bool        const_predict   = false,
                 bool        verbose         = true );

DataFrame<double> CCM( std::string pathIn       = "./data/",
                       std::string dataFile     = "",
                       std::string pathOut      = "./",
                       std::string predictFile  = "",
                       int         E            = 0,
                       int         Tp           = 0,
                       int         knn          = 0,
                       int         tau          = 1,
                       std::string colNames     = "",
                       std::string targetName   = "",
                       std::string libSizes_str = "",
                       int         sample       = 0,
                       bool        random       = true,
                       bool        replacement  = false,
                       unsigned    seed         = 0,     // seed=0: use RNG
                       bool        verbose      = true );

DataFrame<double> CCM( DataFrame< double > dataFrameIn,
                       std::string         pathOut      = "./",
                       std::string         predictFile  = "",
                       int                 E            = 0,
                       int                 Tp           = 0,
                       int                 knn          = 0,
                       int                 tau          = 1,
                       std::string         colNames     = "",
                       std::string         targetName   = "",
                       std::string         libSizes_str = "",
                       int                 sample       = 0,
                       bool                random       = true,
                       bool                replacement  = false,
                       unsigned            seed         = 0, // seed=0: use RNG
                       bool                verbose      = true );

MultiviewValues Multiview( std::string pathIn          = "./",
                           std::string dataFile        = "",
                           std::string pathOut         = "./",
                           std::string predictFile     = "",
                           std::string lib             = "",
                           std::string pred            = "",
                           int         E               = 0,
                           int         Tp              = 1,
                           int         knn             = 0,
                           int         tau             = 1,
                           std::string columns         = "",
                           std::string target          = "",
                           int         multiview       = 0,
                           int         exclusionRadius = 0,
                           bool        verbose         = false,
                           unsigned    nThreads        = 4 );

MultiviewValues Multiview( DataFrame< double >,
                           std::string pathOut         = "./",
                           std::string predictFile     = "",
                           std::string lib             = "",
                           std::string pred            = "",
                           int         E               = 0,
                           int         Tp              = 1,
                           int         knn             = 0,
                           int         tau             = 1,
                           std::string columns         = "",
                           std::string target          = "",
                           int         multiview       = 0,
                           int         exclusionRadius = 0,
                           bool        verbose         = false,
                           unsigned    nThreads        = 4 );

DataFrame<double> EmbedDimension( std::string pathIn      = "./data/",
                                  std::string dataFile    = "",
                                  std::string pathOut     = "./",
                                  std::string predictFile = "",
                                  std::string lib         = "",
                                  std::string pred        = "",
                                  int         maxE        = 10,
                                  int         Tp          = 1,
                                  int         tau         = 1,
                                  std::string colNames    = "",
                                  std::string targetName  = "",
                                  bool        embedded    = false,
                                  bool        verbose     = true,
                                  unsigned    nThreads    = 4 );

DataFrame<double> EmbedDimension( DataFrame< double > &dataFrameIn,
                                  std::string pathOut     = "./",
                                  std::string predictFile = "",
                                  std::string lib         = "",
                                  std::string pred        = "",
                                  int         maxE        = 10,
                                  int         Tp          = 1,
                                  int         tau         = 1,
                                  std::string colNames    = "",
                                  std::string targetName  = "",
                                  bool        embedded    = false,
                                  bool        verbose     = true,
                                  unsigned    nThreads    = 4 );

DataFrame<double> PredictInterval( std::string pathIn      = "./data/",
                                   std::string dataFile    = "",
                                   std::string pathOut     = "./",
                                   std::string predictFile = "",
                                   std::string lib         = "",
                                   std::string pred        = "",
                                   int         maxTp       = 10,
                                   int         E           = 0,
                                   int         tau         = 1,
                                   std::string colNames    = "",
                                   std::string targetName  = "",
                                   bool        embedded    = false,
                                   bool        verbose     = true,
                                   unsigned    nThreads    = 4 );

DataFrame<double> PredictInterval( DataFrame< double > &dataFrameIn,
                                   std::string pathOut     = "./",
                                   std::string predictFile = "",
                                   std::string lib         = "",
                                   std::string pred        = "",
                                   int         maxTp       = 10,
                                   int         E           = 0,
                                   int         tau         = 1,
                                   std::string colNames    = "",
                                   std::string targetName  = "",
                                   bool        embedded    = false,
                                   bool        verbose     = true,
                                   unsigned    nThreads    = 4 );

DataFrame<double> PredictNonlinear( std::string pathIn      = "./data/",
                                    std::string dataFile    = "",
                                    std::string pathOut     = "./",
                                    std::string predictFile = "",
                                    std::string lib         = "",
                                    std::string pred        = "",
                                    std::string theta       = "",
                                    int         E           = 0,
                                    int         Tp          = 1,
                                    int         knn         = 0,
                                    int         tau         = 1,
                                    std::string colNames    = "",
                                    std::string targetName  = "",
                                    bool        embedded    = false,
                                    bool        verbose     = true,
                                    unsigned    nThreads    = 4 );

DataFrame<double> PredictNonlinear( DataFrame< double > &dataFrameIn,
                                    std::string pathOut     = "./",
                                    std::string predictFile = "",
                                    std::string lib         = "",
                                    std::string pred        = "",
                                    std::string theta       = "",
                                    int         E           = 0,
                                    int         Tp          = 1,
                                    int         knn         = 0,
                                    int         tau         = 1,
                                    std::string colNames    = "",
                                    std::string targetName  = "",
                                    bool        embedded    = false,
                                    bool        verbose     = true,
                                    unsigned    nThreads    = 4 );
#endif
