#ifndef EDM_API_H
#define EDM_API_H

#include "Common.h"    // Non DataFrame return struct definitions
#include "Parameter.h"
#include "Simplex.h"
#include "SMap.h"
#include "CCM.h"
#include "Multiview.h"

//-------------------------------------------------------------
// API function declarations.
//
// API functions generally have two call-signatures.
// The first takes a (path, file name) pair specifying the data
// file image on disk to be loaded and converted to a data frame.
// The second replaces these two arguments with a DataFrame object.
//
// NOTE: These are the first declarations seen by the compiler
//       for the API and provide default argument values
//-------------------------------------------------------------

DataFrame< double > Embed( std::string path     = "",
                           std::string dataFile = "",
                           int         E        = 0,
                           int         tau      = 0,
                           std::string columns  = "",
                           bool        verbose  = false );

DataFrame< double > Embed( DataFrame< double > & dataFrame,
                           int                 E       = 0,
                           int                 tau     = 0,
                           std::string         columns = "",
                           bool                verbose = false );

DataFrame< double > MakeBlock( DataFrame< double >      & dataFrame,
                               int                      E,
                               int                      tau,
                               std::vector<std::string> columnNames );

DataFrame< double > Simplex( std::string pathIn          = "./data/",
                             std::string dataFile        = "",
                             std::string pathOut         = "./",
                             std::string predictFile     = "",
                             std::string lib             = "",
                             std::string pred            = "",
                             int         E               = 0,
                             int         Tp              = 1,
                             int         knn             = 0,
                             int         tau             = -1,
                             int         exclusionRadius = 0,
                             std::string colNames        = "",
                             std::string targetName      = "",
                             bool        embedded        = false,
                             bool        const_predict   = false,
                             bool        verbose         = true );

DataFrame< double > Simplex( DataFrame< double > & dataFrameIn,
                             std::string pathOut         = "./",
                             std::string predictFile     = "",
                             std::string lib             = "",
                             std::string pred            = "",
                             int         E               = 0,
                             int         Tp              = 1,
                             int         knn             = 0,
                             int         tau             = -1,
                             int         exclusionRadius = 0,
                             std::string colNames        = "",
                             std::string targetName      = "",
                             bool        embedded        = false,
                             bool        const_predict   = false,
                             bool        verbose         = true );

// SMap is a special case since it can be called with a function pointer
// to the SVD solver. This is done so that interfaces such as pybind11
// can provide their own object for the solver.
// 1) Data path/file with default SVD (LAPACK) assigned in Smap.cc 2)
SMapValues SMap( std::string pathIn          = "./data/",
                 std::string dataFile        = "",
                 std::string pathOut         = "./",
                 std::string predictFile     = "",
                 std::string lib             = "",
                 std::string pred            = "",
                 int         E               = 0,
                 int         Tp              = 1,
                 int         knn             = 0,
                 int         tau             = -1,
                 double      theta           = 0,
                 int         exclusionRadius = 0,
                 std::string columns         = "",
                 std::string target          = "",
                 std::string smapFile        = "",
                 std::string derivatives     = "",
                 bool        embedded        = false,
                 bool        const_predict   = false,
                 bool        verbose         = true );

// 2) DataFrame with default SVD (LAPACK) assigned in Smap.cc 2)
SMapValues SMap( DataFrame< double > &dataFrameIn,
                 std::string pathOut         = "./",
                 std::string predictFile     = "",
                 std::string lib             = "",
                 std::string pred            = "",
                 int         E               = 0,
                 int         Tp              = 1,
                 int         knn             = 0,
                 int         tau             = -1,
                 double      theta           = 0,
                 int         exclusionRadius = 0,
                 std::string columns         = "",
                 std::string target          = "",
                 std::string smapFile        = "",
                 std::string derivatives     = "",
                 bool        embedded        = false,
                 bool        const_predict   = false,
                 bool        verbose         = true );

// 3) Data path/file with external solver object, init to default SVD
SMapValues SMap( std::string pathIn          = "./data/",
                 std::string dataFile        = "",
                 std::string pathOut         = "./",
                 std::string predictFile     = "",
                 std::string lib             = "",
                 std::string pred            = "",
                 int         E               = 0,
                 int         Tp              = 1,
                 int         knn             = 0,
                 int         tau             = -1,
                 double      theta           = 0,
                 int         exclusionRadius = 0,
                 std::string columns         = "",
                 std::string target          = "",
                 std::string smapFile        = "",
                 std::string derivatives     = "",
                 std::valarray< double > (*solver)
                     (DataFrame     < double >,
                      std::valarray < double >) = & SVD,
                 bool        embedded        = false,
                 bool        const_predict   = false,
                 bool        verbose         = true );

// 4) DataFrame with external solver object, init to default SVD
SMapValues SMap( DataFrame< double > &dataFrameIn,
                 std::string pathOut         = "./",
                 std::string predictFile     = "",
                 std::string lib             = "",
                 std::string pred            = "",
                 int         E               = 0,
                 int         Tp              = 1,
                 int         knn             = 0,
                 int         tau             = -1,
                 double      theta           = 0,
                 int         exclusionRadius = 0,
                 std::string columns         = "",
                 std::string target          = "",
                 std::string smapFile        = "",
                 std::string derivatives     = "",
                 std::valarray< double > (*solver)
                     (DataFrame     < double >,
                      std::valarray < double >) = & SVD,
                 bool        embedded        = false,
                 bool        const_predict   = false,
                 bool        verbose         = true );

CCMValues CCM( std::string pathIn       = "./data/",
               std::string dataFile     = "",
               std::string pathOut      = "./",
               std::string predictFile  = "",
               int         E            = 0,
               int         Tp           = 0,
               int         knn          = 0,
               int         tau          = -1,
               std::string colNames     = "",
               std::string targetName   = "",
               std::string libSizes_str = "",
               int         sample       = 0,
               bool        random       = true,
               bool        replacement  = false,
               unsigned    seed         = 0,     // seed=0: use RNG
               bool        includeData  = false,
               bool        verbose      = true );

CCMValues CCM( DataFrame< double > & dataFrameIn,
               std::string         pathOut      = "./",
               std::string         predictFile  = "",
               int                 E            = 0,
               int                 Tp           = 0,
               int                 knn          = 0,
               int                 tau          = -1,
               std::string         colNames     = "",
               std::string         targetName   = "",
               std::string         libSizes_str = "",
               int                 sample       = 0,
               bool                random       = true,
               bool                replacement  = false,
               unsigned            seed         = 0, // seed=0: use RNG
               bool                includeData  = false,
               bool                verbose      = true );

MultiviewValues Multiview( std::string pathIn          = "./",
                           std::string dataFile        = "",
                           std::string pathOut         = "./",
                           std::string predictFile     = "",
                           std::string lib             = "",
                           std::string pred            = "",
                           int         D               = 0,
                           int         E               = 1,
                           int         Tp              = 1,
                           int         knn             = 0,
                           int         tau             = -1,
                           std::string columns         = "",
                           std::string target          = "",
                           int         multiview       = 0,
                           int         exclusionRadius = 0,
                           bool        trainLib        = true,
                           bool        verbose         = false,
                           unsigned    nThreads        = 4 );

MultiviewValues Multiview( DataFrame< double > & dataFrameIn,
                           std::string pathOut         = "./",
                           std::string predictFile     = "",
                           std::string lib             = "",
                           std::string pred            = "",
                           int         D               = 0,
                           int         E               = 1,
                           int         Tp              = 1,
                           int         knn             = 0,
                           int         tau             = -1,
                           std::string columns         = "",
                           std::string target          = "",
                           int         multiview       = 0,
                           int         exclusionRadius = 0,
                           bool        trainLib        = true,
                           bool        verbose         = false,
                           unsigned    nThreads        = 4 );

DataFrame< double > EmbedDimension( std::string pathIn      = "./data/",
                                    std::string dataFile    = "",
                                    std::string pathOut     = "./",
                                    std::string predictFile = "",
                                    std::string lib         = "",
                                    std::string pred        = "",
                                    int         maxE        = 10,
                                    int         Tp          = 1,
                                    int         tau         = -1,
                                    std::string colNames    = "",
                                    std::string targetName  = "",
                                    bool        embedded    = false,
                                    bool        verbose     = true,
                                    unsigned    nThreads    = 4 );

DataFrame< double > EmbedDimension( DataFrame< double > & dataFrameIn,
                                    std::string pathOut     = "./",
                                    std::string predictFile = "",
                                    std::string lib         = "",
                                    std::string pred        = "",
                                    int         maxE        = 10,
                                    int         Tp          = 1,
                                    int         tau         = -1,
                                    std::string colNames    = "",
                                    std::string targetName  = "",
                                    bool        embedded    = false,
                                    bool        verbose     = true,
                                    unsigned    nThreads    = 4 );

DataFrame< double > PredictInterval( std::string pathIn      = "./data/",
                                     std::string dataFile    = "",
                                     std::string pathOut     = "./",
                                     std::string predictFile = "",
                                     std::string lib         = "",
                                     std::string pred        = "",
                                     int         maxTp       = 10,
                                     int         E           = 0,
                                     int         tau         = -1,
                                     std::string colNames    = "",
                                     std::string targetName  = "",
                                     bool        embedded    = false,
                                     bool        verbose     = true,
                                     unsigned    nThreads    = 4 );

DataFrame< double > PredictInterval( DataFrame< double > & dataFrameIn,
                                     std::string pathOut     = "./",
                                     std::string predictFile = "",
                                     std::string lib         = "",
                                     std::string pred        = "",
                                     int         maxTp       = 10,
                                     int         E           = 0,
                                     int         tau         = -1,
                                     std::string colNames    = "",
                                     std::string targetName  = "",
                                     bool        embedded    = false,
                                     bool        verbose     = true,
                                     unsigned    nThreads    = 4 );

DataFrame< double > PredictNonlinear( std::string pathIn      = "./data/",
                                      std::string dataFile    = "",
                                      std::string pathOut     = "./",
                                      std::string predictFile = "",
                                      std::string lib         = "",
                                      std::string pred        = "",
                                      std::string theta       = "",
                                      int         E           = 0,
                                      int         Tp          = 1,
                                      int         knn         = 0,
                                      int         tau         = -1,
                                      std::string colNames    = "",
                                      std::string targetName  = "",
                                      bool        embedded    = false,
                                      bool        verbose     = true,
                                      unsigned    nThreads    = 4 );

DataFrame< double > PredictNonlinear( DataFrame< double > & dataFrameIn,
                                      std::string pathOut     = "./",
                                      std::string predictFile = "",
                                      std::string lib         = "",
                                      std::string pred        = "",
                                      std::string theta       = "",
                                      int         E           = 0,
                                      int         Tp          = 1,
                                      int         knn         = 0,
                                      int         tau         = -1,
                                      std::string colNames    = "",
                                      std::string targetName  = "",
                                      bool        embedded    = false,
                                      bool        verbose     = true,
                                      unsigned    nThreads    = 4 );
#endif
