#ifndef PARAMETER_H
#define PARAMETER_H

#include <algorithm>
#include <numeric>

#include "Common.h"
#include "Version.h"

//------------------------------------------------------------
//
//------------------------------------------------------------
class Parameters {

public:  // Not protected with accessors.
    Method      method;             // Simplex or SMap enum class
    std::vector<size_t> library;    // library row indices
    std::vector<size_t> prediction; // prediction row indices
    int         E;                  // dimension
    int         Tp;                 // prediction interval
    int         knn;                // k nearest neighbors
    int         tau;                // block embedding delay
    int         exclusionRadius;    // temporal rows to ignore in predict
    
    float       theta;              // S Map localization
    float       SVDSignificance;    // SVD singular value cutoff
    std::vector<size_t> derivatives;// list of column indices for derivatives
    float       TikhonovAlpha;      // Initial alpha parameter
    float       ElasticNetAlpha;    // Initial alpha parameter

    std::vector<std::string> columnNames; // column names 
    std::vector<size_t>      columnIndex; // column indices

    std::string targetName;       // target column name
    size_t      targetIndex;      // target column index

    bool        embedded;         // true if data is already embedded/block
    bool        const_predict;    // true to compute non "predictor" stats
    
    int         MultiviewEnsemble;// Number of ensembles in multiview

    std::vector<size_t> librarySizes;// CCM library sizes to evaluate
    int         subSamples;       // CCM number of samples to draw
    bool        randomLib;        // CCM randomly select subsets if true
    bool        replacement;      // CCM random select with replacement if true
    unsigned    seed;             // CCM random selection RNG seed
    
    bool        noNeighborLimit;  // Strictly forbid neighbors outside library
    bool        forwardTau;       // Embed/block with t+tau instead t-tau

    bool        verbose;
    bool        validated;
    
    std::string pathIn;           // path for input dataFile
    std::string pathOut;          // path for output files
    std::string dataFile;         // input dataFile (assumed .csv)
    std::string predictOutputFile;//
    std::string SmapOutputFile;   //
    std::string blockOutputFile;  // Embed() output file

    std::string lib_str;      // String inputs of multi argument parameters
    std::string pred_str;
    std::string columns_str;
    std::string target_str;
    std::string libSizes_str;
    std::string derivatives_str;

    Version version;  // Version object, instantiated in constructor

    friend std::ostream& operator<<(std::ostream &os, Parameters &params);

    // Constructor declaration and default arguments
    Parameters(
        Method      method      = Method::None,
        std::string pathIn      = "./",
        std::string dataFile    = "",
        std::string pathOut     = "./",
        std::string predictFile = "",
        std::string lib_str     = "",
        std::string pred_str    = "",
        int         E           = 0,
        int         Tp          = 0,
        int         knn         = 0,
        int         tau         = 1,
        float       theta       = 0,
        int         exclusionRadius = 0,

        std::string columns_str = "",
        std::string target_str  = "",

        bool        embedded      = false,
        bool        const_predict = false,
        bool        verbose       = false,
        
        std::string SmapFile        = "",
        std::string blockFile       = "",        
        std::string derivatives_str = "",
        
        float       svdSig       = 1E-5,
        float       tikhonov     = 0,
        float       elasticNet   = 0.1,
        
        int         multi        = 0,
        std::string libSizes_str = "",
        int         sample       = 0,
        bool        random       = true,
        bool        replacement  = false,
        unsigned    seed         = 0,   // 0: Generate random seed in CCM
        bool        noNeighbor   = false,
        bool        forwardTau   = false
    );

    ~Parameters();

    void Validate(); // Parameter validation and index offsets
    void Load();     // Populate the parameters from arguments
    void DeleteLibPred( size_t shift ); // Adjust for embedding
    void PrintIndices( std::vector<size_t> library,
                       std::vector<size_t> prediction );
};

#endif
