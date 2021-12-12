#ifndef PARAMETER_H
#define PARAMETER_H

#include <algorithm>
#include <numeric>

#include "Common.h"
#include "Version.h"

class ParameterContainer; // forward declaration

//------------------------------------------------------------
//
//------------------------------------------------------------
class Parameters {

public: // No need for protected or private
    Method      method;             // Simplex or SMap enum class

    std::string pathIn;             // path for input dataFile
    std::string dataFile;           // input dataFile (assumed .csv)
    std::string pathOut;            // path for output files
    std::string predictOutputFile;  // path for output file

    std::string lib_str;            // multi argument parameters for library
    std::string pred_str;           // multi argument parameters for prediction

    std::vector<size_t> library;    // library row indices
    std::vector<size_t> prediction; // prediction row indices

    int         E;                  // dimension
    int         Tp;                 // prediction interval
    int         knn;                // k nearest neighbors
    int         tau;                // block embedding delay
    double      theta;              // S Map localization
    int         exclusionRadius;    // temporal rows to ignore in predict

    std::string                columns_str;
    std::string                target_str;
    std::vector< std::string > columnNames; // column name(s)
    std::string                targetName;  // target column name

    bool        embedded;          // true if data is already embedded/block
    bool        const_predict;     // true to compute non "predictor" stats
    bool        verbose;

    std::vector<bool> validLib;    // maps row to valid library flag

    int         generateSteps;     // Number of timesteps to feedback generate

    bool        parameterList;     // Add parameter list to output

    std::string SmapOutputFile;    // path for output file
    std::string blockOutputFile;   // Embed() output file

    int         multiviewEnsemble; // Number of ensembles in multiview
    int         multiviewD;        // Multiview state-space dimension
    bool        multiviewTrainLib; // Use prediction as training library
    bool        multiviewExcludeTarget; // Exclude target from eval combos

    std::string libSizes_str;
    std::vector< size_t > librarySizes;// CCM library sizes to evaluate
    int         subSamples;       // CCM number of samples to draw
    bool        randomLib;        // CCM randomly select subsets if true
    bool        replacement;      // CCM random select with replacement if true
    unsigned    seed;             // CCM random selection RNG seed
    bool        includeData;      // CCM include all simplex projection results

    bool        validated;

    Version version; // Version object, instantiated in constructor
    
    std::map< std::string, std::string > Map;

    friend std::ostream& operator<<( std::ostream & os, Parameters & params );

    // Constructor declaration and default arguments
    Parameters(
        Method      method            = Method::None,
        std::string pathIn            = "./",
        std::string dataFile          = "",
        std::string pathOut           = "./",
        std::string predictOutputFile = "",

        std::string lib_str           = "",
        std::string pred_str          = "",

        int         E                 = 0,
        int         Tp                = 0,
        int         knn               = 0,
        int         tau               = -1,
        double      theta             = 0,
        int         exclusionRadius   = 0,

        std::string columns_str       = "",
        std::string target_str        = "",

        bool        embedded          = false,
        bool        const_predict     = false,
        bool        verbose           = false,

        std::vector<bool> validLib    = std::vector<bool>(),

        int         generateSteps     = 0,
        bool        parameterList     = false,

        std::string SmapOutputFile    = "",
        std::string blockOutputFile   = "",        

        int         multiviewEnsemble      = 0,
        int         multiviewD             = 0,
        bool        multiviewTrainLib      = true,
        bool        multiviewExcludeTarget = false,

        std::string libSizes_str      = "",
        int         subSamples        = 0,
        bool        randomLib         = true,
        bool        replacement       = false,
        unsigned    seed              = 0,  // 0: Generate random seed in CCM
        bool        includeData       = false
    );

    ~Parameters();

    void Validate();      // Parameter validation and index offsets
    void AdjustLibPred(); // Adjust for embedding
    void FillMap();
    void PrintIndices( std::vector< size_t > library,
                       std::vector< size_t > prediction );
};
#endif
