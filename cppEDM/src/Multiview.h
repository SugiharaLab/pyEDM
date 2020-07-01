
#ifndef EDM_MULTIVIEW_H
#define EDM_MULTIVIEW_H

#include <thread>
#include <atomic>
#include <mutex>
#include <queue>

#include "EDM.h"
#include "Simplex.h"

//----------------------------------------------------------------
// Multiview class inherits from Simplex class and defines
// CCM-specific projection methods
//----------------------------------------------------------------
class MultiviewClass : public SimplexClass {
public:
    std::string          predictOutputFileIn; // copy from parameters
    std::vector<size_t>  predictionIn;        // copy from parameters

    struct MultiviewValues MVvalues; // output structure
    
    // Constructor
    MultiviewClass ( DataFrame< double > & data,
                     Parameters          & parameters );

    // Method declarations
    void Project( unsigned maxThreads );
    void CheckParameters();
    void SetupParameters();
    void Multiview( unsigned maxThreads );
};
#endif
