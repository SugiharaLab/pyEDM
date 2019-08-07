#ifndef AUXFUNC
#define AUXFUNC

#include <mutex>
#include <functional> // std::ref

#include "Common.h"
#include "Neighbors.h"
#include "Embed.h"

//----------------------------------------------------------------
// Data Input, embedding and NN structure to accommodate
// common initial processing in Simplex and Smap
//----------------------------------------------------------------
struct DataEmbedNN {
    DataFrame<double>    *dataIn;
    DataFrame<double>     dataFrame;
    std::valarray<double> targetVec;
    Neighbors             neighbors;
    
    // Constructors
    DataEmbedNN() {}
    
    DataEmbedNN( DataFrame<double>    *dataIn,
                 DataFrame<double>     dataFrame,
                 std::valarray<double> targetVec,
                 Neighbors             neighbors ) :
        dataIn( dataIn ), dataFrame( dataFrame ), targetVec( targetVec ),
        neighbors( neighbors ) {}
};

// Prototypes
DataEmbedNN EmbedNN( DataFrame<double> *dataIn,
                     Parameters        &param,
                     bool               checkDataRows = true );
    
DataFrame<double> FormatOutput( Parameters               param,
                                std::valarray<double>    predictions,
                                std::valarray<double>    const_predictions,
                                std::valarray<double>    target_vec,
                                std::vector<std::string> time,
                                std::string              timeName );

void FillTimes( Parameters                param,
                std::vector<std::string>  time,
                std::vector<std::string> &timeOut );

void CheckDataRows( Parameters         param,
                    DataFrame<double> &dataFrameIn,
                    std::string        call );
#endif
