#ifndef AUXFUNC
#define AUXFUNC

#include "Common.h"

#include "Neighbors.h"
#include "Embed.h"

//----------------------------------------------------------------
// Data Input, embedding and NN structure to accomodate
// common initial processing in Simplex and Smap
//----------------------------------------------------------------
struct DataEmbedNN {
    DataFrame<double>     dataIn;
    DataFrame<double>     dataFrame;
    std::valarray<double> targetVec;
    Neighbors             neighbors;
    
    // Constructors
    DataEmbedNN() {}
    
    DataEmbedNN( DataFrame<double>     dataIn,
                 DataFrame<double>     dataFrame,
                 std::valarray<double> targetVec,
                 Neighbors             neighbors ) :
        dataIn( dataIn ), dataFrame( dataFrame ), targetVec( targetVec ),
        neighbors( neighbors ) {}
};

// Prototypes
DataEmbedNN EmbedNN( DataFrame<double> dataIn,
                     Parameters        param,
                     bool              checkDataRows = true );
    
DataFrame<double> FormatOutput( Parameters            param,
                                size_t                N_row,
                                std::valarray<double> predictions,
                                DataFrame<double>     dataFrameIn,
                                std::valarray<double> target_vec,
                                bool                  checkDataRows = true );

void CheckDataRows( Parameters        param,
                    DataFrame<double> dataFrameIn,
                    std::string       call );
#endif
