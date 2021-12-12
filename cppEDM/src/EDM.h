
#ifndef EDM_H
#define EDM_H

#include <mutex>
#include "Common.h"
#include "Parameter.h"

//---------------------------------------------------------------------
// EDM Class
// Central data object and base class for EDM algorithms.
// Specific algorithm projection methods defined in sub-classes.
//
// NOTE JP: Tony recommends to explicitly define special members:
//          http://www.cplusplus.com/doc/tutorial/classes2/
//---------------------------------------------------------------------
class EDM {

public: // No need for private or protected
    DataFrame< double > data;
    DataFrame< double > embedding;

    DataFrame< size_t > knn_neighbors; // N pred rows, knn columns; sorted
    DataFrame< double > knn_distances; // N pred rows, knn columns; sorted

    DataFrame< size_t > allLibRows;   // 1 row,       N lib columns
    DataFrame< double > allDistances; // N pred rows  N lib columns

    DataFrame< double > projection;   // Simplex & SMap Output
    DataFrame< double > coefficients; // SMap Output

    // Project() vectors to populate projection DataFrame in FormatData()
    // JP Can we do away with these and write directly to projection (+Tp)?
    std::valarray< double > predictions;
    std::valarray< double > const_predictions;
    std::valarray< double > variance;

    // Simplex :: Prediction row accounting of library neighbor ties
    bool                  anyTies;
    std::vector< bool >   ties;          // true/false each prediction row
    std::vector< size_t > tieFirstIndex; // index in knn of first tie
    std::vector< std::vector< std::pair< double, size_t > > > tiePairs;

    // SMap :: Each prediction row can have variable knn
    std::vector< size_t > knnSmap;

    std::valarray< double >    target;  // entire record
    std::vector< std::string > allTime; // entire record

    int embedShift; // number of data rows lost to embedding

    Parameters parameters;

    // Constructor declaration
    EDM ( DataFrame< double > & data, Parameters & parameters );

    // Method declarations
    // EDM.cc
    void GetTarget();
    void EmbedData();
    void Project();  // Simplex.cc : SMap.cc : CCM.cc : Multiview.cc
    void Generate(); // Simplex.cc : SMap.cc

    // EDM_Neighbors.cc
    void PrepareEmbedding( bool checkDataRows = true );
    void Distances();
    void FindNeighbors();

    // EDM_Formatting.cc
    void CheckDataRows( std::string call );
    void CheckValidLib( std::string call );
    void FormatOutput();
    void FillTimes( std::vector< std::string > & timeOut );

    void PrintDataFrameIn(); // EDM_Neighbors.cc #ifdef DEBUG_ALL
    void PrintNeighbors();   // EDM_Neighbors.cc #ifdef DEBUG_ALL
};
#endif
