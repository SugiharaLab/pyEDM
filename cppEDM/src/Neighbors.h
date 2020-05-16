#ifndef NEIGHBORS_H
#define NEIGHBORS_H

#include <cmath>
#include <iterator>
#include <functional>

#include "Common.h"
#include "Parameter.h"

// Return structure of FindNeighbors() & Distances()
struct Neighbors {
    DataFrame<size_t> neighbors;
    DataFrame<double> distances;

    bool anyTies; // Are there ties?
    std::vector< bool > ties; // true : false for each prediction row
    std::vector< std::vector< std::pair< double, size_t > > > tiePairs;
    
    Neighbors();
    ~Neighbors();
};

// Prototypes
Neighbors Distances( const DataFrame< double > &dataBlock,
                           Parameters           param );

Neighbors FindNeighbors( DataFrame<double> dataFrame,
                         Parameters        parameters );

void PrintDataFrameIn( const DataFrame<double> &dataFrame,
                       const Parameters        &parameters );

void PrintNeighborsOut( const Neighbors &neighbors );

double Distance( const std::valarray<double> &v1,
                 const std::valarray<double> &v2,
                 DistanceMetric metric );

#endif
