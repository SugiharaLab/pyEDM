#ifndef NEIGHBORS_H
#define NEIGHBORS_H

#include <cmath>
#include <iterator>

#include "Common.h"
#include "Parameter.h"

// Return structure of FindNeighbors()
struct Neighbors {
    DataFrame<size_t> neighbors;
    DataFrame<double> distances;
    Neighbors();
    ~Neighbors();
};

// Prototypes
Neighbors FindNeighbors( DataFrame<double> dataFrame,
                         Parameters        parameters );

void PrintDataFrameIn( const DataFrame<double> &dataFrame,
                       const Parameters        &parameters );

void PrintNeighborsOut( const Neighbors &neighbors );

double Distance( const std::valarray<double> &v1,
                 const std::valarray<double> &v2,
                 DistanceMetric metric );

#endif
