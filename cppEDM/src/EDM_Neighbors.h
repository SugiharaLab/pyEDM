#ifndef EDM_NEIGHBORS_H
#define EDM_NEIGHBORS_H

#include "EDM.h"

namespace EDM_Distance {
    // Define the initial maximum distance for neigbors
    // DBL_MAX is a Macro equivalent to: std::numeric_limits<double>::max()
    double DistanceMax = std::numeric_limits<double>::max();
}

// Prototypes
double Distance( const std::valarray<double> &v1,
                 const std::valarray<double> &v2,
                 DistanceMetric metric );

bool DistanceCompare( const std::pair<double, size_t> &x,
                      const std::pair<double, size_t> &y );
#endif
