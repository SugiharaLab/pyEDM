
#ifndef EDM_SMAP_H
#define EDM_SMAP_H

#include "EDM.h"

// Prototype & alias of solver function pointer
using Solver = SVDValues (*) ( DataFrame     < double >,
                               std::valarray < double > );

// Prototype declaration of general functions
SVDValues SVD( DataFrame < double > A, std::valarray< double > B );

SVDValues Lapack_SVD( int     m, // number of rows in matrix
                      int     n, // number of columns in matrix
                      double *a, // pointer to top-left corner
                      double *b,
                      double  rcond );

//----------------------------------------------------------------
// SMap class inherits from EDM class and defines
// SMap-specific projection & output methods
//----------------------------------------------------------------
class SMapClass : public EDM {

public:
    // Constructor
    SMapClass ( DataFrame<double> & data,
                Parameters        & parameters );

    // Method declarations
    void Generate( Solver );
    void Project ( Solver );
    void SMap    ( Solver );
    void RecordNan( size_t row, size_t N_SingularValues );
    void WriteOutput();
};
#endif
