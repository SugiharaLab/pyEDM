
#ifndef EDM_SIMPLEX_H
#define EDM_SIMPLEX_H

#include "EDM.h"

//----------------------------------------------------------------
// Simplex class inherits from EDM class and defines
// Simplex-specific projection methods
//----------------------------------------------------------------
class SimplexClass : public EDM {
public:
    // Constructor
    SimplexClass ( DataFrame<double> & data,
                   Parameters        & parameters );

    // Method declarations
    void Project();
    void Simplex();
    void WriteOutput();
};
#endif
