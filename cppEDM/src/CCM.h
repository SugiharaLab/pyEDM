
#ifndef EDM_CCM_H
#define EDM_CCM_H

#include <cstdlib>
#include <random>
#include <unordered_set>
#include <chrono>
#include <queue>
#include <thread>

#include "EDM.h"
#include "Simplex.h"

//----------------------------------------------------------------
// CCM class inherits from Simplex class and defines
// CCM-specific projection methods
//----------------------------------------------------------------
class CCMClass : public SimplexClass {
public:
    // CCM implements two Simplex objects for cross mapping
    SimplexClass colToTarget;
    SimplexClass targetToCol;

    // Cross mapping results are stored here
    DataFrame< double > allLibStats; // CCM unified libsize, rho, RMSE, MAE
    CrossMapValues      colToTargetValues; // CCM CrossMap() thread results
    CrossMapValues      targetToColValues; // CCM CrossMap() thread results

    // Constructor
    CCMClass ( DataFrame< double > & data,
               Parameters          & parameters );

    // Method declarations
    void Project();
    void SetupParameters();
    void CCM();
    void FormatOutput();
    void WriteOutput();
};
#endif
