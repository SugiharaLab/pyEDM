
#include "PyBind.h"

//-----------------------------------------------------------
// 
//-----------------------------------------------------------
py::list CCM_pybind( std::string pathIn, 
                     std::string dataFile,
                     DF          df,
                     std::string pathOut,
                     std::string predictFile,
                     int         E,
                     int         Tp,
                     int         knn,
                     int         tau, 
                     std::string columns,
                     std::string target,
                     std::string libSizes,
                     int         sample,
                     bool        random,
                     bool        replacement,
                     unsigned    seed, 
                     bool        verbose ) {
    
    CCMValues ccmVals;

    if ( dataFile.size() ) {
        // dataFile specified, dispatch overloaded CCM, ignore dataList
        
        ccmVals  = CCM( pathIn,
                         dataFile,
                         pathOut,
                         predictFile,
                         E, 
                         Tp,
                         knn,
                         tau,
                         columns,
                         target, 
                         libSizes,
                         sample,
                         random,
                         replacement,
                         seed,
                         verbose );
    }
    else if ( df.dataList.size() ) {
        DataFrame< double > dataFrame = DFToDataFrame( df );
        
        ccmVals     = CCM( dataFrame,
                         pathOut,
                         predictFile,
                         E, 
                         Tp,
                         knn,
                         tau,
                         columns,
                         target, 
                         libSizes,
                         sample,
                         random,
                         replacement,
                         seed,
                         verbose );
    }
    else {
        throw std::runtime_error( "CCM_pybind(): Invalid input.\n" );
    }

    // Format CCMValues for output list
    
    py::list ccmOutput;

    // all lib stats first

    DF allLibStatsDF     = DataFrameToDF( ccmVals.AllLibStats );
    py::dict allLibStats = DFtoDict( allLibStatsDF );
    ccmOutput.append( allLibStats );

    // Now add each crossmap values' internal DF and [DF]
    for (auto cmapVal : {ccmVals.CrossMap1, ccmVals.CrossMap2} ){

        py::list cmapOut;
        
        // internal pred stats DF
        DF predStatsDF      = DataFrameToDF( cmapVal.PredictStats );
        py::dict statsOut   = DFtoDict( predStatsDF );
        cmapOut.append( statsOut );

        // list of prediction DFs
        py::list preds;
        for (auto predDataFrame : cmapVal.Predictions ) {
            DF predDF           = DataFrameToDF( predDataFrame );
            py::dict predOut    = DFtoDict( predDF );
            preds.append( predOut );
        }
        cmapOut.append( preds );

        ccmOutput.append( cmapOut );
    }
    
    return ccmOutput;
}
