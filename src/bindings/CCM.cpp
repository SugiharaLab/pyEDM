
#include "PyBind.h"

//-----------------------------------------------------------
// 
//-----------------------------------------------------------
std::map< std::string, py::dict > CCM_pybind( std::string pathIn, 
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
                                              bool        includeData,
                                              bool        verbose ) {
    
    CCMValues ccmValues;

    if ( dataFile.size() ) {
        // dataFile specified, dispatch overloaded CCM, ignore dataList
        
        ccmValues = CCM( pathIn,
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
                         includeData,
                         verbose );
    }
    else if ( df.dataList.size() ) {
        DataFrame< double > dataFrame = DFToDataFrame( df );
        
        ccmValues = CCM( dataFrame,
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
                         includeData,
                         verbose );
    }
    else {
        throw std::runtime_error( "CCM_pybind(): Invalid input.\n" );
    }
    
    // Format CCMValues for output dict
    std::map< std::string, py::dict > CCM_;

    // LibStats are always included
    DF allLibStats     = DataFrameToDF( ccmValues.AllLibStats );
    CCM_[ "LibMeans" ] = DFtoDict( allLibStats );
    
    // Now add each crossmap prediction stats if requested
    if ( includeData ) {
        DF predStats1 = DataFrameToDF( ccmValues.CrossMap1.PredictStats );
        DF predStats2 = DataFrameToDF( ccmValues.CrossMap2.PredictStats );
        
        CCM_[ "PredictStats1" ] = DFtoDict( predStats1 );
        CCM_[ "PredictStats2" ] = DFtoDict( predStats2 );
    }

    return CCM_;
}
