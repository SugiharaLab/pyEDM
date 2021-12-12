
#include "PyBind.h"

//-----------------------------------------------------------
// 
//-----------------------------------------------------------
std::map< std::string, py::dict >
    CCM_pybind( std::string pathIn, 
                std::string dataFile,
                DF          df,
                std::string pathOut,
                std::string predictFile,
                int         E,
                int         Tp,
                int         knn,
                int         tau,
                int         exclusionRadius,
                std::string columns,
                std::string target,
                std::string libSizes,
                int         sample,
                bool        random,
                bool        replacement,
                unsigned    seed,
                bool        includeData,
                bool        parameterList,
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
                         exclusionRadius,
                         columns,
                         target, 
                         libSizes,
                         sample,
                         random,
                         replacement,
                         seed,
                         includeData,
                         parameterList,
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
                         exclusionRadius,
                         columns,
                         target, 
                         libSizes,
                         sample,
                         random,
                         replacement,
                         seed,
                         includeData,
                         parameterList,
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

    if ( parameterList ) {
        CCM_["parameters"] = ParamMaptoDict( ccmValues.parameterMap );
    }

    return CCM_;
}
