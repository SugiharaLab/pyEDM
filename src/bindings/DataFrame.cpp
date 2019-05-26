
#include "PyBind.h"

//-----------------------------------------------------------------------
// Convert DF list< pair<string, valarray> > to cppEDM DataFrame<double>
//-----------------------------------------------------------------------
DataFrame< double > DFToDataFrame ( DF df ) {

    // Get number of valarray rows from first pair
    size_t numRows = 0;
    if ( df.size() ) {
        numRows = df.front().second.size();
    }
    
    // Get column names
    std::vector< std::string > colNames;
    for ( auto colPair : df ) {
        colNames.push_back( colPair.first );
    } 

    // Create cpp DataFrame
    DataFrame< double > dataFrame ( numRows, colNames.size(), colNames ); 

    for ( DF::iterator it = df.begin(); it != df.end(); it++ ) {

        dataFrame.WriteColumn( std::distance( df.begin(), it), it->second ); 

    }

    return dataFrame;
}

//---------------------------------------------------------------
// Convert cppEDM DataFrame<double> to DF = 
//   list< pair< string, valarray<double> >>
//---------------------------------------------------------------
DF DataFrameToDF ( DataFrame< double > dataFrame ) {

    DF df;

    // Setup list with dataframe contents to give to python
    for ( std::string colName : dataFrame.ColumnNames() ) {

        std::valarray< double > colData = dataFrame.VectorColumnName( colName );

        df.push_back( std::make_pair( colName, colData ) );
    }

    return df;
}

//---------------------------------------------------------------
// Convert DF = list< pair< string, valarray<double> >>
// to Python dict
//---------------------------------------------------------------
py::dict DFtoDict ( DF df ) {

    py::dict D;

    for ( auto pi = df.begin(); pi != df.end(); pi++ ) {
        // Why is cast required for string key but not valarray? 
        D[ py::str( pi->first ) ] = pi->second;
    }
    
    return D;
}

//---------------------------------------------------------------
// Load path/file into cppEDM DataFrame, convert to Python
// dict{ column : array }
//---------------------------------------------------------------
py::dict ReadDataFrame ( std::string path, std::string file ) {

    DataFrame< double > dataFrame = DataFrame< double >( path, file );

    // list< pair< string, valarray<double> >>
    DF df = DataFrameToDF( dataFrame );

    py::dict D = DFtoDict( df );

    return D;
}
