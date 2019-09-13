
#include "PyBind.h"

//-----------------------------------------------------------------------
// Convert DF struct to cppEDM DataFrame<double>
// 
// Note : cppEDM DataFrame stores time vector in dataFrame.Time()
//        and label in dataFrame.TimeName()
//-----------------------------------------------------------------------
DataFrame< double > DFToDataFrame( DF df ) {

    // Get number of valarray rows from first pair
    size_t numRows = 0;
    if ( df.dataList.size() ) {
        numRows = df.dataList.front().second.size();
    }
    
    // Get data column names
    std::vector< std::string > colNames;
    for ( auto colPair : df.dataList ) {
        colNames.push_back( colPair.first );
    } 

    // Create cpp DataFrame
    DataFrame< double > dataFrame( numRows, colNames.size(), colNames );

    // Insert time into dataFrame.time and dataFrame.timeName
    dataFrame.TimeName() = df.timeName;
    dataFrame.Time()     = df.time;

    // Insert DF data items into dataFrame columns.
    for ( auto it = df.dataList.begin(); it != df.dataList.end(); it++ ) {

        dataFrame.WriteColumn( std::distance( df.dataList.begin(), it ),
                               it->second ); 
    }

    return dataFrame;
}

//---------------------------------------------------------------
// Convert cppEDM DataFrame<double> to DF struct.
// 
// Note : cppEDM DataFrame stores time vector in dataFrame.Time()
//        and label in dataFrame.TimeName()
//---------------------------------------------------------------
DF DataFrameToDF( DataFrame< double > dataFrame ) {

    DF df;

    // Copy time vector and name
    df.time     = dataFrame.Time();
    df.timeName = dataFrame.TimeName();

    // Copy data to DF dataList
    for ( std::string colName : dataFrame.ColumnNames() ) {

        std::valarray< double > colData = dataFrame.VectorColumnName( colName );

        df.dataList.push_back( std::make_pair( colName, colData ) );
    }

    return df;
}

//---------------------------------------------------------------
// Convert DF struct to Python dict
//---------------------------------------------------------------
py::dict DFtoDict( DF df ) {

    py::dict D;

    // Time
    // In PyBind.h struct DF and cppEDM DataFrame time and timeName are
    // separate members, data are a list< pair< string, valarray >>
    // and valarray respectively.  Here, we copy the time vector into
    // the py::dict as just another column.
    
    if ( df.timeName.size() and df.time.size() ) {
        // SMap coef df.data will not be as long as prediction data
        // since SMap coef are only available for observations
        size_t N_time     = df.time.size();
        size_t N_dataList = df.dataList.begin()->second.size();
        size_t offset     = N_time - N_dataList;

        if ( offset > N_dataList ) {
            std::stringstream errMsg;
            errMsg << "DFtoDict(): Invalid offset for time adjustment.";
            throw std::runtime_error( errMsg.str() );
        }
        
        if ( offset ) {
            df.time.erase( df.time.end() - offset, df.time.end() );
        }
        
        D[ py::str( df.timeName ) ] = df.time;
    }
    
    // Data
    for ( auto pi = df.dataList.begin(); pi != df.dataList.end(); pi++ ) {

        // Ignore the time vector if in dataList: already handled above
        // JP ----------------------------------------------------
        //    For SMap coef this overwrites Time, but with what?
        // JP ----------------------------------------------------
        if ( df.timeName == pi->first ) {
            continue;
        }
        
        // Why is cast required for string key but not valarray? 
        D[ py::str( pi->first ) ] = pi->second;
    }
    
    return D;
}

//---------------------------------------------------------------
// Load path/file into cppEDM DataFrame, convert to Python
// dict{ column : array }
//---------------------------------------------------------------
py::dict ReadDataFrame( std::string path, std::string file,
                        bool noTime = false ) {

    // DataFrame stores first column of time into
    // DataFrame.Time() vector< string > and dataFrame.TimeName()
    DataFrame< double > dataFrame = DataFrame< double >( path, file, noTime );

    // list< pair< string, valarray<double> >>
    DF df = DataFrameToDF( dataFrame );

    py::dict D = DFtoDict( df );

    return D;
}
