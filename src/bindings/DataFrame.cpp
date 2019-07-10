
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

        // JP ----------------------------------------------------
        // SMap coeff df.time is wrong size for dataList
        //    Need to fix in cppEDM ?
        //    This is a hack until it's fixed....
        size_t N_time     = df.time.size();
        size_t N_dataList = df.dataList.begin()->second.size();

        if ( N_time != N_dataList ) {
            df.time.erase( df.time.begin(), df.time.begin() + 1 );
        }
        // JP ----------------------------------------------------
        
        D[ py::str( df.timeName ) ] = df.time;
    }

#ifdef DEBUG
    // JP -----------------------------------------------------------------
    std::cout << "DFtoDict() df.time.size=" << df.time.size()
              << " df.dataList[0].size="
              << df.dataList.begin()->second.size() << std::endl;
    // JP -----------------------------------------------------------------
#endif
    
    // Data
    for ( auto pi = df.dataList.begin(); pi != df.dataList.end(); pi++ ) {

        // Ignore the time vector if in dataList: already handled above
        // JP ----------------------------------------------------
        //    For SMap coef this overwrites Time, but with what?
        //    The length of the above coeff df.time vector is wrong
        //    for the SMAP coef dataList
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
py::dict ReadDataFrame( std::string path, std::string file ) {

    // DataFrame stores first column of time into
    // DataFrame.Time() vector< string > and dataFrame.TimeName()
    DataFrame< double > dataFrame = DataFrame< double >( path, file );

    // list< pair< string, valarray<double> >>
    DF df = DataFrameToDF( dataFrame );

    py::dict D = DFtoDict( df );

    return D;
}
