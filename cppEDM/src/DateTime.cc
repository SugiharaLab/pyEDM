
#include "DateTime.h"

//---------------------------------------------------------------------
//  Provide some utility for parsing datetime std::strings
//  to add some tp increment to a datetime std::string past the
//  given range of the time column.
//
//  Fractional seconds not supported by strpftime or IncrementDatetime
//
//  TIME FORMATS supported:
//    YYYY-MM-DD
//    HH:MM:SS
//    YYYY-MM-DD HH:MM:SS  (2019-06-30 10:26:10)
//    YYYY-MM-DDTHH:MM:SS  (2019-06-30T10:26:10)
//    
//---------------------------------------------------------------------
// Time formats
std::string YMD       ("%Y-%m-%d");
std::string HMS       ("%H:%M:%S");
std::string YMD_HMS   ("%Y-%m-%d %H:%M:%S");
std::string YMD_T_HMS ("%Y-%m-%dT%H:%M:%S");

//----------------------------------------------------------------------
// Parse a date or time string into a tm obj
// tm             : tm object to populate
// datetime_str   : date or time string
// isDate         : true if this is a date object
//----------------------------------------------------------------------
void ParseDatetimeString( struct tm & tmStruct,
                          std::string datetime,
                          bool        isDate ) {
    // parsing delimeter is '-' for date, ':' for time
    char delimeter = isDate ? '-' : ':';

    // parse datetime into tokens
    std::stringstream ss( datetime );
    std::string token;
    std::vector<std::string> tokens;

    while( getline( ss, token, delimeter ) ) {
        tokens.push_back( token );
    }

    // populate the tmStruct
    if ( isDate ) {
        tmStruct.tm_mday = stod(tokens[2]);
        tmStruct.tm_mon  = stod(tokens[1]) - ISO_StartMonth;
        tmStruct.tm_year = stod(tokens[0]) - ISO_StartYear;
    }
    else {
        tmStruct.tm_sec  = stod(tokens[2]);
        tmStruct.tm_min  = stod(tokens[1]);
        tmStruct.tm_hour = stod(tokens[0]);
    }

    int err = mktime( &tmStruct );

    if ( err < 0 ) {
        std::stringstream errMsg;
        errMsg << "ParseDatetimeString() mktime failed on " << datetime;
        throw( errMsg.str() );
    }
}

//----------------------------------------------------------------------
// Parse the datetime into a DatetimeInfo struct
// datetime :  datetime to parse
// return   :  DatetimeInfo struct
//----------------------------------------------------------------------
DatetimeInfo ParseDatetime( std::string datetime ) {

    DatetimeInfo output;

    // Detecting the format is based on delimeters to avoid regex:
    //    [ '-' and '-' ] YMD
    //    [ ':' and ':' ] HMS
    //    [ '-' and '-' and ':' and ':' ] YMD_HMS
    //    [ '-' and '-' and ':' and ':' and 'T' ] YMD_T_HMS

    size_t nHypen = std::count( datetime.begin(), datetime.end(), '-' );
    size_t nColon = std::count( datetime.begin(), datetime.end(), ':' );
    size_t nT     = std::count( datetime.begin(), datetime.end(), 'T' );
    
    if ( nHypen == 2 and nColon == 0 ) {
        output.format = YMD;
        ParseDatetimeString( output.time, datetime, true );
    }
    else if ( nHypen == 0 and nColon == 2 ) {
        output.format = HMS;
        ParseDatetimeString( output.time, datetime, false );
    }
    else if ( nHypen == 2 and nColon == 2 and nT == 0 ) {
        output.format = YMD_HMS; 
        // split by " ", then split first by - second by :
        int delim_pos    = datetime.find(' ');
        std::string date = datetime.substr(0, delim_pos);
        std::string time = datetime.substr(delim_pos+1, datetime.size());
        ParseDatetimeString( output.time, date, true );
        ParseDatetimeString( output.time, time, false );
    }
    else if ( nHypen == 2 and nColon == 2 and nT == 1 ) {
        output.format = YMD_T_HMS; 
        // split by T, then split first by - second by :
        int delim_pos    = datetime.find('T');
        std::string date = datetime.substr(0, delim_pos);
        std::string time = datetime.substr(delim_pos+1, datetime.size());
        ParseDatetimeString( output.time, date, true );           
        ParseDatetimeString( output.time, time, false );        
    }
    else {
        output.unrecognized = true;
    }
    return output; 
}

//----------------------------------------------------------------------
// Generate a new datetime + delta past the range of given
//----------------------------------------------------------------------
//
// @params datetime1/2   :  the two last time std::strings
//                          to compute the delta unit
//                          we increment from datetime2
// @param tp             :  the amount to increment the time diff by
// @return               :  the new incremented timestd::string
//----------------------------------------------------------------------
std::string IncrementDatetime( std::string datetime1, 
                               std::string datetime2, int tp ) {
    // parse datetimes
    DatetimeInfo dtinfo1 = ParseDatetime( datetime1 );
    DatetimeInfo dtinfo2 = ParseDatetime( datetime2 );

    if ( dtinfo1.unrecognized or dtinfo2.unrecognized ) {
        // return empty string
        return std::string();
    }

    // get the delta unit between two datetimes in the time col
    size_t seconds_diff = difftime( mktime( &dtinfo2.time ),
                                    mktime( &dtinfo1.time ) );

    if ( seconds_diff == 0 ) {
        seconds_diff = 1; //if millisec, want some update
    }

    // increment the time and format
    dtinfo2.time.tm_sec += tp * seconds_diff;

    int err = mktime( &dtinfo2.time );

    if ( err < 0 ) {
        std::stringstream errMsg;
        errMsg << "increment_datetime_str() mktime failed on "
               << datetime2;
        throw( errMsg.str() );
    }

    // format incremented time
    char tmp_buffer [ BUFSIZ ];
    
    size_t n_char = strftime( tmp_buffer, BUFSIZ,
                              dtinfo2.format.c_str(), &dtinfo2.time );
    if ( n_char == 0 ) {
        std::stringstream errMsg;
        errMsg << "increment_datetime_str(): Failed on "
               << datetime1 << ", " << datetime2 << " tp = " << tp;
        throw( errMsg.str() );
    }

    return std::string( tmp_buffer );
}
