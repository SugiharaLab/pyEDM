
#include "DateTime.h"

//  Provide some utility for parsing datetime std::strings
//  to add some tp increment to a datetime std::string past the given
//  range the time column

//---------------------------------------------------------------------
//  TIME FORMATS supported:
//    YYYY-MM-DD
//    HH:MM:SS
//    YYYY-MM-DDTHH:MM:SS   (2019-06-30T10:26:10)
//    hh:mm:ss.sss
//---------------------------------------------------------------------
// regex's used in parsing and their time formats. in pair for easier checking
std::regex  regEx_yyyymmdd      ("\\d{4}-\\d{2}-\\d{2}");
std::string fmt_yyyymmdd        ("%Y-%m-%d");
std::regex  regEx_hhmmss        ("\\d{2}:\\d{2}:\\d{2}");
std::string fmt_hhmmss          ("%H:%M:%S");
std::regex  regEx_yymmddthhmmss ("\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}");
std::string fmt_yymmddthhmmss   ("%Y-%m-%dT%H:%M:%S");
std::regex  regEx_yymmddhhmmss  ("\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}");
std::string fmt_yymmddhhmmss    ("%Y-%m-%d %H:%M:%S");
std::regex  regEx_hhmmsssss     ("\\d{2}:\\d{2}:\\d{2}\\.\\d{3}");
std::string fmt_hhmmsssss       ("%H:%M:%S");

//----------------------------------------------------------------------
// Parse a date or time string into a tm obj
// @param tm             : the tm object to populate
// @param datetime_str   : the date or time string to populate
// @param date_fmt       : true if this is a date object
// @return               : none, just populates the tm obj
//----------------------------------------------------------------------
void ParseDatetimeString ( struct tm & time_obj, 
                           std::string datetime_str,
                           bool        date_fmt ) {
    // parsing delim is different for date or time
    char parse_delim = date_fmt ? '-' : ':';
    
    // parse the string for it's tokens
    std::stringstream parseable_str ( datetime_str );
    std::string token;
    std::vector<std::string> tokens;
    
    while( getline( parseable_str, token, parse_delim ) ) {
        tokens.push_back( token );
    }
    
    // populate the date time obj
    if ( date_fmt ) {
        time_obj.tm_mday = stod(tokens[2]);
        time_obj.tm_mon  = stod(tokens[1]) - iso_start_month;
        time_obj.tm_year = stod(tokens[0]) - iso_start_year;
    }
    else {
        time_obj.tm_sec  = stod(tokens[2]);
        time_obj.tm_min  = stod(tokens[1]);
        time_obj.tm_hour = stod(tokens[0]);
    }

    int err = mktime( &time_obj );

    if ( err < 0 ) {
        std::stringstream errMsg;
        errMsg << "parse_datetime_str() mktime failed on "
               << datetime_str;
        throw( errMsg.str() );
    }
}

//----------------------------------------------------------------------
// Parse the datetime std::string into a struct tm
// @param  datetime      :  the datetime to parse
// @return datetime      :  the datetime to parse
//----------------------------------------------------------------------
datetime_info ParseDatetime ( std::string datetime ) {
    
    datetime_info output;
    
    // check which time format we have. populate output with each case
    if ( std::regex_match( datetime, regEx_yyyymmdd ) ) {
        output.datetime_fmt = fmt_yyyymmdd; 
        ParseDatetimeString( output.time, datetime, true );            
    }
    else if ( std::regex_match( datetime, regEx_hhmmss )) {
        output.datetime_fmt = fmt_hhmmss; 
        ParseDatetimeString( output.time, datetime, false );            
    }
    else if ( std::regex_match( datetime, regEx_yymmddhhmmss )) {
        output.datetime_fmt = fmt_yymmddhhmmss; 
        // split by " ", then split first by - second by :
        int delim_pos       = datetime.find(' ');
        std::string date    = datetime.substr(0, delim_pos);
        std::string time    = datetime.substr(delim_pos+1, datetime.size());
        ParseDatetimeString( output.time, date, true );            
        ParseDatetimeString( output.time, time, false );            
    }
    else if ( std::regex_match( datetime, regEx_yymmddthhmmss )) {
        output.datetime_fmt = fmt_yymmddthhmmss; 
        // split by T, then split first by - second by :
        int delim_pos       = datetime.find('T');
        std::string date    = datetime.substr(0, delim_pos);
        std::string time    = datetime.substr(delim_pos+1, datetime.size());
        ParseDatetimeString( output.time, date, true );            
        ParseDatetimeString( output.time, time, false );            
    }
    else if ( std::regex_match( datetime, regEx_hhmmsssss )) {
        output.datetime_fmt = fmt_hhmmsssss; 
        // trim the milliseconds off and parse
        datetime = datetime.substr(0,datetime.size()-4);
        ParseDatetimeString( output.time, datetime, false );            
    }
    else {
        output.unrecognized_fmt = true;
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
std::string IncrementDatetime ( std::string datetime1, 
                                std::string datetime2, int tp ) {
    // parse datetimes
    datetime_info dtinfo1 = ParseDatetime( datetime1 );
    datetime_info dtinfo2 = ParseDatetime( datetime2 );
    
    if ( dtinfo1.unrecognized_fmt or dtinfo2.unrecognized_fmt ) {
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
                              dtinfo2.datetime_fmt.c_str(), &dtinfo2.time );
    if ( n_char == 0 ) {
        std::stringstream errMsg;
        errMsg << "increment_datetime_str(): Failed on "
               << datetime1 << ", " << datetime2 << " tp = " << tp;
        throw( errMsg.str() );
    }

    return std::string( tmp_buffer );
}
