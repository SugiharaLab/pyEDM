#ifndef DATETIMEUTIL_H
#define DATETIMEUTIL_H

#include <sstream>
#include <vector>
#include <algorithm> // std::count
#include <time.h>    // mktime

const int ISO_StartYear  = 1900;
const int ISO_StartMonth = 1;

struct DatetimeInfo {
    struct      tm time = {};
    std::string format;
    bool        unrecognized = false;

    // Constructor : setup time struct
    DatetimeInfo() {
        time.tm_sec   = 0;
        time.tm_min   = 0;
        time.tm_hour  = 0;
        time.tm_mday  = 1;
        time.tm_mon   = 0;
        time.tm_year  = 70; // Minimal valid Unix time 1900 + 70
        time.tm_wday  = 0;
        time.tm_yday  = 0;
        time.tm_isdst = 0;
    }
};

// Prototypes
void ParseDatetimeString( struct tm & tmStruct,
                          std::string datetime,
                          bool        isDate );

DatetimeInfo ParseDatetime( std::string datetime );

std::string IncrementDatetime( std::string datetime1,
                               std::string datetime2,
                               int         tp );
#endif
