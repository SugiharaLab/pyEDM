#ifndef DATETIMEUTIL_H
#define DATETIMEUTIL_H

#include <sstream>
#include <vector>
#include <algorithm> // std::count

const int ISO_StartYear  = 1900;
const int ISO_StartMonth = 1;

struct DatetimeInfo {
    struct tm   time = {};
    std::string format;
    bool        unrecognized = false;
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
