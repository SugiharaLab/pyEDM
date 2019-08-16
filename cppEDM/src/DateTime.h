#ifndef DATETIMEUTIL_H
#define DATETIMEUTIL_H

#include <iostream> // for testing
#include <cstdio>   // time formatting
#include <ctime> 
#include <string> 
#include <regex> 
#include <sstream> 
#include <fstream> 

const int iso_start_year  = 1900;
const int iso_start_month = 1;

struct datetime_info { struct tm   time = {};
                       std::string datetime_fmt;
                       bool        unrecognized_fmt; };

// Prototypes
void parse_datetime_str ( struct tm & time_obj, 
                          std::string datetime_str,
                          bool        date_fmt );

datetime_info parse_datetime ( std::string datetime );

std::string increment_datetime_str ( std::string datetime1, 
                                     std::string datetime2,
                                     int         tp );

#endif
