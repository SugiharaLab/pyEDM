#ifndef EMBED_H
#define EMBED_H

#include "Common.h"
#include "Parameter.h"

//----------------------------------------------------------------
// API Overload 1: Explicit data file path/name
//   Implemented as a wrapper to API Overload 2:
//   which is a wrapper for MakeBlock()
//----------------------------------------------------------------
DataFrame< double > Embed ( std::string path     = "",
                            std::string dataFile = "",
                            int         E        = 0,
                            int         tau      = 0,
                            std::string columns  = "",
                            bool        verbose  = false );

//----------------------------------------------------------------
// API Overload 2: DataFrame provided
//   Implemented as a wrapper for MakeBlock()
//----------------------------------------------------------------
DataFrame< double > Embed ( DataFrame< double > dataFrame,
                            int                 E       = 0,
                            int                 tau     = 0,
                            std::string         columns = "",
                            bool                verbose = false );

//----------------------------------------------------------------
//----------------------------------------------------------------
DataFrame< double > MakeBlock ( DataFrame< double >      dataFrame,
                                int                      E,
                                int                      tau,
                                std::vector<std::string> columnNames,
                                bool                     verbose );
#endif
