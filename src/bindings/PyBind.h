#ifndef WRAPPER_COMMON
#define WRAPPER_COMMON

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "Common.h"  // cpp DataFrame

// There are three data frame representations:
//   1) C++ DataFrame class defined in cppEDM DataFrame.h
//   2) DF: C++ struct to hold DataFrame values
//   3) py::dict Python dictionary equivalent of DF

// DF container to hold cppEDM DataFrame values
typedef std::list< std::pair< std::string, std::valarray<double> > > DataList;
struct DF {
    std::string                timeName;
    std::vector< std::string > time;
    DataList                   dataList;
};

// Forward declarations for DataFrameWrapper.cpp convertors
DataFrame< double > DFToDataFrame ( DF df );
DF                  DataFrameToDF ( DataFrame< double > dataFrame );

#endif
