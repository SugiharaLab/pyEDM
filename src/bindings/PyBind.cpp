// Expose cpp wrapper functions to EDM module via pybind11

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "PyBind.h"

#include "DataFrame.cpp"
#include "ComputeError.cpp"
#include "Embed.cpp"
#include "Simplex.cpp"
#include "SMap.cpp"
#include "Multiview.cpp"
#include "CCM.cpp"
#include "EmbedDim.cpp"
#include "PredictInterval.cpp"
#include "PredictNL.cpp"

//-------------------------------------------------------------------------
// PYBIND11_MODULE macro creates entry points invoked when the Python
// interpreter imports the extension module. The module name is given as the
// fist argument and it should not be in quotes. The second macro argument
// defines a variable of type py::module used to initialize the module.
//-------------------------------------------------------------------------
PYBIND11_MODULE( pyBindEDM, pyMod ) {

    pyMod.doc() = "Python bindings to cppEDM via pybind11.";

    py::class_<DF>( pyMod, "DF" )
        .def( py::init() )  // Default constructor (?)
        .def_readwrite( "timeName", &DF::timeName )
        .def_readwrite( "time",     &DF::time     )
        .def_readwrite( "dataList", &DF::dataList );
    
    pyMod.def( "ComputeError", &ComputeError_pybind );

    pyMod.def( "DataFrameToDF", &DataFrameToDF );

    // Load cppEDM DataFrame( path, file ) into py::dict
    pyMod.def( "ReadDataFrame", &ReadDataFrame,
               py::arg("path")   = "",
               py::arg("file")   = "",
               py::arg("noTime") = false );
    
    pyMod.def( "MakeBlock", &MakeBlock_pybind,
               py::arg("pyInput")     = DF(),
               py::arg("E")           = 0,
               py::arg("tau")         = 0,
               py::arg("columnNames") = std::vector<std::string>(),
               py::arg("verbose")     = false );

    //-------------------------------------------------------------
    // In cppEDM, these API's have two call signatures:
    //   1) path and dataFile arguments for data file loading
    //   2) DF object
    // Here, we combine both sets of arguments, and in the .cpp
    // code dispatch the appropriate overloaded cppEDM API.
    //-------------------------------------------------------------
    pyMod.def( "Embed", &Embed_pybind,
               py::arg("path")     = std::string(""),
               py::arg("dataFile") = std::string(""),
               py::arg("pyInput")  = DF(),
               py::arg("E")        = 0,
               py::arg("tau")      = 0,
               py::arg("columns")  = std::string(""),
               py::arg("verbose")  = false );
    
    pyMod.def( "Simplex", &Simplex_pybind,
               py::arg("pathIn")      = std::string("./"),
               py::arg("dataFile")    = std::string(""),
               py::arg("pyInput")     = DF(),
               py::arg("pathOut")     = std::string("./"),
               py::arg("predictFile") = std::string(""),
               py::arg("lib")         = std::string(""),
               py::arg("pred")        = std::string(""),
               py::arg("E")           = 0,
               py::arg("Tp")          = 1,
               py::arg("knn")         = 0,
               py::arg("tau")         = 1,
               py::arg("exclusionRadius") = 0,
               py::arg("columns")     = std::string(""),
               py::arg("target")      = std::string(""),
               py::arg("embedded")    = false,
               py::arg("const_pred")  = false,
               py::arg("verbose")     = false );
    
    pyMod.def( "SMap", &SMap_pybind,
               py::arg("pathIn")      = std::string("./"),
               py::arg("dataFile")    = std::string(""),
               py::arg("pyInput")     = DF(),
               py::arg("pathOut")     = std::string("./"),
               py::arg("predictFile") = std::string(""),
               py::arg("lib")         = std::string(""),
               py::arg("pred")        = std::string(""),
               py::arg("E")           = 0,
               py::arg("Tp")          = 1,
               py::arg("knn")         = 0,
               py::arg("tau")         = 1,
               py::arg("theta")       = 0,
               py::arg("exclusionRadius") = 0,
               py::arg("columns")     = std::string(""),
               py::arg("target")      = std::string(""),
               py::arg("smapFile")    = std::string(""),
               py::arg("derivatives") = std::string(""),
               py::arg("embedded")    = false,
               py::arg("const_pred")  = false,
               py::arg("verbose")     = false );

    pyMod.def( "Multiview", &Multiview_pybind,
               py::arg("pathIn")          = std::string("./"),
               py::arg("dataFile")        = std::string(""),
               py::arg("pyInput")         = DF(),
               py::arg("pathOut")         = std::string("./"),
               py::arg("predictFile")     = std::string(""),
               py::arg("lib")             = std::string(""),
               py::arg("pred")            = std::string(""),
               py::arg("E")               = 0,
               py::arg("Tp")              = 1,
               py::arg("knn")             = 0,
               py::arg("tau")             = 1,
               py::arg("columns")         = std::string(""),
               py::arg("target")          = std::string(""),
               py::arg("multiview")       = 0,
               py::arg("exclusionRadius") = 0,
               py::arg("verbose")         = false,
               py::arg("numThreads")      = 4 );
    
    pyMod.def( "CCM", &CCM_pybind,
               py::arg("pathIn")      = std::string("./"),
               py::arg("dataFile")    = std::string(""),
               py::arg("pyInput")     = DF(),
               py::arg("pathOut")     = std::string("./"),
               py::arg("predictFile") = std::string(""),
               py::arg("E")           = 0,
               py::arg("Tp")          = 0,
               py::arg("knn")         = 0,
               py::arg("tau")         = 1,
               py::arg("columns")     = std::string(""),
               py::arg("target")      = std::string(""),
               py::arg("libSizes")    = std::string(""),
               py::arg("sample")      = 0,
               py::arg("random")      = true,
               py::arg("replacement") = false,
               py::arg("seed")        = 0,
               py::arg("verbose")     = false );
    
    pyMod.def( "EmbedDimension", &EmbedDimension_pybind,
               py::arg("pathIn")      = std::string("./"),
               py::arg("dataFile")    = std::string(""),
               py::arg("pyInput")     = DF(),
               py::arg("pathOut")     = std::string("./"),
               py::arg("predictFile") = std::string(""),
               py::arg("lib")         = std::string(""),
               py::arg("pred")        = std::string(""),
               py::arg("maxE")        = 10,
               py::arg("Tp")          = 1,
               py::arg("tau")         = 1,
               py::arg("columns")     = std::string(""),
               py::arg("target")      = std::string(""),
               py::arg("embedded")    = false,
               py::arg("verbose")     = false,
               py::arg("numThreads")  = 4 );

    pyMod.def( "PredictInterval", &PredictInterval_pybind,
               py::arg("pathIn")      = std::string("./"),
               py::arg("dataFile")    = std::string(""),
               py::arg("pyInput")     = DF(),
               py::arg("pathOut")     = std::string("./"),
               py::arg("predictFile") = std::string(""),
               py::arg("lib")         = std::string(""),
               py::arg("pred")        = std::string(""),
               py::arg("maxTp")       = 10,
               py::arg("E")           = 0,
               py::arg("tau")         = 1,
               py::arg("columns")     = std::string(""),
               py::arg("target")      = std::string(""),
               py::arg("embedded")    = false,
               py::arg("verbose")     = false,
               py::arg("numThreads")  = 4 );

    pyMod.def( "PredictNonlinear", &PredictNonlinear_pybind,
               py::arg("pathIn")      = std::string("./"),
               py::arg("dataFile")    = std::string(""),
               py::arg("pyInput")     = DF(),
               py::arg("pathOut")     = std::string("./"),
               py::arg("predictFile") = std::string(""),
               py::arg("lib")         = std::string(""),
               py::arg("pred")        = std::string(""),
               py::arg("theta")       = std::string(""),
               py::arg("E")           = 0,
               py::arg("Tp")          = 1,
               py::arg("knn")         = 0,
               py::arg("tau")         = 1,
               py::arg("columns")     = std::string(""),
               py::arg("target")      = std::string(""),
               py::arg("embedded")    = false,
               py::arg("verbose")     = false,
               py::arg("numThreads")  = 4 );
}
