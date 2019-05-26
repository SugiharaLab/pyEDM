
#include "PyBind.h"
#include "Embed.h"

//---------------------------------------------------------------
// 
//---------------------------------------------------------------
py::dict Embed_pybind( std::string path,
                       std::string dataFile,
                       DF          dataList,
                       int         E,
                       int         tau,
                       std::string columns,
                       bool        verbose ) {

    DataFrame< double > embedded;
    
    if ( dataFile.size() ) {
        // dataFile specified, dispatch overloaded Embed, ignore dataList
        embedded = Embed( path,
                          dataFile,
                          E,
                          tau,
                          columns,
                          verbose );
    }
    else if ( dataList.size() ) {
        DataFrame< double > dataFrame = DFToDataFrame( dataList );
        
        embedded = Embed( dataFrame,
                          E,
                          tau,
                          columns,
                          verbose );
    }
    else {
        throw std::runtime_error( "Embed_pybind(): Invalid input.\n" );
    }

    DF       df = DataFrameToDF( embedded );
    py::dict D  = DFtoDict( df );
    
    return D;
}

//---------------------------------------------------------------
// 
//---------------------------------------------------------------
py::dict MakeBlock_pybind( DF                       dataList,
                           int                      E,
                           int                      tau,
                           std::vector<std::string> columnNames,
                           bool                     verbose ) {
    
    DataFrame< double > dataFrame = DFToDataFrame( dataList );

    DataFrame< double > block = MakeBlock( dataFrame,
                                           E,
                                           tau,
                                           columnNames,
                                           verbose );

    DF       df = DataFrameToDF( block );
    py::dict D  = DFtoDict( df );
    
    return D;
}
