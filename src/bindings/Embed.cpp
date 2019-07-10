
#include "PyBind.h"
#include "Embed.h"

//---------------------------------------------------------------
// 
//---------------------------------------------------------------
py::dict Embed_pybind( std::string path,
                       std::string dataFile,
                       DF          df,
                       int         E,
                       int         tau,
                       std::string columns,
                       bool        verbose ) {

    DataFrame< double > embedded;
    
    if ( dataFile.size() ) {
        // dataFile specified, ignore df.dataList
        embedded = Embed( path,
                          dataFile,
                          E,
                          tau,
                          columns,
                          verbose );
    }
    else if ( df.dataList.size() ) {
        DataFrame< double > dataFrame = DFToDataFrame( df );
        
        embedded = Embed( dataFrame,
                          E,
                          tau,
                          columns,
                          verbose );
    }
    else {
        throw std::runtime_error( "Embed_pybind(): Invalid input.\n" );
    }

    DF       dfout = DataFrameToDF( embedded );
    py::dict D     = DFtoDict( dfout );
    
    return D;
}

//---------------------------------------------------------------
// 
//---------------------------------------------------------------
py::dict MakeBlock_pybind( DF                       dfin,
                           int                      E,
                           int                      tau,
                           std::vector<std::string> columnNames,
                           bool                     verbose ) {
    
    DataFrame< double > dataFrame = DFToDataFrame( dfin );

    DataFrame< double > block = MakeBlock( dataFrame,
                                           E,
                                           tau,
                                           columnNames,
                                           verbose );

    DF       df = DataFrameToDF( block );
    py::dict D  = DFtoDict( df );
    
    return D;
}
