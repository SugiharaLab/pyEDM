
#include "Embed.h"

// NOTE: The returned data block does NOT have the time column

//----------------------------------------------------------------
// API Overload 1: Explicit data file path/name
//   Implemented as a wrapper to API Overload 2:
//   which is a wrapper for MakeBlock()
//----------------------------------------------------------------
// Embed DataFrame columns in E dimensions.
// Data is read from path/dataFile
//
// side effects:        truncates the array by tau * (E-1) rows 
//                      to remove nan values (partial data rows)
// @param E:            embedding dimension
// @param tau:          time step delay
// @param columns:      column names or indices to embed
// @return:             DataFrame with embedded data block
//----------------------------------------------------------------
DataFrame< double > Embed ( std::string path,
                            std::string dataFile,
                            int         E,
                            int         tau,
                            std::string columns,
                            bool        verbose ) {
    
    DataFrame< double > toEmbed (path, dataFile);
    DataFrame< double > embedded = Embed (toEmbed, E, tau, columns, verbose); 
    return embedded;
}

//----------------------------------------------------------------
// API Overload 2: DataFrame provided
//   Implemented as a wrapper for MakeBlock()
// Note: dataFrame must have the columnNameToIndex map 
//----------------------------------------------------------------
DataFrame< double > Embed ( DataFrame< double > dataFrameIn,
                            int                 E,
                            int                 tau,
                            std::string         columns,
                            bool                verbose ) {
    
    // Parameter.Validate will convert columns into a vector of names
    // or a vector of column indices
    Parameters param = Parameters( Method::Embed, "", "", "", "",
                                   "1 1", "1 1", E, 0, 0, tau, 0,
                                   columns, "", false, verbose );

    if ( not param.columnIndex.size() and
         dataFrameIn.ColumnNameToIndex().empty() ) {
        throw std::runtime_error("Embed(DataFrame): columnNameIndex empty.\n");
    }

    // Get column names for MakeBlock
    std::vector< std::string > colNames;
    if ( param.columnNames.size() ) {
        // column names are strings use as-is
        colNames = param.columnNames;
    }
    else if ( param.columnIndex.size() ) {
        // columns are indices : Create column names for MakeBlock
        for ( size_t i = 0; i < param.columnIndex.size(); i++ ) {
            std::stringstream ss;
            ss << "V" << param.columnIndex[i];
            colNames.push_back( ss.str() );
        }
    }
    else {
        throw std::runtime_error( "Embed(DataFrame): columnNames and "
                                  " columnIndex are empty.\n" );
    }

    // Extract the specified columns (sub)DataFrame from dataFrameIn
    DataFrame< double > dataFrame;
    
    if ( param.columnNames.size() ) {
        // Get a vector of column indices 
        std::vector< size_t > col_i;
        for ( auto colName : param.columnNames ) {
            col_i.push_back( dataFrameIn.ColumnNameToIndex()[ colName ] );
        }
        dataFrame = dataFrameIn.DataFrameFromColumnIndex( col_i );
    }
    else if ( param.columnIndex.size() ) {
        // alread have column indices
        dataFrame = dataFrameIn.DataFrameFromColumnIndex( param.columnIndex );
    }

    DataFrame< double > embedding = MakeBlock( dataFrame, E, tau,
                                               colNames, verbose );
    return embedding;
}

//---------------------------------------------------------
// MakeBlock from dataFrame
//---------------------------------------------------------
DataFrame< double > MakeBlock ( DataFrame< double >      dataFrame,
                                int                      E,
                                int                      tau,
                                std::vector<std::string> columnNames,
                                bool                     verbose ) {

    if ( columnNames.size() != dataFrame.NColumns() ) {
        std::stringstream errMsg;
        errMsg << "MakeBlock: The number of columns in the dataFrame ("
               << dataFrame.NColumns() << ") is not equal to the number "
               << "of columns specified (" << columnNames.size() << ").\n";;
        throw std::runtime_error( errMsg.str() );
    }
    
    size_t NRows    = dataFrame.NRows();        // number of input rows
    size_t NColOut  = dataFrame.NColumns() * E; // number of output columns
    size_t NPartial = tau * (E-1);              // rows to shift & delete

    // temporary data frame to hold the embedded (shifted) data
    DataFrame< double > shiftDataFrame( NRows, NColOut );

    // Create embedded data frame column names X(t-0) X(t-1)...
    std::vector< std::string > newColumnNames( NColOut );
    size_t newCol_i = 0;
    for ( size_t col = 0; col < columnNames.size(); col ++ ) {
        for ( size_t e = 0; e < E; e++ ) {
            std::stringstream ss;
            ss << columnNames[ col ] << "(t-" << e << ")";
            newColumnNames[ newCol_i ] = ss.str();
            newCol_i++;
        }
    }

    // Ouput data frame with tau * E-1 fewer rows
    DataFrame< double > embedding( NRows - NPartial, NColOut, newColumnNames );

    // slice indices for each column of original & shifted data
    std::slice slice_i = std::slice (0, NRows, 1);

    // to keep track of where to insert column in new data frame
    size_t colCount = 0;

    // shift column data and write to temporary data frame
    for ( size_t col = 0; col < dataFrame.NColumns(); col++ ) {
        // for each embedding dimension
        for ( size_t e = 0; e < E; e++ ) {

            std::valarray< double > column = dataFrame.Column( col );
            
            std::valarray< double > tmp = column.shift( -e * tau )[slice_i];

            // These will be deleted, but set to NAN for completeness
            tmp[ std::slice( 0, e * tau, 1 ) ] = NAN;

            shiftDataFrame.WriteColumn( colCount, tmp );
            colCount++;
        }
    }

    // Delete rows with partial data
    slice_i = std::slice ( NPartial, NRows - NPartial, 1 );

    // Write shifted columns to the output embedding DataFrame
    for ( size_t i = 0; i < NColOut; i++ ) {
        std::valarray< double > tmp = shiftDataFrame.Column( i )[ slice_i ];
        embedding.WriteColumn( i, tmp );
    }
    
    return embedding;
}
