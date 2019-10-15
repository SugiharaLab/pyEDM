
#include "Embed.h"

// NOTE: The returned data block does NOT have the time column

//----------------------------------------------------------------
// API Overload 1: Explicit data file path/name
// Embed DataFrame columns in E dimensions.
// Data is read from path/dataFile
// Implemented as a wrapper to API Overload 2:
// which is a wrapper for MakeBlock()
//
// NOTE: Truncates data by tau * (E-1) rows to remove
//       nan values (partial data rows)
//----------------------------------------------------------------
DataFrame< double > Embed( std::string path,
                           std::string dataFile,
                           int         E,         // embedding dimension
                           int         tau,       // time step delay
                           std::string columns,   // column names or indices
                           bool        verbose ) {
    
    DataFrame< double > dataFrame( path, dataFile );
    DataFrame< double > embedded = Embed(dataFrame, E, tau, columns, verbose); 
    return embedded;
}

//----------------------------------------------------------------
// API Overload 2: DataFrame provided
// Implemented as a wrapper for MakeBlock()
// Note: dataFrame must have the columnNameToIndex map 
//----------------------------------------------------------------
DataFrame< double > Embed( DataFrame< double > dataFrameIn,
                           int                 E,
                           int                 tau,
                           std::string         columns,
                           bool                verbose ) {
    
    // Parameter.Validate will convert columns into a vector of names
    // or a vector of column indices
    Parameters param = Parameters( Method::Embed, "", "", "", "",
                                   "1 1", "1 1", E, 0, 0, tau, 0, 0,
                                   columns, "", false, false, verbose );

    if ( not param.columnIndex.size() and
         dataFrameIn.ColumnNameToIndex().empty() ) {
        throw std::runtime_error("Embed(DataFrame): columnNameIndex empty.\n");
    }

    // If columns provided, validate they are in dataFrameIn
    for ( auto colName : param.columnNames ) {
        auto ci = find( dataFrameIn.ColumnNames().begin(),
                        dataFrameIn.ColumnNames().end(), colName );
        
        if ( ci == dataFrameIn.ColumnNames().end() ) {
            std::stringstream errMsg;
            errMsg << "Embed(DataFrame): Failed to find column "
                   << colName << " in dataFrame with columns: [ ";
            for ( auto col : dataFrameIn.ColumnNames() ) {
                errMsg << col << " ";
            } errMsg << " ]\n";
            throw std::runtime_error( errMsg.str() );    
        }
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
        dataFrame = dataFrameIn.DataFrameFromColumnNames( param.columnNames );
    }
    else if ( param.columnIndex.size() ) {
        // already have column indices
        // Note there will be no column names transferred
        dataFrame = dataFrameIn.DataFrameFromColumnIndex( param.columnIndex );
    }
        
    DataFrame< double > embedding = MakeBlock( dataFrame, E, tau,
                                               colNames, verbose );
        
    return embedding;
}

//--------------------------------------------------------------
// MakeBlock from dataFrame
// Ignores the first tau * (E-1) dataFrame rows of partial data.
// Does not validate parameters or columns, use Embed()
//--------------------------------------------------------------
DataFrame< double > MakeBlock( DataFrame< double >      dataFrame,
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

    // To keep track of where to insert column in new data frame
    size_t colCount = 0;

    // Slice to ignore rows with partial data
    std::slice slice_i = std::slice( NPartial, NRows - NPartial, 1 );
    
    // Shift column data and write to embedding data frame
    for ( size_t col = 0; col < dataFrame.NColumns(); col++ ) {
        // for each embedding dimension
        for ( size_t e = 0; e < E; e++ ) {

            std::valarray< double > column = dataFrame.Column( col );
            
            std::valarray< double > tmp = column.shift( -e * tau );

            // Write shifted columns to the output embedding DataFrame
            embedding.WriteColumn( colCount, tmp[ slice_i ] );
            
            colCount++;
        }
    }

#ifdef ADD_EMBEDDING_TIME  // JP Is this needed now that Time is separate?
    // Add time vector with partial rows removed if present
    if ( dataFrame.Time().size() ) {
        embedding.Time() =
            std::vector< std::string >( dataFrame.Time().size() - NPartial );
        
        for ( size_t t = NPartial; t < dataFrame.Time().size(); t++ ) {
            embedding.Time()[ t - NPartial ] = dataFrame.Time()[ t ];
        }
        embedding.TimeName() = dataFrame.TimeName();
    }
#endif
    
    return embedding;
}
