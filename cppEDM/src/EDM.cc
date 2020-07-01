
#include "EDM.h"

// Declared in API.h
extern DataFrame< double > MakeBlock( DataFrame< double > &, int, int,
                                      std::vector< std::string > );

//----------------------------------------------------------------
// Constructors
//----------------------------------------------------------------
EDM::EDM ( DataFrame< double > & data,
           Parameters          & parameters ) :
    data( data ), anyTies( false ), parameters( parameters ) {}

//----------------------------------------------------------------
// Project : Implemented in sub-class
//----------------------------------------------------------------
void EDM::Project () {}

//----------------------------------------------------------------
// Set target (library) vector
//----------------------------------------------------------------
void EDM::GetTarget() {
    if ( parameters.targetIndex ) {
        target = data.Column( parameters.targetIndex );
    }
    else if ( parameters.targetName.size() ) {
        target = data.VectorColumnName( parameters.targetName );
    }
    else {
        // Default to first column
        target = data.Column( 0 );
    }
}

//----------------------------------------------------------------
// Implemented as a wrapper for API MakeBlock()
// Note: dataFrame must have the columnNameToIndex map
//
// NOTE: Truncates data by tau * (E-1) rows to remove
//       nan values (partial data rows)
// NOTE: The returned data block does NOT have the time column
//----------------------------------------------------------------
void EDM::EmbedData() {

    if ( not parameters.columnIndex.size() and
         data.ColumnNameToIndex().empty() ) {
        throw std::runtime_error("EDM::Embed(): columnNameIndex empty.\n");
    }

    // If columns provided, validate they are in dataFrameIn
    for ( auto colName : parameters.columnNames ) {
        auto ci = find( data.ColumnNames().begin(),
                        data.ColumnNames().end(), colName );

        if ( ci == data.ColumnNames().end() ) {
            std::stringstream errMsg;
            errMsg << "EDM::Embed(): Failed to find column "
                   << colName << " in dataFrame with columns: [ ";
            for ( auto col : data.ColumnNames() ) {
                errMsg << col << " ";
            } errMsg << " ]\n";
            throw std::runtime_error( errMsg.str() );    
        }
    }

    // Get column names for MakeBlock
    std::vector< std::string > colNames;
    if ( parameters.columnNames.size() ) {
        // column names are strings use as-is
        colNames = parameters.columnNames;
    }
    else if ( parameters.columnIndex.size() ) {
        // columns are indices : Create column names for MakeBlock
        for ( size_t i = 0; i < parameters.columnIndex.size(); i++ ) {
            std::stringstream ss;
            ss << "V" << parameters.columnIndex[i];
            colNames.push_back( ss.str() );
        }
    }
    else {
        throw std::runtime_error( "EDM::Embed(): columnNames and "
                                  " columnIndex are empty.\n" );
    }

    // Extract the specified columns (sub)DataFrame from dataFrameIn
    DataFrame< double > dataFrame;

    if ( parameters.columnNames.size() ) {
        dataFrame = data.DataFrameFromColumnNames( parameters.columnNames );
    }
    else if ( parameters.columnIndex.size() ) {
        // already have column indices
        // Note there will be no column names transferred
        dataFrame = data.DataFrameFromColumnIndex( parameters.columnIndex );
    }

    embedding = MakeBlock( std::ref( dataFrame ), parameters.E,
                           parameters.tau, colNames );
}
