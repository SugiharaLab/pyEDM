
#include "EDM.h"

// Declared in API.h
extern DataFrame< double > MakeBlock( DataFrame< double > &, int, int,
                                      std::vector< std::string >, bool );

//----------------------------------------------------------------
// Constructors
//----------------------------------------------------------------
EDM::EDM ( DataFrame< double > & data,
           Parameters          & parameters ) :
    data( data ), anyTies( false ), embedShift( 0 ),
    parameters( parameters ) {}

//----------------------------------------------------------------
// Project : Implemented in sub-class
//----------------------------------------------------------------
void EDM::Project () {}

//----------------------------------------------------------------
// Generate : Implemented in sub-class
//----------------------------------------------------------------
void EDM::Generate () {}

//----------------------------------------------------------------
// Set target (library) vector
//----------------------------------------------------------------
void EDM::GetTarget() {
    if ( parameters.targetName.size() ) {
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
// NOTE: The returned data block does NOT have the time column
//----------------------------------------------------------------
void EDM::EmbedData() {

    if ( data.ColumnNameToIndex().empty() ) {
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
        // column names are strings
        colNames = parameters.columnNames;
    }
    else {
        throw std::runtime_error( "EDM::Embed(): columnNames are empty.\n" );
    }

    // Extract the specified columns (sub)DataFrame from dataFrameIn
    DataFrame< double > dataFrame =
        data.DataFrameFromColumnNames( parameters.columnNames );

    // deletePartial = false
    embedding = MakeBlock( std::ref( dataFrame ), parameters.E,
                           parameters.tau, colNames, false );
}
