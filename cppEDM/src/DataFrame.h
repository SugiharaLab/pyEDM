#ifndef DATAFRAME_H
#define DATAFRAME_H

// NOTE: This header deviates from the desired class implementation
// where *.h provides declarations, *.cc methods.  This is solely to
// accomodate the OSX XCode environment which seems unable to deal
// with c++11 standard template implemenations.
// A possible solution is to link against libc++ on OSX.
// See ../etc/Notes, ../etc/libstdc++_Notes.txt.

#include <iomanip>
#include <fstream>
#include <iterator>

#include "Common.h"

// Since #include DataFrame.h is in Common.h, need forward declaration
extern std::vector<std::string> SplitString( std::string inString, 
                                             std::string delimeters = "," );
extern bool OnlyDigits( std::string str, bool integerOnly );

// Type definition for CSV NamedData to pair column names & column data
typedef std::vector<std::pair<std::string, std::vector<double>>> NamedData;

// Container for parsed data file returned by ReadData()
struct ParsedData {
    std::vector< std::string > time;
    std::string                timeName;
    NamedData                  namedData;
};

//----------------------------------------------------------------
// DataFrame class
// Data container is a single, contiguous valarray: elements.
// NOTE: elements are Row Major format, ala C, C++, numpy
// DataFrame element access is through the () operator: (row,col).
// The time column is not processed as data, but as strings.
//----------------------------------------------------------------
template <class T>
class DataFrame {
    
    std::valarray<T> elements;
    size_t           n_columns;
    size_t           n_rows;
    
    std::vector< std::string >      columnNames;
    std::map< std::string, size_t > columnNameToIndex;

    std::vector< std::string > time;
    std::string                timeName;
    
    bool   noTime;
    bool   partialDataRowsDeleted;
    size_t maxRowPrint;
    
public:
    //-----------------------------------------------------------------
    // Destructor
    //-----------------------------------------------------------------
    ~DataFrame() {}
    
    //-----------------------------------------------------------------
    // Constructors
    //-----------------------------------------------------------------
    DataFrame() {}
    
    //-----------------------------------------------------------------
    // Load data from CSV file path/fileName, populate DataFrame
    //-----------------------------------------------------------------
    DataFrame( std::string path, std::string fileName, bool noTime = false ):
        maxRowPrint( 10 ), noTime( noTime ), partialDataRowsDeleted( false )
    {
        ParsedData parsedData = ReadData( path, fileName );
        SetupDataFrame( parsedData ); // Process parsedData into a DataFrame
    }
    
    //-----------------------------------------------------------------
    // Empty DataFrame of size (row, columns), no column names
    //-----------------------------------------------------------------
    DataFrame( size_t rows, size_t columns ):
        n_rows( rows ), n_columns( columns ), elements( columns * rows ),
        maxRowPrint( 10 ), partialDataRowsDeleted( false ) {}
    
    //-----------------------------------------------------------------
    // Empty DataFrame of size (rows, columns) with column names in a
    // single whitespace delimited string. 
    //-----------------------------------------------------------------
    DataFrame( size_t rows, size_t columns, std::string colNames ):
        n_rows( rows ), n_columns( columns ), elements( columns * rows ),
        columnNames( std::vector<std::string>(columns) ), maxRowPrint( 10 ),
        partialDataRowsDeleted( false )
    {
        BuildColumnNameIndex( colNames );
    }
    
    //-----------------------------------------------------------------
    // Empty DataFrame of size (rows, columns) with column names in a
    // string vector.
    //-----------------------------------------------------------------
    DataFrame( size_t rows, size_t columns,
               std::vector< std::string > columnNames ):
        n_rows( rows ), n_columns( columns ), elements( columns * rows ),
        columnNames( columnNames ), maxRowPrint( 10 ),
        partialDataRowsDeleted( false ) 
    {
        BuildColumnNameIndex();
    }
   
    //-----------------------------------------------------------------
    // Fortran style element access operators M(row,col)
    //-----------------------------------------------------------------
    T &operator()( size_t row, size_t column ) {
        return elements[ row * n_columns + column ];
    }
    T operator()( size_t row, size_t column ) const {
        return elements[ row * n_columns + column ];
    }

    //-----------------------------------------------------------------
    // Member Accessors
    //-----------------------------------------------------------------
    size_t NColumns() const { return n_columns;          }
    size_t NRows()    const { return n_rows;             }
    size_t size()     const { return n_rows * n_columns; }
    
    std::valarray<T>  Elements() const { return elements; }
    std::valarray<T> &Elements()       { return elements; }
    
    std::vector< std::string >  Time() const { return time; }
    std::vector< std::string > &Time()       { return time; }
    
    std::string  TimeName() const { return timeName; }
    std::string &TimeName()       { return timeName; }
    
    std::vector< std::string >  ColumnNames() const { return columnNames; }
    std::vector< std::string > &ColumnNames()       { return columnNames; }
    
    std::map< std::string, size_t > ColumnNameToIndex() const {
        return columnNameToIndex;
    }
    std::map< std::string, size_t > &ColumnNameToIndex() {
        return columnNameToIndex;
    }

    size_t  MaxRowPrint() const { return maxRowPrint; }
    size_t &MaxRowPrint()       { return maxRowPrint; }
    
    bool  PartialDataRowsDeleted() const { return partialDataRowsDeleted; }
    bool &PartialDataRowsDeleted()       { return partialDataRowsDeleted; }
    
    //-----------------------------------------------------------------
    // Return column from index col
    //-----------------------------------------------------------------
    std::valarray<T> Column( size_t col ) const {
        // slice( size_t start, size_t length, size_t stride )
        return elements[ std::slice( col, n_rows, n_columns ) ];
    }

    //-----------------------------------------------------------------
    // Return row from index row
    //-----------------------------------------------------------------
    std::valarray<T> Row( size_t row ) const {
        // slice( size_t start, size_t length, size_t stride )
        return elements[ std::slice( row * n_columns, n_columns, 1 ) ];
    }

    //------------------------------------------------------------------
    // Return data column selected by column name
    //------------------------------------------------------------------
    std::valarray< double > VectorColumnName( std::string column ) {
        
        std::vector< std::string >::iterator ci = std::find(columnNames.begin(),
                                                            columnNames.end(),
                                                            column );
        if ( ci == columnNames.end() ) {
            std::stringstream errMsg;
            errMsg << "DataFrame::VectorColumnName() Failed to find column: "
                   << column;
            errMsg << " in DataFrame columns:\n[ ";
            for ( auto cni  = columnNames.begin();
                       cni != columnNames.end(); ++cni ) {
                errMsg << *cni << " ";
            } errMsg << "]" << std::endl;
            throw std::runtime_error( errMsg.str() );
        }
        
        size_t col_i = std::distance( columnNames.begin(), ci );
        
        std::valarray<double> vec = Column( col_i );
        
        return vec;
    }

    //-----------------------------------------------------------------
    // Return (sub)DataFrame of specified column indices
    //-----------------------------------------------------------------
    DataFrame<double> DataFrameFromColumnIndex( std::vector<size_t> column_i ) {
        
        DataFrame<double> M = DataFrame( n_rows, column_i.size() );

        size_t col_j = 0;
        
        for ( size_t i = 0; i < column_i.size(); i++ ) {
            size_t col_i = column_i[ i ];

            if ( col_i >= n_columns ) {
                std::stringstream errMsg;
                errMsg << "DataFrame::DataFrameFromColumnIndex(): "
                       << "A column index ("
                       << col_i << ") exceeds the data frame domain.\n";
                throw std::runtime_error( errMsg.str() );
            }
            
            std::valarray<double> column_vec_i = Column( col_i );

            M.WriteColumn( col_j, column_vec_i );
            col_j++;
        }
        
        // Add time vector if present
        if ( time.size() ) {
            M.Time()     = time;
            M.TimeName() = timeName;
        }
        // Add columnNames if present
        if ( columnNames.size() ) {
            std::vector < std::string > colNames;
            for ( auto col_i : column_i ) {
                colNames.push_back( columnNames.at( col_i ) );
            }
            M.ColumnNames() = colNames;
            M.BuildColumnNameIndex();
        }
        
        return M;
    }

    //------------------------------------------------------------------
    // Return (sub)DataFrame selected by columnNames
    // columnNames converted to column indices for DataFrameFromColumnIndex()
    //------------------------------------------------------------------
    DataFrame< double > DataFrameFromColumnNames(
        std::vector<std::string> colNames ) {

        // vector of column indices for DataFrameFromColumnIndex()
        std::vector<size_t> col_i_vec;
        
        // Map column names to indices
        std::vector<std::string>::iterator si;
        for ( auto ci = colNames.begin(); ci != colNames.end(); ++ci ) {
            auto si = find( columnNames.begin(), columnNames.end(), *ci );
            
            if ( si != columnNames.end() ) {
                col_i_vec.push_back( std::distance( columnNames.begin(), si ) );
            }
        }
        
        // Validation
        if ( col_i_vec.size() != colNames.size() ) {
            std::stringstream errMsg;
            errMsg << "DataFrame::DataFrameFromColumnNames() "
                      "Failed to find columns:\n[ ";
            for ( auto ci = colNames.begin(); ci != colNames.end(); ++ci ){
                errMsg << *ci << " ";
            } errMsg << "]" << std::endl;
            errMsg << "in DataFrame columns:\n[ ";
            for (auto ci = columnNames.begin(); ci != columnNames.end(); ++ci){
                errMsg << *ci << " ";
            } errMsg << "]" << std::endl;
            throw std::runtime_error( errMsg.str() );
        }
        
        DataFrame<double> M_col = DataFrameFromColumnIndex( col_i_vec );
        
        // Insert columnNames if not already present
        if ( not M_col.ColumnNames().size() ) {
            M_col.ColumnNames() = colNames;
            M_col.BuildColumnNameIndex();
        }
        
        return M_col;
    }

    //-----------------------------------------------------------------
    // Return Elements in Column Major order (Fortran)
    //-----------------------------------------------------------------
    std::valarray<T> ColumnMajorData() const {

        std::valarray<T> colMajorElements( elements.size() );
        
        for ( size_t col = 0; col < n_columns; col++ ) {
            // slice( size_t start, size_t length, size_t stride )
            colMajorElements[ std::slice( col * n_rows, n_rows, 1 ) ] =
                Column( col );
        }

        return colMajorElements;
    }

    //-----------------------------------------------------------------
    // Write array to row
    //-----------------------------------------------------------------
    void WriteRow( size_t row, std::valarray<T> array ) {
        size_t N = array.size();
    
        if ( N != n_columns ) {
            std::stringstream errMsg;
            errMsg << "DataFrame::WriteRow(): array must have "
                   << n_columns << " elements. " << N << " were provided.\n";
            throw std::runtime_error( errMsg.str() );
        }
        if ( row >= n_rows ) {
            std::stringstream errMsg;
            errMsg << "DataFrame::WriteRow(): row argument must be less than "
                   << n_rows << ". " << row << " was provided.\n";
            throw std::runtime_error( errMsg.str() );
        }
        for ( size_t i = 0; i < N; i++ ) {
            (*this)( row, i ) = array[ i ];
        }
    }

    //-----------------------------------------------------------------
    // Write array to col
    //-----------------------------------------------------------------
    void WriteColumn( size_t col, std::valarray<T> array ) {
        size_t N = array.size();
    
        if ( N != n_rows ) {
            std::stringstream errMsg;
            errMsg << "DataFrame::WriteColumn(): array must have "
                   << n_rows << " elements. " << N << " were provided.\n";
            throw std::runtime_error( errMsg.str() );
        }
        if ( col >= n_columns ) {
            std::stringstream errMsg;
            errMsg<<"DataFrame::WriteColumn(): col argument must be less than "
                   << n_columns << ". " << col << " was provided.\n";
            throw std::runtime_error( errMsg.str() );
        }
        for ( size_t i = 0; i < N; i++ ) {
            (*this)( i, col ) = array[ i ];
        }
    }

    //-----------------------------------------------------------------
    // Delete nrows from the top
    // Requires that rows are contiguous: [ 0 : nrows ]
    //-----------------------------------------------------------------
    void DeletePartialDataRows( size_t nrows ) {

        // NOTE : Not thread safe

        if ( partialDataRowsDeleted ) {
            std::cout << "DeletePartialDataRows(): Partial data rows have "
                         "already been deleted." << std::endl;
            return;
        }
        
        partialDataRowsDeleted = true;

        if ( nrows > n_rows ) {
            std::stringstream errMsg;
            errMsg << "DataFrame::DeleteContiguousRows() "
                   << " nrows (" << nrows << " larger than DataFrame "
                   << "NRows (" << n_rows << ")" << std::endl;
            throw( std::runtime_error( errMsg.str() ) );
        }
        
        // Update n_rows
        n_rows = n_rows - nrows;

        // Update time
        time.erase( time.begin(), time.begin() + nrows );

        // Copy elements into data
        std::valarray< double > data( elements );
        
        // Resize elements
        size_t n_elements = elements.size() - nrows * n_columns;
        elements.resize( n_elements );

        // Copy non deleted data to resized elements. NOTE: Row major format
        std::slice elements_i( nrows * n_columns, n_elements, 1 );
        // Bogus cast for MSVC 
        elements[ std::slice( 0, n_elements, 1 ) ] =
            ( std::valarray< double > ) data[ elements_i ];
    }
    
    //-----------------------------------------------------------------
    // Build Column Name Index( std::string colNames )
    //-----------------------------------------------------------------
    void BuildColumnNameIndex( std::string colNames ) {
        // If colNames provided populate columnNames, columnNameToIndex
        if ( colNames.size() ) {
            columnNames = SplitString( colNames, " ,\t" );
            if ( columnNames.size() != n_columns ) {
                std::stringstream errMsg;
                errMsg << "DataFrame::BuildColumnNameIndex(s) "
                       << "Number of column names ("
                       << columnNames.size() << ") does not match the"
                       << " number of columns (" << n_columns << ").\n";
                throw std::runtime_error( errMsg.str() );
            }
        }
        for ( size_t i = 0; i < columnNames.size(); i++ ) {
            columnNameToIndex[ columnNames[i] ] = i;
        }
    }

    //-----------------------------------------------------------------
    // Build Column Name Index
    //-----------------------------------------------------------------
    void BuildColumnNameIndex() {
        // If columnNames exist, populate columnNameToIndex
        if ( columnNames.size() ) {
            if ( columnNames.size() != n_columns ) {
                std::stringstream errMsg;
                errMsg << "DataFrame::BuildColumnNameIndex() Number of column "
                       << " names (" << columnNames.size() 
                       << ") does not match the number of columns "
                       << "(" << n_columns << ").\n";
                throw std::runtime_error( errMsg.str() );
            }
        }
        for ( size_t i = 0; i < columnNames.size(); i++ ) {
            columnNameToIndex[ columnNames[i] ] = i;
        }
    }
    
    //------------------------------------------------------------------
    // Stream DataFrame to ostream
    //------------------------------------------------------------------
    friend std::ostream& operator <<( std::ostream& os, const DataFrame& D ) {
        // precision should be a parameter
        os.precision( 4 );
        os.fill( ' ' );
        os.setf( std::ios::fixed, std::ios::floatfield );
        
        os << "DataFrame: -----------------------------------\n";
        os << D.NRows() << " rows, " << D.NColumns() << " columns.\n";
        os << "---------------- First " << D.MaxRowPrint()
           << " rows ---------------\n";
        
        // print names of columns
        if ( D.timeName.size() ) {
            os << std::setw(10) << D.timeName;
        }
        
        for ( size_t i = 0; i < D.ColumnNames().size(); i++ ) {
            os << std::setw(13) << D.ColumnNames()[i];
        } os << std::endl;
        
        os << "----------------------------------------------\n";
 
        // print vec data up to maxRowPrint points
        for ( size_t row = 0; row < D.NRows() and
                              row < D.MaxRowPrint(); row++ ) {

            // row time
            if ( D.time.size() ) {
                os << std::setw(10) << D.time[ row ];
            }
            
            // print data points from each col
            for ( size_t col = 0; col < D.NColumns(); col++ ) {
                os << std::setw(13) << D( row, col );
            }
            os << std::endl;
        }
        os << "----------------------------------------------" << std::endl;

        return os;
    }

    //------------------------------------------------------------------
    //  Write contents to file
    //------------------------------------------------------------------
    void WriteData( std::string outputFilePath, std::string outputFileName ) {
        
        // Vector of strings to hold image of DataFrame for file output
        std::vector< std::string > fileLines;

        // Stream to ingest each line (row) of the DataFrame
        std::stringstream lineStr;

        // Set stream precision. This should be a parameter.
        lineStr.precision( 4 );
        lineStr.setf( std::ios::fixed, std::ios::floatfield );

        // Create column name header if needed
        if ( ColumnNames().size() == 0 ) {
            std::cout << "DataFrame::WriteData(): This data frame has no column"
                      << "names.  Column names will be created.\n";
            for ( size_t i = 0; i < n_columns; i++ ) {
                std::stringstream colname;
                colname << "V" << i;
                ColumnNames().push_back( colname.str() );
            }
        }
        if ( ColumnNames().size() != n_columns ) {
            throw std::runtime_error( "DataFrame::WriteData(): The number of "
                                      " column names does not match the number "
                                      " of data columns.\n" );
        }

        if ( TimeName().size() ) {
            lineStr << TimeName() << ",";
        }
        
        // Push column name from each column into the string stream
        for ( size_t colIdx = 0; colIdx < n_columns; colIdx++ ) {
            lineStr << ColumnNames()[ colIdx ];

            if ( colIdx != n_columns - 1 ) {
                lineStr << ",";
            }
        }

        // Push column name header into fileLines container
        fileLines.push_back( lineStr.str() );

        // Reset the lineStr stringstream content
        lineStr.str( std::string() );

        // Iterate through rows of numerical data
        for ( size_t rowIdx = 0; rowIdx < n_rows; rowIdx++ ) {

            if ( time.size() ) {
                lineStr << time[ rowIdx ] << ",";
            }
            
            for ( size_t colIdx = 0; colIdx < n_columns; colIdx++ ) {

                lineStr << (*this) ( rowIdx, colIdx );

                if ( colIdx != n_columns - 1 ) {
                    lineStr << ",";
                }
            }

            // Push row data into fileLines container
            fileLines.push_back( lineStr.str() );
            lineStr.str( std::string() );
        }

        // Write contents to file
        std::ofstream outputFile( outputFilePath + outputFileName );
    
        if ( outputFile.is_open() ) {
            
            std::copy( fileLines.begin(), fileLines.end(),
                       std::ostream_iterator<std::string>(outputFile,"\n") );

            outputFile.close();
        }
        else {
            // Error in state of open() on outputFile
            std::stringstream errMsg;
            errMsg << "DataFrame::WriteData(): Failed to open file: "
                << outputFilePath + outputFileName << ". \n";
            throw std::runtime_error( errMsg.str() );
        }
    }

private:
    //------------------------------------------------------------------
    // Process parsedData from ReadData() to populate DataFrame
    //------------------------------------------------------------------
    void SetupDataFrame( ParsedData parsedData ) {

        NamedData namedData = parsedData.namedData;

        // Setup column names in same order as dataFrame
        std::vector< std::string > colNames;
        for ( NamedData::iterator iterate = namedData.begin(); 
              iterate != namedData.end(); iterate++ ) {
            colNames.push_back( iterate->first );
        }
        
        // Initialize DataFrame members and storage
        n_rows      = namedData.begin()->second.size();
        n_columns   = namedData.size();
        elements    = std::valarray <T>( n_rows * n_columns );
        columnNames = colNames;
        time        = parsedData.time;
        timeName    = parsedData.timeName;

        BuildColumnNameIndex();

        // Transfer each data value into valarray 
        //    Another option is to use the writeColumn() method
        // NamedData is : pair< string, vector<double> >
        for ( NamedData::iterator iterate  = namedData.begin(); 
                                  iterate != namedData.end(); iterate++ ) {
            
            size_t colIdx = std::distance( namedData.begin(), iterate );

            for ( size_t rowIdx = 0; rowIdx < n_rows; rowIdx++ ) {
                (*this)( rowIdx, colIdx ) = iterate->second[ rowIdx ];
            }
        }
    }
    
    //------------------------------------------------------------------
    // Read disk file. Parse into a NamedData container and time vector.
    //------------------------------------------------------------------
    ParsedData ReadData( std::string path, std::string fileName ) {
        
        // Create input file stream and open file for input
        std::ifstream dataStrm( path + fileName );
        
        // Ensure file access is good before reading
        if ( not dataStrm.is_open() ) {
            std::stringstream errMsg;
            errMsg << "ERROR: DataFrame::ReadData() file " << path + fileName
                   << " is not open for reading." << std::endl;
            throw std::runtime_error( errMsg.str() );
        }
        if ( not dataStrm.good() ) {
            std::stringstream errMsg;
            errMsg << "ERROR: DataFrame::ReadData() file " << path + fileName
                   << " is not ready for reading." << std::endl;
            throw std::runtime_error( errMsg.str() );
        }
        
        // Read into a vector of strings, one line per string
        std::vector< std::string > dataLines;
        std::string tmp;
        
        while( getline( dataStrm, tmp ) ) {
            dataLines.push_back( tmp );
        }
        dataStrm.close();
        
#ifdef DEBUG_ALL
        std::cout << "------- ReadData() Contents of file "
                  << fileName << " -------" << std::endl;
        for( std::vector< std::string >::iterator ci = dataLines.begin();
             ci != dataLines.end(); ++ci ) {
            std::cout << *ci << std::endl;
        }
#endif

        // Vector of times
        std::vector< std::string > time;
        std::string                timeName;
        
        // Container of data name : vector pairs
        NamedData namedData; // vector< pair< string, vector< double >> >;
        
        // Container of column names in the same order as in csv file
        std::vector< std::string > colNames;

        // Check first line to see if it's only numeric digits, or a header
        bool onlyDigits = true;
        std::vector<std::string> firstLineWords = SplitString( dataLines[0] );
        
        for ( auto si =  firstLineWords.begin();
                   si != firstLineWords.end(); ++si ){

            onlyDigits = OnlyDigits( *si, false );
            
            if ( not onlyDigits ) { break; }
        }
        if ( onlyDigits ) {
            // Create named columns with generic col names: V1, V2...
            for (size_t colIdx = 0; colIdx < firstLineWords.size(); colIdx++){
                colNames.push_back( "V" + std::to_string(colIdx) );
            }
        }
        else {
            // Get named columns from header line
            for (size_t colIdx = 0; colIdx < firstLineWords.size(); colIdx++){
                colNames.push_back( firstLineWords[ colIdx ] );  
            }
            // Remove header line from read in lines so only numerical after
            dataLines.erase( dataLines.begin() );
        }

        if ( not noTime ) {
            timeName = colNames[ 0 ];
        }
        
        // Setup each col in namedData with new vec to insert numerical data
        // If noTime true then first column is data
        size_t startCol_i = noTime ? 0 : 1;
        for ( size_t colIdx = startCol_i; colIdx < colNames.size(); colIdx++ ) {
            NamedData::value_type colPair( colNames[ colIdx ],
                                           std::vector<double>() );
            namedData.push_back( colPair );
        }
        
        // Process each line in dataLines to fill in data vectors and time
        for ( size_t lineIdx = 0; lineIdx < dataLines.size(); lineIdx++ ) {
            
            std::vector<std::string> words = SplitString(dataLines[ lineIdx ]);

            if ( words.size() != colNames.size() ) {
                std::stringstream errMsg;
                errMsg << "ERROR: DataFrame::ReadData() Line " << lineIdx + 1
                       << " of file " << path + fileName
                       << " does not have " << colNames.size()
                       << " columns." << std::endl;
                throw std::runtime_error( errMsg.str() );
            }

            // Get time string. Required to be in first column.
            if ( not noTime ) {
                time.push_back( words[ 0 ] );
            }
            
            try {
                // Convert data columns to double, add to namedData
                for ( size_t colIdx = startCol_i;
                             colIdx < colNames.size(); colIdx++ ) {
                    namedData[ colIdx - startCol_i ].second.push_back(
                        std::stod( words[ colIdx ] ) );
                }
            }
            catch ( const std::exception& e ) {
                std::cout << "DataFrame ReadData() Exception:\n";
                std::cout << e.what() << std::endl;
                throw std::runtime_error( e.what() );
            }
        }

#ifdef DEBUG_ALL
        std::cout << "------- ReadData() data from "
                  << fileName << " -------" << std::endl;

        // Header
        if ( not noTime ) {
            std::cout << timeName << " ";
        }
        for ( auto ci = namedData.begin(); ci != namedData.end(); ++ci ) {
            std::cout << ci->first << " ";
        } std::cout << std::endl;

        for (size_t row = 0; row < (namedData.begin()->second).size(); row++){
            if ( not noTime ) {
                std::cout << time[ row ] << " ";
            }
            for ( auto ci = namedData.begin(); ci != namedData.end(); ++ci ) {
                std::vector< double > vec = ci->second;
                std::cout << vec[ row ] << " ";
            } std::cout << std::endl;
        }
#endif

        ParsedData parsedData;
        parsedData.time      = time;
        parsedData.timeName  = timeName;
        parsedData.namedData = namedData;
        
        return parsedData;
    }
};
#endif
