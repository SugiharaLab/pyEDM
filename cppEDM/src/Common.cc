
#include <algorithm>
#include <cstring>

#include "Common.h"

//---------------------------------------------------------------
// Binary sort function for FindNeighbors() & CCMNeighbors()
//---------------------------------------------------------------
bool DistanceCompare( const std::pair<double, size_t> & x,
                      const std::pair<double, size_t> & y ) {
    return x.first < y.first;
}

//----------------------------------------------------------------
// 
//----------------------------------------------------------------
std::string ToLower( std::string str ) {

    std::string lowerStr( str );
    std::transform( lowerStr.begin(), lowerStr.end(),
                    lowerStr.begin(), ::tolower );

    return lowerStr;
}

//-------------------------------------------------------------------
// Called in two different contexts:
// DataFrame.h  ReadData() Are data columns numeric or labels?
// Parameter.cc Validate() Are columns/target integer index or label?
//
// param integer : true: only digits, false: digits plus '-', '.'
// return
//        true  : str has only numeric characters
//        false : str has non-numeric characters
//-------------------------------------------------------------------
bool OnlyDigits( std::string str, bool integer ) {

    if ( not str.size() ) {
        throw std::runtime_error( "OnlyDigits(): String is empty.\n" );
    }

    // Remove whitespace
    std::string str_( str );
    str_.erase( std::remove_if( str_.begin(), str_.end(), ::isspace ),
                str_.end() );

    std::string digits;
    if ( integer ) {
        digits = "0123456789";
    }
    else {
        digits = "-.0123456789";
    }

    // Is str_ purely numeric characters?
    bool onlyDigits = strspn(str_.c_str(), digits.c_str()) == str_.size();

    return onlyDigits;
}

//----------------------------------------------------------------
// SplitString
//
// Purpose: like Python string.split()
//
// Arguments: inString    : string to be split
//            delimeters  : string of delimeters
//
// Note:  A typical delimeter string: delimeters = " \t,\n;"
//           
// Return: vector of tokens
//----------------------------------------------------------------
std::vector<std::string> SplitString( std::string inString, 
                                      std::string delimeters ) {

  size_t pos       = 0;
  size_t eos       = 0;
  size_t wordStart = 0;
  size_t wordEnd   = 0;

  bool foundStart = false;
  bool foundEnd   = false;

  std::vector<std::string> splitString;

  std::string word;

  eos = inString.length();

  while ( pos <= eos ) {
    if ( not foundStart ) {
      if ( delimeters.find( inString[pos] ) == delimeters.npos ) {
	// this char (inString[pos]) is not a delimeter
	wordStart  = pos;
	foundStart = true;
	pos++;
	continue;
      }
    }
    if ( foundStart and not foundEnd ) {
      if ( delimeters.find( inString[pos] ) != delimeters.npos 
	   or pos == eos ) {
	// this char (inString[pos]) is a delimeter or
	// at the end of the string
	wordEnd  = pos;
	foundEnd = true;
      }
    }
    if ( foundStart and foundEnd ) {
      foundStart = false;
      foundEnd   = false;

      word = inString.substr( wordStart, wordEnd - wordStart );

      // remove whitespace
      word.erase( std::remove_if( word.begin(), word.end(), ::isspace ),
                  word.end() );

      splitString.push_back( word );
    }
    if ( pos == eos ) {
      break;
    }
    pos++;
  }

  return splitString;
}

//----------------------------------------------------------------
// 
//----------------------------------------------------------------
VectorError ComputeError( std::valarray< double > obsIn,
                          std::valarray< double > predIn ) {

    if ( obsIn.size() != predIn.size() ) {
        std::stringstream errMsg;
        errMsg << "ComputeError(): Observation size "
               << obsIn.size() << " is not equal to prediction size "
               << predIn.size();
        throw std::runtime_error( errMsg.str() );
    }

    // JP does find work on nan?  Since nan != nan, probably not...
    // Use a slice to extract the overlapping subset of obsIn, PredIn
    // We need to find the appropriate slice parameters

    // To try and be efficient, we first scan for nans, if none: stats
    // If there are nans, then assess the overlapping subset and slice
    bool nanObs  = false;
    bool nanPred = false;

    for ( auto o : obsIn  ) { if ( std::isnan( o ) ) { nanObs = true; break; } }
    for ( auto p : predIn ) { if ( std::isnan( p ) ) { nanPred= true; break; } }
    
    // vectors to hold data with no nans: reassigned below
    std::valarray< double > obs;
    std::valarray< double > pred;
    size_t                  Nin = obsIn.size();

    if ( not nanObs and not nanPred ) {
        obs  = std::valarray< double >( obsIn  );
        pred = std::valarray< double >( predIn );
    }
    else {
        // Handle nans...
        // Build concurrent vector of bool pairs : isnan on obsIn, predIn
        std::vector< std::pair< bool, bool > > nanIndexPairs( Nin );
        for ( size_t i = 0; i < Nin; i++ ) {
            nanIndexPairs[ i ] = std::make_pair( std::isnan( obsIn[i]  ),
                                                 std::isnan( predIn[i] ) );
        }
        // Find overlapping subset indices
        // Condense pairs into one boolean value in nonNanOverlap
        std::vector< bool > nonNanOverlap( Nin );
        for ( size_t i = 0; i < Nin; i++ ) {
            if ( not nanIndexPairs[ i ].first and
                 not nanIndexPairs[ i ].second ) {
                nonNanOverlap[ i ] = true; // Both are not nan, valid index
            }
            else {
                nonNanOverlap[ i ] = false;
            }
        }
        // Get the slice parameters
        auto firstValid = std::find( nonNanOverlap.begin(),
                                     nonNanOverlap.end(), true );
        auto lastValid  = std::find( nonNanOverlap.rbegin(),
                                     nonNanOverlap.rend(), true );
        size_t firstValidIndex = std::distance( nonNanOverlap.begin(),
                                                firstValid );
        size_t lastValidIndex = std::distance( nonNanOverlap.rbegin(),
                                               lastValid );

        lastValidIndex = Nin - lastValidIndex; // reverse iterator used...

        int Nout = (int) lastValidIndex - (int) firstValidIndex;
        
        if ( Nout < 0 ) {
            std::stringstream msg;
            msg << "WARNING: ComputeError(): nan predictions found"
                << " error not computed." << std::endl;
            std::cout << msg.str();

            Nout = 0;
            obs  = std::valarray< double >( 0., 1 ); // vector [0.] N = 1
            pred = std::valarray< double >( 0., 1 ); // vector [0.] N = 1
        }
        else {
            // Allocate the output arrays and fill with slices
            obs  = std::valarray< double >( Nout );
            pred = std::valarray< double >( Nout );
            
            std::slice nonNan = std::slice( firstValidIndex, Nout, 1 );
            obs [ std::slice( 0, Nout, 1 ) ] = obsIn [ nonNan ];
            pred[ std::slice( 0, Nout, 1 ) ] = predIn[ nonNan ];
        }
    }

    size_t N = std::max( 1, (int) pred.size() );
    std::valarray< double > two( 2, N ); // Vector of 2's for squaring

    double sumPred    = pred.sum();
    double sumObs     = obs.sum();
    double meanPred   = sumPred / N;
    double meanObs    = sumObs  / N;
    double sumSqrPred = pow( pred, two ).sum();
    double sumSqrObs  = pow( obs,  two ).sum();
    double sumErr     = abs( obs - pred ).sum();
    double sumSqrErr  = pow( obs - pred, two ).sum();
    double sumProd    = ( obs * pred ).sum();

    double rho; // Pearson correlation coefficient

    double denom = ( std::sqrt( ( sumSqrObs  - N * pow( meanObs,  2 ) ) ) *
                     std::sqrt( ( sumSqrPred - N * pow( meanPred, 2 ) ) ) );

    if ( denom == 0 or std::isnan( denom ) ) {
        rho = 0;
    }
    else {
        rho = ( sumProd - N * meanObs * meanPred ) / denom;
    }

    VectorError vectorError = VectorError();

    vectorError.RMSE = sqrt( sumSqrErr / N );
    vectorError.MAE  = sumErr / N;
    vectorError.rho  = rho;

    return vectorError;
}
