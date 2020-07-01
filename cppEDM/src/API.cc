
//----------------------------------------------------------------
// Functions implemented here:
//     Embed(), MakeBlock(), Simplex(), SMap(), CCM(), Multiview()
//
// Functions implemented in Eval.cc:
//     EmbedDimension(), PredictInterval(), PredictNonlinear()
//----------------------------------------------------------------

#include "API.h"

//----------------------------------------------------------------
// Embed from file path/file input
//----------------------------------------------------------------
DataFrame< double > Embed( std::string path,
                           std::string dataFile,
                           int         E,         // embedding dimension
                           int         tau,       // time step offset
                           std::string columns,   // column names or indices
                           bool        verbose ) {

    DataFrame< double > dataFrame( path, dataFile );
    DataFrame< double > embedded = Embed( std::ref( dataFrame ),
                                          E, tau, columns, verbose ); 
    return embedded;
}

//----------------------------------------------------------------
// Embed from DataFrame input
//----------------------------------------------------------------
DataFrame< double > Embed( DataFrame< double > & dataFrameIn,
                           int                 E,
                           int                 tau,
                           std::string         columns,
                           bool                verbose ) {

    // Parameter.Validate will convert columns into a vector of names
    // or a vector of column indices
    Parameters parameters = Parameters( Method::Embed, "", "", "", "",
                                        "1 1", "1 1", E, 0, 0, tau, 0, 0,
                                        columns, "", false, false, verbose );
    // Instantiate EDM object
    EDM EDM_Embed = EDM( dataFrameIn, std::ref( parameters ) );

    // Perform embedding : calls MakeBlock() API function
    EDM_Embed.EmbedData();

    return EDM_Embed.embedding;
}

//------------------------------------------------------------------------
// MakeBlock from dataFrame :: API function
// Ignores the first (or last) tau * (E-1) dataFrame rows of partial data.
// Does not validate parameters or columns, use EmbedData()
//------------------------------------------------------------------------
DataFrame< double > MakeBlock( DataFrame< double >      & dataFrame,
                               int                      E,
                               int                      tau,
                               std::vector<std::string> columnNames )
{
    if ( columnNames.size() != dataFrame.NColumns() ) {
        std::stringstream errMsg;
        errMsg << "MakeBlock: The number of columns in the dataFrame ("
               << dataFrame.NColumns() << ") is not equal to the number "
               << "of columns specified (" << columnNames.size() << ").\n";;
        throw std::runtime_error( errMsg.str() );
    }

    if ( E < 1 ) {
        std::stringstream errMsg;
        errMsg << "MakeBlock(): E = " << E << " is invalid.\n" ;
        throw std::runtime_error( errMsg.str() );
    }

    size_t NRows    = dataFrame.NRows();        // number of input rows
    size_t NColOut  = dataFrame.NColumns() * E; // number of output columns
    size_t NPartial = abs( tau ) * (E-1);       // rows to shift & delete

    // Create embedded data frame column names X(t-0) X(t-1)...
    std::vector< std::string > newColumnNames( NColOut );
    size_t newCol_i = 0;
    for ( size_t col = 0; col < columnNames.size(); col ++ ) {
        for ( int e = 0; e < E; e++ ) {
            std::stringstream ss;
            if ( tau < 0 ) {
                ss << columnNames[ col ] << "(t-" << e << ")";
            }
            else {
                ss << columnNames[ col ] << "(t+" << e << ")";
            }
            newColumnNames[ newCol_i ] = ss.str();
            newCol_i++;
        }
    }

    // Ouput data frame with tau * E-1 fewer rows
    DataFrame< double > embedding( NRows - NPartial, NColOut, newColumnNames );

    // To keep track of where to insert column in new data frame
    size_t colCount = 0;

    // Slice to ignore rows with partial data
    std::slice slice_i;
    if ( tau < 0 ) {
        slice_i = std::slice( NPartial, NRows - NPartial, 1 );
    }
    else {
        slice_i = std::slice( 0, NRows - NPartial, 1 );
    }

    // Shift column data and write to embedding data frame
    for ( size_t col = 0; col < dataFrame.NColumns(); col++ ) {
        // for each embedding dimension
        for ( int e = 0; e < E; e++ ) {

            std::valarray< double > column = dataFrame.Column( col );
            
            // Returns a copy of the valarray object with its elements
            // shifted left n spaces (or right if n is negative).
            std::valarray< double > tmp = column.shift( e * tau );

            // Write shifted columns to the output embedding DataFrame
            embedding.WriteColumn( colCount, tmp[ slice_i ] );
            
            colCount++;
        }
    }

    return embedding;
}

//----------------------------------------------------------------------
// Simplex with path/file input
//----------------------------------------------------------------------
DataFrame< double > Simplex( std::string pathIn,
                             std::string dataFile,
                             std::string pathOut,
                             std::string predictFile,
                             std::string lib,
                             std::string pred,
                             int         E,
                             int         Tp,
                             int         knn,
                             int         tau,
                             int         exclusionRadius,
                             std::string colNames,
                             std::string targetName,
                             bool        embedded,
                             bool        const_predict,
                             bool        verbose )
{
    // DataFrame constructor loads data
    DataFrame< double > DF( pathIn, dataFile );

    // Pass data frame to Simplex 
    DataFrame< double > simplexProjection = Simplex( std::ref( DF ),
                                                     pathOut,
                                                     predictFile,
                                                     lib,
                                                     pred,
                                                     E,
                                                     Tp,
                                                     knn,
                                                     tau,
                                                     exclusionRadius,
                                                     colNames,
                                                     targetName,
                                                     embedded,
                                                     const_predict,
                                                     verbose );

    return simplexProjection;
}

//----------------------------------------------------------------------
// Simplex with DataFrame input
//----------------------------------------------------------------------
DataFrame<double> Simplex( DataFrame< double > & DF,
                           std::string pathOut,
                           std::string predictFile,
                           std::string lib,
                           std::string pred,
                           int         E,
                           int         Tp,
                           int         knn,
                           int         tau,
                           int         exclusionRadius,
                           std::string colNames,
                           std::string targetName,
                           bool        embedded,
                           bool        const_predict,
                           bool        verbose )
{
    // Instantiate Parameters
    Parameters parameters = Parameters( Method::Simplex, "", "",
                                        pathOut, predictFile,
                                        lib, pred, E, Tp, knn, tau, 0,
                                        exclusionRadius,
                                        colNames, targetName, embedded,
                                        const_predict, verbose );
    
    // Instantiate EDM::SimplexClass object
    SimplexClass SimplexModel = SimplexClass( DF, std::ref( parameters ) );

    SimplexModel.Project();

    return SimplexModel.projection;
}

//----------------------------------------------------------------------------
// 1) SMap with path/file input
//    Default SVD (LAPACK) assigned in SMap() overload 2)
//----------------------------------------------------------------------------
SMapValues SMap( std::string pathIn,
                 std::string dataFile,
                 std::string pathOut,
                 std::string predictFile,
                 std::string lib,
                 std::string pred,
                 int         E,
                 int         Tp,
                 int         knn,
                 int         tau,
                 double      theta,
                 int         exclusionRadius,
                 std::string columns,
                 std::string target,
                 std::string smapFile,
                 std::string derivatives,
                 bool        embedded,
                 bool        const_predict,
                 bool        verbose )
{
    // DataFrame constructor loads data
    DataFrame< double > DF( pathIn, dataFile );

    // Call overload 2) with DataFrame
    SMapValues SMapOutput = SMap( std::ref( DF ), pathOut, predictFile,
                                  lib, pred, E, Tp, knn, tau, theta,
                                  exclusionRadius,
                                  columns, target, smapFile, derivatives, 
                                  embedded, const_predict, verbose );
    return SMapOutput;
}

//----------------------------------------------------------------------------
// 2) SMap with DataFrame
//    Default SVD (LAPACK) assigned in Smap.cc overload 2)
//----------------------------------------------------------------------------
SMapValues SMap( DataFrame< double > & DF,
                 std::string pathOut,
                 std::string predictFile,
                 std::string lib,
                 std::string pred,
                 int         E,
                 int         Tp,
                 int         knn,
                 int         tau,
                 double      theta,
                 int         exclusionRadius,
                 std::string columns,
                 std::string target,
                 std::string smapFile,
                 std::string derivatives,
                 bool        embedded,
                 bool        const_predict,
                 bool        verbose )
{
    // Call overload 4) with default SVD function
    SMapValues SMapOutput = SMap( DF, pathOut, predictFile,
                                  lib, pred, E, Tp, knn, tau, theta, 
                                  exclusionRadius,
                                  columns, target, smapFile, derivatives,
                                  & SVD, // LAPACK SVD default
                                  embedded, const_predict, verbose);

    return SMapOutput;
}

//----------------------------------------------------------------------------
// 3) Data path/file with external solver object
//----------------------------------------------------------------------------
SMapValues SMap( std::string pathIn,
                 std::string dataFile,
                 std::string pathOut,
                 std::string predictFile,
                 std::string lib,
                 std::string pred,
                 int         E,
                 int         Tp,
                 int         knn,
                 int         tau,
                 double      theta,
                 int         exclusionRadius,
                 std::string columns,
                 std::string target,
                 std::string smapFile,
                 std::string derivatives,
                 std::valarray< double > (*solver)(DataFrame < double >,
                                               std::valarray < double >),
                 bool        embedded,
                 bool        const_predict,
                 bool        verbose )
{
    // DataFrame constructor loads data
    DataFrame< double > DF( pathIn, dataFile );
    
    // Call overload 4) with DataFrame and solver object
    SMapValues SMapOutput = SMap( std::ref( DF ), pathOut, predictFile,
                                  lib, pred, E, Tp, knn, tau, theta,
                                  exclusionRadius,
                                  columns, target, smapFile, derivatives, 
                                  solver, embedded, const_predict, verbose );
    return SMapOutput;
}

//----------------------------------------------------------------------------
// 4) DataFrame with external solver object
//----------------------------------------------------------------------------
SMapValues SMap( DataFrame< double > & DF,
                 std::string pathOut,
                 std::string predictFile,
                 std::string lib,
                 std::string pred,
                 int         E,
                 int         Tp,
                 int         knn,
                 int         tau,
                 double      theta,
                 int         exclusionRadius,
                 std::string columns,
                 std::string target,
                 std::string smapFile,
                 std::string derivatives,
                 std::valarray< double > (*solver)(DataFrame < double >,
                                               std::valarray < double >),
                 bool        embedded,
                 bool        const_predict,
                 bool        verbose )
{
    if ( derivatives.size() ) {} // -Wunused-parameter
    
    Parameters parameters = Parameters( Method::SMap, "", "",
                                        pathOut, predictFile,
                                        lib, pred, E, Tp, knn, tau, theta,
                                        exclusionRadius,
                                        columns, target, embedded,
                                        const_predict, verbose,
                                        smapFile );
    
    // Instantiate EDM::SMapClass object
    SMapClass SMapModel = SMapClass( DF, std::ref( parameters ) );

    SMapModel.Project( solver );

    SMapValues values = SMapValues();
    values.predictions  = SMapModel.projection;
    values.coefficients = SMapModel.coefficients;

    return values;    
}

//----------------------------------------------------------------------
// CCM with path/file input
//----------------------------------------------------------------------
CCMValues CCM( std::string pathIn,
               std::string dataFile,
               std::string pathOut,
               std::string predictFile,
               int         E,
               int         Tp,
               int         knn,
               int         tau,
               std::string colNames,
               std::string targetName,
               std::string libSizes_str,
               int         sample,
               bool        random,
               bool        replacement,
               unsigned    seed,
               bool        includeData,
               bool        verbose )
{
    // DataFrame constructor loads data
    DataFrame< double > DF( pathIn, dataFile );

    CCMValues ccmValues = CCM( std::ref( DF ), pathOut, predictFile,
                               E, Tp, knn, tau, colNames, targetName,
                               libSizes_str, sample, random, replacement,
                               seed, includeData, verbose );

    return ccmValues;
}

//----------------------------------------------------------------------
// CCM with DataFrame input
//----------------------------------------------------------------------
CCMValues CCM( DataFrame< double > & DF,
               std::string         pathOut,
               std::string         predictFile,
               int                 E,
               int                 Tp,
               int                 knn,
               int                 tau,
               std::string         colNames,
               std::string         targetName,
               std::string         libSizes_str,
               int                 sample,
               bool                random,
               bool                replacement,
               unsigned            seed,
               bool                includeData,
               bool                verbose )
{
    // Set library and prediction indices to entire library (embedded)
    std::stringstream ss;
    ss << "1 " << DF.NRows();

    Parameters parameters = Parameters( Method::CCM,
                                        "",           // pathIn
                                        "",           // dataFile
                                        pathOut,      // 
                                        predictFile,  // 
                                        ss.str(),     // lib_str
                                        ss.str(),     // pred_str
                                        E,            // 
                                        Tp,           // 
                                        knn,          // 
                                        tau,          // 
                                        0,            // theta
                                        0,            // exclusionRadius
                                        colNames,     // 
                                        targetName,   // 
                                        false,        // embedded
                                        false,        // const_predict
                                        verbose,      // 
                                        "",           // SmapFile
                                        "",           // blockFile
                                        0,            // multiviewEnsemble
                                        0,            // multiviewD
                                        false,        // multiviewTrainLib
                                        libSizes_str, // 
                                        sample,       // 
                                        random,       // 
                                        replacement,  // 
                                        seed,         //
                                        includeData );//

    // Instantiate EDM::Simplex::CCM object
    CCMClass CCMModel = CCMClass( DF, std::ref( parameters ) );

    CCMModel.Project();

    CCMValues values = CCMValues();
    values.AllLibStats = CCMModel.allLibStats;
    values.CrossMap1   = CCMModel.colToTargetValues;
    values.CrossMap2   = CCMModel.targetToColValues;

    return values;    
}

//----------------------------------------------------------------------
// Multiview with path/file input
//----------------------------------------------------------------------
MultiviewValues Multiview( std::string pathIn,
                           std::string dataFile,
                           std::string pathOut,
                           std::string predictFile,
                           std::string lib,
                           std::string pred,
                           int         D,
                           int         E,
                           int         Tp,
                           int         knn,
                           int         tau,
                           std::string columns,
                           std::string target,
                           int         multiview,
                           int         exclusionRadius,
                           bool        trainLib,
                           bool        verbose,
                           unsigned    nThreads )
{
    // DataFrame constructor loads data
    DataFrame< double > DF( pathIn, dataFile );

    MultiviewValues mvValues = Multiview( std::ref( DF ), pathOut, predictFile,
                                          lib, pred, D, E, Tp, knn, tau,
                                          columns, target, multiview,
                                          exclusionRadius, trainLib,
                                          verbose, nThreads);

    return mvValues;
}

//----------------------------------------------------------------------
// Multiview with DataFrame input
//----------------------------------------------------------------------
MultiviewValues Multiview( DataFrame< double > & DF,
                           std::string pathOut,
                           std::string predictFile,
                           std::string lib,
                           std::string pred,
                           int         D,
                           int         E,
                           int         Tp,
                           int         knn,
                           int         tau,
                           std::string columns,
                           std::string target,
                           int         multiview,
                           int         exclusionRadius,
                           bool        trainLib,
                           bool        verbose,
                           unsigned    nThreads )
{
    Parameters parameters = Parameters( Method::Simplex,
                                        "",           // pathIn
                                        "",           // dataFile
                                        pathOut,      // 
                                        predictFile,  // 
                                        lib,          // lib_str
                                        pred    ,     // pred_str
                                        E,            // 
                                        Tp,           // 
                                        knn,          // 
                                        tau,          // 
                                        0,            // theta
                                        exclusionRadius,
                                        columns,      // 
                                        target,       // 
                                        true,         // embedded true
                                        false,        // const_predict
                                        verbose,      // 
                                        "",           // SmapFile
                                        "",           // blockFile
                                        multiview,    // multiviewEnsemble,
                                        D,            // multiviewD
                                        trainLib );   // multiviewTrainLib

    // Instantiate EDM::Simplex::Multiview object
    MultiviewClass MultiviewModel = MultiviewClass( DF, std::ref( parameters ) );

    MultiviewModel.Project( nThreads );

    return MultiviewModel.MVvalues;
}
