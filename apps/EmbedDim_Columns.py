#!/usr/bin/env python3
'''
EmbedDim_Columns : Parallel embedding-dimension estimation across the
columns of an N x M DataFrame.

For each feature column the optimal embedding dimension E is estimated by a
serial Simplex() sweep over [minE : maxE], reducing each E to prediction
skill (rho) exactly as pyEDM.EmbedDimension does
(ComputeError(Simplex(...)['Observations'], ...['Predictions'])['rho']).

Parallelism is at the column level: one task per column, each task running
its own serial E-sweep.  There is no inner pool, so no nested multiprocessing.
The multiprocessing start method is resolved to a non-fork context
(forkserver or spawn); 'fork' is rejected.

Returns a pandas DataFrame with columns ['column','target','E','rho'].
'''

import os
import sys
import time
import warnings
import argparse
import multiprocessing

import numpy  as np
import pandas as pd

from pyEDM import Simplex, ComputeError


#-----------------------------------------------------------------------------
# Multiprocessing context resolution
#-----------------------------------------------------------------------------
def ResolveMPContext( mpMethod = None ):
    '''Return a multiprocessing context that is never 'fork'.

    If mpMethod is given it is honored (except 'fork', which raises).
    Otherwise: query the platform default; if it exists and is not 'fork',
    use it.  If it is 'fork', use 'forkserver' when constructable, else
    fall back to 'spawn'.
    '''
    if mpMethod is not None :
        if mpMethod == 'fork' :
            raise ValueError(
                "EmbedDim_Columns: mpMethod 'fork' is not permitted. "
                "Use 'spawn' or 'forkserver'." )
        return multiprocessing.get_context( mpMethod )

    # No explicit request : query the platform default.
    default = multiprocessing.get_start_method( allow_none = False )

    if default != 'fork' :
        return multiprocessing.get_context( default )

    # Platform default is 'fork' : prefer forkserver, else spawn.
    if 'forkserver' in multiprocessing.get_all_start_methods() :
        try :
            return multiprocessing.get_context( 'forkserver' )
        except ValueError :
            pass

    return multiprocessing.get_context( 'spawn' )


#-----------------------------------------------------------------------------
# Worker : runs entirely inside a child process
#-----------------------------------------------------------------------------
def _ColumnSweep( task ):
    '''Estimate E for one column.

    task = ( index, column, target, subFrame, params )
    Returns ( index, column, target, E, rho ).
    Any failure (degenerate / too-short / Simplex error) yields
    E, rho = NaN, NaN rather than killing the batch.
    '''
    index, column, target, subFrame, P = task

    try :
        # Optional runtime guarantee of single-threaded math.
        try :
            from threadpoolctl import threadpool_limits
            limiter = threadpool_limits( limits = 1 )
        except Exception :
            limiter = None

        try :
            warnings.simplefilter( 'ignore' ) # NaN
            prevE   = None
            prevRho = None
            allE    = []    # (E, rho) global-max
            peakE   = None  # if firstMax True
            peakRho = None  # if firstMax True

            for E in range( P['minE'], P['maxE'] + 1 ):
                smpx = Simplex( dataFrame       = subFrame,
                                columns         = column,
                                target          = target,
                                lib             = P['lib'],
                                pred            = P['pred'],
                                E               = E,
                                Tp              = P['Tp'],
                                tau             = P['tau'],
                                exclusionRadius = P['exclusionRadius'],
                                embedded        = False,
                                validLib        = P['validLib'],
                                noTime          = P['noTime'],
                                ignoreNan       = P['ignoreNan'] )

                err = ComputeError( smpx['Observations'], smpx['Predictions'] )
                rho = err['rho']

                if rho is None or (isinstance(rho, float) and np.isnan( rho )):
                    continue   # skip non-finite rho; do not trip turnover

                allE.append( ( E, rho ) )

                if P['firstMax'] :
                    # First local maximum : stop when rho turns down.
                    if prevRho is not None and rho < prevRho :
                        peakE, peakRho = prevE, prevRho
                        break
                    prevE, prevRho = E, rho

            if P['firstMax'] :
                if peakE is None :
                    # Never turned over (monotonic rise, or single point).
                    if prevRho is not None :
                        peakE, peakRho = prevE, prevRho
                    else :
                        return ( index, column, target, np.nan, np.nan )
                # Estimate is never below the minE floor.
                peakE = max( peakE, P['minE'] )
                return ( index, column, target,
                         np.uint16(peakE),
                         peakRho.round(4).astype(np.float32) )

            # Global maximum, ties broken toward the smaller E.
            if not allE :
                return ( index, column, target, np.nan, np.nan )

            bestE, bestRho = max( allE, key = lambda er: ( er[1], -er[0] ) )
            bestE = max( bestE, P['minE'] )   # never below the minE floor
            return ( index, column, target, np.uint16(bestE),
                     bestRho.round(4).astype(np.float32) )

        finally :
            if limiter is not None :
                limiter.unregister()

    except Exception :
        return ( index, column, target, np.nan, np.nan )


#-----------------------------------------------------------------------------
# Driver
#-----------------------------------------------------------------------------
def EmbedDim_Columns( data,
                      target          = None,
                      maxE            = 15,
                      minE            = 1,
                      lib             = None,
                      pred            = None,
                      Tp              = 1,
                      tau             = -1,
                      exclusionRadius = 0,
                      firstMax        = False,
                      validLib        = [],
                      noTime          = False,
                      ignoreNan       = True,
                      nWorkers        = None,
                      mpMethod        = None,
                      chunksize       = 1,
                      outputFile      = None,
                      verbose         = False,
                      logPct          = 5,
                      plot            = False ):
    '''Estimate embedding dimension for every column of data.

    data       : pandas DataFrame, N observations x M columns.  Unless
                 noTime=True the first column is treated as a time/index
                 vector and is not itself evaluated.
    target     : None  -> each feature is projected to itself.
                 name  -> every feature is projected to this column.
    minE,maxE  : inclusive embedding-dimension sweep.
    lib,pred   : pyEDM library/prediction ranges.  None -> full record [1,N].
    firstMax   : True  -> report the first (lowest-E) local maximum of rho.
                 False -> report the global maximum (ties -> smaller E).
    nWorkers   : column-level processes.  None -> os.cpu_count().
    mpMethod   : start method; None -> resolved non-fork context. 'fork' raises.
    logPct     : when verbose, emit a progress line every logPct percent of
                 columns (default 5).
    Returns DataFrame ['column','target','E','rho'].
    '''

    if not isinstance( data, pd.DataFrame ):
        raise TypeError( "EmbedDim_Columns: data must be a pandas DataFrame." )

    # minE has a hard floor of 1; values below are raised, not rejected.
    if minE < 1 :
        minE = 1
    if maxE < minE :
        raise ValueError( "EmbedDim_Columns: require maxE >= minE." )

    columns = data.columns.to_list()

    # ---- feature / time split ------------------------------------------
    if noTime :
        timeName = None
        features = columns
    else :
        if len( columns ) < 2 :
            raise ValueError( "EmbedDim_Columns: noTime=False but data has "
                              "no feature columns after the time column." )
        timeName = columns[0]
        features = columns[1:]

    if target is not None and target not in columns :
        raise ValueError( f"EmbedDim_Columns: target '{target}' is not a "
                          f"column of data." )

    # ---- lib / pred default : full record ------------------------------
    N = len( data )

    # If no lib and pred, create from full data span
    if lib is None :
        lib = [ 1, N ]
    if pred is None :
        pred = [ 1, N ]

    # ---- worker parameters (identical for every column) ----------------
    params = { 'minE'            : minE,
               'maxE'            : maxE,
               'lib'             : lib,
               'pred'            : pred,
               'Tp'              : Tp,
               'tau'             : tau,
               'exclusionRadius' : exclusionRadius,
               'validLib'        : validLib,
               'noTime'          : noTime,
               'ignoreNan'       : ignoreNan,
               'firstMax'        : firstMax }

    # ---- build minimal per-column tasks --------------------------------
    tasks = []
    for i, feature in enumerate( features ):
        tgt = feature if target is None else target

        needed = []
        if timeName is not None :
            needed.append( timeName )
        needed.append( feature )
        if tgt != feature :
            needed.append( tgt )
        # de-duplicate preserving order
        seen   = set()
        needed = [ c for c in needed if not ( c in seen or seen.add( c ) ) ]

        subFrame = data[ needed ].copy()
        tasks.append( ( i, feature, tgt, subFrame, params ) )

    M = len( tasks )
    if M == 0 :
        return pd.DataFrame( columns = ['column','target','E','rho'] ) # empty

    # ---- worker count ---------------------------------------------------
    if nWorkers is None :
        nWorkers = os.cpu_count() or 1
    nWorkers = max( 1, min( nWorkers, M, os.cpu_count() or 1 ) )

    # ---- pin BLAS threads BEFORE the pool is created (inherited) --------
    for var in ( 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS' ) :
        os.environ.setdefault( var, '1' )

    ctx = ResolveMPContext( mpMethod )

    # ---- progress logging cadence --------------------------------------
    stride = max( 1, round( M * logPct / 100.0 ) )

    if verbose :
        print( f"EmbedDim_Columns: {M} columns, {nWorkers} workers, "
               f"method={ctx.get_start_method()}, E=[{minE}:{maxE}], "
               f"firstMax={firstMax}", flush = True )

    t0 = time.time()

    # ---- run : column-level pool, results as completed -----------------
    resultsByIndex = {}
    done = 0

    with ctx.Pool( processes = nWorkers ) as pool :
        for res in pool.imap_unordered( _ColumnSweep, tasks,
                                        chunksize = chunksize ):
            index, col, tgt, E_, rho_ = res
            resultsByIndex[ index ] = ( col, tgt, E_, rho_ )
            done += 1

            if verbose and ( done % stride == 0 or done == M ):
                pct = 100.0 * done / M
                print( f"  [{done}/{M} {pct:5.1f}%] {col} -> {tgt}  "
                       f"maxE={E_}  maxRho={rho_}", flush = True )

    if verbose :
        print( f"EmbedDim_Columns: finished {M}/{M} in "
               f"{time.time() - t0:.2f}s", flush = True )

    # ---- assemble in original column order -----------------------------
    rows = [ resultsByIndex[ i ] for i in range( M ) ]
    df   = pd.DataFrame( rows,
                         columns = ['column','target','E','rho'] )

    if outputFile :
        if '.csv' in outputFile[-4:] :
            df.to_csv( outputFile, index = False )
        elif '.feather' in outputFile[-8:] :
            df.to_feather( outputFile )
        elif any([_ in outputFile[-4:] for _ in ['.pkl','.gz','.xz','.zip'] ]):
            df.to_pickle( outputFile )
        else :
            err = 'EmbedDim_Columns() ' +\
                f'unrecognized outputFile format in {outputFile}'
            print( err )

        if verbose :
            print( f"EmbedDim_Columns: wrote {outputFile}", flush = True )

    if plot :
        _Plot( df, Tp )

    return df


#-----------------------------------------------------------------------------
def _Plot( df, Tp ):
    '''Bar plot of E per column (parent process only).'''
    try :
        import matplotlib.pyplot as plt
    except Exception as e :
        print( f"EmbedDim_Columns: plot unavailable ({e})", file = sys.stderr )
        return

    valid = df.dropna( subset = ['E'] )
    fig, ax = plt.subplots( figsize = ( max( 6, 0.4 * len( df ) ), 4 ) )
    ax.bar( valid['column'].astype( str ), valid['E'] )
    ax.set_xlabel( 'Column' )
    ax.set_ylabel( 'Embedding Dimension E' )
    ax.set_title( f"EmbedDim_Columns  (Tp={Tp})" )
    plt.xticks( rotation = 90 )
    plt.tight_layout()
    plt.show()


#-----------------------------------------------------------------------------
# Command-line interface
#-----------------------------------------------------------------------------
def ParseArguments():
    parser = argparse.ArgumentParser(
        description = 'Parallel embedding-dimension estimation across '
                      'DataFrame columns.' )

    parser.add_argument( '-i', '--inputFile', dest = 'inputFile',
                         required = True,
                         help = 'CSV input file (N x M).' )
    
    parser.add_argument( '-ip', '--inputPath', dest = 'inputPath',
                         default = './',
                         help = 'Path to input file. Default ./' )
    
    parser.add_argument( '-t', '--target', dest = 'target', default = None,
                         help = 'Target column. None -> each self-targets.' )
    
    parser.add_argument( '-maxE', '--maxE', dest = 'maxE', type = int,
                         default = 15, help = 'Maximum E. Default 15.' )
    
    parser.add_argument( '-minE', '--minE', dest = 'minE', type = int,
                         default = 1, help = 'Minimum E. Default 1.' )
    
    parser.add_argument( '-l', '--lib', dest = 'lib', nargs = '+',
                         type = int, default = None,
                         help = 'Library range, e.g. -l 1 500. Default full.' )
    
    parser.add_argument( '-p', '--pred', dest = 'pred', nargs = '+',
                         type = int, default = None,
                         help = 'Prediction range. Default full.' )
    
    parser.add_argument( '-Tp', '--Tp', dest = 'Tp', type = int,
                         default = 1, help = 'Prediction interval. Default 1.' )
    
    parser.add_argument( '-tau', '--tau', dest = 'tau', type = int,
                         default = -1, help = 'Embedding delay. Default -1.' )
    
    parser.add_argument( '-xr', '--exclusionRadius', dest = 'exclusionRadius',
                         type = int, default = 0,
                         help = 'Exclusion radius. Default 0.' )
    
    parser.add_argument( '-f', '--firstMax', dest = 'firstMax',
                         action = 'store_true',
                         help = 'Report first local maximum of rho.' )
    
    parser.add_argument( '-vl', '--validLib', dest = 'validLib', nargs = '+',
                         type = int, default = [],
                         help = 'Valid library rows. Default [].' )
    
    parser.add_argument( '-noTime', '--noTime', dest = 'noTime',
                         action = 'store_true',
                         help = 'First column is data, not time.' )
    
    parser.add_argument( '-in', '--ignoreNan', dest = 'ignoreNan',
                         action = 'store_false',
                         help = 'Disable NaN handling (default on).' )
    
    parser.add_argument( '-nw', '--nWorkers', dest = 'nWorkers', type = int,
                         default = None,
                         help = 'Column-level workers. Default os.cpu_count().' )
    
    parser.add_argument( '-mp', '--mpMethod', dest = 'mpMethod',
                         default = None,
                         help = "Start method (spawn|forkserver). "
                                "'fork' is rejected." )
    
    parser.add_argument( '-cz', '--chunksize', dest = 'chunksize',
                         type = int, default = 1,
                         help = 'Pool chunksize. Default 1.' )
    
    parser.add_argument( '-logPct', '--logPct', dest = 'logPct', type = float,
                         default = 5,
                         help = 'Percent-of-columns logging cadence. Default 5.')
    
    parser.add_argument( '-o', '--outputFile', dest = 'outputFile',
                         default = None, help = 'CSV output file.' )
    
    parser.add_argument( '-v', '--verbose', dest = 'verbose',
                         action = 'store_true', help = 'Progress logging.' )
    
    parser.add_argument( '-P', '--plot', dest = 'plot',
                         action = 'store_true', help = 'Plot E per column.' )

    return parser.parse_args()


def main():
    '''Command-line entry point.

    Parses CLI flags via ParseArguments(), reads the input .csv .feather from
    inputPath/inputFile into a pandas DataFrame, calls EmbedDim_Columns()
    with the parsed arguments, and either writes the result to outputFile
    or prints it to stdout when no output file is given.

    Must run under the `if __name__ == '__main__':` guard: the resolved
    multiprocessing start method is 'spawn' or 'forkserver', which re-import
    this module in each worker.
    '''
    args = ParseArguments()

    path = os.path.join( args.inputPath, args.inputFile )
    if '.csv' in args.inputFile[-4:] :
        data = pd.read_csv( path )
    elif '.feather' in args.inputFile[-8:] :
        data = pd.read_feather( path )
    else :
        msg = f'Input file {args.inputFile} must be csv or feather'
        raise( RuntimeError( msg ) )

    df = EmbedDim_Columns( data            = data,
                           target          = args.target,
                           maxE            = args.maxE,
                           minE            = args.minE,
                           lib             = args.lib,
                           pred            = args.pred,
                           Tp              = args.Tp,
                           tau             = args.tau,
                           exclusionRadius = args.exclusionRadius,
                           firstMax        = args.firstMax,
                           validLib        = args.validLib,
                           noTime          = args.noTime,
                           ignoreNan       = args.ignoreNan,
                           nWorkers        = args.nWorkers,
                           mpMethod        = args.mpMethod,
                           chunksize       = args.chunksize,
                           outputFile      = args.outputFile,
                           verbose         = args.verbose,
                           logPct          = args.logPct,
                           plot            = args.plot )

    if not args.outputFile :
        print( df.to_string( index = False ) )


if __name__ == '__main__':
    main()
