"""
test_knn_filtering.py — Neighbors.FindNeighbors() knn filtering tests.

Exercises the exclusionRadius / libOverlap / validLib / tieBreak filtering
in FindNeighbors() through the public API (returnObject = True exposes the
resolved knn_neighbors / knn_distances).

The filtering must satisfy one invariant on every path:

    knn_distances and knn_neighbors are (N_pred, knn); a slot is a real
    neighbor iff its distance is finite; finite slots are contiguous at the
    front of each row in ascending-distance order; no finite slot lies
    within exclusionRadius of its prediction (this subsumes the self-match);
    a row with no finite slot yields a nan prediction.

Regression coverage for issue #74 : deficient / fully-excluded rows must be
inf-padded, never back-filled with excluded or self neighbors.
"""

from numpy  import asarray, isfinite, isnan, diff, array_equal, sin, cos, arange, nan
from pandas import DataFrame
import pytest

import pyEDM as EDM

from conftest import SimplexArgs, SMapArgs


# ---------------------------------------------------------------------------
# Shared invariant checker
# ---------------------------------------------------------------------------

def _assert_filter_invariants( obj ):
    '''Assert the FindNeighbors() output invariants; return finite counts.'''
    dist = asarray( obj.knn_distances, dtype = float )
    nbr  = asarray( obj.knn_neighbors )
    pred = asarray( obj.pred_i )
    knn  = obj.knn
    xrad = obj.exclusionRadius

    N_pred = len( pred )

    # (1) Shape is (N_pred, knn) : compaction preserved, no k_query bloat
    assert dist.shape == ( N_pred, knn )
    assert nbr.shape  == dist.shape

    finite = isfinite( dist )
    counts = finite.sum( axis = 1 )

    for r in range( N_pred ):
        f   = finite[ r ]
        n_f = int( f.sum() )

        # (2) Finite slots contiguous at the front (inf padding trails)
        assert f[ :n_f ].all() and not f[ n_f: ].any()

        # (3) Finite distances non-decreasing (ascending)
        assert ( diff( dist[ r ][ f ] ) >= 0 ).all()

        # (4) No leakage : no finite neighbor within exclusionRadius
        within = abs( pred[ r ] - nbr[ r ][ f ] ) <= xrad
        assert not within.any()

    # (5) Never more finite neighbors than knn
    assert ( counts <= knn ).all()

    return counts


# ---------------------------------------------------------------------------
# tieBreak path (top-level Simplex)
# ---------------------------------------------------------------------------

def test_simplex_tiebreak_exclusion_no_leakage():
    '''Simplex tieBreak, exclusionRadius : invariants, no leakage, full knn'''
    data = EDM.sampleData['TentMap']
    kwargs = SimplexArgs.copy()
    kwargs.update( dict( columns         = 'TentMap',
                         target          = 'TentMap',
                         lib             = [1, 500],
                         pred            = [1, 500],
                         E               = 2,
                         Tp              = 1,
                         exclusionRadius = 5,
                         returnObject    = True ) )
    S = EDM.Simplex( data, **kwargs )

    counts = _assert_filter_invariants( S )
    assert ( counts == S.knn ).all()   # dense library : every row full


def test_simplex_tiebreak_deficiency_no_leakage():
    '''Simplex tieBreak tiny lib + large exclusionRadius : inf-padded, no leak'''
    N = 8
    data = DataFrame( { 'time' : range( 1, N + 1 ),
                        'X'    : [ 1., 2, 3, 4, 5, 6, 7, 8 ],
                        'Y'    : [ 1., 4, 9, 16, 25, 36, 49, 64 ] } )
    kwargs = SimplexArgs.copy()
    kwargs.update( dict( columns         = 'Y',
                         target          = 'X',
                         lib             = [1, N],
                         pred            = [1, N],
                         E               = 2,
                         tau             = -1,
                         Tp              = 0,
                         exclusionRadius = 4,
                         embedded        = False,
                         returnObject    = True ) )
    with pytest.warns( UserWarning ):
        S = EDM.Simplex( data, **kwargs )

    counts = _assert_filter_invariants( S )
    assert ( counts == 0 ).any()       # setup forces fully-excluded rows


def test_simplex_tiebreak_deterministic():
    '''tieBreak selection deterministic : identical neighbors across runs'''
    data = EDM.sampleData['TentMapNoise']
    kwargs = SimplexArgs.copy()
    kwargs.update( dict( columns         = 'TentMap',
                         target          = 'TentMap',
                         lib             = [1, 300],
                         pred            = [1, 300],
                         E               = 3,
                         Tp              = 1,
                         exclusionRadius = 3,
                         returnObject    = True ) )
    S1 = EDM.Simplex( data, **kwargs )
    S2 = EDM.Simplex( data, **kwargs )

    assert ( asarray( S1.knn_neighbors ) == asarray( S2.knn_neighbors ) ).all()
    assert array_equal( asarray( S1.knn_distances ),
                        asarray( S2.knn_distances ), equal_nan = True )


# ---------------------------------------------------------------------------
# non-tieBreak path (SMap)
# ---------------------------------------------------------------------------

def test_smap_exclusion_no_leakage():
    '''SMap (non-tieBreak), exclusionRadius : invariants, no leakage, full knn'''
    data = EDM.sampleData['TentMap']
    kwargs = SMapArgs.copy()
    kwargs.update( dict( columns         = 'TentMap',
                         target          = 'TentMap',
                         lib             = [1, 500],
                         pred            = [1, 500],
                         E               = 2,
                         Tp              = 1,
                         theta           = 2,
                         knn             = 20,
                         exclusionRadius = 5,
                         returnObject    = True ) )
    S = EDM.SMap( data, **kwargs )

    counts = _assert_filter_invariants( S )
    assert ( counts == S.knn ).all()


# ---------------------------------------------------------------------------
# libOverlap path (self-match exclusion)
# ---------------------------------------------------------------------------

def test_liboverlap_excludes_self():
    '''lib == pred, exclusionRadius = 0 : self-match dropped, full knn kept'''
    data = EDM.sampleData['TentMap']
    kwargs = SMapArgs.copy()
    kwargs.update( dict( columns         = 'TentMap',
                         target          = 'TentMap',
                         lib             = [1, 300],
                         pred            = [1, 300],
                         E               = 2,
                         Tp              = 1,
                         theta           = 2,
                         knn             = 15,
                         exclusionRadius = 0,
                         returnObject    = True ) )
    S = EDM.SMap( data, **kwargs )
    assert S.libOverlap

    _assert_filter_invariants( S )   # xrad = 0 -> no neighbor == pred

    dist   = asarray( S.knn_distances, dtype = float )
    nbr    = asarray( S.knn_neighbors )
    pred   = asarray( S.pred_i )
    finite = isfinite( dist )

    # No finite slot is the self-match, and the +1 over-query preserves knn
    for r in range( len( pred ) ):
        assert not ( nbr[ r ][ finite[ r ] ] == pred[ r ] ).any()
    assert ( finite.sum( axis = 1 ) == S.knn ).all()


# ---------------------------------------------------------------------------
# no_filtering fast path (disjoint library, no exclusion)
# ---------------------------------------------------------------------------

def test_disjoint_no_filtering_fast_path():
    '''Disjoint lib/pred, exclusionRadius = 0, non-tieBreak : raw KDTree result
       is the answer : all distances finite, shape (N_pred, knn).'''
    data = EDM.sampleData['TentMap']
    kwargs = SMapArgs.copy()
    kwargs.update( dict( columns         = 'TentMap',
                         target          = 'TentMap',
                         lib             = [1, 100],
                         pred            = [201, 500],
                         E               = 2,
                         Tp              = 1,
                         theta           = 2,
                         knn             = 10,
                         exclusionRadius = 0,
                         returnObject    = True ) )
    S = EDM.SMap( data, **kwargs )
    assert not S.libOverlap

    counts = _assert_filter_invariants( S )
    assert isfinite( asarray( S.knn_distances, dtype = float ) ).all()
    assert ( counts == S.knn ).all()


# ---------------------------------------------------------------------------
# Deficiency / fully-excluded rows  (issue #74 regression)
# ---------------------------------------------------------------------------

def test_deficiency_padding_and_nan():
    '''exclusionRadius excludes every neighbor : rows are inf-padded (never
       back-filled with self / excluded points) and predictions are nan.'''
    N = 10
    data = DataFrame( { 'time' : range( 1, N + 1 ),
                        'X'    : sin( arange( N ) / 1.5 ),
                        'Y'    : cos( arange( N ) / 1.5 ) } )
    kwargs = SMapArgs.copy()
    kwargs.update( dict( columns         = 'Y',
                         target          = 'X',
                         lib             = [1, N],
                         pred            = [1, N],
                         E               = 2,
                         tau             = -1,
                         Tp              = 0,
                         theta           = 3,
                         exclusionRadius = 8,
                         embedded        = False,
                         returnObject    = True ) )
    with pytest.warns( UserWarning ):
        S = EDM.SMap( data, **kwargs )

    counts = _assert_filter_invariants( S )   # includes no-leakage check

    dist = asarray( S.knn_distances, dtype = float )
    nbr  = asarray( S.knn_neighbors )
    pred = asarray( S.pred_i )

    # No finite-weight self-match anywhere (the leak this fix removes)
    for r in range( len( pred ) ):
        f = isfinite( dist[ r ] )
        assert not ( nbr[ r ][ f ] == pred[ r ] ).any()

    # Fully-excluded rows : every distance inf -> nan prediction
    zero_valid = ( counts == 0 )
    assert zero_valid.any()

    preds = S.Projection['Predictions'].to_numpy()
    assert int( isnan( preds ).sum() ) >= int( zero_valid.sum() )


# ---------------------------------------------------------------------------
# validLib path
# ---------------------------------------------------------------------------

def test_validlib_filtering_no_leakage():
    '''validLib removes interior library points : filtering still yields the
       (N_pred, knn) shape with no exclusionRadius leakage.'''
    N    = 200
    data = EDM.sampleData['TentMap'].iloc[ :N ].reset_index( drop = True )

    validLib = [ True ] * N
    for i in range( 0, N, 3 ):   # drop every third library point
        validLib[ i ] = False

    kwargs = SimplexArgs.copy()
    kwargs.update( dict( columns         = 'TentMap',
                         target          = 'TentMap',
                         lib             = [1, N],
                         pred            = [1, N],
                         E               = 2,
                         Tp              = 1,
                         exclusionRadius = 2,
                         validLib        = validLib,
                         returnObject    = True ) )
    S = EDM.Simplex( data, **kwargs )

    _assert_filter_invariants( S )


def test_smap_validlib_fewer_than_knn():
    '''validLib leaving fewer valid library points than knn is an invalid
       system : SMap must raise ValueError with resolving information,
       not warn-and-crash or silently reduce knn (issue #74 follow-up).'''
    data = DataFrame( { 'time' : range( 1, 13 ),
                        'X'    : [ 1, 2, 3, nan, nan, 6, 7, 8, 9, 10, 11, 12 ],
                        'Y'    : [ 12, 11, 10, 9, 8, 7, 6, nan, nan, 3, 2, 1 ] } )
    kwargs = SMapArgs.copy()
    kwargs.update( dict( columns         = 'Y',
                         target          = 'X',
                         lib             = [1, 10],
                         pred            = [1, 10],
                         E               = 2,
                         tau             = -1,
                         theta           = 1,
                         exclusionRadius = 2,
                         validLib        = data['X'].notna().values ) )
    with pytest.raises( ValueError ):
        EDM.SMap( data, **kwargs )
