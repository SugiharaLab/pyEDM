# python modules
from warnings import warn

# package modules
from numpy import array, arange, zeros
from numpy import repeat, inf, isfinite, sqrt, lexsort, where
from numpy import abs as npabs, sum as npsum
from scipy.spatial import KDTree

#--------------------------------------------------------------------
# EDM Method
#--------------------------------------------------------------------
def FindNeighbors( self ) :
#--------------------------------------------------------------------
    '''Use Scipy KDTree to find neighbors

       Note: If dimensionality is k, the number of points n in 
       the data should be n >> 2^k, otherwise KDTree efficiency is low. 
       k:2^k pairs { 4 : 16, 5 : 32, 7 : 128, 8 : 256, 10 : 1024 }

       KDTree returns ndarray of knn_neighbors as indices with respect
       to the data array passed to KDTree, not with respect to the lib_i
       of embedding[ lib_i ] passed to KDTree. Since lib_i are generally 
       not [0..N] the knn_neighbors need to be adjusted to lib_i reference
       for use in projections.

       If there are degenerate lib & pred indices (libOverlap) and/or
       exclusionRadius > 0, a vectorised boolean mask is applied to the
       full (N_pred, k_query) neighbor matrix to exclude self-matches
       and temporally proximate library rows. The first knn valid
       neighbors per row are selected via cumulative-sum indexing and
       compacted into dense (N_pred, knn) output arrays.

       If self.tieBreak is True (set only on the top-level Simplex path)
       selection among candidates is made deterministic and backend
       independent using the ordering :
           1. distance ascending
           2. |predRow - libRow| ascending (proximity to prediction)
           3. libRow ascending)
       on original data-row indices. The tertiary condition applies proximal
       tie breaking (|predRow - libRow|) to previous time library indices.
       The common case is vectorized, a per-row full scan runs only for rows
       whose knn-th distance reaches the over-query boundary (possible
       straddling tie) or that are deficient.
       tieBreak defaults False for SMap / CCM / Multiview.

       Writes to EDM object:
         knn_distances : sorted knn distances
         knn_neighbors : library neighbor rows of knn_distances
    '''
    if self.verbose :
        print( f'{self.name}: FindNeighbors()' )

    N_pred_rows = len( self.pred_i )

    tieBreak = getattr( self, 'tieBreak', False )

    #-----------------------------------------------
    # Determine if exclusionRadius filtering needed
    #-----------------------------------------------
    exclusionRadius_knn = False

    if self.exclusionRadius > 0 :
        if self.libOverlap :
            exclusionRadius_knn = True
        else :
            # If no libOverlap and exclusionRadius is less than the
            # distance in rows between lib : pred, no library neighbor
            # exclusion needed.
            excludeRow = 0
            if self.pred_i[0] > self.lib_i[-1] :
                # pred start is beyond lib end
                excludeRow = self.pred_i[0] - self.lib_i[-1]
            elif self.lib_i[0] > self.pred_i[-1] :
                # lib start row is beyond pred end
                excludeRow = self.lib_i[0] - self.pred_i[-1]
            if self.exclusionRadius >= excludeRow :
                exclusionRadius_knn = True

    #-----------------------------------------------
    # Filter library by validLib
    #-----------------------------------------------
    if len( self.validLib ) :
        # Convert self.validLib boolean vector to data indices
        data_i = array( range( self.Data.shape[0] ), dtype = int )
        validLib_i = data_i[ self.validLib ]

        # Filter lib_i to only include valid library points
        lib_i_valid = array( [ i for i in self.lib_i if i in validLib_i ],
                             dtype = int )

        if len( lib_i_valid ) == 0 :
            raise ValueError(
                f'{self.name}: FindNeighbors() : '
                'No valid library points found. '
                'All library points excluded by validLib.' )

        if len( lib_i_valid ) < self.knn :
            warn( f'{self.name}: FindNeighbors() : '
                  f'Only {len(lib_i_valid)} valid library points found, '
                  f'but knn={self.knn}. Reduce knn or check validLib.' )

        # Replace lib_ with lib_i_valid
        self.lib_i = lib_i_valid

    #-----------------------------------------------
    # Determine k_query : neighbors to request
    #-----------------------------------------------
    k_query = self.knn

    if exclusionRadius_knn :
        # knn_neighbors exclusionRadius adjustment required
        # Ask for enough knn to discard exclusionRadius neighbors
        # This is controlled by the factor: self.xRadKnnFactor
        k_query = min( self.knn * self.xRadKnnFactor, len( self.lib_i ) )
    elif self.libOverlap :
        # Increase knn +1 if libOverlap
        # Returns one more column in knn_distances, knn_neighbors
        # The first nn degenerate with the prediction vector
        # is replaced with the 2nd to knn+1 neighbors
        k_query = k_query + 1

    if len( self.validLib ) :
        # Have to examine all knn
        k_query = len( self.lib_i )

    if tieBreak and len( self.lib_i ) > self.knn :
        # tieBreak needs a lookahead column even in the plain (disjoint)
        # case so a boundary tie can be detected and completed.
        k_query = min( max( k_query, self.knn + 1 ), len( self.lib_i ) )

    #-----------------------------------------------
    # Compute KDTree on library of embedding vectors
    #-----------------------------------------------
    self.kdTree = KDTree( self.Embedding.iloc[ self.lib_i, : ].to_numpy(),
                          leafsize      = 20,
                          compact_nodes = True,
                          balanced_tree = True )

    #------------------------------------------------
    # Query prediction set : workers = -1 all threads
    #------------------------------------------------
    knn_distances, knn_neighbors = self.kdTree.query(
        self.Embedding.iloc[ self.pred_i, : ].to_numpy(),
        k = k_query, eps = 0, p = 2, workers = self.kdWorkers )

    # KDTree.query squeezes the last dimension when k == 1
    if k_query == 1 :
        knn_distances = knn_distances[:, None]
        knn_neighbors = knn_neighbors[:, None]

    #-----------------------------------------------
    # Map KDTree indices to lib_i row references
    #-----------------------------------------------
    # KDTree.query returns indices 0..len(lib_i)-1.
    # Use lib_i as a lookup table to recover embedding row indices.
    lib_i_arr     = array( self.lib_i )
    knn_neighbors = lib_i_arr[ knn_neighbors ]

    #-----------------------------------------------
    # Vectorised exclusion mask
    #-----------------------------------------------
    needs_filtering = self.libOverlap or exclusionRadius_knn \
                      or k_query > self.knn or tieBreak

    if needs_filtering :
        pred_col = array( self.pred_i )[:, None]  # (N_pred, 1)

        # Build boolean mask: True = exclude this neighbor
        if exclusionRadius_knn :
            # abs(pred - neighbor) <= exclusionRadius subsumes self-match
            mask = abs( pred_col - knn_neighbors ) <= self.exclusionRadius
        elif self.libOverlap :
            # libOverlap only: exclude the self-match
            mask = ( pred_col == knn_neighbors )
        else :
            # validLib over-query only: no exclusions, trim to knn
            mask = zeros( knn_neighbors.shape, dtype = bool )

        if tieBreak :
            # Deterministic Simplex selection (see _TieBreakSelect below).
            self.knn_neighbors, self.knn_distances = _TieBreakSelect(
                self, knn_neighbors, knn_distances, mask, k_query,
                exclusionRadius_knn )
            return

        # Select the first self.knn valid (unmasked) neighbors per row
        valid   = ~mask
        cs      = valid.cumsum( axis = 1 )
        first_k = valid & ( cs <= self.knn )

        # Check for rows with insufficient valid neighbors
        valid_counts = cs[ :, -1 ]
        deficient    = valid_counts < self.knn

        if deficient.any() :
            warn( f'{self.name}: FindNeighbors() : '
                  'Failed to find knn outside exclusionRadius '
                  f'{self.exclusionRadius} for some predictions. '
                  f'Consider reducing knn {self.knn}.' )

            # Fall back to first knn raw neighbors for deficient rows
            for i in range( N_pred_rows ) :
                if deficient[ i ] :
                    first_k[ i, : ]          = False
                    first_k[ i, :self.knn ]  = True

        # Compact: gather selected entries into dense (N_pred, k_out)
        # k_out guards against validLib leaving fewer points than knn
        k_out = min( self.knn, knn_neighbors.shape[1] )

        # argsort on ~first_k places True (selected) columns first
        order = ( ~first_k ).argsort( axis = 1, kind = 'stable' )
        col   = order[ :, :k_out ]
        row   = arange( N_pred_rows )[:, None]

        self.knn_neighbors = knn_neighbors[ row, col ]
        self.knn_distances = knn_distances[ row, col ]

    else :
        self.knn_neighbors = knn_neighbors
        self.knn_distances = knn_distances


#--------------------------------------------------------------------
# tieBreak helpers (Simplex only). Not used when self.tieBreak is False,
# so SMap / CCM / Multiview paths are byte-for-byte unchanged.
#--------------------------------------------------------------------
def _TieBreakSelect( self, knn_neighbors, knn_distances, mask, k_query,
                     exclusionRadius_knn ) :
    '''Vectorised deterministic knn selection by the ordering key
       (distance asc, |predRow - libRow| asc, libRow asc). A per-row
       full-library scan (_FullScanRow) completes only rows whose knn-th
       distance reaches the over-query boundary (a possible straddling
       tie) or that are deficient; the common case is fully vectorised.'''
    N, k   = knn_neighbors.shape
    knn    = self.knn
    lib_i  = array( self.lib_i )
    nLib   = len( lib_i )
    pred_i = array( self.pred_i )

    # Ordering keys over the (N_pred, k_query) candidate matrix
    prox     = npabs( pred_i[:, None] - knn_neighbors )   # |predRow - libRow|
    dist_key = where( mask, inf, knn_distances )          # excluded sort last
    rowk     = repeat( arange( N )[:, None], k, axis = 1 )

    # Per-row lexsort: primary row, then distance, proximity, library index
    flat = lexsort( ( knn_neighbors.ravel(), prox.ravel(),
                      dist_key.ravel(), rowk.ravel() ) ).reshape( N, k )
    sel  = flat[ :, :knn ]                                 # (N_pred, knn)
    rows = sel // k ; cols = sel % k
    out_nbr  = knn_neighbors[ rows, cols ]
    out_dist = knn_distances[ rows, cols ]
    sel_key  = dist_key[ rows, cols ]

    # Flag rows needing a full scan : boundary straddle or deficiency
    canComplete = k_query < nLib
    finite_ret  = where( isfinite( knn_distances ), knn_distances, -inf )
    maxRet      = finite_ret.max( axis = 1 )
    knnth       = sel_key[ :, knn - 1 ]
    deficient   = ~isfinite( knnth )
    straddle    = canComplete & isfinite( knnth ) & ( knnth >= maxRet )
    flagged     = where( straddle | deficient )[0]

    if flagged.size :
        embLib    = self.Embedding.iloc[ self.lib_i, : ].to_numpy()
        warnedDef = False
        for i in flagged :
            p = pred_i[ i ]
            nbr_c, dst_c = _FullScanRow( self, p, embLib, lib_i, knn,
                                         exclusionRadius_knn, False )
            if nbr_c is None or nbr_c.shape[0] < knn :
                # Deficiency : fall back to nearest knn ignoring exclusion
                nbr_c, dst_c = _FullScanRow( self, p, embLib, lib_i, knn,
                                             exclusionRadius_knn, True )
                warnedDef = True
            out_nbr[ i, : ]  = 0
            out_dist[ i, : ] = 0
            t = min( knn, 0 if nbr_c is None else nbr_c.shape[0] )
            if t :
                out_nbr[ i, :t ]  = nbr_c[ :t ]
                out_dist[ i, :t ] = dst_c[ :t ]
        if warnedDef :
            warn( f'{self.name}: FindNeighbors() : '
                  'Failed to find knn outside exclusionRadius '
                  f'{self.exclusionRadius} for some predictions. '
                  f'Consider reducing knn {self.knn}.' )

    return out_nbr, out_dist


def _FullScanRow( self, p, embLib, lib_i, knn, exclusionRadius_knn,
                  ignoreExclusion ) :
    '''Exact full-library scan for one prediction row p, ordered by the
       key (distance, |p - libRow|, libRow). Returns (neighbors, distances)
       or (None, None) when no library rows remain.'''
    predVec = self.Embedding.iloc[ p, : ].to_numpy()
    d       = sqrt( npsum( ( embLib - predVec ) ** 2, axis = 1 ) )

    if ignoreExclusion :
        keep = None
    elif exclusionRadius_knn :
        keep = npabs( p - lib_i ) > self.exclusionRadius  # subsumes self
    elif self.libOverlap :
        keep = lib_i != p                                 # self-match only
    else :
        keep = None

    if keep is None :
        nbrRow, dRow = lib_i, d
    else :
        nbrRow, dRow = lib_i[ keep ], d[ keep ]

    if nbrRow.shape[0] == 0 :
        return None, None

    order = lexsort( ( nbrRow, npabs( p - nbrRow ), dRow ) )
    sel   = order[ : knn ]
    return nbrRow[ sel ], dRow[ sel ]
