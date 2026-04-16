# python modules
from warnings import warn

# package modules
from numpy import array, arange, zeros
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

       Writes to EDM object:
         knn_distances : sorted knn distances
         knn_neighbors : library neighbor rows of knn_distances
    '''
    if self.verbose :
        print( f'{self.name}: FindNeighbors()' )

    N_pred_rows = len( self.pred_i )

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
        validLib_i = data_i[ self.validLib.to_numpy() ]

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

    #-----------------------------------------------
    # Compute KDTree on library of embedding vectors
    #-----------------------------------------------
    self.kdTree = KDTree( self.Embedding.iloc[ self.lib_i, : ].to_numpy(),
                          leafsize      = 20,
                          compact_nodes = True,
                          balanced_tree = True )

    #-----------------------------------------------
    # Query prediction set
    #-----------------------------------------------
    numThreads = -1 # Use all CPU threads in kdTree.query
    knn_distances, knn_neighbors = self.kdTree.query(
        self.Embedding.iloc[ self.pred_i, : ].to_numpy(),
        k = k_query, eps = 0, p = 2, workers = numThreads )

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
                      or k_query > self.knn

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
