# python modules
from warnings import warn

# package modules
from numpy import array, delete, full, zeros, apply_along_axis
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
       for use in projections. If the the library is unitary this is
       a simple shift by lib_i[0]. If the library has disjoint segments
       or unordered indices, a mapping is needed from KDTree to lib_i.

       If there are degenerate lib & pred indices the first nn will
       be the prediction vector itself with distance 0. These are removed
       to implement "leave-one-out" prediction validation. In this case
       self.libOverlap is set True and the value of knn is increased 
       by 1 to return an additional nn. The first nn is relplaced by 
       shifting the j = 1:knn+1 knn columns into the j = 0:knn columns.

       If exlcusionRadius > 0, and, there are degenerate lib & pred
       indices, or, if there are not degnerate lib & pred but the 
       distance in rows between the lib & pred gap is less than
       exlcusionRadius, knn_neighbors have to be selected for each
       pred row to exclude library neighbors within exlcusionRadius.
       This is done by increasing knn to KDTree.query by a factor of 
       self.xRadKnnFactor, then selecting valid nn.

       Writes to EDM object:
         knn_distances : sorted knn distances
         knn_neighbors : library neighbor rows of knn_distances
    '''
    if self.verbose :
        print( f'{self.name}: FindNeighbors()' )

    N_lib_rows  = len( self.lib_i )
    N_pred_rows = len( self.pred_i )

    # Is knn_neighbors exclusionRadius radius adjustment needed?
    exclusionRadius_knn = False

    if self.exclusionRadius > 0 :
        if self.libOverlap :
            exclusionRadius_knn = True
        else :
            # If no libOverlap and exclusionRadius is less than the
            # distance in rows between lib : pred, no library neighbor
            # exclusion needed. 
            # Find row span between lib & pred
            excludeRow = 0
            if self.pred_i[0] > self.lib_i[-1] :
                # pred start is beyond lib end
                excludeRow = self.pred_i[0] - self.lib_i[-1]
            elif self.lib_i[0] > self.pred_i[-1] :
                # lib start row is beyond pred end
                excludeRow = self.lib_i[0] - self.pred_i[-1]
            if self.exclusionRadius >= excludeRow :
                exclusionRadius_knn = True

    # Local knn_
    knn_ = self.knn
    if self.libOverlap and not exclusionRadius_knn :
        # Increase knn +1 if libOverlap
        # Returns one more column in knn_distances, knn_neighbors
        # The first nn degenerate with the prediction vector
        # is replaced with the 2nd to knn+1 neighbors
        knn_ = knn_ + 1

    elif exclusionRadius_knn :
        # knn_neighbors exclusionRadius adjustment required
        # Ask for enough knn to discard exclusionRadius neighbors
        # This is controlled by the factor: self.xRadKnnFactor
        # JP : Perhaps easier to just compute all neighbors?
        knn_ = min( knn_ * self.xRadKnnFactor, len( self.lib_i ) )

    if len( self.validLib ) :
        # Have to examine all knn
        knn_ = len( self.lib_i )

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
    self.knn_distances, self.knn_neighbors = self.kdTree.query(
        self.Embedding.iloc[ self.pred_i, : ].to_numpy(),
        k = knn_, eps = 0, p = 2, workers = numThreads )

    #-----------------------------------------------
    # Shift knn_neighbors to lib_i reference
    #-----------------------------------------------
    # KDTree.query returns knn referenced to embedding.iloc[self.lib_i,:]
    # where returned knn_neighbors are indexed from 0 : len( lib_i ).
    # Generally, these are different from the knn that refer to prediction
    # library rows since generally lib != pred. Adjust knn from 0-offset 
    # returned by KDTree.query to EDM knn with respect to  embedding rows.
    #
    # If there is only one lib segment with contiguous values, a single
    # adjustment to knn_neighbors based on lib_i[0] suffices
    if not self.disjointLib and \
       self.lib_i[-1] - self.lib_i[0] == len(self.lib_i) + 1 :

        self.knn_neighbors = self.knn_neighbors + self.lib_i[0]

    else :
        # Disjoint library or CCM subset of lib_i.
        # Create mapping from KDTree neighbor indices to knn_neighbors
        knn_lib_map = {} # keys KDTree index : values lib_i index

        for i in range( len(self.lib_i) ) :
            knn_lib_map[ i ] = self.lib_i[ i ]

        #--------------------------------------------------------
        # Function to apply the knn_lib_map in apply_along_axis()
        #--------------------------------------------------------
        def knnMapFunc( knn, knn_lib_map ) :
            '''Function for apply_along_axis() on knn_neighbors.
               Maps the KDTree returned knn_neighbor indices to lib_i'''
            out = zeros( len( knn ), dtype = int )
            for i in range( len( knn ) ) :
                idx    = knn[i]
                out[i] = knn_lib_map[ idx ]
            return out

        # Apply the knn_lib_map to self.knn_neighbors
        # Use numpy apply_along_axis() to transform knn_neighbors from
        # KDTree indices to lib_i indices using the knn_lib_map
        knn_neighbors_ = zeros( self.knn_neighbors.shape, dtype = int )

        for j in range( self.knn_neighbors.shape[1] ) :
            knn_neighbors_[:,j] = \
                apply_along_axis( knnMapFunc, 0,
                                  self.knn_neighbors[:,j], knn_lib_map )

        self.knn_neighbors = knn_neighbors_

    if self.libOverlap :
        # Remove degenerate knn_distances, knn_neighbors
        # Get first column of knn_neighbors with knn_distance = 0
        knn_neighbors_0 = self.knn_neighbors[:,0]

        # If self.pred_i == knn_neighbors[:,0], point is degenerate,
        # distance = 0. Create boolean mask array of rows i_overlap
        # True where self.pred_i == knn_neighbors_0
        i_overlap = [ i == j for i,j in zip( self.pred_i,
                                             knn_neighbors_0 ) ]

        # Shift col = 1:knn_ values into col = 0:(J-1)
        # Use 0:(J-1) instead of 0:self.knn since knn_ may be large
        J = self.knn_distances.shape[1]
        self.knn_distances[ i_overlap, 0:(J-1) ] = \
            self.knn_distances[ i_overlap, 1:knn_ ]

        self.knn_neighbors[ i_overlap, 0:(J-1) ] = \
            self.knn_neighbors[ i_overlap, 1:knn_ ]

        # Delete extra knn_ column
        if not exclusionRadius_knn :
            self.knn_distances = delete(self.knn_distances, self.knn, axis=1)
            self.knn_neighbors = delete(self.knn_neighbors, self.knn, axis=1)

    if exclusionRadius_knn :
        # For each pred row find k nn outside exclusionRadius

        #-----------------------------------------------------------
        # Function to select knn from each row of self.knn_neighbors
        #-----------------------------------------------------------
        def ExclusionRad( knnRow, knnDist, excludeRow ) :
            '''Search excludeRow for each element of knnRow
               If knnRow is in excludeRow : exclude the neighbor
               Return knn length arrays of neighbors, distances'''

            knn_neighbors = full( self.knn, -1E6, dtype = int )
            knn_distances = full( self.knn, -1E6, dtype = float )

            k = 0
            for r in range( len( knnRow ) ) :
                if knnRow[ r ] in excludeRow :
                    # this nn is within exlcusionRadius of pred_i
                    continue

                knn_neighbors[ k ] = knnRow [ r ]
                knn_distances[ k ] = knnDist[ r ]
                k = k + 1

                if k == self.knn :
                    break

            if -1E6 in knn_neighbors :
                knn_neighbors = knnRow [ : self.knn ]
                knn_distances = knnDist[ : self.knn ]
                msg = f'{self.name}: FindNeighbors() : ExclusionRad()' +\
                    ' Failed to find knn outside exclusionRadius ' +\
                    f'{self.exclusionRadius}. Returning orginal knn.'
                warn( msg )

            return knn_neighbors, knn_distances

        # Call ExclusionRad() on each row
        for i in range( N_pred_rows ) :
            # Existing knn_neighbors, knn_distances row i with knn_ values
            knn_neighbors_i = self.knn_neighbors[ i, : ]
            knn_distances_i = self.knn_distances[ i, : ]

            # Create list excludeRow of lib_i nn to be excluded
            pred_i     = self.pred_i[ i ]
            rowLow     = max( 0, pred_i - self.exclusionRadius )
            rowHi      = min( N_lib_rows, pred_i + self.exclusionRadius )
            excludeRow = [ k for k in range( rowLow, rowHi + 1 ) ]

            knn_neighbors, knn_distances = \
                ExclusionRad( knn_neighbors_i, knn_distances_i, excludeRow )

            self.knn_neighbors[ i, range( self.knn ) ] = knn_neighbors
            self.knn_distances[ i, range( self.knn ) ] = knn_distances

        # Delete the extra knn_ columns
        d = [ i for i in range( self.knn, self.knn_distances.shape[1] ) ]
        self.knn_distances = delete( self.knn_distances, d, axis=1 )
        self.knn_neighbors = delete( self.knn_neighbors, d, axis=1 )

    if len( self.validLib ) :
        # Conditional embedding
        # Convert self.validLib boolean vector to lib_i indices
        data_i = array( [ i for i in range( self.Data.shape[0] ) ],
                        dtype = int )
        validLib_i = data_i[ self.validLib.to_numpy() ]

        #-----------------------------------------------------------
        # Function to select knn from each row of self.knn_distances
        #-----------------------------------------------------------
        def ValidLib( knnRow, knnDist, validLib_i ) :
            '''Search validLib for each element of knnRow
               If knnRow is in validLib : add the neighbor
               Return knn length arrays of neighbors, distances'''

            knn_neighbors = full( self.knn, -1E6, dtype = int )
            knn_distances = full( self.knn, -1E6, dtype = float )

            k = 0
            for r in range( len( knnRow ) ) :
                if knnRow[ r ] in validLib_i :
                    knn_neighbors[ k ] = knnRow [ r ]
                    knn_distances[ k ] = knnDist[ r ]
                    k = k + 1
                else :
                    continue

                if k == self.knn :
                    break

            if -1E6 in knn_neighbors :
                knn_neighbors = knnRow [ : self.knn ]
                knn_distances = knnDist[ : self.knn ]
                msg = f'{self.name}: FindNeighbors() : ValidLib()' +\
                    ' Failed to find knn in validLib. Return orginal knn.'
                warn( msg )

            return knn_neighbors, knn_distances

        # Call ValidLib() on each row
        for i in range( N_pred_rows ) :
            # Existing knn_neighbors, knn_distances row i with knn_ values
            knn_neighbors_i = self.knn_neighbors[ i, : ]
            knn_distances_i = self.knn_distances[ i, : ]

            knn_neighbors, knn_distances = \
                ValidLib( knn_neighbors_i, knn_distances_i, validLib_i )

            self.knn_neighbors[ i, range( self.knn ) ] = knn_neighbors
            self.knn_distances[ i, range( self.knn ) ] = knn_distances

        # Delete the extra knn_ columns
        d = [ i for i in range( self.knn, self.knn_distances.shape[1] ) ]
        self.knn_distances = delete( self.knn_distances, d, axis=1 )
        self.knn_neighbors = delete( self.knn_neighbors, d, axis=1 )
