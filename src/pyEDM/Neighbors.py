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

    self.check_lib_valid()

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
    raw_distances, raw_neighbors = self.kdTree.query(
        self.Embedding.iloc[ self.pred_i, : ].to_numpy(),
        k = self.knn_, eps = 0, p = 2, workers = numThreads )

    self.knn_distances, self.knn_neighbors = self.map_knn_indices_to_data(raw_distances, raw_neighbors)


