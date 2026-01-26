from abc import ABC as AbstractBaseClass
from typing import Tuple, Union, Optional

import numpy as np
from scipy.spatial import KDTree, distance


class NeighborFinderBase(AbstractBaseClass):
	"""
	Interface for describing a class to find nearest neighbors
	"""

	def __init__(self, 
				 data: np.ndarray):
		"""
		Constructor
		:param data: data in the shape of [samples, dimensions]
		"""
		self.data = data

	def query(self, 
			  x: np.ndarray, 
			  k: int = 1) -> Tuple[Union[float, np.ndarray], Union[int, np.ndarray]]:
		"""
		Get nearest neighbors for k
		:param x:	data to query
		:param k: 	number of nearest neighbors to get
		:return: distance to each nearest neighbor and index for each neighbor
		"""
		raise NotImplementedError


class KDTreeNeighborFinder(NeighborFinderBase):
	"""
	Scipy KDTree neighbor finder. Used as in original EDM Implementation

	Note: If dimensionality is k, the number of points n in
	the data should be n >> 2^k, otherwise KDTree efficiency is low.
	k:2^k pairs { 4 : 16, 5 : 32, 7 : 128, 8 : 256, 10 : 1024 }
	"""

	def __init__(self, 
				 data: np.ndarray,
				 leafsize = 20, 
				 compact_nodes = True, 
				 copy_data = False, 
				 balanced_tree = True, 
				 boxsize = None):
		super().__init__(data)
		self.tree = KDTree(self.data, leafsize = leafsize, compact_nodes = compact_nodes, copy_data = copy_data,
						   balanced_tree = balanced_tree, boxsize = boxsize)

	def query(self, 
			  x: np.ndarray, 
			  k: int = 1, 
			  eps: float = 0.0,
			  distance_upper_bound = np.inf, 
			  p: float = 2.0, 
			  workers: int = 1) -> Tuple[Union[float, np.ndarray], Union[int, np.ndarray]]:
		return self.tree.query(x, k, eps, p, distance_upper_bound, workers)


class PairwiseDistanceNeighborFinder(NeighborFinderBase):
	"""
	Neighbor finder that uses pairwise euclidean distances and can be updated with new dimensions and re-queried
	"""

	@staticmethod
	def find_neighbors(distances: np.ndarray, k: int):
		neighbors = np.argsort(distances, axis = 0)[:k, :]
		indices = np.arange(distances.shape[1])[np.newaxis, :]
		neighbor_distances = distances[neighbors, indices]
		return np.sqrt(neighbor_distances).transpose().squeeze(), neighbors.transpose().squeeze()

	def __init__(self, 
				 data: np.ndarray, 
				 x: Optional[np.ndarray] = None):
		"""

		:param data: data
		:param x: 	data that we might want to query
		"""
		super().__init__(data)
		self.distanceMatrix = None
		if x is not None:
			self.distanceMatrix = distance.cdist(data, x, 'sqeuclidean')
		self.numNeighbors = None

	def requery(self):
		return self.query(None, self.numNeighbors)

	def update(self, additional_distance: np.ndarray) -> 'PairwiseDistanceNeighborFinder':
		"""
		Update the distance matrix and return a new neighborfinder
		:param additional_distance:
		:return:
		"""
		out = PairwiseDistanceNeighborFinder(None, None)
		out.distanceMatrix = self.distanceMatrix + additional_distance
		out.numNeighbors = self.numNeighbors
		return out

	def query(self, 
			  x: np.ndarray = None, 
			  k: int = 1, 
			  workers = 1) -> Tuple[
		Union[float, np.ndarray], Union[int, np.ndarray]]:
		self.numNeighbors = k
		if x is not None:
			self.distanceMatrix = distance.cdist(self.data, x, 'sqeuclidean')
		return PairwiseDistanceNeighborFinder.find_neighbors(self.distanceMatrix, self.numNeighbors)
