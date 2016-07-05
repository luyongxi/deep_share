# Written by Yongxi Lu

import cv2
import os.path as osp
import os
import datasets

class Imdb(object):
	""" Image database."""
	
	def __init__(self, name):
		self._name = name
		self._classes = []
		self._image_list = []
		self._gtdb = {}
		self._data_path = osp.abspath(osp.join(datasets.ROOT_DIR, 'data'))
		# Options specific to a particular datasets
		self.config = {}

	@property
	def name(self):
		return self._name

	@property
	def num_classes(self):
		return len(self._classes)

	@property
	def gtdb(self):
		return self._gtdb

	@property
	def image_list(self):
		return self._image_list

	@property
	def num_images(self):
		return len(self._image_list)

	def image_size(self, i):
		""" (width, height) """
		assert (i>=0 and i<self.num_images), 'Index out of boundary: {}.'.format(i)

		size = cv2.imread(self.image_path_at(i)).shape
		return (size[1], size[0])

	def image_path_at(self, i):
		""" absolute path """
		return self._image_list[i]

	@property
	def cache_path(self):
		cache_path = osp.abspath(osp.join(datasets.ROOT_DIR, 'data', 'cache'))
		if not osp.exists(cache_path):
			os.makedirs(cache_path)
		return cache_path

	@property
	def data_path(self):
		return self._data_path

	def evaluate(self, res):
		raise NotImplementedError

	def print_info(self, i):
		raise NotImplementedError
