from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import numpy as np
import torch


class ImagesDataset(Dataset):

	def __init__(self, source_root, opts,segm_path, segm_transform, source_transform=None):
		self.source_paths = data_utils.make_dataset(source_root)
		#self.target_paths = data_utils.make_dataset(target_root)
		self.source_transform = source_transform
		#self.target_transform = target_transform
		self.segm_transform = segm_transform
		self.segm_path = data_utils.make_dataset(segm_path)
		self.opts = opts

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		segm_path = self.segm_path[index]
		with open(segm_path, 'rb') as f:
			from_segm = Image.open(f)
			from_segm.load()
		from_segm = self.segm_transform(from_segm)
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')
		from_segm = np.array(from_segm)
		from_segm = torch.from_numpy(from_segm)

		#to_path = self.target_paths[index]
		#to_im = Image.open(to_path).convert('RGB')
		#if self.target_transform:
			#to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		#else:
			#from_im = to_im

		return from_im, from_segm
