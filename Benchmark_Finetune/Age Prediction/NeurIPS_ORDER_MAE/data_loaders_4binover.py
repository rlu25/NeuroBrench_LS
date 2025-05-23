from glob import glob
from torch.utils import data
from monai import transforms as T
import nibabel as nib
import numpy as np
import os
import random, csv
import torch
import pandas as pd

class DataFolder(data.Dataset):
	def __init__(self, image_dir, image_type, transform, mode='train'):
		# Dictionary to map image types to their respective loading functions
		self.__image_reader = {
			'np': lambda url: np.load(url),
			'nii.gz': lambda url: nib.load(url).get_fdata()
		}

		# Supported image extensions
		self.__supported_extensions = self.__image_reader.keys()
		assert image_type in self.__supported_extensions
		assert transform is not None

		self.image_dir = image_dir
		self.image_type = image_type
		self.transform = transform
		self.mode = mode
		self.data_urls = []
		self.data_labels = []
		self.data_index = []
		self.num_classes = 0

		# Process the dataset
		self.__process()

		print("Total data loaded =", len(self.data_urls))

	def __process(self):
		# Get the list of classes (directories) in the image directory
		classes = sorted([c.split("/")[-1] for c in glob(os.path.join(self.image_dir, '*'))])
		self.num_classes = len(classes)
		print("Classes: ", self.num_classes)

		# Read the master data CSV file
		df = pd.read_csv("/opt/localdata/data/usr-envs/wenyi/ORDER/masterdata.csv") 
		age = df['age'].tolist()

		# Create histogram of ages
		# hist, bin_edges = np.histogram(age, bins=[
		# 	4, 8, 12, 16, 20, 22])

		hist, bin_edges = np.histogram(age, bins=[
			0, 4, 8, 12, 16, 20, 24, 28, 
			32, 36, 40, 44, 48, 
			52, 56, 60, 64, 68, 
			72, 76, 80, 84, 88, 92, 96])

		# Calculate the sample multiplier for each bin
		maxval = np.max(hist)
		
		multp = [int(np.floor(maxval / freq)) if freq > 0 else 0 for freq in hist]
		sample_mul = {}

		# Assign sample multipliers to each class
		for i, c in enumerate(classes):
			for j in range(1, len(bin_edges)):
				if int(c) >= bin_edges[j-1] and int(c) < bin_edges[j]:
					if self.mode == "train":
						sample_mul[c] = multp[j-1]
					else:
						sample_mul[c] = 1

		print("Sample multiplier:", sample_mul)

		# Collect data URLs and labels
		for i, c in enumerate(classes):
			temp = glob(os.path.join(self.image_dir, c, f'*.{self.image_type}'))
			self.data_urls += temp * sample_mul[c]
			self.data_labels += [i] * len(temp) * sample_mul[c]

		self.data_urls.sort()

		# Create data index and shuffle if in training mode
		self.data_index = list(range(len(self)))
		if self.mode in ['train']:
			random.seed(3141)
			random.shuffle(self.data_index)

		assert len(self) > 0

	def __read(self, url):
		# Read the image from the URL using the appropriate reader
		return self.__image_reader[self.image_type](url)

	def __getitem__(self, index):
		# Get the image and label at the specified index
		img = self.__read(self.data_urls[self.data_index[index]])
		lbl = self.data_labels[self.data_index[index]]

		# Apply transforms to the image
		img = np.expand_dims(img, 0)
		img = self.transform(img)
		img -= np.min(img)
		img /= np.max(img)
		return torch.FloatTensor(img), torch.LongTensor([lbl])

	def __len__(self):
		# Return the total number of data samples
		return len(self.data_urls)


def get_loader(image_dir, crop_size=101, image_size=101, 
			   batch_size=5, dataset='adni', mode='train', num_workers=16):
	"""Build and return a data loader."""
	transform = []

	# Add data augmentation transforms if in training mode
	if mode == 'train':
		transform.append(T.RandGaussianNoise())
		transform.append(T.RandBiasField())
		transform.append(T.RandScaleIntensity(0.25))
		transform.append(T.RandAdjustContrast())
		transform.append(T.RandGibbsNoise())
		transform.append(T.RandKSpaceSpikeNoise())
		transform.append(T.RandRotate())
		transform.append(T.RandFlip())

	# Convert the image to a tensor
	transform.append(T.ToTensor(dtype=torch.float))
	transform = T.Compose(transform)

	# Create the dataset
	dataset = DataFolder(image_dir, 'nii.gz', transform, mode)

	# Create the data loader
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  drop_last=True,
								  num_workers=num_workers)
	return data_loader


if __name__ == '__main__':
	# Create the data loader
	loader = get_loader('./data/adni/train', mode="train")
	# Iterate through the data loader and print the shape of the images and labels
	for i, x in enumerate(loader):
		print(i, x[0].shape, x[1], torch.min(x[0]), torch.max(x[0]))
		break