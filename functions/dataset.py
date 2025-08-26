# Importing Libraries
import numpy as np
import pandas as pd
import scipy
from PIL import Image
import random
from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms

import os,sys,warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import functions.dataset_utils as dataset_utils
import functions.utils as utils
import pathlib
import defaults


# PyTorch Images Dataset
class Image_Dataset(torch.utils.data.Dataset):
	def __init__(self,
		real_images_paths:list,
		fake_images_paths:list,
		preprocessfn:any,
		dual_scale:bool,
		resize:any,
		separateAugmentation:bool,
		ignore_reconstructed_images:bool=False,
	):
		"""
		Args:
			real_images_paths (str): List of all real-images to consider.
			fake_images_paths (str): List of all fake-images to consider.
			preprocessfn (any): Pre-Process Function.
			dual_scale (bool): Return Images on two scales.
			resize (any): If Tuple i.e (height, width), image will be resize to given dimensions. Else, if it's float, it will resize by a factor.
			seperateAugmentation: If True, then both original samples and data-augmented samples are returned.
			ignore_reconstructed_images (bool): If True, then ignore reconstructed images.
		"""
		# Data Transform
		self.dual_scale = dual_scale
		self.resize = resize
		self.seperateAugmentation = separateAugmentation
		self.image_preprocess_transforms = preprocessfn[0]
		self.PIL_to_Tensor_transforms = preprocessfn[1]

		# Center-Crop Transforms: We need Center-Crop transforms for seperateAugmentation because we are extracting both anchors and data-augmented anchors
		if dual_scale:
			self.center_crop_transform1 = transforms.CenterCrop((224,224))
			self.center_crop_transform2 = transforms.CenterCrop((112,112))
		

		# Dataset
		self.dataset_images_paths = []
		self.dataset_labels = []

		# For each real-image path
		for img_path in real_images_paths:
			# Image-Path
			self.dataset_images_paths.append(img_path)

			# Target
			self.dataset_labels.append([1,0])

		
		# Logging
		if ignore_reconstructed_images:
			print("Ignoring Reconstructed Images")
		
		
		# For each fake-image path
		for img_path in fake_images_paths:
			# Ignore Reconstructed Images
			if ignore_reconstructed_images:
				if img_path.__contains__("_reconstructed_"):
					continue

			# Image-Path
			self.dataset_images_paths.append(img_path)

			# Target
			self.dataset_labels.append([0,1])


		# Assertions
		assert len(self.dataset_images_paths) == len(self.dataset_labels), "No.of features-paths and labels are not equal."


	def __len__(self):
		return len(self.dataset_images_paths)


	def __getitem__(self, idx):
		# Assertions
		img_path = os.path.join(defaults.main_dataset_dir, self.dataset_images_paths[idx])

		# Target
		target = self.dataset_labels[idx]
		target = torch.LongTensor(target)

		# Returned data-augmented samples
		if self.seperateAugmentation == False:
			# Loading Image
			img = Image.open(img_path).convert('RGB')

			# Resize
			if self.resize is not None:
				if isinstance(self.resize, tuple):
					img = img.resize((self.resize[1], self.resize[0]))
				elif isinstance(self.resize, float):
					width, height = img.size[0], img.size[1]
					new_width = int(self.resize * width)
					new_height = int(self.resize * height)
					img = img.resize((new_width, new_height))
				else:
					assert False, "Unknown resize format"


			# Pre-processing
			if type(self.image_preprocess_transforms).__module__.__contains__("torchvision.transforms.transforms"):
				# Torch Transforms
				preprocessed_img = self.image_preprocess_transforms(img)
			elif type(self.image_preprocess_transforms).__module__.__contains__("albumentations.core.composition"):
				# Albumentations
				preprocessed_img = Image.fromarray(self.image_preprocess_transforms(image=np.array(img))["image"])
			else:
				assert False, "Unknown Pre-processing function"

			# Pre-processed image on original scale to Tensor
			img1 = self.PIL_to_Tensor_transforms(preprocessed_img)

			# Downsampled Scale: Half-Scale
			if self.dual_scale:
				# Downscaling
				preprocessed_img_dowsampled = preprocessed_img.resize(
					(preprocessed_img.size[0]//2, preprocessed_img.size[1]//2)
				)

				# Pre-processed image on downsampled scale to Tensor
				img2 = self.PIL_to_Tensor_transforms(preprocessed_img_dowsampled)

				return img1, img2, target
			
			return img1, target
		
		# Returning both original samples and data-augmented samples
		else:
			# Loading Image
			img = Image.open(img_path).convert('RGB')

			# Resize
			if self.resize is not None:
				if isinstance(self.resize, tuple):
					img = img.resize((self.resize[1], self.resize[0]))
				elif isinstance(self.resize, float):
					width, height = img.size[0], img.size[1]
					new_width = int(self.resize * width)
					new_height = int(self.resize * height)
					img = img.resize((new_width, new_height))
				else:
					assert False, "Unknown resize format"


			# Pre-processing
			if type(self.image_preprocess_transforms).__module__.__contains__("torchvision.transforms.transforms"):
				# Torch Transforms
				preprocessed_img = self.image_preprocess_transforms(img)
			elif type(self.image_preprocess_transforms).__module__.__contains__("albumentations.core.composition"):
				# Albumentations
				preprocessed_img = Image.fromarray(self.image_preprocess_transforms(image=np.array(img))["image"])
			else:
				assert False, "Unknown Pre-processing function"

			# Orignal and Pre-processed image on original scale to Tensor
			img1 = self.PIL_to_Tensor_transforms(self.center_crop_transform1(img))
			preprocessed_img1 = self.PIL_to_Tensor_transforms(preprocessed_img)

			# Downsampled Scale: Half-Scale
			if self.dual_scale:
				# Downscaling
				img_downsampled = img.resize(
					(img.size[0]//2, img.size[1]//2)
				)
				preprocessed_img_downsampled = preprocessed_img.resize(
					(preprocessed_img.size[0]//2, preprocessed_img.size[1]//2)
				)

				# Original and Pre-processed image on downsampled scale to Tensor
				img2 = self.PIL_to_Tensor_transforms(self.center_crop_transform2(img_downsampled))
				preprocessed_img2 = self.PIL_to_Tensor_transforms(preprocessed_img_downsampled)

				return img1, preprocessed_img1, img2, preprocessed_img2, target
			
			return img1, preprocessed_img1, target