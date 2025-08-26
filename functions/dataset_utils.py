# Importing Libraires
import numpy as np
import pandas as pd
import math
import scipy
from PIL import Image
import random

import torch
import torchvision
from torchvision import transforms

import os,sys,warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import pathlib
import functions.utils as utils
import defaults


# Saving Images Paths for UnivFD dataset
def save_all_images_path_UnivFD(
	imgs_dir:str,
	status:str,
	save_path:str,
	replace:bool
):
	if os.path.exists(save_path) == False or replace == True:
		# Dataset
		dataset_images_paths = {}

		# For each UnivFD image-source for the given status
		for _,source in tqdm(enumerate(defaults.All_UnivFD_Sources[status])):
			dataset_images_paths[source] = {}

			# Initializing
			for label in ["fake", "real"]:
				dataset_images_paths[source][label] = []

			# Images Source Directory
			source_images_dir = os.path.join(imgs_dir, status, source)

			# For each label or category
			if ("0_real" in os.listdir(source_images_dir)) and ("1_fake" in os.listdir(source_images_dir)):

				# For each label 
				for _,label in enumerate(os.listdir(source_images_dir)):
					if label == "0_real":
						img_label = "real"
					elif label == "1_fake":
						img_label = "fake"
					else:
						assert False, "Unknown Label encountered."

					for fname in os.listdir(os.path.join(source_images_dir, label)):
						# Image Path and Label
						img_path = os.path.join(source_images_dir, label, fname)

						# Appending
						dataset_images_paths[source][img_label].append(os.path.relpath(img_path, defaults.main_dataset_dir))

			else:
				# For each category
				for _,category in enumerate(os.listdir(source_images_dir)):
					
					# For each label 
					for _,label in enumerate(os.listdir(os.path.join(source_images_dir, category))):
						if label == "0_real":
							img_label = "real"
						elif label == "1_fake":
							img_label = "fake"
						else:
							assert False, "Unknown Label encountered."

						for fname in os.listdir(os.path.join(source_images_dir, category, label)):
							# Image Path and Label
							img_path = os.path.join(source_images_dir, category, label, fname)

							# Appending
							dataset_images_paths[source][img_label].append(os.path.relpath(img_path, defaults.main_dataset_dir))
		
		# Saving
		np.save(save_path, dataset_images_paths)


# Saving Images Paths for GenImage dataset
def save_all_images_path_GenImage(
	imgs_dir:str,
	status:str,
	save_path:str,
	replace:bool
):
	if os.path.exists(save_path) == False or replace == True:
		# Dataset
		dataset_images_paths = {}

		# For each GenImage image-source for the given status
		for _,source in tqdm(enumerate(defaults.All_GenImage_Sources[status])):
			dataset_images_paths[source] = {}

			# Initializing
			for label in ["fake", "real"]:
				dataset_images_paths[source][label] = []

			# Images Source Directory
			source_images_dir = os.path.join(imgs_dir, source, status)

			# For each label
			for _,label in enumerate(os.listdir(source_images_dir)):
				if label == "nature":
					img_label = "real"
				elif label == "ai":
					img_label = "fake"
				elif (label == "ai_reconstructed_inpainting" or label == "nature_reconstructed_inpainting") and status == "train" and source == "sdv4":
					print ("Encountered label:{} for status:{} and source:{}".format(label, status, source))
					img_label = "fake"
				else:
					assert False, "Unknown Label encountered."

				for fname in os.listdir(os.path.join(source_images_dir, label)):
					# Image Path and Label
					img_path = os.path.join(source_images_dir, label, fname)

					# Appending
					dataset_images_paths[source][img_label].append(os.path.relpath(img_path, defaults.main_dataset_dir))
		
		# Saving
		np.save(save_path, dataset_images_paths)


# Saving Images Paths for DRCT dataset
def save_all_images_path_DRCT(
	imgs_dir:str,
	status:str,
	save_path:str,
	replace:bool
):
	if os.path.exists(save_path) == False or replace == True:
		# Dataset
		dataset_images_paths = {}

		# For Training Dataset: Real Images, Fake Images, Real Reconstructed Images and Fake Reconstructed Imagees
		if status == "train":
			# For each image-source
			for _,source in tqdm(enumerate(defaults.All_DRCT_Sources[status])):
				dataset_images_paths[source] = {}

				# Initializing
				for label in ["fake", "real"]:
					dataset_images_paths[source][label] = []

				# Images Source Directory
				real_images_dir = os.path.join(imgs_dir, "real_images", "{}2017".format(status))
				fake_images_dirs = [
					os.path.join(imgs_dir, "fake_images", source, "{}2017".format(status)),
					os.path.join(imgs_dir, "fake_reconstructed_images", source, "{}2017".format(status)),
					os.path.join(imgs_dir, "real_reconstructed_images", source, "{}2017".format(status)),
				]

				# Real Images Paths
				img_label = "real"
				for fname in os.listdir(real_images_dir):
					# Image Path and Label
					img_path = os.path.join(real_images_dir, fname)

					# Appending
					dataset_images_paths[source][img_label].append(os.path.relpath(img_path, defaults.main_dataset_dir))

				# Fake Images Paths
				img_label = "fake"
				for i in range(len(fake_images_dirs)):
					for fname in os.listdir(fake_images_dirs[i]):
						# Image Path and Label
						img_path = os.path.join(fake_images_dirs[i], fname)

						# Appending
						dataset_images_paths[source][img_label].append(os.path.relpath(img_path, defaults.main_dataset_dir))

		# For Validation Dataset: Real Images, Fake Images
		else:
			# For each image-source
			for _,source in tqdm(enumerate(defaults.All_DRCT_Sources[status])):
				dataset_images_paths[source] = {}

				# Initializing
				for label in ["fake", "real"]:
					dataset_images_paths[source][label] = []

				# Images Source Directory
				real_images_dir = os.path.join(imgs_dir, "real_images", "{}2017".format(status))
				fake_images_dir = os.path.join(imgs_dir, "fake_images", source, "{}2017".format(status))

				# Real Images Paths
				img_label = "real"
				for fname in os.listdir(real_images_dir):
					# Image Path and Label
					img_path = os.path.join(real_images_dir, fname)

					# Appending
					dataset_images_paths[source][img_label].append(os.path.relpath(img_path, defaults.main_dataset_dir))

				# Fake Images Paths
				img_label = "fake"
				for fname in os.listdir(fake_images_dir):
					# Image Path and Label
					img_path = os.path.join(fake_images_dir, fname)

					# Appending
					dataset_images_paths[source][img_label].append(os.path.relpath(img_path, defaults.main_dataset_dir))
		
		# Saving
		np.save(save_path, dataset_images_paths)


# Saving all paths of image dataset
def save_all_images_paths(
	imgs_dir:str,
	dataset_type:str,
	status:str,
	save_path:str,
	replace:bool
):
	"""
	Saves path info images of a dataset_type, status, image_sources.
	Args:
		imgs_dir (str): Directory of images.
		dataset_type (str): Type of Dataset. Options: ["UnivFD", "GenImage", "DRCT]
		status (str): ["train", "val"]
		save_path (str): Path to save .npy file.
		replace (bool): Replace File if True.
	"""
	# Assertions
	assert dataset_type in ["UnivFD", "GenImage", "DRCT"], "Invalid dataset"
	assert os.path.exists(imgs_dir), f"Image directory {imgs_dir} is not found."
	assert status in ["train", "val"], "Invalid status"

	if dataset_type == "UnivFD":
		save_all_images_path_UnivFD(
			imgs_dir=imgs_dir,
			status=status,
			save_path=save_path,
			replace=replace
		)
	elif dataset_type == "GenImage":
		save_all_images_path_GenImage(
			imgs_dir=imgs_dir,
			status=status,
			save_path=save_path,
			replace=replace
		)
	else:
		save_all_images_path_DRCT(
			imgs_dir=imgs_dir,
			status=status,
			save_path=save_path,
			replace=replace
		)


# Get Images Paths
def get_image_paths(
	dataset_type:str,
	status:str,
	image_sources:str,
	label:str,
):
	"""
	Get path to all images in the folder based on arguments.
	Args:
		dataset_type (str): Type of Dataset. Options: ["UnivFD", "GenImage", "DRCT]
		status (str): ["train", "val"]
		image_sources (list): Image-Sources to consider for dataset.
		label (str): ["real", "fake"]
	"""
	# Assertions
	assert dataset_type in ["UnivFD", "GenImage", "DRCT"], "Invalid dataset"
	assert status in ["train", "val"], "Invalid status"
	assert label in ["real", "fake"], "Invalid label"


	# Loading Paths
	img_dir = os.path.join(defaults.main_dataset_dir, dataset_type, "dataset")
	info_path = os.path.join(defaults.main_dataset_dir, "Info", "{}_{}_image_Paths.npy".format(dataset_type, status))


	# Saving Info File
	if os.path.exists(info_path) == False:
		print ("Saving Info File")

		save_all_images_paths(
			imgs_dir=img_dir,
			dataset_type=dataset_type,
			status=status,
			save_path=info_path,
			replace=False
		)
	

	# Loading Path Info
	Path_Info = np.load(info_path, allow_pickle=True)[()]
	

	# Dataset
	dataset_images_paths = []
	# For each image-source
	for _, source in enumerate(image_sources):
		for img_path in sorted(Path_Info[source][label]):
			# Image-Path
			dataset_images_paths.append(img_path)

	return dataset_images_paths



# Dataset Paths
def dataset_img_paths(
	dataset_type:str,
	status:str
):
	"""
	Returns real_image_paths and fake_image_paths based on arguments.
	Args:
		dataset_type (str): Type of Dataset. Options: ["UnivFD", "GenImage", "DTCT]
		status (str): ["train", "val"]
	"""
	# Assertions
	assert dataset_type in ["UnivFD", "GenImage", "DRCT"], "Invalid dataset"
	assert status in ["train", "val"], "Invalid status"

	# DRCT Dataset
	if dataset_type == "DRCT":
		train_image_sources, test_image_sources = utils.get_DRCT_options()

		if status == "train":
			image_sources = train_image_sources
		else:
			image_sources = test_image_sources

		real_images_paths = get_image_paths(
			dataset_type=dataset_type,
			status=status,
			image_sources=image_sources,
			label="real"
		)

		fake_images_paths = get_image_paths(
			dataset_type=dataset_type,
			status=status,
			image_sources=image_sources,
			label="fake"
		)

	# GenImage Dataset
	elif dataset_type == "GenImage":
		train_image_sources, test_image_sources = utils.get_GenImage_options()

		if status == "train":
			image_sources = train_image_sources
		else:
			image_sources = test_image_sources

		real_images_paths = get_image_paths(
			dataset_type=dataset_type,
			status=status,
			image_sources=image_sources,
			label="real"
		)

		fake_images_paths = get_image_paths(
			dataset_type=dataset_type,
			status=status,
			image_sources=image_sources,
			label="fake"
		)

	# UnivFD Dataset
	elif dataset_type == "UnivFD":
		train_image_sources, test_image_sources = utils.get_UnivFD_options()

		if status == "train":
			image_sources = train_image_sources
		else:
			image_sources = test_image_sources

		real_images_paths = get_image_paths(
			dataset_type=dataset_type,
			status=status,
			image_sources=image_sources,
			label="real"
		)

		fake_images_paths = get_image_paths(
			dataset_type=dataset_type,
			status=status,
			image_sources=image_sources,
			label="fake"
		)

	else:
		assert False, "Unknown dataset_type: {}".format(dataset_type)

	return real_images_paths, fake_images_paths