"""
Pre-Processing Functions
References: https://github.com/WisconsinAIVision/UniversalFakeDetect/blob/main/data/datasets.py
"""
# Importing Libraires
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage
from scipy.ndimage.filters import gaussian_filter
import cv2

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform


import os, sys, warnings
warnings.filterwarnings("ignore")
from io import BytesIO
import random



# Gaussian Blur Function 
def randomGaussianBlur_fn(img, gaussian_blur_range):
	# Selecting standard-deviation randomly
	assert len(gaussian_blur_range) == 1 or len(gaussian_blur_range) == 2, "Invalid length of gaussian_blur_range"
	if len(gaussian_blur_range) == 2:
		sigma = np.random.uniform(low=gaussian_blur_range[0], high=gaussian_blur_range[1])
	else:
		sigma = gaussian_blur_range[0]

	# Applying Gaussian-Blur
	# print ("Sigma:", sigma)
	img = np.array(img)
	gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
	gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
	gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

	# Returning Blurred image
	return Image.fromarray(img)



# JPEG Compression Function
def randomJPEGCompression_PIL_fn(img, jpeg_compression_qfs):
	# Selecting a QF randomly
	qf = int(np.random.choice(jpeg_compression_qfs))
	# print ("QF:", qf)

	# Compressing the image
	outputIoStream = BytesIO()
	img.save(outputIoStream, "JPEG", quality=qf, optimice=True)
	outputIoStream.seek(0)

	# Returning compressed image
	return Image.open(outputIoStream)



# JPEG Compression Function
def randomJPEGCompression_OpenCV_fn(img, jpeg_compression_qfs):
	# Selecting a QF randomly
	qf = int(np.random.choice(jpeg_compression_qfs))
	# print ("QF:", qf)

	# PIL to Array
	img_array = np.array(img)
	
	# Assertions
	assert (img_array.dtype.kind == np.dtype('uint8').kind), "Numpy array is not uint8"

	# Compressing the image
	img_cv2 = img_array[:,:,::-1]
	encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), qf]
	_, encoded_img = cv2.imencode('.jpg', img_cv2, encode_param)
	decoded_image = cv2.imdecode(encoded_img, 1)
	compressed_img_cv2 = decoded_image[:,:,::-1]

	# Assertions
	assert (compressed_img_cv2.dtype.kind == np.dtype('uint8').kind), "Numpy array is not uint8"

	# Array to PIL
	img = Image.fromarray(compressed_img_cv2)

	# Returning compressed image
	return img
	


# Random Blur or/and JPEG Function and Transform
def randomBlurJPEG_fn(img, probability, gaussian_blur_range, jpeg_compression_qfs):
	# Randomly introducing Blur
	if np.random.uniform(low=0, high=1) <= probability:
		# Gaussian Blur
		if gaussian_blur_range is not None:
			img = randomGaussianBlur_fn(img=img, gaussian_blur_range=gaussian_blur_range)

	# Randomly introducing Compression Artifacts
	if np.random.uniform(low=0, high=1) <= probability:
		if np.random.uniform(low=0, high=1) <= 0.5:
			# JPEG Compression using PIL
			if jpeg_compression_qfs is not None:
				# print ("PIL Compression:")
				img = randomJPEGCompression_PIL_fn(img=img, jpeg_compression_qfs=jpeg_compression_qfs)
		else:
			# JPEG Compression using OpenCV
			if jpeg_compression_qfs is not None:
				# print ("OpenCV Compression:")
				img = randomJPEGCompression_OpenCV_fn(img=img, jpeg_compression_qfs=jpeg_compression_qfs)
	return img


class randomBlurJPEG(transforms.Lambda):
	def __init__(self, lambd, probability, gaussian_blur_range, jpeg_compression_qfs):
		super().__init__(lambd)
		self.probability = probability
		self.gaussian_blur_range = gaussian_blur_range
		self.jpeg_compression_qfs = jpeg_compression_qfs

	def __call__(self, img):
		return self.lambd(img, self.probability, self.gaussian_blur_range, self.jpeg_compression_qfs)


class ConvertImageToPIL(ImageOnlyTransform):
	def __init__(self, always_apply=False, p=1.0):
		super(ConvertImageToPIL, self).__init__(always_apply, p)

	def apply(self, img, **params):
		return A.convert_image_to_pil(img)



# Get Preprocessing Function
def get_preprocessfn(
	model_name:str,
	selected_transforms_name:str,
	probability:float,
	gaussian_blur_range:list,
	jpeg_compression_qfs:list,
	input_image_dimensions:tuple,
	resize:any
):
	"""
	Creating a Preprocessing Function.

	Args:
		model_name (str): Feature Extraction Model.
		selected_transforms_name (str): Name of list of Transforms
		probability (float): The probability of applying Blur or JPEG artifacts.
		gaussian_blur_range (list): Uniform Sampling of standard-deivation of Gaussian Blur.
		jpeg_compression_qfs (list): Uniform Sampling of JPEG QFs for JPEG compression.
		input_image_dimensions (tuple): Dimensions (height, width) of Input Image via Center or Random Crop.
		resize (any): Dummy Argument.
	"""
	# Assertions


	# Log
	print (
		"\n",
		"Data-Augmentation-Settings:", "\n",
		" "*2, "model_name: {}".format(model_name), "\n",
		" "*2, "Resizing: {}".format(resize), "\n",
		" "*2, "selected_transforms_name: {}".format(selected_transforms_name), "\n",
		" "*2, "Probability: {}".format(probability), "\n",
		" "*2, "Gaussian-Blur: {}".format(gaussian_blur_range is not None), "\n",
		" "*2, "JPEG-Compression: {}".format(jpeg_compression_qfs is not None), "\n",
		" "*2, "Input Image Dimensions: {}".format(input_image_dimensions), "\n",
		"\n"
	)


	# List of Transforms
	image_preprocessing_transforms = []
	PIL_to_Tensor_transforms = []


	## Image Preprocessing Transforms
	# Gaussian Blur
	if gaussian_blur_range is not None and probability > 0:
		image_preprocessing_transforms.append(
			A.GaussianBlur(sigma_limit=gaussian_blur_range, p=probability)
		)


	# JPEG Compression
	if jpeg_compression_qfs is not None and probability > 0:
		image_preprocessing_transforms.append(
			A.ImageCompression(quality_range=jpeg_compression_qfs, p=probability)
		)

	# Center Crop
	image_preprocessing_transforms.extend([
		A.PadIfNeeded(min_height=input_image_dimensions[0], min_width=input_image_dimensions[1], border_mode=cv2.BORDER_CONSTANT, value=0),
		A.CenterCrop(height=input_image_dimensions[0], width=input_image_dimensions[1]),
		A.PadIfNeeded(min_height=input_image_dimensions[0], min_width=input_image_dimensions[1], border_mode=cv2.BORDER_CONSTANT, value=0)
	])



	# PIL Image to Tensor Transforms
	PIL_to_Tensor_transforms.append(
		transforms.ToTensor()
	)


	# Normalization
	if model_name == "drct-convnext-b" or model_name == "drct-clip-vit-l-14":
		PIL_to_Tensor_transforms.append(
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		)
	elif model_name == "clip-resnet50" or model_name == "clip-vit-l-14":
		PIL_to_Tensor_transforms.append(
			transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
		)
	else:
		assert False, "Invalid model_name"



	# Dual Scale
	if model_name == "clip-resnet50" or model_name == "clip-vit-l-14" or model_name == "drct-clip-vit-l-14" or model_name == "drct-convnext-b":
		dual_scale = False
	else:
		assert False, "Invalid model_name"


	# Returning
	if selected_transforms_name == "extensive":
		return (A.Compose(image_preprocessing_transforms), transforms.Compose(PIL_to_Tensor_transforms), ), dual_scale
	
	return (A.Compose(image_preprocessing_transforms), transforms.Compose(PIL_to_Tensor_transforms), ), dual_scale