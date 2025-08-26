"""
Reference: https://github.com/beibuwandeluori/DRCT/blob/main/data/diffusion_rec.py
"""
# Importing Libraries
import numpy as np
from PIL import Image
import cv2

import torch
from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting
import albumentations as A

import os, sys, warnings
warnings.filterwarnings("ignore")
import random, argparse
import glob
from tqdm import tqdm
import gc


def create_crop_transforms(height=224, width=224):
	aug_list = [
		A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, value=0),
		A.CenterCrop(height=height, width=width)
	]
	return A.Compose(aug_list)


def set_seed(seed: int):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(seed)
	random.seed(seed)


def find_nearest_multiple(a, multiple=8):
	"""
	Find the multiple that is closest to a and is greater than a
	"""
	n = a // multiple
	remainder = a % multiple
	if remainder == 0:
		# If multiple divides a, then n is a multiple of multiple
		return a
	else:
		# Otherwise, we need to find a larger multiple of multiple
		return (n + 1) * multiple


def pad_image_to_size(image, target_width=224, target_height=224, fill_value=255):
	"""
	Pads the image to the target width and height, using the specified padding value (default is 255)
	"""
	height, width = image.shape[:2]

	if height < target_height:
		pad_height = target_height - height
		pad_top = pad_height // 2
		pad_bottom = pad_height - pad_top
	else:
		pad_top = pad_bottom = 0

	if width < target_width:
		pad_width = target_width - width
		pad_left = pad_width // 2
		pad_right = pad_width - pad_left
	else:
		pad_left = pad_right = 0

	padded_image = np.pad(
		image,
		((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
		mode="constant",
		constant_values=fill_value
	)

	return padded_image


def center_crop(image, crop_width, crop_height):
	height, width = image.shape[:2]

	# Calculate the starting and ending points of the cropping area
	if width > crop_width:
		start_x = (width - crop_width) // 2
		end_x = start_x + crop_width
	else:
		start_x, end_x = 0, width
	if height > crop_height:
		start_y = (height - crop_height) // 2
		end_y = start_y + crop_height
	else:
		start_y, end_y = 0, height

	# Using array slicing to implement center crop
	cropped_image = image[start_y:end_y, start_x:end_x]
	if cropped_image.shape[0] < crop_height or cropped_image.shape[1] < crop_width:
		cropped_image = pad_image_to_size(
			cropped_image,
			target_width=crop_width,
			target_height=crop_height,
			fill_value=255
		)

	return cropped_image


def stable_diffusion_inpainting(pipe, image, mask_image, prompt, steps=50, height=512, width=512, seed=2023, guidance_scale=7.5):
	set_seed(int(seed))
	image_pil = Image.fromarray(image)
	mask_image_pil = Image.fromarray(mask_image).convert("L")
	
	# image and mask_image should be PIL images.
	# The mask structure is white for inpainting and black for keeping as it is
	new_image = pipe(
		prompt=prompt,
		image=image_pil,
		mask_image=mask_image_pil,
		height=height, width=width,
		num_inference_steps=steps,
		guidance_scale=guidance_scale
	).images[0]

	return new_image


def read_image(image_path, max_size=512):
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	# Crop Image
	height, width = image.shape[:2]
	height = height if height < max_size else max_size
	width = width if width < max_size else max_size
	transform = create_crop_transforms(height=height, width=width)
	image = transform(image=image)["image"]
	
	# Handle multiples of 8
	original_shape = image.shape
	new_height = find_nearest_multiple(original_shape[0], multiple=8)
	new_width = find_nearest_multiple(original_shape[1], multiple=8)
	new_image = np.zeros(shape=(new_height, new_width, 3), dtype=image.dtype)
	new_image[:original_shape[0], :original_shape[1]] = image

	mask_image = np.zeros_like(image)

	del transform
	del image
	gc.collect()

	return new_image, mask_image, original_shape


def func(pipe, image_path, save_path, step=50, max_size=1024):
	image, mask_image, original_shape = read_image(image_path, max_size)
	new_image = stable_diffusion_inpainting(
		pipe,
		image,
		mask_image,
		prompt='',
		steps=step,
		height=image.shape[0], width=image.shape[1],
		seed=2023,
		guidance_scale=7.5
	)

	# Restore original size
	new_image = new_image.crop(box=(0, 0, original_shape[1], original_shape[0]))

	# Save results
	new_image.save(save_path)


def run_inference(gpu_id, state, label):
	# Device
	device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

	# Paths
	root = '/mnt/LIVELAB2/krishna/Perceptual-Classifiers'

	# Reconstruction Models
	sd_model_names = [
		"runwayml/stable-diffusion-inpainting",
		"stabilityai/stable-diffusion-2-inpainting",
		"diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
	]

	# Selecting SD Model
	index = 0

	# Loading SD MOdel
	sd_model_name = sd_model_names[index]
	if 'xl' in sd_model_name:
		pipe = AutoPipelineForInpainting.from_pretrained(
			sd_model_name,
			torch_dtype=torch.float16,
			variant="fp16",
			safety_checker=None,
			requires_safety_checker=False
		)
		pipe.enable_xformers_memory_efficient_attention()
		pipe = pipe.to(device)
	else:
		pipe = StableDiffusionInpaintPipeline.from_pretrained(
			sd_model_name,
			torch_dtype=torch.float16,
			safety_checker=None,
			requires_safety_checker=False
		)
		pipe.enable_xformers_memory_efficient_attention()
		pipe = pipe.to(device)
	
	print ("\n")
	print(f"Successfully loaded {sd_model_name}")
	print ("\n")

	# Create reconstructed images for the GenImage dataset
	step = 50
	phase = 'train'
	inpainting_dir = {0: 'inpainting', 1: 'inpainting2', 2: 'inpainting_xl'}[index]
	model_name = "sdv4"
	
	# Creating Directories and Path
	images_dir = f'{root}/GenImage/dataset/{model_name}/{phase}/{label}'
	save_dir = f'{root}/GenImage/dataset/{model_name}/{phase}/{label}_reconstructed_{inpainting_dir}'
	os.makedirs(save_dir, exist_ok=True)

	# List of All Images
	image_filenames = os.listdir(images_dir)
	image_filenames.sort()

	# Start State
	start_state = state
	image_filenames_filtered = image_filenames[start_state*40500 : (start_state+1)*40500]

	print ()
	print (gpu_id, label, start_state*40500, (start_state+1)*40500)
	print ()
	
	failed_num = 0
	for i, filename in tqdm(enumerate(image_filenames_filtered)):
		img_name = os.path.splitext(filename)[0]
		image_path = os.path.join(images_dir, filename)
		save_path = os.path.join(save_dir, img_name + '.png')
		
		if os.path.exists(save_path):
			continue
		
		try:
			func(pipe, image_path, save_path, step=step, max_size=1024)
		except Exception as e:
			failed_num += 1
			print(f'Failed to generate image in {image_path}. Error: {e}') 

	print(f'Failed to reconstruct {failed_num} images')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="A simple argparse example.")
	parser.add_argument('--gpu_id', type=str, help='GPU ID', required=True)
	parser.add_argument('--label', type=str, help='Label', required=True)
	parser.add_argument('--state', type=int, help='State', required=True)
	
	# Parsing arguments
	args = parser.parse_args()

	# -----------------------------------------------------------------
	# Flushing Output
	import functools
	print = functools.partial(print, flush=True)

	# Saving stdout
	sys.stdout = open('stdouts/{}_{}_{}.log'.format(args.gpu_id, args.label, args.state), 'w')

	# -----------------------------------------------------------------

	run_inference(
		gpu_id=args.gpu_id,
		state=args.state,
		label=args.label
	)