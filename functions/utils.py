# Importing Libraries
import numpy as np

from pytorch_lightning.callbacks import TQDMProgressBar
import pytorch_lightning as pl

import os, sys, warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import getpass
import argparse, csv
import joblib
import contextlib


# Progress bar
class LitProgressBar(TQDMProgressBar):
	def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
		print ()
		return super().on_validation_epoch_end(trainer, pl_module)


# Allows to use tqdm when extracting features for multiple images in parallel.
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
	"""
	- Allows to use tqdm when extracting features for multiple images in parallel.
	- Useful when multi-threading on CPUs with joblib.
	- Context manager to patch joblib to report into tqdm progress bar given as argument
	"""
	class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
		def __call__(self, *args, **kwargs):
			tqdm_object.update(n=self.batch_size)
			return super().__call__(*args, **kwargs)

	old_batch_callback = joblib.parallel.BatchCompletionCallBack
	joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
	try:
		yield tqdm_object
	finally:
		joblib.parallel.BatchCompletionCallBack = old_batch_callback
		tqdm_object.close()
	

# Parsing Arguments for config files
def parser_args():
	"""
	Parsing Arguments for config files i.e .yaml files
	"""
	def nullable_str(s):
		if s.lower() in ['null', 'none', '']:
			return None
		return s
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', '-c', type=nullable_str, help='config file path')
	return parser.parse_args()


# Options for GenImage dataset
def get_GenImage_options():
	"""
	Get Image Sources
	"""
	# Image Sources
	test_image_sources = ["midjourney", "sdv4", "sdv5", "adm", "glide", "wukong", "vqdm", "biggan"]
	train_image_sources = ["sdv4"]

	return train_image_sources, test_image_sources


# Options for UnivFD dataset
def get_UnivFD_options():
	"""
	Get Image Sources
	"""
	# Image Sources
	test_image_sources = ["progan", "cyclegan", "biggan", "stylegan", "gaugan", "stargan", "deepfake", "seeingdark", "san", "crn", "imle", "guided", "ldm_200", "ldm_200_cfg", "ldm_100", "glide_100_27", "glide_50_27", "glide_100_10", "dalle"]
	train_image_sources = ["progan"]

	return train_image_sources, test_image_sources


# Options for DRCT dataset
def get_DRCT_options():
	"""
	Get Image Sources
	"""
	# Image Sources
	test_image_sources = [
		'ldm-text2im-large-256', 'stable-diffusion-v1-4', 'stable-diffusion-v1-5', 'stable-diffusion-2-1', 'stable-diffusion-xl-base-1.0', 'stable-diffusion-xl-refiner-1.0',
		'sd-turbo', 'sdxl-turbo', 
		'lcm-lora-sdv1-5', 'lcm-lora-sdxl',
		'sd-controlnet-canny', 'sd21-controlnet-canny', 'controlnet-canny-sdxl-1.0',
		'stable-diffusion-inpainting', 'stable-diffusion-2-inpainting', 'stable-diffusion-xl-1.0-inpainting-0.1']
	train_image_sources = ["stable-diffusion-v1-4"]

	return train_image_sources, test_image_sources


# Write Results in a .csv file
def write_results_csv(test_set_metrics, test_image_sources, f_model_name, save_path):
	# Assertions
	assert len(test_set_metrics) == len(test_image_sources), "len(test_set_metrics) and len(test_image_sources), does not match."

	# Create a .csv file if it doesn't exist
	if os.path.exists(save_path) == False:
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		with open(save_path, mode='w') as filename:
			writer = csv.writer(filename, delimiter=',',  quotechar='"', quoting=csv.QUOTE_MINIMAL)
			writer.writerow(["model", "(mAP, mAcc, mAcc_Real, mAcc_fake, mcc)"] + test_image_sources)


	# Results
	Append_List = [f_model_name, (0,0,0,0,0)]

	metrics = []
	for i,_ in enumerate(test_image_sources):
		O = test_set_metrics[i][1:]
		ap = np.round(O[0]*100, decimals=2)
		acc = np.round(O[4]*100, decimals=2)
		r_acc = np.round(O[5]*100, decimals=2)
		f_acc = np.round(O[6]*100, decimals=2)
		mcc = np.round(O[8], decimals=2)

		Append_List.append(tuple([ap, acc, r_acc, f_acc, mcc]))
		metrics.append([ap, acc, r_acc, f_acc, mcc])

	metrics = np.round(np.mean(metrics, axis=0), decimals=2)
	Append_List[1] = tuple(metrics)
	

	# Appending
	assert len(Append_List) == len(test_image_sources) + 2, "len(Append_List) != len(test_image_sources) + 2"
	with open(save_path, mode='a') as filename:
		writer = csv.writer(filename, delimiter=',',  quotechar='"', quoting=csv.QUOTE_MINIMAL)
		writer.writerow(Append_List)