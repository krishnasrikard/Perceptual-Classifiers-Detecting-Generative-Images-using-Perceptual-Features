"""
Testing the performance deep learning models on images/features to predict whether an image is fake/synthetic or real/natural.
"""
# Importing Libraries
import numpy as np

import os, sys, warnings
warnings.filterwarnings("ignore")
from yaml import safe_load
import functions.networks as networks
import functions.module as module
import functions.utils as utils
import defaults


# Calling Main function
if __name__ == '__main__':
	# -----------------------------------------------------------------
	# Flushing Output
	import functools
	print = functools.partial(print, flush=True)

	# Saving stdout
	sys.stdout = open('results/{}_1.log'.format(os.path.basename(__file__)[:-3]), 'w')

	# -----------------------------------------------------------------

	# Parsing Argumens
	args = utils.parser_args()

	# Iterate
	train_test_dataset_types_list = [
		("GenImage", 'GenImage'), 
		("GenImage", "DRCT"), ("GenImage", "UnivFD"),
		("DRCT", "DRCT"),
		("DRCT", "UnivFD"), ("DRCT", 'GenImage')
	]


	# For each train and test datasets
	for train_dataset_type, test_dataset_type in train_test_dataset_types_list:
		# Save folder prefix
		if train_dataset_type == test_dataset_type:
			prefix = ""
		else:
			prefix = 'Cross_'


		# Pre-Process Settings based on dataset
		if train_dataset_type == test_dataset_type:
			if test_dataset_type == "UnivFD":
				preprocess_settings_list = [
					# Default
					({"probability": -1, "gaussian_blur_range": None, "jpeg_compression_qfs": None, "input_image_dimensions": (224,224), "resize": None}, "default"),
				]
			else:
				preprocess_settings_list = [
					# Default
					({"probability": -1, "gaussian_blur_range": None, "jpeg_compression_qfs": None, "input_image_dimensions": (224,224), "resize": None}, "default"),

					# Gaussian-Blur
					({"probability": 1, "gaussian_blur_range": [1,1], "jpeg_compression_qfs": None, "input_image_dimensions": (224,224), "resize": None}, "sigma=1"),
					({"probability": 1, "gaussian_blur_range": [2,2], "jpeg_compression_qfs": None, "input_image_dimensions": (224,224), "resize": None}, "sigma=2"),
					({"probability": 1, "gaussian_blur_range": [3,3], "jpeg_compression_qfs": None, "input_image_dimensions": (224,224), "resize": None}, "sigma=3"),
					({"probability": 1, "gaussian_blur_range": [4,4], "jpeg_compression_qfs": None, "input_image_dimensions": (224,224), "resize": None}, "sigma=4"),
					({"probability": 1, "gaussian_blur_range": [5,5], "jpeg_compression_qfs": None, "input_image_dimensions": (224,224), "resize": None}, "sigma=5"),

					# JPEG-Compression
					({"probability": 1, "gaussian_blur_range": None, "jpeg_compression_qfs": [90,90], "input_image_dimensions": (224,224), "resize": None}, "jpegQF=90"),
					({"probability": 1, "gaussian_blur_range": None, "jpeg_compression_qfs": [80,80], "input_image_dimensions": (224,224), "resize": None}, "jpegQF=80"),
					({"probability": 1, "gaussian_blur_range": None, "jpeg_compression_qfs": [70,70], "input_image_dimensions": (224,224), "resize": None}, "jpegQF=70"),
					({"probability": 1, "gaussian_blur_range": None, "jpeg_compression_qfs": [60,60], "input_image_dimensions": (224,224), "resize": None}, "jpegQF=60"),
					({"probability": 1, "gaussian_blur_range": None, "jpeg_compression_qfs": [50,50], "input_image_dimensions": (224,224), "resize": None}, "jpegQF=50"),
					({"probability": 1, "gaussian_blur_range": None, "jpeg_compression_qfs": [40,40], "input_image_dimensions": (224,224), "resize": None}, "jpegQF=40"),
					({"probability": 1, "gaussian_blur_range": None, "jpeg_compression_qfs": [30,30], "input_image_dimensions": (224,224), "resize": None}, "jpegQF=30"),
				]
		else:
			preprocess_settings_list = [
				# Default
				({"probability": -1, "gaussian_blur_range": None, "jpeg_compression_qfs": None, "input_image_dimensions": (224,224), "resize": None}, "default"),
			]

		
		# For each preprocess_settings
		for preprocess_settings, suffix in preprocess_settings_list:
			# Inference-Restriction-1: Config Files
			"""
			- Inference only on limited feature extractors for various kinds of image distortions
			"""
			# Config Filenames
			config_filenames = [
				"hyperiqa",
				"tres",
				"contrique",
				"reiqa",
				"arniqa"
			]

			# Iterating for each config_filename
			for config_filename in config_filenames:
				# Loading Config file
				dir_path = os.path.dirname(os.path.realpath(__file__))
				args.config = os.path.join(dir_path, "configs/{}.yaml".format(config_filename))
				with open(args.config, 'r') as f:
					config:dict = safe_load(f)


				# Inference-Restriction-2: Variants of Training: Removed
				"""
				- Inference only on limited feature extractors for basic list of image distortions
				"""
				checkpoint_directories = [
					"extensive/MarginContrastiveLoss_CrossEntropy"
				]


				# For each training variant
				for ckpt_dir in checkpoint_directories:
					# Changes: (resume_ckpt_path, checkpoint_dirname, checkpoint_filename, dataset_type)
					config["checkpoints"]["resume_dirname"] = os.path.join(train_dataset_type, ckpt_dir)
					config["checkpoints"]["resume_filename"] = "best_model.ckpt"
					config["checkpoints"]["checkpoint_dirname"] = ckpt_dir
					config["checkpoints"]["checkpoint_filename"] = "best_model.ckpt"
					config["dataset"]["dataset_type"] = test_dataset_type


					# Threshold for calculating metrics
					if test_dataset_type == 'UnivFD':
						best_threshold = None
					else:
						best_threshold = 0.5


					# Setting model_name and preprocess_type for Pre-processing
					preprocess_settings["model_name"] = config["dataset"]["model_name"]
					preprocess_settings["selected_transforms_name"] = "test"

					
					# Dataset-Type
					dataset_type = config["dataset"]["dataset_type"]

					# Model
					model_name = config["dataset"]["model_name"]
					f_model_name = config["dataset"]["f_model_name"]


					# Model
					feature_extractor = networks.get_model(model_name=config["dataset"]["model_name"], device="cuda:{}".format(config["trainer"]["devices"][0]))

					
					# Classifier
					classifier = networks.Classifier_Arch2(
						input_dim=config["classifier"]["input_dim"],
						hidden_layers=config["classifier"]["hidden_layers"]
					)

					# Log
					print (
						"\n",
						"Classifier:", "\n",
						classifier, "\n",
						"\n"
					)


					# Assertions
					for key in ["dataset_type", "model_name"]:
						assert key in config["dataset"], "{} not provided".format(key)

					
					# Image-Sources and Classes
					if config["dataset"]["dataset_type"] == "GenImage":
						# GenImage Dataset
						train_image_sources, test_image_sources = utils.get_GenImage_options()

					elif config["dataset"]["dataset_type"] == "UnivFD":
						# UnivFD Dataset
						train_image_sources, test_image_sources = utils.get_UnivFD_options()

					elif config["dataset"]["dataset_type"] == "DRCT":
						# DRCT Dataset
						train_image_sources, test_image_sources = utils.get_DRCT_options()

					else:
						assert False, "Invalid Dataset"
				
					
					# Log
					print (
						"\n",
						"Test-Settings:", "\n",
						" "*2, "dataset_type:", dataset_type, "\n",
						" "*2, "model_name:", model_name, "\n",
						" "*2, "f_model_name:", f_model_name, "\n",
						" "*2, "train_image_sources:", train_image_sources, "\n",
						" "*2, "test_image_sources:", test_image_sources, "\n",
						" "*2, "resume_dirname", config["checkpoints"]["resume_dirname"], "\n",
						" "*2, "best_threshold", best_threshold, "\n",
						"\n"
					)


					# Testing
					config["train_settings"]["train"] = False
					config["train_loss_fn"]["name"] = "CrossEntropy"
					config["val_loss_fn"]["name"] = "CrossEntropy"
					
					test_set_metrics, best_threshold = module.run(
						feature_extractor=feature_extractor, 
						classifier=classifier, 
						config=config, 
						train_image_sources=train_image_sources,
						test_image_sources=test_image_sources,
						preprocess_settings=preprocess_settings,
						best_threshold=best_threshold,
						verbose=False
					)


					# Saving Results
					utils.write_results_csv(
						test_set_metrics=test_set_metrics,
						test_image_sources=test_image_sources,
						f_model_name=f_model_name,
						save_path=os.path.join("results", "{}{}_{}_{}".format(prefix, ckpt_dir.replace("/", "_"), train_dataset_type, test_dataset_type), "{}.csv".format(suffix)),
					)
					print ("\n"*2)