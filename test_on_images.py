"""
Testing the performance deep learning models on images/features to predict whether an image is fake/synthetic or real/natural.
"""
# Importing Libraries
import numpy as np

import torch
torch.set_float32_matmul_precision('medium')
import os, sys, warnings
warnings.filterwarnings("ignore")
from yaml import safe_load
from functions.loss_optimizers_metrics import *
from functions.run_on_images_fn import run_on_images
import functions.utils as utils
import functions.networks as networks
import defaults

dir_path = "/home/krishna/Perceptual-Classifiers-Working/images/True=Real_Pred=Fake"
test_real_images_files = os.listdir(dir_path)
test_real_images_paths = []
for f in test_real_images_files:
	test_real_images_paths.append(
		os.path.join(
			dir_path, f
		)
	)

test_fake_images_paths = []

# Calling Main function
if __name__ == '__main__':
	# -----------------------------------------------------------------
	# Flushing Output
	import functools
	print = functools.partial(print, flush=True)

	# Saving stdout
	sys.stdout = open('results/{}.log'.format(os.path.basename(__file__)[:-3]), 'w')

	# -----------------------------------------------------------------

	# Parsing Argumens
	args = utils.parser_args()

	# Iterate
	train_test_dataset_types_list = [("GenImage", "GenImage")]


	# For each train and test datasets
	for train_dataset_type, test_dataset_type in train_test_dataset_types_list:
		# Save folder prefix
		if train_dataset_type == test_dataset_type:
			prefix = ""
		else:
			prefix = 'Cross_'


		# Pre-Process Settings based on dataset
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
				# "hyperiqa",
				# "tres",
				"contrique",
				# "reiqa",
				# "arniqa"
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
					feature_extractor = networks.get_model(model_name=config["dataset"]["model_name"], device="cuda")

					
					# Classifier
					config["classifier"]["hidden_layers"] = [1024]
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
					
					test_set_metrics, best_threshold, y_pred, y_true = run_on_images(
						feature_extractor=feature_extractor, 
						classifier=classifier, 
						config=config, 
						test_real_images_paths=test_real_images_paths,
						test_fake_images_paths=test_fake_images_paths,
						preprocess_settings=preprocess_settings,
						best_threshold=best_threshold,
						verbose=False
					)

					print (y_pred)


					# Saving Predictions
					"""
					predictions = []
					for i in range(len(test_real_images_paths)):
						predictions.append(
							[test_real_images_paths[i], str(y_pred[i]), str(y_true[i])]
						)
					np.save("misc/predictions.npy", predictions)
					"""