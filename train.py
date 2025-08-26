"""
Training deep learning models on images/features to predict whether an image is fake/synthetic or real/natural.
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
	# Settings
	settings = [
		("GenImage", "MarginContrastiveLoss_CrossEntropy", "extensive"),
		("DRCT", "MarginContrastiveLoss_CrossEntropy", "extensive"),
	]

	# Config Files to Execute
	config_files = []

	for filename in os.listdir("configs"):
		if "hyperiqa" in filename:
			None
		else:
			continue
		config_files.append(os.path.join("configs", filename))

	for setting in settings:
		for config_file in config_files:
			with open(config_file, 'r') as f:
				config:dict = safe_load(f)

			
			# Updating Config File based on Settings
			config["dataset"]["dataset_type"] = setting[0]
			config["train_loss_fn"]["name"] = setting[1]
			config["val_loss_fn"]["name"] = setting[1]

			
			# -----------------------------------------------------------------
			# Flushing Output
			import functools
			print = functools.partial(print, flush=True)

			# Saving stdout
			sys.stdout = open('stdouts/{}_{}_{}_{}.log'.format(config["dataset"]["dataset_type"], setting[1], setting[2], config["dataset"]["f_model_name"]), 'w')

			# -----------------------------------------------------------------


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

			
			# Preprocess Settings
			preprocess_type = setting[2]

			preprocess_settings = {
				"model_name": model_name,
				"selected_transforms_name": preprocess_type,
				"probability": 0.5,
				"gaussian_blur_range": (0,3),
				"jpeg_compression_qfs": (30,100), 
				"input_image_dimensions": (224,224),
				"resize": None
			}


			# Checkpoints
			# Assertions
			assert (config["checkpoints"]["resume_dirname"] is None and config["checkpoints"]["resume_filename"] is None) or (config["checkpoints"]["resume_dirname"] is not None and config["checkpoints"]["resume_filename"] is not None), "Both resume_dirname and resume_filename should either be None or not None"
			
			# Loading pretraining if required

			# Defaults inputs for saving checkpoints
			if config["checkpoints"]["checkpoint_dirname"] is None:
				config["checkpoints"]["checkpoint_dirname"] = os.path.join(dataset_type, preprocess_type, config["train_loss_fn"]["name"])

			if os.path.exists(os.path.join(defaults.main_checkpoints_dir, config["checkpoints"]["checkpoint_dirname"], f_model_name, "best_model.ckpt")):
				if config["checkpoints"]["resume_dirname"] is None and config["checkpoints"]["resume_filename"] is None:
					config["checkpoints"]["resume_dirname"] = config["checkpoints"]["checkpoint_dirname"]
					config["checkpoints"]["resume_filename"] = "best_model.ckpt"


			# Log
			print (
				"\n",
				"Training-Settings:", "\n",
				" "*2, "dataset_type:", dataset_type, "\n",
				" "*2, "model_name:", model_name, "\n",
				" "*2, "f_model_name:", f_model_name, "\n",
				" "*2, "preprocess_type:", preprocess_type, "\n",
				" "*2, "train_loss_fn", config["train_loss_fn"]["name"], "\n",
				" "*2, "train_image_sources:", train_image_sources, "\n",
				" "*2, "test_image_sources:", test_image_sources, "\n",
				"\n"
			)


			# Training
			module.run(
				feature_extractor=feature_extractor,
				classifier=classifier,
				config=config, 
				train_image_sources=train_image_sources,
				test_image_sources=test_image_sources,
				preprocess_settings=preprocess_settings,
				best_threshold=None,
				verbose=False
			)