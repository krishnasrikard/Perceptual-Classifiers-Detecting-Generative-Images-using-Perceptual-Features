"""
Testing the performance deep learning models on images/features to predict whether an image is fake/synthetic or real/natural.
"""
# Importing Libraries
import numpy as np

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
torch.set_float32_matmul_precision('medium')
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirnae(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from yaml import safe_load
from functions.dataset import Image_Dataset
from functions.loss_optimizers_metrics import *
import functions.utils as utils
import prior_methods.prior_functions.prior_preprocess as prior_preprocess
import prior_methods.prior_functions.prior_module as prior_module
import defaults

# Main Function
def run_on_images(feature_extractor, classifier, config, test_real_images_paths, test_fake_images_paths, preprocess_settings, best_threshold, verbose=True):

	# Parameters
	dataset_type = config["dataset"]["dataset_type"]
	model_name = config["dataset"]["model_name"]
	f_model_name = config["dataset"]["f_model_name"]


	# Paths
	main_dataset_dir = defaults.main_dataset_dir


	# Checkpoints Paths
	if config["checkpoints"]["resume_ckpt_path"] is None:
		resume_ckpt_path = os.path.join(main_dataset_dir, dataset_type, config["checkpoints"]["checkpoint_foldername"], f_model_name, "best_model.ckpt")
	else:
		resume_ckpt_path = config["checkpoints"]["resume_ckpt_path"]

	if config["checkpoints"]["checkpoint_dirpath"] is None:
		checkpoint_dirpath = os.path.join(main_dataset_dir, dataset_type, config["checkpoints"]["checkpoint_foldername"], f_model_name)
	else:
		checkpoint_dirpath = config["checkpoints"]["checkpoint_dirpath"]


	# Global Variables: (feature_extractor)
	global feature_extractor_module
	feature_extractor_module = feature_extractor
	feature_extractor_module.to("cuda")
	feature_extractor_module.eval()
	for params in feature_extractor_module.parameters():
		params.requires_grad = False


	global classifier_module
	classifier_module = classifier
	classifier_module.to("cuda")
	classifier_module.eval()
	for params in classifier_module.parameters():
		classifier_module.requires_grad = False

	# Lightning Module
	Model = prior_module.Model_LightningModule(config)


	# Checkpoint Callbacks
	best_checkpoint_callback = ModelCheckpoint(
		dirpath=checkpoint_dirpath,
		filename="best_model",
		monitor=config["train_settings"]["monitor"],
		mode=config["train_settings"]["mode"]
	)


	# Resuming from checkpoint
	# For Evaluating
	if resume_ckpt_path is not None:
		if os.path.isfile(resume_ckpt_path):
			print ("Found the checkpoint at resume checkpoint (resume_ckpt_path) provided.")
		else:
			assert False, f"File not found at {resume_ckpt_path} for evaluation."
	else:
		assert False, "No path is provided for resume checkpoint (resume_ckpt_path) provided. resume_ckpt_path is required for evaluation."



	# PyTorch Lightning Trainer
	trainer = pl.Trainer(
		**config["trainer"],
		callbacks=[best_checkpoint_callback, utils.LitProgressBar()],
		precision=32
	)
	


	# Pre-processing Functions
	preprocessfn, dual_scale = prior_preprocess.get_preprocessfn(**preprocess_settings)


	# Datasets	
	# Images Test Dataset
	if config["train_settings"]["train"] == False:
		# For images smaller than preprocess_settings["input_image_dimensions"] which only occur for BigGAN fake images in GenImage dataset, we do the following:
		"""
		- During inference, we avoid Resizing to reduce the effect of resizing artifacts.
		- We process the images at (224,224) or their smaller resolution unless the feature extraction model requires (224,224) inputs.
		"""

		if model_name == "clip-resnet50" or model_name == "clip-vit-l-14" or model_name == "drct-clip-vit-l-14" or model_name == "drct-convnext-b":
			# Updated Pre-Processing Settings
			Fixed_Input_preprocess_settings = preprocess_settings.copy()
			Fixed_Input_preprocess_settings["input_image_dimensions"] = (224,224)

			# Preprocessing Function
			Fixed_Input_preprocessfn, Fixed_Input_dual_scale = prior_preprocess.get_preprocessfn(**Fixed_Input_preprocess_settings)

			Test_Dataset = Image_Dataset(
				real_images_paths=test_real_images_paths,
				fake_images_paths=test_fake_images_paths,
				preprocessfn=Fixed_Input_preprocessfn,
				dual_scale=Fixed_Input_dual_scale
			)

		else:
			Test_Dataset = Image_Dataset(
				real_images_paths=test_real_images_paths,
				fake_images_paths=test_fake_images_paths,
				preprocessfn=preprocessfn,
				dual_scale=dual_scale
			)


	# DataLoaders
	if config["train_settings"]["train"] == False:
		# Test DataLoaders
		Test_Dataloader = DataLoader(
			dataset=Test_Dataset,
			batch_size=config["train_settings"]["batch_size"],
			num_workers=config["train_settings"]["num_workers"],
			shuffle=False,
		)


	print ("-"*25 + " Datasets and DataLoaders Ready " + "-"*25)
	

	# Evaluating
	# Predictions on Test Dataset
	test_y_pred_y_true = trainer.predict(
		model=Model,
		dataloaders=Test_Dataloader,
		ckpt_path=resume_ckpt_path
	)
	
	test_set_metrics = []
	y_pred, y_true = concatenate_predictions(y_pred_y_true=test_y_pred_y_true)

	y_pred = y_pred[:, 1]
	y_true = np.argmax(y_true, axis=1) 

	ap, acc0, r_acc0, f_acc0, acc1, r_acc1, f_acc1, mcc0, mcc1, _ = calculate_metrics(y_pred=y_pred, y_true=y_true, threshold=best_threshold)
	test_set_metrics.append([0, ap, acc0, r_acc0, f_acc0, acc1, r_acc1, f_acc1, mcc0, mcc1])

	return test_set_metrics, best_threshold