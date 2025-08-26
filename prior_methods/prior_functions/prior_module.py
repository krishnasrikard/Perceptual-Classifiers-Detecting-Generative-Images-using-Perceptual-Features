"""
PyTorch Lightning Module of training of deep-learning models

Notes:
- Using ".to(torch.float32)" to resolving precision issues while using different models.
"""

# Importing Libraries
import numpy as np
from sklearn.model_selection import train_test_split

import torch
torch.set_float32_matmul_precision('medium')
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import random
from functions.dataset import Image_Dataset
import functions.dataset_utils as dataset_utils
from functions.loss_optimizers_metrics import *
import functions.utils as utils
import prior_methods.prior_functions.prior_preprocess as prior_preprocess
import defaults


# Lightning Module
class Model_LightningModule(pl.LightningModule):
	def __init__(self, config):
		super().__init__()
		self.save_hyperparameters()
		self.config = config

		# Loss
		self.train_lossfn = get_loss_function(**self.config["train_loss_fn"])
		self.val_lossfn = get_loss_function(**self.config["val_loss_fn"])

		# Metrics
		self.train_accuracy_fn = torchmetrics.Accuracy(task="binary")
		self.val_accuracy_fn = torchmetrics.Accuracy(task="binary")


	# Training-Step
	def training_step(self, batch, batch_idx):
		with torch.no_grad():
			X, y_true = batch
			X = feature_extractor_module(X)
			y_true_classes = torch.argmax(y_true, dim=1)

			y_pred = classifier_module(X)
			if y_pred.shape[1] == 1:
				y_pred_classes = y_pred >= 0.5
			else:
				y_pred_classes = torch.argmax(y_pred, dim=1)

		self.train_loss = self.train_lossfn(y_pred, y_true_classes)
		self.train_acc = self.train_accuracy_fn(y_pred_classes, y_true_classes)

		self.log_dict(
			{
				"train_loss": self.train_loss,
				"train_acc": self.train_acc
			}, 
			on_step=True, on_epoch=False, prog_bar=True, sync_dist=True
		)

		return self.train_loss


	# Validation-Step
	def validation_step(self, batch, batch_idx, dataloader_idx=0):
		with torch.no_grad():
			X, y_true = batch
			X = feature_extractor_module(X)
			y_true_classes = torch.argmax(y_true, dim=1)

			y_pred = classifier_module(X)
			if y_pred.shape[1] == 1:
				y_pred_classes = y_pred >= 0.5
			else:
				y_pred_classes = torch.argmax(y_pred, dim=1)

		self.val_loss = self.val_lossfn(y_pred, y_true_classes)
		self.val_acc = self.val_accuracy_fn(y_pred_classes, y_true_classes)

		self.log_dict(
			{
				"val_loss": self.val_loss,
				"val_acc": self.val_acc,
			},
			on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
		)


	# Prediction-Step
	def predict_step(self, batch, batch_idx, dataloader_idx=0):
		with torch.no_grad():
			X, y_true = batch
			X = feature_extractor_module(X)
			y_true_classes = torch.argmax(y_true, dim=1)

			y_pred = classifier_module(X)
			if y_pred.shape[1] == 1:
				y_pred_classes = y_pred >= 0.5
			else:
				y_pred_classes = torch.argmax(y_pred, dim=1)

		return y_pred, y_true


	# Configure Optimizers
	def configure_optimizers(self):
		optimizer = get_optimizer(
			None
			**self.config["optimizer"]
		)
		
		return [optimizer]



# Main Function
def run(feature_extractor, classifier, config, train_image_sources, test_image_sources, preprocess_settings, best_threshold, verbose=True):

	# Parameters
	dataset_type = config["dataset"]["dataset_type"]
	separateAugmentation = config["dataset"]["separateAugmentation"]
	model_name = config["dataset"]["model_name"]
	f_model_name = config["dataset"]["f_model_name"]


	# Paths
	main_dataset_dir = defaults.main_dataset_dir
	main_checkpoints_dir = defaults.main_prior_checkpoints_dir


	# Checkpoints Paths
	# Resume Checkpoints
	if config["checkpoints"]["resume_dirname"] is not None and config["checkpoints"]["resume_filename"] is not None:
		resume_ckpt_path = os.path.join(main_checkpoints_dir, config["checkpoints"]["resume_dirname"], f_model_name, config["checkpoints"]["resume_filename"])
	else:
		resume_ckpt_path = None

	print (resume_ckpt_path)

	# Save Checkpoints
	checkpoint_dirpath = os.path.join(main_checkpoints_dir, dataset_type, config["checkpoints"]["checkpoint_dirname"], f_model_name)
	os.makedirs(checkpoint_dirpath, exist_ok=True)


	# Resuming from checkpoint
	if resume_ckpt_path is not None:
		if os.path.exists(resume_ckpt_path):
			print ("Found the checkpoint at resume_ckpt_path provided.")
		else:
			assert False, "Resume checkpoint not found at resume_ckpt_path provided."
	else:
		if config["train_settings"]["train"]:
			# For Training.
			print ("No path is provided for resume checkpoint (resume_ckpt_path) provided. Starting training from the begining.")
		else:
			assert False, "No path is provided for resume checkpoint (resume_ckpt_path) provided. resume_ckpt_path is required for evaluation."


	# Checkpoint Callbacks
	best_checkpoint_callback = ModelCheckpoint(
		dirpath=checkpoint_dirpath,
		filename="best_model",
		monitor=config["train_settings"]["monitor"],
		mode=config["train_settings"]["mode"]
	)


	# Pre-processing Functions
	preprocessfn, dual_scale = prior_preprocess.get_preprocessfn(**preprocess_settings)

	# Logging
	print ()
	print (preprocessfn)
	print ()


	# Datasets
	# Images Train and Val Paths
	train_val_real_images_paths, train_val_fake_images_paths = dataset_utils.dataset_img_paths(
		dataset_type=dataset_type,
		status="train"
	)

	# Train-Val Split
	train_val_real_images_paths.sort()
	train_val_fake_images_paths.sort()
	random.Random(0).shuffle(train_val_real_images_paths)
	random.Random(0).shuffle(train_val_fake_images_paths)
	train_real_images_paths, val_real_images_paths = train_val_real_images_paths[:int(0.8 * len(train_val_real_images_paths))], train_val_real_images_paths[int(0.8 * len(train_val_real_images_paths)):]
	train_fake_images_paths, val_fake_images_paths = train_val_fake_images_paths[:int(0.8 * len(train_val_fake_images_paths))], train_val_fake_images_paths[int(0.8 * len(train_val_fake_images_paths)):]
	
	
	# Images Validation Dataset
	Val_Dataset = Image_Dataset(
		real_images_paths=val_real_images_paths,
		fake_images_paths=val_fake_images_paths,
		preprocessfn=preprocessfn,
		dual_scale=dual_scale,
		resize=preprocess_settings["resize"],
		separateAugmentation=separateAugmentation
	)
	
	# Images Test Dataset
	if config["train_settings"]["train"] == False:
		Test_Datasets = []
		for _,source in enumerate(test_image_sources):
			test_real_images_paths = dataset_utils.get_image_paths(
				dataset_type=dataset_type,
				status="val",
				image_sources=[source],
				label="real"
			)
			test_fake_images_paths = dataset_utils.get_image_paths(
				dataset_type=dataset_type,
				status="val",
				image_sources=[source],
				label="fake"
			)

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

				Test_Datasets.append(
					Image_Dataset(
						real_images_paths=test_real_images_paths,
						fake_images_paths=test_fake_images_paths,
						preprocessfn=Fixed_Input_preprocessfn,
						dual_scale=Fixed_Input_dual_scale,
						resize=preprocess_settings["resize"],
						separateAugmentation=separateAugmentation
					)
				)

			else:
				Test_Datasets.append(
					Image_Dataset(
						real_images_paths=test_real_images_paths,
						fake_images_paths=test_fake_images_paths,
						preprocessfn=preprocessfn,
						dual_scale=dual_scale,
						resize=preprocess_settings["resize"],
						separateAugmentation=separateAugmentation
					)
				)


	# DataLoaders
	# Val DataLoader
	Val_Dataloader = DataLoader(
		dataset=Val_Dataset,
		batch_size=config["train_settings"]["batch_size"],
		num_workers=config["train_settings"]["num_workers"],
		shuffle=False,
	)

	# Test DataLoaders
	if config["train_settings"]["train"] == False:
		Test_Dataloaders = []
		for i,_ in enumerate(test_image_sources):
			Test_Dataloaders.append(
				DataLoader(
					dataset=Test_Datasets[i],
					batch_size=config["train_settings"]["batch_size"],
					num_workers=config["train_settings"]["num_workers"],
					shuffle=False,
				)
			)

	print ("-"*25 + " Datasets and DataLoaders Ready " + "-"*25)


	# Global Variables: (feature_extractor, classifier)
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
	Model = Model_LightningModule(config)

	# PyTorch Lightning Trainer
	trainer = pl.Trainer(
		**config["trainer"],
		callbacks=[utils.LitProgressBar()],
		precision=32
	)
	

	# Evaluating
	# Finding Best Threshold
	if best_threshold is None:
		print ("-"*10, "Calculating best_threshold", "-"*10)
		# Predictions on Validation Dataset
		val_y_pred_y_true = trainer.predict(
			model=Model,
			dataloaders=Val_Dataloader,
		)
		val_y_pred, val_y_true = concatenate_predictions(y_pred_y_true=val_y_pred_y_true)

		# Calculating Threshold
		val_y_pred = val_y_pred[:, 1]
		val_y_true = np.argmax(val_y_true, axis=1) 
		_, _, _, _, _, _, _, _, _, best_threshold = calculate_metrics(y_pred=val_y_pred, y_true=val_y_true, threshold=None)

	# Predictions on Test Dataset
	test_y_pred_y_true = trainer.predict(
		model=Model,
		dataloaders=Test_Dataloaders,
	)
	
	if len(test_image_sources) == 1:
		test_set_metrics = []
		y_pred, y_true = concatenate_predictions(y_pred_y_true=test_y_pred_y_true)

		y_pred = y_pred[:, 1]
		y_true = np.argmax(y_true, axis=1) 

		ap, acc0, r_acc0, f_acc0, acc1, r_acc1, f_acc1, mcc0, mcc1, _ = calculate_metrics(y_pred=y_pred, y_true=y_true, threshold=best_threshold)
		test_set_metrics.append([0, ap, acc0, r_acc0, f_acc0, acc1, r_acc1, f_acc1, mcc0, mcc1])

		return test_set_metrics, best_threshold
	
	test_set_metrics = []
	for i, _ in enumerate(test_image_sources):
		y_pred, y_true = concatenate_predictions(y_pred_y_true=test_y_pred_y_true[i])

		y_pred = y_pred[:, 1]
		y_true = np.argmax(y_true, axis=1) 

		ap, acc0, r_acc0, f_acc0, acc1, r_acc1, f_acc1, mcc0, mcc1, _ = calculate_metrics(y_pred=y_pred, y_true=y_true, threshold=best_threshold)
		test_set_metrics.append([i, ap, acc0, r_acc0, f_acc0, acc1, r_acc1, f_acc1, mcc0, mcc1])

	return test_set_metrics, best_threshold