# Importing Libraries
import torch
import torch.nn as nn
import timm
from torchinfo import summary

import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from tqdm import tqdm
import joblib, time
import prior_features.CLIP as CLIP
import prior_features.DRCT as DRCT
import defaults


# Get Model for feature extraction
def get_model(model_name, device):
	# Get model
	if model_name == "clip-resnet50":
		model = CLIP.CLIP_Classifier(
			model_name="RN50",
			device=device
		)

	elif model_name == "clip-vit-l-14":
		model = CLIP.CLIP_Classifier(
			model_name="ViT-L/14",
			device=device
		)
		
	elif model_name == "drct-clip-vit-l-14":
		model = DRCT.DRCT_Classifier(
			model_name=model_name,
			device=device
		)

	elif model_name == "drct-convnext-b":
		model = DRCT.DRCT_Classifier(
			model_name=model_name,
			device=device
		)

	else:
		assert False, f"Update get_model() in extract_features.py to work with the model name: '{model_name}'"
		
	return nn.Identity(), model


# Calling Main function
if __name__ == '__main__':
	None