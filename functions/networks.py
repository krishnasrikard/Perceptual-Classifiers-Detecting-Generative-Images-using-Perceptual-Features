# Importing Libraries
import torch
import torch.nn as nn
import timm
from torchinfo import summary

import os, sys, warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import joblib, time
import features.ResNet50 as ResNet50
import features.CLIP as CLIP
import features.CONTRIQUE as CONTRIQUE
import features.ReIQA as ReIQA
import features.ARNIQA as ARNIQA
import features.TReS as TReS
import features.HyperIQA as HyperIQA
import defaults


class Classifier_Arch1(nn.Module):
	def __init__(self,
		input_dim:int,
		hidden_layers:list	
	) -> None:
		super().__init__()

		layers = []

		for i,_ in enumerate(hidden_layers):
			if i==0:
				layers.append(nn.Linear(in_features=input_dim, out_features=hidden_layers[i]))
				layers.append(nn.BatchNorm1d(hidden_layers[i]))
				layers.append(nn.GELU())
			else:
				layers.append(nn.Linear(in_features=hidden_layers[i-1], out_features=hidden_layers[i]))
				layers.append(nn.BatchNorm1d(hidden_layers[i]))
				layers.append(nn.GELU())

		layers.append(nn.Linear(in_features=hidden_layers[i], out_features=2))

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		x = self.layers(x)
		return x
	

class Classifier_Arch2(nn.Module):
	def __init__(self,
		input_dim:int,
		hidden_layers:list,
	) -> None:
		super().__init__()
		assert len(hidden_layers) == 1, "Invalid Hidden Size"

		self.layer1 = nn.Linear(in_features=input_dim, out_features=hidden_layers[0])
		self.act_layer1 = nn.ReLU()
		self.layer2 = nn.Linear(in_features=hidden_layers[0], out_features=2)

	def forward(self, x):
		features = self.act_layer1(self.layer1(x))
		preds = self.layer2(features)
		return features, preds
	

# Get Model for feature extraction
def get_model(model_name, device):
	# Get model
	if model_name == "resnet50":
		model = ResNet50.Compute_ResNet50(
			device=device
		)

	elif model_name == "clip-resnet50":
		model = CLIP.Compute_CLIP(
			model_name="RN50",
			device=device
		)	

	elif model_name == "clip-vit-l-14":
		model = CLIP.Compute_CLIP(
			model_name="ViT-L/14",
			device=device
		)	

	elif model_name == "contrique":
		model = CONTRIQUE.Compute_CONTRIQUE(
			model_path=os.path.join(defaults.main_feature_ckpts_dir, "contrique_feature_extractor", "CONTRIQUE_checkpoint25.tar"),
			device=device
		)

	elif model_name == "reiqa":
		model = ReIQA.Compute_ReIQA(
			model_path=os.path.join(defaults.main_feature_ckpts_dir, "reiqa_feature_extractors"),
			device=device
		)

	elif model_name == "arniqa":
		model = ARNIQA.Compute_ARNIQA(
			device=device
		)
	
	elif model_name == "tres":
		model = TReS.Compute_TReS(
			model_path=os.path.join(defaults.main_feature_ckpts_dir, "tres_model/bestmodel_1_2021.zip"),
			device=device
		)
	
	elif model_name == "hyperiqa":
		model = HyperIQA.Compute_HyperIQA(
			model_path=os.path.join(defaults.main_feature_ckpts_dir, "hyperiqa_model/koniq_pretrained.pkl"),
			device=device
		)

	else:
		assert False, f"Update get_model() in extract_features.py to work with the model name: '{model_name}'"
		
	return model


# Calling Main function
if __name__ == '__main__':
	None