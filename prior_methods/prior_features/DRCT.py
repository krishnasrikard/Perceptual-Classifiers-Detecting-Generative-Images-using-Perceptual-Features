"""
Testing Performance of DRCT on our conditions
"""
# Importing Libraries
import numpy as np

import torch
import torch.nn as nn
import clip
import timm

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append("/home/krishna/Perceptual-Classifers")
from yaml import safe_load
import functions.networks as networks
import functions.module as module
import functions.utils as utils
import defaults


# DRCT CLIP and ConvNext modules
class ConvNext(nn.Module):
	def __init__(self, ):
		super(ConvNext, self).__init__()
		self.net = timm.create_model("convnext_base_in22k", pretrained=True)
		n_features = self.net.head.fc.in_features
		self.net.head.fc = nn.Linear(n_features, 1024)

	def forward(self, x):
		features = self.net(x)
		return features


class CLIPv2(nn.Module):
	def __init__(self, ):
		super(CLIPv2, self).__init__()
		self.model, _ = clip.load("ViT-L/14", device="cpu")
		self.model.eval()
		for params in self.model.parameters():
			params.requires_grad = False
		self.fc = nn.Linear(768, 1024)

	def forward(self, x):
		features = self.model.encode_image(x)
		return self.fc(features)


# DRCT: CLIP and ConvNext models
class DRCT_CLIP(nn.Module):
	def __init__(self):
		super(DRCT_CLIP, self).__init__()
		self.model = CLIPv2()
		self.fc = nn.Linear(1024, 2)

	def forward(self, x):
		feature = self.model(x)
		y_pred = self.fc(feature)

		return y_pred


class DRCT_ConvNext(nn.Module):
	def __init__(self):
		super(DRCT_ConvNext, self).__init__()
		self.model = ConvNext()
		self.fc = nn.Linear(1024, 2)

	def forward(self, x):
		feature = self.model(x)
		y_pred = self.fc(feature)

		return y_pred


class DRCT_Classifier(torch.nn.Module):
	def __init__(self,
		model_name:str,
		device:str
	):
		"""
		Args:
			model_name (str): CLIP backbone. Use `clip.available_models()` to see all backbones.
			device (str): Device used while computing CLIP features.
		"""
		super().__init__()
		# Device
		if device is None:
			self.device = "cuda" if torch.cuda.is_available() else "cpu"
		else:
			self.device = device

		# Model and Preprocessing Function
		if model_name == "drct-clip-vit-l-14":
			# DRCT: CLIP:ViT-L/14
			self.model = DRCT_CLIP()
		else:
			# DRCT: ConvNext-B
			self.model = DRCT_ConvNext()

		# To Device
		self.model = self.model.to(self.device)

	def forward(self, img):
		return self.model(img)
	

# Calling Main function
if __name__ == '__main__':
	F = DRCT_Classifier(model_name="drct-vit-l-14", device="cuda:0")
	O = F.forward(torch.randn(1,3,224,224).cuda())
	print (O.shape)