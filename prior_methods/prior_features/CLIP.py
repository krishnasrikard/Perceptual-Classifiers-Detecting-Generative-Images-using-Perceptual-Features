# Importing Libraries
import numpy as np
from PIL import Image

import torch
from torchinfo import summary
import clip

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Compute_CLIP(torch.nn.Module):
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
		self.clip_model, self.preprocess_fn = clip.load(model_name, device=self.device)
		for param in self.clip_model.parameters():
			param.requires_grad = False


	def forward(self, img):
		return self.clip_model.encode_image(image=img)	


class CLIP_Classifier(torch.nn.Module):
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
		self.clip_model, self.preprocess_fn = clip.load(model_name, device=self.device)
		for param in self.clip_model.parameters():
			param.requires_grad = False

		if model_name == "RN50":
			self.classifier = torch.nn.Linear(1024,1)
		else:
			self.classifier = torch.nn.Linear(768,1)


	def forward(self, img):
		f = self.clip_model.encode_image(image=img).to(torch.float32)
		return self.classifier(f)
	

# Calling Main function
if __name__ == '__main__':
	F = Compute_CLIP(model_name="ViT-L/14", device="cuda:0")
	O = F.forward(torch.randn(1,3,224,224).cuda())
	print (O.shape)