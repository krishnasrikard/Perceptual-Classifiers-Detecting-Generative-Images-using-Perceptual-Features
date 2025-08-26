# Importing Libraries
import numpy as np
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary

import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Compute_ResNet50(torch.nn.Module):
	def __init__(self,
		device:str
	):
		"""
		Args:
			device (str): Device used while computing ResNet50 features.
		"""
		super().__init__()
		# Device
		if device is None:
			self.device = "cuda" if torch.cuda.is_available() else "cpu"
		else:
			self.device = device

		# Model and Preprocessing Function
		self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
		self.model.fc = torch.nn.Identity()
		self.model = self.model.to(self.device)
		self.model.eval()


	def forward(self, img):
		return self.model(img)
	

# Calling Main function
if __name__ == '__main__':
	F = Compute_ResNet50(device="cuda:0")
	O = F.forward(torch.randn(1,3,224,224).cuda())
	print (O.shape)