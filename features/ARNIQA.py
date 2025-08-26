# Importing Libraries
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torchinfo import summary

import os,sys,warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import defaults

class Compute_ARNIQA(torch.nn.Module):
	def __init__(self,
		device:str
	):
		"""
		Args:
			device (str): Device used while computing features.
		"""
		super().__init__()
		# Device
		if device is None:
			self.device = "cuda" if torch.cuda.is_available() else "cpu"
		else:
			self.device = device

		# Load ARNIQA Model
		ARNIQA = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA", regressor_dataset="flive")

		# ARNIQA Feature Extractor
		self.model = ARNIQA.encoder.model
		self.model = self.model.to(self.device)
		self.model.eval()
		for param in self.model.parameters():
			param.requires_grad = False

	
	def forward(self, img1, img2):
		# ARNIQA performs normalization after extracting features
		# https://github.com/miccunifi/ARNIQA/blob/main/models/resnet.py#L43

		feat1_batch = self.model(img1)
		feat1_batch = torch.nn.functional.normalize(feat1_batch, dim=1)
		feat1_batch = torch.flatten(feat1_batch, start_dim=1)

		feat2_batch = self.model(img2)
		feat2_batch = torch.nn.functional.normalize(feat2_batch, dim=1)
		feat2_batch = torch.flatten(feat2_batch, start_dim=1)

		feat_batch = torch.hstack((feat1_batch, feat2_batch))

		return feat_batch
	

# Calling Main function
if __name__ == '__main__':
	F = Compute_ARNIQA(device="cuda:0")
	O = F.forward(torch.randn(1,3,224,224).cuda(), torch.randn(1,3,112,112).cuda())
	print (O.shape)
	print (torch.linalg.norm(O[0,:2048]), torch.linalg.norm(O[0,2048:4096]))