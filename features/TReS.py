# Importing Libraries
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torchinfo import summary

import os,sys,warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import argparse
import functions.tres_functions as tres_functions
import defaults

class Compute_TReS(torch.nn.Module):
	def __init__(self,
		model_path:str,
		device:str
	):
		"""
		Args:
			model_path (str): Path to weights of TReS model.
			device (str): Device used while computing features.
		"""
		super().__init__()
		# Device
		if device is None:
			self.device = "cuda" if torch.cuda.is_available() else "cpu"
		else:
			self.device = device

		# Load TReS Model
		config = argparse.Namespace()
		config.network = 'resnet50'
		config.nheadt = 16
		config.num_encoder_layerst = 2
		config.dim_feedforwardt = 64

		self.model = tres_functions.Net(config, self.device).to(self.device)
		self.model.load_state_dict(torch.load(model_path))
		self.model.eval()
		for param in self.model.parameters():
			param.requires_grad = False

	
	def forward(self, img):
		_, feat_batch = self.model(img)

		return feat_batch
	

# Calling Main function
if __name__ == '__main__':
	F = Compute_TReS(model_path=os.path.join(defaults.main_feature_ckpts_dir, "feature_extractor_checkpoints/tres_model/bestmodel_1_2021.zip"), device="cuda:0")
	O = F.forward(torch.randn(1,3,224,224).cuda())
	print (O.shape)