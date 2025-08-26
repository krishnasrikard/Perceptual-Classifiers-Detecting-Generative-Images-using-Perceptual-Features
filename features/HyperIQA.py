# Importing Libraries
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torchinfo import summary

import os,sys,warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import functions.hyperiqa_functions as hyperiqa_functions
import defaults

class Compute_HyperIQA(torch.nn.Module):
	def __init__(self,
		model_path:str,
		device:str
	):
		"""
		Args:
			model_path (str): Path to weights of HyperIQA model.
			device (str): Device used while computing features.
		"""
		super().__init__()
		# Device
		if device is None:
			self.device = "cuda" if torch.cuda.is_available() else "cpu"
		else:
			self.device = device

		# Load HyperIQA Model
		self.HyperIQA = hyperiqa_functions.HyperNet(16, 112, 224, 112, 56, 28, 14, 7)
		self.HyperIQA = self.HyperIQA.to(self.device)
		self.HyperIQA.load_state_dict(torch.load(model_path, map_location=self.device))
		self.HyperIQA.eval()
		for param in self.HyperIQA.parameters():
			param.requires_grad = False

	
	def forward(self, img):
		# Parameters for Target Network
		parameters = self.HyperIQA(img)
		semantic_features = parameters["pooled_semantic_features"]
		
		# Target Network
		target_model = hyperiqa_functions.TargetNet(parameters).to(self.device)
		for param in target_model.parameters():
			param.requires_grad = False

		q,q1,q2,q3,q4 = target_model(parameters["target_in_vec"])

		feat_batch = torch.hstack((
			torch.flatten(semantic_features, start_dim=1),
			torch.flatten(parameters["target_in_vec"], start_dim=1),
			torch.flatten(q1, start_dim=1),
			torch.flatten(q2, start_dim=1),
			torch.flatten(q3, start_dim=1),
			torch.flatten(q4, start_dim=1),
		))

		return feat_batch
	

# Calling Main function
if __name__ == '__main__':
	F = Compute_HyperIQA(model_path=os.path.join("/mnt/LIVELAB2/Detecting-AI-Generated-Images", "feature_extractor_checkpoints/hyperiqa_model/koniq_pretrained.pkl"), device="cuda:0")
	O = F.forward(torch.randn(1,3,224,224).cuda())
	print (O.shape)