# Importing Libraries
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

import os,sys,warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import functions.contrique_functions as contrique_functions
import defaults


class Compute_CONTRIQUE(torch.nn.Module):
	def __init__(self,
		model_path:str,
		device:str
	):
		"""
		Args:
			model_path (str): Path to weights of Contrique feature extractor.
			device (str): Device used while computing features.
		"""
		super().__init__()
		# Device
		if device is None:
			self.device = "cuda" if torch.cuda.is_available() else "cpu"
		else:
			self.device = device

		# Load CONTRIQUE Model
		encoder = contrique_functions.get_network('resnet50', pretrained=False)
		self.model = contrique_functions.CONTRIQUE_Model(encoder=encoder, n_features=2048)
		self.model.load_state_dict(torch.load(model_path, map_location=self.device))
		self.model = self.model.to(self.device)
		self.model.eval()

	def forward(self, img1, img2):
		# Contrique performs and returns L2-normalized features

		_, _, _, _, feat1_batch, feat2_batch, _, _ = self.model(img1, img2)

		feat1_batch = torch.flatten(feat1_batch, start_dim=1)
		feat2_batch = torch.flatten(feat2_batch, start_dim=1)
		feat_batch = torch.hstack((feat1_batch, feat2_batch))

		return feat_batch
	

# Calling Main function
if __name__ == '__main__':
	F = Compute_CONTRIQUE(model_path=os.path.join(defaults.main_feature_ckpts_dir, "feature_extractor_checkpoints/contrique_feature_extractor/CONTRIQUE_checkpoint25.tar"), device="cuda:0")
	O = F.forward(torch.randn(1,3,224,224).cuda(), torch.randn(1,3,112,112).cuda())
	print (O.shape)
	print (torch.linalg.norm(O[0,:2048]), torch.linalg.norm(O[0,2048:4096]))