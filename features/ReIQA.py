# Importing Libraries
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

import os,sys,warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.ReIQA.options.train_options import TrainOptions
from functions.ReIQA.networks.build_backbone import build_model
from functions.ReIQA.demo_content_aware_feats import run_inference_content
from functions.ReIQA.demo_quality_aware_feats import run_inference_quality
import defaults


class Compute_ReIQA(torch.nn.Module):
	def __init__(self,
		model_path:str,
		device:str
	):
		"""
		Args:
			model_path (str): Path to weights of ReIQA feature extractor.
			device (str): Device used while computing ReIQA features.
		"""
		super().__init__()
		# Device
		if device is None:
			self.device = "cuda" if torch.cuda.is_available() else "cpu"
		else:
			self.device = device

		# Paths
		self.content_model_path = os.path.join(model_path, "content_aware_r50.pth")
		self.quality_model_path = os.path.join(model_path, "quality_aware_r50.pth")
		
		# Arguments
		args = TrainOptions().parse()
		args.device = self.device

		# THE FOLLOWING SETTINGS are set to avoid EXECUTION ISSUES.
		strict = False			# Generally strict = True in `load_state_dict`
		args.feat_dim = 2048	# Added this line to balance dimensions of weights.

		# Content Model
		self.content_model, _ = build_model(args)
		self.content_model = torch.nn.DataParallel(self.content_model)

		checkpoint = torch.load(self.content_model_path, map_location=self.device)
		self.content_model.load_state_dict(checkpoint['model'], strict=strict)

		self.content_model.to(self.device)
		self.content_model.eval()

		# Quality Model
		self.quality_model, _ = build_model(args)
		self.quality_model = torch.nn.DataParallel(self.quality_model)

		checkpoint = torch.load(self.quality_model_path, map_location=self.device)
		self.quality_model.load_state_dict(checkpoint['model'], strict=strict)

		self.quality_model.to(self.device)
		self.quality_model.eval()


	def forward(self, img1, img2):
		# ReIQA does not perform L2-normalization and returns non-normalized features

		# Quality Features
		feat1  = self.quality_model.module.encoder(img1)
		feat1 = torch.flatten(feat1, start_dim=1)
		feat2  = self.quality_model.module.encoder(img2)
		feat2 = torch.flatten(feat2, start_dim=1)
		quality = torch.hstack((feat1, feat2))

		# Normalization
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
		img1 = normalize(img1)
		img2 = normalize(img2)

		# Content Features
		feat1  = self.content_model.module.encoder(img1)
		feat1 = torch.flatten(feat1, start_dim=1)
		feat2  = self.content_model.module.encoder(img2)
		feat2 = torch.flatten(feat2, start_dim=1)
		content = torch.hstack((feat1, feat2))

		# Content and Quality Fatures
		feat = torch.hstack((content, quality))

		return feat


# Calling Main function
if __name__ == '__main__':
	F = Compute_ReIQA(model_path=os.path.join(defaults.main_feature_ckpts_dir, "feature_extractor_checkpoints/reiqa_feature_extractors"), device="cuda:0")
	O = F.forward(torch.randn(1,3,224,224).cuda(), torch.randn(1,3,112,112).cuda())
	print (O.shape)
	print (torch.linalg.norm(O[0,:2048]), torch.linalg.norm(O[0,2048:4096]), torch.linalg.norm(O[0,4096:6144]), torch.linalg.norm(O[0,6144:8192]))