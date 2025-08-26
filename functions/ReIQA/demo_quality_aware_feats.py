# Importing Libraries
from __future__ import print_function
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data.distributed
from torchvision import transforms
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils import data

import os,sys
sys.path.append("functions/ReIQA")
import scipy.io
import time
import csv
import subprocess
import pandas as pd
import pickle
from functions.ReIQA.options.train_options import TrainOptions
from functions.ReIQA.learning.contrast_trainer import ContrastTrainer
from functions.ReIQA.networks.build_backbone import build_model
from functions.ReIQA.datasets.util import build_contrast_loader
from functions.ReIQA.memory.build_memory import build_mem



def run_inference_quality(img, ckpt_path):
	# Arguments
	args = TrainOptions().parse()
	args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# build model
	model, _ = build_model(args)
	model = torch.nn.DataParallel(model)

	# check and resume a model
	# ckpt_path =  './reiqa_ckpts/quality_aware_r50.pth'
	
	checkpoint = torch.load(ckpt_path, map_location='cpu')
	model.load_state_dict(checkpoint['model'])

	model.to(args.device)
	model.eval()


	# Modiying the next few line i.e instead image-path as input considering image as input
	# img_path = img
	# image = Image.open(img_path).convert('RGB')
	
	# PIL Image
	image = Image.fromarray(img)
	image2 = image.resize((image.size[0]//2,image.size[1]//2)) # half-scale
		
	# transform to tensor
	img1 = transforms.ToTensor()(image).unsqueeze(0)
	img2 = transforms.ToTensor()(image2).unsqueeze(0)

	with torch.no_grad():
		feat1  = model.module.encoder(img1.to(args.device)) 
		feat2  = model.module.encoder(img2.to(args.device)) 
		feat = torch.cat((feat1,feat2),dim=1).detach().cpu().numpy()

	# save features 
	return feat
				
	# save_path = "feats_quality_aware/"
	# if not os.path.exists(save_path):
	# 	os.makedirs(save_path)
	# np.save("feats_quality_aware/" + img_path[img_path.rfind("/")+1:-4] + '_quality_aware_features.npy', feat)
	# print('Quality Aware feature Extracted')



# if __name__ == '__main__':
# 	run_inference_quality()
