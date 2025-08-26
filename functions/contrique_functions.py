# Importing Libaries
import torch
import torch.nn as nn
import torchvision

def get_network(name, pretrained=False):
	network = {
		"VGG16": torchvision.models.vgg16(pretrained=pretrained),
		"VGG16_bn": torchvision.models.vgg16_bn(pretrained=pretrained),
		"resnet18": torchvision.models.resnet18(pretrained=pretrained),
		"resnet34": torchvision.models.resnet34(pretrained=pretrained),
		"resnet50": torchvision.models.resnet50(pretrained=pretrained),
		"resnet101": torchvision.models.resnet101(pretrained=pretrained),
		"resnet152": torchvision.models.resnet152(pretrained=pretrained),
	}

	if name not in network.keys():
		raise KeyError(f"{name} is not a valid network architecture")
	
	return network[name]



class CONTRIQUE_Model(nn.Module):
	def __init__(self, 
		encoder,
		n_features,
		patch_dim = (2,2),
		normalize = True,
		projection_dim = 128
	):
		"""
		ResNet50 model for Projector
		"""
		super(CONTRIQUE_Model, self).__init__()

		self.normalize = normalize
		self.encoder = nn.Sequential(*list(encoder.children())[:-2])
		self.n_features = n_features
		self.patch_dim = patch_dim
		
		self.avgpool = nn.AdaptiveAvgPool2d((1,1))
		self.avgpool_patch = nn.AdaptiveAvgPool2d(patch_dim)

		# MLP for Projector
		self.projector = nn.Sequential(
			nn.Linear(self.n_features, self.n_features, bias=False),
			nn.BatchNorm1d(self.n_features),
			nn.ReLU(),
			nn.Linear(self.n_features, projection_dim, bias=False),
			nn.BatchNorm1d(projection_dim),
		)
		
	def forward(self, x_i, x_j):
		# global features
		h_i = self.encoder(x_i)
		h_j = self.encoder(x_j)
		
		# local features
		h_i_patch = self.avgpool_patch(h_i)
		h_j_patch = self.avgpool_patch(h_j)
		
		h_i_patch = h_i_patch.reshape(-1,self.n_features, self.patch_dim[0]*self.patch_dim[1])
		
		h_j_patch = h_j_patch.reshape(-1,self.n_features, self.patch_dim[0]*self.patch_dim[1])
		
		h_i_patch = torch.transpose(h_i_patch,2,1)
		h_i_patch = h_i_patch.reshape(-1, self.n_features)
		
		h_j_patch = torch.transpose(h_j_patch,2,1)
		h_j_patch = h_j_patch.reshape(-1, self.n_features)
		
		h_i = self.avgpool(h_i)
		h_j = self.avgpool(h_j)
		
		h_i = h_i.view(-1, self.n_features)
		h_j = h_j.view(-1, self.n_features)
		
		if self.normalize:
			h_i = nn.functional.normalize(h_i, dim=1)
			h_j = nn.functional.normalize(h_j, dim=1)
			
			h_i_patch = nn.functional.normalize(h_i_patch, dim=1)
			h_j_patch = nn.functional.normalize(h_j_patch, dim=1)
		
		# Global Projections
		z_i = self.projector(h_i)
		z_j = self.projector(h_j)
		
		# Local Projections
		z_i_patch = self.projector(h_i_patch)
		z_j_patch = self.projector(h_j_patch)
		
		return z_i, z_j, z_i_patch, z_j_patch, h_i, h_j, h_i_patch, h_j_patch