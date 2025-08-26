""" 
Plotting Feature Representations
"""
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.manifold import TSNE

import torch
from torch.utils.data import DataLoader

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import random
import prior_methods.prior_functions.prior_preprocess as prior_preprocess
import functions.preprocess as preprocess
import prior_methods.prior_features.CLIP as CLIP
import prior_methods.prior_features.DRCT as DRCT
import functions.networks as networks
import functions.dataset as dataset
import functions.dataset_utils as dataset_utils
import defaults


# Load extracted features
def extract_features(
	feature_extractor:any,
	preprocess_fn:any,
	dual_scale:bool,
	num_images:int,
	use_reconstructed:bool = True,
):
	"""
	Args:
		feature_extractor (any): Feature Extractor.
		dual_scale (bool): Dual Scale
	"""
	# Assertions
	if use_reconstructed:
		assert num_images % 4 == 0, "Number of images must be multiple of 4"
		num_images_per_category = num_images // 4
	else:
		assert num_images % 2 == 0, "Number of images must be multiple of 2"
		num_images_per_category = num_images // 2


	# Random Indices
	np.random.seed(0)
	random_indices = np.arange(5000)
	np.random.shuffle(random_indices)
	fake_random_indices = random_indices[:num_images_per_category]
	real_random_indices = random_indices[:num_images_per_category]


	# Loading DRCT Train Dataset Paths
	real_images = np.array(os.listdir("/mnt/LIVELAB2/krishna/Datasets/DRCT/dataset/real_images/train2017"))[real_random_indices]
	fake_images = np.array(os.listdir("/mnt/LIVELAB2/krishna/Datasets/DRCT/dataset/fake_images/stable-diffusion-v1-4/train2017"))[fake_random_indices]
	fake_reconstructed_images = np.array(os.listdir("/mnt/LIVELAB2/krishna/Datasets/DRCT/dataset/fake_reconstructed_images/stable-diffusion-v1-4/train2017"))[fake_random_indices]
	real_reconstructed_images = np.array(os.listdir("/mnt/LIVELAB2/krishna/Datasets/DRCT/dataset/real_reconstructed_images/stable-diffusion-v1-4/train2017"))[fake_random_indices]


	# Paths
	real_images_path = []
	fake_images_path = []
	fake_reconstructed_images_path = []
	real_reconstructed_images_path = []

	for i in range(num_images_per_category):
		path = os.path.join("DRCT/dataset/real_images/train2017", real_images[i])
		real_images_path.append(path)

	for i in range(num_images_per_category):
		path = os.path.join("DRCT/dataset/fake_images/stable-diffusion-v1-4/train2017", fake_images[i])
		fake_images_path.append(path)

		path = os.path.join("DRCT/dataset/fake_reconstructed_images/stable-diffusion-v1-4/train2017", fake_reconstructed_images[i])
		fake_reconstructed_images_path.append(path)

		path = os.path.join("DRCT/dataset/real_reconstructed_images/stable-diffusion-v1-4/train2017", real_reconstructed_images[i])
		real_reconstructed_images_path.append(path)


	# Calculating Feature Representations
	if use_reconstructed:
		list_images_paths = [real_images_path, fake_images_path, fake_reconstructed_images_path, real_reconstructed_images_path]
	else:
		list_images_paths = [real_images_path, fake_images_path]

	feature_representations = []

	for image_paths in list_images_paths:
		# Loading Dataset
		image_dataset = dataset.Image_Dataset(
			real_images_paths=image_paths,
			fake_images_paths=[],
			preprocessfn=preprocess_fn,
			dual_scale=dual_scale,
			resize=None,
			separateAugmentation=False,
		)
		image_dataloader = DataLoader(
			dataset=image_dataset,
			batch_size=32,
			num_workers=12,
			shuffle=True,
		)

		# To GPU
		feature_extractor.to("cuda")

		# Prediction
		features = []
		for batch in image_dataloader:
			if len(batch) == 2:
				X, y_true = batch
				with torch.no_grad():
					X = feature_extractor(X.to("cuda"))
			else:
				X1, X2, y_true = batch
				with torch.no_grad():
					X = feature_extractor(X1.to("cuda"), X2.to("cuda"))
			
			features.append(X)

		features = torch.concat(features, dim=0)
		features = features.detach().cpu().numpy()
		print ("Features Shape: ", features.shape)
		feature_representations.append(features)


	# Return Assertions
	if use_reconstructed:
		assert len(feature_representations) == 4, "Number of feature representations must be 4"
	else:
		assert len(feature_representations) == 2, "Number of feature representations must be 2"

	return feature_representations


# T-Sne Plot
def tnse_plot(
	feature_extractor:any,
	preprocess_fn:any,
	dual_scale:bool,
	num_images:int,
	use_reconstructed:bool,
	title:str,
	save_path:str,
):
	# Features
	feature_representations = extract_features(
		feature_extractor=feature_extractor,
		preprocess_fn=preprocess_fn,
		dual_scale=dual_scale,
		num_images=num_images,
		use_reconstructed=use_reconstructed,
	)
	if len(feature_representations) == 4 and use_reconstructed:
		num_images_per_category = num_images // 4
		target_labels = [np.zeros((num_images_per_category,1)), np.ones((num_images_per_category,1)), np.ones((num_images_per_category,1)), np.ones((num_images_per_category,1))]
	elif len(feature_representations) == 2 and not use_reconstructed:
		num_images_per_category = num_images // 2
		target_labels = [np.zeros((num_images_per_category,1)), np.ones((num_images_per_category,1))]
	else:
		assert False, "Number of feature representations must be 2 or 4"

	# T-SNE Separation
	tsne = TSNE(n_components=2, perplexity=20, random_state=13, n_jobs=-1)
	data = np.concatenate(feature_representations, axis=0)
	target_labels = np.concatenate(target_labels, axis=0)
	feature_representations_components = tsne.fit_transform(data, target_labels)

	# Plotting
	tab10 = cm.get_cmap('tab10')
	plt.figure(figsize=(8,8))
	plt.grid()

	if use_reconstructed:
		Labels_Colors = [
			("Real", tab10(2)),
			("Fake", tab10(3)),
			("Fake Reconstructed", tab10(1)),
			("Real Reconstructed", tab10(0)),
		]
	else:
		Labels_Colors = [
			("Real", tab10(2)),
			("Fake", tab10(3)),
		]

	for i,(label, color) in enumerate(Labels_Colors):
		f = feature_representations_components[num_images_per_category*i : num_images_per_category*(i+1), :]
		plt.scatter(x=f[:,0], y=f[:,1], label=label, c=color)

	plt.legend()
	plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=750)



# Main Function
if __name__ == "__main__":

	# ----------------------------------------------------------------------------------------------------------

	# Preprocessing-Settings
	preprocess_settings = {
		"selected_transforms_name":"test", 
		"probability": -1, 
		"gaussian_blur_range": None, 
		"jpeg_compression_qfs": None, 
		"input_image_dimensions": (224,224), 
		"resize":None
	}

	# Settings
	num_images = 2000
	use_reconstructed = False
	target_dir = "plots/tsne"

	# Create Directories
	os.makedirs(target_dir, exist_ok=True)

	# ----------------------------------------------------------------------------------------------------------

	# CLIP and Preprocessing-Function
	feature_extractor = CLIP.Compute_CLIP(model_name="ViT-L/14", device="cuda")
	preprocess_fn, dual_scale = prior_preprocess.get_preprocessfn(model_name="clip-vit-l-14", **preprocess_settings)

	tnse_plot(
		feature_extractor=feature_extractor,
		preprocess_fn=preprocess_fn,
		dual_scale=dual_scale,
		num_images=num_images,
		use_reconstructed=use_reconstructed,
		title="CLIP:ViT-L/14",
		save_path=os.path.join(target_dir, "clip-vit-l-14.png"),
	)

	# ----------------------------------------------------------------------------------------------------------

	# DRCT Feature-Extractor and Preprocessing-Function
	feature_extractor = DRCT.CLIPv2()
	preprocess_fn, dual_scale = prior_preprocess.get_preprocessfn(model_name="drct-clip-vit-l-14", **preprocess_settings)

	initial_weights = feature_extractor.state_dict()
	trained_weights = torch.load(os.path.join(defaults.main_prior_checkpoints_dir, "DRCT/extensive/MarginContrastiveLoss_CrossEntropy/drct-clip-vit-l-14/13_acc0.9664.pth"))
	for (k1,v1), (k2,v2) in zip(initial_weights.items(), trained_weights.items()):
		assert v1.shape == v2.shape, ""
		initial_weights[k1] = trained_weights[k2]
	feature_extractor.load_state_dict(initial_weights)

	tnse_plot(
		feature_extractor=feature_extractor,
		preprocess_fn=preprocess_fn,
		dual_scale=dual_scale,
		num_images=num_images,
		use_reconstructed=use_reconstructed,
		title="DRCT CLIP:ViT-L/14",
		save_path=os.path.join(target_dir, "drct-clip-vit-l-14.png"),
	)

	# ----------------------------------------------------------------------------------------------------------

	# IQA Feature-Extractors and Preprocessing-Functions
	for model_name, title in [("contrique", "CONTRIQUE"), ("reiqa", "ReIQA"), ("hyperiqa", "HyperIQA"), ("arniqa", "ARNIQA"), ("tres", "TReS")]:
		feature_extractor = networks.get_model(model_name=model_name, device="cuda")

		preprocess_fn, dual_scale = preprocess.get_preprocessfn(model_name=model_name, **preprocess_settings)

		tnse_plot(
			feature_extractor=feature_extractor,
			preprocess_fn=preprocess_fn,
			dual_scale=dual_scale,
			num_images=num_images,
			use_reconstructed=use_reconstructed,
			title=title,
			save_path=os.path.join(target_dir, "{}_model.png".format(model_name)),
		)

	# ----------------------------------------------------------------------------------------------------------