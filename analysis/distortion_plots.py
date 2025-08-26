# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os, sys, warnings
warnings.filterwarnings("ignore")


# Models
IQAModels2Index = {
	"HyperIQA": 0,
	"TreS": 1,
	"CONTRIQUE": 2,
	"ReIQA": 3,
	"ARNIQA": 4
}

PriorModels2Index = {
	"DRCT/ConvNext": 0,
	"DRCT/UnivFD": 1,
	"UnivFD": 1
}


# Levels of Distortion
levels_distortions_map = {
	"Blur": [1,2,3,4,5],
	"JPEG": [90,80,70,60,50,40,30]
}


# Path to folders of proposed and prior works
proposed_works_dir_map = {
	"GenImage": "../results/extensive_MarginContrastiveLoss_CrossEntropy_GenImage_GenImage",
	"DRCT": "../results/extensive_MarginContrastiveLoss_CrossEntropy_DRCT_DRCT"
}

prior_works_dir_map = {
	"GenImage": {"DRCT/UnivFD": "../prior_methods/prior_results/extensive_GenImage_GenImage"},
	"DRCT": "../prior_methods/prior_results/extensive_DRCT_DRCT"
}


# Functions
def get_Accuracy_distortions(	
	proposed_model_names:list,
	prior_model_names:list,
	dataset_type:str,
	distortion_type:str
):
	"""
	Args:
		proposed_model_names (list): List of proposed model names.
		prior_model_names (list): List of prior model names.
		dataset_type (str): Options: ['GenImage', 'DRCT'].
		distortion_type (str): Options: ['Blur', 'JPEG']
	"""
	# Paths
	proposed_works_dir = proposed_works_dir_map[dataset_type]
	prior_works_dir = prior_works_dir_map[dataset_type]

	# Setting Name
	if distortion_type == "JPEG":
		setting_name = "jpegQF="
	else:
		setting_name = "sigma="

	# Results
	results = {}

	# Proposed Works
	for proposed_model_name in proposed_model_names:
		# Results of Model-Name
		results[proposed_model_name] = []

		for setting_value in levels_distortions_map[distortion_type]:
			# Load the CSV file
			df = pd.read_csv(os.path.join(
				proposed_works_dir, setting_name + "{}.csv".format(setting_value)
			))

			# Accuracy
			Acc = df['(mAP, mAcc, mAcc_Real, mAcc_fake)']

			# Model Index
			model_index = IQAModels2Index[proposed_model_name]

			# Appending
			results[proposed_model_name].append(
				eval(Acc.iloc[model_index])[1]
			)

	# Prior Works
	for prior_model_name in prior_model_names:
		# Results of Model-Name
		results[prior_model_name] = []

		for setting_value in levels_distortions_map[distortion_type]:
			# Load the CSV file
			if dataset_type == "GenImage":
				df = pd.read_csv(os.path.join(
					prior_works_dir[prior_model_name], setting_name + "{}.csv".format(setting_value)
				))
			else:
				df = pd.read_csv(os.path.join(
					prior_works_dir, setting_name + "{}.csv".format(setting_value)
				))

			# Accuracy
			Acc = df['(mAP, mAcc, mAcc_Real, mAcc_fake)']

			# Model Index
			model_index = PriorModels2Index[prior_model_name]

			# Appending
			results[prior_model_name].append(
				eval(Acc.iloc[model_index])[1]
			)

	return results


def plot_results(
	model_names:list,
	results:dict,
	dataset_type:str,
	distortion_type:str
):
	# Parameters
	markers = ['o', 's', 'D', '^', "*", ".", ]
	levels_distortions = levels_distortions_map[distortion_type]

	# Figure
	plt.figure(figsize=(8, 4))
	plt.rc('font', **{'size': 15})
	plt.grid()
	plt.ylim(0,100)

	for i,model_name in enumerate(model_names):
		plt.plot(levels_distortions, results[model_name], label=model_name, linewidth=2)
		plt.scatter(levels_distortions, results[model_name], marker=markers[i])

	if distortion_type == "Blur":
		plt.xlabel(r'sigma ($\sigma$)', fontsize=18)
	else:
		plt.xlabel('QF', fontsize=18)
	plt.ylabel('mAcc', fontsize=18)
	plt.legend(fontsize=14)
	

	plt.savefig('plots/distortions/mAcc_vs_{}_{}_{}.png'.format(distortion_type, dataset_type, dataset_type), bbox_inches='tight', pad_inches=0.1, dpi=750)



for dataset_type in ['DRCT', "GenImage"]:
	for distortion_type in ["JPEG", "Blur"]:
		results = get_Accuracy_distortions(
			proposed_model_names=["CONTRIQUE", "ReIQA"],
			prior_model_names=["DRCT/UnivFD"],
			dataset_type=dataset_type,
			distortion_type=distortion_type
		)

		plot_results(
			model_names=["CONTRIQUE", "ReIQA", "DRCT/UnivFD"],
			results=results,
			dataset_type=dataset_type,
			distortion_type=distortion_type
		)