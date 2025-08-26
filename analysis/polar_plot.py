"""
Plotting Accuracy of different methods for different generative models.
"""
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import os, sys, warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import defaults


# Function to get performance of perceptual models
def get_performance(
	results_dataframe:pd.DataFrame,
	generative_models:list,
	metric:str
):
	"""
	Args:
		results_dataframe (pd.DataFrame): Pandas Dataframe of loaded from .csv file.
		generative_models (list): List of generative models to consider from the dataframe.
		metric (str): Evaluation Metric.
	"""
	# Assertions
	assert all(gen_model in results_dataframe.columns for gen_model in generative_models), "Invalid list of generative models"

	# Metric-Index
	if metric == "mAP":
		metric_index = 0
	elif metric == "mAcc":
		metric_index = 1
	elif metric == "mAcc_Real":
		metric_index = 2
	else:
		metric_index = 3

	# Performance
	performance = {}
	for _, row in results_dataframe.iterrows():
		model = row["model"]

		model_performance = []
		for gen_model in generative_models:
			model_performance.append(
				np.round(float(eval(row[gen_model])[metric_index]), decimals=2)
			)
		model_performance.append(
			np.round(float(eval(row["(mAP, mAcc, mAcc_Real, mAcc_fake)"])[metric_index]), decimals=2)
		)

		performance[model] = model_performance

	return performance



def realign_polar_xticks(ax):
    for theta, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        theta = theta * ax.get_theta_direction() + ax.get_theta_offset()
        theta = np.pi/2 - theta
        y, x = np.cos(theta), np.sin(theta)
        if x >= 0.1:
            label.set_horizontalalignment('left')
        if x <= -0.1:
            label.set_horizontalalignment('right')
        if y >= 0.5:
            label.set_verticalalignment('bottom')
        if y <= -0.5:
            label.set_verticalalignment('top')


# Paths
Our_Results = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "results")
Prior_Results = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "prior_methods/prior_results")



# GenImage Table Results for variants of Data Augmentation
Without_Distortion = []
performance = {}

# Dataframe Paths
Our_Results_Dataframe_path = os.path.join(Our_Results, "extensive_GenImage_GenImage", "default.csv")
Prior_Results_Dataframe_path1 = os.path.join(Prior_Results, "extensive_GenImage_GenImage", "default.csv")
Prior_Results_Dataframe_path2 = os.path.join(Prior_Results, "p=0.5_standard_GenImage_GenImage", "default.csv")

# Dataframe
Our_df = pd.read_csv(Our_Results_Dataframe_path)
Prior_df1 = pd.read_csv(Prior_Results_Dataframe_path1)
Prior_df2 = pd.read_csv(Prior_Results_Dataframe_path2)
df = pd.concat([Our_df, Prior_df1, Prior_df2])

# Getting Performance
performance = get_performance(
	results_dataframe=df,
	generative_models=["midjourney", "sdv4", "sdv5", "adm", "glide", "wukong", "vqdm", "biggan"],
	metric="mAcc"
)

Without_Distortion.append(performance)


# Plotting Polar Plot for different models
categories = ["Midjourney", "SDv1.4", "SDv1.5", "Guided", "Glide", "Wukong", "VQDM", "BigGAN"]

# Plotting
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
plt.rcParams.update({'font.size': 14})
angle = list(np.linspace(0, 2 * np.pi, len(categories), endpoint=False))
angle += angle[:1]
ax.set_yticklabels([])
ax.set_xticks(angle[:-1])
ax.set_xticklabels(categories)
ax.xaxis.set_tick_params(labelsize=14)
realign_polar_xticks(ax)

for model_name, color in [("clip-vit-l-14", "C0"), ("drct-clip-vit-l-14", "C1"), ("reiqa", "C3"), ("contrique", "C2")]:
	accuracy = list(Without_Distortion[0][model_name][:-1])
	accuracy += accuracy[:1]

	ax.fill(angle, accuracy, color=color, alpha=0.25)

	if model_name == "clip-vit-l-14":
		label = "UnivFD"
	elif model_name == "drct-clip-vit-l-14":
		label = "DRCT/UnivFD"
	elif model_name == "contrique":
		label = "CONTRIQUE"
	elif model_name == "reiqa":
		label = "ReIQA"
	else:
		label = model_name

	ax.plot(angle, accuracy, color=color, linewidth=2, label=label)

plt.legend(loc="center")
plt.savefig("plots/GenImage_polar_plot.png", bbox_inches='tight', pad_inches=0.1, dpi=750)