# Importing Libaries
import os

# Paths
if os.path.exists("/mnt/LIVELAB_NAS/krishna/Perceptual-Classifiers"):
	# For darthvader, leia, odin, diochan
	main_dataset_dir = "/mnt/LIVELAB_NAS/krishna/Datasets"
	main_checkpoints_dir = "/mnt/LIVELAB_NAS/krishna/Perceptual-Classifiers/checkpoints"
	main_feature_ckpts_dir = "/mnt/LIVELAB_NAS/krishna/Perceptual-Classifiers/feature_extractor_checkpoints"
	main_prior_checkpoints_dir = "/mnt/LIVELAB_NAS/krishna/Perceptual-Classifiers/prior_methods_checkpoints"
elif os.path.exists("/mnt/LIVELAB2/krishna/Perceptual-Classifiers"):
	# For odin
	main_dataset_dir = "/mnt/LIVELAB2/krishna/Datasets"
	main_checkpoints_dir = "/mnt/LIVELAB2/krishna/Perceptual-Classifiers/checkpoints"
	main_feature_ckpts_dir = "/mnt/LIVELAB2/krishna/Perceptual-Classifiers/feature_extractor_checkpoints"
	main_prior_checkpoints_dir = "/mnt/LIVELAB2/krishna/Perceptual-Classifiers/prior_methods_checkpoints"
elif os.path.exists("/mnt/LIVELAB_NAS2/krishna/Perceptual-Classifiers"):
	# For genesis
	main_dataset_dir = "/mnt/LIVELAB_NAS2/krishna/Datasets"
	main_checkpoints_dir = "/mnt/LIVELAB_NAS2/krishna/Perceptual-Classifiers/checkpoints"
	main_feature_ckpts_dir = "/mnt/LIVELAB_NAS2/krishna/Perceptual-Classifiers/feature_extractor_checkpoints"
	main_prior_checkpoints_dir = "/mnt/LIVELAB_NAS2/krishna/Perceptual-Classifiers/prior_methods_checkpoints"
else:
	assert False, "Invalid Dataset Directory"



# Sources
All_UnivFD_Sources = {
	"train": ["progan"],
	"val": ["progan", "cyclegan", "biggan", "stylegan", "gaugan", "stargan", "deepfake", "seeingdark", "san", "crn", "imle", "guided", "ldm_200", "ldm_200_cfg", "ldm_100", "glide_100_27", "glide_50_27", "glide_100_10", "dalle"]
}

All_GenImage_Sources = {
	"train": ["biggan", "vqdm", "sdv4", "sdv5", "wukong", "adm", "glide", "midjourney"],
	"val": ["biggan", "vqdm", "sdv4", "sdv5", "wukong", "adm", "glide", "midjourney"]
}

All_DRCT_Sources = {
	"train": ['stable-diffusion-v1-4', 'stable-diffusion-2-1'],
	"val": [
		'ldm-text2im-large-256', 'stable-diffusion-v1-4', 'stable-diffusion-v1-5', 'stable-diffusion-2-1', 'stable-diffusion-xl-base-1.0', 'stable-diffusion-xl-refiner-1.0',
		'sd-turbo', 'sdxl-turbo', 
		'lcm-lora-sdv1-5', 'lcm-lora-sdxl',
		'sd-controlnet-canny', 'sd21-controlnet-canny', 'controlnet-canny-sdxl-1.0',
		'stable-diffusion-inpainting', 'stable-diffusion-2-inpainting', 'stable-diffusion-xl-1.0-inpainting-0.1']
}