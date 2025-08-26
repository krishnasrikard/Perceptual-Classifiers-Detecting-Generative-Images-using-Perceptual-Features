# Perceptual Classifiers for Detecting Generative Images

The following is the official implementation of Perceptual Classifiers: Detecting Generative Images using Perceptual Features.

## Abstract
Image Quality Assessment (IQA) models are employed in many practical image and video processing pipelines to reduce storage, minimize transmission costs, and improve the Quality of Experience (QoE) of millions of viewers. These models are sensitive to a diverse range of image distortions and can accurately predict image quality as judged by human viewers. Recent advancements in generative models have resulted in a significant influx of “GenAI” content on the internet. Existing methods for detecting GenAI content have progressed significantly with improved generalization performance on images from unseen generative models. Here, we leverage the capabilities of existing IQA models, which effectively capture the manifold of real images within a bandpass statistical space, to distinguish between real and AI-generated images. We investigate the generalization ability of these perceptual classifiers to the task of GenAI image detection and evaluate their robustness against various image degradations. Our results show that a two-layer network trained on the feature space of IQA models demonstrates state-of-the-art performance in detecting fake images across generative models, while maintaining significant robustness against image degradations.

## Checkpoints
The checkpoints of the classifiers and feature-extractors can be accessed here: [Link](https://utexas-my.sharepoint.com/:u:/g/personal/kd28684_eid_utexas_edu/EToHGgU9qctDj1KewO8VmkQBEBoK9iFBPYJ6swQ3VOloQQ?e=uvK5kV)

## Papers and Citations
Perceptual Classifiers: Detecting Generative Images using Perceptual Features: [https://arxiv.org/pdf/2507.17240](https://arxiv.org/pdf/2507.17240)
```
@article{durbha2025perceptual,
	title        = {Perceptual Classifiers: Detecting Generative Images using Perceptual Features},
	author       = {Durbha, Krishna Srikar and Venkataramanan, Asvin Kumar and Sureddi, Rajesh and Bovik, Alan C},
	year         = 2025,
	journal      = {arXiv preprint arXiv:2507.17240}
}
```