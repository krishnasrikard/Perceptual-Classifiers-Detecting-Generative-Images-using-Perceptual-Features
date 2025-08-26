"""
Loss Functions, Optimizers and Evaluation Metrics
"""
# Importing Libraries
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score, matthews_corrcoef

import torch
import torch.nn as nn

import os, sys, warnings
warnings.filterwarnings("ignore")


# Margin-Based Constrative Loss
class MarginContrastiveLoss(nn.Module):
	def __init__(self, margin=1):
		"""
		Reference: https://github.com/beibuwandeluori/DRCT/blob/main/utils/losses.py
		"""
		super(MarginContrastiveLoss, self).__init__()
		self.margin = margin

	def forward(self, projections, targets):
		"""
		Args:
			projections (torch.Tensor): Projections of shape (batch_size, projection_dim)
			targets (torch.Tensor): Target Predictions of shape (batch_size)
		"""
		# Device
		device = projections.device
		batch_size = projections.shape[0]

		# Pair-wise Distance
		repeat_projections1 = projections.unsqueeze(0).repeat(batch_size, 1, 1)
		repeat_projections2 = projections.unsqueeze(1).repeat(1, 1, 1)
		pairwise_distance = torch.nn.functional.pairwise_distance(repeat_projections2, repeat_projections1, p=2)

		# Mask: Similar Classes
		mask_dissimilar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) != targets).to(device)
		mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)

		# Contrastive Loss
		loss = torch.empty_like(pairwise_distance).to(device)

		loss[mask_similar_class] = pairwise_distance[mask_similar_class]
		loss[mask_dissimilar_class] = torch.clamp(self.margin - pairwise_distance[mask_dissimilar_class], min=0)

		contrastive_loss = torch.mean(torch.pow(loss, exponent=2))

		return contrastive_loss
	

# Margin-Based Constrative Loss with Cross-Entropy
class MarginContrastiveLoss_CrossEntropy(nn.Module):
	def __init__(self, margin=1, lambda_=0.3):
		"""
		Reference: https://github.com/beibuwandeluori/DRCT/blob/main/utils/losses.py
		"""
		super(MarginContrastiveLoss_CrossEntropy, self).__init__()
		self.margin = margin
		self.lambda_ = lambda_
		self.margin_contrastive_loss_fn = MarginContrastiveLoss()
		self.cross_entropy_loss_fn = nn.CrossEntropyLoss()

	def forward(self, projections, preds, targets):
		"""
		Args:
			projections (torch.Tensor): Projections of shape (batch_size, projection_dim)
			targets (torch.Tensor): Target Predictions of shape (batch_size)
			preds (torch.Tensor): Predictions of shape (batch_size, num_classes)
		"""
		# Margin-based Contrastive Loss
		contrastive_loss = self.margin_contrastive_loss_fn(projections, targets)

		# Cross-Entropy Loss
		cross_entropy_loss = self.cross_entropy_loss_fn(preds, targets)

		# Total Loss
		loss = (self.lambda_ * contrastive_loss) + ((1 - self.lambda_) * cross_entropy_loss)

		return loss
	

# Multi-Margin Loss
class MultiMarginLoss_(nn.Module):
	def __init__(self, margin=2, p=2):
		super(MultiMarginLoss_, self).__init__()
		self.loss_fn = nn.MultiMarginLoss(p=p, margin=margin)

	def forward(self, projections, preds, targets):
		"""
		Args:
			projections (torch.Tensor): Projections of shape (batch_size, projection_dim)
			targets (torch.Tensor): Target Predictions of shape (batch_size)
			preds (torch.Tensor): Predictions of shape (batch_size, num_classes)
		"""
		loss = self.loss_fn(preds, targets)

		return loss
	

# Cross-Entropy Loss
class CrossEntropy_(nn.Module):
	def __init__(self):
		super(CrossEntropy_, self).__init__()
		self.loss_fn = nn.CrossEntropyLoss()

	def forward(self, projections, preds, targets):
		"""
		Args:
			projections (torch.Tensor): Projections of shape (batch_size, projection_dim)
			targets (torch.Tensor): Target Predictions of shape (batch_size)
			preds (torch.Tensor): Predictions of shape (batch_size, num_classes)
		"""
		loss = self.loss_fn(preds, targets)

		return loss
	

# Get Loss Function
def get_loss_function(
	**kwargs
):
	if kwargs["name"] == "CrossEntropy":
		return CrossEntropy_()
	elif kwargs["name"] == "MultiMarginLoss":
		return MultiMarginLoss_(margin=1, p=2)
	elif kwargs["name"] == "MarginContrastiveLoss":
		return MarginContrastiveLoss(margin=1)
	elif kwargs["name"] == "MarginContrastiveLoss_CrossEntropy":
		return MarginContrastiveLoss_CrossEntropy(margin=1, lambda_=0.3)
	else:
		assert False, "Invalid Loss Function"


# Get Optimizer
def get_optimizer(
	parameters,
	**kwargs
):
	if kwargs["name"] == "SGD":
		return torch.optim.SGD(params = parameters, lr = kwargs["lr"], weight_decay = kwargs["weight_decay"])
	elif kwargs["name"] == "Adam":
		return torch.optim.Adam(params = parameters, lr = kwargs["lr"], weight_decay = kwargs["weight_decay"])
	elif kwargs["name"] == "AdamW":
		return torch.optim.AdamW(params = parameters, lr = kwargs["lr"], weight_decay = kwargs["weight_decay"])
	else:
		assert False, "Invalid Optimizer"


# Concatenate Predictions
def concatenate_predictions(
	y_pred_y_true:any
):
	"""
	Concatenating predictions and applying necessary post processing on predictions. 
	Args:
		y_pred_y_true (any): Output from Trainer.predict
	"""
	# Concatenating
	y_pred = []
	y_true = []
	for i in range(len(y_pred_y_true)):
		y_pred.append(y_pred_y_true[i][0])
		y_true.append(y_pred_y_true[i][1])

	y_pred = torch.concat(y_pred, dim=0)
	y_true = torch.concat(y_true, dim=0)

	# Post Processing
	"""
	- Converting Logits to Softmax Probabilities as we are either using MultiMarginLoss or CrossEntropy, which means that predictions are logits and are not normalized probabilities
	- If only one prediction as output, we apply ssigmoid and estimate probabilities for both labels
	"""
	if y_pred.shape[1] == 1:
		y_pred = torch.nn.functional.sigmoid(y_pred)
		y_pred = torch.concat([1-y_pred, y_pred], dim=1)
	else:
		y_pred = torch.nn.functional.softmax(y_pred.to(torch.float32), dim=1)

	return y_pred.numpy(), y_true.numpy()


# Finding mAcc threshold.
def find_best_threshold(
	y_true:np.array,
	y_pred:np.array
):
	"""
	- Source: https://github.com/WisconsinAIVision/UniversalFakeDetect/blob/main/validate.py
	- We assume first half of y_true is real 0, and the second half is fake 1
	Args:
		y_true (np.array): True Labels.
		y_pred (np.array): Predicted Labels.
	"""
	# Assertions
	assert np.all((y_pred >= 0) & (y_pred <= 1)), "y_pred does not lie between 0 and 1"
	assert np.all((y_true >= 0) & (y_true <= 1)), "y_true does not lie between 0 and 1"

	N = y_true.shape[0]

	best_acc = 0 
	best_thres = 0 
	for thres in y_pred:
		temp = np.copy(y_pred)
		temp[temp>=thres] = 1 
		temp[temp<thres] = 0 

		acc = np.sum(temp == y_true)/N  
		if acc >= best_acc:
			best_thres = thres
			best_acc = acc 

	return best_thres


# Calculate Accuracy
def calculate_accuracy(y_true, y_pred, thres):
	"""
	- Source: https://github.com/WisconsinAIVision/UniversalFakeDetect/blob/main/validate.py
	- We assume first half of y_true is real 0, and the second half is fake 1
	Args:
		y_true (np.array): True Labels.
		y_pred (np.array): Predicted Labels.
	"""
	r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] >= thres)
	f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] >= thres)
	acc = accuracy_score(y_true, y_pred >= thres)

	return acc, r_acc, f_acc


# Get Metrics
def calculate_metrics(
	y_pred:np.array,
	y_true:np.array,
	threshold:float,
):
	"""
	Calculating Metrics
	Args:
		y_pred (np.array): Predictions Probabilities.
		y_true (np.array): True Labels
		threshold (float): Threshold to calculate accuracy.
	"""
	# Get AP
	ap = average_precision_score(y_true, y_pred)
	ap = np.round(ap, decimals=4)

	# Accuracy when threshold = 0.5
	acc0, r_acc0, f_acc0 = calculate_accuracy(y_true, y_pred, 0.5)
	acc0 = np.round(acc0, decimals=4)
	r_acc0 = np.round(r_acc0, decimals=4)
	f_acc0 = np.round(f_acc0, decimals=4)

	# best threshold
	if threshold is None:
		threshold = find_best_threshold(y_true, y_pred)
		print ()
		print ("Calculated best_threshold =", threshold)
	else:
		print ()
		print ("Using given best_threshold =", threshold)

	# Accuracy based on the best threshold
	acc1, r_acc1, f_acc1 = calculate_accuracy(y_true, y_pred, threshold)
	acc1 = np.round(acc1, decimals=4)
	r_acc1 = np.round(r_acc1, decimals=4)
	f_acc1 = np.round(f_acc1, decimals=4)

	# Mathews Correlation Coefficient when threshold = 0.5
	mcc0 = matthews_corrcoef(y_true, y_pred >= 0.5)
	mcc1 = matthews_corrcoef(y_true, y_pred >= threshold)

	return ap, acc0, r_acc0, f_acc0, acc1, r_acc1, f_acc1, mcc0, mcc1, threshold