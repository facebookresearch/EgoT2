#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

class lossAV(nn.Module):
	def __init__(self, dim=256):
		super(lossAV, self).__init__()
		self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 4]))
		self.FC        = nn.Linear(dim, 2)
		
	def forward(self, x, labels=None):	
		x = x.squeeze(1)
		x = self.FC(x)
		if labels == None:
			predScore = x[:,1]
			predScore = predScore.t()
			predScore = predScore.view(-1).detach().cpu().numpy()
			return predScore
		else:
			nloss = self.criterion(x, labels)
			predScore = F.softmax(x, dim = -1)
			predLabel = torch.round(F.softmax(x, dim = -1))[:,1]
			correctNum = (predLabel == labels).sum().float()
			return nloss, predScore, predLabel, correctNum

class lossA(nn.Module):
	def __init__(self):
		super(lossA, self).__init__()
		self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 4]))
		self.FC        = nn.Linear(128, 2)

	def forward(self, x, labels):	
		x = x.squeeze(1)
		x = self.FC(x)	
		nloss = self.criterion(x, labels)
		return nloss

class lossV(nn.Module):
	def __init__(self):
		super(lossV, self).__init__()

		self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 4]))
		self.FC        = nn.Linear(128, 2)

	def forward(self, x, labels):	
		x = x.squeeze(1)
		x = self.FC(x)
		nloss = self.criterion(x, labels)
		return nloss

