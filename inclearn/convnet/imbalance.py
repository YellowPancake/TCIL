import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR



class CR(object):
    def __init__(self):
        self.gamma = None

    @torch.no_grad()
    def update(self, classifier, task_size):
        old_weight_norm = torch.norm(classifier.weight[:-task_size], p=2, dim=1)
        new_weight_norm = torch.norm(classifier.weight[-task_size:], p=2, dim=1)
        self.gamma = old_weight_norm.mean() / new_weight_norm.mean()
        # print(self.gamma.cpu().item())

    @torch.no_grad()
    def post_process(self, logits, task_size):
        logits[:, -task_size:] = logits[:, -task_size:] * self.gamma
        return logits


class All_av(object):
    def __init__(self):
        self.gamma = []

    @torch.no_grad()
    def update(self, classifier, task_size, classnum_list, taski):
        self.gamma = []
        for i in range(taski+1):
            old_weight_norm = torch.norm(classifier.weight[:-task_size], p=2, dim=1)
            new_weight_norm = torch.norm(classifier.weight[sum(classnum_list[:i]):sum(classnum_list[:i+1])], p=2, dim=1)
            self.gamma.append(old_weight_norm.mean() / new_weight_norm.mean())
        # print(self.gamma)

    @torch.no_grad()
    def post_process(self, logits, task_size, classnum_list, taski):
        for i in range(taski+1):
            logits[:, sum(classnum_list[:i]):sum(classnum_list[:i+1])] = logits[:, sum(classnum_list[:i]):sum(classnum_list[:i+1])] * self.gamma[i]     
        return logits
