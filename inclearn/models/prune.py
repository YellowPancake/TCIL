import argparse
import os, sys
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
# from models import print_log
import numpy as np
from collections import OrderedDict

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Mask:
    def __init__(self, model, cfg):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []
        self.cfg = cfg

    def get_codebook(self, weight_torch, compress_rate, length):
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()

        weight_abs = np.abs(weight_np)
        weight_sort = np.sort(weight_abs)

        threshold = weight_sort[int(length * (1 - compress_rate))]
        weight_np[weight_np <= -threshold] = 1
        weight_np[weight_np >= threshold] = 1
        weight_np[weight_np != 1] = 0

        # print("codebook done")
        return weight_np

    def get_filter_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            # norm1 = torch.norm(weight_vec, 1, 1)
            # norm1_np = norm1.cpu().numpy()
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0

            # print("filter codebook done")
        else:
            pass
        return codebook

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()

        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]

    def init_rate(self, layer_rate):
        if 'vgg' in self.cfg["arch"]:
            cfg_5x = [24, 22, 41, 51, 108, 89, 111, 184, 276, 228, 512, 512, 512]
            cfg_official = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
            # cfg = [32, 64, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256]
            cfg_index = 0
            pre_cfg = True
            for index, item in enumerate(self.model.named_parameters()):
                self.compress_rate[index] = 1
                if len(item[1].size()) == 4:
                    # print(item[1].size())
                    if not pre_cfg:
                        self.compress_rate[index] = layer_rate
                        self.mask_index.append(index)
                        # print(item[0], "self.mask_index", self.mask_index)
                    else:
                        self.compress_rate[index] =  1 - cfg_5x[cfg_index] / item[1].size()[0]
                        self.mask_index.append(index)
                        # print(item[0], "self.mask_index", self.mask_index, cfg_index, cfg_5x[cfg_index], item[1].size()[0],)
                        cfg_index += 1
        elif "resnet" in self.cfg["arch"]:
            for index, item in enumerate(self.model.parameters()):
                self.compress_rate[index] = 1
            for key in range(self.cfg["layer_begin"], self.cfg["layer_end"] + 1, self.cfg["layer_inter"]):
                self.compress_rate[key] = layer_rate
            if self.cfg["arch"] == 'resnet18':
                # last index include last fc layer
                last_index = 59
                skip_list = [21, 36, 51]
            elif self.cfg["arch"] == 'resnet34':
                last_index = 108
                skip_list = [27, 54, 93]
            elif self.cfg["arch"] == 'resnet50':
                last_index = 159
                skip_list = [12, 42, 81, 138]
            elif self.cfg["arch"] == 'resnet101':
                last_index = 312
                skip_list = [12, 42, 81, 291]
            elif self.cfg["arch"] == 'resnet152':
                last_index = 465
                skip_list = [12, 42, 117, 444]
            self.mask_index = [x for x in range(0, last_index, 3)]
            # skip downsample layer
            if self.cfg["skip_downsample"] == 1:
                for x in skip_list:
                    self.compress_rate[x] = 1
                    self.mask_index.remove(x)
                    # print(self.mask_index)
            else:
                pass

    def init_mask(self, layer_rate):
        self.init_rate(layer_rate)
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index],
                                                           self.model_length[index])
                self.mat[index] = self.convert2tensor(self.mat[index])
                if self.cfg["use_cuda"]:
                    self.mat[index] = self.mat[index].cuda()
        print("mask Ready")

    def do_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.model_size[index])
        print("mask Done")

    def if_zero(self):
        for index, item in enumerate(self.model.parameters()):
            #            if(index in self.mask_index):
            if index in [x for x in range(self.cfg["layer_begin"], self.cfg["layer_end"] + 1, self.cfg["layer_inter"])]:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()

                # print("layer: %d, number of nonzero weight is %d, zero is %d" % (index, np.count_nonzero(b), len(b) - np.count_nonzero(b)))
