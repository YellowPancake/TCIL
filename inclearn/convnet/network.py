import copy
import pdb

import torch
from torch import nn
import torch.nn.functional as F

from inclearn.tools import factory
from inclearn.convnet.imbalance import CR, All_av
from inclearn.convnet.classifier import CosineClassifier


class BasicNet(nn.Module):
    def __init__(
        self,
        convnet_type,
        cfg,
        nf=64,
        use_bias=False,
        init="kaiming",
        device=None,
        dataset="cifar100",
    ):
        super(BasicNet, self).__init__()
        self.nf = nf
        self.init = init
        self.convnet_type = convnet_type
        self.dataset = dataset
        self.start_class = cfg['start_class']
        self.weight_normalization = cfg['weight_normalization']
        self.remove_last_relu = True if self.weight_normalization else False
        self.use_bias = use_bias if not self.weight_normalization else False
        self.dea = cfg['dea']
        self.ft_type = cfg.get('feature_type', 'normal')
        self.at_res = cfg.get('attention_use_residual', False)
        self.div_type = cfg['div_type']
        self.reuse_oldfc = cfg['reuse_oldfc']
        self.prune = cfg.get('prune', False)
        self.reset = cfg.get('reset_se', True)


        if self.dea:
            print("Enable dynamical reprensetation expansion!")
            self.convnets = nn.ModuleList()
            self.convnets.append(
                factory.get_convnet(convnet_type,
                                    nf=nf,
                                    dataset=dataset,
                                    start_class=self.start_class,
                                    remove_last_relu=self.remove_last_relu))
            self.out_dim = self.convnets[0].out_dim
        else:
            self.convnet = factory.get_convnet(convnet_type,
                                               nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu)
            self.out_dim = self.convnet.out_dim
        self.classifier = None
        self.se = None
        self.aux_classifier = None

        self.n_classes = 0
        self.ntask = 0
        self.device = device

        if cfg['postprocessor']['enable']:
            if cfg['postprocessor']['type'].lower() == "cr":
                self.postprocessor = CR()
            elif cfg['postprocessor']['type'].lower() == "aver":
                self.postprocessor = All_av()
        else:
            self.postprocessor = None

        self.to(self.device)

    def forward(self, x):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

        if self.dea:
            feature = [convnet(x) for convnet in self.convnets]
            features = torch.cat(feature, 1)
            last_dim = feature[-1].size(1)
            width = features.size(1)

            if self.reset:
                se = factory.get_attention(width, self.ft_type, self.at_res).to(self.device)
                features = se(features)
            else:
                features = self.se(features)
     
        else:
            features = self.convnet(x)

        logits = self.classifier(features)

        div_logits = self.aux_classifier(features[:, -last_dim:]) if self.ntask > 1 else None

        return {'feature': features, 'logit': logits, 'div_logit': div_logits, 'features': feature}

    def caculate_dim(self, x):
        feature = [convnet(x) for convnet in self.convnets]
        features = torch.cat(feature, 1)

        width = features.size(1)

        # se = factory.get_attention(width, self.ft_type, self.at_res).to(self.device)
        se = factory.get_attention(width, "ce", self.at_res).cuda()
        features = se(features)

        # import pdb
        # pdb.set_trace()
        return features.size(1), feature[-1].size(1)     

    @property
    def features_dim(self):
        if self.dea:
            return self.out_dim * len(self.convnets)
        else:
            return self.out_dim

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        self.ntask += 1

        if self.dea:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes):
        if self.ntask > 1:
            new_clf = factory.get_convnet(self.convnet_type,
                                          nf=self.nf,
                                          dataset=self.dataset,
                                          start_class=self.start_class,
                                          remove_last_relu=self.remove_last_relu).to(self.device)
            if self.prune:        
                pass
            else:
                new_clf.load_state_dict(self.convnets[-1].state_dict())
            self.convnets.append(new_clf)

        if not self.reset:
            self.se = factory.get_attention(512*len(self.convnets), self.ft_type, self.at_res)
            self.se.to(self.device)

        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)

        fc = self._gen_classifier(self.out_dim * len(self.convnets), self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            fc.weight.data[:self.n_classes, :self.out_dim * (len(self.convnets) - 1)] = weight
        del self.classifier
        self.classifier = fc

        if self.div_type == "n+1":
            div_fc = self._gen_classifier(self.out_dim, n_classes + 1)
        elif self.div_type == "1+1":
            div_fc = self._gen_classifier(self.out_dim, 2)
        elif self.div_type == "n+t":
            div_fc = self._gen_classifier(self.out_dim, self.ntask + n_classes)
        else:
            div_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
        del self.aux_classifier
        self.aux_classifier = div_fc

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, in_features, n_classes):
        if self.weight_normalization:
            classifier = CosineClassifier(in_features, n_classes).to(self.device)
            # classifier = CosineClassifier(in_features, n_classes).cuda()
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
            # classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).cuda()
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)

        return classifier
