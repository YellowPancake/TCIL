import numpy as np
import torch
from torch import nn
from torch.optim import SGD
import torch.nn.functional as F
from inclearn.tools.metrics import ClassErrorMeter, AverageValueMeter
from inclearn.loss.focalloss import BCEFocalLoss, MultiCEFocalLoss


def finetune_last_layer(
    logger,
    network,
    loader,
    n_class,
    nepoch=30,
    lr=0.1,
    scheduling=[15, 35],
    lr_decay=0.1,
    weight_decay=5e-4,
    loss_type="ce",
    temperature=5.0,
    test_loader=None,
    samples_per_cls = []
):
    network.eval()
    #if hasattr(network.module, "convnets"):
    #    for net in network.module.convnets:
    #        net.eval()
    #else:
    #    network.module.convnet.eval()
    optim = SGD(network.module.classifier.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, scheduling, gamma=lr_decay)

    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    logger.info("Begin finetuning last layer")

    for i in range(nepoch):
        total_loss = 0.0
        total_correct = 0.0
        total_count = 0
        # print(f"dataset loader length {len(loader.dataset)}")
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            if loss_type == "bce":
                targets = to_onehot(targets, n_class)
            outputs = network(inputs)['logit']
            _, preds = outputs.max(1)
            optim.zero_grad()
            if loss_type == "cb":
                loss = CB_loss(targets, outputs / temperature, samples_per_cls, n_class, "focal")
            else:
                loss = criterion(outputs / temperature, targets)
            loss.backward()
            optim.step()
            total_loss += loss * inputs.size(0)
            total_correct += (preds == targets).sum()
            total_count += inputs.size(0)

        if test_loader is not None:
            test_correct = 0.0
            test_count = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = network(inputs.cuda())['logit']
                    _, preds = outputs.max(1)
                    test_correct += (preds.cpu() == targets).sum().item()
                    test_count += inputs.size(0)

        scheduler.step()
        if test_loader is not None:
            logger.info(
                "Epoch %d finetuning loss %.3f acc %.3f Eval %.3f" %
                (i, total_loss.item() / total_count, total_correct.item() / total_count, test_correct / test_count))
        else:
            logger.info("Epoch %d finetuning loss %.3f acc %.3f" %
                        (i, total_loss.item() / total_count, total_correct.item() / total_count))
    return network

def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta=0.99, gamma=2.0):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
    labels: A int tensor of size [batch].
    logits: A float tensor of size [batch, no_of_classes].
    samples_per_cls: A python list of size [no_of_classes].
    no_of_classes: total number of classes. int
    loss_type: string. One of "sigmoid", "focal", "softmax".
    beta: float. Hyperparameter for Class balanced loss.
    gamma: float. Hyperparameter for Focal loss.
    Returns:
    cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    # print(labels.device, logits.device)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float().cpu()

    weights = torch.tensor(weights).float()
    # weights = weights.unsqueeze(0)
    # print(labels_one_hot.device, weights.device)
    # weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    # weights = weights.sum(1)
    # weights = weights.unsqueeze(1)
    # weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        # print(effective_num.shape, weights.shape, len(samples_per_cls))
        focalloss = MultiCEFocalLoss(class_num=no_of_classes, gamma=gamma, alpha=weights).cuda()
        cb_loss = focalloss(logits, labels)
        # cb_loss = self.focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss

def extract_features(model, loader):
    targets, features = [], []
    model.eval()
    with torch.no_grad():
        for _inputs, _targets in loader:
            _inputs = _inputs.cuda()
            _targets = _targets.numpy()
            _features = model(_inputs)['feature'].detach().cpu().numpy()
            features.append(_features)
            targets.append(_targets)

    return np.concatenate(features), np.concatenate(targets)


def calc_class_mean(network, loader, class_idx, metric):
    EPSILON = 1e-8
    features, targets = extract_features(network, loader)
    # norm_feats = features/(np.linalg.norm(features, axis=1)[:,np.newaxis]+EPSILON)
    # examplar_mean = norm_feats.mean(axis=0)
    examplar_mean = features.mean(axis=0)
    if metric == "cosine" or metric == "weight":
        examplar_mean /= (np.linalg.norm(examplar_mean) + EPSILON)
    return examplar_mean


def update_classes_mean(network, inc_dataset, n_classes, task_size, share_memory=None, metric="cosine", EPSILON=1e-8):
    loader = inc_dataset._get_loader(inc_dataset.data_inc,
                                     inc_dataset.targets_inc,
                                     shuffle=False,
                                     share_memory=share_memory,
                                     mode="test")
    class_means = np.zeros((n_classes, network.module.features_dim))
    count = np.zeros(n_classes)
    network.eval()
    with torch.no_grad():
        for x, y in loader:
            feat = network(x.cuda())['feature']
            for lbl in torch.unique(y):
                class_means[lbl] += feat[y == lbl].sum(0).cpu().numpy()
                count[lbl] += feat[y == lbl].shape[0]
        for i in range(n_classes):
            class_means[i] /= count[i]
            if metric == "cosine" or metric == "weight":
                class_means[i] /= (np.linalg.norm(class_means) + EPSILON)
    return class_means
