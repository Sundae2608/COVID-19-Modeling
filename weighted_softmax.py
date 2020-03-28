"""
    Reimplementation of Class-Balanced Loss. Heavily based on the reimplementation by vandit15 at:
    https://github.com/vandit15/Class-balanced-loss-pytorch/

    Used for COVID-19 modeling.

    Reference: "Class-Balanced Loss Based on Effective Number of Samples", CVPR 19'
    Authors: Yin Cui, Menglin Jia, Tsung Yi Lin, Yang Song and Serge J. Belongie.
    Link: https://arxiv.org/abs/1901.05555
"""

import numpy as np
import torch as T
import torch.nn.functional as F
from functools import partial


def cb_softmax_loss(logits, labels, n_classes, n_samples, beta):
    """
    Class balanced loss
    :param labels: Ground truth.
    :param logits: Prediction.
    :param n_classes: Number of classes.
    :param n_samples: Number of samples for each class.
    :param beta: Hyper-parameter for class balanced loss, representing our estimation of growth in sampling volume.
    :return: Float tensor represents the class balanced loss.
    """
    # Calculate the class-balanced weight terms
    weights = (1.0 - beta) / np.array(1.0 - np.power(beta, n_samples))

    # Normalize the weights across number of classes
    weights = weights / sum(weights)

    # Convert labels to one hot vector, and apply the weight terms on each label
    labels_one_hot = F.one_hot(input=labels, num_classes=n_classes).float()
    weights = T.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(len(labels), 1)
    weights = weights * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, n_classes)

    # Calculate and return the weighted softmax loss
    loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
    return loss


def cb_softmax_loss_fn(n_classes, n_samples, beta):
    """
    Return a loss function that is already weighted using class balance
    :param n_classes: Number of classes
    :param n_samples: Number of samples per class
    :param beta: Hyper-parameter beta
    :return:
    """
    return partial(cb_softmax_loss, n_classes=n_classes, n_samples=n_samples, beta=beta)


if __name__ == '__main__':
    n_classes = 5
    n_samples = [2, 3, 1, 2, 2]  # 10 samples
    logits = T.rand(10, n_classes).float()
    labels = T.randint(0, n_classes, size=(10,))
    beta = 0.9999
    gamma = 2.0
    loss_type = "focal"
    cb_loss = cb_softmax_loss(logits, labels, n_classes, n_samples, beta)