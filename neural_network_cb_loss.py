import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from collections import OrderedDict
from weighted_softmax import cb_softmax_loss_fn


class NeuralNetworkCBLoss(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 n_classes,
                 n_samples,
                 lr, beta):
        super(NeuralNetworkCBLoss, self).__init__()

        # Store the variable
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.n_classes = n_classes
        self.lr = lr

        # Initialize weight matrices based on hidden sizes
        self.layers = []
        layer_sizes = [input_size] + hidden_sizes
        for i in range(len(layer_sizes) - 1):
            self.layers.append(('linear{0:d}'.format(i), nn.Linear(layer_sizes[i], layer_sizes[i+1])))
            self.layers.append(('relu{0:d}'.format(i), nn.ReLU()))
        self.layers.append(('logits', nn.Linear(layer_sizes[-1], n_classes)))
        self.model = nn.Sequential(OrderedDict(self.layers))

        # Initialize loss function and optimizer
        self.optimizer = optim.Adagrad(self.model.parameters(), lr=lr)
        self.loss = cb_softmax_loss_fn(n_classes, n_samples, beta)

        # Use GPU if available
        self.device = T.device('cpu')
        self.model.to(self.device)

    def forward(self, X):
        return self.model(X)

    def predict(self, X):
        logits = self.model(X)
        return np.argmax(logits.detach().numpy(), axis=1)

    def train(self, input_data, target):
        # Zero the gradient
        self.optimizer.zero_grad()

        # Forward propagation
        logits = self.forward(input_data)
        loss = self.loss(logits=logits, labels=target)

        # Backward propagation
        loss.backward()
        self.optimizer.step()

