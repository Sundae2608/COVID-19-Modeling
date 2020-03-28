import numpy as np
import pandas as pd
import torch as T

from neural_network_cb_loss import NeuralNetworkCBLoss
from sklearn.metrics import f1_score

# Read the data from the CSV file
data = pd.read_csv("data/creditcard.csv")

# Count the number of samples for each class

# Hyper-parameters
beta = 0.99
lr = 1e-3

# Data processing
p_data = data.copy()

# Amount has very different scale compared to the rest of the variables an needs to be normalized
p_data["Amount"] = (p_data["Amount"] - p_data["Amount"].mean()) / p_data["Amount"].std()

# Remove labels from data
p_label = p_data["Class"]
p_data = p_data.drop(["Class", "Time"], axis=1)

# Split data into 7 / 3 ratio
mask = np.random.rand(len(p_data)) < 0.7
train_data = p_data[mask]
train_label = p_label[mask]

test_data = p_data[~mask]
test_label = p_label[~mask]
t_test_data = T.tensor(test_data.values, dtype=T.float32)
t_test_label = T.tensor(test_label.values, dtype=T.float32)

# Initialize neural networks
n_classes = p_label.nunique()
n_samples = p_label.value_counts()
input_size = len(p_data.count())
hidden_sizes = [50, 50]

model = NeuralNetworkCBLoss(
    input_size=input_size,
    hidden_sizes=hidden_sizes,
    n_classes=n_classes,
    n_samples=n_samples,
    lr=lr, beta=beta
)

# Train the network

num_epochs = 10000
batch_size = 128
num_batches = 20

# Train loader is a nice utility to feed network slowly over time, in a parallel fashion
# Source: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
for epoch in range(num_epochs):
    running_loss = 0.0
    for i in range(num_batches):
        indices = p_data.sample(batch_size).index
        sample_X = T.tensor(p_data.iloc[indices].values, dtype=T.float32)
        sample_y = T.tensor(p_label.iloc[indices].values, dtype=T.int64)
        model.train(sample_X, sample_y)

    test_pred = model.predict(t_test_data)
