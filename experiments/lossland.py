import numpy as np
from your_code import load_data, ZeroOneLoss
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

print('Starting lossland experiment')

train_features, test_features, train_targets, test_targets = load_data('synthetic', fraction=1)
bias = [0.5, -0.5, -1.5, -2.5, -3.5, -4.5, -5.5]
losses = []
ones = np.ones((1,train_features.shape[0]))
for b in range(len(bias)):
    X_bias = np.append(train_features, ones.T, axis=1)
    w_bias = np.array([1, bias[b]])
    loss = ZeroOneLoss().forward(X_bias, w_bias, train_targets)
    losses.append(loss)

np_bias = np.asarray(bias)
np_losses = np.asarray(losses)

plt.plot(np_bias, np_losses)
plt.xlabel("Bias Value")
plt.ylabel("Loss")
plt.title("Loss Landscape of Synthetic Dataset")
plt.savefig("Loss Landscapes - 2a")
#plt.cla()

bias = [0.5, -0.5, -1.5, -2.5, -3.5, -4.5, -5.5]
losses = []
points = np.array([0,2,3,4])
ones = np.ones((1,points.shape[0]))
for b in range(len(bias)):
    X_bias = np.append(train_features[points], ones.T, axis=1)
    w_bias = np.array([1, bias[b]])
    loss = ZeroOneLoss().forward(X_bias, w_bias, train_targets[points])
    losses.append(loss)

np_bias = np.asarray(bias)
np_losses = np.asarray(losses)

plt.plot(np_bias, np_losses)
plt.xlabel("Bias Value")
plt.ylabel("Loss")
plt.title("Loss Landscape of Synthetic Dataset")
plt.savefig("Loss Landscapes - 2b")

print(train_features, train_targets)