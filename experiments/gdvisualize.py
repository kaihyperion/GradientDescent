import numpy as np
from your_code import GradientDescent, load_data
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


print('Starting gdvisualize experiment')

train_features, test_features, train_targets, test_targets = load_data('mnist-binary', fraction=1)

gd = GradientDescent(loss='hinge', learning_rate=1e-4)
gd.fit(train_features, train_targets)

loss_np = np.asarray(gd.lossdata)
accuracy_np = np.asarray(gd.accuracydata)

x_values = np.arange(1, len(gd.lossdata)+1)

plt.plot(x_values, loss_np, label = "Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Mnist Batch Gradient Descent - Loss")
plt.savefig("Batch Gradient - Loss")
plt.cla()

plt.plot(x_values, accuracy_np, label = "Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Mnist Batch Gradient Descent - Accuracy")
plt.savefig("Batch Gradient - Accuracy")
plt.cla()


gd = GradientDescent(loss='hinge', learning_rate=1e-4)
gd.fit(train_features, train_targets, batch_size=1, max_iter=train_features.shape[0]*1000)
loss_np = np.asarray(gd.lossdata)
accuracy_np = np.asarray(gd.accuracydata)

x_values = np.arange(1, len(gd.lossdata)+1)

plt.plot(x_values, loss_np, label = "Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Mnist Stochastic Gradient Descent - Loss")
plt.savefig("Stochastic Gradient - Loss")
plt.cla()

plt.plot(x_values, accuracy_np, label = "Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Mnist Stochastic Gradient Descent - Accuracy")
plt.savefig("Stochastic Gradient - Accuracy")
plt.cla()
