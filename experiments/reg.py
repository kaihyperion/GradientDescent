from your_code import GradientDescent, load_data
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

train_features, test_features, train_targets, test_targets = load_data('mnist-binary', fraction=1)
print(train_features.shape[0])
'''

lam = [1e-3, 1e-2, 1e-1, 1, 10, 100]
nonzerol1 = []
nonzerol2 = []
for l in range(len(lam)):
    gdl1 = GradientDescent(loss='squared',regularization='l1',learning_rate=1e-5, reg_param=lam[l])
    gdl1.fit(train_features, train_targets, max_iter = 2000)
    print(gdl1.it)
    num = 0
    for i in range(gdl1.model.shape[0] - 1):
        if abs(gdl1.model[i]) > 0.001: num +=1
    nonzerol1.append(num)


for l in range(len(lam)):
    gdl2 = GradientDescent(loss='squared',regularization='l2',learning_rate=1e-5, reg_param=lam[l])
    gdl2.fit(train_features, train_targets, max_iter = 2000)
    print(gdl2.it)
    num = 0
    for i in range(gdl2.model.shape[0] - 1):
        if abs(gdl2.model[i]) > 0.001: num +=1
    nonzerol2.append(num)

lamnp = np.asarray(lam)
nzl1np = np.asarray(nonzerol1)
nzl2np = np.asarray(nonzerol2)

plt.plot(lamnp, nzl1np, label="L1 Regularization")
plt.plot(lamnp, nzl2np, label="L2 Regularization")
plt.xlabel("Lambda")
plt.ylabel("Non-Zero Model Weights")
plt.title("Lambda and Non-Zero Model Weights")
plt.legend()
plt.savefig("RegLambda")

plt.cla()

gr = GradientDescent(loss='squared',regularization='l1',learning_rate=1e-5, reg_param=1)
gr.fit(train_features, train_targets, max_iter = 2000)
values = gr.model[:-1]
for v in range(values.shape[0]):
    if values[v] <= 0.001: values[v] = 0
    else: values[v] = 1


plt.imshow(np.array(values).reshape((28,28)), cmap = 'plasma')
plt.colorbar()
plt.title("Zero and Non-Zero Values of Weights in L1 Model with Reg_Param=1")
plt.savefig("Heatmap")
'''