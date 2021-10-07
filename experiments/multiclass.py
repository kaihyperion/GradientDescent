from your_code import load_data, MultiClassGradientDescent, confusion_matrix

train_features, test_features, train_targets, test_targets = load_data('mnist-multiclass', fraction=0.75)

mgd = MultiClassGradientDescent(loss='squared', regularization='l1')

mgd.fit(train_features, train_targets)

predictions = mgd.predict(test_features).astype(int)


cm = confusion_matrix(test_targets, predictions)
print(cm)
