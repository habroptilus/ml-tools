from one_vs_rest_classifier import OneVsRestClassifier
from gaussian_kernel_ls_classifier import GaussianKernelLSBinaryClassifier
from digit_data_loader import DigitDataLoader
CLASS_NUM = 10

data_loader = DigitDataLoader(class_num=CLASS_NUM)
X_train, y_train, X_test, y_test = data_loader.load()
print(X_train.shape, X_train.shape, X_test.shape, X_test.shape)

params = {"h": 1000, "c": 0}

multi_classifier = OneVsRestClassifier(
    class_num=CLASS_NUM, classifier=GaussianKernelLSBinaryClassifier, params=params)

multi_classifier.fit(X_train, y_train)
acc = multi_classifier.evaluate(X_test, y_test)

print(f"Test Accuracy = {acc*100:.2f}%")
