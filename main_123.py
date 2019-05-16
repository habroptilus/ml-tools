from sklearn.datasets import load_boston
from gaussian_kernel_l1 import GaussianKernelNorm1
from gaussian_kernel_l2 import GaussianKernelNorm2
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    boston = load_boston()
    X = boston.data
    y = boston.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = GaussianKernelNorm1(h=1000, c=0.00001)
    model.fit(X_train, y_train)
    train_mse = model.evaluate(X_train, y_train)
    test_mse = model.evaluate(X_test, y_test)
    print(f"train MSE : {train_mse}")
    print(f"test MSE : {test_mse}")
    print(f"{sum(model.coef_ < 1e-9)} / {len(model.coef_)}")
