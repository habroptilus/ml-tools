from anomaly_data_generator import AnomalyDataGenerator
from plot import Plotter
from tukey_regression import TukeyRegression

N = 50
seed = 10

gen = AnomalyDataGenerator(seed=seed)
x, y = gen.run(N)
p = Plotter()
p.plot_data(x, y, gen)

model = TukeyRegression(seed=seed)
model.fit(x, y)
y_hat = model.predict(x)
p.plot_model(x, y, model.theta)
print(model.theta)
