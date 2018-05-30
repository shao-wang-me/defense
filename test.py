from keras.models import Sequential
from keras.layers import Dense
import numpy as np

model = Sequential()
model.add(Dense(units=4, input_dim=1))
model.add(Dense(units=4))
model.add(Dense(units=2))
model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()

# from keras.utils import plot_model
# plot_model(model, to_file='model.png')

# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# SVG(model_to_dot(model).create(prog='dot', format='svg'))

x_train = np.fromfunction(lambda i: i, (10000,), dtype=int)
y_train1 = np.fromfunction(lambda i: 3 * i + 8, (10000,), dtype=int)
y_train2 = np.fromfunction(lambda i: 3 * i + 18, (10000,), dtype=int)
y_train = np.column_stack((y_train1, y_train2))
model.fit(x_train, y_train, epochs=500, batch_size=32)


x_test = np.fromfunction(lambda i: i, (10099,), dtype=int)[10000:]
y_test1 = np.fromfunction(lambda i: 3 * i + 8, (10099,), dtype=int)[10000:]
y_test2 = np.fromfunction(lambda i: 3 * i + 18, (10099,), dtype=int)[10000:]
y_test = np.column_stack((y_test1, y_test2))
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

print(loss_and_metrics)

print(model.predict(np.array([1,2,3,4,5,6,7,8,9,10])))