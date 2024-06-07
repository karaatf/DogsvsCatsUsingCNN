from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd

Data = keras.datasets.fashion_mnist.load_data()
(x_train,y_train),(x_test,y_test)=Data
print(x_train[23])

plt.imshow(x_train[1],cmap="gray")

x_train = x_train/255.0
x_test = x_test/255.0
print(y_train[1])
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
print(y_train[1])

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300,activation="relu"))
model.add(keras.layers.Dense(300,activation="relu"))
model.add(keras.layers.Dense(10,activation="softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer=keras.optimizers.SGD(),
              metrics=["accuracy"])

history = model.fit(x_train,y_train,epochs=30,validation_split=0.1)

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

model.evaluate(x_test,y_test)

a=model.predict(x_test[:2])
print(a)