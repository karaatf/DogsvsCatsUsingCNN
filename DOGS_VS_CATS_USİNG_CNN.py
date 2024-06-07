import os,shutil
from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers,models
from keras import optimizers


print("total cat images",len(os.listdir("/content/drive/MyDrive/Colab Notebooks/dogs_vs_cats_first_1000_examples/train_cats")))
print("total dogs images",len(os.listdir("/content/drive/MyDrive/Colab Notebooks/dogs_vs_cats_first_1000_examples/train_dogs")))
print("total val_cats images",len(os.listdir("/content/drive/MyDrive/Colab Notebooks/dogs_vs_cats_first_1000_examples/val_cats")))
print("total val_dogs images",len(os.listdir("/content/drive/MyDrive/Colab Notebooks/dogs_vs_cats_first_1000_examples/val_dogs")))
print("total test_cats images",len(os.listdir("/content/drive/MyDrive/Colab Notebooks/dogs_vs_cats_first_1000_examples/test_cats")))
print("total test_dogs images",len(os.listdir("/content/drive/MyDrive/Colab Notebooks/dogs_vs_cats_first_1000_examples/test_dogs")))



model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation="relu"))
model.add(layers.Dense(1,activation="sigmoid"))


model.summary()



model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

from keras.preprocessing.image import ImageDataGenerator as IDG


train_data_gen= IDG(rescale=1./255)
test_data_gen= IDG(rescale=1./255)

train_generator=train_data_gen.flow_from_directory(
    "/content/drive/MyDrive/Colab Notebooks/dogs_vs_cats_first_1000_examples/train",
    target_size=(150,150),
    batch_size=20,
    class_mode="binary")

val_generator=test_data_gen.flow_from_directory(
    "/content/drive/MyDrive/Colab Notebooks/dogs_vs_cats_first_1000_examples/val",
    target_size=(150,150),
    batch_size=20,
    class_mode="binary")


for data_batch,label_batch in train_generator:
  print("data_bach_size:",data_batch.shape)
  print("labels batch size:",label_batch.shape)
  break

history = model.fit(train_generator, steps_per_epoch=100, epochs=30, validation_data=val_generator,validation_steps=50)


model.save("cat_and_dog_small_1.h5")

history=history.history
history_dic = history
loss_values = history_dic["loss"]
val_loss_val=history_dic["val_loss"]
epochs = range(1,len(loss_values)+1)


plt.clf()#şekli temizler
plt.plot(epochs,loss_values,"bo",label="Eğitim kaybı")
plt.plot(epochs,val_loss_val,"b",label="Doğruluk kaybı")
plt.title("Eğitim veDoğruluk kaybı")
plt.xlabel("epoklar")
plt.ylabel("kayıp")
plt.legend()
plt.show()
